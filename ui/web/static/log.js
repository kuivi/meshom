// meshom web UI: the action log. The page's clicks, choices, messages, map view changes, slow or failed server calls,
// main-thread stalls and a heartbeat of the map go to the server (POST api/log), which writes them with its own record
// of every request (ui/web/server.py ActionLog): what happened before a freeze can be read back afterwards.
const PAGE = Math.random().toString(36).slice(2, 8);     // one page load (a reload starts another)
const Q = [];
const MAX_Q = 3000;                                      // the server unreachable: keep the newest events only
const FLUSH_MS = 2000, BEAT_MS = 5000, STALL_MS = 1000, SLOW_MS = 1000;
const beats = [];
const t0 = performance.now();

export function ulog(ev, data = {}) {
  Q.push({t: Math.round(performance.now() - t0), ev, ...data});
  if (Q.length > MAX_Q) Q.splice(0, Q.length - MAX_Q);
}
// a part of the page adds its numbers to the heartbeat (e.g. the map: renders, view, zoom)
export function addBeat(fn) { beats.push(fn); }

// one flush at a time, at most CHUNK events a request, no keepalive: Chrome refuses keepalive requests once those in
// flight pass 64 KB, and a queue grown while the server was busy then never went out again (the log stopped
// while the page went on)
const send = window.fetch.bind(window);
const CHUNK = 200;
let sending = false;
async function flush() {
  if (sending || !Q.length) return;
  sending = true;
  try {
    while (Q.length) {
      const events = Q.slice(0, CHUNK);
      const r = await send('api/log', {method: 'POST', body: JSON.stringify({page: PAGE, events})});
      if (!r.ok) break;
      Q.splice(0, events.length);
    }
  } catch (e) { /* the server unreachable: the events wait for the next flush */ }
  sending = false;
}
setInterval(flush, FLUSH_MS);
addEventListener('pagehide', () => {                     // the last events (a beacon also stops at 64 KB)
  ulog('pagehide');
  navigator.sendBeacon && navigator.sendBeacon('api/log', JSON.stringify({page: PAGE, events: Q.splice(0, Q.length).slice(-CHUNK)}));
});
document.addEventListener('visibilitychange', () => ulog('visibility', {state: document.visibilityState}));

// server calls: the open ones go into the heartbeat, the slow and the failed ones get their own line
const open = new Map();
let nextCall = 0;
const short = (u) => String(u).replace(/[?&]token=[^&]*/, '').slice(0, 160);
window.fetch = async (url, opts) => {
  const id = nextCall++, path = short(url && url.url || url), t1 = performance.now();
  if (path.startsWith('api/log')) return send(url, opts);
  open.set(id, {path, t1});
  try {
    const r = await send(url, opts);
    const ms = Math.round(performance.now() - t1);
    if (!r.ok || ms > SLOW_MS) ulog('fetch', {path, status: r.status, ms});
    return r;
  } catch (e) {
    ulog('fetch-failed', {path, ms: Math.round(performance.now() - t1), error: String(e && e.message || e)});
    throw e;
  } finally { open.delete(id); }
};

// the main thread blocked: a timer that fires late by more than STALL_MS
let tick = performance.now();
setInterval(() => {
  const now = performance.now(), late = now - tick - 250;
  if (late > STALL_MS) ulog('stall', {ms: Math.round(late)});
  tick = now;
}, 250);

// animation frames per heartbeat: none while the tab is visible means the browser stopped painting the page
let frames = 0;
const frame = () => { frames++; requestAnimationFrame(frame); };
requestAnimationFrame(frame);

setInterval(() => {
  const now = performance.now(), rec = {frames, open: [...open.values()].map((c) => `${c.path} ${((now - c.t1) / 1000).toFixed(0)} s`)};
  frames = 0;
  for (const f of beats) { try { Object.assign(rec, f()); } catch (e) { rec.beatError = String(e); } }
  ulog('beat', rec);
}, BEAT_MS);

// what the user does: clicks on controls, choices in forms, keys that act (Escape, Enter)
const label = (e) => {
  const c = e.closest('button, a, summary, label, [data-view], [data-tab], select, input, textarea, .row') || e;
  const t = (c.getAttribute('aria-label') || c.title || c.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 60);
  return {tag: c.tagName.toLowerCase(), id: c.id || undefined, on: t || undefined,
    view: c.dataset.view, tab: c.dataset.tab, cls: c.tagName === 'CANVAS' ? undefined : (c.className || '').toString().slice(0, 40) || undefined};
};
document.addEventListener('click', (e) => { if (e.target.tagName !== 'CANVAS') ulog('click', label(e.target)); }, true);
document.addEventListener('change', (e) => {
  const t = e.target;
  ulog('change', {...label(t), value: t.type === 'checkbox' ? t.checked : String(t.value).slice(0, 80)});
}, true);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' || e.key === 'Enter') ulog('key', {key: e.key}); }, true);

ulog('page', {ua: navigator.userAgent, w: innerWidth, h: innerHeight, dpr: devicePixelRatio});
