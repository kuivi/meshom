// Floating windows over the map: a title bar (drag it to move the window), a close button, the corner to resize
// (CSS resize), several at once, the one clicked comes to the front, Escape closes the front one (unless a shape is
// being drawn on the map: Escape cancels that). Position and size are remembered per browser (localStorage
// 'meshom.windows'). A window is an extra view of a panel's data: its module defines it (defineWin: a title and a
// build function over the module's data) and calls refreshWin after its data changed; the rebuild keeps the
// filter box's focus and the scroll position. Not to be confused with the zoom windows of "Size h" (windows.js).
import {$, S, el} from './util.js';

const KEY = 'meshom.windows';
const DEFS = new Map(), OPEN = new Map();          // id -> {title, build, width, height, open}; id -> {w, ro}
let front = 30;                                     // z-index of the front window (over the panel and the map)

function saved() { try { return JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch (e) { return {}; } }
function remember(id, w) {
  if (!w.isConnected || !w.offsetWidth) return;     // removed (closing): keep the last size
  try {
    const all = saved();
    all[id] = {x: parseFloat(w.style.left), y: parseFloat(w.style.top), w: w.offsetWidth, h: w.offsetHeight};
    localStorage.setItem(KEY, JSON.stringify(all));
  } catch (e) { /* private window: not remembered */ }
}
// on screen: the window no larger than the viewport, its title bar reachable
function clamp(x, y, w, h) {
  const vw = innerWidth, vh = innerHeight;
  w = Math.min(w, vw - 8); h = Math.min(h, vh - 8);
  return {x: Math.max(0, Math.min(x, vw - 120)), y: Math.max(0, Math.min(y, vh - 40)), w, h};
}
function place(id, d) {
  const p = saved()[id];
  if (p && p.w) return clamp(p.x, p.y, p.w, p.h);
  if (d.atRight) return clamp(innerWidth - d.width - 16, 56, d.width, d.height);      // a tool used with the map: at the right edge
  const right = $('panel') ? $('panel').getBoundingClientRect().right : 0, k = OPEN.size;   // cascade right of the panel
  return clamp(right + 24 + 28 * k, 56 + 28 * k, d.width, d.height);
}
const toFront = (w) => { w.style.zIndex = ++front; };

function dragBy(bar, w, id) {
  bar.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 || e.target.closest('button, input, select')) return;
    e.preventDefault();
    const dx = e.clientX - parseFloat(w.style.left), dy = e.clientY - parseFloat(w.style.top);
    bar.setPointerCapture(e.pointerId);
    const move = (ev) => {
      const c = clamp(ev.clientX - dx, ev.clientY - dy, w.offsetWidth, w.offsetHeight);
      w.style.left = `${c.x}px`; w.style.top = `${c.y}px`;
    };
    const up = () => {
      bar.removeEventListener('pointermove', move); bar.removeEventListener('pointerup', up); bar.removeEventListener('pointercancel', up);
      remember(id, w);
    };
    bar.addEventListener('pointermove', move); bar.addEventListener('pointerup', up); bar.addEventListener('pointercancel', up);
  });
}

// id: the window's name (one window per id); title: text or a function of nothing; build: () => Node from the
// module's current data; open: called when the window opens (to fetch fresh data, which then calls refreshWin);
// atRight: first placed at the right edge of the screen (the map's middle stays free to click on)
export function defineWin(id, {title, build, width = 640, height = 420, open = null, atRight = false}) {
  DEFS.set(id, {title, build, width, height, open, atRight});
}
export const isOpen = (id) => OPEN.has(id);
export const openWins = () => [...OPEN.keys()];

export function openWin(id) {
  if (OPEN.has(id)) { toFront(OPEN.get(id).w); refreshWin(id); return OPEN.get(id).w; }
  const d = DEFS.get(id), r = place(id, d);
  const bar = el('div.fwin-bar', {title: 'drag to move'}, el('span.fwin-title'),
    el('button.icon.fwin-close', {title: 'close (Esc)', 'aria-label': 'close', on: {click: () => closeWin(id)}}, '✕'));
  const w = el('div.fwin', {role: 'dialog', style: {left: `${r.x}px`, top: `${r.y}px`, width: `${r.w}px`, height: `${r.h}px`}},
    bar, el('div.fwin-body'));
  w.dataset.win = id;
  w.addEventListener('pointerdown', () => toFront(w), true);
  dragBy(bar, w, id);
  document.body.append(w);
  const ro = new ResizeObserver(() => remember(id, w));          // the size after a drag of the corner
  ro.observe(w);
  OPEN.set(id, {w, ro});
  toFront(w); refreshWin(id);
  if (d.open) d.open();
  return w;
}
export function closeWin(id) {
  const o = OPEN.get(id);
  if (!o) return;
  o.ro.disconnect(); o.w.remove(); OPEN.delete(id);
}
// rebuild a window's content from its module's data (no-op when it is closed)
export function refreshWin(id) {
  const o = OPEN.get(id);
  if (!o) return;
  const d = DEFS.get(id), body = o.w.querySelector('.fwin-body');
  const t = typeof d.title === 'function' ? d.title() : d.title;
  o.w.querySelector('.fwin-title').textContent = t; o.w.setAttribute('aria-label', t);
  const a = document.activeElement;
  const keep = a && body.contains(a) && a.dataset.keep ? {k: a.dataset.keep, s: a.selectionStart, e: a.selectionEnd} : null;
  const scrolls = [...body.querySelectorAll('.stbl-scroll')].map((e) => e.scrollTop), top = body.scrollTop;
  body.replaceChildren(d.build());
  body.scrollTop = top;
  body.querySelectorAll('.stbl-scroll').forEach((e, i) => { e.scrollTop = scrolls[i] || 0; });
  if (keep) {
    const n = body.querySelector(`[data-keep="${keep.k}"]`);
    if (n) { n.focus(); try { n.setSelectionRange(keep.s, keep.e); } catch (e) { /* not a text field */ } }
  }
}

// Escape: the front window closes; the map's own Escape (cancel a drawing) goes first
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape' || !OPEN.size || (S.api && S.api.draw && S.api.draw.mode)) return;
  const [id] = [...OPEN.entries()].reduce((a, b) => (+b[1].w.style.zIndex > +a[1].w.style.zIndex ? b : a));
  closeWin(id);
  e.stopPropagation();                        // the map keeps its fly-to ring
}, true);
