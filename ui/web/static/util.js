// meshom web UI: shared state, server I/O, DOM helpers, colour maps and sphere geometry.
import {addBeat, ulog} from './log.js';
export const D = window.deck;
export const DEG = Math.PI / 180;
export const $ = (id) => document.getElementById(id);

export const S = {
  cfg: null, loadedToml: '', view: 'globe', tab: 'h', layerColours: [],
  stats: {loads: [], frames: []},
};
window.__meshom = S;   // used by the headless measurements (ui/web/measure.py)

// ----------------------------------------------------------------------------------------- busy: what computes
// Every server call is named in words; one that takes longer than BUSY_MS shows "Computing <what> … 12 s (last time
// 23 s)" over the map until it returns (busyWhat names it). The last duration of each kind is remembered until the page reloads.
const BUSY_MS = 500;
const BUSY = {items: new Map(), last: {}, timer: null, next: 0};
function busyWhat(path, body) {
  const p = path.split('?')[0].replace(/^\/?/, '');
  if (p === 'api/preview') return body && body.limited ? "Computing the run's h (gradient limiter)" : 'Computing the h preview';
  return {'api/preview/window': 'Computing the zoom window', 'api/rho': 'Computing the ρ_c map', 'api/mesh/load': 'Opening the mesh',
    'api/mesh/info': 'Reading the mesh', 'api/mesh/mesh.bin': 'Loading the mesh elements', 'api/mesh/overview.png': 'Drawing the overview',
    'api/mesh/hist': 'Counting the histograms', 'api/coast.bin': 'Loading the coastline', 'api/runs': 'Reading the run list', 'api/dirs': 'Reading the folder', 'api/profile': 'Computing the profile', 'api/runs/roots': 'Adding the folder',
    'api/run': 'Writing the run folder', 'api/methods': 'Reading the methods', 'api/refs': 'Reading the reference meshes',
    'api/config/open': "Opening the run's config", 'api/compare': 'Reading the quality records', 'api/mesh/edit': 'Writing the edited mesh'}[p] || null;
}
function busyTick() {
  const box = $('busy'), now = performance.now(), shown = [...BUSY.items.values()].filter((b) => now - b.t0 > BUSY_MS);
  box.hidden = !shown.length;
  box.textContent = shown.map((b) => `${b.what} … ${((now - b.t0) / 1000).toFixed(0)} s` +
    (BUSY.last[b.what] > 2 ? ` (last time ${BUSY.last[b.what].toFixed(0)} s)` : '')).join('\n');
  if (!BUSY.items.size) { clearInterval(BUSY.timer); BUSY.timer = null; }
}
addBeat(() => ({busy: [...BUSY.items.values()].map((b) => `${b.what} ${((performance.now() - b.t0) / 1000).toFixed(0)} s`)}));
async function busy(what, f) {
  if (!what) return f();
  const id = BUSY.next++;
  BUSY.items.set(id, {what, t0: performance.now()});
  if (!BUSY.timer) BUSY.timer = setInterval(busyTick, 250);
  try { return await f(); } finally {
    const b = BUSY.items.get(id); BUSY.items.delete(id);
    BUSY.last[what] = (performance.now() - b.t0) / 1000; busyTick();
  }
}

// --------------------------------------------------------------------------------------------------- server I/O
export async function api(path, body) {
  return busy(busyWhat(path, body), async () => {
    const r = await fetch(path, body === undefined ? {} : {method: 'POST', body: JSON.stringify(body)});
    const j = await r.json();
    if (j && j.error) status(`The server could not do this: ${j.error}`, true);
    return j;
  });
}
export async function bin(path) {
  return busy(busyWhat(path), async () => {
    const t0 = performance.now();
    const r = await fetch(path); const buf = await r.arrayBuffer();
    return {buf, ms: performance.now() - t0};
  });
}
export function loadImage(src) {
  return busy(busyWhat(src), () => new Promise((res) => { const i = new Image(); i.onload = () => res(i); i.src = src; }));
}
// a deflated base64 array of the server (ui/web/server.py readout_grid) as a typed array
export async function inflate(b64, T) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const out = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer();
  return new T(out);
}
export const pngImage = (b64) => loadImage('data:image/png;base64,' + b64);

// a message for the user: in the footer and, while it matters, over the map (errors and sticky hints stay until the
// next message; the others fade after NOTICE_MS)
const NOTICE_MS = 6000;
let noticeTimer = null;
export function status(t, err = false, sticky = false) {
  if (t) ulog(err ? 'error' : 'status', {text: t});
  const s = $('status'); s.textContent = t; s.classList.toggle('err', err);
  const n = $('notice'); n.textContent = t; n.hidden = !t; n.classList.toggle('err', err);
  clearTimeout(noticeTimer);
  if (t && !err && !sticky) noticeTimer = setTimeout(() => { n.hidden = true; s.textContent = ''; }, NOTICE_MS);
}

// ---------------------------------------------------------------------------------------------------- DOM helpers
// el('div.cls#id', {attrs, on: {click}}, ...children)
export function el(spec, props = {}, ...kids) {
  const m = spec.match(/^([a-z0-9]+)?((?:[.#][\w-]+)*)$/i);
  const e = document.createElement(m[1] || 'div');
  for (const p of (m[2] || '').match(/[.#][\w-]+/g) || []) {
    if (p[0] === '.') e.classList.add(p.slice(1)); else e.id = p.slice(1);
  }
  for (const [k, v] of Object.entries(props || {})) {
    if (k === 'on') for (const [ev, f] of Object.entries(v)) e.addEventListener(ev, f);
    else if (k === 'style') Object.assign(e.style, v);
    else if (k in e && k !== 'list') e[k] = v; else e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k !== null && k !== undefined && k !== false) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
}
export const km = (m) => +(m / 1000).toPrecision(6);          // form values: what the user typed comes back
export const kmd = (m) => +(m / 1000).toPrecision(3);         // computed numbers shown to the user
export const fmt = (x, d = 3) => (x === null || x === undefined) ? '–' : (+x).toFixed(d);
export const rgbCss = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

// ------------------------------------------------------------------------------------------------------ colours
const RDYLBU_R = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'];
export const GRADIENTS = {
  viridis: ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6ece58', '#b5de2b', '#fde725'],
  rdylbu_r: RDYLBU_R,
  rdbu_r: ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f'],
  // depth: shallow light, deep dark; the 9 stops of matplotlib's YlGnBu (the server's overview raster uses it)
  ylgnbu: ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58'],
};
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
function lut(stops) {
  const c = stops.map(hex), out = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (c.length - 1), k = Math.min(Math.floor(t), c.length - 2), f = t - k;
    for (let j = 0; j < 3; j++) out[4 * i + j] = Math.round(c[k][j] * (1 - f) + c[k + 1][j] * f);
    out[4 * i + 3] = 255;
  }
  return out;
}
export const LUT_Q = lut(RDYLBU_R), LUT_DEPTH = lut(GRADIENTS.ylgnbu), LUT_SIZE = lut(GRADIENTS.viridis);
export const TRI_RGB = [240, 140, 30], QUAD_RGB = [120, 140, 170];
// per-element colour of a mesh field: q in [0, 1] (max angle scaled to its gate), size ratio in [1, 2], kind,
// depth in [0, hi] m (the element's mean node depth), size_km in [lo, hi] km on a log scale (the mean edge)
export function elemColour(field, angle, ratio, isQuad, out, o, val = 0, lo = 0, hi = 1) {
  if (field === 'kind') { const c = isQuad ? QUAD_RGB : TRI_RGB; out[o] = c[0]; out[o + 1] = c[1]; out[o + 2] = c[2]; out[o + 3] = 255; return; }
  const t = field === 'q' ? (isQuad ? (angle - 90) / 60 : (angle - 60) / 60) : field === 'depth' ? val / hi
    : field === 'size_km' ? Math.log(val / lo) / Math.log(hi / lo) : ratio - 1;
  const i = 4 * Math.max(0, Math.min(255, Math.round(t * 255))), L = field === 'depth' ? LUT_DEPTH : field === 'size_km' ? LUT_SIZE : LUT_Q;
  out[o] = L[i]; out[o + 1] = L[i + 1]; out[o + 2] = L[i + 2]; out[o + 3] = 255;
}
export const IRR_COLOURS = {1: [31, 79, 209], 2: [26, 152, 80], 3: [156, 39, 176]};   // as meshom.figures

// ----------------------------------------------------------------------------------------------- sphere geometry
export function xyz(p) { const l = p[0] * DEG, f = p[1] * DEG; return [Math.cos(f) * Math.cos(l), Math.cos(f) * Math.sin(l), Math.sin(f)]; }
// polar stereographic, plane tangent at the pole, the equator at r = 1 (same as server.polar_xy)
export function polarXY(lon, lat, north) {
  const colat = (north ? 90 - lat : 90 + lat) * DEG;
  const r = Math.min(Math.tan(colat / 2), 4.0);
  return [r * Math.sin(lon * DEG), north ? -r * Math.cos(lon * DEG) : r * Math.cos(lon * DEG)];
}
export function polarInv(x, y, north) {
  const r = Math.hypot(x, y), colat = 2 * Math.atan(r) / DEG;
  const lon = north ? Math.atan2(x, -y) / DEG : Math.atan2(x, y) / DEG;
  return [lon, north ? 90 - colat : colat - 90];
}
// great-circle densified path through lon/lat vertices, longitudes kept continuous (no jump at ±180)
export function arcPath(pts, closed) {
  const out = [], n = pts.length;
  for (let i = 0; i < (closed ? n : n - 1); i++) {
    const a = xyz(pts[i]), b = xyz(pts[(i + 1) % n]);
    const w = Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2])));
    const steps = Math.max(1, Math.ceil(w / DEG));
    for (let s = 0; s < steps; s++) {
      const t = s / steps, sa = Math.sin((1 - t) * w), sb = Math.sin(t * w), sw = Math.sin(w) || 1;
      const v = w < 1e-9 ? a : [0, 1, 2].map((k) => (sa * a[k] + sb * b[k]) / sw);
      out.push([Math.atan2(v[1], v[0]) / DEG, Math.asin(Math.max(-1, Math.min(1, v[2]))) / DEG]);
    }
  }
  if (n) out.push(closed ? pts[0].slice() : pts[n - 1].slice());
  unwrapPath(out);
  return out;
}
export function unwrapPath(out) {
  for (let i = 1; i < out.length; i++) {
    while (out[i][0] - out[i - 1][0] > 180) out[i][0] -= 360;
    while (out[i][0] - out[i - 1][0] < -180) out[i][0] += 360;
  }
  return out;
}
// a lon-lat box (bounds w, e, s, n with e > w) as a closed ring densified along the parallels
export function boxRing(b) {
  const [w, e, s, n] = b, k = Math.max(2, Math.ceil((e - w) / 2)), out = [];
  for (let i = 0; i <= k; i++) out.push([w + (e - w) * i / k, s]);
  for (let i = 0; i <= k; i++) out.push([e - (e - w) * i / k, n]);
  return out;
}
export const wrapLon = (x) => ((x + 540) % 360) - 180;
