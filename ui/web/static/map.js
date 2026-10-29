// meshom web UI: the deck.gl map (GlobeView, flat and polar OrthographicView maps), layer providers, fly-to and the
// drawing tool (polygons and boxes) shared by the zone layers and the zoom windows.
import {$, D, DEG, S, arcPath, polarInv, polarXY, status, unwrapPath} from './util.js';
import {addBeat, ulog} from './log.js';

const GlobeView = D.GlobeView || D._GlobeView;
export const isPolar = (v) => v === 'north' || v === 'south';
const makeView = (v) => v === 'globe' ? new GlobeView({id: 'globe', resolution: 5}) : new D.OrthographicView({id: v, flipY: false});

function fitState(v) {
  const el = $('map'), w = el.clientWidth, h = el.clientHeight;
  if (v === 'globe') return {longitude: 20, latitude: 15, zoom: Math.log2(Math.min(w, h) / 190)};   // radius ≈ 0.43 min(w, h)
  if (v === 'flat') return {target: [0, 0, 0], zoom: Math.log2(Math.min(w / 360, h / 180) * 0.97)};
  return {target: [0, 0, 0], zoom: Math.log2(Math.min(w, h) / 2 * 0.97)};
}
export let viewState = null;

// lines are lifted above the surfaces on the globe (metres): at equal depth they z-fight with the mesh and the rasters
// mesh cells are flat chords that sag below the sphere by d²/8R (440 m for 150 km cells): their fill is lifted above
// the backdrop sphere, the edges above the fill
export const ALT = {fill: 1500, coast: 3000, graticule: 2000, zone: 4000, edges: 2500, points: 5000};
// Float32Array (lon, lat)* -> positions of view v (size 3 on the globe when lifted)
export function projectFlat(ll, v, alt = 0) {
  if (v === 'globe' && alt) {
    const out = new Float32Array(ll.length / 2 * 3);
    for (let k = 0, j = 0; k < ll.length; k += 2, j += 3) { out[j] = ll[k]; out[j + 1] = ll[k + 1]; out[j + 2] = alt; }
    return out;
  }
  if (v === 'globe' || v === 'flat') return ll;
  const out = new Float32Array(ll.length), north = v === 'north';
  for (let k = 0; k < ll.length; k += 2) { const p = polarXY(ll[k], ll[k + 1], north); out[k] = p[0]; out[k + 1] = p[1]; }
  return out;
}
export const projectPt = (p, v, alt = 0) => v === 'globe' ? [p[0], p[1], alt] : v === 'flat' ? p : polarXY(p[0], p[1], v === 'north');
export const unproject = (c, v) => (v === 'globe' || v === 'flat') ? [c[0], c[1]] : polarInv(c[0], c[1], v === 'north');
export const sizeOf = (v, alt) => v === 'globe' && alt ? 3 : 2;
// a lon/lat path as PathLayer data for view v: on the flat map with copies shifted by ±360 where it crosses ±180
export function pathCopies(path, v) {
  if (v !== 'flat') return [path];
  const lo = Math.min(...path.map((p) => p[0])), hi = Math.max(...path.map((p) => p[0]));
  const out = [path];
  if (hi > 180) out.push(path.map((p) => [p[0] - 360, p[1]]));
  if (lo < -180) out.push(path.map((p) => [p[0] + 360, p[1]]));
  return out;
}

const graticule = (() => {
  const g = [];
  for (let lon = -180; lon < 180; lon += 30) g.push({p: arcPath([[lon, -89.999], [lon, 0], [lon, 89.999]], false), am: lon === -180});
  for (let lat = -60; lat <= 60; lat += 30) g.push({p: Array.from({length: 361}, (_, i) => [i - 180, lat]), am: false});
  g.push({p: arcPath([[180, -89.999], [180, 0], [180, 89.999]], false).map((q) => [180, q[1]]), am: true});
  return g;
})();
const backdrop = (() => { const c = document.createElement('canvas'); c.width = 2; c.height = 1;
  const g = c.getContext('2d'); g.fillStyle = '#2a2f37'; g.fillRect(0, 0, 2, 1); return c; })();

// a raster on the current view: lon-lat image with bounds [w, s, e, n] (global or a window), or a polar-plane image;
// opacity < 1 dims it (a newer one is being computed)
export function bitmap(id, image, v, bounds = null, polarBounds = null, opacity = 1) {
  const globe = v === 'globe';
  const b = isPolar(v) ? (polarBounds || [-1, -1, 1, 1]) : (bounds || [-180, -90, 180, 90]);
  const props = globe ? {_imageCoordinateSystem: D.COORDINATE_SYSTEM.LNGLAT} : {};
  // layer ids carry the view: a BitmapLayer reused across views keeps its old _imageCoordinateSystem
  const L = [new D.BitmapLayer({id: `${id}-${v}`, image, bounds: b, opacity, ...props})];
  if (v === 'flat' && bounds) {
    if (bounds[2] > 180) L.push(new D.BitmapLayer({id: `${id}-${v}-w`, image, bounds: [b[0] - 360, b[1], b[2] - 360, b[3]]}));
    if (bounds[0] < -180) L.push(new D.BitmapLayer({id: `${id}-${v}-e`, image, bounds: [b[0] + 360, b[1], b[2] + 360, b[3]]}));
  }
  return L;
}

// ------------------------------------------------------------------------------------------- layer providers
// providers: [order, fn(view) -> layers]; drawn in increasing order (rasters < mesh < lines < handles)
const providers = [];
export function addLayers(order, fn) { providers.push([order, fn]); providers.sort((a, b) => a[0] - b[0]); }
// the flat map shows the world once; when the view reaches past ±180 every layer is drawn again shifted by ±360°
// (modelMatrix), so the map is continuous across the antimeridian at any pan
function flatShifts() {
  if (S.view !== 'flat' || !viewState || !viewState.target) return [];
  const half = $('map').clientWidth / 2 / 2 ** viewState.zoom, x = viewState.target[0], out = [];
  if (x + half > 180) out.push(360);
  if (x - half < -180) out.push(-360);
  return out;
}
let shiftKey = '';
function layers() {
  // polar maps keep cells out to r = 2 (37° past the equator, mesh.js): the backdrop covers that disc's square
  const v = S.view, L = [new D.BitmapLayer({id: `backdrop-${v}`, image: backdrop, bounds: isPolar(v) ? [-2, -2, 2, 2] : [-180, -90, 180, 90],
    ...(v === 'globe' ? {_imageCoordinateSystem: D.COORDINATE_SYSTEM.LNGLAT} : {})})];
  for (const [, f] of providers) { const x = f(v); if (x) L.push(...[x].flat().filter(Boolean)); }
  const shifts = flatShifts();
  shiftKey = shifts.join();
  for (const dx of shifts) {
    const m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, dx, 0, 0, 1];
    for (const l of L.slice(0, L.length)) if (!l.id.includes('@')) L.push(l.clone({id: `${l.id}@${dx}`, modelMatrix: m}));
  }
  return L;
}
addLayers(50, (v) => new D.PathLayer({
  id: `graticule-${v}`, data: graticule, getPath: (d) => d.p.map((p) => projectPt(p, v, ALT.graticule)),
  getColor: (d) => d.am ? [255, 80, 200, 220] : [255, 255, 255, 70], getWidth: (d) => d.am ? 2 : 1, widthUnits: 'pixels',
}));

let renderWaiters = [];
export const nextRender = () => new Promise((res) => renderWaiters.push(res));
const tooltips = [];
export function addTooltip(fn) { tooltips.push(fn); }
// pointer hooks of the panels (the hand editor, edit.js): hover, click, dragStart, drag, dragEnd with deck.gl's picking
// info; the drawing tool takes clicks first
const pointer = {hover: [], click: [], dragStart: [], drag: [], dragEnd: []};
export function onPointer(kind, fn) { pointer[kind].push(fn); }
const fire = (kind, info, ev) => { for (const f of pointer[kind]) f(info, ev); };
// a callback that throws inside deck.gl's frame (hover picking and the tooltip run there) stops every later frame: the
// map froze for good (e.g. a tooltip with a stale element index). Every callback of the page runs
// guarded: its error is logged and shown, and the map goes on drawing
const shown = new Set();
const guard = (name, f) => (...a) => {
  try { return f(...a); } catch (e) {
    ulog('callback-error', {name, stack: String(e && e.stack).slice(0, 800)});
    const t = `Map ${name} error: ${e && e.message}`;
    if (!shown.has(t)) { shown.add(t); status(t, true); }
    console.error(e); return null;
  }
};
// the action log (log.js): per heartbeat the map's renders, view changes and the user's wheel and pointer input on it
// (input without renders or view changes is a frozen map), the view, the controller and the layers drawn
const M = {renders: 0, views: 0, zooms: 0, pans: 0, wheels: 0, downs: 0};
const vsRec = () => viewState && {lon: +(+viewState.longitude || 0).toFixed(3), lat: +(+viewState.latitude || 0).toFixed(3),
  zoom: +(+viewState.zoom || 0).toFixed(2), ...(viewState.target ? {target: viewState.target.map((x) => +x.toFixed(4))} : {})};
let lastView = 0;
function logView() { const now = performance.now(); if (now - lastView > 1000) { lastView = now; ulog('view', vsRec()); } }
addBeat(() => {
  const ctl = deckgl.props.controller, ls = deckgl.props.layers || [], rec = {...M, view: S.view, vs: vsRec(),
    controller: typeof ctl === 'object' ? ctl : !!ctl, layers: ls.flat(Infinity).filter(Boolean).map((l) => l.id).join(' '),
    lost: !!(deckgl.getCanvas && deckgl.getCanvas() && deckgl.getCanvas().getContext('webgl2') && deckgl.getCanvas().getContext('webgl2').isContextLost())};
  for (const k in M) M[k] = 0;
  return rec;
});
export const deckgl = new D.Deck({
  parent: $('map'), views: [makeView('globe')], controller: true, layers: [],
  onViewStateChange: guard('view', ({viewState: vs, interactionState: is}) => {
    viewState = vs; deckgl.setProps({viewState}); M.views++;
    if (is) { if (is.isZooming) M.zooms++; if (is.isPanning || is.isDragging) M.pans++; }
    logView();
    if (S.view === 'flat' && flatShifts().join() !== shiftKey) redraw();
    zoomed();
  }),
  onAfterRender: () => { M.renders++; const w = renderWaiters; renderWaiters = []; w.forEach((f) => f(performance.now())); },
  onClick: guard('click', (info) => {
    ulog('map-click', {lonlat: info.coordinate && unproject(info.coordinate, S.view).map((x) => +x.toFixed(3)), layer: info.layer && info.layer.id, draw: draw.mode || undefined});
    if (draw.mode && info.coordinate) { draw.add(unproject(info.coordinate, S.view).map((x) => +x.toFixed(4))); return; }
    fire('click', info);
  }),
  onDragStart: guard('drag', (info, ev) => fire('dragStart', info, ev)),
  onDrag: guard('drag', (info, ev) => fire('drag', info, ev)),
  onDragEnd: guard('drag', (info, ev) => fire('dragEnd', info, ev)),
  onHover: guard('hover', (info) => { readout(info); fire('hover', info); }),
  getTooltip: guard('tooltip', (info) => { for (const f of tooltips) { const t = f(info); if (t) return t; } return null; }),
  onError: (err, layer) => { ulog('deck-error', {layer: layer && layer.id, stack: String(err && err.stack).slice(0, 600)}); status(`Map error${layer ? ' in ' + layer.id : ''}: ${err && err.message}`, true); console.error(err); },
});
// a lost graphics context (the GPU reset it) stops every drawing: say so instead of a silent, frozen map
deckgl.getCanvas && setTimeout(() => {
  const c = deckgl.getCanvas() || document.querySelector('#map canvas');
  if (!c) return;
  try {
    const gl = c.getContext('webgl2') || c.getContext('webgl'), x = gl && gl.getExtension('WEBGL_debug_renderer_info');
    ulog('gl', {renderer: gl && gl.getParameter(x ? x.UNMASKED_RENDERER_WEBGL : gl.RENDERER), maxTexture: gl && gl.getParameter(gl.MAX_TEXTURE_SIZE)});
  } catch (e) { ulog('gl', {error: String(e)}); }
  c.addEventListener('webglcontextrestored', () => ulog('gl-restored'));
  c.addEventListener('wheel', () => { M.wheels++; }, {passive: true, capture: true});
  c.addEventListener('pointerdown', () => { M.downs++; }, {capture: true});
  c.addEventListener('webglcontextlost', () => status('The map lost its graphics context (the GPU reset it): reload the page (Cmd+R). If this repeats, tell which view and mesh you were in.', true));
}, 0);
window.addEventListener('error', (e) => ulog('page-error-stack', {stack: String(e.error && e.error.stack).slice(0, 800)}), true);
window.addEventListener('error', (e) => status(`Page error: ${e.message} (${(e.filename || '').split('/').pop()}:${e.lineno})`, true));
window.addEventListener('unhandledrejection', (e) => ulog('rejection-stack', {stack: String(e.reason && e.reason.stack).slice(0, 800)}));
window.addEventListener('unhandledrejection', (e) => status(`Page error: ${e.reason && (e.reason.message || e.reason)}`, true));
viewState = fitState('globe'); deckgl.setProps({viewState});
export function redraw() { deckgl.setProps({layers: layers()}); }
setTimeout(() => zoomed(), 0);                     // the scale bar of the first view (after the hooks below exist)

// ------------------------------------------------------------------------------- readout and scale bar
// Under the cursor: lon, lat, then what the panels add for that point (h and its layer, a mesh cell); a scale bar
// for the centre of the view (on the globe and the polar maps the scale changes away from the centre).
const readers = [];
export function addReadout(fn) { readers.push(fn); }
const fmtLL = (lon, lat) => `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? 'N' : 'S'} ${Math.abs(lon).toFixed(2)}°${lon >= 0 ? 'E' : 'W'}`;
function readout(info) {
  const box = $('readout');
  if (!info.coordinate) { box.textContent = ''; return; }
  let [lon, lat] = unproject(info.coordinate, S.view);
  lon = ((lon + 540) % 360) - 180;
  if (!(Math.abs(lat) <= 90)) { box.textContent = ''; return; }
  box.textContent = [fmtLL(lon, lat), ...readers.map((f) => f(lon, lat, info)).filter(Boolean)].join(' · ');
}
function scalebar() {
  const ppk = pxPerKm(), box = $('scalebar');
  if (!(ppk > 0)) { box.hidden = true; return; }
  const raw = 120 / ppk, p = 10 ** Math.floor(Math.log10(raw)), km = [5, 2, 1].map((m) => m * p).find((x) => x <= raw) || p;
  box.hidden = false; box.style.width = `${Math.round(km * ppk)}px`;
  box.textContent = (km >= 1 ? `${km.toLocaleString()} km` : `${(km * 1000).toFixed(0)} m`) + (S.view === 'flat' ? ' N–S' : '');
}

const viewHooks = [];
export function onView(fn) { viewHooks.push(fn); }
export async function setView(v, state) {
  ulog('set-view', {v});
  S.view = v;
  document.querySelectorAll('[data-view]').forEach((b) => b.classList.toggle('on', b.dataset.view === v));
  viewState = {...fitState(v), ...(state || {})};
  deckgl.setProps({views: [makeView(v)], viewState});
  for (const f of viewHooks) await f(v);
  const done = nextRender(); redraw(); zoomed(); await done;
}
export function setViewState(vs) { viewState = {...viewState, ...vs}; deckgl.setProps({viewState}); if (flatShifts().join() !== shiftKey) redraw(); zoomed(); }
// callbacks after every change of the view state (pan, zoom, fly-to): e.g. the mesh edges switch with the cell size
const zoomHooks = [scalebar];
export function onZoom(fn) { zoomHooks.push(fn); }
function zoomed() { for (const f of zoomHooks) f(); }
// screen pixels per km at the centre of the view: the globe as deck.gl's GlobeView scales it (web mercator's scale at
// the centre latitude: 512 px per Earth circumference at zoom 0, divided by cos φ); the flat map north-south; polar
// maps the stereographic scale at the centre's colatitude
const KM_ROUND = 2 * Math.PI * 6371;
export function pxPerKm(v = S.view) {
  if (!viewState) return 0;
  const z = 2 ** viewState.zoom;
  if (v === 'globe') return z * 512 / (KM_ROUND * Math.cos(Math.min(Math.abs(viewState.latitude), 85) * DEG));
  if (v === 'flat') return z / 111.2;
  const [, lat] = unproject(viewState.target, v), colat = (v === 'north' ? 90 - lat : 90 + lat) * DEG;
  return z * 0.5 / Math.cos(colat / 2) ** 2 / 6371;
}
export const getViewState = () => viewState;
// the point at the centre of the view (lon, lat)
export function viewCentre() {
  return S.view === 'globe' ? [viewState.longitude, viewState.latitude] : unproject(viewState.target, S.view);
}
// centre the current view on (lon, lat) so that a circle of radius_km fills about the smaller half of the map; mark:
// leave a ring there (the "Fly to" buttons), until the next marked fly-to or Esc
export let flyMark = null;
export function clearFlyMark() { if (flyMark) { flyMark = null; redraw(); } }
export function flyTo(lon, lat, radiusKm = 700, mark = false) {
  if (mark) flyMark = [lon, lat];
  const el = $('map'), m = 0.45 * Math.min(el.clientWidth, el.clientHeight);
  const v = S.view;
  if (v === 'globe') setViewState({longitude: lon, latitude: lat, zoom: Math.log2(m * KM_ROUND * Math.cos(Math.min(Math.abs(lat), 85) * DEG) / (512 * radiusKm))});
  else if (v === 'flat') setViewState({target: [lon, lat, 0], zoom: Math.log2(m * 111.2 / radiusKm)});
  else {
    const colat = (v === 'north' ? 90 - lat : 90 + lat) * DEG, scale = 0.5 / Math.cos(colat / 2) ** 2 / 6371;
    setViewState({target: [...polarXY(lon, lat, v === 'north'), 0], zoom: Math.log2(m / (radiusKm * scale))});
  }
  redraw();
}

// --------------------------------------------------------------------------------------------- drawing tool
// draw.start('poly' | 'box', onDone(points), onAbandon?): clicks add vertices; a box is done after two clicks, a
// polygon by draw.finish(); draw.cancel() (Esc) or a finish with too few points drops it and calls onAbandon. The
// points in progress are drawn by the provider below.
export const draw = {
  mode: null, pts: [], done: null, abandon: null, onAdd: null,
  // mode box (2 clicks), quad (4), poly (Finish closes it), line (Finish or Enter ends it; open); onAdd(pts) after each click
  start(mode, done, abandon = null, onAdd = null) {
    this.cancel(false);
    this.mode = mode; this.pts = []; this.done = done; this.abandon = abandon; this.onAdd = onAdd;
    deckgl.setProps({controller: {doubleClickZoom: false}});   // a double click would zoom and swallow the clicks
    status(mode === 'box' ? 'Box: click two opposite corners on the map (Esc cancels)'
      : mode === 'quad' ? 'Block: click its four corners on the map, counter-clockwise: the block lies on the left of each side (Esc cancels)'
        : mode === 'line' ? 'Line: click its points on the map, then Finish or Enter (Esc cancels)'
        : 'Polygon: click its corners on the map, then Finish (Esc cancels)', false, true); redraw();
  },
  add(p) {
    this.pts.push(p);
    if ((this.mode === 'box' && this.pts.length === 2) || (this.mode === 'quad' && this.pts.length === 4)) return this.finish();
    if (this.onAdd) this.onAdd(this.pts);
    redraw();
  },
  finish() {
    const pts = this.pts, mode = this.mode, done = this.done, abandon = this.abandon;
    this.cancel(false);
    if (mode === 'box' && pts.length === 2) {
      // the shorter way round in longitude: a box across ±180 gets east > 180
      const d = ((pts[1][0] - pts[0][0] + 540) % 360) - 180, w = d >= 0 ? pts[0][0] : pts[0][0] + d;
      done({box: [w, w + Math.abs(d), Math.min(pts[0][1], pts[1][1]), Math.max(pts[0][1], pts[1][1])]});
    } else if (mode === 'poly' && pts.length >= 3) done({polygon: pts});
    else if (mode === 'quad' && pts.length === 4) done({corners: pts});
    else if (mode === 'line' && pts.length >= 2) done({line: pts});
    else if (abandon) abandon();
  },
  cancel(abandoned = true) {
    const abandon = this.abandon, was = this.mode;
    this.mode = null; this.pts = []; this.done = null; this.abandon = null; this.onAdd = null;
    deckgl.setProps({controller: true}); status(''); redraw();
    if (abandoned && was && abandon) abandon();
  },
};
addLayers(90, (v) => {
  if (!draw.mode || !draw.pts.length) return null;
  const path = draw.mode === 'box' ? unwrapPath(draw.pts.map((p) => p.slice())) : arcPath(draw.pts, false);
  return [new D.PathLayer({id: `draw-path-${v}`, data: pathCopies(path, v), getPath: (d) => d.map((p) => projectPt(p, v, ALT.zone)),
    getColor: [255, 255, 0, 255], getWidth: 2, widthUnits: 'pixels'}),
  new D.ScatterplotLayer({id: `draw-pts-${v}`, data: draw.pts.slice(), getPosition: (p) => projectPt(p, v, ALT.zone), getRadius: 5,
    radiusUnits: 'pixels', getFillColor: [255, 255, 0]})];
});
window.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (draw.mode === 'line' || draw.mode === 'poly') && !(e.target.closest && e.target.closest('input, textarea, select'))) { e.preventDefault(); draw.finish(); return; }
  if (e.key !== 'Escape') return;
  if (draw.mode) draw.cancel(); else if (flyMark) { flyMark = null; redraw(); }
});
addLayers(88, (v) => flyMark ? new D.ScatterplotLayer({id: `fly-mark-${v}`, data: [flyMark], getPosition: (p) => projectPt(p, v, ALT.points),
  getRadius: 11, radiusUnits: 'pixels', filled: false, stroked: true, getLineColor: [255, 214, 0], lineWidthUnits: 'pixels', getLineWidth: 3}) : null);

// frame times while the view moves (globe: longitude; maps: pan), for the headless measurements
export async function rotate(n = 60, step = 3) {
  const out = []; let t = performance.now();
  for (let i = 0; i < n; i++) {
    viewState = S.view === 'globe' ? {...viewState, longitude: viewState.longitude + step}
      : {...viewState, target: [viewState.target[0] + (S.view === 'flat' ? step : 0.01), viewState.target[1], 0]};
    const done = nextRender(); deckgl.setProps({viewState}); const t1 = await done;
    out.push(t1 - t); t = t1;
  }
  S.stats.frames.push({view: S.view, ms: out});
  return out;
}

// deck.gl's own frame counter, to compare GPUs (it shows ~0 fps while nothing moves)
setInterval(() => { const m = deckgl.metrics; $('fps').textContent = `${Math.round(m.fps)} fps · `; }, 1000);
