// Zoom windows (in the "Size h" panel): boxes or polygons previewed with their own (finer) step; any of them can become
// a refinement zone (a zone layer appended to [[size.layers]], its transition from grad_max unless typed).
import {$, D, S, api, arcPath, boxRing, el, fmt, km, kmd, pngImage, status, wrapLon} from './util.js';
import {ALT, addLayers, bitmap, draw, flyTo, isPolar, onView, pathCopies, projectPt, redraw} from './map.js';
import {SETUP_TABS, renderLayers, schedulePreview, transitionFor} from './hsize.js';

const W = S.win = {list: [], next: 1};

function ring(w) { if (!w.box) return arcPath(w.polygon, true); const r = boxRing(w.box); r.push(r[0].slice()); return r; }
function boundsOf(w) {
  if (w.box) return w.box;
  const p = arcPath(w.polygon, true), lon = p.map((q) => q[0]), lat = p.map((q) => q[1]);
  let lo = Math.min(...lon), hi = Math.max(...lon);
  if (lo < -180) { lo += 360; hi += 360; }
  return [lo, hi, Math.max(-90, Math.min(...lat)), Math.min(90, Math.max(...lat))];
}
const nPoints = (w) => Math.round((w.bounds[1] - w.bounds[0]) / w.step) * Math.round((w.bounds[3] - w.bounds[2]) / w.step);

export function addWindow(shape, step) {
  const w = {id: W.next++, box: shape.box || null, polygon: shape.polygon || null, step: step || +$('win_step').value, show: true,
    img: null, polar: {}, polarBounds: {}, stats: null, zoneH: null, zoneW: null};       // zoneW null: from grad_max
  w.name = `window ${w.id}`;
  w.bounds = boundsOf(w);
  W.list.push(w); render();
  return previewWindow(w);
}
export async function previewWindow(w) {
  const t0 = performance.now();
  const polar = isPolar(S.view) ? [S.view] : [];
  const h = S.h.last || {};
  const r = await api('api/preview/window', {config: S.cfg, bounds: w.bounds, polygon: w.polygon, step: w.step,
    h_lo: h.h_min, h_hi: h.h_max, polar, limited: S.h.limited});
  if (r.error) return;
  w.img = await pngImage(r.png); w.polar = {}; w.polarBounds = {};
  for (const p of polar) { w.polar[p] = await pngImage(r['png_' + p]); w.polarBounds[p] = r['bounds_' + p]; }
  w.stats = r;
  if (w.zoneH === null && r.h_min) w.zoneH = +(km(r.h_min) / 2).toPrecision(3);
  const ms = performance.now() - t0;
  S.stats.loads.push({what: 'window', ms, server_s: r.seconds, points: r.shape[0] * r.shape[1]});
  render(); redraw();
  return ms;
}
const zoneW = (w) => w.zoneW ?? km(transitionFor(w.zoneH * 1000));
// the window becomes a zone layer: the layer opens in the list above, the window's outline gives way to the zone's
function toZone(w) {
  const ly = {kind: 'zone', name: w.name, h_m: w.zoneH * 1000, transition_m: zoneW(w) * 1000,
    polygon: (w.box ? boxRing(w.box) : w.polygon).map((p) => [+wrapLon(p[0]).toFixed(4), +p[1].toFixed(4)])};
  S.cfg.size.layers.push(ly);
  if (w.zoneW === null) S.h.autoT.add(ly);
  S.h.open = S.cfg.size.layers.length - 1; w.show = false;
  renderLayers(); render(); schedulePreview();
  status(`${w.name} is now refinement zone "${ly.name}" (layer ${S.cfg.size.layers.length}: h ${w.zoneH} km, transition ${zoneW(w)} km), opened above.`);
}
function render() {
  const box = $('windows'); box.replaceChildren();
  for (const w of W.list) {
    const b = w.bounds, s = w.stats;
    const num = (val, f) => el('input', {type: 'number', step: 'any', value: val, on: {change: (e) => f(+e.target.value)}});
    box.append(el('div.layer.open', {},
      el('div.layer-head', {},
        el('input', {type: 'checkbox', checked: w.show, title: 'show', on: {change: (e) => { w.show = e.target.checked; redraw(); }}}),
        el('span.layer-title', {}, el('b', {}, w.name), ' ', el('span.muted', {},
          `${w.box ? 'box' : 'polygon'} ${fmt(b[1] - b[0], 1)}° × ${fmt(b[3] - b[2], 1)}° at ${fmt(wrapLon((b[0] + b[1]) / 2), 1)}°, ${fmt((b[2] + b[3]) / 2, 1)}°`)),
        el('button.small', {on: {click: () => flyTo(wrapLon((b[0] + b[1]) / 2), (b[2] + b[3]) / 2, Math.max((b[3] - b[2]) * 111 * 0.7, (b[1] - b[0]) * 111 * Math.cos((b[2] + b[3]) / 2 * Math.PI / 180) * 0.6, 100), true)}}, 'Fly to'),
        el('button.icon', {title: 'remove', on: {click: () => { W.list.splice(W.list.indexOf(w), 1); render(); redraw(); }}}, '✕')),
      el('div.layer-form', {},
        el('label', {}, 'step, deg', num(w.step, (x) => { w.step = x; render(); })),
        el('div.muted.small', {}, `${nPoints(w).toLocaleString()} grid points`),
        el('div.row', {}, el('button', {on: {click: () => previewWindow(w)}}, 'Preview')),
        s ? el('div.stats', {}, `h ${kmd(s.h_min)}–${kmd(s.h_max)} km · nodes ≈ ${Math.round(s.n_nodes_est).toLocaleString()} · max |∇h| ${s.grad_max.toFixed(3)} · ${s.seconds.toFixed(2)} s`) : null,
        el('label', {}, 'zone h, km', num(w.zoneH ?? '', (x) => { w.zoneH = x; })),
        el('label', {}, 'zone transition, km', Object.assign(num(w.zoneW ?? '', (x) => { w.zoneW = Number.isFinite(x) && x > 0 ? x : null; }),
          {placeholder: w.zoneH ? `${zoneW(w)} (grad_max)` : 'from grad_max'})),
        el('div.row', {}, el('button', {disabled: !w.zoneH, on: {click: () => toZone(w)}}, 'Make a refinement zone')))));
  }
}
export function initWindows() {
  $('win_box').onclick = () => draw.start('box', (s) => addWindow(s));
  $('win_poly').onclick = () => draw.start('poly', (s) => addWindow(s));
  $('win_finish').onclick = () => draw.finish();
  S.h.afterPreview.push(async () => { for (const w of W.list) if (w.show && w.stats) await previewWindow(w); });
}
// window rasters are hidden while a mesh hides the h raster (6 Mesh): no polar preview then
const hShown = () => !(S.mesh && S.mesh.mode !== 'off' && !S.mesh.showH);
onView(async (v) => { if (isPolar(v) && hShown()) for (const w of W.list) if (w.show && w.stats && !w.polar[v]) await previewWindow(w); });

addLayers(15, (v) => !hShown() || !SETUP_TABS.has(S.tab) ? null : W.list.filter((w) => w.show && w.img && (!isPolar(v) || w.polar[v])).flatMap((w) =>
  bitmap(`win-${w.id}`, isPolar(v) ? w.polar[v] : w.img, v, [w.bounds[0], w.bounds[2], w.bounds[1], w.bounds[3]], w.polarBounds[v])));
addLayers(75, (v) => {
  const shown = SETUP_TABS.has(S.tab) ? W.list.filter((w) => w.show) : [];
  if (!shown.length) return null;
  const paths = shown.flatMap((w) => pathCopies(ring(w), v).map((p) => ({p, w})));
  return [new D.PathLayer({id: `win-outline-${v}`, data: paths, getPath: (d) => d.p.map((p) => projectPt(p, v, ALT.zone)),
    getColor: [0, 230, 255, 255], getWidth: 2, widthUnits: 'pixels', updateTriggers: {getPath: [W.list.length, v]}}),
  new D.TextLayer({id: `win-label-${v}`, data: shown, getPosition: (w) => projectPt([w.bounds[0], w.bounds[3]], v, ALT.zone),
    getText: (w) => w.name, getColor: [0, 230, 255, 255], getSize: 13, getTextAnchor: 'start', getAlignmentBaseline: 'bottom',
    fontFamily: 'system-ui, sans-serif', updateTriggers: {getPosition: [W.list.length, v]}})];
});
