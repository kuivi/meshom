// Panel "Coast": source (the shapefiles found, or none = the closed sphere), Antarctica, minimum island area, the coast
// drawn at the coastal h. While a mesh is drawn, the map shows the coast that mesh was built with (its run's config).
// Below it the [bathymetry] keys of the core (docs/config.md) as plain inputs; a key is written only when it is edited;
// then the depth rules by area, [[bathymetry.regions]] (a box or a polygon drawn on the map, set / at least / at most).
import {$, D, S, api, arcPath, bin, boxRing, el, fmt, km, status, wrapLon} from './util.js';
import {ALT, addLayers, draw, flyTo, pathCopies, projectFlat, projectPt, redraw, sizeOf} from './map.js';
import {preview} from './hsize.js';

const C = S.coast = {data: null, atH: false, show: true, sources: null, current: null, proj: {}, mesh: null};

function parseCoast(buf) {
  const n = new Uint32Array(buf, 0, 1)[0];
  const starts = new Uint32Array(buf, 4, n + 1);
  const ll = new Float32Array(buf, 4 + 4 * (n + 1), starts[n] * 2);
  return {n, starts, ll};
}
// the h of the coast layer: the first enabled coast_distance layer, else the smallest constant
function coastH() {
  const L = S.cfg.size.layers.filter((l) => l.enabled !== false);
  const c = L.find((l) => l.kind === 'coast_distance');
  return c ? c.h_m : Math.min(...L.filter((l) => l.kind === 'constant').map((l) => l.h_m));
}
export async function loadCoast() {
  const sp = C.atH ? `?spacing_m=${coastH()}` : '';
  const {buf, ms} = await bin('api/coast.bin' + sp);
  C.data = parseCoast(buf); C.proj = {};
  $('coast_stats').replaceChildren(el('div', {}, `${C.data.n.toLocaleString()} coast lines, ${C.data.starts[C.data.n].toLocaleString()} points` +
    (C.atH ? ` (drawn at ${km(coastH())} km)` : '')), el('div.dev', {}, `${(buf.byteLength / 1e6).toFixed(2)} MB in ${ms.toFixed(0)} ms`));
  redraw();
}
// the coast of the mesh being drawn: its run's config (info.coast); none for a mesh built without a coast; null when
// the mesh came without a config (the page's coast is drawn then)
export async function loadMeshCoast(info) {
  if (!info.config) { C.mesh = null; redraw(); return; }
  if (C.mesh && C.mesh.config === info.config) return;
  C.mesh = {config: info.config, data: info.coast ? parseCoast((await bin('api/coast.bin?of=mesh')).buf) : null, proj: {}};
  redraw();
}
async function loadSources() {
  const r = await api('api/coast/sources');
  C.sources = r.sources; C.current = r.current;
  const sel = $('coast_source'); sel.replaceChildren();
  sel.append(el('option', {value: 'none', selected: r.current === 'none'}, 'none (the closed sphere)'));
  for (const s of r.sources.filter((x) => x.usable || x.path === r.current)) {      // the rest: "coast files found"
    sel.append(el('option', {value: s.path, selected: s.path === r.current}, s.label || s.name));
  }
  const cur = r.sources.find((s) => s.path === r.current);
  $('coast_note').textContent = cur ? (cur.note || cur.label || cur.kind) : '';
  $('coast_list').textContent = r.sources.map((s) => `${s.usable ? '✓' : '·'} ${s.kind}: ${s.path}`).join('\n');
}
export function initCoast() {
  const c = S.cfg.coast;
  $('min_area').value = c.min_area_km2;
  const keepNote = () => { $('keep_note').textContent = c.keep_basins_km2 ? '' : 'Empty: only the largest connected sea is kept (FESOM2\'s rule); closed seas such as the Mediterranean become land in the run, though the h map still shows them as water.'; };
  $('keep_basins').value = c.keep_basins_km2 ?? '';
  $('keep_basins').onchange = (e) => { if (e.target.value === '') delete c.keep_basins_km2; else c.keep_basins_km2 = +e.target.value; keepNote(); S.showToml(); };
  keepNote();
  $('antarctica').value = c.antarctica || 'ice_front';
  $('antarctica').onchange = async (e) => { c.antarctica = e.target.value; await preview(); await loadCoast(); };
  $('min_area').onchange = async (e) => { c.min_area_km2 = +e.target.value; await preview(); await loadCoast(); };
  $('coast_at_h').onchange = (e) => { C.atH = e.target.checked; loadCoast(); };
  $('coast_show').onchange = (e) => { C.show = e.target.checked; redraw(); };
  $('coast_source').onchange = async (e) => {
    if (e.target.value !== C.current) { c.source = e.target.value; C.current = e.target.value; }
    const cur = C.sources.find((s) => s.path === e.target.value);
    $('coast_note').textContent = cur ? (cur.note || cur.label || cur.kind) : '';
    await preview(); await loadCoast();
  };
  initBathymetry();
  return loadSources();
}
function initBathymetry() {
  const b = S.cfg.bathymetry;
  if (!b) return;
  $('bathy_source').value = b.source === 'none' ? '' : b.source;
  $('bathy_variable').value = b.variable;
  $('bathy_min_depth').value = b.min_depth_m;
  $('bathy_depth_at').value = b.depth_at;
  $('bathy_levels').value = Array.isArray(b.levels) ? JSON.stringify(b.levels) : b.levels;
  const set = (id, key, parse) => { $(id).onchange = (e) => {
    try { b[key] = parse(e.target.value); S.showToml(); } catch (err) { status(`[bathymetry] ${key}: ${err.message}`, true); }
  }; };
  set('bathy_source', 'source', (v) => v.trim() || 'none');
  set('bathy_variable', 'variable', (v) => v.trim());
  set('bathy_min_depth', 'min_depth_m', (v) => +v);
  set('bathy_depth_at', 'depth_at', (v) => v);
  set('bathy_levels', 'levels', (v) => (v.trim().startsWith('[') ? JSON.parse(v) : v.trim()));   // a list, core2 or a file
  if (!b.regions) b.regions = [];
  const add = (mode) => draw.start(mode, (a) => {
    b.regions.push({name: `region ${b.regions.length + 1}`, ...areaOf(a), min_depth_m: 50.0});
    renderRegions(); S.showToml(); redraw();
  });
  $('bathy_region_box').onclick = () => add('box');
  $('bathy_region_poly').onclick = () => add('poly');
  $('bathy_region_finish').onclick = () => draw.finish();
  renderRegions();
}

// ------------------------------------------------------------------------------- depth rules by area (C1)
// the map's area as a region's: a box [west, east, south, north] -> [lon_min, lat_min, lon_max, lat_max] (lon_min >
// lon_max across 180), a polygon as it is
export function areaOf(a) {
  if (a.box) { const [w, e, s, n] = a.box; return {box: [+wrapLon(w).toFixed(4), +s.toFixed(4), +wrapLon(e).toFixed(4), +n.toFixed(4)]}; }
  return {polygon: a.polygon.map((p) => [+(+p[0]).toFixed(4), +(+p[1]).toFixed(4)])};
}
// a region's outline for the map (a box along its meridians and parallels)
function ringOf(r) {
  if (!r.box) return arcPath(r.polygon, true);
  const [lo0, la0, lo1, la1] = r.box, ring = boxRing([lo0, lo1 >= lo0 ? lo1 : lo1 + 360, la0, la1]);
  ring.push(ring[0].slice());
  return ring;
}
const RULES = [['depth_m', 'set to, m', 'the nodes inside get this depth'], ['min_depth_m', 'at least, m', 'nodes shallower are deepened to it'],
  ['max_depth_m', 'at most, m', 'nodes deeper are made this shallow']];
export function renderRegions() {
  const b = S.cfg.bathymetry, box = $('bathy_regions');
  if (!box || !b) return;
  box.replaceChildren(...(b.regions || []).map((r, i) => {
    const area = r.box ? `box ${fmt(r.box[0], 2)}…${fmt(r.box[2], 2)}° E, ${fmt(r.box[1], 2)}…${fmt(r.box[3], 2)}° N` : `polygon, ${r.polygon.length} corners`;
    const changed = () => { S.showToml(); redraw(); };
    return el('div.layer.open', {},
      el('div.layer-head', {}, el('input', {type: 'text', value: r.name || '', title: 'the region\'s name (shown in the write\'s record)', spellcheck: false,
        on: {change: (e) => { r.name = e.target.value; changed(); }}}),
      el('button.icon', {title: 'fly there', on: {click: () => { const p = ringOf(r), c = p[0]; flyTo(wrapLon(c[0]), c[1], 400, true); }}}, '⌖'),
      el('button.icon', {title: 'remove this rule', on: {click: () => { b.regions.splice(i, 1); renderRegions(); changed(); }}}, '✕')),
      el('div.layer-form', {}, el('div.muted.small', {}, area),
        ...RULES.map(([k, label, tip]) => el('label', {title: tip}, label, el('input', {type: 'number', step: 'any', placeholder: '–', value: r[k] ?? '',
          on: {change: (e) => { if (e.target.value === '') delete r[k]; else r[k] = +e.target.value; changed(); }}})))));
  }));
}
addLayers(71, (v) => {
  if (S.tab !== 'coast' || !S.cfg.bathymetry || !(S.cfg.bathymetry.regions || []).length) return null;
  return S.cfg.bathymetry.regions.map((r, i) => new D.PathLayer({id: `depth-region-${i}-${v}`, data: pathCopies(ringOf(r), v),
    getPath: (d) => d.map((p) => projectPt(p, v, ALT.zone)), getColor: [0, 200, 230, 255], getWidth: 2, widthUnits: 'pixels'}));
});
addLayers(60, (v) => {
  const ofMesh = S.mesh && S.mesh.mode !== 'off' && C.mesh, src = ofMesh ? C.mesh : C;
  if (!C.show || !src.data) return null;
  const d = src.data;         // data projected once per load and view: a new data object would re-upload every redraw
  const data = src.proj[v] || (src.proj[v] = {length: d.n, startIndices: d.starts,
    attributes: {getPath: {value: projectFlat(d.ll, v, ALT.coast), size: sizeOf(v, ALT.coast)}}});
  return new D.PathLayer({
    id: `coast-${ofMesh ? 'mesh' : 'page'}-${v}`, data,
    _pathType: 'open', getColor: [15, 15, 15, 255], getWidth: 1, widthUnits: 'pixels',
  });
});
