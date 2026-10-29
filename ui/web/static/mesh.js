// Panel "Mesh": overview raster (worst value per pixel), the worst elements, a zoom window or the full mesh from the
// indexed binary payload (ui/web/payload.py), irregular nodes, triangles, the gates as a table, histograms, the
// worst-element list. Opening a mesh (Runs, Reference meshes) shows its overview with the whole mesh in view.
import {$, D, IRR_COLOURS, S, api, arcPath, bin, el, elemColour, fmt, loadImage, status} from './util.js';
import {ALT, addLayers, addTooltip, flyTo, isPolar, nextRender, onView, onZoom, pathCopies, projectPt, pxPerKm, redraw, setView, sizeOf, viewCentre} from './map.js';
import {polarXY} from './util.js';
import {loadBlocks} from './blocks.js';
import {SETUP_TABS, ensurePolarH, legend} from './hsize.js';
import {loadMeshCoast} from './coastpanel.js';
import {defineWin, openWin, refreshWin} from './floatwin.js';
import {sortTable} from './sorttable.js';

// callbacks after loadMesh put new elements in X.data (or none): the hand editor (edit.js) starts over on them
const dataHooks = [];
export function onMeshData(fn) { dataHooks.push(fn); }
// "Full" draws every element; a mesh loaded while it is on switches to the overview above this many (the 10^7-element
// reference meshes: 0.5 GB payloads; README "Reference meshes")
const FULL_MAX = 2000000;
const X = S.mesh = {mode: 'off', field: 'q', data: null, geo: {}, colours: {}, overview: {}, info: null, irr: null, irrShow: {1: true, 2: true, 3: true},
  showIrr: true, tris: null, showTris: false, highlight: null, rKm: 700, showH: false, showEdges: true, cellKm: null, edgesOn: false};

// ------------------------------------------------------------------------------------------- edges and legend
// Rule: element edges are drawn when the "edges" box is on and a typical cell (the median mean edge length of the
// loaded elements, great-circle) covers at least EDGE_MIN_PX screen pixels at the centre of the view; below that the
// lines would merge into a grey veil, so they are skipped (zoom in, or use a zoom window).
export const EDGE_MIN_PX = 6;
function cellKm(m) {
  const T = m.nt, E = T + m.nq, n = Math.min(E, 4000), out = [];
  const R = 6371, d = Math.PI / 180;
  for (let i = 0; i < n; i++) {
    const e = Math.floor(i * E / n), k = e < T ? 3 : 4, ids = e < T ? m.tris.subarray(3 * e, 3 * e + 3) : m.quads.subarray(4 * (e - T), 4 * (e - T) + 4);
    let s = 0;
    for (let j = 0; j < k; j++) {
      const a = ids[j], b = ids[(j + 1) % k], la1 = m.ll[2 * a + 1] * d, la2 = m.ll[2 * b + 1] * d, dl = (m.ll[2 * b] - m.ll[2 * a]) * d;
      s += R * Math.acos(Math.max(-1, Math.min(1, Math.sin(la1) * Math.sin(la2) + Math.cos(la1) * Math.cos(la2) * Math.cos(dl))));
    }
    out.push(s / k);
  }
  out.sort((a, b) => a - b);
  return out.length ? out[out.length >> 1] : null;
}
// the same rule thins the irregular-node dots: drawn when cells cover EDGE_MIN_PX, else they hide the mesh
const cellPx = () => (X.cellKm || (X.info && X.info.cell_km) || 0) * pxPerKm();
const edgesVisible = () => X.showEdges && cellPx() >= EDGE_MIN_PX;
const irrVisible = () => cellPx() >= EDGE_MIN_PX;
onZoom(() => {
  const e = edgesVisible(), i = irrVisible();
  if ((X.data && e !== X.edgesOn) || (X.irr && i !== X.irrOn)) { X.irrOn = i; redraw(); }
  if (`${e}${i}` !== X.legendKey) { X.legendKey = `${e}${i}`; legend(); }       // after view switches too
});
// what the map shows, in words (first lines of the legend)
X.describe = () => {
  const lines = [];
  const what = {q: 'max angle / gate (dark blue = ideal, red = at the hard gate)', size_ratio: 'size ratio to the neighbours (blue = 1, red = 2)',
    kind: 'kind (grey-blue quads, orange triangles)', depth: "depth, the mean of the element's nodes (light = shallow, dark blue = deep)",
    size_km: 'cell size, the mean edge of each element (log scale: dark = small, bright = large)'}[X.field];
  const n = X.data ? (X.data.nt + X.data.nq).toLocaleString() : '';
  if (X.mode === 'overview') lines.push(`raster: the ${{kind: 'worst kind', depth: 'deepest element', size_km: 'largest element'}[X.field] || 'worst value'} per pixel, ${what}`);
  else lines.push(`${X.mode === 'full' ? 'all' : X.mode === 'worst' ? 'the worst' : 'zoom window:'} ${n} cells coloured by ${what}` +
    (X.mode === 'worst' ? ', over the dimmed overview raster' : ''));
  if (X.data) {
    const px = cellPx();
    lines.push(!X.showEdges ? 'edges: off (6 Mesh, "edges")' : px >= EDGE_MIN_PX ? `edges: thin black lines (cells ≥ ${EDGE_MIN_PX} px here)`
      : `edges: hidden, cells < ${EDGE_MIN_PX} px here (zoom in)`);
  }
  if (X.showIrr && X.irr && X.irr.n) lines.push(X.irrOn ? 'dots: irregular interior nodes' : `irregular nodes: hidden, cells < ${EDGE_MIN_PX} px here (zoom in)`);
  lines.push(X.showH ? 'under the cells: the h raster' : 'dark ground: no cells there (land, or outside the mesh)');
  if (S.coast && S.coast.show) lines.push(X.info && X.info.config ? (X.info.coast ? `black line: the coast this mesh was built with (islands ≥ ${X.info.coast.min_area_km2} km²)` : 'no coast: this mesh was built without one')
    : "black line: the page's coast (this mesh came without a config)");
  if (X.highlight) lines.push('magenta ring: the element picked in the worst list');
  if (S.blocks && S.blocks.onMesh && S.blocks.data && S.blocks.data.blocks.length)
    lines.push('white outlines: the m14 blocks of this run (grey: dropped), yellow dots: their corners, B#: seeds (the Blocks tab lists them)');
  if (S.edit && S.edit.on && X.mode === 'window') lines.push('editing by hand: white dots = nodes you can drag (not coast, boundary or the window\'s rim); ' +
    'white outlines = elements you changed, coloured by their quality recomputed in the page');
  if (X.pair && X.pair.repair.polygon) lines.push('white line: the remeshed zone');
  if (S.blocks && S.blocks.show && S.blocks.data && S.blocks.data.blocks.length && S.tab === 'blocks')
    lines.push('white discs B0, B1, …: m14 block seeds; white lines: block outlines, yellow: corners');
  lines.push('faint lines: graticule every 30°; magenta line: the antimeridian (±180°)');
  return lines.join('\n');
};

// ----------------------------------------------------------------------------------------- payload decoding
export function decodeMesh(buf) {
  const dv = new DataView(buf);
  const hl = dv.getUint32(4, true), h = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hl)));
  let o = 8 + hl;
  const take = (T, n) => { const a = new T(buf, o, n); o += a.byteLength; return a; };
  const m = {h, nn: h.n_nodes, nt: h.n_tris, nq: h.n_quads};
  m.ll = take(Float32Array, 2 * m.nn); m.tris = take(Uint32Array, 3 * m.nt); m.quads = take(Uint32Array, 4 * m.nq);
  m.eid = h.subset ? take(Uint32Array, m.nt + m.nq) : null;
  m.f = {}; for (const f of h.fields) m.f[f] = take(Float32Array, m.nt + m.nq);
  if (h.node_fields) { m.nid = take(Uint32Array, m.nn); m.nflag = take(Uint8Array, m.nn); }   // the hand editor's
  return m;
}
// per-element vertices (flat colours) with pre-triangulated indices: deck.gl then triangulates nothing
function expand(m) {
  const T = m.nt, Q = m.nq, E = T + Q, V = 3 * T + 4 * Q;
  const vnode = new Uint32Array(V); vnode.set(m.tris, 0); vnode.set(m.quads, 3 * T);
  const starts = new Uint32Array(E + 1), idx = new Uint32Array(3 * T + 6 * Q);
  for (let k = 0; k < T; k++) { starts[k] = 3 * k; idx[3 * k] = 3 * k; idx[3 * k + 1] = 3 * k + 1; idx[3 * k + 2] = 3 * k + 2; }
  for (let k = 0; k <= Q; k++) starts[T + k] = 3 * T + 4 * k;
  for (let k = 0, j = 3 * T; k < Q; k++, j += 6) {
    const s = 3 * T + 4 * k; idx[j] = s; idx[j + 1] = s + 1; idx[j + 2] = s + 2; idx[j + 3] = s; idx[j + 4] = s + 2; idx[j + 5] = s + 3;
  }
  return {E, V, vnode, starts, idx};
}
// positions of view v: the globe takes lon/lat as they are (deck projects every vertex, the triangles come from the
// indices), polar maps project the nodes once, the flat map unwraps each element and copies the ones across ±180
function geometry(m, v, cache = X.geo) {
  if (cache[v]) return cache[v];
  const t0 = performance.now(), G = geometry_(m, v);
  S.stats.loads.push({what: 'geometry', view: v, n: G.length, ms: performance.now() - t0});
  return (cache[v] = G);
}
function geometry_(m, v) {
  const g = m.g || (m.g = expand(m)), {E, V, vnode, starts, idx} = g, ll = m.ll;
  const pos = new Float32Array(2 * V);
  if (isPolar(v)) {
    const P = new Float32Array(2 * m.nn), north = v === 'north';
    for (let i = 0; i < m.nn; i++) { const p = polarXY(ll[2 * i], ll[2 * i + 1], north); P[2 * i] = p[0]; P[2 * i + 1] = p[1]; }
    for (let k = 0; k < V; k++) { pos[2 * k] = P[2 * vnode[k]]; pos[2 * k + 1] = P[2 * vnode[k] + 1]; }
    // cells reaching past r = 2 (37° beyond the equator) are collapsed to a point: polarXY clamps r at 4, so a cell
    // around the far pole would become a disc over the whole map (it showed through the holes, e.g. over Antarctica)
    for (let e = 0; e < E; e++) {
      const s = starts[e], t = starts[e + 1];
      let far = false;
      for (let k = s; k < t; k++) if (pos[2 * k] * pos[2 * k] + pos[2 * k + 1] * pos[2 * k + 1] > 4) { far = true; break; }
      if (far) for (let k = s + 1; k < t; k++) { pos[2 * k] = pos[2 * s]; pos[2 * k + 1] = pos[2 * s + 1]; }
    }
    return ({length: E, starts, pos, idx, eOf: null, V});
  }
  for (let k = 0; k < V; k++) { pos[2 * k] = ll[2 * vnode[k]]; pos[2 * k + 1] = ll[2 * vnode[k] + 1]; }
  if (v === 'globe') return ({length: E, starts, pos, idx, eOf: null, V});
  const cross = [];
  for (let e = 0; e < E; e++) {
    const s = starts[e], t = starts[e + 1];
    let ref = NaN;
    for (let k = s; k < t; k++) if (Math.abs(pos[2 * k + 1]) < 89.9999) { ref = pos[2 * k]; break; }
    let sum = 0, cnt = 0, lo = 1e9, hi = -1e9, pole = false;
    for (let k = s; k < t; k++) {
      if (Math.abs(pos[2 * k + 1]) >= 89.9999) { pole = true; continue; }
      let x = pos[2 * k]; if (x - ref > 180) x -= 360; else if (x - ref < -180) x += 360;
      pos[2 * k] = x; sum += x; cnt++; lo = Math.min(lo, x); hi = Math.max(hi, x);
    }
    if (pole) for (let k = s; k < t; k++) if (Math.abs(pos[2 * k + 1]) >= 89.9999) pos[2 * k] = sum / Math.max(cnt, 1);
    if (hi > 180) cross.push(e, -360); else if (lo < -180) cross.push(e, 360);
  }
  if (!cross.length) return ({length: E, starts, pos, idx, eOf: null, V});
  // ±360° copies of the elements across the antimeridian, appended
  const nc = cross.length / 2;
  let vc = 0, ic = 0;
  for (let c = 0; c < nc; c++) { const e = cross[2 * c], n = starts[e + 1] - starts[e]; vc += n; ic += 3 * (n - 2); }
  const pos2 = new Float32Array(2 * (V + vc)); pos2.set(pos);
  const starts2 = new Uint32Array(E + nc + 1); starts2.set(starts);
  const idx2 = new Uint32Array(idx.length + ic); idx2.set(idx);
  const eOf = new Uint32Array(E + nc); for (let e = 0; e < E; e++) eOf[e] = e;
  let vo = V, io = idx.length;
  for (let c = 0; c < nc; c++) {
    const e = cross[2 * c], dx = cross[2 * c + 1], s = starts[e], n = starts[e + 1] - s;
    for (let k = 0; k < n; k++) { pos2[2 * (vo + k)] = pos[2 * (s + k)] + dx; pos2[2 * (vo + k) + 1] = pos[2 * (s + k) + 1]; }
    for (let k = 1; k < n - 1; k++) { idx2[io++] = vo; idx2[io++] = vo + k; idx2[io++] = vo + k + 1; }
    eOf[E + c] = e; vo += n; starts2[E + c + 1] = vo;
  }
  return ({length: E + nc, starts: starts2, pos: pos2, idx: idx2, eOf, V: V + vc, copies: nc});
}
function colours(m, G, v, field) {
  const key = `${v}-${field}`;
  if (X.colours[key]) return X.colours[key];
  const t0 = performance.now();
  const out = new Uint8Array(4 * G.V), a = m.f.angle, r = m.f.size_ratio, c = new Uint8Array(4);
  const i = X.info || {}, d = field === 'depth' ? m.f.depth : field === 'size_km' ? m.f.size_km : null;
  const [lo, hi] = field === 'depth' && i.depth ? [0, i.depth.hi] : field === 'size_km' && i.size_km ? [i.size_km.lo, i.size_km.hi] : [0, 1];
  for (let p = 0; p < G.length; p++) {
    const e = G.eOf ? G.eOf[p] : p;
    elemColour(field, a[e], r[e], e >= m.nt, c, 0, d ? d[e] : 0, lo, hi);
    for (let k = 4 * G.starts[p], k1 = 4 * G.starts[p + 1]; k < k1; k += 4) { out[k] = c[0]; out[k + 1] = c[1]; out[k + 2] = c[2]; out[k + 3] = c[3]; }
  }
  S.stats.loads.push({what: 'colours', view: v, field, ms: performance.now() - t0});
  return (X.colours[key] = out);
}
// the binary path data of n outlines (vertices src[starts[e] .. starts[e + 1]), size numbers each), each closed by its
// first vertex again
export function closedOutlines(src, size, starts, n) {
  const out = new src.constructor(src.length + n * size), st = new Uint32Array(n + 1);
  let o = 0;
  for (let e = 0; e < n; e++) {
    const a = starts[e] * size, b = starts[e + 1] * size;
    st[e] = o / size;
    out.set(src.subarray(a, b), o); o += b - a;
    out.set(src.subarray(a, a + size), o); o += size;
  }
  st[n] = o / size;
  return {length: n, startIndices: st, attributes: {getPath: {value: out, size}}};
}
function lifted(G, alt) {
  const L = G.lift || (G.lift = {});
  if (L[alt]) return L[alt];
  const n = G.pos.length / 2, out = new Float32Array(3 * n);
  for (let k = 0; k < n; k++) { out[3 * k] = G.pos[2 * k]; out[3 * k + 1] = G.pos[2 * k + 1]; out[3 * k + 2] = alt; }
  return (L[alt] = out);
}

// --------------------------------------------------------------------------------------------------- loading
export async function loadMesh(mode) {
  const seq = X.loadSeq = (X.loadSeq || 0) + 1, stale = () => seq !== X.loadSeq;
  X.mode = mode;
  $('h_mesh_row').hidden = mode === 'off';
  document.querySelectorAll('[data-mesh]').forEach((b) => b.classList.toggle('on', b.dataset.mesh === mode));
  const t0 = performance.now(), rec = {what: `mesh-${mode}`};
  X.data = null; X.geo = {}; X.colours = {}; X.cellKm = null;
  if (mode !== 'window') X.highlight = null;           // the ring marks a picked worst element only in its zoom window
  if (mode === 'off') { for (const f of dataHooks) f(mode); legend(); redraw(); ensurePolarH(); return rec; }
  if (!X.info) await loadInfo();
  if (mode === 'overview' || mode === 'worst') await loadOverview();
  if (stale()) return rec;
  let q = null;
  if (mode === 'worst') q = `sel=worst&n=2000&field=${rankField()}`;
  if (mode === 'full') q = 'sel=all';
  // a zoom window carries the global node ids and node_flag (5 bytes a node) for the hand editor (edit.js)
  if (mode === 'window') { const c = viewCentre(); q = `sel=window&lon=${c[0].toFixed(3)}&lat=${c[1].toFixed(3)}&r_km=${X.rKm}&edit=1`; }
  if (q !== null) {
    const {buf, ms} = await bin(`api/mesh/mesh.bin?${q}`);
    if (stale()) return rec;
    const t1 = performance.now();
    X.data = decodeMesh(buf);
    X.cellKm = cellKm(X.data);
    const t2 = performance.now();
    const G = geometry(X.data, S.view); colours(X.data, G, S.view, X.field);
    Object.assign(rec, {fetch_ms: ms, bytes: buf.byteLength, decode_ms: t2 - t1, expand_ms: performance.now() - t2,
      n_elems: X.data.nt + X.data.nq, n_nodes: X.data.nn, copies: G.copies || 0});
  }
  for (const f of dataHooks) f(mode);
  const t3 = performance.now();
  legend();
  const done = nextRender(); redraw(); await done;
  Object.assign(rec, {first_frame_ms: performance.now() - t3, total_ms: performance.now() - t0});
  S.stats.loads.push(rec);
  $('mesh_load').textContent = X.data ? `${mode}: ${(X.data.nt + X.data.nq).toLocaleString()} elements, ${(rec.bytes / 1e6).toFixed(1)} MB` +
    `\nfetch ${rec.fetch_ms.toFixed(0)} ms, decode ${rec.decode_ms.toFixed(0)} ms, expand ${rec.expand_ms.toFixed(0)} ms, first frame ${rec.first_frame_ms.toFixed(0)} ms` +
    (rec.copies ? `\n${rec.copies} elements copied across ±180°` : '') : `${mode}: raster ${S.view}`;
  return rec;
}
async function loadOverview() {
  const proj = isPolar(S.view) ? S.view : null, key = proj || 'll';
  X.overview[key] = await loadImage(`api/mesh/overview.png?field=${X.field}${proj ? '&proj=' + proj : ''}&t=${Date.now()}`);
}
// the mesh's name as the user knows it: the run folder and raw / smooth, or the reference mesh's folder
export const meshName = (p) => { const s = p.split('/'), f = s.at(-1).replace('.npz', ''); return f === 'mesh' ? s.at(-2) : `${s.at(-2)} / ${f}`; };
export async function loadInfo() {
  const prev = X.info && X.info.path;
  X.info = await api('api/mesh/info');
  const i = X.info, c = i.irregular, Q = i.quality;
  if (i.path !== prev) {                    // a new mesh: irregular nodes on for quad meshes, off for triangle meshes
    X.showIrr = c.regular === 4; $('show_irr').checked = X.showIrr;
  }
  const dopt = $('field').querySelector('option[value="depth"]');      // colour "depth": only with a depth (its write, depth.npy)
  dopt.disabled = !i.has_depth;
  dopt.title = i.has_depth ? `depth per element: the mean of its nodes' depths (${i.depth_source}), 0–${i.depth.hi} m`
    : 'no depth for this mesh yet: write it as model files (Write model files…), or a run\'s final mesh with [bathymetry] source set has depth.npy';
  if (!i.has_depth && X.field === 'depth') { X.field = 'q'; $('field').value = 'q'; X.overview = {}; X.colours = {}; }
  $('mesh_now').replaceChildren(el('b', {}, meshName(i.path)), ` · ${i.n_nodes.toLocaleString()} nodes`,
    el('div.muted.small', {}, 'Open another mesh in Runs (raw / smooth) or in Reference meshes.'));
  $('mesh_info').textContent = `${i.n_quads.toLocaleString()} quads, ${i.n_tris.toLocaleString()} triangles, typical edge ${i.cell_km.toPrecision(3)} km` +
    `\nirregular interior nodes (valence ≠ ${c.regular}): n${c.regular - 1} ${c['n' + (c.regular - 1)]}, n${c.regular + 1} ${c['n' + (c.regular + 1)]}, other ${c.other}` +
    (Q && Q.irregular && Q.irregular.components ? `\nwater: ${Q.irregular.components} connected ${Q.irregular.components === 1 ? 'part' : 'parts (the ocean and closed seas)'}` : '') +
    (Q ? `\n${[Q.family, Q.config || Q.generator, Q.res].filter(Boolean).join(' · ') || (Q.method || '')}` : '');
  renderGates(i.gates, Q);
  const runCfg = !!(i.config && i.config.endsWith('config.resolved.toml'));         // Write model files needs a `meshom run` folder
  $('mesh_write').disabled = !runCfg;
  $('mesh_write').title = runCfg ? "write this mesh (or another of its run) as model files: the run's format, depth and gates, the format's checks"
    : 'only for a mesh of a `meshom run` folder (config.resolved.toml next to it gives the format, the bathymetry and the gates)';
  if (i.repair) X.pair = {after: i.path, before: `${i.path.split('/').slice(0, -1).join('/')}/${i.repair.input}`, repair: i.repair};
  else if (X.pair && i.path !== X.pair.before) X.pair = null;
  renderRepair();
  for (const k of [1, 2]) $(`irr_${k}_name`).textContent = `n${c.regular + (k === 1 ? -1 : 1)}`;
  X.irr = null; X.tris = null;
  await Promise.all([loadIrregular(), loadTris(), loadHist(), loadWorst(), loadMeshCoast(i)]);
  await loadBlocks(i.m14 ? i.run_dir : null);
}
// the gates of the mesh's quality.json as a table: what, where, value, limit, passed; the failed rows first, the
// passed ones folded
function renderGates(gates, Q) {
  const box = $('mesh_gates'); box.replaceChildren();
  if (!gates || !gates.length) { box.append(el('div.muted.small', {}, 'No quality.json next to this mesh: no gates to show.')); return; }
  box.append(gatesView(gates, Q));
}
// rows of runs.gate_table (server) as that table; Q: the record (its gates_passed / gates_total when scored)
export function gatesView(gates, Q) {
  const box = el('div');
  const failed = gates.filter((g) => !g.passed), passed = gates.filter((g) => g.passed);
  const val = (g, v) => typeof v === 'boolean' ? (v ? 'yes' : 'no') : g.count ? String(Math.round(v)) : `${fmt(v, g.unit === '°' ? 1 : 3)}${g.unit}`;
  const row = (g) => el('tr', {title: g.gate}, el('td.' + (g.passed ? 'pass' : 'fail'), {}, g.passed ? '✓ passed' : '✗ failed'),
    el('td', {}, g.what, el('span.muted', {}, g.where ? ` · ${g.where}` : '')), el('td.num', {}, val(g, g.value)), el('td.num', {}, `${g.op} ${val(g, g.limit)}`));
  const head = () => el('tr', {}, ['', 'gate', 'value', 'limit'].map((t) => el('th', {}, t)));
  box.append(el('div', {}, el('b', {}, `Gates: ${passed.length} of ${gates.length} passed`),
    Q && Q.gates_total && Q.gates_total !== gates.length ? el('span.muted.small', {title: 'quality.json was scored with the gates of its day; this table checks today\'s gates on the same numbers'},
      ` (when scored: ${Q.gates_passed} of ${Q.gates_total})`) : null));
  if (failed.length) box.append(el('table.tbl.gates', {}, head(), failed.map(row)));
  if (passed.length) box.append(el('details', {}, el('summary.small', {}, `the ${passed.length} passed gates`), el('table.tbl.gates', {}, head(), passed.map(row))));
  return box;
}
async function loadIrregular() {
  const {buf} = await bin('api/mesh/irregular.bin');
  const n = new Uint32Array(buf, 0, 1)[0];
  X.irr = {n, ll: new Float32Array(buf, 4, 2 * n), cls: new Uint8Array(buf, 4 + 8 * n, n), data: {}};
}
async function loadTris() {
  if (!X.info.n_tris || X.info.n_tris > X.info.n_quads) return;      // a triangle mesh: the whole mesh, not an overlay
  const {buf} = await bin('api/mesh/mesh.bin?sel=tris');
  X.tris = {m: decodeMesh(buf), geo: {}, col: {}};
}
// ------------------------------------------------------------------------------------------------ histograms
async function loadHist() {
  const r = await api('api/mesh/hist');
  const box = $('mesh_hist'); box.replaceChildren();
  for (const h of r.histograms) if (h.n || !h.counts.length) box.append(histogram(h));     // no empty quad θmax on a triangle mesh
  if (!X.info.has_ln_sh && !r.histograms.some((h) => h.name.startsWith('|ln'))) box.append(el('div.muted.small', {}, '|ln s/h| per edge: h of this mesh unknown (no run config, not a compete folder)'));
  if (r.quality) box.append(el('div.muted.small', {}, 'bins: the overlay arrays; "quality.json" lines: the scored record (the numbers the gates use)'));
}
function histogram(h) {
  const W = 330, Hh = 70, pad = 4, n = h.counts.length, max = Math.log10(1 + Math.max(...h.counts, 1));
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${Hh + 14}`); svg.setAttribute('class', 'hist');
  const add = (tag, attrs, text) => { const e = document.createElementNS(svg.namespaceURI, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); if (text) e.textContent = text; svg.append(e); return e; };
  const x = (val) => pad + (val - h.lo) / (h.hi - h.lo) * (W - 2 * pad);
  h.counts.forEach((c, i) => {
    if (!c) return;
    const bh = Math.max(1, Math.log10(1 + c) / max * Hh);
    add('rect', {x: x(h.lo + i * h.step) + 0.3, y: Hh - bh, width: Math.max(0.5, (W - 2 * pad) / n - 0.6), height: bh, class: 'bar'}).append(
      Object.assign(document.createElementNS(svg.namespaceURI, 'title'), {textContent: `${fmt(h.lo + i * h.step, 3)}–${fmt(h.lo + (i + 1) * h.step, 3)}: ${c}`}));
  });
  h.gates.forEach(([label, g], k) => {
    add('line', {x1: x(g), x2: x(g), y1: 0, y2: Hh, class: 'gate'});
    add('text', {x: x(g) + 2, y: 9 + 9 * (k % 2), class: 'gate-label'}, `${label} ${g}`);
  });
  add('text', {x: pad, y: Hh + 12, class: 'axis'}, `${h.lo}${h.unit}`);
  add('text', {x: W - pad, y: Hh + 12, class: 'axis', 'text-anchor': 'end'}, `≥ ${h.hi}${h.unit}`);
  const rec = h.record ? el('div.small.rec', {}, 'quality.json: ' + Object.entries(h.record).map(([k, v]) => `${k} ${fmt(v, 3)}`).join(' · ')) : null;
  const note = h.note ? el('div.muted.small', {}, h.note) : null;
  if (!h.counts.length) return el('div.histbox', {}, el('div.small', {}, el('b', {}, h.name), el('span.muted', {}, ' (no bins: h unknown to the UI)')), rec, note);
  return el('div.histbox', {}, el('div.small', {}, el('b', {}, h.name), ` · n ${h.n.toLocaleString()} · p99 ${fmt(h.p99, 3)} · p99.9 ${fmt(h.p99_9, 3)} · max ${fmt(h.max, 3)}`,
    el('span.muted', {}, ' (log counts)')), svg, rec, note);
}
// -------------------------------------------------------------------------------------------- worst elements
async function loadWorst() {
  const r = await api(`api/mesh/worst.json?n=40&field=${rankField()}`);
  const tb = el('table.tbl', {}, el('tr', {}, ['#', 'id', 'kind', 'θmax °', 'size ratio', 'lon', 'lat'].map((t) => el('th', {}, t))));
  r.worst.forEach((w, i) => tb.append(el('tr', {title: 'fly to and load a zoom window', on: {click: () => flyToElement(w)}},
    el('td', {}, i + 1), el('td', {}, w.id), el('td', {}, w.kind), el('td', {}, fmt(w.angle, 1)), el('td', {}, fmt(w.size_ratio, 2)),
    el('td', {}, fmt(w.lon, 2)), el('td', {}, fmt(w.lat, 2)))));
  $('mesh_worst').replaceChildren(tb);
  X.worst = r.worst;
  refreshWin('worst');
}
// the same list in a floating window, sortable (by θmax, size ratio, lon, lat, …); a click on a row flies there as
// in the panel
const WORST_COLS = [
  {label: '#', num: true, value: (r) => r.i + 1},
  {label: 'id', num: true, value: (r) => r.w.id},
  {label: 'kind', value: (r) => r.w.kind},
  {label: 'θmax °', num: true, value: (r) => r.w.angle, cell: (r) => fmt(r.w.angle, 1)},
  {label: 'size ratio', num: true, value: (r) => r.w.size_ratio, cell: (r) => fmt(r.w.size_ratio, 2)},
  {label: 'lon', num: true, value: (r) => r.w.lon, cell: (r) => fmt(r.w.lon, 2)},
  {label: 'lat', num: true, value: (r) => r.w.lat, cell: (r) => fmt(r.w.lat, 2)},
];
defineWin('worst', {width: 470, height: 440,
  title: () => `Worst elements${X.info ? ' · ' + meshName(X.info.path) : ''}`,
  build: () => X.worst && X.info ? el('div.fwin-stack', {}, el('div.muted.small', {}, `the ${X.worst.length} worst by ${rankField() === 'size_ratio' ? 'size ratio' : 'max angle'} (the colour); click a row: fly to it`),
    sortTable('worst', WORST_COLS, X.worst.map((w, i) => ({w, i})), {onRow: (r) => flyToElement(r.w), rowTitle: () => 'fly to and load a zoom window', cls: 'worst-win'}))
    : el('div.muted', {}, 'No mesh open: open one in Runs (raw / smooth) or in Reference meshes.')});
export async function flyToElement(w, rKm = 250) {
  X.highlight = w; flyTo(w.lon, w.lat, rKm);
  const keep = X.rKm; X.rKm = Math.max(rKm * 1.6, 200);
  const rec = await loadMesh('window'); X.rKm = keep;
  return rec;
}

// ------------------------------------------------------------------------------------------------ map layers
addLayers(20, (v) => {
  const ov = X.overview[isPolar(v) ? v : 'll'];
  return meshDrawn() && (X.mode === 'overview' || X.mode === 'worst') && ov ? new D.BitmapLayer({id: `overview-${v}`, image: ov, bounds: isPolar(v) ? [-1, -1, 1, 1] : [-180, -90, 180, 90],
    opacity: (X.mode === 'worst' ? 0.3 : 1) * meshOpacity(),     // Worst: the worst cells stand out over a dimmed raster
    ...(v === 'globe' ? {_imageCoordinateSystem: D.COORDINATE_SYSTEM.LNGLAT} : {})}) : null;
});
// the open mesh belongs to the Inspect tabs; in the set-up tabs (h, coast, method) h is drawn and the mesh only when
// asked for (Size h, "the open mesh over h"), faint over h
export const meshDrawn = () => X.mode !== 'off' && (!SETUP_TABS.has(S.tab) || X.overH);
const meshOpacity = () => (SETUP_TABS.has(S.tab) ? 0.35 : 1);
function meshLayers(m, v, id, colourFn, edges, pickable, cache = X.geo, colourKey = '') {
  // one data object per geometry and one layer id per mode and view: a new colour field swaps only the colour array
  // (updateTriggers, 70-95 ms for 10^6 elements); a new layer id or data object makes deck.gl rebuild and upload the
  // whole layer (with SwiftShader 0.3-50 s, erratic)
  const G = geometry(m, v, cache), L = [], d = G.d || (G.d = {});
  const fill = d[id] || (d[id] = {length: G.length, startIndices: G.starts, mesh: m, geo: G, attributes: {indices: G.idx,
    getPolygon: v === 'globe' ? {value: lifted(G, ALT.fill), size: 3} : {value: G.pos, size: 2}}});
  const col = colourFn(G);
  if (!fill.attributes.getFillColor || fill.attributes.getFillColor.value !== col) fill.attributes.getFillColor = {value: col, size: 4, normalized: true};
  L.push(new D.SolidPolygonLayer({id: `${id}-${v}`, data: fill, _normalize: false, filled: true, extruded: false, pickable, opacity: meshOpacity(),
    updateTriggers: {getFillColor: colourKey}}));
  if (edges) {
    // every element's outline closed by hand (its first vertex again at the end) and drawn as an open path: deck.gl's
    // _pathType 'loop' does not close paths given as binary attributes (its tesselator does not read them), so the
    // edge from the last vertex to the first was missing, and gone where it was that edge for both neighbours
    const k = sizeOf(v, ALT.edges);
    const path = d.edges || (d.edges = closedOutlines(v === 'globe' ? lifted(G, ALT.edges) : G.pos, k, G.starts, G.length));
    L.push(new D.PathLayer({id: `${id}-edges-${v}`, data: path, _pathType: 'open', getColor: [0, 0, 0, 150], getWidth: 0.7, widthUnits: 'pixels'}));
  }
  return L;
}
// after a change of view the geometry and colours of the other views are dropped (rebuilt in 0.1-0.2 s if needed):
// one view's vertex arrays (~70 MB for 10^6 elements) are kept, not three
function freshData(v) {
  if (X.lastView === v) return;
  X.lastView = v;
  for (const k of Object.keys(X.geo)) if (k !== v) delete X.geo[k];
  for (const k of Object.keys(X.colours)) if (!k.startsWith(v + '-')) delete X.colours[k];
  if (X.tris) { X.tris.geo = {}; X.tris.col = {}; }
}
addLayers(30, (v) => {
  freshData(v);
  if (!X.data || !meshDrawn()) return null;
  const small = X.data.nt + X.data.nq <= 50000;          // picking for windows, the worst set and small meshes
  X.edgesOn = edgesVisible();                            // the rule above (EDGE_MIN_PX)
  return meshLayers(X.data, v, `mesh-${X.mode}`, (G) => colours(X.data, G, v, X.field), X.edgesOn,
    X.mode !== 'full' || small, X.geo, X.field);
});
addLayers(32, (v) => {
  if (!X.showTris || !X.tris || !meshDrawn()) return null;
  const T = X.tris;                        // the triangle subset has its own geometry and colour caches
  return meshLayers(T.m, v, 'tris', (G) => T.col[v] || (T.col[v] = new Uint8Array(4 * G.V).map((_, k) => [240, 140, 30, 255][k % 4])),
    true, false, T.geo);
});
addLayers(85, (v) => {
  const L = [];
  X.irrOn = irrVisible();
  if (X.showIrr && X.irrOn && X.irr && X.irr.n && meshDrawn()) {
    const key = `${v}-${[1, 2, 3].map((c) => +X.irrShow[c]).join('')}`;
    if (!X.irr.data[key]) {
      const I = X.irr, sel = [], far = (i) => isPolar(v) && Math.hypot(...polarXY(I.ll[2 * i], I.ll[2 * i + 1], v === 'north')) > 2;
      for (let i = 0; i < I.n; i++) if (X.irrShow[I.cls[i]] && !far(i)) sel.push(i);      // polar maps: one hemisphere + 37°
      const sz = v === 'globe' ? 3 : 2, pos = new Float32Array(sz * sel.length), col = new Uint8Array(4 * sel.length);
      sel.forEach((i, k) => {
        const lon = I.ll[2 * i], lat = I.ll[2 * i + 1], p = isPolar(v) ? polarXY(lon, lat, v === 'north') : [lon, lat];
        pos[sz * k] = p[0]; pos[sz * k + 1] = p[1]; if (sz === 3) pos[3 * k + 2] = ALT.points;
        col.set([...IRR_COLOURS[I.cls[i]], 255], 4 * k);
      });
      X.irr.data[key] = {length: sel.length, attributes: {getPosition: {value: pos, size: sz}, getFillColor: {value: col, size: 4, normalized: true}}};
    }
    L.push(new D.ScatterplotLayer({id: `irregular-${key}`, data: X.irr.data[key], getRadius: 3, radiusUnits: 'pixels', radiusMinPixels: 2,
      stroked: true, getLineColor: [255, 255, 255], lineWidthUnits: 'pixels', getLineWidth: 0.5}));
  }
  if (X.highlight) {
    const w = X.highlight, p = isPolar(v) ? polarXY(w.lon, w.lat, v === 'north') : [w.lon, w.lat, ALT.points];
    L.push(new D.ScatterplotLayer({id: `highlight-${v}`, data: [p], getPosition: (d) => d, getRadius: 14, radiusUnits: 'pixels',
      filled: false, stroked: true, getLineColor: [255, 0, 200], lineWidthUnits: 'pixels', getLineWidth: 3}));
  }
  return L;
});
// the picked layer's own mesh and geometry (its data object), not X.data: a hover on a layer drawn before the mesh or the
// window changed gives an index into that layer's elements
addTooltip(({layer, index}) => {
  const d = layer && layer.id.startsWith('mesh-') && layer.props.data, m = d && d.mesh;
  if (!m || index < 0) return null;
  const e = d.geo.eOf ? d.geo.eOf[index] : index;
  if (!(e < m.nt + m.nq)) return null;
  const km = m.f.size_km ? m.f.size_km[e] : null;
  return `element ${m.eid ? m.eid[e] : e} (${e >= m.nt ? 'quad' : 'triangle'})\nθmax ${m.f.angle[e].toFixed(1)}°\nsize ratio ${m.f.size_ratio[e].toFixed(3)}` +
    (km !== null ? `\ncell ${km < 10 ? km.toFixed(2) : km.toFixed(1)} km (mean edge)` : '') +
    (m.f.depth ? `\ndepth ${Math.round(m.f.depth[e])} m (mean of its nodes)` : '');
});
onView(async (v) => {
  if ((X.mode === 'overview' || X.mode === 'worst') && !X.overview[isPolar(v) ? v : 'll']) await loadOverview();
});

// ---------------------------------------------------------------------------------------------- zone remesh
// A mesh repaired by `meshom repair` (repaired-NN.npz / .json next to the run's meshes) opens like raw / smooth, with
// the box below: accepted or not, the zone's and the whole mesh's numbers before -> after (worse ones marked), and
// Show before / Show after on the same view. Remeshing from the page is planned, greyed out (decision U27).
function renderRepair() {
  const box = $('repair_box'); box.replaceChildren();
  const P = X.pair;
  if (!P || !X.info || (X.info.path !== P.after && X.info.path !== P.before)) return;
  const R = P.repair, after = X.info.path === P.after;
  const num = (v) => v === null || v === undefined ? '–' : Number.isInteger(v) ? v.toLocaleString() : fmt(v, Math.abs(v) >= 10 ? 1 : 3);
  const cell = (pair, row) => {
    if (!pair) return el('td.num.muted', {}, '–');
    const [b, a] = pair, worse = row.what !== 'elements' && typeof a === 'number' && typeof b === 'number' && (row.higher_is_better ? a < b : a > b);
    return el('td.num' + (worse ? '.fail' : ''), {title: worse ? 'worse after the repair' : ''}, `${num(b)} → ${num(a)}${worse ? ' ▲' : ''}`);
  };
  const go = async (path) => { await api('api/mesh/load', {path}); await reloadMesh(false); };
  box.append(...[el('div', {}, el('b', {}, `Zone remesh of ${R.input}: `),
    R.accepted ? `applied (${R.forced ? 'forced, ' : ''}fill "${R.kind || 'mixed'}")` : `not applied${R.reason ? ': ' + R.reason : ''}; the mesh is unchanged`),
    R.n_core ? el('div.muted.small', {}, `${R.n_core} elements in the zone, ${R.n_cavity || '–'} with the rings`) : null,
    el('table.tbl.gates', {}, el('tr', {}, ['', R.polygon ? 'zone' : '', 'whole mesh'].map((t) => el('th', {}, t))),
      R.rows.map((row) => el('tr', {}, el('td', {}, row.what), R.polygon ? cell(row.zone, row) : el('td'), cell(row.whole, row)))),
    el('div.row', {},
      el('button', {disabled: !after, on: {click: () => go(P.before)}}, 'Show before'),
      el('button', {disabled: after, on: {click: () => go(P.after)}}, 'Show after'),
      R.polygon ? el('button', {on: {click: () => flyToZone(R.polygon)}}, 'Fly to the zone') : null,
      el('span.muted.small', {}, after ? 'showing: after' : 'showing: before'))].filter(Boolean));
}
function flyToZone(poly) {
  const p = arcPath(poly, true), lon = p.reduce((a, q) => a + q[0], 0) / p.length, lat = p.reduce((a, q) => a + q[1], 0) / p.length;
  const r = Math.max(...p.map((q) => Math.hypot((q[0] - lon) * Math.cos(lat * Math.PI / 180), q[1] - lat))) * 111;
  flyTo(((lon + 540) % 360) - 180, lat, Math.max(r * 1.5, 200));
}
addLayers(83, (v) => {
  const L = [], line = (id, poly, colour) => new D.PathLayer({id: `${id}-${v}`, data: pathCopies(arcPath(poly, true), v),
    getPath: (d) => d.map((p) => projectPt(p, v, ALT.zone)), getColor: colour, getWidth: 2.5, widthUnits: 'pixels'});
  if (X.pair && X.pair.repair.polygon && meshDrawn()) L.push(line('zone-done', X.pair.repair.polygon, [255, 255, 255, 240]));
  return L;
});

// "worst" by the colour field: max angle (q) or size ratio; tri / quad, depth and cell size rank by angle
const rankField = () => (X.field === 'size_ratio' ? 'size_ratio' : 'q');
// the whole mesh in view: the fitted view for a global mesh, its lon-lat box for a regional one
export async function frameMesh() {
  const e = X.info && X.info.extent;
  if (!e) return;
  if (e.global || isPolar(S.view)) { await setView(S.view); return; }
  const lat = (e.south + e.north) / 2, lon = ((e.west + e.width / 2 + 540) % 360) - 180;
  flyTo(lon, lat, Math.max(e.width * 111 * Math.cos(lat * Math.PI / 180), (e.north - e.south) * 111) * 0.55 + 50);
}
export function initMesh() {
  document.querySelectorAll('[data-mesh]').forEach((b) => b.onclick = async () => {
    const m = b.dataset.mesh;
    if (m === 'overview' || m === 'worst') { if (!X.info) await loadInfo(); await frameMesh(); }
    await loadMesh(m);
  });
  $('field').onchange = async (e) => { await setField(e.target.value); await loadWorst(); if (X.mode === 'worst') await loadMesh('worst'); };
  $('show_irr').onchange = (e) => { X.showIrr = e.target.checked; legend(); redraw(); };
  $('mesh_show_h').onchange = (e) => { X.showH = e.target.checked; legend(); redraw(); ensurePolarH(); };
  $('h_mesh_over').onchange = (e) => { X.overH = e.target.checked; legend(); redraw(); };
  for (const c of [1, 2, 3]) $(`irr_${c}`).onchange = (e) => { X.irrShow[c] = e.target.checked; redraw(); };
  $('show_tris').onchange = (e) => { X.showTris = e.target.checked; redraw(); };
  $('show_edges').onchange = (e) => { X.showEdges = e.target.checked; legend(); redraw(); };
  $('win_r').onchange = (e) => { X.rKm = +e.target.value; };
  $('worst_win').onclick = () => openWin('worst');
}
// a new colour field: new colours only (the overview raster is reloaded, the elements are not fetched again)
export async function setField(f) {
  const t0 = performance.now();
  X.field = f; $('field').value = f; X.overview = {}; X.colours = {};
  if (X.mode === 'overview' || X.mode === 'worst') await loadOverview();
  legend();
  const done = nextRender(); redraw(); await done;
  const ms = performance.now() - t0;
  S.stats.loads.push({what: 'field', field: f, ms});
  return ms;
}
// after another mesh was opened: its info, then (show) its overview with the whole mesh in view
export async function reloadMesh(show = false) {
  X.overview = {}; X.highlight = null; await loadInfo();
  if (X.mode === 'full' && X.info.n_tris + X.info.n_quads > FULL_MAX) {
    X.mode = 'overview'; status(`${(X.info.n_tris + X.info.n_quads).toLocaleString()} elements: shown as the overview ("Full" draws every element)`);
  }
  if (show) { if (X.mode === 'off' || X.mode === 'window') X.mode = 'overview'; await frameMesh(); }
  if (X.mode !== 'off') await loadMesh(X.mode);
}
