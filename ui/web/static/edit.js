// Panel "Mesh", section "Edit by hand" (an option, off by default; only in a zoom window, the few thousand elements
// around the view centre). Drag a node: its elements and edges follow at once and show their quality, recomputed here
// on the sphere as meshom.quality computes it (max angle: corner angles in each element's gnomonic tangent plane about
// its normalised corner mean; size ratio: the largest ratio of mean great-circle edge lengths across the element's
// edges). Add a node: a click inside an element splits it around the new node (a triangle into 3 triangles, a quad into
// 4). Undo drops the last edit (the window is rebuilt from the loaded one by replaying the others). "Save as
// edited-NN" sends the edits to the server, which replays them on the full mesh and only writes edited-NN.npz next to
// it (ui/web/edit.py: the same numbering and splitting rules; raw / smooth / earlier files are never overwritten);
// nothing is recomputed: the edits stay in the page, a later save writes all of them into the
// next file, and the saved file opens from Runs like raw / smooth; "Write model files…" writes it for the model (write.js, U32). "Remesh with these edits" is planned. Coast and open-boundary nodes (node_flag != 0) and the nodes on the window's rim (the
// mesh's own boundary is part of it) do not move. Flips, collapses and other topology belong to the core.
import {$, D, DEG, S, api, el, fmt, status} from './util.js';
import {openWrite} from './write.js';
import {ALT, addLayers, deckgl, onPointer, projectPt, redraw, unproject, viewCentre} from './map.js';
import {onMeshData} from './mesh.js';
import {legend} from './hsize.js';

const X = S.mesh;
// base: the zoom window as loaded (decodeMesh with node ids and flags); ops: the edits in the server's numbering
// (ui/web/edit.py); W: the editable window = base + ops
const E = S.edit = {on: false, tool: 'move', base: null, ops: [], W: null, drag: null, panOff: false, saving: false, saved: null, file: null};   // file: the last saved edited-NN.npz
const active = () => E.on && X.mode === 'window' && E.W !== null;
const R_KM = 6371;
const wrap = (lon) => ((lon + 540) % 360) - 180;

// ------------------------------------------------------------------------------------- quality on the sphere
const xyz = (lon, lat) => { const a = lon * DEG, b = lat * DEG, c = Math.cos(b); return [c * Math.cos(a), c * Math.sin(a), Math.sin(b)]; };
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = (a) => { const n = Math.hypot(a[0], a[1], a[2]); return [a[0] / n, a[1] / n, a[2] / n]; };
// largest corner angle (degrees) of an element with unit-vector corners P, CCW seen from outside (meshom/quality/kernel.py)
export function maxAngle(P) {
  const k = P.length, c = unit(P.reduce((s, p) => [s[0] + p[0], s[1] + p[1], s[2] + p[2]], [0, 0, 0]));
  let e1 = Math.abs(c[2]) < 0.9 ? [-c[1], c[0], 0] : [1, 0, 0];           // east, or x near a pole
  const d = dot(e1, c); e1 = unit([e1[0] - d * c[0], e1[1] - d * c[1], e1[2] - d * c[2]]);
  const e2 = cross(c, e1);                                                  // e1 x e2 = c: seen from outside
  const q = P.map((p) => { const s = dot(p, c); return [dot(p, e1) / s, dot(p, e2) / s]; });
  let mx = 0;
  for (let i = 0; i < k; i++) {
    const o = q[i], n = q[(i + 1) % k], p = q[(i + k - 1) % k];
    const a = [n[0] - o[0], n[1] - o[1]], b = [p[0] - o[0], p[1] - o[1]];
    let t = Math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]);
    if (t < 0) t += 2 * Math.PI;
    mx = Math.max(mx, t);
  }
  return mx / DEG;
}
const arcKm = (a, b) => R_KM * Math.atan2(Math.hypot(...cross(a, b)), dot(a, b));
const meanEdge = (P) => P.reduce((s, p, i) => s + arcKm(p, P[(i + 1) % P.length]), 0) / P.length;
const corners = (W, e) => W.elems[e].n.map((i) => xyz(W.ll[2 * i], W.ll[2 * i + 1]));
const edgeKey = (a, b) => (a < b ? `${a},${b}` : `${b},${a}`);
// max angle, size ratio and cell size (mean edge, km) of element e (neighbours outside the window are not known: their
// edges are left out of the size ratio)
function quality(W, e) {
  const el = W.elems[e], P = corners(W, e), me = meanEdge(P), c = el.n;
  el.angle = maxAngle(P); el.km = me;
  let sr = 1;
  for (let k = 0; k < c.length; k++) {
    for (const f of W.edges.get(edgeKey(c[k], c[(k + 1) % c.length])) || []) {
      if (f === e) continue;
      const mf = meanEdge(corners(W, f));
      sr = Math.max(sr, Math.max(me, mf) / Math.min(me, mf));
    }
  }
  el.sr = sr;
}

// -------------------------------------------------------------------------------------- the editable window
// node -> its elements, edge -> its elements, the rim (nodes on an edge with one element in the window)
function topology(W) {
  const nn = W.ll.length / 2;
  W.inc = Array.from({length: nn}, () => []); W.edges = new Map(); W.rim = new Uint8Array(nn);
  W.elems.forEach((el, e) => {
    if (!el.alive) return;
    const c = el.n;
    for (let k = 0; k < c.length; k++) {
      W.inc[c[k]].push(e);
      const key = edgeKey(c[k], c[(k + 1) % c.length]);
      W.edges.has(key) ? W.edges.get(key).push(e) : W.edges.set(key, [e]);
    }
  });
  for (const [key, es] of W.edges) if (es.length === 1) for (const i of key.split(',')) W.rim[+i] = 1;
}
const movable = (W, i) => !W.flag[i] && !W.rim[i];
function applyOp(W, op) {
  if (op.op === 'move') {
    const i = W.nodeOf.get(op.node);
    W.ll[2 * i] = op.lon; W.ll[2 * i + 1] = op.lat;
    for (const e of W.inc[i]) W.touched.add(e);
    return;
  }
  const e = W.elemOf.get(op.elem), c = W.elems[e].n, p = W.ll.length / 2;
  W.elems[e].alive = false; W.touched.delete(e);
  W.ll.push(op.lon, op.lat); W.gid.push(W.nextNode); W.flag.push(0); W.nodeOf.set(W.nextNode++, p);
  for (let k = 0; k < c.length; k++) {                // corner order kept: (a, b, p), (b, c, p), ... as the server does
    W.elemOf.set(W.nextElem, W.elems.length); W.touched.add(W.elems.length);
    W.elems.push({n: [c[k], c[(k + 1) % c.length], p], gid: W.nextElem++, alive: true});
  }
  topology(W);
}
function build() {
  const m = E.base, i = X.info;
  const W = {ll: Array.from(m.ll), gid: Array.from(m.nid), flag: Array.from(m.nflag), elems: [], nodeOf: new Map(),
    elemOf: new Map(), nextNode: i.n_nodes, nextElem: i.n_tris + i.n_quads, touched: new Set()};
  W.gid.forEach((g, k) => W.nodeOf.set(g, k));
  for (let e = 0; e < m.nt + m.nq; e++) {
    const n = e < m.nt ? Array.from(m.tris.subarray(3 * e, 3 * e + 3)) : Array.from(m.quads.subarray(4 * (e - m.nt), 4 * (e - m.nt) + 4));
    W.elems.push({n, gid: m.eid[e], angle: m.f.angle[e], sr: m.f.size_ratio[e], km: m.f.size_km[e], alive: true});
    W.elemOf.set(m.eid[e], e);
  }
  topology(W);
  for (const op of E.ops) applyOp(W, op);
  for (const e of W.touched) quality(W, e);
  return W;
}
// the window as mesh.js draws it (decodeMesh's layout: triangles first, then quads); welem: drawn element -> W element
function toData(W) {
  const te = [], qe = [];
  W.elems.forEach((el, e) => { if (el.alive) (el.n.length === 3 ? te : qe).push(e); });
  const order = te.concat(qe), nt = te.length, nq = qe.length, n = nt + nq;
  const m = {h: {}, nn: W.ll.length / 2, nt, nq, ll: Float32Array.from(W.ll), tris: new Uint32Array(3 * nt), quads: new Uint32Array(4 * nq),
    eid: new Uint32Array(n), f: {angle: new Float32Array(n), size_ratio: new Float32Array(n), size_km: new Float32Array(n)}, welem: order,
    nid: Uint32Array.from(W.gid), nflag: Uint8Array.from(W.flag)};
  order.forEach((e, j) => {
    const el = W.elems[e];
    if (j < nt) m.tris.set(el.n, 3 * j); else m.quads.set(el.n, 4 * (j - nt));
    m.eid[j] = el.gid; m.f.angle[j] = el.angle; m.f.size_ratio[j] = el.sr; m.f.size_km[j] = el.km;
  });
  return m;
}
function show() { X.data = toData(E.W); X.geo = {}; X.colours = {}; redraw(); panel(); }
let raf = 0;
const showSoon = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; show(); }); };

// ------------------------------------------------------------------------------------------------- the panel
function panel() {
  const W = E.W, n = E.ops.length, moves = E.ops.filter((o) => o.op === 'move').length;
  const fresh = n && !(E.saved && E.saved.n === n && E.saved.last === E.ops[n - 1]);    // edits since the last save
  $('edit_undo').disabled = !n || E.saving; $('edit_save').disabled = !fresh || E.saving;
  $('edit_write').disabled = !E.file;
  const box = $('edit_info'); box.replaceChildren();
  if (!E.on) return;
  if (X.mode !== 'window' || !W) { box.append(el('div.warn.small', {}, 'Open a zoom window first ("Zoom window" above, or a row of the worst list): edits are made there.')); return; }
  box.append(el('div', {}, n ? `${n} edit${n > 1 ? 's' : ''}: ${moves} node${moves === 1 ? '' : 's'} moved, ${n - moves} added` : 'No edits yet.',
    E.saved && n ? el('span.muted', {}, fresh ? ` · last saved: ${E.saved.name} (${E.saved.n} edits)` : ` · saved as ${E.saved.name}`) : null));
  const t = [...W.touched].filter((e) => W.elems[e].alive).map((e) => W.elems[e]);
  if (!t.length) return;
  const worst = (k) => t.filter((x) => x.n.length === k).reduce((a, x) => Math.max(a, x.angle), 0);
  const wq = worst(4), wt = worst(3), sr = t.reduce((a, x) => Math.max(a, x.sr), 1);
  box.append(el('div', {}, `${t.length} changed elements · largest angle ` +
    [wq ? `quads ${fmt(wq, 1)}° (hard gate 150°)` : '', wt ? `triangles ${fmt(wt, 1)}° (hard gate 120°)` : ''].filter(Boolean).join(', ') +
    ` · largest size ratio ${fmt(sr, 3)}`));
  const rows = t.sort((a, b) => b.angle - a.angle).slice(0, 8).map((x) => el('tr', {}, el('td', {}, x.gid), el('td', {}, x.n.length === 3 ? 'triangle' : 'quad'),
    el('td.num' + (x.angle > (x.n.length === 3 ? 120 : 150) ? '.fail' : ''), {}, fmt(x.angle, 1)), el('td.num', {}, fmt(x.sr, 3))));
  box.append(el('table.tbl.gates', {}, el('tr', {}, ['element', 'kind', 'θmax °', 'size ratio'].map((h) => el('th', {}, h))), rows));
}
function setOn(on) {
  E.on = on; E.panOff = false;
  deckgl.setProps({controller: on ? {doubleClickZoom: false} : true});       // two quick clicks would zoom
  panel(); legend(); redraw();
}
function undo() {
  if (!E.ops.length) return;
  E.ops.pop(); E.W = build(); show();
}
// the server writes the file only; the page keeps the edits (nothing is recomputed, the saved file is not opened)
async function save() {
  if (!E.ops.length) return;
  E.saving = true; panel();
  const r = await api('api/mesh/edit', {ops: E.ops});
  E.saving = false;
  if (!r.error) {
    E.saved = {name: r.name, n: E.ops.length, last: E.ops[E.ops.length - 1]}; E.file = r.saved;
    status(`Saved ${r.name} next to the mesh (${r.n_nodes.toLocaleString()} nodes, ${r.n_tris.toLocaleString()} triangles, ` +
      `${r.n_quads.toLocaleString()} quads); it opens from Runs like raw / smooth.`, false, true);
  }
  panel();
}
export function initEdit() {
  $('edit_on').onchange = (e) => setOn(e.target.checked);
  document.querySelectorAll('input[name=edit_tool]').forEach((r) => r.onchange = (e) => { E.tool = e.target.value; E.panOff = false; if (E.on) deckgl.setProps({controller: {doubleClickZoom: false}}); redraw(); });
  $('edit_undo').onclick = undo;
  $('edit_save').onclick = save;
  $('edit_write').onclick = () => E.file && openWrite(E.file);
  panel();
}
// a new view of the mesh: edits not saved are dropped (they belong to the window they were made in)
onMeshData((mode) => {
  const unsaved = E.ops.length - (E.saved ? E.saved.n : 0);
  if (unsaved > 0) status(`${unsaved} hand edit${unsaved > 1 ? 's' : ''} not saved were dropped: another view of the mesh was loaded.`, true);
  E.ops = []; E.drag = null; E.saved = null;
  if (E.file && S.mesh.info && E.file.split('/').slice(0, -1).join('/') !== S.mesh.info.path.split('/').slice(0, -1).join('/')) E.file = null;   // another run's mesh
  E.base = mode === 'window' && X.data && X.data.nid ? X.data : null;
  E.W = E.base ? build() : null;
  panel();
});

// ---------------------------------------------------------------------------------------- mouse and layers
const onNodes = (info) => info.layer && info.layer.id.startsWith('edit-nodes');
onPointer('hover', (info) => {                        // over a draggable node the map does not pan: the node moves
  if (!active() || E.drag) return;
  const over = E.tool === 'move' && onNodes(info);
  if (over !== E.panOff) { E.panOff = over; deckgl.setProps({controller: {doubleClickZoom: false, dragPan: !over}}); }
});
onPointer('dragStart', (info) => {
  if (!active() || E.tool !== 'move' || !onNodes(info)) return;
  const i = info.object;
  E.drag = {node: i, from: [E.W.ll[2 * i], E.W.ll[2 * i + 1]]};
});
onPointer('drag', (info) => {
  if (!E.drag || !info.coordinate) return;
  const W = E.W, i = E.drag.node, [lon, lat] = unproject(info.coordinate, S.view);
  if (!(Math.abs(lat) <= 90)) return;
  W.ll[2 * i] = wrap(lon); W.ll[2 * i + 1] = lat;
  for (const e of W.inc[i]) { W.touched.add(e); quality(W, e); }
  showSoon();
});
onPointer('dragEnd', () => {
  if (!E.drag) return;
  const W = E.W, i = E.drag.node, lon = W.ll[2 * i], lat = W.ll[2 * i + 1];
  if (lon !== E.drag.from[0] || lat !== E.drag.from[1]) E.ops.push({op: 'move', node: W.gid[i], lon, lat});
  E.drag = null; E.panOff = false;
  deckgl.setProps({controller: {doubleClickZoom: false}});
  show();
});
onPointer('click', (info) => {
  if (!active() || E.tool !== 'add' || !info.layer || !info.layer.id.startsWith('mesh-window') || info.index < 0 || !info.coordinate) return;
  const G = X.geo[S.view], p = G && G.eOf ? G.eOf[info.index] : info.index;
  const e = X.data.welem ? X.data.welem[p] : p, [lon, lat] = unproject(info.coordinate, S.view);
  const op = {op: 'split', elem: E.W.elems[e].gid, lon: wrap(lon), lat};
  E.ops.push(op); applyOp(E.W, op);
  for (const t of E.W.touched) quality(E.W, t);
  show();
});
addLayers(84, (v) => {                                // the changed elements, outlined
  if (!E.W || X.mode !== 'window' || !E.W.touched.size) return null;
  const W = E.W, paths = [];
  for (const e of W.touched) {
    if (!W.elems[e].alive) continue;
    const p = W.elems[e].n.map((i) => [W.ll[2 * i], W.ll[2 * i + 1]]);
    for (const q of p) q[0] = p[0][0] + wrap(q[0] - p[0][0]);       // unwrapped across ±180 like the cells
    paths.push([...p, p[0]]);                                        // closed by hand (no _pathType 'loop' on plain arrays)
  }
  return new D.PathLayer({id: `edit-touched-${v}`, data: paths, getPath: (d) => d.map((q) => projectPt(q, v, ALT.zone)),
    getColor: [255, 255, 255, 230], getWidth: 2, widthUnits: 'pixels'});
});
addLayers(86, (v) => {                                // the nodes that can be dragged
  if (!active() || E.tool !== 'move') return null;
  const W = E.W, ids = [];
  for (let i = 0; i < W.ll.length / 2; i++) if (movable(W, i)) ids.push(i);
  const drag = E.drag ? E.drag.node : -1;
  return new D.ScatterplotLayer({id: `edit-nodes-${v}`, data: ids, pickable: true, getPosition: (i) => projectPt([W.ll[2 * i], W.ll[2 * i + 1]], v, ALT.points),
    getRadius: 4, radiusUnits: 'pixels', getFillColor: (i) => i === drag ? [255, 214, 0] : [255, 255, 255], stroked: true,
    getLineColor: [0, 0, 0], lineWidthUnits: 'pixels', getLineWidth: 1});
});

// for the headless walkthrough (ui/web/walkthrough.py): page pixels of the draggable node and of an element centre
// nearest the view centre
E.targets = () => {
  const W = E.W, c = xyz(...viewCentre()), vp = deckgl.getViewports()[0], box = $('map').getBoundingClientRect();
  const px = (ll) => { const p = vp.project(projectPt(ll, S.view, ALT.points)); return [box.left + p[0], box.top + p[1]]; };
  let bn = -1, be = -1, dn = -2, de = -2;
  for (let i = 0; i < W.ll.length / 2; i++) {
    const d = dot(xyz(W.ll[2 * i], W.ll[2 * i + 1]), c);
    if (movable(W, i) && d > dn) { dn = d; bn = i; }
  }
  const mid = (e) => unit(corners(W, e).reduce((s, p) => [s[0] + p[0], s[1] + p[1], s[2] + p[2]], [0, 0, 0]));
  W.elems.forEach((el, e) => { if (el.alive && !el.n.includes(bn)) { const d = dot(mid(e), c); if (d > de) { de = d; be = e; } } });
  const ll = (p) => [Math.atan2(p[1], p[0]) / DEG, Math.asin(p[2]) / DEG];
  return {node: px([W.ll[2 * bn], W.ll[2 * bn + 1]]), elem: px(ll(mid(be))), nodeId: W.gid[bn], elemId: W.elems[be].gid};
};
