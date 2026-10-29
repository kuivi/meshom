// The Tools window (the map bar's "Tools", in every tab): a ruler - a line clicked on the map, its legs and total
// length along great circles - and the profile of h and of the bottom along that line (server: /api/profile, the h of
// the page's config: the run's h when Size h shows it, else the layers' h; the bottom of [bathymetry] source, else
// GEBCO 2024 next to the repository). The two profiles are two charts on one distance axis (never two y-scales);
// hovering one shows the values of both and the point on the map. With a mesh open, the bottom chart also shows the
// nodes of the elements the line crosses as dots at their mesh depth (its last write, else the run's depth.npy), and
// the hover names the nearest one.
import {$, D, S, api, arcPath, boxRing, el, fmt, status, xyz} from './util.js';
import {ALT, addLayers, draw, pathCopies, projectPt, redraw} from './map.js';
import {defineWin, isOpen, openWin, refreshWin} from './floatwin.js';
import {reloadMesh} from './mesh.js';
import {areaOf, renderRegions} from './coastpanel.js';

const T = S.tools = {pts: [], prof: null, hover: -1, node: -1, drawing: false, sel: null, selFor: null, area: null, selecting: false,
  op: 'min', D: 50, edits: [], editsFor: null};
const OPS = {set: 'set to', min: 'at least', max: 'at most'};
const R_KM = 6371.0;                                     // meshom.geo.R_EARTH_M: the ruler and the profile agree
const SVG = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs = {}, ...kids) => {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  for (const k of kids) if (k) e.append(k);
  return e;
};
function legKm(a, b) {
  const p = xyz(a), q = xyz(b);
  return R_KM * Math.acos(Math.max(-1, Math.min(1, p[0] * q[0] + p[1] * q[1] + p[2] * q[2])));
}
const cumKm = (pts) => pts.reduce((c, p, i) => (c.push(i ? c[i - 1] + legKm(pts[i - 1], p) : 0), c), []);
const kmText = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2));

// ----------------------------------------------------------------------------------------------------- the line
function drawLine() {
  T.drawing = true; T.prof = null; T.hover = -1;
  draw.start('line', ({line}) => { T.pts = line; T.drawing = false; refreshWin('tools'); redraw(); },
    () => { T.drawing = false; refreshWin('tools'); redraw(); },
    (pts) => { T.pts = pts.slice(); refreshWin('tools'); });
  refreshWin('tools');
}
function clearLine() { if (draw.mode === 'line') draw.cancel(false); T.pts = []; T.prof = null; T.hover = -1; T.drawing = false; refreshWin('tools'); redraw(); }
async function profile() {
  if (T.pts.length < 2) return;
  const limited = !!($('limited') && $('limited').checked);
  const mesh = S.mesh && S.mesh.info ? S.mesh.info.path : null;           // the mesh open: its nodes along the line
  const r = await api('api/profile', {config: S.cfg, line: T.pts, n: 600, limited, mesh});
  if (r.error) return;
  T.prof = r; T.hover = -1; T.node = -1; refreshWin('tools'); redraw();
  if (r.note) status(r.note, false, true);
}
function csv() {
  const p = T.prof, rows = ['distance_km,lon,lat,h_km,bottom_m,land'];
  p.d_km.forEach((d, i) => rows.push([d.toFixed(3), p.lon[i].toFixed(5), p.lat[i].toFixed(5), p.h_km[i].toFixed(4),
    p.elev_m ? p.elev_m[i] : '', p.land[i] ? 1 : 0].join(',')));
  const a = el('a', {href: URL.createObjectURL(new Blob([rows.join('\n') + '\n'], {type: 'text/csv'})), download: 'profile.csv'});
  document.body.append(a); a.click(); a.remove();
}
function nodesCsv() {
  const q = T.prof.nodes, rows = ['node,distance_km,offset_km,lon,lat,mesh_depth_m,bottom_m'];
  q.id.forEach((id, i) => rows.push([id, q.d_km[i].toFixed(3), q.off_km[i].toFixed(3), q.lon[i].toFixed(5), q.lat[i].toFixed(5),
    q.depth_m ? q.depth_m[i].toFixed(2) : '', q.elev_m ? q.elev_m[i] : ''].join(',')));
  const a = el('a', {href: URL.createObjectURL(new Blob([rows.join('\n') + '\n'], {type: 'text/csv'})), download: 'profile-nodes.csv'});
  document.body.append(a); a.click(); a.remove();
}
// the node nearest along the line to distance v (km), within 2 % of the line's length; -1 without one
function nearestNode(v) {
  const q = T.prof && T.prof.nodes, d = T.prof && T.prof.d_km;
  if (!q || !q.id.length) return -1;
  let lo = 0, hi = q.d_km.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (q.d_km[m] < v) lo = m; else hi = m; }
  const k = Math.abs(q.d_km[lo] - v) <= Math.abs(q.d_km[hi] - v) ? lo : hi;
  return Math.abs(q.d_km[k] - v) <= 0.02 * d[d.length - 1] ? k : -1;
}

// ---------------------------------------------------------------------------------------------------- the charts
// a nice step for about n ticks over span
function step(span, n = 5) {
  const raw = span / n, p = 10 ** Math.floor(Math.log10(raw)), m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}
const W = 560, Hc = 150, M = {l: 52, r: 10, t: 22, b: 22};
const xhairs = [];                                     // the crosshairs of the drawn charts (moved on hover)
function chart(title, y, {lo, hi, unit, water, line, dots = null}) {
  const d = T.prof.d_km, n = d.length, X = (v) => M.l + (v / d[n - 1]) * (W - M.l - M.r), Y = (v) => M.t + (hi - v) / (hi - lo) * (Hc - M.t - M.b);
  const g = svg('g');
  const sy = step(hi - lo, 4), sx = step(d[n - 1], 6);
  for (let v = Math.ceil(lo / sy) * sy; v <= hi + 1e-9; v += sy) {
    g.append(svg('line', {class: 'grid', x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v)}),
      svg('text', {x: M.l - 6, y: Y(v) + 3, 'text-anchor': 'end'}, document.createTextNode(Math.abs(v) < 1e-9 ? '0' : (+v.toPrecision(4)).toLocaleString())));
  }
  for (let v = 0; v <= d[n - 1] + 1e-9; v += sx) g.append(svg('text', {x: X(v), y: Hc - 6, 'text-anchor': 'middle'}, document.createTextNode(`${(+v.toPrecision(4)).toLocaleString()}`)));
  // land: grey bands where the coast has land under the line
  let a = -1;
  for (let i = 0; i <= n; i++) {
    const on = i < n && T.prof.land[i];
    if (on && a < 0) a = i;
    if (!on && a >= 0) { g.append(svg('rect', {class: 'land', x: X(d[a]), y: M.t, width: Math.max(1, X(d[i - 1]) - X(d[a])), height: Hc - M.t - M.b})); a = -1; }
  }
  if (water && y) {                                    // the water column between sea level and the bottom
    const pts = d.map((v, i) => `${X(v)},${Y(Math.min(y[i], 0))}`).join(' ');
    g.append(svg('polygon', {class: 'water', points: `${X(d[0])},${Y(0)} ${pts} ${X(d[n - 1])},${Y(0)}`}),
      svg('line', {class: 'sea', x1: M.l, x2: W - M.r, y1: Y(0), y2: Y(0)}));
  }
  if (y) g.append(svg('polyline', {class: line, points: d.map((v, i) => `${X(v)},${Y(y[i])}`).join(' ')}));
  if (dots) dots.d.forEach((v, i) => g.append(svg('circle', {class: 'node-dot', r: 2.5, cx: X(v), cy: Y(dots.y[i])})));
  const xh = svg('line', {class: 'xhair', y1: M.t, y2: Hc - M.b, x1: -10, x2: -10});
  const dot = svg('circle', {class: 'xdot', r: 4, cx: -10, cy: -10});
  const ring = dots ? svg('circle', {class: 'node-ring', r: 6, cx: -10, cy: -10}) : null;
  g.append(xh, dot);
  if (ring) g.append(ring);
  xhairs.push({xh, dot, ring, X, Y, y, dots});
  g.append(svg('text', {x: M.l, y: 13, class: 'title'}, document.createTextNode(`${title}, ${unit}`)));
  const s = svg('svg', {viewBox: `0 0 ${W} ${Hc}`, class: 'tools-chart', role: 'img', 'aria-label': `${title} along the line`}, g);
  const hit = svg('rect', {x: M.l, y: M.t, width: W - M.l - M.r, height: Hc - M.t - M.b, fill: 'transparent'});
  hit.addEventListener('mousemove', (e) => {
    const r = s.getBoundingClientRect(), v = ((e.clientX - r.left) * W / r.width - M.l) / (W - M.l - M.r) * d[n - 1];
    let k = 0, lo2 = 0, hi2 = n - 1;
    while (hi2 - lo2 > 1) { const m = (lo2 + hi2) >> 1; if (d[m] < v) lo2 = m; else hi2 = m; }
    k = Math.abs(d[lo2] - v) < Math.abs(d[hi2] - v) ? lo2 : hi2;
    hover(k);
  });
  hit.addEventListener('mouseleave', () => hover(-1));
  s.append(hit);
  return s;
}
function hover(k) {
  if (k === T.hover) return;
  T.hover = k;
  const j = T.node = k < 0 ? -1 : nearestNode(T.prof.d_km[k]);
  for (const c of xhairs) {
    const x = k < 0 ? -10 : c.X(T.prof.d_km[k]);
    c.xh.setAttribute('x1', x); c.xh.setAttribute('x2', x);
    c.dot.setAttribute('cx', c.y ? x : -10); c.dot.setAttribute('cy', k < 0 || !c.y ? -10 : c.Y(c.y[k]));
    if (c.ring) { c.ring.setAttribute('cx', j < 0 ? -10 : c.X(c.dots.d[j])); c.ring.setAttribute('cy', j < 0 ? -10 : c.Y(c.dots.y[j])); }
  }
  const p = T.prof, q = p.nodes, box = $('tools_read');
  const deep = (z) => (z < 0 ? `${(-z).toLocaleString()} m deep` : `land ${z.toLocaleString()} m`);
  if (box) box.textContent = k < 0 ? 'Hover a chart: the values there and the point on the map.' :
    `${kmText(p.d_km[k])} km from point 1 · ${fmt(p.lon[k], 3)}, ${fmt(p.lat[k], 3)} · h ${(+p.h_km[k].toPrecision(4))} km` +
    (p.elev_m ? ` · bottom ${deep(p.elev_m[k])}` : '') + (p.land[k] ? ' · land (coast)' : '') +
    (j < 0 ? '' : `\nnode ${q.id[j]}: mesh depth ${q.depth_m ? `${Math.round(q.depth_m[j]).toLocaleString()} m` : '–'}` +
      (q.elev_m ? ` · bottom at the node ${deep(q.elev_m[j])}` : '') + ` · ${kmText(q.off_km[j])} km off the line`);
  redraw();
}
function charts() {
  xhairs.length = 0;
  const p = T.prof, h = p.h_km, hmax = Math.max(...h), hmin = Math.min(...h);
  const out = [chart(p.limited ? "h, the run's (after the gradient limiter)" : "h, the layers'", h,
    {lo: Math.max(0, hmin - 0.08 * (hmax - hmin + 1e-9)), hi: hmax + 0.08 * (hmax - hmin + 1e-9) + (hmax === hmin ? 1 : 0), unit: 'km', line: 'h-line'})];
  const q = p.nodes && p.nodes.depth_m && p.nodes.id.length ? p.nodes : null;
  if (p.elev_m || q) {
    const zn = q ? q.depth_m.map((v) => -v) : [], all = (p.elev_m || []).concat(zn);
    const zmin = Math.min(...all, 0), zmax = Math.max(...all, 0);
    out.push(chart((p.elev_m ? 'bottom (GEBCO elevation: below 0 the sea floor)' : 'the bottom') + (q ? "; dots: the mesh's nodes at their depth" : ''),
      p.elev_m, {lo: zmin - 0.05 * (zmax - zmin + 1), hi: zmax + 0.05 * (zmax - zmin + 1), unit: 'm', water: true, line: 'elev-line',
        dots: q ? {d: q.d_km, y: zn} : null}));
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------- the window
defineWin('tools', {title: () => 'Tools · ruler, profile, mesh depth', width: 560, height: 640, atRight: true, build: () => {
  const cum = cumKm(T.pts), total = cum.length ? cum[cum.length - 1] : 0;
  const pts = T.pts.length ? el('table.tbl', {}, el('tr', {}, ['point', 'lon, lat', 'leg, km', 'from 1, km'].map((t) => el('th', {}, t))),
    T.pts.map((p, i) => el('tr', {}, el('td', {}, `${i + 1}`), el('td', {}, `${fmt(p[0], 3)}, ${fmt(p[1], 3)}`),
      el('td.num', {}, i ? kmText(cum[i] - cum[i - 1]) : '–'), el('td.num', {}, kmText(cum[i])))))
    : el('div.muted.small', {}, '"Draw a line", then click its points on the map; Finish (or Enter) ends it.');
  return el('div.tools', {},
    el('div.row', {},
      el('button.primary', {title: 'click the line\'s points on the map; Finish or Enter ends it, Esc cancels', disabled: T.drawing, on: {click: drawLine}}, 'Draw a line'),
      el('button', {disabled: !T.drawing, on: {click: () => draw.finish()}}, 'Finish'),
      el('button', {disabled: !T.pts.length, on: {click: clearLine}}, 'Clear'),
      el('span.tools-total', {}, T.pts.length > 1 ? `${kmText(total)} km` : '')),
    el('div.muted.small', {}, 'Lengths along great circles on a sphere of radius 6371 km, as the mesh is measured.'),
    pts,
    el('h4', {}, 'Profile along the line'),
    el('div.row', {},
      el('button', {disabled: T.pts.length < 2 || T.drawing, title: 'h of the page\'s config and the bottom along the line, 600 points', on: {click: profile}}, T.prof ? 'Profile again' : 'Profile'),
      T.prof ? el('button', {title: 'distance, lon, lat, h, bottom, land at the 600 points', on: {click: csv}}, 'Download CSV') : null,
      T.prof && T.prof.nodes && T.prof.nodes.id.length ? el('button', {title: 'the nodes along the line: id, distance, offset, lon, lat, mesh depth, bottom at the node', on: {click: nodesCsv}}, 'Nodes CSV') : null),
    T.prof ? [el('div.tools-charts', {}, charts()),
      el('div.small.pre-line#tools_read', {}, 'Hover a chart: the values there and the point on the map.'),
      el('div.muted.small', {}, T.prof.bathy ? `bottom: ${T.prof.bathy}` : (T.prof.note || '')), nodesNote(T.prof.nodes)] :
      el('div.muted.small', {}, 'h follows Size h ("show the run\'s h" ticked: the run\'s h after the gradient limiter, else the layers\' h); the bottom is the config\'s [bathymetry] source, else GEBCO 2024. Grey bands: land.'),
    el('h4', {}, 'Mesh depth: nodes in an area'), depthSection());
}});

// ------------------------------------------------------------------------------------------ mesh depth: nodes
// the node tool (server: /api/mesh/depth/*, ui/web/depth.py): the nodes of the mesh shown inside a box or a polygon
// drawn on the map, their mesh depth and the bottom grid there; "set to" / "at least" / "at most" D on them goes to
// the mesh's <stem>.depth_edits.json, which the core applies at the next write (C2); the colours show it at once
// (the last write's depth before its edits, with the current ones); "Save as a config rule" adds the area and rule to
// the page's config ([[bathymetry.regions]], C1) for later runs
const meshPath = () => S.mesh.info.path;
function selectNodes(mode) {
  T.selecting = true; refreshWin('tools');
  draw.start(mode, async (a) => {
    T.selecting = false;
    const r = await api('api/mesh/depth/select', {path: meshPath(), ...a});
    if (!r.error) { T.sel = r; T.area = a; T.selFor = meshPath(); }
    refreshWin('tools'); redraw();
  }, () => { T.selecting = false; refreshWin('tools'); });
}
function clearSel() { T.sel = null; T.area = null; refreshWin('tools'); redraw(); }
async function loadEdits() {
  const path = T.editsFor = meshPath();
  const r = await fetch(`api/mesh/depth/edits?path=${encodeURIComponent(path)}`).then((x) => x.json()).catch(() => null);
  if (path !== T.editsFor) return;
  T.edits = r && r.edits ? r.edits : []; refreshWin('tools');
}
async function applyEdit() {
  const r = await api('api/mesh/depth/edit', {path: meshPath(), op: T.op, depth_m: T.D, ...T.area});
  if (r.error) return;
  T.edits = r.edits; T.sel = r.area;
  await reloadMesh(); refreshWin('tools'); redraw();
  status(`${OPS[T.op]} ${T.D} m on ${r.area.n.toLocaleString()} nodes: shown now, written into the model files at the next Write model files.`, false, true);
}
async function dropEdit(index) {                        // index null: the last one
  const r = await api(index === null ? 'api/mesh/depth/undo' : 'api/mesh/depth/remove', {path: meshPath(), index});
  if (r.error) return;
  T.edits = r.edits;
  await reloadMesh();
  if (T.area) { const s = await api('api/mesh/depth/select', {path: meshPath(), ...T.area}); if (!s.error) T.sel = s; }
  refreshWin('tools'); redraw();
}
function saveRule() {
  const b = S.cfg.bathymetry, key = {set: 'depth_m', min: 'min_depth_m', max: 'max_depth_m'}[T.op];
  if (!b.regions) b.regions = [];
  const name = `region ${b.regions.length + 1}`;
  b.regions.push({name, ...areaOf(T.area), [key]: T.D});
  renderRegions(); S.showToml();
  status(`"${name}" (${OPS[T.op]} ${T.D} m) is in the page's config now: Coast, "Depth rules by area" (rename it there); save the config to keep it. It reaches runs made from this config.`, false, true);
}
function editsView() {
  if (!T.edits.length) return null;
  return [el('div.small', {}, el('b', {}, `This mesh's depth edits (${T.edits.length})`),
    el('span.muted', {}, ': applied in this order at the next Write model files')),
  el('table.tbl', {}, el('tr', {}, ['#', 'rule', 'nodes', 'date', ''].map((h) => el('th', {}, h))),
    T.edits.map((x, i) => el('tr', {}, el('td', {}, `${i + 1}`), el('td', {}, `${OPS[x.op] || x.op} ${metres(x.depth_m)}`),
      el('td.num', {}, x.n.toLocaleString()), el('td.small', {}, (x.date || '').replace('T', ' ')),
      el('td', {}, el('button.icon', {title: 'remove this edit', on: {click: () => dropEdit(i)}}, '✕'))))),
  el('div.row', {}, el('button', {on: {click: () => dropEdit(null)}}, 'Undo the last edit'))];
}
const metres = (x) => `${Math.round(x).toLocaleString()} m`;
function depthSection() {
  const open = S.mesh && S.mesh.info;
  if (open && T.editsFor !== meshPath()) loadEdits();
  if (T.sel && (!open || T.selFor !== meshPath())) { T.sel = null; T.area = null; }      // another mesh now
  const r = T.sel;
  const head = el('div.row', {},
    el('button', {disabled: !open || T.selecting, title: 'click two opposite corners on the map', on: {click: () => selectNodes('box')}}, 'Select nodes: box'),
    el('button', {disabled: !open || T.selecting, title: 'click the corners on the map, then Finish', on: {click: () => selectNodes('poly')}}, 'polygon'),
    el('button', {disabled: !T.selecting, on: {click: () => draw.finish()}}, 'Finish'),
    el('button', {disabled: !r, on: {click: clearSel}}, 'Clear'));
  if (!open) return [head, el('div.muted.small', {}, 'Open a mesh (Runs or Reference meshes) to select its nodes.')];
  if (!r) return [head, el('div.muted.small', {}, `The nodes of ${meshPath().split('/').at(-1)} inside a box or a polygon: their mesh depth and the bottom there; then set their depth.`), ...(editsView() || [])];
  if (!r.n) return [head, el('div.muted.small', {}, 'No node of the mesh inside.'), ...(editsView() || [])];
  const inRun = (S.mesh.info.config || '').endsWith('config.resolved.toml');
  const row = (k, v) => el('tr', {}, el('td', {}, k), el('td.num', {}, v));
  const d = r.depth, b = r.bottom;
  return [head, el('table.tbl', {},
    row('nodes', r.n.toLocaleString()),
    d ? row('mesh depth: min · median · mean · max', `${metres(d.min)} · ${metres(d.median)} · ${metres(d.mean)} · ${metres(d.max)}`) : row('mesh depth', 'none yet: write the mesh (Write model files…)'),
    b ? row(`bottom grid at the nodes: shallowest · median · deepest${r.bottom_points < r.n ? ` (${r.bottom_points.toLocaleString()} of them)` : ''}`,
      [b.max, b.median, b.min].map((z) => (z < 0 ? metres(-z) : `land ${metres(z)}`)).join(' · ')) : null,
    r.floor_m !== null && r.at_floor !== null ? row(`at the run's floor (${metres(r.floor_m)})`, r.at_floor.toLocaleString()) : null),
  d ? el('div.muted.small', {}, `mesh depth from ${r.depth_source}`) : null,
  el('div.row.wrap', {},
    el('select#depth_op', {title: 'set to: every node gets D; at least: shallower nodes are deepened to D; at most: deeper nodes are made D deep',
      on: {change: (e) => { T.op = e.target.value; }}}, Object.entries(OPS).map(([k, t]) => el('option', {value: k, selected: T.op === k || null}, t))),
    el('input#depth_d', {type: 'number', step: 'any', value: T.D, style: {width: '6em'}, on: {change: (e) => { T.D = +e.target.value; }}}), 'm',
    el('button.primary#depth_apply', {disabled: !inRun, title: inRun ? "an edit of this mesh: shown at once, written into its model files at the next Write model files"
      : 'this mesh is not in a `meshom run` folder: no write would apply its edits', on: {click: applyEdit}}, `Apply to these ${r.n.toLocaleString()} nodes`),
    el('button#depth_rule', {title: "add this area and rule to the page's config ([[bathymetry.regions]]) for the runs made from it", on: {click: saveRule}}, 'Save as a config rule')),
  r.floor_m !== null ? el('div.muted.small', {}, `A depth below the run's floor (${metres(r.floor_m)}) gives the floor: the model raises such nodes.`) : null,
  ...(editsView() || [])];
}

// what the dots are (or why there are none)
function nodesNote(q) {
  if (!S.mesh || !S.mesh.info) return el('div.muted.small', {}, 'Open a mesh (Runs or Reference meshes) to see its nodes along the line with their depth.');
  if (!q) return null;
  if (!q.id.length) return el('div.muted.small', {}, `${q.mesh}: the line crosses none of its elements.`);
  return el('div.muted.small', {}, `dots: the ${q.id.length.toLocaleString()} nodes of ${q.mesh}'s elements the line crosses, at their distance along the line` +
    (q.depth_m ? `; mesh depth from ${q.depth_source}` : '; no depth for this mesh yet: write it as model files (Mesh, Write model files…)'));
}

// --------------------------------------------------------------------------------------------------------- the map
addLayers(93, (v) => {
  if (T.pts.length < 2 || T.drawing) return null;
  const cum = cumKm(T.pts), at = (p) => projectPt(p, v, ALT.points);
  const L = [new D.PathLayer({id: `tools-line-${v}`, data: pathCopies(arcPath(T.pts, false), v), getPath: (d) => d.map((p) => projectPt(p, v, ALT.zone)),
      getColor: [255, 64, 160, 255], getWidth: 2.5, widthUnits: 'pixels'}),
    new D.ScatterplotLayer({id: `tools-pts-${v}`, data: T.pts.map((p, i) => ({p, i})), getPosition: (x) => at(x.p), getRadius: 5, radiusUnits: 'pixels',
      getFillColor: [255, 64, 160], stroked: true, getLineColor: [255, 255, 255], lineWidthUnits: 'pixels', getLineWidth: 1.5}),
    new D.TextLayer({id: `tools-km-${v}`, data: T.pts.map((p, i) => ({p, t: i ? `${kmText(cum[i])} km` : '0'})), getPosition: (x) => at(x.p),
      getText: (x) => x.t, getSize: 12, getColor: [255, 255, 255], getPixelOffset: [8, -8], getTextAnchor: 'start', fontFamily: 'system-ui, sans-serif',
      outlineWidth: 3, outlineColor: [0, 0, 0, 220], fontSettings: {sdf: true}})];
  const q = T.prof && T.prof.nodes;
  if (q && q.id.length) L.push(new D.ScatterplotLayer({id: `tools-nodes-${v}`, data: q.lon.map((lon, i) => [lon, q.lat[i]]),
    getPosition: (p) => at(p), getRadius: 3, radiusUnits: 'pixels', getFillColor: [216, 27, 96], stroked: true, getLineColor: [255, 255, 255],
    lineWidthUnits: 'pixels', getLineWidth: 1}));
  if (T.prof && T.hover >= 0) L.push(new D.ScatterplotLayer({id: `tools-hover-${v}`, data: [[T.prof.lon[T.hover], T.prof.lat[T.hover]]],
    getPosition: (p) => at(p), getRadius: 8, radiusUnits: 'pixels', filled: false, stroked: true, getLineColor: [255, 64, 160], lineWidthUnits: 'pixels', getLineWidth: 3}));
  if (q && T.node >= 0) L.push(new D.ScatterplotLayer({id: `tools-node-${v}`, data: [[q.lon[T.node], q.lat[T.node]]],
    getPosition: (p) => at(p), getRadius: 7, radiusUnits: 'pixels', filled: false, stroked: true, getLineColor: [216, 27, 96], lineWidthUnits: 'pixels', getLineWidth: 2.5}));
  return L;
});

export function initTools() { $('tools_open').onclick = () => (isOpen('tools') ? refreshWin('tools') : openWin('tools')); }
// the node tool's area (a box follows its parallels and meridians, a polygon great circles) and the selected nodes
addLayers(94, (v) => {
  if (!T.sel || !T.area) return null;
  const a = T.area, at = (p) => projectPt(p, v, ALT.points);
  let path;
  if (a.box) { path = boxRing(a.box); path.push(path[0].slice()); } else path = arcPath(a.polygon, true);
  return [new D.PathLayer({id: `tools-area-${v}`, data: pathCopies(path, v), getPath: (d) => d.map((p) => projectPt(p, v, ALT.zone)),
      getColor: [0, 200, 230, 255], getWidth: 2, widthUnits: 'pixels'}),
    new D.ScatterplotLayer({id: `tools-sel-${v}`, data: T.sel.lon.map((lon, i) => [lon, T.sel.lat[i]]), getPosition: (p) => at(p),
      getRadius: 2.5, radiusUnits: 'pixels', getFillColor: [0, 200, 230], stroked: false})];
});
