// Panel 4 · Blocks: the hand-placed blocks of the config ([[blocks]], built by the m14 methods as fixed blocks;
// docs/config.md "[[blocks]]"), drawn on the map (four corners clicked, made counter-clockwise; dragged by their numbered
// dots) and edited in a form (the sides' kinds or the core's rule, rows / cols / h_m) or as TOML text; and the m14 block
// layout of the loaded run (m14.json next to the mesh): seeds,
// sizes, outlines in lon-lat and their corners, drawn in this panel and, with the Mesh panel's "m14 blocks" tick, over
// the mesh.
import {$, D, S, api, arcPath, el, fmt, status, unwrapPath, wrapLon, xyz} from './util.js';
import {ALT, addLayers, deckgl, draw, flyTo, onPointer, pathCopies, projectPt, redraw, unproject} from './map.js';
import {legend} from './hsize.js';

const B = S.blocks = {run: null, data: null, show: true, onMesh: false};
const EXAMPLE = `# hand-placed blocks, one [[blocks]] table each (the m14 methods build them first; docs/config.md)
# [[blocks]]
# name = "Atlantic"
# corners = [[-42.0, -20.0], [15.0, -20.0], [-7.0, 40.0], [-76.0, 40.0]]   # [lon, lat], counter-clockwise
# sides = ["lat", "cut", "lat", "cut"]   # coast / cut / lat / offset; default: coast along a coast, else cut
# rows = 64                              # cells along side 1-2 (default: from h)
# cols = 70                              # cells along side 2-3
`;

export async function showBlocksText() {
  if (!S.cfg.blocks) { $('blocks_text').value = EXAMPLE; return; }
  const r = await api('api/toml', {config: {blocks: S.cfg.blocks}});
  $('blocks_text').value = r.toml;
}
async function apply() {
  const r = await api('api/toml', {text: $('blocks_text').value});
  if (r.error) { status(`[blocks]: ${r.error}`, true); return; }
  if (r.value.blocks && Object.keys(r.value.blocks).length) S.cfg.blocks = r.value.blocks; else delete S.cfg.blocks;
  status('[blocks] applied to the config'); S.showToml(); renderHand(); redraw();
}
export async function loadBlocks(runDir) {
  B.run = runDir;
  B.data = runDir ? await api(`api/runs/blocks?path=${encodeURIComponent(runDir)}`) : null;
  $('mesh_blocks_row').hidden = !(B.data && B.data.blocks.length);
  render(); redraw();
}
function render() {
  const box = $('blocks_run'); box.replaceChildren();
  const d = B.data;
  if (!d || !d.blocks.length) { box.append(el('div.muted', {}, B.run ? `no m14.json in ${B.run}` : 'load a run of an m14 method (Runs) to see its blocks')); return; }
  box.append(el('div', {}, `${d.method}: ${d.n_blocks} blocks kept of ${d.blocks.length} regions, coverage ${fmt(d.coverage_area * 100, 1)} % of the water`),
    d.note ? el('div.muted.small', {}, d.note) : el('div.muted.small', {}, 'outlines: white (kept) / grey (dropped), corners: yellow'),
    el('table.tbl', {}, el('tr', {}, el('th', {}, '#'), el('th', {}, 'kept'), el('th', {}, 'seed'), el('th', {}, 'cells'), el('th', {}, '∫ρ_c'), el('th', {}, 'outline')),
      d.blocks.map((b) => el('tr', {on: {click: () => b.seed && flyTo(b.seed[0], b.seed[1], 1500)}},
        el('td', {}, b.i), el('td', {}, b.kept ? '✓' : '·'), el('td', {}, b.seed ? `${fmt(b.seed[0], 1)}, ${fmt(b.seed[1], 1)}` : '–'),
        el('td', {}, b.cells ?? '–'), el('td', {}, fmt(b.B, 2)),
        el('td', {}, b.outline ? `${b.outline.length} nodes, ${(b.corners || []).length} corners` : '–')))));
}
export function initBlocks() {
  $('blocks_apply').onclick = apply;
  $('blocks_draw').onclick = drawBlock;
  renderHand();
  $('blocks_show').onchange = (e) => { B.show = e.target.checked; redraw(); };
  $('mesh_show_blocks').onchange = (e) => { B.onMesh = e.target.checked; redraw(); legend(); };
  showBlocksText(); render();
}
addLayers(80, (v) => {
  const d = B.data;
  if (!d || !d.blocks.length || !(S.tab === 'blocks' ? B.show : S.tab === 'mesh' && B.onMesh)) return null;   // panel 4, or Mesh ticked
  const seeds = d.blocks.filter((b) => b.seed);
  const L = [new D.ScatterplotLayer({id: `block-seeds-${v}`, data: seeds, getPosition: (b) => projectPt(b.seed, v, ALT.points),
    getRadius: 7, radiusUnits: 'pixels', getFillColor: (b) => b.kept ? [255, 255, 255] : [150, 150, 150], getLineColor: [0, 0, 0],
    stroked: true, lineWidthUnits: 'pixels', getLineWidth: 2}),
  new D.TextLayer({id: `block-labels-${v}`, data: seeds, getPosition: (b) => projectPt(b.seed, v, ALT.points), getText: (b) => `B${b.i}`,
    getSize: 12, getColor: [255, 255, 255], getPixelOffset: [10, 0], getTextAnchor: 'start', fontFamily: 'system-ui, sans-serif'})];
  const outl = d.blocks.filter((b) => b.outline && b.outline.length > 2);
  if (outl.length) {
    // outline vertices are block-grid nodes (spacing ~h): unwrap the longitudes so a block across 180° stays whole
    const paths = outl.flatMap((b) => pathCopies(unwrapPath([...b.outline, b.outline[0]].map((p) => p.slice())), v).map((p) => ({p, kept: b.kept})));
    L.push(new D.PathLayer({id: `block-outlines-${v}`, data: paths, getPath: (x) => x.p.map((q) => projectPt(q, v, ALT.zone)),
      getColor: (x) => x.kept ? [255, 255, 255, 235] : [170, 170, 170, 200], getWidth: (x) => x.kept ? 2 : 1, widthUnits: 'pixels'}));
    const corners = outl.flatMap((b) => (b.corners || []).map((k) => b.outline[k]));
    L.push(new D.ScatterplotLayer({id: `block-corners-${v}`, data: corners, getPosition: (p) => projectPt(p, v, ALT.points),
      getRadius: 3.5, radiusUnits: 'pixels', getFillColor: [255, 210, 0], stroked: true, getLineColor: [0, 0, 0], lineWidthUnits: 'pixels', getLineWidth: 1}));
  }
  return L;
});

// ------------------------------------------------------------------------------ hand blocks of the config ([[blocks]])
// S.cfg.blocks holds the set keys of each [[blocks]] table only: an unset key is the core's default (sides by its rule,
// rows / cols from h, h_m the run's h). The sides are drawn by kind: cut a great-circle arc, lat a parallel (linear in
// lon-lat), coast and offset straight here (the run lays them along the coast), by the rule grey.
const SIDE_KINDS = ['cut', 'coast', 'lat', 'offset'];
const SIDE_RGB = {cut: [255, 255, 255], coast: [0, 200, 255], lat: [255, 214, 0], offset: [255, 140, 0], rule: [175, 175, 175]};
const HB = {drag: null, panOff: false};
const hand = () => S.cfg.blocks || [];
// corners counter-clockwise seen from outside the sphere (the block on the left of each side): the sum of the sides'
// cross products points outwards
function ccw(c) {
  const p = c.map(xyz), m = [0, 1, 2].map((k) => p.reduce((s, q) => s + q[k], 0));
  let s = 0;
  for (let i = 0; i < p.length; i++) {
    const a = p[i], b = p[(i + 1) % p.length];
    s += (a[1] * b[2] - a[2] * b[1]) * m[0] + (a[2] * b[0] - a[0] * b[2]) * m[1] + (a[0] * b[1] - a[1] * b[0]) * m[2];
  }
  return s > 0;
}
const round4 = (p) => [+wrapLon(p[0]).toFixed(4), +p[1].toFixed(4)];
const centre = (c) => { const v = c.map(xyz).reduce((s, q) => s.map((x, k) => x + q[k]), [0, 0, 0]);
  return [Math.atan2(v[1], v[0]) * 180 / Math.PI, Math.atan2(v[2], Math.hypot(v[0], v[1])) * 180 / Math.PI]; };
function changed(msg) { showBlocksText(); S.showToml(); renderHand(); redraw(); if (msg) status(msg); }
function drawBlock() {
  draw.start('quad', ({corners}) => {
    let c = corners.map(round4), note = '';
    if (!ccw(c)) { c = [c[0], c[3], c[2], c[1]]; note = ' Its corners came clockwise: their order was reversed (corner 1 kept).'; }
    S.cfg.blocks = [...hand(), {name: `block ${hand().length + 1}`, corners: c}];
    changed(`Block ${S.cfg.blocks.length} added to [[blocks]].${note} Set its sides and counts below; drag a numbered corner to move it.`);
  });
}
function setKey(b, k, v) { if (v === null || Number.isNaN(v)) delete b[k]; else b[k] = v; changed(); }
function renderHand() {
  const box = $('blocks_hand');
  if (!box) return;
  const bs = hand();
  if (!bs.length) { box.replaceChildren(el('div.muted.small', {}, 'No hand blocks in this config. "Draw a block", then click its four corners on the map.')); return; }
  box.replaceChildren(...bs.map((b, i) => {
    const rule = !b.sides;
    const num = (k, label, tip, toKey, fromKey) => el('label.inline.small', {title: tip}, label + ' ',
      el('input', {type: 'number', step: 'any', value: b[k] === undefined || b[k] === null ? '' : fromKey(b[k]), placeholder: 'auto', style: {width: '5em'},
        on: {change: (e) => setKey(b, k, e.target.value === '' ? null : toKey(+e.target.value))}}));
    return el('div.hand-block', {},
      el('div.row', {},
        el('input', {type: 'text', value: b.name || '', title: 'the block\'s name (shown in the records)', style: {flex: '1'},
          on: {change: (e) => { b.name = e.target.value; changed(); }}}),
        el('button.small', {title: 'fly to this block', on: {click: () => flyTo(...centre(b.corners), 900, true)}}, 'fly to'),
        el('button.small', {title: 'remove this block from [[blocks]]', on: {click: () => {
          bs.splice(i, 1); if (!bs.length) delete S.cfg.blocks; changed(`Block ${i + 1} removed from [[blocks]].`);
        }}}, 'remove')),
      el('div.muted.small', {}, b.corners.map((p, k) => `${k + 1}: ${fmt(p[0], 2)}, ${fmt(p[1], 2)}`).join('   ')),
      el('label.check.small', {title: 'the core\'s rule per side: coast when both corners lie within 1 h of one coast loop and the coast between them is shorter than 3 × the chord, else cut'},
        el('input', {type: 'checkbox', checked: rule || null, on: {change: (e) => {
          if (e.target.checked) delete b.sides; else b.sides = ['cut', 'cut', 'cut', 'cut']; changed();
        }}}), 'sides by the core\'s rule'),
      rule ? null : el('div.row.wrap', {}, [0, 1, 2, 3].map((k) => el('label.inline.small', {title: `side ${k + 1}: from corner ${k + 1} to corner ${(k + 1) % 4 + 1}`},
        `${k + 1}→${(k + 1) % 4 + 1} `, el('select', {on: {change: (e) => { b.sides = b.sides.map((s, j) => j === k ? e.target.value : s); changed(); }}},
          SIDE_KINDS.map((s) => el('option', {value: s, selected: b.sides[k] === s || null}, s)))))),
      el('div.row.wrap', {},
        num('rows', 'rows', 'cells along sides 1→2 and 3→4 (empty: from h)', Math.round, (x) => x),
        num('cols', 'cols', 'cells along sides 2→3 and 4→1 (empty: from h)', Math.round, (x) => x),
        num('h_m', 'h km', 'the cell size inside, km (empty: the run\'s h)', (x) => x * 1000, (x) => +(x / 1000).toPrecision(6))));
  }));
}
function sidePath(a, b, kind) {
  if (kind !== 'lat') return arcPath([a, b], false);
  const d = ((b[0] - a[0] + 540) % 360) - 180, n = Math.max(2, Math.ceil(Math.abs(d)));
  return Array.from({length: n + 1}, (_, s) => [a[0] + d * s / n, a[1] + (b[1] - a[1]) * s / n]);
}
addLayers(81, (v) => {
  const bs = hand();
  if (S.tab !== 'blocks' || !bs.length) return null;
  const sides = bs.flatMap((b, i) => b.corners.map((a, k) => ({i, kind: b.sides ? b.sides[k] : 'rule', path: sidePath(a, b.corners[(k + 1) % 4], b.sides ? b.sides[k] : 'cut')})))
    .flatMap((s) => pathCopies(s.path, v).map((p) => ({...s, p})));
  const corners = bs.flatMap((b, i) => b.corners.map((p, k) => ({i, k, p})));
  const at = (p) => projectPt(p, v, ALT.points);
  return [
    new D.PathLayer({id: `hand-sides-${v}`, data: sides, getPath: (x) => x.p.map((q) => projectPt(q, v, ALT.zone)),
      getColor: (x) => SIDE_RGB[x.kind], getWidth: 2.5, widthUnits: 'pixels'}),
    new D.ScatterplotLayer({id: `hand-corners-${v}`, data: corners, pickable: true, getPosition: (c) => at(c.p), getRadius: 8,
      radiusUnits: 'pixels', getFillColor: [255, 255, 255], stroked: true, getLineColor: [0, 0, 0], lineWidthUnits: 'pixels', getLineWidth: 2}),
    new D.TextLayer({id: `hand-numbers-${v}`, data: corners, getPosition: (c) => at(c.p), getText: (c) => `${c.k + 1}`, getSize: 11,
      getColor: [0, 0, 0], getTextAnchor: 'middle', getAlignmentBaseline: 'center', fontFamily: 'system-ui, sans-serif', fontWeight: 700}),
    new D.TextLayer({id: `hand-names-${v}`, data: bs.map((b, i) => ({c: centre(b.corners), t: b.name || `block ${i + 1}`})),
      getPosition: (x) => at(x.c), getText: (x) => x.t, getSize: 12, getColor: [255, 255, 255], fontFamily: 'system-ui, sans-serif',
      getTextAnchor: 'middle', getAlignmentBaseline: 'center', outlineWidth: 3, outlineColor: [0, 0, 0, 200], fontSettings: {sdf: true}}),
  ];
});
// a numbered corner is dragged (the map does not pan while the pointer is over one)
const onCorner = (info) => (info.layer && info.layer.id.startsWith('hand-corners') && info.object) || null;
function pan(on) { if (HB.panOff === !on) return; HB.panOff = !on; deckgl.setProps({controller: on ? true : {dragPan: false}}); }
onPointer('hover', (info) => { if (!HB.drag) pan(!(S.tab === 'blocks' && !draw.mode && onCorner(info))); });
onPointer('dragStart', (info) => { const c = S.tab === 'blocks' && !draw.mode && onCorner(info); if (c) HB.drag = {i: c.i, k: c.k}; });
onPointer('drag', (info) => {
  if (!HB.drag || !info.coordinate) return;
  const p = unproject(info.coordinate, S.view);
  if (!(Math.abs(p[1]) <= 90)) return;
  hand()[HB.drag.i].corners[HB.drag.k] = round4(p); redraw();
});
onPointer('dragEnd', () => {
  if (!HB.drag) return;
  const b = hand()[HB.drag.i];
  HB.drag = null; pan(true);
  changed(ccw(b.corners) ? '' : `Block "${b.name}": its corners are now clockwise; the core needs them counter-clockwise (the block on the left of each side): move the corner back.`);
});
