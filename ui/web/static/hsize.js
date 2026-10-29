// Panel "Size h": the [[size.layers]] list (add, remove, reorder, enable), one form per kind, zones drawn as polygons,
// the preview (h, which layer sets h, rho_c h^2), the node estimate and max |grad h|. After each change the layers' h
// comes first (fast); the run's h (the core's gradient limiter on top, seconds to half a minute) follows in the
// background and replaces it while "show the run's h" is on.
import {$, D, S, api, arcPath, el, inflate, km, kmd, pngImage, rgbCss, status} from './util.js';
import {ALT, addLayers, addReadout, bitmap, deckgl, draw, flyTo, isPolar, onView, pathCopies, projectPt, redraw, unproject} from './map.js';

const KINDS = {
  constant: {title: 'constant', fields: [['h_m', 'h, km', 'km']], make: () => ({kind: 'constant', h_m: 100000.0})},
  latitude: {title: 'latitude law  h = max(h_eq cos φ, h_min)', fields: [['h_m', 'h at the equator, km', 'km'], ['h_min_m', 'h min, km', 'km']],
    make: () => ({kind: 'latitude', h_m: 100000.0, h_min_m: 25000.0})},
  coast_distance: {title: 'distance to coast  h = h_c + g·max(d − plateau, 0)', fields: [['h_m', 'h at the coast, km', 'km'],
    ['plateau_m', 'plateau, km', 'km'], ['grad', 'ramp g = |∇h|', 'num'],
    ['island_max_km2', 'islands: land smaller than, km²', 'num', (ly) => (ly.coasts || 'all') !== 'all']],
  make: () => ({kind: 'coast_distance', h_m: 20000.0, plateau_m: 30000.0, grad: 0.05})},
  zone: {title: 'zone (polygon, great-circle edges)', fields: [['name', 'name', 'text'],
    ['h_m', 'h inside, km', 'km', (ly) => ly.mode !== 'smooth'], ['smooth_m', 'smoothing radius, km', 'km', (ly) => ly.mode === 'smooth'],
    ['transition_m', 'transition width, km', 'km', (ly) => ly.mode !== 'blend']],
  make: () => ({kind: 'zone', name: `zone ${S.cfg.size.layers.filter((l) => l.kind === 'zone').length + 1}`, h_m: 10000.0,
    transition_m: transitionFor(10000.0), polygon: []})},
  orca: {title: "ORCA grid: h = the preset's cell size (√area), at least h min (B101)", fields: [['h_min_m', 'h min, km', 'km']],
    make: () => ({kind: 'orca', orca: 'ORCA1', h_min_m: 10000.0})},
};
const ORCA_PRESETS = ['ORCA2', 'ORCA1', 'ORCA05', 'ORCA025', 'ORCA12'];      // meshom.orca.presets.PRESETS
const H = S.h = {open: -1, overlay: 'h', img: null, polar: {}, winner: null, winnerPolar: {}, rho: null, rhoPolar: {}, rhoStep: 1.0,
  last: null, raw: null, run: null, grid: null, limited: true, pending: false, seq: 0, afterPreview: [], dragIdx: -1,
  autoT: new WeakSet()};   // autoT: zones whose transition follows grad_max until the user types one

// the panels whose overlays (zones, zoom windows) are drawn: setting up h
export const SETUP_TABS = new Set(['h', 'coast', 'method']);
const layersOf = () => S.cfg.size.layers;
// max |∇h| is red above the [size] grad_max the run's limiter enforces; the preview's finite differences read up to
// GRAD_TOL × grad_max on a field limited to grad_max (0.103 for 0.1)
const GRAD_TOL = 1.1;
const gradLimit = () => +S.cfg.size.grad_max;                    // NaN for "inf"
const gradBad = (g) => Number.isFinite(gradLimit()) && g > GRAD_TOL * gradLimit();
// the h raster is drawn in the set-up tabs (h is edited there; an open mesh only over it, faint, when asked for), and
// elsewhere unless a mesh hides it (Mesh, "h raster under the mesh" off)
const hShown = () => SETUP_TABS.has(S.tab) || !(S.mesh && S.mesh.mode !== 'off' && !S.mesh.showH);
// what the run's h depends on: a run result for another key is stale
const hKey = () => JSON.stringify([S.cfg.size, S.cfg.coast, S.cfg.mesh.scale]);

// a new zone's transition: from the base h down to h inside at the steepest slope the limiter allows, (h_out − h_in) /
// grad_max, rounded to 10 km; 300 km when grad_max is inf
function baseH() {
  const c = layersOf().filter((l) => l.kind === 'constant' && l.enabled !== false).map((l) => l.h_m);
  return c.length ? Math.max(...c) : (H.raw ? H.raw.h_max : 100000.0);
}
export function transitionFor(hIn) {
  const g = gradLimit(), base = baseH();
  return Number.isFinite(g) && g > 0 && base > hIn ? Math.max(Math.round((base - hIn) / g / 10000) * 10000, 10000) : 300000.0;
}
// a zone left without a polygon (drawing cancelled, or finished with fewer than 3 clicks) is removed
function dropIfEmpty(ly) {
  const L = layersOf(), i = L.indexOf(ly);
  if (i < 0 || ly.polygon.length) return;
  L.splice(i, 1); H.open = -1; renderLayers(); status('Zone not drawn, so it was removed: a zone needs at least 3 clicks on the map.');
}
const colourOf = (i) => S.layerColours[i % S.layerColours.length];
// zone modes (docs/config.md "Layer order and zone modes"): [the select's text, the summary's word, the tip]
const ZONE_MODES = {
  min: ['refine: h = min(h above, h inside)', 'refine to', 'the default: h inside where it is smaller than the h of the layers above'],
  set: ['set h: h inside whatever the layers above give', 'set h', 'h inside the polygon is this h, also coarser than the layers above; later layers still apply'],
  smooth: ['smooth h: ln h averaged over a radius', 'smooth', 'the h of the layers above, averaged (ln h) over the smoothing radius inside the polygon'],
  blend: ['blend: h inside, the h above outside an outer polygon', 'blend to', 'h inside the polygon; outside the outer polygon the h of the layers above; between the two, ln h changes linearly with the distance (no transition width)'],
};
const COASTS = {all: 'all coasts', continents: 'continents only', islands: 'islands only'};
function summary(ly) {
  const k = ly.kind;
  if (k === 'constant') return `${km(ly.h_m)} km`;
  if (k === 'latitude') return `${km(ly.h_m)} km · cos φ, ≥ ${km(ly.h_min_m)} km`;
  if (k === 'coast_distance') return `${km(ly.h_m)} km + ${ly.grad}·(d − ${km(ly.plateau_m)} km)` + ((ly.coasts || 'all') === 'all' ? '' : ` · ${ly.coasts} only`);
  if (k === 'orca') return `${ly.orca} cells, ≥ ${km(ly.h_min_m)} km`;
  if (k === 'zone' && ly.mode === 'blend') return `${ly.name || ''} · blend to ${km(ly.h_m)} km · ${ly.polygon.length} vertices, outer ${(ly.outer || []).length}`;
  if (k === 'zone') return `${ly.name || ''} · ${ZONE_MODES[ly.mode || 'min'][1]}` +
    (ly.mode === 'smooth' ? ` over ${km(ly.smooth_m)} km` : ` ${km(ly.h_m)} km`) + `, ${km(ly.transition_m)} km ramp · ${ly.polygon.length} vertices`;
  return '';
}

export function renderLayers() {
  const box = $('layers'); box.replaceChildren();
  layersOf().forEach((ly, i) => {
    const on = ly.enabled !== false, open = i === H.open;
    const toggle = () => { H.open = open ? -1 : i; renderLayers(); redraw(); };
    const row = el('div.layer' + (open ? '.open' : '') + (on ? '' : '.off'), {},
      el('div.layer-head', {},
        el('input', {type: 'checkbox', checked: on, title: 'use this layer', on: {change: (e) => {
          if (e.target.checked) delete ly.enabled; else ly.enabled = false; renderLayers(); schedulePreview();
        }}}),
        el('span.swatch', {style: {background: rgbCss(colourOf(i))}}),
        el('span.layer-title', {title: open ? 'close the form' : 'edit this layer', on: {click: toggle}},
          el('span.chev', {}, open ? '▾ ' : '▸ '), el('b', {}, ly.kind), ' ', el('span.muted', {}, summary(ly))),
        el('button.icon', {title: 'move up: the layers apply in this order (a refine layer takes the smaller h; a set or smooth zone acts on the h of the layers above it)', disabled: i === 0, on: {click: () => move(i, -1)}}, '▲'),
        el('button.icon', {title: 'move down', disabled: i === layersOf().length - 1, on: {click: () => move(i, 1)}}, '▼'),
        el('button.icon', {title: 'remove this layer', on: {click: () => { layersOf().splice(i, 1); H.open = -1; renderLayers(); schedulePreview(); }}}, '✕')),
      open ? form(ly, i) : null);
    box.append(row);
  });
}
function move(i, d) {
  const L = layersOf(), [x] = L.splice(i, 1); L.splice(i + d, 0, x);
  if (H.open === i) H.open = i + d;
  renderLayers(); schedulePreview();
}
function form(ly, i) {
  const spec = KINDS[ly.kind] || {fields: []};
  const f = el('div.layer-form', {}, el('div.muted.small', {}, spec.title || ly.kind));
  const inputs = {};
  const hint = el('div.muted.small');
  const showHint = () => {
    hint.textContent = ly.kind === 'zone' && H.autoT.has(ly) && Number.isFinite(gradLimit())
      ? `transition from grad_max: (${kmd(baseH())} − ${kmd(ly.h_m)}) km / ${gradLimit()} ≈ ${km(ly.transition_m)} km` : '';
  };
  if (ly.kind === 'zone') f.append(el('label', {title: ZONE_MODES[ly.mode || 'min'][2]}, 'mode',
    el('select', {on: {change: (e) => {
      const m = e.target.value;
      if (m === 'min') delete ly.mode; else ly.mode = m;
      if (m === 'smooth' && !ly.smooth_m) ly.smooth_m = 500000.0;
      if (m !== 'smooth' && !ly.h_m) ly.h_m = 100000.0;
      if (m === 'blend') { H.autoT.delete(ly); if (!ly.outer) ly.outer = []; }   // blend: no transition width
      if (m !== 'blend') delete ly.outer;
      renderLayers(); schedulePreview();
      if (m === 'blend' && ly.outer.length < 3) status('Blend: draw the outer polygon (the button in the form) around the zone; the h above holds outside it.');
    }}}, Object.entries(ZONE_MODES).map(([k, t]) => el('option', {value: k, selected: (ly.mode || 'min') === k || null, title: t[2]}, t[0])))));
  if (ly.kind === 'orca') f.append(el('label', {title: 'the NEMO ORCA grid whose cell size this layer gives (meshom.orca presets)'}, 'grid',
    el('select', {on: {change: (e) => { ly.orca = e.target.value; renderSummary(i, ly); schedulePreview(); }}},
      [...new Set([...ORCA_PRESETS, ly.orca])].map((k) => el('option', {value: k, selected: ly.orca === k || null}, k)))));
  if (ly.kind === 'coast_distance') f.append(el('label', {title: 'which coasts this layer measures the distance to: islands are land polygons smaller than the threshold below (Antarctica is a continent)'}, 'coasts',
    el('select', {on: {change: (e) => { if (e.target.value === 'all') delete ly.coasts; else ly.coasts = e.target.value; renderLayers(); schedulePreview(); }}},
      Object.entries(COASTS).map(([k, t]) => el('option', {value: k, selected: (ly.coasts || 'all') === k || null}, t)))));
  for (const [key, label, type, show] of spec.fields) {
    if (show && !show(ly)) continue;
    const val = ly[key] ?? (key === 'island_max_km2' ? 100000 : undefined);
    const input = inputs[key] = el('input', type === 'text' ? {type: 'text', value: val ?? ''} : {type: 'number', step: 'any', value: type === 'km' ? km(val) : val});
    // only the edited key is written: untouched values keep their exact TOML value
    input.addEventListener('input', () => {
      ly[key] = type === 'text' ? input.value : type === 'km' ? +input.value * 1000 : +input.value;
      if (ly.kind === 'zone' && key === 'transition_m') H.autoT.delete(ly);
      if (ly.kind === 'zone' && key === 'h_m' && H.autoT.has(ly) && inputs.transition_m) { ly.transition_m = transitionFor(ly.h_m); inputs.transition_m.value = km(ly.transition_m); }
      showHint(); schedulePreview(); renderSummary(i, ly);
    });
    f.append(el('label', {}, label, input));
  }
  if (ly.kind === 'zone') {
    f.append(hint, el('div.row', {},
      el('button', {on: {click: () => draw.start('poly', ({polygon}) => { ly.polygon = polygon; renderLayers(); schedulePreview(); },
        () => dropIfEmpty(ly))}}, 'Draw polygon'),
      ly.mode === 'blend' ? el('button', {title: 'the outer polygon: outside it the h of the layers above holds', on: {click: () => draw.start('poly',
        ({polygon}) => { ly.outer = polygon; renderLayers(); schedulePreview(); })}}, 'Draw outer polygon') : null,
      el('button', {on: {click: () => draw.finish()}}, 'Finish'),
      el('button', {disabled: !ly.polygon.length, on: {click: () => flyToZone(ly)}}, 'Fly to'),
      ly.mode === 'blend' ? null : el('button', {title: 'transition = (base h − h inside) / grad_max', on: {click: () => {
        H.autoT.add(ly); ly.transition_m = transitionFor(ly.h_m); inputs.transition_m.value = km(ly.transition_m);
        showHint(); schedulePreview(); renderSummary(i, ly);
      }}}, 'Transition from grad_max')),
    el('div.muted.small', {}, 'Click on the map to add corners, Finish to close (Esc cancels). While this layer is open, drag its corners to move them.'));
    showHint();
  }
  if (ly.kind === 'coast_distance' && (ly.coasts || 'all') === 'all') f.append(el('div.row', {},
    el('button', {title: 'make this layer "continents only" and add a copy below it for "islands only", to give islands their own h (e.g. coarser, or none)',
      on: {click: () => {
        ly.coasts = 'continents';
        const isl = {...ly, coasts: 'islands', name: 'islands'}; delete isl.enabled;
        layersOf().splice(i + 1, 0, isl); renderLayers(); schedulePreview();
        status('Split: this layer refines at the continents, the new one below at the islands; give the islands their own h (or switch that layer off).');
      }}}, 'Split into continents + islands')));
  return f;
}
function renderSummary(i, ly) { const t = $('layers').children[i]?.querySelector('.layer-title .muted'); if (t) t.textContent = summary(ly); }
function flyToZone(ly) {
  const p = arcPath(ly.polygon, true), lon = p.reduce((a, q) => a + q[0], 0) / p.length, lat = p.reduce((a, q) => a + q[1], 0) / p.length;
  const r = Math.max(...p.map((q) => Math.hypot((q[0] - lon) * Math.cos(lat * Math.PI / 180), q[1] - lat))) * 111;
  flyTo(((lon + 540) % 360) - 180, lat, Math.max(r * 1.3, 200), true);
}

// ---------------------------------------------------------------------------------------------------- preview
// apply(r): a preview answer becomes the drawn raster (its images, polar images and the readout grid)
async function apply(r, polar) {
  H.last = r; H.polar = {}; H.winnerPolar = {};
  [H.img, H.winner] = await Promise.all([pngImage(r.png), pngImage(r.png_winner)]);
  for (const p of polar) if (r['png_' + p]) { H.polar[p] = await pngImage(r['png_' + p]); H.winnerPolar[p] = await pngImage(r['png_winner_' + p]); }
  const g = r.grid;
  H.grid = g ? {...g, h: await inflate(g.h, Float32Array), w: await inflate(g.winner, Uint8Array), names: r.layers, index: r.layer_index,
    run: !!r.limited_applied} : null;
}
let timer = null, runTimer = null;
export function schedulePreview() { clearTimeout(timer); timer = setTimeout(preview, 250); variantNow(); }
export async function preview() {
  const t0 = performance.now(), seq = ++H.seq, polar = isPolar(S.view) ? [S.view] : [];
  H.pending = true; redraw();
  const r = await api('api/preview', {config: S.cfg, polar, limited: false});
  if (seq !== H.seq) return;                         // a newer change is on its way
  H.pending = false;
  if (r.error) { redraw(); return; }
  H.raw = {...r, key: hKey()};
  const run = H.limited && H.run && H.run.key === hKey() && polar.every((p) => H.run['png_' + p]) ? H.run : null;
  await apply(run || r, polar);
  if (H.overlay !== 'rho') { H.rho = null; H.rhoPolar = {}; }
  const ms = performance.now() - t0;
  S.stats.loads.push({what: 'preview', ms, server_s: r.seconds});
  renderStats(ms);
  if (H.overlay === 'rho') await rho();
  for (const f of H.afterPreview) await f(r);
  redraw(); S.showToml();
  if (H.limited && !run) scheduleRun();
  return ms;
}
// the run's h in the background: one request at a time; a change while it runs asks for another when it returns
function scheduleRun() { clearTimeout(runTimer); runTimer = setTimeout(runH, 600); }
async function runH() {
  if (H.runBusy) { H.runAgain = true; return; }
  if (!Number.isFinite(gradLimit()) && S.cfg.mesh.scale === 1) { H.run = {...H.raw}; renderStats(); return; }   // nothing to limit
  H.runBusy = true; renderStats();
  const key = hKey(), polar = isPolar(S.view) ? [S.view] : [], t0 = performance.now();
  const r = await api('api/preview', {config: S.cfg, polar, limited: true});
  H.runBusy = false;
  if (H.runAgain) { H.runAgain = false; scheduleRun(); }
  if (r.error || key !== hKey()) { renderStats(); return; }
  H.run = {...r, key};
  S.stats.loads.push({what: 'preview-run', ms: performance.now() - t0, server_s: r.seconds});
  if (H.limited) { await apply(H.run, polar); redraw(); }
  renderStats();
}
// for scripts (ui/web/walkthrough.py): the run's h of the current config is drawn
H.runReady = () => !H.runBusy && !!H.run && H.run.key === hKey() && H.last === H.run;
export async function rho() {
  const t0 = performance.now();
  const polar = isPolar(S.view) ? [S.view] : [];
  const r = await api('api/rho', {config: S.cfg, step: H.rhoStep, polar, limited: H.limited});
  if (r.error) return;
  H.rho = await pngImage(r.png); H.rhoInfo = r; H.rhoPolar = {};
  for (const p of polar) H.rhoPolar[p] = await pngImage(r['png_' + p]);
  const ms = performance.now() - t0;
  S.stats.loads.push({what: 'rho', ms, server_s: r.seconds, step: H.rhoStep});
  renderStats(); redraw();
  return ms;
}
function renderStats(ms) {
  const raw = H.raw; if (!raw) return;
  const run = H.run && H.run.key === hKey() ? H.run : null, r = H.limited && run ? run : raw;
  const box = $('h_stats'); box.replaceChildren();
  const g = (run || raw).grad_at, grad = (run || raw).grad_max;
  box.append(el('div', {}, `h ${kmd(r.h_min)}–${kmd(r.h_max)} km over water` + (H.limited && run ? " (the run's h)" : H.limited ? " (the layers' h; the run's h follows)" : " (the layers' h)")),
    el('div', {}, `nodes ≈ ${(r.n_nodes_est / 1e6).toFixed(3)} M (∫h⁻² dA)`),
    el('div', {}, 'steepest change of h: ', run && run !== raw ? `|∇h| ${raw.grad_max.toFixed(3)} in the layers → ` : '',
      el('span.' + (gradBad(grad) ? 'bad' : 'good'), {title: `red above grad_max ${S.cfg.size.grad_max} (× ${GRAD_TOL} for the preview's grid)`},
        run && run !== raw ? `${grad.toFixed(3)} in the run` : `|∇h| ${grad.toFixed(3)}`), ` at ${g[0].toFixed(1)}°, ${g[1].toFixed(1)}° `,
      el('button.small', {on: {click: () => flyTo(g[0], g[1], 800, true)}}, 'Fly to')));
  if (H.runBusy) box.append(el('div.muted', {}, "the run's h: computing in the background …"));
  else if (run && run.limited_applied) {
    const L = run.limiter, q = L.limiter;
    box.append(el('div.muted', {}, `the run's h: limited to |∇h| ≤ ${run.grad_limit}` + (run.scale !== 1 ? `, × scale ${run.scale}` : '')),
      el('div.dev.muted', {}, q ? `limiter: cubed-sphere grid n ${q.n}, ${(q.nodes / 1e6).toFixed(2)} M nodes, ${kmd(q.spacing_m)} km` +
        `${q.capped ? ' (capped)' : ''}, ${q.nodes_lowered.toLocaleString()} nodes lowered; ` + (L.cached ? `cached (built in ${L.seconds.toFixed(1)} s)` : `built in ${L.seconds.toFixed(1)} s`) : ''));
  } else if (run) box.append(el('div.muted', {}, "the run's h = the layers' h (grad_max = inf, scale 1)"));
  box.append(el('div.dev.muted', {}, `preview ${raw.seconds.toFixed(2)} s server` + (ms ? `, ${(ms / 1000).toFixed(2)} s round trip` : '')));
  if (!hShown()) box.append(el('div.muted', {}, 'The h raster is hidden while a mesh is drawn (Mesh: "h raster under the mesh").'));
  if (H.overlay === 'rho' && H.rhoInfo) {
    const q = H.rhoInfo;
    box.append(el('div', {}, `ρ_c on ${q.step}° (${q.n_points} points, ${q.seconds.toFixed(2)} s): water ∫ρ⁺ = ${q.plus.toFixed(1)}, ∫ρ⁻ = ${q.minus.toFixed(1)} (cone units)`));
  }
  legend();
}
// colour bar of the current raster (bottom left of the map)
const MESH_LEGEND = {q: ['max angle / gate: (θmax − 90°)/60° quads, (θmax − 60°)/60° triangles', '0 ideal', '1 hard gate'],
  size_ratio: ['element size ratio across edges', '1', '2']};
export function legend() {
  const lg = $('legend'); lg.replaceChildren();
  const m = S.mesh, setup = SETUP_TABS.has(S.tab);
  if (m && m.mode !== 'off' && (!setup || m.overH)) {
    if (m.describe) lg.append(el('div.shown', {}, m.describe()));
    if (m.field === 'kind') lg.append(el('div', {}, el('span.swatch', {style: {background: 'rgb(120,140,170)'}}), ' quad  ',
      el('span.swatch', {style: {background: 'rgb(240,140,30)'}}), ' triangle'));
    else if (m.field === 'depth') lg.append(el('div', {}, "depth, m: the mean of each element's nodes"), el('div.bar.ylgnbu'),
      el('div.ends', {}, el('span', {}, '0 m'), el('span', {}, `≥ ${(m.info && m.info.depth ? m.info.depth.hi : 0).toLocaleString()} m`)));
    else if (m.field === 'size_km') {
      const r = (m.info && m.info.size_km) || {lo: 1, hi: 10}, k = (x) => (+x.toPrecision(2)).toLocaleString();
      lg.append(el('div', {}, "cell size, km: the mean edge of each element (log scale)"), el('div.bar.viridis'),
        el('div.ends', {}, el('span', {}, `≤ ${k(r.lo)}`), el('span', {}, k(Math.sqrt(r.lo * r.hi))), el('span', {}, `≥ ${k(r.hi)} km`)));
    } else lg.append(el('div', {}, MESH_LEGEND[m.field][0]), el('div.bar.rdylbu'),
      el('div.ends', {}, el('span', {}, MESH_LEGEND[m.field][1]), el('span', {}, MESH_LEGEND[m.field][2])));
    const reg = (m.info && m.info.irregular && m.info.irregular.regular) || 4;
    if (m.showIrr && m.irrOn) lg.append(el('div', {}, `irregular nodes (interior, valence ≠ ${reg}): `, el('span.dot.n3'), `n${reg - 1} `, el('span.dot.n5'), `n${reg + 1} `, el('span.dot.nx'), 'other'));
    if (!m.showH && !setup) return;
  }
  const r = H.last; if (!r) return;
  const which = H.limited && H.run && H.last === H.run ? "the run's h" : "the layers' h";
  if (H.overlay === 'h') lg.append(el('div', {}, `h, km (log scale): ${which}`), el('div.bar.viridis'), el('div.ends', {}, el('span', {}, kmd(r.h_min)), el('span', {}, kmd(r.h_max))));
  if (H.overlay === 'winner') {
    lg.append(el('div', {}, 'the layer that sets h'));
    r.layer_index.forEach((li, k) => lg.append(el('div', {}, el('span.swatch', {style: {background: rgbCss(colourOf(li))}}), ` ${li + 1} · ${r.layers[k]}`)));
  }
  if (H.overlay === 'rho' && H.rhoInfo) lg.append(el('div', {}, 'ρ_c h² (irregular spots per cell)'), el('div.bar.rdbu'),
    el('div.ends', {}, el('span', {}, `−${(2 * H.rhoInfo.v0).toPrecision(2)} (n5)`), el('span', {}, `+${(2 * H.rhoInfo.v0).toPrecision(2)} (n3)`)));
  if (SETUP_TABS.has(S.tab) && layersOf().some((l) => l.kind === 'zone' && l.enabled !== false && l.polygon.length > 1))
    lg.append(el('div', {}, 'white lines: refinement zones (yellow: the one being edited; thin: a blend zone\'s outer polygon)'));
  const c = S.cfg.coast;
  if (c.source !== 'none' && S.coast && S.coast.show)
    lg.append(el('div', {}, `black line: the coast (Antarctica: ${{ice_front: 'ice front', grounding: 'grounding line', none: 'left out'}[c.antarctica || 'ice_front']})`));
  if (!(m && m.mode !== 'off')) lg.append(el('div', {}, 'faint lines: graticule every 30°; magenta line: the antimeridian (±180°)'));
}

// ------------------------------------------------------------------------------------------- readout of h
// h under the cursor from the preview's grid, with the layer that sets it (before the limiter)
addReadout((lon, lat) => {
  const G = H.grid;
  if (!G || !hShown() || H.overlay === 'none') return null;
  const i = Math.floor((((lon - G.lon0) % 360) + 360) % 360 / G.step), j = Math.floor((lat - G.lat0) / G.step);
  if (i < 0 || i >= G.nx || j < 0 || j >= G.ny) return null;
  const h = G.h[j * G.nx + i], w = G.w[j * G.nx + i];
  if (!Number.isFinite(h)) return 'land';
  return `h ${h.toPrecision(3)} km${G.run ? " (the run's)" : ''}` + (w < 255 ? ` · set by ${G.index[w] + 1} ${G.names[w]}` : '');
});

// --------------------------------------------------------------------------------------------- map layers
addLayers(10, (v) => {
  if (!hShown()) return null;                                               // a mesh is drawn: no h raster under it
  const polar = isPolar(v);
  const img = {h: polar ? H.polar[v] : H.img, winner: polar ? H.winnerPolar[v] : H.winner, rho: polar ? H.rhoPolar[v] : H.rho}[H.overlay];
  return img ? bitmap(`h-${H.overlay}`, img, v, null, null, H.pending ? 0.45 : 1) : null;   // dimmed while a new one computes
});
addLayers(70, (v) => {
  if (!SETUP_TABS.has(S.tab)) return null;                                  // zones belong to setting up h
  const L = [];
  layersOf().forEach((ly, i) => {
    if (ly.kind !== 'zone' || ly.enabled === false || ly.polygon.length < 2) return;
    const sel = i === H.open, path = arcPath(ly.polygon, !(sel && draw.mode));
    L.push(new D.PathLayer({id: `zone-${i}-${v}`, data: pathCopies(path, v), getPath: (d) => d.map((p) => projectPt(p, v, ALT.zone)),
      getColor: sel ? [255, 220, 0, 255] : [255, 255, 255, 230], getWidth: sel ? 2.5 : 1.5, widthUnits: 'pixels',
      updateTriggers: {getPath: [JSON.stringify(ly.polygon), v]}}));
    if (ly.mode === 'blend' && (ly.outer || []).length > 1) L.push(new D.PathLayer({id: `zone-outer-${i}-${v}`,      // blend: its outer polygon
      data: pathCopies(arcPath(ly.outer, true), v), getPath: (d) => d.map((p) => projectPt(p, v, ALT.zone)),
      getColor: sel ? [255, 220, 0, 160] : [255, 255, 255, 140], getWidth: 1, widthUnits: 'pixels', updateTriggers: {getPath: [JSON.stringify(ly.outer), v]}}));
    if (sel && !draw.mode && S.tab === 'h') L.push(new D.ScatterplotLayer({      // handles only where the form is
      id: `zone-vertices-${v}`, data: ly.polygon.map((p, k) => ({p, k})), getPosition: (d) => projectPt(d.p, v, ALT.zone),
      getRadius: 6, radiusUnits: 'pixels', getFillColor: [255, 210, 0], getLineColor: [0, 0, 0], stroked: true,
      lineWidthUnits: 'pixels', getLineWidth: 1, pickable: true, updateTriggers: {getPosition: [JSON.stringify(ly.polygon), v]},
      onDragStart: (info) => { H.dragIdx = info.index; deckgl.setProps({controller: false}); return true; },
      onDrag: (info) => {
        if (H.dragIdx < 0 || !info.coordinate) return true;
        ly.polygon[H.dragIdx] = unproject(info.coordinate, v).map((x) => +x.toFixed(4)); redraw(); return true;
      },
      onDragEnd: () => { H.dragIdx = -1; deckgl.setProps({controller: true}); renderLayers(); schedulePreview(); return true; },
    }));
  });
  return L;
});
onView(async (v) => {
  if (isPolar(v) && hShown() && (!H.polar[v] || (H.overlay === 'rho' && !H.rhoPolar[v]))) await preview();
});
// the polar raster of the current view, when it was skipped while a mesh hid the h raster
export function ensurePolarH() { if (isPolar(S.view) && hShown() && !H.polar[S.view]) schedulePreview(); }

// ----------------------------------------------------------------------------------------------------- wiring
export function initH() {
  const sel = $('add_kind');
  for (const k of Object.keys(KINDS)) sel.append(el('option', {value: k}, k));
  $('add_layer').onclick = () => {
    const ly = KINDS[sel.value].make(); layersOf().push(ly); H.open = layersOf().length - 1;
    if (ly.kind === 'zone') H.autoT.add(ly);
    renderLayers();
    if (ly.kind === 'zone') draw.start('poly', ({polygon}) => { ly.polygon = polygon; renderLayers(); schedulePreview(); }, () => dropIfEmpty(ly));
    else schedulePreview();
  };
  document.querySelectorAll('input[name=h_overlay]').forEach((r) => r.onchange = async () => {
    H.overlay = r.value; document.body.classList.toggle('rho-on', H.overlay === 'rho');
    if (H.overlay === 'rho' && !H.rho) await rho(); renderStats(); legend(); redraw();
  });
  $('rho_step').onchange = async (e) => { H.rhoStep = +e.target.value; H.rho = null; H.rhoPolar = {}; if (H.overlay === 'rho') await rho(); };
  H.limited = $('limited').checked;
  $('limited').onchange = async (e) => {
    H.limited = e.target.checked; H.rho = null; H.rhoPolar = {};
    const polar = isPolar(S.view) ? [S.view] : [], run = H.run && H.run.key === hKey() ? H.run : null;
    if (H.limited && !run) scheduleRun();
    else if (H.raw) { await apply(H.limited ? run : H.raw, polar); renderStats(); redraw(); }
  };
  $('grad_max').value = S.cfg.size.grad_max;
  $('grad_max').onchange = (e) => { const v = e.target.value.trim(); S.cfg.size.grad_max = v === 'inf' ? 'inf' : +v; schedulePreview(); };
  renderLayers();
}

// ----------------------------------------------------------------------------------------------- h variants (S35)
// The page's [size] saved as its own file (only [size] and its layers) and loaded back, to keep variants of h and
// compare them. While the page's h is a variant as saved or loaded (unchanged), Save writes the config with
// [size] from = that file instead of the layers; once h changes, the layers are written into the config again.
const V = S.variant = {path: null, json: null, dir: ''};
const sizeJson = () => JSON.stringify(S.cfg.size);
export const variantRef = () => (V.path && sizeJson() === V.json ? V.path : null);
function variantNow() {
  const box = $('h_variant_now');
  if (!box) return;
  const name = V.path ? V.path.split('/').at(-1) : '';
  box.textContent = !V.path ? '' : variantRef() ? `· h is variant ${name}` : `· h changed since variant ${name}`;
  S.showToml && S.showToml();
}
async function listVariants() {
  const r = await api('api/size/variants');
  if (r.error) return;
  $('variant_list').replaceChildren(...(r.variants.length ? r.variants.map((v) => el('div.row', {},
    el('span.small.name', {title: v.path, style: {flex: '1'}}, `${v.name} · ${v.layers} layers`),
    el('button.small', {title: `replace the page's h by ${v.path}`, on: {click: () => loadVariant(v.path)}}, 'load'))) :
    [el('div.muted.small', {}, `No variants yet in ${V.dir}.`)]));
}
async function loadVariant(path) {
  const r = await api('api/size/load', {path});
  if (r.error) return;
  S.cfg.size = r.size; V.path = r.path; V.json = sizeJson();
  H.open = -1; renderLayers(); schedulePreview();
  status(`h is now variant ${r.path.split('/').at(-1)}; Save writes the config with [size] from = that file.`);
}
async function saveVariant() {
  const name = $('variant_name').value.trim().replace(/[^\w.-]+/g, '-').replace(/\.toml$/, '');
  if (!name) { status('Give the variant a name first.', true); return; }
  const r = await api('api/size/save', {config: S.cfg, path: `h-${name}.toml`});
  if (r.error) return;
  V.path = r.saved; V.json = sizeJson(); variantNow(); listVariants();
  status(`Saved h as ${r.saved}. Save writes the config with [size] from = that file while h stays as it is.`);
}
export async function initVariants(r) {
  V.dir = r.variant_dir || '';
  $('variant_save').onclick = saveVariant;
  $('h_variants').addEventListener('toggle', () => { if ($('h_variants').open) listVariants(); });
  if (r.size_from) {                                   // the open config refers to a variant: the variant's own h
    const v = await api('api/size/load', {path: r.size_from});
    if (!v.error) { V.path = v.path; V.json = JSON.stringify(v.size); }
  }
  variantNow();
}
