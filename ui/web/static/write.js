// "Write model files" (Mesh panel, and the hand editor after a save): a saved mesh of a `meshom run` folder (raw,
// smooth, optimized, repaired-NN, edited-NN) written as model files by the core's `meshom write` (server:
// /api/mesh/write, ui/web/write.py): the run's format or another, into <run>/fesom-<mesh>/, the depth from the run's
// [bathymetry], the mesh scored with the run's gates; FESOM3's fesom_ini after it when ticked ([mesh] fesom3_ini). The
// write starts only from the window's Write button and runs on the server, one at a time (the first write of a run
// computes its size field and depth, minutes for a large mesh: 288 s for a 281 k-node Arctic mesh on 8 CPUs; later
// writes read them from the run's cache/, ~30 s after a small edit); while it runs the window tells the stage. Then it
// shows the folder, the format's checks (what stops the model first), the gates, the straits and the narrow passages.
// A mesh written before shows its last record from the folder.
import {$, S, api, el, fmt, status} from './util.js';
import {flyTo} from './map.js';
import {gatesView, reloadMesh} from './mesh.js';
import {defineWin, isOpen, openWin, refreshWin} from './floatwin.js';

const W = S.write = {path: null, t: null, err: null, fmt: null, ini: true, job: null, timer: null};
const POLL_MS = 1000;
const base = (p) => p.split('/').at(-1);
const dir = (p) => p.split('/').slice(0, -1).join('/');
// write_mesh's stages (its progress callback) in words, for the busy line
const STAGE = {size: "the run's size field h", score: "the scores with the run's gates", bathymetry: 'the depth at the nodes',
  write: "the files and the format's checks", straits: 'the straits', passages: 'the narrow passages', fesom_ini: 'fesom_ini'};
const PASSAGE_KIND = {pools: 'between two basins', island: 'around an island'};

// the window for a mesh file (path): the run's meshes, the format, the last write
export function openWrite(path) {
  if (path !== W.path) { W.path = path; W.t = null; W.err = null; W.fmt = null; }
  openWin('write');
}
async function load() {
  if (!W.path) return;
  const r = await fetch(`api/mesh/write?path=${encodeURIComponent(W.path)}`).then((x) => x.json()).catch((e) => ({error: String(e)}));
  if (r.error) { W.t = null; W.err = r.error; refreshWin('write'); return; }
  W.t = r; W.err = null; W.job = r.job;
  if (!W.fmt) W.fmt = r.format;
  refreshWin('write');
  if (r.job.state === 'running') poll();
}
// the job, every POLL_MS while it runs (plain fetch: no "busy" sign for the polls)
function poll() {
  if (W.timer) return;
  W.timer = setInterval(async () => {
    const j = await fetch('api/mesh/write/job').then((x) => x.json()).catch(() => null);
    if (!j) return;
    W.job = j;
    if (j.state !== 'running') {
      clearInterval(W.timer); W.timer = null;
      if (j.state === 'done') {
        const s = j.summary, c = s.checks;
        status(`Wrote ${s.mesh} as ${s.format} into ${s.out}: ` + (c ? (c.ok ? 'the checks pass' : `${c.stop.length} check(s) stop the model`) : 'no checks for this format') +
          `, gates ${s.gates_passed} of ${s.gates_total} passed.`, !!(c && !c.ok), true);
        if (S.mesh.info && S.mesh.info.path === j.mesh) await reloadMesh();     // the mesh shown: its depth is this write's
      } else if (j.state === 'error') status(`The write of ${base(j.mesh)} failed: ${j.error}`, true, true);
      if (isOpen('write')) await load();
    }
    refreshWin('write');
  }, POLL_MS);
}
async function start() {
  const t = W.t, ini = W.fmt === 'fesom3' && !!t.fesom3_ini && W.ini;
  const r = await api('api/mesh/write', {path: W.path, format: W.fmt, fesom_ini: ini});
  if (r.error) return;
  W.job = r; refreshWin('write'); poll();
}

// ------------------------------------------------------------------------------------------------ the record
const yes = (b) => (b ? 'yes' : 'no');
function checksView(c, format) {
  if (!c) return el('div.small', {}, `${format}: this format has no model checks (FESOM-C reads the files as they are).`);
  const name = c.name.replace('_checks', '').toUpperCase();
  const items = el('details', {}, el('summary.small', {}, `every ${name} check (${c.items.length})`),
    el('table.tbl.gates', {}, c.items.map((x) => el('tr', {}, el('td.small', {}, x.key), el('td.num', {}, Array.isArray(x.value) ? x.value.join(', ') : String(x.value))))));
  if (c.ok) return el('div', {}, el('div', {}, el('span.pass-text', {}, '✓ '), el('b', {}, `${name} checks: nothing stops the model`)), items);
  return el('div', {}, el('div', {}, el('span.fail-text', {}, '✗ '), el('b', {}, `${name} checks: ${c.stop.length} stop the model`)),
    el('table.tbl.gates', {}, el('tr', {}, ['', 'check', 'value'].map((h) => el('th', {}, h))),
      c.stop.map((x) => el('tr', {title: x.key}, el('td.fail', {}, '✗ stops'), el('td', {}, x.what), el('td.num', {}, String(x.value))))), items);
}
function straitsView(st) {
  if (!st) return null;
  const bad = st.rows.filter((r) => !r.ok);
  const head = st.not_connected.length ? `${st.not_connected.length} not connected: ${st.not_connected.join(', ')}`
    : bad.length ? `all connected, ${bad.length} narrower than asked` : 'all connected and wide enough';
  const head2 = () => el('tr', {}, ['', 'strait', 'connected', 'cells across', 'narrowest', 'depth'].map((h) => el('th', {}, h)));
  const good = st.rows.filter((r) => r.ok);
  return el('div', {}, el('b', {}, `Straits (${st.rows.length}): ${head}`),
    el('div.muted.small', {}, `water: ${st.n_components} connected part${st.n_components === 1 ? '' : 's'}` + (st.connectivity_ok ? '' : ' · connectivity check failed')),
    bad.length ? el('table.tbl.gates', {}, head2(), bad.map(straitRow)) : null,
    good.length ? el('details', {}, el('summary.small', {}, `the ${good.length} straits that pass`), el('table.tbl.gates', {}, head2(), good.map(straitRow))) : null);
}
// a strait's row: a click flies to its narrowest section
function straitRow(r) {
  return el('tr', {title: r.at ? `click: fly to the narrowest section (${fmt(r.at[0], 2)}, ${fmt(r.at[1], 2)})` : '',
    on: {click: () => r.at && flyTo(r.at[0], r.at[1], 300, true)}},
    el('td.' + (r.ok ? 'pass' : 'fail'), {}, r.ok ? '✓' : '✗'), el('td', {}, r.name), el('td', {}, yes(r.connected)),
    el('td.num', {}, `${fmt(r.cells_min, 1)} of ${r.cells_target}` + (r.closed_sections ? ` · ${r.closed_sections} closed` : '')),
    el('td.num', {}, `${fmt(r.width_min_m / 1000, 1)} km`), el('td.num', {}, r.depth_min_m === null || r.depth_min_m === undefined ? '–' : `${Math.round(r.depth_min_m)} m`));
}
// the through-passages narrower than [coast] passage_min_cells (C51): a warning (the mesh is written as it is); a row
// flies there; a long list is folded
function passagesView(p) {
  if (!p) return null;
  if (!p.n_narrow) return el('div', {}, el('span.pass-text', {}, '✓ '), el('b', {}, `Narrow passages: none below ${p.cells} cells across`));
  const rows = el('table.tbl.gates', {}, el('tr', {}, ['where', 'cells across', 'h', 'kind', 'strait'].map((h) => el('th', {}, h))),
    p.narrow.map((x) => el('tr', {title: x.at ? 'click: fly there' : '', on: {click: () => x.at && flyTo(x.at[0], x.at[1], 300, true)}},
      el('td.num', {}, x.at ? `${fmt(x.at[0], 2)}, ${fmt(x.at[1], 2)}` : '?'), el('td.num', {}, String(x.cells_across ?? '?')),
      el('td.num', {}, x.h_m ? `${fmt(x.h_m / 1000, 1)} km` : '–'), el('td', {}, PASSAGE_KIND[x.kind] || x.kind || ''), el('td', {}, x.near || ''))));
  return el('div', {}, el('div', {}, el('span.warn', {}, '⚠ '), el('b', {}, `Narrow passages: ${p.n_narrow} below ${p.cells} cells across`),
    el('span.muted.small', {}, " ([coast] passage_min_cells; the mesh is written as it is, a passage this narrow may close in the model)")),
  p.n_narrow <= 10 ? rows : el('details', {}, el('summary.small', {}, `the ${p.n_narrow} passages`), rows));
}
function iniView(f) {
  if (!f) return null;
  return el('div', {}, el('div', {}, el('span.' + (f.ok ? 'pass-text' : 'fail-text'), {}, f.ok ? '✓ ' : '✗ '),
    el('b', {}, f.ok ? 'fesom_ini ran: the derived files (edges, levels) are written' : 'fesom_ini did not finish'),
    el('span.muted.small', {}, ` · ${fmt(f.wall_s, 1)} s, exit ${f.exit_status}`)),
  f.stop_lines && f.stop_lines.length ? el('div.small', {}, f.ok ? el('span.muted', {}, 'its last words, as expected with 1 partition (it stops at the partitioning):') : 'where it stopped:',
    el('pre.small', {}, f.stop_lines.join('\n'))) : null,
  el('div.muted.small', {}, `log: ${f.log}`));
}
// the depth rules by area ([[bathymetry.regions]]) and the node edits the write applied
function depthView(s) {
  const reg = s.regions || [], de = s.depth_edits;
  if (!reg.length && !de) return null;
  const floor = (k) => (k ? `, ${k} raised to the floor` : '');
  return el('div', {}, el('b', {}, 'Depth rules and edits applied'),
    reg.length ? el('div.small', {}, reg.map((r) => `${r.name || 'region'}: ${r.n_changed} of ${r.n_nodes} nodes changed${floor(r.n_clamped_to_floor)}`).join(' · ')) : null,
    de ? el('div.small', {}, `node edits: ${de.n_edits}, ${de.n_nodes_changed} nodes changed${floor(de.n_clamped_to_floor)}${de.n_clamped_to_floor ? ` (${Math.round(de.floor_m)} m)` : ''}`) : null);
}
function recordView(s) {
  return el('div.write-rec', {},
    el('div', {title: s.out}, el('b', {}, `${s.mesh} → ${base(s.out)}/`), el('span.muted.small', {}, dir(s.out) === dir(W.path) ? ' in the run folder' : ` in ${dir(s.out)}`)),
    el('div.muted.small', {}, `${s.format} · ${(s.n_nodes ?? 0).toLocaleString()} nodes, ${(s.n_quads ?? 0).toLocaleString()} quads, ${(s.n_tris ?? 0).toLocaleString()} triangles` +
      ` · Euler ${s.euler_ok ? 'ok' : 'FAILED'} · ${s.date ? s.date.replace('T', ' ') : ''} · ${fmt(s.wall_s, 1)} s`),
    checksView(s.checks, s.format), iniView(s.fesom_ini), depthView(s),
    s.gates && s.gates.length ? gatesView(s.gates, s) : el('div.muted.small', {}, 'no gates'),
    straitsView(s.straits), passagesView(s.passages));
}

// ------------------------------------------------------------------------------------------------ the window
function build() {
  if (W.err) return el('div.write', {}, el('div.warn', {}, W.err),
    el('div.muted.small', {}, 'Model files are written from a `meshom run` folder: its config.resolved.toml gives the format, the bathymetry and the gates.'));
  const t = W.t;
  if (!t) return el('div.muted', {}, 'Reading the run folder…');
  const name = base(W.path), m = t.meshes.find((x) => x.name === name) || {out: `${t.run_dir}/fesom-${name.replace('.npz', '')}`, written: null};
  const j = W.job || {state: 'idle'}, running = j.state === 'running';
  const mine = j.mesh === W.path && j.format === W.fmt;          // the job is about this mesh and format
  // the job's record when it is about this mesh and format; the last write's otherwise (not while it is being replaced)
  const shown = j.state === 'done' && mine ? j.summary : running && mine ? null : m.written && m.written.format === W.fmt ? m.written : null;
  const sel = (id, opts, val, on) => el('select#' + id, {on: {change: (e) => on(e.target.value)}},
    opts.map(([v, txt]) => el('option', {value: v, selected: v === val || null}, txt)));
  const iniOk = W.fmt === 'fesom3' && !!t.fesom3_ini;
  return el('div.write', {},
    el('div.muted.small', {}, `run folder ${t.run_dir}`),
    el('div.row.wrap', {},
      el('label', {}, 'mesh ', sel('write_mesh', t.meshes.map((x) => [x.name, x.name.replace('.npz', '') + (x.written ? ' · written' : '')]), name,
        (v) => { W.path = `${dir(W.path)}/${v}`; refreshWin('write'); })),
      el('label', {}, 'format ', sel('write_format', t.formats.map((f) => [f, f + (f === t.format ? " (the run's)" : '')]), W.fmt,
        (v) => { W.fmt = v; refreshWin('write'); })),
      iniOk ? el('label.check', {title: `run ${t.fesom3_ini} on the written folder (1 partition): it writes the edges and the levels and stops at the partitioning`},
        el('input#write_ini', {type: 'checkbox', checked: W.ini || null, on: {change: (e) => { W.ini = e.target.checked; }}}), 'then fesom_ini') : null,
      el('button.primary#write_go', {disabled: running, on: {click: start},
        title: "write the files: the run's size field and the depth for this mesh, the scores with the run's gates, the format's checks (the first write of a run takes minutes for a large mesh, later ones read the run's cache; one write at a time)"},
      running ? 'Writing…' : 'Write')),
    el('div.small', {title: m.out}, `writes into ${base(m.out)}/ in the run folder`, m.written ? el('span.muted', {}, ` · written before (${(m.written.date || '').replace('T', ' ')}, ${m.written.format}): writing again replaces its files`) : null),
    m.edits && m.edits_newer ? el('div.warn.small', {}, `${m.edits} depth edit${m.edits === 1 ? '' : 's'} of this mesh (Tools, Mesh depth) ${m.written ? 'changed after its last write' : 'not written yet'}: Write applies them.`) : null,
    running ? el('div.busy-line', {}, `Writing ${base(j.mesh)} as ${j.format}… ${fmt(j.elapsed_s, 0)} s` + (j.stage ? ` · now ${STAGE[j.stage] || j.stage}` : ''),
      el('div.muted.small', {}, `the run's size field and the depth for this mesh, the scores, the files${j.fesom_ini ? ', fesom_ini' : ''}: ` +
        "the first write of a run takes minutes for a large mesh, later ones read the run's cache/")) : null,
    j.state === 'error' && mine ? el('div.warn', {}, `The write failed: ${j.error}`) : null,
    shown ? recordView(shown) : running ? null : el('div.muted.small', {}, m.written ? `Written before as ${m.written.format}; pick that format to see its record, or Write.` : 'Not written yet.'));
}
// the Mesh panel's button: the mesh shown (a `meshom run` folder's mesh; the others are greyed out by mesh.js)
export function initWrite() {
  $('mesh_write').onclick = () => S.mesh.info && openWrite(S.mesh.info.path);
}
defineWin('write', {title: () => `Model files · ${W.path ? base(W.path).replace('.npz', '') : ''}`, width: 620, height: 600, open: load, build});
