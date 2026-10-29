// Panel "Runs": "Run" writes the page's config into a new run folder with an sbatch script that runs `meshom run`
// (submitted only when the server runs with --allow-submit), the run list (state, stages, the quality.json summary,
// the log tail), opening a run's mesh or its config; panel "Reference meshes": the meshes other models built
// (curvilinear grids, triangle meshes). Opening a mesh switches to the Mesh panel and shows it whole. Both lists also
// open in a floating window (floatwin.js) as sortable tables: every run (state, nodes, gates, the config's h, date)
// and every reference mesh (family, resolution, nodes), with the panel's buttons. "Add folder…" opens a window of the
// folders searched for runs (the runs right in them; ticked "subfolders": at any depth): a folder browser of the
// server's disk adds more, remembered per browser.
import {$, S, api, el, fmt, status} from './util.js';
import {meshName, reloadMesh} from './mesh.js';
import {addCompare, compareFull, inCompare} from './compare.js';
import {defineWin, isOpen, openWin, refreshWin} from './floatwin.js';
import {sortTable} from './sorttable.js';

const R = S.runs = {list: [], refs: [], roots: [], deep: [], scans: [], tag: null, timer: null, logPath: null, armed: null};   // armed: {path, until} of "open config"
const ARM_MS = 6000;                       // "click again" holds this long, across the list's 10 s re-renders

// one line of a quality.json record: status, nodes, quads / triangles, gates, wall time
export function recordLine(q) {
  if (!q) return '';
  if (q.status && q.status !== 'ok') return `${q.status}${q.error ? ': ' + q.error.split('\n').filter(Boolean).pop().slice(0, 90) : ''}`;
  const n = (x) => (x ?? 0).toLocaleString();
  return `${n(q.n_nodes)} nodes · ${n(q.n_quads)} quads · ${n(q.n_tris)} tris` +
    (q.gates_total ? ` · gates ${q.gates_passed}/${q.gates_total}` : '') + (q.wall_s ? ` · ${fmt(q.wall_s, 1)} s` : '');
}
// a folder name that may wrap after its underscores, not inside a word
const breakable = (name) => name.split('_').flatMap((p, i) => i ? [el('wbr'), '_' + p] : [p]);
// "+ compare": puts a mesh into the Compare tab (greyed out when it is there or the tab is full); again: the list
// that re-renders with the button ticked (its panel and window)
function compareButton(path, name, text = '+ compare', again = refreshRuns) {
  const there = inCompare(path);
  return el('button.small.add', {disabled: there || compareFull(),
    title: there ? `${name} is in Compare` : compareFull() ? 'Compare is full (4 meshes): remove one there' : `add ${name} to the comparison (tab Compare)`,
    on: {click: () => { addCompare(path, name); again(); }}}, there ? text.replace('+', '✓') : text);
}
// a run's buttons: one per mesh (raw, smooth, repaired-NN: open it in Mesh); its "+ raw" … picks for Compare
const meshButtons = (x) => x.meshes.map((m) => el('button.small', {title: `show ${x.path}/${m} in Mesh`, on: {click: () => loadRunMesh(x.path + '/' + m)}}, m.replace('.npz', '')));
const cmpButtons = (x) => x.meshes.length ? el('div.cmp.small', {}, el('span.muted', {}, 'compare '), x.meshes.map((m) =>
  compareButton(x.path + '/' + m, `${x.name} / ${m.replace('.npz', '')}`, '+ ' + m.replace('.npz', '')))) : null;
const stateCls = (x) => 'state-' + x.state.split(' ')[0].replace('?', '');
export async function refreshRuns() {
  const r = await api('api/runs' + (R.tag ? `?known=${R.tag}` : ''));
  if (r.same || r.error) return R.list.length;          // unchanged since the last list: nothing to redraw
  R.list = r.runs; R.roots = r.roots; R.deep = r.deep; R.scans = r.scans; R.tag = r.tag || null;
  refreshWin('runs'); refreshWin('folders');
  $('runs_roots').textContent = `run folders under ${r.roots.join(', ')}` + (r.allow_submit ? '' : ' · submit off (dry run)');
  const tb = el('table.tbl.runs', {}, el('tr', {}, ['run', 'state', 'stages', 'mesh'].map((t) => el('th', {}, t))));
  if (!r.runs.length) { $('runs_list').replaceChildren(el('div.muted', {}, 'No runs yet. Set up h, the coast and the method, then press Run.')); return 0; }
  for (const x of r.runs.slice(0, 60)) {
    const st = x.stages.map((s) => `${s.stage}${s.mesh ? ' ' + s.mesh : ''} ${fmt(s.wall_s, 1)} s`).join(', ');
    const q = x.quality ? Object.entries(x.quality).map(([k, v]) => `${k}: ${recordLine(v)}`).join('\n') : '';
    tb.append(el('tr' + (q ? '.head' : ''), {},
      el('td.name', {title: x.path}, breakable(x.name), el('div.muted.small', {}, x.folder), x.job ? el('div.muted.small', {}, `job ${x.job.job_id}`) : null),
      el('td.' + stateCls(x), {}, x.state, x.m14 ? el('div.muted.small', {}, 'm14 blocks') : null),
      el('td.small', {}, st),
      el('td', {}, meshButtons(x),
        el('button.small', {disabled: !x.log, on: {click: () => showLog(x.path)}}, 'log'),
        x.config ? el('button.small', {title: "make this run's config the page's config, to change it and run again", on: {click: () => openConfig(x)}},
          armedFor(x.path) ? 'click again: unsaved changes are lost' : 'open config') : null,
        cmpButtons(x))));
    if (q) tb.append(el('tr.sub', {}, el('td.muted.small.pre', {colSpan: 4}, q)));
  }
  $('runs_list').replaceChildren(tb);
  if (R.logPath) await showLog(R.logPath, false);
  return r.runs.length;
}

// ------------------------------------------------------------------------- the run list in a floating window
// the run's last record: optimized when the run optimised, else smooth when it was smoothed, else raw (its state's record)
const lastRec = (x) => (x.quality ? x.quality.optimized || x.quality.smooth || x.quality.raw || null : null);
const count = (v) => (v === null || v === undefined) ? '–' : v.toLocaleString();
const km3 = (v) => `${+v.toPrecision(3)}`;
// the config's h (finest–base, km, after the scale) and the scale when it is not 1
function hCell(x) {
  const h = x.h_km ? (x.h_km[0] === x.h_km[1] ? `${km3(x.h_km[0])} km` : `${km3(x.h_km[0])}–${km3(x.h_km[1])} km`) : '–';
  return [h, x.scale !== null && x.scale !== undefined && x.scale !== 1 ? el('div.muted.small', {}, `scale ${x.scale}`) : null];
}
const RUN_COLS = [
  {label: 'run', value: (x) => x.name, cls: 'name', cell: (x) => [breakable(x.name), x.job ? el('div.muted.small', {}, `job ${x.job.job_id}`) : null]},
  {label: 'folder', value: (x) => x.folder, cls: 'small', cell: (x) => breakable(x.folder),
    title: 'sort by the folder the run sits in (relative to the searched folder it was found under)'},
  {label: 'state', value: (x) => x.state, cls: stateCls},
  {label: 'nodes', num: true, value: (x) => lastRec(x)?.n_nodes, cell: (x) => count(lastRec(x)?.n_nodes)},
  {label: 'gates passed', num: true, value: (x) => lastRec(x)?.gates_passed, title: 'sort by the gates passed (of the last record: optimized, else smooth, else raw)',
    cell: (x) => { const q = lastRec(x); return q && q.gates_total ? `${q.gates_passed} / ${q.gates_total}` : '–'; }},
  {label: 'h', num: true, value: (x) => (x.h_km ? x.h_km[1] : null), cell: hCell,
    title: "sort by the base h of the run's config (the h column: finest–base, km, after [mesh] scale)"},
  {label: 'date', value: (x) => x.date, cls: 'nowrap', cell: (x) => (x.date ? x.date.replace('T', ' ').slice(0, 16) : '–')},
  {label: 'open', sort: false, cell: (x) => [meshButtons(x), cmpButtons(x)]},
];
defineWin('runs', {title: () => `Runs · ${R.list.length}`, width: 760, height: 460, open: () => refreshRuns(),
  build: () => R.list.length ? sortTable('runs', RUN_COLS, R.list, {rowTitle: (x) => x.path, cls: 'runs-win'})
    : el('div.muted', {}, 'No runs yet. Set up h, the coast and the method, then press Run.')});

// ----------------------------------------------------------- the folders searched for runs ("Add folder…" window)
// The server lists the run folders right in every folder of the list, and with the folder's "subfolders" tick those at
// any depth below it; the first folder is where new runs go. Folders added or ticked here are remembered in this
// browser (localStorage, [{path, deep}]) and added again when the page loads.
const ROOTS_KEY = 'meshom.runRoots';
function savedRoots() {
  try { return (JSON.parse(localStorage.getItem(ROOTS_KEY) || '[]') || []).map((x) => typeof x === 'string' ? {path: x, deep: false} : x); }
  catch (e) { return []; }
}
function keepRoots(list) {
  const by = new Map(list.map((x) => [x.path, x]));        // the last entry of a path wins
  try { localStorage.setItem(ROOTS_KEY, JSON.stringify([...by.values()])); } catch (e) { /* not remembered */ }
}
const F = {path: null, parent: null, dirs: null, more: 0, places: [], typed: '', deep: false};
const tail = (p, n = 3) => p.split('/').filter(Boolean).slice(-n).join('/');
async function browse(path) {
  const r = await api(path ? `api/dirs?path=${encodeURIComponent(path)}` : 'api/dirs');
  if (r.error) return;
  if (path) Object.assign(F, {path: r.path, parent: r.parent, dirs: r.dirs, more: r.more, typed: r.path});
  else F.places = r.places;
  refreshWin('folders');
}
async function addRoot(path, deep = false) {
  if (!path) return;
  const r = await api('api/runs/roots', {add: path, deep});
  if (r.error) return;
  keepRoots([...savedRoots(), {path: r.path, deep}]);
  status(deep ? `Searching ${r.path} and all its subfolders for run folders. A large folder takes minutes the first time; the list fills in then.`
    : `Listing the run folders in ${r.path}.`);
  R.tag = null; await refreshRuns(); browse();
}
async function removeRoot(path) {
  const r = await api('api/runs/roots', {remove: path});
  if (r.error) return;
  keepRoots(savedRoots().filter((x) => x.path !== path));
  R.tag = null; await refreshRuns(); browse();
}
function scanCell(s) {
  if (!s) return '–';
  if (s.scanning) return 'searching…';
  if (s.missing) return 'folder missing';
  return `${s.runs.toLocaleString()} runs` + (s.truncated ? ` (stopped after ${s.dirs.toLocaleString()} folders)` : '');
}
defineWin('folders', {title: () => 'Folders searched for runs', width: 640, height: 560, open: () => { browse(); browse(F.path || R.roots[0]); },
  build: () => {
    const scans = new Map(R.scans.map((s) => [s.path, s]));
    const roots = el('table.tbl', {}, el('tr', {}, ['folder', 'subfolders', 'found', ''].map((t) => el('th', {}, t))), R.roots.map((p, i) =>
      el('tr', {}, el('td.name', {title: p}, p, i ? null : el('div.muted.small', {}, 'new runs are written here')),
        el('td', {}, el('input', {type: 'checkbox', checked: R.deep.includes(p) || null,
          title: 'ticked: also the run folders in all subfolders, at any depth (a large folder takes minutes the first time); not ticked: only the run folders right in this folder',
          on: {change: (e) => addRoot(p, e.target.checked)}})),
        el('td.small', {}, scanCell(scans.get(p))),
        el('td', {}, i ? el('button.small', {title: 'stop searching this folder (nothing is deleted)', on: {click: () => removeRoot(p)}}, 'remove') : null))));
    const input = el('input#folders_path', {type: 'text', value: F.typed, placeholder: 'a folder on the server, e.g. /work/…/meshom/.worktrees',
      on: {input: (e) => { F.typed = e.target.value; }, keydown: (e) => { if (e.key === 'Enter') browse(F.typed.trim()); }}});
    const here = F.path && R.roots.includes(F.path);
    return el('div.folders', {},
      roots,
      el('h4', {}, 'Add a folder'),
      el('div.muted.small', {}, 'The run folders right in it are listed; tick "subfolders" for those at any depth below it too (hidden folders such as .worktrees and .tmp included). Open a folder below or type its path.'),
      el('div.row', {}, input, el('button.small', {title: 'show the subfolders of the typed folder', on: {click: () => browse(F.typed.trim())}}, 'Open'),
        el('label.inline.small', {title: 'also the run folders in all its subfolders, at any depth'},
          el('input', {type: 'checkbox', checked: F.deep || null, on: {change: (e) => { F.deep = e.target.checked; }}}), 'subfolders'),
        el('button.small.primary', {title: 'list the run folders of the typed folder', on: {click: () => addRoot(F.typed.trim(), F.deep)}}, 'Add')),
      el('div.row.wrap', {}, el('span.muted.small', {}, 'go to '), F.places.map((p) => el('button.small', {title: p.path, on: {click: () => browse(p.path)}}, tail(p.path, 2)))),
      F.dirs ? [
        el('div.row', {}, F.parent ? el('button.small', {title: F.parent, on: {click: () => browse(F.parent)}}, '↑ up') : null,
          el('span.small.name', {}, F.path),
          el('button.small.primary', {disabled: here, title: 'list the run folders of this folder (with "subfolders" ticked above: at any depth)',
            on: {click: () => addRoot(F.path, F.deep)}}, here ? 'in the list' : 'Add this folder')),
        el('div.dirlist', {}, F.dirs.length ? F.dirs.map((d) => el('button.dir' + (d.run ? '.run' : ''),
          {title: d.run ? 'a run folder: Add lists it' : 'open', on: {click: () => browse(`${F.path}/${d.name}`)}}, d.name + (d.run ? '  · run' : '')))
          : el('div.muted.small', {}, 'no subfolders')),
        F.more ? el('div.muted.small', {}, `${F.more} more subfolders not shown`) : null] : null);
  }});

// FESOM2 meshes first, then the other triangle meshes, then the curvilinear grids
const refRank = (x) => (/fesom2|fmg_/.test(x.name) ? 0 : x.path.includes('triangle_meshes') ? 1 : 2);
const refsInOrder = () => [...R.refs].sort((a, b) => refRank(a) - refRank(b) || a.name.localeCompare(b.name));
const refButtons = (x) => [el('button.small', {title: `load ${x.path}`, on: {click: () => loadRunMesh(x.path)}}, 'load'),
  compareButton(x.path, x.name, '+ compare', refreshRefs)];
export async function refreshRefs() {
  const r = await api('api/refs');
  R.refs = r.refs;
  refreshWin('refs');
  $('refs_roots').textContent = `folders with mesh.npz under ${r.roots.join(', ')}`;
  const tb = el('table.tbl.runs', {}, el('tr', {}, ['mesh', 'what', 'size', ''].map((t) => el('th', {}, t))));
  for (const x of refsInOrder()) {
    const q = x.quality || {};
    tb.append(el('tr', {},
      el('td.name', {title: x.path}, breakable(x.name)),
      el('td.small', {}, [q.family, q.config || q.generator, q.res].filter(Boolean).join(' · ')),
      el('td.small', {}, recordLine(q), el('div.muted.small', {}, `${x.mb} MB`)),
      el('td', {}, refButtons(x))));
  }
  $('refs_list').replaceChildren(tb);
  return r.refs.length;
}
// the reference meshes in a floating window: resolution as the source names it, sorted by the typical element size
// the server measured (res_km: median sqrt(area) of the elements; for a triangle mesh the equilateral edge)
const REF_COLS = [
  {label: 'mesh', value: (x) => x.name, cls: 'name', cell: (x) => breakable(x.name)},
  {label: 'family', value: (x) => (x.quality || {}).family},
  {label: 'resolution', num: true, value: (x) => x.res_km, title: 'sort by the typical element size (median √area of the elements; triangle meshes: the equilateral edge)',
    cell: (x) => [(x.quality || {}).res || '', x.res_km ? el('div.muted.small', {}, `≈ ${km3(x.res_km)} km`) : (x.quality || {}).res ? null : '–']},
  {label: 'nodes', num: true, value: (x) => (x.quality || {}).n_nodes, cell: (x) => count((x.quality || {}).n_nodes)},
  {label: 'triangles', num: true, value: (x) => { const q = x.quality || {}; return q.n_tris === undefined ? null : q.n_tris / Math.max(1, q.n_tris + (q.n_quads || 0)); },
    title: 'sort by the share of triangles', cell: (x) => { const q = x.quality || {}; return q.n_tris === undefined ? '–'
      : `${(100 * q.n_tris / Math.max(1, q.n_tris + (q.n_quads || 0))).toFixed(0)} %`; }},
  {label: 'MB', num: true, value: (x) => x.mb},
  {label: '', sort: false, cell: refButtons},
];
defineWin('refs', {title: () => `Reference meshes · ${R.refs.length}`, width: 760, height: 480, open: () => (R.refs.length ? null : refreshRefs()),
  build: () => R.refs.length ? sortTable('refs', REF_COLS, refsInOrder(), {cls: 'refs-win',
    rowTitle: (x) => [x.path, (x.quality || {}).config || (x.quality || {}).generator].filter(Boolean).join('\n')})
    : el('div.muted', {}, 'No reference meshes found.')});
async function showLog(path, scroll = true) {
  R.logPath = path;
  const r = await api(`api/runs/log?path=${encodeURIComponent(path)}&n=60`);
  $('runs_log_name').textContent = r.log || '(no log)';
  $('runs_log').textContent = r.tail;
  if (scroll) $('runs_log').scrollTop = 1e9;
}
export async function loadRunMesh(path) {
  const r = await api('api/mesh/load', {path});
  if (r.error) return;
  S.api.tab('mesh');
  await reloadMesh(true);
  status(`Showing ${meshName(S.mesh.info.path)}: ${(S.mesh.info.n_tris + S.mesh.info.n_quads).toLocaleString()} elements.`);
}
// a run's config replaces the page's (the page reloads with it); with unsaved changes the first click only warns
const armedFor = (path) => R.armed && R.armed.path === path && Date.now() < R.armed.until;
async function openConfig(x) {
  if (!$('unsaved').hidden && !armedFor(x.path)) {
    R.armed = {path: x.path, until: Date.now() + ARM_MS}; refreshRuns();
    setTimeout(() => { if (R.armed && Date.now() >= R.armed.until) { R.armed = null; refreshRuns(); } }, ARM_MS + 50);
    return;
  }
  R.armed = null;
  const r = await api('api/config/open', {path: x.path});
  if (r.error) return;
  try { sessionStorage.setItem('meshom.notice', `Opened the config of run ${x.name}. Save writes ${r.save_to}; the run's folder is not touched.`); } catch (e) { /* no storage: no message */ }
  location.reload();
}
async function run() {
  const res = {partition: $('run_partition').value, cpus: +$('run_cpus').value, mem: $('run_mem').value, time: $('run_time').value};
  $('run').disabled = true;                   // one folder (one job) per click
  const r = await api('api/run', {config: S.cfg, resources: res}).finally(() => { $('run').disabled = false; });
  if (r.error) return;
  $('run_result').textContent = (r.submitted ? `Submitted: job ${r.job_id}. It appears in the list below; the list refreshes every 10 s.`
    : r.error ? `sbatch failed: ${r.error}` : `Run folder written, not submitted: this page was started without --allow-submit.` +
      `\nTo submit it yourself: sbatch ${r.script}`) + `\nfolder: ${r.dir}`;
  status(r.submitted ? `Job ${r.job_id} submitted.` : 'Run folder written (not submitted).');
  await refreshRuns();
}
export function initRuns(info) {
  $('run').onclick = run;
  $('run').textContent = info.allow_submit ? 'Run on Levante (sbatch)' : 'Write the run folder (not submitted)';
  $('runs_refresh').onclick = () => { refreshRuns(); refreshRefs(); };
  $('runs_win').onclick = () => openWin('runs');
  $('runs_folders').onclick = () => openWin('folders');
  // the folders added in this browser before: added again (quietly: one that is gone is forgotten)
  Promise.all(savedRoots().map((x) => fetch('api/runs/roots', {method: 'POST', body: JSON.stringify({add: x.path, deep: !!x.deep})})
    .then((r) => { if (r.status === 400) keepRoots(savedRoots().filter((q) => q.path !== x.path)); }).catch(() => null)))
    .then(() => { R.tag = null; return refreshRuns(); });
  $('refs_win').onclick = () => openWin('refs');
  $('run_partition').value = info.resources.partition; $('run_cpus').value = info.resources.cpus;
  $('run_mem').value = info.resources.mem; $('run_time').value = info.resources.time;
  $('run_cmd').textContent = info.run_cmd;
  // the list refreshes every 10 s while the Runs tab or its window is open
  R.timer = setInterval(() => { if (S.tab === 'runs' || isOpen('runs') || isOpen('folders')) refreshRuns(); }, 10000);
}
