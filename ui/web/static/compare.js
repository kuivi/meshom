// Panel "Compare": 2-4 meshes picked with "+ compare" in Runs and Reference meshes, one column each, numbers from
// their quality records (POST api/compare: quality.json / repaired-NN.json, no mesh is read), the best of a row marked;
// "show" opens a column's mesh in Mesh. The table also opens in a floating window (floatwin.js) over the map.
import {$, S, api, el, fmt, status} from './util.js';
import {defineWin, isOpen, openWin, refreshWin} from './floatwin.js';

export const MAX_COMPARE = 4;
const C = S.compare = {paths: [], last: null};      // last: the api/compare answer shown

const count = (v) => (v === null || v === undefined) ? '–' : v.toLocaleString();
const FORMAT = {
  'nodes': count, 'quads': count, 'triangles': count,
  'triangle share': (v) => v === null ? '–' : `${(100 * v).toFixed(v > 0 && v < 0.001 ? 3 : 1)} %`,
  'n3 per 1000 nodes': (v) => fmt(v, 2), 'n5 per 1000 nodes': (v) => fmt(v, 2),
  'size jump across an edge, p99': (v) => fmt(v, 3), '|ln(edge / h)|, p99': (v) => fmt(v, 3),
};
const show = (what, v) => (FORMAT[what] || ((x) => x === null ? '–' : `${fmt(x, 1)}°`))(v);
// a mesh's name that may wrap after its underscores and slashes
const breakable = (name) => name.split(/(?=[_/])/).flatMap((p, i) => i ? [el('wbr'), p] : [p]);

function label() {
  const b = document.querySelector('button[data-tab=compare]');
  if (b) b.textContent = C.paths.length ? `Compare (${C.paths.length})` : 'Compare';
}
export const inCompare = (path) => C.paths.includes(path);
export const compareFull = () => C.paths.length >= MAX_COMPARE;
export function addCompare(path, name) {
  if (inCompare(path)) return status(`${name} is already in Compare.`);
  if (compareFull()) return status(`Compare holds ${MAX_COMPARE} meshes: remove one there first.`, true);
  C.paths.push(path); label();
  if (isOpen('compare')) refreshCompare();
  status(`Added ${name} to Compare (${C.paths.length} of ${MAX_COMPARE})` +
    (C.paths.length < 2 ? ': pick one more.' : ': open the tab Compare.'));
}
function remove(path) {
  C.paths = C.paths.filter((p) => p !== path); label();
  refreshCompare();
}

const EMPTY = 'Nothing picked yet. Add 2–4 meshes with "+ compare" in Runs (raw, smooth, repaired) or in Reference meshes.';
// the table of an api/compare answer and the note under it
function compareView(r) {
  const cols = r.meshes;
  const tb = el('table.tbl.compare', {}, el('tr', {}, el('th', {}, ''), cols.map((c) => el('th', {title: c.file},
    el('div.cname', {}, breakable(c.name)),
    c.has_record ? null : el('div.warn.small', {}, 'no quality record'),
    el('div', {}, el('button.small', {title: `open ${c.file} in Mesh`, on: {click: () => S.api.loadRunMesh(c.path)}}, 'show'),
      el('button.small.icon', {title: 'remove from the comparison', on: {click: () => remove(c.path)}}, '✕'))))));
  for (const row of r.rows) {
    tb.append(el('tr', {}, el('td.what', {}, row.what), row.values.map((v, i) =>
      el('td.num' + (row.best.includes(i) ? '.best' : ''), {}, show(row.what, v)))));
  }
  const g = r.gates;
  tb.append(el('tr', {}, el('td.what', {title: g.names.join('\n')}, `gates all of them have (${g.names.length})`),
    g.columns.map((c, i) => el('td.num' + (g.best.includes(i) ? '.best' : ''), {}, c === null ? '–' : `${c.passed} / ${c.total}`))));
  tb.append(el('tr', {}, el('td.what.muted', {}, 'failed'), g.columns.map((c) => el('td.failed', {},
    c === null ? '' : c.failed.length ? c.failed.map((f) => el('div', {title: `${f.what}${f.where ? ', ' + f.where : ''}`}, f.gate)) : 'none'))));
  return [tb, el('div.muted.small', {}, 'Bold green: the best value of a row (lower is better; for gates, more passed). ' +
    'Counts are not ranked. – : the record has no such number (no quads, no triangles, or h unknown for |ln(edge / h)|).' +
    (cols.length < 2 ? ' Pick one more mesh to compare.' : ''))];
}
export async function refreshCompare() {
  const box = $('compare_table');
  if (!C.paths.length) {
    C.last = null; box.replaceChildren(el('div.muted', {}, EMPTY)); refreshWin('compare');
    return 0;
  }
  const r = await api('api/compare', {paths: C.paths});
  if (r.error) return 0;
  C.last = r;
  box.replaceChildren(...compareView(r));
  refreshWin('compare');
  return r.meshes.length;
}
defineWin('compare', {title: () => `Compare · ${C.paths.length} of ${MAX_COMPARE}`, width: 680, height: 520, open: () => refreshCompare(),
  build: () => el('div.compare-win', {}, C.paths.length && C.last ? compareView(C.last) : el('div.muted', {}, EMPTY))});
export function initCompare() { label(); $('compare_win').onclick = () => openWin('compare'); }
