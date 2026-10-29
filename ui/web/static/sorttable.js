// A table that sorts by a click on a column header (ascending, again: descending; an arrow shows which) and filters
// its rows by the text in the box above it (a substring of the row's text, any case). Numbers sort as numbers;
// empty values ('–', null, NaN, '') go last in both directions. The sort and the filter are kept per table name for
// the page session, so a table rebuilt with new data (the run list every 10 s) keeps them.
//
//   sortTable(name, cols, rows, {onRow, rowTitle, cls})
//   cols: [{label, value: (row) => number | string (the sort key), cell: (row) => content (default: the value or '–'),
//           num: right-aligned, cls: td class or (row) => class, title: the header's tip, sort: false (not sortable)}]
//   onRow: (row) => …, called on a click on the row outside its buttons
import {el} from './util.js';

const STATE = new Map();                    // name -> {col, dir, filter}

const asNum = (v) => typeof v === 'number' ? v : (typeof v === 'string' && v.trim() !== '' && Number.isFinite(+v) ? +v : null);
const isEmpty = (v) => v === null || v === undefined || v === '' || v === '–' || (typeof v === 'number' && !Number.isFinite(v));
export function compareValues(a, b) {        // a, b not empty
  const x = asNum(a), y = asNum(b);
  if (x !== null && y !== null) return x - y;
  return String(a).localeCompare(String(b), undefined, {numeric: true, sensitivity: 'base'});
}

export function sortTable(name, cols, rows, {onRow = null, rowTitle = null, cls = ''} = {}) {
  const st = STATE.get(name) || {col: -1, dir: 1, filter: ''};
  STATE.set(name, st);
  const ths = cols.map((c, j) => {
    const sortable = c.sort !== false;
    return el('th' + (sortable ? '.sortable' : '') + (c.num ? '.num' : ''), {title: c.title || (sortable ? `sort by ${c.label}` : ''),
      on: sortable ? {click: () => { st.dir = st.col === j ? -st.dir : 1; st.col = j; apply(); }} : {}},
    c.label, sortable ? el('span.sort-arrow') : null);
  });
  const items = rows.map((r, i) => {
    const tds = cols.map((c) => {
      const v = c.value ? c.value(r) : null, k = typeof c.cls === 'function' ? c.cls(r) : c.cls;
      // a cell may nest lists (a name broken at its underscores, a list of buttons): el() flattens one level only
      return el('td' + (c.num ? '.num' : '') + (k ? '.' + k : ''), {}, ...[c.cell ? c.cell(r) : (isEmpty(v) ? '–' : v)].flat(Infinity));
    });
    const tr = el('tr', {title: rowTitle ? rowTitle(r) : '', on: onRow ? {click: (e) => { if (!e.target.closest('button, a, input, select')) onRow(r); }} : {}}, tds);
    // the filter reads the cells' text, not the buttons of an action column
    return {r, i, tr, text: tds.filter((td, j) => cols[j].sort !== false).map((td) => td.textContent).join(' ').toLowerCase()};
  });
  const tbody = el('tbody'), count = el('span.muted.small.stbl-count');
  const filter = el('input.stbl-filter', {type: 'search', placeholder: 'filter rows', value: st.filter, spellcheck: false,
    title: 'show only the rows whose text contains this', on: {input: (e) => { st.filter = e.target.value; apply(); }}});
  filter.dataset.keep = `${name}-filter`;
  function apply() {
    const c = cols[st.col], f = st.filter.trim().toLowerCase();
    const order = c ? [...items].sort((p, q) => {
      const a = c.value(p.r), b = c.value(q.r), ea = isEmpty(a), eb = isEmpty(b);
      return ea || eb ? (ea - eb) || (p.i - q.i) : st.dir * compareValues(a, b) || (p.i - q.i);
    }) : items;
    let shown = 0;
    for (const it of order) { it.tr.hidden = !!f && !it.text.includes(f); shown += !it.tr.hidden; }
    tbody.replaceChildren(...order.map((it) => it.tr));
    ths.forEach((th, j) => {
      const on = j === st.col;
      th.classList.toggle('sorted', on);
      th.setAttribute('aria-sort', on ? (st.dir > 0 ? 'ascending' : 'descending') : 'none');
      const a = th.querySelector('.sort-arrow');
      if (a) a.textContent = on ? (st.dir > 0 ? '▲' : '▼') : '';
    });
    count.textContent = f ? `${shown} of ${items.length} rows` : `${items.length} rows`;
  }
  apply();
  return el('div.stbl', {}, el('div.stbl-tools', {}, filter, count),
    el('div.stbl-scroll', {}, el('table.tbl.stbl-table' + (onRow ? '.clickable' : '') + (cls ? '.' + cls : ''), {},
      el('thead', {}, el('tr', {}, ths)), tbody)));
}
