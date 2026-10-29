// Panel 5 · Method: meshom.methods.METHODS with each module's DEFAULTS as a form; the choice goes to [mesh] method and
// the options that differ from the defaults to [mesh] opts (TOML has no null, so a None default stays unset).
import {$, S, api, el, status} from './util.js';

const M = S.method = {list: [], byName: {}, kept: {}};      // kept: [mesh] opts of the methods left

export async function initMethod() {
  const r = await api('api/methods');
  M.list = r.methods; M.byName = Object.fromEntries(r.methods.map((m) => [m.name, m]));
  const sel = $('method'); sel.replaceChildren();
  for (const m of M.list) sel.append(el('option', {value: m.name, selected: m.name === S.cfg.mesh.method}, `${m.name}${m.module !== m.name ? '  (' + m.module + ')' : ''}`));
  sel.onchange = () => {
    const old = S.cfg.mesh.method, had = S.cfg.mesh.opts && Object.keys(S.cfg.mesh.opts).length;
    if (had) M.kept[old] = S.cfg.mesh.opts;
    S.cfg.mesh.method = sel.value; S.cfg.mesh.opts = M.kept[sel.value] || {};
    if (had || Object.keys(S.cfg.mesh.opts).length) status(`options of ${old} kept for when you come back; ${sel.value} uses ${Object.keys(S.cfg.mesh.opts).length ? 'the options you set before' : 'its defaults'}`);
    render(); S.showToml();
  };
  render();
}
function opts() { return S.cfg.mesh.opts || (S.cfg.mesh.opts = {}); }
function setOpt(k, v) { opts()[k] = v; S.showToml(); render(false); }
function clearOpt(k) { if (S.cfg.mesh.opts) delete S.cfg.mesh.opts[k]; S.showToml(); render(); }

function render(rebuild = true) {
  const m = M.byName[S.cfg.mesh.method];
  const box = $('method_form');
  if (!m) { box.replaceChildren(el('div.muted', {}, `unknown method ${S.cfg.mesh.method}`)); return; }
  const cur = S.cfg.mesh.opts || {};
  if (!rebuild) {           // only the "changed" marks
    box.querySelectorAll('label[data-key]').forEach((l) => l.classList.toggle('changed', l.dataset.key in cur));
    return;
  }
  const fixed = Object.entries(m.overrides || {});
  box.replaceChildren(...[el('div.muted.small', {}, m.doc || ''),
    el('div.muted.small', {}, `${Object.keys(m.defaults).length} options; the ones you change are written to [mesh] opts`),
    m.composed ? el('div.small.warn', {}, `${m.name} is made of another method plus a stage: the form shows the options of ${m.module}, which this variant may not take as they are`)
      : fixed.length ? el('div.muted.small', {}, `${m.name} is a variant of ${m.module} that sets ${fixed.map(([k, v]) => `${k} = ${JSON.stringify(v)}`).join(', ')} (marked ◆ below)`) : null].filter(Boolean));
  for (const [k, d] of Object.entries(m.defaults)) {
    const v = k in cur ? cur[k] : d;
    let input;
    if (typeof d === 'boolean') {
      input = el('input', {type: 'checkbox', checked: !!v, on: {change: (e) => setOpt(k, e.target.checked)}});
    } else if (typeof d === 'number') {
      input = el('input', {type: 'number', step: Number.isInteger(d) ? '1' : 'any', value: v,
        on: {change: (e) => setOpt(k, Number.isInteger(d) && !e.target.value.includes('.') ? parseInt(e.target.value, 10) : +e.target.value)}});
    } else if (typeof d === 'string') {
      input = el('input', {type: 'text', value: v, on: {change: (e) => setOpt(k, e.target.value)}});
    } else {                // None, lists, dicts: JSON text
      input = el('input', {type: 'text', value: k in cur ? JSON.stringify(v) : '', placeholder: JSON.stringify(d), title: 'JSON',
        on: {change: (e) => {
          if (e.target.value.trim() === '') return clearOpt(k);
          try { setOpt(k, JSON.parse(e.target.value)); } catch (err) { status(`${k}: not JSON (${err.message})`, true); }
        }}});
    }
    box.append(el('label' + (k in cur ? '.changed' : ''), {'data-key': k, title: k in (m.overrides || {}) ? `set by the variant ${m.name}` : ''},
      el('span', {}, k in (m.overrides || {}) ? `◆ ${k}` : k),
      el('span.opt', {}, input, el('button.icon', {title: 'back to the default', on: {click: () => clearOpt(k)}}, '↺'))));
  }
}
