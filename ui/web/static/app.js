// meshom web UI: boot, tabs, the config box (Save, Save as, the unsaved mark) and the API used by the headless runs.
// The Python server (ui/web/server.py) evaluates h, reads coastlines, meshes and runs; this page only draws.
import {$, S, api, status} from './util.js';
import {clearFlyMark, deckgl, draw, flyTo, getViewState, nextRender, redraw, rotate, setView} from './map.js';
import {ensurePolarH, initH, legend, preview, rho, schedulePreview, variantRef, initVariants} from './hsize.js';
import {initCoast, loadCoast} from './coastpanel.js';
import {addWindow, initWindows, previewWindow} from './windows.js';
import {initBlocks, showBlocksText} from './blocks.js';
import {initMethod} from './method.js';
import {flyToElement, initMesh, loadMesh, setField} from './mesh.js';
import {initEdit} from './edit.js';
import {initRuns, loadRunMesh, refreshRefs, refreshRuns} from './runs.js';
import {addCompare, initCompare, refreshCompare} from './compare.js';
import {closeWin, openWin, openWins} from './floatwin.js';
import {initTools} from './tools.js';
import {initWrite} from './write.js';

// the TOML of the page's config; "unsaved changes" while it differs from the file's last saved or loaded text
S.showToml = () => api('api/toml', {config: S.cfg, size_from: variantRef()}).then((r) => {
  $('toml').textContent = r.toml; $('unsaved').hidden = r.toml === S.savedToml;
});
function showPath(p) {
  S.configPath = p;
  $('config_path').textContent = (S.openedFrom ? `config from run ${S.openedFrom.split('/').at(-1)}; ` : '') + `saved to: ${p}`;
}
async function saveToml(path) {
  const r = await api('api/config', {config: S.cfg, size_from: variantRef(), ...(path ? {path} : {})});
  if (r.error) return;
  $('toml').textContent = S.savedToml = r.toml; $('unsaved').hidden = true; $('save_as_row').hidden = true;
  showPath(r.saved); status(`Saved to ${r.saved}`);
}
export function tab(name) {
  S.tab = name;
  document.querySelectorAll('[data-tab]').forEach((b) => b.classList.toggle('on', b.dataset.tab === name));
  document.querySelectorAll('.tab').forEach((t) => t.hidden = t.id !== `tab-${name}`);
  if (name === 'runs') refreshRuns();
  if (name === 'refs') refreshRefs();
  if (name === 'compare') refreshCompare();
  clearFlyMark();                              // the ring marks a place for the panel that asked for it
  redraw(); legend(); ensurePolarH();         // overlays that belong to one panel (zones, windows, blocks, h, the mesh) follow the tab
  if (name === 'blocks') showBlocksText();
}
// developer numbers (timings, payload sizes, paths, the batch template) only on request; remembered per browser
function showDev(on) {
  document.body.classList.toggle('show-dev', on); $('show_dev').checked = on;
  try { localStorage.setItem('meshom.dev', on ? '1' : ''); } catch (e) { /* private window: not remembered */ }
}

document.querySelectorAll('[data-view]').forEach((b) => b.onclick = () => setView(b.dataset.view));
document.querySelectorAll('[data-tab]').forEach((b) => b.onclick = () => tab(b.dataset.tab));
$('save').onclick = () => saveToml();
$('save_as').onclick = () => { $('save_as_row').hidden = !$('save_as_row').hidden; $('save_path').value = S.configPath || ''; $('save_path').focus(); };
$('save_as_go').onclick = () => saveToml($('save_path').value.trim());
$('show_dev').onchange = (e) => showDev(e.target.checked);
// the look: one stylesheet of static/ (skin-daylight.css, skin-glass.css or the classic app.css), remembered per browser;
// the map is resized after a switch (Glass lays it under the whole window)
const SKINS = {daylight: 'static/skin-daylight.css', glass: 'static/skin-glass.css', app: 'static/app.css'};
function useSkin(k) {
  if (!SKINS[k]) k = 'daylight';
  $('skin').href = SKINS[k]; $('skin_pick').value = k;
  try { localStorage.setItem('meshom.skin', k); } catch (e) { /* not remembered */ }
  $('skin').onload = () => window.dispatchEvent(new Event('resize'));
}
$('skin_pick').onchange = (e) => useSkin(e.target.value);
try { useSkin(localStorage.getItem('meshom.skin') || 'daylight'); } catch (e) { useSkin('daylight'); }
// a short intro on the first visit (remembered per browser); "?" opens it again
function showIntro(on) { $('intro').hidden = !on; try { localStorage.setItem('meshom.intro', on ? '' : 'seen'); } catch (e) { /* not remembered */ } }
$('intro_open').onclick = () => showIntro($('intro').hidden);
$('intro_close').onclick = () => showIntro(false);
try { if (localStorage.getItem('meshom.intro') !== 'seen') $('intro').hidden = false; } catch (e) { $('intro').hidden = false; }
try { showDev(localStorage.getItem('meshom.dev') === '1'); } catch (e) { showDev(false); }

S.api = {preview, rho, loadMesh, setField, loadCoast, rotate, setView, redraw, nextRender, flyTo, flyToElement, tab, addWindow, previewWindow,
  loadRunMesh, refreshRuns, refreshRefs, refreshCompare, addCompare, schedulePreview, draw, deck: () => deckgl, getViewState,
  openWin, closeWin, openWins};
S.ready = (async () => {
  const r = await api('api/config');
  S.cfg = r.config; S.layerColours = r.layer_colours; S.savedToml = r.toml; S.openedFrom = r.opened_from;
  showPath(r.path);
  initH(); initVariants(r); initTools(); initWindows(); initBlocks(); initMesh(); initEdit(); initWrite(); initRuns(r); initCompare();
  await Promise.all([initCoast(), initMethod()]);
  await Promise.all([loadCoast(), preview()]);
  tab('h');
  try { const t = sessionStorage.getItem('meshom.notice'); if (t) { sessionStorage.removeItem('meshom.notice'); status(t, false, true); } } catch (e) { /* no storage */ }
})();
