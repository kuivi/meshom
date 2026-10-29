"""Walk user tasks (TASKS below: a goal, then the steps a user takes) through the web UI in headless Chromium and
record what a user sees at every step: a screenshot, the text of the open panel, the legend and the status line,
console and page errors, failed or slow requests, controls that stick out of the panel.

    scripts/on_node.sh -p interactive -c 8 -m 15G -t 00:40:00 -- env -u DISPLAY PYTHONPATH=$PWD \\
        python -m ui.web.walkthrough [--out DIR] [--tasks T0,T1,...]

Output: DIR/shots/<task>-<nn>.png, DIR/steps.json, DIR/index.md; default DIR .tmp/ui/review/<stamp>. The server code
runs in this process behind Playwright's request routing (ui/web/inproc.py): no port is opened. Map navigation (pan,
zoom) goes through the page's API, as a user would do it with the mouse; everything else is clicks and typing on
the visible controls. The page runs without --allow-submit: "Run" writes a run folder only.
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

from ui.common import scenario
from ui.web import inproc
from ui.web.server import make_handler

ORIGIN = "http://meshom.local"             # never resolved: every request to it is routed to the handler
CHROME_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
API = "window.__meshom.api"
SLOW_S = 2.0                               # a request slower than this is listed with its step
PANEL_CHARS = 8000                         # the open panel's text kept per step (the Mesh panel runs to ~5000)

# what the script reads off the page after each step
VISIBLE_PANEL = "(() => { const t = [...document.querySelectorAll('section.tab')].find(s => !s.hidden); return t ? t.innerText : ''; })()"
STICKING_OUT = """(() => {
  const p = document.getElementById('panel'), R = p.getBoundingClientRect().right, out = [];
  for (const e of p.querySelectorAll('input, select, button, label, td, pre, code, span, div')) {
    if (!e.offsetParent || e.children.length > 3) continue;
    const r = e.getBoundingClientRect();
    if (r.width && r.right > R + 1) out.push(`${e.tagName.toLowerCase()}${e.id ? '#' + e.id : ''} "${(e.innerText || e.value || '').slice(0, 40)}" ends ${Math.round(r.right - R)} px past the panel`);
  }
  if (p.scrollWidth > p.clientWidth + 1) out.unshift(`panel scrolls sideways: ${p.scrollWidth} > ${p.clientWidth} px`);
  return out.slice(0, 12);
})()"""


class Walk:
    """The browser page, the step records and the request log of one walkthrough."""

    def __init__(self, pg, out: Path):
        self.pg, self.out = pg, out
        self.cdp = pg.context.new_cdp_session(pg)
        self.steps: list[dict] = []
        self.tasks: list[dict] = []
        self.errors: list[str] = []
        self.requests: list[dict] = []
        self.t0 = time.perf_counter()
        box = pg.locator("#map").bounding_box()
        self.cx, self.cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

    def begin(self, tid: str, title: str, goal: str) -> None:
        self.tasks.append({"task": tid, "title": title, "goal": goal, "steps": 0, "error": None})

    def shot(self, intent: str, did: str, notes: str = "") -> None:
        """Wait for the page to settle, then record the step."""
        pg, task = self.pg, self.tasks[-1]
        task["steps"] += 1
        pg.wait_for_timeout(700)
        pg.evaluate(f"(async () => {{ const d = {API}.nextRender(); {API}.redraw(); await d; }})()")
        name = f"{task['task']}-{task['steps']:02d}.png"
        png = base64.b64decode(self.cdp.send("Page.captureScreenshot", {"format": "png", "fromSurface": True})["data"])
        (self.out / "shots" / name).write_bytes(png)
        failed = [r for r in self.requests if r["status"] >= 400 or r.get("error")]
        slow = [f"{r['path'][:90]} {r['s']:.1f} s" for r in self.requests if r["s"] > SLOW_S]
        self.steps.append({
            "task": task["task"], "step": task["steps"], "intent": intent, "did": did, "notes": notes,
            "shot": f"shots/{name}", "t_s": round(time.perf_counter() - self.t0, 1),
            "status_line": pg.locator("#status").inner_text(), "legend": pg.locator("#legend").inner_text(),
            "panel_text": pg.evaluate(VISIBLE_PANEL)[:PANEL_CHARS], "sticking_out": pg.evaluate(STICKING_OUT),
            "console_and_page_errors": self.errors[:], "failed_requests": failed, "slow_requests": slow,
            "n_requests": len(self.requests)})
        self.errors.clear()
        self.requests.clear()
        print(f"{task['task']}-{task['steps']:02d} {intent}", flush=True)

    def tap(self, dx: float, dy: float) -> None:
        """A click on the map at (dx, dy) px from its centre: press, 50 ms, release (deck.gl takes an instant click
        for no tap), then the double-click window."""
        pg = self.pg
        pg.mouse.move(self.cx + dx, self.cy + dy)
        pg.mouse.down()
        pg.wait_for_timeout(50)
        pg.mouse.up()
        pg.wait_for_timeout(450)

    def ev(self, js: str):
        return self.pg.evaluate(js)

    def loads(self, what: str) -> int:
        return self.ev(f"window.__meshom.stats.loads.filter(l => l.what === '{what}').length")

    def wait_loads(self, what: str, n0: int, timeout_ms: int = 120000) -> None:
        self.pg.wait_for_function(f"window.__meshom.stats.loads.filter(l => l.what === '{what}').length > {n0}",
                                  timeout=timeout_ms)

    def tab(self, name: str) -> None:
        self.pg.click(f"nav.tabs button[data-tab={name}]")


# ------------------------------------------------------------------------------------------------------ the tasks
def t0_first_look(w: Walk) -> None:
    w.begin("T0", "First look", "A new user opens the page for the first time, without instructions: what is this "
            "tool, what can be done here, where to start?")
    w.shot("See what the tool is and where to start", "opened the page")
    if w.pg.locator("#intro_close").is_visible():
        w.pg.mouse.move(w.cx, w.cy)
        w.pg.hover("button[data-tab=coast]")
        w.pg.wait_for_timeout(1200)
        w.shot("Read the intro and a button's hover tip", "hovered over 'Coast' (its tip)")
        w.pg.click("#intro_close")
    for name, label in (("coast", "Coast"), ("method", "Method"), ("blocks", "Blocks"), ("runs", "Runs"),
                        ("mesh", "Mesh"), ("refs", "Reference meshes")):
        w.tab(name)
        w.shot(f"Find out what the panel '{label}' is for", f"clicked the tab '{label}'")
    w.tab("h")
    w.pg.click("button[data-view=flat]")
    w.shot("See the whole world at once", "back to 'Size h', clicked 'Flat'")
    w.pg.click("button[data-view=globe]")
    w.pg.select_option("#skin_pick", "glass")
    w.pg.wait_for_timeout(1500)
    w.shot("Try the other look", "chose 'Glass' under 'look' in the footer")
    w.pg.select_option("#skin_pick", "daylight")
    w.pg.wait_for_timeout(1000)


def t1_size_field(w: Walk) -> None:
    w.begin("T1", "Size field", "Set up the size field of a global mesh: 100 km in the open ocean, 20 km along the "
            "coasts and a 10 km refinement zone in the Nordic Seas (between Greenland and Norway). Find out how many "
            "nodes the mesh will have and whether the size changes smoothly everywhere.")
    pg = w.pg
    pg.locator("#layers .layer").nth(0).locator(".layer-title").click()
    w.shot("Check the open-ocean size (100 km)", "clicked the first layer ('constant') to open it")
    pg.locator("#layers .layer").nth(1).locator(".layer-title").click()
    w.shot("Check the coastal size (20 km)", "clicked the second layer ('coast_distance') to open it")
    pg.select_option("#add_kind", "zone")
    pg.click("#add_layer")
    w.shot("Add a refinement zone", "chose 'zone' in the list next to 'Add layer', clicked 'Add layer'")
    w.ev(f"{API}.flyTo(0, 70, 1500)")
    pg.wait_for_timeout(800)
    for dx, dy in ((-130, 70), (-50, -100), (120, -80), (130, 80)):
        w.tap(dx, dy)
    w.shot("Draw the zone around the Nordic Seas", "moved the view to the Nordic Seas (script), clicked four points "
           "on the map")
    n0 = w.loads("preview")
    pg.locator(".layer.open button", has_text="Finish").click()
    w.wait_loads("preview", n0)
    w.shot("Close the zone", "clicked 'Finish' in the zone's form")
    pg.locator(".layer.open label", has_text="h inside").locator("input").fill("10")     # the default: no new preview
    pg.wait_for_timeout(1000)
    w.shot("Set 10 km inside the zone", "typed 10 into 'h inside, km'")
    pg.check("input[name=h_overlay][value=winner]")
    w.shot("See which layer sets the size where", "chose 'which layer sets h' above the map")
    pg.check("input[name=h_overlay][value=h]")
    pg.wait_for_function("window.__meshom.h.runReady()", timeout=300000)
    w.shot("See the size the run will use, with smooth transitions", "chose 'h' again and waited: the page computes "
           "the run's h in the background after each change")
    pg.mouse.move(w.cx + 60, w.cy + 30)
    pg.wait_for_timeout(300)
    w.shot("Read h at one point", "moved the mouse over the map")
    pg.locator("#h_stats button", has_text="Fly to").click()
    pg.wait_for_timeout(800)
    w.shot("Find where the size changes fastest", "clicked 'Fly to' next to the steepest change of h")
    pg.click("#save")
    pg.locator("#config summary").click()
    w.shot("Save the setup", "clicked 'Save', opened 'config (TOML)'")
    pg.locator("#config summary").click()


def t6_window_to_zone(w: Walk) -> None:
    w.begin("T6", "Finer look, then a refinement zone", "Look at the size field around Italy at a finer step, then "
            "make that area a 15 km refinement zone.")
    pg = w.pg
    w.tab("h")
    pg.click("#zoom_section summary")
    w.ev(f"{API}.flyTo(15, 40, 1400)")
    pg.click("#win_box")
    w.shot("Start a box", "in 'Size h' opened 'Look closer at a region (zoom windows)', moved the view to Italy "
           "(script), clicked 'Box'")
    n0 = w.loads("window")
    w.tap(-150, 120)
    w.tap(140, -110)
    w.wait_loads("window", n0, 300000)
    w.shot("See the finer preview of the box", "clicked two opposite corners on the map")
    form = pg.locator("#windows .layer").last
    form.locator("label", has_text="zone h").locator("input").fill("15")
    form.locator("label", has_text="zone h").locator("input").press("Tab")
    form.locator("button", has_text="Make a refinement zone").click()
    pg.wait_for_timeout(1500)
    w.shot("Make the box a refinement zone", "typed 15 into 'zone h, km', clicked 'Make a refinement zone'")
    pg.locator("#layers .layer.open").scroll_into_view_if_needed()
    w.shot("Find the new zone among the layers", "scrolled up to the layers")


def t2_coast(w: Walk) -> None:
    w.begin("T2", "Coastline", "Check the coastline: which data set is used, whether Antarctica is in, drop the "
            "islands smaller than 1000 km², look at a complicated coast up close.")
    pg = w.pg
    w.tab("coast")
    opts = pg.eval_on_selector_all("#coast_source option", "os => os.map(o => o.textContent)")
    w.shot("See which coastline is used and what else there is", "opened 'Coast'",
           f"the 'source' list has {len(opts)} entries: " + " | ".join(opts[:14]) + (" | …" if len(opts) > 14 else ""))
    n0 = w.loads("preview")
    pg.fill("#min_area", "1000")
    pg.press("#min_area", "Tab")
    w.wait_loads("preview", n0)
    w.shot("Drop islands smaller than 1000 km²", "typed 1000 into 'drop islands smaller than, km²'")
    pg.click("button[data-view=south]")
    pg.wait_for_timeout(1500)
    w.shot("Check Antarctica", "clicked 'South'")
    n0 = w.loads("preview")
    pg.select_option("#antarctica", "grounding")
    w.wait_loads("preview", n0)
    w.shot("Use the grounding line for Antarctica", "chose 'grounding line (L6)' under 'Antarctica'")
    pg.select_option("#antarctica", "ice_front")
    pg.click("button[data-view=globe]")
    w.ev(f"{API}.flyTo(-95, 72, 900)")
    pg.check("#coast_at_h")
    pg.wait_for_timeout(1500)
    w.shot("See the Canadian Archipelago as the mesh will see it", "Antarctica back to 'ice front', 'Globe', moved "
           "to the Canadian Archipelago (script), ticked 'draw the coast at the coastal h'")


def t3_method_run(w: Walk) -> None:
    w.begin("T3", "Method and run", "Choose the block method (m14), look at its options, change one and start the "
            "mesh run on Levante.")
    pg = w.pg
    w.tab("method")
    m14 = next(v for v in pg.eval_on_selector_all("#method option", "os => os.map(o => o.value)") if v.startswith("m14"))
    pg.select_option("#method", m14)
    w.shot("Choose the block method", f"opened 'Method', chose '{m14}'")
    lab = pg.locator("#method_form label[data-key]").filter(has=pg.locator("input[type=number]")).first
    key, inp = lab.get_attribute("data-key"), lab.locator("input")
    old = inp.input_value()
    new = str(float(old) * 2 if "." in old else int(old) + 1)
    inp.fill(new)
    inp.press("Tab")
    w.shot("Change one option", f"changed '{key}' from {old} to {new}")
    w.tab("runs")
    w.shot("Find where to start the run", "opened 'Runs'")
    pg.click("#run")
    pg.wait_for_function("document.getElementById('run_result').textContent.length > 0")
    w.shot("Start the run", "clicked the run button (this page runs without --allow-submit)")


def _wait_mesh(pg, name: str) -> None:
    pg.wait_for_function("(() => { const m = window.__meshom.mesh; return m.info && m.info.path.includes('%s') "
                         "&& m.mode !== 'off'; })()" % name, timeout=180000)


def t4_judge_mesh(w: Walk) -> None:
    w.begin("T4", "Judge a mesh", "Open the finished demo run global_150km_m14 and judge its quality: where are the "
            "worst elements, what do they look like up close, where are the triangles and the irregular nodes.")
    pg = w.pg
    w.tab("runs")
    n0 = w.loads("mesh-overview")
    pg.locator("#runs_list tr", has_text="global_150km_m14").locator("button:not(.add)", has_text="raw").click()
    _wait_mesh(pg, "global_150km_m14")
    w.wait_loads("mesh-overview", n0)
    w.shot("Open the run's mesh and see it whole", "clicked 'raw' in the row of global_150km_m14")
    n0 = w.loads("mesh-worst")
    pg.click("button[data-mesh=worst]")
    w.wait_loads("mesh-worst", n0)
    w.shot("See the worst elements", "clicked 'Worst'")
    n0 = w.loads("mesh-window")
    pg.locator("#mesh_worst tr").nth(1).click()
    w.wait_loads("mesh-window", n0)
    w.shot("Look at the worst element up close", "clicked the first row of 'worst elements'")
    n0 = w.loads("mesh-full")
    pg.click("button[data-mesh=full]")
    w.wait_loads("mesh-full", n0)
    w.shot("See all elements", "clicked 'Full'")
    w.ev(f"{API}.flyTo(8, 62, 500)")
    pg.wait_for_timeout(800)
    w.shot("Look closely at the Norwegian coast", "moved the view to Norway (script)")
    pg.select_option("#field", "kind")
    w.shot("See which elements are triangles", "chose 'tri / quad' under 'colour'")
    pg.select_option("#field", "q")
    pg.click("button[data-view=north]")
    pg.wait_for_timeout(1500)
    w.shot("See the Arctic", "colour back to 'max angle / gate', clicked 'North'")
    pg.locator("#mesh_gates").scroll_into_view_if_needed()
    w.shot("Read the quality numbers", "scrolled the panel down to the gates")


def t5_reference(w: Walk) -> None:
    w.begin("T5", "Compare with a reference mesh", "Compare our mesh with the FESOM2 CORE2 mesh: how do its quality "
            "numbers and its worst elements look next to ours?")
    pg = w.pg
    pg.click("button[data-view=globe]")
    w.tab("refs")
    w.shot("Find the reference meshes", "opened 'Reference meshes'")
    n0 = w.loads("mesh-overview") + w.loads("mesh-full")
    pg.locator("#refs_list tr", has_text="fesom2_core2").locator("button", has_text="load").click()
    _wait_mesh(pg, "fesom2_core2")
    pg.wait_for_function("window.__meshom.stats.loads.filter(l => l.what === 'mesh-overview' || l.what === 'mesh-full')"
                         f".length > {n0}", timeout=180000)
    w.shot("Load the FESOM2 CORE2 mesh", "clicked 'load' in the row of fesom2_core2")
    pg.locator("#mesh_gates").scroll_into_view_if_needed()
    w.shot("See its quality", "scrolled the panel down to the gates")


def t7_rerun(w: Walk) -> None:
    w.begin("T7", "Change a finished run and run it again", "Take the config of the finished run global_150km_m02, "
            "make the open ocean coarser (120 km), save it under a new name and write a new run.")
    pg = w.pg
    w.tab("runs")
    pg.locator("#runs_list tr", has_text="global_150km_m02").locator("button", has_text="open config").click()
    again = pg.locator("#runs_list button", has_text="click again")      # only with unsaved changes
    try:
        again.first.wait_for(timeout=10000)
        again.first.click()
    except PlaywrightTimeout:
        pass
    pg.wait_for_function("window.__meshom && window.__meshom.cfg && window.__meshom.tab === 'h' && "
                         "window.__meshom.stats.loads.some(l => l.what === 'preview')", timeout=180000)
    w.shot("Open the run's config", "clicked 'open config' in the row of global_150km_m02 (the page reloads with it)")
    pg.locator("#layers .layer").nth(0).locator(".layer-title").click()
    pg.locator(".layer.open label", has_text="h, km").locator("input").fill("120")
    pg.wait_for_timeout(1500)
    w.shot("Make the open ocean coarser", "opened the first layer, typed 120 into 'h, km'")
    pg.click("#save_as")
    pg.fill("#save_path", str(w.out / "config-rerun.toml"))
    pg.click("#save_as_go")
    pg.wait_for_timeout(800)
    w.shot("Save it under a new name", "clicked 'Save as …', typed a new path, clicked 'Save here'")
    w.tab("runs")
    pg.click("#run")
    pg.wait_for_function("document.getElementById('run_result').textContent.length > 0")
    w.shot("Write the new run", "opened 'Runs', clicked the run button")


def t8_remesh_zone(w: Walk) -> None:
    w.begin("T8", "Look at a remeshed zone", "A zone around the worst element of global_150km_m14 was remeshed with "
            "`meshom repair` (the page cannot start it yet: greyed out). Open the result and compare before and after.")
    pg = w.pg
    run = w.out / "runs" / "t8_m14"                    # a private copy: repeated walks do not pile up repaired files
    if not run.exists():
        shutil.copytree(Path(".tmp/runs/global_150km_m14"), run)
    worst = w.ev("window.__meshom.mesh.worst ? window.__meshom.mesh.worst[0] : null")
    lon, lat = (worst["lon"], worst["lat"]) if worst else (170.6, -36.4)
    poly = [[lon - 3, lat - 2], [lon + 3, lat - 2], [lon + 3, lat + 2], [lon - 3, lat + 2]]
    cmd = [sys.executable, "-m", "meshom.cli", "repair", str(run), "--mesh", "raw.npz", "--out", "repaired-01",
           "--polygon", json.dumps(poly)]
    t = time.perf_counter()
    r = subprocess.run(cmd, cwd=scenario.REPO, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(scenario.REPO)))
    (w.out / "t8_repair.log").write_text(r.stdout + r.stderr)
    w.tab("mesh")
    w.shot("See that remeshing is planned", "opened 'Mesh' (the section 'Remesh a zone' is greyed out)")
    pg.click("button[data-view=globe]")
    w.tab("runs")
    pg.click("#runs_refresh")
    btn = pg.locator("#runs_list tr", has_text="t8_m14").locator("button:not(.add)", has_text="repaired-01")
    btn.wait_for()
    btn.click()
    _wait_mesh(pg, "repaired-01")
    w.shot("Open the remeshed mesh", f"meshom repair ran on this node ({time.perf_counter() - t:.0f} s, exit "
           f"{r.returncode}); clicked 'repaired-01' in the row of t8_m14", (r.stdout + r.stderr)[-500:])
    pg.locator("#repair_box button", has_text="Fly to the zone").click()
    pg.wait_for_timeout(1000)
    n0 = w.loads("mesh-full")
    pg.click("button[data-mesh=full]")
    w.wait_loads("mesh-full", n0)
    pg.locator("#repair_box").scroll_into_view_if_needed()
    w.shot("Read before and after", "clicked 'Fly to the zone', then 'Full'")
    pg.locator("#repair_box button", has_text="Show before").click()
    _wait_mesh(pg, "raw.npz")
    pg.wait_for_timeout(800)
    w.shot("See the zone before", "clicked 'Show before'")
    pg.locator("#repair_box button", has_text="Show after").click()
    _wait_mesh(pg, "repaired-01")
    pg.wait_for_timeout(800)
    w.shot("See the zone after", "clicked 'Show after'")


def t9_compare(w: Walk) -> None:
    w.begin("T9", "Compare meshes", "Put the raw meshes of global_150km_m14 and global_150km_m02 and the FESOM2 CORE2 "
            "mesh side by side: which one is best on which number, which gates fail where?")
    pg = w.pg
    pg.click("button[data-view=globe]")
    w.tab("runs")
    for run in ("global_150km_m14", "global_150km_m02"):
        pg.locator("#runs_list tr", has_text=run).locator("button.add", has_text="+ raw").click()
        pg.wait_for_timeout(500)                       # the list re-renders with the button ticked
    w.shot("Pick two runs to compare", "clicked '+ raw' (under 'compare') in the rows of global_150km_m14 and "
           "global_150km_m02")
    w.tab("refs")
    pg.locator("#refs_list tr", has_text="fesom2_core2").locator("button.add").click()
    pg.wait_for_timeout(500)
    w.shot("Add the FESOM2 CORE2 mesh", "opened 'Reference meshes', clicked '+ compare' in the row of fesom2_core2")
    w.tab("compare")
    pg.wait_for_selector("#compare_table table.compare")
    w.shot("Read the comparison", "opened 'Compare'")
    pg.locator("#compare_table th", has_text="fesom2_core2").locator("button", has_text="show").click()
    _wait_mesh(pg, "fesom2_core2")
    w.shot("Open one of them", "clicked 'show' under fesom2_core2")


def t10_edit_by_hand(w: Walk) -> None:
    w.begin("T10", "Edit a mesh by hand", "Open global_150km_m14 raw, look at its worst element up close, drag one node "
            "and add one node there, then save the result as a new file next to the mesh.")
    pg = w.pg
    if pg.locator("#intro_close").is_visible():
        pg.click("#intro_close")
    run = w.out / "runs" / "t9_m14"                    # a private copy: the saved edited-NN files stay in the walk
    if not run.exists():
        run.mkdir(parents=True)
        src = scenario.main_checkout() / ".tmp" / "runs" / "global_150km_m14"
        for f in ("raw.npz", "quality.json", "config.resolved.toml", "stages.json"):
            shutil.copy(src / f, run / f)
    w.tab("runs")
    pg.click("#runs_refresh")
    drawn = "window.__meshom.stats.loads.filter(l => /^mesh-(overview|worst|window|full)$/.test(l.what)).length"
    n0 = w.ev(drawn)                                   # the mesh opens in the mode the page is in (overview or full)
    pg.locator("#runs_list tr", has_text="t9_m14").locator("button:not(.add)", has_text="raw").click()
    _wait_mesh(pg, "t9_m14")
    pg.wait_for_function(f"{drawn} > {n0}", timeout=120000)
    n0 = w.loads("mesh-window")
    pg.locator("#mesh_worst tr").nth(1).click()
    w.wait_loads("mesh-window", n0)
    w.shot("Look at the worst element up close", "opened raw of t9_m14 (a copy of global_150km_m14), clicked the "
           "first row of 'worst elements'")
    pg.click("#edit_section summary")
    pg.check("#edit_on")
    pg.locator("#edit_section").scroll_into_view_if_needed()
    w.shot("Switch on editing", "opened 'Edit by hand', ticked 'edit the zoom window by hand'")
    t = w.ev("window.__meshom.edit.targets()")
    (x, y) = t["node"]
    pg.mouse.move(x + 6, y + 6)
    pg.mouse.move(x, y)                                 # the hover over the dot stops the map from panning
    pg.wait_for_timeout(300)
    pg.mouse.down()
    for k in range(1, 11):
        pg.mouse.move(x + 2.5 * k, y - 1.5 * k)
        pg.wait_for_timeout(40)
    w.shot("Drag a node", f"pressed the white dot of node {t['nodeId']} nearest the centre and moved the mouse "
           "25 px right, 15 px up (button still down)",
           w.ev("JSON.stringify(window.__meshom.edit.drag)"))
    pg.mouse.up()
    pg.wait_for_timeout(300)
    w.shot("See the moved node's elements and their quality", "released the mouse",
           f"edits: {w.ev('JSON.stringify(window.__meshom.edit.ops)')}")
    pg.check("input[name=edit_tool][value=add]")
    pg.wait_for_timeout(500)                           # the map redraws without the dots first
    t = w.ev("window.__meshom.edit.targets()")
    w.tap(t["elem"][0] - w.cx, t["elem"][1] - w.cy)
    w.shot("Add a node", f"chose 'add a node', clicked inside element {t['elemId']} near the centre",
           f"edits: {w.ev('JSON.stringify(window.__meshom.edit.ops)')}")
    pg.click("#edit_save")
    pg.wait_for_function("window.__meshom.edit.saved !== null", timeout=120000)
    saved = sorted(p.name for p in run.glob("edited-*"))
    w.shot("Save the edits", "clicked 'Save as edited-NN'", f"files in the run folder: {saved}")
    w.tab("runs")
    pg.click("#runs_refresh")
    btn = pg.locator("#runs_list tr", has_text="t9_m14").locator("button:not(.add)", has_text="edited-01")
    btn.wait_for()
    w.shot("Find the saved mesh in Runs", "opened 'Runs'")


def t11_windows(w: Walk) -> None:
    w.begin("T11", "Tables in windows", "Open the run list and the reference meshes in their own windows over the map, "
            "sort them, keep only the FESOM2 meshes, load one from the window, then look at its worst elements in a "
            "window sorted by latitude.")
    pg = w.pg
    pg.click("button[data-view=globe]")
    w.tab("runs")
    pg.click("#runs_win")
    runs = pg.locator(".fwin[data-win=runs]")
    runs.locator("tbody tr").first.wait_for()
    runs.locator("th", has_text="nodes").click()
    runs.locator("th", has_text="nodes").click()
    w.shot("Open the run list in a window, the largest mesh first", "opened 'Runs', clicked 'Open in a window', "
           "clicked the column 'nodes' twice")
    w.tab("refs")
    pg.click("#refs_win")
    refs = pg.locator(".fwin[data-win=refs]")
    refs.locator("tbody tr").first.wait_for()
    refs.locator("th", has_text="resolution").click()
    w.shot("Open the reference meshes in a window, the finest first", "opened 'Reference meshes', clicked 'Open in a "
           "window', clicked the column 'resolution'")
    refs.locator("input.stbl-filter").fill("fesom2")
    w.shot("Keep only the FESOM2 meshes", "typed 'fesom2' into the window's filter box")
    bar = refs.locator(".fwin-bar").bounding_box()
    x, y = bar["x"] + 120, bar["y"] + bar["height"] / 2
    pg.mouse.move(x, y)
    pg.mouse.down()
    pg.mouse.move(x - 60, y + 200, steps=8)
    pg.mouse.up()
    w.shot("Move the window out of the way", "dragged the window's title bar down by 200 px")
    n0 = w.loads("mesh-overview") + w.loads("mesh-full")
    refs.locator("tr", has_text="fesom2_core2").locator("button", has_text="load").click()
    _wait_mesh(pg, "fesom2_core2")
    pg.wait_for_function("window.__meshom.stats.loads.filter(l => l.what === 'mesh-overview' || l.what === 'mesh-full')"
                         f".length > {n0}", timeout=180000)
    w.shot("Load a mesh from the window", "clicked 'load' in the row of fesom2_core2 in the window")
    pg.click("#worst_win")
    worst = pg.locator(".fwin[data-win=worst]")
    worst.locator("tbody tr").first.wait_for()
    worst.locator("th", has_text="lat").click()
    w.shot("See the worst elements in a window, from the south", "clicked 'Open in a window' above the worst "
           "elements, clicked the column 'lat'")
    n0 = w.loads("mesh-window")
    worst.locator("tbody tr").first.click()
    w.wait_loads("mesh-window", n0)
    w.shot("Fly to the southernmost of them", "clicked the first row of the window")
    w.tab("compare")
    pg.click("#compare_win")
    pg.wait_for_timeout(800)
    w.shot("Keep the comparison in view", "opened 'Compare', clicked 'Open in a window'")
    for look, name in (("glass", "Glass"), ("app", "Classic")):
        pg.select_option("#skin_pick", look)
        pg.wait_for_timeout(1500)
        w.shot(f"See the windows in the look '{name}'", f"chose '{name}' under 'look' in the footer")
    pg.select_option("#skin_pick", "daylight")
    pg.wait_for_timeout(1000)
    pg.keyboard.press("Escape")
    w.shot("Close the front window", "pressed Escape (the Compare window closes)")
    w.ev(f"{API}.openWins().forEach((id) => {API}.closeWin(id))")


TASKS = {"T0": t0_first_look, "T1": t1_size_field, "T6": t6_window_to_zone, "T2": t2_coast, "T3": t3_method_run,
         "T4": t4_judge_mesh, "T5": t5_reference, "T7": t7_rerun, "T8": t8_remesh_zone, "T9": t9_compare, "T10": t10_edit_by_hand, "T11": t11_windows}


# --------------------------------------------------------------------------------------------------------- output
def index_md(steps: list[dict], tasks: list[dict], meta: dict) -> str:
    dirty = f", with uncommitted changes in {len(meta['uncommitted'])} files" if meta["uncommitted"] else ""
    out = [f"# UI walkthrough {meta['stamp']}", "",
           f"Commit {meta['commit']}{dirty}. Viewport {meta['viewport'][0]} × {meta['viewport'][1]}, renderer "
           f"{meta['renderer']}, page ready in {meta['page_ready_s']} s, {meta['wall_s']} s in all. The tasks ran in "
           "this order in one page: each starts from the state the previous one left. Data per step: steps.json "
           f"(the panel text cut at {PANEL_CHARS} characters).", ""]
    for t in tasks:
        out += [f"## {t['task']} · {t['title']}", "", f"Goal: {t['goal']}", ""]
        if t["error"]:
            out += [f"**The script stopped here:** `{t['error'].splitlines()[-1]}`", ""]
        for s in (s for s in steps if s["task"] == t["task"]):
            flags = [f"{len(s[k])} {k.replace('_', ' ')}" for k in ("console_and_page_errors", "failed_requests",
                                                                     "slow_requests", "sticking_out") if s[k]]
            out.append(f"{s['step']}. **{s['intent']}** — {s['did']}. `{s['shot']}`" + (f" ({', '.join(flags)})" if flags else ""))
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(".tmp/ui/review") / time.strftime("%Y%m%d-%H%M"))
    ap.add_argument("--mesh", type=Path, default=Path(".tmp/ui/mesh"))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=880)
    a = ap.parse_args(argv)
    os.environ.pop("DISPLAY", None)
    out = a.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "shots").mkdir(parents=True)
    cfg = out / "config.toml"                              # "Save TOML" writes here, not into the repository
    scenario.save(scenario.load(), cfg)
    handler = make_handler(cfg, a.mesh, run_roots=[out / "runs", Path(".tmp/runs"), Path(".tmp/compete")])
    t_start = time.perf_counter()

    with sync_playwright() as p:
        b = p.chromium.launch(args=CHROME_ARGS)
        pg = b.new_page(viewport={"width": a.width, "height": a.height})
        pg.set_default_timeout(120000)
        w: Walk | None = None
        log: list[dict] = []

        def serve(route):
            req = route.request
            u = urlsplit(req.url)
            path = u.path + (f"?{u.query}" if u.query else "")
            t = time.perf_counter()
            try:
                s, hs, body = inproc.call(handler, req.method, path, req.post_data_buffer or b"")
            except Exception as e:                         # the handler reports its own errors; this is the transport's
                s, hs, body = 500, {"Content-Type": "text/plain"}, f"{type(e).__name__}: {e}".encode()
            rec = {"path": path, "status": s, "s": round(time.perf_counter() - t, 2)}
            if body.startswith(b'{"error"') or s >= 400:
                rec["error"] = body[:300].decode(errors="replace")
            (w.requests if w else log).append(rec)
            route.fulfill(status=s, body=body, headers={k: v for k, v in hs.items()
                                                        if k.lower() in ("content-type", "set-cookie", "cache-control")})

        pg.route(ORIGIN + "/**", serve)
        errs: list[str] = []
        pg.on("pageerror", lambda e: (w.errors if w else errs).append("pageerror: " + str(e)[:400]))
        pg.on("console", lambda m: (w.errors if w else errs).append(f"console.{m.type}: {m.text[:300]}")
              if m.type in ("error", "warning") else None)
        pg.on("requestfailed", lambda r: (w.errors if w else errs).append(f"request failed: {r.url[:120]} {r.failure}"))
        t = time.perf_counter()
        pg.goto(ORIGIN + "/")
        pg.wait_for_function("window.__meshom && window.__meshom.ready")
        pg.evaluate("window.__meshom.ready")
        ready_s = round(time.perf_counter() - t, 2)
        renderer = pg.evaluate("""(() => { const gl = document.createElement('canvas').getContext('webgl2');
            const d = gl.getExtension('WEBGL_debug_renderer_info'); return gl.getParameter(d.UNMASKED_RENDERER_WEBGL); })()""")
        w = Walk(pg, out)
        w.errors += errs
        w.requests += log
        for tid in a.tasks.split(","):
            try:
                TASKS[tid](w)
            except Exception:                              # record, show the page as it is, go on with the next task
                if not w.tasks or w.tasks[-1]["task"] != tid:
                    w.begin(tid, TASKS[tid].__name__, "")
                w.tasks[-1]["error"] = traceback.format_exc(limit=3)
                print(w.tasks[-1]["error"], flush=True)
                try:
                    w.shot("(the script failed here)", "see the task's error")
                except Exception:
                    pass
        b.close()

    git = lambda *c: subprocess.run(["git", *c], cwd=scenario.REPO, capture_output=True, text=True).stdout.strip()  # noqa: E731
    meta = {"stamp": out.name, "viewport": [a.width, a.height], "renderer": renderer, "page_ready_s": ready_s,
            "wall_s": round(time.perf_counter() - t_start, 1), "commit": git("rev-parse", "--short", "HEAD"),
            "uncommitted": git("status", "--porcelain", "--untracked-files=no").splitlines()}
    (out / "steps.json").write_text(json.dumps({"meta": meta, "tasks": w.tasks, "steps": w.steps}, indent=1,
                                               ensure_ascii=False))
    (out / "index.md").write_text(index_md(w.steps, w.tasks, meta))
    print(f"{len(w.steps)} steps, {sum(bool(t['error']) for t in w.tasks)} tasks stopped -> {out}", flush=True)


if __name__ == "__main__":
    main()
