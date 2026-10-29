"""Headless measurements of the web UI v2: timings into a JSON file, one screenshot per panel into
docs/figures/ui/web-v2/ (extra shots into ../ui_screenshots/web-v2/ outside the repository).

    scripts/on_node.sh -p interactive -c 8 -m 15G -t 00:40:00 -- env -u DISPLAY python -m ui.web.measure [--quick]

Chromium runs headless with SwiftShader (software WebGL on the CPU): frame times are an upper bound for a laptop GPU.
DISPLAY must be unset (an ssh X forwarding display makes Chromium's GPU process fail and disables WebGL). The server
runs in this process; "server" timings are taken with urllib before the browser starts.
"""
import argparse
import base64
import json
import os
import statistics
import threading
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

from ui.common import scenario
from ui.common.shots import save_small
from ui.web.server import make_server

FIG = scenario.REPO / "docs" / "figures" / "ui" / "web-v2"
EXTRA = scenario.main_checkout().parent / "ui_screenshots" / "web-v2"
CHROME_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
NE = "naturalearth/ne_50m_admin_0_countries.shp"


def frames(ms: list) -> dict:
    return {"median_ms": round(statistics.median(ms), 1), "p90_ms": round(sorted(ms)[int(0.9 * (len(ms) - 1))], 1),
            "n": len(ms)}


def server_timings(url: str, cfg: dict) -> dict:
    """Wall time of the main requests as the server answers them (no browser)."""
    def get(path):
        t = time.perf_counter()
        b = urllib.request.urlopen(url + path, timeout=600).read()
        return round(time.perf_counter() - t, 3), len(b)

    def post(path, obj):
        t = time.perf_counter()
        req = urllib.request.Request(url + path, data=json.dumps(obj).encode(), method="POST")
        b = urllib.request.urlopen(req, timeout=600).read()
        return round(time.perf_counter() - t, 3), len(b)
    R = {"mesh_info_first_s": get("/api/mesh/info")}
    for p in ("/api/mesh/mesh.bin", "/api/mesh/mesh.bin?sel=window&lon=60&lat=10&r_km=700",
              "/api/mesh/mesh.bin?sel=worst&n=2000", "/api/mesh/overview.png", "/api/mesh/irregular.bin",
              "/api/mesh/hist", "/api/mesh/worst.json?n=40", "/api/methods", "/api/coast/sources"):
        R[p] = get(p)
    for step in (0.5, 1.0):
        R[f"preview_{step}"] = [post("/api/preview", {"config": cfg, "step": step}) for _ in range(3)]
    for step in (2.0, 1.0, 0.5):
        R[f"rho_{step}"] = post("/api/rho", {"config": cfg, "step": step})
    R["window_box_0.05deg"] = post("/api/preview/window", {"config": cfg, "bounds": [170, 200, 55, 72], "step": 0.05})
    R["window_box_0.02deg"] = post("/api/preview/window", {"config": cfg, "bounds": [170, 200, 55, 72], "step": 0.02})
    return R


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, default=Path(".tmp/ui/mesh"))
    ap.add_argument("--out", type=Path, default=Path(".tmp/ui/measure_web_v2.json"))
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=880)
    ap.add_argument("--quick", action="store_true", help="no screenshots: check the panels for errors and time loads")
    a = ap.parse_args(argv)
    os.environ.pop("DISPLAY", None)
    cfg_copy = Path(".tmp/ui/web_scenario.toml")
    scenario.save(scenario.load(), cfg_copy)
    srv = make_server(cfg_copy, a.mesh, 0, run_roots=[Path(".tmp/ui/runs"), Path(".tmp/compete")])
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    R: dict = {"viewport": [a.width, a.height]}
    R["server"] = server_timings(url, scenario.to_json(scenario.load(cfg_copy)))
    print(json.dumps(R["server"]), flush=True)
    errors: list = []

    with sync_playwright() as p:
        b = p.chromium.launch(args=CHROME_ARGS)
        pg = b.new_page(viewport={"width": a.width, "height": a.height})
        cdp = pg.context.new_cdp_session(pg)
        pg.on("pageerror", lambda e: errors.append("pageerror: " + str(e)[:400]))
        pg.on("console", lambda m: errors.append(f"console.{m.type}: {m.text[:300]}") if m.type in ("error", "warning") else None)
        ev = pg.evaluate

        def shot(name, extra=False):
            if a.quick:
                return
            pg.wait_for_timeout(400)
            ev("(async () => { const d = window.__meshom.api.nextRender(); window.__meshom.api.redraw(); await d; })()")
            png = base64.b64decode(cdp.send("Page.captureScreenshot", {"format": "png", "fromSurface": True})["data"])
            kb = save_small(png, (EXTRA if extra else FIG) / f"{name}.png") / 1024
            print(f"  {name}.png {kb:.0f} KB", flush=True)

        def tap(x, y):
            # press, 50 ms, release: an instant mouse.click() is not taken as a tap by deck.gl here
            pg.mouse.move(x, y)
            pg.mouse.down()
            pg.wait_for_timeout(50)
            pg.mouse.up()
            pg.wait_for_timeout(450)                    # a tap is reported after the double-click window

        t0 = time.perf_counter()
        pg.goto(url + "/")
        pg.wait_for_function("window.__meshom && window.__meshom.ready")
        ev("window.__meshom.ready")
        R["page_ready_s"] = round(time.perf_counter() - t0, 2)
        R["renderer"] = ev("""(() => { const gl = document.createElement('canvas').getContext('webgl2');
            const d = gl.getExtension('WEBGL_debug_renderer_info'); return gl.getParameter(d.UNMASKED_RENDERER_WEBGL); })()""")
        R["preview_roundtrip_ms"] = [round(ev("window.__meshom.api.preview()")) for _ in range(5)]
        box = pg.locator("#map").bounding_box()
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

        # ---- 1 · h builder: open the Bering zone, add a latitude layer (disabled) and a second zone drawn by clicks
        api = "window.__meshom.api"
        ev(f"{api}.setView('globe', {{longitude: -35, latitude: 45, zoom: 1.6}})")
        pg.select_option("#add_kind", "latitude")
        pg.click("#add_layer")
        pg.wait_for_timeout(800)
        pg.locator(".layer").nth(3).locator("input[type=checkbox]").uncheck()
        pg.select_option("#add_kind", "zone")
        pg.click("#add_layer")
        for dx, dy in ((-90, -50), (60, -70), (90, 40), (-40, 80)):
            tap(cx + dx, cy + dy)
        pg.locator(".layer.open button", has_text="Finish").click()
        pg.wait_for_timeout(1500)
        R["drawn_zone"] = ev("window.__meshom.cfg.size.layers.at(-1).polygon")
        pg.wait_for_function("window.__meshom.stats.loads.filter(l => l.what === 'preview').length >= 7")
        shot("h_layers_globe")
        pg.check("input[name=h_overlay][value=winner]")
        ev(f"{api}.setView('flat')")
        shot("h_which_layer_wins_flat")
        t = time.perf_counter()
        pg.check("input[name=h_overlay][value=rho]")
        pg.wait_for_function("window.__meshom.h.rhoInfo")
        R["rho_1deg_click_s"] = round(time.perf_counter() - t, 2)
        pg.select_option("#rho_step", "0.5")
        pg.wait_for_function("window.__meshom.h.rhoInfo.step === 0.5")
        R["rho"] = ev("window.__meshom.stats.loads.filter(l => l.what === 'rho')")
        shot("h_rho_flat")
        ev(f"{api}.setView('north')")
        shot("h_rho_north", extra=True)
        pg.select_option("#rho_step", "1")
        pg.check("input[name=h_overlay][value=h]")
        pg.check("#limited")
        pg.wait_for_timeout(1500)

        # ---- 2 · coastline: Natural Earth, resampled at the coastal h, over Europe
        ev(f"{api}.tab('coast')")
        ne = ev("window.__meshom.coast.sources.find(s => s.path.endsWith('%s'))" % NE)
        if ne:
            pg.select_option("#coast_source", ne["path"])
            pg.wait_for_timeout(3000)
        pg.check("#coast_at_h")
        pg.wait_for_timeout(1500)
        ev(f"{api}.setView('globe', {{longitude: 15, latitude: 52, zoom: 3.2}})")
        shot("coast_natural_earth")
        R["coast_ne"] = ev("document.getElementById('coast_stats').textContent")
        gshhs = ev("window.__meshom.coast.sources.find(s => s.kind === 'GSHHS').path")
        pg.select_option("#coast_source", gshhs)
        pg.uncheck("#coast_at_h")
        pg.wait_for_timeout(3000)

        # ---- 3 · zoom windows: a box across 180° drawn by two clicks, a polygon over Indonesia, one made a zone
        ev(f"{api}.tab('zoom')")
        ev(f"{api}.setView('globe', {{longitude: 180, latitude: 62, zoom: 2.3}})")
        pg.fill("#win_step", "0.05")
        pg.click("#win_box")
        tap(cx - 160, cy - 90)
        tap(cx + 150, cy + 80)
        pg.wait_for_function("window.__meshom.win.list.length === 1 && window.__meshom.win.list[0].stats")
        R["window_box"] = {"bounds": ev("window.__meshom.win.list[0].bounds"),
                           "load": ev("window.__meshom.stats.loads.filter(l => l.what === 'window').at(-1)")}
        ev(f"{api}.addWindow({{polygon: [[96, -8], [128, -11], [141, 4], [112, 14]]}}, 0.1)")
        pg.wait_for_function("window.__meshom.win.list.length === 2 && window.__meshom.win.list[1].stats")
        pg.locator("#windows .layer").nth(1).locator("button", has_text="Make a refinement zone").click()
        pg.wait_for_timeout(2500)
        shot("zoom_windows_globe")
        ev(f"{api}.setView('flat')")
        ev(f"{api}.flyTo(180, 62, 1400)")
        pg.wait_for_timeout(800)
        shot("zoom_windows_flat_antimeridian")
        ev(f"{api}.setView('globe', {{longitude: 118, latitude: 2, zoom: 2.6}})")
        shot("zoom_window_polygon", extra=True)

        # ---- 5 · method
        ev(f"{api}.tab('method')")
        pg.select_option("#method", "m14_blocks_cut")
        pg.locator("#method_form label[data-key=cut] input").uncheck() if pg.locator("#method_form label[data-key=cut] input").count() else None
        shot("method_form")
        pg.select_option("#method", "m13_squadgen")

        # ---- 6 · mesh view on the 10^6-element mesh
        ev(f"{api}.tab('mesh')")
        ev(f"{api}.setView('globe', {{longitude: 40, latitude: 20}})")
        R["mesh_overview"] = ev(f"{api}.loadMesh('overview')")
        shot("mesh_overview_irregular")
        R["mesh_worst"] = ev(f"{api}.loadMesh('worst')")
        pg.check("#show_tris")
        t = time.perf_counter()
        pg.locator("#mesh_worst tr").nth(1).click()
        pg.wait_for_function("window.__meshom.stats.loads.filter(l => l.what === 'mesh-window').length >= 1")
        pg.wait_for_timeout(1000)
        R["worst_flyto_s"] = round(time.perf_counter() - t, 2)
        R["worst_first"] = ev("window.__meshom.mesh.worst[0]")
        shot("mesh_worst_flyto_tris_hist")
        pg.uncheck("#show_tris")
        ev(f"{api}.setView('globe', {{longitude: 40, latitude: 20}})")
        R["mesh_full_globe"] = ev(f"{api}.loadMesh('full')")
        print("full globe", json.dumps(R["mesh_full_globe"]), flush=True)
        R["frames_full_globe"] = frames(ev(f"{api}.rotate(20, 2)"))
        shot("mesh_full_globe")
        t = time.perf_counter()
        ev(f"{api}.setView('flat')")                   # expands 10^6 elements for the flat map (unwrap + copies)
        R["mesh_full_flat_switch_s"] = round(time.perf_counter() - t, 2)
        R["flat_copies"] = ev("window.__meshom.mesh.geo.flat && window.__meshom.mesh.geo.flat.copies")
        R["frames_full_flat"] = frames(ev(f"{api}.rotate(10, 1)"))
        R["recolour_kind_ms"] = round(ev(f"{api}.setField('kind')"))
        R["recolour_q_ms"] = round(ev(f"{api}.setField('q')"))
        shot("mesh_full_flat", extra=True)
        t = time.perf_counter()
        ev(f"{api}.setView('north')")
        R["mesh_full_north_switch_s"] = round(time.perf_counter() - t, 2)
        shot("mesh_full_north")
        ev(f"{api}.setView('flat')")
        ev(f"{api}.flyTo(180, -5, 450)")
        R["mesh_window_flat"] = ev(f"{api}.loadMesh('window')")
        R["window_flat_copies"] = ev("window.__meshom.mesh.geo.flat && window.__meshom.mesh.geo.flat.copies")
        shot("mesh_window_flat_antimeridian")
        R["js_heap_mb"] = ev("performance.memory ? performance.memory.usedJSHeapSize / 1e6 : null")
        ev(f"{api}.loadMesh('off')")

        # ---- 7 · runs: a dry run, the list, the log of the echo job; 4 · blocks of an m14 run
        ev(f"{api}.tab('runs')")
        pg.click("#run")
        pg.wait_for_timeout(1500)
        logs = pg.locator("#runs_list button", has_text="log")
        for i in range(logs.count()):
            if logs.nth(i).is_enabled():
                logs.nth(i).click()
                break
        pg.wait_for_timeout(1000)
        R["runs"] = ev("window.__meshom.runs.list.map(r => [r.name, r.state])")
        ev(f"{api}.setView('globe', {{longitude: -60, latitude: 10}})")
        shot("runs_panel")
        m14 = ev("(window.__meshom.runs.list.find(r => r.m14) || {}).path")
        if m14:
            ev(f"{api}.loadRunMesh('{m14}/raw.npz')")
            ev(f"{api}.tab('blocks')")
            ev(f"{api}.setField('q')")
            seed = ev("window.__meshom.blocks.data.blocks[0].seed")
            ev(f"{api}.setView('globe', {{longitude: {seed[0] + 25}, latitude: {seed[1] + 10}, zoom: 2.0}})")
            R["m14_full"] = ev(f"{api}.loadMesh('full')")
            shot("blocks_m14_run")
        b.close()
    srv.shutdown()
    R["errors"] = errors
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=1))
    print(json.dumps({k: v for k, v in R.items() if k != "server"}, indent=1))


if __name__ == "__main__":
    main()
