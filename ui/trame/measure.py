"""Headless measurements of the trame prototype: timings into a JSON file, screenshots into docs/figures/ui/trame/.

    scripts/on_node.sh -c 16 -t 00:30:00 -- env -u DISPLAY LP_NUM_THREADS=16 python -m ui.trame.measure

The app runs as a child process (VTK renders offscreen through EGL / Mesa llvmpipe on the CPU here); Chromium shows
the page and receives the JPEG stream over the websocket, which is what an ssh tunnel would carry.
"""
import argparse
import base64
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from playwright.sync_api import sync_playwright

from ui.common import scenario
from ui.common.shots import save_small

FIG = scenario.REPO / "docs" / "figures" / "ui" / "trame"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, default=Path(".tmp/ui/mesh"))
    ap.add_argument("--out", type=Path, default=Path(".tmp/ui/measure_trame.json"))
    a = ap.parse_args(argv)
    os.environ.pop("DISPLAY", None)
    cfg = Path(".tmp/ui/trame_scenario.toml")
    scenario.save(scenario.load(), cfg)
    port = free_port()
    t_start = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-u", "-m", "ui.trame.app", "--server", "--port", str(port),
                             "--config", str(cfg), "--mesh", str(a.mesh)],
                            stdout=open(".tmp/ui/trame_app.log", "w"), stderr=subprocess.STDOUT)
    R: dict = {}
    frames: list = []          # (time, bytes) of binary websocket frames = rendered images

    def settle(t0: float, quiet: float = 0.8, limit: float = 60.0) -> dict:
        """Wait until no image arrived for `quiet` s; first / last image after t0."""
        while True:
            time.sleep(0.05)
            pg.wait_for_timeout(1)
            now = time.perf_counter()
            after = [f for f in frames if f[0] > t0]
            if (after and now - after[-1][0] > quiet) or now - t0 > limit:
                break
        return {"first_ms": round(1e3 * (after[0][0] - t0)) if after else None,
                "last_ms": round(1e3 * (after[-1][0] - t0)) if after else None,
                "n_images": len(after), "kb": round(sum(f[1] for f in after) / 1024)}

    def shot(name: str) -> None:
        png = base64.b64decode(cdp.send("Page.captureScreenshot", {"format": "png", "fromSurface": True})["data"])
        print(f"  {name}.png {save_small(png, FIG / f'{name}.png') / 1024:.0f} KB", flush=True)

    def setstate(**kw) -> dict:
        t0 = time.perf_counter()
        pg.evaluate(f"window.trame.state.update({json.dumps(kw)})")
        return settle(t0)

    def trig(name: str, **kw):
        t0 = time.perf_counter()
        r = pg.evaluate(f"window.trame.trigger('{name}', [], {json.dumps(kw)})")
        s = settle(t0)
        return r, s

    try:
        for _ in range(240):
            try:
                socket.create_connection(("127.0.0.1", port), 0.5).close()
                break
            except OSError:
                time.sleep(0.5)
        R["server_start_s"] = round(time.perf_counter() - t_start, 2)
        with sync_playwright() as p:
            b = p.chromium.launch(args=["--disable-gpu-compositing"])
            pg = b.new_page(viewport={"width": 1400, "height": 850})
            cdp = pg.context.new_cdp_session(pg)
            pg.on("websocket", lambda ws: ws.on("framereceived", lambda d: frames.append(
                (time.perf_counter(), len(d))) if isinstance(d, bytes) else None))
            t0 = time.perf_counter()
            pg.goto(f"http://127.0.0.1:{port}/index.html")
            pg.wait_for_function("window.trame && window.trame.state && window.trame.state.get('h_stats')")
            R["page_first_image"] = settle(t0)
            R["page_ready_s"] = round(time.perf_counter() - t0, 2)

            # h preview round trips (state change -> server preview -> new image)
            R["preview_roundtrip"] = [setstate(base_h=v) for v in (90, 110, 100)]
            print(json.dumps(R), flush=True)
            trig("look", lon=-10, lat=30)
            shot("globe_h")
            R["bench_globe_h"], _ = trig("bench", view="globe", mode="off", lon=-10, lat=30, n=10)

            # interaction stream: a mouse drag over the view
            box = pg.locator(".v-main img").bounding_box()
            cx, cy = box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5
            t0 = time.perf_counter()
            pg.mouse.move(cx, cy)
            pg.mouse.down()
            for k in range(30):
                pg.mouse.move(cx + 8 * k, cy + 2 * k)
                pg.wait_for_timeout(30)
            t_up = time.perf_counter()
            pg.mouse.up()
            s = settle(t0)
            during = [f for f in frames if t0 < f[0] <= t_up]
            R["drag_h"] = {**s, "drag_s": round(t_up - t0, 2), "images_during_drag": len(during),
                           "fps_during_drag": round(len(during) / (t_up - t0), 1)}
            trig("look", lon=180, lat=58, dist=2.6)
            shot("antimeridian_h")
            trig("look", lon=0, lat=89.9)
            shot("globe_north_pole_h")
            trig("look", lon=0, lat=-89.9)
            shot("globe_south_pole_h")
            for v in ("flat", "north", "south"):
                R[f"view_{v}"] = setstate(view=v)
                shot(f"{v}_h")
            setstate(view="globe")

            # drawing: Draw polygon, four clicks on the globe, Finish; then drag the first vertex handle
            trig("look", lon=-30, lat=50, dist=3.0)
            pg.get_by_role("button", name="Draw polygon").click()
            settle(time.perf_counter())
            img = pg.locator(".v-main img").bounding_box()
            cx, cy = img["x"] + img["width"] / 2, img["y"] + img["height"] / 2
            for dx, dy in ((-80, -60), (90, -40), (60, 80), (-70, 70)):
                pg.mouse.click(cx + dx, cy + dy)
                pg.wait_for_timeout(500)
            pg.get_by_role("button", name="Finish").click()
            settle(time.perf_counter())
            pv, _ = trig("project", i=0)
            R["draw_polygon"] = pv["polygon"]
            x0, y0 = img["x"] + pv["x"] * img["width"] / pv["size"][0], img["y"] + pv["y"] * img["height"] / pv["size"][1]
            t0 = time.perf_counter()
            pg.mouse.move(x0, y0)
            pg.mouse.down()
            for k in range(1, 11):
                pg.mouse.move(x0 - 6 * k, y0 - 4 * k)
                pg.wait_for_timeout(40)
            pg.mouse.up()
            R["drag_vertex"] = settle(t0)
            R["drag_vertex_after"] = tomllib.loads(pg.evaluate("window.trame.state.get('toml')"))["size"]["layers"][-1]["polygon"][0]
            shot("draw_zone")
            pg.get_by_role("button", name="Save TOML").click()
            pg.wait_for_timeout(500)
            R["saved_toml_has_zone"] = "polygon" in cfg.read_text()

            # mesh: overview, window (island, antimeridian, pole), full
            setstate(show_h=False)
            trig("look", lon=40, lat=20)
            R["mesh_overview"] = setstate(mesh_mode="overview")
            shot("mesh_overview")
            R["bench_overview"], _ = trig("bench", view="globe", mode="overview", lon=40, lat=20, n=10)
            for name, lon, lat in (("island", 60, 10), ("antimeridian", 180, 0), ("north_pole", 0, 89.9)):
                R[f"bench_window_{name}"], _ = trig("bench", view="globe", mode="window", lon=lon, lat=lat,
                                                    dist=1.25, n=10)
                shot(f"mesh_window_{name}")
            R["bench_full"], _ = trig("bench", view="globe", mode="full", lon=40, lat=20, n=10)
            shot("mesh_full")
            t0 = time.perf_counter()
            pg.mouse.move(cx, cy)
            pg.mouse.down()
            for k in range(30):
                pg.mouse.move(cx + 8 * k, cy + 2 * k)
                pg.wait_for_timeout(30)
            t_up = time.perf_counter()
            pg.mouse.up()
            s = settle(t0)
            during = [f for f in frames if t0 < f[0] <= t_up]
            R["drag_full"] = {**s, "drag_s": round(t_up - t0, 2), "images_during_drag": len(during),
                              "fps_during_drag": round(len(during) / (t_up - t0), 1)}
            R["bench_full_north_pole"], _ = trig("bench", view="globe", mode="full", lon=0, lat=89.9, n=5)
            shot("mesh_full_north_pole")
            R["bench_full_north_map"], _ = trig("bench", view="north", mode="full", n=5)
            shot("mesh_full_north_map")
            R["bench_full_flat"], _ = trig("bench", view="flat", mode="full", n=5)
            shot("mesh_full_flat")
            b.close()
    finally:
        proc.terminate()
        proc.wait(10)
    a.out.write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
