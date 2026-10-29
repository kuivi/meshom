"""Headless measurements of the Qt prototype under Xvfb: timings into a JSON file, screenshots into docs/figures/ui/qt/.

    scripts/on_node.sh -c 16 -t 00:30:00 -- env LP_NUM_THREADS=16 python -m ui.qt.measure

Rendering is Mesa llvmpipe in a virtual X server (no GPU): frame times are an upper bound for a laptop GPU.
"""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from ui.common import scenario
from ui.common.shots import save_small
from ui.qt import xvfb

FIG = scenario.REPO / "docs" / "figures" / "ui" / "qt"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=Path, default=Path(".tmp/ui/mesh"))
    ap.add_argument("--out", type=Path, default=Path(".tmp/ui/measure_qt.json"))
    ap.add_argument("--display", type=int, default=90 + os.getpid() % 9)
    a = ap.parse_args(argv)
    xvfb.start(a.display)                                # exits by itself after Qt disconnects
    run(a)


def run(a) -> None:
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtTest import QTest

    from ui.qt.app import MainWindow

    cfg = Path(".tmp/ui/qt_scenario.toml")
    scenario.save(scenario.load(), cfg)
    R: dict = {}
    t0 = time.perf_counter()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["meshom-ui-qt"])
    w = MainWindow(cfg, a.mesh)
    w.resize(1400, 850)
    w.show()
    app.processEvents()
    w.plotter.render()
    app.processEvents()
    R["startup_s"] = round(time.perf_counter() - t0, 2)
    R["gl_renderer"] = w.plotter.ren_win.ReportCapabilities().split("OpenGL renderer string:")[-1].split("\n")[0].strip()
    sc, it = w.sc, w.plotter.interactor

    def shot(name: str) -> None:
        w.plotter.render()
        app.processEvents()
        pm = w.screen().grabWindow(w.winId())
        buf = QtCore.QBuffer()
        buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
        pm.save(buf, "PNG")
        print(f"  {name}.png {save_small(bytes(buf.data()), FIG / f'{name}.png') / 1024:.0f} KB", flush=True)

    def frames(n=10, step=2.0) -> dict:
        ms = [1e3 * x for x in sc.rotate_frames(n, step)]
        app.processEvents()
        return {"median_ms": round(statistics.median(ms), 1), "max_ms": round(max(ms), 1), "n": n}

    def timed(f, *args) -> float:
        t = time.perf_counter()
        f(*args)
        app.processEvents()
        w.plotter.render()
        return round(time.perf_counter() - t, 3)

    R["preview_s"] = []
    for v in (90.0, 110.0, 100.0):
        w.inputs["base_h"].blockSignals(True)
        w.inputs["base_h"].setValue(v)
        w.inputs["base_h"].blockSignals(False)
        R["preview_s"].append(timed(w.preview))
    sc.look(-10, 30)
    shot("globe_h")
    R["frames_globe_h"] = frames()
    sc.look(180, 58, 2.6)
    shot("antimeridian_h")
    sc.look(0, 89.9)
    shot("globe_north_pole_h")
    sc.look(0, -89.9)
    shot("globe_south_pole_h")
    for v in ("flat", "north", "south"):
        R[f"view_{v}_s"] = timed(w.set_view, v)
        shot(f"{v}_h")
    w.set_view("globe")

    # drawing: four clicks on the globe, Finish, then drag the first vertex handle
    sc.look(-30, 50, 3.0)
    w.plotter.render()
    w.draw()
    app.processEvents()
    W, H = it.width(), it.height()
    for dx, dy in ((-80, -60), (90, -40), (60, 80), (-70, 70)):
        QTest.mouseClick(it, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
                         QtCore.QPoint(W // 2 + dx, H // 2 + dy))
        app.processEvents()
    w.finish()
    app.processEvents()
    R["draw_polygon"] = [list(v) for v in sc.zone().polygon]        # a copy: the drag below edits the list
    lon, lat = sc.zone().polygon[0]
    wp = sc.to_view(np.array([[lon, lat]]), "zone")[0]
    ren = w.plotter.renderer
    ren.SetWorldPoint(*wp, 1.0)
    ren.WorldToDisplay()
    x, y, _ = ren.GetDisplayPoint()
    # drag the handle with VTK interactor events: QMouseEvents sent to the QVTK widget did not move the handle in
    # this headless setup, while real mouse events reach VTK the same way as these
    iren = w.plotter.iren.interactor
    t = time.perf_counter()
    iren.SetEventInformation(int(x), int(y))
    iren.LeftButtonPressEvent()
    for k in range(1, 11):
        iren.SetEventInformation(int(x) - 6 * k, int(y) + 4 * k)
        iren.MouseMoveEvent()
        app.processEvents()
    iren.LeftButtonReleaseEvent()
    t_end = time.perf_counter()
    while w.timer.isActive():                       # the debounced preview after the last move
        app.processEvents()
    R["drag_vertex_move_ms"] = round(1e3 * (t_end - t) / 10, 1)
    R["drag_vertex_s"] = round(time.perf_counter() - t, 2)          # moves + debounced preview
    R["drag_vertex_after"] = sc.zone().polygon[0]
    shot("draw_zone")
    w.save()
    R["saved_toml_has_zone"] = "polygon" in cfg.read_text()
    print(json.dumps(R), flush=True)

    # mesh
    w.show_h.setChecked(False)
    w.preview()
    sc.look(40, 20)
    R["mesh_overview_s"] = timed(w.set_mesh, "overview")
    shot("mesh_overview")
    R["frames_overview"] = frames()
    for name, lon, lat in (("island", 60, 10), ("antimeridian", 180, 0), ("north_pole", 0, 89.9)):
        sc.look(lon, lat, 1.25)
        R[f"mesh_window_{name}_s"] = timed(w.set_mesh, "window")
        shot(f"mesh_window_{name}")
    R["frames_window"] = frames(10, 0.2)
    sc.look(40, 20)
    R["mesh_full_s"] = timed(w.set_mesh, "full")
    shot("mesh_full")
    R["frames_full"] = frames()
    sc.look(0, 89.9)
    shot("mesh_full_north_pole")
    for v in ("north", "flat"):
        R[f"mesh_full_{v}_s"] = timed(w.set_view, v)
        shot(f"mesh_full_{v}_map" if v == "north" else "mesh_full_flat")
        R[f"frames_full_{v}"] = frames(5)
    w.close()
    a.out.write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
