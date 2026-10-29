"""Prototype 2: Python only, in the browser — trame + PyVista. VTK renders on the server; the browser gets images.

    python -m ui.trame.app [--config FILE] [--mesh DIR] [--port 8766] [--server]

`--server` does not open a browser (for ssh tunnels and headless runs). The same PyVista scene (ui/common/vtkscene.py)
runs in the Qt prototype; this file is only the trame shell: controls, state and the remote view.
"""
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import html
from trame.widgets import vtk as vtkw
from trame.widgets import vuetify3 as v3

from meshom.config import ZoneLayer
from ui.common import scenario
from ui.common.vtkscene import Scene


def build(server, config: Path, mesh: Path, size=(1100, 800)) -> Scene:
    state, ctrl = server.state, server.controller
    pl = pv.Plotter(off_screen=True, window_size=size)
    sc = Scene(pl, config, mesh)
    L = sc.cfg.size.layers
    get = lambda kind: next(ly for ly in L if ly.kind == kind)  # noqa: E731
    base, coast = get("constant"), get("coast_distance")
    zone = sc.zone() or ZoneLayer(h_m=10000.0, transition_m=300000.0)
    state.update(base_h=base.h_m / 1e3, coast_h=coast.h_m / 1e3, plateau=coast.plateau_m / 1e3,
                 grad=coast.grad, zone_h=zone.h_m / 1e3, zone_w=zone.transition_m / 1e3,
                 min_area=sc.cfg.coast.min_area_km2, view="globe", mesh_mode="off", field="q", show_h=True,
                 coast_at_h=False, drawing=False, h_stats="", mesh_stats="", toml="", status="")

    def push():
        ctrl.view_update()

    def stats():
        p = sc.preview
        state.h_stats = (f"h {p.h[p.water].min() / 1e3:.1f}-{p.h[p.water].max() / 1e3:.1f} km over water\n"
                         f"nodes ≈ ∫h⁻² dA = {p.n_nodes_est / 1e6:.3f} M\nmax |∇h| = {p.grad_max:.3f}\n"
                         f"preview {sc.timings['preview_s']:.2f} s")
        state.toml = scenario.dumps(sc.cfg)

    pending: dict = {"h": None}

    def zone_edited():
        """Every handle move: redraw the outline now, recompute h 0.3 s after the last move."""
        sc.update_zone(widgets=False)
        push()
        if pending["h"] is not None:
            pending["h"].cancel()

        def later():
            sc.update_h()
            with state:
                stats()
            push()
        pending["h"] = asyncio.get_event_loop().call_later(0.3, later)
    sc.on_zone_edit = zone_edited

    @state.change("base_h", "coast_h", "plateau", "grad", "zone_h", "zone_w", "show_h")
    def on_h(base_h, coast_h, plateau, grad, zone_h, zone_w, show_h, **_):
        base.h_m, coast.h_m = float(base_h) * 1e3, float(coast_h) * 1e3
        coast.plateau_m, coast.grad = float(plateau) * 1e3, float(grad)
        if sc.zone():
            sc.zone().h_m, sc.zone().transition_m = float(zone_h) * 1e3, float(zone_w) * 1e3
        sc.show_h = bool(show_h)
        sc.update_h()
        stats()
        push()

    @state.change("min_area", "coast_at_h")
    def on_coast(min_area, coast_at_h, **_):
        sc.cfg.coast.min_area_km2 = float(min_area)
        sc.coast_at_h = bool(coast_at_h)
        sc.update_coast()
        sc.update_h()
        stats()
        push()

    @state.change("view")
    def on_view(view, **_):
        sc.set_view(view)
        push()

    @state.change("mesh_mode", "field")
    def on_mesh(mesh_mode, field, **_):
        sc.mesh_mode, sc.field = mesh_mode, field
        r = sc.update_mesh()
        state.mesh_stats = json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
        push()

    @ctrl.set("draw")
    def draw():
        sc.start_drawing()
        state.drawing = True
        state.status = "drawing: click on the globe to add vertices"
        push()

    @ctrl.set("finish")
    def finish():
        sc.finish_drawing()
        state.drawing = False
        state.status = ""
        stats()
        push()

    @ctrl.set("save")
    def save():
        state.status = f"saved {sc.save_config()}"
        state.toml = scenario.dumps(sc.cfg)

    @ctrl.set("run")
    def run():
        state.status = ("dry run (the core has no `meshom run CONFIG` yet): scripts/on_node.sh -c 8 -t 01:00:00 -- "
                        f"python -m ui.common.prepare_mesh --method {sc.cfg.mesh.method}")

    @ctrl.trigger("bench")
    def bench(view="globe", mode="full", lon=40.0, lat=20.0, dist=3.6, n=10):
        """Server-side timing for the measurements: switch view and mesh mode, render n turned frames."""
        t0 = time.perf_counter()
        if view != sc.view:
            sc.set_view(view)
        sc.look(lon, lat, dist)
        sc.mesh_mode = mode
        r = sc.update_mesh()
        t1 = time.perf_counter()
        pl.ren_win.Render()
        first = time.perf_counter() - t1
        fr = sc.rotate_frames(int(n), 2.0)
        push()
        return {"build_s": time.perf_counter() - t0 - first - sum(fr), "first_render_s": first,
                "frame_ms_median": 1e3 * float(np.median(fr)), "n_elems": r.get("n_elems", 0)}

    @ctrl.trigger("project")
    def project(i=0):
        """Display position (pixels from the top left) and window size, for a zone vertex: used by the tests."""
        lon, lat = sc.zone().polygon[int(i)]
        w = sc.to_view(np.array([[lon, lat]]), "zone")[0]
        pl.renderer.SetWorldPoint(*w, 1.0)
        pl.renderer.WorldToDisplay()
        x, y, _ = pl.renderer.GetDisplayPoint()
        W, H = pl.ren_win.GetSize()
        return {"x": x, "y": H - y, "size": [W, H], "polygon": sc.zone().polygon}

    @ctrl.trigger("look")
    def look(lon=20.0, lat=15.0, dist=3.6):
        sc.look(lon, lat, dist)
        push()
        return {"camera": [list(pl.camera.position), list(pl.camera.focal_point)], "focus": sc.focus_lonlat()}

    with SinglePageWithDrawerLayout(server, full_height=True) as layout:
        layout.title.set_text("meshom · UI prototype · trame")
        with layout.drawer as drawer:
            drawer.width = 320
            with v3.VContainer(classes="pa-2", style="font-size: 13px"):
                with v3.VBtnToggle(v_model=("view",), mandatory=True, density="compact", divided=True,
                                   variant="outlined"):
                    for v in ("globe", "flat", "north", "south"):
                        v3.VBtn(v, value=v, size="small")
                html.Div("1 · h builder (min of layers)", classes="text-subtitle-2 mt-3")
                for key, label, step in (("base_h", "base h, km", 5), ("coast_h", "h at coast, km", 1),
                                         ("plateau", "plateau, km", 5), ("grad", "coast |∇h|", 0.01),
                                         ("zone_h", "zone h, km", 1), ("zone_w", "zone transition, km", 25)):
                    v3.VTextField(v_model=(key,), label=label, type="number", step=step, density="compact",
                                  hide_details=True, classes="my-1")
                with html.Div(classes="d-flex ga-1 my-1"):
                    v3.VBtn("Draw polygon", click=ctrl.draw, size="small", disabled=("drawing",))
                    v3.VBtn("Finish", click=ctrl.finish, size="small", disabled=("!drawing",))
                v3.VCheckbox(v_model=("show_h",), label="show h", density="compact", hide_details=True)
                html.Pre("{{ h_stats }}", style="font-size: 11px")
                html.Div("2 · Coastline", classes="text-subtitle-2 mt-2")
                v3.VTextField(v_model=("min_area",), label="min island area, km²", type="number", step=50,
                              density="compact", hide_details=True)
                v3.VCheckbox(v_model=("coast_at_h",), label="resample at h at coast", density="compact",
                             hide_details=True)
                html.Div("6 · Mesh view", classes="text-subtitle-2 mt-2")
                v3.VSelect(v_model=("field",), items=("['q', 'size_ratio', 'kind']",), label="overlay",
                           density="compact", hide_details=True)
                with v3.VBtnToggle(v_model=("mesh_mode",), mandatory=True, density="compact", divided=True,
                                   variant="outlined", classes="mt-1"):
                    for m in ("off", "overview", "worst", "window", "full"):
                        v3.VBtn(m, value=m, size="x-small")
                html.Pre("{{ mesh_stats }}", style="font-size: 11px; white-space: pre-wrap")
                with html.Div(classes="d-flex ga-1 mt-2"):
                    v3.VBtn("Save TOML", click=ctrl.save, size="small")
                    v3.VBtn("Run (batch)", click=ctrl.run, size="small")
                html.Div("{{ status }}", style="font-size: 11px")
                html.Pre("{{ toml }}", style="font-size: 10px; max-height: 160px; overflow: auto")
        with layout.content:
            with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                view = vtkw.VtkRemoteView(pl.ren_win, ref="view", interactive_ratio=1, interactive_quality=70,
                                          still_quality=95)
                ctrl.view_update = view.update
    sc.rebuild()
    sc.look()
    stats()
    return sc


def main() -> None:
    server = get_server(client_type="vue3")
    server.cli.add_argument("--config", type=Path, default=scenario.EXAMPLE)
    server.cli.add_argument("--mesh", type=Path, default=Path(".tmp/ui/mesh"))
    args, _ = server.cli.parse_known_args()
    build(server, args.config, args.mesh)
    server.start()


if __name__ == "__main__":
    main()
