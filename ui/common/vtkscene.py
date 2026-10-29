"""The PyVista scene shared by the trame (browser) and Qt (desktop) prototypes: the same plotter code, two shells.

Views: "globe" (unit sphere in 3D, perspective camera), "flat" (x = lon, y = lat, parallel projection), "north" and
"south" (polar stereographic, the equator at r = 1, same formula as the web prototype). Layers: h preview (cell data
on the lon-lat preview grid), coastline, graticule (the antimeridian in magenta), the zone polygon with draggable
vertex handles, and the mesh (overview raster, worst elements, zoom window with edges, or every element).
"""
import time
from pathlib import Path

import numpy as np
import pyvista as pv

from meshom.config import ZoneLayer
from meshom.geo import lonlat_to_xyz
from ui.common import coast as coastmod
from ui.common import meshview, scenario

MESH_MODES = ("off", "overview", "worst", "window", "full")
VIEWS = ("globe", "flat", "north", "south")
LIFT = {"h": 1.0, "overview": 1.0, "mesh": 1.0004, "graticule": 1.0006, "coast": 1.0008, "zone": 1.0012}
ZOFF = {"h": 0.0, "overview": 0.001, "mesh": 0.002, "graticule": 0.003, "coast": 0.004, "zone": 0.005}


def great_circle(pts: np.ndarray, closed: bool, step_deg: float = 1.0) -> np.ndarray:
    """Lon/lat vertices -> densified lon/lat path along great circles, longitudes continuous."""
    v = lonlat_to_xyz(pts[:, 0], pts[:, 1])
    idx = list(range(len(v))) + ([0] if closed else [])
    out = []
    for i, j in zip(idx[:-1], idx[1:]):
        a, b = v[i], v[j]
        w = np.arccos(np.clip(a @ b, -1, 1))
        n = max(1, int(np.ceil(np.degrees(w) / step_deg)))
        t = np.arange(n)[:, None] / n
        s = np.sin(w) if w > 1e-12 else 1.0
        seg = (np.sin((1 - t) * w) * a + np.sin(t * w) * b) / s if w > 1e-12 else np.repeat(a[None], n, 0)
        out.append(seg)
    out.append(v[idx[-1]][None])
    p = np.vstack(out)
    lon = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(p[:, 2], -1, 1)))
    lon = np.degrees(np.unwrap(np.radians(lon)))
    return np.column_stack([lon, lat])


class Scene:
    def __init__(self, plotter: pv.Plotter, config: Path, mesh_dir: Path):
        self.p = plotter
        self.config_path = Path(config)
        self.cfg = scenario.load(self.config_path)
        self.mesh_dir = Path(mesh_dir)
        self.view = "globe"
        self.mesh_mode = "off"
        self.field = "q"
        self.show_h = True
        self.coast_at_h = False
        self.drawing = False
        self.preview: scenario.Preview | None = None
        self.timings: dict = {}
        self._widgets: list = []
        self.p.set_background("#0e1116")

    # ------------------------------------------------------------------------------------------------- geometry
    def coast(self) -> coastmod.Coast:
        c = self.cfg.coast
        return coastmod.load(str(scenario.resolve(c.source)), float(c.min_area_km2), c.antarctica)

    def zone(self) -> ZoneLayer | None:
        z = [ly for ly in self.cfg.size.layers if ly.kind == "zone"]
        return z[0] if z else None

    def to_view(self, ll: np.ndarray, layer: str) -> np.ndarray:
        """(V,2) lon/lat -> (V,3) points of the current view."""
        lon, lat = ll[:, 0], ll[:, 1]
        if self.view == "globe":
            return lonlat_to_xyz(lon, lat) * LIFT[layer]
        z = np.full(len(ll), ZOFF[layer])
        if self.view == "flat":
            return np.column_stack([lon, lat, z * 100])
        north = self.view == "north"
        colat = np.radians(90 - lat if north else 90 + lat)
        r = np.minimum(np.tan(colat / 2), 4.0)
        y = -r * np.cos(np.radians(lon)) if north else r * np.cos(np.radians(lon))
        return np.column_stack([r * np.sin(np.radians(lon)), y, z])

    def from_view(self, xyz: np.ndarray) -> list:
        x, y, z = xyz
        if self.view == "globe":
            n = np.linalg.norm(xyz)
            return [round(float(np.degrees(np.arctan2(y, x))), 4), round(float(np.degrees(np.arcsin(z / n))), 4)]
        if self.view == "flat":
            return [round(float(x), 4), round(float(y), 4)]
        north = self.view == "north"
        colat = np.degrees(2 * np.arctan(np.hypot(x, y)))
        lon = np.degrees(np.arctan2(x, -y) if north else np.arctan2(x, y))
        return [round(float(lon), 4), round(float(90 - colat if north else colat - 90), 4)]

    def _grid(self, values: np.ndarray, step: float, layer: str) -> pv.StructuredGrid:
        """Cell data on the global lon-lat grid (rows south to north) as a surface of the current view."""
        ny, nx = values.shape
        lon_e = np.linspace(-180, 180, nx + 1)
        lat_e = np.linspace(-90, 90, ny + 1)
        rows = slice(None)
        if self.view in ("north", "south"):                     # one hemisphere and a margin
            keep = lat_e >= -20 if self.view == "north" else lat_e <= 20
            rows = np.flatnonzero(keep)
            lat_e = lat_e[rows]
            values = values[rows[:-1]] if self.view == "north" else values[rows[:-1]]
        LON, LAT = np.meshgrid(lon_e, lat_e)
        pts = self.to_view(np.column_stack([LON.ravel(), LAT.ravel()]), layer)
        g = pv.StructuredGrid(pts[:, 0].reshape(LON.shape).T, pts[:, 1].reshape(LON.shape).T,
                              pts[:, 2].reshape(LON.shape).T)
        g.cell_data["v"] = np.ascontiguousarray(values.T).ravel(order="F")
        return g

    # ----------------------------------------------------------------------------------------------- the layers
    def update_h(self) -> float:
        t0 = time.perf_counter()
        self.preview = scenario.preview(self.cfg, self.coast())
        pr = self.preview
        if self.show_h:
            v = np.where(pr.water, np.log10(pr.h / 1000.0), np.nan)
            lo, hi = np.nanmin(v), np.nanmax(v)
            self.p.add_mesh(self._grid(v, pr.lon[1] - pr.lon[0], "h"), scalars="v", cmap="viridis", clim=(lo, hi),
                            nan_color="#5a5a5a", name="h", show_scalar_bar=False, lighting=False, reset_camera=False)
        else:
            self.p.remove_actor("h")
        self.timings["preview_s"] = time.perf_counter() - t0
        return self.timings["preview_s"]

    def update_coast(self) -> None:
        sp = None
        if self.coast_at_h:
            sp = next(ly.h_m for ly in self.cfg.size.layers if ly.kind == "coast_distance")
        paths = self.coast().polylines(sp)
        pts = self.to_view(np.vstack(paths), "coast")
        n = np.array([len(q) for q in paths])
        off = np.r_[0, np.cumsum(n)[:-1]]
        lines = np.concatenate([np.r_[k, np.arange(o, o + k)] for k, o in zip(n, off)])
        self.p.add_mesh(pv.PolyData(pts, lines=lines), color="black", line_width=1.2, name="coast",
                        lighting=False, reset_camera=False)

    def update_graticule(self) -> None:
        parts, am = [], []
        for lon in range(-180, 180, 30):
            parts.append(great_circle(np.array([[lon, -89.99], [lon, 0], [lon, 89.99]]), False))
            am.append(lon == -180)
        for lat in range(-60, 61, 30):
            parts.append(np.column_stack([np.linspace(-180, 180, 181), np.full(181, lat)]))
            am.append(False)
        for name, sel, color, width in (("graticule", [not a for a in am], "white", 1.0),
                                        ("antimeridian", am, "magenta", 2.5)):
            ps = [q for q, s in zip(parts, sel) if s]
            ps = ps + [q + [360, 0] for q in ps] if (name == "antimeridian" and self.view == "flat") else ps
            pts = self.to_view(np.vstack(ps), "graticule")
            n = np.array([len(q) for q in ps])
            off = np.r_[0, np.cumsum(n)[:-1]]
            lines = np.concatenate([np.r_[k, np.arange(o, o + k)] for k, o in zip(n, off)])
            self.p.add_mesh(pv.PolyData(pts, lines=lines), color=color, line_width=width, opacity=0.5 if
                            name == "graticule" else 1.0, name=name, lighting=False, reset_camera=False)

    def update_zone(self, widgets: bool = True) -> None:
        """Redraw the zone outline; widgets=True also recreates the vertex handles. Never recreate the handles from
        inside a handle's own callback: VTK frees the widget that is still running (a segfault in Qt and trame)."""
        if widgets:
            for w in self._widgets:
                w.Off()
            if hasattr(self.p, "clear_sphere_widgets"):
                self.p.clear_sphere_widgets()
            self._widgets = []
        z = self.zone()
        if z is None or len(z.polygon) < 2:
            self.p.remove_actor("zone")
            return
        poly = np.asarray(z.polygon, dtype=np.float64)
        path = great_circle(poly, closed=not self.drawing)
        self.p.add_mesh(pv.lines_from_points(self.to_view(path, "zone")), color="white", line_width=3,
                        name="zone", lighting=False, reset_camera=False)
        if widgets and not self.drawing:
            centers = self.to_view(poly, "zone")
            r = 0.012 if self.view == "globe" else (1.2 if self.view == "flat" else 0.01)

            def moved(center, index, widget):
                z.polygon[index] = ll = self.from_view(np.asarray(center))
                widget.SetCenter(*self.to_view(np.array([ll]), "zone")[0])     # back onto the surface
                self.on_zone_edit()
            # "always": the callback runs on every move. With "end" the trame remote view never delivered the button
            # release to the widget (the handle moved, stayed selected, the callback never ran).
            ws = self.p.add_sphere_widget(moved, center=centers, radius=r, color="yellow", style="surface",
                                          pass_widget=True, interaction_event="always", test_callback=False)
            self._widgets = ws if isinstance(ws, list) else [ws]

    def on_zone_edit(self) -> None:
        """Hook on every move of a vertex handle; the shells redraw the outline at once and debounce the preview."""
        self.update_zone(widgets=False)
        self.update_h()

    def start_drawing(self) -> None:
        z = self.zone()
        if z is None:
            z = ZoneLayer(name="zone", h_m=10000.0, transition_m=300000.0, polygon=[])
            self.cfg.size.layers.append(z)
        z.polygon = []
        self.drawing = True
        self.update_zone()

        def picked(point):
            if self.drawing and point is not None:
                z.polygon.append(self.from_view(np.asarray(point)))
                self.update_zone(widgets=False)
        self.p.enable_surface_point_picking(callback=picked, left_clicking=True, show_point=False,
                                            show_message=False, picker="cell")

    def finish_drawing(self) -> None:
        self.drawing = False
        self.p.disable_picking()
        self.update_h()
        self.update_zone()

    def update_mesh(self) -> dict:
        t0 = time.perf_counter()
        rec: dict = {"mode": self.mesh_mode}
        for name in ("mesh", "overview", "worst"):
            self.p.remove_actor(name)
        if self.mesh_mode == "off":
            return rec
        mv = meshview.load(str(self.mesh_dir))
        clim = (0.0, 1.0) if self.field == "q" else ((1.0, 2.0) if self.field == "size_ratio" else (0, 1))
        cmap = "RdYlBu_r" if self.field != "kind" else ["#f08c1e", "#788caa"]
        if self.mesh_mode in ("overview", "worst"):
            ov = mv.overview(self.field)
            self.p.add_mesh(self._grid(ov, 0.5, "overview"), scalars="v", cmap=cmap, clim=clim, nan_opacity=0.0,
                            name="overview", show_scalar_bar=False, lighting=False, reset_camera=False)
        elems = None
        if self.mesh_mode == "worst":
            elems = mv.worst(2000)
        elif self.mesh_mode == "window":
            lon, lat = self.focus_lonlat()
            elems = mv.window(lon, lat, 700.0)
        if self.mesh_mode != "overview":
            if self.view == "globe":                           # shared nodes: compact
                tri, quad = mv.tris, mv.quads
                if elems is not None:
                    T = len(tri)
                    tri, quad = tri[elems[elems < T]], quad[elems[elems >= T] - T]
                    eid = np.r_[elems[elems < T], elems[elems >= T]]
                else:
                    eid = np.arange(mv.n_elems)
                faces = np.r_[np.column_stack([np.full(len(tri), 3), tri]).ravel(),
                              np.column_stack([np.full(len(quad), 4), quad]).ravel()]
                pts = mv.xyz.astype(np.float64) * LIFT["mesh"]
            else:                                              # per-element vertices, unwrapped longitudes
                ll, starts, eid = mv.unwrapped(elems)
                k = np.diff(starts)
                pts = self.to_view(ll, "mesh")
                faces = np.insert(np.arange(len(ll)), starts[:-1], k)
            m = pv.PolyData(pts, faces=faces)
            m.cell_data["v"] = mv.fields[self.field][eid]
            self.p.add_mesh(m, scalars="v", cmap=cmap, clim=clim, name="mesh", show_scalar_bar=False,
                            show_edges=self.mesh_mode in ("window", "worst"), edge_color="black", line_width=0.6,
                            lighting=False, reset_camera=False)
            rec["n_elems"] = int(m.n_cells)
        rec["build_s"] = time.perf_counter() - t0
        self.timings["mesh"] = rec
        return rec

    def focus_lonlat(self) -> tuple[float, float]:
        cam = self.p.camera
        if self.view == "globe":
            d = np.asarray(cam.position) / np.linalg.norm(cam.position)
            return float(np.degrees(np.arctan2(d[1], d[0]))), float(np.degrees(np.arcsin(d[2])))
        return tuple(self.from_view(np.asarray(cam.focal_point)))

    # ---------------------------------------------------------------------------------------------- the camera
    def look(self, lon: float = 20.0, lat: float = 15.0, dist: float = 3.6) -> None:
        """Globe: camera above (lon, lat) at `dist` radii; maps: centred, whole map."""
        cam = self.p.camera
        if self.view == "globe":
            self.p.disable_parallel_projection()
            d = lonlat_to_xyz(np.array([lon]), np.array([lat]))[0]
            cam.focal_point = (0.0, 0.0, 0.0)
            cam.position = tuple(d * dist)
            up = np.array([0, 0, 1.0]) if abs(lat) < 80 else -np.sign(lat) * lonlat_to_xyz(np.array([lon]),
                                                                                          np.array([0.0]))[0]
            cam.up = tuple(up)
            cam.view_angle = 30.0
        else:
            self.p.enable_parallel_projection()
            w, h = self.p.ren_win.GetSize()
            aspect = w / max(h, 1)
            half = max(92.0, 182.0 / aspect) if self.view == "flat" else max(1.02, 1.02 / aspect)
            cx = 0.0
            cam.focal_point = (cx, 0.0, 0.0)
            cam.position = (cx, 0.0, 10.0 * half)
            cam.up = (0.0, 1.0, 0.0)
            cam.parallel_scale = half
        self.p.reset_camera_clipping_range()

    def set_view(self, view: str) -> None:
        self.view = view
        self.rebuild()
        self.look()

    def rebuild(self) -> None:
        self.update_h()
        self.update_coast()
        self.update_graticule()
        self.update_zone()
        self.update_mesh()

    def save_config(self, path: Path | None = None) -> Path:
        path = Path(path or self.config_path)
        scenario.save(self.cfg, path)
        return path

    def rotate_frames(self, n: int = 20, step_deg: float = 3.0) -> list:
        """Render n frames while the camera turns (globe) or pans (maps); seconds per frame. The camera is restored."""
        cam = self.p.camera
        saved = (cam.position, cam.focal_point, cam.up)
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            if self.view == "globe":
                self.p.camera.Azimuth(step_deg)
            else:
                fp = np.array(self.p.camera.focal_point)
                dx = step_deg if self.view == "flat" else 0.01
                self.p.camera.focal_point = tuple(fp + [dx, 0, 0])
                self.p.camera.position = tuple(np.array(self.p.camera.position) + [dx, 0, 0])
            self.p.ren_win.Render()              # Plotter.render() is a no-op before the first show / screenshot
            out.append(time.perf_counter() - t0)
        cam.position, cam.focal_point, cam.up = saved
        self.p.reset_camera_clipping_range()
        return out
