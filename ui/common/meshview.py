"""A finished mesh for the UI: node lon/lat, elements, the quality overlay, a decimated overview, zoom windows,
irregular nodes, histograms and the worst elements.

A mesh is read from a file (`.npz` with points, tris, quads[, node_flag], as `meshom compete` and prepare_mesh write
it) or a run folder (the first of mesh.npz, smooth.npz, raw.npz). Its overlay (prepare_mesh.overlay_arrays) is read
from overlay.npz next to a mesh.npz, else computed once and cached under .tmp/ui/cache/ (keyed by the mesh path; a
newer mesh file is recomputed). The per-edge ln(s/h) needs h: the caller passes it when it knows it (the run's config
or the compete case), else the |ln s/h| histogram is left out.

The overview is the worst quality value per pixel of a lon-lat raster whose pixels widen toward the poles (k = ceil(1 /
cos lat) columns share one bin), so no row is left with empty pixels near a pole. Quality colour: the max interior
angle scaled to its gate, q = (theta - 90) / 60 for quads and (theta - 60) / 60 for triangles: 0 ideal, 1 at the hard
gate (150° quads, 120° triangles).

Depth (depth_of): the last write of the mesh file as model files (<run>/fesom-<mesh stem>/, `meshom write` or the
page's Write: depth_base.npy, the depth before the node edits; for a write without it, depth.out for fesom-c, whose
aux3d.out is a placeholder, else aux3d.out), else the run folder's depth.npy (the core's bathymetry stage: the run's
final mesh), each when its length matches the mesh's nodes; then the mesh's node edits (<mesh stem>.depth_edits.json)
as the next write applies them (meshom.bathymetry.apply_node_edits at the run's floor). They give the node depths and
the element field "depth", the mean depth of the element's nodes. Its colour scale runs
from 0 to depth_hi (the 99th percentile rounded up to 500 m), YlGnBu (shallow light, deep dark).

Cell size: the element field "size_km", the mean great-circle length of the element's edges in km (what h asks for:
an edge length). Its colour scale is logarithmic between the 1st and 99th percentiles rounded out to two significant
digits (size_range; ORCA1: 27-110 km), viridis (small dark, large bright): a structured grid such as ORCA has near-ideal angles and
neighbour ratios everywhere, so only this field shows how its resolution changes with latitude.
"""
import hashlib
import threading
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from meshom.geo import lonlat_to_xyz, xyz_to_lonlat

QUALITY_FIELDS = ("q", "size_ratio", "kind")
SIZE_RATIO = "size ratio, worst edge of each element"      # the histogram's name (ui/web/runs.py RECORD_STATS)
MESH_NAMES = ("mesh.npz", "optimized.npz", "smooth.npz", "raw.npz")    # a folder opens its final mesh
DEPTH_NAME = "depth.npy"
CACHE = Path(__file__).resolve().parents[2] / ".tmp" / "ui" / "cache"


@dataclass
class MeshView:
    lonlat: np.ndarray       # (N,2) float32
    xyz: np.ndarray          # (N,3) float32 unit vectors
    tris: np.ndarray         # (T,3) int64
    quads: np.ndarray        # (Q,4) int64
    fields: dict             # per element (T+Q,): q, size_ratio, kind, angle, size_km[, depth]
    centroid: np.ndarray     # (T+Q,2) float32 lon/lat
    node_class: np.ndarray | None = None   # (N,) int8: 0 regular / boundary, 1 n3, 2 n5, 3 other
    regular: int = 4
    edge_ln: np.ndarray | None = None      # (E,) float32 ln(s/h), None without h
    path: str = ""
    node_flag: np.ndarray | None = None    # (N,) int8 of the file (coast / open boundary nodes: != 0)
    node_depth: np.ndarray | None = None   # (N,) m positive down (depth_of), None without one
    depth_source: str | None = None        # where node_depth comes from, in words

    @property
    def depth_hi(self) -> float | None:
        """The deep end of the depth colour scale (m): the 99th percentile of the element depths rounded up to 500 m."""
        if "depth" not in self.fields:
            return None
        return float(max(500.0, np.ceil(np.percentile(self.fields["depth"], 99) / 500.0) * 500.0))

    @cached_property
    def size_range(self) -> tuple[float, float]:
        """(lo, hi) km of the log colour scale of "size_km": the 1st percentile rounded down and the 99th rounded up to
        two significant digits."""
        p1, p99 = np.percentile(self.fields["size_km"], [1, 99])
        return round2(p1, np.floor), round2(max(p99, p1 * 1.01), np.ceil)

    @property
    def n_elems(self) -> int:
        return len(self.tris) + len(self.quads)

    def triangles(self, elems: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Elements split into triangles (quads along the 0-2 diagonal): (K,3) node ids and (K,) element ids."""
        T = len(self.tris)
        e = np.arange(self.n_elems) if elems is None else np.asarray(elems)
        et, eq = e[e < T], e[e >= T] - T
        q = self.quads[eq]
        tri = np.vstack([self.tris[et], q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        eid = np.r_[et, eq + T, eq + T]
        return tri, eid

    def polygons(self, elems: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flat polygon vertex list for binary renderers: (V,) node ids, (n+1,) start indices, (n,) element ids."""
        T = len(self.tris)
        e = np.arange(self.n_elems) if elems is None else np.asarray(elems)
        et, eq = e[e < T], e[e >= T] - T
        nodes = np.r_[self.tris[et].ravel(), self.quads[eq].ravel()]
        starts = np.r_[np.arange(len(et)) * 3, 3 * len(et) + np.arange(len(eq) + 1) * 4]
        return nodes, starts, np.r_[et, eq + T]

    def unwrapped(self, elems: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-element vertex lon/lat (V,2) float64 for flat maps and lon-lat triangulators, start indices, element ids.

        Longitudes are unwrapped per element (within ±180° of its first vertex, so an element on the antimeridian
        stays small) and a pole vertex takes the mean longitude of the element's other vertices (so the polygon is
        simple in lon-lat at the poles). The page does the same in JavaScript on the indexed payload.
        """
        nodes, starts, eid = self.polygons(elems)
        ll = self.lonlat[nodes].astype(np.float64)
        k = np.diff(starts)
        pid = np.repeat(np.arange(len(k)), k)
        pole = np.abs(ll[:, 1]) > 89.9999
        lon = np.where(pole, np.nan, ll[:, 0])
        ref = lon[starts[:-1]]
        ref = np.where(np.isnan(ref), lon[starts[:-1] + 1], ref)[pid]          # first vertex at a pole
        lon = np.where(lon - ref > 180, lon - 360, np.where(lon - ref < -180, lon + 360, lon))
        if pole.any():
            s = np.bincount(pid, weights=np.nan_to_num(lon), minlength=len(k))
            c = np.bincount(pid, weights=~np.isnan(lon), minlength=len(k))
            lon = np.where(np.isnan(lon), (s / np.maximum(c, 1))[pid], lon)
        ll[:, 0] = lon
        return ll, starts, eid

    def window(self, lon0: float, lat0: float, radius_km: float) -> np.ndarray:
        """Element ids whose centroid lies within radius_km of (lon0, lat0) (great-circle)."""
        lo, la = np.radians(self.centroid[:, 0]), np.radians(self.centroid[:, 1])
        lo0, la0 = np.radians(lon0), np.radians(lat0)
        cosd = np.sin(la) * np.sin(la0) + np.cos(la) * np.cos(la0) * np.cos(lo - lo0)
        return np.flatnonzero(cosd >= np.cos(radius_km / 6371.0))

    def worst(self, n: int = 2000, field: str = "q") -> np.ndarray:
        return np.argsort(self.fields[field])[-n:]

    @cached_property
    def _cover(self):
        """KD-tree of the element centroids (unit vectors) and each element's reach: the chord from its centroid to its
        farthest node."""
        from scipy.spatial import cKDTree
        c = lonlat_to_xyz(self.centroid[:, 0].astype(np.float64), self.centroid[:, 1].astype(np.float64))
        T = len(self.tris)
        reach = np.empty(self.n_elems)
        reach[:T] = np.linalg.norm(self.xyz[self.tris] - c[:T, None], axis=2).max(axis=1)
        reach[T:] = np.linalg.norm(self.xyz[self.quads] - c[T:, None], axis=2).max(axis=1)
        return cKDTree(c), reach

    def overview(self, field: str = "q", step_deg: float = 0.5) -> np.ndarray:
        """(ny, nx) float32 worst (max) value per pixel, NaN where no element; rows south to north. A pixel that no
        centroid falls into takes, of the 8 elements whose centroids are nearest to the pixel centre, the nearest one
        whose reach covers the centre: elements larger than a pixel cover all their pixels, a hole of the mesh stays
        empty (up to one element's reach from its rim)."""
        nx, ny = int(round(360 / step_deg)), int(round(180 / step_deg))
        lon, lat = self.centroid[:, 0].astype(np.float64), self.centroid[:, 1].astype(np.float64)
        r = np.clip(((lat + 90.0) / step_deg).astype(np.int64), 0, ny - 1)
        lat_r = -90.0 + (np.arange(ny) + 0.5) * step_deg
        k = np.minimum(np.ceil(1.0 / np.maximum(np.cos(np.radians(lat_r)), 1e-6)).astype(np.int64), nx)
        c = np.clip(((lon + 180.0) / step_deg).astype(np.int64), 0, nx - 1)
        c0 = (c // k[r]) * k[r]
        out = np.full(nx * ny, -np.inf)
        np.maximum.at(out, r * nx + c0, self.fields[field].astype(np.float64))
        out = out.reshape(ny, nx)
        cols = np.arange(nx)
        img = out[np.arange(ny)[:, None], (cols[None, :] // k[:, None]) * k[:, None]]
        j, i = np.nonzero(~np.isfinite(img))
        if len(j):
            tree, reach = self._cover
            nk = min(8, self.n_elems)
            d, e = tree.query(lonlat_to_xyz(-180.0 + (i + 0.5) * step_deg, -90.0 + (j + 0.5) * step_deg), k=nk)
            d, e = d.reshape(len(j), nk), e.reshape(len(j), nk)
            cover = d <= reach[e]                         # the element holding the pixel always covers it
            ok, first = cover.any(axis=1), cover.argmax(axis=1)
            img[j[ok], i[ok]] = self.fields[field][e[ok, first[ok]]]
        return np.where(np.isfinite(img), img, np.nan).astype(np.float32)

    # ------------------------------------------------------------------------------------------ v2 views

    def irregular(self) -> tuple[np.ndarray, np.ndarray]:
        """Irregular interior nodes: (K,2) lon/lat float32 and (K,) class 1 (n3), 2 (n5), 3 (other)."""
        k = np.flatnonzero(self.node_class > 0)
        return self.lonlat[k], self.node_class[k]

    def cell_km(self) -> float:
        """The median edge length in km (great circle) over up to 20 000 elements spread through the mesh."""
        el = self.quads if len(self.quads) >= len(self.tris) else self.tris
        e = el[:: max(1, len(el) // 20000)]
        a, b = self.xyz[e].astype(np.float64), self.xyz[np.roll(e, -1, axis=1)].astype(np.float64)
        return float(np.median(2.0 * np.arcsin(np.clip(np.linalg.norm(a - b, axis=2) / 2.0, 0.0, 1.0)))) * 6371.0

    def counts(self) -> dict:
        c = np.bincount(self.node_class, minlength=4) if self.node_class is not None else np.zeros(4, int)
        r = self.regular
        return {"regular": r, f"n{r - 1}": int(c[1]), f"n{r + 1}": int(c[2]), "other": int(c[3])}

    def histograms(self) -> list[dict]:
        """Distributions of the gate metrics: quad and triangle max angle, element size ratio, |ln s/h| per edge.
        Counts per bin (the last bin also holds everything above), the gate lines of meshom.config and percentiles."""
        from meshom.config import DEFAULT_GATES as G
        kind, ang = self.fields["kind"], self.fields["angle"]

        def h(name, x, lo, hi, step, gates, unit):
            x = np.asarray(x, np.float64)
            edges = np.arange(lo, hi + step / 2, step)
            cnt = np.histogram(np.clip(x, lo, hi - 1e-9), edges)[0] if x.size else np.zeros(len(edges) - 1, int)
            pct = {k: (float(np.percentile(x, v)) if x.size else None)
                   for k, v in (("p99", 99), ("p99_9", 99.9), ("max", 100))}
            return {"name": name, "unit": unit, "lo": lo, "hi": hi, "step": step, "counts": cnt.tolist(),
                    "n": int(x.size), "gates": gates, **pct}
        out = [h("quad θmax", ang[kind == 1], 90.0, 180.0, 2.0,
                 [["p99.9 target", G["quad_max_angle_target_deg"]], ["hard", G["quad_max_angle_hard_deg"]]], "°"),
               h("triangle θmax", ang[kind == 0], 60.0, 180.0, 2.0,      # the p99 target holds for meshes without quads
                 [["p99 target", G["tri_max_angle_target_deg"]]] * (not (kind == 1).any()) +
                 [["hard", G["tri_max_angle_hard_deg"]]], "°"),
               h(SIZE_RATIO, self.fields["size_ratio"], 1.0, 2.5, 0.025,
                 [["p99", G["elem_size_ratio_p99"]], ["max", G["elem_size_ratio_max"]]], "")]
        out[-1]["note"] = "bins: the largest ratio over each element's edges; the gate lines and quality.json: per edge"
        if self.edge_ln is not None:
            out.append(h("|ln s/h| per edge", np.abs(self.edge_ln), 0.0, 1.0, 0.02,
                         [["p99", G["size_fidelity_p99"]]], ""))
        return out

    def worst_list(self, n: int = 50, field: str = "q") -> list[dict]:
        e = self.worst(n, field)[::-1]
        T = len(self.tris)
        return [{"id": int(i), "kind": "quad" if i >= T else "tri", "lon": round(float(self.centroid[i, 0]), 4),
                 "lat": round(float(self.centroid[i, 1]), 4), "angle": round(float(self.fields["angle"][i]), 2),
                 "size_ratio": round(float(self.fields["size_ratio"][i]), 4), "q": round(float(self.fields["q"][i]), 4)}
                for i in e]


def round2(x: float, how=np.floor) -> float:
    """x > 0 rounded to two significant digits, down (np.floor) or up (np.ceil): 27.0 -> 27, 105.7 -> 110 (up)."""
    k = 10.0 ** (np.floor(np.log10(x)) - 1)
    return float(how(x / k * (1 - 1e-12) if how is np.ceil else x / k * (1 + 1e-12)) * k)


def edge_km(xyz: np.ndarray, el: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
    """(len(el),) float32: the mean great-circle length of each element's edges, km (in chunks: a 4 M quad mesh
    would need 0.4 GB at once)."""
    out = np.empty(len(el), np.float32)
    for i in range(0, len(el), chunk):
        e = el[i:i + chunk]
        a, b = xyz[e].astype(np.float64), xyz[np.roll(e, -1, axis=1)].astype(np.float64)
        out[i:i + chunk] = 2.0 * np.arcsin(np.clip(np.linalg.norm(a - b, axis=2) / 2.0, 0.0, 1.0)).mean(axis=1) * 6371.0
    return out


def quality_rgba(v: np.ndarray, field: str = "q", depth_hi: float | None = None,
                 size_range: tuple[float, float] | None = None) -> np.ndarray:
    """(..., 4) uint8 colours: q in [0, 1] blue -> red; size_ratio in [1, 2]; kind: tri orange, quad grey-blue;
    depth in [0, depth_hi] m, YlGnBu (the page's GRADIENTS.ylgnbu); size_km log between size_range, viridis."""
    from matplotlib import colormaps
    if field == "kind":
        out = np.empty(v.shape + (4,), np.uint8)
        out[...] = (120, 140, 170, 255)
        out[v == 0] = (240, 140, 30, 255)
        out[np.isnan(v)] = (0, 0, 0, 0)
        return out
    if field == "size_km":
        lo, hi = size_range
        t = np.clip(np.log(np.nan_to_num(v, nan=lo).clip(min=1e-9) / lo) / np.log(hi / lo), 0, 1)
    else:
        lo, hi = {"q": (0.0, 1.0), "depth": (0.0, depth_hi)}.get(field, (1.0, 2.0))
        t = np.clip((np.nan_to_num(v, nan=lo) - lo) / (hi - lo), 0, 1)
    out = (colormaps[{"depth": "YlGnBu", "size_km": "viridis"}.get(field, "RdYlBu_r")](t) * 255).astype(np.uint8)
    out[np.isnan(v)] = (0, 0, 0, 0)
    return out


def edits_file(mf: Path) -> Path:
    """The node edits of mesh file mf (the core's C2, decisions O37): <mesh stem>.depth_edits.json next to it."""
    return mf.parent / f"{mf.stem}.depth_edits.json"


def depth_files(mf: Path) -> tuple[Path, ...]:
    """The files whose change changes depth_of: the last write's record and base, the run's depth.npy, the edits."""
    out = mf.parent / f"fesom-{mf.stem}"
    return out / "quality.json", out / "depth_base.npy", mf.parent / DEPTH_NAME, edits_file(mf)


def depth_of(mf: Path, n: int) -> tuple[np.ndarray | None, str | None]:
    """(node depths m positive down, the source in words) of mesh file mf with n nodes: its last write's base (else
    its written files), else the run's depth.npy, with the mesh's node edits applied; (None, None) without one of
    length n."""
    import json

    from meshom.io.fesom import read_aux3d
    rec_f, base_f, df, ef = depth_files(mf)
    try:
        rec = json.loads(rec_f.read_text()).get(mf.stem) or {}
    except (OSError, ValueError):
        rec = {}
    d, src, fmt = None, None, None
    if rec:
        fmt, when = rec.get("format"), (rec.get("date") or "").replace("T", " ")
        f = rec_f.parent / ("depth.out" if fmt == "fesom-c" else "aux3d.out")
        if base_f.exists():
            d, src = np.load(base_f), f"{rec_f.parent.name}/depth_base.npy: the write of {when}, {fmt}, before the node edits"
        elif f.exists():
            d = np.loadtxt(f, ndmin=1) if fmt == "fesom-c" else read_aux3d(f)[1]
            src = f"{rec_f.parent.name}/{f.name}: the write of {when}, {fmt}"
        if d is not None and len(d) != n:
            d = None
    if d is None and df.exists():
        d, fmt = np.load(df), None
        src = "the run's depth.npy (its final mesh)"
        if len(d) != n:                     # depth.npy is the run's final mesh's; a repaired mesh has other nodes
            d = None
    if d is None:
        return None, None
    if ef.exists() and (mf.parent / "config.resolved.toml").is_file():
        from meshom import config
        from meshom.bathymetry import apply_node_edits
        from meshom.run import depth_floor_m
        cfg = config.load(mf.parent / "config.resolved.toml")
        try:
            d, r = apply_node_edits(d, ef, depth_floor_m(cfg, fmt or cfg.mesh.format))
            k = r["n_edits"]
            src += f" + {k} node edit{'' if k == 1 else 's'} (as the next write applies them)"
        except ValueError:
            src += f" ({ef.name} belongs to another mesh: not applied)"
    return d, src


def mesh_file(path) -> Path:
    p = Path(path)
    if p.is_dir():
        for n in MESH_NAMES:
            if (p / n).exists():
                return p / n
        raise FileNotFoundError(f"no {' / '.join(MESH_NAMES)} in {p}")
    return p


def overlay_file(mf: Path) -> Path:
    if mf.name == "mesh.npz" and (mf.parent / "overlay.npz").exists():
        return mf.parent / "overlay.npz"
    key = hashlib.md5(str(mf.resolve()).encode()).hexdigest()[:12]
    return CACHE / f"{key}_{mf.stem}_overlay.npz"


_LOADED: dict = {}
_LOAD_LOCK = threading.Lock()      # one load at a time: the page asks for a new mesh's info, overview and elements at
#                                    once; three threads each reading a 3.9 M node mesh and computing its overlay took the
#                                    server past its memory. The others wait and get the loaded mesh.


def load(path, h_provider=None) -> MeshView:
    """The mesh of a file or run folder with its overlay (computed and cached when missing). h_provider() returns the
    run's SizeField or None; it is called only when ln s/h has to be computed. Thread-safe (_LOAD_LOCK)."""
    with _LOAD_LOCK:
        return _load(path, h_provider)


def _load(path, h_provider=None) -> MeshView:
    mf = mesh_file(path)
    key = (str(mf.resolve()), mf.stat().st_mtime) + tuple(f.stat().st_mtime if f.exists() else None for f in depth_files(mf))
    if key in _LOADED:
        return _LOADED[key]
    with np.load(mf) as z:
        pts, tris, quads = z["points"], z["tris"], z["quads"]
        flag = z["node_flag"] if "node_flag" in z.files else np.zeros(len(pts), np.int8)
    of = overlay_file(mf)
    stale = not of.exists() or of.stat().st_mtime < mf.stat().st_mtime
    hf = None
    if not stale:
        with np.load(of) as z:
            stale = "node_class" not in z.files
            if not stale and "edge_ln_sh" not in z.files and h_provider is not None:
                hf = h_provider()
                stale = hf is not None
    if stale:
        from meshom.mesh import Mesh
        from ui.common.prepare_mesh import write_overlay
        if hf is None and h_provider is not None:
            hf = h_provider()
        of.parent.mkdir(parents=True, exist_ok=True)
        write_overlay(Mesh(pts, tris, quads, flag), of.parent, hf, of.name)
    with np.load(of) as z:
        ang, sr, kind, clon, clat = z["max_angle_deg"], z["size_ratio"], z["kind"], z["lon"], z["lat"]
        cls, reg = z["node_class"], int(z["regular"])
        eln = z["edge_ln_sh"] if "edge_ln_sh" in z.files else None
    lon, lat = xyz_to_lonlat(pts)
    q = np.where(kind == 1, (ang - 90.0) / 60.0, (ang - 60.0) / 60.0).astype(np.float32)
    fields = {"q": q, "size_ratio": sr, "kind": kind, "angle": ang,
              "size_km": np.r_[edge_km(pts, tris), edge_km(pts, quads)].astype(np.float32)}
    d, src = depth_of(mf, len(pts))
    if d is not None:
        fields["depth"] = np.r_[d[tris].mean(axis=1), d[quads].mean(axis=1)].astype(np.float32)
    mv = MeshView(np.column_stack([lon, lat]).astype(np.float32), pts.astype(np.float32), tris, quads, fields,
                  np.column_stack([clon, clat]).astype(np.float32), cls, reg, eln, str(mf), flag, d, src)
    _LOADED.clear()                                   # keep one mesh in memory (10^6 elements: ~0.2 GB)
    _LOADED[key] = mv
    return mv

