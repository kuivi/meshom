"""Straits: channels kept open along a centreline (`[coast] straits`, `[[straits]]`; docs/config.md "Straits").

The strait rules (the band, the cells across, the carve, the protected elements of the land cut, the strait check)
and the built-in strait list are after D. Sein (AWI), fesom2_mesh_gen.

A strait is a centreline `path` [[lon, lat], ...] (from one basin into the other) with
- `width_m`: the channel width to keep open. The **corridor** is W = max(width_m, cells h_m s + h_m / sqrt(3)) wide,
  centred on the path: fesom2_mesh_gen's band of n nodes at size h, b(n, h) = (n - 1) h s with s = 1.15 (15 % margin
  for irregular node placement), here with n - 1 = cells, plus
  CENTROID_MARGIN h: an element is in the corridor when its centroid is, and a centroid sits a third of the row height
  (sqrt(3) h / 6) inside its row of triangles, so the wet band can be up to 2 sqrt(3) h / 6 = h / sqrt(3) narrower than
  W (measured on the CORE2-like run without the margin: 1.6-2.0 cells at 2 x 1.15 x 10 km = 23 km);
  on a mesh with quads (a method not in TRI_METHODS) W = max(width_m, (cells s + sqrt 2) e^t h_m): edges up to
  e^t h_m (the run's size tolerance t) and a quad's centroid up to half its diagonal from its far corner
  (Strait.corridor_m; m14 at t = 0.20 kept Franklin_Strait at 1.56 h with the triangle margin);
- `h_m`: the target size along the path, a size zone: h_m within W/2 of the path, then linear in the distance up to
  the base h over (base - h_m) / grad_max (meshom.run.size_field applies it after [size] h_min_m / h_max_m and before
  the gradient limiter, so the global clip does not undo it);
- `cells`: cells across wanted (default 2: fesom2_mesh_gen's coarse test mesh opens Gibraltar with the path elements
  plus one neighbour ring = about 2-3 cells; its production rule at 0.3-2.5 km is 5-7 nodes = 4-6 cells);
- `depth_m`: a depth floor (carving, the bathymetry stage, `depth_floor`): full within h_m of the path, a cosine
  taper over `[bathymetry] carve_taper` x h_m beyond, as fesom2_mesh_gen carves the Bosphorus and the Dardanelles.

The land cut (meshom.run.cut_land) makes every element whose centroid lies within W/2 of the path water and protects
those elements from the clean-up's spur and pinch rules, as fesom2_mesh_gen protects forced elements;
one unprotected pass follows (FESOM2 rejects elements with one neighbour).

`strait_report` measures what a mesh made of each strait, after fesom2_mesh_gen's strait check
(transects, nodes across): transects perpendicular to the path at N_SECTIONS points of its
inner 80 % (the outer 10 % at each end are approaches), the wet run through the path point, cells across =
run width / mean edge length of the elements on the run; the narrowest section; the water component of the elements
at both ends of the path, connected = both in the main water (`water_components`: 0 = the largest; basin to basin, so a
basin split between two straits shows, decisions C44); the smallest element depth (mean of its node depths, FESOM2's
which_depth_n2e = 'mean') on the inner path.

Profiles (`[coast] strait_profile`, `[[strait_regions]]`, a `[[straits]]` entry's `profile`; decisions C45-C47): a
strait is either **resolve**d (the above: the size zone h_m along the path, corridor at h_m) or **widen**ed (CORE2 and
NG5 style: no size zone, h_m = the background h, the size field
without any strait zone, at its largest over the inner 80 % of the path, and the corridor
max(width_m, (cells 1.15 + 1/sqrt 3) h_m) with cells = `[coast] strait_widen_cells` forced to water: at least that
many cells of the background h across, the channel widened into the land on both sides of the path). With
`[[strait_regions]]` the first region holding a strait's middle point (half its arc length) gives its profile ("off":
not opened); a strait in no region is not opened (left as the coast gives it); without regions every strait takes
`[coast] strait_profile`. A `[[straits]]` entry's own `profile` wins over both and opens it wherever it lies.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M, lonlat_to_xyz, xyz_to_lonlat
from meshom.mesh import Mesh, build_topology
from meshom.size import polygon_inside

BAND_SAFETY = 1.15                       # fesom2_mesh_gen's band margin s (b = (n - 1) h s)
CENTROID_MARGIN = 1.0 / np.sqrt(3.0)     # x h: the centroid rule's loss of width (module docstring)
QUAD_MARGIN = np.sqrt(2.0)               # x quad side: the loss on quads (a half diagonal per side, C45)
TRI_METHODS = frozenset({"m00_background", "m01_triangles", "m01_triangles_fit", "m02_linf_frontal_tri",
                         "m02_linf_frontal_tri_fit", "m11_local_ops_tri", "m15_tri", "ref_jigsaw",
                         "ref_jigsaw_fit"})  # methods whose meshes hold triangles only; every other one makes quads
BUILTIN = {"fesom2_mesh_gen": Path(__file__).parent / "data" / "straits_fesom2_mesh_gen.json"}
MANDATORY = ("Gibraltar",)               # opened on every mesh with a real coast, at least widened (decisions
                                         # C49); only an explicit [[straits]] enabled = false closes one
INNER = (0.1, 0.9)                       # the inner 80 % of a path, as fesom2_mesh_gen's strait check
N_SECTIONS = 40
END_REACH_M = 20e3                       # a path end on land takes the nearest element centroid within this
PROFILES = ("resolve", "widen")          # decisions C45
REGION_PROFILES = PROFILES + ("off",)    # decisions C46
BG_STEP_M = 1000.0                       # widen: the background h sampled along the path at this spacing at most


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _chord(d_m):
    return 2.0 * np.sin(0.5 * np.minimum(np.asarray(d_m, dtype=np.float64) / R_EARTH_M, np.pi))


def _arc_m(chord):
    return 2.0 * R_EARTH_M * np.arcsin(np.clip(0.5 * np.asarray(chord), 0.0, 1.0))


def densify(path: np.ndarray, step_m: float) -> np.ndarray:
    """(K,3) polyline -> points along it at spacing <= step_m (great-circle segments, ends kept)."""
    a, b = path[:-1], path[1:]
    k = np.maximum(1, np.ceil(_arc_m(np.linalg.norm(b - a, axis=1)) / step_m).astype(np.int64))
    out = [a[i] + (b[i] - a[i]) * (np.arange(k[i])[:, None] / k[i]) for i in range(len(a))]
    return _unit(np.vstack(out + [path[-1:]]))


@dataclass(frozen=True, eq=False)
class Strait:
    name: str
    path: np.ndarray                     # (K,3) unit vectors
    width_m: float
    h_m: float
    cells: int = 2
    depth_m: float | None = None
    profile: str = "resolve"             # "resolve": size zone h_m; "widen": h_m = background h, no zone (C45)
    quads: bool = False                  # the mesh holds quads (a method not in TRI_METHODS)
    size_tol: float = 0.0                # the method's size tolerance t (methods.effective_tolerance; 0 = none)

    @property
    def corridor_m(self) -> float:
        """Triangles: max(width_m, (cells BAND_SAFETY + CENTROID_MARGIN) h_m). Quads: the edges may reach e^t h_m
        (the size tolerance) and a quad's centroid lies up to half its diagonal from its far corner across the
        path, so max(width_m, (cells BAND_SAFETY + QUAD_MARGIN) e^t h_m): `cells` cells of the longest permitted
        edge across at every lattice angle and offset (C45)."""
        if self.quads:
            return max(self.width_m, (self.cells * BAND_SAFETY + QUAD_MARGIN) * np.exp(self.size_tol) * self.h_m)
        return max(self.width_m, (self.cells * BAND_SAFETY + CENTROID_MARGIN) * self.h_m)

    def dense(self, step_m: float | None = None) -> np.ndarray:
        """Path points at a spacing of W/8 (5 km at most, 50 m at least) unless given: the nearest one gives the
        distance to the path within W/256 at the corridor's edge."""
        return densify(self.path, step_m or float(np.clip(self.corridor_m / 8.0, 50.0, 5000.0)))

    def distance_m(self, xyz: np.ndarray, reach_m: float = np.inf) -> np.ndarray:
        """(N,) geodesic distance (m) to the path (to its nearest densified point); inf beyond reach_m."""
        d = self.dense()
        c = _unit(d.mean(axis=0))
        rad = float(np.arccos(np.clip((d @ c).min(), -1.0, 1.0))) + min(reach_m / R_EARTH_M, np.pi)
        out = np.full(len(xyz), np.inf)
        near = np.flatnonzero(xyz @ c >= np.cos(min(rad, np.pi)))
        if len(near):
            ch, _ = cKDTree(d).query(xyz[near], distance_upper_bound=_chord(reach_m) if np.isfinite(reach_m)
                                     else np.inf)
            out[near] = np.where(np.isfinite(ch), _arc_m(ch), np.inf)
        return out


def builtin(name: str) -> list[dict]:
    """The entries of a built-in list (BUILTIN)."""
    return json.loads(BUILTIN[name].read_text())["straits"]


def at_fraction(path: np.ndarray, f: np.ndarray) -> np.ndarray:
    """(F,3) the points at fractions f (F,) of the arc length of a (K,3) polyline of great-circle segments
    (spherical interpolation within the segment)."""
    w = 2.0 * np.arcsin(np.clip(0.5 * np.linalg.norm(path[1:] - path[:-1], axis=1), 0.0, 1.0))   # segment angles
    cum = np.r_[0.0, np.cumsum(w)]
    s = np.asarray(f, dtype=np.float64) * cum[-1]
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(w) - 1)
    wi = w[i]
    t = np.clip((s - cum[i]) / np.maximum(wi, 1e-300), 0.0, 1.0)
    sw = np.sin(wi)
    lin = sw < 1e-12                                       # a (near) zero segment: linear, then normalised
    ca = np.where(lin, 1.0 - t, np.sin((1.0 - t) * wi) / np.where(lin, 1.0, sw))
    cb = np.where(lin, t, np.sin(t * wi) / np.where(lin, 1.0, sw))
    return _unit(ca[:, None] * path[i] + cb[:, None] * path[i + 1])


def midpoint(path: np.ndarray) -> np.ndarray:
    """(3,) the point at half the arc length of a (K,3) polyline."""
    return at_fraction(path, np.array([0.5]))[0]


def in_region(xyz: np.ndarray, region) -> bool:
    """A point (3,) inside a [[strait_regions]] entry: its `box` [west, east, south, north] in degrees (west > east:
    across 180), else its spherical `polygon` [[lon, lat], ...] (meshom.size.polygon_inside)."""
    if region.box:
        w, e, so, no = region.box
        lon, lat = (float(v[0]) for v in xyz_to_lonlat(xyz[None, :]))
        return bool(so <= lat <= no and ((w <= lon <= e) if w <= e else (lon >= w or lon <= e)))
    p = np.asarray(region.polygon, dtype=np.float64)
    return bool(polygon_inside(xyz[None, :], lonlat_to_xyz(p[:, 0], p[:, 1]))[0])


def background_h_m(path: np.ndarray, h_bg) -> float:
    """widen: the largest background h (a SizeField) over the inner part (INNER) of the path, sampled every
    BG_STEP_M at most (C45: the corridor then holds `cells` cells of that h at every section)."""
    length = float(_arc_m(np.linalg.norm(path[1:] - path[:-1], axis=1)).sum())
    n = max(2, int(np.ceil((INNER[1] - INNER[0]) * length / BG_STEP_M)) + 1)    # the inner part's ends included
    return float(np.max(h_bg(at_fraction(path, np.linspace(INNER[0], INNER[1], n)))))


def profile_of(path: np.ndarray, own: str | None, coast, regions: list) -> str:
    """The strait's profile (module docstring, C46): its own [[straits]] `profile`, else the first region holding the
    path's middle point ("off" when none does), else `[coast] strait_profile`."""
    if own:
        return own
    if not regions:
        return coast.strait_profile
    m = midpoint(path)
    return next((r.profile for r in regions if in_region(m, r)), "off")


def straits_of(cfg, h_bg=None) -> list[Strait]:
    """[coast] straits (a BUILTIN name or "none") merged with the [[straits]] entries by name (an entry overrides the
    fields it gives; `enabled = false` drops the strait), each with its profile (profile_of; "off" drops it):
    resolve: h_m raised to [coast] strait_h_min_m, cells default 2; widen: h_m = background_h_m over `h_bg` (the size
    field without strait zones, meshom.run.background_h; needed when a strait is widened), cells default
    [coast] strait_widen_cells. quads / size_tol (the corridor on quads, C45) from [mesh] method and the run's size
    tolerance (Config.size_tolerance)."""
    entries = {} if cfg.coast.straits == "none" else {d["name"]: dict(d) for d in builtin(cfg.coast.straits)}
    mandatory = set(MANDATORY) if cfg.coast.source != "none" else set()
    for name in mandatory - set(entries):                 # C49: from the built-in list when no list is chosen, widened
        entries[name] = {**next(d for d in builtin("fesom2_mesh_gen") if d["name"] == name), "profile": "widen"}
    off = set()
    for s in cfg.straits:
        d = entries.setdefault(s.name, {"name": s.name})
        for k in ("path", "width_m", "h_m", "cells", "depth_m", "profile"):
            v = getattr(s, k)
            if v is not None and not (k == "path" and not v):
                d[k] = v
        if not s.enabled:
            off.add(s.name)
    quads, tol = cfg.mesh.method not in TRI_METHODS, float(cfg.size_tolerance() or 0.0)
    out = []
    for d in entries.values():
        if d["name"] in off:
            continue
        p = np.asarray(d["path"], dtype=np.float64)
        xyz = lonlat_to_xyz(p[:, 0], p[:, 1])
        prof = profile_of(xyz, d.get("profile"), cfg.coast, cfg.strait_regions)
        if prof == "off" and d["name"] in mandatory and d.get("profile") != "off":
            prof = "widen"                                # C49: a region leaving it closed does not close it
        if prof == "off":
            continue
        dep = d.get("depth_m")
        if prof == "widen":
            if h_bg is None:
                raise ValueError(f"strait {d['name']}: profile widen needs the background h (straits_of(cfg, h_bg))")
            h, cells = background_h_m(xyz, h_bg), int(d.get("cells") or cfg.coast.strait_widen_cells)
        else:
            h, cells = max(float(d["h_m"]), cfg.coast.strait_h_min_m), int(d.get("cells") or 2)
        out.append(Strait(d["name"], xyz, float(d.get("width_m") or 0.0), h, cells,
                          float(dep) if dep else None, prof, quads, tol))
    return out


# ---------------------------------------------------------------------------------------------- size zone

@dataclass(frozen=True, eq=False)
class StraitsField:
    """min over the straits of h_m within W/2 of the path, linear up to h_out_m over (h_out_m - h_m) / grad beyond;
    h_out_m elsewhere. A running minimum, so memory stays O(N) for any number of straits."""
    straits: tuple
    h_out_m: float
    grad: float

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        out = np.full(len(xyz), self.h_out_m)
        for s in self.straits:
            half = 0.5 * s.corridor_m
            tr = (self.h_out_m - s.h_m) / self.grad if np.isfinite(self.h_out_m) and self.grad > 0 else 0.0
            d = s.distance_m(xyz, half + tr)
            sel = np.flatnonzero(np.isfinite(d))
            if not len(sel):
                continue
            x = np.maximum(d[sel] - half, 0.0)
            t = np.clip(x / tr, 0.0, 1.0) if tr > 0 else (x > 0).astype(np.float64)
            v = s.h_m + (self.h_out_m - s.h_m) * t if np.isfinite(self.h_out_m) else np.where(t > 0, np.inf, s.h_m)
            out[sel] = np.minimum(out[sel], v)
        return out


# ------------------------------------------------------------------------------------------- cut and depth

def centroids(mesh: Mesh) -> np.ndarray:
    P = mesh.points
    return _unit(np.r_[P[mesh.tris].sum(axis=1), P[mesh.quads].sum(axis=1)])


def corridor_mask(cent: np.ndarray, straits: list[Strait]) -> tuple[np.ndarray, dict]:
    """(M,) elements whose centroid lies within W/2 of a strait's path, and the count per strait."""
    forced = np.zeros(len(cent), dtype=bool)
    per = {}
    for s in straits:
        m = s.distance_m(cent, 0.5 * s.corridor_m) <= 0.5 * s.corridor_m
        per[s.name] = int(m.sum())
        forced |= m
    return forced, per


def _elem_tris(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """The elements as triangles: (T + 2Q, 3) the triangles and each quad cut along its 0-2 diagonal (a convex quad
    is the union of the two), and (T + 2Q,) the element of each (tris first, then quads: water_components' order)."""
    q = mesh.quads
    T = np.vstack([mesh.tris, q[:, [0, 1, 2]], q[:, [0, 2, 3]]]).astype(np.int64)
    qi = mesh.n_tris + np.arange(mesh.n_quads, dtype=np.int64)
    return T, np.r_[np.arange(mesh.n_tris, dtype=np.int64), qi, qi]


def _elem_tree(mesh: Mesh) -> tuple[cKDTree, np.ndarray, np.ndarray]:
    T, own = _elem_tris(mesh)
    return cKDTree(_unit(mesh.points[T].sum(axis=1))), T, own


def locate(mesh: Mesh, q: np.ndarray, tree: tuple | None = None, k: int = 16) -> np.ndarray:
    """(Q,) the element of `mesh` (CCW; tris first, then quads) containing each point, -1 if none of the k nearest
    centroids' triangles does (a quad as its two triangles, `_elem_tris`; tree: `_elem_tree(mesh)`)."""
    P = mesh.points
    tree, T, own = _elem_tree(mesh) if tree is None else tree
    k = min(k, len(T))
    _, cand = tree.query(q, k=k)
    cand = cand.reshape(len(q), k)
    a, b, c = P[T[cand, 0]], P[T[cand, 1]], P[T[cand, 2]]
    qq = q[:, None, :]
    ins = ((np.einsum("ijk,ijk->ij", np.cross(a, b), qq) >= 0) & (np.einsum("ijk,ijk->ij", np.cross(b, c), qq) >= 0)
           & (np.einsum("ijk,ijk->ij", np.cross(c, a), qq) >= 0) & (np.einsum("ijk,ijk->ij", a + b + c, qq) > 0))
    first = np.argmax(ins, axis=1)
    return np.where(ins.any(axis=1), own[cand[np.arange(len(q)), first]], -1)


def depth_floor(mesh: Mesh, depth: np.ndarray, straits: list[Strait], taper: float = 1.0) -> tuple[np.ndarray, dict]:
    """Carving: depth = max(depth, depth_m w) with w = 1 within one strait size h_m of the path (every node of a
    triangle the path crosses is closer than that: at most sqrt(3)/2 h_m away) and a cosine taper
    w = (1 + cos(pi x / T)) / 2 over T = taper h_m beyond it (x = distance - h_m), 0 farther: fesom2_mesh_gen's
    carve (full floor within a half-width, then a cosine taper). The floor is the sill
    (the deepest point of the shallowest section), so it is kept to the path's band rather than the whole corridor.
    Returns the new depth and the nodes deepened per strait."""
    depth = depth.copy()
    per = {}
    for s in straits:
        if not s.depth_m:
            continue
        half, tl = s.h_m, taper * s.h_m
        d = s.distance_m(mesh.points, half + tl)
        x = np.maximum(d - half, 0.0)
        w = np.where(x <= 0.0, 1.0, np.where(x < tl, 0.5 * (1.0 + np.cos(np.pi * np.minimum(x, tl) / max(tl, 1e-9))), 0.0))
        target = s.depth_m * w
        low = np.flatnonzero(target > depth)
        depth[low] = target[low]
        per[s.name] = int(len(low))
    return depth, {"taper": taper, "n_raised": int(sum(per.values())), "raised_per_strait": per}


# ------------------------------------------------------------------------------------------------- report

def _components(mesh: Mesh) -> np.ndarray:
    ee = build_topology(mesh).edge_elems
    ee = ee[(ee >= 0).all(axis=1)]
    g = coo_matrix((np.ones(len(ee)), (ee[:, 0], ee[:, 1])), shape=(mesh.n_elems, mesh.n_elems))
    return connected_components(g, directed=False)[1]


def _lon_box(lon: np.ndarray) -> tuple[float, float]:
    """[west, east] of the longitudes: min and max, or, when the shortest interval holding them all (the complement
    of the largest gap between neighbouring longitudes on the circle) crosses 180 degrees and spans less than 180,
    that interval (west > east)."""
    s = np.sort(lon)
    gaps = np.r_[np.diff(s), s[0] + 360.0 - s[-1]]
    i = int(np.argmax(gaps))
    if i == len(s) - 1 or gaps[i] <= 180.0:
        return float(s[0]), float(s[-1])
    return float(s[i + 1]), float(s[i])


def water_components(mesh: Mesh) -> tuple[np.ndarray, list[dict]]:
    """The edge-connected water components of a mesh (decisions C44): (M,) component of every element (tris first),
    0 = the largest by area, and per component {"id", "n_elems", "n_nodes", "area_km2", "box": [west, east, south,
    north] in degrees (west > east: across 180)}, largest first."""
    from meshom.methods.m14.cut import element_areas

    lab = _components(mesh)
    area = np.bincount(lab, weights=element_areas(mesh))
    order = np.argsort(-area, kind="stable")
    rank = np.empty(len(area), dtype=np.int64)
    rank[order] = np.arange(len(area))
    lab = rank[lab]
    n = mesh.n_nodes
    pairs = np.unique(np.concatenate([np.repeat(lab[:mesh.n_tris], 3) * n + mesh.tris.ravel(),
                                      np.repeat(lab[mesh.n_tris:], 4) * n + mesh.quads.ravel()]))
    node = pairs % n
    cuts = np.searchsorted(pairs // n, np.arange(1, len(area)))
    lon, lat = xyz_to_lonlat(mesh.points)
    n_el = np.bincount(lab, minlength=len(area))
    out = []
    for k, nd in enumerate(np.split(node, cuts)):
        w, e = _lon_box(lon[nd])
        out.append({"id": k, "n_elems": int(n_el[k]), "n_nodes": len(nd),
                    "area_km2": round(float(area[order[k]]) / 1e6, 1),
                    "box": [round(w, 2), round(e, 2), round(float(lat[nd].min()), 2), round(float(lat[nd].max()), 2)]})
    return lab, out


def _edge_mean(mesh: Mesh) -> np.ndarray:
    """(T + Q,) the mean edge length (m) of every element."""
    P = mesh.points
    out = []
    for cells in (mesh.tris, mesh.quads):
        k = cells.shape[1]
        out.append(sum(_arc_m(np.linalg.norm(P[cells[:, i]] - P[cells[:, (i + 1) % k]], axis=1))
                       for i in range(k)) / k if len(cells) else np.zeros(0))
    return np.concatenate(out)


def _sections(path: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n points evenly spaced along the inner part (INNER) of the path and their unit normals (path x tangent)."""
    seg = _arc_m(np.linalg.norm(path[1:] - path[:-1], axis=1))
    cum = np.r_[0.0, np.cumsum(seg)]
    s = np.linspace(INNER[0], INNER[1], n) * cum[-1]
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    t = ((s - cum[i]) / np.maximum(seg[i], 1e-9))[:, None]
    p = _unit(path[i] * (1 - t) + path[i + 1] * t)
    tang = path[i + 1] - path[i]
    tang = _unit(tang - np.einsum("ij,ij->i", tang, p)[:, None] * p)
    return p, _unit(np.cross(p, tang))


def strait_report(mesh: Mesh, straits: list[Strait], depth: np.ndarray | None = None,
                  half_length_m: float = 150e3, comp: np.ndarray | None = None) -> list[dict]:
    """Per strait: cells across at the narrowest inner section, the sections whose path point is not in the mesh,
    the water component of each path end (`comp`: water_components' labels, computed when not given) and
    `connected` = both ends in the main water (component 0, the largest: basin to basin, decisions C44), the smallest
    element depth on the inner path (module docstring). Triangles and quads (a quad as its two triangles)."""
    tree = _elem_tree(mesh)
    comp = water_components(mesh)[0] if comp is None else comp
    emean = _edge_mean(mesh)
    edep = (np.r_[depth[mesh.tris].mean(axis=1), depth[mesh.quads].mean(axis=1)] if depth is not None else None)
    out = []
    for s in straits:
        p, nrm = _sections(s.path, N_SECTIONS)
        ds = float(np.clip(min(s.width_m or np.inf, s.corridor_m) / 20.0, 25.0, 1000.0))
        sv = np.arange(-half_length_m, half_length_m + 0.5 * ds, ds)
        i0 = int(np.argmin(np.abs(sv)))
        ang = sv / R_EARTH_M
        y = np.cos(ang)[None, :, None] * p[:, None, :] + np.sin(ang)[None, :, None] * nrm[:, None, :]
        el = locate(mesh, y.reshape(-1, 3), tree).reshape(len(p), len(sv))
        cells, widths, lbar = np.zeros(len(p)), np.zeros(len(p)), np.full(len(p), np.nan)
        for j in range(len(p)):
            wet = el[j] >= 0
            if not wet[i0]:
                continue
            dry = np.flatnonzero(~wet)
            lo = dry[dry < i0].max() + 1 if (dry < i0).any() else 0
            hi = dry[dry > i0].min() - 1 if (dry > i0).any() else len(sv) - 1
            widths[j] = (hi - lo + 1) * ds
            lbar[j] = emean[np.unique(el[j, lo:hi + 1])].mean()
            cells[j] = widths[j] / lbar[j]
        jmin = int(np.argmin(cells))
        lon, lat = xyz_to_lonlat(p[jmin:jmin + 1])
        ends = []
        for q in (s.path[0], s.path[-1]):
            e = int(locate(mesh, q[None, :], tree)[0])
            if e < 0:
                dd, ii = tree[0].query(q, k=1)
                e = int(tree[2][ii]) if _arc_m(dd) <= END_REACH_M else -1
            ends.append(e)
        rec = {"name": s.name, "corridor_m": round(s.corridor_m, 1), "h_m": s.h_m, "cells_target": s.cells,
               "n_sections": len(p), "closed_sections": int((cells == 0).sum()),
               "cells_min": round(float(cells[jmin]), 2), "width_min_m": round(float(widths[jmin]), 1),
               "edge_mean_m": None if np.isnan(lbar[jmin]) else round(float(lbar[jmin]), 1),
               "at": [round(float(lon[0]), 4), round(float(lat[0]), 4)],
               "ends_wet": [e >= 0 for e in ends],
               "ends_component": [int(comp[e]) if e >= 0 else -1 for e in ends],
               "connected": bool(ends[0] >= 0 and ends[1] >= 0 and comp[ends[0]] == comp[ends[1]] == 0)}
        if edep is not None:
            ep = locate(mesh, densify(s.path, 500.0), tree)
            frac = np.linspace(0.0, 1.0, len(ep))
            ep = ep[(ep >= 0) & (frac >= INNER[0]) & (frac <= INNER[1])]
            rec["depth_min_m"] = round(float(edep[ep].min()), 1) if len(ep) else None
        rec["ok"] = bool(rec["connected"] and rec["closed_sections"] == 0 and rec["cells_min"] >= s.cells)
        out.append(rec)
    return out
