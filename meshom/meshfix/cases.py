"""The test cases of the mesh-fixing competition (decisions B120): build, save, load.

A case is a finished meshom mesh plus everything a method and the scorer need, stored in
<MESHFIX_DIR>/cases/<name>/:

  mesh.npz      the mesh in meshom's Mesh npz format (points, tris, quads, node_flag; compete._load reads it)
  case.npz      kind (N,) int8 per node: 0 interior, 1 coast, 2 artificial boundary (cut-outs only; fixed for every
                method and no sink for defects); h_node (N,) m, the run's h at the nodes; coast_seg (S,2,3) the
                reference coast (the input's real coast edges, for coast fidelity); dist_seg (D,2,3) the coast for
                the open-ocean rule (cut-outs: also the real coast just outside the window); basin_pts (K,3) one
                point per water body of the input (for "no basin lost"); art_edges (A,2) node ids of the
                artificial boundary edges
  case.json     name, kind, source run, config path, h pickle path, size tolerance, gates, rho_c, figure windows,
                input totals (area, water bodies), the gates the input itself fails
  h.pkl         the run's size field (a pickled SizeField: h(xyz (M,3) unit vectors) -> (M,) metres), whole cases
                only; cut-outs point to their parent's
  config.resolved.toml   a copy of the run's config

Cases (CASE_SPECS): v4 (ORCA025 + Arctic v4, whole), v4_pacific and v4_atlantic (cut-outs of v4's 56-62 N band in
the windows of `probe_window`), v5 (whole, the control), core2 (core2_A2, a
non-ORCA mesh with coasts and islands).

Build: python -m meshom.meshfix.cases [NAME ...]   (on a compute node: v4 / v5 take a few minutes each)
Load:  meshom.meshfix.cases.load(NAME) -> Case
"""
import json
import os
import pickle
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import meshom
from meshom.geo import R_EARTH_M, xyz_to_lonlat
from meshom.mesh import Mesh, build_topology

OPEN_H = 2.0            # B120: a point is open ocean when its distance to the coast is > OPEN_H * h there
CUTOUT_MIN_ELEMS = 100  # B120: a cut-out keeps only its water pieces of at least this many elements
DIST_MARGIN_KM = 300.0  # B120: a cut-out's dist_seg = the parent's coast edges within this of the window's nodes


def _main_root() -> Path:
    """The main checkout (worktrees under .worktrees/ share its .tmp/)."""
    p = Path(meshom.__file__).resolve().parents[1]
    return p.parent.parent if p.parent.name == ".worktrees" else p


def meshfix_dir() -> Path:
    """<main checkout>/.tmp/meshfix, or $MESHFIX_DIR."""
    return Path(os.environ.get("MESHFIX_DIR", _main_root() / ".tmp" / "meshfix"))


MESHES = _main_root() / ".tmp" / "meshes"

# figure windows (those of the v4 maps): name -> projection spec
WIN_PACIFIC = {"name": "pacific", "proj": "stereo", "lon0": 180.0, "lon": [130.0, 230.0], "lat": [44.0, 68.0],
               "lats": [56.0, 62.0]}
WIN_ATLANTIC = {"name": "atlantic", "proj": "stereo", "lon0": -25.0, "lon": [-70.0, 25.0], "lat": [48.0, 74.0],
                "lats": [56.0, 62.0]}
WIN_GLOBAL = {"name": "global", "proj": "plate", "lon": [-180.0, 180.0], "lat": [-80.0, 90.0], "lats": []}
WIN_BEAUFORT = {"name": "beaufort", "proj": "stereo", "lon0": -145.0, "lon": [-152.0, -138.0], "lat": [69.5, 80.5],
                "lats": []}
WIN_NORTH = {"name": "north", "proj": "stereo", "lon0": 0.0, "lon": [-180.0, 180.0], "lat": [35.0, 90.0],
             "lats": [40.0, 65.0]}

# small meshes with many defects, so a change shows; glob100 is the fast case
CASE_SPECS = {
    "v4": {"run": "orca025_arctic_v4", "windows": [WIN_PACIFIC, WIN_ATLANTIC]},
    "v4_pacific": {"parent": "v4", "window": "pacific", "windows": [WIN_PACIFIC]},
    "v4_atlantic": {"parent": "v4", "window": "atlantic", "windows": [WIN_ATLANTIC]},
    "v5": {"run": "orca025_arctic_v5", "windows": [WIN_PACIFIC, WIN_ATLANTIC]},
    "core2": {"run": "core2_A2", "windows": [WIN_GLOBAL]},
    "glob100": {"run": "glob100_coast_bands", "windows": [WIN_GLOBAL]},
    "core2_mixed": {"run": "core2_mixed_coast_nobands", "windows": [WIN_GLOBAL]},
    "orca1_arctic": {"run": "orca1_40n_arctic10_fesom3", "windows": [WIN_GLOBAL, WIN_NORTH]},
    # B154: pure-triangle cases. meshom's own triangle runs with their real h ...
    "tri_v2": {"run": "core2_A2_tri_v2", "windows": [WIN_GLOBAL]},          # the must-not-get-worse control
    "tri_coast": {"run": "core2_tri_coast", "windows": [WIN_GLOBAL]},
    "tri_arctic": {"run": "glob50_arctic10_tri", "windows": [WIN_GLOBAL, WIN_NORTH]},
    # ... and FESOM2's meshes (comparison and hard test beds only), h estimated (tri/hest.py)
    "farc": {"fesom": "farc", "windows": [WIN_GLOBAL, WIN_NORTH]},
    "farc_beaufort": {"parent": "farc", "window": "beaufort", "windows": [WIN_BEAUFORT]},
    "core2f": {"fesom": "core2", "windows": [WIN_GLOBAL]},
}
FESOM2_MESHES = Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1")     # read only


def probe_window(name: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """The cut-out windows: Pacific 155 E - 140 W, Atlantic 62 W - 10 E, both 54.5-64 N."""
    if name == "pacific":
        return ((lon > 155) | (lon < -140)) & (lat > 54.5) & (lat < 64)
    if name == "atlantic":
        return (lon > -62) & (lon < 10) & (lat > 54.5) & (lat < 64)
    if name == "beaufort":                       # farc's bad Beaufort area
        return (lon > -150) & (lon < -140) & (lat > 70) & (lat < 80)
    raise KeyError(name)


# ------------------------------------------------------------------------------------------------ geometry helpers

def boundary_edges(mesh: Mesh, topo=None) -> np.ndarray:
    """(B,2) node ids of the boundary edges, oriented with the water on the left (the element's own direction)."""
    topo = build_topology(mesh) if topo is None else topo
    a, b = [], []
    for cells in (mesh.tris, mesh.quads):
        if len(cells):
            a.append(cells.ravel())
            b.append(np.roll(cells, -1, axis=1).ravel())
    a, b = np.concatenate(a), np.concatenate(b)
    n = mesh.n_nodes
    be = topo.edges[topo.edge_elems[:, 1] < 0]
    bkey = np.sort(be[:, 0] * n + be[:, 1])
    k = np.minimum(a, b) * n + np.maximum(a, b)
    i = np.searchsorted(bkey, k)
    hit = (i < len(bkey)) & (bkey[np.minimum(i, len(bkey) - 1)] == k)
    return np.column_stack([a[hit], b[hit]])


def elem_areas_m2(mesh: Mesh) -> np.ndarray:
    """(T+Q,) signed tangent-plane areas (m^2, > 0 for CCW), tris first."""
    from meshom.mesh import signed_area

    return signed_area(mesh)


def elem_centroids(mesh: Mesh) -> np.ndarray:
    parts = [mesh.points[c].mean(axis=1) for c in (mesh.tris, mesh.quads) if len(c)]
    c = np.vstack(parts) if parts else np.zeros((0, 3))
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def components(mesh: Mesh, topo=None) -> np.ndarray:
    """(T+Q,) label of the edge-connected water piece of each element."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    topo = build_topology(mesh) if topo is None else topo
    inner = topo.edge_elems[topo.edge_elems[:, 1] >= 0]
    g = coo_matrix((np.ones(len(inner)), (inner[:, 0], inner[:, 1])), shape=(mesh.n_elems, mesh.n_elems))
    return connected_components(g, directed=False)[1]


def segment_distance_km(xyz: np.ndarray, seg: np.ndarray, tree: cKDTree | None = None, k: int = 16,
                        chunk: int = 200_000) -> np.ndarray:
    """(M,) great-circle distance (km) of unit vectors xyz to the nearest of the segments seg (S,2,3): exact
    point-segment distance in chord space over the k segments with the nearest midpoints (inf without segments)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(seg) == 0:
        return np.full(len(xyz), np.inf)
    if tree is None:
        tree = cKDTree(_midpoints(seg))
    k = min(k, len(seg))
    a_all, d_all = seg[:, 0], seg[:, 1] - seg[:, 0]
    dd = np.einsum("ij,ij->i", d_all, d_all)
    out = np.empty(len(xyz))
    for s in range(0, len(xyz), chunk):
        p = xyz[s:s + chunk]
        _, idx = tree.query(p, k=k)
        idx = idx.reshape(len(p), k)
        a, d = a_all[idx], d_all[idx]
        t = np.einsum("mki,mki->mk", p[:, None, :] - a, d) / np.maximum(dd[idx], 1e-300)
        q = a + np.clip(t, 0.0, 1.0)[..., None] * d
        c = np.linalg.norm(p[:, None, :] - q, axis=2).min(axis=1)
        out[s:s + chunk] = 2.0 * np.arcsin(np.minimum(c / 2.0, 1.0)) * R_EARTH_M / 1e3
    return out


def _midpoints(seg: np.ndarray) -> np.ndarray:
    m = seg[:, 0] + seg[:, 1]
    return m / np.linalg.norm(m, axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------------------- the case

@dataclass
class Case:
    name: str
    mesh: Mesh
    kind: np.ndarray                  # (N,) int8: 0 interior, 1 coast, 2 artificial boundary
    h: object                         # SizeField: (M,3) unit vectors -> (M,) metres
    h_node: np.ndarray                # (N,) m
    coast_seg: np.ndarray             # (S,2,3) reference coast of coast fidelity
    dist_seg: np.ndarray              # (D,2,3) coast of the open-ocean rule
    basin_pts: np.ndarray             # (K,3) one point per water body of the input
    art_edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))   # (A,2) node ids
    tol: float | None = 0.20          # the run's size tolerance
    gates: dict = field(default_factory=dict)
    rho_c: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)   # case.json content (config, windows, input totals, ...)
    _dist_tree: cKDTree | None = None

    @property
    def fixed(self) -> np.ndarray:
        """(N,) bool: nodes every method must keep in place (the artificial boundary)."""
        return self.kind == 2

    def dist_coast_km(self, xyz: np.ndarray) -> np.ndarray:
        """Distance (km) of unit vectors to the case's coast (dist_seg), fixed geography independent of a
        candidate's boundary."""
        if self._dist_tree is None and len(self.dist_seg):
            self._dist_tree = cKDTree(_midpoints(self.dist_seg))
        return segment_distance_km(xyz, self.dist_seg, self._dist_tree)

    def is_open(self, xyz: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        """Open ocean (B120): distance to the coast > OPEN_H * h at the point."""
        h = self.h(xyz) if h is None else h
        return self.dist_coast_km(xyz) * 1e3 > OPEN_H * h


def case_dir(name: str) -> Path:
    return meshfix_dir() / "cases" / name


def h_file(case: Case) -> Path | None:
    """The pickle case.h was loaded from (set by load), for worker processes to unpickle; None for a case built in
    memory (pipeline --run), whose h the engines pickle themselves. Never looked up by the case's name: a run folder
    may share a name with a bench case."""
    f = case.meta.get("h_file")
    return Path(f) if f and Path(f).exists() else None


def save(case: Case, d: Path, write_h: bool = False) -> None:
    """Write mesh.npz, case.npz, case.json into d; write_h: also pickle case.h to d / meta["h_pkl"] (the builder
    writes the runs' h itself)."""
    from meshom.compete import _save

    d.mkdir(parents=True, exist_ok=True)
    _save(case.mesh, d / "mesh.npz")
    np.savez(d / "case.npz", kind=case.kind.astype(np.int8), h_node=case.h_node, coast_seg=case.coast_seg,
             dist_seg=case.dist_seg, basin_pts=case.basin_pts, art_edges=case.art_edges)
    meta = dict(case.meta, name=case.name, tol=case.tol, gates=case.gates, rho_c=case.rho_c)
    meta.pop("h_file", None)
    (d / "case.json").write_text(json.dumps(meta, indent=1))
    if write_h:
        with open(d / meta["h_pkl"], "wb") as f:
            pickle.dump(case.h, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(name: str, d: Path | None = None, with_h: bool = True) -> Case:
    """The case NAME from <MESHFIX_DIR>/cases/NAME (or the folder d). with_h=False skips unpickling h (~150 MB for
    the ORCA cases; Case.h is then None)."""
    from meshom.compete import _load

    d = case_dir(name) if d is None else Path(d)
    meta = json.loads((d / "case.json").read_text())
    mesh = _load(d / "mesh.npz")
    z = np.load(d / "case.npz")
    h = None
    if with_h:
        with open((d / meta["h_pkl"]).resolve(), "rb") as f:
            h = pickle.load(f)
    meta["h_file"] = str((d / meta["h_pkl"]).resolve())
    return Case(meta["name"], mesh, z["kind"], h, z["h_node"], z["coast_seg"], z["dist_seg"], z["basin_pts"],
                z["art_edges"], meta.get("tol"), meta.get("gates", {}), meta.get("rho_c", {}), meta)


# --------------------------------------------------------------------------------------------------- the cut-out

def cutout(mesh: Mesh, keep: np.ndarray, min_elems: int = CUTOUT_MIN_ELEMS) -> tuple[Mesh, np.ndarray, np.ndarray,
                                                                                     np.ndarray]:
    """Sub-mesh of the kept elements (keep: (T+Q,) bool, tris first) for a cut-out case.

    Elements at a pinch node of the sub-mesh (more than 2 boundary edges, where the window's staircase touches
    itself at a corner) are dropped until there is none; then the water pieces with fewer than min_elems elements.
    Returns (sub, kind, old_id, art_edges): kind per new node (2 = on an edge that is a boundary edge of the
    sub-mesh but an interior edge of `mesh`: the artificial boundary; 1 = the other boundary nodes, the real coast;
    0 = interior), old_id the parent's node id per new node, art_edges (A,2) new node ids. node_flag = 1 on every
    boundary node (meshom's convention)."""
    keep = np.asarray(keep, dtype=bool).copy()
    T = mesh.n_tris
    topo_full = build_topology(mesh)
    n = mesh.n_nodes
    full_b = topo_full.edges[topo_full.edge_elems[:, 1] < 0]
    full_bkey = np.sort(full_b[:, 0] * n + full_b[:, 1])

    def sub_of(k):
        st, sq = k[:T], k[T:]
        return Mesh(mesh.points, mesh.tris[st], mesh.quads[sq], mesh.node_flag)

    for _ in range(1000):
        s = sub_of(keep)
        tp = build_topology(s)
        be = tp.edges[tp.edge_elems[:, 1] < 0]
        cnt = np.bincount(be.ravel(), minlength=n)
        pinch = cnt > 2
        if not pinch.any():
            break
        keep &= ~np.concatenate([pinch[mesh.tris].any(axis=1), pinch[mesh.quads].any(axis=1)])
    s = sub_of(keep)
    lab = components(s)
    size = np.bincount(lab)
    ok = size[lab] >= min_elems
    ids = np.flatnonzero(keep)
    keep[ids[~ok]] = False
    s = sub_of(keep)
    used = np.zeros(n, dtype=bool)
    used[s.tris.ravel()] = True
    used[s.quads.ravel()] = True
    old_id = np.flatnonzero(used)
    loc = np.full(n, -1, dtype=np.int64)
    loc[old_id] = np.arange(len(old_id))
    sub = Mesh(mesh.points[old_id].copy(), loc[s.tris].reshape(-1, 3), loc[s.quads].reshape(-1, 4),
               np.zeros(len(old_id), dtype=np.int8),
               None if mesh.tri_zone is None else np.asarray(mesh.tri_zone)[keep[:T]])
    tp = build_topology(sub)
    be = tp.edges[tp.edge_elems[:, 1] < 0]
    ob = np.sort(old_id[be], axis=1)
    k = ob[:, 0] * n + ob[:, 1]
    i = np.searchsorted(full_bkey, k)
    real = (i < len(full_bkey)) & (full_bkey[np.minimum(i, len(full_bkey) - 1)] == k)
    art = be[~real]
    kind = np.zeros(sub.n_nodes, dtype=np.int8)
    kind[tp.boundary_node] = 1
    kind[art.ravel()] = 2
    sub.node_flag = tp.boundary_node.astype(np.int8)
    return sub, kind, old_id, art


# ---------------------------------------------------------------------------------------------------- the builder

def _basin_points(mesh: Mesh) -> np.ndarray:
    """One point per water body: the centroid of its largest element."""
    lab = components(mesh)
    area = np.abs(elem_areas_m2(mesh))
    c = elem_centroids(mesh)
    pts = []
    for k in range(lab.max() + 1):
        ids = np.flatnonzero(lab == k)
        pts.append(c[ids[np.argmax(area[ids])]])
    return np.array(pts).reshape(-1, 3)


def _segments(mesh: Mesh, edges: np.ndarray) -> np.ndarray:
    return np.stack([mesh.points[edges[:, 0]], mesh.points[edges[:, 1]]], axis=1)


def _run_h(run_dir: Path, cfg):
    """The run's h: <run>/cache/size.pkl when present (write_inputs' cache), else built as the run builds it."""
    f = run_dir / "cache" / "size.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)["h"], "cache/size.pkl"
    from meshom.run import coast_of, run_straits, size_field

    coast = coast_of(cfg)
    return size_field(cfg, coast, None, run_straits(cfg, coast)), "built from the config (run.size_field)"


def finish(case: Case, d: Path | None = None) -> Case:
    """Totals of the input (area, water bodies, the gates the input fails) into meta; saved into d when given."""
    from meshom.meshfix import score as sc

    area = float(np.abs(elem_areas_m2(case.mesh)).sum()) / 1e6
    case.meta.update(input_area_km2=area, input_water_bodies=int(len(case.basin_pts)),
                     n_nodes=case.mesh.n_nodes, n_quads=case.mesh.n_quads, n_tris=case.mesh.n_tris,
                     n_artificial_nodes=int((case.kind == 2).sum()), n_coast_nodes=int((case.kind == 1).sum()),
                     open_rule=f"distance to dist_seg > {OPEN_H:g} h(x)")
    case.meta["input_gates_failed"] = []
    case.meta["input_gates_failed"] = sorted(sc.gates_block(case, case.mesh)["failed"])
    if d is not None:
        save(case, d)
    return case


def whole_case(name: str, mesh: Mesh, h, tol: float | None, gates: dict, rho_c: dict | None = None,
               meta: dict | None = None) -> Case:
    """A case of a whole mesh: kind 1 on the boundary, the reference coast = every boundary edge (finish() not
    called)."""
    topo = build_topology(mesh)
    kind = topo.boundary_node.astype(np.int8)
    seg = _segments(mesh, boundary_edges(mesh, topo))
    return Case(name, mesh, kind, h, h(mesh.points), seg, seg, _basin_points(mesh), np.zeros((0, 2), np.int64),
                tol, gates, rho_c or {}, dict({"kind": "whole", "h_pkl": "h.pkl"}, **(meta or {})))


def cutout_case(name: str, parent: Case, keep: np.ndarray, meta: dict | None = None,
                min_elems: int = CUTOUT_MIN_ELEMS) -> tuple[Case, np.ndarray]:
    """A cut-out case of the parent's kept elements (cutout()); the reference coast = its real coast edges,
    dist_seg = the parent's coast within DIST_MARGIN_KM of its nodes. Returns (case, parent node id per node)."""
    sub, kind, old_id, art = cutout(parent.mesh, keep, min_elems)
    be = boundary_edges(sub)
    n = sub.n_nodes
    akey = np.sort(np.minimum(art[:, 0], art[:, 1]) * n + np.maximum(art[:, 0], art[:, 1]))
    bk = np.minimum(be[:, 0], be[:, 1]) * n + np.maximum(be[:, 0], be[:, 1])
    i = np.searchsorted(akey, bk)
    is_art = (i < len(akey)) & (akey[np.minimum(i, max(len(akey) - 1, 0))] == bk) if len(akey) else \
        np.zeros(len(be), dtype=bool)
    coast_seg = _segments(sub, be[~is_art])
    dist_seg = parent.coast_seg
    if len(dist_seg):
        dch, _ = cKDTree(sub.points).query(_midpoints(dist_seg),
                                           distance_upper_bound=DIST_MARGIN_KM * 1e3 / R_EARTH_M)
        dist_seg = dist_seg[np.isfinite(dch)]
    m = {"kind": "cutout", "parent": parent.name, "h_pkl": f"../{parent.name}/h.pkl",
         **{k: parent.meta[k] for k in ("source", "config", "h_source", "coast_source", "coast_treatment")
            if k in parent.meta}, **(meta or {})}
    case = Case(name, sub, kind, parent.h, parent.h_node[old_id], coast_seg, dist_seg, _basin_points(sub), art,
                parent.tol, parent.gates, {}, m)
    return case, old_id


def build_whole(name: str) -> Case:
    from meshom.compete import _load
    from meshom.config import load as load_cfg

    spec = CASE_SPECS[name]
    run = MESHES / spec["run"]
    d = case_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(run / "config.resolved.toml")
    shutil.copy(run / "config.resolved.toml", d / "config.resolved.toml")
    mesh = _load(run / "raw.npz")
    h, h_src = _run_h(run, cfg)
    with open(d / "h.pkl", "wb") as f:
        pickle.dump(h, f, protocol=pickle.HIGHEST_PROTOCOL)
    q = json.loads((run / "quality.json").read_text())
    rec = q.get("optimized", q.get("smooth", q["raw"]))
    case = whole_case(name, mesh, h, cfg.size_tolerance(), cfg.gates_all(), rec.get("rho_c", {}),
                      {"source": str(run / "raw.npz"), "config": str(run / "config.resolved.toml"),
                       "h_source": f"{run.name}: {h_src}", "coast_source": cfg.coast.source,
                       "coast_treatment": cfg.coast.treatment, "windows": spec["windows"],
                       "input_record": {k: rec[k] for k in ("gates_passed", "gates_total") if k in rec}
                       | {"size_fidelity_p99": rec["quality"]["size_fidelity"]["p99"]}})
    return finish(case, d)


def build_fesom(name: str) -> Case:
    """A whole case of a FESOM2 mesh (read only from FESOM2_MESHES): no stored h, so h = tri.hest.EstimatedH of the
    mesh (recorded in case.json); the gates are meshom's defaults (DEFAULT_GATES, no size tolerance), so only
    gates.new_failed means anything."""
    from meshom.config import DEFAULT_GATES, tolerance_gates
    from meshom.io.fesom import read_fesom
    from meshom.meshfix.tri.hest import EstimatedH

    spec = CASE_SPECS[name]
    src = FESOM2_MESHES / spec["fesom"]
    d = case_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    mesh = read_fesom(src)
    mesh.node_flag = build_topology(mesh).boundary_node.astype(np.int8)
    h = EstimatedH(mesh)
    with open(d / "h.pkl", "wb") as f:
        pickle.dump(h, f, protocol=pickle.HIGHEST_PROTOCOL)
    case = whole_case(name, mesh, h, None, tolerance_gates(dict(DEFAULT_GATES), None, {}), {},
                      {"source": str(src), "config": None, "h_source": h.describe(), "windows": spec["windows"]})
    return finish(case, d)


def build_cutout(name: str) -> Case:
    spec = CASE_SPECS[name]
    parent = load(spec["parent"])
    m = parent.mesh
    lon, lat = xyz_to_lonlat(m.points)
    inwin = probe_window(spec["window"], lon, lat)
    keep = np.concatenate([inwin[m.tris].all(axis=1), inwin[m.quads].all(axis=1)])
    win = {"probe": spec["window"], "rule": "elements with every node in the probe window",
           "min_elems": CUTOUT_MIN_ELEMS, "dist_margin_km": DIST_MARGIN_KM}
    case, old_id = cutout_case(name, parent, keep, {"window": win, "windows": spec["windows"]})
    d = case_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "parent_node_id.npy", old_id)
    return finish(case, d)


def build(name: str) -> Case:
    spec = CASE_SPECS[name]
    return build_cutout(name) if "parent" in spec else build_fesom(name) if "fesom" in spec else build_whole(name)


if __name__ == "__main__":
    import time

    for nm in sys.argv[1:] or list(CASE_SPECS):
        t = time.perf_counter()
        c = build(nm)
        print(f"{nm}: {c.mesh.n_nodes} nodes, {c.mesh.n_quads} quads, {c.mesh.n_tris} tris, kinds "
              f"{np.bincount(c.kind, minlength=3).tolist()}, water bodies {len(c.basin_pts)}, input gates failed "
              f"{c.meta['input_gates_failed']}, {time.perf_counter() - t:.0f} s", flush=True)
