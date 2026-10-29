"""The node tool for the mesh depth (Tools, "Mesh depth"; decision U35): the nodes of the
mesh shown inside a box or a polygon drawn on the map (the core's region_mask, so a config
rule saved from the area selects the same nodes), their depth statistics (the mesh depth of meshview.depth_of: the
last write's base with the current edits; the bottom grid at the nodes, point samples as the profile's; how many sit
at the run's floor), and their edits: "set" / "min" / "max" D appended to the mesh's <mesh stem>.depth_edits.json,
which the core applies at the next write (meshom.bathymetry.apply_node_edits, C2, decisions O37).

    POST /api/mesh/depth/select   {path, box: [west, east, south, north] | polygon: [[lon, lat], ...]} -> summary()
    POST /api/mesh/depth/edit     {path, op, depth_m, box | polygon}   -> {edits: edits_list(), area: summary()}
    POST /api/mesh/depth/undo     {path}  /  /api/mesh/depth/remove {path, index}   -> {edits}
    GET  /api/mesh/depth/edits?path=                                   -> {file, edits}
"""
import datetime
import json
from pathlib import Path

import numpy as np

from meshom.bathymetry import region_mask
from meshom.config import DepthRegionCfg
from ui.common.meshview import edits_file

BOTTOM_POINTS = 2000          # the bottom grid is read at most at this many of the selected nodes (point by point)
MAP_POINTS = 50000            # at most this many selected nodes are sent for the map (every k-th beyond)


def region_of(box=None, polygon=None) -> DepthRegionCfg:
    """The page's area as a [[bathymetry.regions]] area: the map's box [west, east, south, north] (east may exceed 180
    across the antimeridian) becomes the config's [lon_min, lat_min, lon_max, lat_max]; a polygon stays."""
    if box is not None:
        w, e, s, n = (float(x) for x in box)
        return DepthRegionCfg(box=[w, s, e, n])
    return DepthRegionCfg(polygon=[[float(a), float(b)] for a, b in polygon])


def outline_of(box=None, polygon=None) -> list:
    """The area's outline for the edits file: a box's four corners (longitudes in [-180, 180)), else the polygon."""
    if box is None:
        return [[float(a), float(b)] for a, b in polygon]
    w, e, s, n = (float(x) for x in box)
    wl = lambda x: (x + 180.0) % 360.0 - 180.0  # noqa: E731
    return [[wl(w), s], [wl(e), s], [wl(e), n], [wl(w), n]]


_POINTS: dict = {}


def points64(mv) -> np.ndarray:
    """The mesh file's node positions in float64 (the view keeps float32: a node on a box edge would fall out where
    the core, selecting on the file's points at 1e-9 degrees, keeps it); the last mesh's kept."""
    mf = Path(mv.path)
    key = (str(mf), mf.stat().st_mtime)
    if key not in _POINTS:
        with np.load(mf) as z:
            pts = np.asarray(z["points"], dtype=np.float64)
        _POINTS.clear()
        _POINTS[key] = pts
    return _POINTS[key]


def select(mv, box=None, polygon=None) -> np.ndarray:
    """Node ids of mesh view mv inside the area (meshom.bathymetry.region_mask on the file's points, as a config rule
    would select them at the write)."""
    return np.flatnonzero(region_mask(points64(mv), region_of(box, polygon)))


def run_mesh(mesh_path) -> Path:
    """The mesh file, when it is in a `meshom run` folder (the core applies its edits at `meshom write`, which writes
    with the folder's config.resolved.toml); else ValueError."""
    mf = Path(mesh_path)
    if not (mf.parent / "config.resolved.toml").is_file():
        raise ValueError(f"{mf.parent.name} is not a `meshom run` folder (no config.resolved.toml): nothing would write "
                         "this mesh with its depth edits")
    return mf


def floor_of(mesh_path: Path) -> float | None:
    """The depth floor the core clamps edits to (meshom.run.depth_floor_m) in the format of the mesh's last write,
    else its run's; None outside a `meshom run` folder."""
    from meshom import config
    from meshom.run import depth_floor_m

    mf = Path(mesh_path)
    f = mf.parent / "config.resolved.toml"
    if not f.is_file():
        return None
    cfg = config.load(f)
    try:
        fmt = json.loads((mf.parent / f"fesom-{mf.stem}" / "quality.json").read_text())[mf.stem]["format"]
    except (OSError, ValueError, KeyError):
        fmt = cfg.mesh.format
    return depth_floor_m(cfg, fmt)


def read_edits(mf: Path) -> dict | None:
    f = edits_file(Path(mf))
    return json.loads(f.read_text()) if f.exists() else None


def edits_list(mf: Path) -> list:
    """The mesh's edits for the page: rule, D, how many nodes, date, outline."""
    e = read_edits(mf) or {"edits": []}
    return [{"op": x["op"], "depth_m": x["depth_m"], "n": len(x["nodes"]), "date": x.get("date"),
             "outline": x.get("outline")} for x in e["edits"]]


def _store(mf: Path, e: dict) -> None:
    f = edits_file(mf)
    if e["edits"]:
        f.write_text(json.dumps(e))
    else:                                              # nothing left: no file, nothing for the write to apply
        f.unlink(missing_ok=True)


def add_edit(mv, op: str, depth_m: float, box=None, polygon=None) -> np.ndarray:
    """Append {op, depth_m, nodes, outline, date} to the shown mesh's edits file (created with its name and node
    count, as the core checks them); returns the node ids. ValueError outside a run folder or for an empty area."""
    mf = run_mesh(mv.path)
    ids = select(mv, box, polygon)
    if not len(ids):
        raise ValueError("no node of the mesh inside the area")
    e = read_edits(mf) or {"mesh": mf.name, "n_nodes": int(len(mv.lonlat)), "edits": []}
    e["edits"].append({"op": op, "depth_m": float(depth_m), "nodes": ids.tolist(), "outline": outline_of(box, polygon),
                       "date": datetime.datetime.now().isoformat(timespec="seconds")})
    _store(mf, e)
    return ids


def remove_edit(mesh_path, index: int | None = None) -> None:
    """Remove edit `index` (None: the last) of the mesh's edits file."""
    mf = run_mesh(mesh_path)
    e = read_edits(mf)
    if e and e["edits"]:
        e["edits"].pop(-1 if index is None else int(index))
        _store(mf, e)


def summary(mv, ids: np.ndarray, src: Path | None, variable: str = "elevation") -> dict:
    """What the page shows of a selection: the count, the node ids, the mesh depth's min / median / mean / max and
    its source, the bottom grid's min / median / max at the nodes (at most BOTTOM_POINTS of them, evenly spread),
    the run's floor and how many nodes sit on it, the nodes' positions (at most MAP_POINTS)."""
    from ui.common import profile

    out = {"n": int(len(ids)), "ids": ids.tolist(), "depth": None, "depth_source": mv.depth_source, "bottom": None,
           "bottom_points": 0, "floor_m": None, "at_floor": None}
    if len(ids) and mv.node_depth is not None:
        d = mv.node_depth[ids]
        out["depth"] = {"min": float(d.min()), "median": float(np.median(d)), "mean": float(d.mean()), "max": float(d.max())}
        out["floor_m"] = floor_of(Path(mv.path))
        if out["floor_m"] is not None:
            out["at_floor"] = int((d <= out["floor_m"] + 0.01).sum())
    if len(ids) and src is not None:
        k = ids[np.linspace(0, len(ids) - 1, min(len(ids), BOTTOM_POINTS)).round().astype(np.int64)]
        try:
            z = profile.elevation_at(src, mv.lonlat[k, 0].astype(np.float64), mv.lonlat[k, 1].astype(np.float64), variable)
            out["bottom"] = {"min": float(z.min()), "median": float(np.median(z)), "max": float(z.max())}
            out["bottom_points"] = int(len(k))
        except ImportError:                     # no netCDF4 in the server's Python: no bottom
            pass
    m = ids[:: max(1, int(np.ceil(len(ids) / MAP_POINTS)))]
    out["lon"], out["lat"] = mv.lonlat[m, 0].tolist(), mv.lonlat[m, 1].tolist()
    return out
