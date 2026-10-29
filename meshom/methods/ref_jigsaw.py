"""ref_jigsaw, ref_jigsaw_fit: JIGSAW (Engwirda 2017, JIGSAW-GEO) as a REFERENCE BASELINE, not a meshom method.

The `ref_` prefix marks an external generator run as is (external generators may be run as reference baselines,
never as a part of meshom); nothing of JIGSAW is linked or pasted, it runs as a separate executable on text files.

What is called: the `jigsaw` executable (0.9.14, conda-forge) of the separate env `meshom-jigsaw` (the main env has no
JIGSAW; the ctypes path of jigsawpy 0.3.3 segfaults with a lon-lat hfun, the file + executable path works).
The input files are written here in JIGSAW's text format (no jigsawpy import), into <run dir>/jigsaw/ (the run dir is
MESHOM_RUN_DIR, set by `meshom compete`, else .tmp/ref_jigsaw/<pid>/<variant>); the output mesh file is read back and
deleted (raw.npz holds the mesh), the inputs and JIGSAW's log stay.

Settings, those of fesom2_mesh_gen (D. Sein, AWI):
- geom: ELLIPSOID-MESH, radii R = 6371 km (R_EARTH_M) on all three axes;
- hfun: ELLIPSOID-GRID, h in km on a lon-lat grid (bilinear in lon-lat inside JIGSAW), HFUN_SCAL absolute,
  HFUN_HMIN / HFUN_HMAX = min / max of the grid values;
- MESH_DIMS 2, MESH_KERN default (delfront), OPTM_ITER 32, OPTM_QTOL 1e-5, VERBOSITY 0, every other option default;
- no gradient limiting: fesom2_mesh_gen runs jigsawpy's `marche` (|grad h| <= a limit) only when a limit is configured;
  meshom's h is already limited (cases: g <= 0.1; `meshom run`: size.limit_gradient), and jigsaw() itself does not
  limit.

The lon-lat h grid (`hfun_grid`): lon -180..180 and lat -90..90 inclusive (JIGSAW detects the periodic seam from
cos(lon_first) = cos(lon_last); the 180 column is set equal to the -180 column), start step 0.5 deg, then every lon and
lat interval is halved while the bilinear interpolation misses h by more than `grid_tol` (1 %, relative) at the edge
midpoints or the cell centres of any row / column, down to `grid_min_step_deg` (0.005 deg). A tensor grid, so a
refined lon interval is refined on every latitude. Poles: the lat = +-90 rows hold h at the pole (one value per row;
JIGSAW's pole extrapolation is not used because the grid reaches the poles). The bilinear error of the final grid is
measured on 500 000 Fibonacci points and recorded (`grid.err_rel_max`, `p99`).

Coast:
- `ref_jigsaw` (treatment A, what fesom2_mesh_gen does): JIGSAW meshes the closed sphere, then the elements whose
  centroid is on land are removed and FESOM2's clean-up (spurs, pinched nodes, the largest basin) applied:
  meshom.run.cut_land = meshom.methods.m14.cut (the rules are described there).
- `ref_jigsaw_fit` (treatment B, conforming): the coast loops resampled at h (domain.resample_loops) are given to
  JIGSAW as geometry: POINT (lon, lat in radians) and EDGE2 (each loop closed) of the ELLIPSOID-MESH, with
  GEOM_FEAT and MESH_TOP1 on (JIGSAW's own examples pair them with edge geometry; corners sharper than GEOM_ETA1 = 45
  deg become fixed nodes, the other loop vertices are not kept: JIGSAW resamples the arcs by its h). JIGSAW meshes the
  whole sphere with the coast as embedded edges; the elements with the centroid on land (left of no loop) are dropped,
  no clean-up. Without loops the input files are byte-identical to `ref_jigsaw`'s, so both give the same mesh.

Output: unit vectors (JIGSAW's points in km divided by their norm, i.e. projected exactly onto the unit sphere),
triangles made CCW seen from outside, unused points dropped, boundary nodes first with node_flag 1 (m14.cut.extract).
Records (`ref_jigsaw.json` next to the mesh, also returned by `run`): the options, the grid, the time split
(h grid / write / JIGSAW wall / read / cut / coast fit), JIGSAW's own stage times parsed from its log, JIGSAW's peak
RSS (GNU time's %M of the JIGSAW process: the compete record's peak_rss_mb is the Python process only and misses it;
getrusage of the children would count the Python image shared at vfork), node and element counts, the cut counts
and the coast fit (m14.cut.coast_fit) on coast cases. The compete wall time of the method includes all of it (JIGSAW
runs inside the method call).
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from meshom.domain import Domain, in_water, loop_edges, resample_loops
from meshom.geo import R_EARTH_M, lonlat_to_xyz, xyz_to_lonlat
from meshom.mesh import Mesh
from meshom.methods import register
from meshom.size import SizeField, evaluate

JIGSAW_BIN = "/home/a/a270126/conda/envs/meshom-jigsaw/bin/jigsaw"
R_KM = R_EARTH_M / 1e3

DEFAULTS = {
    "optm_iter": 32,            # fesom2_mesh_gen's default
    "optm_qtol": 1.0e-5,        # fesom2_mesh_gen's default
    "grid_tol": 0.01,           # relative bilinear error of the lon-lat h grid at midpoints / cell centres
    "grid_step0_deg": 0.5,
    "grid_min_step_deg": 0.005,
    "fix_coast_nodes": False,   # _fit: tag the given coast vertices -1 (user features: kept as mesh nodes)
    "coast_fit": True,          # measure the boundary against the true coast (m14.cut.coast_fit), timed separately
    "jigsaw": JIGSAW_BIN,
}


def _opts(opts: dict | None) -> dict:
    o = dict(DEFAULTS)
    o.update(opts or {})
    return o


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ------------------------------------------------------------------------------------------------------ h grid

def _sample(h: SizeField, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """(len(lat), len(lon)) h in km; rows at lat = +-90 hold h at the pole."""
    LON, LAT = np.meshgrid(lon, lat)
    v = evaluate(h, lonlat_to_xyz(LON.ravel(), LAT.ravel())).reshape(LON.shape) / 1e3
    for i, z in ((0, -1.0), (len(lat) - 1, 1.0)):
        if abs(lat[i]) == 90.0:
            v[i] = float(h(np.array([[0.0, 0.0, z]]))[0]) / 1e3
    return v


def _rel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b) / a


def hfun_grid(h: SizeField, tol: float = 0.01, step0: float = 0.5, min_step: float = 0.005) -> tuple:
    """(lon, lat, val km (ny, nx), passes): the lon-lat grid refined until bilinear interpolation reproduces h to `tol`
    at every edge midpoint and cell centre (module docstring)."""
    lon = np.linspace(-180.0, 180.0, int(round(360.0 / step0)) + 1)
    lat = np.linspace(-90.0, 90.0, int(round(180.0 / step0)) + 1)
    passes = 0
    while True:
        passes += 1
        v = _sample(h, lon, lat)
        lonm, latm = 0.5 * (lon[1:] + lon[:-1]), 0.5 * (lat[1:] + lat[:-1])
        ex = _rel(_sample(h, lonm, lat), 0.5 * (v[:, 1:] + v[:, :-1]))
        ey = _rel(_sample(h, lon, latm), 0.5 * (v[1:] + v[:-1]))
        ec = _rel(_sample(h, lonm, latm), 0.25 * (v[1:, 1:] + v[1:, :-1] + v[:-1, 1:] + v[:-1, :-1]))
        rx = ((ex.max(axis=0) > tol) | (ec.max(axis=0) > tol)) & (np.diff(lon) >= 2.0 * min_step)
        ry = ((ey.max(axis=1) > tol) | (ec.max(axis=1) > tol)) & (np.diff(lat) >= 2.0 * min_step)
        if not rx.any() and not ry.any():
            v[:, -1] = v[:, 0]                                  # lon 180 = lon -180 (the same points)
            return lon, lat, v, passes
        lon = np.sort(np.r_[lon, lonm[rx]])
        lat = np.sort(np.r_[lat, latm[ry]])


def _bilinear(lon: np.ndarray, lat: np.ndarray, v: np.ndarray, lo: np.ndarray, la: np.ndarray) -> np.ndarray:
    j = np.clip(np.searchsorted(lon, lo, side="right") - 1, 0, len(lon) - 2)
    i = np.clip(np.searchsorted(lat, la, side="right") - 1, 0, len(lat) - 2)
    fx = ((lo - lon[j]) / (lon[j + 1] - lon[j]))
    fy = ((la - lat[i]) / (lat[i + 1] - lat[i]))
    return ((1 - fx) * (1 - fy) * v[i, j] + fx * (1 - fy) * v[i, j + 1]
            + (1 - fx) * fy * v[i + 1, j] + fx * fy * v[i + 1, j + 1])


def grid_error(h: SizeField, lon: np.ndarray, lat: np.ndarray, v: np.ndarray, m: int = 500_000) -> dict:
    """Relative error of the bilinear lon-lat interpolation against h on m Fibonacci points."""
    from meshom.size import _fibonacci

    p = _fibonacci(m)
    lo, la = xyz_to_lonlat(p)
    e = _rel(evaluate(h, p) / 1e3, _bilinear(lon, lat, v, lo, la))
    return {"err_rel_max": float(e.max()), "err_rel_p99": float(np.percentile(e, 99)), "n_check": m}


# ------------------------------------------------------------------------------------------------ JIGSAW files

def _block(f, header: str, rows: np.ndarray, fmt: str) -> None:
    f.write(header)
    if len(rows):
        f.write((fmt * len(rows)) % tuple(np.asarray(rows).ravel()))


def write_hfun(path: Path, lon: np.ndarray, lat: np.ndarray, v: np.ndarray) -> None:
    """ELLIPSOID-GRID, h in km; VALUE in column-major order of (lat, lon), the layout of JIGSAW's grid files."""
    with open(path, "w") as f:
        f.write(f"# hfun.msh; meshom ref_jigsaw\nMSHID=3;ellipsoid-grid\nRADII={R_KM:.17g};{R_KM:.17g};{R_KM:.17g}\n"
                "NDIMS=2\n")
        _block(f, f"COORD=1;{len(lon)}\n", np.radians(lon), "%.17g\n")
        _block(f, f"COORD=2;{len(lat)}\n", np.radians(lat), "%.17g\n")
        _block(f, f"VALUE={v.size};1\n", v.reshape(-1, order="F"), "%.9g\n")


def write_geom(path: Path, loops: list[np.ndarray], point_tag: int = 0) -> None:
    """ELLIPSOID-MESH of radius R_KM; with loops: POINT (lon, lat radians, tag point_tag; -1 = a user feature that
    JIGSAW keeps as a mesh node) and EDGE2 (closed loops, tag = loop index)."""
    with open(path, "w") as f:
        f.write("# geom.msh; meshom ref_jigsaw\nMSHID=3;ellipsoid-mesh\n")
        if loops:
            f.write("NDIMS=2\n")
        f.write(f"RADII={R_KM:.17g};{R_KM:.17g};{R_KM:.17g}\n")
        if loops:
            P = np.vstack(loops)
            lo, la = xyz_to_lonlat(P)
            pts = np.column_stack([np.radians(lo), np.radians(la), np.full(len(P), point_tag)])
            _block(f, f"POINT={len(P)}\n", pts, "%.17g;%.17g;%d\n")
            e = loop_edges(loops)
            tag = np.repeat(np.arange(len(loops)), [len(lp) for lp in loops])
            _block(f, f"EDGE2={len(e)}\n", np.column_stack([e, tag]), "%d;%d;%d\n")


def write_jig(path: Path, wd: Path, hmin: float, hmax: float, o: dict, features: bool) -> None:
    lines = [f"GEOM_FILE={wd / 'geom.msh'}", f"HFUN_FILE={wd / 'hfun.msh'}", f"MESH_FILE={wd / 'mesh.msh'}",
             "HFUN_SCAL=absolute", f"HFUN_HMAX={hmax!r}", f"HFUN_HMIN={hmin!r}", "MESH_DIMS=2",
             f"OPTM_ITER={int(o['optm_iter'])}", f"OPTM_QTOL={float(o['optm_qtol'])!r}", "VERBOSITY=0"]
    if features:
        lines += ["GEOM_FEAT=TRUE", "MESH_TOP1=TRUE"]
    path.write_text("# opts.jig; meshom ref_jigsaw\n" + "".join(f"  {s}\n" for s in lines))


def read_mesh(path: Path) -> dict:
    """{"POINT": (N,4) x;y;z;tag km, "TRIA3": (T,4), "EDGE2": (E,3) when present} of a JIGSAW output .msh."""
    lines = Path(path).read_text().splitlines()
    out, k = {}, 0
    while k < len(lines):
        s = lines[k].strip()
        key, _, val = s.partition("=")
        key = key.strip().upper()
        if not s.startswith("#") and key in ("POINT", "EDGE2", "TRIA3", "QUAD4"):
            n = int(val.split(";")[0])
            body = ";".join(lines[k + 1:k + 1 + n])
            out[key] = np.array(body.split(";"), dtype=np.float64).reshape(n, -1) if n else np.zeros((0, 4))
            k += 1 + n
        else:
            k += 1
    return out


_STAGE = re.compile(r"^\s*(\S.*?)\.\.\.\s*$")
_DONE = re.compile(r"Done\.\s*\(([-+0-9.eE]+)\s*sec\)")


def parse_times(log: str) -> dict:
    """JIGSAW's own stage times (s) from its stdout: each "<stage>..." line paired with the next "Done. (t sec)"."""
    stages, cur = {}, None
    for line in log.splitlines():
        m = _STAGE.match(line)
        if m:
            cur = m.group(1).strip()
            continue
        m = _DONE.search(line)
        if m and cur is not None:
            stages[cur] = stages.get(cur, 0.0) + float(m.group(1))
            cur = None
    return {"stages": stages, "total_s": float(sum(stages.values()))}


# --------------------------------------------------------------------------------------------------------- run

def _run_dir(variant: str) -> Path:
    d = os.environ.get("MESHOM_RUN_DIR")
    if d:
        return Path(d).resolve()
    import meshom

    return Path(meshom.__file__).resolve().parents[1] / ".tmp" / "ref_jigsaw" / str(os.getpid()) / variant


def jigsaw_sphere(h: SizeField, loops: list[np.ndarray], wd: Path, o: dict, t: dict) -> tuple:
    """Run JIGSAW on the sphere with h and optional coast edges; (points (N,3) unit, tris (T,3) CCW, feature edges
    (E,2), info). t receives the time split."""
    t0 = time.perf_counter()
    lon, lat, v, passes = hfun_grid(h, o["grid_tol"], o["grid_step0_deg"], o["grid_min_step_deg"])
    t["h_grid_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    gerr = grid_error(h, lon, lat, v)
    t["h_grid_check_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    wd.mkdir(parents=True, exist_ok=True)
    write_hfun(wd / "hfun.msh", lon, lat, v)
    write_geom(wd / "geom.msh", loops, -1 if o["fix_coast_nodes"] else 0)
    write_jig(wd / "opts.jig", wd, float(v.min()), float(v.max()), o, bool(loops))
    t["write_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    r = subprocess.run(["/usr/bin/time", "-f", "MESHOM_MAXRSS_KB %M", o["jigsaw"], str(wd / "opts.jig")], cwd=wd,
                       capture_output=True, text=True)
    t["jigsaw_s"] = time.perf_counter() - t0
    rss_mb = int(re.search(r"MESHOM_MAXRSS_KB (\d+)", r.stderr).group(1)) / 1024.0    # JIGSAW's own peak (GNU time)
    (wd / "jigsaw.log").write_text(r.stdout + r.stderr)
    if r.returncode or not (wd / "mesh.msh").exists():
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-20:])
        raise RuntimeError(f"jigsaw failed (rc {r.returncode}):\n{tail}")
    t0 = time.perf_counter()
    msh = read_mesh(wd / "mesh.msh")
    (wd / "mesh.msh").unlink()
    P = msh["POINT"][:, :3]
    T = msh["TRIA3"][:, :3].astype(np.int64)
    E = msh["EDGE2"][:, :2].astype(np.int64) if "EDGE2" in msh else np.zeros((0, 2), dtype=np.int64)
    used = np.zeros(len(P), dtype=bool)
    used[T.ravel()] = True
    nmap = np.cumsum(used) - 1
    P = _unit(P[used])
    T, E = nmap[T], nmap[E]
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    neg = np.einsum("ij,ij->i", a, np.cross(b, c)) < 0
    T[neg] = T[neg][:, [0, 2, 1]]
    t["read_s"] = time.perf_counter() - t0
    info = {"grid": {"n_lon": len(lon), "n_lat": len(lat), "n_values": int(v.size), "passes": passes,
                     "dlon_min_deg": float(np.diff(lon).min()), "dlat_min_deg": float(np.diff(lat).min()),
                     "h_min_km": float(v.min()), "h_max_km": float(v.max()), **gerr},
            "jigsaw_reported": parse_times(r.stdout), "jigsaw_peak_rss_mb": rss_mb, "n_points_jigsaw": int(len(used)),
            "n_points_unused": int((~used).sum()), "n_tris_jigsaw": int(len(T)), "n_feature_edges": int(len(E)),
            "n_flipped": int(neg.sum())}
    return P, T, E, info


def run(domain: Domain, h: SizeField, opts: dict | None = None, fit: bool = False) -> tuple[Mesh, dict]:
    o = _opts(opts)
    variant = "ref_jigsaw_fit" if fit else "ref_jigsaw"
    t_start = time.perf_counter()
    rd = _run_dir(variant)
    t: dict = {}
    loops = resample_loops(domain, h) if (fit and domain.loops) else []
    P, T, E, info = jigsaw_sphere(h, loops, rd / "jigsaw", o, t)
    mesh = Mesh(P, T, np.zeros((0, 4), dtype=np.int64), np.zeros(len(P), dtype=np.int8))
    if domain.loops:
        t0 = time.perf_counter()
        if fit:
            from meshom.methods.m14.cut import extract

            cent = _unit(P[T].sum(axis=1))
            keep = in_water(Domain(loops, "fit"), cent)
            n_before = mesh.n_elems
            mesh, nmap, bl = extract(mesh, keep)
            fe = nmap[E] if len(E) else E
            fe = fe[(fe >= 0).all(axis=1)] if len(fe) else fe
            bset = {tuple(sorted(x)) for lp in bl for x in zip(lp, np.roll(lp, -1))}
            on_feat = sum(tuple(sorted(x)) in bset for x in fe.tolist())
            info["cut"] = {"n_elems_before": n_before, "n_removed_land": int((~keep).sum()), "n_loops": len(bl),
                           "n_boundary_edges": len(bset), "n_feature_edges_kept": int(len(fe)),
                           "boundary_edges_on_features": int(on_feat)}
        else:
            from meshom.run import cut_land

            mesh, info["cut"] = cut_land(mesh, domain)
        t["cut_s"] = time.perf_counter() - t0
        if o["coast_fit"]:
            from meshom.methods.m14.cut import coast_fit

            t0 = time.perf_counter()
            info["coast_fit"] = coast_fit(mesh, domain, h)
            t["coast_fit_s"] = time.perf_counter() - t0
    t["total_s"] = time.perf_counter() - t_start
    info = {"method": variant, "domain": domain.name, "h": repr(h), "params": o,
            "n_nodes": mesh.n_nodes, "n_tris": mesh.n_tris, "n_boundary_nodes": int((mesh.node_flag == 1).sum()),
            "n_coast_points_given": int(sum(len(lp) for lp in loops)), "timings": t, **info}
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "ref_jigsaw.json").write_text(json.dumps(info, indent=1))
    return mesh, info


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """JIGSAW on the closed sphere, land cut away with FESOM2's rules (fesom2_mesh_gen's pipeline)."""
    return run(domain, h, opts, fit=False)[0]


@register("ref_jigsaw_fit", overrides={})          # run(..., fit=True): an argument, not an option
def generate_fit(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """JIGSAW with the coast loops (resampled at h) as embedded edges; the land side dropped (conforming coast)."""
    return run(domain, h, opts, fit=True)[0]
