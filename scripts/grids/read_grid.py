"""Read public curvilinear ocean grids (cell corners) into a meshom Mesh of quads on the unit sphere.

Each family reader returns a Grid: per-cell corners (M,4) lon/lat in degrees, a wet mask and the logical position of
every cell (block, j, i). `to_mesh` merges corners that coincide on the sphere (within 1e-3 of the median edge), so
periodic seams, tripolar folds, halo copies and block joins become shared nodes; a cell that repeats another (halo or
fold copy) is dropped; corners are put in CCW order; a cell with only 3 distinct corners (a collapsed pole cell)
becomes a triangle. Sources, licences and download dates: SOURCES.md in the data folder (GRIDS below).

    python scripts/grids/read_grid.py NAME        # one-line summary of the full and the wet mesh
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from meshom.geo import lonlat_to_xyz, tangent_coords
from meshom.mesh import Mesh

DATA = Path(__file__).resolve().parents[2]
DATA = (DATA.parent.parent if DATA.parent.name == ".worktrees" else DATA).parent / "curvilinear_grids"


@dataclass
class Grid:
    lon: np.ndarray            # (M,4) deg, corners of each cell
    lat: np.ndarray            # (M,4) deg
    wet: np.ndarray            # (M,) bool
    block: np.ndarray          # (M,) int, logical block (face, tile)
    j: np.ndarray              # (M,) int, row in the block
    i: np.ndarray              # (M,) int, column in the block
    shapes: list = field(default_factory=list)   # (ny, nx) per block


def _cells_from_nodes(x, y, wet, block=0, wrap_i=False):
    """Node arrays (ny+1, nx+1) [or (ny, nx+...) with wrap] -> Grid of the ny x nx cells; wet (ny, nx)."""
    if wrap_i:
        x, y = np.concatenate([x, x[:, :1]], axis=1), np.concatenate([y, y[:, :1]], axis=1)
    ny, nx = x.shape[0] - 1, x.shape[1] - 1
    c = [(slice(0, -1), slice(0, -1)), (slice(0, -1), slice(1, None)), (slice(1, None), slice(1, None)),
         (slice(1, None), slice(0, -1))]
    lon = np.stack([x[a, b].ravel() for a, b in c], axis=1)
    lat = np.stack([y[a, b].ravel() for a, b in c], axis=1)
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    return Grid(lon, lat, np.asarray(wet, bool)[:ny, :nx].ravel(), np.full(ny * nx, block), jj.ravel(), ii.ravel(),
                [(ny, nx)])


def _concat(grids: list) -> Grid:
    return Grid(*(np.concatenate([getattr(g, k) for g in grids]) for k in ("lon", "lat", "wet", "block", "j", "i")),
                sum((g.shapes for g in grids), []))


def _nc(path):
    import netCDF4
    ds = netCDF4.Dataset(path)
    ds.set_auto_mask(False)
    return ds


def _2d(v):
    a = np.asarray(v[:], dtype=np.float64)
    while a.ndim > 2:
        a = a[0]
    return a


# ------------------------------------------------------------------------------------------------------ readers

def read_nemo(coords, mask_file=None, mask_var=None):
    """NEMO: F points (glamf, gphif) are the NE corners of the T cells: cell (j, i) has corners f(j-1, i-1), f(j-1, i),
    f(j, i), f(j, i-1); row 0 is dropped. With halo columns (glamt[:, 0] == glamt[:, -2]) column 0 is a copy and is
    dropped too; without halos (NEMO 5 files) the west corners of column 0 wrap to the last column when the grid is
    global. Mask: tmask (level 0), top_level/bottom_level > 0, or a bathymetry > 0 on the T grid."""
    ds = np.load(coords) if str(coords).endswith(".npz") else _nc(coords)
    fx, fy, tx = _2d(ds["glamf"]), _2d(ds["gphif"]), _2d(ds["glamt"])
    m = np.load(mask_file) if str(mask_file).endswith(".npz") else _nc(mask_file or coords)
    for v in ([mask_var] if mask_var else ["tmask", "top_level", "bottom_level", "Bathymetry"]):
        if v in (m.files if hasattr(m, "files") else m.variables):
            wet = _2d(m[v]) > 0
            break
    halo = bool(np.all(np.abs((tx[:, 0] - tx[:, -2] + 180.0) % 360.0 - 180.0) < 1e-4))
    lon = np.sort(np.mod(tx[tx.shape[0] // 2], 360.0))
    regional = np.max(np.diff(np.r_[lon, lon[0] + 360.0])) > 20.0      # a gap in longitude: not a global row
    if halo or regional:              # regional or haloed global: drop column 0
        return _cells_from_nodes(fx, fy, wet[1:, 1:])
    g = _cells_from_nodes(np.concatenate([fx[:, -1:], fx], axis=1), np.concatenate([fy[:, -1:], fy], axis=1),
                          wet[1:, :])
    return g


def read_cmip6(path, mask_file=None, drop_rows=0):
    """CMIP6 ocean fx file on the native grid (gn): per-cell vertices (vertices_*, *_bnds or bounds_nav_*).
    Wet: sftof > 0 (sea area %), else a valid deptho / areacello value. drop_rows: leading rows to drop (MPIOM TP04:
    rows 0-1 are halo copies of the fold whose vertices only partly coincide with the rows they copy)."""
    ds = _nc(path)
    v = ds.variables
    for a, b in (("vertices_longitude", "vertices_latitude"), ("lon_bnds", "lat_bnds"),
                 ("bounds_nav_lon", "bounds_nav_lat")):
        if a in v:
            lon, lat = np.asarray(v[a][:], np.float64), np.asarray(v[b][:], np.float64)
            break
    ny, nx = lon.shape[:2]
    m = _nc(mask_file).variables if mask_file else v
    for name in ("sftof", "deptho", "areacello"):
        if name in m:
            x = _2d(m[name])
            fill = getattr(m[name], "_FillValue", 1e20)
            wet = (x > 0) & (x < 0.5 * fill) & np.isfinite(x)
            break
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    ok = np.all(np.abs(lat.reshape(-1, 4)) <= 90.0, axis=1) & (jj.ravel() >= drop_rows)
    return Grid(lon.reshape(-1, 4)[ok], lat.reshape(-1, 4)[ok], wet.ravel()[ok], np.zeros(ok.sum(), int),
                jj.ravel()[ok], ii.ravel()[ok], [(ny, nx)])


def read_scrip(path):
    """SCRIP grid file: grid_corner_lon/lat (cells, 4), grid_imask, grid_dims = (nx, ny), i fastest."""
    ds = _nc(path)
    v = ds.variables
    f = np.degrees(1.0) if "rad" in v["grid_corner_lat"].units.lower() else 1.0
    nx, ny = (int(a) for a in v["grid_dims"][:])
    n = np.arange(nx * ny)
    return Grid(v["grid_corner_lon"][:] * f, v["grid_corner_lat"][:] * f, v["grid_imask"][:] > 0,
                np.zeros(nx * ny, int), n // nx, n % nx, [(ny, nx)])


def _mds(path):
    """MITgcm .data with its .meta: big-endian array shaped by dimList (slowest first)."""
    meta = Path(path).with_suffix(".meta").read_text()
    dims = [int(a) for a in meta.split("dimList")[1].split("[")[1].split("]")[0].replace(",", " ").split()][::3]
    dt = ">f4" if "float32" in meta else ">f8"
    return np.fromfile(path, dt).reshape(dims[::-1])


def _llc_sizes(n):
    """(ny, nx) of the 5 LLC faces in the compact layout: faces 1-2 (3n, n), cap (n, n), faces 4-5 (n, 3n)."""
    return [(3 * n, n), (3 * n, n), (n, n), (n, 3 * n), (n, 3 * n)]


def read_mitgcm_faces(files, depth, sizes):
    """MITgcm tile/face grid files (mitgrid or grid_cs*.face*.bin): 16 or 18 fields of (ny+1, nx+1) float64
    big-endian (Fortran (nx+1, ny+1)); xG and yG are fields 5 and 6 (corner nodes). Depth (MITgcm .data) gives the
    wet mask in the compact layout: the faces one after another, each a (ny, nx) array."""
    d = _mds(depth).ravel()
    out, off = [], 0
    for b, (f, (ny, nx)) in enumerate(zip(files, sizes)):
        raw = np.fromfile(f, ">f8")
        nf = raw.size // ((ny + 1) * (nx + 1))
        a = raw.reshape(nf, ny + 1, nx + 1)
        wet = d[off: off + ny * nx].reshape(ny, nx) > 0
        off += ny * nx
        out.append(_cells_from_nodes(a[5], a[6], wet, block=b))
    return _concat(out)


def read_llc_compact(xg, yg, depth, n):
    """MITgcm LLC grid as compact XG/YG (SW corner of each cell, no closing row/column). Faces 1-2: (3n, n),
    face 3: (n, n), faces 4-5: data reshaped (n, 3n). The closing corners (east column, north row of each face) are
    extrapolated in 3D and snapped (< 0.45 cell) to the nearest SW corner of any face (a neighbour face's corner);
    the rest (domain edge in the south, cap corners) keep the extrapolated value, copies of one cap corner merged.
    Checked on LLC90 against its exact mitgrid files: same joins and 4 cap corners; the south fold (land) differs by
    2 collapsed cells."""
    X, Y, D = (_mds(p).ravel() for p in (xg, yg, depth))
    allp = lonlat_to_xyz(X, Y)
    tree = cKDTree(allp)
    faces, loose, off = [], [], 0
    for b, (ny, nx) in enumerate(_llc_sizes(n)):
        p = allp[off: off + ny * nx].reshape(ny, nx, 3)
        wet = D[off: off + ny * nx].reshape(ny, nx) > 0
        off += ny * nx
        q = np.empty((ny + 1, nx + 1, 3))
        q[:ny, :nx] = p
        q[:ny, nx] = 2 * p[:, -1] - p[:, -2]
        q[ny, :nx] = 2 * p[-1] - p[-2]
        q[ny, nx] = 2 * q[ny - 1, nx] - q[ny - 2, nx]
        edge = np.concatenate([q[:ny, nx], q[ny, :]])
        edge /= np.linalg.norm(edge, axis=1, keepdims=True)
        dist, idx = tree.query(edge)
        # local spacing of each closing corner: the last cell edge along its extrapolation direction
        sp_e = np.linalg.norm(p[:, -1] - p[:, -2], axis=-1)
        sp_n = np.linalg.norm(p[-1] - p[-2], axis=-1)
        spacing = np.r_[sp_e, sp_n, sp_n[-1]]
        snap = dist < 0.45 * spacing
        edge = np.where(snap[:, None], allp[idx], edge)
        q[:ny, nx], q[ny, :] = edge[:ny], edge[ny:]
        faces.append((q, wet))
        lid = np.concatenate([np.c_[np.arange(ny), np.full(ny, nx)], np.c_[np.full(nx + 1, ny), np.arange(nx + 1)]])
        # face corners (cube corners of the cap): extrapolated from two sides, 0.6-0.7 cell apart between faces
        rad = np.where(((lid[:, 0] == ny) & (lid[:, 1] == 0)) | ((lid[:, 0] == 0) & (lid[:, 1] == nx))
                       | ((lid[:, 0] == ny) & (lid[:, 1] == nx)), 0.8, 0.45) * spacing
        loose.append((lid[~snap], rad[~snap]))
    # unsnapped closing corners: the domain's south edge (unique) and the cube corners of the cap (one copy per
    # face, each extrapolated slightly differently): clusters closer than the radius (0.45 cell, 0.8 cell at face
    # corners) take their mean
    pts = np.concatenate([q[a[:, 0], a[:, 1]] for (q, _), (a, _) in zip(faces, loose)])
    sp = np.concatenate([x for _, x in loose])
    nb = cKDTree(pts).query_ball_point(pts, sp)
    rows = np.repeat(np.arange(len(pts)), [len(x) for x in nb])
    cols = np.concatenate([np.asarray(x, int) for x in nb]) if len(nb) else np.zeros(0, int)
    lab = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(pts),) * 2),
                               directed=False)[1]
    mean = np.zeros((lab.max() + 1, 3))
    np.add.at(mean, lab, pts)
    pts = mean[lab]
    out, k = [], 0
    for b, ((q, wet), (a, _)) in enumerate(zip(faces, loose)):
        q[a[:, 0], a[:, 1]] = pts[k: k + len(a)]
        k += len(a)
        x = np.degrees(np.arctan2(q[..., 1], q[..., 0]))
        y = np.degrees(np.arcsin(np.clip(q[..., 2] / np.linalg.norm(q, axis=-1), -1, 1)))
        out.append(_cells_from_nodes(x, y, wet, block=b))
    return _concat(out)


def read_hycom(q_a, depth_a, idm=4500, jdm=3298):
    """HYCOM grid: q points are the SW corners of the p cells. q_a holds the qlon and qlat fields of regional.grid.a
    (fields 3-4 of 19, big-endian float32, each padded to a multiple of 4096 values; fetched by an HTTP range
    request). The east column wraps periodically; the last row (the Arctic fold copy) has no north corners and is
    dropped. Wet: depth .a > 0 and below the 2^100 land flag."""
    n = idm * jdm
    pad = (n + 4095) // 4096 * 4096
    q = np.fromfile(q_a, ">f4").reshape(2, pad)[:, :n].reshape(2, jdm, idm).astype(np.float64)
    d = np.fromfile(depth_a, ">f4", count=n).reshape(jdm, idm)
    wet = (d > 0) & (d < 1e20)
    return _cells_from_nodes(q[0], q[1], wet[:-1], wrap_i=True)


def read_roms(path):
    """ROMS grid: psi points (lon_psi, lat_psi) are the corners of the interior rho cells (1..L-1, 1..M-1); a file
    without psi points (history output) gets them as the normalised 3D mean of the 4 surrounding rho points."""
    ds = _nc(path)
    v = ds.variables
    wet = _2d(v["mask_rho"])[1:-1, 1:-1] > 0
    if "lon_psi" in v:
        return _cells_from_nodes(_2d(v["lon_psi"]), _2d(v["lat_psi"]), wet)
    r = lonlat_to_xyz(_2d(v["lon_rho"]), _2d(v["lat_rho"]))
    q = r[:-1, :-1] + r[1:, :-1] + r[:-1, 1:] + r[1:, 1:]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return _cells_from_nodes(np.degrees(np.arctan2(q[..., 1], q[..., 0])), np.degrees(np.arcsin(q[..., 2])), wet)


def read_delft3d(grds, utm_epsg):
    """Delft3D-FLOW .grd files (RGFGRID), one block each (domain decomposition): ETA rows of x, then of y; the
    missing value marks inactive nodes; every cell with 4 active nodes is wet. Cartesian UTM coordinates go to the
    sphere through pyproj (EPSG code utm_epsg)."""
    return _concat([_read_grd(f, b, utm_epsg) for b, f in enumerate(grds)])


def _read_grd(grd, block, utm_epsg):
    txt = Path(grd).read_text(errors="replace").splitlines()
    spherical = any("spherical" in s.lower() for s in txt[:10])
    miss = next((float(s.split("=")[1]) for s in txt[:10] if "missing value" in s.lower()), 0.0)
    k = next(n for n, s in enumerate(txt) if not s.startswith("*") and "=" not in s and s.split())
    mmax, nmax = (int(a) for a in txt[k].split()[:2])
    vals = [float(t) for s in txt[k + 2:] for t in s.replace("ETA=", " ").split()[(1 if "ETA=" in s else 0):]]
    a = np.array(vals[: 2 * mmax * nmax]).reshape(2, nmax, mmax)
    x, y = a
    act = ~((x == miss) & (y == miss))
    if not spherical:
        import pyproj
        if (Path(sys.prefix) / "share/proj/proj.db").exists():        # env used without activation
            pyproj.datadir.set_data_dir(str(Path(sys.prefix) / "share/proj"))
        from pyproj import Transformer
        x, y = (np.asarray(a) for a in Transformer.from_crs(utm_epsg, 4326, always_xy=True).transform(x, y))
    g = _cells_from_nodes(x, y, np.ones((nmax - 1, mmax - 1), bool), block=block)
    ok = act[:-1, :-1].ravel() & act[:-1, 1:].ravel() & act[1:, 1:].ravel() & act[1:, :-1].ravel()
    return Grid(g.lon[ok], g.lat[ok], g.wet[ok], g.block[ok], g.j[ok], g.i[ok], g.shapes)


# ------------------------------------------------------------------------------------------------- mesh building

@dataclass
class GridMesh:
    mesh: Mesh                 # tris first, then quads (global element order)
    wet: np.ndarray            # (T+Q,) bool
    block: np.ndarray          # (T+Q,)
    j: np.ndarray
    i: np.ndarray
    shapes: list
    n_cells_in: int            # cells read (with halos and fold copies)
    n_duplicates: int          # cells dropped as copies of another cell
    n_collapsed: int           # cells dropped with fewer than 3 distinct corners
    tol_m: float               # corner merge tolerance


def _merge(xyz: np.ndarray, tol: float) -> np.ndarray:
    """Labels 0..U-1 of points closer than ~tol (unit sphere): grid quantisation, then KD-tree pairs across bins."""
    k = np.floor(xyz / tol).astype(np.int64)
    order = np.lexsort(k.T[::-1])
    ks = k[order]
    new = np.r_[True, np.any(ks[1:] != ks[:-1], axis=1)]
    lab = np.empty(len(xyz), np.int64)
    lab[order] = np.cumsum(new) - 1
    rep = order[new]
    pairs = cKDTree(xyz[rep]).query_pairs(tol, output_type="ndarray")
    u = len(rep)
    if len(pairs):
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(u, u))
        lab = connected_components(g, directed=False)[1][lab]
    return lab


def to_mesh(g: Grid) -> GridMesh:
    """All cells of the grid (land included) as one mesh; the wet flag per element selects the ocean mesh."""
    m = len(g.lon)
    p = lonlat_to_xyz(g.lon, g.lat)                                       # (M,4,3)
    e = np.linalg.norm(p - np.roll(p, -1, axis=1), axis=-1)
    tol = 1e-3 * float(np.median(e[e > 0]))
    lab = _merge(p.reshape(-1, 3), tol).reshape(m, 4)
    # CCW corner order: sort by angle about the corner mean in the tangent plane
    q = tangent_coords(p)
    ang = np.arctan2(q[..., 1] - q[..., 1].mean(1, keepdims=True), q[..., 0] - q[..., 0].mean(1, keepdims=True))
    o = np.argsort(ang, axis=1)
    lab = np.take_along_axis(lab, o, axis=1)
    s = np.sort(lab, axis=1)
    ndist = 1 + (s[:, 1:] != s[:, :-1]).sum(axis=1)
    keep = ndist >= 3
    n_collapsed = int((~keep).sum())
    # duplicates (halo columns, fold copies): same sorted corner set; keep the first, wet if any copy is wet
    idx = np.flatnonzero(keep)
    so = idx[np.lexsort(s[idx].T[::-1])]
    new = np.r_[True, np.any(s[so[1:]] != s[so[:-1]], axis=1)]
    grp = np.cumsum(new) - 1
    wet_any = np.zeros(int(grp[-1]) + 1, bool)
    np.logical_or.at(wet_any, grp, g.wet[so])
    cells = so[new]
    n_dup = int(len(idx) - len(cells))
    lab, wet = lab[cells], wet_any
    tri = ndist[cells] == 3
    # triangles: drop the repeated corner, keep the cyclic order
    t = lab[tri]
    rep = t == np.roll(t, 1, axis=1)
    tris = t[~rep].reshape(-1, 3) if len(t) else np.zeros((0, 3), np.int64)
    quads = lab[~tri]
    used, inv_n = np.unique(np.concatenate([tris.ravel(), quads.ravel()]), return_inverse=True)
    pts = np.zeros((lab.max() + 1, 3))
    pts[lab.ravel()] = p.reshape(-1, 3)[(cells[:, None] * 4 + o[cells]).ravel()]
    pts = pts[used]
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    tris, quads = inv_n[: tris.size].reshape(-1, 3), inv_n[tris.size:].reshape(-1, 4)
    order = np.r_[np.flatnonzero(tri), np.flatnonzero(~tri)]
    c = cells[order]
    mesh = Mesh(pts, tris.astype(np.int64), quads.astype(np.int64), np.zeros(len(pts), np.int8))
    return GridMesh(mesh, wet[order], g.block[c], g.j[c], g.i[c], g.shapes, m, n_dup, n_collapsed,
                    tol * 6_371_000.0)


def submesh(gm: GridMesh, sel: np.ndarray) -> tuple[Mesh, np.ndarray]:
    """Elements `sel` (bool over T+Q) as their own mesh; returns the mesh and the element ids kept."""
    T = gm.mesh.n_tris
    st, sq = sel[:T], sel[T:]
    tris, quads = gm.mesh.tris[st], gm.mesh.quads[sq]
    used, inv = np.unique(np.concatenate([tris.ravel(), quads.ravel()]), return_inverse=True)
    m = Mesh(gm.mesh.points[used], inv[: tris.size].reshape(-1, 3), inv[tris.size:].reshape(-1, 4),
             np.zeros(len(used), np.int8))
    return m, np.r_[np.flatnonzero(st), T + np.flatnonzero(sq)]


# ------------------------------------------------------------------------------------------------------ registry

D = DATA
SETTE = "https://gws-access.jasmin.ac.uk/public/nemo/sette_inputs/r5.0.0"
CEDA = "https://esgf.ceda.ac.uk/thredds/fileServer/esg_cmip6/CMIP6/CMIP"
SCRIP = "https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/share/scripgrids"
CMIP6_SA = "CC BY-SA 4.0 (file attribute `license`)"
NONE = "none stated"
RJ = D / "delft3d_rongjiang/unzipped/Grids"


def _g(read, family, config, res, layout, url, version, licence, downloaded="2026-10-22"):
    return dict(read=read, family=family, config=config, res=res, layout=layout, url=url, version=version,
                licence=licence, downloaded=downloaded)


GRIDS = {
    "nemo_orca2": _g(lambda: read_nemo(D / "nemo_orca2/ORCA_R2_zps_domcfg.nc"), "NEMO",
                     "ORCA2 (ORCA_R2, NEMO 5.0 reference config)", "2°", "tripolar (T-pivot fold), no halo",
                     f"{SETTE}/ORCA2_ICE_v5.0.0/ORCA_R2_zps_domcfg.nc", "NEMO 5.0.0 SETTE inputs, file of 2024-07-04",
                     NONE + " (NEMO code: CeCILL)"),
    "nemo_eorca1": _g(lambda: read_nemo(D / "nemo_eorca1/grid_mask_eORCA1-GO6.nc"), "NEMO",
                      "eORCA1 v2.2x (Met Office GO6)", "1°", "tripolar (F-pivot fold), halos",
                      "https://zenodo.org/records/22010172", "doi:10.5281/zenodo.22010172 (GLOBAL_BATHYMETRY v1.1.0)",
                      NONE),
    "nemo_eorca025": _g(lambda: read_nemo(D / "nemo_eorca025/grid_mask_eORCA025-GO6.nc"), "NEMO",
                        "eORCA025 (Met Office GO6)", "1/4°", "tripolar (T-pivot fold), halos",
                        "https://zenodo.org/records/15494369", "doi:10.5281/zenodo.15494369 (GLOBAL_BATHYMETRY v1.0.0)",
                        NONE),
    "nemo_eorca12": _g(lambda: read_nemo(D / "nemo_eorca12/grid_mask_eORCA12-GO6.nc"), "NEMO",
                       "eORCA12 (Met Office GO6)", "1/12°", "tripolar (T-pivot fold), halos",
                       "https://zenodo.org/records/15495870", "doi:10.5281/zenodo.15495870 (GLOBAL_BATHYMETRY v1.0.0)",
                       NONE),
    "nemo_amm12": _g(lambda: read_nemo(D / "nemo_amm12/AMM_R12_sco_domcfg.nc"), "NEMO regional",
                     "AMM12 (NW European shelf, NEMO 5.0 reference config)", "12 km", "rotated pole",
                     f"{SETTE}/AMM12_v5.0.0/AMM_R12_sco_domcfg.nc", "NEMO 5.0.0 SETTE inputs, file of 2024-07-04",
                     NONE + " (NEMO code: CeCILL)"),
    "nemo_amm15": _g(lambda: read_nemo(D / "nemo_amm15/amm15.coordinates.nc",
                                       D / "nemo_amm15/amm15.bathydepth.co7.cs3x.cs20.nc", "Bathymetry"),
                     "NEMO regional", "AMM15 (NW European shelf, CO7/CO9)", "1.5 km", "rotated pole",
                     "https://gws-access.jasmin.ac.uk/public/jmmp/AMM15/COORDINATES/amm15.coordinates.nc",
                     "JMMP CO_AMM15 inputs (github.com/JMMP-Group/CO_AMM15), file of 2023-01-16; mask from "
                     "AMM15_BATHY/...amm15.bathydepth.co7.cs3x.cs20.nc", NONE),
    "nemo_bal_mfc": _g(lambda: read_nemo(D / "nemo_bal_mfc/domain_cfg_2d_extract.npz",
                                         D / "nemo_bal_mfc/domain_cfg_2d_extract.npz", "bottom_level"),
                       "NEMO regional", "Baltic MFC EIS-202411 (NEMO-Nordic, Baltic + North Sea)", "1 nmi",
                       "regular lon-lat", "https://zenodo.org/records/14507734",
                       "doi:10.5281/zenodo.14507734 (domain_cfg.nc, 4.3 GB; 2D fields read by HTTP ranges, "
                       "scripts/grids/cdf_range.py)", "CC BY 4.0"),
    "mitgcm_cs32": _g(lambda: read_mitgcm_faces([D / f"mitgcm_cs32/grid_cs32.face00{k}.bin" for k in range(1, 7)],
                                                D / "mitgcm_cs32/Depth.data", [(32, 32)] * 6),
                      "MITgcm", "CS32 cubed sphere (MITgcm/ECCO unit test)", "2.8°", "cubed sphere, 6 faces",
                      "https://github.com/gaelforget/GRID_CS32", "doi:10.5281/zenodo.3966408",
                      NONE + " (Zenodo: other-open)"),
    "mitgcm_llc90": _g(lambda: read_mitgcm_faces([D / f"mitgcm_llc90/tile00{k}.mitgrid" for k in range(1, 6)],
                                                 D / "mitgcm_llc90/Depth.data", _llc_sizes(90)),
                       "MITgcm", "LLC90 (ECCO v4)", "1°", "lat-lon-cap, 5 faces",
                       "https://github.com/gaelforget/GRID_LLC90", "doi:10.5281/zenodo.3966420",
                       NONE + " (Zenodo: other-open)"),
    "mitgcm_llc270": _g(lambda: read_llc_compact(D / "mitgcm_llc270/XG.data", D / "mitgcm_llc270/YG.data",
                                                 D / "mitgcm_llc270/Depth.data", 270),
                        "MITgcm", "LLC270 (ECCO)", "1/3°", "lat-lon-cap, 5 faces",
                        "https://data.nas.nasa.gov/ecco/llc_270/grid/", "NAS ECCO portal, files of 2025-06-27",
                        NONE + " (NASA ECCO data portal)"),
    "mom6_om4p5": _g(lambda: read_cmip6(D / "mom6_om4p5/areacello_Ofx_GFDL-ESM4_piControl_r1i1p1f1_gn.nc",
                                        D / "mom6_om4p5/deptho_Ofx_GFDL-ESM4_historical_r1i1p1f1_gn.nc"),
                     "MOM6", "OM4p5 (GFDL-ESM4, CMIP6)", "1/2°", "tripolar",
                     f"{CEDA}/NOAA-GFDL/GFDL-ESM4/piControl/r1i1p1f1/Ofx/areacello/gn/v20180701/",
                     "CMIP6 GFDL-ESM4 Ofx areacello v20180701 + deptho (historical) v20190726", CMIP6_SA),
    "mom6_om4p25": _g(lambda: read_cmip6(D / "mom6_om4p25/sftof_Ofx_GFDL-CM4_piControl_r1i1p1f1_gn.nc"),
                      "MOM6", "OM4p25 (GFDL-CM4, CMIP6)", "1/4°", "tripolar",
                      f"{CEDA}/NOAA-GFDL/GFDL-CM4/piControl/r1i1p1f1/Ofx/sftof/gn/v20180701/",
                      "CMIP6 GFDL-CM4 Ofx sftof v20180701", CMIP6_SA),
    "pop_gx3v7": _g(lambda: read_scrip(D / "pop_gx3v7/gx3v7_120309.nc"), "POP", "gx3v7 (CESM)", "3°",
                    "displaced pole (Greenland)", f"{SCRIP}/gx3v7_120309.nc", "CESM inputdata SCRIP file of 2012-03-09",
                    NONE + " (CESM inputdata)"),
    "pop_gx1v6": _g(lambda: read_scrip(D / "pop_gx1v6/gx1v6_090205.nc"), "POP", "gx1v6 (CESM1)", "1°",
                    "displaced pole (Greenland)", f"{SCRIP}/gx1v6_090205.nc", "CESM inputdata SCRIP file of 2009-02-05",
                    NONE + " (CESM inputdata)"),
    "pop_tx0.1v3": _g(lambda: read_scrip(D / "pop_tx0.1v3/tx0.1v3_170728.nc"), "POP", "tx0.1v3 (CESM)", "1/10°",
                      "tripolar", f"{SCRIP}/tx0.1v3_170728.nc",
                      "CESM inputdata SCRIP file of 2017-07-28 (title 'Pop tripole x0.1 v2')", NONE + " (CESM inputdata)"),
    "mpiom_gr15": _g(lambda: read_cmip6(D / "mpiom_gr15/sftof_Ofx_MPI-ESM1-2-LR_piControl_r1i1p1f1_gn.nc"),
                     "MPIOM", "GR15 (MPI-ESM1.2-LR, CMIP6)", "1.5°", "dipolar (poles on Greenland, Antarctica)",
                     f"{CEDA}/MPI-M/MPI-ESM1-2-LR/piControl/r1i1p1f1/Ofx/sftof/gn/v20190710/",
                     "CMIP6 MPI-ESM1-2-LR Ofx sftof v20190710", CMIP6_SA),
    "mpiom_tp04": _g(lambda: read_cmip6(D / "mpiom_tp04/sftof_Ofx_MPI-ESM1-2-HR_piControl_r1i1p1f1_gn.nc",
                                        drop_rows=2),
                     "MPIOM", "TP04 (MPI-ESM1.2-HR, CMIP6)", "0.4°", "tripolar",
                     f"{CEDA}/MPI-M/MPI-ESM1-2-HR/piControl/r1i1p1f1/Ofx/sftof/gn/v20190710/",
                     "CMIP6 MPI-ESM1-2-HR Ofx sftof v20190710", CMIP6_SA),
    "hycom_glbb0.08": _g(lambda: read_hycom(D / "hycom_glbb0.08/regional.grid.qlon_qlat.a",
                                            D / "hycom_glbb0.08/depth_GLBb0.08_09m11.a"),
                         "HYCOM", "GLBb0.08 (global, expt 93.0)", "1/12°", "tripolar (Mercator + bipolar Arctic)",
                         "https://data.hycom.org/datasets/GLBb0.08/expt_93.0/topo/",
                         "regional.grid.a fields qlon, qlat (HTTP range) + depth_GLBb0.08_09m11.a/.b",
                         NONE + " (HYCOM consortium data server)"),
    "roms_txla": _g(lambda: read_roms(D / "roms_txla/ROMS_example_full_grid.nc"), "ROMS",
                    "TXLA (Texas-Louisiana shelf; xroms example output)", "0.5-3 km", "boundary-fitted curvilinear",
                    "https://github.com/xoceanmodel/xroms/raw/main/xroms/data/ROMS_example_full_grid.nc",
                    "xroms main, 2026-10 (grid txla_grd_v4_new)", "MIT (xroms repository)"),
    "roms_wc13": _g(lambda: read_roms(D / "roms_wc13/wc13_grd.nc"), "ROMS", "WC13 (US West Coast, ROMS test case)",
                    "30 km", "lon-lat (Mercator-like)",
                    "https://github.com/myroms/roms_test/blob/main/WC13/Data/wc13_grd.nc", "roms_test main (Git LFS)",
                    NONE + " (ROMS code: MIT/X)"),
    "roms_usec3km": _g(lambda: read_roms(D / "roms_usec3km/usec3km_roms_grd.nc4"), "ROMS",
                       "USEC 3 km (US East Coast, ROMS test case)", "3 km", "rotated, coast-parallel",
                       "https://github.com/myroms/roms_test/blob/main/USEC/Data/GRD/usec3km_roms_grd.nc4",
                       "roms_test main (Git LFS)", NONE + " (ROMS code: MIT/X)"),
    "delft3d_rongjiang": _g(lambda: read_delft3d([RJ / f"{k}.grd" for k in ("2d-bei", "2d-nan", "2d-shui",
                                                                            "3d-junction")], "EPSG:32650"),
                            "Delft3D-FLOW", "Rongjiang estuary (4 domains, DD coupling)", "15-60 m",
                            "boundary-fitted, 4 blocks", "https://zenodo.org/records/8420090",
                            "doi:10.5281/zenodo.8420090", "CC BY 4.0"),
    "delft3d_caernarvon": _g(lambda: read_delft3d(
        [D / "delft3d_caernarvon/unzipped/Run-31-Shared/BRE_C_cut5_div.grd"], "EPSG:32615"),
        "Delft3D-FLOW", "Caernarvon crevasse 1927, Breton Sound (Louisiana)", "~50 m", "boundary-fitted",
        "https://zenodo.org/records/12639858", "doi:10.5281/zenodo.12639858", "CC BY 4.0"),
}


def read(name: str) -> GridMesh:
    return to_mesh(GRIDS[name]["read"]())


if __name__ == "__main__":
    gm = read(sys.argv[1])
    print(f"{sys.argv[1]}: {gm.n_cells_in} cells read, {gm.mesh.n_quads} quads + {gm.mesh.n_tris} tris, "
          f"{gm.wet.sum()} wet, {gm.n_duplicates} duplicates, {gm.n_collapsed} collapsed, tol {gm.tol_m:.2f} m")
