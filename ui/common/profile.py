"""Profiles along a line on the sphere, for the page's Tools window (ruler and profile): points spaced evenly along the
great-circle legs of a polyline, the land mask of the UI coast and the bottom elevation of a bathymetry grid there.

    lon, lat, d_m = sample_line([[lon, lat], ...], n)
    z = elevation_at(path, lon, lat)        # a regular lon-lat netCDF grid (GEBCO 2024): positive up, ocean < 0
    nd = line_nodes(mv, line)               # the nodes of the mesh's elements the line crosses
"""
from pathlib import Path

import numpy as np

from meshom.geo import R_EARTH_M, lonlat_to_xyz, xyz_to_lonlat

LAND_STEP_DEG = 0.05                       # the land raster of a coast along a profile (~5 km at the equator)
_LAND: dict = {}


def sample_line(line, n: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lon, lat, distance from the first vertex in metres) of n points evenly spaced along the polyline's
    great-circle legs, both ends included."""
    p = np.asarray(line, dtype=np.float64)
    v = lonlat_to_xyz(p[:, 0], p[:, 1])
    w = np.arccos(np.clip(np.einsum("ij,ij->i", v[:-1], v[1:]), -1.0, 1.0))      # leg angles, radians
    cum = np.r_[0.0, np.cumsum(w)]
    s = np.linspace(0.0, cum[-1], max(int(n), 2))
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(w) - 1)
    t = np.where(w[k] > 0, (s - cum[k]) / np.where(w[k] > 0, w[k], 1.0), 0.0)
    a, b, wk = v[k], v[k + 1], w[k]
    sw = np.sin(wk)
    flat = sw < 1e-12                                                               # a zero-length leg
    sw = np.where(flat, 1.0, sw)
    fa = np.where(flat, 1.0 - t, np.sin((1.0 - t) * wk) / sw)[:, None]
    fb = np.where(flat, t, np.sin(t * wk) / sw)[:, None]
    x = fa * a + fb * b
    x /= np.linalg.norm(x, axis=1)[:, None]
    lon, lat = xyz_to_lonlat(x)
    return lon, lat, s * R_EARTH_M


def land_at(coast, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """True on land (the UI coast's polygons on a LAND_STEP_DEG raster, built once per coast)."""
    if coast is None:
        return np.zeros(len(lon), bool)
    key = id(coast)
    if key not in _LAND:
        gx = -180.0 + (np.arange(int(360 / LAND_STEP_DEG)) + 0.5) * LAND_STEP_DEG
        gy = -90.0 + (np.arange(int(180 / LAND_STEP_DEG)) + 0.5) * LAND_STEP_DEG
        _LAND.clear()
        _LAND[key] = (coast, coast.land_mask(gx, gy))                 # coast kept: its id stays its own
    m = _LAND[key][1]
    i = np.clip(((np.asarray(lat) + 90.0) / LAND_STEP_DEG).astype(int), 0, m.shape[0] - 1)
    j = np.clip(((np.asarray(lon) + 180.0) % 360.0 / LAND_STEP_DEG).astype(int), 0, m.shape[1] - 1)
    return m[i, j]


def elevation_at(path, lon: np.ndarray, lat: np.ndarray, variable: str = "elevation") -> np.ndarray:
    """The grid's value at the cell nearest each point (the grid's two dimensions are latitude and longitude, cell
    centres, as meshom.bathymetry reads it); read point by point, never whole (GEBCO 2024 is 7.5 GB)."""
    import netCDF4

    with netCDF4.Dataset(Path(path)) as ds:
        var = ds[variable]
        var.set_auto_maskandscale(False)
        glat = np.asarray(ds[var.dimensions[0]][:], dtype=np.float64)
        glon = np.asarray(ds[var.dimensions[1]][:], dtype=np.float64)
        ny, nx = len(glat), len(glon)
        dlat, dlon = (glat[-1] - glat[0]) / (ny - 1), (glon[-1] - glon[0]) / (nx - 1)
        ci = np.clip(np.rint((np.asarray(lat) - glat[0]) / dlat).astype(np.int64), 0, ny - 1)
        cj = np.rint((np.asarray(lon) - glon[0]) / dlon).astype(np.int64) % nx
        return np.array([float(var[i, j]) for i, j in zip(ci, cj)])


NODE_SAMPLES_MAX = 20000                  # line_nodes: at most this many points along the line


def line_nodes(mv, line) -> dict:
    """The nodes of the elements of mesh view `mv` that the polyline crosses: the line sampled at a third of the mesh's
    small cells (1st percentile of size_km; at most NODE_SAMPLES_MAX points), the element holding each sample (of the
    8 with the nearest centroids, the one whose great-circle edges all have the sample on their left: elements are
    counter-clockwise seen from outside), its nodes. Returns {"id", "d_m" (along the line, to the nearest sample),
    "off_m" (from that sample), "lon", "lat"} as arrays sorted by d_m."""
    from scipy.spatial import cKDTree

    p = np.asarray(line, dtype=np.float64)
    v = lonlat_to_xyz(p[:, 0], p[:, 1])
    length = R_EARTH_M * np.arccos(np.clip(np.einsum("ij,ij->i", v[:-1], v[1:]), -1.0, 1.0)).sum()
    step = max(length / NODE_SAMPLES_MAX, np.percentile(mv.fields["size_km"], 1) * 1000.0 / 3.0)
    lon, lat, d = sample_line(line, int(np.ceil(length / step)) + 1)
    x = lonlat_to_xyz(lon, lat)
    tree, _ = mv._cover
    k = min(8, mv.n_elems)
    _, cand = tree.query(x, k=k)
    cand = cand.reshape(len(x), k)
    T = len(mv.tris)
    corners = np.empty((mv.n_elems, 4), np.int64)                     # a triangle repeats its first node
    corners[:T, :3], corners[:T, 3] = mv.tris, mv.tris[:, 0]
    corners[T:] = mv.quads
    P = mv.xyz.astype(np.float64)[corners[cand]]                      # (S, k, 4, 3)
    side = np.einsum("skij,sj->ski", np.cross(P, np.roll(P, -1, axis=2)), x)
    inside = (side >= -1e-12).all(axis=2)
    hit = inside.any(axis=1)
    elems = np.unique(cand[hit, inside[hit].argmax(axis=1)])
    ids = np.unique(corners[elems])
    nx = mv.xyz[ids].astype(np.float64)
    off, near = cKDTree(x).query(nx)
    order = np.argsort(d[near], kind="stable")
    nlon, nlat = xyz_to_lonlat(nx[order])
    return {"id": ids[order], "d_m": d[near][order], "off_m": 2.0 * R_EARTH_M * np.arcsin(np.clip(off[order] / 2.0, 0.0, 1.0)),
            "lon": nlon, "lat": nlat}
