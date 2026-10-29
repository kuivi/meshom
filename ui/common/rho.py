"""rho_c of the configured h on the preview grid: where a quad mesh following h must have irregular nodes.

h is the config's size field as the preview shows it (scenario.h_field: the core's layers, clip and scale, and the
core's gradient limiter when the page asks for the limited h).

rho_c = (2/pi)(K + Laplace ln h) (meshom.compete). It is evaluated with compete's own discretisation
(`_node_rho`: spherical dual area + cotangent Laplacian of ln h per node) on a Delaunay triangulation of Fibonacci points
with the density of the preview grid at the equator, then shown on the lon-lat grid by nearest point. The map shows
rho_c h^2 = irregular spots per cell of size h (> 0: valence-3 nodes or triangles, < 0: valence-5 nodes); a block of
square cells fits where it is ~0. The integrals (cone units) are taken over the water points only and leave out the
coast flux term of compete.rho_c_integrals (the preview has no coast boundary), so they are indicative.
"""
import time

import numpy as np
from scipy.spatial import cKDTree

from meshom.compete import _node_rho
from meshom.delaunay import sphere_delaunay
from meshom.geo import R_EARTH_M, lonlat_to_xyz, xyz_to_lonlat
from ui.common.scenario import grid


def fibonacci(m: int) -> np.ndarray:
    i = np.arange(m) + 0.5
    z = 1.0 - 2.0 * i / m
    ph = np.pi * (1.0 + 5**0.5) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(ph), r * np.sin(ph), z])


def rho_preview(h, coast, step_deg: float) -> dict:
    """h: a SizeField; coast: the UI coast (land mask) or None (no land)."""
    t0 = time.perf_counter()
    m = int(4 * np.pi / np.radians(step_deg) ** 2)
    pts = fibonacci(m)
    hv = h(pts)
    tris = sphere_delaunay(pts)
    area, lap = _node_rho(pts, tris, np.zeros((0, 2), np.int64), np.log(hv))
    rho = (2.0 / np.pi) * (area + lap)                       # cone units per dual cell
    per_cell = rho / area * (hv / R_EARTH_M) ** 2             # rho_c h^2
    lon, lat = grid(step_deg)
    water_grid = ~coast.land_mask(lon, lat) if coast is not None else np.ones((len(lat), len(lon)), bool)
    plon, plat = xyz_to_lonlat(pts)
    i = np.clip(((plat + 90.0) / step_deg).astype(int), 0, len(lat) - 1)
    j = np.clip(((plon + 180.0) / step_deg).astype(int), 0, len(lon) - 1)
    wet = water_grid[i, j]
    LON, LAT = np.meshgrid(lon, lat)
    _, near = cKDTree(pts).query(lonlat_to_xyz(LON.ravel(), LAT.ravel()), workers=4)
    field = per_cell[near].reshape(LON.shape)
    v0 = float(np.percentile(np.abs(per_cell[wet]), 95)) if wet.any() else 1.0
    r = rho[wet]
    return {"field": field, "water": water_grid, "v0": max(v0, 1e-6), "n_points": m,
            "plus": float(r[r > 0].sum()), "minus": float(r[r < 0].sum()), "total": float(r.sum()),
            "curv": float((2.0 / np.pi) * area[wet].sum()), "seconds": time.perf_counter() - t0}


def rho_rgba(r: dict) -> np.ndarray:
    """(ny, nx, 4) uint8, north row first: rho_c h^2 through a diverging map, saturating at about 2 v0 (tanh)."""
    from matplotlib import colormaps
    t = 0.5 + 0.5 * np.tanh(r["field"] / (2.0 * r["v0"]))
    rgba = (colormaps["RdBu_r"](t) * 255).astype(np.uint8)
    rgba[~r["water"]] = (60, 60, 60, 255)
    return rgba[::-1]
