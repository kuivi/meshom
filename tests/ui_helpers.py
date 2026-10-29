"""Shared fixtures of the UI prototype tests: a tiny global mesh (lon-lat quads, triangle fans at the poles) that
crosses the antimeridian and touches both poles, saved the way a batch run saves it."""
from pathlib import Path

import numpy as np

from meshom.geo import lonlat_to_xyz
from meshom.mesh import Mesh, orient_ccw



def tiny_mesh(step: float = 20.0) -> Mesh:
    lon = np.arange(-180.0, 180.0, step)
    lat = np.arange(-80.0, 80.0 + 1e-9, step)
    nx, ny = len(lon), len(lat)
    LON, LAT = np.meshgrid(lon, lat)
    pts = np.vstack([lonlat_to_xyz(LON.ravel(), LAT.ravel()), [[0, 0, -1.0], [0, 0, 1.0]]])
    sp, npole = nx * ny, nx * ny + 1
    idx = lambda j, i: j * nx + i % nx  # noqa: E731
    quads = [[idx(j, i), idx(j, i + 1), idx(j + 1, i + 1), idx(j + 1, i)] for j in range(ny - 1) for i in range(nx)]
    tris = [[sp, idx(0, i + 1), idx(0, i)] for i in range(nx)] + \
           [[npole, idx(ny - 1, i), idx(ny - 1, i + 1)] for i in range(nx)]
    m = Mesh(pts, np.array(tris, np.int64), np.array(quads, np.int64), np.zeros(len(pts), np.int8))
    return orient_ccw(m)


def write_tiny_mesh(out: Path) -> Path:
    from ui.common.prepare_mesh import write_overlay
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m = tiny_mesh()
    np.savez(out / "mesh.npz", points=m.points, tris=m.tris, quads=m.quads, node_flag=m.node_flag)
    write_overlay(m, out)
    return out
