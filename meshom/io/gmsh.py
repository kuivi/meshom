"""Gmsh MSH 4.1 ASCII reader (the Gmsh reference manual, gmsh.info, "MSH file format").

Node x, y are metres in a stereographic plane centred at (lon0, lat0); z is ignored.
"""
from pathlib import Path

import numpy as np

from meshom.geo import stereographic_inverse
from meshom.mesh import Mesh, orient_ccw

# nodes per element type (the manual's "MSH file format", element types); only 2 (triangle) and 3 (quad) are kept
_NNODES = {1: 2, 2: 3, 3: 4, 4: 4, 5: 8, 6: 6, 7: 5, 8: 3, 9: 6, 10: 9, 11: 10, 12: 27, 13: 18, 14: 14, 15: 1,
           16: 8, 17: 20, 18: 15, 19: 13, 20: 9, 21: 10, 22: 12, 23: 15, 24: 15, 25: 21, 26: 4, 27: 5, 28: 6,
           29: 20, 30: 35, 31: 56, 92: 64, 93: 125}


def _section(text: str, name: str) -> str:
    i = text.index("\n", text.index(f"${name}")) + 1
    return text[i:text.index(f"$End{name}", i)]


def read_gmsh41(path: Path, lon0_deg: float, lat0_deg: float) -> Mesh:
    """Triangles (type 2) and quads (type 3) of dimension 2; nodes used by no 2-D element are dropped."""
    text = Path(path).read_text()

    v = np.array(_section(text, "Nodes").split(), dtype=np.float64)    # tags < 2**53, exact in float64
    nblocks, i = int(v[0]), 4
    tags, xy = [], []
    for _ in range(nblocks):
        dim, par, n = int(v[i]), int(v[i + 2]), int(v[i + 3])
        i += 4
        tags.append(v[i:i + n].astype(np.int64))
        i += n
        w = 3 + (dim if par else 0)
        xy.append(v[i:i + n * w].reshape(n, w)[:, :2])
        i += n * w
    tags, xy = np.concatenate(tags), np.vstack(xy)

    v = np.array(_section(text, "Elements").split(), dtype=np.int64)
    nblocks, i = int(v[0]), 4
    cells = {2: [], 3: []}
    for _ in range(nblocks):
        dim, typ, n = int(v[i]), int(v[i + 2]), int(v[i + 3])
        i += 4
        w = 1 + _NNODES[typ]
        if dim == 2 and typ in cells:
            cells[typ].append(v[i:i + n * w].reshape(n, w)[:, 1:])
        i += n * w
    tri = np.vstack(cells[2]) if cells[2] else np.zeros((0, 3), dtype=np.int64)
    quad = np.vstack(cells[3]) if cells[3] else np.zeros((0, 4), dtype=np.int64)

    order = np.argsort(tags)
    row = order[np.searchsorted(tags[order], np.concatenate([tri.ravel(), quad.ravel()]))]
    used, inv = np.unique(row, return_inverse=True)
    pts = stereographic_inverse(xy[used, 0], xy[used, 1], lon0_deg, lat0_deg)
    tris, quads = inv[:tri.size].reshape(-1, 3), inv[tri.size:].reshape(-1, 4)
    return orient_ccw(Mesh(pts, tris.astype(np.int64), quads.astype(np.int64), np.zeros(len(pts), dtype=np.int8)))
