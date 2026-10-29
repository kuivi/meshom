# SPDX-License-Identifier: GPL-3.0-or-later
"""Straits of an ORCA mesh measured cell by cell: open or closed, the cells across, the sill (decision B93).

A strait is two water points (its ends, one in each basin) and a lon-lat box around it. On a mesh (any elements:
FESOM3's quads, meshom's mixed elements) the cells are the elements whose centre lies in the box, linked across
shared edges (FESOM3 links cells only across edges: a corner does not connect). Each end is the box cell nearest to its
point (the mesh's cells are all water), at most END_DEG away (else that basin has no water: closed).
- open: the two end cells are linked within the box;
- cells across: the fewest cells whose removal separates the ends (vertex-disjoint paths, Menger; max-flow with each
  cell split into an in and an out node of capacity 1);
- sill (m): the largest d such that a path of cells at least d deep joins the ends (the bottleneck path; a cell's
  depth = the shallowest of its nodes, as FESOM3's element level is the minimum over its vertices);
- the cut's cells: one narrowest section (the cells of a minimum vertex cut), their node depths.
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, maximum_flow

from meshom.geo import lonlat_to_xyz, xyz_to_lonlat

END_DEG = 1.5           # an end's cell lies within this of its point (ORCA1's cells are 1 degree)

# name: (end A lon, lat), (end B lon, lat), box (lon0, lon1, lat0, lat1)
STRAITS = {
    "Gibraltar": ((-7.5, 36.0), (-3.5, 36.2), (-8.5, -2.5, 34.6, 37.6)),
    "Bosphorus": ((28.2, 40.8), (29.9, 41.7), (27.6, 30.6, 40.3, 42.2)),
    "Dardanelles": ((25.4, 39.9), (28.0, 40.75), (24.8, 28.6, 39.4, 41.1)),
    "Bab-el-Mandeb": ((41.8, 14.3), (44.8, 12.2), (41.0, 45.6, 11.3, 15.0)),
    "Danish straits": ((11.3, 57.2), (14.2, 55.1), (9.5, 15.2, 54.0, 58.0)),
    "Bering": ((-171.0, 63.8), (-168.0, 67.8), (-176.0, -162.0, 63.0, 68.6)),
    "Nares Strait": ((-57.0, 82.8), (-72.0, 78.6), (-78.0, -50.0, 78.0, 83.6)),
    "Fury and Hecla Strait": ((-81.3, 69.6), (-87.0, 70.0), (-88.5, -80.0, 68.8, 70.8)),
    "Hell Gate": ((-90.5, 77.4), (-85.0, 76.2), (-93.0, -83.0, 75.7, 78.2)),
    "Nevelskoy Strait": ((140.8, 50.6), (141.7, 53.3), (139.5, 143.2, 49.5, 54.2)),
    "Lombok": ((115.7, -7.9), (115.7, -9.4), (115.2, 116.3, -9.9, -7.4)),
    "Makassar": ((120.5, 3.0), (118.3, -4.6), (115.5, 122.0, -5.5, 4.0)),
    "Ombai": ((125.6, -7.0), (123.6, -9.1), (122.6, 126.6, -9.8, -6.4)),
    "Torres": ((141.0, -10.2), (144.5, -10.0), (140.4, 145.2, -11.6, -8.6)),
}
GROUPS = {"Gibraltar": ["Gibraltar"], "Turkish straits": ["Bosphorus", "Dardanelles"],
          "Bab-el-Mandeb": ["Bab-el-Mandeb"], "Danish straits": ["Danish straits"], "Bering": ["Bering"],
          "Canadian Arctic Archipelago": ["Nares Strait", "Fury and Hecla Strait", "Hell Gate"],
          "Sakhalin": ["Nevelskoy Strait"], "Indonesian throughflow": ["Lombok", "Makassar", "Ombai"],
          "Torres": ["Torres"]}


def cell_links(elems: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of elements (k, 3 or 4 node ids; a triangle may repeat its first node as a 4th) sharing an edge."""
    E = np.asarray(elems)
    k = E.shape[1]
    a, b = E, np.roll(E, -1, axis=1)
    real = a != b
    key = (np.minimum(a, b) * np.int64(n_nodes) + np.maximum(a, b))[real]
    el = np.broadcast_to(np.arange(len(E))[:, None], (len(E), k))[real]
    o = np.argsort(key, kind="stable")
    key, el = key[o], el[o]
    same = key[1:] == key[:-1]
    return el[:-1][same], el[1:][same]


def centres(lon: np.ndarray, lat: np.ndarray, elems: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = lonlat_to_xyz(np.asarray(lon, float), np.asarray(lat, float))
    c = p[elems].mean(axis=1)
    return xyz_to_lonlat(c / np.linalg.norm(c, axis=1, keepdims=True))


def _in_box(lon, lat, box):
    lo0, lo1, la0, la1 = box
    return (((lon - lo0) % 360.0) <= ((lo1 - lo0) % 360.0)) & (lat >= la0) & (lat <= la1)


def _nearest(lon, lat, idx, pt):
    if not len(idx):
        return -1
    p = lonlat_to_xyz(lon[idx], lat[idx])
    q = lonlat_to_xyz(np.array([pt[0]]), np.array([pt[1]]))[0]
    k = int(np.argmax(p @ q))
    return int(idx[k]) if np.degrees(np.arccos(np.clip(p[k] @ q, -1, 1))) <= END_DEG else -1


def measure(name: str, clon: np.ndarray, clat: np.ndarray, links: tuple, cell_depth: np.ndarray | None = None,
            node_depth_of_cell=None) -> dict:
    """One strait (STRAITS[name]) on the cells (centres clon, clat; edge links (a, b)): module docstring.
    node_depth_of_cell(cells) -> list of node depth lists (for the cut's report), optional."""
    A, B, box = STRAITS[name]
    inb = _in_box(clon, clat, box)
    idx = np.flatnonzero(inb)
    rec = {"strait": name, "box_cells": int(len(idx))}
    ca, cb = _nearest(clon, clat, idx, A), _nearest(clon, clat, idx, B)
    if ca < 0 or cb < 0 or ca == cb:
        rec.update(open=False, cells_across=0, sill_m=None, note="no water cell at an end")
        return rec
    loc = np.full(len(clon), -1, np.int64)
    loc[idx] = np.arange(len(idx))
    a, b = links
    ok = inb[a] & inb[b]
    la, lb = loc[a[ok]], loc[b[ok]]
    n = len(idx)
    g = coo_matrix((np.ones(len(la)), (la, lb)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    sa, sb = int(loc[ca]), int(loc[cb])
    rec["end_cells_at"] = [[round(float(clon[c]), 3), round(float(clat[c]), 3)] for c in (ca, cb)]
    if lab[sa] != lab[sb]:
        rec.update(open=False, cells_across=0, sill_m=None)
        return rec
    # max flow, vertex capacities 1 (the ends unbounded): node v -> in 2v, out 2v + 1
    big = 1 << 20
    cap_v = np.ones(n, np.int64)
    cap_v[[sa, sb]] = big
    rows = np.r_[2 * np.arange(n), 2 * la + 1, 2 * lb + 1]
    cols = np.r_[2 * np.arange(n) + 1, 2 * lb, 2 * la]
    lk = np.full(len(la), big)
    lk[((la == sa) & (lb == sb)) | ((la == sb) & (lb == sa))] = 1     # the ends side by side: that link is one path
    caps = np.r_[cap_v, lk, lk]
    G = csr_matrix((caps.astype(np.int32), (rows, cols)), shape=(2 * n, 2 * n))
    res = maximum_flow(G, 2 * sa + 1, 2 * sb)
    rec.update(open=True, cells_across=int(res.flow_value))
    # one minimum cut: the cells whose in node is reachable from the source in the residual graph and out node not
    F = res.flow
    R = (G - F).tocsr()
    R.data = np.where(R.data > 0, 1, 0)
    R.eliminate_zeros()
    from scipy.sparse.csgraph import breadth_first_order

    reach = np.zeros(2 * n, bool)
    reach[breadth_first_order(R, 2 * sa + 1, directed=True, return_predecessors=False)] = True
    cut = np.flatnonzero(reach[2 * np.arange(n)] & ~reach[2 * np.arange(n) + 1])
    cut_cells = idx[cut]
    rec["cut_at"] = [[round(float(clon[c]), 3), round(float(clat[c]), 3)] for c in cut_cells]
    if cell_depth is not None:
        d = cell_depth[idx]
        order = np.argsort(-d, kind="stable")
        parent = np.arange(n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        nb = [[] for _ in range(n)]
        for x, y in zip(la.tolist(), lb.tolist()):
            nb[x].append(y)
            nb[y].append(x)
        on = np.zeros(n, bool)
        sill = None
        for v in order:
            on[v] = True
            for w in nb[v]:
                if on[w]:
                    parent[find(v)] = find(w)
            if on[sa] and on[sb] and find(sa) == find(sb):
                sill = float(d[v])
                break
        rec["sill_m"] = sill
        rec["cut_cell_depth_m"] = [round(float(cell_depth[c]), 1) for c in cut_cells]
    if node_depth_of_cell is not None:
        rec["cut_node_depth_m"] = node_depth_of_cell(cut_cells)
    return rec
