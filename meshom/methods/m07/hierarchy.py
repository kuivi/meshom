"""Multiresolution graph hierarchy of Instant Meshes (Jakob 2015 §3.5), on the unit sphere.

Level 0 is the input graph (the edges of the background triangle mesh, link weight 1, vertex area = a third of the
incident triangle areas). Each coarsening step matches neighbours greedily in decreasing order of a score (below),
merges each matched pair into one vertex (area-weighted mean position, re-normalised onto the sphere, so the normal is the position), copies
unmatched vertices, and maps the edges (weights summed, self loops dropped). It stops at one vertex per connected
component. The levels are not meshes; they only carry the neighbour graph for the Gauss-Seidel sweeps.

Score: <n_i, n_j> max(A_i / A_j, A_j / A_i), as the released code, not the paper's
min ratio: with the min ratio the small vertices next to grown clusters are never matched, the graph turns star-like
and the coarsening stalls (uniform, scale 4: 12048 -> ~800 vertices after 15 levels, then -4 per level, 316 levels);
with the max ratio it halves every level (17 levels).
"""
from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass
class Level:
    V: np.ndarray                    # (n,3) unit vectors (positions = normals)
    A: np.ndarray                    # (n,) area weights
    ptr: np.ndarray                  # (n+1,) CSR row pointers of the symmetric neighbour graph
    idx: np.ndarray                  # neighbours
    w: np.ndarray                    # link weights
    edges: np.ndarray                # (E,2) unique undirected edges i < j
    parent: np.ndarray | None = None     # (n,) vertex of the next coarser level (None on the coarsest)
    children: np.ndarray | None = None   # (n,2) vertices of the next finer level, -1 if single (None on level 0)

    @property
    def n(self) -> int:
        return len(self.V)


def _csr(n: int, i: np.ndarray, j: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, ww = np.r_[i, j], np.r_[j, i], np.r_[w, w]
    order = np.lexsort((b, a))
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(a, minlength=n))
    return ptr, b[order].astype(np.int64), ww[order]


@njit(cache=True)
def _match(order, ei, ej, n):
    taken = np.zeros(n, dtype=np.bool_)
    pairs = np.empty((len(order), 2), dtype=np.int64)
    m = 0
    for k in order:
        i, j = ei[k], ej[k]
        if taken[i] or taken[j]:
            continue
        taken[i] = True
        taken[j] = True
        pairs[m, 0] = i
        pairs[m, 1] = j
        m += 1
    return pairs[:m], taken


def unique_edges(tris: np.ndarray, n: int) -> np.ndarray:
    a, b = tris.ravel(), np.roll(tris, -1, axis=1).ravel()
    key = np.unique(np.minimum(a, b) * n + np.maximum(a, b))
    return np.column_stack([key // n, key % n]).astype(np.int64)


def vertex_areas(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """(N,) a third of the (flat) area of the incident triangles, unit sphere."""
    q = points[tris]
    a = 0.5 * np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=1)
    return np.bincount(tris.ravel(), np.repeat(a / 3.0, 3), minlength=len(points))


def build_hierarchy(points: np.ndarray, tris: np.ndarray) -> list[Level]:
    """Levels fine -> coarse; levels[0] is the background mesh graph."""
    n = len(points)
    e = unique_edges(tris, n)
    V, A, w = points, vertex_areas(points, tris), np.ones(len(e))
    levels: list[Level] = []
    children = None
    while True:
        ptr, idx, ww = _csr(len(V), e[:, 0], e[:, 1], w)
        lev = Level(V, A, ptr, idx, ww, e, None, children)
        levels.append(lev)
        if len(e) == 0:
            break
        i, j = e[:, 0], e[:, 1]
        score = np.einsum("ij,ij->i", V[i], V[j]) * np.maximum(A[i] / A[j], A[j] / A[i])
        pairs, taken = _match(np.argsort(-score, kind="stable"), i, j, len(V))
        single = np.flatnonzero(~taken)
        npair = len(pairs)
        nc = npair + len(single)
        parent = np.empty(len(V), dtype=np.int64)
        parent[pairs[:, 0]] = np.arange(npair)
        parent[pairs[:, 1]] = np.arange(npair)
        parent[single] = npair + np.arange(len(single))
        children = np.full((nc, 2), -1, dtype=np.int64)
        children[:npair] = pairs
        children[npair:, 0] = single
        lev.parent = parent
        Ac = np.bincount(parent, A, minlength=nc)
        Vc = np.column_stack([np.bincount(parent, A * V[:, k], minlength=nc) for k in range(3)])
        Vc /= np.linalg.norm(Vc, axis=1, keepdims=True)
        ci, cj = parent[i], parent[j]
        keep = ci != cj
        lo, hi = np.minimum(ci[keep], cj[keep]), np.maximum(ci[keep], cj[keep])
        key, inv = np.unique(lo * nc + hi, return_inverse=True)
        w = np.bincount(inv, w[keep], minlength=len(key))
        e = np.column_stack([key // nc, key % nc]).astype(np.int64)
        V, A = Vc, Ac
    return levels
