"""Triangle -> quad recombination by matching on the dual graph.

Objective: min  sum c(quads) + sum p(unmatched triangles). Merging the pair (a, b) instead of leaving both alone
gains g_ab = p_a + p_b - c_ab, so the optimum is a max-weight matching of the dual graph with weights g_ab
(pairs with g_ab <= 0 are never merged). This is the same problem as the min-cost perfect matching on the doubled
graph (copy edges of cost 2p); a perfect-matching solver needs the doubling, a max-weight matching
solver does not.

Cost c = max_k |theta_k - 90 deg| / 90 deg of the merged quad, angles in its tangent plane (quality.kernel
conventions): 0 for a square, -> 1 for a degenerate convex quad, inf for a non-convex one (a corner >= 180 deg).

"greedy": pairs taken in order of decreasing gain (= increasing cost for a scalar price), then augmenting paths
between leftover triangles (alternating paths of up to AUGMENT_DEPTH matched pairs, iterative deepening, best gain
per start), Numba. "exact": networkx max_weight_matching, small meshes only (reference for the gap).
"""
import networkx as nx
import numpy as np
from numba import njit

from meshom.delaunay import adjacency
from meshom.geo import tangent_coords
from meshom.mesh import Mesh
from meshom.quality.kernel import _angles

AUGMENT_DEPTH = 10        # longest augmenting path: this many matched pairs traversed (2 * depth + 1 dual edges)
_CHUNK = 1 << 20


def _merge(tri_a: np.ndarray, tri_b: np.ndarray) -> np.ndarray:
    """(M,3) CCW triangles sharing an edge -> (M,4) CCW quads (u, x, v, w): a = (u, v, w) rotated, x opposite in b."""
    in_b = (tri_a[:, :, None] == tri_b[:, None, :]).any(axis=2)          # (M,3) node of a also in b
    i = np.argmin(in_b, axis=1)                                          # a's node not in b
    r = np.arange(len(tri_a))
    w, u, v = tri_a[r, i], tri_a[r, (i + 1) % 3], tri_a[r, (i + 2) % 3]
    j = np.argmin((tri_b[:, :, None] == tri_a[:, None, :]).any(axis=2), axis=1)
    x = tri_b[r, j]
    return np.column_stack([u, x, v, w])


def _quad_cost(points: np.ndarray, quads: np.ndarray) -> np.ndarray:
    out = np.empty(len(quads))
    for s in range(0, len(quads), _CHUNK):
        ang = _angles(tangent_coords(points[quads[s:s + _CHUNK]]))
        c = np.abs(ang - 0.5 * np.pi).max(axis=1) / (0.5 * np.pi)
        out[s:s + _CHUNK] = np.where((ang >= np.pi).any(axis=1), np.inf, c)
    return out


def quad_cost(points: np.ndarray, tri_a: np.ndarray, tri_b: np.ndarray) -> np.ndarray:
    """(M,) cost of merging tri_a[m] and tri_b[m] (CCW, sharing an edge): max |theta - 90 deg| / 90 deg, inf if
    the quad is non-convex."""
    a, b = np.atleast_2d(tri_a).astype(np.int64), np.atleast_2d(tri_b).astype(np.int64)
    return _quad_cost(points, _merge(a, b))


@njit(cache=True)
def _greedy_take(order, ea, eb, mate):
    for k in order:
        a, b = ea[k], eb[k]
        if mate[a] < 0 and mate[b] < 0:
            mate[a] = b
            mate[b] = a


@njit(cache=True)
def _pair_gain(nbr, gain, a, b):
    for s in range(3):
        if nbr[a, s] == b:
            return gain[a, s]
    return -1.0


@njit(cache=True)
def _augment(nbr, gain, mate, max_depth):
    """Augmenting paths u - x1 = y1 - x2 = y2 ... - z between free u, z (='' matched, '-' new); gain must rise."""
    t_n = len(mate)
    onpath = np.zeros(t_n, dtype=np.bool_)
    outer = np.empty(max_depth + 1, dtype=np.int64)
    mid = np.empty(max_depth + 1, dtype=np.int64)
    slot = np.empty(max_depth + 1, dtype=np.int64)
    acc = np.empty(max_depth + 1, dtype=np.float64)
    b_outer = np.empty(max_depth + 1, dtype=np.int64)
    b_mid = np.empty(max_depth + 1, dtype=np.int64)
    n_aug = 0
    for lim in range(1, max_depth + 1):
        for u in range(t_n):
            if mate[u] >= 0:
                continue
            best, b_len, b_end = 0.0, -1, -1
            d = 0
            outer[0], slot[0], acc[0] = u, 0, 0.0
            onpath[u] = True
            while d >= 0:
                w, s = outer[d], slot[d]
                if s == 3:
                    onpath[w] = False
                    if d > 0:
                        onpath[mid[d]] = False
                    d -= 1
                    continue
                slot[d] = s + 1
                c = nbr[w, s]
                if c < 0 or gain[w, s] <= 0.0 or onpath[c]:
                    continue
                g = acc[d] + gain[w, s]
                if mate[c] < 0:
                    if g > best:
                        best, b_len, b_end = g, d, c
                        b_outer[: d + 1] = outer[: d + 1]
                        b_mid[1: d + 1] = mid[1: d + 1]
                    continue
                if d + 1 > lim:
                    continue
                y = mate[c]
                d += 1
                mid[d], outer[d], slot[d] = c, y, 0
                acc[d] = g - _pair_gain(nbr, gain, c, y)
                onpath[c] = True
                onpath[y] = True
            if b_len >= 0:
                for i in range(b_len + 1):
                    a = b_outer[i]
                    b = b_mid[i + 1] if i < b_len else b_end
                    mate[a] = b
                    mate[b] = a
                n_aug += 1
    return n_aug


def _match_exact(t_n: int, ea: np.ndarray, eb: np.ndarray, g: np.ndarray) -> np.ndarray:
    graph = nx.Graph()
    graph.add_nodes_from(range(t_n))
    graph.add_weighted_edges_from(zip(ea.tolist(), eb.tolist(), g.tolist()))
    mate = np.full(t_n, -1, dtype=np.int64)
    for a, b in nx.max_weight_matching(graph):
        mate[a], mate[b] = b, a
    return mate


def recombine(mesh: Mesh, price: float | np.ndarray = 1.0, method: str = "greedy") -> Mesh:
    """Pair edge-adjacent triangles into quads; a triangle left alone costs price (scalar or (T,) per triangle).

    Nodes and node_flag are unchanged; existing quads are kept (after the new ones); leftover triangles keep their
    input order and orientation; new quads are CCW.
    """
    tris = mesh.tris
    t_n = len(tris)
    p = np.broadcast_to(np.asarray(price, dtype=np.float64), (t_n,))
    nbr = adjacency(tris)
    t_idx, s_idx = np.nonzero(nbr > np.arange(t_n)[:, None])               # each dual edge once
    ea, eb = t_idx.astype(np.int64), nbr[t_idx, s_idx]
    cost = _quad_cost(mesh.points, _merge(tris[ea], tris[eb]))
    g = p[ea] + p[eb] - cost
    ok = g > 0.0
    ea, eb, g = ea[ok], eb[ok], g[ok]

    if method == "exact":
        mate = _match_exact(t_n, ea, eb, g)
    else:
        mate = np.full(t_n, -1, dtype=np.int64)
        _greedy_take(np.argsort(-g, kind="stable"), ea, eb, mate)
        gain = np.full((t_n, 3), -1.0)
        t_ok, s_ok = t_idx[ok], s_idx[ok]
        gain[t_ok, s_ok] = g
        gain[eb, np.argmax(nbr[eb] == ea[:, None], axis=1)] = g
        _augment(nbr, gain, mate, AUGMENT_DEPTH)

    lo = np.flatnonzero(mate > np.arange(t_n))
    quads = _merge(tris[lo], tris[mate[lo]])
    return Mesh(mesh.points, tris[mate < 0], np.vstack([quads, mesh.quads]).astype(np.int64), mesh.node_flag)
