"""Triangulation on the sphere: Delaunay hull, boundary-edge recovery, land removal, Poisson-disc points, background
mesh.

Conventions: points are (N,3) float64 unit vectors; triangles (T,3) int64 are CCW seen from outside the sphere
(det[a,b,c] > 0). Boundary edges are directed with water on the LEFT (domain.loop_edges). Lengths in metres.

Predicates are fp64 with a relative tie tolerance (not exact arithmetic):
- orient(a,b,c) = ((b-a) x (c-a)) . a, > 0 when c is left of the arc a->b; |.| <= 1e-12 |b-a||c-a| is a tie (0).
- incircle(a,b,c,d) = (d-a) . ((b-a) x (c-a)) for CCW (a,b,c), > 0 when d is inside the circumcircle (above the
  plane of a,b,c); a tie (co-circular points, common on regular point sets) never flips, so flip loops terminate and
  ties keep whatever diagonal the hull gave.
"""
import warnings

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError, cKDTree

from meshom.domain import Domain, in_water, loop_edges, resample_loops
from meshom.geo import R_EARTH_M, tangent_basis
from meshom.mesh import Mesh
from meshom.size import SizeField, scaled

_TOL = 1e-12
HULL_MARGIN = 1e-9      # helpers when a hull facet's plane passes within this of the origin, or beyond it
HELPER_RING = 6         # ring points of far_cap_helpers (with the antipode: 7 helpers)


# ---------------------------------------------------------------------------------------------------- topology

def adjacency(tris: np.ndarray) -> np.ndarray:
    """(T,3) int64: nbr[t,i] = triangle across edge (tris[t,i], tris[t,i+1]), -1 on a boundary edge."""
    t_n = len(tris)
    n = int(tris.max()) + 1 if t_n else 0
    a = tris.ravel()
    b = np.roll(tris, -1, axis=1).ravel()
    key = a * n + b
    rkey = b * n + a
    order = np.argsort(key)
    pos = np.searchsorted(key[order], rkey)
    pos = np.minimum(pos, len(key) - 1)
    hit = key[order][pos] == rkey
    nbr = np.where(hit, order[pos] // 3, -1)
    return nbr.reshape(t_n, 3).astype(np.int64)


def _vertex_triangle(tris: np.ndarray, n: int) -> np.ndarray:
    vt = np.full(n, -1, dtype=np.int64)
    vt[tris.ravel()] = np.repeat(np.arange(len(tris), dtype=np.int64), 3)
    return vt


# ---------------------------------------------------------------------------------------------------- predicates

@njit(cache=True, inline="always")
def _orient(P, a, b, c):
    ux, uy, uz = P[b, 0] - P[a, 0], P[b, 1] - P[a, 1], P[b, 2] - P[a, 2]
    vx, vy, vz = P[c, 0] - P[a, 0], P[c, 1] - P[a, 1], P[c, 2] - P[a, 2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    d = nx * P[a, 0] + ny * P[a, 1] + nz * P[a, 2]
    tol = _TOL * np.sqrt((ux * ux + uy * uy + uz * uz) * (vx * vx + vy * vy + vz * vz))
    if d > tol:
        return 1
    if d < -tol:
        return -1
    return 0


@njit(cache=True, inline="always")
def _incircle(P, a, b, c, d):
    """1 if d is strictly inside the circumcircle of CCW (a,b,c), else 0 (outside or tie)."""
    ux, uy, uz = P[b, 0] - P[a, 0], P[b, 1] - P[a, 1], P[b, 2] - P[a, 2]
    vx, vy, vz = P[c, 0] - P[a, 0], P[c, 1] - P[a, 1], P[c, 2] - P[a, 2]
    wx, wy, wz = P[d, 0] - P[a, 0], P[d, 1] - P[a, 1], P[d, 2] - P[a, 2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    s = wx * nx + wy * ny + wz * nz
    tol = _TOL * np.sqrt((ux * ux + uy * uy + uz * uz) * (vx * vx + vy * vy + vz * vz)
                         * (wx * wx + wy * wy + wz * wz))
    return 1 if s > tol else 0


# ---------------------------------------------------------------------------------------------------- flips

@njit(cache=True)
def _index(tris, t, v):
    for k in range(3):
        if tris[t, k] == v:
            return k
    return -1


@njit(cache=True)
def _find_edge(tris, nbr, vt, p, q):
    """(t, k) with tris[t,k] == p and tris[t,k+1] == q, or (-1, -1). Walks the fan of p (both ways at a boundary).
    A p in no triangle (vt[p] < 0: a point the hull left out, e.g. a duplicate) gives (-1, -1); the walk is bounded by
    the triangle count, so a broken fan cannot spin forever."""
    t0 = vt[p]
    if t0 < 0:
        return -1, -1
    for direction in range(2):
        t = t0
        steps = 0
        while True:
            k = _index(tris, t, p)
            steps += 1
            if k < 0 or steps > len(tris):
                return -1, -1
            if tris[t, (k + 1) % 3] == q:
                return t, k
            if direction == 0:
                t = nbr[t, (k + 2) % 3]          # next triangle CCW around p
            else:
                t = nbr[t, k]                    # next triangle CW around p
            if t < 0 or t == t0:
                break
        if t == t0:
            break
    return -1, -1


@njit(cache=True)
def _replace_nbr(tris, nbr, t, a, b, new):
    """In triangle t, the neighbour across edge (a -> b) becomes `new`."""
    if t < 0:
        return
    for k in range(3):
        if tris[t, k] == a and tris[t, (k + 1) % 3] == b:
            nbr[t, k] = new
            return


@njit(cache=True)
def _flip(tris, nbr, vt, t, i):
    """Flip edge i of t: (a,b,c) + (b,a,d) -> t = (c,a,d), u = (d,b,c). Returns (c, d)."""
    a, b, c = tris[t, i], tris[t, (i + 1) % 3], tris[t, (i + 2) % 3]
    u = nbr[t, i]
    j = 0
    for k in range(3):
        if tris[u, k] == b and tris[u, (k + 1) % 3] == a:
            j = k
    d = tris[u, (j + 2) % 3]
    n_bc, n_ca = nbr[t, (i + 1) % 3], nbr[t, (i + 2) % 3]
    n_ad, n_db = nbr[u, (j + 1) % 3], nbr[u, (j + 2) % 3]
    tris[t, 0], tris[t, 1], tris[t, 2] = c, a, d
    nbr[t, 0], nbr[t, 1], nbr[t, 2] = n_ca, n_ad, u
    tris[u, 0], tris[u, 1], tris[u, 2] = d, b, c
    nbr[u, 0], nbr[u, 1], nbr[u, 2] = n_db, n_bc, t
    _replace_nbr(tris, nbr, n_ad, d, a, t)
    _replace_nbr(tris, nbr, n_bc, c, b, u)
    vt[a] = t
    vt[b] = u
    vt[c] = t
    vt[d] = u
    return c, d


@njit(cache=True)
def _is_cons(keys, n, p, q):
    lo, hi = min(p, q), max(p, q)
    k = lo * n + hi
    i = np.searchsorted(keys, k)
    return i < len(keys) and keys[i] == k


@njit(cache=True)
def _crosses(P, a, b, c, d):
    if c == a or c == b or d == a or d == b:
        return False
    return _orient(P, a, b, c) * _orient(P, a, b, d) < 0 and _orient(P, c, d, a) * _orient(P, c, d, b) < 0


@njit(cache=True)
def _recover(P, tris, nbr, vt, edges, keys, max_iter, on_vertex):
    """Sloan-style flip recovery of every constraint edge, then Lawson flips (constraints fixed). Returns #failed.
    A constraint a->b that runs through a vertex w (orient tie) is skipped with on_vertex[e] = w (recover_edges splits
    it there, decisions M40)."""
    n = len(P)
    cap = 4096
    qa = np.empty(cap, np.int64)
    qb = np.empty(cap, np.int64)
    cap_new = 4 * len(edges) + cap
    new_a = np.empty(cap_new, np.int64)
    new_b = np.empty(cap_new, np.int64)
    n_new = 0
    failed = 0
    for e in range(len(edges)):
        a, b = edges[e, 0], edges[e, 1]
        t, k = _find_edge(tris, nbr, vt, a, b)
        if t >= 0:
            continue
        t, k = _find_edge(tris, nbr, vt, b, a)
        if t >= 0:
            continue
        # first crossing edge: the triangle (a,u,v) of a's fan whose wedge contains b
        t0 = vt[a]
        t = t0
        cu, cv, ct, cj = -1, -1, -1, -1
        steps = 0
        while t0 >= 0:
            k = _index(tris, t, a)
            steps += 1
            if k < 0 or steps > len(tris):
                break
            u, v = tris[t, (k + 1) % 3], tris[t, (k + 2) % 3]
            ou = _orient(P, a, u, b)
            if ou > 0 and _orient(P, a, v, b) < 0:
                cu, cv, ct, cj = u, v, t, (k + 1) % 3
                break
            if ou == 0 and ((P[u, 0] - P[a, 0]) * (P[b, 0] - P[a, 0]) + (P[u, 1] - P[a, 1]) * (P[b, 1] - P[a, 1])
                            + (P[u, 2] - P[a, 2]) * (P[b, 2] - P[a, 2])) > 0:
                on_vertex[e] = u                 # the edge a-u runs along a->b: u is on the constraint
                break
            t = nbr[t, (k + 2) % 3]
            if t == t0:
                break
        if ct < 0:
            failed += 1
            continue
        # walk along a->b collecting crossing edges (u right of a->b, v left)
        head, tail = 0, 0
        ok = True
        while True:
            if tail - head >= cap - 1:
                ok = False
                break
            qa[tail % cap] = cu
            qb[tail % cap] = cv
            tail += 1
            nb = nbr[ct, cj]
            jj = 0
            for kk in range(3):
                if tris[nb, kk] == cv and tris[nb, (kk + 1) % 3] == cu:
                    jj = kk
            w = tris[nb, (jj + 2) % 3]
            if w == b:
                break
            o = _orient(P, a, b, w)
            if o > 0:
                cv, ct, cj = w, nb, (jj + 1) % 3
            elif o < 0:
                cu, ct, cj = w, nb, (jj + 2) % 3
            else:                                # a vertex on the constraint: split there (recover_edges)
                on_vertex[e] = w
                ok = False
                break
        if not ok:
            failed += 1
            continue
        # flip until no edge crosses a->b
        stall = 0
        it = 0
        while tail > head and it < max_iter:
            it += 1
            p, q = qa[head % cap], qb[head % cap]
            head += 1
            t, k = _find_edge(tris, nbr, vt, p, q)
            if t < 0:
                continue
            u = nbr[t, k]
            c = tris[t, (k + 2) % 3]
            j = 0
            for kk in range(3):
                if tris[u, kk] == q and tris[u, (kk + 1) % 3] == p:
                    j = kk
            d = tris[u, (j + 2) % 3]
            if _orient(P, c, d, p) * _orient(P, c, d, q) < 0:   # strictly convex quad
                c, d = _flip(tris, nbr, vt, t, k)
                stall = 0
                if _crosses(P, a, b, c, d):
                    qa[tail % cap] = c
                    qb[tail % cap] = d
                    tail += 1
                elif n_new < cap_new:
                    new_a[n_new] = c
                    new_b[n_new] = d
                    n_new += 1
            else:
                qa[tail % cap] = p
                qb[tail % cap] = q
                tail += 1
                stall += 1
                if stall > 2 * (tail - head) + 2:
                    break
        if tail > head:
            failed += 1
    # Lawson: restore the (constrained) Delaunay property around the new edges
    m = 4 * cap_new + 4 * cap
    sa = np.empty(m, np.int64)
    sb = np.empty(m, np.int64)
    top = 0
    for i in range(n_new):
        sa[top] = new_a[i]
        sb[top] = new_b[i]
        top += 1
    it = 0
    while top > 0 and it < max_iter:
        it += 1
        top -= 1
        p, q = sa[top], sb[top]
        if _is_cons(keys, n, p, q):
            continue
        t, k = _find_edge(tris, nbr, vt, p, q)
        if t < 0:
            continue
        u = nbr[t, k]
        if u < 0:
            continue
        c = tris[t, (k + 2) % 3]
        j = 0
        for kk in range(3):
            if tris[u, kk] == q and tris[u, (kk + 1) % 3] == p:
                j = kk
        d = tris[u, (j + 2) % 3]
        if _incircle(P, p, q, c, d) == 1 and _orient(P, c, d, p) * _orient(P, c, d, q) < 0:
            c, d = _flip(tris, nbr, vt, t, k)
            if top + 4 <= m:
                sa[top], sb[top] = q, c
                sa[top + 1], sb[top + 1] = c, p
                sa[top + 2], sb[top + 2] = p, d
                sa[top + 3], sb[top + 3] = d, q
                top += 4
    return failed


# ---------------------------------------------------------------------------------------------------- public API

def far_cap_helpers(points: np.ndarray) -> np.ndarray:
    """(7,3) helper points on the far cap of a point set that lies within one hemisphere (decisions M36): the antipode
    of the mean direction m and a ring of HELPER_RING points around it at rho = (pi - r_max) / 2, r_max = the largest
    angle from m to a point, so the ring lies as far from the antipode as from the points' cap."""
    m = points.sum(axis=0)
    m /= np.linalg.norm(m)
    rho = 0.5 * (np.pi - np.arccos(np.clip((points @ m).min(), -1.0, 1.0)))
    e1, e2 = tangent_basis(m[None])
    a = 2.0 * np.pi * np.arange(HELPER_RING) / HELPER_RING
    ring = -np.cos(rho) * m + np.sin(rho) * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2)
    return np.vstack([-m, ring])


def sphere_delaunay(points: np.ndarray, edges: np.ndarray | None = None, coast: np.ndarray | None = None
                    ) -> np.ndarray:
    """(T,3) int64 CCW Delaunay triangulation of points on the unit sphere = their convex hull (Qhull); with `edges`
    the constrained one of the water: recover_edges(edges), then remove_land(coast) (default coast = edges; an empty
    coast keeps every triangle).

    Points around the origin: the hull covers the whole sphere (T = 2N - 4) and the result is that of the hull,
    recover_edges and remove_land called in turn. Points within one hemisphere (a lake alone, a regional domain: a hull
    facet with the origin on or beyond its outer side, offset > -HULL_MARGIN) would give a hull whose far facets
    double-cover the cap with orientations set by noise; then far_cap_helpers close the hull first, recover_edges and
    remove_land run on the closed hull, and the helpers' triangles are dropped at the end (without a coast the result
    is open along the points' hull). Points with no 3-d hull (fewer than 4, or all on one circle: a lake of 3 nodes,
    a cavity of 4 co-circular nodes; Qhull raises) lie on one cap and get the helpers too. Co-circular ties are split
    as Qhull triangulates the merged facet. A constraint through another vertex is split there (recover_edges, M40),
    and the land removal uses the pieces (those of the coast edges when `coast` is given).
    """
    n = len(points)
    allp = points
    try:
        hull = ConvexHull(points)
        cap = hull.equations[:, 3].max() > -HULL_MARGIN
    except QhullError:                           # < 4 points or all on one circle: within one cap
        cap = True
    if cap:
        allp = np.vstack([points, far_cap_helpers(points)])
        hull = ConvexHull(allp)
    s = hull.simplices.astype(np.int64)
    a, b, c = allp[s[:, 0]], allp[s[:, 1]], allp[s[:, 2]]
    neg = np.einsum("ij,ij->i", a, np.cross(b, c)) < 0
    s[neg] = s[neg][:, [0, 2, 1]]
    if edges is not None and len(edges):
        s, pieces, parent = recover_edges(allp, s, edges, return_edges=True)
        if coast is None:
            coast = pieces
        elif len(pieces) > len(edges):           # coast edges split at a vertex: their pieces
            m = len(allp)
            ek = np.asarray(edges, dtype=np.int64) @ np.array([m, 1])
            ck = np.asarray(coast, dtype=np.int64).reshape(-1, 2) @ np.array([m, 1])
            coast = np.vstack([pieces[np.isin(ek[parent], ck)], np.asarray(coast).reshape(-1, 2)[~np.isin(ck, ek)]])
        s = remove_land(allp, s, coast)
    return s if len(allp) == n else s[(s < n).all(axis=1)]


def recover_edges(points: np.ndarray, tris: np.ndarray, edges: np.ndarray, max_iter: int = 10**8,
                  return_edges: bool = False):
    """Flips until every (n0, n1) of edges is a mesh edge; then Lawson flips restore the constrained Delaunay property.

    tris must be a closed CCW triangulation (the hull of sphere_delaunay; for points within one hemisphere call
    sphere_delaunay(points, edges), which closes the hull first). A constraint passing exactly through another vertex w
    (an orient tie; a self-touching coast, L15) is split into (n0, w), (w, n1) in its place and those are recovered
    (decisions M40); return_edges=True also returns these constraints and, per row, the index of the edge it comes
    from (the land removal needs the pieces). Failures are reported with a warning. A constraint whose end is in no
    triangle (a point the hull left out: a duplicate of another point) cannot be recovered: RuntimeError naming it and
    the point it coincides with (decisions M39; the fan walk from such a node used to spin forever).
    """
    tris = np.ascontiguousarray(tris, dtype=np.int64).copy()
    edges = np.ascontiguousarray(edges, dtype=np.int64)
    if len(edges) == 0:
        return tris
    n = len(points)
    nbr = adjacency(tris)
    vt = _vertex_triangle(tris, n)
    lost = vt[edges] < 0
    if lost.any():
        e = int(np.flatnonzero(lost.any(axis=1))[0])
        v = int(edges[e][lost[e]][0])
        used = np.unique(tris)
        d, j = cKDTree(points[used]).query(points[v])
        raise RuntimeError(f"recover_edges: constraint {e} ({edges[e, 0]}, {edges[e, 1]}): node {v} is in no triangle; "
                           f"it coincides with node {int(used[j])} ({d * R_EARTH_M:.3g} m apart); "
                           f"{int(lost.any(axis=1).sum())} constraints touch such nodes")
    P = np.ascontiguousarray(points, dtype=np.float64)
    parent = np.arange(len(edges), dtype=np.int64)
    while True:
        keys = np.unique(np.minimum(edges[:, 0], edges[:, 1]) * n + np.maximum(edges[:, 0], edges[:, 1]))
        on = np.full(len(edges), -1, dtype=np.int64)
        _recover(P, tris, nbr, vt, edges, keys, max_iter, on)
        a, b = tris.ravel(), np.roll(tris, -1, axis=1).ravel()          # counted at the end: a later constraint
        have = np.minimum(a, b) * n + np.maximum(a, b)                  # can flip a crossing one away
        k = np.flatnonzero((on >= 0) & ~np.isin(np.minimum(edges[:, 0], edges[:, 1]) * n
                                                + np.maximum(edges[:, 0], edges[:, 1]), have))
        if len(k) == 0:
            break
        rep = np.ones(len(edges), dtype=np.int64)                       # split at the vertex on it, in place
        rep[k] = 2
        first = np.cumsum(rep) - rep
        edges, parent = np.repeat(edges, rep, axis=0), np.repeat(parent, rep)
        edges[first[k], 1] = on[k]
        edges[first[k] + 1, 0] = on[k]
    failed = int((~np.isin(keys, have)).sum())
    if failed:
        warnings.warn(f"recover_edges: {failed} of {len(keys)} constraint edges not recovered")
    return (tris, edges, parent) if return_edges else tris


def remove_land(points: np.ndarray, tris: np.ndarray, boundary_edges: np.ndarray) -> np.ndarray:
    """Triangles reachable from the water side of the boundary without crossing a boundary edge.

    boundary_edges (B,2) are directed with water on the left, so the CCW triangle holding (n0 -> n1) is water; the
    flood fill (connected components of the triangle graph cut at boundary edges) keeps every component that holds
    such a triangle. With no boundary edges all triangles are kept.
    """
    if len(boundary_edges) == 0:
        return tris
    n = len(points)
    nbr = adjacency(tris)
    a = tris.ravel()
    b = np.roll(tris, -1, axis=1).ravel()
    be = np.asarray(boundary_edges, dtype=np.int64)
    ukeys = np.minimum(be[:, 0], be[:, 1]) * n + np.maximum(be[:, 0], be[:, 1])
    blocked = np.isin(np.minimum(a, b) * n + np.maximum(a, b), ukeys)
    nb = nbr.ravel()
    ok = (nb >= 0) & ~blocked
    t_of = np.repeat(np.arange(len(tris)), 3)
    g = coo_matrix((np.ones(ok.sum()), (t_of[ok], nb[ok])), shape=(len(tris), len(tris))).tocsr()
    _, lab = connected_components(g, directed=False)
    seeds = t_of[np.isin(a * n + b, be[:, 0] * n + be[:, 1])]
    return tris[np.isin(lab, np.unique(lab[seeds]))]


# ---------------------------------------------------------------------------------------------------- points

@njit(cache=True)
def _greedy(order, nn_idx, nn_d, hc, a_over_r, accepted_in):
    """Random sequential adsorption: accept candidate i (in `order`) if no accepted neighbour is within
    alpha (h_i + h_j) / 2. nn_* are k-nearest lists (chord distances) among the candidates."""
    n = len(hc)
    acc = accepted_in.copy()
    for ii in range(len(order)):
        i = order[ii]
        ok = True
        for k in range(nn_idx.shape[1]):
            j = nn_idx[i, k]
            if j >= n:
                break
            if j == i or not acc[j]:
                continue
            if nn_d[i, k] < a_over_r * 0.5 * (hc[i] + hc[j]):
                ok = False
                break
        acc[i] = ok
    return acc


def _free_of(tree, pts_h, cand, hc, a_over_r, k=12):
    """Candidates without a conflicting point in the tree (points with sizes pts_h)."""
    if tree is None or len(cand) == 0:
        return np.ones(len(cand), dtype=bool)
    d, j = tree.query(cand, k=k, distance_upper_bound=a_over_r * (hc.max() + pts_h.max()) * 0.5 * 1.001)
    hj = np.concatenate([pts_h, [np.inf]])[j]
    return ~np.any(d < a_over_r * 0.5 * (hc[:, None] + hj), axis=1)


def _select(cand, hc, a_over_r, rng, k=24, order=None):
    """Conflict-free subset of the candidates (greedy in `order`, default random)."""
    if len(cand) == 0:
        return np.zeros(0, dtype=bool)
    tree = cKDTree(cand)
    d, j = tree.query(cand, k=min(k, len(cand)), distance_upper_bound=a_over_r * hc.max() * 1.001)
    if d.ndim == 1:
        d, j = d[:, None], j[:, None]
    order = rng.permutation(len(cand)) if order is None else order
    return _greedy(order, j.astype(np.int64), d, hc, a_over_r, np.zeros(len(cand), dtype=bool))


def _fibonacci(m: int) -> np.ndarray:
    i = np.arange(m) + 0.5
    z = 1.0 - 2.0 * i / m
    ph = np.pi * (1.0 + 5**0.5) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(ph), r * np.sin(ph), z])


def poisson_points(domain: Domain, h: SizeField, boundary: list[np.ndarray], alpha: float = 0.85,
                   seed: int = 0) -> np.ndarray:
    """Variable-radius Poisson-disc points in the water: two points are at least alpha (h_p + h_q) / 2 apart.

    Returns np.vstack(boundary) first (the boundary nodes, unchanged, in loop order: loop_edges(boundary) indexes
    them), then the interior points. The water test uses the boundary polygons (domain.loops if boundary is []),
    so interior points never fall between a coast curve and its resampled chord.

    Method: density-weighted random candidates selected greedily in random order (random sequential adsorption),
    then rounds that insert the circumcentres of Delaunay triangles whose empty circle exceeds the exclusion radius,
    largest first, until none is left: the set is maximal, every water point lies within ~alpha h of a sample.
    Measured on `disc` at alpha = 0.85: nearest-neighbour distance p1/p50/p99 = 0.85/0.90/1.15 h; Delaunay edges
    p1/p50/p99 = 0.85/1.10/1.62 h (theory: [alpha, 2 alpha] h). alpha ~ 0.75 centres the Delaunay edges on h.
    """
    rng = np.random.default_rng(seed)
    a_over_r = alpha / R_EARTH_M                        # chord ~ arc for h << R
    bnd = np.vstack(boundary) if len(boundary) else np.zeros((0, 3))
    water_dom = Domain(list(boundary), domain.name) if len(boundary) else domain

    def water(x):
        return in_water(water_dom, x) if water_dom.loops else np.ones(len(x), dtype=bool)

    pts, ph_ = bnd, h(bnd) if len(bnd) else np.zeros(0)
    # round 0: candidates with density ~ 2x the packing density 0.7 / (alpha h)^2, by Fibonacci-lattice cells
    m = 200_000
    lat = _fibonacci(m)
    cell = 4.0 * np.pi / m                                       # sr per cell
    lam = 2.0 * 0.7 * cell / (a_over_r * h(lat)) ** 2
    src = np.repeat(np.arange(m), rng.poisson(lam))
    e1, e2 = tangent_basis(lat[src])
    rr = np.sqrt(cell / np.pi) * np.sqrt(rng.uniform(0, 1, len(src)))
    ang = rng.uniform(0, 2 * np.pi, len(src))
    cand = lat[src] + rr[:, None] * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    cand = cand[water(cand)]
    hc = h(cand)
    free = _free_of(cKDTree(pts) if len(pts) else None, ph_, cand, hc, a_over_r)
    cand, hc = cand[free], hc[free]
    keep = _select(cand, hc, a_over_r, rng)
    pts, ph_ = np.vstack([pts, cand[keep]]), np.r_[ph_, hc[keep]]
    # fill: circumcentres of Delaunay triangles whose empty circle is wider than the exclusion radius, largest first
    for _ in range(200):
        tris = sphere_delaunay(pts)
        a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
        cc = np.cross(b - a, c - a)
        cc /= np.linalg.norm(cc, axis=1, keepdims=True)
        rc = np.linalg.norm(cc - a, axis=1)
        hc = h(cc)
        hv = np.max(ph_[tris], axis=1)
        big = rc > a_over_r * 0.5 * (hc + hv)
        if not big.any():
            break
        cand, hc, score = cc[big], hc[big], rc[big] / hc[big]
        wet = water(cand)
        cand, hc, score = cand[wet], hc[wet], score[wet]
        if len(cand) == 0:
            break
        free = _free_of(cKDTree(pts), ph_, cand, hc, a_over_r)
        cand, hc, score = cand[free], hc[free], score[free]
        if len(cand) == 0:
            break
        keep = _select(cand, hc, a_over_r, rng, order=np.argsort(-score))
        pts, ph_ = np.vstack([pts, cand[keep]]), np.r_[ph_, hc[keep]]
    return pts


# ---------------------------------------------------------------------------------------------------- background

def background_mesh(domain: Domain, h: SizeField, factor: float = 0.5) -> Mesh:
    """Isotropic triangle mesh of the water at spacing factor * h (for fields and extraction).

    resample_loops -> poisson_points -> sphere_delaunay (coast recovered, land removed). Boundary nodes come first
    (np.vstack of the resampled loops) and have node_flag 1; interior nodes 0. No smoothing.
    """
    hf = scaled(h, factor)
    loops = resample_loops(domain, hf)
    pts = poisson_points(domain, hf, loops)
    tris = sphere_delaunay(pts, loop_edges(loops) if loops else None)
    flag = np.zeros(len(pts), dtype=np.int8)
    flag[: sum(len(lp) for lp in loops)] = 1
    return Mesh(pts, tris, np.zeros((0, 4), dtype=np.int64), flag)
