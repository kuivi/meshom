"""Population control of a triangle mesh on the sphere.

Bubble-mesh / pliant style (Shimada & Gossard 1995, Bossen & Heckbert 1996) in rounds; per round:
1. edges and r_e = ln(s_e / h_e) as the score measures them (great-circle length, h at the normalised midpoint);
2. deletions: edges with r < t_del (-0.18) in increasing r (ties by edge index); the free endpoint with the smaller mean
   r of its edges is deleted unless an endpoint is blocked; the deleted node and its ring are blocked (an independent
   set: no two deletions share a ring);
3. insertions: edges with r > t_ins (+0.22) in decreasing r, not on the boundary, neither endpoint blocked: the
   normalised midpoint is added and both endpoints are blocked;
4. re-triangulation: sphere Delaunay (hull) of the kept and new points, the constraint edges recovered and the land
   removed when the caller passes them (coast loops, block outlines), unreferenced nodes dropped;
5. `sweeps` sweeps of meshom.smooth kind "size" with the fixed nodes fixed.
Free interior nodes with valence <= max_deg_del (4) are deleted first in every round (node order, same independent-set
rule): a valence-4 node keeps its fan at 90 deg whatever the fit does (measured with m15: all band angles outside
[40, 100] deg sat at the 43 valence-4 nodes population control had left). Before the first round the same `sweeps`
sweeps run once. Rounds stop when nothing is selected; then up to `cleanup` rounds delete only
low-valence nodes. (In a gap between fixed outlines the rounds cycle, ~5 % of the fill nodes deleted and inserted per
round, and the p99 right after a round rises 0.30 -> 0.42; keeping the round with the lowest p99 kept the start and
the fit then ended worse, 0.159 vs 0.144 on m14_blocks_tri uniform: the count and the defects matter, not that p99.)
Fixed nodes (node_flag != 0, the boundary, the caller's mask) are never deleted or moved and keep their order and
indices at the front when they come first (the kept nodes keep their order, new nodes are appended).
Thresholds: -0.14 / +0.18 make defects (p99 0.195 / 0.199 instead of 0.114-0.129).
"""
import time

import numpy as np
from numba import njit

from meshom.delaunay import _flip, _orient, adjacency, sphere_delaunay
from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh, build_topology
from meshom.size_fit import edge_ln_sh

T_DEL = -0.18
T_INS = 0.22


@njit(cache=True)
def _select(edges, r, order_del, order_ins, fixed, bnd_edge, ptr, nbrs, rmean, t_del, t_ins, max_deg_del):
    """(deleted node mask, inserted edge ids) of one round: independent sets in the given orders. Free nodes with at
    most max_deg_del edges go first (node order), then the short edges, then the long ones."""
    n = len(fixed)
    blocked = np.zeros(n, dtype=np.bool_)
    dele = np.zeros(n, dtype=np.bool_)
    for v in range(n):
        if fixed[v] or blocked[v] or ptr[v + 1] - ptr[v] > max_deg_del:
            continue
        dele[v] = True
        blocked[v] = True
        for j in range(ptr[v], ptr[v + 1]):
            blocked[nbrs[j]] = True
    for ii in range(len(order_del)):
        k = order_del[ii]
        if r[k] >= t_del:
            break
        a, b = edges[k, 0], edges[k, 1]
        if blocked[a] or blocked[b] or (fixed[a] and fixed[b]):
            continue
        if fixed[a]:
            v = b
        elif fixed[b]:
            v = a
        else:
            v = a if rmean[a] <= rmean[b] else b
        dele[v] = True
        blocked[v] = True
        for j in range(ptr[v], ptr[v + 1]):
            blocked[nbrs[j]] = True
    ins = np.empty(len(order_ins), dtype=np.int64)
    m = 0
    for ii in range(len(order_ins)):
        k = order_ins[ii]
        if r[k] <= t_ins:
            break
        if bnd_edge[k]:
            continue
        a, b = edges[k, 0], edges[k, 1]
        if blocked[a] or blocked[b]:
            continue
        ins[m] = k
        m += 1
        blocked[a] = True
        blocked[b] = True
    return dele, ins[:m]


@njit(cache=True)
def _vflips(P, tris, nbr, vt, deg, interior, lh_tol, R_unit_h, max_passes):
    """Valence flips: edge (a,b) with opposite corners c, d flips to (c,d) when that lowers sum (deg - 6)^2 over the
    interior nodes among a, b, c, d (a + b >= c + d + 3 when all four are interior), the quad is strictly convex, and
    the new edge is within |ln(len / h)| <= lh_tol (h of its ends, R_unit_h = h / R per node). Boundary edges never
    flip. Returns the number of flips."""
    total = 0
    for _ in range(max_passes):
        nf = 0
        for t in range(len(tris)):
            for k in range(3):
                u = nbr[t, k]
                if u < t:
                    continue
                a, b, c = tris[t, k], tris[t, (k + 1) % 3], tris[t, (k + 2) % 3]
                j = 0
                for kk in range(3):
                    if tris[u, kk] == b and tris[u, (kk + 1) % 3] == a:
                        j = kk
                d = tris[u, (j + 2) % 3]
                gain = 0.0
                if interior[a]:
                    gain += (deg[a] - 6) ** 2 - (deg[a] - 7) ** 2
                if interior[b]:
                    gain += (deg[b] - 6) ** 2 - (deg[b] - 7) ** 2
                if interior[c]:
                    gain += (deg[c] - 6) ** 2 - (deg[c] - 5) ** 2
                if interior[d]:
                    gain += (deg[d] - 6) ** 2 - (deg[d] - 5) ** 2
                if gain <= 0.0 or deg[a] <= 4 or deg[b] <= 4:
                    continue
                if _orient(P, c, d, a) * _orient(P, c, d, b) >= 0:
                    continue
                x0, x1, x2 = P[c, 0] - P[d, 0], P[c, 1] - P[d, 1], P[c, 2] - P[d, 2]
                ln = 0.5 * np.log(x0 * x0 + x1 * x1 + x2 * x2) - 0.5 * np.log(R_unit_h[c] * R_unit_h[d])
                if abs(ln) > lh_tol:
                    continue
                _flip(tris, nbr, vt, t, k)
                deg[a] -= 1
                deg[b] -= 1
                deg[c] += 1
                deg[d] += 1
                nf += 1
                break
        total += nf
        if nf == 0:
            break
    return total


def valence_flips(mesh: Mesh, h, fixed: np.ndarray, lh_tol: float = 0.35, max_passes: int = 20) -> tuple[Mesh, int]:
    """Copy of the triangle mesh with valence flips (_vflips); fixed and boundary nodes do not count."""
    tris = np.ascontiguousarray(mesh.tris, dtype=np.int64).copy()
    P = np.ascontiguousarray(mesh.points, dtype=np.float64)
    nbr = adjacency(tris)
    vt = np.full(len(P), -1, dtype=np.int64)
    vt[tris.ravel()] = np.repeat(np.arange(len(tris), dtype=np.int64), 3)
    topo = build_topology(mesh)
    deg = np.bincount(topo.edges.ravel(), minlength=len(P)).astype(np.int64)
    interior = ~(topo.boundary_node | fixed)
    n = _vflips(P, tris, nbr, vt, deg, interior, lh_tol, h(P) / R_EARTH_M, max_passes)
    return Mesh(mesh.points, tris, mesh.quads, mesh.node_flag), int(n)


def _triangulate(P: np.ndarray, bedges: np.ndarray | None) -> np.ndarray:
    return sphere_delaunay(P, bedges)


def _compact(P, tris, keep_first: int, flag):
    """Drop nodes no triangle uses (the first keep_first nodes are always kept)."""
    used = np.zeros(len(P), dtype=bool)
    used[tris.ravel()] = True
    used[:keep_first] = True
    if used.all():
        return P, tris, flag
    new = np.cumsum(used) - 1
    return P[used], new[tris], flag[used]


def _analyse(cur: Mesh, h):
    topo = build_topology(cur)
    e = topo.edges
    r = edge_ln_sh(cur.points, e, h)
    return e, topo.edge_elems[:, 1] < 0, r, float(np.percentile(np.abs(r), 99))


def popctl(mesh: Mesh, h, fixed=None, bedges: np.ndarray | None = None, rounds: int = 8, t_del: float = T_DEL,
           t_ins: float = T_INS, sweeps: int = 10, max_deg_del: int = 4, cleanup: int = 2) -> tuple[Mesh, dict]:
    """Copy of the triangle mesh after population-control rounds (module docstring) and info (per-round counts, p99).
    bedges: (B,2) constraint edges into the mesh's node ids (water on the left; the first nodes, which are fixed),
    None on the closed sphere."""
    from meshom.smooth import smooth

    t_start = time.perf_counter()
    topo = build_topology(mesh)
    fixed_m = (np.asarray(mesh.node_flag) != 0) | topo.boundary_node
    if fixed is not None:
        fixed = np.asarray(fixed)
        if fixed.dtype == bool:
            fixed_m |= fixed
        else:
            fixed_m[fixed.astype(np.int64)] = True
    # fixed nodes keep their indices when they form a prefix; the constraint edges index that prefix
    keep_first = int(np.argmin(fixed_m)) if not fixed_m.all() else len(fixed_m)
    flag = np.asarray(mesh.node_flag).copy()
    cur = Mesh(np.ascontiguousarray(mesh.points, dtype=np.float64), mesh.tris, mesh.quads, flag)
    if sweeps:
        cur = smooth(cur, h, fixed_m, iters=sweeps, kind="size")

    def one_round(cur, flag, fixed_m, e, bnd_edge, r, td, ti):
        P = cur.points
        n = len(P)
        deg = np.bincount(e.ravel(), minlength=n)
        rmean = np.bincount(e.ravel(), np.repeat(r, 2), minlength=n) / np.maximum(deg, 1)
        ee = np.vstack([e, e[:, ::-1]])
        ee = ee[np.argsort(ee[:, 0], kind="stable")]
        ptr = np.searchsorted(ee[:, 0], np.arange(n + 1)).astype(np.int64)
        dele, ins = _select(e, r, np.argsort(r, kind="stable"), np.argsort(-r, kind="stable"), fixed_m, bnd_edge,
                            ptr, np.ascontiguousarray(ee[:, 1]), rmean, td, ti, max_deg_del)
        nd, ni = int(dele.sum()), int(len(ins))
        if nd == 0 and ni == 0:
            return None
        keep = ~dele
        mid = P[e[ins, 0]] + P[e[ins, 1]]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        P = np.vstack([P[keep], mid])
        flag = np.r_[flag[keep], np.zeros(ni, dtype=flag.dtype)]
        fixed_m = np.r_[fixed_m[keep], np.zeros(ni, dtype=bool)]
        tris = _triangulate(P, bedges)
        P, tris, both = _compact(P, tris, keep_first, np.c_[flag, fixed_m.astype(flag.dtype)])
        flag, fixed_m = both[:, 0].copy(), both[:, 1].astype(bool)
        cur = Mesh(P, tris, np.zeros((0, 4), dtype=np.int64), flag)
        if sweeps:
            cur = smooth(cur, h, fixed_m, iters=sweeps, kind="size")
        return cur, flag, fixed_m, nd, ni

    info = {"rounds": [], "cleanup": []}
    for rnd in range(rounds):
        t0 = time.perf_counter()
        e, bnd_edge, r, p99 = _analyse(cur, h)
        out = one_round(cur, flag, fixed_m, e, bnd_edge, r, t_del, t_ins)
        if out is None:
            break
        cur, flag, fixed_m, nd, ni = out
        info["rounds"].append({"deleted": nd, "inserted": ni, "p99_before": p99, "s": time.perf_counter() - t0})
    for _ in range(cleanup):                     # low-valence nodes only
        e, bnd_edge, r, _ = _analyse(cur, h)
        out = one_round(cur, flag, fixed_m, e, bnd_edge, r, -np.inf, np.inf)
        if out is None:
            break
        cur, flag, fixed_m, nd, _ = out
        info["cleanup"].append(nd)
    info.update(p99_after=_analyse(cur, h)[3], total_s=time.perf_counter() - t_start)
    return cur, info
