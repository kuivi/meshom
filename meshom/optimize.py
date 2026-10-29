"""Method-independent mesh optimisation after the generation (decisions Z1-Z13).

optimize(mesh, h, tol=None, opts=None, fixed=None, frozen=None) -> (Mesh, info)

Works on any mesh of the sphere (triangles, quads or both). Fixed nodes: the topological boundary (the coast loops of
a conforming mesh, the cut boundary of a cut one), nodes with node_flag != 0 and the caller's `fixed`; a fixed node is
never moved or removed and every boundary edge stays. Triangles of a declared triangle zone (Mesh.tri_zone)
take part in no operation but the flips among themselves. Frozen cells (`frozen`, the fixed blocks of
m14_blocks_rep, decisions R24; quads) take part in no operation and their nodes are fixed. Steps (opts["steps"], run in this order):

1. "flip" (Numba): the interior edge of two triangles is flipped when the convex quad's other diagonal lowers the
   larger of the two triangles' largest corner angles and keeps their smallest angle at least min(tri_min, the old
   pair's smallest); no triangle gets two boundary edges, no node more than 32 elements, an interior node keeps >= 3.
   Each flip lowers the pair's largest angle, so the passes terminate. Triangles only (quads keep their diagonals).
2. "merge": two edge-adjacent triangles into a quad whose corners stay within [180 - quad_max, quad_max] (135 deg:
   the quad target gate), cheapest quad first (greedy), each kept by the local acceptance (mode "merge").
3. "cluster": every edge-connected cluster of 3 .. cluster_size_max triangles (the tri_cluster_max gate; a larger one is
   a triangle region, left to the merges): candidates are the unions of
   connected element sets with a cluster triangle (2 .. cluster_sets elements, any of them quads: two triangles and a
   quad -> two quads, a triangle and a quad -> a quad and the triangle one step away, the whole cluster's polygon) and
   the stars of free interior nodes at the cluster (the node collapsed), each a simple polygon of <= poly_max nodes
   re-split in every way into quads and at most one triangle (the union's interior nodes removed). Every candidate is
   measured as is, and, when none is accepted, again after relaxing the free nodes within relax_rings rings; when
   still none is, the fit_try best near misses (failing only on the size-jump / bad-edge counts) once more after a
   local size_fit.smooth_fit of those nodes at the tolerance; the best accepted one per cluster and pass is kept (mode
   "cluster").
4. "valence": interior free nodes whose elements are all quads and whose valence is not 4: removal of a valence-2 or
   -3 node (its star re-split), the collapse of a quad's diagonal between two such nodes (their neighbours shared by
   exactly the quad's other two corners), the edge rotation of two quads (their hexagon re-split); kept only where the
   gross charge falls by >= 2 (an irregular pair cancelled: mode "valence").
5. "fit": meshom.size_fit.smooth_fit at the size tolerance (the run's, else the size gate 0.15) on the free nodes
   within fit_rings rings of every node an operation touched ("local"; "all": every free node; "off"), kept only if
   the region's measure is not worse (nothing inverted, no hard-gate count, bad element or bad edge more).
The topology steps 2-4 repeat while a pass accepts something (<= passes); then the flips once more and the fit.

Local acceptance of a topology operation (remove elements rm, add new ones): the patch must keep the mesh valid —
the patch's boundary half-edges unchanged, no repeated node in a cell or directed half-edge, the same Euler
characteristic, every new interior edge new to the mesh, removed nodes free, interior and with their whole star
replaced — so the global boundary and the Euler sum never change. The region Z = the nodes of rm and the free nodes
within relax_rings rings is measured before and after (repair._measure on the elements at Z and their first ring:
hard-gate counts, inverted elements, bad quads > quad_max, bad edges |ln s/h| > tol, element size ratio > 1.15, gross
and net charge; plus the triangles in clusters >= 3 reachable from the region, elements with >= 2 boundary edges and
the worst corner deviation). Every mode needs: nothing inverted, no hard-gate count up, no element with two boundary
edges more, the net charge kept, no bad element more, no size jump next to the coast row more, and no bad edge or
size jump more, counted on the region and on its part in the gates' _coast3 subset — except that a cluster operation
may add up to budget_op of each while the whole mesh and its _coast3 subset keep below budget_share of their edges
(the gates are p99; the input's counts when it is already above); then "merge": fewer triangles, no more cluster
triangles, gross charge not up; "cluster": fewer cluster triangles, gross charge not up by more than gross_slack;
"valence": gross charge down by >= 2, no more triangles or cluster triangles. Among a cluster's (or a node's)
accepted candidates the best has the fewest cluster triangles, then bad elements, then bad edges + size jumps, then
the smallest worst corner deviation. A rejected operation is undone (topology and positions); the gross-charge rule
is checked first from the topology alone (Engine.dgross, exact), so most candidates never reach the geometry.

The loops over candidates run in Python on a working mesh with a node -> element table; their count is set by the
triangles and irregular nodes (2-5 % of a mixed mesh), the flips and the fit run in Numba. JAX / GPU for the fit at
10^6-10^7 nodes (mesh B) is a later step: the fit's energy is size_fit's (Numba, L-BFGS on the CPU).
"""
import time

import numpy as np
from numba import njit

from meshom.delaunay import _flip, _is_cons, _orient, adjacency
from meshom.geo import tangent_coords
from meshom.mesh import Mesh, build_topology
from meshom.quality.kernel import _angles

SIZE_GATE = 0.15            # the size gate when the run has no tolerance (DEFAULT_GATES size_fidelity_p99, G9)

DEFAULTS = dict(
    steps=("flip", "merge", "cluster", "valence", "fit"),   # Z2: the steps run, in this order
    passes=3,               # Z3: rounds of merge / cluster / valence; stop when a round accepts nothing
    quad_max=135.0,         # Z7: corner angle limit of a quad an operation makes (the quad target gate)
    tri_min=20.0,           # Z5: a flip keeps every triangle angle >= min(this, the old pair's smallest)
    flip_passes=50,         # Z5: passes of the flip kernel over the triangles
    relax_rings=1,          # Z9: free nodes within this many rings of an operation relaxed when it is evaluated
    relax_size=20,          # Z9: "size" sweeps of that relaxation (meshom.smooth), then ...
    relax_angle=3,          # ... "angle" sweeps (repair R11's values)
    cluster_sets=3,         # Z9: element sets of up to this many elements re-split around a cluster
    poly_max=8,             # Z9: largest polygon re-split (nodes)
    cluster_size_max=12,    # Z9: larger clusters (a triangle region) are left to the merges (counted as skipped)
    gross_slack=0,          # Z8: a cluster operation may raise the gross charge by at most this
    fit="local",            # Z12: "local" | "all" | "off"
    fit_rings=3,            # Z12: the local fit frees the nodes within this many rings of every touched node
    fit_opts={},            # options of size_fit.smooth_fit
    bad_ratio=1.15,         # Z8: an edge with element size ratio above this is a size jump (the p99 gate's value)
    budget_share=0.009,     # Z11: cluster operations may add size jumps / bad edges up to this share of the edges
    budget_op=4,            # Z11: ... and at most this many per operation
    fit_try=8,              # Z13: a cluster without an accepted candidate: its best near misses (failing only on the
                            # size-jump / bad-edge counts after the relaxation) evaluated again after a local fit
)


def _measure_opts(o: dict, tol: float) -> dict:
    """Options of repair._measure: the hard gates (repair's copies of DEFAULT_GATES), bad quad > quad_max, bad edge
    |ln s/h| > tol, size jump > bad_ratio."""
    from meshom.repair import DEFAULTS as RD

    return {**RD, "bad_quad_deg": float(o["quad_max"]), "bad_edge": float(tol), "bad_ratio": float(o["bad_ratio"])}


# ------------------------------------------------------------------------------------------------------- flips

@njit(cache=True)
def _corner(P, a, b, c):
    """Angle at a of the triangle (a, b, c), in a's tangent plane (rad, unsigned)."""
    ux, uy, uz = P[b, 0] - P[a, 0], P[b, 1] - P[a, 1], P[b, 2] - P[a, 2]
    vx, vy, vz = P[c, 0] - P[a, 0], P[c, 1] - P[a, 1], P[c, 2] - P[a, 2]
    su = ux * P[a, 0] + uy * P[a, 1] + uz * P[a, 2]
    sv = vx * P[a, 0] + vy * P[a, 1] + vz * P[a, 2]
    ux, uy, uz = ux - su * P[a, 0], uy - su * P[a, 1], uz - su * P[a, 2]
    vx, vy, vz = vx - sv * P[a, 0], vy - sv * P[a, 1], vz - sv * P[a, 2]
    cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return np.arctan2(np.sqrt(cx * cx + cy * cy + cz * cz), ux * vx + uy * vy + uz * vz)


@njit(cache=True)
def _minmax(P, a, b, c):
    x, y, z = _corner(P, a, b, c), _corner(P, b, c, a), _corner(P, c, a, b)
    return min(x, min(y, z)), max(x, max(y, z))


@njit(cache=True)
def _angle_flips(P, tris, nbr, vt, zone, epn, bnode, bkeys, n, tri_min, epn_max, max_passes):
    """Flips by angle quality (module docstring, step 1). Returns the number of flips."""
    total = 0
    for _ in range(max_passes):
        nf = 0
        for t in range(len(tris)):
            for k in range(3):
                u = nbr[t, k]
                if u < t or zone[t] != zone[u]:
                    continue
                a, b, c = tris[t, k], tris[t, (k + 1) % 3], tris[t, (k + 2) % 3]
                j = 0
                for kk in range(3):
                    if tris[u, kk] == b and tris[u, (kk + 1) % 3] == a:
                        j = kk
                d = tris[u, (j + 2) % 3]
                if _orient(P, c, d, a) * _orient(P, c, d, b) >= 0:       # not strictly convex
                    continue
                if _orient(P, c, a, d) <= 0 or _orient(P, d, b, c) <= 0:
                    continue
                if epn[c] + 1 > epn_max or epn[d] + 1 > epn_max:
                    continue
                if (not bnode[a] and epn[a] - 1 < 3) or (not bnode[b] and epn[b] - 1 < 3):
                    continue
                if (_is_cons(bkeys, n, c, a) and _is_cons(bkeys, n, a, d)) or \
                        (_is_cons(bkeys, n, d, b) and _is_cons(bkeys, n, b, c)):
                    continue                                              # a triangle with two boundary edges
                lo1, hi1 = _minmax(P, a, b, c)
                lo2, hi2 = _minmax(P, b, a, d)
                lo3, hi3 = _minmax(P, c, a, d)
                lo4, hi4 = _minmax(P, d, b, c)
                old_hi, old_lo = max(hi1, hi2), min(lo1, lo2)
                new_hi, new_lo = max(hi3, hi4), min(lo3, lo4)
                if new_hi < old_hi - 1e-9 and new_lo >= min(old_lo, tri_min):
                    _flip(tris, nbr, vt, t, k)
                    epn[a] -= 1
                    epn[b] -= 1
                    epn[c] += 1
                    epn[d] += 1
                    nf += 1
                    break
        total += nf
        if nf == 0:
            break
    return total


def flip_triangles(mesh: Mesh, tri_min_deg: float = 20.0, max_passes: int = 50, epn_max: int = 32) -> tuple[Mesh, int]:
    """Copy of mesh with the triangle edges flipped by angle quality (step 1); (mesh, number of flips)."""
    T = mesh.n_tris
    if T < 2:
        return mesh, 0
    n = mesh.n_nodes
    tris = np.ascontiguousarray(mesh.tris, dtype=np.int64).copy()
    nbr = adjacency(tris)
    topo = build_topology(mesh)
    bnd = topo.edges[topo.edge_elems[:, 1] < 0]
    bkeys = np.sort(np.minimum(bnd[:, 0], bnd[:, 1]) * n + np.maximum(bnd[:, 0], bnd[:, 1])).astype(np.int64)
    vt = np.full(n, -1, dtype=np.int64)
    vt[tris.ravel()] = np.repeat(np.arange(T, dtype=np.int64), 3)
    zone = np.zeros(T, dtype=np.bool_) if mesh.tri_zone is None else np.asarray(mesh.tri_zone, dtype=np.bool_)
    epn = topo.elems_per_node.astype(np.int64).copy()
    nf = _angle_flips(np.ascontiguousarray(mesh.points, dtype=np.float64), tris, nbr, vt, zone, epn,
                      topo.boundary_node, bkeys, n, np.radians(tri_min_deg), epn_max, max_passes)
    return Mesh(mesh.points, tris, mesh.quads, mesh.node_flag, mesh.tri_zone), int(nf)


# ------------------------------------------------------------------------------------------------ polygon splits

def poly_splits(p: tuple, max_tris: int) -> list[tuple]:
    """Every split of the CCW polygon p (node ids) into quads and at most max_tris triangles by non-crossing
    diagonals: a list of tuples of cells (CCW tuples). Each split has one cell on the edge p0 -> p1, so each is listed
    once (an octagon has 12 quad splits)."""
    return [c for c, _ in _splits(tuple(p), max_tris)]


def _splits(p: tuple, budget: int) -> list:
    n = len(p)
    if n == 2:
        return [((), 0)]
    out = []
    if budget >= 1:
        for k in range(2, n):                                   # triangle (p0, p1, pk)
            for cl, tl in _splits(p[1:k + 1], budget - 1):
                for cr, tr in _splits(p[k:] + p[:1], budget - 1 - tl):
                    out.append((((p[0], p[1], p[k]),) + cl + cr, 1 + tl + tr))
    for j in range(2, n - 1):                                   # quad (p0, p1, pj, pk)
        for k in range(j + 1, n):
            for c1, t1 in _splits(p[1:j + 1], budget):
                for c2, t2 in _splits(p[j:k + 1], budget - t1):
                    for c3, t3 in _splits(p[k:] + p[:1], budget - t1 - t2):
                        out.append((((p[0], p[1], p[j], p[k]),) + c1 + c2 + c3, t1 + t2 + t3))
    return out


def boundary_loop(cells: list) -> tuple | None:
    """The one boundary loop (CCW, region on the left) of a set of cells, or None (several loops, a pinch)."""
    fw = {(c[i], c[(i + 1) % len(c)]) for c in cells for i in range(len(c))}
    nxt = {}
    for a, b in fw:
        if (b, a) not in fw:
            if a in nxt:
                return None
            nxt[a] = b
    if not nxt:
        return None
    s0 = min(nxt)
    lp, x = [s0], nxt[s0]
    while x != s0:
        if x not in nxt or len(lp) > len(nxt):
            return None
        lp.append(x)
        x = nxt[x]
    if len(lp) != len(nxt):
        return None
    return tuple(lp)


def _half_edges(cells) -> list:
    return [(c[i], c[(i + 1) % len(c)]) for c in cells for i in range(len(c))]


def _convex_ok(P: np.ndarray, cells) -> bool:
    """Every triangle CCW and every quad strictly convex (all corner triple products > 0) at the positions P."""
    for c in cells:
        x = P[list(c)]
        k = len(c)
        for i in range(k if k == 4 else 1):
            a, b, d = x[i], x[(i + 1) % k], x[i - 1]
            if np.dot(np.cross(b - a, d - a), a) <= 0:
                return False
    return True


# ------------------------------------------------------------------------------------------------ working mesh

def _subset_counts(P: np.ndarray, el: np.ndarray, nv: np.ndarray, is_c: np.ndarray, c3: np.ndarray | None, h,
                   mo: dict) -> dict:
    """Bad edges (|ln s/h| > bad_edge, the edges of the measured elements) and size jumps (element size ratio >
    bad_ratio, the interior edges with a measured element on a side) as repair._measure counts them, the number of
    such edges and pairs, and the same restricted to the coast3 subset (an element on either side with a c3 node)."""
    from meshom.geo import R_EARTH_M, arc_length
    from meshom.repair import _half_edges as he

    n = len(P)
    a, b, e = he(el, nv)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    uk, inv, cnt = np.unique(lo * n + hi, return_inverse=True, return_counts=True)
    eu = np.column_stack([uk // n, uk % n])
    length = R_EARTH_M * arc_length(P[eu[:, 0]], P[eu[:, 1]])
    ec3 = np.zeros(len(el), dtype=bool) if c3 is None else np.any((el >= 0) & c3[np.maximum(el, 0)], axis=1)
    cav = np.zeros(len(uk), dtype=bool)
    cav[inv[is_c[e]]] = True
    on3 = np.zeros(len(uk), dtype=bool)
    on3[inv[ec3[e]]] = True
    mid = P[eu[:, 0]] + P[eu[:, 1]]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    ci = np.flatnonzero(cav)
    bad = np.abs(np.log(length[ci] / h(mid[ci]))) > mo["bad_edge"]
    mean_edge = np.bincount(e, length[inv], minlength=len(el)) / nv
    order = np.argsort(inv, kind="stable")
    si = inv[order]
    first = np.flatnonzero(np.r_[True, si[1:] != si[:-1]])
    two = cnt[si[first]] == 2
    ea, eb = e[order[first[two]]], e[order[first[two] + 1]]
    pc = is_c[ea] | is_c[eb]
    ma, mb = mean_edge[ea[pc]], mean_edge[eb[pc]]
    jump = np.maximum(ma, mb) / np.minimum(ma, mb) > mo["bad_ratio"]
    p3 = (ec3[ea] | ec3[eb])[pc]
    inner = cnt[ci] == 2
    return {"n_edge": int(len(ci)), "n_pair": int(pc.sum()),
            "bad_edge_c3": int(np.sum(bad & on3[ci] & inner)), "n_edge_c3": int(np.sum(on3[ci] & inner)),
            "bad_ratio_c3": int(np.sum(jump & p3)), "n_pair_c3": int(p3.sum())}


class _Work:
    """Mutable mixed mesh: padded cells el (cap,4), corner counts nv (0 = removed), zone flags, node -> element
    table inc (N, K) with counts ninc, positions P; the boundary edges (never change), fixed and boundary nodes."""

    def __init__(self, mesh: Mesh, fixed: np.ndarray | None = None, frozen: np.ndarray | None = None):
        N, T, Q = mesh.n_nodes, mesh.n_tris, mesh.n_quads
        M = T + Q
        cap = M + max(1024, M // 10)
        self.el = np.full((cap, 4), -1, dtype=np.int64)
        self.el[:T, :3] = mesh.tris
        self.el[T:M] = mesh.quads
        self.nv = np.zeros(cap, dtype=np.int64)
        self.nv[:T], self.nv[T:M] = 3, 4
        self.zone = np.zeros(cap, dtype=bool)
        if mesh.tri_zone is not None:
            self.zone[:T] = np.asarray(mesh.tri_zone, dtype=bool)
        self.frozen = None                                  # R24: frozen cells (never in an operation's rm)
        if frozen is not None and np.any(frozen):
            self.frozen = np.zeros(cap, dtype=bool)
            self.frozen[:M] = np.asarray(frozen, dtype=bool)
        self.m = M
        flat = self.el[:M].ravel()
        ok = flat >= 0
        nodes, elems = flat[ok], np.repeat(np.arange(M, dtype=np.int64), 4)[ok]
        order = np.argsort(nodes, kind="stable")
        ns, es = nodes[order], elems[order]
        cnt = np.bincount(nodes, minlength=N)
        start = np.r_[0, np.cumsum(cnt)[:-1]]
        self.K = int(max(cnt.max(initial=0), 16)) + 8
        self.inc = np.full((N, self.K), -1, dtype=np.int64)
        self.inc[ns, np.arange(len(ns)) - start[ns]] = es
        self.ninc = cnt.astype(np.int64)
        self.P = np.ascontiguousarray(mesh.points, dtype=np.float64).copy()
        topo = build_topology(mesh)
        self.bnode = topo.boundary_node.copy()
        be = topo.edges[topo.edge_elems[:, 1] < 0]
        self.bedges = {(int(a), int(b)) for a, b in be}
        self.fixed = self.bnode | (np.asarray(mesh.node_flag) != 0)
        if fixed is not None:
            f = np.asarray(fixed)
            if f.dtype != bool:
                g = np.zeros(N, dtype=bool)
                g[f] = True
                f = g
            self.fixed |= f
        self.alive = np.ones(N, dtype=bool)
        self.touched = np.zeros(N, dtype=bool)
        self.quad_dom = Q >= T
        self.node_flag = mesh.node_flag
        self._loc = np.full(N, -1, dtype=np.int64)
        self.c3 = None                                      # node mask of the gates' _coast3 rows (set by _Engine)

    # -------------------------------------------------------------------------------------------- queries
    def cells(self, e: int) -> tuple:
        return tuple(self.el[e, : self.nv[e]].tolist())

    def at(self, v: int) -> list:
        return self.inc[v, : self.ninc[v]].tolist()

    def edge_elems(self, a: int, b: int) -> list:
        out = []
        for e in self.at(a):
            c = self.cells(e)
            i = c.index(a)
            if c[(i + 1) % len(c)] == b or c[i - 1] == b:
                out.append(e)
        return out

    def nbrs(self, e: int) -> list:
        """Edge neighbours of element e."""
        c = self.cells(e)
        out = []
        for i in range(len(c)):
            for x in self.edge_elems(c[i], c[(i + 1) % len(c)]):
                if x != e:
                    out.append(x)
        return out

    def is_bedge(self, a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in self.bedges

    def free(self, v: int) -> bool:
        return not self.fixed[v]

    def deg(self, v: int) -> int:
        """Edges at v (= elements for an interior node)."""
        s = set()
        for e in self.at(v):
            c = self.cells(e)
            i = c.index(v)
            s.add(c[(i + 1) % len(c)])
            s.add(c[i - 1])
        return len(s)

    def rings(self, nodes, r: int) -> set:
        """Nodes within r element rings of `nodes` (ring 0 = the nodes themselves)."""
        cur = set(int(v) for v in nodes)
        front = set(cur)
        for _ in range(r):
            nxt = set()
            for v in front:
                for e in self.at(v):
                    nxt.update(self.cells(e))
            front = nxt - cur
            cur |= nxt
        return cur

    def tri_nbrs(self, t: int) -> list:
        return [x for x in self.nbrs(t) if self.nv[x] == 3 and not self.zone[x]]

    def cluster_excess(self, seeds, cap: int = 64) -> int:
        """Triangles (outside the zones) in clusters of >= 3 reachable from the seed triangles (each cluster counted
        once, at most cap)."""
        seen, total = set(), 0
        for t in seeds:
            if t in seen or self.nv[t] != 3 or self.zone[t]:
                continue
            comp, i = [t], 0
            seen.add(t)
            while i < len(comp) and len(comp) < cap:
                for y in self.tri_nbrs(comp[i]):
                    if y not in seen:
                        seen.add(y)
                        comp.append(y)
                i += 1
            if len(comp) >= 3:
                total += len(comp)
        return total

    def tri_clusters(self) -> list[list[int]]:
        """Edge-connected clusters of >= 3 triangles outside the zones, each a sorted id list, sorted by first id."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        ids = np.flatnonzero((self.nv[: self.m] == 3) & ~self.zone[: self.m])
        if len(ids) < 3:
            return []
        nbr = adjacency(self.el[ids, :3])
        t_, k_ = np.nonzero(nbr > np.arange(len(ids))[:, None])
        a, b = t_, nbr[t_, k_]
        g = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(ids), len(ids)))
        _, lab = connected_components(g, directed=False)
        size = np.bincount(lab)
        out = []
        for c in np.flatnonzero(size >= 3):
            out.append(sorted(ids[lab == c].tolist()))
        return sorted(out)

    # ------------------------------------------------------------------------------------------- changes
    def _remove(self, e: int) -> None:
        for v in self.el[e, : self.nv[e]]:
            row = self.inc[v]
            k = int(self.ninc[v])
            i = int(np.flatnonzero(row[:k] == e)[0])
            row[i] = row[k - 1]
            row[k - 1] = -1
            self.ninc[v] = k - 1
        self.nv[e] = 0

    def _put(self, e: int, cell: tuple, zone: bool) -> None:
        self.el[e] = -1
        self.el[e, : len(cell)] = cell
        self.nv[e] = len(cell)
        self.zone[e] = zone
        for v in cell:
            self.inc[v, self.ninc[v]] = e
            self.ninc[v] += 1

    def _grow(self, k: int) -> None:
        if self.m + k <= len(self.nv):
            return
        cap = 2 * len(self.nv) + k
        el = np.full((cap, 4), -1, dtype=np.int64)
        el[: self.m] = self.el[: self.m]
        nv = np.zeros(cap, dtype=np.int64)
        nv[: self.m] = self.nv[: self.m]
        zone = np.zeros(cap, dtype=bool)
        zone[: self.m] = self.zone[: self.m]
        self.el, self.nv, self.zone = el, nv, zone
        if self.frozen is not None:
            frz = np.zeros(cap, dtype=bool)
            frz[: self.m] = self.frozen[: self.m]
            self.frozen = frz

    def apply(self, rm: list, new: list, moves: dict | None = None, zone: bool = False):
        """Replace the elements rm by the cells new, move nodes (moves: node -> position); the undo record, or None
        when a node would exceed the table width K."""
        d = {}
        for e in rm:
            for v in self.el[e, : self.nv[e]].tolist():
                d[v] = d.get(v, 0) - 1
        for c in new:
            for v in c:
                d[v] = d.get(v, 0) + 1
        if any(self.ninc[v] + k > self.K for v, k in d.items()):
            return None
        old = [(e, self.cells(e), bool(self.zone[e])) for e in rm]
        for e in rm:
            self._remove(e)
        self._grow(len(new))
        ids = []
        for c in new:
            self._put(self.m, c, zone)
            ids.append(self.m)
            self.m += 1
        mv = []
        for v, x in (moves or {}).items():
            mv.append((v, self.P[v].copy()))
            self.P[v] = x
        gone = [v for v in d if self.ninc[v] == 0]
        self.alive[gone] = False
        return old, ids, mv, gone

    def undo(self, rec) -> None:
        old, ids, mv, gone = rec
        for e in reversed(ids):
            self._remove(e)
        if ids and ids[-1] == self.m - 1:
            self.m = ids[0]
        for e, c, z in old:
            self._put(e, c, z)
        for v, x in mv:
            self.P[v] = x
        self.alive[gone] = True

    def mark(self, rec) -> None:
        old, ids, mv, gone = rec
        for e in ids:
            self.touched[self.el[e, : self.nv[e]]] = True
        for _, c, _ in old:
            self.touched[list(c)] = True
        for v, _ in mv:
            self.touched[v] = True

    # ------------------------------------------------------------------------------------------- validity
    def patch_ok(self, rm: list, new: list) -> bool:
        """The patch rules of the module docstring (validity of replacing rm by new)."""
        if self.frozen is not None and any(self.frozen[e] for e in rm):
            return False
        old = [self.cells(e) for e in rm]
        if any(len(set(c)) != len(c) for c in new):
            return False
        ho, hn = _half_edges(old), _half_edges(new)
        so, sn = set(ho), set(hn)
        if len(sn) != len(hn):
            return False
        if {e for e in ho if (e[1], e[0]) not in so} != {e for e in hn if (e[1], e[0]) not in sn}:
            return False
        vo = {v for c in old for v in c}
        vn = {v for c in new for v in c}
        if not vn <= vo:
            return False
        rms = set(rm)
        for v in vo - vn:                                   # removed nodes: free, interior, whole star replaced
            if self.fixed[v] or self.bnode[v] or any(e not in rms for e in self.at(v)):
                return False
        eo = {(min(a, b), max(a, b)) for a, b in ho}
        en = {(min(a, b), max(a, b)) for a, b in hn}
        if len(vo) - len(eo) + len(old) != len(vn) - len(en) + len(new):
            return False
        for a, b in en - eo:                                # a new edge must not exist outside the patch
            if any(x not in rms for x in self.edge_elems(a, b)):
                return False
        return True

    # ------------------------------------------------------------------------------------------- measures
    def measure(self, Z, h, mo: dict) -> dict | None:
        """repair._measure of the elements at the nodes Z (with their first ring) plus: cl (triangles in clusters >= 3
        reachable from the region), bnd2_tri / bnd2_quad (elements with >= 2 boundary edges), worst (the largest
        corner deviation from 60 / 90 deg), bad_edge_c3 / bad_ratio_c3 (bad edges and size jumps of the gates'
        _coast3 subset: an element on either side with a node of the coast3 mask), and the counts of the edges
        and element pairs measured (n_edge, n_pair, n_edge_c3, n_pair_c3). Z = None: the whole mesh (cl, bnd2 and
        worst not computed)."""
        from meshom.repair import _measure

        if Z is None:
            C = np.flatnonzero(self.nv[: self.m] > 0).tolist()
            ring = []
        else:
            Zl = [v for v in Z if self.alive[v]]
            C = sorted({e for v in Zl for e in self.at(v)})
            if not C:
                return None
        cl = np.unique(self.el[C][self.el[C] >= 0])
        if Z is not None:
            ring = sorted({e for v in cl.tolist() for e in self.at(v)} - set(C))
        rn = np.unique(self.el[ring][self.el[ring] >= 0]) if ring else np.zeros(0, dtype=np.int64)
        rn = rn[~np.isin(rn, cl)]
        g = np.r_[cl, rn].astype(np.int64)
        self._loc[g] = np.arange(len(g))
        ids = C + ring
        el = self.el[ids].copy()
        okm = el >= 0
        el[okm] = self._loc[el[okm]]
        self._loc[g] = -1
        nv = self.nv[ids]
        is_c = np.r_[np.ones(len(C), dtype=bool), np.zeros(len(ring), dtype=bool)]
        P = self.P[g]
        r = _measure(P, el, nv, is_c, len(cl), self.bnode[g], h, mo, self.quad_dom)
        r.update(_subset_counts(P, el, nv, is_c, self.c3[g] if self.c3 is not None else None, h, mo))
        if Z is None:
            r.update(cl=0, bnd2_tri=0, bnd2_quad=0, worst=0.0)
            return r
        r["cl"] = self.cluster_excess([e for e in ids if self.nv[e] == 3])
        b2t = b2q = 0
        for e in C:
            c = self.cells(e)
            nb = sum(self.is_bedge(c[i], c[(i + 1) % len(c)]) for i in range(len(c)))
            if nb >= 2:
                b2t += len(c) == 3
                b2q += len(c) == 4
        r["bnd2_tri"], r["bnd2_quad"] = b2t, b2q
        worst = 0.0
        for k in (3, 4):
            sel = [e for e in C if self.nv[e] == k]
            if sel:
                a = np.degrees(_angles(tangent_coords(self.P[self.el[sel, :k]])))
                worst = max(worst, float(np.abs(a - (60.0 if k == 3 else 90.0)).max()))
        r["worst"] = worst
        return r

    def relax(self, M, h, o: dict) -> dict:
        """Relax the free nodes M (meshom.smooth "size" then "angle" on the elements at them, the rest fixed);
        returns node -> new position (positions in P updated)."""
        from meshom.smooth import smooth

        M = sorted(v for v in M if self.alive[v] and not self.fixed[v])
        if not M:
            return {}
        E = sorted({e for v in M for e in self.at(v)})
        nodes = np.unique(self.el[E][self.el[E] >= 0])
        self._loc[nodes] = np.arange(len(nodes))
        el = self.el[E]
        nv = self.nv[E]
        tl = self._loc[el[nv == 3, :3]]
        ql = self._loc[el[nv == 4]]
        mloc = self._loc[np.asarray(M)]
        self._loc[nodes] = -1
        fx = np.ones(len(nodes), dtype=bool)
        fx[mloc] = False
        m = Mesh(self.P[nodes], tl.reshape(-1, 3), ql.reshape(-1, 4), np.zeros(len(nodes), dtype=np.int8))
        if o["relax_size"]:
            m = smooth(m, h, fx, iters=int(o["relax_size"]), kind="size")
        if o["relax_angle"]:
            m = smooth(m, h, fx, iters=int(o["relax_angle"]), kind="angle")
        out = {}
        for v, i in zip(M, mloc.tolist()):
            out[v] = m.points[i].copy()
            self.P[v] = m.points[i]
        return out

    def fit(self, M, h, tol: float, fit_opts: dict) -> dict:
        """size_fit.smooth_fit at tol with only the free nodes M moving, on the elements at the nodes of the elements
        at M (the same energy as on the whole mesh, _fit_sub); returns node -> new position (P updated)."""
        from meshom.size_fit import smooth_fit

        M = sorted(v for v in M if self.alive[v] and not self.fixed[v])
        if not M:
            return {}
        act = sorted({e for v in M for e in self.at(v)})
        ring = {v for e in act for v in self.cells(e)}
        E = sorted({e for v in ring for e in self.at(v)})
        nodes = np.unique(self.el[E][self.el[E] >= 0])
        self._loc[nodes] = np.arange(len(nodes))
        el, nv = self.el[E], self.nv[E]
        tl, ql = self._loc[el[nv == 3, :3]], self._loc[el[nv == 4]]
        mloc = self._loc[np.asarray(M)]
        self._loc[nodes] = -1
        free = np.zeros(len(nodes), dtype=bool)
        free[mloc] = True
        sub = Mesh(self.P[nodes], tl.reshape(-1, 3), ql.reshape(-1, 4), np.zeros(len(nodes), dtype=np.int8))
        out, _ = smooth_fit(sub, h, tol, ~free, **fit_opts)
        res = {}
        for v, i in zip(M, mloc.tolist()):
            res[v] = out.points[i].copy()
            self.P[v] = out.points[i]
        return res

    # ------------------------------------------------------------------------------------------- output
    def to_mesh(self) -> tuple[Mesh, np.ndarray, np.ndarray]:
        """(the compact mesh, old node id per new node, touched mask of the new nodes); surviving nodes keep their
        order (boundary nodes stay first)."""
        live = self.nv[: self.m] > 0
        el, nv, zone = self.el[: self.m][live], self.nv[: self.m][live], self.zone[: self.m][live]
        used = np.zeros(len(self.P), dtype=bool)
        used[el[el >= 0]] = True
        nid = np.cumsum(used) - 1
        tris = nid[el[nv == 3, :3]].reshape(-1, 3)
        quads = nid[el[nv == 4]].reshape(-1, 4)
        old = np.flatnonzero(used)
        return (Mesh(self.P[old], tris.astype(np.int64), quads.astype(np.int64), np.asarray(self.node_flag)[old],
                     zone[nv == 3].copy() if zone.any() else None), old, self.touched[old])


# ------------------------------------------------------------------------------------------------ acceptance

def accept(old: dict, new: dict | None, mode: str, o: dict, room: dict | None = None, counts: bool = True) -> bool:
    """The local acceptance of the module docstring; room: how many size jumps / bad edges (bad_ratio, bad_edge, and
    their _coast3 counts bad_ratio_c3, bad_edge_c3) the operation may add (the budget left to a cluster operation;
    None = none); counts=False: those four counts are not checked (a near miss, Z13)."""
    room = room or {}
    if new is None or new["inverted"] > 0 or new["net"] != old["net"]:
        return False
    if any(new["hard"][k] > old["hard"][k] for k in old["hard"]):
        return False
    if new["bnd2_tri"] > old["bnd2_tri"] or new["bnd2_quad"] > old["bnd2_quad"]:
        return False
    if new["bad_elem"] > old["bad_elem"] or new["bad_ratio_coast"] > old["bad_ratio_coast"]:
        return False
    if counts and any(new[k] > old[k] + max(0, room.get(k, 0))
                      for k in ("bad_ratio", "bad_edge", "bad_ratio_c3", "bad_edge_c3")):
        return False
    if mode == "merge":
        return new["T"] < old["T"] and new["cl"] <= old["cl"] and new["gross"] <= old["gross"]
    if mode == "cluster":
        return new["cl"] < old["cl"] and new["gross"] <= old["gross"] + int(o["gross_slack"])
    if mode == "valence":
        return new["gross"] <= old["gross"] - 2 and new["T"] <= old["T"] and new["cl"] <= old["cl"]
    raise KeyError(mode)


def _rank(m: dict) -> tuple:
    return (m["cl"], m["bad_elem"], m["bad_edge"] + m["bad_ratio"], round(m["worst"], 3), m["gross"], m["T"])


class _Engine:
    """The topology steps 2-4 on a working mesh."""

    def __init__(self, W: _Work, h, tol: float, o: dict, mesh: Mesh):
        self.W, self.h, self.o, self.tol = W, h, o, float(tol)
        self.mo = _measure_opts(o, tol)
        self.lim = (float(o["quad_max"]) - 90.0) / 90.0
        self.stats = {k: {"tried": 0, "valid": 0, "accepted": 0} for k in ("merge", "cluster", "valence")}
        # the budget (Z11): cluster operations may add size jumps and bad edges while the whole mesh and its _coast3
        # subset keep below budget_share of their edges (when the input is below it; else no growth), at most
        # budget_op per operation
        d = np.where(W.bnode, 0, 3)                             # node hops from the boundary, capped at 3
        te = build_topology(mesh).edges
        for k in range(1, 3):
            near = (d[te[:, 0]] == k - 1) | (d[te[:, 1]] == k - 1)
            hit = np.zeros(len(d), dtype=bool)
            hit[te[near].ravel()] = True
            d = np.where(hit & (d > k), k, d)
        W.c3 = d <= 2                                           # the gates' _coast3 rows (metrics.element_regions)
        g = W.measure(None, h, self.mo)
        share = float(o["budget_share"])
        self.count = {k: g[k] for k in ("bad_ratio", "bad_edge", "bad_ratio_c3", "bad_edge_c3")}
        self.count0 = dict(self.count)
        size = {"bad_ratio": g["n_pair"], "bad_edge": g["n_edge"], "bad_ratio_c3": g["n_pair_c3"],
                "bad_edge_c3": g["n_edge_c3"]}
        self.budget = {k: max(self.count[k], int(share * size[k])) for k in self.count}

    def dgross(self, rm, new) -> int:
        """Exact change of repair._measure's gross charge when rm is replaced by new (positions play no part): the
        valence terms of the patch's non-boundary nodes (removed nodes drop out) plus the triangles of a
        quad-dominant mesh."""
        W = self.W
        old = [W.cells(e) for e in rm]

        def inner(cells):
            he = set(_half_edges(cells))
            return {(min(a, b), max(a, b)) for a, b in he if (b, a) in he}

        eo, en = inner(old), inner(new)
        nodes = {v for c in old for v in c}
        dq = {}
        for sgn, cells in ((-1, old), (1, new)):
            for c in cells:
                if len(c) == 4:
                    for v in c:
                        dq[v] = dq.get(v, 0) + sgn
        dd = {}
        for a, b in eo - en:
            dd[a], dd[b] = dd.get(a, 0) - 1, dd.get(b, 0) - 1
        for a, b in en - eo:
            dd[a], dd[b] = dd.get(a, 0) + 1, dd.get(b, 0) + 1
        vn = {v for c in new for v in c}
        out = 0
        for v in nodes:
            if W.bnode[v]:
                continue
            d0 = len(self._adjacent(v))
            q0 = sum(W.nv[e] == 4 for e in W.at(v))
            out -= abs(d0 - (4 if q0 else 6))
            if v in vn:
                d1, q1 = d0 + dd.get(v, 0), q0 + dq.get(v, 0)
                out += abs(d1 - (4 if q1 else 6))
        if W.quad_dom:
            out += sum(len(c) == 3 for c in new) - sum(len(c) == 3 for c in old)
        return out

    def _eval(self, rm, new, Z, old, mode, relaxM=None, fit=False, near=None):
        """Apply, relax the nodes relaxM (meshom.smooth; fit=True: size_fit.smooth_fit at the tolerance, Z13),
        measure, undo: (rank, moves, count changes) when accepted, else None; `near` (a list) receives the rank of a
        candidate that fails only on the size-jump / bad-edge counts."""
        W = self.W
        st = self.stats[mode]
        st["tried"] += 1
        lim = {"merge": 0, "cluster": int(self.o["gross_slack"]), "valence": -2}[mode]
        if self.dgross(rm, new) > lim:                            # exact: the gross rule fails whatever moves
            return None
        if not W.patch_ok(rm, new):
            return None
        if relaxM is None and not _convex_ok(W.P, new):
            return None
        st["valid"] += 1
        rec = W.apply(rm, new)
        if rec is None:
            return None
        if fit:
            moves = W.fit(relaxM, self.h, self.tol, dict(self.o["fit_opts"]))
        else:
            moves = W.relax(relaxM, self.h, self.o) if relaxM else {}
        nm = W.measure(Z, self.h, self.mo)
        room = None
        if mode == "cluster":
            cap = int(self.o["budget_op"])
            room = {k: min(cap, self.budget[k] - self.count[k]) for k in self.count}
        ok = accept(old, nm, mode, self.o, room)
        if not ok and near is not None and accept(old, nm, mode, self.o, room, counts=False):
            near.append(_rank(nm))
        W.undo(rec)                                               # the relaxed positions: restored by the caller
        if not ok:
            return None
        return _rank(nm), moves, {k: nm[k] - old[k] for k in self.count}

    def _commit(self, rm, new, moves, mode, delta: dict | None = None) -> None:
        rec = self.W.apply(rm, new, moves)
        self.W.mark(rec)
        self.stats[mode]["accepted"] += 1
        for k, v in (delta or {}).items():
            self.count[k] += v

    def _best(self, cands, mode) -> bool:
        """cands: [(rm, new, Z)]: the best accepted one (unrelaxed first, then relaxed) applied; True if one was."""
        W, o = self.W, self.o
        olds = {}
        near = []                                           # (rank, index) of the relaxed near misses
        phases = ["as is"] + (["relaxed"] if o["relax_rings"] else [])
        if mode == "cluster" and o["relax_rings"] and int(o["fit_try"]) > 0:
            phases.append("fit")
        for phase in phases:
            best = None
            if phase == "fit":
                todo = [i for _, i in sorted(near)[: int(o["fit_try"])]]
            else:
                todo = range(len(cands))
            for i in todo:
                rm, new, Z, M = cands[i]
                key = (tuple(rm), frozenset(Z))
                if key not in olds:
                    olds[key] = W.measure(Z, self.h, self.mo)
                old = olds[key]
                if old is None:
                    continue
                saved = {v: W.P[v].copy() for v in M} if phase != "as is" else None
                nr = [] if phase == "relaxed" and mode == "cluster" else None
                r = self._eval(rm, new, Z, old, mode, M if phase != "as is" else None, fit=phase == "fit", near=nr)
                if nr:
                    near.append((nr[0], i))
                if saved:
                    for v, x in saved.items():
                        W.P[v] = x
                if r is not None and (best is None or (r[0], i) < (best[0][0], best[1])):
                    best = (r, i)
            if best is not None:
                rm, new, Z, M = cands[best[1]]
                self._commit(rm, new, best[0][1], mode, best[0][2])
                if phase == "fit":
                    self.stats[mode]["by_fit"] = self.stats[mode].get("by_fit", 0) + 1
                return True
        return False

    def _region(self, rm) -> tuple[set, set]:
        """(Z, M): the nodes of rm plus the free nodes within relax_rings rings, and those free nodes."""
        W = self.W
        n0 = {v for e in rm for v in W.cells(e)}
        R = W.rings(n0, int(self.o["relax_rings"]))
        M = {v for v in R if not W.fixed[v]}
        return n0 | M, M

    def _resplits(self, S: list) -> list:
        """Candidates (rm, new, Z, M) re-splitting the union of the elements S."""
        W = self.W
        if any(W.zone[e] for e in S):
            return []
        cells = [W.cells(e) for e in S]
        lp = boundary_loop(cells)
        if lp is None or len(lp) > int(self.o["poly_max"]) or len(lp) < 3:
            return []
        cur = {frozenset(c) for c in cells}
        Z, M = self._region(S)
        out = []
        for split in poly_splits(lp, len(lp) % 2):
            if {frozenset(c) for c in split} == cur:
                continue
            out.append((list(S), list(split), Z, M))
        return out

    # ------------------------------------------------------------------------------------------- steps
    def merge(self) -> int:
        """Step 2: greedy pair merges, cheapest quad first."""
        from meshom.recombine import _merge, _quad_cost

        W = self.W
        ids = np.flatnonzero((W.nv[: W.m] == 3) & ~W.zone[: W.m])
        if len(ids) < 2:
            return 0
        tris = W.el[ids, :3]
        nbr = adjacency(tris)
        t_, k_ = np.nonzero(nbr > np.arange(len(ids))[:, None])
        a, b = t_, nbr[t_, k_]
        q = _merge(tris[a], tris[b])
        cost = _quad_cost(W.P, q)
        sel = np.flatnonzero(cost < self.lim)
        n = 0
        for i in sel[np.argsort(cost[sel], kind="stable")]:
            ta, tb = int(ids[a[i]]), int(ids[b[i]])
            if W.nv[ta] != 3 or W.nv[tb] != 3:
                continue
            quad = tuple(int(x) for x in q[i])
            if set(quad) != set(W.cells(ta)) | set(W.cells(tb)):
                continue
            Z = {v for e in (ta, tb) for v in W.cells(e)}
            old = W.measure(Z, self.h, self.mo)
            r = self._eval([ta, tb], [quad], Z, old, "merge")
            if r is not None:
                self._commit([ta, tb], [quad], {}, "merge", r[2])
                n += 1
        return n

    def clusters(self) -> int:
        """Step 3: one accepted operation per cluster (the best), clusters taken in id order."""
        W, o = self.W, self.o
        n = 0
        for cl in W.tri_clusters():
            if any(W.nv[t] != 3 for t in cl):                   # changed by an earlier cluster's operation
                continue
            if len(cl) > int(o["cluster_size_max"]):
                self.stats["cluster"]["skipped"] = self.stats["cluster"].get("skipped", 0) + 1
                continue
            sets = set()
            for t in cl:
                for e in W.nbrs(t):
                    if not W.zone[e]:
                        sets.add(frozenset((t, e)))
            if int(o["cluster_sets"]) >= 3:
                for s in list(sets):
                    for x in s:
                        for e in W.nbrs(x):
                            if e not in s and not W.zone[e]:
                                sets.add(s | {e})
            if len(cl) <= 4:
                sets.add(frozenset(cl))
                for t in cl:
                    for e in W.nbrs(t):
                        if e not in cl and not W.zone[e]:
                            sets.add(frozenset(cl) | {e})
            for v in sorted({v for t in cl for v in W.cells(t)}):  # the star of a free interior node
                if W.fixed[v]:
                    continue
                star = W.at(v)
                if len(star) <= int(o["poly_max"]) and not any(W.zone[e] for e in star):
                    sets.add(frozenset(star))
            cands = []
            for s in sorted(sets, key=lambda x: sorted(x)):
                cands += self._resplits(sorted(s))
            if cands and self._best(cands, "cluster"):
                n += 1
        return n

    def valence(self) -> int:
        """Step 4: around interior free all-quad nodes with valence != 4."""
        W = self.W
        n = 0
        inc = W.inc
        ntri = ((inc >= 0) & (W.nv[np.maximum(inc, 0)] == 3)).sum(axis=1)
        irr = np.flatnonzero(~W.fixed & W.alive & (W.ninc != 4) & (W.ninc > 0) & (ntri == 0)).tolist()
        for v in irr:
            if not W.alive[v] or W.fixed[v]:
                continue
            star = W.at(v)
            if not star or any(W.nv[e] != 4 for e in star) or len(star) == 4:
                continue
            cands = []
            if len(star) <= 3:                                  # node removal (valence 2 or 3)
                cands += self._resplits(sorted(star))
            for q in star if len(star) == 3 else ():        # diagonal collapse of two valence-3 nodes: v <- w
                c = W.cells(q)
                i = c.index(v)
                w, b, d = c[(i + 2) % 4], c[(i + 1) % 4], c[(i + 3) % 4]
                if W.fixed[w] or W.ninc[w] != 3 or any(W.nv[e] != 4 for e in W.at(w)):
                    continue
                adj_v, adj_w = self._adjacent(v), self._adjacent(w)
                if adj_v & adj_w != {b, d} or w in adj_v:
                    continue
                rm = sorted(set(star) | set(W.at(w)))
                new = [tuple(v if x == w else x for x in W.cells(e)) for e in rm if e != q]
                Z, M = self._region(rm)
                cands.append((rm, new, Z, M | {v}))
            for x in sorted(self._adjacent(v)):                 # edge rotation of the two quads at (v, x)
                ee = W.edge_elems(v, x)
                if len(ee) == 2 and all(W.nv[e] == 4 for e in ee):
                    cands += self._resplits(sorted(ee))
            if cands and self._best(cands, "valence"):
                n += 1
        return n

    def _adjacent(self, v: int) -> set:
        s = set()
        for e in self.W.at(v):
            c = self.W.cells(e)
            i = c.index(v)
            s.add(c[(i + 1) % len(c)])
            s.add(c[i - 1])
        return s


# ------------------------------------------------------------------------------------------------ summaries

def summary(mesh: Mesh) -> dict:
    """Counts reported before / after: triangles, quads, triangle share, the largest triangle cluster outside the
    zones and the clusters >= 3, irregular nodes (valence 3 / 5 / other of interior nodes, compete.irregular_counts),
    the Euler sum."""
    from meshom.compete import irregular_counts
    from meshom.quality.metrics import _components

    T = mesh.n_tris
    out = {"n_nodes": mesh.n_nodes, "n_tris": T, "n_quads": mesh.n_quads,
           "tri_share": T / max(mesh.n_elems, 1)}
    if T:
        zone = np.zeros(T, dtype=bool) if mesh.tri_zone is None else np.asarray(mesh.tri_zone, dtype=bool)
        nbr = adjacency(mesh.tris)
        t_, k_ = np.nonzero(nbr > np.arange(T)[:, None])
        a, b = t_, nbr[t_, k_]
        keep = ~zone[a] & ~zone[b]
        lab = _components(T, a[keep], b[keep])
        size = np.bincount(lab[~zone], minlength=T)
        out["tri_cluster_max"] = int(size.max()) if (~zone).any() else 0
        out["tri_clusters_ge3"] = int((size >= 3).sum())
    else:
        out["tri_cluster_max"], out["tri_clusters_ge3"] = 0, 0
    ic = irregular_counts(mesh)
    out.update({k: ic[k] for k in ("n3", "n5", "n_other", "euler_sum", "chi")})
    return out


def _boundary_set(mesh: Mesh, ids: np.ndarray | None = None) -> set:
    topo = build_topology(mesh)
    be = topo.edges[topo.edge_elems[:, 1] < 0]
    if ids is not None:
        be = ids[be]
    return {(int(min(a, b)), int(max(a, b))) for a, b in be}


# ------------------------------------------------------------------------------------------------ the stage

def optimize(mesh: Mesh, h, tol: float | None = None, opts: dict | None = None,
             fixed: np.ndarray | None = None, frozen: np.ndarray | None = None) -> tuple[Mesh, dict]:
    """The optimisation stage (module docstring). tol: the run's size tolerance (methods.effective_tolerance; None
    = the size gate 0.15); fixed: extra fixed nodes (mask or ids); frozen: (T+Q,) cells no operation takes, their
    nodes fixed (R24; quads: the flips see triangles only). Returns (mesh, info): info has the options, the
    tolerance used, per-step counts and times, summary() before / after, and the checks (Euler sum and characteristic,
    boundary edges, inverted elements: unchanged)."""
    from meshom.mesh import signed_area

    o = {**DEFAULTS, **(opts or {})}
    t_all = time.perf_counter()
    fx = np.zeros(mesh.n_nodes, dtype=bool)
    if fixed is not None:
        f = np.asarray(fixed)
        if f.dtype == bool:
            fx |= f
        else:
            fx[f] = True
    frz = None
    if frozen is not None and np.any(frozen):
        frz = np.asarray(frozen, dtype=bool)
        fx[mesh.tris[frz[: mesh.n_tris]].ravel()] = True
        fx[mesh.quads[frz[mesh.n_tris:]].ravel()] = True
    tol_used = float(tol) if tol else SIZE_GATE
    steps = tuple(o["steps"])
    info = {"opts": {k: (list(v) if isinstance(v, tuple) else v) for k, v in o.items()}, "tol": tol_used,
            "before": summary(mesh), "steps": {}, "time_s": {}}
    inv0 = int((signed_area(mesh) <= 0).sum())
    b0 = _boundary_set(mesh)
    cur = mesh
    flipped = np.zeros(cur.n_nodes, dtype=bool)                   # nodes of flipped triangles (for the fit)
    if "flip" in steps and cur.n_tris >= 2:
        t0 = time.perf_counter()
        prev = cur.tris
        cur, nf = flip_triangles(cur, o["tri_min"], int(o["flip_passes"]))
        flipped[cur.tris[(cur.tris != prev).any(axis=1)].ravel()] = True
        info["steps"]["flip"] = {"flips": nf}
        info["time_s"]["flip"] = time.perf_counter() - t0
    topo_steps = [s for s in ("merge", "cluster", "valence") if s in steps]
    touched = flipped
    old_id = np.arange(cur.n_nodes)
    if topo_steps and cur.n_quads > 0:
        W = _Work(cur, fx, frz)
        eng = _Engine(W, h, tol_used, o, cur)
        rounds = []
        for _ in range(int(o["passes"])):
            rec = {}
            for s in topo_steps:
                t0 = time.perf_counter()
                rec[s] = getattr(eng, {"merge": "merge", "cluster": "clusters", "valence": "valence"}[s])()
                info["time_s"][s] = info["time_s"].get(s, 0.0) + time.perf_counter() - t0
            rounds.append(rec)
            if not any(rec.values()):
                break
        cur, old_id, touched = W.to_mesh()
        touched = touched | flipped[old_id]
        info["steps"]["topology"] = {"rounds": rounds, **eng.stats, "budget": eng.budget, "count_before": eng.count0,
                                     "count_after": eng.count}
        if "flip" in steps and touched.any() and cur.n_tris >= 2:
            t0 = time.perf_counter()
            prev = cur.tris
            cur, nf = flip_triangles(cur, o["tri_min"], int(o["flip_passes"]))
            touched[cur.tris[(cur.tris != prev).any(axis=1)].ravel()] = True
            info["steps"]["flip_after"] = {"flips": nf}
            info["time_s"]["flip"] = info["time_s"].get("flip", 0.0) + time.perf_counter() - t0
    if "fit" in steps and o["fit"] != "off":
        t0 = time.perf_counter()
        cur, info["steps"]["fit"] = _fit(cur, h, tol_used, o, touched, fx[old_id])
        info["time_s"]["fit"] = time.perf_counter() - t0
    info["after"] = summary(cur)
    b1 = _boundary_set(cur, old_id)
    info["checks"] = {"euler_kept": info["after"]["euler_sum"] == info["before"]["euler_sum"]
                      and info["after"]["chi"] == info["before"]["chi"],
                      "boundary_kept": b1 == b0,
                      "inverted_before": inv0, "inverted_after": int((signed_area(cur) <= 0).sum())}
    info["time_s"]["total"] = time.perf_counter() - t_all
    return cur, info


def _fit(mesh: Mesh, h, tol: float, o: dict, touched: np.ndarray, fixed: np.ndarray) -> tuple[Mesh, dict]:
    """Step 5: smooth_fit on the free nodes near the touched ones (or all), kept unless the region got worse."""
    from meshom.size_fit import smooth_fit

    W = _Work(mesh, fixed)
    if o["fit"] == "all":
        free = ~W.fixed
    else:
        if not touched.any():
            return mesh, {"free_nodes": 0, "kept": False}
        R = W.rings(np.flatnonzero(touched), int(o["fit_rings"]))
        free = np.zeros(mesh.n_nodes, dtype=bool)
        free[list(R)] = True
        free &= ~W.fixed
    if not free.any():
        return mesh, {"free_nodes": 0, "kept": False}
    mo = _measure_opts(o, tol)
    Z = np.flatnonzero(free).tolist()
    old = W.measure(Z, h, mo)
    out, r = _fit_sub(mesh, h, tol, free, dict(o["fit_opts"])) if o["fit"] != "all" else \
        smooth_fit(mesh, h, tol, ~free, **dict(o["fit_opts"]))
    out.tri_zone = mesh.tri_zone
    W.P = np.ascontiguousarray(out.points, dtype=np.float64).copy()
    new = W.measure(Z, h, mo)
    ok = (new["inverted"] == 0 and all(new["hard"][k] <= old["hard"][k] for k in old["hard"])
          and new["bad_elem"] <= old["bad_elem"] and new["bad_edge"] + new["bad_ratio"] <= old["bad_edge"] + old["bad_ratio"])
    rec = {"free_nodes": int(free.sum()), "kept": bool(ok), "tol": tol,
           "region_before": {k: old[k] for k in ("bad_elem", "bad_edge", "bad_ratio", "inverted")},
           "region_after": {k: new[k] for k in ("bad_elem", "bad_edge", "bad_ratio", "inverted")},
           **{k: r[k] for k in ("before", "after", "total_s")}}
    return (out if ok else mesh), rec


def _fit_sub(mesh: Mesh, h, tol: float, free: np.ndarray, fit_opts: dict) -> tuple[Mesh, dict]:
    """smooth_fit with only the free nodes moving, run on the submesh of the elements at the nodes of the elements at
    the free nodes: every term of its energy (edges with a free end, element pairs and corners of the elements with a
    free node, barrier triangles with a free corner; each averaged over its own set) is the same as on the whole
    mesh, at the cost of the region."""
    from meshom.size_fit import smooth_fit

    act_n = np.zeros(mesh.n_nodes, dtype=bool)
    for cells in (mesh.tris, mesh.quads):
        if len(cells):
            act_n[cells[free[cells].any(axis=1)].ravel()] = True
    sel_t = act_n[mesh.tris].any(axis=1) if mesh.n_tris else np.zeros(0, dtype=bool)
    sel_q = act_n[mesh.quads].any(axis=1) if mesh.n_quads else np.zeros(0, dtype=bool)
    nodes = np.unique(np.r_[mesh.tris[sel_t].ravel(), mesh.quads[sel_q].ravel()])
    loc = np.full(mesh.n_nodes, -1, dtype=np.int64)
    loc[nodes] = np.arange(len(nodes))
    sub = Mesh(mesh.points[nodes], loc[mesh.tris[sel_t]].reshape(-1, 3), loc[mesh.quads[sel_q]].reshape(-1, 4),
               np.asarray(mesh.node_flag)[nodes])
    out, r = smooth_fit(sub, h, tol, ~free[nodes], **fit_opts)
    P = mesh.points.copy()
    P[nodes] = out.points
    return Mesh(P, mesh.tris, mesh.quads, mesh.node_flag, mesh.tri_zone), r


def record(info: dict) -> dict:
    """The JSON part of optimize's info kept in the records (quality.json "optimize", stages.json): tolerance,
    before / after counts, checks, times, per-step counts (the fit without its per-round detail)."""
    steps = dict(info["steps"])
    if "fit" in steps:
        f = steps["fit"]
        steps["fit"] = {k: f[k] for k in ("free_nodes", "kept", "tol", "region_before", "region_after") if k in f}
        for k in ("before", "after"):
            if k in f:
                steps["fit"][k] = {x: f[k][x] for x in ("ratio_p99", "ratio_max", "ln_sh_p99", "share_over_tol")
                                   if x in f[k]}
    return {"tol": info["tol"], "before": info["before"], "after": info["after"], "checks": info["checks"],
            "time_s": info["time_s"], "steps": steps}
