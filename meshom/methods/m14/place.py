"""m14 step 2: greedy block placement over the cone-free part of the fitness map, with the acceptance rule.

Seed: the admissible bg node farthest (in units of h) from everything not admissible. Growth: bg triangles in order of
their distance from the seed (Dijkstra on the bg edges; "linf": the L-infinity norm of each edge in the cross-field
frame, so the region grows as a square along the field lines; "metric": the isotropic metric |dx|/h), only
admissible triangles, keeping the region a closed disc (a triangle joins along one edge with its third node outside,
or along two edges). Growth stops when the pre-screen |B| <= min(b_cap, b_slope L) fails (B = integral of rho_c,
L = equivalent-disc diameter in cells) or when the next triangle lies farther from the seed than compact times the
equivalent-disc radius (a square ends at 0.89, a 3:1 rectangle at 1.54: no tails into the corridors left between
earlier blocks). Acceptance: the largest growth prefix whose conformal size correction lambda
(Laplace lambda = -(K + Laplace ln h) on R, lambda = 0 on its boundary; cotangent Laplacian on the bg triangles) has
F = (max - min) / 2 <= f_max and max over the bg triangles of h |grad lambda| <= grad_max (the P1 gradient of lambda
on each chord triangle, h = the mean of its nodes; binary search over the prefix). Regions
smaller than min_cells cells are rejected (their seed neighbourhood is not tried again). An accepted region and its
block_gap dilation leave the admissible set. Fixed blocks (m14/strait.py) come first (B59): every node within
block_gap of one of their nodes (their cells and the gap around them) is not admissible; within band_gap of a node of
a coastal band (m14/coastband.py, C50).

Small regions (`drop_small`, B68): after the greedy pass the regions below drop_small (0.25) times the median region's
cells are dropped; their seeds' neighbourhoods (min_clear) and their nodes may not seed again, and the greedy runs
again (at most DROP_RERUNS times, then the remaining small regions just go). Their area goes to the fill. Not with
given seeds.

Water under a land cut (`water_place`, B70): when fit.wet is set (the land is cut away after the method), seeds are
taken only at wet bg nodes (the greedy's and the given ones) and a region's value is its water cells (Region.water:
each triangle's cells times its share of wet nodes): min_cells and drop_small count it. The growth and the clearance
of a seed ignore the land as before (its cells are cut afterwards; an island inside a block is cut out of it).

Placement region (`place_region`, B106): a spherical polygon ([[lon, lat], ...], great-circle edges, as a size zone's
`polygon`); only bg nodes inside it are admissible, so blocks are seeded and grown only there (the greedy, given
seeds, the shared front, the direct join's corridor) and everything outside goes to the fill. With `max_blocks` 1 and
the zone's polygon of a uniform h it makes one block over that zone (the ORCA025 + Arctic cap).

No cap (`max_blocks` None / 0, B8; m14_blocks_rep's default): the greedy runs until no seed is min_clear from the
non-admissible nodes.

Shared growth front (`front`, B67; `_front_pass`): given seeds ("seeds", the default; True: the greedy's accepted
seeds too) are grown sharing the space: each in its own Voronoi cell of the seeds in units of h, more than block_gap / 2
from the cell boundary, without grow_limit, so the gap between two neighbours is centred on where their fronts meet.
With `front_peer` (6 h, the default; B69, `_peer_pass`) only peers share a front: the seeds grow one after the other,
and a pair whose gap then lies within front_peer of its bisector is kept on the two sides of that bisector (the same
block_gap / 2 rule); a later seed beside a much larger region keeps the order's layout, so the greedy's own seeds
given back give the greedy's layout (the Voronoi cells cut its large first regions: disc coverage 0.88 -> 0.73).

Direct join (`union_region`, B61): two accepted regions and the corridor between them (bg triangles between the two,
admissible, the partition's corridor rules) form one region when the union is a closed disc that passes the same
acceptance rule (F, gradient); m14/sync.py merge decides which pairs are tried.
"""
import heapq
from dataclasses import dataclass, field

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import spsolve

from meshom.geo import R_EARTH_M
from meshom.methods.m14.fitness import Fitness, dist_h


PEER_ROUNDS = 3     # B69: front_peer finds peers again at most this often
DROP_RERUNS = 3     # B68: drop_small reruns the greedy at most this often (then the small regions just go)


@dataclass
class Region:
    tris: np.ndarray         # bg triangle ids (a closed disc)
    nodes: np.ndarray        # bg node ids of those triangles (sorted)
    lam: np.ndarray          # (len(nodes),) conformal size correction, 0 on the region boundary
    F: float                 # (max lam - min lam) / 2
    grad: float              # max over the bg triangles of h |grad lam| (P1 gradient)
    B: float                 # integral of rho_c over the region (cone units)
    cells: float             # area / h^2
    seed: int                # bg node id
    water: float | None = None   # cells in the water of a land cut (B70: triangle cells x wet-node share), None: cells
    info: dict = field(default_factory=dict)

    @property
    def value(self) -> float:
        """The region's cells as the placement counts them: its water cells under a land cut (B70), else all."""
        return self.cells if self.water is None else self.water


def _tri_excess(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    return 2.0 * np.arctan2(det, 1.0 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c)
                            + np.einsum("ij,ij->i", c, a))


def edge_list(T: np.ndarray, n: int) -> np.ndarray:
    e = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    k = np.unique(e[:, 0] * n + e[:, 1])
    return np.column_stack([k // n, k % n])


def edge_cost(P: np.ndarray, E: np.ndarray, u: np.ndarray, hv: np.ndarray, mode: str) -> np.ndarray:
    """Length of each edge in units of h: isotropic ("metric") or the L-infinity norm in the cross frame ("linf",
    the mean of the norms in the frames of both ends)."""
    i, j = E[:, 0], E[:, 1]
    d = P[j] - P[i]
    he = 0.5 * (hv[i] + hv[j])
    if mode == "metric":
        return np.linalg.norm(d, axis=1) * R_EARTH_M / he
    out = np.zeros(len(E))
    for s in (i, j):
        w = np.cross(P[s], u[s])
        out += np.maximum(np.abs(np.einsum("ij,ij->i", d, u[s])), np.abs(np.einsum("ij,ij->i", d, w)))
    return 0.5 * out * R_EARTH_M / he


@njit(cache=True)
def _grow(seed_tri, prio, adm, nbr, tris, n_nodes, q, a, b_cap, b_slope, a_max, compact, a_compact):
    """Disc growth in order of prio; returns (order, cumulative B, cumulative cells) of the added triangles."""
    nt = len(prio)
    in_r = np.zeros(nt, np.bool_)
    cnt = np.zeros(n_nodes, np.int64)
    order = np.empty(nt, np.int64)
    cum_b = np.empty(nt)
    cum_a = np.empty(nt)
    n = 0
    bsum = 0.0
    asum = 0.0
    heap = [(prio[seed_tri], seed_tri)]
    while len(heap) > 0:
        item = heapq.heappop(heap)
        t = item[1]
        if in_r[t]:
            continue
        if n > 0:
            k = 0
            for s in range(3):
                g = nbr[t, s]
                if g >= 0 and in_r[g]:
                    k += 1
            nv = 0
            for s in range(3):
                if cnt[tris[t, s]] > 0:
                    nv += 1
            if not ((k == 1 and nv == 2) or k == 2):            # k == 3 would close the sphere
                continue
        b2 = bsum + q[t]
        a2 = asum + a[t]
        lim = min(b_cap, b_slope * 2.0 * np.sqrt(a2 / np.pi))
        if abs(b2) > lim or a2 > a_max:
            break
        if a2 > a_compact and item[0] > compact * np.sqrt(a2 / np.pi):   # compact: no tails into corridors
            break
        bsum, asum = b2, a2
        in_r[t] = True
        for s in range(3):
            cnt[tris[t, s]] += 1
        order[n] = t
        cum_b[n] = bsum
        cum_a[n] = asum
        n += 1
        for s in range(3):
            g = nbr[t, s]
            if g >= 0 and adm[g] and not in_r[g]:
                heapq.heappush(heap, (prio[g], g))
    return order[:n], cum_b[:n], cum_a[:n]


def cot_laplacian(P: np.ndarray, T: np.ndarray):
    """(N,N) positive semidefinite cotangent Laplacian of the chord triangles (L lam)_i = sum w_ij (lam_i - lam_j)."""
    n = len(P)
    rows, cols, vals = [], [], []
    q = P[T]
    for k in range(3):
        u = q[:, (k + 1) % 3] - q[:, k]
        v = q[:, (k + 2) % 3] - q[:, k]
        w = 0.5 * np.einsum("ij,ij->i", u, v) / np.linalg.norm(np.cross(u, v), axis=1)
        i, j = T[:, (k + 1) % 3], T[:, (k + 2) % 3]
        rows += [i, j, i, j]
        cols += [i, j, j, i]
        vals += [w, w, -w, -w]
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()


def border(E: np.ndarray, adm: np.ndarray) -> np.ndarray:
    """Non-admissible nodes with an admissible neighbour (the nearest non-admissible node is one of them)."""
    i, j = E[:, 0], E[:, 1]
    out = np.zeros(len(adm), dtype=bool)
    x = adm[i] != adm[j]
    out[np.where(adm[i[x]], j[x], i[x])] = True
    return out


def boundary_nodes(T: np.ndarray, n: int) -> np.ndarray:
    """(n,) bool: node on a half-edge without twin."""
    a = T.ravel()
    b = np.roll(T, -1, axis=1).ravel()
    key, rkey = a * n + b, b * n + a
    sk = np.sort(key)
    pos = np.minimum(np.searchsorted(sk, rkey), len(sk) - 1)
    out = np.zeros(n, dtype=bool)
    out[a[sk[pos] != rkey]] = True
    return out


def tri_grad(P: np.ndarray, T: np.ndarray, lam: np.ndarray, hv: np.ndarray) -> np.ndarray:
    """(T,) h |grad lam| per triangle: the gradient of the P1 interpolant of lam on the chord triangle (metres via
    R_EARTH_M), h = the mean of the triangle's node values hv."""
    q = P[T]
    nrm = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
    two_a = np.linalg.norm(nrm, axis=1)
    nh = nrm / two_a[:, None]
    g = np.zeros((len(T), 3))
    for m in range(3):                                   # grad phi_m = n x (edge opposite m) / 2A
        g += lam[T[:, m]][:, None] * np.cross(nh, q[:, (m + 2) % 3] - q[:, (m + 1) % 3])
    return np.linalg.norm(g, axis=1) / two_a / R_EARTH_M * hv[T].mean(axis=1)


def tri_capacity(fit: Fitness) -> tuple[np.ndarray, np.ndarray]:
    """Per bg triangle: q = integral of rho_c (cone units) and a = area in cells (area / h^2, h = mean of the nodes)."""
    P, T, hv = fit.bg.points, fit.bg.tris, fit.hv
    exc = _tri_excess(P, T)
    q = ((fit.rho / np.maximum(fit.area, 1e-300))[T] * (exc / 3.0)[:, None]).sum(axis=1)
    h_t = hv[T].mean(axis=1)
    return q, exc * R_EARTH_M ** 2 / h_t ** 2


def lambda_solve(P: np.ndarray, T: np.ndarray, rhs: np.ndarray, hv: np.ndarray) -> tuple:
    """Dirichlet solve L lam = rhs (lam = 0 on the boundary) on the local mesh (P, T); returns (lam, F, grad), grad =
    max over the triangles of h |grad lam| (tri_grad)."""
    n = len(P)
    bnd = boundary_nodes(T, n)
    inn = np.flatnonzero(~bnd)
    lam = np.zeros(n)
    if len(inn):
        from meshom.field import _AMG

        L = cot_laplacian(P, T)
        A = L[inn][:, inn].tocsr()
        lam[inn] = _AMG(A).solve(rhs[inn], rtol=1e-9) if len(inn) > 5000 else spsolve(A.tocsc(), rhs[inn])
    grad = float(np.max(tri_grad(P, T, lam, hv)))
    return lam, 0.5 * float(lam.max() - lam.min()), grad


def region_lambda(fit: Fitness, tris: np.ndarray) -> tuple:
    """lambda on the region made of the bg triangles `tris`: (nodes, lam, F, grad)."""
    nodes, Tl = np.unique(fit.bg.tris[tris], return_inverse=True)
    Tl = Tl.reshape(-1, 3)
    lam, F, grad = lambda_solve(fit.bg.points[nodes], Tl, (fit.area + fit.lap)[nodes], fit.hv[nodes])
    return nodes, lam, F, grad


def front_cells(G, seeds: list) -> np.ndarray:
    """(n,) index into `seeds` of the seed whose front reaches each node first (one Dijkstra from all seeds at once on
    the edge-length graph G: fronts advancing at the same speed meet on the shared front); -1 unreached."""
    seeds = np.asarray(seeds, dtype=np.int64)
    _, _, src = dijkstra(G, directed=False, indices=seeds, min_only=True, return_predecessors=True)
    pos = np.full(G.shape[0], -1, dtype=np.int64)
    pos[seeds] = np.arange(len(seeds))
    return np.where(src >= 0, pos[np.maximum(src, 0)], -1)


def front_points(P: np.ndarray, E: np.ndarray, lab: np.ndarray, i: int) -> np.ndarray:
    """The shared front of cell i: the midpoints of the bg edges between a node of cell i and a node of another
    seed's cell."""
    a, b = lab[E[:, 0]], lab[E[:, 1]]
    x = ((a == i) & (b >= 0) & (b != i)) | ((b == i) & (a >= 0) & (a != i))
    m = P[E[x, 0]] + P[E[x, 1]]
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def clear_of_fixed(P: np.ndarray, hv: np.ndarray, o: dict, fixed_pts: np.ndarray | None = None,
                   band_pts: np.ndarray | None = None) -> np.ndarray:
    """B59: the nodes P (local h hv) more than block_gap h from every node of the fixed blocks fixed_pts, and (C50)
    more than band_gap h from every node of the coastal bands band_pts (None: block_gap)."""
    ok = np.ones(len(P), dtype=bool)
    if fixed_pts is not None and len(fixed_pts):
        ok &= dist_h(P, fixed_pts, hv) > o["block_gap"]
    if band_pts is not None and len(band_pts):
        ok &= dist_h(P, band_pts, hv) > (o["block_gap"] if o.get("band_gap") is None else o["band_gap"])
    return ok


def in_place_region(P: np.ndarray, o: dict) -> np.ndarray:
    """B106: (N,) bool, the nodes P inside the placement region `place_region` ([[lon, lat], ...], a spherical
    polygon with great-circle edges, as a size zone's `polygon`); all True when it is None."""
    if o.get("place_region") is None:
        return np.ones(len(P), dtype=bool)
    from meshom.geo import lonlat_to_xyz
    from meshom.size import polygon_inside

    pr = np.asarray(o["place_region"], dtype=np.float64)
    return polygon_inside(P, lonlat_to_xyz(pr[:, 0], pr[:, 1]))


def place(fit: Fitness, o: dict, cap: float = 1.0, fixed_pts: np.ndarray | None = None,
          band_pts: np.ndarray | None = None) -> tuple[list[Region], dict]:
    """Greedy placement, largest first. cap scales the capacity thresholds (f_max, b_cap, b_slope, grad_max; the
    aspect-allowance scan). fixed_pts: the nodes of the fixed blocks (B59), band_pts: those of the coastal bands (their
    own gap band_gap, C50; clear_of_fixed). o["front"] (B67): "seeds" = given seeds
    grow with a shared growth front (`_front_pass`) instead of one after the other; True = the greedy placement's
    accepted seeds are grown again that way too; False = off."""
    P, T, hv = fit.bg.points, fit.bg.tris, fit.hv
    n = len(P)
    adm = (fit.d_cone >= o["cone_margin"]) & (fit.d_coast >= o["coast_margin"]) & \
          (fit.d_complex >= o["complex_margin"])
    adm &= clear_of_fixed(P, hv, o, fixed_pts, band_pts)
    adm &= in_place_region(P, o)
    E = edge_list(T, n)
    wet = fit.wet if o.get("water_place", True) and fit.wet is not None else None     # B70: under a land cut
    adm0 = adm.copy()
    cost = edge_cost(P, E, fit.u, hv, o["grow"])
    G = coo_matrix((cost, (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    q, a = tri_capacity(fit)
    aw = None if wet is None else a * wet[T].mean(axis=1)       # B70: each triangle's water cells (wet-node share)
    seedable = np.ones(n, dtype=bool) if wet is None else wet   # B70: seeds only in the water
    tried = np.zeros(n, dtype=bool)
    vt = np.full(n, -1, dtype=np.int64)
    vt[T.ravel()] = np.repeat(np.arange(len(T)), 3)
    log = {"rejected": 0, "lambda_solves": 0}
    f_max, grad_max = o["f_max"] * cap, o["grad_max"] * cap

    def grow(seed: int, adm: np.ndarray, limit=o["grow_limit"]):
        """The region grown from `seed` over the admissible nodes `adm` and accepted (the acceptance rule), or None."""
        dist = dijkstra(G, directed=False, indices=seed, limit=np.inf if limit is None else limit)
        prio = np.where(np.isfinite(dist[T]).all(axis=1), dist[T].max(axis=1), np.inf)
        adm_t = adm[T].all(axis=1) & np.isfinite(prio)
        order, cum_b, cum_a = _grow(int(vt[seed]), prio, adm_t, fit.nbr, T, n, q, a, o["b_cap"] * cap,
                                    o["b_slope"] * cap, o["a_max"], o["compact"], 0.25 * o["min_cells"])
        cum_w = cum_a if aw is None else np.cumsum(aw[order])    # B70: min_cells counted in water cells
        ok = None
        if len(order) and cum_w[-1] >= o["min_cells"]:
            # largest passing prefix: the full region first, then bisection between min_cells and it
            lo, hi = int(np.searchsorted(cum_w, o["min_cells"])), len(order) - 1
            m, first = hi, True
            for _ in range(o["bisect"] + 1):
                nodes, lam, F, grad = region_lambda(fit, order[: m + 1])
                log["lambda_solves"] += 1
                if F <= f_max and grad <= grad_max:
                    ok = (m, nodes, lam, F, grad)
                    lo = m + 1
                else:
                    hi = m - 1
                if first and ok is not None:
                    break
                first = False
                if lo > hi:
                    break
                m = (lo + hi + 1) // 2
        if ok is None:
            return None
        m, nodes, lam, F, grad = ok
        return Region(order[: m + 1], nodes, lam, F, grad, float(cum_b[m]), float(cum_a[m]), seed,
                      None if aw is None else float(cum_w[m]))

    given = None
    if o.get("seeds"):                                    # seeds given (lon, lat): exactly these, in order
        from meshom.geo import lonlat_to_xyz

        sd = np.asarray(o["seeds"], dtype=np.float64).reshape(-1, 2)
        given = list(lonlat_to_xyz(sd[:, 0], sd[:, 1]))
        if o.get("front") in (True, "seeds") and len(given) > 1:     # B67: the given seeds share the space
            cand = np.flatnonzero(adm0 & seedable)
            nodes = [int(cand[np.argmin(np.linalg.norm(P[cand] - x, axis=1))])
                     for x in given[: o["max_blocks"] or None]]
            seeds = list(dict.fromkeys(nodes))
            lim = o.get("seed_grow_limit")                # per-seed Dijkstra limits (the advisor's lever), None = none
            limits = {nd: lim[i] for i, nd in enumerate(nodes) if lim and i < len(lim)} if lim else None
            regions = _front_pass(seeds, grow, E, P, T, hv, adm0, o, log, limits)
            log["n_regions"] = len(regions)
            return regions, log

    def greedy(tried: np.ndarray) -> list:
        """One placement pass (the given seeds in order, else the clearest admissible node first); tried: the nodes
        that may not seed (updated with the rejected seeds' neighbourhoods)."""
        adm, regions = adm0.copy(), []
        cap_n = o["max_blocks"] or np.inf                 # None / 0: no cap (B70)
        it = 0
        while len(regions) < cap_n and it < 4 * cap_n:       # at most 4 x max_blocks attempts per pass
            it += 1
            cand = np.flatnonzero(adm & ~tried & seedable)
            if len(cand) == 0:
                break
            if given is not None:
                if not given:
                    break
                x = given.pop(0)
                seed = int(cand[np.argmin(np.linalg.norm(P[cand] - x, axis=1))])
            else:
                clear = dist_h(P[cand], P[border(E, adm)], hv[cand])
                k = int(np.argmax(clear))
                if clear[k] < o["min_clear"]:
                    break
                seed = int(cand[k])
            r = grow(seed, adm)
            if r is None:
                log["rejected"] += 1
                near = dist_h(P, P[[seed]], hv) <= o["min_clear"]
                tried |= near
                continue
            regions.append(r)
            adm[r.nodes] = False
            adm &= dist_h(P, P[boundary_nodes(T[r.tris], n)], hv) > o["block_gap"]      # the region's outline
        return regions

    regions = greedy(tried.copy())
    if o.get("drop_small") and given is None and len(regions) > 2:      # B68: small blocks left to the fill
        med = float(np.median([r.value for r in regions]))      # B70: water cells under a land cut
        lim = float(o["drop_small"]) * med
        rec = {"median_cells": round(med, 1), "limit_cells": round(lim, 1), "dropped": [], "reruns": 0}
        for rnd in range(DROP_RERUNS + 1):
            small = [r for r in regions if r.value < lim]
            if not small:
                break
            rec["dropped"] += [round(r.value, 1) for r in small]
            if rnd == DROP_RERUNS:                        # the last pass's small regions go without another rerun
                regions = [r for r in regions if r.value >= lim]
                break
            for r in small:                               # neither the seed's neighbourhood nor the region seeds again
                tried |= dist_h(P, P[[r.seed]], hv) <= o["min_clear"]
                tried[r.nodes] = True
            regions = greedy(tried.copy())
            rec["reruns"] += 1
        log["drop_small"] = rec
    if o.get("front") is True and len(regions) > 1:      # B67: the greedy's accepted seeds share the space
        regions = _front_pass([r.seed for r in regions], grow, E, P, T, hv, adm0, o, log)
    log["n_regions"] = len(regions)
    return regions, log


def _front_pass(seeds: list, grow, E, P, T, hv, adm0, o: dict, log: dict, limits: dict | None = None) -> list:
    """Shared growth front (B67): the seeds (given, or the greedy first pass's accepted ones) grown sharing the space
    between them. Each bg node belongs to the cell of the seed whose front reaches it first when all seeds' fronts
    advance at once at the same speed in the isotropic metric |dx| / h (front_cells: the seeds' Voronoi cells in units
    of h, so the front between two seeds placed symmetrically about a bisector is that bisector, whatever the field
    does in either cell); region i grows from its seed in its own cell and more than block_gap / 2 from the shared
    front (the midpoints of the bg edges between its cell and another's), so the gap between two neighbours is centred
    on the front. The growth (order, b_cap / compact / a_max, acceptance) is the first pass's, without grow_limit: the
    cell is the limit (the layout hint's grow_limit only kept a block in its arm). Regions in the seeds' order, each
    accepted region and its block_gap dilation leaving the admissible set, as in the first pass. A seed whose region
    is rejected leaves the cells, which are formed again without it (a greedy pass's region is not kept)."""
    n = len(P)
    seeds = list(seeds)
    Gi = coo_matrix((edge_cost(P, E, None, hv, "metric"), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    rec = {"seeds": len(seeds), "dropped": 0, "rounds": 0}
    if o.get("front_peer") is not None:
        out = _peer_pass(seeds, grow, Gi, E, P, T, hv, adm0, o, limits, rec)
        log["front"] = rec
        return out
    while seeds:
        rec["rounds"] += 1
        lab = front_cells(Gi, seeds)
        adm = adm0.copy()
        out, bad = [], None
        for i, s in enumerate(seeds):
            adm_i = adm & (lab == i) & (dist_h(P, front_points(P, E, lab, i), hv) > 0.5 * o["block_gap"])
            r = grow(s, adm_i, (limits or {}).get(s))     # limits: per-seed grow limits (seed_grow_limit)
            if r is None:
                bad = i
                break
            out.append(r)
            adm[r.nodes] = False
            adm &= dist_h(P, P[boundary_nodes(T[r.tris], n)], hv) > o["block_gap"]
        if bad is None:
            rec["cells"] = [float(c) for c in (r.cells for r in out)]
            log["front"] = rec
            return out
        rec["dropped"] += 1
        seeds.pop(bad)
    log["front"] = rec
    return []


def _peer_pass(seeds: list, grow, Gi, E, P, T, hv, adm0, o: dict, limits: dict | None, rec: dict) -> list:
    """The shared front between peers only (option front_peer = X, B69). The seeds grow one after the other, in
    order (each accepted region and its block_gap dilation leaving the admissible set, a rejected seed skipped). Two
    seeds are peers when the gap their two regions leave lies within X h of their bisector in the isotropic metric:
    |c_ij - c_ji| / 2 <= X, c_ij = how far region i reaches past the bisector toward seed j, max over its nodes of
    (d_i - d_j) / 2 (d: Dijkstra distances in units of h from each seed; negative when it stops short). The growth is
    then repeated with each region kept on its own side of every peer's bisector (the nodes nearer to its seed than to
    the peer's, d_i < d_j), more than block_gap / 2 from it (the chord distance to the midpoints of the bg edges that
    cross it, front_points of the pair: the shared front's rule restricted to the pair), so
    a symmetric pair's gap is centred on the bisector (B67) while a seed placed beside a much larger region (the greedy
    placement's later seeds, whose region stops a gap away from that region's edge, far from the bisector) keeps the
    order's layout. Peers are found again on the new regions (never removed) until none is added, at most
    PEER_ROUNDS passes. grow_limit applies to a seed without a peer (the plain sequential growth); a seed with a peer
    grows without it, as in the shared front (its bisectors are its limit); seed_grow_limit overrides both."""
    n = len(P)
    k = len(seeds)
    D = np.atleast_2d(dijkstra(Gi, directed=False, indices=np.asarray(seeds, dtype=np.int64)))
    X = float(o["front_peer"])

    def seq(keep, has_peer):
        adm, out = adm0.copy(), []
        for i, s in enumerate(seeds):
            lim = (limits or {}).get(s, None if has_peer[i] else o["grow_limit"])    # a peer's limit: its bisectors
            r = grow(s, adm & keep[i], lim)
            out.append(r)
            if r is not None:
                adm[r.nodes] = False
                adm &= dist_h(P, P[boundary_nodes(T[r.tris], n)], hv) > o["block_gap"]
        return out

    def reach(regs):
        c = np.full((k, k), -np.inf)
        for i, r in enumerate(regs):
            if r is not None:
                c[i] = 0.5 * np.max(D[i, r.nodes][None, :] - D[:, r.nodes], axis=1)
        return c

    keep = [np.ones(n, dtype=bool) for _ in seeds]
    peers: set = set()
    has_peer = np.zeros(k, dtype=bool)
    regs = seq(keep, has_peer)
    for rnd in range(PEER_ROUNDS):
        c = reach(regs)
        new = {(i, j) for i in range(k) for j in range(i + 1, k)
               if np.isfinite(c[i, j]) and np.isfinite(c[j, i]) and abs(c[i, j] - c[j, i]) / 2 <= X} - peers
        if not new:
            break
        peers |= new
        for i, j in new:
            lab = np.where(D[i] < D[j], 0, 1)               # the pair's two cells
            near = np.abs(D[j] - D[i]) <= 4.0 * o["block_gap"]     # (farther from the bisector: far from its front)
            far = np.ones(n, dtype=bool)
            far[near] = dist_h(P[near], front_points(P, E, lab, 0), hv[near]) > 0.5 * o["block_gap"]
            keep[i] &= (lab == 0) & far
            keep[j] &= (lab == 1) & far
            has_peer[[i, j]] = True
        regs = seq(keep, has_peer)
        rec["rounds"] = rnd + 1
    rec["peers"] = sorted([int(i), int(j)] for i, j in peers)
    rec["dropped"] = sum(r is None for r in regs)
    out = [r for r in regs if r is not None]
    rec["cells"] = [float(r.cells) for r in out]
    return out


BETWEEN_DEG = 150.0     # a corridor node of the union sees the nearest nodes of its two regions this far apart


def _comp(sel: np.ndarray, nbr: np.ndarray) -> np.ndarray:
    """Edge-connected component label of each triangle of the selection sel (others: their own components)."""
    t, k = np.nonzero(nbr >= 0)
    u = nbr[t, k]
    m = sel[t] == sel[u]
    n = len(sel)
    return connected_components(coo_matrix((np.ones(int(m.sum())), (t[m], u[m])), shape=(n, n)), directed=False)[1]


def union_region(fit: Fitness, ta: np.ndarray, tb: np.ndarray, others: list, o: dict,
                 fixed_pts: np.ndarray | None = None) -> tuple:
    """The direct join's union test (B61): regions ta, tb (bg triangle ids) plus the corridor between them. Corridor =
    the bg triangles whose nodes all lie between the two regions (the great-circle directions to the nearest node of
    each at least BETWEEN_DEG apart, so the corridor ends with the facing sides) where the stitch's partition would put
    the pair's corridor (its rules on distances in h: d_a + d_b <= w_open; the third object, another block of `others`
    or the coast, >= j_margin farther than the second) and that stay admissible for a block (cone, coast and
    complex-coast margins, > block_gap from every other block's outline and from every node of the fixed blocks,
    `fixed_pts`, B59; a fixed block is a third object too). The edge-connected union holding both regions, its holes
    filled and the triangle fan added at a pinch node, must keep that gap at every node (why "gap"), be a closed disc
    and pass the placement's acceptance rule (F <= f_max, h |grad lambda| <= grad_max; B12, B13; no B14 pre-screen:
    the union is not grown). Returns (Region or None, info)."""
    from scipy.spatial import cKDTree

    P, T, hv = fit.bg.points, fit.bg.tris, fit.hv
    n = len(P)
    side = []
    r = 2.0 * np.sin(0.5 * min(np.pi, o["w_open"] * float(hv.max()) / R_EARTH_M))   # farther: d_a + d_b > w_open
    for t_ in (ta, tb):
        q_ = P[boundary_nodes(T[t_], n)]                 # the nearest node of a region to a node outside it
        d_, j_ = cKDTree(q_).query(P, distance_upper_bound=r, workers=-1)
        far = ~np.isfinite(d_)
        side.append((np.where(far, np.inf, 2.0 * np.arcsin(np.clip(0.5 * d_, 0.0, 1.0)) * R_EARTH_M / hv),
                     np.where(far[:, None], 0.0, q_[np.minimum(j_, len(q_) - 1)] - P)))
    (da, va), (db, vb) = side
    cos = np.einsum("ij,ij->i", va, vb) / np.maximum(np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1), 1e-300)
    between = (cos <= np.cos(np.radians(BETWEEN_DEG))) | (np.minimum(da, db) < 1e-9)
    ob = [P[boundary_nodes(T[t], n)] for t in others]
    do = dist_h(P, np.vstack(ob) if ob else np.zeros((0, 3)), hv)
    if fixed_pts is not None and len(fixed_pts):
        do = np.minimum(do, dist_h(P, fixed_pts, hv))      # the fixed blocks' exclusion (B59), as place(fixed_pts)
    adm = (fit.d_cone >= o["cone_margin"]) & (fit.d_coast >= o["coast_margin"]) & \
          (fit.d_complex >= o["complex_margin"]) & (do > o["block_gap"]) & in_place_region(P, o)
    corr = adm & between & (da + db <= o["w_open"]) & \
        (np.maximum(da, db) + o["j_margin"] <= np.minimum(do, fit.d_coast))
    sel = corr[T].all(axis=1)
    sel[ta] = sel[tb] = True
    lab = _comp(sel, fit.nbr)
    sel &= lab == lab[ta[0]]
    info = {"corridor_tris": int(sel.sum() - len(ta) - len(tb))}
    if lab[tb[0]] != lab[ta[0]] or info["corridor_tris"] <= 0:
        return None, dict(info, why="no corridor")
    for _ in range(2):                                   # holes filled, pinch nodes closed by their triangle fans
        lab = _comp(sel, fit.nbr)
        out = np.bincount(lab[~sel]).argmax()
        sel |= lab != out
        sel |= pinch_nodes(T[sel], n)[T].any(axis=1)
    tris = np.flatnonzero(sel)
    info["union_tris_added"] = int(len(tris) - len(ta) - len(tb) - info["corridor_tris"])
    if (do[np.unique(T[tris])] <= o["block_gap"]).any():  # a filled hole or a pinch fan reached another block
        return None, dict(info, why="gap")
    if not is_disc(T[tris]):
        return None, dict(info, why="not a disc")
    nodes, lam, F, grad = region_lambda(fit, tris)
    info.update(F=F, grad=grad)
    if F > o["f_max"] or grad > o["grad_max"]:
        return None, dict(info, why="lambda")
    q, a = tri_capacity(fit)
    return Region(tris, nodes, lam, F, grad, float(q[tris].sum()), float(a[tris].sum()), -1), info


def pinch_nodes(T: np.ndarray, n: int) -> np.ndarray:
    """(n,) bool: nodes with more than one outgoing boundary half-edge of the triangles T (two fans touching)."""
    a = T.ravel()
    b = np.roll(T, -1, axis=1).ravel()
    key, rkey = a * n + b, b * n + a
    sk = np.sort(key)
    pos = np.minimum(np.searchsorted(sk, rkey), len(sk) - 1)
    return np.bincount(a[sk[pos] != rkey], minlength=n) > 1


def is_disc(T: np.ndarray) -> bool:
    """The triangles form a closed disc: Euler characteristic 1 and one boundary loop without pinch nodes."""
    n = int(T.max()) + 1
    E = edge_list(T, n)
    nodes = np.unique(T)
    if len(nodes) - len(E) + len(T) != 1:
        return False
    a = T.ravel()
    b = np.roll(T, -1, axis=1).ravel()
    key, rkey = a * n + b, b * n + a
    sk = np.sort(key)
    pos = np.minimum(np.searchsorted(sk, rkey), len(sk) - 1)
    bnd = sk[pos] != rkey
    deg = np.bincount(a[bnd], minlength=n)
    if (deg > 1).any():
        return False
    nxt = np.full(n, -1, dtype=np.int64)
    nxt[a[bnd]] = b[bnd]
    s0 = int(a[bnd][0])
    x, k = int(nxt[s0]), 1
    while x != s0 and k <= bnd.sum():
        x, k = int(nxt[x]), k + 1
    return k == int(bnd.sum())


def with_margin(T: np.ndarray, tris: np.ndarray, rings: int) -> np.ndarray:
    """The region's triangles plus up to `rings` vertex rings of bg triangles, as long as the result is a disc."""
    sel = np.zeros(len(T), dtype=bool)
    sel[tris] = True
    for _ in range(rings):
        mark = np.zeros(int(T.max()) + 1, dtype=bool)
        mark[T[sel].ravel()] = True
        new = sel | mark[T].any(axis=1)
        if not is_disc(T[new]):
            break
        sel = new
    return np.flatnonzero(sel)
