"""Coast filter (decisions C16, C40, C41): the coast smoothed to the local h before it becomes the domain boundary
(`[coast] treatment = "loops"`).

The water of the coast loops is opened at the radius ro(x) = OPEN_H h(x), then closed at rc(x) = CLOSE_H h(x)
(`[coast] smoothing_m` > 0: rc = smoothing_m everywhere, ro = 2 rc), on the sphere:

1. Opening: water that no disc of radius ro inside the water covers becomes land. Bays, straits and gaps between
   islands narrower than 2 ro = 2 h close, and water corners are rounded, so every passage left open is at least two
   cells wide (one-cell passages left triangles with two coast edges and quads with three: measured, C41). The strait
   corridors (meshom.straits: within W/2 of a listed strait's path) are water after this step whatever the coast
   says, as in the cut. A passage resolved by h (an automatic strait zone h = W/n, meshom.lfs) stays open by itself,
   since ro follows h. The source's land connectivity is kept first (C40): the land points nearest to one source
   loop are one piece of land in the source, so where a neck or a land strip thinner than the point spacing left
   them in several land components, a bridge of points along the cheapest path joins them (`_bridges`).
2. Closing: land that no disc of radius rc inside the land covers becomes water. Capes, spits and islands narrower
   than 2 rc = h go, and land corners are rounded. A piece of land that this step would remove is kept when removing
   it would change the topology (C40): it would split a land mass (an isthmus: Panama and Kra at h = 100 km; the plug
   the opening put into a short strait) or join two waters that the opening left apart (a spit in front of a lagoon
   whose mouth has closed). So the closing never opens a passage, and the opening closes every passage narrower than
   2 ro that is not a corridor (S24).
3. Basins: water that the opening cut off (the Mediterranean behind Gibraltar at h = 100 km) becomes land unless it is
   the largest water body or at least `keep_basins_km2` (the cut's rule, meshom.methods.m14.cut).

Representation: signed distances (m, water positive) on an adaptive point set: icosahedral refinement down to a
spacing of about min(ro, rc) / Q within BAND (ro + rc) of the coast and of the corridor edges, coarse elsewhere,
triangulated by its convex hull (the spherical Delaunay triangulation). The signed distance to the coast is exact,
side included (`signed_distance`: a nearest-sample search widened until no unseen segment can be nearer, C43); each
later step is a distance to the zero set of the previous field (its crossing points on the triangulation edges in a
k-d tree). The result's zero set is traced by marching triangles into closed loops with water on the left. There are
no charts and no tiles, so the antimeridian and the poles need no special case.
"""
import time

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import ConvexHull, cKDTree

from meshom.coast import Coast
from meshom.geo import R_EARTH_M
from meshom.size import SizeField, evaluate

OPEN_H = 1.0          # opening radius ro = OPEN_H h: water narrower than 2 h closes, every passage left open is
                      # at least 2 cells wide (C16, C30)
CLOSE_H = 0.5         # closing radius rc = CLOSE_H h: land narrower than h goes (C16)
Q = 4                 # point spacing r / Q near the coast
BAND = 1.25           # ... within BAND (ro + rc) + 2 spacings of the coast and the corridor edges (the fields'
                      # zero sets lie within ro + rc of the coast)
SLOPE = 0.25          # |grad h| assumed within a face while refining (h is limited to grad_max = 0.1 by default)
K_NEAR = 6            # coast samples per point in the first round of the exact segment distance (doubled for the
                      # points where an unseen segment could be nearer, C43)
CHUNK = 200_000       # points per block of the signed distance


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _arc_m(chord: np.ndarray) -> np.ndarray:
    return 2.0 * R_EARTH_M * np.arcsin(np.clip(0.5 * np.asarray(chord), 0.0, 1.0))


# ------------------------------------------------------------------------------------- coast segments

def _segments(loops: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Directed segments of the loops (water on the left): start, end, previous and next segment of the same loop."""
    off = np.cumsum([0] + [len(lp) for lp in loops])
    A = np.vstack(loops)
    idx = np.arange(off[-1])
    nxt, prv = idx + 1, idx - 1
    nxt[off[1:] - 1] = off[:-1]
    prv[off[:-1]] = off[1:] - 1
    return A, A[nxt], prv, nxt


def _samples(A: np.ndarray, B: np.ndarray, step_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points along every segment at spacing <= step_m (its start included) and their segment ids."""
    L = _arc_m(np.linalg.norm(B - A, axis=1))
    k = np.maximum(1, np.ceil(L / step_m).astype(np.int64))
    seg = np.repeat(np.arange(len(A)), k)
    t = (np.arange(len(seg)) - np.repeat(np.cumsum(k) - k, k)) / np.repeat(k, k)
    return _unit((1.0 - t)[:, None] * A[seg] + t[:, None] * B[seg]), seg


def sample_gap(samp: np.ndarray, sseg: np.ndarray, B: np.ndarray) -> float:
    """The longest arc (rad) between consecutive samples of a segment (its last sample to the segment's end)."""
    same = np.r_[sseg[1:] == sseg[:-1], False]
    q = np.where(same[:, None], np.roll(samp, -1, axis=0), B[sseg])
    return float(2.0 * np.arcsin(min(1.0, 0.5 * float(np.linalg.norm(samp - q, axis=1).max()))))


def signed_distance(x: np.ndarray, A: np.ndarray, B: np.ndarray, prv: np.ndarray, nxt: np.ndarray,
                    tree: cKDTree, sseg: np.ndarray, gap: float) -> tuple[np.ndarray, np.ndarray]:
    """(N,) geodesic distance (m) of x to the coast segments, > 0 in the water, and (N,) the nearest segment. Exact:
    the candidates are the segments of the k nearest samples and the segments before them (a segment's end is the
    next one's first sample); a segment none of whose samples or end is among them has every sample at least rho_k
    (the k-th sample's angle) away, so it lies at least arccos(cos rho_k / cos(gap / 2)) away (spherical Pythagoras;
    gap: sample_gap). A point whose best candidate is farther than that bound is searched again with twice k (from
    K_NEAR = 6; a dense islet's vertices took all six slots and turned land 131 km inland into water: C43). On the
    nearest segment the arc distance is exact; the side is that segment's (water on its left), or at its end vertex
    v: left of both segments around v at a left turn (water convex there), of either at a right turn."""
    out = np.empty(len(x))
    seg = np.empty(len(x), dtype=np.int64)
    nA = _unit(np.cross(A, B))
    cg = np.cos(0.5 * gap)
    todo, k = np.arange(len(x)), K_NEAR
    while len(todo):
        k = min(k, len(sseg))
        step = max(1, CHUNK * K_NEAR // (2 * k))                          # 2k candidates: the memory of k
        again = []
        for a0 in range(0, len(todo), step):
            ii = todo[a0:a0 + step]
            p = x[ii]
            dk, idx = tree.query(p, k)
            dk, idx = dk.reshape(len(p), k), idx.reshape(len(p), k)
            cand = np.concatenate([sseg[idx], prv[sseg[idx]]], axis=1)    # (n, 2k)
            a, b, n = A[cand], B[cand], nA[cand]
            z = np.einsum("ij,ikj->ik", p, n)
            w = p[:, None, :] - z[..., None] * n
            between = ((np.einsum("ikj,ikj->ik", np.cross(a, w), n) >= 0)
                       & (np.einsum("ikj,ikj->ik", np.cross(w, b), n) >= 0))
            da = 2.0 * np.arcsin(np.clip(0.5 * np.linalg.norm(p[:, None, :] - a, axis=2), 0.0, 1.0))
            db = 2.0 * np.arcsin(np.clip(0.5 * np.linalg.norm(p[:, None, :] - b, axis=2), 0.0, 1.0))
            d = np.where(between, np.arcsin(np.clip(np.abs(z), 0.0, 1.0)), np.minimum(da, db))
            j = np.argmin(d, axis=1)
            r = np.arange(len(p))
            s, dist, inner = cand[r, j], d[r, j], between[r, j]
            rho = 2.0 * np.arcsin(np.clip(0.5 * dk[:, -1], 0.0, 1.0))
            cr = np.cos(rho)
            bound = np.where(cr >= 0.0, np.arccos(np.clip(cr / cg, -1.0, 1.0)), rho)
            ok = (dist <= bound) | (k == len(sseg))
            again.append(ii[~ok])
            water = z[r, j] > 0
            v = ~inner
            if v.any():
                at_a = (da[r, j] <= db[r, j])[v]
                e1 = np.where(at_a, prv[s[v]], s[v])                      # the segment into the vertex
                e2 = np.where(at_a, s[v], nxt[s[v]])                      # ... and out of it
                pv = p[v]
                l1 = np.einsum("ij,ij->i", pv, nA[e1]) > 0
                l2 = np.einsum("ij,ij->i", pv, nA[e2]) > 0
                left_turn = np.einsum("ij,ij->i", nA[e1], B[e2]) > 0
                water[v] = np.where(left_turn, l1 & l2, l1 | l2)
            out[ii[ok]] = (np.where(water, dist, -dist) * R_EARTH_M)[ok]
            seg[ii[ok]] = s[ok]
        todo = np.concatenate(again)
        k *= 2
    return out, seg


# ---------------------------------------------------------------------------------------------- points

_T = (1.0 + 5.0**0.5) / 2.0
_ICO_V = np.array([(-1, _T, 0), (1, _T, 0), (-1, -_T, 0), (1, -_T, 0), (0, -1, _T), (0, 1, _T), (0, -1, -_T),
                   (0, 1, -_T), (_T, 0, -1), (_T, 0, 1), (-_T, 0, -1), (-_T, 0, 1)], dtype=np.float64)
_ICO_F = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
                   (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
                   (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)], dtype=np.int64)


def refine_points(split) -> np.ndarray:
    """Vertices of the icosahedron refined 1 -> 4 wherever split(centre (F,3), radius_m (F,), edge_m (F,)) says so;
    a midpoint shared by two faces is one vertex (hanging vertices are fine: the hull triangulates them)."""
    V = [_unit(_ICO_V)]
    n = len(V[0])
    keys = np.zeros(0, dtype=np.int64)
    kid = np.zeros(0, dtype=np.int64)
    active = _ICO_F
    while len(active):
        P = np.vstack(V) if len(V) > 1 else V[0]
        V = [P]
        a, b, c = P[active[:, 0]], P[active[:, 1]], P[active[:, 2]]
        cen = _unit(a + b + c)
        rad = _arc_m(np.max([np.linalg.norm(cen - a, axis=1), np.linalg.norm(cen - b, axis=1),
                             np.linalg.norm(cen - c, axis=1)], axis=0))
        edge = _arc_m(np.linalg.norm(a - b, axis=1))
        active = active[split(cen, rad, edge)]
        if not len(active):
            break
        e = np.concatenate([active[:, [0, 1]], active[:, [1, 2]], active[:, [2, 0]]])
        k = np.minimum(e[:, 0], e[:, 1]) * (1 << 32) + np.maximum(e[:, 0], e[:, 1])
        uk, inv = np.unique(k, return_inverse=True)
        pos = np.searchsorted(keys, uk)
        old = (pos < len(keys)) & (keys[np.minimum(pos, len(keys) - 1)] == uk) if len(keys) else np.zeros(len(uk), bool)
        ids = np.empty(len(uk), dtype=np.int64)
        ids[old] = kid[pos[old]]
        new = np.flatnonzero(~old)
        ids[new] = n + np.arange(len(new))
        i, j = uk[new] >> 32, uk[new] & ((1 << 32) - 1)
        V.append(_unit(P[i] + P[j]))
        n += len(new)
        allk = np.concatenate([keys, uk[new]])
        o = np.argsort(allk, kind="stable")
        keys, kid = allk[o], np.concatenate([kid, ids[new]])[o]
        m = ids[inv].reshape(3, -1)
        ab, bc, ca = m
        A0, B0, C0 = active.T
        active = np.concatenate([np.column_stack([A0, ab, ca]), np.column_stack([ab, B0, bc]),
                                 np.column_stack([ca, bc, C0]), np.column_stack([ab, bc, ca])])
    return np.vstack(V)


def _hull(P: np.ndarray) -> np.ndarray:
    """(T,3) CCW (seen from outside) convex hull triangles of points around the origin (meshom.delaunay)."""
    s = ConvexHull(P).simplices.astype(np.int64)
    neg = np.einsum("ij,ij->i", P[s[:, 0]], np.cross(P[s[:, 1]], P[s[:, 2]])) < 0
    s[neg] = s[neg][:, [0, 2, 1]]
    return s


# ------------------------------------------------------------------------------------------ field steps

def _crossings(P: np.ndarray, E: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Points where f (linear along each edge) changes sign (> 0 against <= 0) on the edges E."""
    i, j = E[:, 0], E[:, 1]
    x = (f[i] > 0) != (f[j] > 0)
    i, j = i[x], j[x]
    t = (f[i] / (f[i] - f[j]))[:, None]
    return _unit(P[i] + t * (P[j] - P[i]))


def _distance_to_zero(P: np.ndarray, E: np.ndarray, f: np.ndarray, where: np.ndarray) -> np.ndarray:
    """(N,) geodesic distance (m) to the zero set of f at the points `where`, 0 elsewhere (inf without a zero set)."""
    X = _crossings(P, E, f)
    out = np.zeros(len(P))
    if not len(X):
        out[where] = np.inf
        return out
    ch, _ = cKDTree(X).query(P[where])
    out[where] = _arc_m(ch)
    return out


def _labels(E: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Connected components of the points in mask over the edges E (-1 outside the mask)."""
    n = len(mask)
    k = mask[E[:, 0]] & mask[E[:, 1]]
    g = coo_matrix((np.ones(int(k.sum())), (E[k, 0], E[k, 1])), shape=(n, n)).tocsr()
    _, lab = connected_components(g, directed=False)
    out = np.full(n, -1, dtype=np.int64)
    out[mask] = np.unique(lab[mask], return_inverse=True)[1]
    return out


def _touching(E: np.ndarray, la: np.ndarray, lb: np.ndarray, n_a: int) -> np.ndarray:
    """(n_a,) for each component of la: the number of distinct components of lb it shares an edge with."""
    i, j = E[:, 0], E[:, 1]
    pa = np.concatenate([la[i], la[j]])
    pb = np.concatenate([lb[j], lb[i]])
    k = (pa >= 0) & (pb >= 0)
    pairs = np.unique(pa[k] * (int(lb.max()) + 1) + pb[k]) if k.any() else np.zeros(0, dtype=np.int64)
    return np.bincount(pairs // (int(lb.max()) + 1), minlength=n_a) if len(pairs) else np.zeros(n_a, dtype=np.int64)


def _bridges(P: np.ndarray, E: np.ndarray, s0: np.ndarray, g1: np.ndarray, loop: np.ndarray,
             floor: np.ndarray) -> tuple[np.ndarray, int, int]:
    """The source's land connectivity kept through the point set (C40). The land points whose nearest coast segment
    lies on one source loop are connected in the source (along that loop's land side), and the opening only adds
    land, so they must lie in one land component of g1. Where they do not (a neck or a land strip thinner than the
    point spacing: no chain of land points along it), the other pieces are joined to the largest one along the
    cheapest path of edges within the points nearest to that loop (cost of a water point: its distance to the coast,
    of a land point: 0), and the path's water points become land, g1 = -max(|g1|, floor). Returns the new g1, the
    number of pieces joined and of points made land."""
    land = g1 <= 0
    lab = _labels(E, land)
    src = np.flatnonzero((s0 <= 0) & land)
    if not len(src):
        return g1, 0, 0
    nl = int(lab.max()) + 1
    pairs = np.unique(loop[src] * nl + lab[src])
    lp_of = pairs // nl
    split = np.unique(lp_of[np.r_[lp_of[1:] == lp_of[:-1], False]])
    if not len(split):
        return g1, 0, 0
    g1 = g1.copy()
    cost = np.where(land, 0.0, np.maximum(s0, 0.0)) + 1.0                 # + 1 m: shortest of equal paths
    n_joined, n_points = 0, 0
    for L in split:
        sub = np.flatnonzero(loop == L)
        loc = np.full(len(P), -1, dtype=np.int64)
        loc[sub] = np.arange(len(sub))
        k = (loc[E[:, 0]] >= 0) & (loc[E[:, 1]] >= 0)
        i, j = loc[E[k, 0]], loc[E[k, 1]]
        G = coo_matrix((np.r_[cost[sub[j]], cost[sub[i]]], (np.r_[i, j], np.r_[j, i])),
                       shape=(len(sub), len(sub))).tocsr()
        own = (s0[sub] <= 0) & land[sub]
        pieces = np.unique(lab[sub][own])
        big = int(np.argmax(np.bincount(lab[sub][own])))
        dist, pred = dijkstra(G, indices=np.flatnonzero(lab[sub] == big), min_only=True,
                              return_predecessors=True)[:2]
        for c in pieces[pieces != big]:
            m = np.flatnonzero(own & (lab[sub] == c))
            v = int(m[np.argmin(dist[m])])
            if not np.isfinite(dist[v]):
                continue
            n_joined += 1
            while v >= 0:
                g = sub[v]
                if g1[g] > 0:
                    g1[g] = -max(abs(g1[g]), float(floor[g]))
                    n_points += 1
                v = int(pred[v])
    return g1, n_joined, n_points


def _point_areas(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """(N,) a third of the spherical area (m^2) of every triangle at each vertex."""
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    ex = 2.0 * np.arctan2(np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))),
                          1.0 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c) + np.einsum("ij,ij->i", c, a))
    return np.bincount(T.ravel(), np.repeat(ex * R_EARTH_M**2 / 3.0, 3), minlength=len(P))


# ------------------------------------------------------------------------------------------- tracing

@njit(cache=True)
def _cycles(succ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The cycles of the partial permutation succ (-1 = not in it): their elements in order and the start offsets."""
    n = len(succ)
    seen = np.zeros(n, dtype=np.bool_)
    order = np.empty(n, dtype=np.int64)
    starts = np.empty(n + 1, dtype=np.int64)
    pos, nc = 0, 0
    for e in range(n):
        if succ[e] < 0 or seen[e]:
            continue
        starts[nc] = pos
        nc += 1
        x = e
        while not seen[x]:
            seen[x] = True
            order[pos] = x
            pos += 1
            x = succ[x]
    starts[nc] = pos
    return order[:pos], starts[:nc + 1]


def trace(P: np.ndarray, T: np.ndarray, f: np.ndarray) -> list[np.ndarray]:
    """Closed loops (K,3) of the zero set of f (piecewise linear on the CCW triangles T), water (f > 0) on the left.
    In a triangle whose odd vertex v_k is water the segment runs from edge (v_k, v_k+1) to edge (v_k+2, v_k), the
    other way when it is land; every crossed edge starts one segment and ends one, so the segments chain into cycles."""
    w = f > 0
    s = w[T].sum(axis=1)
    mixed = np.flatnonzero((s == 1) | (s == 2))
    if not len(mixed):
        return []
    Tm = T[mixed]
    he = np.stack([Tm, np.roll(Tm, -1, axis=1)], axis=2).reshape(-1, 2)            # half-edges (k, k+1)
    key = np.minimum(he[:, 0], he[:, 1]) * len(P) + np.maximum(he[:, 0], he[:, 1])
    uk, inv = np.unique(key, return_inverse=True)
    eid = inv.reshape(-1, 3)
    wm = w[Tm]
    odd_is_water = s[mixed] == 1
    k = np.where(odd_is_water, np.argmax(wm, axis=1), np.argmin(wm, axis=1))
    r = np.arange(len(mixed))
    e_k, e_prev = eid[r, k], eid[r, (k + 2) % 3]
    src = np.where(odd_is_water, e_k, e_prev)
    dst = np.where(odd_is_water, e_prev, e_k)
    succ = np.full(len(uk), -1, dtype=np.int64)
    succ[src] = dst
    i, j = uk // len(P), uk % len(P)
    cut = w[i] != w[j]                                                          # the other edges are never visited
    t = np.where(cut, f[i] / np.where(cut, f[i] - f[j], 1.0), 0.0)[:, None]
    X = _unit(P[i] + t * (P[j] - P[i]))
    order, starts = _cycles(succ)
    out = []
    for a, b in zip(starts[:-1], starts[1:]):
        lp = X[order[a:b]]
        keep = np.linalg.norm(lp - np.roll(lp, 1, axis=0), axis=1) > 1e-10      # crossings at a shared vertex
        lp = lp[keep]
        if len(lp) >= 3:
            out.append(lp)
    return out


# ------------------------------------------------------------------------------------------- the filter

def filter_coast(coast: Coast, h: SizeField, straits=(), radius_m: float | None = None,
                 keep_basins_km2: float | None = None) -> tuple[Coast, dict]:
    """The coast opened at ro = OPEN_H h, then closed at rc = CLOSE_H h (radius_m: rc = radius_m, ro = 2 rc),
    corridors protected, topology kept by the closing, cut-off basins below keep_basins_km2 dropped (module
    docstring); returns the filtered Coast (area_km2 NaN: the loops no longer come from one polygon each) and a record
    of the steps."""
    fl, info = fields(coast, h, straits, radius_m, keep_basins_km2)
    t0 = time.perf_counter()
    loops = trace(fl["P"], fl["T"], fl["F"])
    info["steps_s"]["trace_s"] = round(info["steps_s"]["topology_s"] + time.perf_counter() - t0, 3)
    info.update(n_loops=len(loops), n_vertices=int(sum(len(lp) for lp in loops)))
    return Coast(loops, np.full(len(loops), np.nan), coast.antarctica), info


def fields(coast: Coast, h: SizeField, straits=(), radius_m: float | None = None,
           keep_basins_km2: float | None = None) -> tuple[dict, dict]:
    """The filter's point set and fields (filter_coast without the tracing): P points, T triangles, E edges, ro / rc
    the opening / closing radius, s0 the signed distance to the coast, g1 after the opening, F0 after the closing, F
    with the kept pieces and the basins (m, water positive), and the record."""
    t0 = time.perf_counter()
    tm = {}
    k_open = OPEN_H / CLOSE_H                                      # ro / rc

    def rc_of(x):
        return np.full(len(x), float(radius_m)) if radius_m else CLOSE_H * evaluate(h, x)

    A, B, prv, nxt = _segments(coast.loops)
    r_seg = rc_of(_unit(A + B))
    lo = min(1.0, k_open)                                          # the smaller radius / rc: sets the spacing
    hi = 1.0 + k_open                                              # the zero sets lie within ro + rc of the coast
    samp, sseg = _samples(A, B, lo * r_seg / Q / 2.0)
    tree = cKDTree(samp)
    walls = [(s, 0.5 * s.corridor_m) for s in straits]
    r_floor = float(r_seg.min())
    for s, _ in walls:
        r_floor = min(r_floor, float(rc_of(s.dense()).min()))
    tm["samples_s"] = time.perf_counter() - t0

    def split(cen, rad, edge):
        if radius_m:
            r_lo = r_hi = np.full(len(cen), float(radius_m))
        else:
            hc = CLOSE_H * evaluate(h, cen)
            r_lo = np.maximum(hc - CLOSE_H * SLOPE * rad, r_floor)
            r_hi = hc + CLOSE_H * SLOPE * rad
        reach = rad + BAND * hi * r_hi + 2.0 * lo * r_hi / Q
        ch, _ = tree.query(cen, distance_upper_bound=float(2.0 * np.sin(min(reach.max() / R_EARTH_M, np.pi) / 2)))
        near = np.where(np.isfinite(ch), _arc_m(np.where(np.isfinite(ch), ch, 0.0)), np.inf) <= reach
        for s, half in walls:
            d = s.distance_m(cen, half + float(reach.max()))
            near |= np.abs(d - half) <= reach
        return near & (edge > np.sqrt(2.0) * lo * r_lo / Q)

    P = refine_points(split)
    tm["points_s"] = time.perf_counter() - t0
    T = _hull(P)
    E = np.unique(np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1), axis=0)
    tm["hull_s"] = time.perf_counter() - t0
    rc = rc_of(P)
    ro = k_open * rc
    s0, near = signed_distance(P, A, B, prv, nxt, tree, sseg, sample_gap(samp, sseg, B))
    tm["distance_s"] = time.perf_counter() - t0
    # 1. opening: E0 = the centres of discs of radius ro inside the water, O = E0 dilated by ro, the source's land
    # connectivity kept (bridges), plus the corridors
    e0 = s0 > ro
    g1 = np.where(e0, s0, ro - _distance_to_zero(P, E, s0 - ro, ~e0))
    loop_of = np.repeat(np.arange(len(coast.loops)), [len(lp) for lp in coast.loops])
    g1, n_joined, n_bridge = _bridges(P, E, s0, g1, loop_of[near], lo * rc / Q / 2.0)
    tm["bridges_s"] = time.perf_counter() - t0
    for s, half in walls:
        g1 = np.maximum(g1, half - s.distance_m(P, half + 2.0 * hi * float(rc.max())))
    # 2. closing: EL = the centres of discs of radius rc inside the land of O (-g1 = the distance to O), land = EL
    # dilated by rc
    el = g1 < -rc
    F = np.where(el, g1, _distance_to_zero(P, E, -(g1 + rc), ~el) - rc)
    F0 = F.copy()
    tm["morphology_s"] = time.perf_counter() - t0
    removed = (g1 <= 0) & (F > 0)                                 # land of O the closing makes water
    lab_r = _labels(E, removed)
    n_r = int(lab_r.max()) + 1
    n_restored = 0
    if n_r:
        lab_l, lab_o = _labels(E, F <= 0), _labels(E, g1 > 0)
        bad = (_touching(E, lab_r, lab_l, n_r) >= 2) | (_touching(E, lab_r, lab_o, n_r) >= 2)
        back = removed & bad[np.maximum(lab_r, 0)]
        n_restored = int(np.unique(lab_r[back]).size)
        ring = back.copy()
        ring[E[back[E[:, 0]], 1]] = True
        ring[E[back[E[:, 1]], 0]] = True
        F[ring] = np.minimum(F[ring], g1[ring])
    # 3. basins
    wet = F > 0
    lab_w = _labels(E, wet)
    area = np.bincount(lab_w[wet], _point_areas(P, T)[wet]) * 1e-6
    keep = area >= (keep_basins_km2 if keep_basins_km2 else np.inf)
    keep[np.argmax(area)] = True
    drop = wet & ~keep[np.maximum(lab_w, 0)]
    F[drop] = -rc[drop]
    tm["topology_s"] = time.perf_counter() - t0
    info = {"radius": (f"smoothing_m (closing), {k_open:g} x it (opening)" if radius_m
                       else f"opening {OPEN_H:g} h, closing {CLOSE_H:g} h"),
            "r_close_min_m": float(rc.min()), "r_close_max_m": float(rc.max()), "n_points": int(len(P)),
            "n_samples": int(len(samp)), "n_loops_in": len(coast.loops), "n_vertices_in": int(len(A)),
            "n_bridged_pieces": n_joined, "n_bridge_points": n_bridge,
            "n_removed_pieces": n_r, "n_kept_pieces": n_restored,
            "basins_km2": sorted((float(a) for a in area[keep]), reverse=True)[:20],
            "n_basins_dropped": int((~keep).sum()), "dropped_km2": float(area[~keep].sum()),
            "corridors": len(walls), "steps_s": {k: round(v, 3) for k, v in tm.items()}}
    return {"P": P, "T": T, "E": E, "ro": ro, "rc": rc, "s0": s0, "g1": g1, "F0": F0, "F": F}, info
