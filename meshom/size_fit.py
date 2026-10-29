"""Per-edge size fit on the sphere (the common final stage of the triangle methods).

Node positions that minimise, over the free nodes and for a fixed connectivity per round,

    E = (1/n_e) sum_e r_e^p  +  mu (1/n_e) sum_t max(0, tau - q_t)^2,      r_e = ln(l_e / h_e),  p even (6).

- l_e: chord of the edge on the unit sphere (chord vs arc: < 2e-4 in ln at h <= 400 km); h_e = h at the normalised
  edge midpoint, **frozen per round**: evaluated at the positions the round starts from (as `size_fidelity` evaluates
  it), so the gradient is that of the lengths only; the next round re-evaluates h_e, and on a ramp the fit converges
  over the rounds to the midpoint h of the score.
- q_t = det(a, b, c) / ((sqrt 3 / 2) h_t^2): twice the triangle's area (triple product = signed area in the local
  frame times 2, > 0 for CCW) in units of the equilateral triangle at h_t = h at the centroid (frozen per round).
  tau = 0.5: a triangle whose edges are within e^(+-0.15) h has q >= 0.85, so the term is inactive on meshes that pass
  the size gate and pushes flattened or folding triangles back (an unguarded fit folds strained seams).
- Variables: tangent offsets of the free nodes in units of h(x0), x = normalise(x0 + h(x0)/R (v1 e1 + v2 e2)); x0 and
  the chart are reset every round.
- Solver: L-BFGS (memory m, own two-loop recursion), first step scaled to `step0` h, every trial step capped at
  `max_step` h per node; Armijo backtracking that also rejects any trial where a triangle with a free corner has
  det <= 0, so a valid start never yields an inverted triangle. Energy and gradient in Numba: per-edge and
  per-triangle maps, then a per-node gather over a CSR of each free node's edges, sums in fixed blocks, so the result
  is bit-identical for any thread count (band, 0.48 M nodes: one round of 150 iterations 20 s on 1 thread, 11 s on 4).
- Fixed nodes: node_flag != 0, the topological boundary, and the caller's mask (index array or bool).
- Rounds (<= `rounds`): freeze h, run L-BFGS (<= max_iter iterations; stop when the energy falls by less than ftol
  relative in 5 consecutive iterations, or when the frozen p99 of |r| reaches target_p99), then Lawson flips
  (sphere incircle test on interior edges; boundary edges never flip). Stop early when the p99 of |ln s/h| (score's
  definition) is <= target_p99, or when a round neither flips nor lowers the energy by 1e-3 relative.
Triangle meshes; quads are carried along (their edges enter the energy, their 4 corner triangles the barrier, no flips).

smooth_fit (decisions G38, M38; block comment before it): the same machinery with the energy "smoothness first within a
size tolerance": the element size ratio between neighbours as small as possible and the corner angles as regular as
possible while every edge stays within |ln s/h| <= tol (the `size_tolerance` of meshom run, m14 and m15_tri).
"""
import time

import numpy as np
from numba import njit, prange

from meshom.delaunay import _flip, _incircle, _orient, adjacency
from meshom.geo import R_EARTH_M, arc_length, tangent_basis
from meshom.mesh import Mesh, build_topology

Q_EQ = np.sqrt(3.0) / 2.0          # twice the area of the unit equilateral triangle


BLOCK = 4096                       # blocked sums: the same partial sums whatever the thread count


@njit(cache=True, parallel=True)
def _bsum(a):
    """Deterministic parallel sum: serial within fixed blocks, then over the block sums in order."""
    nb = (len(a) + BLOCK - 1) // BLOCK
    part = np.zeros(nb)
    for b in prange(nb):
        s = 0.0
        for i in range(b * BLOCK, min(len(a), (b + 1) * BLOCK)):
            s += a[i]
        part[b] = s
    s = 0.0
    for b in range(nb):
        s += part[b]
    return s


@njit(cache=True, parallel=True)
def _dot(a, b):
    nb = (len(a) + BLOCK - 1) // BLOCK
    part = np.zeros(nb)
    for k in prange(nb):
        s = 0.0
        for i in range(k * BLOCK, min(len(a), (k + 1) * BLOCK)):
            s += a[i] * b[i]
        part[k] = s
    s = 0.0
    for k in range(nb):
        s += part[k]
    return s


@njit(cache=True, parallel=True)
def _axpy(a, x, y):
    """y += a x"""
    for i in prange(len(y)):
        y[i] += a * x[i]


@njit(cache=True, parallel=True)
def _positions(V, free, X0, E1, E2, S, X, Yn):
    for k in prange(len(free)):
        i = free[k]
        y0 = X0[i, 0] + S[k] * (V[k, 0] * E1[k, 0] + V[k, 1] * E2[k, 0])
        y1 = X0[i, 1] + S[k] * (V[k, 0] * E1[k, 1] + V[k, 1] * E2[k, 1])
        y2 = X0[i, 2] + S[k] * (V[k, 0] * E1[k, 2] + V[k, 1] * E2[k, 2])
        r = np.sqrt(y0 * y0 + y1 * y1 + y2 * y2)
        X[i, 0], X[i, 1], X[i, 2] = y0 / r, y1 / r, y2 / r
        Yn[k] = r


@njit(cache=True, parallel=True)
def _edge_pass(X, edges, lh, w, p, R, FE, CE):
    for e in prange(len(edges)):
        i, j = edges[e, 0], edges[e, 1]
        d0, d1, d2 = X[i, 0] - X[j, 0], X[i, 1] - X[j, 1], X[i, 2] - X[j, 2]
        c2 = d0 * d0 + d1 * d1 + d2 * d2
        r = 0.5 * np.log(c2) - lh[e]
        R[e] = r
        rp1 = r ** (p - 1)
        FE[e] = w[e] * rp1 * r
        CE[e] = w[e] * p * rp1 / c2


@njit(cache=True, parallel=True)
def _tri_pass(X, tris, ref, tau, Q, FB):
    for t in prange(len(tris)):
        a, b, c = tris[t, 0], tris[t, 1], tris[t, 2]
        bc0 = X[b, 1] * X[c, 2] - X[b, 2] * X[c, 1]
        bc1 = X[b, 2] * X[c, 0] - X[b, 0] * X[c, 2]
        bc2 = X[b, 0] * X[c, 1] - X[b, 1] * X[c, 0]
        q = (X[a, 0] * bc0 + X[a, 1] * bc1 + X[a, 2] * bc2) / ref[t]
        Q[t] = q
        FB[t] = (tau - q) * (tau - q) if q < tau else 0.0


@njit(cache=True)
def _barrier_grad(X, tris, ref, tau, Q, g_scale, GX):
    """Barrier gradient of the (few) triangles with q < tau into GX (serial, triangle order)."""
    for t in range(len(tris)):
        q = Q[t]
        if q >= tau:
            continue
        a, b, c = tris[t, 0], tris[t, 1], tris[t, 2]
        g = -2.0 * g_scale * (tau - q) / ref[t]
        GX[a, 0] += g * (X[b, 1] * X[c, 2] - X[b, 2] * X[c, 1])
        GX[a, 1] += g * (X[b, 2] * X[c, 0] - X[b, 0] * X[c, 2])
        GX[a, 2] += g * (X[b, 0] * X[c, 1] - X[b, 1] * X[c, 0])
        GX[b, 0] += g * (X[c, 1] * X[a, 2] - X[c, 2] * X[a, 1])
        GX[b, 1] += g * (X[c, 2] * X[a, 0] - X[c, 0] * X[a, 2])
        GX[b, 2] += g * (X[c, 0] * X[a, 1] - X[c, 1] * X[a, 0])
        GX[c, 0] += g * (X[a, 1] * X[b, 2] - X[a, 2] * X[b, 1])
        GX[c, 1] += g * (X[a, 2] * X[b, 0] - X[a, 0] * X[b, 2])
        GX[c, 2] += g * (X[a, 0] * X[b, 1] - X[a, 1] * X[b, 0])


@njit(cache=True, parallel=True)
def _gather(X, Yn, free, E1, E2, S, edges, CE, ptr, inc, inv_ne, GX, Gv):
    """Per free node: barrier part (GX) + its edges in CSR order, projected to the tangent chart (deterministic)."""
    for k in prange(len(free)):
        i = free[k]
        g0, g1, g2 = GX[i, 0], GX[i, 1], GX[i, 2]
        for jj in range(ptr[k], ptr[k + 1]):
            e = inc[jj]
            j = edges[e, 0] + edges[e, 1] - i
            c = CE[e] * inv_ne
            g0 += c * (X[i, 0] - X[j, 0])
            g1 += c * (X[i, 1] - X[j, 1])
            g2 += c * (X[i, 2] - X[j, 2])
        s = g0 * X[i, 0] + g1 * X[i, 1] + g2 * X[i, 2]
        g0 = (g0 - s * X[i, 0]) / Yn[k]
        g1 = (g1 - s * X[i, 1]) / Yn[k]
        g2 = (g2 - s * X[i, 2]) / Yn[k]
        Gv[k, 0] = S[k] * (g0 * E1[k, 0] + g1 * E1[k, 1] + g2 * E1[k, 2])
        Gv[k, 1] = S[k] * (g0 * E2[k, 0] + g1 * E2[k, 1] + g2 * E2[k, 2])


def _energy(V, free, X0, E1, E2, S, X, Yn, edges, lh, w, tris, ref, p, mu, tau, inv_ne, GX, Gv, R, FE, CE, Q, FB,
            ptr, inc):
    """f and its gradient Gv (F,2) at the offsets V (F,2); X, Yn, GX, R, FE, CE, Q, FB are work arrays (R gets r_e).
    Returns (f, ok); ok = False (and f = inf) when a triangle has det <= 0. Bit-identical for any thread count."""
    _positions(V, free, X0, E1, E2, S, X, Yn)
    _edge_pass(X, edges, lh, w, p, R, FE, CE)
    if len(tris):
        _tri_pass(X, tris, ref, tau, Q, FB)
        if Q.min() <= 0.0:
            return np.inf, False
    f = _bsum(FE) * inv_ne
    GX[free] = 0.0
    if len(tris):
        f += mu * _bsum(FB) * inv_ne
        _barrier_grad(X, tris, ref, tau, Q, mu * inv_ne, GX)
    _gather(X, Yn, free, E1, E2, S, edges, CE, ptr, inc, inv_ne, GX, Gv)
    return f, True


@njit(cache=True)
def _lawson(P, tris, nbr, vt, max_passes):
    """Lawson flips to the (locally) Delaunay triangulation, pass by pass over the triangles in index order; boundary
    edges (no neighbour) never flip; only strictly convex quads flip. Returns the number of flips."""
    total = 0
    for _ in range(max_passes):
        nf = 0
        for t in range(len(tris)):
            for k in range(3):
                u = nbr[t, k]
                if u < t:                      # each interior edge once; -1 = boundary
                    continue
                p, q, c = tris[t, k], tris[t, (k + 1) % 3], tris[t, (k + 2) % 3]
                j = 0
                for kk in range(3):
                    if tris[u, kk] == q and tris[u, (kk + 1) % 3] == p:
                        j = kk
                d = tris[u, (j + 2) % 3]
                if _incircle(P, p, q, c, d) == 1 and _orient(P, c, d, p) * _orient(P, c, d, q) < 0:
                    _flip(tris, nbr, vt, t, k)
                    nf += 1
                    break                      # t changed: its other edges are seen in the next pass
        total += nf
        if nf == 0:
            break
    return total


def lawson_flips(points: np.ndarray, tris: np.ndarray, max_passes: int = 100) -> tuple[np.ndarray, int]:
    """Copy of tris flipped toward the Delaunay triangulation (boundary edges kept); (tris, number of flips)."""
    tris = np.ascontiguousarray(tris, dtype=np.int64).copy()
    if len(tris) == 0:
        return tris, 0
    nbr = adjacency(tris)
    vt = np.full(len(points), -1, dtype=np.int64)
    vt[tris.ravel()] = np.repeat(np.arange(len(tris), dtype=np.int64), 3)
    n = _lawson(np.ascontiguousarray(points, dtype=np.float64), tris, nbr, vt, max_passes)
    return tris, int(n)


def edge_ln_sh(points: np.ndarray, edges: np.ndarray, h) -> np.ndarray:
    """Signed ln(s/h) per edge as the score measures it: great-circle length, h at the normalised midpoint."""
    p, q = points[edges[:, 0]], points[edges[:, 1]]
    mid = p + q
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    return np.log(R_EARTH_M * arc_length(p, q) / h(mid))


def _fixed_mask(mesh: Mesh, fixed) -> np.ndarray:
    n = mesh.n_nodes
    m = np.asarray(mesh.node_flag) != 0
    m = m | build_topology(mesh).boundary_node
    if fixed is not None:
        fixed = np.asarray(fixed)
        if fixed.dtype == bool:
            m = m | fixed
        else:
            m[fixed.astype(np.int64)] = True
    return m


def _barrier_tris(mesh: Mesh) -> np.ndarray:
    """Triangles of the barrier: the mesh triangles plus the 4 corner triangles of each quad."""
    q = mesh.quads
    parts = [mesh.tris] + [q[:, [k, (k + 1) % 4, (k + 3) % 4]] for k in range(4)] if len(q) else [mesh.tris]
    return np.ascontiguousarray(np.vstack(parts), dtype=np.int64)


@njit(cache=True, parallel=True)
def _two_loop(g, S, Y, RHO, head, cnt, d):
    """d = -H g by the L-BFGS two-loop recursion over the cnt newest pairs of the ring buffers (newest at head - 1)."""
    m = len(RHO)
    n = len(g)
    for i in prange(n):
        d[i] = g[i]
    al = np.zeros(m)
    for k in range(cnt):
        j = (head - 1 - k) % m
        a = RHO[j] * _dot(S[j], d)
        al[j] = a
        for i in prange(n):
            d[i] -= a * Y[j, i]
    j = (head - 1) % m
    gam = _dot(S[j], Y[j]) / _dot(Y[j], Y[j])
    for i in prange(n):
        d[i] *= gam
    for k in range(cnt - 1, -1, -1):
        j = (head - 1 - k) % m
        b = RHO[j] * _dot(Y[j], d)
        for i in prange(n):
            d[i] += (al[j] - b) * S[j, i]
    for i in prange(n):
        d[i] = -d[i]


def _lbfgs(fg, n, max_iter, m, step0, max_step, ftol, check=None):
    """Minimise fg (returns f, g, ok) from 0 in R^n. Returns (x, f, iterations, stop reason)."""
    x = np.zeros(n)
    f, g, ok = fg(x)
    S = np.zeros((m, n))
    Y = np.zeros((m, n))
    RHO = np.zeros(m)
    head, cnt = 0, 0
    d = np.empty(n)
    sv = np.empty(n)
    yv = np.empty(n)
    small = 0
    it = 0
    reason = "max_iter"
    for it in range(1, max_iter + 1):
        gmax = float(np.abs(g).max()) if n else 0.0
        if gmax == 0.0:
            reason = "gradient"
            break
        if cnt:
            _two_loop(g, S, Y, RHO, head, cnt, d)
            gd = _dot(g, d)
        if not cnt or not gd < 0.0:
            cnt = 0
            np.multiply(g, -step0 / gmax, out=d)
            gd = _dot(g, d)
        dm = float(np.abs(d).max())
        if dm > max_step:
            d *= max_step / dm
            gd *= max_step / dm
        t = 1.0
        for _ in range(40):
            xn = x + t * d
            fn, gn, ok = fg(xn)
            if ok and fn <= f + 1e-4 * t * gd:
                break
            t *= 0.5
        else:
            reason = "line_search"
            break
        np.subtract(xn, x, out=sv)
        np.subtract(gn, g, out=yv)
        sy = _dot(sv, yv)
        if sy > 1e-12 * np.sqrt(_dot(sv, sv) * _dot(yv, yv)):
            S[head] = sv
            Y[head] = yv
            RHO[head] = 1.0 / sy
            head = (head + 1) % m
            cnt = min(cnt + 1, m)
        rel = (f - fn) / max(abs(f), 1e-300)
        x, f, g = xn, fn, gn
        small = small + 1 if rel < ftol else 0
        if small >= 5:
            reason = "ftol"
            break
        if check is not None and it % 10 == 0 and check():
            reason = "target"
            break
    return x, f, it, reason


def size_fit(mesh: Mesh, h, fixed=None, p: int = 6, max_iter: int = 150, target_p99: float | None = None,
             rounds: int = 3, mu: float = 1.0, tau: float = 0.5, flips: bool = True, m: int = 10,
             step0: float = 0.05, max_step: float = 0.25, ftol: float = 1e-6) -> tuple[Mesh, dict]:
    """Copy of mesh with the free nodes moved to minimise the per-edge size energy (module docstring); info dict with
    per-round iterations, energies, flips, |ln s/h| p99 before / after, times."""
    t_start = time.perf_counter()
    fixed_m = _fixed_mask(mesh, fixed)
    free = np.flatnonzero(~fixed_m).astype(np.int64)
    P = np.ascontiguousarray(mesh.points, dtype=np.float64).copy()
    tris = np.ascontiguousarray(mesh.tris, dtype=np.int64).copy()
    cur = Mesh(P, tris, mesh.quads, mesh.node_flag)
    edges_all = build_topology(cur).edges
    r0 = edge_ln_sh(P, edges_all, h)
    info = {"p": p, "free_nodes": int(len(free)), "p99_before": float(np.percentile(np.abs(r0), 99)),
            "rounds": []}
    if len(free) == 0:
        info.update(p99_after=info["p99_before"], total_s=time.perf_counter() - t_start)
        return cur, info
    e_prev = None
    for rnd in range(rounds):
        t0 = time.perf_counter()
        topo_edges = edges_all
        movable = ~fixed_m[topo_edges[:, 0]] | ~fixed_m[topo_edges[:, 1]]
        edges = np.ascontiguousarray(topo_edges[movable])
        w = np.ones(len(edges))
        pm = P[edges[:, 0]] + P[edges[:, 1]]
        pm /= np.linalg.norm(pm, axis=1, keepdims=True)
        lh = np.log(h(pm) / R_EARTH_M)
        bt = _barrier_tris(cur)
        bt = np.ascontiguousarray(bt[~fixed_m[bt].all(axis=1)])
        cen = P[bt].sum(axis=1)
        cen /= np.linalg.norm(cen, axis=1, keepdims=True)
        ref = Q_EQ * (h(cen) / R_EARTH_M) ** 2
        X0 = P.copy()
        E1, E2 = tangent_basis(X0[free])
        E1, E2 = np.ascontiguousarray(E1), np.ascontiguousarray(E2)
        S = h(X0[free]) / R_EARTH_M
        X = X0.copy()
        Yn = np.ones(len(free))
        GX = np.zeros_like(X0)
        Gv = np.zeros((len(free), 2))
        R, FE, CE = np.zeros(len(edges)), np.zeros(len(edges)), np.zeros(len(edges))
        Q, FB = np.zeros(len(bt)), np.zeros(len(bt))
        inv_ne = 1.0 / max(len(edges), 1)
        pos = np.full(len(P), -1, dtype=np.int64)               # free node -> its CSR row of incident edges
        pos[free] = np.arange(len(free))
        ke = np.r_[pos[edges[:, 0]], pos[edges[:, 1]]]
        ee = np.r_[np.arange(len(edges)), np.arange(len(edges))]
        ok_ = ke >= 0
        ke, ee = ke[ok_], ee[ok_]
        o_ = np.argsort(ke, kind="stable")
        inc = np.ascontiguousarray(ee[o_])
        ptr = np.searchsorted(ke[o_], np.arange(len(free) + 1)).astype(np.int64)

        def fg(v):
            f, ok = _energy(v.reshape(-1, 2), free, X0, E1, E2, S, X, Yn, edges, lh, w, bt, ref, int(p),
                            float(mu), float(tau), inv_ne, GX, Gv, R, FE, CE, Q, FB, ptr, inc)
            return f, Gv.ravel().copy(), ok

        f0, _, ok0 = fg(np.zeros(2 * len(free)))
        check = None
        if target_p99 is not None:
            def check():
                return float(np.percentile(np.abs(R), 99)) <= target_p99
        v, f1, nit, reason = _lbfgs(fg, 2 * len(free), max_iter, m, step0, max_step, ftol, check)
        _positions(v.reshape(-1, 2), free, X0, E1, E2, S, P, Yn)
        nflip = 0
        if flips:
            tris, nflip = lawson_flips(P, tris)
        cur = Mesh(P, tris, mesh.quads, mesh.node_flag)
        if nflip:
            edges_all = build_topology(cur).edges
        r = edge_ln_sh(P, edges_all, h)
        p99 = float(np.percentile(np.abs(r), 99))
        info["rounds"].append({"iters": int(nit), "stop": reason, "E0": float(f0), "E": float(f1), "flips": int(nflip),
                               "p99": p99, "s": time.perf_counter() - t0, "start_ok": bool(ok0)})
        if target_p99 is not None and p99 <= target_p99:
            break
        if e_prev is not None and nflip == 0 and (e_prev - f1) < 1e-3 * abs(e_prev):
            break
        e_prev = f1
    info.update(p99_after=info["rounds"][-1]["p99"], total_s=time.perf_counter() - t_start)
    return cur, info


# ---------------------------------------------------------------------------------------- smoothness within a tolerance
#
# smooth_fit: the same chart, L-BFGS, barrier and rounds as size_fit, with another energy (decisions G38, M38):
#
#     E = (1/n_p) sum_pairs rho^p  +  w_tol (1/n_e) sum_e Huber(max(0, |r_e| - tol + margin))  +  w_h (1/n_e) sum_e r_e^2
#         + (1/n_c) sum_corners [w_angle (theta - theta*)^2 + w_wall (theta outside the wall)^2
#                                + w_hard (theta beyond the hard gates)^2]
#         + mu (1/n_e) sum_t max(0, tau - q_t)^2
#
# - rho = ln M_a - ln M_b over each interior edge between elements a, b (M = the element's mean edge chord): the
#   element size ratio of the quality gates (metrics.compute_quality, elem_size_ratio), as a smooth energy;
# - r_e = ln(l_e / h_e) as in size_fit (h_e frozen per round): the tolerance is a soft wall from |r| = tol - margin
#   on (margin 0.01: the soft wall then holds the edges at |r| <= tol, the gate's limit), inside it the size is free
#   (w_h = 0) or weakly pulled to h;
# - theta = every corner angle of an element with a free node (chord vectors, in (0, 2 pi) about the outward normal),
#   target 90 deg (quads) and 60 deg (triangles), and a wall at quad corners outside 60-120 deg and triangle corners
#   outside 35-100 deg: keeps the shapes (and the angle tails of the gates) the size smoothing would otherwise give
#   away (a cosine form has no restoring force at 180 deg: on the CORE2-like fill it let quads flatten to 180 deg);
# - 5 deg inside the hard angle gates a second, stiff wall (w_hard; at the gates themselves the soft wall let a
#   quad reach 150.1 deg), and the tolerance's pull is bounded (linear beyond
#   |r| = tol - margin + huber): a far-off tolerance (0.10 on the CORE2-like fill, 12 % of the edges beyond it) pulled
#   quad corners to 157 deg with the soft wall alone; a hard angle guard in the line search instead stalled the
#   L-BFGS at the first corner it reached (or at once, where a corner started beyond it);
# - the barrier and the inversion guard of the line search are size_fit's.
# Pairs, corners and edges enter only when they touch a free node (the others are constants).


QUAD_LO, QUAD_HI = np.radians(60.0), np.radians(120.0)     # the angle wall (M38): quad corners 60-120 deg,
TRI_LO, TRI_HI = np.radians(35.0), np.radians(100.0)        # triangle corners 35-100 deg
QUAD_LO2, QUAD_HI2 = np.radians(35.0), np.radians(145.0)  # the stiff second wall (w_hard), 5 deg inside the hard
TRI_LO2, TRI_HI2 = np.radians(25.0), np.radians(115.0)      # angle gates (G1 150, G3 120, G5 20; quads: SJ 0.57 > G6)


@njit(cache=True, parallel=True)
def _elem_pass(X, el, nv, LEN, SUM, LM):
    """Per element: its edge chords LEN[e, k] = |x_(k+1) - x_k|, their sum and ln(mean)."""
    for e in prange(len(el)):
        m = nv[e]
        s = 0.0
        for k in range(m):
            a, b = el[e, k], el[e, (k + 1) % m]
            d0, d1, d2 = X[a, 0] - X[b, 0], X[a, 1] - X[b, 1], X[a, 2] - X[b, 2]
            ll = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
            LEN[e, k] = ll
            s += ll
        SUM[e] = s
        LM[e] = np.log(s / m)


@njit(cache=True, parallel=True)
def _pair_pass(LM, pa, pb, p, FP, CP):
    for k in prange(len(pa)):
        r = LM[pa[k]] - LM[pb[k]]
        rp1 = r ** (p - 1)
        FP[k] = rp1 * r
        CP[k] = p * rp1


@njit(cache=True, parallel=True)
def _elem_coef(CP, eptr, epair, esgn, scale, SUM, CEL):
    """CEL[e] = scale * d/d ln M_e of the pair sum / sum of e's chords = the factor on d l / d x of e's edges."""
    for e in prange(len(CEL)):
        s = 0.0
        for j in range(eptr[e], eptr[e + 1]):
            s += esgn[j] * CP[epair[j]]
        CEL[e] = scale * s / SUM[e]


@njit(cache=True, inline="always")
def _theta(X, i, a, b, lu, lv):
    """Corner angle a-i-b in (0, 2 pi), CCW from u = x_a - x_i to v = x_b - x_i about the outward x_i: atan2(s, c) with
    c = u.v / (|u||v|) and s = det(x_i, x_a, x_b) / (|u||v|) (= (u x v).x_i / (|u||v|) on the unit sphere)."""
    c = ((X[a, 0] - X[i, 0]) * (X[b, 0] - X[i, 0]) + (X[a, 1] - X[i, 1]) * (X[b, 1] - X[i, 1])
         + (X[a, 2] - X[i, 2]) * (X[b, 2] - X[i, 2])) / (lu * lv)
    d = (X[i, 0] * (X[a, 1] * X[b, 2] - X[a, 2] * X[b, 1]) + X[i, 1] * (X[a, 2] * X[b, 0] - X[a, 0] * X[b, 2])
         + X[i, 2] * (X[a, 0] * X[b, 1] - X[a, 1] * X[b, 0]))
    t = np.arctan2(d / (lu * lv), c)
    return t + 2.0 * np.pi if t < 0.0 else t


@njit(cache=True, parallel=True)
def _corner_pass(X, el, nv, act, LEN, w_a, w_w, w_h2, scale, FA, CA):
    """Per active element: sum over its corners of w_a (theta - theta*)^2 + w_w (theta outside the wall)^2 (FA, rad) and
    scale times its derivative in theta per corner (CA); theta* = 90 deg (quad) / 60 deg (triangle), the wall at
    [QUAD_LO, QUAD_HI] / [TRI_LO, TRI_HI], plus w_h2 (theta beyond the hard gates)^2 (QUAD_LO2 ... TRI_HI2). The angle
    itself (not its cosine) keeps a restoring force at 180 deg."""
    for e in prange(len(el)):
        FA[e] = 0.0
        if not act[e]:
            for k in range(4):
                CA[e, k] = 0.0
            continue
        m = nv[e]
        if m == 3:
            t0, lo, hi, lo2, hi2 = np.pi / 3.0, TRI_LO, TRI_HI, TRI_LO2, TRI_HI2
        else:
            t0, lo, hi, lo2, hi2 = np.pi / 2.0, QUAD_LO, QUAD_HI, QUAD_LO2, QUAD_HI2
        f = 0.0
        for k in range(m):
            t = _theta(X, el[e, k], el[e, (k + 1) % m], el[e, (k + m - 1) % m], LEN[e, k], LEN[e, (k + m - 1) % m])
            f += w_a * (t - t0) * (t - t0)
            g = 2.0 * w_a * (t - t0)
            if t > hi:
                f += w_w * (t - hi) * (t - hi)
                g += 2.0 * w_w * (t - hi)
            elif t < lo:
                f += w_w * (lo - t) * (lo - t)
                g -= 2.0 * w_w * (lo - t)
            if t > hi2:
                f += w_h2 * (t - hi2) * (t - hi2)
                g += 2.0 * w_h2 * (t - hi2)
            elif t < lo2:
                f += w_h2 * (lo2 - t) * (lo2 - t)
                g -= 2.0 * w_h2 * (lo2 - t)
            CA[e, k] = scale * g
        FA[e] = f


@njit(cache=True, inline="always")
def _dtheta(X, i, a, b, lu, lv, which, out):
    """out[0:3] += out[3] * d theta(a-i-b) / d x_i (which 0), d x_a (1) or d x_b (2); theta as in _theta:
    d theta = (c ds - s dc) / (s^2 + c^2)."""
    w = out[3]
    luv = lu * lv
    u0, u1, u2 = X[a, 0] - X[i, 0], X[a, 1] - X[i, 1], X[a, 2] - X[i, 2]
    v0, v1, v2 = X[b, 0] - X[i, 0], X[b, 1] - X[i, 1], X[b, 2] - X[i, 2]
    c = (u0 * v0 + u1 * v1 + u2 * v2) / luv
    d = (X[i, 0] * (X[a, 1] * X[b, 2] - X[a, 2] * X[b, 1]) + X[i, 1] * (X[a, 2] * X[b, 0] - X[a, 0] * X[b, 2])
         + X[i, 2] * (X[a, 0] * X[b, 1] - X[a, 1] * X[b, 0]))
    s = d / luv
    q = 1.0 / (s * s + c * c)
    iu, iv = 1.0 / (lu * lu), 1.0 / (lv * lv)
    if which == 0:      # dc = -(dc/du + dc/dv); ds = (x_a x x_b)/luv + s u/lu^2 + s v/lv^2
        dc0 = -(v0 / luv - c * u0 * iu + u0 / luv - c * v0 * iv)
        dc1 = -(v1 / luv - c * u1 * iu + u1 / luv - c * v1 * iv)
        dc2 = -(v2 / luv - c * u2 * iu + u2 / luv - c * v2 * iv)
        ds0 = (X[a, 1] * X[b, 2] - X[a, 2] * X[b, 1]) / luv + s * (u0 * iu + v0 * iv)
        ds1 = (X[a, 2] * X[b, 0] - X[a, 0] * X[b, 2]) / luv + s * (u1 * iu + v1 * iv)
        ds2 = (X[a, 0] * X[b, 1] - X[a, 1] * X[b, 0]) / luv + s * (u2 * iu + v2 * iv)
    elif which == 1:    # dc/du; ds = (x_b x x_i)/luv - s u/lu^2
        dc0, dc1, dc2 = v0 / luv - c * u0 * iu, v1 / luv - c * u1 * iu, v2 / luv - c * u2 * iu
        ds0 = (X[b, 1] * X[i, 2] - X[b, 2] * X[i, 1]) / luv - s * u0 * iu
        ds1 = (X[b, 2] * X[i, 0] - X[b, 0] * X[i, 2]) / luv - s * u1 * iu
        ds2 = (X[b, 0] * X[i, 1] - X[b, 1] * X[i, 0]) / luv - s * u2 * iu
    else:               # dc/dv; ds = (x_i x x_a)/luv - s v/lv^2
        dc0, dc1, dc2 = u0 / luv - c * v0 * iv, u1 / luv - c * v1 * iv, u2 / luv - c * v2 * iv
        ds0 = (X[i, 1] * X[a, 2] - X[i, 2] * X[a, 1]) / luv - s * v0 * iv
        ds1 = (X[i, 2] * X[a, 0] - X[i, 0] * X[a, 2]) / luv - s * v1 * iv
        ds2 = (X[i, 0] * X[a, 1] - X[i, 1] * X[a, 0]) / luv - s * v2 * iv
    out[0] += w * q * (c * ds0 - s * dc0)
    out[1] += w * q * (c * ds1 - s * dc1)
    out[2] += w * q * (c * ds2 - s * dc2)


@njit(cache=True, parallel=True)
def _node_gather(X, el, nv, free, nptr, inc_e, inc_k, LEN, CEL, CA, GX):
    """Per free node: the ratio and corner terms of its elements, added to GX[i] (deterministic, no races)."""
    for f in prange(len(free)):
        i = free[f]
        out = np.zeros(4)
        for j in range(nptr[i], nptr[i + 1]):
            e, k = inc_e[j], inc_k[j]
            m = nv[e]
            kn, kp = (k + 1) % m, (k + m - 1) % m
            nx, pv = el[e, kn], el[e, kp]
            c = CEL[e]
            l1, l2 = LEN[e, k], LEN[e, kp]
            for d in range(3):
                out[d] += c * ((X[i, d] - X[nx, d]) / l1 + (X[i, d] - X[pv, d]) / l2)
            # corner at i (u = i->nx, v = i->pv)
            out[3] = CA[e, k]
            if out[3] != 0.0:
                _dtheta(X, i, nx, pv, l1, l2, 0, out)
            # corner at nx: u = nx->(k+2), v = nx->i, i is its b
            out[3] = CA[e, kn]
            if out[3] != 0.0:
                _dtheta(X, nx, el[e, (k + 2) % m], i, LEN[e, kn], l1, 2, out)
            # corner at pv: u = pv->i (i is its a), v = pv->(k-2)
            out[3] = CA[e, kp]
            if out[3] != 0.0:
                _dtheta(X, pv, i, el[e, (k + m - 2) % m], l2, LEN[e, (k + m - 2) % m], 1, out)
        GX[i, 0] += out[0]
        GX[i, 1] += out[1]
        GX[i, 2] += out[2]


@njit(cache=True, parallel=True)
def _tol_pass(X, edges, lh, tol, w_tol, huber, w_h, R, FE, CE):
    """Per edge: r = ln(l / h), the tolerance wall w_tol x^2 for x = |r| - tol > 0, linear beyond x = huber (a bounded
    pull where the tolerance cannot be met), and w_h r^2."""
    for e in prange(len(edges)):
        i, j = edges[e, 0], edges[e, 1]
        d0, d1, d2 = X[i, 0] - X[j, 0], X[i, 1] - X[j, 1], X[i, 2] - X[j, 2]
        c2 = d0 * d0 + d1 * d1 + d2 * d2
        r = 0.5 * np.log(c2) - lh[e]
        R[e] = r
        x = abs(r) - tol
        f, g = w_h * r * r, 2.0 * w_h * r
        if x > 0.0:
            sg = 1.0 if r > 0.0 else -1.0
            if x < huber:
                f += w_tol * x * x
                g += 2.0 * w_tol * x * sg
            else:
                f += w_tol * huber * (2.0 * x - huber)
                g += 2.0 * w_tol * huber * sg
        FE[e] = f
        CE[e] = g / c2


def ratio_pairs(mesh: Mesh) -> np.ndarray:
    """(P,2) element ids (triangles first, then quads) on the two sides of every interior edge."""
    ee = build_topology(mesh).edge_elems
    return np.ascontiguousarray(ee[ee[:, 1] >= 0], dtype=np.int64)


def ln_ratio(mesh: Mesh) -> np.ndarray:
    """|ln(M_a / M_b)| per interior edge (M = mean edge chord): ln of the gates' element size ratio."""
    from meshom.smooth import _elements

    el, nv = _elements(mesh)
    LEN, SUM, LM = np.zeros((len(el), 4)), np.zeros(len(el)), np.zeros(len(el))
    _elem_pass(np.ascontiguousarray(mesh.points, dtype=np.float64), el, nv, LEN, SUM, LM)
    pr = ratio_pairs(mesh)
    return np.abs(LM[pr[:, 0]] - LM[pr[:, 1]])


def smooth_problem(cur: Mesh, h, fixed_m: np.ndarray, free: np.ndarray, tol: float, p: int = 2, w_tol: float = 1e3,
                   w_h: float = 0.0, w_angle: float = 0.1, w_wall: float = 100.0, mu: float = 1.0, tau: float = 0.5,
                   margin: float = 0.01, huber: float = 0.05, w_hard: float = 1e4):
    """One round's energy of smooth_fit at the positions of cur: (fg, X0, E1, E2, S, Yn); fg(v) -> (f, gradient,
    ok) at the tangent offsets v (2F,) of the free nodes (h frozen at cur)."""
    from meshom.smooth import _elements, _incidence

    P = cur.points
    el, nv = _elements(cur)
    nptr, inc_e, inc_k = _incidence(el, nv, len(P))
    act = np.zeros(len(el), dtype=bool)
    act[np.repeat(np.arange(len(el)), 4)[(el.ravel() >= 0) & ~fixed_m[np.maximum(el.ravel(), 0)]]] = True
    pr = ratio_pairs(cur)
    pr = np.ascontiguousarray(pr[act[pr[:, 0]] | act[pr[:, 1]]])
    pa, pb = np.ascontiguousarray(pr[:, 0]), np.ascontiguousarray(pr[:, 1])
    ke = np.r_[pa, pb]                                         # element -> (pair, sign) CSR
    kp = np.r_[np.arange(len(pa)), np.arange(len(pa))]
    ks = np.r_[np.ones(len(pa)), -np.ones(len(pa))]
    o_ = np.argsort(ke, kind="stable")
    epair, esgn = np.ascontiguousarray(kp[o_]), np.ascontiguousarray(ks[o_])
    eptr = np.searchsorted(ke[o_], np.arange(len(el) + 1)).astype(np.int64)
    topo_edges = build_topology(cur).edges
    movable = ~fixed_m[topo_edges[:, 0]] | ~fixed_m[topo_edges[:, 1]]
    edges = np.ascontiguousarray(topo_edges[movable])
    pm = P[edges[:, 0]] + P[edges[:, 1]]
    pm /= np.linalg.norm(pm, axis=1, keepdims=True)
    lh = np.log(h(pm) / R_EARTH_M)
    bt = _barrier_tris(cur)
    bt = np.ascontiguousarray(bt[~fixed_m[bt].all(axis=1)])
    cen = P[bt].sum(axis=1)
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    ref = Q_EQ * (h(cen) / R_EARTH_M) ** 2
    X0 = P.copy()
    E1, E2 = tangent_basis(X0[free])
    E1, E2 = np.ascontiguousarray(E1), np.ascontiguousarray(E2)
    S = h(X0[free]) / R_EARTH_M
    X = X0.copy()
    Yn = np.ones(len(free))
    GX = np.zeros_like(X0)
    Gv = np.zeros((len(free), 2))
    R, FE, CE = np.zeros(len(edges)), np.zeros(len(edges)), np.zeros(len(edges))
    Q, FB = np.zeros(len(bt)), np.zeros(len(bt))
    LEN, SUM, LM = np.zeros((len(el), 4)), np.ones(len(el)), np.zeros(len(el))
    FP, CP = np.zeros(len(pa)), np.zeros(len(pa))
    CEL = np.zeros(len(el))
    FA, CA = np.zeros(len(el)), np.zeros((len(el), 4))
    inv_ne = 1.0 / max(len(edges), 1)
    inv_np = 1.0 / max(len(pa), 1)
    inv_nc = 1.0 / max(int(nv[act].sum()), 1)
    pos = np.full(len(P), -1, dtype=np.int64)
    pos[free] = np.arange(len(free))
    kk = np.r_[pos[edges[:, 0]], pos[edges[:, 1]]]
    ee = np.r_[np.arange(len(edges)), np.arange(len(edges))]
    ok_ = kk >= 0
    kk, ee = kk[ok_], ee[ok_]
    o_ = np.argsort(kk, kind="stable")
    inc = np.ascontiguousarray(ee[o_])
    ptr = np.searchsorted(kk[o_], np.arange(len(free) + 1)).astype(np.int64)
    pi = int(p)

    def fg(v):
        _positions(v.reshape(-1, 2), free, X0, E1, E2, S, X, Yn)
        if len(bt):
            _tri_pass(X, bt, ref, tau, Q, FB)
            if Q.min() <= 0.0:
                return np.inf, Gv.ravel().copy(), False
        _elem_pass(X, el, nv, LEN, SUM, LM)
        _pair_pass(LM, pa, pb, pi, FP, CP)
        _elem_coef(CP, eptr, epair, esgn, inv_np, SUM, CEL)
        _corner_pass(X, el, nv, act, LEN, w_angle, w_wall, w_hard, inv_nc, FA, CA)
        _tol_pass(X, edges, lh, tol - margin, w_tol, huber, w_h, R, FE, CE)
        f = _bsum(FP) * inv_np + _bsum(FE) * inv_ne + inv_nc * _bsum(FA)
        GX[free] = 0.0
        if len(bt):
            f += mu * _bsum(FB) * inv_ne
            _barrier_grad(X, bt, ref, tau, Q, mu * inv_ne, GX)
        _node_gather(X, el, nv, free, nptr, inc_e, inc_k, LEN, CEL, CA, GX)
        _gather(X, Yn, free, E1, E2, S, edges, CE, ptr, inc, inv_ne, GX, Gv)
        return f, Gv.ravel().copy(), True

    return fg, X0, E1, E2, S, Yn


def smooth_fit(mesh: Mesh, h, tol: float, fixed=None, p: int = 2, w_tol: float = 1e3, w_h: float = 0.0,
               w_angle: float = 0.1, w_wall: float = 100.0, margin: float = 0.01, huber: float = 0.05,
               w_hard: float = 1e4, max_iter: int = 500, rounds: int = 5, mu: float = 1.0, tau: float = 0.5,
               flips: bool = False, m: int = 10, step0: float = 0.05, max_step: float = 0.25,
               ftol: float = 1e-10) -> tuple[Mesh, dict]:
    """Copy of mesh with the free nodes moved to make the element sizes vary as gently as possible between
    neighbours while every edge stays within |ln s/h| <= tol (the soft wall; block comment above); info dict with the
    ratio and |ln s/h| p99 before / after per round, iterations, times. Fixed nodes as in size_fit."""
    t_start = time.perf_counter()
    fixed_m = _fixed_mask(mesh, fixed)
    free = np.flatnonzero(~fixed_m).astype(np.int64)
    P = np.ascontiguousarray(mesh.points, dtype=np.float64).copy()
    tris = np.ascontiguousarray(mesh.tris, dtype=np.int64).copy()
    cur = Mesh(P, tris, mesh.quads, mesh.node_flag)

    def stats(msh):
        r = np.abs(edge_ln_sh(msh.points, build_topology(msh).edges, h))
        lr = ln_ratio(msh)
        return {"ratio_p99": float(np.exp(np.percentile(lr, 99))) if len(lr) else None,
                "ratio_max": float(np.exp(lr.max())) if len(lr) else None,
                "ln_sh_p99": float(np.percentile(r, 99)), "share_over_tol": float(np.mean(r > tol))}

    info = {"tol": tol, "p": p, "w_tol": w_tol, "w_h": w_h, "w_angle": w_angle, "w_wall": w_wall, "margin": margin,
            "huber": huber, "w_hard": w_hard,
            "free_nodes": int(len(free)),
            "before": stats(cur), "rounds": []}
    if len(free) == 0:
        info.update(after=info["before"], total_s=time.perf_counter() - t_start)
        return cur, info
    e_prev = None
    for rnd in range(rounds):
        t0 = time.perf_counter()
        fg, X0, E1, E2, S, Yn = smooth_problem(cur, h, fixed_m, free, tol, p, w_tol, w_h, w_angle, w_wall, mu, tau,
                                               margin, huber, w_hard)
        f0, _, ok0 = fg(np.zeros(2 * len(free)))
        v, f1, nit, reason = _lbfgs(fg, 2 * len(free), max_iter, m, step0, max_step, ftol)
        _positions(v.reshape(-1, 2), free, X0, E1, E2, S, P, Yn)
        nflip = 0
        if flips and len(tris):
            tris, nflip = lawson_flips(P, tris)
        cur = Mesh(P, tris, mesh.quads, mesh.node_flag)
        rec = {"iters": int(nit), "stop": reason, "E0": float(f0), "E": float(f1), "flips": int(nflip),
               "s": time.perf_counter() - t0, "start_ok": bool(ok0), **stats(cur)}
        info["rounds"].append(rec)
        if e_prev is not None and nflip == 0 and (e_prev - f1) < 1e-3 * abs(e_prev):
            break
        e_prev = f1
    info.update(after=info["rounds"][-1] if info["rounds"] else info["before"],
                total_s=time.perf_counter() - t_start)
    return cur, info
