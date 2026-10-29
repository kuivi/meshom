"""Free-node geometric optimisation on the sphere (mesh fixing, competitor A; decisions B122-B125).

optimize_geometry(mesh, h, fixed, slide, opts) moves nodes only (no topology change). Every node that is not fixed
is free (ORCA nodes included); `slide` nodes (the coast) move along the reference coast. Per round, at the positions
the round starts from (the chart, h at the edge midpoints and the coast candidates frozen), L-BFGS minimises over the
tangent offsets v of the free nodes, x = normalise(x0 + s (v1 e1 + v2 e2)), s = h(x0) / R:

  E = sum over corners of active elements (theta in rad, at the corner node's tangent plane)
        w_shape (theta - theta*)^2                                  theta* = 90 deg quads, 60 deg triangles
      + w_tail  max(0, theta - T)^2  + w_tail2 max(0, theta - T2)^2 + w_tail3 max(0, theta - T3)^2
                                                     quads T = 100, T2 = 110, T3 = 120 deg (the tail)
      + w_tail_t max(0, theta - Tt)^2 + w_tail_t max(0, Tt_lo - theta)^2      triangles Tt = 80, Tt_lo = 40 deg
      + w_guard max(0, theta - GHI_e)^2 + w_guard max(0, GLO_e - theta)^2      the per-element no-worse guard:
            GHI_e = max(input max angle of e, guard_q | guard_t), GLO_e = min(input min angle of e, lo_q | lo_t)
  + sum over edges with a free end
        w_size (ln l - ln l_in)^2 + w_wall max(0, |ln l/h_e| - W_e)^2      W_e = max(wall - margin, |input ln s/h|)
                                       (margin_coast instead of margin on the edges of the gates' _coast3 elements)
        + w_wall_max max(0, |ln l/h_e| - W2_e)^2                          W2_e = max(wall + wall_over, |input ln s/h|)
  + sum over interior edges between elements a, b (rho = ln M_a - ln M_b, M = mean edge chord)
        w_ratio max(0, |rho| - ln ratio_soft)^2 + w_wall max(0, |rho| - RW_k)^2
                                                                    RW_k = max(ln ratio_wall, |input rho|)
  + sum over slide nodes (d = chord distance to the nearest reference coast segment, u = displacement from the input,
    both in units of s)
        w_coast d^2 + w_wall max(0, d - D_k)^2 + w_wall max(0, u - coast_move)^2
                                                                    D_k = max(coast_dist, the input's own d)
  + with a reference coast `ref` (B146; optimize_geometry(..., ref=case.coast_seg)): the slide nodes' d is measured
    to the reference segments instead of the input's boundary, and per reference vertex V near a slide node
    (dv = chord distance of V to the current coast edges, in units of h(V) / R; frozen candidate edges per round)
        w_wall max(0, dv - RV_V)^2 + w_wall_max max(0, dv - RV2_V)^2
                    RV_V = max(coast_move - ref_margin, input dv), RV2_V = max(coast_move - ref_margin2, input dv)
    so that the scorer's backward coast shift stays within coast_move h of the case's coast whatever the stages
    before A moved (A's own displacement wall u is still measured from its input).

The line search rejects any trial with a reflex or inverted corner in an active element (a valid start stays
valid), and a corner more than guard_hard (1 deg) beyond its guard (a hard limit at the guard itself stalled L-BFGS;
1 deg beyond the soft guard it rarely binds, and it keeps the post-round revert, which breaks the size walls between
reverted and moved nodes, from firing). L-BFGS stops when the energy fell by less than wtol relative over `window`
iterations; a failed line search restarts it from steepest descent. After each round a hard check with the scorer's
kernel (element tangent planes) reverts the nodes of every element that is invalid or breaks its guard by more than
guard_tol to the round's start, until none is left. Energy and gradient in Numba (deterministic per-node gathers,
size_fit's kernels). active_deg > 0 frees only the nodes within active_rings rings of a defect or a quad above
active_deg (active_nodes), for large meshes whose regular parts need no work.
"""
import time

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M, tangent_basis
from meshom.mesh import Mesh, build_topology
from meshom.quality.kernel import element_geometry
from meshom.size_fit import (_bsum, _dot, _elem_coef, _elem_pass, _gather, _node_gather, _positions, _theta, _two_loop,
                             edge_ln_sh, ratio_pairs)
from meshom.smooth import _elements, _incidence

DEG = np.pi / 180.0
REF_REVERT_H = 1.5   # B146: a reference vertex beyond the coast-shift limit puts back the slide nodes within this (h)

# B122-B124: the energy's thresholds and weights (first guesses, tuned on the cut-outs; see docs/decisions.md)
DEFAULTS = {
    "w_shape": 0.3, "w_shape_t": 0.5,
    "tail_q": 100.0, "w_tail": 50.0, "tail_q2": 110.0, "w_tail2": 200.0, "tail_q3": 120.0, "w_tail3": 1000.0,
    "tail_t": 80.0, "tail_t_lo": 40.0, "w_tail_t": 50.0,
    "guard_q": 120.0, "guard_t": 90.0, "lo_q": 45.0, "lo_t": 35.0, "w_guard": 1e3, "guard_hard": 1.0,
    "guard_tol": 1.1, "near": 0.02,
    "wall": 0.20, "margin": 0.02, "margin_coast": 0.04, "w_size": 0.2, "w_wall": 1e3, "wall_over": 0.06,
    "w_wall_max": 1e5,
    "ratio_soft": 1.05, "w_ratio": 1.0, "ratio_wall": 1.12,
    "coast_dist": 0.10, "coast_move": 0.25, "w_coast": 10.0, "coast_k": 8,
    "ref_margin": 0.03, "ref_margin2": 0.005, "ref_reach": 2.0,
    "active_deg": 100.0, "active_rings": 8,
    "rounds": 4, "max_iter": 1500, "m": 10, "step0": 0.05, "max_step": 0.25, "window": 50, "wtol": 3e-4,
    "restarts": 50, "round_tol": 1e-3,
}


# ------------------------------------------------------------------------------------------------ Numba kernels

@njit(cache=True, parallel=True)
def _corners(X, el, nv, act, LEN, prm, GHI, GLO, HHI, HLO, FA, CA, BAD):
    """Per active element: its corner energy FA[e] and dF/dtheta per corner CA[e, k]; BAD[e] = 1 when a corner is
    reflex or inverted (theta >= pi) or outside its hard limits [HLO_e, HHI_e], 2 when a corner is within prm[12] of
    them (near: the nodes a blocked line search freezes). prm: w_shape, w_shape_t, tq, w_tq, tq2, w_tq2, tt, tt_lo,
    w_tt, w_guard, tq3, w_tq3, near (rad)."""
    w_sq, w_st, tq, w_tq, tq2, w_tq2, tt, tt_lo, w_tt, w_g = (prm[0], prm[1], prm[2], prm[3], prm[4], prm[5],
                                                             prm[6], prm[7], prm[8], prm[9])
    tq3, w_tq3, nm = prm[10], prm[11], prm[12]
    for e in prange(len(el)):
        FA[e] = 0.0
        BAD[e] = 0
        for k in range(4):
            CA[e, k] = 0.0
        if not act[e]:
            continue
        m = nv[e]
        f = 0.0
        ghi, glo = GHI[e], GLO[e]
        for k in range(m):
            t = _theta(X, el[e, k], el[e, (k + 1) % m], el[e, (k + m - 1) % m], LEN[e, k], LEN[e, (k + m - 1) % m])
            if t >= np.pi or t > HHI[e] or t < HLO[e]:
                BAD[e] = 1
            elif BAD[e] == 0 and (t >= np.pi - nm or t > HHI[e] - nm or t < HLO[e] + nm):
                BAD[e] = 2
            if m == 4:
                x = t - 0.5 * np.pi
                f += w_sq * x * x
                g = 2.0 * w_sq * x
                x = t - tq
                if x > 0.0:
                    f += w_tq * x * x
                    g += 2.0 * w_tq * x
                x = t - tq2
                if x > 0.0:
                    f += w_tq2 * x * x
                    g += 2.0 * w_tq2 * x
                x = t - tq3
                if x > 0.0:
                    f += w_tq3 * x * x
                    g += 2.0 * w_tq3 * x
            else:
                x = t - np.pi / 3.0
                f += w_st * x * x
                g = 2.0 * w_st * x
                x = t - tt
                if x > 0.0:
                    f += w_tt * x * x
                    g += 2.0 * w_tt * x
                x = tt_lo - t
                if x > 0.0:
                    f += w_tt * x * x
                    g -= 2.0 * w_tt * x
            x = t - ghi
            if x > 0.0:
                f += w_g * x * x
                g += 2.0 * w_g * x
            x = glo - t
            if x > 0.0:
                f += w_g * x * x
                g -= 2.0 * w_g * x
            CA[e, k] = g
        FA[e] = f


@njit(cache=True, parallel=True)
def _edges(X, edges, lh, ll0, W, W2, w_s, w_w, w_w2, R, FE, CE):
    """Per edge: r = ln(l / h_e); w_s (ln l - ln l_in)^2 + w_w max(0, |r| - W_e)^2 + w_w2 max(0, |r| - W2_e)^2;
    CE = dF/dr / l^2."""
    for e in prange(len(edges)):
        i, j = edges[e, 0], edges[e, 1]
        d0, d1, d2 = X[i, 0] - X[j, 0], X[i, 1] - X[j, 1], X[i, 2] - X[j, 2]
        c2 = d0 * d0 + d1 * d1 + d2 * d2
        ll = 0.5 * np.log(c2)
        r = ll - lh[e]
        R[e] = r
        y = ll - ll0[e]
        f = w_s * y * y
        g = 2.0 * w_s * y
        x = abs(r) - W[e]
        if x > 0.0:
            sg = 1.0 if r > 0.0 else -1.0
            f += w_w * x * x
            g += 2.0 * w_w * x * sg
        x = abs(r) - W2[e]
        if x > 0.0:
            sg = 1.0 if r > 0.0 else -1.0
            f += w_w2 * x * x
            g += 2.0 * w_w2 * x * sg
        FE[e] = f
        CE[e] = g / c2


@njit(cache=True, parallel=True)
def _pairs(LM, pa, pb, RW, rs, w_rs, w_rw, FP, CP):
    """Per interior edge: rho = ln M_a - ln M_b; w_rs max(0, |rho| - rs)^2 + w_rw max(0, |rho| - RW_k)^2;
    CP = dF/drho."""
    for k in prange(len(pa)):
        r = LM[pa[k]] - LM[pb[k]]
        a = abs(r)
        sg = 1.0 if r > 0.0 else -1.0
        f, g = 0.0, 0.0
        x = a - rs
        if x > 0.0:
            f += w_rs * x * x
            g += 2.0 * w_rs * x * sg
        x = a - RW[k]
        if x > 0.0:
            f += w_rw * x * x
            g += 2.0 * w_rw * x * sg
        FP[k] = f
        CP[k] = g


@njit(cache=True, parallel=True)
def _coast(X, cn, X_in, S, SA, SB, cptr, cseg, dmax, umax, w_d, w_w, FC, GX):
    """Per slide node: nearest reference segment among its candidates; the coast terms (module docstring) into FC and
    their gradient (in x) added to GX (each node once: no races)."""
    for k in prange(len(cn)):
        i = cn[k]
        s = S[k]
        best = 1e300
        p0, p1, p2 = X[i, 0], X[i, 1], X[i, 2]
        q0, q1, q2 = p0, p1, p2
        for jj in range(cptr[k], cptr[k + 1]):
            j = cseg[jj]
            a0, a1, a2 = SA[j, 0], SA[j, 1], SA[j, 2]
            b0, b1, b2 = SB[j, 0] - a0, SB[j, 1] - a1, SB[j, 2] - a2
            bb = b0 * b0 + b1 * b1 + b2 * b2
            t = ((p0 - a0) * b0 + (p1 - a1) * b1 + (p2 - a2) * b2) / bb if bb > 0.0 else 0.0
            t = min(1.0, max(0.0, t))
            c0, c1, c2 = a0 + t * b0, a1 + t * b1, a2 + t * b2
            d2 = (p0 - c0) ** 2 + (p1 - c1) ** 2 + (p2 - c2) ** 2
            if d2 < best:
                best = d2
                q0, q1, q2 = c0, c1, c2
        dd = np.sqrt(best)
        d = dd / s
        f = w_d * d * d
        gs = 2.0 * w_d / (s * s)                      # gradient = gs * (x - q)
        x = d - dmax[k]
        if x > 0.0 and dd > 0.0:
            f += w_w * x * x
            gs += 2.0 * w_w * x / (s * dd)
        g0, g1, g2 = gs * (p0 - q0), gs * (p1 - q1), gs * (p2 - q2)
        u0, u1, u2 = p0 - X_in[k, 0], p1 - X_in[k, 1], p2 - X_in[k, 2]
        uu = np.sqrt(u0 * u0 + u1 * u1 + u2 * u2)
        x = uu / s - umax
        if x > 0.0:
            f += w_w * x * x
            c = 2.0 * w_w * x / (s * uu)
            g0 += c * u0
            g1 += c * u1
            g2 += c * u2
        FC[k] = f
        GX[i, 0] += g0
        GX[i, 1] += g1
        GX[i, 2] += g2


@njit(cache=True)
def _refwall(X, V, SV, LIM, LIM2, vptr, vedge, CE, isfree, w1, w2, GX):
    """B146: per reference vertex V[k] the chord distance to the nearest of its candidate coast edges CE[vedge[...]]
    at X, in units of SV[k]; the two walls (module docstring) summed and their gradient (in x) added to GX at the free
    edge ends (serial: an edge end is shared by several vertices)."""
    f = 0.0
    for k in range(len(V)):
        v0, v1, v2 = V[k, 0], V[k, 1], V[k, 2]
        best = 1e300
        ba, bb, bt = -1, -1, 0.0
        q0, q1, q2 = v0, v1, v2
        for jj in range(vptr[k], vptr[k + 1]):
            a, b = CE[vedge[jj], 0], CE[vedge[jj], 1]
            a0, a1, a2 = X[a, 0], X[a, 1], X[a, 2]
            d0, d1, d2 = X[b, 0] - a0, X[b, 1] - a1, X[b, 2] - a2
            dd = d0 * d0 + d1 * d1 + d2 * d2
            t = ((v0 - a0) * d0 + (v1 - a1) * d1 + (v2 - a2) * d2) / dd if dd > 0.0 else 0.0
            t = min(1.0, max(0.0, t))
            c0, c1, c2 = a0 + t * d0, a1 + t * d1, a2 + t * d2
            e2 = (v0 - c0) ** 2 + (v1 - c1) ** 2 + (v2 - c2) ** 2
            if e2 < best:
                best, ba, bb, bt = e2, a, b, t
                q0, q1, q2 = c0, c1, c2
        if ba < 0:
            continue
        dist = np.sqrt(best)
        s = SV[k]
        dv = dist / s
        x1, x2 = dv - LIM[k], dv - LIM2[k]
        if x1 <= 0.0 and x2 <= 0.0:
            continue
        gd = 0.0                                            # dE / d dv
        if x1 > 0.0:
            f += w1 * x1 * x1
            gd += 2.0 * w1 * x1
        if x2 > 0.0:
            f += w2 * x2 * x2
            gd += 2.0 * w2 * x2
        if dist <= 0.0:
            continue
        c = -gd / (s * dist)                                # d dist / d q = -(V - q) / dist; q = (1-t) A + t B
        u0, u1, u2 = c * (v0 - q0), c * (v1 - q1), c * (v2 - q2)
        if isfree[ba]:
            GX[ba, 0] += (1.0 - bt) * u0
            GX[ba, 1] += (1.0 - bt) * u1
            GX[ba, 2] += (1.0 - bt) * u2
        if isfree[bb]:
            GX[bb, 0] += bt * u0
            GX[bb, 1] += bt * u1
            GX[bb, 2] += bt * u2
    return f


@njit(cache=True, parallel=True)
def _corner_range(X, el, nv, act, LEN, HHI, HLO):
    """Widen [HLO_e, HHI_e] to contain the corners of e at X (plus 1e-9 rad)."""
    for e in prange(len(el)):
        if not act[e]:
            continue
        m = nv[e]
        for k in range(m):
            t = _theta(X, el[e, k], el[e, (k + 1) % m], el[e, (k + m - 1) % m], LEN[e, k], LEN[e, (k + m - 1) % m])
            if t + 1e-9 > HHI[e]:
                HHI[e] = t + 1e-9
            if t - 1e-9 < HLO[e]:
                HLO[e] = t - 1e-9


def lbfgs(fg, n, max_iter, m, step0, max_step, window: int = 50, wtol: float = 1e-4, restarts: int = 20,
          freeze=None):
    """L-BFGS from 0 in R^n (size_fit._lbfgs's two-loop recursion, first step and per-node step cap, Armijo
    backtracking that also rejects a trial with ok = False). Stops when the energy fell by less than wtol relative
    over the last `window` iterations, or at max_iter. A failed line search clears the memory and restarts with a
    steepest-descent step (up to `restarts` times); before that, freeze(x) may fix the variables that block it (a
    corner at a hard limit): when it freezes any, the memory is cleared and the search goes on. Returns (x, f,
    iterations, stop reason, restarts used)."""
    x = np.zeros(n)
    f, g, ok = fg(x)
    S, Y, RHO = np.zeros((m, n)), np.zeros((m, n)), np.zeros(m)
    head, cnt, used = 0, 0, 0
    d, sv, yv = np.empty(n), np.empty(n), np.empty(n)
    hist = [f]
    reason = "max_iter"
    it = 0
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
            if freeze is not None and used < restarts and freeze(x) > 0:
                cnt = 0
                used += 1
                f, g, ok = fg(x)
                continue
            if cnt and used < restarts:
                cnt = 0
                used += 1
                continue
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
        x, f, g = xn, fn, gn
        hist.append(f)
        if len(hist) > window and hist[-1 - window] - f < wtol * abs(f):
            reason = "window"
            break
    return x, f, it, reason, used


# ------------------------------------------------------------------------------------------------ helpers

def active_nodes(mesh: Mesh, deg: float, rings: int, tri_hi: float = 80.0, tri_lo: float = 40.0) -> np.ndarray:
    """(N,) nodes within `rings` element rings of a seed element: a quad whose largest corner is above deg, any
    triangle, or an element at an interior node with != 4 quads. The active set of a large mesh: regular parts far
    from any defect (ORCA blocks) stay put. A pure-triangle mesh (B157) seeds at its triangles with a corner
    above tri_hi or below tri_lo (A's triangle tail thresholds): there every element is a triangle and valence
    defects are spread everywhere (FESOM2's farc: 43 % of its nodes), so the quad rule would free every node."""
    hi, lo = _elem_angles(mesh)
    T = mesh.n_tris
    el, nv = _elements(mesh)
    N = mesh.n_nodes
    if mesh.n_quads == 0:
        seed = (hi > tri_hi) | (lo < tri_lo)
    else:
        seed = np.r_[hi[:T] > deg - 30.0, hi[T:] > deg]
        nq = np.bincount(mesh.quads.ravel(), minlength=N)
        irregular = (nq != 4) & ~build_topology(mesh).boundary_node
        seed |= ((el >= 0) & irregular[np.maximum(el, 0)]).any(axis=1)
        seed[:T] = True
    node = np.zeros(N, bool)
    node[el[seed][el[seed] >= 0]] = True
    for _ in range(rings):
        e_on = ((el >= 0) & node[np.maximum(el, 0)]).any(axis=1)
        node[el[e_on][el[e_on] >= 0]] = True
    return node



def _elem_angles(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """(T+Q,) largest and smallest corner angle (deg) per element in its tangent plane (the scorer's kernel)."""
    g = element_geometry(mesh)
    parts_hi, parts_lo = [], []
    for a in (g.tri_angles, g.quad_angles):
        if len(a):
            d = np.degrees(a)
            parts_hi.append(d.max(axis=1))
            parts_lo.append(d.min(axis=1))
    return np.concatenate(parts_hi), np.concatenate(parts_lo)


def _bad_elements(mesh: Mesh, act: np.ndarray, ghi: np.ndarray, glo: np.ndarray, tol: float) -> np.ndarray:
    """Active elements that are inverted, reflex, degenerate or break their guard by more than tol deg."""
    from meshom.mesh import signed_area

    hi, lo = _elem_angles(mesh)
    area = signed_area(mesh)
    bad = (area <= 0) | (hi >= 180.0) | (lo < 1.0) | (hi > ghi + tol) | (lo < glo - tol)
    return bad & act


def _coast_candidates(P: np.ndarray, cn: np.ndarray, SA: np.ndarray, SB: np.ndarray, k: int):
    """CSR (ptr, seg) of the k reference segments nearest (by midpoint) to each slide node."""
    if len(cn) == 0 or len(SA) == 0:
        return np.zeros(len(cn) + 1, np.int64), np.zeros(0, np.int64)
    mid = SA + SB
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    k = min(k, len(SA))
    _, j = cKDTree(mid).query(P[cn], k=k)
    j = np.asarray(j, dtype=np.int64).reshape(len(cn), k)
    return np.arange(0, k * len(cn) + 1, k, dtype=np.int64), np.ascontiguousarray(j.ravel())


def stats(mesh: Mesh, h, edges: np.ndarray, pairs: np.ndarray, tol: float) -> dict:
    """Quick measures of a round: quad max angle tail, |ln s/h| p99, element ratio p99 / max."""
    hi, _ = _elem_angles(mesh)
    q = hi[mesh.n_tris:]
    r = np.abs(edge_ln_sh(mesh.points, edges, h))
    LEN, SUM, LM = np.zeros((mesh.n_elems, 4)), np.zeros(mesh.n_elems), np.zeros(mesh.n_elems)
    el, nv = _elements(mesh)
    _elem_pass(np.ascontiguousarray(mesh.points), el, nv, LEN, SUM, LM)
    lr = np.abs(LM[pairs[:, 0]] - LM[pairs[:, 1]]) if len(pairs) else np.zeros(1)
    return {"q_gt110": int((q > 110).sum()), "q_gt120": int((q > 120).sum()),
            "q_p999": float(np.percentile(q, 99.9)) if len(q) else None, "q_max": float(q.max()) if len(q) else None,
            "ln_sh_p99": float(np.percentile(r, 99)), "ln_sh_max": float(r.max()),
            "share_gt_tol": float(np.mean(r > tol)),
            "ratio_p99": float(np.exp(np.percentile(lr, 99))), "ratio_max": float(np.exp(lr.max()))}


# ------------------------------------------------------------------------------------------------ driver

class Problem:
    """The static part of the energy for one mesh (connectivity, masks, the input's guards and walls) and the work
    arrays; round(X0) freezes the chart, h and the coast candidates at X0 and returns fg(v) -> (f, grad, ok)."""

    def __init__(self, mesh: Mesh, h, fixed=None, slide=None, opts: dict | None = None, ref=None):
        o = self.o = {**DEFAULTS, **(opts or {})}
        self.mesh, self.h = mesh, h
        N = mesh.n_nodes
        topo = build_topology(mesh)
        bnd = topo.boundary_node
        fixed = np.zeros(N, bool) if fixed is None else np.asarray(fixed, bool).copy()
        slide = (bnd & ~fixed) if slide is None else (np.asarray(slide, bool) & bnd & ~fixed)
        fixed |= bnd & ~slide
        if o["active_deg"] > 0:
            fixed |= ~active_nodes(mesh, o["active_deg"], int(o["active_rings"]), o["tail_t"], o["tail_t_lo"])
        slide = slide & ~fixed
        self.fixed = fixed
        self.free = free = np.flatnonzero(~fixed).astype(np.int64)
        pos = np.full(N, -1, np.int64)
        pos[free] = np.arange(len(free))
        self.pos = pos
        self.P_in = P_in = np.ascontiguousarray(mesh.points, dtype=np.float64)
        el_all, nv_all = _elements(mesh)
        self.el_all = el_all
        act_all = ((el_all >= 0) & ~fixed[np.maximum(el_all, 0)]).any(axis=1)
        self.act_all = act_all
        hi0, lo0 = _elem_angles(mesh)
        T = mesh.n_tris
        self.ghi = np.r_[np.maximum(hi0[:T], o["guard_t"]), np.maximum(hi0[T:], o["guard_q"])]
        self.glo = np.r_[np.minimum(lo0[:T], o["lo_t"]), np.minimum(lo0[T:], o["lo_q"])]
        # the energy runs over a compact subset of the elements: the active ones (a free corner) and their edge
        # neighbours (the ratio terms need their mean edge); a large mesh with a small active set stays cheap
        self.pr_all = pr_all = ratio_pairs(mesh)
        pr = pr_all[act_all[pr_all[:, 0]] | act_all[pr_all[:, 1]]]
        sub = act_all.copy()
        sub[pr.ravel()] = True
        eid = np.flatnonzero(sub)
        remap = np.full(len(el_all), -1, np.int64)
        remap[eid] = np.arange(len(eid))
        self.el, self.nv = el, nv = np.ascontiguousarray(el_all[eid]), np.ascontiguousarray(nv_all[eid])
        self.act = act = np.ascontiguousarray(act_all[eid])
        self.nptr, self.inc_e, self.inc_k = _incidence(el, nv, N)
        self.GHI = np.ascontiguousarray(self.ghi[eid] * DEG)
        self.GLO = np.ascontiguousarray(self.glo[eid] * DEG)

        self.all_edges = all_edges = topo.edges
        movable = ~fixed[all_edges[:, 0]] | ~fixed[all_edges[:, 1]]
        self.edges = edges = np.ascontiguousarray(all_edges[movable], dtype=np.int64)
        d = P_in[edges[:, 0]] - P_in[edges[:, 1]]
        self.ll0 = np.log(np.linalg.norm(d, axis=1))
        r0 = edge_ln_sh(P_in, edges, h)
        # the gates take the p99 also over the edges of the elements within 3 rows of the boundary (_coast3, which
        # holds _coast): a smaller population, so a wider margin there (margin_coast)
        from meshom.quality.metrics import element_regions

        c3 = element_regions(mesh, topo)["_coast3"]
        ee = topo.edge_elems[movable]
        near = c3[ee[:, 0]] | ((ee[:, 1] >= 0) & c3[np.maximum(ee[:, 1], 0)])
        self.W = np.maximum(o["wall"] - np.where(near, o["margin_coast"], o["margin"]), np.abs(r0))
        self.W2 = np.maximum(o["wall"] + o["wall_over"], np.abs(r0))

        self.pa, self.pb = pa, pb = np.ascontiguousarray(remap[pr[:, 0]]), np.ascontiguousarray(remap[pr[:, 1]])
        self.LEN, self.SUM, self.LM = np.zeros((len(el), 4)), np.ones(len(el)), np.zeros(len(el))
        _elem_pass(P_in, el, nv, self.LEN, self.SUM, self.LM)
        self.RW = np.maximum(np.log(o["ratio_wall"]), np.abs(self.LM[pa] - self.LM[pb]))
        ke = np.r_[pa, pb]
        kp = np.r_[np.arange(len(pa)), np.arange(len(pa))]
        ks = np.r_[np.ones(len(pa)), -np.ones(len(pa))]
        o_ = np.argsort(ke, kind="stable")
        self.epair, self.esgn = np.ascontiguousarray(kp[o_]), np.ascontiguousarray(ks[o_])
        self.eptr = np.searchsorted(ke[o_], np.arange(len(el) + 1)).astype(np.int64)

        kk = np.r_[pos[edges[:, 0]], pos[edges[:, 1]]]
        ee = np.r_[np.arange(len(edges)), np.arange(len(edges))]
        ok_ = kk >= 0
        o_ = np.argsort(kk[ok_], kind="stable")
        self.inc = np.ascontiguousarray(ee[ok_][o_])
        self.ptr = np.searchsorted(kk[ok_][o_], np.arange(len(free) + 1)).astype(np.int64)

        self.cn = cn = np.flatnonzero(slide).astype(np.int64)
        ball = all_edges[topo.edge_elems[:, 1] < 0]
        be = ball[slide[ball[:, 0]] | slide[ball[:, 1]]]
        self.X_cin = np.ascontiguousarray(P_in[cn])
        self.cpos = pos[cn]
        self.ref = ref
        if ref is None:                                     # the reference coast = the input's boundary
            self.SA, self.SB = np.ascontiguousarray(P_in[be[:, 0]]), np.ascontiguousarray(P_in[be[:, 1]])
            self.dmax = np.full(len(cn), float(o["coast_dist"]))
        else:                                               # B146: the case's reference coast
            self._ref_setup(np.asarray(ref, dtype=np.float64).reshape(-1, 2, 3), ball, h, P_in, pos, o)

        self.prm = np.array([o["w_shape"], o["w_shape_t"], o["tail_q"] * DEG, o["w_tail"], o["tail_q2"] * DEG,
                             o["w_tail2"], o["tail_t"] * DEG, o["tail_t_lo"] * DEG, o["w_tail_t"], o["w_guard"],
                             o["tail_q3"] * DEG, o["w_tail3"], o["near"] * DEG])
        self.FA, self.CA, self.BAD = np.zeros(len(el)), np.zeros((len(el), 4)), np.zeros(len(el), np.int8)
        self.R, self.FE, self.CE = np.zeros(len(edges)), np.zeros(len(edges)), np.zeros(len(edges))
        self.FP, self.CP, self.CEL = np.zeros(len(pa)), np.zeros(len(pa)), np.zeros(len(el))
        self.FC = np.zeros(len(cn))
        self.GX = np.zeros_like(P_in)
        self.Gv = np.zeros((len(free), 2))
        self.X = P_in.copy()
        self.rs = np.log(o["ratio_soft"])

    def _ref_setup(self, ref: np.ndarray, ball: np.ndarray, h, P_in: np.ndarray, pos: np.ndarray, o: dict) -> None:
        """B146: slide nodes measured against the reference segments ref (S,2,3); the reference vertices within
        ref_reach h of a slide node get the backward wall against the current coast edges (every boundary edge)."""
        from meshom.meshfix.cases import segment_distance_km

        cn = self.cn
        self.SA, self.SB = np.ascontiguousarray(ref[:, 0]), np.ascontiguousarray(ref[:, 1])
        d0 = segment_distance_km(P_in[cn], ref) * 1e3 / h(P_in[cn]) if len(cn) and len(ref) else np.zeros(len(cn))
        self.dmax = np.ascontiguousarray(np.maximum(float(o["coast_dist"]), d0))
        self.BE = np.ascontiguousarray(ball, dtype=np.int64)
        self.isfree = pos >= 0
        V = np.unique(ref.reshape(-1, 3), axis=0)
        if len(cn) and len(V) and len(ball):
            hv = h(V)
            dc, _ = cKDTree(P_in[cn]).query(V)
            V, hv = V[dc * R_EARTH_M <= o["ref_reach"] * hv], hv[dc * R_EARTH_M <= o["ref_reach"] * hv]
        else:
            V, hv = np.zeros((0, 3)), np.zeros(0)
        self.V = np.ascontiguousarray(V)
        self.SV = np.ascontiguousarray(hv / R_EARTH_M)
        dv0 = self.ref_shift(P_in)
        self.ref_seg, self.d0, self.dv0 = ref, d0, dv0
        self.LIM = np.ascontiguousarray(np.maximum(o["coast_move"] - o["ref_margin"], dv0))
        self.LIM2 = np.ascontiguousarray(np.maximum(o["coast_move"] - o["ref_margin2"], dv0))
        self.FV = 0.0

    def ref_shift(self, X: np.ndarray) -> np.ndarray:
        """(len(V),) distance of each wall vertex to the coast edges of X (exact over the 16 nearest by midpoint),
        in units of h(V)."""
        if self.ref is None or len(self.V) == 0:
            return np.zeros(0)
        seg = np.stack([X[self.BE[:, 0]], X[self.BE[:, 1]]], axis=1)
        from meshom.meshfix.cases import segment_distance_km

        return segment_distance_km(self.V, seg) * 1e3 / (self.SV * R_EARTH_M)

    def ref_violations(self, X: np.ndarray) -> np.ndarray:
        """B146 hard check (the scorer's coast shift, both directions): the slide nodes to put back, i.e. those
        farther than max(coast_move, their input distance) h from the reference coast (forward) and those within
        REF_REVERT_H h of a reference vertex farther than max(coast_move, its input distance) h from the coast edges
        of X (backward)."""
        from meshom.meshfix.cases import segment_distance_km

        o, cn = self.o, self.cn
        if self.ref is None or len(cn) == 0:
            return np.zeros(0, np.int64)
        Xc = X[cn]
        fwd = segment_distance_km(Xc, self.ref_seg) * 1e3 / self.h(Xc)
        out = set(cn[fwd > np.maximum(o["coast_move"], self.d0)].tolist())
        if len(self.V):
            bad = np.flatnonzero(self.ref_shift(X) > np.maximum(o["coast_move"], self.dv0))
            if len(bad):
                tree = cKDTree(Xc)
                for k in bad:
                    out.update(cn[tree.query_ball_point(self.V[k], REF_REVERT_H * self.SV[k])].tolist())
        return np.array(sorted(out), dtype=np.int64)

    def round(self, X0: np.ndarray):
        """Freeze the chart, h at the edge midpoints and the coast candidates at X0; returns fg."""
        o, h, free, edges = self.o, self.h, self.free, self.edges
        self.X0 = X0 = np.ascontiguousarray(X0)
        E1, E2 = tangent_basis(X0[free])
        self.E1, self.E2 = np.ascontiguousarray(E1), np.ascontiguousarray(E2)
        self.S = np.ascontiguousarray(h(X0[free]) / R_EARTH_M)
        self.Sc = np.ascontiguousarray(self.S[self.cpos]) if len(self.cn) else np.zeros(0)
        pm = X0[edges[:, 0]] + X0[edges[:, 1]]
        pm /= np.linalg.norm(pm, axis=1, keepdims=True)
        self.lh = np.ascontiguousarray(np.log(h(pm) / R_EARTH_M))
        self.cptr, self.cseg = _coast_candidates(X0, self.cn, self.SA, self.SB, int(o["coast_k"]))
        if self.ref is not None:                            # B146: the wall vertices' candidate coast edges
            self.vptr, self.vedge = _coast_candidates(self.V, np.arange(len(self.V)), X0[self.BE[:, 0]],
                                                      X0[self.BE[:, 1]], int(o["coast_k"]))
        self.X[:] = X0
        self.Yn = np.ones(len(free))
        self.frozen = np.zeros(len(free), bool)
        # hard limits of the line search: the guard plus guard_hard, widened to the start's own angles (a start
        # that sits on a limit to rounding never blocks the round)
        hm = o["guard_hard"] * DEG
        self.HHI, self.HLO = np.full(len(self.el), 1e9), np.full(len(self.el), -1e9)
        self.HHI[:] = self.GHI + hm
        self.HLO[:] = self.GLO - hm
        self.HHI, self.HLO = np.ascontiguousarray(self.HHI), np.ascontiguousarray(self.HLO)
        _elem_pass(X0, self.el, self.nv, self.LEN, self.SUM, self.LM)
        _corner_range(X0, self.el, self.nv, self.act, self.LEN, self.HHI, self.HLO)
        return self.fg

    def fg(self, v, parts: bool = False):
        o = self.o
        X, free, el, nv = self.X, self.free, self.el, self.nv
        _positions(v.reshape(-1, 2), free, self.X0, self.E1, self.E2, self.S, X, self.Yn)
        _elem_pass(X, el, nv, self.LEN, self.SUM, self.LM)
        _corners(X, el, nv, self.act, self.LEN, self.prm, self.GHI, self.GLO, self.HHI, self.HLO, self.FA, self.CA,
                 self.BAD)
        if not parts and (self.BAD == 1).any():
            return np.inf, self.Gv.ravel().copy(), False
        _edges(X, self.edges, self.lh, self.ll0, self.W, self.W2, o["w_size"], o["w_wall"], o["w_wall_max"], self.R,
               self.FE, self.CE)
        _pairs(self.LM, self.pa, self.pb, self.RW, self.rs, o["w_ratio"], o["w_wall"], self.FP, self.CP)
        _elem_coef(self.CP, self.eptr, self.epair, self.esgn, 1.0, self.SUM, self.CEL)
        self.GX[free] = 0.0
        fa, fe, fp, fc = _bsum(self.FA), _bsum(self.FE), _bsum(self.FP), 0.0
        if len(self.cn):
            _coast(X, self.cn, self.X_cin, self.Sc, self.SA, self.SB, self.cptr, self.cseg, self.dmax,
                   o["coast_move"], o["w_coast"], o["w_wall"], self.FC, self.GX)
            fc = _bsum(self.FC)
        fv = 0.0
        if self.ref is not None and len(self.V):
            fv = _refwall(X, self.V, self.SV, self.LIM, self.LIM2, self.vptr, self.vedge, self.BE, self.isfree,
                          o["w_wall"], o["w_wall_max"], self.GX)
        if parts:
            return {"corners": fa, "edges": fe, "pairs": fp, "coast": fc, "ref": fv,
                    "bad": int((self.BAD == 1).sum())}
        _node_gather(X, el, nv, free, self.nptr, self.inc_e, self.inc_k, self.LEN, self.CEL, self.CA, self.GX)
        _gather(X, self.Yn, free, self.E1, self.E2, self.S, self.edges, self.CE, self.ptr, self.inc, 1.0, self.GX,
                self.Gv)
        self.Gv[self.frozen] = 0.0
        return fa + fe + fp + fc + fv, self.Gv.ravel().copy(), True

    def freeze(self, v) -> int:
        """A blocked line search: freeze (zero gradient, so no move for the rest of the round) the free nodes of the
        elements with a corner near a hard limit at v; returns how many nodes were newly frozen."""
        self.fg(v)
        near = self.el[self.BAD > 0]
        k = self.pos[near[near >= 0]]
        k = k[k >= 0]
        new = int((~self.frozen[k]).sum())
        self.frozen[k] = True
        return new


def optimize_geometry(mesh: Mesh, h, fixed=None, slide=None, opts: dict | None = None,
                      log=print, ref=None) -> tuple[Mesh, dict]:
    """Copy of mesh with its free nodes moved to minimise the energy of the module docstring; (mesh, info).

    fixed: bool (N,) mask of nodes that never move (e.g. a cut-out's artificial boundary, or the nodes another method
    wants kept). slide: bool (N,) mask of boundary nodes that may slide along the coast (default: every boundary node
    not fixed); boundary nodes outside `slide` are fixed. The reference coast is the input's boundary edges with a
    slide end, or (B146) `ref`, the case's reference coast segments (S,2,3) such as Case.coast_seg: then the slide
    nodes stay near it and its vertices near the new coast (module docstring) whatever moved the coast before A.
    opts: overrides of DEFAULTS.
    """
    t_start = time.perf_counter()
    pb = Problem(mesh, h, fixed, slide, opts, ref)
    o, free, el = pb.o, pb.free, pb.el_all
    tol = o["wall"]
    P = pb.P_in.copy()
    info = {"opts": o, "free_nodes": int(len(free)), "slide_nodes": int(len(pb.cn)), "ref": ref is not None,
            "ref_vertices": int(len(pb.V)) if ref is not None else 0,
            "active_elems": int(pb.act_all.sum()), "edges": int(len(pb.edges)), "pairs": int(len(pb.pa)),
            "before": stats(mesh, h, pb.all_edges, pb.pr_all, tol), "rounds": []}
    log(f"geom: free {len(free)} (slide {len(pb.cn)}), active elements {int(pb.act_all.sum())}, "
        f"before {info['before']}")
    if len(free) == 0:
        info.update(after=info["before"], total_s=time.perf_counter() - t_start)
        return Mesh(P, mesh.tris, mesh.quads, mesh.node_flag, mesh.tri_zone), info
    e_prev = None
    for rnd in range(int(o["rounds"])):
        t0 = time.perf_counter()
        X0 = P.copy()
        fg = pb.round(X0)
        f0, _, ok0 = fg(np.zeros(2 * len(free)))
        v, f1, nit, reason, nrs = lbfgs(fg, 2 * len(free), int(o["max_iter"]), int(o["m"]), o["step0"],
                                        o["max_step"], int(o["window"]), o["wtol"], int(o["restarts"]), pb.freeze)
        t_opt = time.perf_counter() - t0
        _positions(v.reshape(-1, 2), free, X0, pb.E1, pb.E2, pb.S, P, pb.Yn)
        # hard check: revert the nodes of every bad element (and, with a reference coast, the coast nodes that break
        # the coast-shift rule, B146) to the round's start until none is left
        reverted, ref_rev = set(), set()
        for _ in range(100):
            cur = Mesh(P, mesh.tris, mesh.quads, mesh.node_flag)
            bad = _bad_elements(cur, pb.act_all, pb.ghi, pb.glo, o["guard_tol"])
            rv = pb.ref_violations(P) if ref is not None else np.zeros(0, np.int64)
            if not bad.any() and len(rv) == 0:
                break
            nodes = np.unique(np.r_[el[bad][el[bad] >= 0], rv])
            nodes = nodes[~pb.fixed[nodes]]
            moved = nodes[np.any(P[nodes] != X0[nodes], axis=1)]
            if len(moved) == 0:
                break
            P[moved] = X0[moved]
            reverted.update(moved.tolist())
            ref_rev.update(np.intersect1d(moved, rv).tolist())
        cur = Mesh(P.copy(), mesh.tris, mesh.quads, mesh.node_flag, mesh.tri_zone)
        rec = {"round": rnd, "iters": int(nit), "stop": reason, "restarts": nrs, "frozen": int(pb.frozen.sum()),
               "E0": float(f0), "E": float(f1), "start_ok": bool(ok0),
               "reverted": len(reverted), "ref_reverted": len(ref_rev), "bad_left": int(bad.sum()), "opt_s": t_opt,
               "s": time.perf_counter() - t0,
               **stats(cur, h, pb.all_edges, pb.pr_all, tol)}
        info["rounds"].append(rec)
        log(f"geom round {rnd}: {rec}")
        if e_prev is not None and (e_prev - f1) < o["round_tol"] * abs(e_prev) and not reverted:
            break
        e_prev = f1
    info.update(after=info["rounds"][-1], total_s=time.perf_counter() - t_start)
    if ref is not None and len(pb.V):
        info.update(ref_shift_in=float(pb.ref_shift(pb.P_in).max()), ref_shift_out=float(pb.ref_shift(P).max()))
    return Mesh(P, mesh.tris, mesh.quads, mesh.node_flag, mesh.tri_zone), info
