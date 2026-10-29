"""B's local relaxation in one Numba call (decision B151): size_fit.smooth_fit with rounds=1 on a small patch.

Work.fit (B128) ran `size_fit.smooth_fit(sub, h, tol, fixed, max_iter=100, rounds=1, w_angle=2, w_wall=200, w_tol=50,
ftol=1e-9)` on the 2-ring patch of every trial: per L-BFGS iteration ~10 parallel Numba kernel launches, numpy
temporaries and a Python line search, plus two `stats` passes (topology, percentiles) whose output B never read. Here
the same round runs as one serial Numba function: the set-up of `size_fit.smooth_problem` (same arrays in the same
order), the energy and gradient of its `fg` (each kernel's loop in the same order: `_bsum` and `_dot` are plain
sequential sums below BLOCK = 4096 terms, the per-node gathers keep their CSR order) and `size_fit._lbfgs` (the same
two-loop recursion, Armijo backtracking with the inversion guard, the same stopping rules). The result equals
smooth_fit's to the last bit on the bench patches (test `test_nfit_equals_smooth_fit`).
"""
import numpy as np
from numba import njit

from meshom.geo import R_EARTH_M, tangent_basis
from meshom.size_fit import (Q_EQ, QUAD_HI, QUAD_HI2, QUAD_LO, QUAD_LO2, TRI_HI, TRI_HI2, TRI_LO, TRI_LO2, _dtheta,
                             _theta)


@njit(cache=True)
def _fg(V, free, X0, E1, E2, S, X, Yn, el, nv, act, pa, pb, p, eptr, epair, esgn, edges, lh, bt, ref, ptr, inc,
        nptr, inc_e, inc_k, inv_ne, inv_np, inv_nc, w_angle, w_wall, w_hard, tolm, w_tol, huber, w_h, mu, tau,
        GX, Gv, R, FE, CE, Q, FB, LEN, SUM, LM, FP, CP, CEL, FA, CA, out4):
    """smooth_problem's fg at the offsets V (F,2): returns (f, ok); the gradient into Gv (left as it was when not
    ok, as fg returns the stale Gv)."""
    nf = len(free)
    # _positions
    for k in range(nf):
        i = free[k]
        y0 = X0[i, 0] + S[k] * (V[k, 0] * E1[k, 0] + V[k, 1] * E2[k, 0])
        y1 = X0[i, 1] + S[k] * (V[k, 0] * E1[k, 1] + V[k, 1] * E2[k, 1])
        y2 = X0[i, 2] + S[k] * (V[k, 0] * E1[k, 2] + V[k, 1] * E2[k, 2])
        r = np.sqrt(y0 * y0 + y1 * y1 + y2 * y2)
        X[i, 0], X[i, 1], X[i, 2] = y0 / r, y1 / r, y2 / r
        Yn[k] = r
    nb = len(bt)
    if nb:
        # _tri_pass
        qmin = np.inf
        for t in range(nb):
            a, b, c = bt[t, 0], bt[t, 1], bt[t, 2]
            bc0 = X[b, 1] * X[c, 2] - X[b, 2] * X[c, 1]
            bc1 = X[b, 2] * X[c, 0] - X[b, 0] * X[c, 2]
            bc2 = X[b, 0] * X[c, 1] - X[b, 1] * X[c, 0]
            q = (X[a, 0] * bc0 + X[a, 1] * bc1 + X[a, 2] * bc2) / ref[t]
            Q[t] = q
            FB[t] = (tau - q) * (tau - q) if q < tau else 0.0
            if q < qmin:
                qmin = q
        if qmin <= 0.0:
            return np.inf, False
    # _elem_pass
    ne_ = len(el)
    for e in range(ne_):
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
    # _pair_pass
    for k in range(len(pa)):
        r = LM[pa[k]] - LM[pb[k]]
        rp1 = r ** (p - 1)
        FP[k] = rp1 * r
        CP[k] = p * rp1
    # _elem_coef
    for e in range(ne_):
        s = 0.0
        for j in range(eptr[e], eptr[e + 1]):
            s += esgn[j] * CP[epair[j]]
        CEL[e] = inv_np * s / SUM[e]
    # _corner_pass
    for e in range(ne_):
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
            f += w_angle * (t - t0) * (t - t0)
            g = 2.0 * w_angle * (t - t0)
            if t > hi:
                f += w_wall * (t - hi) * (t - hi)
                g += 2.0 * w_wall * (t - hi)
            elif t < lo:
                f += w_wall * (lo - t) * (lo - t)
                g -= 2.0 * w_wall * (lo - t)
            if t > hi2:
                f += w_hard * (t - hi2) * (t - hi2)
                g += 2.0 * w_hard * (t - hi2)
            elif t < lo2:
                f += w_hard * (lo2 - t) * (lo2 - t)
                g -= 2.0 * w_hard * (lo2 - t)
            CA[e, k] = inv_nc * g
        FA[e] = f
    # _tol_pass
    for e in range(len(edges)):
        i, j = edges[e, 0], edges[e, 1]
        d0, d1, d2 = X[i, 0] - X[j, 0], X[i, 1] - X[j, 1], X[i, 2] - X[j, 2]
        c2 = d0 * d0 + d1 * d1 + d2 * d2
        r = 0.5 * np.log(c2) - lh[e]
        R[e] = r
        x = abs(r) - tolm
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
    # f = _bsum(FP) * inv_np + _bsum(FE) * inv_ne + inv_nc * _bsum(FA)  (sequential sums: < BLOCK terms)
    s1 = 0.0
    for k in range(len(FP)):
        s1 += FP[k]
    s2 = 0.0
    for k in range(len(FE)):
        s2 += FE[k]
    s3 = 0.0
    for k in range(len(FA)):
        s3 += FA[k]
    fval = s1 * inv_np + s2 * inv_ne + inv_nc * s3
    for k in range(nf):
        i = free[k]
        GX[i, 0] = 0.0
        GX[i, 1] = 0.0
        GX[i, 2] = 0.0
    if nb:
        s4 = 0.0
        for k in range(nb):
            s4 += FB[k]
        fval += mu * s4 * inv_ne
        # _barrier_grad
        g_scale = mu * inv_ne
        for t in range(nb):
            q = Q[t]
            if q >= tau:
                continue
            a, b, c = bt[t, 0], bt[t, 1], bt[t, 2]
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
    # _node_gather
    for f_ in range(nf):
        i = free[f_]
        out4[0] = 0.0
        out4[1] = 0.0
        out4[2] = 0.0
        out4[3] = 0.0
        for j in range(nptr[i], nptr[i + 1]):
            e, k = inc_e[j], inc_k[j]
            m = nv[e]
            kn, kp = (k + 1) % m, (k + m - 1) % m
            nx, pv = el[e, kn], el[e, kp]
            c = CEL[e]
            l1, l2 = LEN[e, k], LEN[e, kp]
            for d in range(3):
                out4[d] += c * ((X[i, d] - X[nx, d]) / l1 + (X[i, d] - X[pv, d]) / l2)
            out4[3] = CA[e, k]
            if out4[3] != 0.0:
                _dtheta(X, i, nx, pv, l1, l2, 0, out4)
            out4[3] = CA[e, kn]
            if out4[3] != 0.0:
                _dtheta(X, nx, el[e, (k + 2) % m], i, LEN[e, kn], l1, 2, out4)
            out4[3] = CA[e, kp]
            if out4[3] != 0.0:
                _dtheta(X, pv, i, el[e, (k + m - 2) % m], l2, LEN[e, (k + m - 2) % m], 1, out4)
        GX[i, 0] += out4[0]
        GX[i, 1] += out4[1]
        GX[i, 2] += out4[2]
    # _gather
    for k in range(nf):
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
    return fval, True


@njit(cache=True)
def _dotv(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


@njit(cache=True)
def _solve(free, X0, E1, E2, S, X, Yn, el, nv, act, pa, pb, p, eptr, epair, esgn, edges, lh, bt, ref, ptr, inc,
           nptr, inc_e, inc_k, inv_ne, inv_np, inv_nc, w_angle, w_wall, w_hard, tolm, w_tol, huber, w_h, mu, tau,
           max_iter, m, step0, max_step, ftol, P):
    """size_fit._lbfgs on smooth_problem's fg, then the positions of the result into P. Returns (f0, f, iterations,
    stop: 0 max_iter, 1 gradient, 2 line_search, 3 ftol)."""
    nf = len(free)
    n = 2 * nf
    ne_ = len(el)
    GX = np.zeros_like(X0)
    Gv = np.zeros((nf, 2))
    R = np.zeros(len(edges))
    FE = np.zeros(len(edges))
    CE = np.zeros(len(edges))
    Q = np.zeros(len(bt))
    FB = np.zeros(len(bt))
    LEN = np.zeros((ne_, 4))
    SUM = np.ones(ne_)
    LM = np.zeros(ne_)
    FP = np.zeros(len(pa))
    CP = np.zeros(len(pa))
    CEL = np.zeros(ne_)
    FA = np.zeros(ne_)
    CA = np.zeros((ne_, 4))
    out4 = np.zeros(4)
    x = np.zeros(n)
    V = np.zeros((nf, 2))
    g = np.zeros(n)
    gn = np.zeros(n)

    def fg(xv, gout):
        for k in range(nf):
            V[k, 0] = xv[2 * k]
            V[k, 1] = xv[2 * k + 1]
        fv, ok = _fg(V, free, X0, E1, E2, S, X, Yn, el, nv, act, pa, pb, p, eptr, epair, esgn, edges, lh, bt, ref,
                     ptr, inc, nptr, inc_e, inc_k, inv_ne, inv_np, inv_nc, w_angle, w_wall, w_hard, tolm, w_tol,
                     huber, w_h, mu, tau, GX, Gv, R, FE, CE, Q, FB, LEN, SUM, LM, FP, CP, CEL, FA, CA, out4)
        for k in range(nf):
            gout[2 * k] = Gv[k, 0]
            gout[2 * k + 1] = Gv[k, 1]
        return fv, ok

    f, ok = fg(x, g)
    f0 = f
    Sm = np.zeros((m, n))
    Ym = np.zeros((m, n))
    RHO = np.zeros(m)
    al = np.zeros(m)
    head, cnt = 0, 0
    d = np.empty(n)
    sv = np.empty(n)
    yv = np.empty(n)
    xn = np.empty(n)
    small = 0
    it = 0
    reason = 0
    gd = 0.0
    fn = f
    for it in range(1, max_iter + 1):
        gmax = 0.0
        for i in range(n):
            a = abs(g[i])
            if a > gmax:
                gmax = a
        if gmax == 0.0:
            reason = 1
            break
        if cnt:
            # _two_loop
            for i in range(n):
                d[i] = g[i]
            for i in range(m):
                al[i] = 0.0
            for k in range(cnt):
                j = (head - 1 - k) % m
                a = RHO[j] * _dotv(Sm[j], d)
                al[j] = a
                for i in range(n):
                    d[i] -= a * Ym[j, i]
            j = (head - 1) % m
            gam = _dotv(Sm[j], Ym[j]) / _dotv(Ym[j], Ym[j])
            for i in range(n):
                d[i] *= gam
            for k in range(cnt - 1, -1, -1):
                j = (head - 1 - k) % m
                b = RHO[j] * _dotv(Ym[j], d)
                for i in range(n):
                    d[i] += (al[j] - b) * Sm[j, i]
            for i in range(n):
                d[i] = -d[i]
            gd = _dotv(g, d)
        if not cnt or not gd < 0.0:
            cnt = 0
            c = -step0 / gmax
            for i in range(n):
                d[i] = g[i] * c
            gd = _dotv(g, d)
        dm = 0.0
        for i in range(n):
            a = abs(d[i])
            if a > dm:
                dm = a
        if dm > max_step:
            c = max_step / dm
            for i in range(n):
                d[i] *= c
            gd *= max_step / dm
        t = 1.0
        found = False
        for _ in range(40):
            for i in range(n):
                xn[i] = x[i] + t * d[i]
            fn, ok = fg(xn, gn)
            if ok and fn <= f + 1e-4 * t * gd:
                found = True
                break
            t *= 0.5
        if not found:
            reason = 2
            break
        for i in range(n):
            sv[i] = xn[i] - x[i]
            yv[i] = gn[i] - g[i]
        sy = _dotv(sv, yv)
        if sy > 1e-12 * np.sqrt(_dotv(sv, sv) * _dotv(yv, yv)):
            for i in range(n):
                Sm[head, i] = sv[i]
                Ym[head, i] = yv[i]
            RHO[head] = 1.0 / sy
            head = (head + 1) % m
            cnt = min(cnt + 1, m)
        rel = (f - fn) / max(abs(f), 1e-300)
        for i in range(n):
            x[i] = xn[i]
            g[i] = gn[i]
        f = fn
        small = small + 1 if rel < ftol else 0
        if small >= 5:
            reason = 3
            break
    # _positions(v, free, X0, E1, E2, S, P, Yn)
    for k in range(nf):
        i = free[k]
        y0 = X0[i, 0] + S[k] * (x[2 * k] * E1[k, 0] + x[2 * k + 1] * E2[k, 0])
        y1 = X0[i, 1] + S[k] * (x[2 * k] * E1[k, 1] + x[2 * k + 1] * E2[k, 1])
        y2 = X0[i, 2] + S[k] * (x[2 * k] * E1[k, 2] + x[2 * k + 1] * E2[k, 2])
        r = np.sqrt(y0 * y0 + y1 * y1 + y2 * y2)
        P[i, 0], P[i, 1], P[i, 2] = y0 / r, y1 / r, y2 / r
    return f0, f, it, reason


def _incidence(el, nv, n):
    """meshom.smooth._incidence (CSR node -> (element, corner) in element order), in numpy."""
    ok = el >= 0
    node = el[ok]
    e = np.repeat(np.arange(len(el)), 4).reshape(-1, 4)[ok]
    k = np.tile(np.arange(4), len(el)).reshape(-1, 4)[ok]
    o = np.argsort(node, kind="stable")
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(node, minlength=n), out=ptr[1:])
    return ptr, e[o].astype(np.int64), k[o].astype(np.int64)


def fit_sub(P, tris, quads, fixed, h, tol, max_iter=500, w_angle=0.1, w_wall=100.0, w_tol=1e3, w_h=0.0,
            margin=0.01, huber=0.05, w_hard=1e4, mu=1.0, tau=0.5, m=10, step0=0.05, max_step=0.25, ftol=1e-10,
            p=2):
    """smooth_fit(Mesh(P, tris, quads), h, tol, fixed, rounds=1, flips=False, ...) on a small local mesh: the moved
    copy of P (B151). fixed: bool mask; the patch's own boundary nodes are fixed as smooth_fit fixes them."""
    n = len(P)
    nt = len(tris)
    el = np.full((nt + len(quads), 4), -1, dtype=np.int64)
    el[:nt, :3] = tris
    el[nt:] = quads
    nv = np.r_[np.full(nt, 3), np.full(len(quads), 4)].astype(np.int64)
    # build_topology: edges (sorted keys) and their elements
    ok = el >= 0
    a = el[ok]
    b = np.where(nv[:, None] == 4, np.roll(el, -1, axis=1), np.c_[el[:, 1:3], el[:, :1], el[:, 3:]])[ok]
    eid = np.repeat(np.arange(len(el), dtype=np.int64), 4).reshape(-1, 4)[ok]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo * n + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    first = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    count = np.diff(np.r_[first, len(ks)])
    topo_edges = np.column_stack([ks[first] // n, ks[first] % n])
    second = np.minimum(first + 1, len(ks) - 1)
    ee2 = np.column_stack([eid[order[first]], np.where(count >= 2, eid[order[second]], -1)])
    fixed_m = np.asarray(fixed, dtype=bool).copy()
    fixed_m[topo_edges[ee2[:, 1] < 0].ravel()] = True
    free = np.flatnonzero(~fixed_m).astype(np.int64)
    out = np.ascontiguousarray(P, dtype=np.float64).copy()
    if len(free) == 0:
        return out, (0.0, 0.0, 0, -1)
    nptr, inc_e, inc_k = _incidence(el, nv, n)
    act = np.zeros(len(el), dtype=bool)
    act[np.repeat(np.arange(len(el)), 4)[(el.ravel() >= 0) & ~fixed_m[np.maximum(el.ravel(), 0)]]] = True
    pr = ee2[ee2[:, 1] >= 0]
    pr = np.ascontiguousarray(pr[act[pr[:, 0]] | act[pr[:, 1]]])
    pa, pb = np.ascontiguousarray(pr[:, 0]), np.ascontiguousarray(pr[:, 1])
    ke = np.r_[pa, pb]
    kp = np.r_[np.arange(len(pa)), np.arange(len(pa))]
    ksg = np.r_[np.ones(len(pa)), -np.ones(len(pa))]
    o_ = np.argsort(ke, kind="stable")
    epair, esgn = np.ascontiguousarray(kp[o_]), np.ascontiguousarray(ksg[o_])
    eptr = np.searchsorted(ke[o_], np.arange(len(el) + 1)).astype(np.int64)
    movable = ~fixed_m[topo_edges[:, 0]] | ~fixed_m[topo_edges[:, 1]]
    edges = np.ascontiguousarray(topo_edges[movable])
    bt_parts = [tris] + [quads[:, [k, (k + 1) % 4, (k + 3) % 4]] for k in range(4)] if len(quads) else [tris]
    bt = np.ascontiguousarray(np.vstack(bt_parts), dtype=np.int64)
    bt = np.ascontiguousarray(bt[~fixed_m[bt].all(axis=1)])
    X0 = out.copy()
    pm = X0[edges[:, 0]] + X0[edges[:, 1]]
    pm /= np.linalg.norm(pm, axis=1, keepdims=True)
    cen = X0[bt].sum(axis=1)
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    hv = h(np.concatenate([pm, cen, X0[free]]))                    # one call: h is evaluated point by point
    lh = np.log(hv[: len(pm)] / R_EARTH_M)
    ref = Q_EQ * (hv[len(pm): len(pm) + len(cen)] / R_EARTH_M) ** 2
    S = hv[len(pm) + len(cen):] / R_EARTH_M
    E1, E2 = tangent_basis(X0[free])
    E1, E2 = np.ascontiguousarray(E1), np.ascontiguousarray(E2)
    inv_ne = 1.0 / max(len(edges), 1)
    inv_np = 1.0 / max(len(pa), 1)
    inv_nc = 1.0 / max(int(nv[act].sum()), 1)
    pos = np.full(n, -1, dtype=np.int64)
    pos[free] = np.arange(len(free))
    kk = np.r_[pos[edges[:, 0]], pos[edges[:, 1]]]
    ee = np.r_[np.arange(len(edges)), np.arange(len(edges))]
    ok_ = kk >= 0
    kk, ee = kk[ok_], ee[ok_]
    o_ = np.argsort(kk, kind="stable")
    inc = np.ascontiguousarray(ee[o_]).astype(np.int64)
    ptr = np.searchsorted(kk[o_], np.arange(len(free) + 1)).astype(np.int64)
    X = X0.copy()
    Yn = np.ones(len(free))
    info = _solve(free, X0, E1, E2, S, X, Yn, el, nv, act, pa, pb, int(p), eptr, epair, esgn, edges, lh, bt, ref,
                  ptr, inc, nptr, inc_e, inc_k, inv_ne, inv_np, inv_nc, float(w_angle), float(w_wall),
                  float(w_hard), float(tol - margin), float(w_tol), float(huber), float(w_h), float(mu), float(tau),
                  int(max_iter), int(m), float(step0), float(max_step), float(ftol), out)
    return out, info
