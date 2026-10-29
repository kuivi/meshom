"""Numba relaxation of e_coast (decisions B144, B147): coast/relax.py's JAX energy, hand-differentiated.

The energy of coast/relax.py (D's B136 energy plus the stiff size wall W_CAP at CAP, the element-ratio wall W_RWALL at
RWALL, the size wall with weight w_wall, and the sliding coast nodes: W_COAST sum (d / s)^2, d = chord distance to
the nearest of the node's allowed reference segments, s its mean edge) on the unpadded zone. Gradient by hand: D's
corner terms (nrelax._corner_acc), the edge and ratio terms with their walls, the coast term (gradient 2 (x - q) /
s^2 at the nearest point q, the clamp of q to the segment and the min over segments are piecewise constant in the
envelope), and the retraction. The minimiser is D's Numba L-BFGS (nrelax.lbfgs, B139: scipy's L-BFGS-B on an
unbounded problem) with the coast energy in place of D's: the line search and the iteration are copied verbatim
below (Numba binds the energy at compile time; D's kernels stay as they are). After the minimisation the sliding
nodes are projected onto their nearest allowed segment (coast/relax.project).
"""
import numpy as np
from numba import njit

from meshom.meshfix.cavity.nrelax import (EPSMCH, FACTR, MAXLS, PGTOL, W_RATIO, W_SIZE, _basis, _corner_acc,
                                          _dcstep, _positions, edges_of, elem_pairs)

# coast/relax.py's constants (B144), the JAX reference imports nothing from here: they are checked equal in the tests
W_CAP = 1e4
CAP = 0.27
W_RWALL = 1e3
RWALL = float(np.log(1.12))
W_COAST = 1e3
WALL_MARGIN = 0.02


@njit(cache=True, fastmath=False)
def energy_grad(v, P0, fidx, sc, E1, E2, quads, tris, Ed, lh, bias, wall, w_wall, pa, pb, ci, cA, cB, cs, X, G, Ynorm,
                M, GM, g):
    """The coast energy at v and its gradient into g (len 2 * len(fidx)); X, G, Ynorm, M, GM are work arrays; ci
    (C,) local ids of the sliding nodes, cA, cB (C, K, 3) their allowed segments, cs (C,) their scales."""
    _positions(v, P0, fidx, sc, E1, E2, X, Ynorm)
    G[:] = 0.0
    e = 0.0
    for k in range(len(quads)):
        for i in range(4):
            e += _corner_acc(X, G, quads[k, i], quads[k, (i + 1) % 4], quads[k, (i + 3) % 4], True)
    for k in range(len(tris)):
        for i in range(3):
            e += _corner_acc(X, G, tris[k, i], tris[k, (i + 1) % 3], tris[k, (i + 2) % 3], False)
    for k in range(len(Ed)):
        a, b = Ed[k, 0], Ed[k, 1]
        d0, d1, d2 = X[a, 0] - X[b, 0], X[a, 1] - X[b, 1], X[a, 2] - X[b, 2]
        l = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2) + 1e-30
        r = np.log(l) - lh[k]
        e += W_SIZE * (r - bias) ** 2
        de = 2.0 * W_SIZE * (r - bias)
        sg = 1.0 if r > 0 else -1.0
        x = abs(r) - wall
        if x > 0.0:
            e += w_wall * x * x
            de += 2.0 * w_wall * x * sg
        x = abs(r) - CAP
        if x > 0.0:
            e += W_CAP * x * x
            de += 2.0 * W_CAP * x * sg
        f = de / (l * l)
        G[a, 0] += f * d0
        G[a, 1] += f * d1
        G[a, 2] += f * d2
        G[b, 0] -= f * d0
        G[b, 1] -= f * d1
        G[b, 2] -= f * d2
    nq = len(quads)
    if len(pa):
        for k in range(nq):
            s = 0.0
            for i in range(4):
                a, b = quads[k, i], quads[k, (i + 1) % 4]
                s += np.sqrt((X[a, 0] - X[b, 0]) ** 2 + (X[a, 1] - X[b, 1]) ** 2 + (X[a, 2] - X[b, 2]) ** 2)
            M[k] = s / 4.0 + 1e-30
        for k in range(len(tris)):
            s = 0.0
            for i in range(3):
                a, b = tris[k, i], tris[k, (i + 1) % 3]
                s += np.sqrt((X[a, 0] - X[b, 0]) ** 2 + (X[a, 1] - X[b, 1]) ** 2 + (X[a, 2] - X[b, 2]) ** 2)
            M[nq + k] = s / 3.0 + 1e-30
        GM[:] = 0.0
        for k in range(len(pa)):
            d = np.log(M[pa[k]]) - np.log(M[pb[k]])
            e += W_RATIO * d * d
            gd = 2.0 * W_RATIO * d
            x = abs(d) - RWALL
            if x > 0.0:
                e += W_RWALL * x * x
                gd += 2.0 * W_RWALL * x * (1.0 if d > 0 else -1.0)
            GM[pa[k]] += gd / M[pa[k]]
            GM[pb[k]] -= gd / M[pb[k]]
        for k in range(nq + len(tris)):
            if GM[k] == 0.0:
                continue
            if k < nq:
                nvk = 4
            else:
                nvk = 3
            for i in range(nvk):
                if k < nq:
                    a, b = quads[k, i], quads[k, (i + 1) % 4]
                else:
                    a, b = tris[k - nq, i], tris[k - nq, (i + 1) % 3]
                d0, d1, d2 = X[a, 0] - X[b, 0], X[a, 1] - X[b, 1], X[a, 2] - X[b, 2]
                l = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2) + 1e-30
                f = GM[k] / nvk / l
                G[a, 0] += f * d0
                G[a, 1] += f * d1
                G[a, 2] += f * d2
                G[b, 0] -= f * d0
                G[b, 1] -= f * d1
                G[b, 2] -= f * d2
    # sliding coast nodes: squared chord distance to the nearest allowed segment over the squared scale
    for k in range(len(ci)):
        i = ci[k]
        p0, p1, p2 = X[i, 0], X[i, 1], X[i, 2]
        best = 1e300
        q0, q1, q2 = p0, p1, p2
        for j in range(cA.shape[1]):
            a0, a1, a2 = cA[k, j, 0], cA[k, j, 1], cA[k, j, 2]
            b0, b1, b2 = cB[k, j, 0] - a0, cB[k, j, 1] - a1, cB[k, j, 2] - a2
            bb = max(b0 * b0 + b1 * b1 + b2 * b2, 1e-300)
            t = ((p0 - a0) * b0 + (p1 - a1) * b1 + (p2 - a2) * b2) / bb
            t = min(1.0, max(0.0, t))
            c0, c1, c2 = a0 + t * b0, a1 + t * b1, a2 + t * b2
            d2 = (p0 - c0) ** 2 + (p1 - c1) ** 2 + (p2 - c2) ** 2
            if d2 < best:
                best = d2
                q0, q1, q2 = c0, c1, c2
        w = W_COAST / (cs[k] * cs[k])
        e += w * best
        G[i, 0] += 2.0 * w * (p0 - q0)
        G[i, 1] += 2.0 * w * (p1 - q1)
        G[i, 2] += 2.0 * w * (p2 - q2)
    # retraction: x = y / |y|, y = p0 + s (v1 e1 + v2 e2)
    for k in range(len(fidx)):
        i = fidx[k]
        gx = G[i, 0] * X[i, 0] + G[i, 1] * X[i, 1] + G[i, 2] * X[i, 2]
        t0 = (G[i, 0] - gx * X[i, 0]) / Ynorm[k]
        t1 = (G[i, 1] - gx * X[i, 1]) / Ynorm[k]
        t2 = (G[i, 2] - gx * X[i, 2]) / Ynorm[k]
        g[2 * k] = sc[k] * (t0 * E1[k, 0] + t1 * E1[k, 1] + t2 * E1[k, 2])
        g[2 * k + 1] = sc[k] * (t0 * E2[k, 0] + t1 * E2[k, 1] + t2 * E2[k, 2])
    return e


def prepare(X0: np.ndarray, free: np.ndarray, quads: np.ndarray, tris: np.ndarray, h, tol: float,
            slide: np.ndarray | None = None, SA: np.ndarray | None = None, SB: np.ndarray | None = None,
            bias: float | None = None, w_wall: float = 200.0):
    """The kernel arguments of one coast zone (coast/relax.relax's set-up without padding): a dict."""
    from meshom.geo import R_EARTH_M

    n = len(X0)
    Ed = edges_of(quads, tris)
    l0 = np.linalg.norm(X0[Ed[:, 0]] - X0[Ed[:, 1]], axis=1)
    mid = X0[Ed[:, 0]] + X0[Ed[:, 1]]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    lh = np.log(h(mid) / R_EARTH_M)
    if bias is None:
        bias = float(np.median(np.log(l0) - lh))
    fidx = np.flatnonzero(free)
    s = np.zeros(n)
    c = np.zeros(n)
    np.add.at(s, Ed[:, 0], l0)
    np.add.at(s, Ed[:, 1], l0)
    np.add.at(c, Ed[:, 0], 1)
    np.add.at(c, Ed[:, 1], 1)
    mean_e = s / np.maximum(c, 1)
    E1, E2 = _basis(X0[fidx])
    pairs = elem_pairs(quads, tris, n)
    nel = len(quads) + len(tris)
    slide = np.zeros(0, np.int64) if slide is None else np.asarray(slide, np.int64)
    if slide.size:
        cA, cB = np.ascontiguousarray(SA, dtype=np.float64), np.ascontiguousarray(SB, dtype=np.float64)
        cs = np.maximum(mean_e[slide], 1e-12)
    else:
        cA = cB = np.zeros((0, 1, 3))
        cs = np.zeros(0)
    return dict(P0=np.ascontiguousarray(X0, dtype=np.float64), fidx=fidx.astype(np.int64),
                sc=np.ascontiguousarray(mean_e[fidx]), E1=np.ascontiguousarray(E1), E2=np.ascontiguousarray(E2),
                quads=np.ascontiguousarray(quads, dtype=np.int64).reshape(-1, 4),
                tris=np.ascontiguousarray(tris, dtype=np.int64).reshape(-1, 3),
                Ed=np.ascontiguousarray(Ed, dtype=np.int64), lh=np.ascontiguousarray(lh, dtype=np.float64),
                bias=float(bias), wall=float(tol - WALL_MARGIN), w_wall=float(w_wall),
                pa=np.ascontiguousarray(pairs[:, 0], dtype=np.int64),
                pb=np.ascontiguousarray(pairs[:, 1], dtype=np.int64),
                ci=np.ascontiguousarray(slide, dtype=np.int64), cA=cA, cB=cB, cs=np.ascontiguousarray(cs),
                X=np.zeros((n, 3)), G=np.zeros((n, 3)), Ynorm=np.zeros(len(fidx)), M=np.zeros(max(nel, 1)),
                GM=np.zeros(max(nel, 1)))


_ORDER = ("P0", "fidx", "sc", "E1", "E2", "quads", "tris", "Ed", "lh", "bias", "wall", "w_wall", "pa", "pb", "ci",
          "cA", "cB", "cs", "X", "G", "Ynorm", "M", "GM")


def fg_of(k: dict):
    """f(v) -> (energy, gradient) for scipy."""
    args = tuple(k[x] for x in _ORDER)

    def f(v):
        g = np.empty(len(v))
        e = energy_grad(v, *args, g)
        return e, g

    return f


def relax(X0: np.ndarray, free: np.ndarray, quads: np.ndarray, tris: np.ndarray, h, tol: float,
          slide: np.ndarray | None = None, SA: np.ndarray | None = None, SB: np.ndarray | None = None,
          bias: float | None = None, max_iter: int = 300, w_wall: float = 200.0,
          solver: str = "numba") -> tuple[np.ndarray, dict]:
    """coast/relax.relax with the Numba energy and L-BFGS (memory 20); solver "scipy": scipy's L-BFGS-B on it."""
    from meshom.meshfix.coast.refcoast import project

    k = prepare(X0, free, quads, tris, h, tol, slide, SA, SB, bias, w_wall)
    fidx, sc, E1, E2 = k["fidx"], k["sc"], k["E1"], k["E2"]
    v0 = np.zeros(2 * len(fidx))
    if solver == "scipy":
        from scipy.optimize import minimize

        f = fg_of(k)
        e0 = f(v0)[0]
        res = minimize(f, v0, jac=True, method="L-BFGS-B", options=dict(maxiter=max_iter, maxcor=20))
        vx, e1, nit = res.x, float(res.fun), int(res.nit)
    else:
        vx, e1, e0, nit, _ = lbfgs(v0, *(k[x] for x in _ORDER), int(max_iter), 20)
    v = vx.reshape(-1, 2)
    X = X0.copy()
    X[fidx] += sc[:, None] * (v[:, 0:1] * E1 + v[:, 1:2] * E2)
    X[fidx] /= np.linalg.norm(X[fidx], axis=1, keepdims=True)
    sl = k["ci"]
    if sl.size:
        X[sl] = project(X[sl], SA, SB)
    return X, {"e0": float(e0), "e1": float(e1), "nit": int(nit), "bias": k["bias"]}


# ------------------------------------------------------------------------------- L-BFGS in Numba (copied, B139)
# nrelax.py's _dcstep is shared; _linesearch and lbfgs below are nrelax.py's with the coast energy's arguments.


@njit(cache=True)
def _linesearch(x0, f0, g0, d, stp, stpmax, P0, fidx, sc, E1, E2, quads, tris, Ed, lh, bias, wall, w_wall, pa,
                pb, ci, cA, cB, cs, X, G, Ynorm, M, GM, x, g):
    """dcsrch along d from x0 (f0, g0); x, g receive the last point. Returns (status 0 converged / 1 warning /
    2 failed (MAXLS evaluations or an ascent direction), f at x, stp, evaluations)."""
    ftol, gtol, xtol, stpmin = 1e-3, 0.9, 0.1, 0.0
    ginit = 0.0
    for i in range(len(d)):
        ginit += g0[i] * d[i]
    if ginit >= 0.0:
        return 2, f0, stp, 0
    brackt = False
    stage = 1
    finit = f0
    gtest = ftol * ginit
    width = stpmax - stpmin
    width1 = width / 0.5
    stx, fx, gx = 0.0, finit, ginit
    sty, fy, gy = 0.0, finit, ginit
    stmin = 0.0
    stmax = stp + 4.0 * stp
    nfev = 0
    f = f0
    while True:
        for i in range(len(d)):
            x[i] = x0[i] + stp * d[i]
        f = energy_grad(x, P0, fidx, sc, E1, E2, quads, tris, Ed, lh, bias, wall, w_wall, pa, pb, ci, cA, cB, cs, X,
                        G, Ynorm, M, GM, g)
        nfev += 1
        gd = 0.0
        for i in range(len(d)):
            gd += g[i] * d[i]
        ftest = finit + stp * gtest
        if stage == 1 and f <= ftest and gd >= 0:
            stage = 2
        warn = False
        if brackt and (stp <= stmin or stp >= stmax):
            warn = True
        if brackt and stmax - stmin <= xtol * stmax:
            warn = True
        if stp == stpmax and f <= ftest and gd <= gtest:
            warn = True
        if stp == stpmin and (f > ftest or gd >= gtest):
            warn = True
        if f <= ftest and abs(gd) <= gtol * -ginit:
            return 0, f, stp, nfev
        if warn:
            return 1, f, stp, nfev
        if nfev >= MAXLS:
            return 2, f, stp, nfev
        if stage == 1 and f <= fx and f > ftest:
            fm = f - stp * gtest
            fxm = fx - stx * gtest
            fym = fy - sty * gtest
            gm = gd - gtest
            gxm = gx - gtest
            gym = gy - gtest
            stx, fxm, gxm, sty, fym, gym, stp, brackt = _dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt,
                                                                 stmin, stmax)
            fx = fxm + stx * gtest
            fy = fym + sty * gtest
            gx = gxm + gtest
            gy = gym + gtest
        else:
            stx, fx, gx, sty, fy, gy, stp, brackt = _dcstep(stx, fx, gx, sty, fy, gy, stp, f, gd, brackt, stmin,
                                                            stmax)
        if brackt:
            if abs(sty - stx) >= 0.66 * width1:
                stp = stx + 0.5 * (sty - stx)
            width1 = width
            width = abs(sty - stx)
        if brackt:
            stmin = min(stx, sty)
            stmax = max(stx, sty)
        else:
            stmin = stp + 1.1 * (stp - stx)
            stmax = stp + 4.0 * (stp - stx)
        stp = min(max(stp, stpmin), stpmax)
        if (brackt and (stp <= stmin or stp >= stmax)) or (brackt and stmax - stmin <= xtol * stmax):
            stp = stx


@njit(cache=True)
def lbfgs(v0, P0, fidx, sc, E1, E2, quads, tris, Ed, lh, bias, wall, w_wall, pa, pb, ci, cA, cB, cs, X, G, Ynorm,
          M, GM, max_iter, m):
    """Minimise the energy from v0 (module comment above): (v, f, f0, iterations, evaluations)."""
    n = len(v0)
    x = v0.copy()
    g = np.empty(n)
    f = energy_grad(x, P0, fidx, sc, E1, E2, quads, tris, Ed, lh, bias, wall, w_wall, pa, pb, ci, cA, cB, cs, X,
                        G, Ynorm, M, GM, g)
    f_start = f
    nfev = 1
    Sm = np.zeros((m, n))
    Ym = np.zeros((m, n))
    rho = np.zeros(m)
    al = np.zeros(m)
    col = 0
    head = 0
    it = 0
    d = np.empty(n)
    x0 = np.empty(n)
    g0 = np.empty(n)
    xt = np.empty(n)
    gt = np.empty(n)
    gmax = 0.0
    for i in range(n):
        gmax = max(gmax, abs(g[i]))
    if gmax <= PGTOL or n == 0:
        return x, f, f_start, 0, nfev
    tol = FACTR * EPSMCH
    while True:
        # direction -H g (two-loop; H0 = s'y / y'y of the newest pair)
        for i in range(n):
            d[i] = -g[i]
        if col > 0:
            for k in range(col):
                j = (head + col - 1 - k) % m
                a = 0.0
                for i in range(n):
                    a += Sm[j, i] * d[i]
                a *= rho[j]
                al[j] = a
                for i in range(n):
                    d[i] -= a * Ym[j, i]
            jn = (head + col - 1) % m
            yy = 0.0
            for i in range(n):
                yy += Ym[jn, i] * Ym[jn, i]
            gam = 1.0 / (rho[jn] * yy)
            for i in range(n):
                d[i] *= gam
            for k in range(col):
                j = (head + k) % m
                b = 0.0
                for i in range(n):
                    b += Ym[j, i] * d[i]
                b *= rho[j]
                for i in range(n):
                    d[i] += (al[j] - b) * Sm[j, i]
        dn = 0.0
        for i in range(n):
            dn += d[i] * d[i]
        dn = np.sqrt(dn)
        stp = min(1.0 / dn, 1e10) if (it == 0 and col == 0) else 1.0
        for i in range(n):
            x0[i] = x[i]
            g0[i] = g[i]
        f0 = f
        status, ft, stp, ne = _linesearch(x0, f0, g0, d, stp, 1e10, P0, fidx, sc, E1, E2, quads, tris, Ed, lh,
                                          bias, wall, w_wall, pa, pb, ci, cA, cB, cs, X, G, Ynorm, M, GM, xt, gt)
        nfev += ne
        if status == 2:                                     # restore; restart with an empty memory or stop
            if col == 0:
                break
            col = 0
            head = 0
            continue
        for i in range(n):
            x[i] = xt[i]
            g[i] = gt[i]
        f = ft
        it += 1
        if it >= max_iter:
            break
        gmax = 0.0
        for i in range(n):
            gmax = max(gmax, abs(g[i]))
        if gmax <= PGTOL:
            break
        if (f0 - f) <= tol * max(abs(f0), abs(f), 1.0):
            break
        # the new pair s = stp d, y = g - g0
        gd0 = 0.0
        sy = 0.0
        for i in range(n):
            gd0 += g0[i] * d[i]
            sy += (g[i] - g0[i]) * d[i]
        sy *= stp
        if sy <= EPSMCH * (-gd0 * stp):
            continue
        if col < m:
            j = (head + col) % m
            col += 1
        else:
            j = head
            head = (head + 1) % m
        for i in range(n):
            Sm[j, i] = stp * d[i]
            Ym[j, i] = g[i] - g0[i]
        rho[j] = 1.0 / sy
    return x, f, f_start, it, nfev
