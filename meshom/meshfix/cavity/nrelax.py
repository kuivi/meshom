"""Numba energy and gradient of the cavity relaxation (decisions B136, B139): the energy of relax.py, hand-differentiated.

The same energy as relax._energy (module docstring of relax.py), evaluated without padding on the unpadded local
mesh. Gradient: reverse mode by hand through the corner terms (tangent projections at the corner node), the edge
terms, the mean-edge ratio terms and the retraction x = normalise(x0 + s (v1 e1 + v2 e2)). Serial kernels (one
candidate per call; parallelism is over candidates, engine.py).
"""
import numpy as np
from numba import njit

# the energy's weights (B136; relax.py, the JAX reference, imports them from here)
W_HINGE = 50.0
TH_HINGE = 100.0
W_BAR = 1e3
SIN_MIN = 0.2
W_TRI = 0.5
W_SIZE = 0.2
W_WALL = 200.0
WALL_MARGIN = 0.02
W_RATIO = 2.0
MAX_ITER = 300
C_LIM = float(np.cos(np.radians(TH_HINGE)))


def _basis(p):
    k = np.argmin(np.abs(p), axis=1)
    a = np.zeros_like(p)
    a[np.arange(len(p)), k] = 1.0
    e1 = a - (a * p).sum(1, keepdims=True) * p
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(p, e1)


def edges_of(quads: np.ndarray, tris: np.ndarray) -> np.ndarray:
    e = [np.stack([c, np.roll(c, -1, 1)], -1).reshape(-1, 2) for c in (quads, tris) if len(c)]
    e = np.sort(np.concatenate(e), axis=1)
    return np.unique(e, axis=0)


def elem_pairs(quads: np.ndarray, tris: np.ndarray, n: int) -> np.ndarray:
    """(K,2) element index pairs (quads first, then tris) across interior edges."""
    cells = [(c, np.roll(c, -1, 1)) for c in (quads, tris) if len(c)]
    a = np.concatenate([x.ravel() for x, _ in cells])
    b = np.concatenate([y.ravel() for _, y in cells])
    eid = np.concatenate([np.repeat(np.arange(len(quads)), 4), len(quads) + np.repeat(np.arange(len(tris)), 3)]) \
        if len(tris) else np.repeat(np.arange(len(quads)), 4)
    key = np.minimum(a, b) * n + np.maximum(a, b)
    o = np.argsort(key, kind="stable")
    ks = key[o]
    same = np.flatnonzero(ks[1:] == ks[:-1])
    return np.column_stack([eid[o[same]], eid[o[same + 1]]])




@njit(cache=True, fastmath=False)
def _positions(v, P0, fidx, sc, E1, E2, X, Ynorm):
    X[:] = P0
    for k in range(len(fidx)):
        i = fidx[k]
        y0 = P0[i, 0] + sc[k] * (v[2 * k] * E1[k, 0] + v[2 * k + 1] * E2[k, 0])
        y1 = P0[i, 1] + sc[k] * (v[2 * k] * E1[k, 1] + v[2 * k + 1] * E2[k, 1])
        y2 = P0[i, 2] + sc[k] * (v[2 * k] * E1[k, 2] + v[2 * k + 1] * E2[k, 2])
        nr = np.sqrt(y0 * y0 + y1 * y1 + y2 * y2)
        Ynorm[k] = nr
        X[i, 0] = y0 / nr
        X[i, 1] = y1 / nr
        X[i, 2] = y2 / nr


@njit(cache=True, fastmath=False)
def _corner_acc(X, G, ip, inx, iq, quad):
    """Energy of one corner at node ip (next node inx, previous iq) and its gradient added to G."""
    p0, p1, p2 = X[ip, 0], X[ip, 1], X[ip, 2]
    a0, a1, a2 = X[inx, 0] - p0, X[inx, 1] - p1, X[inx, 2] - p2
    b0, b1, b2 = X[iq, 0] - p0, X[iq, 1] - p1, X[iq, 2] - p2
    ap = a0 * p0 + a1 * p1 + a2 * p2
    bp = b0 * p0 + b1 * p1 + b2 * p2
    u0, u1, u2 = a0 - ap * p0, a1 - ap * p1, a2 - ap * p2
    w0, w1, w2 = b0 - bp * p0, b1 - bp * p1, b2 - bp * p2
    A = np.sqrt(u0 * u0 + u1 * u1 + u2 * u2) + 1e-30
    B = np.sqrt(w0 * w0 + w1 * w1 + w2 * w2) + 1e-30
    N = A * B
    dot = u0 * w0 + u1 * w1 + u2 * w2
    c0, c1, c2 = u1 * w2 - u2 * w1, u2 * w0 - u0 * w2, u0 * w1 - u1 * w0          # u x w
    t = c0 * p0 + c1 * p1 + c2 * p2
    cs = dot / N
    sn = t / N
    e = 0.0
    dcs = 0.0
    dsn = 0.0
    if quad:
        e += cs * cs
        dcs += 2.0 * cs
        hh = C_LIM - cs
        if hh > 0.0:
            e += W_HINGE * hh * hh
            dcs -= 2.0 * W_HINGE * hh
    else:
        e += W_TRI * (cs - 0.5) ** 2
        dcs += 2.0 * W_TRI * (cs - 0.5)
    bb = SIN_MIN - sn
    if bb > 0.0:
        e += W_BAR * bb * bb
        dsn -= 2.0 * W_BAR * bb
    if dcs == 0.0 and dsn == 0.0:
        return e
    # d/du, d/dw of cs and sn
    A2, B2 = A * A, B * B
    # w x p, p x u
    wp0, wp1, wp2 = w1 * p2 - w2 * p1, w2 * p0 - w0 * p2, w0 * p1 - w1 * p0
    pu0, pu1, pu2 = p1 * u2 - p2 * u1, p2 * u0 - p0 * u2, p0 * u1 - p1 * u0
    gu0 = dcs * (w0 / N - cs * u0 / A2) + dsn * (wp0 / N - sn * u0 / A2)
    gu1 = dcs * (w1 / N - cs * u1 / A2) + dsn * (wp1 / N - sn * u1 / A2)
    gu2 = dcs * (w2 / N - cs * u2 / A2) + dsn * (wp2 / N - sn * u2 / A2)
    gw0 = dcs * (u0 / N - cs * w0 / B2) + dsn * (pu0 / N - sn * w0 / B2)
    gw1 = dcs * (u1 / N - cs * w1 / B2) + dsn * (pu1 / N - sn * w1 / B2)
    gw2 = dcs * (u2 / N - cs * w2 / B2) + dsn * (pu2 / N - sn * w2 / B2)
    gp0 = dsn * c0 / N
    gp1 = dsn * c1 / N
    gp2 = dsn * c2 / N
    # u = a - (a.p) p
    gup = gu0 * p0 + gu1 * p1 + gu2 * p2
    ga0, ga1, ga2 = gu0 - gup * p0, gu1 - gup * p1, gu2 - gup * p2
    gp0 -= ap * gu0 + gup * a0
    gp1 -= ap * gu1 + gup * a1
    gp2 -= ap * gu2 + gup * a2
    gwp = gw0 * p0 + gw1 * p1 + gw2 * p2
    gb0, gb1, gb2 = gw0 - gwp * p0, gw1 - gwp * p1, gw2 - gwp * p2
    gp0 -= bp * gw0 + gwp * b0
    gp1 -= bp * gw1 + gwp * b1
    gp2 -= bp * gw2 + gwp * b2
    # a = n - p, b = q - p
    G[inx, 0] += ga0
    G[inx, 1] += ga1
    G[inx, 2] += ga2
    G[iq, 0] += gb0
    G[iq, 1] += gb1
    G[iq, 2] += gb2
    G[ip, 0] += gp0 - ga0 - gb0
    G[ip, 1] += gp1 - ga1 - gb1
    G[ip, 2] += gp2 - ga2 - gb2
    return e


@njit(cache=True, fastmath=False)
def energy_grad(v, P0, fidx, sc, E1, E2, quads, tris, Ed, lh, bias, wall, pa, pb, X, G, Ynorm, M, GM, g):
    """Energy at v and its gradient into g (len 2 * len(fidx)); X, G, Ynorm, M, GM are work arrays."""
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
        x = abs(r) - wall
        if x > 0.0:
            e += W_WALL * x * x
            de += 2.0 * W_WALL * x * (1.0 if r > 0 else -1.0)
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
            GM[pa[k]] += 2.0 * W_RATIO * d / M[pa[k]]
            GM[pb[k]] -= 2.0 * W_RATIO * d / M[pb[k]]
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
            bias: float | None = None, lh: np.ndarray | None = None):
    """The kernel arguments of one zone (relax.py's set-up without padding): a dict; lh (optional) the edges'
    ln(h / R) when the caller has it."""
    from meshom.geo import R_EARTH_M

    n = len(X0)
    Ed = edges_of(quads, tris)
    l0 = np.linalg.norm(X0[Ed[:, 0]] - X0[Ed[:, 1]], axis=1)
    if lh is None:
        mid = X0[Ed[:, 0]] + X0[Ed[:, 1]]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        lh = np.log(h(mid) / R_EARTH_M)
    if bias is None:
        bias = float(np.median(np.log(l0) - lh))
    fidx = np.flatnonzero(free)
    s = np.bincount(Ed.ravel(), weights=np.repeat(l0, 2), minlength=n)
    c = np.bincount(Ed.ravel(), minlength=n)
    sc = s[fidx] / np.maximum(c[fidx], 1)
    E1, E2 = _basis(X0[fidx])
    pairs = elem_pairs(quads, tris, n)
    nel = len(quads) + len(tris)
    return dict(P0=np.ascontiguousarray(X0, dtype=np.float64), fidx=fidx.astype(np.int64), sc=sc,
                E1=np.ascontiguousarray(E1), E2=np.ascontiguousarray(E2),
                quads=np.ascontiguousarray(quads, dtype=np.int64).reshape(-1, 4),
                tris=np.ascontiguousarray(tris, dtype=np.int64).reshape(-1, 3),
                Ed=np.ascontiguousarray(Ed, dtype=np.int64), lh=np.ascontiguousarray(lh, dtype=np.float64),
                bias=float(bias), wall=float(tol - WALL_MARGIN),
                pa=np.ascontiguousarray(pairs[:, 0], dtype=np.int64),
                pb=np.ascontiguousarray(pairs[:, 1], dtype=np.int64),
                X=np.zeros((n, 3)), G=np.zeros((n, 3)), Ynorm=np.zeros(len(fidx)), M=np.zeros(max(nel, 1)),
                GM=np.zeros(max(nel, 1)))


_ORDER = ("P0", "fidx", "sc", "E1", "E2", "quads", "tris", "Ed", "lh", "bias", "wall", "pa", "pb", "X", "G", "Ynorm",
          "M", "GM")


def fg_of(k: dict):
    """f(v) -> (energy, gradient) for scipy."""
    args = tuple(k[x] for x in _ORDER)

    def f(v):
        g = np.empty(len(v))
        e = energy_grad(v, *args, g)
        return e, g

    return f


def relax(X0: np.ndarray, free: np.ndarray, quads: np.ndarray, tris: np.ndarray, h, tol: float,
          bias: float | None = None, max_iter: int = 300, lh: np.ndarray | None = None,
          solver: str = "numba") -> tuple[np.ndarray, dict]:
    """relax.relax with the Numba energy (the same energy and stopping rule, memory 20): solver "numba" the L-BFGS
    above, "scipy" scipy's L-BFGS-B on the Numba energy."""
    k = prepare(X0, free, quads, tris, h, tol, bias, lh)
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
    return X, {"e0": float(e0), "e1": float(e1), "nit": int(nit), "bias": k["bias"]}


# ------------------------------------------------------------------------------- L-BFGS in Numba (B139)
# scipy's L-BFGS-B (v3.0, the C port of scipy 1.18) on an unbounded problem, rewritten: the direction -H g by the
# two-loop recursion (= the compact-form subspace step of L-BFGS-B when no bound is active; H0 = s'y / y'y of the
# newest pair), the MINPACK-2 line search dcsrch / dcstep (ftol 1e-3, gtol 0.9, xtol 0.1, at most MAXLS
# evaluations; first step 1 / |d|, then 1), the pair skipped when s'y <= eps (-g'd stp), the memory cleared and the
# iteration restarted after a failed line search (stop if the memory was empty), stop when max |g| <= PGTOL, when
# (f_old - f) <= FACTR eps max(|f_old|, |f|, 1), or after max_iter iterations. Same answer as scipy up to rounding.

FACTR = 1e7
PGTOL = 1e-5
MAXLS = 20
EPSMCH = float(np.finfo(float).eps)


@njit(cache=True)
def _dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    sgnd = np.sign(dp) * np.sign(dx)
    if fp > fx:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp < stx:
            gamma = -gamma
        p = (gamma - dx) + theta
        q = ((gamma - dx) + gamma) + dp
        r = p / q
        stpc = stx + r * (stp - stx)
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx)
        if abs(stpc - stx) <= abs(stpq - stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq - stpc) / 2.0
        brackt = True
    elif sgnd < 0.0:
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + dx
        r = p / q
        stpc = stp + r * (stx - stp)
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if abs(stpc - stp) > abs(stpq - stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True
    elif abs(dp) < abs(dx):
        theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt(max(0.0, (theta / s) ** 2 - (dx / s) * (dp / s)))
        if stp > stx:
            gamma = -gamma
        p = (gamma - dp) + theta
        q = (gamma + (dx - dp)) + gamma
        r = p / q
        if r < 0 and gamma != 0:
            stpc = stp + r * (stx - stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp / (dp - dx)) * (stx - stp)
        if brackt:
            if abs(stpc - stp) < abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            if stp > stx:
                stpf = min(stp + 0.66 * (sty - stp), stpf)
            else:
                stpf = max(stp + 0.66 * (sty - stp), stpf)
        else:
            if abs(stpc - stp) > abs(stpq - stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = min(max(stpf, stpmin), stpmax)
    else:
        if brackt:
            theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))
            gamma = s * np.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))
            if stp > sty:
                gamma = -gamma
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + dy
            r = p / q
            stpc = stp + r * (sty - stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin
    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp
    return stx, fx, dx, sty, fy, dy, stpf, brackt


@njit(cache=True)
def _linesearch(x0, f0, g0, d, stp, stpmax, P0, fidx, sc, E1, E2, quads, tris, Ed, lh, bias, wall, pa, pb,
                X, G, Ynorm, M, GM, x, g):
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
        f = energy_grad(x, P0, fidx, sc, E1, E2, quads, tris, Ed, lh, bias, wall, pa, pb, X, G, Ynorm, M, GM, g)
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
def lbfgs(v0, P0, fidx, sc, E1, E2, quads, tris, Ed, lh, bias, wall, pa, pb, X, G, Ynorm, M, GM, max_iter, m):
    """Minimise the energy from v0 (module comment above): (v, f, f0, iterations, evaluations)."""
    n = len(v0)
    x = v0.copy()
    g = np.empty(n)
    f = energy_grad(x, P0, fidx, sc, E1, E2, quads, tris, Ed, lh, bias, wall, pa, pb, X, G, Ynorm, M, GM, g)
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
                                          bias, wall, pa, pb, X, G, Ynorm, M, GM, xt, gt)
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
