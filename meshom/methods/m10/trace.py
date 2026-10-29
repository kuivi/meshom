"""Separatrix tracing as a motorcycle graph on the background mesh (m10 kernels, Numba).

Traces start at the cones of the cross field (one per separatrix direction, 4 (1 - k) of them for a cone of index k)
and at reflex coast corners (one per extra quarter turn of the water angle). All traces advance together, one step of
step_frac * h each per round (so they grow at the same speed in the metric g/h^2), and each stops at the first of:

- JOIN: an active trace's tip lies head-on within eps_join * h ahead of its own tip (the two are one separatrix traced
  from both ends; they are joined tip to tip);
- HIT: its step crosses an existing trace segment (any trace, its own older part included): a T-junction (motorcycle
  graph, Eppstein et al. 2008, as QGP / QuadWild build their T-meshes);
- COAST: the step leaves the water (the walk on the bg mesh leaves through a coast edge): snapped to the nearer coast
  node of that edge;
- LOST: it exceeds max_len.

Steps follow the field with a midpoint (RK2) rule: the cross direction closest to the current heading is taken at the
point and at the half step. The first `straight` radians from a cone go straight along the separatrix direction found
on a circle around it (the field is singular in the cone triangle). Lengths are angles on the unit sphere.
"""
import numpy as np
from numba import njit

from meshom.methods.m02.packing import bary, field_at, walk

END_ACTIVE, END_HIT, END_JOIN, END_COAST, END_LOST = 0, 1, 2, 3, 4


@njit(cache=True, inline="always")
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def _cross(a, b, out):
    out[0] = a[1] * b[2] - a[2] * b[1]
    out[1] = a[2] * b[0] - a[0] * b[2]
    out[2] = a[0] * b[1] - a[1] * b[0]


@njit(cache=True)
def _normalize(v):
    r = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    v[0] /= r
    v[1] /= r
    v[2] /= r


@njit(cache=True)
def gstep(p, d, ang, q):
    """q = point at angle ang along the great circle from p with initial unit tangent d."""
    c, s = np.cos(ang), np.sin(ang)
    for k in range(3):
        q[k] = c * p[k] + s * d[k]
    _normalize(q)


@njit(cache=True)
def _heading(p, q, out):
    """Unit tangent at q of the great circle p -> q, continuing away from p."""
    c = _dot(p, q)
    for k in range(3):
        out[k] = c * q[k] - p[k]
    _normalize(out)


@njit(cache=True)
def _to_tangent(x, v):
    """v projected onto the tangent plane at x, normalised (in place)."""
    s = _dot(v, x)
    for k in range(3):
        v[k] -= s * x[k]
    _normalize(v)


@njit(cache=True)
def near_dir(Pb, Tb, ub, lhb, t, x, v, w, d, e, out):
    """Field direction at x (bg triangle t) closest to the heading v -> out; returns h(x) (radians)."""
    bary(Pb, Tb, t, x, w)
    hh = field_at(Pb, Tb, ub, lhb, t, x, w, d, 4)
    _cross(x, d, e)
    a0 = _dot(v, d)
    a1 = _dot(v, e)
    if abs(a0) >= abs(a1):
        s = 1.0 if a0 >= 0 else -1.0
        for k in range(3):
            out[k] = s * d[k]
    else:
        s = 1.0 if a1 >= 0 else -1.0
        for k in range(3):
            out[k] = s * e[k]
    return hh


@njit(cache=True)
def h_at(Pb, Tb, lhb, t, x, w):
    bary(Pb, Tb, t, x, w)
    s = 0.0
    for k in range(3):
        s += w[k] * lhb[Tb[t, k]]
    return np.exp(s)


@njit(cache=True)
def _seg_cross(a, b, c, d, n1, n2):
    """(t, u): crossing of the short arcs a -> b and c -> d at a + t (b - a), c + u (d - c); t = -1 if none."""
    _cross(a, b, n1)
    sc = _dot(n1, c)
    sd = _dot(n1, d)
    if (sc > 0 and sd > 0) or (sc < 0 and sd < 0) or (sc == 0 and sd == 0):
        return -1.0, -1.0
    _cross(c, d, n2)
    sa = _dot(n2, a)
    sb = _dot(n2, b)
    if (sa > 0 and sb > 0) or (sa < 0 and sb < 0) or sa == sb:
        return -1.0, -1.0
    if _dot(a, c) <= 0:
        return -1.0, -1.0
    t = sa / (sa - sb)
    u = sc / (sc - sd)
    # nearly collinear arcs: both side tests are ~0 with random signs; accept only if the two estimates agree
    # (compare the directions: the two chord points lie on the same line through the centre, at different radii)
    r1 = 0.0
    r2 = 0.0
    for k in range(3):
        r1 += (a[k] + t * (b[k] - a[k])) ** 2
        r2 += (c[k] + u * (d[k] - c[k])) ** 2
    r1 = np.sqrt(r1)
    r2 = np.sqrt(r2)
    e2 = 0.0
    l2 = 0.0
    for k in range(3):
        e2 += ((a[k] + t * (b[k] - a[k])) / r1 - (c[k] + u * (d[k] - c[k])) / r2) ** 2
        l2 += (b[k] - a[k]) ** 2 + (d[k] - c[k]) ** 2
    if e2 > 1e-6 * l2:
        return -1.0, -1.0
    return t, u


@njit(cache=True)
def _gather(nbr, cen, rad, head, nxt, stamp, sid, stack, out, t0, x, r):
    """Point ids bucketed in the bg triangles within chord radius r of x (BFS from t0) -> out; returns the count."""
    stack[0] = t0
    stamp[t0] = sid
    top = 1
    pos = 0
    m = 0
    while pos < top:
        t = stack[pos]
        pos += 1
        j = head[t]
        while j >= 0:
            if m < len(out):
                out[m] = j
                m += 1
            j = nxt[j]
        for k in range(3):
            nt = nbr[t, k]
            if nt >= 0 and stamp[nt] != sid:
                stamp[nt] = sid
                c0, c1, c2 = cen[nt, 0] - x[0], cen[nt, 1] - x[1], cen[nt, 2] - x[2]
                if np.sqrt(c0 * c0 + c1 * c1 + c2 * c2) <= r + rad[nt]:
                    stack[top] = nt
                    top += 1
    return m


@njit(cache=True)
def _append(TP, prev, nextp, ptrc, ppos, ptri, head, nxt, n, a, x, t, tip):
    TP[n, 0], TP[n, 1], TP[n, 2] = x[0], x[1], x[2]
    j = tip[a]
    prev[n] = j
    nextp[n] = -1
    if j >= 0:
        nextp[j] = n
        ppos[n] = ppos[j] + 1
    else:
        ppos[n] = 0
    ptrc[n] = a
    ptri[n] = t
    nxt[n] = head[t]
    head[t] = n
    tip[a] = n


@njit(cache=True)
def _first_cross(TP, prev, nextp, ptrc, ppos, jlink, s_org, s_org_pos, found, m, a, ja, p, q, hh, skip, n1, n2, x):
    """First crossing of the arc p -> q with a trace segment among the points found[:m]: (t, c0, c1, u), t = 2 if
    none. Skipped: trace a's own last segment, siblings touching at their common origin, segments with an end at
    point `skip` (>= 0)."""
    best_t = 2.0
    best_c = -1
    best_d = -1
    best_u = 0.0
    for ii in range(m):
        j = found[ii]
        for kk in range(3):
            if kk == 0:
                c0, c1 = prev[j], j
            elif kk == 1:
                c0, c1 = j, nextp[j]
            else:
                c0, c1 = j, jlink[j]
            if c0 < 0 or c1 < 0:
                continue
            if c0 == skip or c1 == skip:
                continue
            b = ptrc[c0]
            if b == a and kk != 2 and ppos[c1] >= ppos[ja] - 1 and ptrc[c1] == a:
                continue                                   # own last segment
            if c1 == ja or c0 == ja:
                continue
            tt, uu = _seg_cross(p, q, TP[c0], TP[c1], n1, n2)
            if tt < 1e-9 or tt > 1.0 or tt >= best_t:
                continue
            if s_org[ptrc[c0]] == s_org[a]:                  # siblings touching at their common origin
                for k in range(3):
                    x[k] = p[k] + tt * (q[k] - p[k])
                _normalize(x)
                o = s_org_pos[a]
                dx = np.sqrt((x[0] - o[0]) ** 2 + (x[1] - o[1]) ** 2 + (x[2] - o[2]) ** 2)
                if dx < 0.05 * hh:
                    continue
            best_t, best_c, best_d, best_u = tt, c0, c1, uu
    return best_t, best_c, best_d, best_u


@njit(cache=True)
def motorcycle(Pb, Tb, nbr, ub, lhb, cen, rad, s_pos, s_dir, s_tri, s_org, s_org_pos, s_straight, step_frac,
               eps_join, max_len, cap):
    """Trace all starts together. Returns (n, TP, prev, nextp, ptrc, jlink, first, tip, etype, ea, eb, eu, full).

    Points 0..n-1: TP positions, prev / nextp along their trace, ptrc their trace, jlink the partner tip of a JOIN.
    Per trace: first / tip point, etype (END_*), and for HIT the crossed segment (ea -> eb) at fraction eu; for JOIN
    ea = partner trace; for COAST ea = the bg coast node."""
    S = len(s_pos)
    nT = len(Tb)
    TP = np.zeros((cap, 3))
    prev = np.full(cap, -1, dtype=np.int64)
    nextp = np.full(cap, -1, dtype=np.int64)
    ptrc = np.full(cap, -1, dtype=np.int64)
    ppos = np.zeros(cap, dtype=np.int64)
    ptri = np.zeros(cap, dtype=np.int64)
    jlink = np.full(cap, -1, dtype=np.int64)
    head = np.full(nT, -1, dtype=np.int64)
    nxt = np.full(cap, -1, dtype=np.int64)
    stamp = np.zeros(nT, dtype=np.int64)
    stack = np.zeros(nT, dtype=np.int64)
    found = np.zeros(4096, dtype=np.int64)
    first = np.full(S, -1, dtype=np.int64)
    tip = np.full(S, -1, dtype=np.int64)
    etype = np.zeros(S, dtype=np.int64)
    ea = np.full(S, -1, dtype=np.int64)
    eb = np.full(S, -1, dtype=np.int64)
    eu = np.zeros(S)
    tdir = s_dir.copy()
    ttri = s_tri.copy()
    tlen = np.zeros(S)
    w = np.zeros(3)
    d = np.zeros(3)
    e = np.zeros(3)
    d1 = np.zeros(3)
    d2 = np.zeros(3)
    q = np.zeros(3)
    mid = np.zeros(3)
    x = np.zeros(3)
    hd = np.zeros(3)
    n1 = np.zeros(3)
    n2 = np.zeros(3)
    n = 0
    for a in range(S):
        _append(TP, prev, nextp, ptrc, ppos, ptri, head, nxt, n, a, s_pos[a], s_tri[a], tip)
        first[a] = n
        n += 1
    sid = 0
    n_active = S
    full = False
    while n_active > 0:
        for a in range(S):
            if etype[a] != END_ACTIVE:
                continue
            if n + 2 >= cap:
                full = True
                return n, TP, prev, nextp, ptrc, jlink, first, tip, etype, ea, eb, eu, full
            ja = tip[a]
            p = TP[ja]
            t = ttri[a]
            v = tdir[a]
            if tlen[a] < s_straight[a]:
                hh = h_at(Pb, Tb, lhb, t, p, w)
                ds = step_frac * hh
                gstep(p, v, ds, q)
            else:
                hh = near_dir(Pb, Tb, ub, lhb, t, p, v, w, d, e, d1)
                ds = step_frac * hh
                gstep(p, d1, 0.5 * ds, mid)
                tm = walk(Pb, Tb, nbr, t, mid)
                if tm < 0:
                    tm = t
                near_dir(Pb, Tb, ub, lhb, tm, mid, d1, w, d, e, d2)
                _to_tangent(p, d2)
                gstep(p, d2, ds, q)
            tq = walk(Pb, Tb, nbr, t, q)
            # ---- JOIN: an active tip head-on ahead, within eps_join h
            sid += 1
            m = _gather(nbr, cen, rad, head, nxt, stamp, sid, stack, found, t, p, (eps_join + step_frac) * hh)
            best_b = -1
            best_dd = 1e300
            for ii in range(m):
                j = found[ii]
                b = ptrc[j]
                if b == a or etype[b] != END_ACTIVE or tip[b] != j:
                    continue
                if _dot(tdir[b], v) > -0.7:
                    continue
                ahead = (TP[j, 0] - p[0]) * v[0] + (TP[j, 1] - p[1]) * v[1] + (TP[j, 2] - p[2]) * v[2]
                if ahead <= 0:
                    continue
                dq = np.sqrt((TP[j, 0] - q[0]) ** 2 + (TP[j, 1] - q[1]) ** 2 + (TP[j, 2] - q[2]) ** 2)
                if dq < eps_join * hh and dq < best_dd:
                    best_dd = dq
                    best_b = b
            if best_b >= 0:
                tb = tip[best_b]
                sid += 1
                m = _gather(nbr, cen, rad, head, nxt, stamp, sid, stack, found, t, p, (eps_join + 0.5) * hh)
                lt, _c, _d, _u = _first_cross(TP, prev, nextp, ptrc, ppos, jlink, s_org, s_org_pos, found, m, a, ja,
                                              p, TP[tb], hh, tb, n1, n2, x)
                if lt <= 1.0:
                    best_b = -1                                # the link would cross a trace: no join
            if best_b >= 0:
                b = best_b
                etype[a] = END_JOIN
                etype[b] = END_JOIN
                ea[a] = b
                ea[b] = a
                jlink[tip[a]] = tip[b]
                jlink[tip[b]] = tip[a]
                n_active -= 2
                continue
            # ---- HIT: the first crossing of p -> q with an existing segment
            sid += 1
            m = _gather(nbr, cen, rad, head, nxt, stamp, sid, stack, found, t, p, ds + (eps_join + 0.5) * hh)
            best_t, best_c, best_d, best_u = _first_cross(TP, prev, nextp, ptrc, ppos, jlink, s_org, s_org_pos,
                                                          found, m, a, ja, p, q, hh, -1, n1, n2, x)
            if best_c >= 0:
                for k in range(3):
                    x[k] = p[k] + best_t * (q[k] - p[k])
                _normalize(x)
                tx = walk(Pb, Tb, nbr, t, x)
                if tx < 0:
                    tx = t
                _append(TP, prev, nextp, ptrc, ppos, ptri, head, nxt, n, a, x, tx, tip)
                n += 1
                etype[a] = END_HIT
                ea[a] = best_c
                eb[a] = best_d
                eu[a] = best_u
                n_active -= 1
                continue
            # ---- COAST
            if tq < 0:
                if tq == -1:
                    etype[a] = END_LOST
                    n_active -= 1
                    continue
                tl = -2 - tq
                bn = -1
                bd = 1e300
                for k in range(3):
                    if nbr[tl, k] >= 0:
                        continue
                    for kk in range(2):
                        c0 = Tb[tl, (k + kk) % 3]
                        dq = (Pb[c0, 0] - q[0]) ** 2 + (Pb[c0, 1] - q[1]) ** 2 + (Pb[c0, 2] - q[2]) ** 2
                        if dq < bd:
                            bd = dq
                            bn = c0
                _append(TP, prev, nextp, ptrc, ppos, ptri, head, nxt, n, a, Pb[bn], tl, tip)
                n += 1
                etype[a] = END_COAST
                ea[a] = bn
                n_active -= 1
                continue
            # ---- advance
            _heading(p, q, hd)
            _append(TP, prev, nextp, ptrc, ppos, ptri, head, nxt, n, a, q, tq, tip)
            n += 1
            tdir[a, 0], tdir[a, 1], tdir[a, 2] = hd[0], hd[1], hd[2]
            ttri[a] = tq
            tlen[a] += ds
            if tlen[a] > max_len:
                etype[a] = END_LOST
                n_active -= 1
    return n, TP, prev, nextp, ptrc, jlink, first, tip, etype, ea, eb, eu, full


@njit(cache=True)
def separatrix_angles(Pb, Tb, nbr, ub, lhb, t0, c, e1, e2, r, m, k):
    """Angles theta (basis e1, e2 at c) of the 4 (1 - k) separatrices of a cone of index k at c (bg triangle t0).

    For an ideal cone the angle beta from the radial direction to the field on a circle around it is
    beta = phi0 - (1 - k) theta, and the separatrices are where 4 beta = 0 mod 2 pi. The field is sampled on the
    circle of radius r (m points) and only the phase 4 phi0 is fitted (mean of exp(i (4 beta + 4 (1 - k) theta))),
    so the count is always 4 (1 - k) and the directions are equally spaced, even when another cone lies within r."""
    w = np.zeros(3)
    d = np.zeros(3)
    q = np.zeros(3)
    dr = np.zeros(3)
    rr = np.zeros(3)
    xx = np.zeros(3)
    nsep = int(np.rint(4.0 * (1.0 - k)))
    acc = 0.0 + 0.0j
    for i in range(m):
        th = 2.0 * np.pi * i / m
        for kk in range(3):
            dr[kk] = np.cos(th) * e1[kk] + np.sin(th) * e2[kk]
        gstep(c, dr, r, q)
        t = walk(Pb, Tb, nbr, t0, q)
        if t < 0:
            t = -2 - t if t <= -2 else t0
        bary(Pb, Tb, t, q, w)
        field_at(Pb, Tb, ub, lhb, t, q, w, d, 4)
        _heading(c, q, rr)
        _cross(rr, d, xx)
        b = np.arctan2(_dot(xx, q), _dot(rr, d))
        acc += np.exp(1j * (4.0 * b + nsep * th))
    ph = np.angle(acc)
    out = np.zeros(max(nsep, 0))
    for j in range(nsep):
        out[j] = (ph - 2.0 * np.pi * j) / nsep
    return out


@njit(cache=True)
def field_at_points(Pb, Tb, nbr, ub, lhb, start, X):
    """(M,3) one field direction at each point X (walk from the bg triangles `start`)."""
    out = np.zeros((len(X), 3))
    w = np.zeros(3)
    d = np.zeros(3)
    for i in range(len(X)):
        t = walk(Pb, Tb, nbr, start[i], X[i])
        if t < 0:
            t = -2 - t if t <= -2 else start[i]
        bary(Pb, Tb, t, X[i], w)
        field_at(Pb, Tb, ub, lhb, t, X[i], w, d, 4)
        out[i, 0], out[i, 1], out[i, 2] = d[0], d[1], d[2]
    return out
