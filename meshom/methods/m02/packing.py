"""Frontal packing of points along a cross field (n = 4) or an asterisk field (n = 6) on the sphere (m02 kernels).

After Baudouin et al. 2014 (packing of parallelograms, Gmsh algorithm 9)
and Georgiadis et al. 2017 (cross / asterisk fields, coastal ocean, sphere; walks on the background mesh instead of an
R-tree, Algorithm 1). The points live on a background triangle mesh (bg) that carries the cross field u and ln h at
its nodes:

- point location: visibility walk from the parent's bg triangle (Georgiadis §2.3); a walk that leaves the bg mesh
  through a boundary edge means the candidate is on land and it is dropped;
- interpolation at x in bg triangle t: gnomonic barycentric weights; ln h linear; the field as the weighted mean of the
  corners' n-th power vectors, each corner's u projected onto the tangent plane at x (O(edge^2) from transport);
- candidates of a popped point p: the n field directions d_k at p; a geodesic step of length s along d_k, where d_k is
  turned by the field's rotation seen at the half step (midpoint rule, so the step follows the field's integral
  line) and s = logmean(h(p), h(q)) (metric length 1 along the arc, two fixed-point updates);
- exclusion: a candidate q is rejected when an existing point (its parent excepted, as in Gmsh) lies inside q's
  exclusion zone of half-width lim = alpha (h_q + h_j) / 2 in q's field frame: n = 4, the L-inf box |x|, |y| < lim
  (Gmsh FACTOR = 0.7 with the box of the existing point); n = 6, the hexagon |v . d_k| < lim for the three field
  directions d_k (the L-inf analogue for 6-fold symmetry: all six lattice neighbours at distance h have norm h);
  neighbours are found by a breadth-first search over bg triangles (points are bucketed by bg triangle) within the
  chord radius c alpha h_q / (1 - c alpha g / 2), c = sqrt(2) (box) or 2 / sqrt(3) (hexagon), g = max |grad h|;
- FIFO order = insertion order: the queue is the point array itself;
- gap filling: when the queue is empty, every bg node (in the given order) that is not excluded is inserted and the
  queue runs again from it, so gaps left where fronts collide get points that continue the lattice of the field.
"""
import numpy as np
from numba import njit


@njit(cache=True, inline="always")
def _det(ax, ay, az, bx, by, bz, cx, cy, cz):
    return ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)


@njit(cache=True)
def walk(Pb, Tb, nbr, t, x, max_steps=100000):
    """bg triangle containing the unit vector x, walking from t; -2 - t_last when the walk leaves the mesh through a
    boundary edge of t_last; -1 when it does not end (never seen)."""
    for _ in range(max_steps):
        best = -1
        bestv = 0.0
        for k in range(3):
            a = Tb[t, k]
            b = Tb[t, (k + 1) % 3]
            o = _det(Pb[a, 0], Pb[a, 1], Pb[a, 2], Pb[b, 0], Pb[b, 1], Pb[b, 2], x[0], x[1], x[2])
            if o < bestv:
                bestv = o
                best = k
        if best < 0:
            return t
        nt = nbr[t, best]
        if nt < 0:
            return -2 - t
        t = nt
    return -1


@njit(cache=True)
def bary(Pb, Tb, t, x, w):
    """Gnomonic barycentric weights of x in bg triangle t into w (3,), clamped to >= 0 and normalised."""
    a, b, c = Tb[t, 0], Tb[t, 1], Tb[t, 2]
    w[0] = _det(x[0], x[1], x[2], Pb[b, 0], Pb[b, 1], Pb[b, 2], Pb[c, 0], Pb[c, 1], Pb[c, 2])
    w[1] = _det(Pb[a, 0], Pb[a, 1], Pb[a, 2], x[0], x[1], x[2], Pb[c, 0], Pb[c, 1], Pb[c, 2])
    w[2] = _det(Pb[a, 0], Pb[a, 1], Pb[a, 2], Pb[b, 0], Pb[b, 1], Pb[b, 2], x[0], x[1], x[2])
    s = 0.0
    for k in range(3):
        if w[k] < 0.0:
            w[k] = 0.0
        s += w[k]
    if s <= 0.0:
        w[0] = w[1] = w[2] = 1.0 / 3.0
    else:
        for k in range(3):
            w[k] /= s


@njit(cache=True)
def field_at(Pb, Tb, ub, lhb, t, x, w, d, nd):
    """Field direction d (3,) (one of the nd) and h at x in bg triangle t (weights w from bary)."""
    # reference axis f1: corner 0's u in the tangent plane at x
    u0 = Tb[t, 0]
    s = ub[u0, 0] * x[0] + ub[u0, 1] * x[1] + ub[u0, 2] * x[2]
    f10, f11, f12 = ub[u0, 0] - s * x[0], ub[u0, 1] - s * x[1], ub[u0, 2] - s * x[2]
    r = np.sqrt(f10 * f10 + f11 * f11 + f12 * f12)
    f10, f11, f12 = f10 / r, f11 / r, f12 / r
    f20 = x[1] * f12 - x[2] * f11
    f21 = x[2] * f10 - x[0] * f12
    f22 = x[0] * f11 - x[1] * f10
    cs, sn, lh = 0.0, 0.0, 0.0
    for k in range(3):
        v = Tb[t, k]
        a = ub[v, 0] * f10 + ub[v, 1] * f11 + ub[v, 2] * f12
        b = ub[v, 0] * f20 + ub[v, 1] * f21 + ub[v, 2] * f22
        th = nd * np.arctan2(b, a)
        cs += w[k] * np.cos(th)
        sn += w[k] * np.sin(th)
        lh += w[k] * lhb[v]
    th = np.arctan2(sn, cs) / nd
    c, sth = np.cos(th), np.sin(th)
    d[0] = c * f10 + sth * f20
    d[1] = c * f11 + sth * f21
    d[2] = c * f12 + sth * f22
    return np.exp(lh)


@njit(cache=True, inline="always")
def _step(p, d, ang, q):
    """q = point at angle ang along the great circle from p with initial unit tangent d."""
    c, s = np.cos(ang), np.sin(ang)
    for k in range(3):
        q[k] = c * p[k] + s * d[k]
    r = np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2])
    for k in range(3):
        q[k] /= r


@njit(cache=True, inline="always")
def _logmean(a, b):
    if abs(a - b) <= 1e-9 * (a + b):
        return 0.5 * (a + b)
    return (a - b) / np.log(a / b)


@njit(cache=True)
def excluded(P, ph, head, nxt, nbr, cen, rad, stamp, sid, stack, q, e1, hq, tq, parent, alpha, rq, nd):
    """Index of a point (other than parent) in q's exclusion zone of half-width alpha (h_q + h_j) / 2 (frame e1,
    q x e1; box for nd = 4, hexagon for nd = 6), or -1 if it is empty. Lengths are angles (radians on the unit
    sphere): hq, ph, rq, rad."""
    e20 = q[1] * e1[2] - q[2] * e1[1]
    e21 = q[2] * e1[0] - q[0] * e1[2]
    e22 = q[0] * e1[1] - q[1] * e1[0]
    hx = np.sqrt(3.0) / 2.0
    stack[0] = tq
    stamp[tq] = sid
    top = 1
    pos = 0
    while pos < top:
        t = stack[pos]
        pos += 1
        j = head[t]
        while j >= 0:
            if j != parent:
                d0, d1, d2 = P[j, 0] - q[0], P[j, 1] - q[1], P[j, 2] - q[2]
                lim = alpha * 0.5 * (hq + ph[j])
                x = d0 * e1[0] + d1 * e1[1] + d2 * e1[2]
                if abs(x) < lim:
                    y = d0 * e20 + d1 * e21 + d2 * e22
                    if nd == 4:
                        if abs(y) < lim:
                            return j
                    elif abs(0.5 * x + hx * y) < lim and abs(-0.5 * x + hx * y) < lim:
                        return j
            j = nxt[j]
        for k in range(3):
            nt = nbr[t, k]
            if nt >= 0 and stamp[nt] != sid:
                stamp[nt] = sid
                c0, c1, c2 = cen[nt, 0] - q[0], cen[nt, 1] - q[1], cen[nt, 2] - q[2]
                if np.sqrt(c0 * c0 + c1 * c1 + c2 * c2) <= rq + rad[nt]:
                    stack[top] = nt
                    top += 1
    return -1


@njit(cache=True)
def _insert(P, ph, ptri, head, nxt, n, q, hq, tq):
    P[n, 0], P[n, 1], P[n, 2] = q[0], q[1], q[2]
    ph[n] = hq
    ptri[n] = tq
    nxt[n] = head[tq]
    head[tq] = n


@njit(cache=True)
def _run_queue(Pb, Tb, nbr, ub, lhb, cen, rad, P, ph, ptri, head, nxt, stamp, sid, stack, start, n,
               alpha, rfac, midpoint, seam, alpha_seam, nd):
    """Process the FIFO (= points start .. n-1, growing) until empty; returns (n, sid, full).

    seam > 0: a step blocked by a point j whose distance ahead along the step, D = (x_j - p) . d, is at least
    seam * s (a gap left between colliding fronts) is replaced by a point at D / 2, accepted with the smaller
    exclusion alpha_seam (an L-inf off-centre, Remacle 2013 §5): gaps up to (1 + alpha) h become two of D / 2."""
    cap = len(P)
    w = np.empty(3)
    d1 = np.empty(3)
    d = np.empty(3)
    dm = np.empty(3)
    q = np.empty(3)
    m = np.empty(3)
    e1 = np.empty(3)
    p = np.empty(3)
    i = start
    while i < n:
        p[0], p[1], p[2] = P[i, 0], P[i, 1], P[i, 2]
        t = ptri[i]
        hp = ph[i]
        bary(Pb, Tb, t, p, w)
        field_at(Pb, Tb, ub, lhb, t, p, w, d1, nd)
        d20 = p[1] * d1[2] - p[2] * d1[1]
        d21 = p[2] * d1[0] - p[0] * d1[2]
        d22 = p[0] * d1[1] - p[1] * d1[0]
        for k in range(nd):
            ck = np.cos(2.0 * np.pi * k / nd)
            sk = np.sin(2.0 * np.pi * k / nd)
            d[0] = ck * d1[0] + sk * d20
            d[1] = ck * d1[1] + sk * d21
            d[2] = ck * d1[2] + sk * d22
            s = hp
            ok = True
            if midpoint:
                # field direction seen at the half step, transported back: turn d by that angle
                _step(p, d, 0.5 * s, m)
                tm = walk(Pb, Tb, nbr, t, m)
                if tm >= 0:
                    bary(Pb, Tb, tm, m, w)
                    field_at(Pb, Tb, ub, lhb, tm, m, w, dm, nd)
                    c, sn = np.cos(0.5 * s), np.sin(0.5 * s)
                    tt0 = -sn * p[0] + c * d[0]           # d transported to m along the arc
                    tt1 = -sn * p[1] + c * d[1]
                    tt2 = -sn * p[2] + c * d[2]
                    n0 = m[1] * tt2 - m[2] * tt1          # m x tt
                    n1 = m[2] * tt0 - m[0] * tt2
                    n2 = m[0] * tt1 - m[1] * tt0
                    a = dm[0] * tt0 + dm[1] * tt1 + dm[2] * tt2
                    b = dm[0] * n0 + dm[1] * n1 + dm[2] * n2
                    psi = nd * np.arctan2(b, a)
                    phi = np.arctan2(np.sin(psi), np.cos(psi)) / nd   # the field branch nearest to tt
                    cph, sph = np.cos(phi), np.sin(phi)
                    pd0 = p[1] * d[2] - p[2] * d[1]
                    pd1 = p[2] * d[0] - p[0] * d[2]
                    pd2 = p[0] * d[1] - p[1] * d[0]
                    d[0] = cph * d[0] + sph * pd0
                    d[1] = cph * d[1] + sph * pd1
                    d[2] = cph * d[2] + sph * pd2
            tq = t
            hq = hp
            for it in range(3):
                _step(p, d, s, q)
                tq = walk(Pb, Tb, nbr, t, q)
                if tq < 0:
                    ok = False
                    break
                bary(Pb, Tb, tq, q, w)
                hq = field_at(Pb, Tb, ub, lhb, tq, q, w, e1, nd)
                if it < 2:
                    s = _logmean(hp, hq)
            if not ok:
                continue
            sid += 1
            j = excluded(P, ph, head, nxt, nbr, cen, rad, stamp, sid, stack, q, e1, hq, tq, i, alpha, rfac * hq, nd)
            if j >= 0:
                if seam <= 0.0:
                    continue
                dd = (P[j, 0] - p[0]) * d[0] + (P[j, 1] - p[1]) * d[1] + (P[j, 2] - p[2]) * d[2]
                if dd < seam * s:
                    continue
                _step(p, d, 0.5 * dd, q)
                tq = walk(Pb, Tb, nbr, t, q)
                if tq < 0:
                    continue
                bary(Pb, Tb, tq, q, w)
                hq = field_at(Pb, Tb, ub, lhb, tq, q, w, e1, nd)
                sid += 1
                if excluded(P, ph, head, nxt, nbr, cen, rad, stamp, sid, stack, q, e1, hq, tq, i, alpha_seam,
                            rfac * hq, nd) >= 0:
                    continue
            if n >= cap:
                return n, sid, True
            _insert(P, ph, ptri, head, nxt, n, q, hq, tq)
            n += 1
        i += 1
    return n, sid, False


@njit(cache=True)
def pack(Pb, Tb, nbr, ub, lhb, cen, rad, P, ph, ptri, n0, fill_nodes, fill_tri, alpha, rfac, midpoint, seam,
         alpha_seam, nd, q0=0):
    """Run the packing. P[:n0], ph[:n0], ptri[:n0] hold the seeds (fixed coast nodes or one seed); only P[q0:n0]
    start fronts (P[:q0] are fixed points that exclude but do not grow, B112). Returns (n, full, n_gap) with P[:n] the
    points; n_gap = points inserted by gap filling (each then grows its own front)."""
    n_tb = len(Tb)
    head = np.full(n_tb, -1, dtype=np.int64)
    nxt = np.full(len(P), -1, dtype=np.int64)
    stamp = np.zeros(n_tb, dtype=np.int64)
    stack = np.empty(n_tb, dtype=np.int64)
    for i in range(n0):
        nxt[i] = head[ptri[i]]
        head[ptri[i]] = i
    sid = 0
    n, sid, full = _run_queue(Pb, Tb, nbr, ub, lhb, cen, rad, P, ph, ptri, head, nxt, stamp, sid, stack, q0, n0,
                              alpha, rfac, midpoint, seam, alpha_seam, nd)
    n_gap = 0
    q = np.empty(3)
    e1 = np.empty(3)
    w = np.empty(3)
    for g in range(len(fill_nodes)):
        if full:
            break
        v = fill_nodes[g]
        t = fill_tri[g]
        q[0], q[1], q[2] = Pb[v, 0], Pb[v, 1], Pb[v, 2]
        bary(Pb, Tb, t, q, w)
        hq = field_at(Pb, Tb, ub, lhb, t, q, w, e1, nd)
        sid += 1
        if excluded(P, ph, head, nxt, nbr, cen, rad, stamp, sid, stack, q, e1, hq, t, -1, alpha, rfac * hq, nd) >= 0:
            continue
        if n >= len(P):
            full = True
            break
        _insert(P, ph, ptri, head, nxt, n, q, hq, t)
        n += 1
        n_gap += 1
        n, sid, full = _run_queue(Pb, Tb, nbr, ub, lhb, cen, rad, P, ph, ptri, head, nxt, stamp, sid, stack,
                                  n - 1, n, alpha, rfac, midpoint, seam, alpha_seam, nd)
    return n, full, n_gap
