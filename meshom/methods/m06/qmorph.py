"""Q-Morph kernels (Numba): indirect advancing-front quad formation over a triangle mesh on the unit sphere.

Owen, Staten, Canann & Saigal 1999 (IJNME 44:1317-1340). Written from the paper's
description.

State during the advance:
- The *region*: triangles not yet converted (tris, nbr, talive). Its boundary half-edges (nbr = -1) are the front:
  the triangles are CCW, so the region lies on the left of every front edge a -> b. Coast edges start as the front.
- Elements made so far: quads, and triangles where the front changes its edge count (out of the region).
- emap: half-edge key u * NN + v -> 3 t + i of the alive triangle holding u -> v; re-registered whenever a triangle
  changes and verified on lookup, so stale keys are harmless.
- A binary heap of front edges, key = 3 (level + PEN fails) + 2 - (number of ends in state 1): lowest level first,
  within a level the edges with both ends reusing a neighbour front edge first (Owen's state order 11, 10/01, 00).

Per front edge a -> b (region on its left):
1. The region sectors at a (from a->b CCW to the previous front edge x->a) and at b (from b->a CW to the next front
   edge b->y) give the front angles theta_a, theta_b.
2. Side at a: theta_a < STATE1 (3 pi / 4) -> the previous front edge (a, x). Otherwise k = round(theta / (pi/2)) >= 2
   elements meet at a and the side direction is theta / k from a->b (the bisector for k = 2; or the cross-field
   direction nearest to it). The side is the region edge nearest to that direction if within side_eps; else the edge
   opposite a in the wedge holding the direction is swapped; else that edge is split on the ray. A side edge longer
   than split_long h is split at h. Same at b, mirrored.
3. New top nodes (not yet on any front, not coast) are moved to distance side_len h along the side direction (step
   halved until no region triangle around them inverts).
4. Top edge c-d: |cd| < contract h -> one triangle (a, b, c or d) (the front loses an edge); |cd| > expand h -> the
   top edge is split at its midpoint m and the pentagon (a, b, d, m, c) becomes a quad and a triangle (the front gains
   an edge); else c-d is recovered by swaps and the quad (a, b, d, c) is formed. The triangles inside are deleted with
   their interior nodes; elements outside the angle bounds are refused.
5. The new front edges and their neighbours are pushed with fresh keys; a refused edge is pushed back with a level
   penalty, at most max_fail times. Triangles never reached stay as they are (mixed output).
"""
import numpy as np
from numba import njit, types
from numba.typed import Dict

BIG = np.int64(1) << np.int64(40)
MAXS = 64            # longest node star / sector walked
PEN = 2              # level penalty per refusal

# prm indices (float array)
P_STATE1, P_EPS, P_SPLIT_LONG, P_SIDE_LEN, P_EXPAND, P_CONTRACT, P_QMIN, P_QMAX, P_TMIN, P_TMAX, P_FIELD_TOL, \
    P_MOVE, P_USE_FIELD, P_MAX_FAIL, P_TQ_TOL, P_SNAP, P_GAP2, P_QLONG = range(18)
N_PRM = 18


# ------------------------------------------------------------------------------------------------- geometry

@njit(cache=True, inline="always")
def _orient(P, a, b, c):
    ux, uy, uz = P[b, 0] - P[a, 0], P[b, 1] - P[a, 1], P[b, 2] - P[a, 2]
    vx, vy, vz = P[c, 0] - P[a, 0], P[c, 1] - P[a, 1], P[c, 2] - P[a, 2]
    return (uy * vz - uz * vy) * P[a, 0] + (uz * vx - ux * vz) * P[a, 1] + (ux * vy - uy * vx) * P[a, 2]


@njit(cache=True, inline="always")
def _d2(P, a, b):
    return (P[a, 0] - P[b, 0]) ** 2 + (P[a, 1] - P[b, 1]) ** 2 + (P[a, 2] - P[b, 2]) ** 2


@njit(cache=True)
def _tq(P, a, b, c):
    """orient / largest squared edge: > 0 for a CCW triangle (0.87 equilateral), -> 0 for slivers."""
    return _orient(P, a, b, c) / max(_d2(P, a, b), _d2(P, b, c), _d2(P, c, a), 1e-300)


@njit(cache=True)
def _ang(P, a, p, q):
    """CCW angle at node a from direction a->p to a->q (seen from outside), in (-pi, pi]."""
    ax, ay, az = P[a, 0], P[a, 1], P[a, 2]
    ux, uy, uz = P[p, 0] - ax, P[p, 1] - ay, P[p, 2] - az
    vx, vy, vz = P[q, 0] - ax, P[q, 1] - ay, P[q, 2] - az
    du = ux * ax + uy * ay + uz * az
    dv = vx * ax + vy * ay + vz * az
    ux, uy, uz = ux - du * ax, uy - du * ay, uz - du * az
    vx, vy, vz = vx - dv * ax, vy - dv * ay, vz - dv * az
    s = (uy * vz - uz * vy) * ax + (uz * vx - ux * vz) * ay + (ux * vy - uy * vx) * az
    return np.arctan2(s, ux * vx + uy * vy + uz * vz)


@njit(cache=True)
def _arc(P, a, b):
    return 2.0 * np.arcsin(min(1.0, 0.5 * np.sqrt(_d2(P, a, b))))


@njit(cache=True)
def _rot_dir(P, a, p, phi):
    """Unit tangent at a: the direction a->p turned by phi CCW (seen from outside)."""
    ax, ay, az = P[a, 0], P[a, 1], P[a, 2]
    ux, uy, uz = P[p, 0] - ax, P[p, 1] - ay, P[p, 2] - az
    du = ux * ax + uy * ay + uz * az
    ux, uy, uz = ux - du * ax, uy - du * ay, uz - du * az
    n = np.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / n, uy / n, uz / n
    wx, wy, wz = ay * uz - az * uy, az * ux - ax * uz, ax * uy - ay * ux          # a x u: u turned by +90 deg
    c, s = np.cos(phi), np.sin(phi)
    return c * ux + s * wx, c * uy + s * wy, c * uz + s * wz


@njit(cache=True)
def _geo_target(P, a, p, phi, L, out):
    """Point at arc L from a along a->p turned by phi."""
    wx, wy, wz = _rot_dir(P, a, p, phi)
    c, s = np.cos(L), np.sin(L)
    out[0] = c * P[a, 0] + s * wx
    out[1] = c * P[a, 1] + s * wy
    out[2] = c * P[a, 2] + s * wz


@njit(cache=True)
def _field_angle(P, U, v, p, sign):
    """Angle of the cross-field direction U[v] measured from v->p (CCW if sign = 1, CW if sign = -1), in [0, pi/2)."""
    ax, ay, az = P[v, 0], P[v, 1], P[v, 2]
    ux, uy, uz = P[p, 0] - ax, P[p, 1] - ay, P[p, 2] - az
    du = ux * ax + uy * ay + uz * az
    ux, uy, uz = ux - du * ax, uy - du * ay, uz - du * az
    fx, fy, fz = U[v, 0], U[v, 1], U[v, 2]
    s = (uy * fz - uz * fy) * ax + (uz * fx - ux * fz) * ay + (ux * fy - uy * fx) * az
    a = sign * np.arctan2(s, ux * fx + uy * fy + uz * fz)
    q = 0.5 * np.pi
    return a - q * np.floor(a / q)


# ------------------------------------------------------------------------------------------------- topology

@njit(cache=True, inline="always")
def _idx(tris, t, v):
    for k in range(3):
        if tris[t, k] == v:
            return k
    return -1


@njit(cache=True)
def _reg(tris, emap, vt, NN, t):
    for i in range(3):
        u = tris[t, i]
        emap[u * NN + tris[t, (i + 1) % 3]] = 3 * t + i
        vt[u] = t


@njit(cache=True)
def _he(tris, talive, emap, NN, u, v):
    """Slot 3 t + i of the alive half-edge u -> v, or -1."""
    s = emap.get(u * NN + v, -1)
    if s < 0:
        return -1
    t, i = s // 3, s % 3
    if talive[t] and tris[t, i] == u and tris[t, (i + 1) % 3] == v:
        return s
    return -1


@njit(cache=True)
def _has_edge(tris, talive, emap, NN, u, v):
    return _he(tris, talive, emap, NN, u, v) >= 0 or _he(tris, talive, emap, NN, v, u) >= 0


@njit(cache=True)
def _setnbr(tris, nbr, t, u, v, new):
    """In triangle t, the neighbour across half-edge u -> v becomes new."""
    for k in range(3):
        if tris[t, k] == u and tris[t, (k + 1) % 3] == v:
            nbr[t, k] = new
            return


@njit(cache=True)
def _vtri(tris, talive, vt, v):
    t = vt[v]
    if t >= 0 and talive[t] and _idx(tris, t, v) >= 0:
        return t
    return -1


@njit(cache=True)
def _flip(P, tris, nbr, talive, vt, emap, NN, t, i, tol):
    """Swap the diagonal of the two triangles across half-edge i of t; False if not convex enough or p-q exists."""
    t2 = nbr[t, i]
    if t2 < 0:
        return False
    u, v, p = tris[t, i], tris[t, (i + 1) % 3], tris[t, (i + 2) % 3]
    j = -1
    for k in range(3):
        if tris[t2, k] == v and tris[t2, (k + 1) % 3] == u:
            j = k
    if j < 0:
        return False
    q = tris[t2, (j + 2) % 3]
    if p == q or _has_edge(tris, talive, emap, NN, p, q):
        return False
    if _tq(P, u, q, p) <= tol or _tq(P, v, p, q) <= tol:
        return False
    n_vp, n_pu = nbr[t, (i + 1) % 3], nbr[t, (i + 2) % 3]
    n_uq, n_qv = nbr[t2, (j + 1) % 3], nbr[t2, (j + 2) % 3]
    tris[t, 0], tris[t, 1], tris[t, 2] = u, q, p
    nbr[t, 0], nbr[t, 1], nbr[t, 2] = n_uq, t2, n_pu
    tris[t2, 0], tris[t2, 1], tris[t2, 2] = v, p, q
    nbr[t2, 0], nbr[t2, 1], nbr[t2, 2] = n_vp, t, n_qv
    if n_uq >= 0:
        _setnbr(tris, nbr, n_uq, q, u, t)
    if n_vp >= 0:
        _setnbr(tris, nbr, n_vp, p, v, t2)
    _reg(tris, emap, vt, NN, t)
    _reg(tris, emap, vt, NN, t2)
    return True


@njit(cache=True)
def _split(P, hv, U, level, tris, nbr, talive, vt, emap, NN, cnt, t, i, X, tol):
    """Split the interior half-edge i of t (u -> v) at the unit point X; returns the new node or -1."""
    t2 = nbr[t, i]
    if t2 < 0 or cnt[0] >= P.shape[0] or cnt[1] + 2 > tris.shape[0]:
        return -1
    u, v, p = tris[t, i], tris[t, (i + 1) % 3], tris[t, (i + 2) % 3]
    j = -1
    for k in range(3):
        if tris[t2, k] == v and tris[t2, (k + 1) % 3] == u:
            j = k
    if j < 0:
        return -1
    q = tris[t2, (j + 2) % 3]
    s = cnt[0]
    P[s, 0], P[s, 1], P[s, 2] = X[0], X[1], X[2]
    if _tq(P, u, s, p) <= tol or _tq(P, s, v, p) <= tol or _tq(P, v, s, q) <= tol or _tq(P, s, u, q) <= tol:
        return -1
    cnt[0] += 1
    du, dv = np.sqrt(_d2(P, s, u)), np.sqrt(_d2(P, s, v))
    w = du / max(du + dv, 1e-300)
    hv[s] = (1.0 - w) * hv[u] + w * hv[v]
    near = u if w < 0.5 else v
    d = U[near, 0] * X[0] + U[near, 1] * X[1] + U[near, 2] * X[2]
    fx, fy, fz = U[near, 0] - d * X[0], U[near, 1] - d * X[1], U[near, 2] - d * X[2]
    fn = np.sqrt(fx * fx + fy * fy + fz * fz)
    U[s, 0], U[s, 1], U[s, 2] = fx / fn, fy / fn, fz / fn
    level[s] = BIG
    t3, t4 = cnt[1], cnt[1] + 1
    cnt[1] += 2
    talive[t3] = True
    talive[t4] = True
    n_vp, n_pu = nbr[t, (i + 1) % 3], nbr[t, (i + 2) % 3]
    n_uq, n_qv = nbr[t2, (j + 1) % 3], nbr[t2, (j + 2) % 3]
    tris[t, 0], tris[t, 1], tris[t, 2] = u, s, p
    nbr[t, 0], nbr[t, 1], nbr[t, 2] = t4, t3, n_pu
    tris[t3, 0], tris[t3, 1], tris[t3, 2] = s, v, p
    nbr[t3, 0], nbr[t3, 1], nbr[t3, 2] = t2, n_vp, t
    tris[t2, 0], tris[t2, 1], tris[t2, 2] = v, s, q
    nbr[t2, 0], nbr[t2, 1], nbr[t2, 2] = t3, t4, n_qv
    tris[t4, 0], tris[t4, 1], tris[t4, 2] = s, u, q
    nbr[t4, 0], nbr[t4, 1], nbr[t4, 2] = t, n_uq, t2
    if n_vp >= 0:
        _setnbr(tris, nbr, n_vp, p, v, t3)
    if n_uq >= 0:
        _setnbr(tris, nbr, n_uq, q, u, t4)
    for tt in (t, t2, t3, t4):
        _reg(tris, emap, vt, NN, tt)
    return s


@njit(cache=True)
def _incircle(P, a, b, c, d):
    """> 0 when d lies inside the circumcircle of the CCW triangle (a, b, c) on the sphere."""
    ux, uy, uz = P[b, 0] - P[a, 0], P[b, 1] - P[a, 1], P[b, 2] - P[a, 2]
    vx, vy, vz = P[c, 0] - P[a, 0], P[c, 1] - P[a, 1], P[c, 2] - P[a, 2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    return (P[d, 0] - P[a, 0]) * nx + (P[d, 1] - P[a, 1]) * ny + (P[d, 2] - P[a, 2]) * nz


@njit(cache=True)
def _delaunay_fix(P, tris, nbr, talive, vt, emap, NN, seeds, ns, trs, tol, stats):
    """Lawson flips of the region edges around the nodes seeds[:ns] (undoes slivers left by refused operations)."""
    for _ in range(4):
        nfl = 0
        for q in range(ns):
            v = seeds[q]
            t0 = _vtri(tris, talive, vt, v)
            if t0 < 0:
                continue
            k = abs(_star(tris, nbr, t0, v, trs))
            for j in range(k):
                t = trs[j]
                if not talive[t] or _idx(tris, t, v) < 0:
                    continue
                for i in range(3):
                    nb = nbr[t, i]
                    if nb < 0:
                        continue
                    u, w, p = tris[t, i], tris[t, (i + 1) % 3], tris[t, (i + 2) % 3]
                    z = tris[nb, (_idx(tris, nb, u) + 1) % 3]
                    scale = _d2(P, u, w) ** 1.5
                    if _incircle(P, u, w, p, z) > 1e-9 * scale:
                        if _flip(P, tris, nbr, talive, vt, emap, NN, t, i, tol):
                            nfl += 1
                            stats[0] += 1
                            break
        if nfl == 0:
            return


# ------------------------------------------------------------------------------------------------- walks

@njit(cache=True)
def _sector_a(P, tris, nbr, t, i, nodes, angs, trs):
    """Region sector at a = tris[t, i] from a->b CCW to the previous front edge x->a. nodes[0] = b, nodes[k] = x,
    angs = cumulative CCW angles, trs[j] = triangle (a, nodes[j], nodes[j+1]). Returns k, or -1."""
    a = tris[t, i]
    nodes[0] = tris[t, (i + 1) % 3]
    angs[0] = 0.0
    cur, ci, k = t, i, 0
    while k < MAXS:
        c = tris[cur, (ci + 2) % 3]
        trs[k] = cur
        angs[k + 1] = angs[k] + _ang(P, a, nodes[k], c)
        nodes[k + 1] = c
        k += 1
        nb = nbr[cur, (ci + 2) % 3]
        if nb < 0:
            return k
        if nb == t:
            return -1
        cur = nb
        ci = _idx(tris, nb, a)
    return -1


@njit(cache=True)
def _sector_b(P, tris, nbr, t, i, nodes, angs, trs):
    """Region sector at b = tris[t, i+1] from b->a CW to the next front edge b->y. nodes[0] = a, nodes[k] = y,
    angs = cumulative CW angles, trs[j] = triangle (b, nodes[j+1], nodes[j]). Returns k, or -1."""
    b = tris[t, (i + 1) % 3]
    nodes[0] = tris[t, i]
    angs[0] = 0.0
    cur, bi, k = t, (i + 1) % 3, 0
    while k < MAXS:
        c = tris[cur, (bi + 1) % 3]
        trs[k] = cur
        angs[k + 1] = angs[k] + _ang(P, b, c, nodes[k])
        nodes[k + 1] = c
        k += 1
        nb = nbr[cur, bi]
        if nb < 0:
            return k
        if nb == t:
            return -1
        cur = nb
        bi = _idx(tris, nb, b)
    return -1


@njit(cache=True)
def _star(tris, nbr, t0, v, trs):
    """Triangles around v from t0: count if the star is closed, -count if open (both directions walked)."""
    k, cur = 0, t0
    while k < MAXS:
        trs[k] = cur
        k += 1
        nb = nbr[cur, (_idx(tris, cur, v) + 2) % 3]
        if nb == t0:
            return k
        if nb < 0:
            break
        cur = nb
    cur = t0
    while k < MAXS:
        nb = nbr[cur, _idx(tris, cur, v)]
        if nb < 0:
            break
        cur = nb
        trs[k] = cur
        k += 1
    return -k


@njit(cache=True)
def _move(P, tris, trs, k, v, X, tol):
    """Move node v toward X; the step is halved until every triangle in trs[:k] keeps _tq > tol. Returns the step."""
    ox, oy, oz = P[v, 0], P[v, 1], P[v, 2]
    lam = 1.0
    for _ in range(5):
        x, y, z = ox + lam * (X[0] - ox), oy + lam * (X[1] - oy), oz + lam * (X[2] - oz)
        n = np.sqrt(x * x + y * y + z * z)
        P[v, 0], P[v, 1], P[v, 2] = x / n, y / n, z / n
        ok = True
        for j in range(k):
            t = trs[j]
            if _tq(P, tris[t, 0], tris[t, 1], tris[t, 2]) <= tol:
                ok = False
                break
        if ok:
            return lam
        lam *= 0.5
    P[v, 0], P[v, 1], P[v, 2] = ox, oy, oz
    return 0.0


@njit(cache=True)
def _crosses(P, m, n, c, d):
    return (_orient(P, c, d, m) * _orient(P, c, d, n) < 0.0) and (_orient(P, m, n, c) * _orient(P, m, n, d) < 0.0)


@njit(cache=True)
def _recover(P, tris, nbr, talive, vt, emap, NN, c, d, trs, qa, qb, tol, stats):
    """Make c-d an edge of the region by swaps of the region edges it crosses (none may be a front edge)."""
    if _has_edge(tris, talive, emap, NN, c, d):
        return True
    t0 = _vtri(tris, talive, vt, c)
    if t0 < 0:
        return False
    k = abs(_star(tris, nbr, t0, c, trs))
    t = -1
    for j in range(k):
        tt = trs[j]
        ci = _idx(tris, tt, c)
        m, n = tris[tt, (ci + 1) % 3], tris[tt, (ci + 2) % 3]
        if _orient(P, c, m, d) > 0.0 and _orient(P, c, d, n) > 0.0:
            t = tt
            break
    if t < 0:
        return False
    ci = _idx(tris, t, c)
    m, n = tris[t, (ci + 1) % 3], tris[t, (ci + 2) % 3]
    nq = 0
    while True:                                   # crossed edges along the arc, m right / n left of c -> d
        if nq >= len(qa) // 2:
            return False
        qa[nq], qb[nq] = m, n
        nq += 1
        nb = nbr[t, _idx(tris, t, m)]
        if nb < 0:
            return False
        z = tris[nb, (_idx(tris, nb, m) + 1) % 3]
        if z == d:
            break
        o = _orient(P, c, d, z)
        if o > 0.0:
            n = z
        elif o < 0.0:
            m = z
        else:
            return False
        t = nb
    head, it = 0, 0
    while head < nq and it < 400:
        it += 1
        u, v = qa[head], qb[head]
        head += 1
        s = _he(tris, talive, emap, NN, u, v)
        if s < 0:
            continue
        t, i = s // 3, s % 3
        if nbr[t, i] < 0:
            return False
        if _flip(P, tris, nbr, talive, vt, emap, NN, t, i, tol):
            stats[0] += 1
            p, q = tris[t, 2], tris[t, 1]
            if (p == c and q == d) or (p == d and q == c):
                continue
            if _crosses(P, p, q, c, d):
                if nq >= len(qa):
                    return False
                qa[nq], qb[nq] = p, q
                nq += 1
        else:
            if nq >= len(qa):
                return False
            qa[nq], qb[nq] = u, v
            nq += 1
    return _has_edge(tris, talive, emap, NN, c, d)


@njit(cache=True)
def _collect(tris, nbr, talive, emap, NN, fixed, poly, npoly, R, maxR):
    """Region triangles inside the CCW polygon poly[:npoly] (its edges must be region edges); their count, or -1 if
    the polygon does not bound a disc of region triangles (front inside, wrong side, interior coast node)."""
    s = _he(tris, talive, emap, NN, poly[0], poly[1])
    if s < 0:
        return -1
    R[0] = s // 3
    nR, head, seen = 1, 0, 0
    while head < nR:
        r = R[head]
        head += 1
        for j in range(3):
            u, v = tris[r, j], tris[r, (j + 1) % 3]
            pe = -1
            for k in range(npoly):
                if poly[k] == u and poly[(k + 1) % npoly] == v:
                    pe = k
                elif poly[k] == v and poly[(k + 1) % npoly] == u:
                    return -1
            if pe >= 0:
                seen |= 1 << pe
                continue
            nb = nbr[r, j]
            if nb < 0:
                return -1
            new = True
            for q in range(nR):
                if R[q] == nb:
                    new = False
                    break
            if new:
                if nR >= maxR:
                    return -1
                R[nR] = nb
                nR += 1
    if seen != (1 << npoly) - 1:
        return -1
    n_int = 0                                      # disc check: nR = npoly - 2 + 2 * interior nodes
    for q in range(nR):
        for j in range(3):
            v = tris[R[q], j]
            inside = True
            for k in range(npoly):
                if poly[k] == v:
                    inside = False
            if not inside:
                continue
            for q2 in range(q):                    # count each node once
                if _idx(tris, R[q2], v) >= 0:
                    inside = False
                    break
            if inside:
                for j2 in range(j):
                    if tris[R[q], j2] == v:
                        inside = False
            if inside:
                if fixed[v]:
                    return -1
                n_int += 1
    if nR != npoly - 2 + 2 * n_int:
        return -1
    return nR


@njit(cache=True)
def _poly_ok(P, poly, npoly, amin, amax):
    for k in range(npoly):
        a = _ang(P, poly[k], poly[(k + 1) % npoly], poly[(k - 1) % npoly])
        if a < amin or a > amax:
            return False
    return True


@njit(cache=True)
def _poly_long(P, poly, npoly, lmax):
    for k in range(npoly):
        if _arc(P, poly[k], poly[(k + 1) % npoly]) > lmax:
            return True
    return False


@njit(cache=True)
def _poly_dev(P, poly, npoly, ideal):
    w = 0.0
    for k in range(npoly):
        a = _ang(P, poly[k], poly[(k + 1) % npoly], poly[(k - 1) % npoly])
        if a <= 0.0:
            return 10.0
        w = max(w, abs(a - ideal))
    return w


@njit(cache=True)
def _commit(tris, nbr, talive, vt, emap, NN, dead, R, nR, poly, npoly, fa, fb, nf):
    """Remove the triangles R[:nR] (the inside of poly): their interior nodes die, the region edges around become
    front edges (recorded in fa/fb from index nf). Returns the new nf."""
    for q in range(nR):
        talive[R[q]] = False
    for q in range(nR):
        r = R[q]
        for j in range(3):
            v = tris[r, j]
            onp = False
            for k in range(npoly):
                if poly[k] == v:
                    onp = True
            if not onp:
                dead[v] = True
            nb = nbr[r, j]
            if nb >= 0 and talive[nb]:
                u, w = tris[r, j], tris[r, (j + 1) % 3]
                _setnbr(tris, nbr, nb, w, u, -1)
                _reg(tris, emap, vt, NN, nb)
                if nf < len(fa):
                    fa[nf], fb[nf] = w, u
                    nf += 1
    return nf


# ------------------------------------------------------------------------------------------------- heap

@njit(cache=True)
def _hpush(hk, ha, hb, hn, key, a, b):
    i = hn[0]
    if i >= len(hk):
        return
    hk[i], ha[i], hb[i] = key, a, b
    hn[0] += 1
    while i > 0:
        p = (i - 1) // 2
        if hk[p] <= hk[i]:
            break
        hk[p], hk[i] = hk[i], hk[p]
        ha[p], ha[i] = ha[i], ha[p]
        hb[p], hb[i] = hb[i], hb[p]
        i = p


@njit(cache=True)
def _hpop(hk, ha, hb, hn):
    key, a, b = hk[0], ha[0], hb[0]
    hn[0] -= 1
    n = hn[0]
    hk[0], ha[0], hb[0] = hk[n], ha[n], hb[n]
    i = 0
    while True:
        l, r, m = 2 * i + 1, 2 * i + 2, i
        if l < n and hk[l] < hk[m]:
            m = l
        if r < n and hk[r] < hk[m]:
            m = r
        if m == i:
            break
        hk[m], hk[i] = hk[i], hk[m]
        ha[m], ha[i] = ha[i], ha[m]
        hb[m], hb[i] = hb[i], hb[m]
        i = m
    return key, a, b


# ------------------------------------------------------------------------------------------------- side edges

@njit(cache=True)
def _ray_gap(P, tris, nbr, nodes, angs, trs, k, v, o, psi, sign, maxd, hit):
    """Distance from v along the side direction psi (sector arrays of v) to the first front edge the ray meets
    inside the region, or 1e9 beyond maxd; hit[0], hit[1] = that edge's nodes."""
    m = 0
    while m < k - 1 and angs[m + 1] < psi:
        m += 1
    t = trs[m]
    vi = _idx(tris, t, v)
    r, l = tris[t, (vi + 1) % 3], tris[t, (vi + 2) % 3]     # right / left of the ray
    wx, wy, wz = _rot_dir(P, v, o, sign * psi)
    nx = P[v, 1] * wz - P[v, 2] * wy
    ny = P[v, 2] * wx - P[v, 0] * wz
    nz = P[v, 0] * wy - P[v, 1] * wx
    for _ in range(40):
        nb = nbr[t, _idx(tris, t, r)]
        if nb < 0:
            fr = P[r, 0] * nx + P[r, 1] * ny + P[r, 2] * nz
            fl = P[l, 0] * nx + P[l, 1] * ny + P[l, 2] * nz
            lam = fr / (fr - fl) if fr != fl else 0.5
            lam = min(1.0, max(0.0, lam))
            x = P[r, 0] + lam * (P[l, 0] - P[r, 0])
            y = P[r, 1] + lam * (P[l, 1] - P[r, 1])
            z = P[r, 2] + lam * (P[l, 2] - P[r, 2])
            nrm = np.sqrt(x * x + y * y + z * z)
            d2 = (x / nrm - P[v, 0]) ** 2 + (y / nrm - P[v, 1]) ** 2 + (z / nrm - P[v, 2]) ** 2
            hit[0], hit[1] = r, l
            return 2.0 * np.arcsin(min(1.0, 0.5 * np.sqrt(d2)))
        z = tris[nb, (_idx(tris, nb, r) + 1) % 3]
        if P[z, 0] * nx + P[z, 1] * ny + P[z, 2] * nz > 0.0:
            l = z
        else:
            r = z
        t = nb
        if min(_arc(P, v, r), _arc(P, v, l)) > maxd:
            return 1e9
    return 1e9


@njit(cache=True)
def _pick(nodes, angs, k, psi, excl):
    """Index j in 1..k of the sector edge nearest to angle psi (node excl skipped), or -1."""
    best, bj = 1e9, -1
    for j in range(1, k + 1):
        if nodes[j] == excl:
            continue
        d = abs(angs[j] - psi)
        if d < best:
            best, bj = d, j
    return bj


@njit(cache=True)
def _side(P, hv, U, level, tris, nbr, talive, vt, emap, NN, cnt, fixed, a, b, at_a, psi, excl, prm,
          nodes, angs, trs, X, stats):
    """Side edge at a (at_a) or b for the front edge a -> b, direction psi into the region (CCW from a->b at a, CW
    from b->a at b). Returns the far node or -1. May swap or split region edges."""
    eps, tol = prm[P_EPS], prm[P_TQ_TOL]
    v = a if at_a else b
    o = b if at_a else a
    w = -1
    for attempt in range(3):                      # 0: nearest edge; 1: after a swap; 2: split on the ray
        s = _he(tris, talive, emap, NN, a, b)
        if s < 0:
            return -1
        t, i = s // 3, s % 3
        k = _sector_a(P, tris, nbr, t, i, nodes, angs, trs) if at_a else \
            _sector_b(P, tris, nbr, t, i, nodes, angs, trs)
        if k < 0:
            return -1
        j = _pick(nodes, angs, k, psi, excl)
        if j < 0:
            return -1
        dev = abs(angs[j] - psi)
        if dev <= eps:
            w = nodes[j]
            break
        m = 0                                     # wedge holding psi
        while m < k - 1 and angs[m + 1] < psi:
            m += 1
        tw = trs[m]
        opp = (_idx(tris, tw, v) + 1) % 3         # half-edge opposite v
        if attempt == 0:
            if nbr[tw, opp] >= 0 and _flip(P, tris, nbr, talive, vt, emap, NN, tw, opp, tol):
                stats[0] += 1
            continue
        if attempt == 1 and nbr[tw, opp] >= 0:    # split the opposite edge where the ray from v at psi crosses it
            sg = 1.0 if at_a else -1.0
            wx, wy, wz = _rot_dir(P, v, o, sg * psi)
            nx = P[v, 1] * wz - P[v, 2] * wy
            ny = P[v, 2] * wx - P[v, 0] * wz
            nz = P[v, 0] * wy - P[v, 1] * wx
            p0, p1 = tris[tw, opp], tris[tw, (opp + 1) % 3]
            f0 = P[p0, 0] * nx + P[p0, 1] * ny + P[p0, 2] * nz
            f1 = P[p1, 0] * nx + P[p1, 1] * ny + P[p1, 2] * nz
            if f0 != f1:
                lam = min(0.8, max(0.2, f0 / (f0 - f1)))
                for q in range(3):
                    X[q] = P[p0, q] + lam * (P[p1, q] - P[p0, q])
                nrm = np.sqrt(X[0] ** 2 + X[1] ** 2 + X[2] ** 2)
                X[0], X[1], X[2] = X[0] / nrm, X[1] / nrm, X[2] / nrm
                w = _split(P, hv, U, level, tris, nbr, talive, vt, emap, NN, cnt, tw, opp, X, tol)
                if w >= 0:
                    stats[1] += 1
                    break
        if dev <= 2.0 * eps:                      # no swap or split possible: the nearest edge if not too far
            w = nodes[j]
            break
        if attempt == 2:
            return -1
    if w < 0:
        return -1
    # a long side edge is split at the target length
    L = np.sqrt(hv[v] * hv[w]) * prm[P_SIDE_LEN]
    lw = _arc(P, v, w)
    if lw > prm[P_SPLIT_LONG] * L:
        s = _he(tris, talive, emap, NN, v, w)
        if s >= 0 and nbr[s // 3, s % 3] >= 0:
            lam = L / lw
            for q in range(3):
                X[q] = P[v, q] + lam * (P[w, q] - P[v, q])
            nrm = np.sqrt(X[0] ** 2 + X[1] ** 2 + X[2] ** 2)
            X[0], X[1], X[2] = X[0] / nrm, X[1] / nrm, X[2] / nrm
            w2 = _split(P, hv, U, level, tris, nbr, talive, vt, emap, NN, cnt, s // 3, s % 3, X, tol)
            if w2 >= 0:
                stats[1] += 1
                w = w2
    return w


@njit(cache=True)
def _side_angle(P, U, v, o, theta, sign, use_field):
    """Side angle at v (from v->o, CCW if sign = 1 else CW) for a front angle theta >= STATE1: theta / k with
    k = round(theta / (pi/2)) >= 2, and the cross-field direction nearest to it inside (0, theta) with its distance."""
    kq = max(2, int(np.floor(theta / (0.5 * np.pi) + 0.5)))
    psi = theta / kq
    if not use_field:
        return psi, psi, 0.0
    f = _field_angle(P, U, v, o, sign)
    best, bd = psi, 1e9
    for r in range(5):
        cand = f + r * 0.5 * np.pi
        if 0.0 < cand < theta and abs(cand - psi) < bd:
            bd = abs(cand - psi)
            best = cand
    return psi, best, bd


# ------------------------------------------------------------------------------------------------- main loop

@njit(cache=True)
def _edge_key(P, tris, nbr, level, t, i, fails, NN, prm, nodes, angs, trs):
    a, b = tris[t, i], tris[t, (i + 1) % 3]
    ka = _sector_a(P, tris, nbr, t, i, nodes, angs, trs)
    s = 0
    if ka > 0 and angs[ka] < prm[P_STATE1]:
        s += 1
    kb = _sector_b(P, tris, nbr, t, i, nodes, angs, trs)
    if kb > 0 and angs[kb] < prm[P_STATE1]:
        s += 1
    lev = min(level[a], level[b])
    if lev >= BIG:
        lev = 0
    return 3 * (lev + PEN * fails.get(a * NN + b, 0)) + 2 - s


@njit(cache=True)
def _push_front(P, tris, nbr, talive, emap, level, fails, NN, prm, hk, ha, hb, hn, u, v, nodes, angs, trs):
    s = _he(tris, talive, emap, NN, u, v)
    if s < 0 or nbr[s // 3, s % 3] >= 0:
        return
    _hpush(hk, ha, hb, hn, _edge_key(P, tris, nbr, level, s // 3, s % 3, fails, NN, prm, nodes, angs, trs), u, v)


@njit(cache=True)
def advance(P, hv, U, fixed, level, tris, nbr, talive, n_nodes, n_tris, quads, n_quads, prm):
    """Run Q-Morph over the region (talive triangles). P/hv/U/fixed/level have spare rows for split nodes, tris/nbr/
    talive spare rows for split triangles, quads spare rows. hv: target edge length in radians.

    Returns (n_nodes, n_tris, n_quads, otris, okind, olev, qlev, dead, stats); okind: 1 contraction (both sides at
    one node), 2 contraction (short top edge), 3 expansion, 4 isolated triangle; olev/qlev: front level."""
    NN = P.shape[0]
    tol = prm[P_TQ_TOL]
    cnt = np.zeros(2, dtype=np.int64)
    cnt[0], cnt[1] = n_nodes, n_tris
    otris = np.empty((max(1024, n_tris // 2), 3), dtype=np.int64)
    okind = np.zeros(len(otris), dtype=np.int64)
    olev = np.zeros(len(otris), dtype=np.int64)
    qlev = np.zeros(quads.shape[0], dtype=np.int64)
    n_ot = 0
    nq = n_quads
    dead = np.zeros(NN, dtype=np.bool_)
    vt = np.full(NN, -1, dtype=np.int64)
    emap = Dict.empty(key_type=types.int64, value_type=types.int64)
    fails = Dict.empty(key_type=types.int64, value_type=types.int64)
    for t in range(n_tris):
        if talive[t]:
            _reg(tris, emap, vt, NN, t)
    stats = np.zeros(13, dtype=np.int64)  # flips, splits, quads, tri_contract, tri_expand, tri_isolated, fails,
    #                                       moves, pops, stale, dropped, quad_refused, snaps
    fr = np.zeros(8, dtype=np.int64)       # refusal reasons: side, contract, recover top, expand, quad angle,
    #                                        quad region, isolated
    hit_a = np.zeros(2, dtype=np.int64)
    seeds = np.zeros(4, dtype=np.int64)
    hit_b = np.zeros(2, dtype=np.int64)
    nodes = np.empty(MAXS + 1, dtype=np.int64)
    angs = np.empty(MAXS + 1)
    trs = np.empty(MAXS, dtype=np.int64)
    na = np.empty(MAXS + 1, dtype=np.int64)
    aa = np.empty(MAXS + 1)
    ta = np.empty(MAXS, dtype=np.int64)
    nb_ = np.empty(MAXS + 1, dtype=np.int64)
    ab = np.empty(MAXS + 1)
    tb = np.empty(MAXS, dtype=np.int64)
    R = np.empty(64, dtype=np.int64)
    R2 = np.empty(64, dtype=np.int64)
    qa = np.empty(256, dtype=np.int64)
    qb = np.empty(256, dtype=np.int64)
    fa = np.empty(64, dtype=np.int64)
    fb = np.empty(64, dtype=np.int64)
    poly = np.empty(5, dtype=np.int64)
    poly2 = np.empty(5, dtype=np.int64)
    X = np.empty(3)
    Tc = np.empty(3)
    Td = np.empty(3)
    cap = 8 * n_tris + 1024
    hk = np.empty(cap, dtype=np.int64)
    ha = np.empty(cap, dtype=np.int64)
    hb = np.empty(cap, dtype=np.int64)
    hn = np.zeros(1, dtype=np.int64)
    for t in range(n_tris):
        if talive[t]:
            for i in range(3):
                if nbr[t, i] < 0:
                    _hpush(hk, ha, hb, hn, _edge_key(P, tris, nbr, level, t, i, fails, NN, prm, nodes, angs, trs),
                           tris[t, i], tris[t, (i + 1) % 3])
    max_fail = int(prm[P_MAX_FAIL])
    state1 = prm[P_STATE1]
    it = 0
    while hn[0] > 0 and it < 50 * n_tris + 1000:
        it += 1
        key, a, b = _hpop(hk, ha, hb, hn)
        stats[8] += 1
        c, d = -1, -1
        s = _he(tris, talive, emap, NN, a, b)
        if s < 0:
            continue
        t, i = s // 3, s % 3
        if nbr[t, i] >= 0:
            continue
        cur = _edge_key(P, tris, nbr, level, t, i, fails, NN, prm, nodes, angs, trs)
        if cur > key:
            stats[9] += 1
            _hpush(hk, ha, hb, hn, cur, a, b)
            continue
        lev = min(level[a], level[b])
        ka = _sector_a(P, tris, nbr, t, i, na, aa, ta)
        kb = _sector_b(P, tris, nbr, t, i, nb_, ab, tb)
        ok = ka > 0 and kb > 0
        nf = 0
        if ok:
            th_a, th_b = aa[ka], ab[kb]
            xa, yb = na[ka], nb_[kb]
            if ka == 1 and kb == 1:                                    # isolated triangle: emit it
                poly[0], poly[1], poly[2] = a, b, xa
                nR = _collect(tris, nbr, talive, emap, NN, fixed, poly, 3, R, 64)
                if nR > 0 and n_ot < len(otris):
                    nf = _commit(tris, nbr, talive, vt, emap, NN, dead, R, nR, poly, 3, fa, fb, nf)
                    otris[n_ot, 0], otris[n_ot, 1], otris[n_ot, 2] = a, b, xa
                    okind[n_ot], olev[n_ot] = 4, lev
                    n_ot += 1
                    stats[5] += 1
                else:
                    ok = False
                    fr[6] += 1
            else:
                # side directions and the gap to the next front along them (Q-Morph seams: a front closer than
                # snap h is joined directly; within gap2 h the next row takes half of the gap)
                psi_a, psi_b = th_a, th_b
                Da, Db = 1e9, 1e9
                use_f = prm[P_USE_FIELD] > 0.0
                if th_a >= state1:
                    psi_a, pfa, dva = _side_angle(P, U, a, b, th_a, 1.0, use_f)
                    use_f = use_f and dva < prm[P_FIELD_TOL]
                if th_b >= state1:
                    psi_b, pfb, dvb = _side_angle(P, U, b, a, th_b, -1.0, use_f)
                    use_f = use_f and dvb < prm[P_FIELD_TOL]
                if use_f:                                    # the field at both ends or at neither
                    if th_a >= state1:
                        psi_a = pfa
                    if th_b >= state1:
                        psi_b = pfb
                if th_a >= state1:
                    Da = _ray_gap(P, tris, nbr, na, aa, ta, ka, a, b, psi_a, 1.0, prm[P_GAP2] * hv[a], hit_a)
                if th_b >= state1:
                    Db = _ray_gap(P, tris, nbr, nb_, ab, tb, kb, b, a, psi_b, -1.0, prm[P_GAP2] * hv[b], hit_b)
                c, d = -1, -1
                if th_a < state1:
                    c = xa
                if th_b < state1:
                    d = yb
                snap_a = th_a >= state1 and Da < prm[P_SNAP] * hv[a]
                snap_b = th_b >= state1 and Db < prm[P_SNAP] * hv[b]
                if snap_a or snap_b:
                    # join the other front: the pair (c, d) with d -> c a front edge and the squarest quad
                    best = 1e9
                    bc, bd = -1, -1
                    for qa_i in range(2):
                        cc = hit_a[qa_i] if snap_a else c
                        if cc < 0 or cc == a or cc == b:
                            continue
                        for qb_i in range(2):
                            dd = hit_b[qb_i] if snap_b else d
                            if dd < 0 or dd == a or dd == b or dd == cc:
                                continue
                            sd = _he(tris, talive, emap, NN, dd, cc)
                            if sd < 0 or nbr[sd // 3, sd % 3] >= 0:
                                continue
                            poly[0], poly[1], poly[2], poly[3] = a, b, dd, cc
                            w = _poly_dev(P, poly, 4, 0.5 * np.pi)
                            if w < best:
                                best, bc, bd = w, cc, dd
                    if bc >= 0 and (not snap_a or _recover(P, tris, nbr, talive, vt, emap, NN, a, bc, trs, qa, qb,
                                                           tol, stats)) and \
                            (not snap_b or _recover(P, tris, nbr, talive, vt, emap, NN, b, bd, trs, qa, qb, tol,
                                                    stats)):
                        c, d = bc, bd
                        stats[12] += 1
                    else:                                    # each end on its own: the hit node nearest in angle
                        if snap_a:
                            e0 = abs(_ang(P, a, b, hit_a[0]) % (2.0 * np.pi) - psi_a)
                            e1 = abs(_ang(P, a, b, hit_a[1]) % (2.0 * np.pi) - psi_a)
                            cs = hit_a[0] if e0 <= e1 else hit_a[1]
                            if cs != a and cs != b and _recover(P, tris, nbr, talive, vt, emap, NN, a, cs, trs, qa,
                                                                qb, tol, stats):
                                c = cs
                        if snap_b:
                            e0 = abs((-_ang(P, b, a, hit_b[0])) % (2.0 * np.pi) - psi_b)
                            e1 = abs((-_ang(P, b, a, hit_b[1])) % (2.0 * np.pi) - psi_b)
                            ds = hit_b[0] if e0 <= e1 else hit_b[1]
                            if ds != a and ds != b and _recover(P, tris, nbr, talive, vt, emap, NN, b, ds, trs, qa,
                                                                qb, tol, stats):
                                d = ds
                if c < 0:
                    c = _side(P, hv, U, level, tris, nbr, talive, vt, emap, NN, cnt, fixed, a, b, True, psi_a,
                              -1, prm, nodes, angs, trs, X, stats)
                if d < 0:
                    d = _side(P, hv, U, level, tris, nbr, talive, vt, emap, NN, cnt, fixed, a, b, False, psi_b,
                              -1, prm, nodes, angs, trs, X, stats)
                if c < 0 or d < 0:
                    ok = False
                    fr[0] += 1
                if ok and c == d:                                       # try the other side choice once
                    if th_a >= state1:
                        c2 = _side(P, hv, U, level, tris, nbr, talive, vt, emap, NN, cnt, fixed, a, b, True,
                                   psi_a, d, prm, nodes, angs, trs, X, stats)
                        if c2 >= 0:
                            c = c2
                    if c == d and th_b >= state1:
                        d2 = _side(P, hv, U, level, tris, nbr, talive, vt, emap, NN, cnt, fixed, a, b, False,
                                   psi_b, c, prm, nodes, angs, trs, X, stats)
                        if d2 >= 0:
                            d = d2
                if ok:
                    c_new = level[c] >= BIG and not fixed[c]
                    d_new = level[d] >= BIG and not fixed[d]
                    # targets of new top nodes: side_len h along the side (half the gap near another front)
                    La = np.sqrt(hv[a] * hv[c]) * prm[P_SIDE_LEN]
                    if Da < prm[P_GAP2] * La:
                        La = min(La, 0.5 * Da)
                    Lb = np.sqrt(hv[b] * hv[d]) * prm[P_SIDE_LEN]
                    if Db < prm[P_GAP2] * Lb:
                        Lb = min(Lb, 0.5 * Db)
                    for q in range(3):
                        Tc[q], Td[q] = P[c, q], P[d, q]
                    if c_new:
                        _geo_target(P, a, b, psi_a, La, Tc)
                    if d_new:
                        _geo_target(P, b, a, -psi_b, Lb, Td)
                    if c == d and c_new:                                # one apex: the mean of both targets
                        for q in range(3):
                            Tc[q] = 0.5 * (Tc[q] + Td[q])
                        nrm = np.sqrt(Tc[0] ** 2 + Tc[1] ** 2 + Tc[2] ** 2)
                        for q in range(3):
                            Tc[q] /= nrm
                    if prm[P_MOVE] > 0.0:                               # move them (step halved against inversion)
                        for end in range(2):
                            v = c if end == 0 else d
                            if not (c_new if end == 0 else d_new) or (end == 1 and d == c):
                                continue
                            t0 = _vtri(tris, talive, vt, v)
                            if t0 >= 0:
                                k = _star(tris, nbr, t0, v, trs)
                                if k > 0 and _move(P, tris, trs, k, v, Tc if end == 0 else Td, tol) > 0.0:
                                    stats[7] += 1
                    dx, dy, dz = Tc[0] - Td[0], Tc[1] - Td[1], Tc[2] - Td[2]      # top length at the targets
                    lcd = 2.0 * np.arcsin(min(1.0, 0.5 * np.sqrt(dx * dx + dy * dy + dz * dz)))
                    htop = np.sqrt(hv[c] * hv[d])
                    mode = 0                                            # 0 quad, 1 contract, 2 expand
                    kind = 2
                    if c == d:
                        mode = 1
                        kind = 1
                    elif lcd < prm[P_CONTRACT] * htop and (c_new or d_new):
                        mode = 1
                        # apex: the better triangle (a, b, c) or (a, b, d)
                        poly[0], poly[1], poly[2] = a, b, c
                        poly2[0], poly2[1], poly2[2] = a, b, d
                        if _poly_dev(P, poly2, 3, np.pi / 3) < _poly_dev(P, poly, 3, np.pi / 3):
                            c = d
                        else:
                            d = c
                    elif lcd > prm[P_EXPAND] * htop and (c_new or d_new):
                        mode = 2
                    if mode == 1:
                        ok = _recover(P, tris, nbr, talive, vt, emap, NN, a, c, trs, qa, qb, tol, stats) and \
                            _recover(P, tris, nbr, talive, vt, emap, NN, b, c, trs, qa, qb, tol, stats)
                        poly[0], poly[1], poly[2] = a, b, c
                        if ok:
                            ok = _poly_ok(P, poly, 3, prm[P_TMIN], prm[P_TMAX])
                        nR = _collect(tris, nbr, talive, emap, NN, fixed, poly, 3, R, 64) if ok else -1
                        if nR > 0 and n_ot < len(otris):
                            nf = _commit(tris, nbr, talive, vt, emap, NN, dead, R, nR, poly, 3, fa, fb, nf)
                            otris[n_ot, 0], otris[n_ot, 1], otris[n_ot, 2] = a, b, c
                            okind[n_ot], olev[n_ot] = kind, lev
                            n_ot += 1
                            stats[3] += 1
                            if level[c] >= BIG:
                                level[c] = lev + 1
                        else:
                            ok = False
                            fr[1] += 1
                    else:
                        ok = _recover(P, tris, nbr, talive, vt, emap, NN, c, d, trs, qa, qb, tol, stats)
                        if not ok:
                            fr[2] += 1
                        m = -1
                        if ok and mode == 2:
                            s2 = _he(tris, talive, emap, NN, d, c)          # the inside half-edge d -> c
                            if s2 >= 0 and nbr[s2 // 3, s2 % 3] >= 0:
                                for q in range(3):
                                    X[q] = 0.5 * (P[c, q] + P[d, q])
                                nrm = np.sqrt(X[0] ** 2 + X[1] ** 2 + X[2] ** 2)
                                X[0], X[1], X[2] = X[0] / nrm, X[1] / nrm, X[2] / nrm
                                m = _split(P, hv, U, level, tris, nbr, talive, vt, emap, NN, cnt, s2 // 3, s2 % 3,
                                           X, tol)
                                if m >= 0:
                                    stats[1] += 1
                        if ok and m >= 0:
                            # pentagon (a, b, d, m, c): A = quad (a,b,d,m) + tri (a,m,c); B = quad (a,b,m,c) + tri (b,d,m)
                            poly[0], poly[1], poly[2], poly[3] = a, b, d, m
                            poly2[0], poly2[1], poly2[2] = a, m, c
                            wa = max(_poly_dev(P, poly, 4, 0.5 * np.pi), _poly_dev(P, poly2, 3, np.pi / 3))
                            poly[0], poly[1], poly[2], poly[3] = a, b, m, c
                            poly2[0], poly2[1], poly2[2] = b, d, m
                            wb = max(_poly_dev(P, poly, 4, 0.5 * np.pi), _poly_dev(P, poly2, 3, np.pi / 3))
                            if wa <= wb:
                                poly[0], poly[1], poly[2], poly[3] = a, b, d, m
                                poly2[0], poly2[1], poly2[2] = a, m, c
                                ok = _recover(P, tris, nbr, talive, vt, emap, NN, a, m, trs, qa, qb, tol, stats)
                            else:
                                ok = _recover(P, tris, nbr, talive, vt, emap, NN, b, m, trs, qa, qb, tol, stats)
                            if ok:
                                ok = _poly_ok(P, poly, 4, prm[P_QMIN], prm[P_QMAX]) and \
                                    _poly_ok(P, poly2, 3, prm[P_TMIN], prm[P_TMAX])
                            nR = _collect(tris, nbr, talive, emap, NN, fixed, poly, 4, R, 64) if ok else -1
                            nR2 = _collect(tris, nbr, talive, emap, NN, fixed, poly2, 3, R2, 64) if nR > 0 else -1
                            if nR2 > 0 and nq < quads.shape[0] and n_ot < len(otris):
                                nf = _commit(tris, nbr, talive, vt, emap, NN, dead, R, nR, poly, 4, fa, fb, nf)
                                nf = _commit(tris, nbr, talive, vt, emap, NN, dead, R2, nR2, poly2, 3, fa, fb, nf)
                                for q in range(4):
                                    quads[nq, q] = poly[q]
                                qlev[nq] = lev
                                nq += 1
                                otris[n_ot, 0], otris[n_ot, 1], otris[n_ot, 2] = poly2[0], poly2[1], poly2[2]
                                okind[n_ot], olev[n_ot] = 3, lev
                                n_ot += 1
                                stats[2] += 1
                                stats[4] += 1
                                for v in (c, d, m):
                                    if level[v] >= BIG:
                                        level[v] = lev + 1
                            else:
                                ok = False
                                fr[3] += 1
                        elif ok:
                            poly[0], poly[1], poly[2], poly[3] = a, b, d, c
                            ok = _poly_ok(P, poly, 4, prm[P_QMIN], prm[P_QMAX]) and not _poly_long(
                                P, poly, 4, prm[P_QLONG] * max(hv[a], hv[b], hv[c], hv[d]))
                            if not ok:
                                stats[11] += 1
                                fr[4] += 1
                            nR = _collect(tris, nbr, talive, emap, NN, fixed, poly, 4, R, 64) if ok else -1
                            if ok and nR < 0:
                                fr[5] += 1
                            if nR > 0 and nq < quads.shape[0]:
                                nf = _commit(tris, nbr, talive, vt, emap, NN, dead, R, nR, poly, 4, fa, fb, nf)
                                for q in range(4):
                                    quads[nq, q] = poly[q]
                                qlev[nq] = lev
                                nq += 1
                                stats[2] += 1
                                for v in (c, d):
                                    if level[v] >= BIG:
                                        level[v] = lev + 1
                            else:
                                ok = False
        if not ok:
            seeds[0], seeds[1], ns = a, b, 2
            if c >= 0:
                seeds[ns] = c
                ns += 1
            if d >= 0:
                seeds[ns] = d
                ns += 1
            _delaunay_fix(P, tris, nbr, talive, vt, emap, NN, seeds, ns, trs, tol, stats)
            stats[6] += 1
            kf = a * NN + b
            f = fails.get(kf, 0) + 1
            fails[kf] = f
            if f <= max_fail:
                _hpush(hk, ha, hb, hn, key + 3 * PEN, a, b)
            else:
                stats[10] += 1
            continue
        # push the new front edges and their front neighbours
        for e in range(nf):
            u, v = fa[e], fb[e]
            s = _he(tris, talive, emap, NN, u, v)
            if s < 0 or nbr[s // 3, s % 3] >= 0:
                continue
            _push_front(P, tris, nbr, talive, emap, level, fails, NN, prm, hk, ha, hb, hn, u, v, nodes, angs, trs)
            t2, i2 = s // 3, s % 3
            k = _sector_a(P, tris, nbr, t2, i2, nodes, angs, trs)
            if k > 0:
                _push_front(P, tris, nbr, talive, emap, level, fails, NN, prm, hk, ha, hb, hn, nodes[k], u,
                            na, aa, ta)
            s = _he(tris, talive, emap, NN, u, v)
            k = _sector_b(P, tris, nbr, s // 3, s % 3, nodes, angs, trs)
            if k > 0:
                _push_front(P, tris, nbr, talive, emap, level, fails, NN, prm, hk, ha, hb, hn, v, nodes[k],
                            na, aa, ta)
    return cnt[0], cnt[1], nq, otris[:n_ot].copy(), okind[:n_ot].copy(), olev[:n_ot].copy(), qlev[:nq].copy(), \
        dead, np.concatenate((stats, fr))
