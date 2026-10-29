"""Lp-CVT energy and its exact gradient on the sphere (Levy & Liu 2010, §3-4, Appendix A-B).

F(X) = sum over owner cells a of  w_a * integral over Vor(a) of ||M_a (y - x_a)||_p^p dA / c_p

- Vor(a) is the spherical Voronoi cell (dual of the convex hull = sphere_delaunay). It is split per Delaunay triangle
  t = (a, b, c) into two signed integration triangles (a, m_ab, C_t) and (a, C_t, m_ca): m_ab = direction of a + b
  (on the bisector great circle of a, b), C_t = direction of (b - a) x (c - a) (the circumcentre). Summed over the
  triangles around a they give the cell exactly (signed areas handle obtuse triangles).
- Everything is mapped to the plane tangent at x_a by the gnomonic projection G_a(y) = y / (y . x_a) - x_a. It maps
  great circles to lines, so the pieces of neighbouring triangles tile the projected cell exactly, and it ignores
  the length of y (no normalisation of a + b or of the cross product is needed).
- M_a = (R / h_a) [e1_a; e2_a]: (e1_a, e2_a) is the cross-field frame of a (frozen 3D vectors), h_a the size. The
  L_p norm with p = 8 approximates L_inf, whose minimisers are square cells of side h aligned with the frame.
- w_a = (h_a / h_ref)^2 makes the integral an area integral in units of h_ref^2 (cell size ~ h follows from it);
  c_p = 2 (1/2)^p / (p + 1) is the integral over the unit square, so an ideal cell of side h_a contributes w_a.

Integral of x^p over the triangle (0, w1, w2) (Lasserre-Avrachenkov polar form, Levy-Liu Appendix A, n = 2):
|T| 2 / ((p+1)(p+2)) sum_{alpha=0..p} w1^alpha w2^(p-alpha); |T| = det(w1, w2) / 2 in units of h_a^2.

The gradient follows the chain rule of Levy-Liu Appendix B through the gnomonic map and the two Voronoi-vertex
constructions (here closed-form vector derivatives instead of their bisector-plane systems):
  P = y / s - a, s = y . a:   g_y = g_P / s - a (g_P . y) / s^2,   g_a += -y (g_P . y) / s^2 - g_P
  y = (b - a) x (c - a):      g_a += (b - c) x g_y,  g_b += (c - a) x g_y,  g_c += (a - b) x g_y
Only owners with own[a] contribute (coast nodes are fixed and their cells, which reach over land, are not counted).
"""
import numpy as np
from numba import njit, prange

from meshom.delaunay import _find_edge, _flip, _incircle, _vertex_triangle, adjacency, sphere_delaunay


def unit_square_integral(p: int) -> float:
    """c_p = integral of |x|^p + |y|^p over [-1/2, 1/2]^2."""
    return 2.0 * 0.5**p / (p + 1.0)


@njit(cache=True, inline="always")
def _sp(x, y, p):
    """S = sum_{alpha=0..p} x^alpha y^(p-alpha) and its partial derivatives (recurrence S_q = y S_{q-1} + x^q)."""
    s, dsx, dsy, xq = 1.0, 0.0, 0.0, 1.0
    for q in range(1, p + 1):
        dsx = y * dsx + q * xq
        dsy = s + y * dsy
        xq = xq * x
        s = y * s + xq
    return s, dsx, dsy


@njit(cache=True, inline="always")
def _tri(x1, y1, x2, y2, p):
    """E = D (S(x1,x2) + S(y1,y2)), D = det / 2, and dE / d(x1, y1, x2, y2)."""
    d = 0.5 * (x1 * y2 - y1 * x2)
    sx, sx1, sx2 = _sp(x1, x2, p)
    sy, sy1, sy2 = _sp(y1, y2, p)
    s = sx + sy
    return d * s, 0.5 * y2 * s + d * sx1, -0.5 * x2 * s + d * sy1, -0.5 * y1 * s + d * sx2, 0.5 * x1 * s + d * sy2


@njit(cache=True, inline="always")
def _proj(yx, yy, yz, ax, ay, az, f1x, f1y, f1z, f2x, f2y, f2z, ih):
    """Gnomonic image of direction y at a in frame units: s = y . a, w = ih (e1 . P, e2 . P), P = y / s - a."""
    s = yx * ax + yy * ay + yz * az
    qx, qy, qz = yx / s - ax, yy / s - ay, yz / s - az
    return s, ih * (f1x * qx + f1y * qy + f1z * qz), ih * (f2x * qx + f2y * qy + f2z * qz)


@njit(cache=True, inline="always")
def _back(gwx, gwy, yx, yy, yz, s, ax, ay, az, f1x, f1y, f1z, f2x, f2y, f2z, ih):
    """Gradient w.r.t. y and the direct part w.r.t. a, from the gradient w.r.t. w = (wx, wy)."""
    px = ih * (gwx * f1x + gwy * f2x)
    py = ih * (gwx * f1y + gwy * f2y)
    pz = ih * (gwx * f1z + gwy * f2z)
    r = (px * yx + py * yy + pz * yz) / (s * s)
    return (px / s - ax * r, py / s - ay * r, pz / s - az * r,
            -yx * r - px, -yy * r - py, -yz * r - pz)


@njit(parallel=True, cache=True)
def _corner_terms(P, tris, own, e1, e2, inv_h, wgt, p, et, gt):
    """Per triangle t and corner k (owner a = tris[t,k]): energy et[t,k] and gradient gt[t,k,m,:] w.r.t. corner m."""
    kp = 2.0 / ((p + 1.0) * (p + 2.0))
    for t in prange(len(tris)):
        for k in range(3):
            ia, ib, ic = tris[t, k], tris[t, (k + 1) % 3], tris[t, (k + 2) % 3]
            kb, kc = (k + 1) % 3, (k + 2) % 3
            for m in range(3):
                for d in range(3):
                    gt[t, k, m, d] = 0.0
            et[t, k] = 0.0
            if not own[ia]:
                continue
            ax, ay, az = P[ia, 0], P[ia, 1], P[ia, 2]
            bx, by, bz = P[ib, 0], P[ib, 1], P[ib, 2]
            cx, cy, cz = P[ic, 0], P[ic, 1], P[ic, 2]
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            f1x, f1y, f1z = e1[ia, 0], e1[ia, 1], e1[ia, 2]
            f2x, f2y, f2z = e2[ia, 0], e2[ia, 1], e2[ia, 2]
            ih = inv_h[ia]
            # y1 = a + b (edge midpoint direction), y2 = (b - a) x (c - a) (circumcentre), y3 = a + c
            y1x, y1y, y1z = ax + bx, ay + by, az + bz
            y2x, y2y, y2z = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            y3x, y3y, y3z = ax + cx, ay + cy, az + cz
            s1, w1x, w1y = _proj(y1x, y1y, y1z, ax, ay, az, f1x, f1y, f1z, f2x, f2y, f2z, ih)
            s2, w2x, w2y = _proj(y2x, y2y, y2z, ax, ay, az, f1x, f1y, f1z, f2x, f2y, f2z, ih)
            s3, w3x, w3y = _proj(y3x, y3y, y3z, ax, ay, az, f1x, f1y, f1z, f2x, f2y, f2z, ih)
            ea, g1x, g1y, g2x, g2y = _tri(w1x, w1y, w2x, w2y, p)
            eb, h2x, h2y, h3x, h3y = _tri(w2x, w2y, w3x, w3y, p)
            c = wgt[ia] * kp
            et[t, k] = c * (ea + eb)
            q1x, q1y, q1z, a1x, a1y, a1z = _back(c * g1x, c * g1y, y1x, y1y, y1z, s1, ax, ay, az,
                                                 f1x, f1y, f1z, f2x, f2y, f2z, ih)
            q2x, q2y, q2z, a2x, a2y, a2z = _back(c * (g2x + h2x), c * (g2y + h2y), y2x, y2y, y2z, s2, ax, ay, az,
                                                 f1x, f1y, f1z, f2x, f2y, f2z, ih)
            q3x, q3y, q3z, a3x, a3y, a3z = _back(c * h3x, c * h3y, y3x, y3y, y3z, s3, ax, ay, az,
                                                 f1x, f1y, f1z, f2x, f2y, f2z, ih)
            # y1 = a + b, y3 = a + c
            gax = a1x + a2x + a3x + q1x + q3x
            gay = a1y + a2y + a3y + q1y + q3y
            gaz = a1z + a2z + a3z + q1z + q3z
            gbx, gby, gbz = q1x, q1y, q1z
            gcx, gcy, gcz = q3x, q3y, q3z
            # y2 = (b - a) x (c - a): g_a += (b - c) x g, g_b += (c - a) x g, g_c += (a - b) x g
            rx, ry, rz = bx - cx, by - cy, bz - cz
            gax += ry * q2z - rz * q2y
            gay += rz * q2x - rx * q2z
            gaz += rx * q2y - ry * q2x
            rx, ry, rz = cx - ax, cy - ay, cz - az
            gbx += ry * q2z - rz * q2y
            gby += rz * q2x - rx * q2z
            gbz += rx * q2y - ry * q2x
            rx, ry, rz = ax - bx, ay - by, az - bz
            gcx += ry * q2z - rz * q2y
            gcy += rz * q2x - rx * q2z
            gcz += rx * q2y - ry * q2x
            gt[t, k, k, 0], gt[t, k, k, 1], gt[t, k, k, 2] = gax, gay, gaz
            gt[t, k, kb, 0], gt[t, k, kb, 1], gt[t, k, kb, 2] = gbx, gby, gbz
            gt[t, k, kc, 0], gt[t, k, kc, 1], gt[t, k, kc, 2] = gcx, gcy, gcz


@njit(cache=True)
def _scatter(tris, gt, n):
    g = np.zeros((n, 3))
    for t in range(len(tris)):
        for k in range(3):
            for m in range(3):
                i = tris[t, m]
                g[i, 0] += gt[t, k, m, 0]
                g[i, 1] += gt[t, k, m, 1]
                g[i, 2] += gt[t, k, m, 2]
    return g


def lp_energy(P: np.ndarray, tris: np.ndarray, own: np.ndarray, e1: np.ndarray, e2: np.ndarray, inv_h: np.ndarray,
              wgt: np.ndarray, p: int) -> tuple[float, np.ndarray, np.ndarray]:
    """(F, dF/dP (N,3) in R^3, per-node cell energy (N,)). inv_h = R / h_a; wgt = (h_a / h_ref)^2 / c_p."""
    t_n = len(tris)
    et = np.empty((t_n, 3))
    gt = np.empty((t_n, 3, 3, 3))
    _corner_terms(P, tris, own, e1, e2, inv_h, wgt, int(p), et, gt)
    cell = np.bincount(tris.ravel(), et.ravel(), minlength=len(P))
    return float(cell.sum()), _scatter(tris, gt, len(P)), cell


# ------------------------------------------------------------------------------------------- Delaunay maintenance

@njit(cache=True)
def _lawson(P, tris, nbr, vt, max_flips):
    """Lawson flips on a valid closed CCW sphere triangulation until every edge is locally Delaunay (= the convex
    hull). Returns the number of flips, or -1 when max_flips or the stack is exhausted (caller falls back to Qhull)."""
    t_n = len(tris)
    cap = 2 * t_n + 4 * max_flips + 16
    sa = np.empty(cap, np.int64)
    sb = np.empty(cap, np.int64)
    top = 0
    for t in range(t_n):
        for i in range(3):
            if nbr[t, i] > t:
                sa[top] = tris[t, i]
                sb[top] = tris[t, (i + 1) % 3]
                top += 1
    flips = 0
    while top > 0:
        top -= 1
        p, q = sa[top], sb[top]
        t, k = _find_edge(tris, nbr, vt, p, q)
        if t < 0:
            continue
        u = nbr[t, k]
        c = tris[t, (k + 2) % 3]
        d = -1
        for kk in range(3):
            if tris[u, kk] == q and tris[u, (kk + 1) % 3] == p:
                d = tris[u, (kk + 2) % 3]
        if _incircle(P, p, q, c, d) == 1:
            c, d = _flip(tris, nbr, vt, t, k)
            flips += 1
            if flips > max_flips or top + 4 > cap:
                return -1
            sa[top], sb[top] = q, c
            sa[top + 1], sb[top + 1] = c, p
            sa[top + 2], sb[top + 2] = p, d
            sa[top + 3], sb[top + 3] = d, q
            top += 4
    return flips


def delaunay_update(P: np.ndarray, state: tuple | None) -> tuple[tuple, bool]:
    """Delaunay triangulation of P (closed sphere) as state = (tris, nbr, vt): Lawson flips from the state of nearby
    positions when all its triangles are still CCW, else Qhull. Returns (state, used_qhull); state[0] is tris."""
    if state is not None:
        tris = state[0]
        a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
        if (np.einsum("ij,ij->i", a, np.cross(b, c)) > 0.0).all():
            t, nb, vt = (x.copy() for x in state)
            if _lawson(P, t, nb, vt, max(1000, len(t) // 4)) >= 0:
                return (t, nb, vt), False
    tris = sphere_delaunay(P)
    return (tris, adjacency(tris), _vertex_triangle(tris, len(P))), True
