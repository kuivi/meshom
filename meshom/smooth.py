"""Node smoothing of mixed triangle/quad meshes on the sphere: Jacobi sweeps with relaxation and a smart guard.

Every movable node gets a target from its star; the step relax * (target - x) is projected onto the tangent plane at
x and the node is re-normalised to the unit sphere. Fixed nodes never move (the caller fixes coast nodes).

kinds:
- "laplace": the mean of the edge neighbours (h-blind; flattens grading over many sweeps).
- "angle": Zhou & Shimada (2000) angle-based smoothing on the ring of edge neighbours, for mixed meshes (as Gmsh's
  quad-dominant smoothing, ring without quad diagonals). For an edge N-M, N's ring neighbours S_L, S_R
  on both sides of the edge give the angle S_L-M-S_R; the geodesic M->N, at its current length, is turned about M to
  split that angle in the ratio of the ideal angles at M between M->N and M->S (60 deg triangle, 45 deg square). The
  candidates of all two-sided edges are averaged with equal weights. On a regular quad grid this is a damped Laplacian
  (relax 1 ~ Laplace with relax 1/2); it is h-blind, so many sweeps flatten grading.
  (Turning M->N to bisect M's own edges instead, i.e. the quad's corner angle at M, is unstable under Jacobi on a
  regular grid: a shear mode at wavenumber pi/3 grows by 1.125 per sweep.)
- "size": as "angle", with the ideal angles and the length taken from h. Target edge lengths are lam * h_e (h_e = mean
  of the end values); the ideal angle at M is that of the side triangle with those edge lengths (law of cosines), or
  atan(l_NS / l_NM) for a quad (right angle at N); the candidate sits at distance lam * h_NM from M. A graded mesh is
  then a fixed point. (With the fixed 60/45 deg ratios of "angle" it is not: on 'disc' at scale 4 the fine disc
  expands, edge p50 / h inside it 1.00 -> 1.45 in 1000 sweeps; with the h ratios it stays at 1.02.) The node count
  fixes the global scale, so lam = sqrt(sum area / sum ideal area at h) (sqrt(3)/4 h^2 per triangle, h^2 per quad) is
  computed once per call: the smoother fits L ~ lam h, lam = 1 when the count matches h.

Smart guard: badness of an element = max over corners |angle - ideal| (spherical corner angles, ideal 60 / 90 deg; a
reflex or inverted corner counts > 90 deg). A move is put back when the worst element around the node gets worse and
exceeds tol_deg (default 45: triangle angles in [15, 105] and quad angles in [45, 135] are free; 135 is the quad
target gate). tol_deg <= 30 freezes r-adaptation on a Poisson-Delaunay mesh within ~10 sweeps: 59 % of its nodes
touch a triangle beyond 30 deg. Independently of that comparison, a move is put back when an element around the node
that had every corner below 180 deg gets one at or above it (inverted, or a non-convex quad): badness >= 180 -
ideal (decisions M3; the star's worst element can improve while another one breaks). Put-backs
repeat until every remaining move passes (monotone, so it terminates; at worst nothing moves).
"""
import numpy as np
from numba import njit, prange

from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh

KINDS = {"laplace": 0, "angle": 1, "size": 2}


def _elements(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """(T+Q,4) int64 corners (triangles padded with -1) and (T+Q,) corner counts, global element order."""
    t, q = mesh.n_tris, mesh.n_quads
    el = np.full((t + q, 4), -1, dtype=np.int64)
    el[:t, :3] = mesh.tris
    el[t:] = mesh.quads
    nv = np.r_[np.full(t, 3), np.full(q, 4)].astype(np.int64)
    return el, nv


@njit(cache=True)
def _incidence(el, nv, n):
    """CSR node -> (element, corner) incidences by counting sort: ptr (N+1,), inc_e, inc_k (element order)."""
    ptr = np.zeros(n + 1, dtype=np.int64)
    for e in range(len(el)):
        for k in range(nv[e]):
            ptr[el[e, k] + 1] += 1
    for i in range(n):
        ptr[i + 1] += ptr[i]
    fill = ptr[:-1].copy()
    inc_e = np.empty(ptr[n], dtype=np.int64)
    inc_k = np.empty(ptr[n], dtype=np.int64)
    for e in range(len(el)):
        for k in range(nv[e]):
            i = el[e, k]
            inc_e[fill[i]] = e
            inc_k[fill[i]] = k
            fill[i] += 1
    return ptr, inc_e, inc_k


def size_scale(mesh: Mesh, h) -> float:
    """lam = sqrt(sum area / sum ideal area at h): edge length of ideal elements at the current node count, over h.
    Element area = |vector area| (half the cross product of the diagonals), centroid = normalised corner mean."""
    p = mesh.points
    t, q = p[mesh.tris], p[mesh.quads]
    area = np.r_[np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1),
                 np.linalg.norm(np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1]), axis=1)] * 0.5 * R_EARTH_M ** 2
    c = np.vstack([t.sum(axis=1), q.sum(axis=1)])
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    w = np.r_[np.full(mesh.n_tris, np.sqrt(3.0) / 4.0), np.ones(mesh.n_quads)]
    return float(np.sqrt(area.sum() / np.sum(w * h(c) ** 2)))


@njit(cache=True, inline="always")
def _ideal(m):
    return np.pi / 3.0 if m == 3 else np.pi / 2.0


@njit(cache=True, inline="always")
def _side_angle(m, l_nm, l_ms, l_ns):
    """Ideal angle at M between M->N and M->S in the side element with target edge lengths l: triangle (N, M, S) by
    the law of cosines; quad with a right angle at N: atan(l_NS / l_NM). Unit lengths give 60 / 45 deg."""
    if m == 3:
        c = (l_nm * l_nm + l_ms * l_ms - l_ns * l_ns) / (2.0 * l_nm * l_ms)
        return np.arccos(min(1.0, max(-1.0, c)))
    return np.arctan2(l_ns, l_nm)


@njit(cache=True, inline="always")
def _tangent(P, a, f):
    """Unit initial direction of the geodesic a -> f (tangent plane at a), from the difference for accuracy."""
    d0, d1, d2 = P[f, 0] - P[a, 0], P[f, 1] - P[a, 1], P[f, 2] - P[a, 2]
    s = d0 * P[a, 0] + d1 * P[a, 1] + d2 * P[a, 2]
    d0, d1, d2 = d0 - s * P[a, 0], d1 - s * P[a, 1], d2 - s * P[a, 2]
    r = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    return d0 / r, d1 / r, d2 / r


@njit(cache=True, parallel=True)
def _propose(P, el, nv, ptr, inc_e, inc_k, movable, kind, bh, relax, out, moved):
    """Jacobi step: out[i] = node i moved by relax * (target - x) in its tangent plane, re-normalised; moved[i] set.
    bh: (N,) lam * h / R per node (angular target length), used by kind 2."""
    for i in prange(len(P)):
        x0, x1, x2 = P[i, 0], P[i, 1], P[i, 2]
        out[i, 0], out[i, 1], out[i, 2] = x0, x1, x2
        moved[i] = False
        if not movable[i]:
            continue
        s0, s1, s2 = 0.0, 0.0, 0.0
        cnt = 0
        for j in range(ptr[i], ptr[i + 1]):
            e, k = inc_e[j], inc_k[j]
            m = nv[e]
            a = el[e, (k + 1) % m]
            if kind == 0:
                b = el[e, (k + m - 1) % m]
                s0 += P[a, 0] + P[b, 0]
                s1 += P[a, 1] + P[b, 1]
                s2 += P[a, 2] + P[b, 2]
                cnt += 2
                continue
            # edge i -> a: e is on its left, its ring side is i's corner before i in e; the element with a as the
            # corner before i is on its right, its ring side is i's corner after i there
            sl = el[e, (k + m - 1) % m]
            sr, mr = -1, 0
            for j2 in range(ptr[i], ptr[i + 1]):
                e2, k2 = inc_e[j2], inc_k[j2]
                m2 = nv[e2]
                if el[e2, (k2 + m2 - 1) % m2] == a:
                    sr, mr = el[e2, (k2 + 1) % m2], m2
                    break
            if sr < 0:
                continue
            if kind == 1:
                al, ar = _side_angle(m, 1.0, 1.0, 1.0), _side_angle(mr, 1.0, 1.0, 1.0)
            else:
                l_na = 0.5 * (bh[i] + bh[a])
                al = _side_angle(m, l_na, 0.5 * (bh[a] + bh[sl]), 0.5 * (bh[i] + bh[sl]))
                ar = _side_angle(mr, l_na, 0.5 * (bh[a] + bh[sr]), 0.5 * (bh[i] + bh[sr]))
            ul0, ul1, ul2 = _tangent(P, a, sl)
            ur0, ur1, ur2 = _tangent(P, a, sr)
            ma0, ma1, ma2 = P[a, 0], P[a, 1], P[a, 2]
            w0, w1, w2 = ma1 * ul2 - ma2 * ul1, ma2 * ul0 - ma0 * ul2, ma0 * ul1 - ma1 * ul0   # u_L turned +90 deg
            cr = (ul1 * ur2 - ul2 * ur1) * ma0 + (ul2 * ur0 - ul0 * ur2) * ma1 + (ul0 * ur1 - ul1 * ur0) * ma2
            th = np.arctan2(cr, ul0 * ur0 + ul1 * ur1 + ul2 * ur2)
            if th < 0.0:
                th += 2.0 * np.pi                      # CCW angle u_L -> u_R, the side that holds N
            ph = th * al / (al + ar)
            c, s = np.cos(ph), np.sin(ph)
            d0, d1, d2 = c * ul0 + s * w0, c * ul1 + s * w1, c * ul2 + s * w2
            if kind == 1:
                q0, q1, q2 = ma1 * x2 - ma2 * x1, ma2 * x0 - ma0 * x2, ma0 * x1 - ma1 * x0
                beta = np.arctan2(np.sqrt(q0 * q0 + q1 * q1 + q2 * q2), ma0 * x0 + ma1 * x1 + ma2 * x2)
            else:
                beta = 0.5 * (bh[i] + bh[a])
            cb, sb = np.cos(beta), np.sin(beta)
            s0 += cb * ma0 + sb * d0
            s1 += cb * ma1 + sb * d1
            s2 += cb * ma2 + sb * d2
            cnt += 1
        if cnt == 0:
            continue
        t0, t1, t2 = relax * (s0 / cnt - x0), relax * (s1 / cnt - x1), relax * (s2 / cnt - x2)
        r = t0 * x0 + t1 * x1 + t2 * x2
        y0, y1, y2 = x0 + t0 - r * x0, x1 + t1 - r * x1, x2 + t2 - r * x2
        nn = np.sqrt(y0 * y0 + y1 * y1 + y2 * y2)
        out[i, 0], out[i, 1], out[i, 2] = y0 / nn, y1 / nn, y2 / nn
        moved[i] = True


@njit(cache=True, inline="always")
def _elem_bad(P, el, nv, e):
    """max over corners |spherical corner angle - ideal|, angle CCW in (0, 2 pi)."""
    m = nv[e]
    ideal = _ideal(m)
    worst = 0.0
    for k in range(m):
        p = el[e, k]
        u0, u1, u2 = _tangent(P, p, el[e, (k + 1) % m])
        v0, v1, v2 = _tangent(P, p, el[e, (k + m - 1) % m])
        cr = (u1 * v2 - u2 * v1) * P[p, 0] + (u2 * v0 - u0 * v2) * P[p, 1] + (u0 * v1 - u1 * v0) * P[p, 2]
        ang = np.arctan2(cr, u0 * v0 + u1 * v1 + u2 * v2)
        if ang < 0.0:
            ang += 2.0 * np.pi
        worst = max(worst, abs(ang - ideal))
    return worst


@njit(cache=True, parallel=True)
def _badness(P, el, nv, out):
    for e in prange(len(el)):
        out[e] = _elem_bad(P, el, nv, e)


@njit(cache=True)
def _rebad(P, el, nv, ptr, inc_e, rev, out):
    """Recompute out for the elements around the nodes flagged in rev (put back), and clear the flags."""
    for i in range(len(rev)):
        if rev[i]:
            for j in range(ptr[i], ptr[i + 1]):
                out[inc_e[j]] = _elem_bad(P, el, nv, inc_e[j])
            rev[i] = False


@njit(cache=True, parallel=True)
def _revert(P, Pn, nv, ptr, inc_e, bad_old, bad_new, moved, tol, rev):
    """Put back every moved node whose worst incident element got worse and exceeds tol, or whose incident element
    got a corner >= 180 deg that it had not (badness >= pi - ideal: inverted or non-convex); flagged in rev; returns
    how many."""
    cnt = 0
    for i in prange(len(P)):
        if not moved[i]:
            continue
        wo, wn = 0.0, 0.0
        broke = False
        for j in range(ptr[i], ptr[i + 1]):
            e = inc_e[j]
            wo = max(wo, bad_old[e])
            wn = max(wn, bad_new[e])
            lim = np.pi - _ideal(nv[e])
            if bad_new[e] >= lim and bad_old[e] < lim:
                broke = True
        if broke or wn > max(wo, tol):
            Pn[i, 0], Pn[i, 1], Pn[i, 2] = P[i, 0], P[i, 1], P[i, 2]
            moved[i] = False
            rev[i] = True
            cnt += 1
    return cnt


def smooth(mesh: Mesh, h, fixed: np.ndarray, iters: int = 10, kind: str = "angle", relax: float | None = None,
           smart: bool = True, tol_deg: float = 45.0) -> Mesh:
    """Copy of mesh with smoothed node positions (same connectivity).

    fixed: (N,) bool mask or an index array of nodes that never move. h is used by kind="size" only. relax defaults to
    0.5 for "laplace" (damps the checkerboard mode) and 1.0 for "angle" / "size" (self-damped).
    """
    n = mesh.n_nodes
    fixed = np.asarray(fixed)
    if fixed.dtype != bool:
        mask = np.zeros(n, dtype=bool)
        mask[fixed] = True
        fixed = mask
    movable = ~fixed
    code = KINDS[kind]
    relax = (0.5 if code == 0 else 1.0) if relax is None else relax
    tol = np.radians(tol_deg)
    P = np.ascontiguousarray(mesh.points, dtype=np.float64).copy()
    el, nv = _elements(mesh)
    ptr, inc_e, inc_k = _incidence(el, nv, n)
    lam = size_scale(mesh, h) if code == 2 else 1.0
    bh = np.zeros(n)
    bad = np.empty(len(el))
    bad_new = np.empty(len(el))
    moved = np.zeros(n, dtype=bool)
    rev = np.zeros(n, dtype=bool)
    if smart:
        _badness(P, el, nv, bad)
    for _ in range(iters):
        if code == 2:
            bh = lam * h(P) / R_EARTH_M
        Pn = np.empty_like(P)
        _propose(P, el, nv, ptr, inc_e, inc_k, movable, code, bh, relax, Pn, moved)
        if smart:
            _badness(Pn, el, nv, bad_new)
            while _revert(P, Pn, nv, ptr, inc_e, bad, bad_new, moved, tol, rev):
                _rebad(Pn, el, nv, ptr, inc_e, rev, bad_new)
            bad, bad_new = bad_new, bad
        P = Pn
    return Mesh(P, mesh.tris, mesh.quads, mesh.node_flag, mesh.tri_zone)       # nodes move only: the zone stays
