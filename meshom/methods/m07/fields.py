"""Orientation (4-RoSy) and position (4-PoSy) fields of Instant Meshes (Jakob 2015 §3.2-3.5) on the unit sphere.

Representation (paper §3.2-3.3): vertex i of a hierarchy level has its position v_i (= normal n_i on the sphere), a
representative direction q_i (unit tangent; the cross is q_i rotated by k 90° about n_i) and a representative
position o_i in its tangent plane: a point of the local lattice o_i + rho_i (a q_i + b t_i), t_i = n_i x q_i, a, b in
Z, chosen as the lattice point nearest v_i. Everything is extrinsic (3D vectors compared directly), as in the paper
and the released code (github.com/wjakob/instant-meshes).

Orientation: nonlinear Gauss-Seidel, the running sum is merged with each neighbour after the brute-force match of
the 2 x 2 representatives, projected and normalised after every neighbour (eq. 3).
Or the shared `meshom.field.cross_field` ("cones": parallel in g/h^2 except at cones that follow rho_c) as a fixed
orientation; the coarse levels then get its restriction.

Position: the same sweep; for neighbour j the lattice points of i and of j around the edge midpoint are matched by a
4 x 4 search (the floor cell of the midpoint on each lattice), the running mean moves to the matched pair, and at the
end the result is rounded to the lattice point nearest v_i.

Per-vertex scale (the extension of this method; the paper and the released code use one global rho). rho_i =
h(v_i) / R. In the matching step of edge (i, j) both lattices use the edge scale rho_ij = (rho_i + rho_j) / 2
("mean", default) - a lattice of spacing rho_i extrapolated to the midpoint would sit off by |rho_i - rho_ij| per
step, which pulls the field toward the coarser side; with the common edge scale two neighbours agree when their
lattice points are one rho_ij apart, i.e. the lattice spacing follows h at the edge midpoints (first order in |grad
h|). The final rounding at i uses rho_i. Variant "own": each lattice with its own spacing (an unused switch of
QuadriFlow, github.com/hjwdzh/QuadriFlow). The orientation step does not see h unless the cones field is
used: a lattice of squares of size h with an h-blind orientation is not integrable where h varies, and the mismatch
comes out as position singularities (T-junctions: triangles and pentagons), the mechanism of paper §3.6 / Fig. 10.

Constraints (paper §3.7, coast = "edge brush" of the released batch mode): boundary vertices get CQ = the boundary
tangent (orientation, weight 1) and CO = v_i (position, weight 1): the component of o_i normal to CQ is set to that of
v_i, so one lattice line runs along the coast through the coast vertex. They are propagated up the hierarchy.
"""
import numpy as np
from numba import njit

from meshom.methods.m07.hierarchy import Level

TINY = 1e-300


# ------------------------------------------------------------------------------------------------ 3-vector helpers

@njit(cache=True, inline="always")
def _row(a, i):
    return (a[i, 0], a[i, 1], a[i, 2])


@njit(cache=True, inline="always")
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True, inline="always")
def _crs(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


@njit(cache=True, inline="always")
def _lin(s, a, r, b):
    """s a + r b."""
    return (s * a[0] + r * b[0], s * a[1] + r * b[1], s * a[2] + r * b[2])


@njit(cache=True, inline="always")
def _unit(a):
    m = np.sqrt(_dot(a, a))
    if m > TINY:
        return (a[0] / m, a[1] / m, a[2] / m)
    return a


@njit(cache=True, inline="always")
def _tangent(a, n):
    """a projected onto the plane normal to the unit n."""
    return _lin(1.0, a, -_dot(a, n), n)


@njit(cache=True, inline="always")
def _compat_orient(q0, n0, q1, n1):
    """Best-matching representatives (a of cross 0, b of cross 1, a.b > 0): the 2 x 2 search of the paper."""
    A0, A1 = q0, _crs(n0, q0)
    B0, B1 = q1, _crs(n1, q1)
    best, ia, ib = -1.0, 0, 0
    for i in range(2):
        a = A0 if i == 0 else A1
        for j in range(2):
            b = B0 if j == 0 else B1
            s = abs(_dot(a, b))
            if s > best:
                best, ia, ib = s, i, j
    a = A0 if ia == 0 else A1
    b = B0 if ib == 0 else B1
    if _dot(a, b) < 0.0:
        b = (-b[0], -b[1], -b[2])
    return a, b


@njit(cache=True, inline="always")
def _floor_idx(o, q, t, p, s):
    d = _lin(1.0, p, -1.0, o)
    return np.floor(_dot(q, d) / s), np.floor(_dot(t, d) / s)


@njit(cache=True, inline="always")
def _lattice(o, q, t, a, b, s):
    return (o[0] + s * (a * q[0] + b * t[0]), o[1] + s * (a * q[1] + b * t[1]), o[2] + s * (a * q[2] + b * t[2]))


@njit(cache=True, inline="always")
def _compat_pos(v0, q0, o0, s0, v1, q1, o1, s1):
    """Integer lattice coordinates (a0, b0), (a1, b1) of the closest pair of lattice points of the two vertices
    around the edge midpoint (4 x 4 search over the floor cells), and the squared distance."""
    t0, t1 = _crs(v0, q0), _crs(v1, q1)
    mid = _lin(0.5, v0, 0.5, v1)
    fa0, fb0 = _floor_idx(o0, q0, t0, mid, s0)
    fa1, fb1 = _floor_idx(o1, q1, t1, mid, s1)
    best = np.inf
    ra0, rb0, ra1, rb1 = 0.0, 0.0, 0.0, 0.0
    for i in range(4):
        a0, b0 = fa0 + (i & 1), fb0 + (i >> 1)
        p0 = _lattice(o0, q0, t0, a0, b0, s0)
        for j in range(4):
            a1, b1 = fa1 + (j & 1), fb1 + (j >> 1)
            p1 = _lattice(o1, q1, t1, a1, b1, s1)
            d = _lin(1.0, p0, -1.0, p1)
            c = _dot(d, d)
            if c < best:
                best, ra0, rb0, ra1, rb1 = c, a0, b0, a1, b1
    return ra0, rb0, ra1, rb1, best


@njit(cache=True, inline="always")
def _edge_scale(ri, rj, own):
    if own:
        return ri, rj
    s = 0.5 * (ri + rj)
    return s, s


# ------------------------------------------------------------------------------------------------ Gauss-Seidel

@njit(cache=True)
def _orient_sweeps(V, ptr, idx, w, Q, CQ, CQw, iters):
    for _ in range(iters):
        for i in range(len(V)):
            n = _row(V, i)
            s = _row(Q, i)
            ws = 0.0
            for k in range(ptr[i], ptr[i + 1]):
                j = idx[k]
                a, b = _compat_orient(s, n, _row(Q, j), _row(V, j))
                s = _unit(_tangent(_lin(ws, a, w[k], b), n))
                ws += w[k]
            if CQw[i] > 0.0:
                a, b = _compat_orient(s, n, _row(CQ, i), n)
                s = _unit(_tangent(_lin(1.0 - CQw[i], a, CQw[i], b), n))
            if ws > 0.0:
                Q[i, 0], Q[i, 1], Q[i, 2] = s


@njit(cache=True)
def _round_pos(o, q, t, v, s):
    d = _lin(1.0, v, -1.0, o)
    return _lattice(o, q, t, np.round(_dot(q, d) / s), np.round(_dot(t, d) / s), s)


@njit(cache=True)
def _position_sweeps(V, ptr, idx, w, Q, O, rho, CQ, CO, COw, iters, own):
    for _ in range(iters):
        for i in range(len(V)):
            v = _row(V, i)
            q = _unit(_row(Q, i))
            s = _row(O, i)
            ws = 0.0
            for k in range(ptr[i], ptr[i + 1]):
                j = idx[k]
                vj = _row(V, j)
                qj = _unit(_row(Q, j))
                s0, s1 = _edge_scale(rho[i], rho[j], own)
                a0, b0, a1, b1, _ = _compat_pos(v, q, s, s0, vj, qj, _row(O, j), s1)
                p0 = _lattice(s, q, _crs(v, q), a0, b0, s0)
                p1 = _lattice(_row(O, j), qj, _crs(vj, qj), a1, b1, s1)
                tot = ws + w[k]
                s = _lin(ws / tot, p0, w[k] / tot, p1)
                ws = tot
                s = _lin(1.0, s, -_dot(_lin(1.0, s, -1.0, v), v), v)           # back into the tangent plane at v
            if COw[i] > 0.0:
                cq = _row(CQ, i)
                d = _lin(1.0, _row(CO, i), -1.0, s)
                d = _lin(1.0, d, -_dot(cq, d), cq)
                s = _lin(1.0, s, COw[i], d)
                s = _lin(1.0, s, -_dot(_lin(1.0, s, -1.0, v), v), v)
            if ws > 0.0:
                O[i, 0], O[i, 1], O[i, 2] = _round_pos(s, q, _crs(v, q), v, rho[i])


@njit(cache=True)
def _edge_jumps(V, E, Q, O, rho, own):
    """Per edge (i, j): integer jump diff = (a_i - a_j, b_i - b_j) of the matched lattice points in the matched frame
    of i (0 = same output vertex, a unit vector = one output edge), and the matching error (squared distance)."""
    m = len(E)
    diff = np.empty((m, 2), dtype=np.int64)
    err = np.empty(m)
    for e in range(m):
        i, j = E[e, 0], E[e, 1]
        vi, vj = _row(V, i), _row(V, j)
        qa, qb = _compat_orient(_unit(_row(Q, i)), vi, _unit(_row(Q, j)), vj)
        s0, s1 = _edge_scale(rho[i], rho[j], own)
        a0, b0, a1, b1, c = _compat_pos(vi, qa, _row(O, i), s0, vj, qb, _row(O, j), s1)
        diff[e, 0] = np.int64(a0 - a1)
        diff[e, 1] = np.int64(b0 - b1)
        err[e] = c
    return diff, err


# ------------------------------------------------------------------------------------------------ hierarchy transfer

@njit(cache=True)
def _restrict_q(Q, children, Vc, out):
    for c in range(len(children)):
        i0, i1 = children[c, 0], children[c, 1]
        q = _row(Q, i0)
        if i1 >= 0:
            a, b = _compat_orient(q, _row(Vc, c), _row(Q, i1), _row(Vc, c))
            q = _lin(1.0, a, 1.0, b)
        q = _unit(_tangent(q, _row(Vc, c)))
        out[c, 0], out[c, 1], out[c, 2] = q


def _project(x: np.ndarray, V: np.ndarray, point: bool) -> np.ndarray:
    """Directions (point=False) or points (point=True) projected onto the tangent planes at V."""
    if point:
        return x - np.einsum("ij,ij->i", x - V, V)[:, None] * V
    y = x - np.einsum("ij,ij->i", x, V)[:, None] * V
    return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), TINY)


def restrict_orientation(levels: list[Level], Q0: np.ndarray) -> list[np.ndarray]:
    """Fine field Q0 -> one field per level (as in the paper: the matched sum of the children)."""
    out = [Q0]
    for lev in levels[1:]:
        q = np.empty((lev.n, 3))
        _restrict_q(out[-1], lev.children, lev.V, q)
        out.append(q)
    return out


def restrict_constraints(levels: list[Level], CQ: np.ndarray, CQw: np.ndarray, CO: np.ndarray,
                         COw: np.ndarray) -> list[tuple]:
    """Boundary constraints on every level: a coarse vertex is constrained if a child is; it takes the first
    constrained child's direction and the mean of the constrained children's points (projected)."""
    out = [(CQ, CQw, CO, COw)]
    for lev in levels[1:]:
        cq, cqw, co, cow = out[-1]
        ch = lev.children
        c0, c1 = ch[:, 0], np.where(ch[:, 1] >= 0, ch[:, 1], ch[:, 0])
        w0, w1 = cow[c0], cow[c1] * (ch[:, 1] >= 0)
        has = (w0 + w1) > 0
        first = np.where(w0 > 0, c0, c1)
        q = _project(cq[first], lev.V, point=False)
        p = (w0[:, None] * co[c0] + w1[:, None] * co[c1]) / np.maximum(w0 + w1, TINY)[:, None]
        p = _project(p, lev.V, point=True)
        hw = has.astype(np.float64)
        out.append((np.where(has[:, None], q, 0.0), hw, np.where(has[:, None], p, 0.0), hw))
    return out


def random_tangent(V: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    from meshom.geo import tangent_basis

    e1, e2 = tangent_basis(V)
    a = rng.uniform(0.0, 2.0 * np.pi, len(V))
    return np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2


# ------------------------------------------------------------------------------------------------ public steps

def orientation_field(levels: list[Level], constraints: list[tuple], iters: int = 6, fine_iters: int = 6,
                      seed: int = 0) -> list[np.ndarray]:
    """Instant Meshes extrinsic 4-RoSy field: random start on the coarsest level, `iters` Gauss-Seidel sweeps per
    level, copied down (projected) to the next finer level; `fine_iters` sweeps on level 0. Returns the field on every
    level (the coarse levels as restricted from the final fine field, as the paper's batch mode does before the
    position solve)."""
    rng = np.random.default_rng(seed)
    q = random_tangent(levels[-1].V, rng)
    for k in range(len(levels) - 1, -1, -1):
        lev = levels[k]
        cq, cqw, _, _ = constraints[k]
        _orient_sweeps(lev.V, lev.ptr, lev.idx, lev.w, q, cq, cqw, fine_iters if k == 0 else iters)
        if k > 0:
            q = _project(q[levels[k - 1].parent], levels[k - 1].V, point=False)
    return restrict_orientation(levels, q)


def position_field(levels: list[Level], Q: list[np.ndarray], rho: list[np.ndarray], constraints: list[tuple],
                   iters: int = 6, fine_iters: int = 6, own_scale: bool = False, seed: int = 0) -> np.ndarray:
    """4-PoSy position field with per-vertex scale rho (unit sphere), coarse to fine; returns O on level 0."""
    rng = np.random.default_rng(seed + 1)
    from meshom.geo import tangent_basis

    top = levels[-1]
    e1, e2 = tangent_basis(top.V)
    xy = rng.uniform(-1.0, 1.0, (top.n, 2)) * rho[-1][:, None]
    o = top.V + xy[:, :1] * e1 + xy[:, 1:] * e2
    for k in range(len(levels) - 1, -1, -1):
        lev = levels[k]
        cq, _, co, cow = constraints[k]
        _position_sweeps(lev.V, lev.ptr, lev.idx, lev.w, Q[k], o, rho[k], cq, co, cow,
                         fine_iters if k == 0 else iters, own_scale)
        if k > 0:
            o = _project(o[levels[k - 1].parent], levels[k - 1].V, point=True)
    return o


def edge_jumps(lev: Level, Q: np.ndarray, O: np.ndarray, rho: np.ndarray, own_scale: bool = False) -> tuple:
    """(diff (E,2) int64, err (E,)) for lev.edges: see _edge_jumps."""
    return _edge_jumps(lev.V, lev.edges, Q, O, rho, own_scale)


@njit(cache=True)
def _edge_frames(V, E, Q):
    """Per edge (i, j): rotation indices (ki, kj) in 0..3 of the matched representatives used by _edge_jumps: the
    lattice frame of the jump diff is q_i turned by ki x 90 deg about n_i, and it matches q_j turned by kj x 90 deg."""
    m = len(E)
    k = np.empty((m, 2), dtype=np.int64)
    for e in range(m):
        i, j = E[e, 0], E[e, 1]
        vi, vj = _row(V, i), _row(V, j)
        qi, qj = _unit(_row(Q, i)), _unit(_row(Q, j))
        A1, B1 = _crs(vi, qi), _crs(vj, qj)
        best, ia, ib = -1.0, 0, 0
        for a in range(2):
            x = qi if a == 0 else A1
            for b in range(2):
                y = qj if b == 0 else B1
                s = abs(_dot(x, y))
                if s > best:
                    best, ia, ib = s, a, b
        x = qi if ia == 0 else A1
        y = qj if ib == 0 else B1
        k[e, 0] = ia
        k[e, 1] = ib + (2 if _dot(x, y) < 0.0 else 0)
    return k


def edge_frames(lev: Level, Q: np.ndarray) -> np.ndarray:
    """(E,2) rotation indices of the frames in which edge_jumps expresses its integer jumps (for m08: jumps in a
    vertex's own frame are diff rotated back by -ki quarter turns)."""
    return _edge_frames(lev.V, lev.edges, Q)


def restrict_scale(levels: list[Level], rho0: np.ndarray) -> list[np.ndarray]:
    """rho on every level: area-weighted geometric mean of the children."""
    out = [rho0]
    for f, c in zip(levels[:-1], levels[1:]):
        out.append(np.exp(np.bincount(f.parent, f.A * np.log(out[-1]), minlength=c.n) / c.A))
    return out


def conformal_factor(mesh, h, u: np.ndarray) -> np.ndarray:
    """(N,) best-fit log scale f of a 4-direction field u on the triangle mesh: the field rotates, relative to the
    transport of g/h^2 (meshom.field.connection), by omega_ij along each edge; a lattice of spacing h e^f is
    integrable with it when omega_ij = flux of grad f through the edge (the conformal term of field.py with ln h ->
    f). f per triangle (dual vertex), whose flux through a primal edge is (f_R - f_L) / w_e (w_e = cotan weight =
    dual / primal length), from min sum_e w_e ((f_R - f_L) / w_e - omega_e)^2 (dual Laplacian with weights 1 / w_e,
    CG), then averaged to the vertices. For the "cones" field f is the size distortion of its cones (log
    singular at each cone: the trivial connection spreads the curvature, field.py); a uniform h with this field needs
    position singularities (T-junction dipoles) to stay at h."""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import cg

    from meshom.field import connection

    p, tris = mesh.points, mesh.tris
    edges, he_edge, he_sign, rho, (e1, e2) = connection(mesh, np.log(h(p)))
    phi = np.arctan2(np.einsum("ij,ij->i", u, e2), np.einsum("ij,ij->i", u, e1))
    i, j = edges[:, 0], edges[:, 1]
    om = np.angle(np.exp(4j * (phi[j] - phi[i] - rho))) / 4.0
    ne, nt = len(edges), len(tris)
    q = p[tris]
    w = np.zeros(ne)
    for k in range(3):
        a, b = q[:, (k + 1) % 3] - q[:, k], q[:, (k + 2) % 3] - q[:, k]
        cot = np.einsum("ij,ij->i", a, b) / np.linalg.norm(np.cross(a, b), axis=1)
        w += np.bincount(he_edge[:, (k + 1) % 3], 0.5 * cot, minlength=ne)
    w = np.maximum(w, 1e-3 * np.median(w))
    # half-edge (t, k) runs i -> j when he_sign = +1: t is left of the edge n0 -> n1, else right
    left = np.full(ne, -1, dtype=np.int64)
    right = np.full(ne, -1, dtype=np.int64)
    t_id = np.repeat(np.arange(nt), 3)
    s = he_sign.ravel() > 0
    left[he_edge.ravel()[s]] = t_id[s]
    right[he_edge.ravel()[~s]] = t_id[~s]
    inner = (left >= 0) & (right >= 0)
    L, R, wi, oi = left[inner], right[inner], 1.0 / w[inner], om[inner]
    A = coo_matrix((np.r_[wi, wi, -wi, -wi], (np.r_[L, R, L, R], np.r_[L, R, R, L])), shape=(nt, nt)).tocsr()
    b = np.bincount(R, oi, minlength=nt) - np.bincount(L, oi, minlength=nt)
    A = A + diags(np.full(nt, 1e-9 * A.diagonal().mean()))
    ft, _ = cg(A, b - b.mean(), rtol=1e-8, maxiter=5000)
    area = 0.5 * np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=1)
    fv = np.bincount(tris.ravel(), np.repeat(area * ft, 3), minlength=len(p)) / np.maximum(
        np.bincount(tris.ravel(), np.repeat(area, 3), minlength=len(p)), 1e-300)
    return fv
