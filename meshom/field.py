"""Cross fields (n = 4) and asterisk fields (n = 6) on a triangle mesh of the sphere, built on the connection of the
conformal metric g^ = g / h^2.

Representation (Knöppel 2013, power vectors). Node i has the tangent basis (e1_i, e2_i) = geo.tangent_basis(p_i) and
z_i = |z_i| exp(i psi_i), psi_i = n phi_i; the n directions of the field are phi_i + k 2 pi / n in that basis.

Connection. rho_ij is the angle by which a direction measured in basis i is shifted when it is transported along edge
i -> j and measured in basis j: a field parallel along the edge has phi_j = phi_i + rho_ij. Two parts:

1. Levi-Civita of g (round sphere), exact along the great-circle arc: transport keeps the angle to the arc. With beta_i
   the angle (basis i) of the arc's direction at p_i toward p_j and gamma_j the angle (basis j) of the arc's direction
   at p_j continuing away from p_i:  alpha_ij = gamma_j - beta_i.  Around a triangle the alphas sum to its spherical
   excess (= integral of K dA) mod 2 pi.

2. Conformal term (g^ = exp(2u) g, u = -ln h; the connection 1-form gains the Hodge star of d ln h). Derivation: a grid
   of square cells of size h is a conformal map f with |f'| = 1/h and grid angle theta = -arg f'; log f' = -ln h -
   i theta is holomorphic, so Cauchy-Riemann gives  grad theta = J grad ln h  (J = +90° about the outward normal N;
   check: polar grid f = log z has h ~ r and theta = polar angle). Along a path with unit tangent t the grid turns by
        d theta / ds = (J grad ln h) . t = grad ln h . n_R,      n_R = t x N (right normal),
   so edge i -> j gets the extra angle  delta_ij = integral_{i->j} d ln h / d n_R ds  = the flux of grad ln h through
   the edge, to its right, discretised as  l_ij n_R . (mean P1 gradient of ln h over the edge's triangles).
   delta is antisymmetric; around a CCW triangle it sums to the outward flux = integral of Laplace ln h dA, so the
   holonomy of rho = alpha + delta over a triangle is integral (K + Laplace ln h) dA = integral K^ dA^ (curvature of
   g^): the field must carry index where rho_c = (2/pi)(K + Laplace ln h) is nonzero (+1/4 on the fine rim of a
   refinement zone, -1/4 over its ramp) and is free of singularities where g^ is flat. Without delta the field ignores h.

Solvers (`method`):

- "cones" (default): the integrable field. Charges per triangle in units of cones, (n / 2 pi) times: (1) K dA (the
  Levi-Civita part of the holonomy, diffuse), (2) Laplace ln h dA (the conformal part without the flux through fixed
  coast edges; concentrated on rims and ramps), (3) the coast line charge (turning of the fixed boundary tangents
  relative to the transport plus that flux, i.e. the coast's geodesic curvature in g^). Each is smoothed by a heat
  kernel of CONE_SMOOTH_CELLS cells (cancels triangle-scale noise of the discrete Laplace ln h); with a fixed coast,
  all charge within COAST_CONE_DISTANCE h of it moves to the nearest triangle of the shell at that distance (no cone in
  the coastal rows; in a strait narrower than twice that distance the cones go to its mouths). Each charge is then
  quantised on its own (charges of different spatial scale must not share leaves): round(sum q+) cones of +1/n and
  the rest of round(sum q) as -1/n, by a mass-balanced k-d partition (each leaf holds one unit; its cone goes to the
  heavy triangle nearest the leaf's weighted centre); rounding left over goes to the part with the largest residual.
  Several water components (a lake, a closed sea kept as its own loop): each triangle component is quantised on its
  own with its own total (its charge is an integer when its boundary is all fixed, and x below is solvable only with
  exactly that many cones in it), its coastal charge pushed to its own shell.
  Then the trivial connection (Crane, Desbrun, Schröder 2010): the smallest correction x on edges (min sum w_e x_e^2,
  w = cotan weights) such that rho + x has holonomy 2 pi k_t / n on every triangle: x = W^-1 d^T lambda with lambda
  from the triangle-graph Laplacian (weights 1/w_e; Neumann, grounded through free boundary edges), solved by CG with a
  smoothed-aggregation multigrid preconditioner; the power field is integrated exactly along a BFS forest, one tree
  per connected component rooted at its first fixed node (else the first corner of its first triangle) and anchored
  there. Separate fixed boundary pieces of a component come out turned by constant angles (their relative phase is
  not a triangle constraint); a harmonic g (vertex Laplacian, constant on each fixed piece) adds dg to x and removes
  them. The result is parallel in g^ except at the cones: its singularities follow rho_c and a mesh of square cells of
  size ~h can follow it (the size distortion of each cone stays local). Measured (factor 0.5): `disc` scale 3: rim +16,
  ramp -16, +8 elsewhere (a cube-like set); `band` scale 4: +75 on the fine edge, -76 over the band, 8 elsewhere;
  uniform sphere: a cube (n = 4), 12 cones (n = 6); `continent` scale 4/2: index sum 0, all cones >= 3 h from the
  coast.
- "smoothest": Knöppel's globally optimal field for the same connection, energy sum w_ij |z_j - r_ij z_i|^2,
  r_ij = exp(i n rho_ij), mass M = diag(A_i / h_i^2): without fixed nodes the eigenvector of the smallest eigenvalue
  of L z = lambda M z (inverse power iteration, one sparse LU); with fixed nodes the harmonic extension
  L_II z_I = -L_IB z_B (sparse LU; slow beyond ~3e5 nodes). On `disc` it makes only 2-4 of the ~20 rim pairs: the
  eigenvector nearly vanishes over the refined disc (|z| ~ 1e-7 there), because twisting costs less Dirichlet energy
  than 40 cones (~480 vs ~520-870 in units of the cone core energy, growing with resolution); MBO annealing from a
  random start gives ~10-11 pairs. Smoothest fields are therefore not integrable with h where h varies.

Fixed nodes (boundary_nodes) get the boundary tangent: the normalised sum of t^n of their boundary half-edges (half-
edges without twin), so a corner turning by a multiple of 2 pi / n is consistent. smooth_steps: MBO / Ginzburg-Landau
steps (M + tau L) z <- M z, z <- z / |z|, tau = (2 l^)^2 with l^ the median edge length in g^, fixed nodes
kept, CG solves. They smooth the field near the cones but also pull +- pairs together (10 steps on `disc` scale 4:
rim +17 -> +10), i.e. back toward the Dirichlet optimum.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, diags
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.sparse.linalg import LinearOperator, cg, splu
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M, tangent_basis
from meshom.mesh import Mesh
from meshom.size import SizeField


CONE_SMOOTH_CELLS = 0.5      # heat-kernel width (cells of the background mesh) applied to the g^ curvature charge
COAST_CONE_DISTANCE = 3.0    # no cone within this distance (units of h) of a fixed coast: its charge moves to this shell


@dataclass
class CrossField:
    mesh: Mesh               # the triangle mesh the field lives on
    u: np.ndarray            # (N,3) unit tangent vector per node; the n directions are u rotated by k 2 pi / n
    n: int                   # 4 (quads) or 6 (triangles)
    logh: np.ndarray | None = None   # (N,) ln h at the nodes (connection of g^); None = Levi-Civita of g only


# ------------------------------------------------------------------------------------------------ discrete geometry

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _angle_in_basis(v: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    return np.arctan2(np.einsum("ij,ij->i", v, e2), np.einsum("ij,ij->i", v, e1))


def _wrap(a: np.ndarray) -> np.ndarray:
    """Angle into (-pi, pi]."""
    return np.angle(np.exp(1j * a))


def _edges(tris: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique edges (E,2) with n0 < n1; for every triangle half-edge (T,3) (corner k -> k+1) its edge id and sign
    (+1 when the half-edge runs n0 -> n1)."""
    a = tris
    b = np.roll(tris, -1, axis=1)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    uniq, inv = np.unique((lo * n_nodes + hi).ravel(), return_inverse=True)
    edges = np.column_stack([uniq // n_nodes, uniq % n_nodes]).astype(np.int64)
    return edges, inv.reshape(tris.shape), np.where(a < b, 1.0, -1.0)


def _levi_civita(p: np.ndarray, e1: np.ndarray, e2: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """(E,) alpha_ij for edges i -> j (i = edges[:,0]): exact parallel transport along the great-circle arc."""
    i, j = edges[:, 0], edges[:, 1]
    pi, pj = p[i], p[j]
    c = np.einsum("ij,ij->i", pi, pj)[:, None]
    t_i = _unit(pj - c * pi)                     # arc direction at p_i toward p_j
    t_j = _unit(c * pj - pi)                     # arc direction at p_j, continuing away from p_i
    return _angle_in_basis(t_j, e1[j], e2[j]) - _angle_in_basis(t_i, e1[i], e2[i])


def _conformal_term(p: np.ndarray, tris: np.ndarray, edges: np.ndarray, he_edge: np.ndarray,
                    f: np.ndarray) -> np.ndarray:
    """(E,) delta_ij = l_ij n_R . <grad f> for edges i -> j: flux of the P1 gradient of f (mean over the adjacent
    triangles) through the edge, positive to the right of i -> j (radians when f = ln h)."""
    q = p[tris]                                                            # (T,3,3)
    nrm = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
    two_a = np.linalg.norm(nrm, axis=1)
    nh = nrm / two_a[:, None]
    grad = np.zeros((len(tris), 3))
    for m in range(3):                                                     # P1 gradient: sum f_m (N x e_opp) / 2A
        e = q[:, (m + 2) % 3] - q[:, (m + 1) % 3]
        grad += f[tris[:, m]][:, None] * np.cross(nh, e) / two_a[:, None]
    n_e = len(edges)
    cnt = np.bincount(he_edge.ravel(), minlength=n_e).astype(np.float64)
    g_mean = np.stack([np.bincount(he_edge.ravel(), np.repeat(grad[:, d], 3), minlength=n_e) for d in range(3)],
                      axis=1) / cnt[:, None]
    pi, pj = p[edges[:, 0]], p[edges[:, 1]]
    n_right = np.cross(pj - pi, _unit(pi + pj))                            # |d| * (t x N)
    return np.einsum("ij,ij->i", g_mean, n_right)


def _cot_weights(p: np.ndarray, tris: np.ndarray, he_edge: np.ndarray, n_e: int) -> np.ndarray:
    """(E,) cotangent weights 0.5 (cot a + cot b) of the chord triangles, clamped at 0."""
    w = np.zeros(n_e)
    q = p[tris]
    for k in range(3):                                                     # corner k is opposite half-edge k+1 -> k+2
        u = q[:, (k + 1) % 3] - q[:, k]
        v = q[:, (k + 2) % 3] - q[:, k]
        cot = np.einsum("ij,ij->i", u, v) / np.linalg.norm(np.cross(u, v), axis=1)
        w += np.bincount(he_edge[:, (k + 1) % 3], 0.5 * cot, minlength=n_e)
    return np.maximum(w, 0.0)


def _tri_area(p: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """(T,) spherical area, m^2."""
    a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    exc = 2.0 * np.arctan2(det, 1.0 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c)
                           + np.einsum("ij,ij->i", c, a))
    return exc * R_EARTH_M**2


def connection(mesh: Mesh, logh: np.ndarray | None) -> tuple:
    """(edges (E,2), he_edge (T,3), he_sign (T,3), rho (E,), (e1, e2)): transport angles for edges n0 -> n1,
    Levi-Civita of g plus, when logh is given, the conformal term of g^ = g/h^2."""
    p, tris = mesh.points, mesh.tris
    edges, he_edge, he_sign = _edges(tris, len(p))
    e1, e2 = tangent_basis(p)
    rho = _levi_civita(p, e1, e2, edges)
    if logh is not None:
        rho = rho + _conformal_term(p, tris, edges, he_edge, logh)
    return edges, he_edge, he_sign, rho, (e1, e2)


def _holonomy(rho: np.ndarray, he_edge: np.ndarray, he_sign: np.ndarray) -> np.ndarray:
    """(T,) holonomy of rho around each CCW triangle, wrapped to (-pi, pi] (the true, small curvature angle)."""
    return _wrap((rho[he_edge] * he_sign).sum(axis=1))


def _boundary_halfedges(tris: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Half-edges a -> b without a twin (the mesh boundary, water on the left)."""
    a = tris.ravel()
    b = np.roll(tris, -1, axis=1).ravel()
    key, rkey = a * n_nodes + b, b * n_nodes + a
    sk = np.sort(key)
    pos = np.minimum(np.searchsorted(sk, rkey), len(sk) - 1)
    bnd = sk[pos] != rkey
    return a[bnd], b[bnd]


def _boundary_tangent_power(mesh: Mesh, nodes: np.ndarray, e1: np.ndarray, e2: np.ndarray, n: int) -> np.ndarray:
    """Unit power vectors (len(nodes),): normalised sum of t^n over the node's boundary half-edges, t in its basis."""
    p = mesh.points
    a, b = _boundary_halfedges(mesh.tris, len(p))
    d = p[b] - p[a]                                                        # direction of travel a -> b
    acc = np.zeros(len(p), dtype=np.complex128)
    for s in (a, b):                                                       # its tangent at both ends
        t = d - np.einsum("ij,ij->i", d, p[s])[:, None] * p[s]
        np.add.at(acc, s, np.exp(1j * n * _angle_in_basis(t, e1[s], e2[s])))
    z = acc[nodes]
    return z / np.abs(z)


def _assemble(n_nodes: int, edges: np.ndarray, w: np.ndarray, r: np.ndarray) -> csc_matrix:
    """Hermitian L with z^H L z = sum w |z_j - r z_i|^2 over edges (i, j)."""
    i, j = edges[:, 0], edges[:, 1]
    wc = w.astype(np.complex128)
    vals = np.concatenate([wc, wc, -w * r, -w * np.conj(r)])
    return coo_matrix((vals, (np.concatenate([i, j, j, i]), np.concatenate([i, j, i, j]))),
                      shape=(n_nodes, n_nodes)).tocsc()


# ------------------------------------------------------------------------------------------------ sparse solvers

@njit(cache=True)
def _aggregate(indptr, indices, n):
    """Standard aggregation of a graph (CSR pattern): seeds with a fully free neighbourhood, then join a neighbour's
    aggregate, then singletons."""
    agg = np.full(n, -1, dtype=np.int64)
    na = 0
    for i in range(n):
        if agg[i] >= 0:
            continue
        free = True
        for k in range(indptr[i], indptr[i + 1]):
            if agg[indices[k]] >= 0:
                free = False
                break
        if free:
            for k in range(indptr[i], indptr[i + 1]):
                agg[indices[k]] = na
            agg[i] = na
            na += 1
    for i in range(n):
        if agg[i] < 0:
            for k in range(indptr[i], indptr[i + 1]):
                if agg[indices[k]] >= 0:
                    agg[i] = agg[indices[k]]
                    break
    for i in range(n):
        if agg[i] < 0:
            agg[i] = na
            na += 1
    return agg, na


@njit(cache=True)
def _gauss_seidel(indptr, indices, data, diag, x, b, backward):
    """One Gauss-Seidel sweep in place (forward, or backward for the symmetric cycle)."""
    n = len(b)
    for ii in range(n):
        i = n - 1 - ii if backward else ii
        s = b[i]
        for k in range(indptr[i], indptr[i + 1]):
            s -= data[k] * x[indices[k]]
        x[i] += s / diag[i]


def _strong(a: csr_matrix, theta: float) -> csr_matrix:
    """Pattern of the strong off-diagonal couplings |a_ij| >= theta sqrt(a_ii a_jj)."""
    c = a.tocoo()
    d = a.diagonal()
    keep = (c.row != c.col) & (np.abs(c.data) >= theta * np.sqrt(np.abs(d[c.row] * d[c.col])))
    return csr_matrix((np.ones(int(keep.sum())), (c.row[keep], c.col[keep])), shape=a.shape)


class _AMG:
    """Smoothed-aggregation multigrid V-cycle (symmetric Gauss-Seidel, strength 0.08, Jacobi-smoothed prolongator
    with omega = 2/3) for a symmetric weakly diagonally dominant M-matrix (graph Laplacian, possibly singular with
    the constant null space); used as a CG preconditioner."""

    def __init__(self, a: csr_matrix, coarse_size: int = 3000, theta: float = 0.08):
        self.levels = []
        a = a.tocsr()
        while a.shape[0] > coarse_size:
            n = a.shape[0]
            s = _strong(a, theta)
            agg, na = _aggregate(s.indptr, s.indices.astype(np.int64), n)
            pt = csr_matrix((np.ones(n), (np.arange(n), agg)), shape=(n, na))
            dinv = 1.0 / a.diagonal()
            p = (pt - (2.0 / 3.0) * (diags(dinv) @ (a @ pt))).tocsr()
            a.sort_indices()
            self.levels.append((a, a.diagonal().copy(), p, p.T.tocsr()))
            a = (p.T @ a @ p).tocsr()
        self.singular = bool(np.abs(a @ np.ones(a.shape[0])).max() <= 1e-9 * np.abs(a.diagonal()).max())
        self.coarse = splu((a + 1e-10 * diags(a.diagonal())).tocsc())

    def _coarse_solve(self, b: np.ndarray) -> np.ndarray:
        if not self.singular:
            return self.coarse.solve(b)
        x = self.coarse.solve(b - b.mean())                               # null space = constants: project it out
        return x - x.mean()

    def _cycle(self, b: np.ndarray, lvl: int) -> np.ndarray:
        if lvl == len(self.levels):
            return self._coarse_solve(b)
        a, d, p, pt = self.levels[lvl]
        x = np.zeros_like(b)
        _gauss_seidel(a.indptr, a.indices, a.data, d, x, b, False)
        x += p @ self._cycle(pt @ (b - a @ x), lvl + 1)
        _gauss_seidel(a.indptr, a.indices, a.data, d, x, b, True)
        return x

    def solve(self, b: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
        if not self.levels:
            return self._coarse_solve(b)
        a = self.levels[0][0]
        x, _ = cg(a, b, rtol=rtol, maxiter=500, M=LinearOperator(a.shape, lambda r: self._cycle(r, 0)))
        return x


def _spd_solve(a: csr_matrix, b: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """CG with a Jacobi preconditioner, for well-conditioned SPD / Hermitian systems (mass + small tau Laplacian)."""
    x, _ = cg(a, b, rtol=rtol, maxiter=2000, M=diags(1.0 / a.diagonal()))
    return x


# ------------------------------------------------------------------------------------------------ cone placement

def _kd_cones(pos: np.ndarray, mass: np.ndarray, count: int) -> np.ndarray:
    """Triangle ids of `count` cones for a nonnegative mass on triangles (centroids pos): recursive mass-balanced
    splits along the axis of largest extent; each leaf holds ~1/count of the mass and gets one cone at the triangle
    nearest its weighted centre."""
    out = []
    stack = [(np.flatnonzero(mass > 0), count)]
    while stack:
        idx, m = stack.pop()
        if m <= 0 or len(idx) == 0:
            continue
        if m == 1 or len(idx) == 1:
            c = (mass[idx, None] * pos[idx]).sum(axis=0)
            heavy = idx[mass[idx] >= 0.1 * mass[idx].max()]               # not a light triangle off the support
            out.extend([heavy[np.argmax(pos[heavy] @ c)]] * m)
            continue
        q = pos[idx]
        ax = int(np.argmax(q.max(axis=0) - q.min(axis=0)))
        order = np.argsort(q[:, ax], kind="stable")
        cum = np.cumsum(mass[idx[order]])
        m_left = m // 2
        cut = int(np.clip(np.searchsorted(cum, cum[-1] * m_left / m), 1, len(idx) - 1))
        stack.append((idx[order[:cut]], m_left))
        stack.append((idx[order[cut:]], m - m_left))
    return np.asarray(out, dtype=np.int64)


def _place_cones(parts: list, pos: np.ndarray, area_hat: np.ndarray, lap_d: csr_matrix, tau: float, total: int,
                 push: tuple | None = None) -> np.ndarray:
    """(T,) integer cone indices (units of 1/n) with sum = total.

    parts: charges (T,) in units of cones, quantised one by one so that charges of different spatial scale (the diffuse
    sphere curvature, the concentrated Laplace ln h, the coast line charge) do not share k-d leaves. Each is smoothed by
    the heat kernel of width ~sqrt(tau) (g^ units); push = (near, target) moves the charge of the triangles near a fixed
    coast to their target triangles inland. Cone count of a part: round(sum); the rounding left over to reach `total`
    goes to the part with the largest residual of that sign; within a part the positive and negative masses are
    partitioned separately (round(sum q+) cones of +1, the rest -1)."""
    a = (diags(area_hat) + tau * lap_d).tocsr()
    qs = []
    for charge in parts:
        q = area_hat * _spd_solve(a, charge) if np.any(charge) else np.zeros(len(pos))
        if push is not None:
            near, target = push
            moved = q[near]
            q[near] = 0.0
            np.add.at(q, target, moved)
        qs.append(q)
    sums = np.array([q.sum() for q in qs])
    counts = np.rint(sums).astype(np.int64)
    diff = total - int(counts.sum())
    while diff != 0:
        s = int(np.sign(diff))
        i = int(np.argmax(s * (sums - counts)))
        counts[i] += s
        diff -= s
    k = np.zeros(len(pos), dtype=np.int64)
    for q, c in zip(qs, counts):
        n_plus = int(round(q[q > 0].sum()))
        n_minus = n_plus - c
        if n_minus < 0:
            n_plus, n_minus = c, 0
        np.add.at(k, _kd_cones(pos, np.maximum(q, 0.0), n_plus), 1)
        np.add.at(k, _kd_cones(pos, np.maximum(-q, 0.0), n_minus), -1)
    return k


@njit(cache=True)
def _integrate(order, pred, nbr_ptr, nbr_idx, nbr_ang, psi):
    """psi[v] = psi[pred[v]] + angle(pred -> v) in BFS order."""
    for v in order:
        u = pred[v]
        if u < 0:
            continue
        for k in range(nbr_ptr[u], nbr_ptr[u + 1]):
            if nbr_idx[k] == v:
                psi[v] = psi[u] + nbr_ang[k]
                break


def _trivial_connection_field(mesh: Mesh, n: int, edges, he_edge, he_sign, rho, delta, w, hv, fixed, z_fixed,
                              tau_cells: float, cones: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Power field z (N,) parallel for rho + x except at cones (rho includes the conformal term delta); returns
    (z, cone indices per triangle). cones=False: no cone anywhere (k = 0; the curvature is absorbed by x, grounded
    through the free boundary edges)."""
    p, tris = mesh.points, mesh.tris
    nn, nt, ne = len(p), len(tris), len(edges)
    in_tri = np.bincount(tris.ravel(), minlength=nn) > 0
    # triangles of each edge
    t_of = np.repeat(np.arange(nt), 3)
    he = he_edge.ravel()
    sg = he_sign.ravel()
    order = np.argsort(he, kind="stable")
    first = np.r_[True, he[order][1:] != he[order][:-1]]
    cnt = np.bincount(he, minlength=ne)
    # fixed boundary edges: x set so that rho + x maps z_i to z_j
    xfix = np.zeros(ne)
    is_fixed_edge = np.zeros(ne, dtype=bool)
    if fixed.any():
        bnd_edge = cnt == 1
        is_fixed_edge = bnd_edge & fixed[edges[:, 0]] & fixed[edges[:, 1]]
        zi, zj = z_fixed[edges[is_fixed_edge, 0]], z_fixed[edges[is_fixed_edge, 1]]
        xfix[is_fixed_edge] = _wrap(np.angle(zj) - np.angle(zi) - n * rho[is_fixed_edge]) / n
    # charge per triangle (units of cones): g^ holonomy, and the turning of the fixed boundary tangents relative to the
    # transport (the coast's geodesic curvature in g^), which sits on the boundary triangles
    hol = _holonomy(rho, he_edge, he_sign)
    turn = (xfix[he_edge] * he_sign).sum(axis=1)
    charge = (n / (2.0 * np.pi)) * (hol + turn)
    # triangle graph: interior edges, weight 1 / w_e; free boundary edges ground the triangle (outside = 0)
    w_c = np.maximum(w, 0.05 * np.median(w))
    interior = cnt == 2
    ta = t_of[order][first]                                                # first triangle of each edge (by edge id)
    tb = np.full(ne, -1)
    second = order[~first]
    tb[he[second]] = t_of[second]
    ie = np.flatnonzero(interior)
    c = 1.0 / w_c[ie]
    free_bnd = np.flatnonzero((cnt == 1) & ~is_fixed_edge)
    diag = np.bincount(ta[ie], c, minlength=nt) + np.bincount(tb[ie], c, minlength=nt) \
        + np.bincount(ta[free_bnd], 1.0 / w_c[free_bnd], minlength=nt)
    lap_d = coo_matrix((np.r_[diag, -c, -c], (np.r_[np.arange(nt), ta[ie], tb[ie]],
                                               np.r_[np.arange(nt), tb[ie], ta[ie]])), shape=(nt, nt)).tocsr()
    # triangle components (no interior edge joins two): x is solved on each separately, so each holds the cones of its
    # own charge (an integer for a component whose boundary is all fixed) and is quantised on its own
    tcomp = connected_components(lap_d, directed=False)[1]
    groups = np.split(np.argsort(tcomp, kind="stable"), np.flatnonzero(np.diff(np.sort(tcomp))) + 1)
    area = _tri_area(p, tris)
    h_tri = np.exp(np.log(hv)[tris].mean(axis=1))
    area_hat = area / h_tri**2
    pos = _unit(p[tris].mean(axis=1))
    # three charges of different scale: K dA (Levi-Civita part of hol), Laplace ln h dA without the flux through fixed
    # coast edges, and the coast line charge (turning of the fixed tangents + that flux)
    excess = area / R_EARTH_M**2
    bflux = ((delta * is_fixed_edge)[he_edge] * he_sign).sum(axis=1)
    c_n = n / (2.0 * np.pi)
    near = shell = None
    if is_fixed_edge.any():
        # charges within COAST_CONE_DISTANCE h of a fixed coast go to the nearest triangle of the shell at that distance
        # in their own component (inland of a strait narrower than twice it: to the strait's mouths), so no cone sits
        # in the coastal rows
        coast = np.unique(edges[is_fixed_edge])
        d_coast = cKDTree(p[coast]).query(pos)[0] * R_EARTH_M / h_tri
        near = d_coast < COAST_CONE_DISTANCE
        shell = (d_coast >= COAST_CONE_DISTANCE) & (d_coast < COAST_CONE_DISTANCE + 1.0)
    k = np.zeros(nt, dtype=np.int64)
    parts = [c_n * excess, c_n * (hol - excess - bflux), c_n * (turn + bflux)]
    for t in groups if cones else []:
        push = None
        if near is not None:
            nr, sh = np.flatnonzero(near[t]), np.flatnonzero(shell[t])
            if len(nr) and len(sh):
                push = (nr, sh[cKDTree(pos[t][sh]).query(pos[t][nr])[1]])
        k[t] = _place_cones([q[t] for q in parts], pos[t], area_hat[t], lap_d if len(groups) == 1 else lap_d[t][:, t],
                            tau_cells, int(round(charge[t].sum())), push)
    # trivial connection: min sum w x^2 with (d x)_t = 2 pi k_t / n - hol_t - fixed part; x = W^-1 d^T lam
    rhs = (2.0 * np.pi / n) * k - hol - turn
    grounded = np.zeros(nt, dtype=bool)
    grounded[ta[free_bnd]] = True
    for t in groups:
        if not grounded[t].any():                                          # Neumann: consistent rhs, any mean
            rhs[t] -= rhs[t].mean()
    lam = _AMG(lap_d).solve(rhs)
    # x_e = (1/w_e) (d^T lam)_e: sign of edge e in its triangles
    sg_ta = np.zeros(ne)
    sg_ta[he[order][first]] = sg[order][first]
    x = np.where(interior, sg_ta * (lam[ta] - lam[np.maximum(tb, 0)]), sg_ta * lam[ta]) / w_c
    x = np.where(is_fixed_edge, xfix, x)
    # integrate along a BFS forest: one tree per connected component of the edge graph (a closed sea kept as its own
    # loop, a lake), rooted at the component's first fixed node, else at the first corner of its first triangle, and
    # anchored there (psi = the fixed value, else 0)
    anchors = np.flatnonzero(fixed & in_tri)
    ncomp = connected_components(csr_matrix((np.ones(ne), (edges[:, 0], edges[:, 1])), shape=(nn, nn)),
                                 directed=False)[1]
    root = np.full(ncomp.max() + 1, -1)
    cc, first = np.unique(ncomp[tris[:, 0]], return_index=True)
    root[cc] = tris[first, 0]
    cc, first = np.unique(ncomp[anchors], return_index=True)
    root[cc] = anchors[first]
    roots = root[root >= 0]
    tree = _BFSTree(edges, nn, roots)
    psi = tree.integrate(n * (rho + x), np.angle(z_fixed[roots]))
    if len(anchors):
        # every fixed boundary piece but each component's first comes out turned by a constant angle: the relative
        # phase of separate boundary pieces is not a triangle constraint. Remove it with a harmonic g (vertex
        # Laplacian), g = -offset / n on each piece: x += dg changes no triangle holonomy and no fixed boundary edge.
        fe = np.flatnonzero(is_fixed_edge)
        comp = connected_components(csr_matrix((np.ones(len(fe)), (edges[fe, 0], edges[fe, 1])), shape=(nn, nn)),
                                    directed=False)[1]
        d = psi[anchors] - np.angle(z_fixed[anchors])
        nc = comp.max() + 1
        off = np.angle(np.bincount(comp[anchors], np.cos(d), minlength=nc)
                       + 1j * np.bincount(comp[anchors], np.sin(d), minlength=nc))
        if np.abs(off).max() > 1e-9:
            g = np.zeros(nn)
            g[anchors] = -off[comp[anchors]] / n
            free = ~fixed & in_tri
            lap_v = coo_matrix((np.r_[w_c, w_c, -w_c, -w_c], (np.r_[edges[:, 0], edges[:, 1], edges[:, 0], edges[:, 1]],
                                np.r_[edges[:, 0], edges[:, 1], edges[:, 1], edges[:, 0]])), shape=(nn, nn)).tocsr()
            g[free] = _AMG(lap_v[free][:, free].tocsr()).solve(-(lap_v[free][:, anchors] @ g[anchors]))
            x = x + g[edges[:, 1]] - g[edges[:, 0]]
            psi = tree.integrate(n * (rho + x), np.angle(z_fixed[roots]))
    z = np.exp(1j * psi)
    z[fixed] = z_fixed[fixed]
    return z, k


class _BFSTree:
    """Breadth-first spanning forest of the edge graph, one tree per root (one root per connected component), for
    integrating transport angles (n0 -> n1 per edge). A virtual node joined to every root makes it one BFS: within a
    component the tree is the one a BFS from its root alone gives."""

    def __init__(self, edges: np.ndarray, nn: int, roots: np.ndarray):
        ne = len(edges)
        rows = np.r_[edges[:, 0], edges[:, 1]]
        cols = np.r_[edges[:, 1], edges[:, 0]]
        perm = np.lexsort((cols, rows))                                    # CSR order: neighbours of each node
        self.cols = cols[perm].astype(np.int64)
        self.sign = np.r_[np.ones(ne), -np.ones(ne)][perm]
        self.eid = np.r_[np.arange(ne), np.arange(ne)][perm]
        self.ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=nn))].astype(np.int64)
        roots = np.asarray(roots, dtype=np.int64)
        g = csr_matrix((np.ones(2 * ne + len(roots)), (np.r_[rows, np.full(len(roots), nn)], np.r_[cols, roots])),
                       shape=(nn + 1, nn + 1))
        bfs, pred = breadth_first_order(g, nn, directed=False, return_predecessors=True)
        pred = pred[:nn].astype(np.int64)
        pred[roots] = -1
        self.order, self.pred, self.roots, self.nn = bfs[1:].astype(np.int64), pred, roots, nn

    def integrate(self, ang: np.ndarray, value0: np.ndarray) -> np.ndarray:
        """psi (N,) with psi[roots] = value0 and psi[v] = psi[parent] + ang(parent -> v) down each tree."""
        psi = np.zeros(self.nn)
        psi[self.roots] = value0
        _integrate(self.order, self.pred, self.ptr, self.cols, self.sign * ang[self.eid], psi)
        return psi


# ------------------------------------------------------------------------------------------------ public API

def cross_field(mesh: Mesh, h: SizeField, n: int = 4, boundary_nodes: np.ndarray | None = None,
                smooth_steps: int = 0, method: str = "cones", cones: bool = True) -> CrossField:
    """n-direction field for the connection of g^ = g/h^2 (see the module docstring).

    boundary_nodes: node ids whose field is fixed to the tangent of the mesh boundary; None = free.
    smooth_steps: MBO steps after the solve. method: "cones" (integrable field, singularities follow rho_c) or
    "smoothest" (Knöppel). Pass a uniform h to get the plain Levi-Civita (h-blind) field. cones=False (method
    "cones" only): the trivial connection with no cone at all, for a disc-like mesh with a free boundary (m14's block
    frame: parallel in the flat metric g / (h e^lambda)^2, lambda = 0 on the boundary).
    """
    p = mesh.points
    nn = len(p)
    hv = h(p)
    logh = np.log(hv)
    edges, he_edge, he_sign, rho, (e1, e2) = connection(mesh, None)
    delta = _conformal_term(p, mesh.tris, edges, he_edge, logh)
    rho = rho + delta
    w = _cot_weights(p, mesh.tris, he_edge, len(edges))
    l_hat = float(np.median(np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1) * R_EARTH_M
                            * 2.0 / (hv[edges[:, 0]] + hv[edges[:, 1]])))
    fixed = np.zeros(nn, dtype=bool)
    if boundary_nodes is not None:
        fixed[boundary_nodes] = True
    fixed |= np.bincount(mesh.tris.ravel(), minlength=nn) == 0             # nodes in no triangle: frozen
    z_fixed = np.ones(nn, dtype=np.complex128)
    if boundary_nodes is not None and len(boundary_nodes):
        z_fixed[boundary_nodes] = _boundary_tangent_power(mesh, np.asarray(boundary_nodes), e1, e2, n)
    mass = np.bincount(mesh.tris.ravel(), np.repeat(_tri_area(p, mesh.tris) / 3.0, 3), minlength=nn) / hv**2
    lap = None

    if method == "cones":
        z, _ = _trivial_connection_field(mesh, n, edges, he_edge, he_sign, rho, delta, w, hv, fixed, z_fixed,
                                         (CONE_SMOOTH_CELLS * l_hat) ** 2, cones)
    else:
        lap = _assemble(nn, edges, w, np.exp(1j * n * rho))
        free = ~fixed
        if not fixed.any():
            shift = 1e-10 * float(np.mean(lap.diagonal().real) / np.mean(mass))
            lu = splu((lap + shift * diags(mass)).tocsc())
            z = np.random.default_rng(0).standard_normal(nn) + 1j * np.random.default_rng(1).standard_normal(nn)
            lam_old = np.inf
            for _ in range(50):
                z = lu.solve(mass * z)
                z /= np.sqrt(np.sum(mass * np.abs(z) ** 2))
                lam = float(np.real(np.vdot(z, lap @ z)))
                if abs(lam_old - lam) <= 1e-8 * abs(lam):
                    break
                lam_old = lam
        else:
            z = z_fixed.copy()
            z[free] = splu(lap[free][:, free].tocsc()).solve(-(lap[free][:, fixed] @ z[fixed]))

    if smooth_steps:
        if lap is None:
            lap = _assemble(nn, edges, w, np.exp(1j * n * rho))
        free = ~fixed
        a = (diags(mass) + (2.0 * l_hat) ** 2 * lap).tocsr()
        a_ff = a[free][:, free].tocsr()
        a_fb = a[free][:, fixed]
        for _ in range(smooth_steps):
            z = z / np.maximum(np.abs(z), 1e-300)
            z[free] = _spd_solve(a_ff, mass[free] * z[free] - a_fb @ z[fixed], rtol=1e-8)

    phi = np.angle(z) / n
    u = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
    return CrossField(mesh, u, n, logh)


def singularities(field: CrossField) -> tuple[np.ndarray, np.ndarray]:
    """Triangles with nonzero index and their indices (multiples of 1/n, float).

    Index of triangle (i,j,k) (Knöppel 2013 eq. 19): (omega_ij + omega_jk + omega_ki + Omega) / (2 pi n), omega_ij in
    (-pi, pi] the rotation of the power field across edge i -> j relative to the transport, Omega the triangle's
    holonomy. The transport is the field's connection (g^ when field.logh is set); the index does not depend on it
    unless a rotation sits near +-pi. On the closed sphere the indices sum to chi = 2.
    """
    mesh, n = field.mesh, field.n
    tris = mesh.tris
    edges, he_edge, he_sign, rho, (e1, e2) = connection(mesh, field.logh)
    psi = n * np.arctan2(np.einsum("ij,ij->i", field.u, e2), np.einsum("ij,ij->i", field.u, e1))
    rh = n * rho[he_edge] * he_sign                                        # transport of half-edge k -> k+1
    om = _wrap(psi[np.roll(tris, -1, axis=1)] - psi[tris] - rh)
    k = np.rint((om.sum(axis=1) + _wrap(rh.sum(axis=1))) / (2.0 * np.pi)).astype(np.int64)
    ids = np.flatnonzero(k)
    return ids, k[ids] / float(n)
