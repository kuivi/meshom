"""m12: fixed cubed-sphere quad mesh moved to follow h (r-adaptation by optimal transport), all quads.

1. Equiangular cubed sphere, n x n cells per face, n = round(sqrt(integral of h^-2 dA over the sphere / 6)).
2. Nodes moved with the topology fixed, x = exp_xi(grad phi(xi)), so that every node's dual area (a quarter of each
   adjacent quad) equals the quarter sum of the adjacent quads' targets c h(centroid)^2: the Monge-Ampere-type
   equation of Weller, Browne, Budd and Cullen (2016) solved by their fixed-point iteration
   (1 + a) Lap(phi^{k+1}) = (1 + a) Lap(phi^k) - s(phi^k), eq. (22), with the under-relaxation (34).
   Discretisation (ours, node based): P1 cotangent Laplacian and area-averaged P1 gradient on both diagonal splits of
   the quads, lumped mass = computational dual area V_xi; the Laplacian is factorised once. Deviation from the paper:
   s = (V_x - target) / V_xi instead of V_x / V_xi - c / m, i.e. the target does not scale with the computational
   cell area, so the 1.37x area spread of the equiangular grid is removed too (quads -> area c h^2).
3. Coast (domain with loops): cells move on the closed sphere first; cells kept if their nodes are in water and at
   least `coast_gap` * h from the coast; edge-connected main piece, pinch nodes removed; then one coastal row between
   the staircase boundary and the resampled coast (`strip`, default: quads where both polylines advance together,
   triangles elsewhere, every resampled coast node kept) or the staircase nodes projected onto the coast (`snap`);
   then guarded Laplacian sweeps on the rows next to the coast (m12/coast.py).
4. Optional (`polish` = lam): L-BFGS on a size+shape (TMOP-type) energy before the cut (m12/polish.py).
"""
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

from meshom.domain import Domain
from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh
from meshom.methods import OVERRIDES, register
from meshom.size import SizeField

DEFAULTS = {
    "n": None,              # cells per face edge; None = from integral of h^-2
    "max_iter": 400,        # fixed-point iterations
    "tol": 0.01,            # stop when max |V_x / (c h^2) - 1| < tol
    "coast": "strip",       # "strip" | "snap" (see m12/coast.py)
    "coast_gap": 0.4,       # strip: kept cells have every node >= coast_gap * h from the coast
    "smooth_rows": 3,       # Laplacian smoothing of nodes within this many rows of the coast
    "smooth_iters": 30,
    "polish": None,         # None | lam: size+shape L-BFGS polish after OT (m12/polish.py), lam = size weight
    "polish_rounds": 3,
    "polish_iters": 150,
    "verbose": False,
}


# ---------------------------------------------------------------------------------------------------- cubed sphere

def cubed_sphere(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Equiangular cubed sphere: (6 n^2 + 2, 3) unit vectors and (6 n^2, 4) CCW quads."""
    t = np.tan(np.linspace(-np.pi / 4, np.pi / 4, n + 1))
    U, V = np.meshgrid(t, t, indexing="ij")                       # U[i, j] = t[i] (along u), V = t[j] (along v)
    e = np.eye(3)
    faces = [(0, 1, 2, 1), (0, 2, 1, -1), (1, 2, 0, 1), (1, 0, 2, -1), (2, 0, 1, 1), (2, 1, 0, -1)]
    pts, quads = [], []
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    i, j = i.ravel(), j.ravel()
    for f, (k, a, b, s) in enumerate(faces):
        p = s * e[k] + U.ravel()[:, None] * e[a] + V.ravel()[:, None] * e[b]     # u x v = s e_k: CCW from outside
        pts.append(p / np.linalg.norm(p, axis=1, keepdims=True))
        base = f * (n + 1) ** 2
        node = lambda ii, jj: base + ii * (n + 1) + jj                             # noqa: E731
        quads.append(np.column_stack([node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]))
    pts = np.vstack(pts)
    _, first, inv = np.unique(np.round(pts, 9), axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)                                     # keep first-seen order
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    quads = rank[inv.ravel()][np.vstack(quads)]
    return pts[first[order]], quads.astype(np.int64)


# ---------------------------------------------------------------------------------------------------- geometry

def _tri_excess(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Signed spherical area (sr) of triangles (a, b, c), > 0 for CCW seen from outside."""
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = 1.0 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c) + np.einsum("ij,ij->i", c, a)
    return 2.0 * np.arctan2(det, den)


def quad_area(x: np.ndarray, quads: np.ndarray) -> np.ndarray:
    p = [x[quads[:, k]] for k in range(4)]
    return _tri_excess(p[0], p[1], p[2]) + _tri_excess(p[0], p[2], p[3])


def corner_area(x: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """(Q,) min over corners of the signed area of the corner triangle (p[k-1], p[k], p[k+1]); < 0 = tangled."""
    p = [x[quads[:, k]] for k in range(4)]
    return np.min([_tri_excess(p[k - 1], p[k], p[(k + 1) % 4]) for k in range(4)], axis=0)


def node_area(x: np.ndarray, quads: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(quads.ravel(), np.repeat(quad_area(x, quads) / 4.0, 4), minlength=n)


def _centroids(x: np.ndarray, quads: np.ndarray) -> np.ndarray:
    c = x[quads].sum(axis=1)
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def expmap(xi: np.ndarray, v: np.ndarray) -> np.ndarray:
    """exp_xi(v) for tangent vectors v (unit sphere)."""
    th = np.linalg.norm(v, axis=1)
    f = np.where(th > 0, np.sin(th) / np.where(th > 0, th, 1.0), 1.0)
    x = np.cos(th)[:, None] * xi + f[:, None] * v
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _splits(quads: np.ndarray) -> np.ndarray:
    return np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]], quads[:, [0, 1, 3]], quads[:, [1, 2, 3]]])


def operators(xi: np.ndarray, quads: np.ndarray):
    """Stiffness K (cotangent, both diagonal splits averaged) and the node gradient G = (Gx, Gy, Gz)."""
    n = len(xi)
    tris = _splits(quads)
    p = xi[tris]                                                    # (T,3,3)
    nrm = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    two_a = np.linalg.norm(nrm, axis=1)
    nh = nrm / two_a[:, None]
    rows, cols, vals = [], [], []
    for k in range(3):
        i, j = tris[:, (k + 1) % 3], tris[:, (k + 2) % 3]
        u, v = p[:, (k + 1) % 3] - p[:, k], p[:, (k + 2) % 3] - p[:, k]
        w = 0.25 * np.einsum("ij,ij->i", u, v) / np.linalg.norm(np.cross(u, v), axis=1)   # 0.5 cot, half weight
        rows += [i, j, i, j]
        cols += [j, i, i, j]
        vals += [-w, -w, w, w]
    K = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsc()
    # gradient of P1 on each triangle: sum_k phi_k (n x e_k) / 2A, e_k opposite corner k; node = area-weighted mean
    wsum = np.bincount(tris.ravel(), np.repeat(0.5 * two_a, 3), minlength=n)
    G = []
    for d in range(3):
        r, c, v = [], [], []
        for k in range(3):
            ek = p[:, (k + 2) % 3] - p[:, (k + 1) % 3]
            gk = np.cross(nh, ek)[:, d] / two_a                      # d-component of grad lambda_k
            for m in range(3):
                r.append(tris[:, m])
                c.append(tris[:, k])
                v.append(0.5 * two_a * gk / wsum[tris[:, m]])
        G.append(coo_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(n, n)).tocsr())
    return K, G


class _Power:
    """h^t (the normalisation of the target makes the units irrelevant): continuation from uniform (t=0) to h."""

    def __init__(self, h: SizeField, t: float):
        self.h, self.t = h, t

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        return self.h(xyz) ** self.t


def ot_move(xi: np.ndarray, quads: np.ndarray, h: SizeField, max_iter: int = 400, tol: float = 0.01,
            verbose: bool = False, stages: int = 1) -> tuple[np.ndarray, dict]:
    """Continuation over `stages` (h^(k/stages), k = 1..stages, warm start); see _ot_stage."""
    n = len(xi)
    K, G = operators(xi, quads)
    lu = splu(csc_matrix(K[1:, 1:]), permc_spec="MMD_AT_PLUS_A")   # phi pinned at node 0 (K singular on S^2)
    phi = np.zeros(n)
    info = {}
    per = max(1, max_iter // stages)
    for k in range(1, stages + 1):
        hk = h if k == stages else _Power(h, k / stages)
        x, phi, info = _ot_stage(xi, quads, hk, K, G, lu, phi, per if k < stages else max_iter - per * (stages - 1),
                                 tol, verbose)
    return x, info


def _ot_stage(xi, quads, h, K, G, lu, phi, max_iter, tol, verbose):
    """Weller et al. (2016) fixed-point Monge-Ampere iteration. Returns (x, info).

    Equidistributed quantity: node dual area V_x (quarter quads) against the quarter sum of the quad targets
    c h(centroid)^2, so at convergence quads have area ~ c h^2 also next to the 8 valence-3 cube corners.
    """
    n = len(xi)
    v_xi = node_area(xi, quads, n)
    phi = phi.copy()
    x = _map(xi, G, phi)
    relax = 1.0
    hist = []
    best = (np.inf, x, 0, phi.copy())
    for it in range(max_iter + 1):
        a_q = quad_area(x, quads)
        v_x = np.bincount(quads.ravel(), np.repeat(a_q / 4.0, 4), minlength=n)
        h2 = (h(_centroids(x, quads)) / R_EARTH_M) ** 2           # target quad area c h^2 at the quad centroid
        h2 *= a_q.sum() / h2.sum()
        tgt = np.bincount(quads.ravel(), np.repeat(h2 / 4.0, 4), minlength=n)
        err = np.abs(v_x / tgt - 1.0)
        tangled = int((corner_area(x, quads) <= 0).sum())
        hist.append((float(err.max()), float(np.percentile(err, 99)), tangled))
        if verbose and (it % 20 == 0 or it == max_iter):
            print(f"  it {it:4d}  max|e| {err.max():.4f}  p99 {np.percentile(err, 99):.4f}  relax {relax:.2f}"
                  f"  tangled {tangled}", flush=True)
        if tangled == 0 and err.max() < best[0]:
            best = (float(err.max()), x, it, phi.copy())
        if err.max() < tol or it == max_iter:
            break
        s = (v_x - tgt) / v_xi
        relax = max(relax, 4.0 * max(0.25, float(np.abs(s).max())))
        rhs = v_xi * s
        rhs -= v_xi * (rhs.sum() / v_xi.sum())
        d = np.zeros(n)
        d[1:] = lu.solve(rhs[1:])
        phi += d / relax
        x = _map(xi, G, phi)
    info = {"iters": it, "last_err_max": hist[-1][0], "last_tangled": hist[-1][2], "iter": it,
            "err_max": hist[-1][0], "err_p99": hist[-1][1], "tangled": hist[-1][2]}
    if hist[-1][2] > 0 and np.isfinite(best[0]):                  # never return a tangled mesh if an untangled one was seen
        x, phi = best[1], best[3]
        info.update(iter=best[2], err_max=hist[best[2]][0], err_p99=hist[best[2]][1], tangled=0)
    return x, phi, info


def _map(xi: np.ndarray, G, phi: np.ndarray) -> np.ndarray:
    g = np.column_stack([Gd @ phi for Gd in G])
    g -= np.einsum("ij,ij->i", g, xi)[:, None] * xi
    return expmap(xi, g)


def cells_for(h: SizeField, m: int = 200_000) -> float:
    """Integral of h^-2 over the sphere (= quad count of an ideal mesh), Fibonacci quadrature."""
    i = np.arange(m) + 0.5
    z = 1.0 - 2.0 * i / m
    ph = np.pi * (1.0 + 5**0.5) * i
    r = np.sqrt(1.0 - z * z)
    p = np.column_stack([r * np.cos(ph), r * np.sin(ph), z])
    return float(4.0 * np.pi * R_EARTH_M**2 / m * np.sum(h(p) ** -2.0))


# ---------------------------------------------------------------------------------------------------- generate

@register()
def generate(domain: Domain, h: SizeField, opts: dict | None = None) -> Mesh:
    o = {**DEFAULTS, **(opts or {})}
    n = o["n"] or max(2, int(round(np.sqrt(cells_for(h) / 6.0))))
    xi, quads = cubed_sphere(n)
    x, info = ot_move(xi, quads, h, o["max_iter"], o["tol"], o["verbose"])
    if o["verbose"]:
        print(f"m12: n = {n}, nodes {len(xi)}, OT {info}", flush=True)
    if o["polish"] is not None:
        from meshom.methods.m12.polish import polish
        x = polish(x, quads, h, float(o["polish"]), o["polish_rounds"], o["polish_iters"], o["verbose"])
    if not domain.loops:
        return Mesh(x, np.zeros((0, 3), dtype=np.int64), quads, np.zeros(len(x), dtype=np.int8))
    from meshom.methods.m12.coast import cut_to_coast
    return cut_to_coast(x, quads, domain, h, o)


@register("m12_cubed_ot_polish", overrides={"polish": 1.0})
def generate_polish(domain: Domain, h: SizeField, opts: dict | None = None) -> Mesh:
    """m12 + the size+shape polish (lam = 1) before the coast cut: trades size fidelity for angles."""
    return generate(domain, h, {**OVERRIDES["m12_cubed_ot_polish"], **(opts or {})})
