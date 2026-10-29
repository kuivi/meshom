"""The region and the two fields of c_placement (B130, B131).

Region R (`region`): the elements that are re-extracted. Seeds: elements at an irregular interior node or a
triangle, quads with a corner above `bad_deg`, and elements where the size field ramps (|d ln h| along an edge above
`ramp`: the transition band). The seeds are grown by `grow` node rings; kept pieces smaller than `min_keep` elements
are swallowed (holes). The kept elements within `collar` rings of R form the collar: an exact lattice the new rows
are stitched to.

Fields on the nodes of R + collar:
- the row direction n (unit tangent, the direction rows are stacked in): the direction of -grad ln h, smoothed by a
  vector diffusion (a band refines towards n);
- the row coordinate v (`row_field`): least squares d v = (dx . n) / h on R's edges; on the collar's lattice edges
  d v = 0 (row edge) or +-1 (column edge) with a large weight; then the collar is rounded to integers and v solved
  again with the collar fixed: rows are the integer level sets of v, the collar's rows among them;
- the column stripes psi (`stripes`, Knoppel et al. 2015 with Dirichlet values): psi complex per node minimising
  sum_edges |psi_b - exp(i w_ab) psi_a|^2, w_ab = 2 pi (dx . e) / h (e = n x p, along the rows), psi = 1 on the
  collar (its nodes are lattice points). Column lines are the level sets arg psi = 0 mod 2 pi; a triangle around
  which the phase winds is a zero of psi: a column ends there (a dislocation). The zeros' layout is the energy's:
  even, staggered, and free to leave through the coast (natural boundary).
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.sparse.linalg import spsolve

from meshom.mesh import Mesh, build_topology
from meshom.meshfix.placement.chart import defect_nodes, node_counts

DEFAULTS = dict(
    bad_deg=100.0,     # B130: quads with a corner above this seed R
    ramp=0.006,        # B130: |d ln h| along an edge above this marks the transition band
    grow=3,            # B130: node rings R grows by
    min_keep=300,      # B130: kept pieces with fewer elements join R
    collar=2,          # B130: kept rings fixed around R (the stitch lattice)
    coast_keep=0,      # B130: rings of elements at the coast (and the artificial boundary) kept as they are
    lattice_w=100.0,   # B131: weight of the collar's lattice edges in the first v solve
)


def elements(mesh: Mesh):
    """(E,4) element nodes (triangles padded with -1), tris first, and nv (E,)."""
    t = np.c_[mesh.tris, -np.ones(len(mesh.tris), np.int64)] if len(mesh.tris) else np.zeros((0, 4), np.int64)
    el = np.vstack([t, mesh.quads]).astype(np.int64)
    nv = np.r_[np.full(len(mesh.tris), 3), np.full(len(mesh.quads), 4)]
    return el, nv


def tri_split(mesh: Mesh) -> np.ndarray:
    """(T,3) triangles of the mesh (quads split along 0-2) and (T,) the element of each."""
    q = mesh.quads
    tr = np.vstack([mesh.tris, q[:, [0, 1, 2]], q[:, [0, 2, 3]]]).astype(np.int64)
    nt = len(mesh.tris)
    owner = np.r_[np.arange(nt), nt + np.arange(len(q)), nt + np.arange(len(q))]
    return tr, owner


def corner_max_angle(mesh: Mesh) -> np.ndarray:
    """(Q,) largest corner angle (deg) of each quad, tangent plane."""
    from meshom.quality.kernel import element_geometry

    g = element_geometry(mesh)
    return np.degrees(g.quad_angles).max(axis=1)


def _node_elem_csr(el, nv, n):
    rows = np.repeat(np.arange(len(el)), nv)
    cols = el[el >= 0]
    return csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(el), n))


def region(mesh: Mesh, kind: np.ndarray, h_node: np.ndarray, opts: dict | None = None, topo=None) -> dict:
    """R (E,) bool, collar (E,) bool over the elements (tris first); and the seeds' parts for the record."""
    o = {**DEFAULTS, **(opts or {})}
    topo = build_topology(mesh) if topo is None else topo
    n = mesh.n_nodes
    el, nv = elements(mesh)
    M = _node_elem_csr(el, nv, n)                      # elem x node
    bnd = topo.boundary_node
    dn = defect_nodes(mesh, bnd)
    seed = (M @ dn.astype(float)) > 0
    seed[: mesh.n_tris] = True
    bad = np.zeros(len(el), bool)
    bad[mesh.n_tris:] = corner_max_angle(mesh) > o["bad_deg"]
    E = topo.edges
    lh = np.log(h_node)
    ramp_e = np.abs(lh[E[:, 0]] - lh[E[:, 1]]) > o["ramp"]
    ramp_n = np.zeros(n, bool)
    ramp_n[E[ramp_e].ravel()] = True
    # an element ramps when all its nodes touch a ramping edge (keeps single noisy edges out)
    ramp = (M @ ramp_n.astype(float)) >= nv
    R = seed | bad | ramp
    MT = M.T.tocsr()
    for _ in range(o["grow"]):
        nodes = (MT @ R.astype(float)) > 0
        R = (M @ nodes.astype(float)) > 0
    # swallow small kept pieces
    inner = topo.edge_elems[topo.edge_elems[:, 1] >= 0]
    keep = ~R
    ok = keep[inner[:, 0]] & keep[inner[:, 1]]
    g = coo_matrix((np.ones(ok.sum()), (inner[ok, 0], inner[ok, 1])), shape=(len(el), len(el)))
    nc, lab = connected_components(g, directed=False)
    size = np.bincount(lab[keep], minlength=nc)
    small = keep & (size[lab] < o["min_keep"])
    R |= small
    # the coastal ring stays (frozen): elements within coast_keep rings of a boundary node (coast or artificial);
    # its inner nodes may still move in the relaxation, its outline toward R is a free boundary of the fields
    frozen = np.zeros(len(el), bool)
    if o["coast_keep"] > 0:
        cur = (M @ (bnd & (kind == 1)).astype(float)) > 0
        for _ in range(int(o["coast_keep"]) - 1):
            nodes = (MT @ cur.astype(float)) > 0
            cur = (M @ nodes.astype(float)) > 0
        frozen = cur
        R &= ~frozen
    lattice = ~R & ~frozen
    cur = R.copy()
    for _ in range(o["collar"]):
        nodes = (MT @ cur.astype(float)) > 0
        cur = (M @ nodes.astype(float)) > 0
    collar = cur & lattice
    return dict(R=R, collar=collar, frozen=frozen, seed=int(seed.sum()), bad=int(bad.sum()), ramp=int(ramp.sum()),
                swallowed=int(small.sum()), el=el, nv=nv)


# ------------------------------------------------------------------------------------------------------ geometry

def tangent_frame(p: np.ndarray):
    """East / north unit vectors at unit points p (fallback near the poles)."""
    z = np.array([0.0, 0.0, 1.0])
    e = np.cross(z, p)
    nrm = np.linalg.norm(e, axis=1)
    bad = nrm < 1e-8
    e[bad] = np.cross(np.array([1.0, 0, 0]), p[bad])
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    nn = np.cross(p, e)
    return e, nn


def node_gradient(points: np.ndarray, edges: np.ndarray, f: np.ndarray) -> np.ndarray:
    """(N,3) tangent gradient of f per node by least squares over its edges (per unit of chord length)."""
    n = len(points)
    a, b = np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]
    d = points[b] - points[a]
    pa = points[a]
    d -= np.einsum("ij,ij->i", d, pa)[:, None] * pa
    df = f[b] - f[a]
    # normal equations per node: (sum d d^T) g = sum d df (2x2 in the tangent basis)
    e1, e2 = tangent_frame(points)
    x, y = np.einsum("ij,ij->i", d, e1[a]), np.einsum("ij,ij->i", d, e2[a])
    Sxx = np.bincount(a, x * x, n) + 1e-30
    Sxy = np.bincount(a, x * y, n)
    Syy = np.bincount(a, y * y, n) + 1e-30
    bx = np.bincount(a, x * df, n)
    by = np.bincount(a, y * df, n)
    det = Sxx * Syy - Sxy ** 2
    gx = (Syy * bx - Sxy * by) / det
    gy = (Sxx * by - Sxy * bx) / det
    return gx[:, None] * e1 + gy[:, None] * e2


def _laplacian(edges, w, n):
    a, b = edges[:, 0], edges[:, 1]
    L = coo_matrix((np.r_[w, w, -w, -w], (np.r_[a, b, a, b], np.r_[a, b, b, a])), shape=(n, n)).tocsr()
    return L


def row_direction(points, edges, h_node, free_mask, smooth=20.0):
    """(N,3) unit row direction n: -grad ln h (dimensionless, * h), diffused: min sum |n - g|^2 + smooth sum_edges
    |n_a - n_b|^2 over the nodes in free_mask (others keep g)."""
    g = -node_gradient(points, edges, np.log(h_node)) * h_node[:, None] / 6.371e6
    n = len(points)
    L = _laplacian(edges, np.ones(len(edges)), n)
    A = (coo_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n)) + smooth * L).tocsc()
    out = np.column_stack([spsolve(A, g[:, k]) for k in range(3)])
    out -= np.einsum("ij,ij->i", out, points)[:, None] * points
    nrm = np.linalg.norm(out, axis=1)
    weak = nrm < 1e-12
    e, nn = tangent_frame(points)
    out[weak] = nn[weak]
    return out / np.linalg.norm(out, axis=1, keepdims=True), nrm


def _solve_dirichlet(L, b, fixed, val):
    """Solve L x = b with x[fixed] = val."""
    n = L.shape[0]
    free = np.flatnonzero(~fixed)
    x = np.zeros(n, dtype=np.result_type(L.dtype, b.dtype, val.dtype))
    x[fixed] = val
    rhs = b[free] - L[free][:, fixed] @ val
    x[free] = spsolve(L[free][:, free].tocsc(), rhs)
    return x


def row_field(points, edges, edge_in_R, h_node, ndir, collar_node, opts=None, edge_kept=None):
    """v per node (sub-arrays over the local node set). edges (E,2) local; edge_in_R (E,) bool (edge of an R element);
    collar_node (N,) bool. Returns v (N,), the rounding residual on the collar (max)."""
    o = {**DEFAULTS, **(opts or {})}
    n = len(points)
    a, b = edges[:, 0], edges[:, 1]
    d = points[b] - points[a]
    mid = points[a] + points[b]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    nm = ndir[a] + ndir[b]
    nm -= np.einsum("ij,ij->i", nm, mid)[:, None] * mid
    nm /= np.linalg.norm(nm, axis=1, keepdims=True)
    hm = 0.5 * (h_node[a] + h_node[b]) / 6.371e6
    proj = np.einsum("ij,ij->i", d, nm)
    t = proj / hm
    w = np.ones(len(edges))
    lat = ~edge_in_R if edge_kept is None else edge_kept     # lattice edges: edges of kept (collar) elements
    c = np.abs(proj) / np.linalg.norm(d, axis=1)
    t[lat] = np.where(c[lat] > np.sqrt(0.5), np.sign(proj[lat]), 0.0)
    w[lat] = o["lattice_w"]
    L = _laplacian(edges, w, n)
    # rhs: sum over edges w t (dv = v_b - v_a = t): grad of sum w (v_b - v_a - t)^2 -> L v = B^T W t
    rhs = np.bincount(b, w * t, n) - np.bincount(a, w * t, n)
    g = coo_matrix((np.ones(len(edges)), (a, b)), shape=(n, n))
    nc, lab = connected_components(g, directed=False)
    fixed = np.zeros(n, bool)
    first = np.unique(lab, return_index=True)[1]
    fixed[first] = True
    v0 = _solve_dirichlet(L.tocsr(), rhs, fixed, np.zeros(len(first)))
    # each collar piece (connected through collar-only edges) is a lattice of its own: its own offset, so that it
    # rounds best (offset = median fractional part)
    ce = collar_node[a] & collar_node[b] & lat
    gc = coo_matrix((np.ones(ce.sum()), (a[ce], b[ce])), shape=(n, n))
    _, labc = connected_components(gc, directed=False)
    # exact integer labels per piece by a BFS over its lattice edges (d label = t), then the piece's offset
    tl = np.round(t).astype(np.int64)
    G = coo_matrix((np.ones(2 * ce.sum()), (np.r_[a[ce], b[ce]], np.r_[b[ce], a[ce]])), shape=(n, n)).tocsr()
    dkey = np.r_[a[ce] * n + b[ce], b[ce] * n + a[ce]]
    dval = np.r_[tl[ce], -tl[ce]]
    so = np.argsort(dkey)
    dkey, dval = dkey[so], dval[so]
    lab_int = np.zeros(n, np.int64)
    for k in np.unique(labc[collar_node]):
        s = np.flatnonzero((labc == k) & collar_node)
        order, pred = breadth_first_order(G, s[0], directed=False, return_predecessors=True)
        st = dval[np.searchsorted(dkey, pred[order[1:]] * n + order[1:])]
        for node, p, d in zip(order[1:], pred[order[1:]], st):     # BFS order: the parent is labelled first
            lab_int[node] = lab_int[p] + d
        off = np.round(np.median(v0[s] - lab_int[s]))
        v0[s] = lab_int[s] + off
    res = np.abs(v0[collar_node] - np.round(v0[collar_node]))
    vc = np.round(v0[collar_node])
    Lr = _laplacian(edges[edge_in_R], w[edge_in_R], n)
    rhs_r = np.bincount(b[edge_in_R], t[edge_in_R], n) - np.bincount(a[edge_in_R], t[edge_in_R], n)
    fixed2 = collar_node.copy()
    # components of R without any collar node keep v0
    gR = coo_matrix((np.ones(edge_in_R.sum()), (a[edge_in_R], b[edge_in_R])), shape=(n, n))
    ncR, labR = connected_components(gR, directed=False)
    has = np.bincount(labR[collar_node], minlength=ncR) > 0
    nofix = ~has[labR] & ~collar_node
    fixed2 |= nofix
    val = v0[fixed2].copy()
    val[collar_node[fixed2]] = np.round(v0[fixed2][collar_node[fixed2]])
    v = _solve_dirichlet(Lr.tocsr() + 1e-12 * coo_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n)),
                         rhs_r, fixed2, val)
    return v, (float(res.max()) if len(res) else 0.0), float(np.percentile(res, 99)) if len(res) else 0.0


def collar_width(points, edges, edge_kept, h_node, ndir, collar_node):
    """(N,) column width for the stripes: h times exp(rho), rho = ln(actual column width / h) measured on the collar's
    row edges (edges of kept elements across n) and extended harmonically into R (Laplace, the collar fixed), so the
    stripes meet the collar's lattice without a misfit row of dislocations along the stitch (B131)."""
    n = len(points)
    a, b = edges[:, 0], edges[:, 1]
    d = points[b] - points[a]
    ln_ = np.linalg.norm(d, axis=1)
    c = np.abs(np.einsum("ij,ij->i", d, 0.5 * (ndir[a] + ndir[b]))) / ln_
    row = edge_kept & collar_node[a] & collar_node[b] & (c < np.sqrt(0.5))
    hm = 0.5 * (h_node[a] + h_node[b]) / 6.371e6
    r = np.log(ln_ / hm)
    s_ = np.bincount(a[row], r[row], n) + np.bincount(b[row], r[row], n)
    k_ = np.bincount(a[row], minlength=n) + np.bincount(b[row], minlength=n)
    known = k_ > 0
    rho = np.zeros(n)
    rho[known] = s_[known] / k_[known]
    if known.any() and (~known).any():
        L = _laplacian(edges, np.ones(len(edges)), n)
        L = L + 1e-9 * coo_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))
        rho = _solve_dirichlet(L.tocsr(), np.zeros(n), known, rho[known])
    return h_node * np.exp(rho)


def stripes(points, edges, h_node, ndir, fixed, width=None):
    """psi (N,) complex, omega (E,) per edge (a -> b). fixed nodes get psi = 1."""
    n = len(points)
    a, b = edges[:, 0], edges[:, 1]
    d = points[b] - points[a]
    mid = points[a] + points[b]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    nm = ndir[a] + ndir[b]
    nm -= np.einsum("ij,ij->i", nm, mid)[:, None] * mid
    nm /= np.linalg.norm(nm, axis=1, keepdims=True)
    em = np.cross(nm, mid)
    wid = (h_node if width is None else width)
    hm = 0.5 * (wid[a] + wid[b]) / 6.371e6
    om = 2.0 * np.pi * np.einsum("ij,ij->i", d, em) / hm
    w = np.ones(len(edges))
    z = np.exp(1j * om)
    L = coo_matrix((np.r_[w, w, -w * np.conj(z), -w * z], (np.r_[a, b, a, b], np.r_[a, b, b, a])),
                   shape=(n, n)).tocsr()
    psi = _solve_dirichlet(L, np.zeros(n, complex), fixed, np.ones(int(fixed.sum()), complex))
    return psi, om


def ginzburg_landau(L, fixed, psi0, kappa=1.0, iters=3000, tol=1e-10):
    """Minimise psi^H L psi + kappa/2 sum (1 - |psi|^2)^2 over the free nodes (L Hermitian, the connection
    Laplacian), fixed nodes keep psi0. L-BFGS on (Re, Im). Returns psi and the optimiser's record."""
    from scipy.optimize import minimize

    free = np.flatnonzero(~fixed)
    Lff = L[free][:, free].tocsr()
    rhs = L[free][:, fixed] @ psi0[fixed]
    nf = len(free)

    def fg(x):
        p = x[:nf] + 1j * x[nf:]
        Lp = Lff @ p + rhs
        m2 = np.abs(p) ** 2
        e = np.real(np.vdot(p, Lff @ p)) + 2.0 * np.real(np.vdot(p, rhs)) + 0.5 * kappa * np.sum((1.0 - m2) ** 2)
        g = 2.0 * (Lp + kappa * (m2 - 1.0) * p)
        return e, np.r_[g.real, g.imag]

    p0 = psi0[free]
    r = minimize(fg, np.r_[p0.real, p0.imag], jac=True, method="L-BFGS-B",
                 options=dict(maxiter=iters, maxcor=20, ftol=tol, gtol=1e-8))
    psi = psi0.copy()
    psi[free] = r.x[:nf] + 1j * r.x[nf:]
    return psi, dict(nit=int(r.nit), msg=str(r.message), e=float(r.fun))


def connection_laplacian(edges, om, n, w=None):
    a, b = edges[:, 0], edges[:, 1]
    w = np.ones(len(edges)) if w is None else w
    z = np.exp(1j * om)
    return coo_matrix((np.r_[w, w, -w * np.conj(z), -w * z], (np.r_[a, b, a, b], np.r_[a, b, b, a])),
                      shape=(n, n)).tocsr()


def winding(tris, edges, om, psi):
    """(T,) integer winding of the phase around each triangle (a -> b -> c), the connection om per edge a < b."""
    n = len(psi)
    ekey = edges[:, 0].astype(np.int64) * n + edges[:, 1]
    srt = np.argsort(ekey)
    ek = ekey[srt]

    def inc(a, b):
        k = np.minimum(a, b) * n + np.maximum(a, b)
        i = srt[np.searchsorted(ek, k)]
        w = np.where(a < b, om[i], -om[i])
        return w + np.angle(psi[b] * np.conj(psi[a]) * np.exp(-1j * w))

    s = inc(tris[:, 0], tris[:, 1]) + inc(tris[:, 1], tris[:, 2]) + inc(tris[:, 2], tris[:, 0])
    return np.round(s / (2 * np.pi)).astype(np.int64)
