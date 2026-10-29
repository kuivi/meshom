"""m09 extraction (in the spirit of QEx, Ebke et al. 2013, without its exact predicates): quads of an integer-grid map.

Every unit cell [i, i+1] x [j, j+1] of the map is found from its centre (i+1/2, j+1/2), which lies in exactly one
face; its four corners are reached by straight walks in the parameter domain from the centre (across seams with the
transition functions), so a cell that crosses a seam or touches a cone is built like any other. Corners are merged by
their 3D position (an integer point reached from different charts lands on the same point when the map is seamless).
Cells that fail (walk leaves the water, a corner repeated, a non-convex or inverted quad, an edge used twice in the
same direction) are dropped; the holes they leave and any uncovered rest are filled with a constrained Delaunay
triangulation of the quad vertices (quad edges and the coast as constraints), whose leftover triangles may be
recombined into quads.
"""
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from meshom.delaunay import sphere_delaunay

MERGE_TOL = 1e-9          # radians: two corners closer than this are one node (~6 mm)


@njit(cache=True)
def _bary(Ur, Ui, f, qr, qi):
    u0, v0 = Ur[f, 0], Ui[f, 0]
    u1, v1 = Ur[f, 1], Ui[f, 1]
    u2, v2 = Ur[f, 2], Ui[f, 2]
    det = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)
    if det == 0.0:
        return -1.0, -1.0, 3.0                                       # degenerate face: never contains q
    l1 = ((qr - u0) * (v2 - v0) - (u2 - u0) * (qi - v0)) / det
    l2 = ((u1 - u0) * (qi - v0) - (qr - u0) * (v1 - v0)) / det
    return 1.0 - l1 - l2, l1, l2


@njit(cache=True)
def _cells_pass(Ur, Ui, of, ou, ov, fill):
    """Cell centres (i+1/2, j+1/2) inside each positively oriented face (closed, tiny tolerance): count, or fill."""
    cnt = 0
    for f in range(Ur.shape[0]):
        det = (Ur[f, 1] - Ur[f, 0]) * (Ui[f, 2] - Ui[f, 0]) - (Ur[f, 2] - Ur[f, 0]) * (Ui[f, 1] - Ui[f, 0])
        if det <= 0.0:
            continue
        umin = min(Ur[f, 0], min(Ur[f, 1], Ur[f, 2]))
        umax = max(Ur[f, 0], max(Ur[f, 1], Ur[f, 2]))
        vmin = min(Ui[f, 0], min(Ui[f, 1], Ui[f, 2]))
        vmax = max(Ui[f, 0], max(Ui[f, 1], Ui[f, 2]))
        for i in range(int(np.ceil(umin - 0.5)), int(np.floor(umax - 0.5)) + 1):
            for j in range(int(np.ceil(vmin - 0.5)), int(np.floor(vmax - 0.5)) + 1):
                b0, b1, b2 = _bary(Ur, Ui, f, i + 0.5, j + 0.5)
                if b0 >= -1e-12 and b1 >= -1e-12 and b2 >= -1e-12:
                    if fill:
                        of[cnt] = f
                        ou[cnt] = i + 0.5
                        ov[cnt] = j + 0.5
                    cnt += 1
    return cnt


def _cells(Ur, Ui):
    e = np.zeros(1, dtype=np.int64)
    n = _cells_pass(Ur, Ui, e, np.zeros(1), np.zeros(1), False)
    of, ou, ov = np.empty(n, dtype=np.int64), np.empty(n), np.empty(n)
    _cells_pass(Ur, Ui, of, ou, ov, True)
    return of, ou, ov


@njit(cache=True)
def _rot(rr, x, y):
    if rr == 0:
        return x, y
    if rr == 1:
        return -y, x
    if rr == 2:
        return -x, -y
    return y, -x


@njit(cache=True)
def _walk(Ur, Ui, nbr, rho, taur, taui, f, pr, pi_, qr, qi, maxit):
    """Straight walk in the parameter domain from p (in face f's chart) to q; returns (face, b0, b1, b2) of q, face
    -1 when the walk leaves the mesh, -2 when it does not end."""
    eps = 1e-10
    for _ in range(maxit):
        b0, b1, b2 = _bary(Ur, Ui, f, qr, qi)
        bmin = min(b0, min(b1, b2))
        if bmin >= -eps:
            return f, b0, b1, b2
        best = -1
        for k in range(3):
            bo = b2 if k == 0 else (b0 if k == 1 else b1)          # barycentric of the corner opposite edge k
            if bo >= -eps:
                continue
            k1 = (k + 1) % 3
            ax, ay = Ur[f, k] - pr, Ui[f, k] - pi_
            bx, by = Ur[f, k1] - pr, Ui[f, k1] - pi_
            dx, dy = qr - pr, qi - pi_
            o1 = dx * ay - dy * ax
            o2 = dx * by - dy * bx
            if o1 * o2 <= 0.0:
                best = k
                break
        if best < 0:
            if b0 <= b1 and b0 <= b2:
                best = 1
            elif b1 <= b2:
                best = 2
            else:
                best = 0
        g = nbr[f, best]
        if g < 0:
            if bmin > -1e-6:
                s = max(b0, 0.0) + max(b1, 0.0) + max(b2, 0.0)
                return f, max(b0, 0.0) / s, max(b1, 0.0) / s, max(b2, 0.0) / s
            return -1, 0.0, 0.0, 0.0
        rr = rho[f, best]
        x, y = _rot(rr, pr, pi_)
        pr, pi_ = x + taur[f, best], y + taui[f, best]
        x, y = _rot(rr, qr, qi)
        qr, qi = x + taur[f, best], y + taui[f, best]
        f = g
    return -2, 0.0, 0.0, 0.0


@njit(cache=True)
def _corners(P, T, Ur, Ui, nbr, rho, taur, taui, cf, cu, cv, maxit):
    """(C,4,3) corner positions of the cells (CCW in the map), (C,4) their faces and (C,) ok flags."""
    nc = len(cf)
    out = np.zeros((nc, 4, 3))
    face = np.full((nc, 4), -1, dtype=np.int64)
    ok = np.ones(nc, dtype=np.bool_)
    du = (-0.5, 0.5, 0.5, -0.5)
    dv = (-0.5, -0.5, 0.5, 0.5)
    for c in range(nc):
        for m in range(4):
            g, b0, b1, b2 = _walk(Ur, Ui, nbr, rho, taur, taui, cf[c], cu[c], cv[c], cu[c] + du[m], cv[c] + dv[m],
                                  maxit)
            if g < 0:
                ok[c] = False
                break
            x = b0 * P[T[g, 0]] + b1 * P[T[g, 1]] + b2 * P[T[g, 2]]
            out[c, m] = x / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
            face[c, m] = g
    return out, face, ok


def _cluster(x: np.ndarray, tol: float) -> np.ndarray:
    """Labels of points closer than tol (connected components of the close pairs)."""
    pairs = cKDTree(x).query_pairs(tol, output_type="ndarray")
    n = len(x)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def _valid_quads(pts: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Mask of quads with 4 distinct nodes, strictly convex and CCW (seen from outside)."""
    distinct = np.ones(len(quads), dtype=bool)
    for i in range(4):
        for j in range(i + 1, 4):
            distinct &= quads[:, i] != quads[:, j]
    q = pts[quads]
    c = q.sum(axis=1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    conv = np.ones(len(quads), dtype=bool)
    for k in range(4):
        e1 = q[:, (k + 1) % 4] - q[:, k]
        e2 = q[:, (k + 2) % 4] - q[:, (k + 1) % 4]
        conv &= np.einsum("ij,ij->i", np.cross(e1, e2), c) > 0
    return distinct & conv


def _manifold_quads(quads: np.ndarray, n: int) -> np.ndarray:
    """Mask: drop quads that share a directed edge with another quad, or an edge used by more than two quads."""
    keep = np.ones(len(quads), dtype=bool)
    while True:
        idx = np.flatnonzero(keep)
        a = quads[idx].ravel()
        b = np.roll(quads[idx], -1, axis=1).ravel()
        owner = np.repeat(idx, 4)
        dkey = a * n + b
        ukey = np.minimum(a, b) * n + np.maximum(a, b)
        _, dinv, dcnt = np.unique(dkey, return_inverse=True, return_counts=True)
        _, uinv, ucnt = np.unique(ukey, return_inverse=True, return_counts=True)
        bad = (dcnt[dinv] > 1) | (ucnt[uinv] > 2)
        if not bad.any():
            return keep
        keep[owner[bad]] = False


def extract(P: np.ndarray, T: np.ndarray, prm, coast: np.ndarray, coast_edges: np.ndarray, fold_rings: int = 2,
            maxit: int = 2000):
    """Quad mesh of the integer-grid map prm on the background mesh (P, T).

    coast: (B,3) coast nodes (they become nodes 0..B-1), coast_edges (B,2) into them (water on the left).
    Cells touching a folded face or its fold_rings face rings are dropped. Returns (points, tris, quads, info); tris
    fill what the quads leave.
    """
    info = {}
    Ur = np.ascontiguousarray(prm.W.real)
    Ui = np.ascontiguousarray(prm.W.imag)
    taur = np.ascontiguousarray(prm.tau.real)
    taui = np.ascontiguousarray(prm.tau.imag)
    bad = prm.flipped.copy()
    for _ in range(fold_rings):
        nb_ = prm.nbr[bad].ravel()
        bad[nb_[nb_ >= 0]] = True
    cf, cu, cv = _cells(Ur, Ui)
    # a centre on a shared edge is found from both faces: keep one
    bc = _bary_np(Ur, Ui, cf, cu, cv)
    cpos = np.einsum("ck,ckd->cd", bc, P[T[cf]])
    cpos /= np.linalg.norm(cpos, axis=1, keepdims=True)
    lab = _cluster(cpos, MERGE_TOL)
    _, first = np.unique(lab, return_index=True)
    cf, cu, cv = cf[first], cu[first], cv[first]
    info["n_cells"] = len(cf)
    corners, cface, ok = _corners(P, T, Ur, Ui, prm.nbr, prm.rho.astype(np.int64), taur, taui, cf, cu, cv, maxit)
    info["n_cells_walk_failed"] = int((~ok).sum())
    near_fold = bad[cf] | (bad[np.maximum(cface, 0)] & (cface >= 0)).any(axis=1)
    info["n_cells_near_fold"] = int((ok & near_fold).sum())
    corners = corners[ok & ~near_fold]
    nb = len(coast)
    # merge corners by position; coast nodes first
    allp = np.vstack([coast, corners.reshape(-1, 3)]) if nb else corners.reshape(-1, 3)
    lab = _cluster(allp, MERGE_TOL)
    ulab, first, inv = np.unique(lab, return_index=True, return_inverse=True)
    # node order: clusters holding a coast node first, in coast order
    rank = np.full(len(ulab), -1, dtype=np.int64)
    if nb:
        cl_coast = inv[:nb]
        info["coast_merged_dup"] = int(nb - len(np.unique(cl_coast)))
        rank[cl_coast] = np.arange(nb)
    rest = np.flatnonzero(rank < 0)
    rank[rest] = nb + np.arange(len(rest))
    pts = np.empty((nb + len(rest), 3))
    pts[rank] = allp[first]
    if nb:
        pts[:nb] = coast
    quads = rank[inv[nb:]].reshape(-1, 4)
    v = _valid_quads(pts, quads)
    info["n_quads_invalid"] = int((~v).sum())
    quads = quads[v]
    m = _manifold_quads(quads, len(pts))
    info["n_quads_nonmanifold"] = int((~m).sum())
    quads = quads[m]
    # fill, then drop the quads that the constrained triangulation does not split into exactly two triangles (an
    # overlap or a node inside), and fill again
    n_drop = 0
    for _ in range(6):
        used = np.zeros(len(pts), dtype=bool)
        used[:nb] = True
        used[quads.ravel()] = True
        new = np.cumsum(used) - 1
        pts, quads = pts[used], new[quads]
        tris, cover, finfo = fill(pts, quads, coast_edges, nb)
        if (cover == 2).all():
            break
        n_drop += int((cover != 2).sum())
        quads = quads[cover == 2]
    info.update(finfo, n_quads_dropped_fill=n_drop)
    info["n_coast_in_quads"] = int(np.isin(np.arange(nb), quads.ravel()).sum()) if nb else 0
    return pts, tris, quads, info


def _bary_np(Ur, Ui, f, qr, qi):
    u0, v0 = Ur[f, 0], Ui[f, 0]
    u1, v1 = Ur[f, 1], Ui[f, 1]
    u2, v2 = Ur[f, 2], Ui[f, 2]
    det = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)
    l1 = ((qr - u0) * (v2 - v0) - (u2 - u0) * (qi - v0)) / det
    l2 = ((u1 - u0) * (qi - v0) - (qr - u0) * (v1 - v0)) / det
    return np.column_stack([1.0 - l1 - l2, l1, l2])


def fill(pts: np.ndarray, quads: np.ndarray, coast_edges: np.ndarray, nb: int):
    """Triangles covering what the quads leave uncovered: Delaunay of all nodes with the quad edges and the coast as
    constraints, land removed, the triangles inside each quad removed. Returns (tris, cover (Q,) number of triangles
    inside each quad (2 when it is a cell of the triangulation), info)."""
    n = len(pts)
    qe = np.column_stack([quads.ravel(), np.roll(quads, -1, axis=1).ravel()])
    cons = np.vstack([qe, coast_edges]) if nb else qe
    cons = np.unique(np.sort(cons, axis=1), axis=0)
    tris = sphere_delaunay(pts, cons, coast_edges if nb else cons[:0])

    def key(t):
        s = np.sort(t, axis=1)
        return (s[:, 0] * n + s[:, 1]) * n + s[:, 2]

    nq = len(quads)
    qk = np.concatenate([key(quads[:, [0, 1, 2]]), key(quads[:, [0, 2, 3]]), key(quads[:, [0, 1, 3]]),
                         key(quads[:, [1, 2, 3]])])
    qid = np.tile(np.arange(nq), 4)
    order = np.argsort(qk, kind="stable")
    sk = qk[order]
    tk = key(tris)
    pos = np.minimum(np.searchsorted(sk, tk), max(len(sk) - 1, 0))
    covered = (sk[pos] == tk) if nq else np.zeros(len(tris), dtype=bool)
    cover = np.bincount(qid[order[pos[covered]]], minlength=nq)
    info = {"n_fill_tris": int((~covered).sum())}
    return tris[~covered], cover, info
