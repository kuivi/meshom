"""m09 core: face cross field, cut graph, seamless parametrisation with integer rounding (MIQ, Bommes et al. 2009;
QuadCover, Kälberer et al. 2007) on a background triangle mesh of the sphere.

Notation. The background mesh is a polyhedron of flat chord triangles. Face f has an orthonormal frame (b1, b2) in its
plane (outward normal) and complex 2D corner coordinates x[f, k]. The map is complex, w = u + i v. In face f the target
is the conformal map with |dw/dx| = 1/h along the face's cross direction Theta_f:  dw = s_f e^{-i Theta_f} dx,
s_f = R / h_f. The least-squares energy sum_f A_f |G_f W_f - c_f|^2, c_f = s_f e^{-i Theta_f} (1, i), is a Hermitian
form with the real cotangent Laplacian (|grad u - s d|^2 + |grad v - s d^perp|^2 in complex form).

Seams. The field is combed along a breadth-first dual spanning tree; the remaining edges are the cut candidates, pruned
at non-singular leaves (closed sphere: a tree through the cones; with a coast: the coast loops, the trees joining them
and the cones). Across a cut edge from its left face f to its right face g the charts differ by w_g = rho w_f + t,
rho = i^{-r} (r = the field's matching) and t a translation, constant along each chain of the cut graph.

Unknowns: one w per sector (the corners of a vertex between two cut edges), one t per chain. Eliminated directly:
the right-side sector of every chain-interior vertex (w_R = rho w_L + t) and every coast sector after the first of its
coast piece (the coast is an integer iso-line: each coast edge is one unit step delta in {1, i, -1, -i}, so the coast
nodes resampled at h land on integer points). The rest - the chain-end constraints at cut-graph nodes - is a small
sparse system eliminated by Gauss-Jordan over Gaussian integers (pivots: continuous sectors first, then chain
translations, then integer sectors), so the dependent translations stay integer combinations of the free integers.

Integers: one sector per cone, one per coast piece and the chain translations left free. MIQ greedy rounding on the
Schur complement of the continuous unknowns (one sparse LU): the integer unknown nearest to a Gaussian integer is fixed,
the dense reduced system re-solved, repeat.
"""
from dataclasses import dataclass, field

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.sparse.linalg import splu

from meshom.geo import R_EARTH_M

HALF_PI = 0.5 * np.pi
# Ncc is Hermitian positive definite: geometric nested-dissection order, natural column order, diagonal pivots
SPLU_KW = dict(permc_spec="NATURAL", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
ND_LEAF = 256
SCHUR_BLOCK = 16          # dense right-hand sides per solve for the Schur complement (memory: 16 B x n x block)


def nd_order(xyz: np.ndarray, a: csr_matrix) -> np.ndarray:
    """Nested-dissection ordering of the unknowns at positions xyz with the sparsity of a: median plane cuts along the
    widest axis; the left vertices touching the right part form the separator, numbered after both halves."""
    a = a.tocsr()
    side = np.zeros(len(xyz), dtype=np.int8)
    out = []
    stack = [(np.arange(len(xyz)), False)]
    while stack:
        idx, emit = stack.pop()
        if emit:
            out.append(idx)
            continue
        if len(idx) <= ND_LEAF:
            out.append(idx)
            continue
        q = xyz[idx]
        ax = int(np.argmax(q.max(axis=0) - q.min(axis=0)))
        med = np.median(q[:, ax])
        lt = q[:, ax] < med
        left, right = idx[lt], idx[~lt]
        if len(left) == 0 or len(right) == 0:
            out.append(idx)
            continue
        side[right] = 1
        sub = a[left]
        touch = np.zeros(len(left), dtype=bool)
        rows = np.repeat(np.arange(len(left)), np.diff(sub.indptr))
        touch[rows[side[sub.indices] == 1]] = True
        side[right] = 0
        stack.append((left[touch], True))                                  # separator: emitted last
        stack.append((right, False))
        stack.append((left[~touch], False))
    return np.concatenate(out)


class _LU:
    """splu of a Hermitian positive definite matrix in nested-dissection order (unknowns marked tail go last)."""

    def __init__(self, a, xyz, tail):
        a = a.tocsr()
        body = np.flatnonzero(~tail)
        self.p = np.r_[body[nd_order(xyz[body], a[body][:, body])], np.flatnonzero(tail)]
        self.lu = splu(a[self.p][:, self.p].tocsc(), **SPLU_KW)

    def solve(self, b):
        x = np.empty_like(b, dtype=np.complex128)
        x[self.p] = self.lu.solve(np.asarray(b, dtype=np.complex128)[self.p])
        return x


@dataclass
class Param:
    W: np.ndarray            # (F,3) complex corner coordinates of the map
    nbr: np.ndarray          # (F,3) face across half-edge k (T[f,k] -> T[f,k+1]); -1 = boundary
    rho: np.ndarray          # (F,3) int, transition rotation power: w_g = i^rho w_f + tau
    tau: np.ndarray          # (F,3) complex transition translation
    flipped: np.ndarray      # (F,) bool: face mapped with non-positive orientation
    info: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ geometry

def face_geometry(P: np.ndarray, T: np.ndarray):
    """Frames (b1, b2), complex corner coordinates x (F,3), areas (F,), P1 gradients G (F,2,3) of the chord faces."""
    q = P[T]
    e01 = q[:, 1] - q[:, 0]
    e02 = q[:, 2] - q[:, 0]
    nrm = np.cross(e01, e02)
    two_a = np.linalg.norm(nrm, axis=1)
    n = nrm / two_a[:, None]
    l01 = np.linalg.norm(e01, axis=1)
    b1 = e01 / l01[:, None]
    b2 = np.cross(n, b1)
    x = np.zeros((len(T), 3), dtype=np.complex128)
    x[:, 1] = l01
    x[:, 2] = np.einsum("ij,ij->i", e02, b1) + 1j * np.einsum("ij,ij->i", e02, b2)
    G = np.zeros((len(T), 2, 3))
    for k in range(3):
        d = x[:, (k + 2) % 3] - x[:, (k + 1) % 3]                       # grad phi_k = rot90(opposite edge) / 2A
        G[:, 0, k] = -d.imag / two_a
        G[:, 1, k] = d.real / two_a
    return b1, b2, x, 0.5 * two_a, G


def half_edges(T: np.ndarray, n: int):
    """Half-edge 3f+k runs T[f,k] -> T[f,k+1]: (a, b, twin (-1 on the boundary), undirected edge id)."""
    a = T.ravel()
    b = np.roll(T, -1, axis=1).ravel()
    key, rkey = a * n + b, b * n + a
    order = np.argsort(key, kind="stable")
    sk = key[order]
    pos = np.minimum(np.searchsorted(sk, rkey), len(sk) - 1)
    twin = np.where(sk[pos] == rkey, order[pos], -1)
    _, eid = np.unique(np.minimum(a, b) * n + np.maximum(a, b), return_inverse=True)
    return a, b, twin, eid.ravel()


def face_field(T: np.ndarray, b1: np.ndarray, b2: np.ndarray, u: np.ndarray) -> np.ndarray:
    """(F,) angle of a representative cross direction per face: mean of the 4th powers of its nodes' directions."""
    z = np.zeros(len(T), dtype=np.complex128)
    for k in range(3):
        d = u[T[:, k]]
        z += np.exp(4j * np.arctan2(np.einsum("ij,ij->i", d, b2), np.einsum("ij,ij->i", d, b1)))
    return np.angle(z) / 4.0


@njit(cache=True)
def _comb(order, pred_he, twin, theta, tau):
    """Turn each face's representative by multiples of pi/2 so that the matching to its BFS parent is 0."""
    for f in order:
        he = pred_he[f]
        if he < 0:
            continue
        tw = twin[he]
        p = tw // 3
        r = np.round((theta[f] - theta[p] - tau[tw]) / (0.5 * np.pi))
        theta[f] -= r * 0.5 * np.pi


@njit(cache=True)
def _prune(n, eu, ev, keep, ptr, inc, protected):
    """Remove cut-candidate edges hanging at unprotected degree-1 vertices, repeatedly (in place on keep)."""
    deg = np.zeros(n, dtype=np.int64)
    for e in range(len(eu)):
        if keep[e]:
            deg[eu[e]] += 1
            deg[ev[e]] += 1
    stack = np.empty(n, dtype=np.int64)
    top = 0
    for v in range(n):
        if deg[v] == 1 and not protected[v]:
            stack[top] = v
            top += 1
    while top > 0:
        top -= 1
        v = stack[top]
        if deg[v] != 1 or protected[v]:
            continue
        for k in range(ptr[v], ptr[v + 1]):
            e = inc[k]
            if keep[e]:
                keep[e] = False
                w = ev[e] if eu[e] == v else eu[e]
                deg[v] -= 1
                deg[w] -= 1
                if deg[w] == 1 and not protected[w]:
                    stack[top] = w
                    top += 1
                break


def _csr_incidence(n, eu, ev, sel):
    idx = np.flatnonzero(sel)
    rows = np.r_[eu[idx], ev[idx]]
    cols = np.r_[idx, idx]
    order = np.argsort(rows, kind="stable")
    ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=n))].astype(np.int64)
    return ptr, cols[order].astype(np.int64)


# ------------------------------------------------------------------------------------------------ Gaussian elimination

def _gauss_jordan(rows: list, kind: np.ndarray, tol: float = 1e-9):
    """rows: list of (dict var -> complex coeff, const) meaning sum coeff var = const. Pivot preference: continuous
    (kind 0) unit coefficients, then translations (1), then integer sectors (2); any nonzero coefficient after the
    units. Returns (pivots {var: row}, rows reduced, n_redundant, max inconsistency, n_nonunit)."""
    col_rows: dict[int, set] = {}
    for r, (d, _) in enumerate(rows):
        for v in d:
            col_rows.setdefault(v, set()).add(r)
    pivots = {}
    n_red, bad, n_nonunit = 0, 0.0, 0
    for r in range(len(rows)):
        d, c = rows[r]
        for v in [v for v, a in d.items() if abs(a) < tol]:
            del d[v]
            col_rows[v].discard(r)
        if not d:
            n_red += 1
            bad = max(bad, abs(c))
            continue
        best, key = None, None
        for v, a in d.items():
            k = (0 if abs(abs(a) - 1.0) < 1e-9 else 1, int(kind[v]), v)
            if key is None or k < key:
                best, key = v, k
        n_nonunit += key[0]
        a = d[best]
        d = {v: x / a for v, x in d.items()}
        c = c / a
        rows[r] = (d, c)
        for r2 in list(col_rows[best]):
            if r2 == r:
                continue
            d2, c2 = rows[r2]
            f = d2[best]
            for v, x in d.items():
                y = d2.get(v, 0.0) - f * x
                if abs(y) < tol:
                    if v in d2:
                        del d2[v]
                        col_rows[v].discard(r2)
                else:
                    if v not in d2:
                        col_rows.setdefault(v, set()).add(r2)
                    d2[v] = y
            rows[r2] = (d2, c2 - f * c)
        pivots[best] = r
    return pivots, rows, n_red, bad, n_nonunit


# ------------------------------------------------------------------------------------------------ main

def parametrize(P: np.ndarray, T: np.ndarray, u: np.ndarray, hv: np.ndarray, rounding: str = "greedy",
                batch_frac: float = 0.0, stiffen: int = 0, stiffen_post: int = 0, stiffen_c: float = 1.0,
                free_boundary: bool = False) -> Param:
    """Seamless integer-grid map of the background mesh (P, T) for the node cross field u and node sizes hv (m).

    free_boundary=True (m14's block map): the mesh boundary is not an integer iso-line (no coast pieces, no unit
    steps); cut chains still end on it. Without cones the map is then translation-free, so one sector is pinned at
    w = 0 (a disc without cones has no cuts: one least-squares solve, no integers)."""
    nn, nf = len(P), len(T)
    info = {}
    b1, b2, x, area, G = face_geometry(P, T)
    a, b, twin, eid = half_edges(T, nn)
    ne = int(eid.max()) + 1
    he_f = np.arange(3 * nf) // 3
    interior = twin >= 0
    nbr = np.where(interior, twin // 3, -1).reshape(nf, 3)
    ang = np.angle(np.roll(x, -1, axis=1) - x).ravel()                    # direction of each half-edge in its face
    tau = np.zeros(3 * nf)
    tau[interior] = ang[twin[interior]] + np.pi - ang[interior]           # frame change f -> g across the half-edge

    # face field, combed along a BFS dual tree
    theta = face_field(T, b1, b2, u)
    g = csr_matrix((np.ones(int(interior.sum())), (he_f[interior], twin[interior] // 3)), shape=(nf, nf))
    order, pred = breadth_first_order(g, 0, directed=False, return_predecessors=True)
    info["dual_components"] = int(connected_components(g, directed=False)[0])
    pred_he = np.full(nf, -1, dtype=np.int64)
    has = pred >= 0
    kk = np.argmax(nbr[has] == pred[has, None], axis=1)
    pred_he[has] = 3 * np.flatnonzero(has) + kk
    _comb(order.astype(np.int64), pred_he, twin.astype(np.int64), theta, tau)

    # matchings and singular vertices
    tg = np.zeros(3 * nf)
    tg[interior] = theta[twin[interior] // 3]
    dth = tg - theta[he_f] - tau
    r = np.where(interior, np.round(dth / HALF_PI), 0).astype(np.int64)
    resid = np.where(interior, dth - r * HALF_PI, 0.0)
    bnd_v = np.zeros(nn, dtype=bool)
    bnd_v[a[~interior]] = True
    crossed = (3 * he_f + (np.arange(3 * nf) % 3 + 2) % 3)                 # corner k of f crosses half-edge k+2
    corner_ang = np.abs(np.angle((np.roll(x, -1, axis=1) - x) / (np.roll(x, 1, axis=1) - x))).ravel()
    defect = 2.0 * np.pi - np.bincount(T.ravel(), corner_ang, minlength=nn)
    index = (defect + np.bincount(T.ravel(), resid[crossed], minlength=nn)) / (2.0 * np.pi)
    rsum = np.bincount(T.ravel(), r[crossed], minlength=nn)
    singular = (np.mod(rsum, 4) != 0) & ~bnd_v
    index[bnd_v] = 0.0
    info["n_cones"] = int(singular.sum())
    info["cone_index_sum"] = float(np.round(4 * index[singular]).sum() / 4)
    info["n_cones_plus"] = int((index[singular] > 0).sum())

    # cut graph: edges not crossed by the dual tree, pruned at non-singular leaves
    eu = np.zeros(ne, dtype=np.int64)
    ev = np.zeros(ne, dtype=np.int64)
    eu[eid], ev[eid] = a, b
    in_tree = np.zeros(ne, dtype=bool)
    in_tree[eid[pred_he[has]]] = True
    keep = ~in_tree
    ptr, inc = _csr_incidence(nn, eu, ev, keep)
    _prune(nn, eu, ev, keep, ptr, inc, singular | bnd_v)
    e_bnd = np.zeros(ne, dtype=bool)
    e_bnd[eid[~interior]] = True
    cut = keep & ~e_bnd                                                    # interior seams
    info["n_cut_edges"] = int(cut.sum())
    info["_cut"] = np.column_stack([eu[cut], ev[cut]])

    # sectors: corners joined across non-cut interior edges
    he_cut = cut[eid]
    sel = np.flatnonzero(interior & ~he_cut)
    tw = twin[sel]
    ca = np.r_[sel, 3 * (sel // 3) + (sel % 3 + 1) % 3]                     # corners of a and b in f
    cb = np.r_[3 * (tw // 3) + (tw % 3 + 1) % 3, tw]                        # the same vertices in g
    sec = connected_components(coo_matrix((np.ones(len(ca)), (ca, cb)), shape=(3 * nf, 3 * nf)),
                               directed=False)[1].astype(np.int64)
    ns = int(sec.max()) + 1
    sec_v = np.zeros(ns, dtype=np.int64)
    sec_v[sec] = T.ravel()

    # chains of the cut graph between nodes (degree != 2, cones, coast vertices)
    cdeg = np.bincount(np.r_[eu[cut], ev[cut]], minlength=nn)
    node = (cdeg > 0) & ((cdeg != 2) | singular | bnd_v)
    cptr, cinc = _csr_incidence(nn, eu, ev, cut)
    he_of_edge = np.full((ne, 2), -1, dtype=np.int64)                      # half-edge a->b with a = eu, and its twin
    fwd = a == eu[eid]
    he_of_edge[eid[fwd], 0] = np.flatnonzero(fwd)
    he_of_edge[eid[~fwd], 1] = np.flatnonzero(~fwd)
    visited = np.zeros(ne, dtype=bool)
    chains = []

    def trace(v, e):
        seq = []
        cur = v
        while True:
            visited[e] = True
            h = he_of_edge[e, 0] if eu[e] == cur else he_of_edge[e, 1]
            seq.append(h)
            cur = b[h]
            if node[cur]:
                return seq
            nxt = [cinc[k] for k in range(cptr[cur], cptr[cur + 1]) if not visited[cinc[k]]]
            if not nxt:
                return seq
            e = nxt[0]

    for v in np.flatnonzero(node):
        for k in range(cptr[v], cptr[v + 1]):
            if not visited[cinc[k]]:
                chains.append(trace(v, cinc[k]))
    for e in np.flatnonzero(cut & ~visited):                              # cut cycles without a node
        if not visited[e]:
            node[eu[e]] = True
            chains.append(trace(eu[e], e))
    nc = len(chains)
    info["n_chains"] = nc

    # sector types: 0 own unknown, 1 = rho * w_L + t_c (chain interior, right side), 2 = anchor + offset (coast)
    stype = np.zeros(ns, dtype=np.int64)
    s_ref = np.full(ns, -1, dtype=np.int64)
    s_rho = np.ones(ns, dtype=np.complex128)
    s_chain = np.full(ns, -1, dtype=np.int64)
    s_off = np.zeros(ns, dtype=np.complex128)
    ends = []                                                             # (R sector, L sector, rho, chain)
    rho_bad = 0
    units = np.array([1, 1j, -1, -1j])
    for c, seq in enumerate(chains):
        seq = np.asarray(seq)
        tws = twin[seq]
        rr = np.mod(-r[seq], 4)
        rho_bad += int((rr != rr[0]).sum())
        rho = units[rr[0]]
        lA = sec[seq]                                                     # L at a_j
        lB = sec[3 * (seq // 3) + (seq % 3 + 1) % 3]                      # L at b_j
        rA = sec[3 * (tws // 3) + (tws % 3 + 1) % 3]                      # R at a_j
        rB = sec[tws]                                                     # R at b_j
        ends.append((rA[0], lA[0], rho, c))
        ends.append((rB[-1], lB[-1], rho, c))
        mid_r, mid_l = rB[:-1], lB[:-1]
        stype[mid_r] = 1
        s_ref[mid_r] = mid_l
        s_rho[mid_r] = rho
        s_chain[mid_r] = c
    info["rho_inconsistent"] = rho_bad

    # coast pieces: delta = unit step per coast edge, sectors after the first of each piece eliminated
    bhe = np.flatnonzero(~interior)
    if len(bhe) and not free_boundary:
        f_b = bhe // 3
        dx = (np.roll(x, -1, axis=1) - x).ravel()[bhe]
        s_f = R_EARTH_M / np.exp(np.log(hv)[T]).mean(axis=1)
        dw = s_f[f_b] * np.exp(-1j * theta[f_b]) * dx
        q = np.mod(np.round(np.angle(dw) / HALF_PI).astype(np.int64), 4)
        delta = units[q]
        info["coast_misalign_deg"] = float(np.degrees(np.abs(np.angle(dw / delta))).max())
        info["coast_step_p99"] = float(np.percentile(np.abs(dw), 99))
        start_s = sec[bhe]
        end_s = sec[3 * f_b + (bhe % 3 + 1) % 3]
        out_of = np.full(nn, -1, dtype=np.int64)
        out_of[a[bhe]] = np.arange(len(bhe))
        nxt = out_of[b[bhe]]
        prv = np.empty_like(nxt)
        prv[nxt] = np.arange(len(bhe))
        brk = end_s[prv] != start_s                                        # a piece starts at this coast edge
        done = np.zeros(len(bhe), dtype=bool)
        anchors = []
        closing_bad = 0.0
        # pieces start at breaks (a cut chain meets the coast); a loop without a break is one closed piece
        for i0 in list(np.flatnonzero(brk)) + list(range(len(bhe))):
            if done[i0]:
                continue
            anchor = start_s[i0]
            anchors.append(anchor)
            off = 0.0 + 0.0j
            j = i0
            while True:
                done[j] = True
                off += delta[j]
                s = end_s[j]
                if s == anchor:
                    closing_bad = max(closing_bad, abs(off))
                    break
                stype[s] = 2
                s_ref[s] = anchor
                s_off[s] = off
                j = nxt[j]
                if done[j] or brk[j]:
                    break
        info["coast_closing_bad"] = closing_bad
        anchors = np.asarray(anchors, dtype=np.int64)
    else:
        anchors = np.zeros(0, dtype=np.int64)
    info["n_coast_pieces"] = len(anchors)

    # stage-1 unknowns: own sectors, then chain translations
    own = np.flatnonzero(stype == 0)
    var_of = np.full(ns, -1, dtype=np.int64)
    var_of[own] = np.arange(len(own))
    n1 = len(own) + nc
    kind = np.zeros(n1, dtype=np.int64)                                    # 0 continuous, 1 translation, 2 integer
    kind[len(own):] = 1
    kind[var_of[anchors]] = 2
    cone_sec = {}
    for s in own:
        v = sec_v[s]
        if singular[v] and v not in cone_sec:
            cone_sec[v] = s
    kind[var_of[np.fromiter(cone_sec.values(), dtype=np.int64, count=len(cone_sec))]] = 2
    t_var = len(own) + np.arange(nc)

    rows_e, cols_e, vals_e = [], [], []
    k1 = np.zeros(ns, dtype=np.complex128)
    rows_e.append(own); cols_e.append(var_of[own]); vals_e.append(np.ones(len(own), dtype=np.complex128))
    m1 = np.flatnonzero(stype == 1)
    rows_e += [m1, m1]
    cols_e += [var_of[s_ref[m1]], t_var[s_chain[m1]]]
    vals_e += [s_rho[m1], np.ones(len(m1), dtype=np.complex128)]
    m2 = np.flatnonzero(stype == 2)
    rows_e.append(m2); cols_e.append(var_of[s_ref[m2]]); vals_e.append(np.ones(len(m2), dtype=np.complex128))
    k1[m2] = s_off[m2]
    E1 = coo_matrix((np.concatenate(vals_e), (np.concatenate(rows_e), np.concatenate(cols_e))),
                    shape=(ns, n1)).tocsr()

    # stage 2: chain-end constraints  w_R - rho w_L - t_c = 0
    def expand(s):
        if stype[s] == 0:
            return {int(var_of[s]): 1.0 + 0j}, 0j
        if stype[s] == 2:
            return {int(var_of[s_ref[s]]): 1.0 + 0j}, s_off[s]
        raise RuntimeError("chain-interior sector at a chain end")

    rows, seen = [], set()
    for sR, sL, rho, c in ends:
        key = (int(sR), int(sL), int(c))
        if key in seen:
            continue
        seen.add(key)
        dR, cR = expand(sR)
        dL, cL = expand(sL)
        d = dict(dR)
        for v, x_ in dL.items():
            d[v] = d.get(v, 0.0) - rho * x_
        d[int(t_var[c])] = d.get(int(t_var[c]), 0.0) - 1.0
        rows.append((d, -(cR - rho * cL)))
    if free_boundary and not (kind > 1).any():                             # pin the translation: w(sector 0) = 0
        rows.append(({int(var_of[own[0]]): 1.0 + 0j}, 0j))
    pivots, rows, n_red, bad, n_nonunit = _gauss_jordan(rows, kind)
    info.update(n_constraints=len(rows), n_redundant=n_red, inconsistency=float(bad), n_nonunit_pivots=n_nonunit)

    free = np.ones(n1, dtype=bool)
    free[list(pivots)] = False
    col = np.full(n1, -1, dtype=np.int64)
    col[free] = np.arange(int(free.sum()))
    r2, c2, v2 = list(np.flatnonzero(free)), list(col[free]), [1.0 + 0j] * int(free.sum())
    k2 = np.zeros(n1, dtype=np.complex128)
    for p, ri in pivots.items():
        d, c = rows[ri]
        k2[p] = c
        for v, x_ in d.items():
            if v != p:
                r2.append(p); c2.append(col[v]); v2.append(-x_)
    nfree = int(free.sum())
    S2 = coo_matrix((np.asarray(v2), (np.asarray(r2), np.asarray(c2))), shape=(n1, nfree)).tocsr()
    E = (E1 @ S2).tocsr()
    k = E1 @ k2 + k1
    is_int = (kind[free] > 0)
    info.update(n_unknowns=nfree, n_integer=int(is_int.sum()), n_sectors=ns)

    # energy: sum_f wt_f A_f |G_f W_f - c_f|^2
    s_f = R_EARTH_M / np.exp(np.log(hv)[T]).mean(axis=1)
    cf = s_f[:, None] * np.exp(-1j * theta)[:, None] * np.array([1.0, 1j])[None, :]
    K0 = area[:, None, None] * np.einsum("fdi,fdj->fij", G, G)
    b0 = area[:, None] * np.einsum("fdi,fd->fi", G, cf)
    sc = sec.reshape(nf, 3)
    li, lj = np.repeat(sc, 3, axis=1).ravel(), np.tile(sc, (1, 3)).ravel()
    EH = E.conj().T.tocsr()
    ci = np.flatnonzero(~is_int)
    ii = np.flatnonzero(is_int)
    # dependent integers (pivots of kind > 0) as functions of the free integers: z = D y_i + kd must be in Z[i]
    col_i = np.full(nfree, -1, dtype=np.int64)
    col_i[ii] = np.arange(len(ii))
    dep = [p for p in pivots if kind[p] > 0]
    D = np.zeros((len(dep), len(ii)), dtype=np.complex128)
    kd = np.zeros(len(dep), dtype=np.complex128)
    for m_, p in enumerate(dep):
        d, c = rows[pivots[p]]
        kd[m_] = c
        for v, x_ in d.items():
            if v != p and col_i[col[v]] >= 0:
                D[m_, col_i[col[v]]] = -x_

    # positions of the free unknowns for the nested-dissection order
    var_pos = np.zeros((n1, 3))
    var_pos[var_of[own]] = P[sec_v[own]]
    pos_free = var_pos[np.flatnonzero(free)]

    def normal_eq(wt):
        L = coo_matrix(((wt[:, None, None] * K0).ravel(), (li, lj)), shape=(ns, ns)).tocsr()
        bw = wt[:, None] * b0
        bvec = np.bincount(sc.ravel(), bw.real.ravel(), minlength=ns) \
            + 1j * np.bincount(sc.ravel(), bw.imag.ravel(), minlength=ns)
        return (EH @ L @ E).tocsr(), EH @ (bvec - L @ k)

    def schur(N, rhs):
        Ncc = N[ci][:, ci]
        Nci = N[ci][:, ii].tocsc()
        lu = _LU(Ncc, pos_free[ci], np.zeros(len(ci), dtype=bool))
        zc = lu.solve(rhs[ci])
        S = N[ii][:, ii].toarray()
        for j0 in range(0, len(ii), SCHUR_BLOCK):
            S[:, j0:j0 + SCHUR_BLOCK] -= Nci.conj().T @ lu.solve(Nci[:, j0:j0 + SCHUR_BLOCK].toarray())
        S = 0.5 * (S + S.conj().T)
        ri = rhs[ii] - Nci.conj().T @ zc
        eps = 1e-10 * max(float(np.abs(np.diag(S)).mean()) if len(ii) else 1.0, 1e-300)
        return lu, Nci, S, ri, eps

    def with_integers(yi, lu, Nci, rhs):
        y = np.zeros(nfree, dtype=np.complex128)
        y[ii] = yi
        y[ci] = lu.solve(rhs[ci] - Nci @ yi)
        return y, (E @ y + k)

    def distortion(w):
        J = np.einsum("fdi,fi->fd", G, w[sc]) / s_f[:, None]               # (grad u + i grad v) / s
        gu, gv = J.real, J.imag
        det = gu[:, 0] * gv[:, 1] - gu[:, 1] * gv[:, 0]
        fro = (gu ** 2 + gv ** 2).sum(axis=1)
        dsc = np.sqrt(np.maximum(fro ** 2 - 4 * det ** 2, 0.0))
        s1 = np.sqrt(0.5 * (fro + dsc))
        s2 = np.sign(det) * np.sqrt(np.maximum(0.5 * (fro - dsc), 0.0))
        return np.abs(s1 - 1.0) + np.abs(s2 - 1.0), det

    # local stiffening (MIQ §5) on the flipped faces and their rings: continuous map first (integers relaxed, through
    # the Schur complement), then with the integers fixed
    wt = np.ones(nf)
    for it in range(stiffen + 1):
        N, rhs = normal_eq(wt)
        lu = None                                                         # one factorisation in memory at a time
        lu, Nci, S, ri, eps = schur(N, rhs)
        yi0 = np.linalg.solve(S + eps * np.eye(len(ii)), ri) if len(ii) else np.zeros(0, dtype=np.complex128)
        dist, det = distortion(with_integers(yi0, lu, Nci, rhs)[1])
        if it == 0:
            info["n_flipped_continuous0"] = int((det <= 0).sum())
        if it == stiffen or not (det <= 0).any():
            break
        wt = _stiffen(wt, dist, det, nbr, stiffen_c)
    info.update(n_flipped_continuous=int((det <= 0).sum()), stiffen_iters=it)
    yi = _round(S, ri, eps, rounding, batch_frac, info, D, kd)
    y, w = with_integers(yi, lu, Nci, rhs)
    del S
    n_post, prev = 0, None
    for it in range(stiffen_post):
        dist, det = distortion(w)
        nflip = int((det <= 0).sum())
        if nflip == 0 or (prev is not None and nflip >= prev):            # done, or stuck at the rounded cones
            break
        prev = nflip
        n_post += 1
        wt = _stiffen(wt, dist, det, nbr, stiffen_c)
        N, rhs = normal_eq(wt)
        del lu                                                            # one factorisation in memory at a time
        lu = _LU(N[ci][:, ci], pos_free[ci], np.zeros(len(ci), dtype=bool))
        y, w = with_integers(yi, lu, N[ci][:, ii].tocsc(), rhs)
    info["stiffen_post_iters"] = n_post
    info["stiffen_max_weight"] = float(wt.max())

    # integrality of the dependent integer unknowns (translations, cones, coast anchors)
    y1 = S2 @ y + k2
    iv = kind > 0
    frac = np.abs(y1[iv].real - np.round(y1[iv].real)) + np.abs(y1[iv].imag - np.round(y1[iv].imag))
    info["int_frac_max"] = float(frac.max()) if len(frac) else 0.0
    info["int_nonint"] = int((frac > 1e-6).sum())
    vid = np.flatnonzero(iv)[frac > 1e-6]
    var_sec = np.full(n1, -1, dtype=np.int64)
    var_sec[var_of[own]] = own
    info["_nonint"] = [(int(kind[j]), int(sec_v[var_sec[j]]) if var_sec[j] >= 0 else -1, complex(y1[j]))
                       for j in vid]
    Wt = w[sec.reshape(nf, 3)]
    info["_flipped_v"] = T[np.flatnonzero(((Wt[:, 1] - Wt[:, 0]).conj() * (Wt[:, 2] - Wt[:, 0])).imag <= 0), 0]
    info["_singular_v"] = np.flatnonzero(singular)
    info["_index"] = index[singular]

    W = w[sc]
    # transitions per half-edge from the corner values: w_g = i^rho w_f + tau
    rho = np.where(interior, np.mod(-r, 4), 0).reshape(nf, 3)
    tauv = np.zeros(3 * nf, dtype=np.complex128)
    hi = np.flatnonzero(interior)
    twi = twin[hi]
    wa_f = w[sec[hi]]                                                     # vertex a in f
    wa_g = w[sec[3 * (twi // 3) + (twi % 3 + 1) % 3]]                     # vertex a in g
    tauv[hi] = wa_g - units[rho.ravel()[hi]] * wa_f

    # distortion of the map against the target (per face, area weighted)
    J = np.einsum("fdi,fi->fd", G, W)                                     # complex gradient: (grad u + i grad v)
    gu, gv = J.real, J.imag
    det = gu[:, 0] * gv[:, 1] - gu[:, 1] * gv[:, 0]
    fro = (gu ** 2 + gv ** 2).sum(axis=1)
    disc = np.sqrt(np.maximum(fro ** 2 - 4 * det ** 2, 0.0))
    s1 = np.sqrt(0.5 * (fro + disc))
    s2 = np.sqrt(np.maximum(0.5 * (fro - disc), 0.0))
    info["n_flipped"] = int((det <= 0).sum())
    ok = det > 0
    lsc = np.log(np.sqrt(np.abs(det[ok])) / s_f[ok])
    info["log_scale_p50"] = float(np.percentile(np.abs(lsc), 50))
    info["log_scale_p99"] = float(np.percentile(np.abs(lsc), 99))
    info["aniso_p99"] = float(np.percentile(s1[ok] / np.maximum(s2[ok], 1e-300), 99))
    info["energy"] = float(np.sum(area * np.sum(np.abs(np.einsum("fdi,fi->fd", G, W) - cf) ** 2, axis=1)))
    info["energy_target"] = float(np.sum(area * np.sum(np.abs(cf) ** 2, axis=1)))
    return Param(W, nbr, rho, tauv.reshape(nf, 3), det <= 0, info)


def _stiffen(wt, dist, det, nbr, c, rings=2, cap=1e3):
    """Weights x (1 + c * distortion) on the flipped faces and `rings` face rings around them."""
    sel = det <= 0
    for _ in range(rings):
        nb = nbr[sel].ravel()
        sel = sel.copy()
        sel[nb[nb >= 0]] = True
    out = wt.copy()
    out[sel] = np.minimum(out[sel] * (1.0 + c * dist[sel]), cap)
    return out


def _frac(z: np.ndarray) -> np.ndarray:
    return np.abs(z.real - np.round(z.real)) + np.abs(z.imag - np.round(z.imag))


def _round(S: np.ndarray, r: np.ndarray, eps: float, rounding: str, batch_frac: float, info: dict,
           D: np.ndarray, kd: np.ndarray) -> np.ndarray:
    """Gaussian-integer values of the Schur-complement unknowns (energy y^H S y - 2 Re y^H r), MIQ greedy: fix the
    unknown nearest to Z[i], re-solve, repeat. Then the parity repair: a dependent integer z = D y + kd that came out
    fractional (a non-unit pivot, e.g. (1 - i) at the last cone) is fixed by the cheapest unit step of one free integer
    entering it with a fractional coefficient."""
    n = len(r)
    if n == 0:
        return np.zeros(0, dtype=np.complex128)
    A = S + eps * np.eye(n)
    y = np.linalg.solve(A, r)
    info["int_dist_before"] = float(np.mean(_frac(y)))
    if rounding == "direct":
        val = np.round(y.real) + 1j * np.round(y.imag)
    else:
        fixed = np.zeros(n, dtype=bool)
        val = np.zeros(n, dtype=np.complex128)
        steps = 0
        while not fixed.all():
            fr = np.flatnonzero(~fixed)
            yf = np.linalg.solve(A[np.ix_(fr, fr)], r[fr] - S[np.ix_(fr, np.flatnonzero(fixed))] @ val[fixed])
            m = max(1, int(batch_frac * len(fr))) if batch_frac > 0 else 1
            pick = np.argsort(_frac(yf), kind="stable")[:m]
            val[fr[pick]] = np.round(yf[pick].real) + 1j * np.round(yf[pick].imag)
            fixed[fr[pick]] = True
            steps += 1
        info["round_steps"] = steps

    def energy(v):
        return float(np.real(np.vdot(v, S @ v)) - 2.0 * np.real(np.vdot(v, r)))

    steps = [1, -1, 1j, -1j, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
    n_fix = 0
    for _ in range(50):
        bad = np.flatnonzero(_frac(D @ val + kd) > 1e-6)
        if len(bad) == 0:
            break
        best, best_key = None, None
        cand = np.unique(np.nonzero(_frac(D[bad]) > 1e-9)[1])
        for q in cand:
            for dlt in steps:
                v2 = val.copy()
                v2[q] += dlt
                key = (int((_frac(D @ v2 + kd) > 1e-6).sum()), energy(v2))
                if best_key is None or key < best_key:
                    best, best_key = v2, key
        if best is None or best_key[0] >= len(bad):
            info["parity_stuck"] = [complex(z) for z in (D @ val + kd)[bad]]
            info["parity_cand"] = int(len(cand))
            break
        val = best
        n_fix += 1
    info["parity_fixes"] = n_fix
    return val
