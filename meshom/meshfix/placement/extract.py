"""Re-extraction of R from the row field v and the column stripes psi (B132).

1. Rows: the level sets v = k - eps (k integer) over the triangles of R + collar (quads split), chained into
   polylines (an open polyline ends on the region's outline; closed ones are loops).
2. Columns: along each polyline the phase of psi is unwrapped (per point on an edge a -> b: arg psi_a + t inc_ab,
   inc_ab = om_ab + wrap(arg psi_b - arg psi_a - om_ab); between consecutive points the step is the geometric
   estimate 2 pi (dq . e) / h plus the wrapped residual); a node at every multiple of 2 pi.
3. Snapping: a node within `snap` h of a fixed node of the outline (stitch, coast, artificial) or of a collar node
   becomes that node; a node within `clear` h of a coast / artificial outline edge is dropped (the fringe).
4. Strips: consecutive rows are paired by a dynamic programme (monotone rungs; a step on both rows is a quad, on one
   row a triangle; tilt of a rung penalised) per pair of row pieces that face each other.
5. The explicit elements inside R (centroid in R, valid, convex) are constraints of a constrained Delaunay
   triangulation of R's nodes; the leftover triangles (the fringe at the coast and the window sides) are recombined.
"""
import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M

DEFAULTS = dict(
    snap=0.35,         # B132: generated node -> fixed outline / collar node within this * h
    snap_soft=0.35,    # B132: ... within this * h of a coast / artificial outline node
    clear=0.5,          # B132: generated nodes nearer than this * h to a coast / artificial outline edge are dropped
    tilt_w=2.0,        # B132: DP penalty per (rung tilt / h)^2
    tri_cost=0.45,     # B132: DP cost of a triangle relative to a quad (both earn 1)
    max_rung=2.2,      # B132: a rung longer than this * h is not used
    eps=1e-6,          # level offset
)


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def edge_lookup(edges, n):
    key = edges[:, 0].astype(np.int64) * n + edges[:, 1]
    srt = np.argsort(key)
    ks = key[srt]

    def find(a, b):
        k = np.minimum(a, b) * n + np.maximum(a, b)
        i = np.searchsorted(ks, k)
        i = np.minimum(i, len(ks) - 1)
        ok = ks[i] == k
        return np.where(ok, srt[i], -1)
    return find


def row_polylines(P, tris, edges, find, v, level):
    """Polylines of {v = level} over tris: list of (edge ids (m,), t (m,), closed bool). A crossing on edge (a<b) is at
    parameter t from a. Convention: v < level vs v >= level."""
    up = v >= level
    tu = up[tris]
    cnt = tu.sum(1)
    sel = np.flatnonzero((cnt == 1) | (cnt == 2))
    if len(sel) == 0:
        return []
    tt = tris[sel]
    segs = []
    for k in range(3):
        a, b = tt[:, k], tt[:, (k + 1) % 3]
        segs.append(np.where(up[a] != up[b], find(a, b), -1))
    S = np.stack(segs, 1)                                  # (m,3) edge id or -1, exactly two per row
    ok = (S >= 0).sum(1) == 2
    S = np.sort(S[ok], 1)[:, 1:]                           # the two edge ids
    # chain: each edge id appears in 1 (outline) or 2 segments
    ne = len(edges)
    m = len(S)
    occ = np.full((ne, 2), -1, np.int64)
    flat = S.ravel()
    order = np.argsort(flat, kind="stable")
    fs = flat[order]
    firstpos = np.r_[True, fs[1:] != fs[:-1]]
    idx_in = np.arange(len(fs)) - np.maximum.accumulate(np.where(firstpos, np.arange(len(fs)), 0))
    keep2 = idx_in < 2
    occ[fs[keep2], idx_in[keep2]] = order[keep2] // 2
    lines = _chain(S, occ)
    a, b = edges[:, 0], edges[:, 1]
    out = []
    for es, closed in lines:
        es = np.asarray(es, np.int64)
        va, vb = v[a[es]], v[b[es]]
        t = (level - va) / (vb - va)
        out.append((es, np.clip(t, 0.0, 1.0), closed))
    return out


def _chain(S, occ):
    m = len(S)
    used = np.zeros(m, bool)
    lines = []

    def other_seg(e, s):
        o0, o1 = occ[e]
        return o1 if o0 == s else o0

    # open chains start at an edge with a single segment
    deg = (occ >= 0).sum(1)
    starts = np.flatnonzero(deg == 1)
    for e0 in starts:
        s = occ[e0, 0]
        if used[s]:
            continue
        es = [e0]
        e = e0
        while s >= 0 and not used[s]:
            used[s] = True
            e1 = S[s, 0] if S[s, 1] == e else S[s, 1]
            es.append(e1)
            s = other_seg(e1, s)
            e = e1
        lines.append((es, False))
    for s0 in range(m):
        if used[s0]:
            continue
        e = S[s0, 0]
        es = [e]
        s = s0
        while not used[s]:
            used[s] = True
            e1 = S[s, 0] if S[s, 1] == e else S[s, 1]
            es.append(e1)
            s = other_seg(e1, s)
            e = e1
            if s < 0:
                break
        lines.append((es, True))
    return lines


def row_nodes(P, edges, es, t, psi, om, h_loc, edir):
    """Points along one polyline where the unwrapped phase passes 2 pi m: (K,3) unit points, and their position s
    along the polyline (h units). Orient so the phase grows along the polyline (returns reversed flag)."""
    a, b = edges[es, 0], edges[es, 1]
    X = (1 - t)[:, None] * P[a] + t[:, None] * P[b]
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ia = om[es] + _wrap(np.angle(psi[b]) - np.angle(psi[a]) - om[es])
    th = np.angle(psi[a]) + t * ia                         # phase at each point (mod 2 pi)
    hh = ((1 - t) * h_loc[a] + t * h_loc[b]) / R_EARTH_M
    e = (1 - t)[:, None] * edir[a] + t[:, None] * edir[b]
    dX = np.diff(X, axis=0)
    em = 0.5 * (e[1:] + e[:-1])
    hm = 0.5 * (hh[1:] + hh[:-1])
    est = 2 * np.pi * np.einsum("ij,ij->i", dX, em) / hm
    step = est + _wrap(np.diff(th) - est)
    Th = np.r_[0.0, np.cumsum(step)] + th[0]
    rev = Th[-1] < Th[0]
    if rev:
        X, Th, hh = X[::-1], Th[::-1], hh[::-1]
    # crossings of 2 pi m
    k0 = np.ceil(Th[:-1] / (2 * np.pi) - 1e-9)
    k1 = np.floor(Th[1:] / (2 * np.pi) + 1e-9)
    pts = []
    for i in np.flatnonzero(k1 >= k0):
        for k in np.arange(k0[i], k1[i] + 1):
            d = Th[i + 1] - Th[i]
            s = 0.0 if abs(d) < 1e-12 else (2 * np.pi * k - Th[i]) / d
            if pts and s <= 1e-9 and i > 0:
                pass
            pts.append((i, min(max(s, 0.0), 1.0), k))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, np.int64), rev
    # unique by phase index k
    ks = np.array([p[2] for p in pts])
    _, first = np.unique(ks, return_index=True)
    pts = [pts[j] for j in np.sort(first)]
    Y = np.array([(1 - s) * X[i] + s * X[i + 1] for i, s, _ in pts])
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y, np.array([p[2] for p in pts], np.int64), rev


@njit(cache=True)
def _dp(A, B, ha, hb, ea, eb, tilt_w, tri_cost, max_rung):
    """Monotone rung chain between rows A (na,3) and B (nb,3) maximising sum(1 - cost) over elements; local (free
    start / end). Rung (i, j) allowed when |A_i - B_j| <= max_rung * mean h. Returns rungs (k,2) of the best chain."""
    na, nb = A.shape[0], B.shape[0]
    NEG = -1e18
    S = np.full((na, nb), NEG)
    back = np.full((na, nb), -1, np.int64)       # 0 start, 1 diag, 2 from (i-1, j), 3 from (i, j-1)
    best, bi, bj = NEG, -1, -1
    for i in range(na):
        for j in range(nb):
            d = A[i] - B[j]
            l = np.sqrt((d * d).sum())
            hm = 0.5 * (ha[i] + hb[j])
            if l > max_rung * hm:
                continue
            e = 0.5 * (ea[i] + eb[j])
            tl = (d * e).sum() / hm
            c = tilt_w * tl * tl
            s, bk = 0.0 - c, 0
            if i > 0 and j > 0 and S[i - 1, j - 1] > NEG:
                v = S[i - 1, j - 1] + 1.0 - c
                if v > s:
                    s, bk = v, 1
            if i > 0 and S[i - 1, j] > NEG:
                v = S[i - 1, j] + 1.0 - tri_cost - c
                if v > s:
                    s, bk = v, 2
            if j > 0 and S[i, j - 1] > NEG:
                v = S[i, j - 1] + 1.0 - tri_cost - c
                if v > s:
                    s, bk = v, 3
            S[i, j] = s
            back[i, j] = bk
            if s > best:
                best, bi, bj = s, i, j
    out = np.empty((na + nb, 2), np.int64)
    k = 0
    i, j = bi, bj
    while i >= 0 and j >= 0:
        out[k, 0], out[k, 1] = i, j
        k += 1
        bk = back[i, j]
        if bk == 1:
            i, j = i - 1, j - 1
        elif bk == 2:
            i -= 1
        elif bk == 3:
            j -= 1
        else:
            break
    return out[:k][::-1].copy()


@njit(cache=True)
def _dp_band(A, B, ha, hb, ea, eb, tilt_w, tri_cost, max_rung, jlo, W):
    """_dp restricted to j in [jlo[i], jlo[i] + W) (jlo non-decreasing): O(na W) for long rows."""
    na, nb = A.shape[0], B.shape[0]
    NEG = -1e18
    S = np.full((na, W), NEG)
    back = np.full((na, W), -1, np.int64)
    best, bi, bj = NEG, -1, -1
    for i in range(na):
        for jj in range(W):
            j = jlo[i] + jj
            if j >= nb:
                break
            d = A[i] - B[j]
            l = np.sqrt((d * d).sum())
            hm = 0.5 * (ha[i] + hb[j])
            if l > max_rung * hm:
                continue
            e = 0.5 * (ea[i] + eb[j])
            tl = (d * e).sum() / hm
            c = tilt_w * tl * tl
            s, bk = 0.0 - c, 0
            if i > 0:
                k1 = j - 1 - jlo[i - 1]
                if j > 0 and 0 <= k1 < W and S[i - 1, k1] > NEG:
                    v = S[i - 1, k1] + 1.0 - c
                    if v > s:
                        s, bk = v, 1
                k2 = j - jlo[i - 1]
                if 0 <= k2 < W and S[i - 1, k2] > NEG:
                    v = S[i - 1, k2] + 1.0 - tri_cost - c
                    if v > s:
                        s, bk = v, 2
            if jj > 0 and S[i, jj - 1] > NEG:
                v = S[i, jj - 1] + 1.0 - tri_cost - c
                if v > s:
                    s, bk = v, 3
            S[i, jj] = s
            back[i, jj] = bk
            if s > best:
                best, bi, bj = s, i, j
    out = np.empty((na + nb, 2), np.int64)
    k = 0
    i, j = bi, bj
    while i >= 0 and j >= 0:
        out[k, 0], out[k, 1] = i, j
        k += 1
        bk = back[i, j - jlo[i]]
        if bk == 1:
            i, j = i - 1, j - 1
        elif bk == 2:
            i -= 1
        elif bk == 3:
            j -= 1
        else:
            break
    return out[:k][::-1].copy()


def strip_elements(rows_k, rows_k1, o):
    """Elements between two rows. rows_*: list of pieces, each dict(ids (m,), X (m,3), h (m,), e (m,3)) in phase
    order. Returns quads (Q,4) and tris (T,3) in global node ids, CCW for a row direction e and rows stacked along
    n = p x e (row k below k1)."""
    quads, tris = [], []
    if not rows_k or not rows_k1:
        return quads, tris
    allA = np.vstack([p["X"] for p in rows_k])
    pid = np.concatenate([np.full(len(p["ids"]), i) for i, p in enumerate(rows_k)])
    tree = cKDTree(allA)
    for pb in rows_k1:
        if len(pb["ids"]) < 2:
            continue
        _, nn = tree.query(pb["X"])
        for ia in np.unique(pid[nn]):
            pa = rows_k[ia]
            if len(pa["ids"]) < 2:
                continue
            if len(pa["ids"]) * len(pb["ids"]) <= 250_000:
                r = _dp(pa["X"], pb["X"], pa["h"] / R_EARTH_M, pb["h"] / R_EARTH_M, pa["e"], pb["e"],
                        o["tilt_w"], o["tri_cost"], o["max_rung"])
            else:                                 # long rows: a band around the nearest B of each A
                W = 16
                _, jn = cKDTree(pb["X"]).query(pa["X"])
                jlo = np.maximum.accumulate(np.clip(jn - W // 2, 0, max(len(pb["ids"]) - W, 0))).astype(np.int64)
                r = _dp_band(pa["X"], pb["X"], pa["h"] / R_EARTH_M, pb["h"] / R_EARTH_M, pa["e"], pb["e"],
                             o["tilt_w"], o["tri_cost"], o["max_rung"], jlo, W)
            for (i0, j0), (i1, j1) in zip(r[:-1], r[1:]):
                a0, a1 = pa["ids"][i0], pa["ids"][i1]
                b0, b1 = pb["ids"][j0], pb["ids"][j1]
                if i1 == i0 + 1 and j1 == j0 + 1:
                    quads.append((a0, a1, b1, b0))
                elif i1 == i0 + 1:
                    tris.append((a0, a1, b0))
                else:
                    tris.append((a0, b1, b0))
    return quads, tris


def fix_spikes(mesh):
    """A triangle with two boundary edges (the boundary_edges gate) is merged with the element across its third
    edge: a triangle -> a quad (if convex), a quad -> the pentagon re-split into the quad + triangle with the best worst
    angle among the splits whose triangle has at most one boundary edge. Disjoint pairs, one pass. Returns
    (mesh, changed)."""
    from meshom.mesh import Mesh, build_topology
    from meshom.methods.m07.extract import _badness

    tp = build_topology(mesh)
    n = mesh.n_nodes
    nt = mesh.n_tris
    be = tp.edges[tp.edge_elems[:, 1] < 0]
    bkey = set((be[:, 0] * n + be[:, 1]).tolist())

    def isb(a, b):
        return (min(a, b) * n + max(a, b)) in bkey

    ekey = tp.edges[:, 0] * n + tp.edges[:, 1]
    so = np.argsort(ekey)
    tris, quads = mesh.tris.copy(), mesh.quads.copy()
    drop_t, drop_q, add_t, add_q = set(), set(), [], []
    used = set()
    for t in range(nt):
        a, b, c = tris[t]
        eb = [isb(a, b), isb(b, c), isb(c, a)]
        if sum(eb) < 2:
            continue
        k = eb.index(False)                      # the interior edge (u -> v), apex w
        u, v, w = tris[t, k], tris[t, (k + 1) % 3], tris[t, (k + 2) % 3]
        key = min(u, v) * n + max(u, v)
        i = so[np.searchsorted(ekey[so], key)]
        e0, e1 = tp.edge_elems[i]
        other = e1 if e0 == t else e0
        if other < 0 or other in used or t in used:
            continue
        if other < nt:                           # triangle: quad (u, w, v, x) with x the far apex
            ot = tris[other]
            x = [z for z in ot if z != u and z != v][0]
            q = np.array([[w, u, x, v]])
            if np.isfinite(_badness(mesh.points, np.zeros((0, 3), np.int64), q)).all():
                drop_t |= {t, other}
                add_q.append(q[0])
                used |= {t, other}
            continue
        qi = other - nt
        qq = quads[qi]
        j = int(np.flatnonzero(qq == v)[0])
        if qq[(j + 1) % 4] != u:
            continue
        # pentagon CCW: v -> w -> u (the triangle's outer edges), then the quad's u -> y -> z -> v
        y, z = qq[(j + 2) % 4], qq[(j + 3) % 4]
        Pg = np.array([v, w, u, y, z])
        best, bs = np.inf, None
        for s in range(5):
            idx = (np.arange(5) + s) % 5
            tri_s = Pg[idx[:3]]
            if sum(isb(tri_s[r], tri_s[(r + 1) % 3]) for r in range(3)) >= 2:
                continue
            quad_s = Pg[[idx[2], idx[3], idx[4], idx[0]]]
            val = _badness(mesh.points, tri_s[None], quad_s[None])[0]
            if val < best:
                best, bs = val, (tri_s, quad_s)
        if bs is None or not np.isfinite(best):
            continue
        drop_t.add(t)
        drop_q.add(qi)
        add_t.append(bs[0])
        add_q.append(bs[1])
        used |= {t, other}
    if not used:
        return mesh, 0
    kt = np.ones(nt, bool)
    kt[list(drop_t)] = False
    kq = np.ones(len(quads), bool)
    kq[list(drop_q)] = False
    T2 = np.vstack([tris[kt]] + ([np.array(add_t)] if add_t else []))
    Q2 = np.vstack([quads[kq]] + ([np.array(add_q)] if add_q else []))
    return Mesh(mesh.points, T2.astype(np.int64), Q2.astype(np.int64), mesh.node_flag), len(used) // 2


def split_bad_quads(mesh, deg=145.0):
    """A quad with a corner above deg is split into two triangles along the diagonal from that corner when the worse
    triangle is better than the quad (worst angle relative to the hard gates) and neither triangle gets two boundary
    edges. Returns (mesh, count)."""
    from meshom.mesh import Mesh, build_topology
    from meshom.methods.m07.extract import _corner_angles

    if mesh.n_quads == 0:
        return mesh, 0
    ang = np.degrees(_corner_angles(mesh.points, mesh.quads))
    cand = np.flatnonzero(ang.max(1) > deg)
    if len(cand) == 0:
        return mesh, 0
    tp = build_topology(mesh)
    n = mesh.n_nodes
    be = tp.edges[tp.edge_elems[:, 1] < 0]
    bset = set((be[:, 0] * n + be[:, 1]).tolist())

    def nb(t):
        return sum(((min(t[r], t[(r + 1) % 3]) * n + max(t[r], t[(r + 1) % 3])) in bset) for r in range(3))

    keep = np.ones(mesh.n_quads, bool)
    add = []
    for qi in cand:
        q = mesh.quads[qi]
        k = int(np.argmax(ang[qi]))
        t1 = np.array([q[k], q[(k + 1) % 4], q[(k + 2) % 4]])
        t2 = np.array([q[k], q[(k + 2) % 4], q[(k + 3) % 4]])
        if nb(t1) >= 2 or nb(t2) >= 2:
            continue
        bq = _bad_q(mesh.points, q[None])[0]
        bt = max(_bad_t(mesh.points, t1[None])[0], _bad_t(mesh.points, t2[None])[0])
        if bt < bq:
            keep[qi] = False
            add += [t1, t2]
    if not add:
        return mesh, 0
    T2 = np.vstack([mesh.tris, np.array(add)]).astype(np.int64)
    return Mesh(mesh.points, T2, mesh.quads[keep], mesh.node_flag), int((~keep).sum())


def _bad_q(points, quads):
    from meshom.methods.m07.extract import _corner_angles

    a = _corner_angles(points, quads)
    return np.where((a >= np.pi).any(1), np.inf, np.abs(a - 0.5 * np.pi).max(1) / (np.pi / 3))


def _bad_t(points, tris):
    from meshom.methods.m07.extract import _corner_angles

    a = _corner_angles(points, tris)
    b = np.maximum((a.max(1) - np.pi / 3) / (np.pi / 3), (np.pi / 3 - a.min(1)) / np.radians(40.0))
    return np.where((a >= np.pi).any(1), np.inf, b)
