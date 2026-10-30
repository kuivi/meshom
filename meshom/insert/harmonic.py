"""The band block of a coastal strip from a harmonic map: rows that cannot fold (decisions B195-B198).

The band region (chart plane; the stereographic chart is conformal, so a harmonic map there is harmonic on the
sphere) is a curvilinear quadrilateral: the spine (row 0, P -> Q: the coast smoothed at the fine size), the bank
beyond Q (the coast Q -> B0), the host chain (B0 -> A0, row K, fixed host nodes) and the bank before P (A0 -> P).

1. Harmonic coordinates (B195). A triangulation of the region (boundary and interior points at HB_CELLS fine
   cells, constrained Delaunay; every interior edge with both ends on one side of the square split at its midpoint,
   `_split_chords`); two Laplace problems on it (P1 finite elements, cotangent weights): t = 0 on the
   spine, 1 on the chain, no flux through the banks; s = 0 on the bank before P, 1 on the bank beyond Q, no flux
   through the spine and the chain. Together (s, t) is the conformal map of the quadrilateral onto a rectangle (the
   orthogonal grids of coastal models by conformal mapping, after P. Sakov's gridgen, github.com/sakov/gridgen-c).
   The boundary values found there (s on the spine and the chain, t on the banks, each side made strictly
   monotone) are then kept, and the interior is solved again with mean value weights (Floater 2003,
   doi:10.1016/S0167-8396(03)00002-5): positive weights and a convex image (the unit square) make the discrete map
   one-to-one (Tutte 1963, doi:10.1112/plms/s3-13.1.743; Floater 2003) when no interior edge has both ends on one
   side of the square, so its piecewise linear inverse X(s, t) is defined everywhere on the square and never
   folds. A map with a flipped or flat triangle in the square is refused (MapError), never masked.
2. Rows (B196). Every column of the lattice is a line s = const of the square (a field line of t: it leaves the
   coast at a right angle). Rows are graphs t = T_k(s) increasing in k at every s, so the cells of the lattice tile
   the square and their images tile the region: a fold can only come from the straight edges of a cell, not from
   the lattice. Near the coast a row lies at the arc length d_k along each column (rows at the distances of the
   size profile, following the coast and its capes without the swallowtails of an offset curve); toward the chain
   it blends (smoothstep of the distance over the growth zone) into the level line t = tau_k (tau_k: the level at
   the distance d_k on the middle columns), which follows the chain's steps for about a cell and smooths them out.
3. Columns (B197). Row 0's columns are the spine nodes; each next row keeps its inner neighbour's columns and
   splits or merges a few of them where its cells along the row are too wide or too narrow for its size target (a
   Y node of valence 5 and one triangle per event, as the ORCA ladder, B114-B116): a split when the mean of a
   column's two cells exceeds SPLIT x h, a merge when the mean of three cells is below h / SPLIT; at most
   max(1, BIRTH_FRAC N) events per row, none at a column or next to one that changed within GAP_ROWS rows; a split's
   two arms start THETA_DEG from the parent's line; then the row's columns are placed by least squares near their
   straight continuation with smooth widths along the row (RELAX_BETA; the ladder's relaxation, B115). In the last
   ALIGN_ROWS rows before the chain the columns are matched to the chain's nodes (a monotone alignment by dynamic
   programming), the unmatched ones merged or split, the matched ones drawn onto their node's s (the last row on it,
   so the last cell row is quads); a reflex step of the chain takes two columns, a convex one none.
4. Corners. P and Q sit where the coast runs straight: the square's corner there would be a cell with a 180 deg
   corner, so P (Q) is merged into row 1's end on the bank and that cell becomes a triangle.
5. Polish (B197, `polish`, called by meshom.insert.band after the smoothing): the free nodes of the quads above
   POLISH_DEG move to lower the squared excess of their cells' largest corners, cells kept convex.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from meshom.insert.bandblock import BandBlock, _isotonic
from meshom.insert.block import BIRTH_FRAC, GAP_ROWS, zip_rows
from meshom.insert.geom import Polyline, point_in_polygon, resample

HB_CELLS = 0.4               # B195: the map's triangulation spacing, in fine cells (boundary and interior points)
MAX_POINTS = 150_000         # B195: the triangulation's spacing grows to keep under this many points
SPLIT = 1.22                 # B197: a split (merge) when the cells along a row are SPLIT x (1 / SPLIT x) the target
THETA_DEG = 18.0             # B197: a split's two arms start this far from the parent column's line each
TILT_DEG = 20.0              # B197: per row a column moves at most the row step x tan(TILT_DEG) along its row
RELAX_BETA = 1.0             # B197: the weight of smooth widths against straight columns in a row's relaxation (0:
                             # the drift below instead)
EQUALIZE = 0.5               # B197: drift: a column moves EQUALIZE of the way toward the middle of its neighbours ...
DRIFT_ROWS = 4               # B197: ... when within DRIFT_COLS columns of a split or merge of the last DRIFT_ROWS
DRIFT_COLS = 2               # rows (elsewhere a column stays on its line s = const)
ALIGN_ROWS = 4               # B197: rows before the chain whose columns are matched to the chain's nodes
REFLEX_DEG = 60.0            # B197: a host chain node turning away from the band by this much takes two columns
ARC_SHARE = 0.1              # B195: the boundary values blend this share of the normalised arc length into the
                             # conformal ones (a narrow inlet compresses the conformal s exponentially)
ROW_SMOOTH = 2.0             # B196: a distance row's t is smoothed along it with a Gaussian of this x its distance
POLISH_ROUNDS = 20           # B197: rounds of the corner polish of the band's quads above the angle target
POLISH_STEPS = (0.5, 0.25, 0.1, 0.04)   # B197: its trial moves, in the node's shortest incident edge
POLISH_FREE_DEG = 100.0      # B197: the polish lowers the squared excess of the quads' largest corners over this ...
POLISH_TRI_DEG = 140.0       # B197: a triangle's largest corner counts in the polish above this
POLISH_EDGE = 0.8            # B197: a polish move keeps every edge above this x the shorter end's shortest edge before
POLISH_DEG = 120.0           # B197: the polish works on quads with a corner above this
T_SAMPLES = 400              # samples of each column in t (stretched toward the coast)
S_PER_CELL = 4               # samples of the rows in s per spine cell


def _poly_area(p):
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)))


def _split_edges(a, b, step):
    """Points from a toward b (a included, b excluded) at a spacing <= step."""
    n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
    return a + np.linspace(0.0, 1.0, n + 1)[:-1, None] * (b - a)


def _cot_laplacian(xy, tri, n):
    """The P1 stiffness matrix (cotangent weights) of the plane triangulation."""
    rows, cols, vals = [], [], []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u = xy[tri[:, a]] - xy[tri[:, c]]
        v = xy[tri[:, b]] - xy[tri[:, c]]
        cr = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        w = 0.5 * (u * v).sum(1) / np.maximum(cr, 1e-30)
        rows += [tri[:, a], tri[:, b], tri[:, a], tri[:, b]]
        cols += [tri[:, b], tri[:, a], tri[:, a], tri[:, b]]
        vals += [-w, -w, w, w]
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()


def _mvc_matrix(xy, tri, n):
    """Mean value weights (Floater 2003): w_ij = (tan(a/2) + tan(b/2)) / |x_j - x_i|, a, b the angles at i of the
    two triangles at the edge ij; the matrix of sum_j w_ij (u_j - u_i)."""
    rows, cols, vals = [], [], []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        e1 = xy[tri[:, b]] - xy[tri[:, a]]
        e2 = xy[tri[:, c]] - xy[tri[:, a]]
        l1 = np.linalg.norm(e1, axis=1)
        l2 = np.linalg.norm(e2, axis=1)
        ang = np.arctan2(np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]), (e1 * e2).sum(1))
        tn = np.tan(0.5 * ang)
        for o, ln in ((b, l1), (c, l2)):
            w = tn / np.maximum(ln, 1e-30)
            rows += [tri[:, a], tri[:, a]]
            cols += [tri[:, o], tri[:, a]]
            vals += [w, -w]
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()


def _dirichlet(A, fixed, val):
    """Solve A u = 0 on the free nodes with u = val on the fixed ones."""
    u = np.where(fixed, val, 0.0).astype(float)
    f = ~fixed
    rhs = -A[f][:, fixed] @ u[fixed]
    u[f] = spsolve(A[f][:, f].tocsc(), rhs)
    return u


def _strict(y, lo, hi):
    """y made strictly increasing from lo to hi (pool adjacent violators, then a tiny floor on the steps)."""
    y = np.clip(_isotonic(np.asarray(y, float)), lo, hi)
    y[0], y[-1] = lo, hi
    n = len(y)
    eps = 1e-6 * (hi - lo) / max(n, 1)
    for k in range(1, n):
        y[k] = max(y[k], y[k - 1] + eps)
    if y[-1] > hi:                                          # squeeze back under hi
        y = lo + (y - lo) * (hi - lo) / (y[-1] - lo)
    y[-1] = hi
    return y


class MapError(ValueError):
    """The band's map is not one-to-one (a flipped or flat triangle in the square): no inverse is built."""


def _split_chords(xy, tri, nb, on):
    """Every interior edge whose two ends lie on one side of the square (on: (nb, 4) side membership of the boundary
    points 0..nb-1) split at its midpoint, the two triangles at it split in two each, until none is left. Such an
    edge would map into that side (a boundary ear, its three corners on one side, maps to zero area; so does a
    point whose neighbours all lie on one side); without one, positive weights onto the convex square give a
    one-to-one map with no degenerate triangle (Floater 2003, doi:10.1090/S0025-5718-02-01466-7). Returns (xy,
    tri, number of splits)."""
    xy, tri = np.asarray(xy, float), np.asarray(tri, np.int64).copy()
    n_split = 0
    while True:
        e = np.sort(np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=1)
        u, cnt = np.unique(e, axis=0, return_counts=True)
        u = u[(cnt == 2) & (u[:, 1] < nb)]
        u = u[(on[u[:, 0]] & on[u[:, 1]]).any(1)]
        if not len(u):
            return xy, tri, n_split
        touched = np.zeros(len(tri), bool)
        new_xy, new_tri = [], []
        for a, b in u:
            ts = np.flatnonzero(((tri == a) | (tri == b)).sum(1) == 2)
            if touched[ts].any():                           # one split per triangle and pass
                continue
            touched[ts] = True
            m = len(xy) + len(new_xy)
            new_xy.append(0.5 * (xy[a] + xy[b]))
            for k in ts:
                r = tri[k]
                i = int(np.flatnonzero((r != a) & (r != b))[0])
                p_, q_, o_ = r[(i + 1) % 3], r[(i + 2) % 3], r[i]   # the edge p -> q, o opposite (orientation kept)
                tri[k] = (p_, m, o_)
                new_tri.append((m, q_, o_))
            n_split += 1
        xy = np.vstack([xy, new_xy])
        tri = np.vstack([tri, np.array(new_tri, np.int64)])


@dataclass
class HarmonicMap:
    """The band's harmonic coordinates (module docstring, step 1)."""
    xy: np.ndarray               # (V,2) triangulation points (chart)
    tri: np.ndarray              # (T,3)
    st: np.ndarray               # (V,2) (s, t) of every point
    spine_s: np.ndarray          # s of the spine nodes (row 0)
    chain_s: np.ndarray          # s of the chain nodes (row K)
    bank_p: tuple                # (t increasing from P to A0, points)
    bank_q: tuple                # (t increasing from Q to B0, points)
    info: dict

    def __post_init__(self):
        import matplotlib.tri as mtri

        self._T = mtri.Triangulation(self.st[:, 0], self.st[:, 1], self.tri)
        from scipy.spatial import cKDTree

        self._near = cKDTree(self.st)
        self._fx = mtri.LinearTriInterpolator(self._T, self.xy[:, 0])
        self._fy = mtri.LinearTriInterpolator(self._T, self.xy[:, 1])

    def bank(self, t, end: str) -> np.ndarray:
        tt, pts = self.bank_p if end == "p" else self.bank_q
        t = np.atleast_1d(np.asarray(t, float))
        return np.c_[np.interp(t, tt, pts[:, 0]), np.interp(t, tt, pts[:, 1])]

    def X(self, s, t) -> np.ndarray:
        """The inverse map: chart points of (s, t) (s = 0 / 1 on the banks exactly)."""
        s = np.atleast_1d(np.asarray(s, float))
        t = np.atleast_1d(np.asarray(t, float))
        s, t = np.broadcast_arrays(s, t)
        out = np.empty((len(s), 2))
        e0, e1 = s <= 0.0, s >= 1.0
        if e0.any():
            out[e0] = self.bank(t[e0], "p")
        if e1.any():
            out[e1] = self.bank(t[e1], "q")
        m = ~(e0 | e1)
        if m.any():
            ss = np.clip(s[m], 1e-12, 1 - 1e-12)
            tt = np.clip(t[m], 1e-12, 1 - 1e-12)
            x = self._fx(ss, tt)
            y = self._fy(ss, tt)
            bad = np.ma.getmaskarray(x) | np.ma.getmaskarray(y)
            r = np.c_[np.ma.filled(x, np.nan), np.ma.filled(y, np.nan)]
            if bad.any():                                   # on the square's edge by round-off: nudge inside
                for f in (1e-9, 1e-7, 1e-5):
                    ss2 = np.clip(ss[bad], f, 1 - f)
                    tt2 = np.clip(tt[bad], f, 1 - f)
                    x2, y2 = self._fx(ss2, tt2), self._fy(ss2, tt2)
                    ok = ~(np.ma.getmaskarray(x2) | np.ma.getmaskarray(y2))
                    idx = np.flatnonzero(bad)
                    r[idx[ok]] = np.c_[np.ma.filled(x2, 0)[ok], np.ma.filled(y2, 0)[ok]]
                    bad[idx[ok]] = False
                    if not bad.any():
                        break
                if bad.any():                               # not found by round-off: the nearest map point
                    idx = np.flatnonzero(bad)
                    r[idx] = self.xy[self._near.query(np.c_[ss[idx], tt[idx]])[1]]
            out[m] = r
        return out


def harmonic_map(spine, bank_p: Polyline, bank_q: Polyline, chain_xy, h_fine: float, chart) -> HarmonicMap:
    """Step 1 of the module docstring. spine (n,2) P -> Q, bank_p A0 -> P, bank_q Q -> B0, chain_xy A0 -> B0."""
    from scipy.spatial import cKDTree

    from meshom.delaunay import sphere_delaunay

    L_tot = (Polyline(spine).L + bank_p.L + bank_q.L + Polyline(chain_xy).L)
    hb = HB_CELLS * h_fine
    pts, side, sp_idx, ch_idx = [], [], [], []

    def add(p, sd):
        pts.append(p)
        side.append(np.full(len(p), sd))

    for _ in range(4):                                      # coarser when the region is large
        pts, side, sp_idx, ch_idx = [], [], [], []
        n0 = 0
        for i in range(len(spine) - 1):
            sp_idx.append(n0)
            q = _split_edges(spine[i], spine[i + 1], hb)
            add(q, 0)
            n0 += len(q)
        sp_idx.append(n0)
        q = resample(bank_q.xy, hb)[:-1]
        add(q, 1)
        n0 += len(q)
        rc = chain_xy[::-1]
        ch_rev = []
        for i in range(len(rc) - 1):
            ch_rev.append(n0)
            q = _split_edges(rc[i], rc[i + 1], hb)
            add(q, 2)
            n0 += len(q)
        ch_rev.append(n0)                                   # A0: the first point of the bank before P
        q = resample(bank_p.xy, hb)[:-1]
        add(q, 3)
        n0 += len(q)
        loop = np.vstack(pts)
        area = abs(_poly_area(loop))
        if area / (0.866 * hb * hb) + len(loop) <= MAX_POINTS:
            break
        hb *= 1.6
    side = np.concatenate(side)
    ch_idx = np.array(ch_rev[::-1])
    sp_idx = np.array(sp_idx)
    nb = len(loop)
    ccw = _poly_area(loop) > 0
    # interior points: a triangular lattice, away from the boundary
    lo, hi = loop.min(0), loop.max(0)
    ny = int(np.ceil((hi[1] - lo[1]) / (0.866 * hb))) + 1
    nx = int(np.ceil((hi[0] - lo[0]) / hb)) + 2
    gy, gx = np.mgrid[0:ny, 0:nx].astype(float)
    gx = lo[0] - hb + hb * (gx + 0.5 * (gy % 2))
    gy = lo[1] + 0.866 * hb * gy
    g = np.c_[gx.ravel(), gy.ravel()]
    g = g[point_in_polygon(g, loop)]
    dense = resample(np.vstack([loop, loop[:1]]), 0.25 * hb)
    if len(g):
        dmin = cKDTree(dense).query(g)[0]
        g = g[dmin >= 0.55 * hb]
    E = np.c_[np.arange(nb), (np.arange(nb) + 1) % nb]
    if not ccw:
        E = E[:, ::-1]
    P = np.vstack([loop, g])
    tri = sphere_delaunay(chart.inv(P), edges=E, coast=E)
    used = np.zeros(len(P), bool)
    used[tri.ravel()] = True
    if not used[:nb].all():
        raise ValueError("the band's triangulation lost boundary points")
    keep = np.flatnonzero(used)
    remap = np.full(len(P), -1)
    remap[keep] = np.arange(len(keep))
    P, tri = P[keep], remap[tri]
    on = np.zeros((nb, 4), bool)                            # the square's sides each boundary point lies on
    on[np.arange(nb), side] = True
    on[[0, sp_idx[-1], ch_rev[0], ch_rev[-1]], [3, 0, 1, 2]] = True   # the corners P, Q, B0, A0 on two sides
    P, tri, n_split = _split_chords(P, tri, nb, on)
    n = len(P)
    # 1a. the conformal pair: t (spine 0, chain 1, banks free), s (bank before P 0, beyond Q 1, spine / chain free)
    K = _cot_laplacian(P, tri, n)
    q_i, b_i, a_i = sp_idx[-1], ch_rev[0], ch_rev[-1]       # Q, B0, A0 in the loop
    fix_t = np.zeros(n, bool)
    fix_t[:nb] = (side == 0) | (side == 2)
    fix_t[[q_i, a_i]] = True
    val_t = np.zeros(n)
    val_t[:nb][side == 2] = 1.0
    val_t[a_i] = 1.0
    val_t[q_i] = 0.0
    t_c = _dirichlet(K, fix_t, val_t)
    fix_s = np.zeros(n, bool)
    fix_s[:nb] = (side == 1) | (side == 3)
    fix_s[[0, b_i]] = True
    val_s = np.zeros(n)
    val_s[:nb][side == 1] = 1.0
    val_s[b_i] = 1.0
    val_s[0] = 0.0
    s_c = _dirichlet(K, fix_s, val_s)
    # 1b. the boundary values, ARC_SHARE of them by arc length, made strictly monotone per side
    def side_values(idx, u):
        a = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P[idx], axis=0), axis=1))]
        u = (u - u[0]) / max(u[-1] - u[0], 1e-300)
        return _strict((1.0 - ARC_SHARE) * u + ARC_SHARE * a / max(a[-1], 1e-300), 0.0, 1.0)

    sb, tb = np.zeros(nb), np.zeros(nb)
    i_sp = np.arange(0, q_i + 1)                            # P .. Q
    sb[i_sp] = side_values(i_sp, s_c[i_sp])
    tb[i_sp] = 0.0
    i_bq = np.arange(q_i, b_i + 1)                          # Q .. B0
    tb[i_bq] = side_values(i_bq, t_c[i_bq])
    sb[i_bq] = 1.0
    i_ch = np.arange(a_i, b_i - 1, -1)                      # A0 .. B0 (the loop runs B0 -> A0)
    sb[i_ch] = side_values(i_ch, s_c[i_ch])
    tb[i_ch] = 1.0
    i_bp = np.r_[0, np.arange(nb - 1, a_i - 1, -1)]         # P .. A0 (the loop runs A0 -> P)
    tb[i_bp] = side_values(i_bp, t_c[i_bp])
    sb[i_bp] = 0.0
    # 1c. the one-to-one map: mean value weights, the boundary on the square
    A = _mvc_matrix(P, tri, n)
    fixed = np.zeros(n, bool)
    fixed[:nb] = True
    s = _dirichlet(A, fixed, np.r_[sb, np.zeros(n - nb)])
    t = _dirichlet(A, fixed, np.r_[tb, np.zeros(n - nb)])
    st = np.c_[s, t]
    pa = st[tri]
    ar = 0.5 * ((pa[:, 1, 0] - pa[:, 0, 0]) * (pa[:, 2, 1] - pa[:, 0, 1])
                - (pa[:, 1, 1] - pa[:, 0, 1]) * (pa[:, 2, 0] - pa[:, 0, 0]))
    sgn = np.sign(np.median(ar))
    info = {"points": int(n), "triangles": int(len(tri)), "spacing_km": round(hb / 1e3, 3),
            "chords_split": int(n_split), "flipped": int((ar * sgn < 0).sum()), "flat": int((ar == 0).sum()),
            "boundary_km": round(float(L_tot) / 1e3, 1)}
    if info["flipped"] or info["flat"]:                     # no inverse without the one-to-one argument
        raise MapError(f"the band's harmonic map has {info['flipped']} flipped and {info['flat']} flat triangles "
                       f"in the square (of {info['triangles']})")
    bp_t = tb[i_bp]
    bp_x = P[i_bp]
    bq_t = tb[i_bq]
    bq_x = P[i_bq]
    return HarmonicMap(P, tri, st, sb[sp_idx], sb[ch_idx], (bp_t, bp_x), (bq_t, bq_x), info)


# ------------------------------------------------------------------------------------------------- rows

def _t_grid(n=T_SAMPLES, a=8.0):
    j = np.arange(n + 1) / n
    return (np.exp(a * j) - 1.0) / (np.exp(a) - 1.0)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def rows_t(hm: HarmonicMap, d: np.ndarray, band: float, n_spine: int):
    """Step 2: (S_g, T (K+1, len S_g)): the rows as t over the s samples S_g."""
    S = np.unique(np.r_[np.linspace(0.0, 1.0, max(S_PER_CELL * n_spine, 200) + 1), hm.spine_s])
    tg = _t_grid()
    ss, tt = np.meshgrid(S, tg, indexing="ij")
    Xg = hm.X(ss.ravel(), tt.ravel()).reshape(len(S), len(tg), 2)
    Dg = np.concatenate([np.zeros((len(S), 1)), np.cumsum(np.linalg.norm(np.diff(Xg, axis=1), axis=2), axis=1)], 1)
    K = len(d) - 1
    D = float(d[-1])
    mid = (S >= hm.spine_s[max(1, n_spine // 10)]) & (S <= hm.spine_s[min(n_spine - 1, n_spine - n_spine // 10)])
    if mid.sum() < 3:
        mid = np.ones(len(S), bool)
    eta = np.median(Dg[mid] / Dg[mid, -1:], axis=0)
    eta = np.maximum.accumulate(eta)
    T = np.zeros((K + 1, len(S)))
    T[K] = 1.0
    d_out = d[K - 1] if K >= 2 else D
    for k in range(1, K):
        Td = np.array([np.interp(min(d[k], 0.98 * Dg[i, -1]), Dg[i], tg) for i in range(len(S))])
        if ROW_SMOOTH > 0:                                  # B196: smoothed along the row over ROW_SMOOTH x d
            Td = _smooth_row(hm, S, Td, ROW_SMOOTH * d[k])
        tau = float(np.interp(d[k] / D, eta, tg))
        w = float(_smoothstep((d[k] - band) / max(d_out - band, 1e-9))) if d_out > band else 1.0
        T[k] = (1.0 - w) * Td + w * tau
    for k in range(1, K + 1):                               # rows increasing in k at every s (never cross)
        gap = 0.25 * (T[k] - T[k - 1]).clip(min=0).mean() if k < K else 0.0
        T[k] = np.maximum(T[k], T[k - 1] + max(gap, 1e-7)) if k < K else T[k]
        if k < K:
            T[k] = np.minimum(T[k], 1.0 - (K - k) * 1e-6)
    return S, T


def _smooth_row(hm, S, Tk, sigma):
    """log t of a row smoothed with a Gaussian of sigma (m) along the row's arc (its ends held by 'nearest')."""
    from scipy.ndimage import gaussian_filter1d

    X = hm.X(S, Tk)
    a = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
    if a[-1] <= 0 or sigma <= 0:
        return Tk
    m = max(int(np.ceil(a[-1] / (0.25 * sigma))), 8)
    ag = np.linspace(0.0, a[-1], m + 1)
    a = np.maximum.accumulate(a + 1e-9 * np.arange(len(a)))
    lg = np.interp(ag, a, np.log(np.maximum(Tk, 1e-300)))
    lg = gaussian_filter1d(lg, sigma / (ag[1] - ag[0]), mode="nearest")
    return np.exp(np.interp(a, ag, lg))


class _Row:
    """Row k: positions and arc length over s."""

    def __init__(self, hm, S, Tk):
        self.hm, self.S, self.Tk = hm, S, Tk
        X = hm.X(S, Tk)
        U = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
        self.U = np.maximum.accumulate(U + 1e-9 * np.arange(len(U)))

    def u(self, s):
        return np.interp(s, self.S, self.U)

    def s(self, u):
        return np.interp(u, self.U, self.S)

    def xy(self, s):
        s = np.asarray(s, float)
        return self.hm.X(s, np.interp(s, self.S, self.Tk))


def _align(a, b, w):
    """Monotone matching of sorted a to sorted b (ends matched): cost ((a_i - b_j) / w)^2 per match, 1 per unmatched
    one. Returns (pairs list of (i, j), unmatched a, unmatched b)."""
    n, m = len(a), len(b)
    INF = 1e18
    C = np.full((n + 1, m + 1), INF)
    bt = np.zeros((n + 1, m + 1), np.int8)
    C[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best, how = INF, 0
            if i > 0 and j > 0:
                c = C[i - 1, j - 1] + ((a[i - 1] - b[j - 1]) / w) ** 2
                if c < best:
                    best, how = c, 1
            if 0 < i < n and C[i - 1, j] + 1.0 < best:      # skip a (not an end)
                best, how = C[i - 1, j] + 1.0, 2
            if 0 < j < m and C[i, j - 1] + 1.0 < best:      # skip b (not an end)
                best, how = C[i, j - 1] + 1.0, 3
            C[i, j], bt[i, j] = best, how
    i, j = n, m
    pairs, ua, ub = [], [], []
    while i > 0 or j > 0:
        h = bt[i, j]
        if h == 1:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif h == 2:
            ua.append(i - 1)
            i -= 1
        else:
            ub.append(j - 1)
            j -= 1
    return pairs[::-1], sorted(ua), sorted(ub)


def _events(u, age, h, step, lim, want=None):
    """Splits and merges of the columns at arc positions u (ends fixed) for the size target h: (new u, new age,
    splits, merges). want: (extra columns to merge, missing positions to split near) from the chain alignment."""
    n = len(u) - 1
    g = np.diff(u)
    busy = np.zeros(n + 1, bool)
    busy[age < GAP_ROWS] = True
    busy = busy | np.r_[busy[1:], False] | np.r_[False, busy[:-1]]
    split, merge = [], []
    cand = []
    if want is None:
        for j in range(1, n):
            r = 0.5 * (g[j - 1] + g[j]) / h
            if r > SPLIT:
                cand.append((np.log(r), "s", j))
        for j in range(1, n - 1):                           # merge columns j, j + 1
            r = (g[j - 1] + g[j] + g[j + 1]) / (3 * h)
            if r < 1 / SPLIT and g[j] <= min(g[j - 1], g[j + 1]):
                cand.append((-np.log(r), "m", j))
    else:
        extra, missing = want
        for j in extra:                                      # an extra column: merged with its nearer neighbour
            if 0 < j < n:
                jj = j - 1 if (j == n - 1 or (j > 1 and g[j - 1] <= g[j])) else j
                if 1 <= jj and jj + 1 <= n - 1:
                    cand.append((2.0, "m", jj))
        for x in missing:                                    # a missing column: the nearest column split
            j = int(np.clip(np.argmin(np.abs(u - x)), 1, n - 1))
            cand.append((2.0, "s", j))
    cand.sort(key=lambda c: -c[0])
    taken = np.zeros(n + 1, bool)
    for _, kind, j in cand:
        if len(split) + len(merge) >= lim:
            break
        cols = [j] if kind == "s" else [j, j + 1]
        lo_, hi_ = max(cols[0] - 1, 0), min(cols[-1] + 1, n)
        if taken[lo_:hi_ + 1].any() or (want is None and busy[cols].any()):
            continue
        taken[lo_:hi_ + 1] = True
        (split if kind == "s" else merge).append(j)
    if not split and not merge:
        return u, age + 1, 0, 0
    new_u, new_age = [], []
    sp, mg = set(split), set(merge)
    j = 0
    delta = np.tan(np.radians(THETA_DEG)) * step
    while j <= n:
        if j in sp:
            dl = min(delta, 0.3 * g[j - 1], 0.3 * g[j])
            new_u += [u[j] - dl, u[j] + dl]
            new_age += [0, 0]
            j += 1
        elif j in mg:
            new_u.append(0.5 * (u[j] + u[j + 1]))
            new_age.append(0)
            j += 2
        else:
            new_u.append(u[j])
            new_age.append(age[j] + 1)
            j += 1
    return np.array(new_u), np.array(new_age), len(split), len(merge)


def _drift(u, step, target=None, frac=0.0, age=None):
    """Columns near a recent event (age < DRIFT_ROWS within DRIFT_COLS columns) toward the middle of their
    neighbours (EQUALIZE, at most step x tan(TILT_DEG)); the others stay on their line s = const (orthogonal to
    the coast); with target (the chain's matched positions, nan where none) toward it by frac instead."""
    u = u.copy()
    cap = np.tan(np.radians(TILT_DEG)) * step
    n = len(u) - 1
    near = np.ones(n + 1, bool)
    if age is not None:
        rec = age < DRIFT_ROWS
        near = rec.copy()
        for k in range(1, DRIFT_COLS + 1):
            near[k:] |= rec[:-k]
            near[:-k] |= rec[k:]
    for j in range(1, n):
        if target is None and not near[j]:
            continue
        if target is not None and np.isfinite(target[j]):
            mv = frac * (target[j] - u[j])
            if frac < 1.0:
                mv = float(np.clip(mv, -cap, cap))
        else:
            mv = float(np.clip(EQUALIZE * (0.5 * (u[j - 1] + u[j + 1]) - u[j]), -cap, cap))
        lo_ = u[j - 1] + 0.05 * (u[j + 1] - u[j - 1])
        hi_ = u[j + 1] - 0.05 * (u[j + 1] - u[j - 1])
        u[j] = float(np.clip(u[j] + mv, lo_, hi_)) if hi_ > lo_ else u[j]
    return u


def _relax(c, beta, lo_end, hi_end):
    """Column positions u on a row near the targets c (their straight continuation, a split's arms, the chain's
    nodes) with smooth widths: min sum (u_i - c_i)^2 + beta sum (u_{i+1} - 2 u_i + u_{i-1})^2, the ends fixed
    (the ORCA ladder's relaxation, B115, along one row); kept in order."""
    from scipy.sparse import diags

    n = len(c)
    if n < 4 or beta <= 0:
        return c.copy()
    D = diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n)).tocsr()
    A = (diags(np.ones(n)) + beta * (D.T @ D)).tolil()
    b = c.astype(float).copy()
    for i, v in ((0, lo_end), (n - 1, hi_end)):
        A[i, :] = 0
        A[i, i] = 1.0
        b[i] = v
    u = spsolve(A.tocsc(), b)
    if np.any(np.diff(u) <= 0):                             # out of order: back toward the targets
        for w in (0.5, 0.25, 0.0):
            v = w * u + (1 - w) * c
            if np.all(np.diff(v) > 0):
                return v
        return c.copy()
    return u


def build_harmonic(hm: HarmonicMap, spine_nodes: np.ndarray, chain_xy: np.ndarray, d: np.ndarray, h: np.ndarray,
                   band: float) -> BandBlock:
    """Steps 2-4: the band block on the harmonic map (rows, columns, cells, the corners at P and Q)."""
    K = len(d) - 1
    n_sp = len(spine_nodes) - 1
    S, T = rows_t(hm, d, band, n_sp)
    cols = hm.spine_s.copy()
    age = np.full(len(cols), 99)
    xs, ts = [spine_nodes.copy()], [cols.copy()]
    n_split = n_merge = 0
    chain_s = hm.chain_s
    refl = reflex_corners(np.asarray(chain_xy, float), spine_nodes[len(spine_nodes) // 2])
    conv = reflex_corners(np.asarray(chain_xy, float), spine_nodes[len(spine_nodes) // 2], convex=True)
    for k in range(1, K):
        row = _Row(hm, S, T[k])
        step = float(d[k] - d[k - 1])
        u = row.u(cols)
        n = len(u) - 1
        lim = max(1, int(BIRTH_FRAC * n))
        left = K - k                                       # rows to the chain (1 = the last row before it)
        if left <= ALIGN_ROWS:
            uc = row.u(chain_s)
            w = max(float(np.median(np.diff(uc))), 1e-9)
            if refl or conv:                               # a reflex corner of the chain takes two columns, a
                g = np.diff(uc)                            # convex one none
                uc = np.sort(np.r_[np.delete(uc, list(refl) + list(conv)), [uc[j] - 0.3 * g[j - 1] for j in refl],
                                   [uc[j] + 0.3 * g[j] for j in refl]])
            pairs, ua, ub = _align(u, uc, w)
            if ua or ub:
                u, age, a_, b_ = _events(u, age, h[k], step, max(lim, int(np.ceil((len(ua) + len(ub)) / left))),
                                         want=(ua, [uc[j] for j in ub]))
                n_split += a_
                n_merge += b_
            pairs, ua, ub = _align(u, uc, w)
            tgt = np.full(len(u), np.nan)
            for i, j in pairs:
                tgt[i] = uc[j]
            u = _drift(u, step, tgt, 1.0 / left)
        else:
            u, age, a_, b_ = _events(u, age, h[k], step, lim)
            n_split += a_
            n_merge += b_
            if RELAX_BETA > 0:
                u = _relax(u, RELAX_BETA, row.U[0], row.U[-1])
            else:
                u = _drift(u, step, age=age)
        u[0], u[-1] = row.U[0], row.U[-1]
        cols = row.s(u)
        cols[0], cols[-1] = 0.0, 1.0
        cols = np.maximum.accumulate(cols)
        xs.append(row.xy(cols))
        ts.append(cols.copy())
    xs.append(np.asarray(chain_xy, float))
    ts.append(chain_s.copy())
    N = np.array([len(t) - 1 for t in ts])
    off = np.r_[0, np.cumsum(N + 1)]
    allxy = np.vstack(xs)
    ql, tl = [], []
    for k in range(K):
        qq, tt = zip_rows(ts[k], ts[k + 1], np.arange(off[k], off[k + 1]), np.arange(off[k + 1], off[k + 2]), allxy)
        ql += qq
        tl += tt
    quads = np.array(ql, np.int64).reshape(-1, 4)
    tris = np.array(tl, np.int64).reshape(-1, 3)
    row = np.concatenate([np.full(N[k] + 1, k) for k in range(K + 1)])
    # 4. P and Q merged into row 1's ends on the banks
    P, Q = off[0], off[1] - 1
    Pn, Qn = off[1], off[2] - 1
    quads, tris = _collapse(quads, tris, {P: Pn, Q: Qn})
    keep = np.ones(len(allxy), bool)
    keep[[P, Q]] = False
    new = np.cumsum(keep) - 1
    quads, tris = new[quads], new[tris]
    allxy, row = allxy[keep], row[keep]
    off2 = off.copy()
    off2[1:] -= 2
    ts[0] = ts[0][1:-1]
    info = {"rows": int(K), "counts": [int(x) for x in N], "d_km": [round(float(x) / 1e3, 3) for x in d],
            "h_km": [round(float(x) / 1e3, 3) for x in h], "splits": int(n_split), "merges": int(n_merge),
            "map": hm.info, "rows_kind": "harmonic"}
    return BandBlock(allxy, quads, tris, row, off2, N, d, h, ts, info)


def reflex_corners(chain_xy: np.ndarray, inside: np.ndarray, deg: float = None, w: int = None,
                   convex: bool = False) -> list:
    """B197: the reflex bends of the host chain (it turns away from the band by at least REFLEX_DEG between the
    chords over block.CORNER_W edges before and after the node: a step of a grid-line staircase); the band's side
    from the point `inside`. Such a node takes two columns of the last row (a triangle between them). convex: the
    bends toward the band instead (no column ends there: the last row cuts the corner, B180's corner triangle)."""
    from meshom.insert.block import CORNER_W

    deg = REFLEX_DEG if deg is None else deg
    w = CORNER_W if w is None else w
    n = len(chain_xy) - 1
    if n < 2 * w:
        return []
    mid = chain_xy[n // 2]
    t = chain_xy[min(n // 2 + 1, n)] - chain_xy[max(n // 2 - 1, 0)]
    side = np.sign(t[0] * (inside[1] - mid[1]) - t[1] * (inside[0] - mid[0]))
    out = []
    for k in range(w, n - w + 1):
        a, b = chain_xy[k] - chain_xy[k - w], chain_xy[k + w] - chain_xy[k]
        turn = np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], (a * b).sum())) * side
        if (turn <= -deg if not convex else turn >= deg) and all(abs(k - j) >= 3 for j in out):
            out.append(int(k))
    return out


def _collapse(quads, tris, into: dict):
    """Each node a of `into` merged into into[a]: cells with both lose a corner (a quad -> a triangle, a triangle
    dropped), the others renamed."""
    q_out, t_out = [], list(tris)
    for c in quads:
        c = [into.get(int(v), int(v)) for v in c]
        u = [v for i, v in enumerate(c) if v != c[i - 1]]
        if len(u) == 4:
            q_out.append(u)
        elif len(u) == 3:
            t_out.append(u)
    t2 = []
    for c in t_out:
        c = [into.get(int(v), int(v)) for v in c]
        if len(set(c)) == 3:
            t2.append(c)
    return np.array(q_out, np.int64).reshape(-1, 4), np.array(t2, np.int64).reshape(-1, 3)


def _max_angles(xy, cell):
    """Largest corner (deg) of one cell, inf if it is not convex (counter-clockwise expected)."""
    p = xy[cell]
    a = np.roll(p, -1, 0) - p
    b = np.roll(p, 1, 0) - p
    cr = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    if np.any(cr <= 0):
        return np.inf
    return float(np.degrees(np.arctan2(cr, (a * b).sum(1))).max())


def polish(xy, quads, tris, fixed, deg: float, keep_rows=None, rounds: int = POLISH_ROUNDS):
    """B197: the free nodes of every quad with a corner above deg (and one node ring round it) moved one at a time
    to the trial position (POLISH_STEPS x the shortest incident edge in 8 directions, or the neighbours' mean) that
    lowers most the sum over their cells of the squared excess of the largest corner over POLISH_FREE_DEG (quads) or
    POLISH_TRI_DEG (triangles), every cell kept convex and every edge above POLISH_EDGE x its ends' shortest; at
    most `rounds` rounds. keep_rows: a mask of the cells it may judge (others are left to the ring cleaning).
    Returns (xy, record)."""
    xy = xy.copy()
    n = len(xy)
    cells = [q for q in quads] + [t for t in tris]
    inc = [[] for _ in range(n)]
    nbr = [set() for _ in range(n)]
    for k, c in enumerate(cells):
        m = len(c)
        for i in range(m):
            inc[int(c[i])].append(k)
            nbr[int(c[i])].update((int(c[(i + 1) % m]), int(c[(i - 1) % m])))
    judge = np.ones(len(cells), bool) if keep_rows is None else np.asarray(keep_rows, bool)
    dirs = np.c_[np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)]

    nq = len(quads)

    def worst(v, pos):                                      # the sum of squared excesses of the cells' largest
        old = xy[v].copy()                                  # corners over POLISH_FREE_DEG (quads) and
        xy[v] = pos                                         # POLISH_TRI_DEG (triangles); inf when one is invalid
        w = 0.0
        for k in inc[v]:
            a = _max_angles(xy, cells[k])
            w += max(0.0, a - (POLISH_FREE_DEG if k < nq else POLISH_TRI_DEG)) ** 2
        xy[v] = old
        return w

    e0 = np.array([min((np.linalg.norm(xy[w] - xy[v]) for w in nbr[v]), default=0.0) for v in range(n)])

    def short_ok(v, pos):                                   # no incident edge below POLISH_EDGE x the node's
        for w in nbr[v]:                                    # shortest edge before the polish (both ends)
            L = np.linalg.norm(xy[w] - pos)
            if L < POLISH_EDGE * e0[v] or L < POLISH_EDGE * e0[w]:
                return False
        return True

    def bad_quads():
        qa = np.array([_max_angles(xy, c) for c in cells[:nq]] + [0.0] * len(tris))
        return np.flatnonzero((qa > deg) & judge)

    bad = bad_quads()
    rec = {"bad_before": int(len(bad)), "bad_after": 0, "rounds": 0}
    for rnd in range(rounds):
        if rnd:
            bad = bad_quads()
        if not len(bad):
            break
        rec["rounds"] = rnd + 1
        core = {int(v) for k in bad for v in cells[k]}
        nodes = sorted(v for v in core | {w for v in core for w in nbr[v]} if not fixed[v])
        moved = False
        for v in nodes:
            nb = list(nbr[v])
            L = float(np.min(np.linalg.norm(xy[nb] - xy[v], axis=1)))
            s0 = worst(v, xy[v])
            best, bp = s0, None
            for c in [xy[nb].mean(0)] + [xy[v] + f * L * d for f in POLISH_STEPS for d in dirs]:
                sc = worst(v, c)
                if sc < best - 1e-9 and short_ok(v, c):
                    best, bp = sc, c
            if bp is not None:
                xy[v] = bp
                moved = True
        if not moved:
            break
    rec["bad_after"] = int(len(bad_quads()))                # what the returned positions leave
    return xy, rec
