"""m14 fixed blocks in straits (decisions S31, C35, B59): one boundary-fitted block per
passage of the water's medial axis, built before the automatic placement.

1. Passages: meshom.lfs (the pruned lambda-medial axis of the domain's coast, lambda = lfs.LAM h; the runs with
   W <= W_MAX h and |dW/ds| <= n g whose two ends are mouths), n = max(n_min, round(median W/h)) cells across. A
   passage whose block size W/n departs from h by more than e^0.10 in the median gets no block (h does not resolve it
   with n cells: the S22 zone of meshom.run, `[coast] auto_straits`, makes h = W/n there); nor does a block that
   passes 6. but whose edges depart from h itself by more than that at the 99th percentile (e.g. a basin block inside
   the graded zone of a narrower strait next to it).
2. Ends: at every station the cross line (the great circle through it, orthogonal to the centreline) is cut with the
   coast: the nearest crossing on each side, on the passage's left and right bank loops (the loops of most of the
   stations' Delaunay corners on that side), within HIT_REACH r; a section counts when both crossings exist and the
   chord between them is at most e^0.10 W (B22's tolerance: the cross line meets the banks about orthogonally). The
   block's ends are the cross chords at the first and the last station with a section (the stations between need
   none: the banks are the coast between the ends); a chord end within SNAP h_b of a coast corner, a loop vertex
   turning by more than 30 degrees (C17), moves onto it. Fewer than MIN_M cells along: no block.
3. Sides (counter-clockwise, block on the left): the right bank forward along its loop (water on the left, so the
   loop runs with the centreline there), the far end chord, the left bank forward along its loop, the near end chord;
   the banks Gaussian-filtered along their arc length at sigma = FILTER h_b with the ends kept (C35: 2 h). A vertex of
   another coast loop inside this outline (an island in the strait) leaves the passage to the fill.
4. Size inside: h_b = W(s) / n (S24), W smoothed along the centreline at sigma = FILTER h_b, s the station nearest to
   the point; m = round(integral ds / h_b along the centreline).
5. Map (a planar mesh in the gnomonic plane of the block, not sphere_delaunay):
   the sides sampled at BG h_b with the Dirichlet data xi = the normalised integral of ds/h_b along the S and N sides
   (0 / 1 on the W / E sides), eta likewise across; interior points from the Coons patch of the sides at BG h_b; a
   conforming Delaunay triangulation of the polygon in the gnomonic plane (scipy; a missing side segment loses the
   interior points in its diametral circle or is split, at most CDT_ROUNDS rounds); xi, eta harmonic (cotangent
   weights of the chord triangles, clipped >= 0: one-to-one onto the unit square, Tutte / Floater); the interior nodes
   where (m xi, n eta) is an integer pair (blockmap.lattice), the sides resampled at h_b with m and n steps; then 30
   "size" sweeps (B25) with the sides fixed. With h_b = W/n plain (unit) weights make square cells: the flux
   W xi'(s) = const gives xi' ~ 1/W ~ 1/h_b.
6. Acceptance (B22): every block edge |ln s/h_b| <= 0.10, every quad angle <= 135 degrees, no inverted cell. A block
   that fails is cut at its worst cross line (the column of cells with the largest excess), the stations within W/2 of
   it dropped, and both parts tried again (depth <= SPLIT_DEPTH); what still fails is left to the fill.
7. Blocks are taken largest first; one that comes within block_gap h of an accepted block of another passage is
   dropped (the two pieces of a split are W apart by construction and both stay).
A FixedBlock carries the BlockGrid (outline with the fill on the left), the coast mask (bank nodes) and its non-coast
outline runs (the end chords) with their end points located on the domain loops (loop, segment, t), which
stitch.merged_loops needs; the record goes to m14.json ("fixed_blocks"), the ends as "mouths" (as corridors).
"""
from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay, cKDTree

from meshom import lfs
from meshom.domain import Domain, _turn_angles
from meshom.geo import R_EARTH_M, arc_length, tangent_basis, xyz_to_lonlat
from meshom.mesh import Mesh, signed_area
from meshom.methods.m14.blockmap import BlockGrid, lattice
from meshom.quality.kernel import element_geometry
from meshom.size import SizeField, scaled

HIT_REACH = 1.5          # a bank crossing counts within this many medial radii of the station
SNAP = 0.5               # an end within this many h_b of a coast corner moves onto it
CORNER_DEG = 30.0        # a coast corner: a loop vertex turning by more than this (C17)
FILTER = 2.0             # C35: banks and W(s) filtered at sigma = FILTER h_b
BG = 0.33                # map mesh spacing (units of h_b)
CDT_ROUNDS = 30
SPLIT_DEPTH = 2
MIN_M = 2                # S31: a block has at least this many cells along
TRIM = 0.10              # B22
THETA_MAX = 135.0        # B22


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _seg_m(p: np.ndarray) -> np.ndarray:
    return R_EARTH_M * arc_length(p[:-1], p[1:])


def densify(p: np.ndarray, step_m: float) -> np.ndarray:
    """Open polyline with extra points so no segment is longer than step_m."""
    k = np.maximum(1, np.ceil(_seg_m(p) / step_m).astype(np.int64))
    out = [p[i] + (p[i + 1] - p[i]) * (np.arange(k[i])[:, None] / k[i]) for i in range(len(p) - 1)]
    return _unit(np.vstack(out + [p[-1:]]))


def h_integral(p: np.ndarray, h) -> np.ndarray:
    """Cumulative integral of ds/h along an open polyline (trapezoid)."""
    hv = h(p)
    return np.r_[0.0, np.cumsum(_seg_m(p) * 0.5 * (1.0 / hv[:-1] + 1.0 / hv[1:]))]


def resample(p: np.ndarray, h, n: int, step_m: float) -> np.ndarray:
    """n equal steps of the integral of ds/h along the open polyline p (densified at step_m): (n+1,3), ends kept."""
    q = densify(p, step_m)
    c = h_integral(q, h)
    u = np.linspace(0.0, c[-1], n + 1)
    j = np.clip(np.searchsorted(c, u, side="right") - 1, 0, len(q) - 2)
    f = ((u - c[j]) / np.maximum(c[j + 1] - c[j], 1e-300))[:, None]
    out = _unit((1.0 - f) * q[j] + f * q[j + 1])
    out[0], out[-1] = q[0], q[-1]
    return out


def smooth_polyline(p: np.ndarray, sigma_m: float, step_m: float) -> np.ndarray:
    """Gaussian smoothing of an open polyline along its arc length on the sphere, ends kept (odd reflection)."""
    q = densify(p, step_m)
    s = np.r_[0.0, np.cumsum(_seg_m(q))]
    if s[-1] <= 2 * step_m or sigma_m <= 0:
        return q
    su = np.linspace(0.0, s[-1], int(np.ceil(s[-1] / step_m)) + 1)
    qu = _unit(np.column_stack([np.interp(su, s, q[:, d]) for d in range(3)]))
    w = min(int(np.ceil(3.0 * sigma_m / (su[1] - su[0]))), len(qu) - 1)
    ker = np.exp(-0.5 * (np.arange(-w, w + 1) * (su[1] - su[0]) / sigma_m) ** 2)
    ker /= ker.sum()
    pad = np.vstack([2 * qu[:1] - qu[w:0:-1], qu, 2 * qu[-1:] - qu[-2:-w - 2:-1]])
    out = _unit(np.column_stack([np.convolve(pad[:, d], ker, mode="valid") for d in range(3)]))
    out[0], out[-1] = qu[0], qu[-1]
    return out


def _gauss(s: np.ndarray, v: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of v(s) on irregular stations s (normalised weights, so the ends are not biased)."""
    if sigma <= 0 or len(s) < 3:
        return v.copy()
    out = np.empty_like(v)
    for i in range(len(s)):
        lo, hi = np.searchsorted(s, s[i] - 3 * sigma), np.searchsorted(s, s[i] + 3 * sigma, side="right")
        w = np.exp(-0.5 * ((s[lo:hi] - s[i]) / sigma) ** 2)
        out[i] = np.sum(w * v[lo:hi]) / np.sum(w)
    return out


@dataclass(frozen=True, eq=False)
class WidthField:
    """h_b = W(s)/n at the station nearest to the point (S24)."""
    stations: np.ndarray
    W: np.ndarray
    n: int

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        _, j = cKDTree(self.stations).query(xyz)
        return self.W[j] / self.n


@dataclass
class FixedBlock:
    grid: BlockGrid
    coast: np.ndarray        # (Nb,) bool: node on a coast side
    runs: list               # [(node ids, (loop, seg, t) of the first node, (loop, seg, t) of the last)]: the
                             # non-coast outline runs, in outline order (fill on the left), coast node to coast node
    record: dict = field(default_factory=dict)

    def bank(self) -> np.ndarray:
        """(K,) bool over the outline edges (edge k: outline[k] -> outline[k+1]): the edge lies on a coast side (a
        bank). Both its nodes are coast nodes, and it is no edge of a non-coast side: of a run, or, for a hand block
        built without runs (cut mode, m14/hand.py), of a side whose kind is not "coast" (side 0: j = 0, 1: i = m,
        2: j = n, 3: i = 0 of grid.ij). The node mask alone cannot tell them apart where a non-coast side is one
        cell long between two coast sides (a block one cell wide across a strait: its mouths)."""
        o = self.grid.outline
        a, b = o, np.roll(o, -1)
        bank = self.coast[a] & self.coast[b]
        if self.runs:
            nb = len(self.coast)
            ids = [r[0] for r in self.runs]
            key = np.concatenate([np.minimum(x[:-1], x[1:]) * nb + np.maximum(x[:-1], x[1:]) for x in ids])
            bank &= ~np.isin(np.minimum(a, b) * nb + np.maximum(a, b), key)
        elif self.record.get("kind") == "hand":
            i, j = self.grid.ij[:, 0], self.grid.ij[:, 1]
            on = [j == 0, i == i.max(), j == j.max(), i == 0]
            for k, kind in enumerate(self.record["sides"]):
                if kind != "coast":
                    bank &= ~(on[k][a] & on[k][b])
        return bank


# ---------------------------------------------------------------------------------------------- cross sections

class _Coast:
    """The domain loops as segments with a k-d tree over their vertices."""

    def __init__(self, loops: list):
        self.loops = loops
        self.V = np.vstack(loops)
        self.loop = np.repeat(np.arange(len(loops)), [len(lp) for lp in loops])
        self.idx = np.concatenate([np.arange(len(lp)) for lp in loops])
        self.off = np.r_[0, np.cumsum([len(lp) for lp in loops])]
        nxt = np.vstack([np.roll(lp, -1, axis=0) for lp in loops])
        self.seg = float((R_EARTH_M * arc_length(self.V, nxt)).max())
        self.tree = cKDTree(self.V)
        self.corner = np.concatenate([_turn_angles(lp) > np.radians(CORNER_DEG) for lp in loops])

    def crossings(self, c: np.ndarray, t: np.ndarray, reach_m: float):
        """Crossings of the great circle through c orthogonal to t with the coast within reach_m of c:
        (points (k,3), loop (k,), seg (k,), param (k,), side (k,): +1 left of t, -1 right)."""
        rad = (reach_m + self.seg) / R_EARTH_M
        cand = np.asarray(self.tree.query_ball_point(c, 2 * np.sin(0.5 * rad)), dtype=np.int64)
        if len(cand) == 0:
            return None
        lp, k = self.loop[cand], self.idx[cand]
        segs = np.unique(np.r_[self.off[lp] + k, self.off[lp] + (k - 1) % (self.off[lp + 1] - self.off[lp])])
        L = self.loop[segs]
        a = self.V[segs]
        b = self.V[self.off[L] + (self.idx[segs] + 1) % (self.off[L + 1] - self.off[L])]
        sa, sb = a @ t, b @ t
        x = (sa * sb <= 0) & (sa != sb)
        if not x.any():
            return None
        segs, L, a, b, sa, sb = segs[x], L[x], a[x], b[x], sa[x], sb[x]
        u = sa / (sa - sb)
        p = _unit(a + u[:, None] * (b - a))
        near = (p @ c > 0) & (R_EARTH_M * arc_length(p, c[None, :]) <= reach_m)
        side = np.sign(p @ np.cross(c, t))
        return p[near], L[near], self.idx[segs][near], u[near], side[near]

    def islands_inside(self, poly: np.ndarray, banks: set) -> int:
        """Vertices of loops other than `banks` inside the closed polygon poly (a block outline): an island in the
        strait, which a block must not cover."""
        c0 = _unit(poly.sum(axis=0))
        rad = float(arc_length(poly, c0[None, :]).max())
        cand = np.asarray(self.tree.query_ball_point(c0, 2 * np.sin(0.5 * rad)), dtype=np.int64)
        cand = cand[~np.isin(self.loop[cand], list(banks))] if len(cand) else cand
        if len(cand) == 0:
            return 0
        proj = _gnomonic(c0)
        return int(Path(proj(poly)).contains_points(proj(self.V[cand])).sum())


def _tangents(P: np.ndarray, s: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Unit centreline tangents (along increasing s) by differences over W/2 on each side, in the tangent planes."""
    i = np.arange(len(s))
    lo = np.minimum(np.searchsorted(s, s - 0.5 * W), np.maximum(i - 1, 0))          # at least the neighbours
    hi = np.maximum(np.minimum(np.searchsorted(s, s + 0.5 * W, side="right") - 1, len(s) - 1),
                    np.minimum(i + 1, len(s) - 1))
    d = P[hi] - P[lo]
    d -= np.einsum("ij,ij->i", d, P)[:, None] * P
    return _unit(d)


def _sections(ma: lfs.MedialAxis, p: lfs.Passage, coast: _Coast):
    """Per station: (left hit, right hit) as (point, loop, seg, t) or None; the bank loops."""
    P, W = p.points, p.W
    T = _tangents(P, p.s, W)
    left = np.cross(P, T)
    tri = ma.tri[p.ids]
    sgn = np.einsum("skd,sd->sk", ma.samples[tri] - P[:, None, :], left)
    lo = ma.loop[tri]
    bank_l = np.bincount(lo[sgn > 0], minlength=1).argmax() if (sgn > 0).any() else -1
    bank_r = np.bincount(lo[sgn < 0], minlength=1).argmax() if (sgn < 0).any() else -1
    hits = []
    for i in range(len(P)):
        x = coast.crossings(P[i], T[i], HIT_REACH * 0.5 * W[i])
        hit = [None, None]
        if x is not None:
            pts, L, seg, u, side = x
            d = arc_length(pts, P[i][None, :])
            for k, (sd, bank) in enumerate(((1.0, bank_l), (-1.0, bank_r))):
                m = side == sd
                if m.any():
                    j = np.flatnonzero(m)[np.argmin(d[m])]
                    if L[j] == bank:
                        hit[k] = (pts[j], int(L[j]), int(seg[j]), float(u[j]))
        if hit[0] is not None and hit[1] is not None and \
                R_EARTH_M * float(arc_length(hit[0][0][None], hit[1][0][None])[0]) > np.exp(TRIM) * W[i]:
            hit = [None, None]                        # the cross line leaves the water obliquely: no section here
        hits.append(hit)
    return T, hits, int(bank_l), int(bank_r)


def _snap(hit: tuple, coast: _Coast, hb: float) -> tuple:
    """Move a crossing onto a coast corner within SNAP hb of it (C17 corners)."""
    pt, L, seg, u = hit
    K = coast.off[L + 1] - coast.off[L]
    for v in (seg, (seg + 1) % K):
        g = coast.off[L] + v
        if coast.corner[g] and R_EARTH_M * float(arc_length(coast.V[g][None], pt[None])[0]) <= SNAP * hb:
            return coast.V[g].copy(), L, int(v), 0.0
    return hit


def _bank(coast: _Coast, a: tuple, b: tuple) -> np.ndarray:
    """The coast from crossing a forward along its loop to crossing b (same loop): points, a and b included."""
    L = a[1]
    lp = coast.loops[L]
    K = len(lp)
    cnt = (b[2] - a[2]) % K
    if cnt == 0 and b[3] < a[3]:
        cnt = K
    idx = (a[2] + 1 + np.arange(cnt)) % K
    q = np.vstack([a[0][None], lp[idx], b[0][None]])
    keep = np.r_[True, np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-12]
    return q[keep]


# ----------------------------------------------------------------------------------------------- harmonic map

def _gnomonic(c0: np.ndarray):
    e1, e2 = tangent_basis(c0[None])
    e1, e2 = e1[0], e2[0]

    def f(p):
        d = p @ c0
        return np.column_stack([p @ e1 / d, p @ e2 / d])
    return f


def _cdt(B: np.ndarray, I: np.ndarray, data: np.ndarray, proj):
    """Conforming Delaunay triangulation of the closed polygon B (sphere points, CCW) with interior points I in the
    gnomonic plane: a missing polygon segment drops the interior points in its diametral circle, else it is split
    (data (len(B), k) interpolated linearly). Returns (points, tris (CCW, inside the polygon), boundary count, data)."""
    for _ in range(CDT_ROUNDS):
        X = proj(np.vstack([B, I]))
        T = Delaunay(X).simplices
        nb = len(B)
        e = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
        have = set(map(tuple, np.unique(e, axis=0).tolist()))
        k = np.arange(nb)
        seg = np.sort(np.column_stack([k, (k + 1) % nb]), axis=1)
        miss = np.array([tuple(s) not in have for s in seg.tolist()])
        if not miss.any():
            break
        Xb, Xi = X[:nb], X[nb:]
        drop = np.zeros(len(I), bool)
        split = []
        for j in np.flatnonzero(miss):
            a, b = Xb[j], Xb[(j + 1) % nb]
            m, r2 = 0.5 * (a + b), 0.25 * ((b - a) ** 2).sum()
            enc = ((Xi - m) ** 2).sum(axis=1) < r2
            if enc.any():
                drop |= enc
            else:
                split.append(j)
        I = I[~drop]
        if split:
            nB, nD = [], []
            sp = set(split)
            for j in range(nb):
                nB.append(B[j][None])
                nD.append(data[j][None])
                if j in sp:
                    nB.append(_unit(B[j] + B[(j + 1) % nb])[None])
                    nD.append(0.5 * (data[j] + data[(j + 1) % nb])[None])
            B, data = np.vstack(nB), np.vstack(nD)
    pts = np.vstack([B, I])
    X = proj(pts)
    T = Delaunay(X).simplices
    cen = X[T].mean(axis=1)
    T = T[Path(X[:len(B)]).contains_points(cen)]
    a, b, c = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
    neg = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]) < 0
    T[neg] = T[neg][:, [0, 2, 1]]
    return pts, T, len(B), data


def _interior(sides: list, hb: SizeField, m: int, n: int, B: np.ndarray, proj, c0: np.ndarray, step: float):
    """Interior points of the map mesh: the Coons patch (TFI) of the sides at (m, n) / BG cells in the gnomonic
    plane, inside the polygon, at least BG h_b / 2 from its samples and BG h_b / 2 from each other."""
    k = int(np.ceil(1.0 / BG))
    S = proj(resample(sides[0], hb, k * m, step))
    E = proj(resample(sides[1], hb, k * n, step))
    N = proj(resample(sides[2], hb, k * m, step)[::-1])
    Wd = proj(resample(sides[3], hb, k * n, step)[::-1])
    u = np.linspace(0, 1, k * m + 1)[1:-1, None, None]
    v = np.linspace(0, 1, k * n + 1)[None, 1:-1, None]
    X = ((1 - v) * S[1:-1, None] + v * N[1:-1, None] + (1 - u) * Wd[None, 1:-1] + u * E[None, 1:-1]
         - ((1 - u) * (1 - v) * S[0] + u * (1 - v) * S[-1] + (1 - u) * v * N[0] + u * v * N[-1])).reshape(-1, 2)
    X = X[Path(proj(B)).contains_points(X)]
    e1, e2 = tangent_basis(c0[None])
    I = _unit(c0[None] + X[:, :1] * e1 + X[:, 1:] * e2)
    r = 0.5 * BG * hb(I) / R_EARTH_M
    d, _ = cKDTree(B).query(I)
    I, r = I[d > r], r[d > r]
    drop = np.zeros(len(I), bool)
    near = cKDTree(I).query_ball_point(I, r)             # local radii: h_b may vary 10-fold along a strait
    for i in range(len(I)):
        if drop[i]:
            continue
        for j in near[i]:
            if j > i and not drop[j] and np.linalg.norm(I[i] - I[j]) < r[j]:
                drop[j] = True
    return I[~drop]


def harmonic_grid(sides: list, hb: SizeField, m: int, n: int) -> tuple[np.ndarray | None, dict]:
    """The harmonic map on the sides (4 open polylines, CCW, block on the left; S and N with m cells, E and W with n):
    returns (P (m+1, n+1, 3) or None when the map misses a lattice point, info)."""
    hf = scaled(hb, BG)
    step = float(np.min(hb(np.vstack(sides)))) * BG / 3.0
    bl, bd = [], []
    for k, sd in enumerate(sides):
        q = densify(sd, step)
        nk = max(2, int(round(h_integral(q, hf)[-1])))
        r = resample(q, hf, nk, step)
        u = h_integral(r, hb)
        u = (u / u[-1])[:-1]
        r = r[:-1]
        xi, et = [(u, 0 * u), (1 + 0 * u, u), (1 - u, 1 + 0 * u), (0 * u, 1 - u)][k]
        bl.append(r)
        bd.append(np.column_stack([xi, et]))
    B, D = np.vstack(bl), np.vstack(bd)
    c0 = _unit(B.sum(axis=0))
    proj = _gnomonic(c0)
    I = _interior(sides, hb, m, n, B, proj, c0, step)
    pts, T, nb, D = _cdt(B, I, D, proj)
    q = pts[T]
    rows, cols, vals = [], [], []
    for k in range(3):
        u_ = q[:, (k + 1) % 3] - q[:, k]
        v_ = q[:, (k + 2) % 3] - q[:, k]
        w = 0.5 * np.einsum("ij,ij->i", u_, v_) / np.linalg.norm(np.cross(u_, v_), axis=1)
        i, j = T[:, (k + 1) % 3], T[:, (k + 2) % 3]
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    Wm = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                    shape=(len(pts), len(pts))).tocsr()
    neg = int((Wm.data < 0).sum())
    Wm.data = np.maximum(Wm.data, 1e-12 * np.abs(Wm.data).max())       # Tutte: non-negative weights
    Lp = diags(np.asarray(Wm.sum(axis=1)).ravel()) - Wm
    inn, bnd = np.arange(nb, len(pts)), np.arange(nb)
    A = Lp[inn][:, inn].tocsc()
    sol = []
    for d in range(2):
        x = np.zeros(len(pts))
        x[bnd] = D[:, d]
        x[inn] = spsolve(A, -Lp[inn][:, bnd] @ D[:, d])
        sol.append(x)
    Wc = (m * sol[0] + 1j * n * sol[1])[T]
    ij, xyz, _ = lattice(pts, T, Wc, np.ones(len(T), bool))
    P = np.full((m + 1, n + 1, 3), np.nan)
    sel = (ij[:, 0] > 0) & (ij[:, 0] < m) & (ij[:, 1] > 0) & (ij[:, 1] < n)
    P[ij[sel, 0], ij[sel, 1]] = xyz[sel]
    S = resample(sides[0], hb, m, step)
    E = resample(sides[1], hb, n, step)
    N = resample(sides[2], hb, m, step)[::-1]
    Wd = resample(sides[3], hb, n, step)[::-1]
    P[:, 0], P[:, -1], P[0, :], P[-1, :] = S, N, Wd, E
    miss = int(np.isnan(P[..., 0]).sum())
    flipped = int((np.imag(np.conj(Wc[:, 1] - Wc[:, 0]) * (Wc[:, 2] - Wc[:, 0])) <= 0).sum())
    info = dict(map_nodes=len(pts), map_tris=len(T), map_neg_weights=neg, map_flipped=flipped, missing=miss)
    return (None if miss else P), info


def grid_block(P: np.ndarray, hb: SizeField, iters: int) -> tuple[BlockGrid, np.ndarray, dict]:
    """(m+1, n+1, 3) nodes -> BlockGrid (quads CCW, outline with the fill on the left), coast mask (j = 0 and j = n),
    after `iters` "size" sweeps with the sides fixed; and the B22 numbers."""
    from meshom.smooth import smooth

    m, n = P.shape[0] - 1, P.shape[1] - 1
    nid = np.arange((m + 1) * (n + 1)).reshape(m + 1, n + 1)
    q = np.column_stack([nid[:-1, :-1].ravel(), nid[1:, :-1].ravel(), nid[1:, 1:].ravel(), nid[:-1, 1:].ravel()])
    pts = _unit(P.reshape(-1, 3))
    mesh = Mesh(pts, np.zeros((0, 3), np.int64), q, np.zeros(len(pts), np.int8))
    side = np.zeros((m + 1, n + 1), bool)
    side[0, :] = side[-1, :] = side[:, 0] = side[:, -1] = True
    if iters:
        mesh = smooth(mesh, hb, side.ravel(), iters=iters, kind="size")
    ccw = np.r_[nid[:m, 0], nid[m, :n], nid[m:0:-1, n], nid[0, n:0:-1]]
    outline = ccw[::-1].copy()
    coast = np.zeros((m + 1, n + 1), bool)
    coast[:, 0] = coast[:, n] = True
    ij = np.column_stack(np.divmod(np.arange(len(pts)), n + 1))
    g = element_geometry(mesh)
    e = np.vstack([q[:, [0, 1]], q[:, [1, 2]], q[:, [2, 3]], q[:, [3, 0]]])
    a, b = mesh.points[e[:, 0]], mesh.points[e[:, 1]]
    lnsh = np.abs(np.log(R_EARTH_M * arc_length(a, b) / hb(_unit(a + b))))
    th = np.degrees(g.quad_angles).max(axis=1)
    acc = dict(lnsh_p99=float(np.percentile(lnsh, 99)), lnsh_max=float(lnsh.max()),
               quad_max_deg=float(th.max()), quad_p999_deg=float(np.percentile(th, 99.9)),
               inverted=int((signed_area(mesh) <= 0).sum()))
    # per column of cells: the worst excess over B22 (for the split)
    cell_l = lnsh.reshape(4, -1).max(axis=0)
    excess = np.maximum(cell_l / TRIM, th / THETA_MAX).reshape(m, n).max(axis=1)
    acc["worst_column"] = int(np.argmax(excess))
    acc["worst_excess"] = float(excess.max())
    grid = BlockGrid(mesh.points, q, ij, outline, dict(n_cells=m * n, inradius_cells=float(min(m, n)) / 2))
    return grid, coast.ravel(), acc


# ------------------------------------------------------------------------------------------------- the blocks

@dataclass
class _Ctx:
    p: lfs.Passage
    Ws: np.ndarray           # W smoothed along the centreline
    hits: list               # per station [left, right] crossing or None
    coast: _Coast
    h: SizeField
    o: dict
    base: dict


def _block(c: _Ctx, a: int, b: int, kinds: tuple, depth: int) -> list:
    """Fixed blocks of the passage stations [a, b] (recursive split); returns [(FixedBlock | None, record)]."""
    p, W, s, n = c.p, c.Ws, c.p.s, c.p.n
    rec = dict(c.base, stations=[int(a), int(b)], depth=depth)
    hbf = WidthField(p.points[a:b + 1], W[a:b + 1], n)
    hb_med = float(np.median(W[a:b + 1])) / n
    m = int(round(float(h_integral(p.points[a:b + 1], hbf)[-1])))
    rec.update(n=n, m=m, hb_min_m=round(float(W[a:b + 1].min()) / n, 1),
               hb_max_m=round(float(W[a:b + 1].max()) / n, 1), length_m=round(float(s[b] - s[a]), 1))
    if m < MIN_M or b - a < 2 or c.hits[a][0] is None or c.hits[b][0] is None:
        rec["status"] = "short"
        return [(None, rec)]
    (l0, r0), (l1, r1) = [[_snap(c.hits[i][k], c.coast, float(W[i]) / n) for k in (0, 1)] for i in (a, b)]
    step = hb_med / 4.0
    bank_r = smooth_polyline(_bank(c.coast, r0, r1), FILTER * hb_med, step)
    bank_l = smooth_polyline(_bank(c.coast, l1, l0), FILTER * hb_med, step)
    lim = 3.0 * float(s[b] - s[a]) + 4.0 * hb_med
    if _seg_m(bank_r).sum() > lim or _seg_m(bank_l).sum() > lim:
        rec["status"] = "bank"                        # a crossing on the far side of the loop: no bank piece
        return [(None, rec)]
    sides = [bank_r, densify(np.vstack([r1[0], l1[0]]), step), bank_l, densify(np.vstack([l0[0], r0[0]]), step)]
    if min(float(h_integral(sd, hbf)[-1]) for sd in sides) < 0.5:
        rec["status"] = "degenerate"                  # a side shorter than half a cell (ends snapped together)
        return [(None, rec)]
    isl = c.coast.islands_inside(np.vstack([sd[:-1] for sd in sides]), {l0[1], r0[1], l1[1], r1[1]})
    if isl:
        rec.update(status="island", island_vertices=isl)   # an island inside the block: left to the fill
        return [(None, rec)]
    P, minfo = harmonic_grid(sides, hbf, m, n)
    rec.update(minfo)
    if P is None:
        rec["status"] = "map"
        return [(None, rec)]
    grid, cm, acc = grid_block(P, hbf, c.o["block_size_iters"])
    rec.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in acc.items()})
    if acc["lnsh_max"] <= TRIM and acc["quad_max_deg"] <= THETA_MAX and acc["inverted"] == 0:
        e = np.vstack([grid.quads[:, [0, 1]], grid.quads[:, [1, 2]]])
        pa, pb = grid.points[e[:, 0]], grid.points[e[:, 1]]
        rec["lnsh_h_p99"] = round(float(np.percentile(np.abs(np.log(R_EARTH_M * arc_length(pa, pb)
                                                                       / c.h(_unit(pa + pb)))), 99)), 4)
        if rec["lnsh_h_p99"] > TRIM:                  # the block does not follow h (another zone's grading)
            rec["status"] = "h_mismatch"
            return [(None, rec)]
        rec["status"] = "accepted"
        nid = np.arange((m + 1) * (n + 1)).reshape(m + 1, n + 1)
        runs = [(nid[0, :].copy(), r0[1:], l0[1:]), (nid[m, ::-1].copy(), l1[1:], r1[1:])]
        rec["mouths"] = []
        for i, (lh, rh), kind in ((a, (l0, r0), kinds[0]), (b, (l1, r1), kinds[1])):
            lo, la = xyz_to_lonlat(np.vstack([rh[0], lh[0]]))
            rec["mouths"].append({"lonlat": np.round(np.column_stack([lo, la]), 4).tolist(), "cells": n,
                                  "W_m": round(float(W[i]), 1), "kind": kind})
        return [(FixedBlock(grid, cm, runs, rec), rec)]
    rec["status"] = "rejected"
    if depth >= SPLIT_DEPTH:
        return [(None, rec)]
    # split at the worst cross line: the station at its share of the integral ds/h_b, +- W/2 dropped
    cum = h_integral(p.points[a:b + 1], hbf)
    k = a + int(np.argmin(np.abs(cum - (acc["worst_column"] + 0.5) / m * cum[-1])))
    lo = int(np.searchsorted(s, s[k] - 0.5 * W[k]))
    hi = int(np.searchsorted(s, s[k] + 0.5 * W[k], side="right")) - 1
    rec["split_at"] = int(k)
    out = [(None, rec)]
    ok = np.flatnonzero([x[0] is not None and x[1] is not None for x in c.hits])
    for a2, b2, kk in ((a, lo - 1, (kinds[0], "split")), (hi + 1, b, ("split", kinds[1]))):
        sec = ok[(ok >= a2) & (ok <= b2)]                # the piece ends at stations with a section
        if len(sec) >= 2 and sec[-1] - sec[0] >= 2:
            out += _block(c, int(sec[0]), int(sec[-1]), kk, depth + 1)
    return out


def strait_blocks(domain: Domain, h: SizeField, o: dict, avoid: list = ()) -> tuple[list[FixedBlock], dict]:
    """The fixed strait blocks of the domain (module docstring) and a record of every passage piece tried; avoid:
    the nodes of fixed blocks placed before (the hand-placed `[[blocks]]`, m14/hand.py), kept block_gap h away too."""
    ma = lfs.medial_axis(domain.loops, scaled(h, lfs.STEP), scaled(h, lfs.LAM))
    ps = lfs.passages(ma, h, o["strait_n_min"], o["strait_w_max"], o["strait_grad"])
    coast = _Coast(domain.loops)
    tried = []
    for pi, p in enumerate(ps):
        dev = abs(float(np.median(np.log(p.W / p.n / h(p.points)))))
        if dev > TRIM:                                # h does not resolve the strait at n cells (no S22 zone)
            tried.append((None, {"passage": pi, **p.record(), "ln_hb_h_median": round(dev, 4),
                                 "status": "h_mismatch"}))
            continue
        T, hits, bl, br = _sections(ma, p, coast)
        valid = np.flatnonzero([x[0] is not None and x[1] is not None for x in hits])
        base = {"passage": pi, **p.record(), "bank_loops": [bl, br], "ln_hb_h_median": round(dev, 4)}
        if len(valid) < 2:
            tried.append((None, dict(base, status="no_section")))
            continue
        a, b = int(valid[0]), int(valid[-1])
        Ws = _gauss(p.s, p.W, FILTER * float(np.median(p.W)) / p.n)
        tried += _block(_Ctx(p, Ws, hits, coast, h, o, base), a, b, p.ends, 0)
    order = sorted([t for t in tried if t[0] is not None], key=lambda t: -len(t[0].grid.quads))
    blocks, pts = [], [(q, -1) for q in avoid]
    for fb, rec in order:                             # largest first; one within block_gap h of another passage's
        other = [q for q, pi in pts if pi != rec["passage"]]    # block drops (the pieces of a split are W apart)
        if other:
            d, _ = cKDTree(np.vstack(other)).query(fb.grid.points)
            if (R_EARTH_M * 2 * np.arcsin(np.clip(0.5 * d, 0, 1)) / h(fb.grid.points)).min() < o["block_gap"]:
                rec["status"] = "overlap"
                continue
        rec["fixed_index"] = len(blocks)
        blocks.append(fb)
        pts.append((fb.grid.points, rec["passage"]))
    return blocks, {"axis": ma.info, "n_passages": len(ps), "tried": [r for _, r in tried]}
