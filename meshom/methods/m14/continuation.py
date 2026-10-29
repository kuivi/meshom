"""Lattice continuation: the fill of a corridor between two block lattices by the integer points of a blended map
("cont", the lattice continuation of the quantised join, B57).

Chart. Everything runs in the stereographic plane at the corridor's mean point (conformal: a block lattice stays a
square lattice); the cell unit hc is the median cell size of the two outline faces in that plane.

Block maps, extrapolated (`Side`). Per outline-run node k of a block, the affine map of its outline face
u = l_k + G_k (x - x_k): l_k its lattice index, G_k by least squares on the block nodes within one lattice step of it
(index box). A point x in the gap takes the Gaussian-weighted mean (sigma = 1 hc, the 8 nearest run nodes) of those
maps evaluated at x: exact for a lattice that is affine near the outline, an affine extrapolation of a curved one.
B's indices are turned into A's frame by the 90-degree rotation M nearest to G_A G_B^-1 (both lattices CCW).

Blended map. t = d_A / (d_A + d_B) (distances to the two runs), u = (1 - t) u_A + t (M u_B + T), T an integer vector
per corridor position s (arc length of the mid-line between the runs, in cells): it equals A's indices on A's run and
B's (shifted by T) on B's. T steps by one unit vector at each event of the schedule (`schedule`); an event is one
dislocation, its core where the step meets t = 1/2, i.e. on the corridor's middle row:
  "even"  (B58) each component's events where delta = u_A - M u_B crosses a half-integer (hysteresis eta), then
          all events re-spaced evenly in the accrued count (the L1 variation of delta along s), their order kept: the
          |b|_1 dislocations at spacing L / |b|_1, along and normal kinds alternating as they accrue (Bresenham);
  "round" the events where they fall (T = round(delta) per component);
  "none"  one integer transition for the whole corridor (T = round(delta) at its middle): the mismatch as a shear,
          every dislocation pushed to the corridor ends (for b_perp = 0, |b_par| <= w);
  "rule"  (default) the corridor rule (stitch_mix.quantise, B47) on b from the maps (`Corridor.burgers_split`):
          "none" for b = 0 and for the shear, else "even".

Points (`gap_points`). Per constant-T segment the integer points of its map whose s lies within cut_win cells of the
segment (Newton on the blended Jacobian from the segment's affine guess), inside the corridor loop, >= d_loop cells
from it; then thinned to >= min_sep cells apart, the points deepest inside their own segment first (so a lattice
site cut off by the step of T on both sides is still made once). Each keeps its label (its integer u) and a second label v = u - [t > 1/2] T: across a step of T the
lattice sites match by u below the middle row and by v above it.

Fill (`fill`). Constrained Delaunay of the loop and the points; label quads first (`pair_labels`: a triangle pair whose
four labels, in u or in v, form a unit CCW square, by increasing quad cost), the rest by recombine (greedy, price; a
triangle touching an outline run costs more); a triangle still touching a run is moved one element inward
(`lift_triangles`); then m02's smoothing ("size", "angle"; loop nodes fixed). The triangulation closes the hull with one antipodal helper
point (a local CDT).
"""
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

from meshom.geo import tangent_basis
from meshom.mesh import Mesh

DEFAULTS = dict(
    schedule="rule",    # "rule" (the corridor rule on the maps' b: "none" for b = 0 and the shear, else "even") |
                        # "even" | "round" | "none" (module docstring)
    eta=0.15,           # hysteresis of a component's event (cells of index): noise in delta makes no +- pairs
                        # end_margin (no default, the caller's): "even": no event within this many cells of a corridor
                        # end, (at C0, at C1) (B58; stitch_mix._cont_opts sets it from m14_blocks' end_rows)
    sigma=1.0,          # Gaussian width (cells) of the outline-face maps' blend (on the band ramp a = 0.1 per
                        # cell: each side's error on the mid-line p99 0.21-0.27 cells; 0.5 / 1.5 / 2: 0.25-0.33 /
                        # 0.20-0.24 / 0.21-0.22)
    face_rad=1,         # outline face: block nodes within this many lattice steps (index box) of the run node
    run_smooth=4,       # the mid-line: each run averaged over +-this many nodes (its lattice steps smoothed out)
    min_sep=0.6,        # gap points closer than this (cells) are thinned (the deeper inside its own segment kept)
    cut_win=1.5,        # a segment's points are made up to this many cells beyond its cuts (they fill the cracks)
    d_loop=0.35,        # gap points nearer than this (cells) to the corridor loop are dropped
    newton=30,          # Newton steps per point
    price=1.0 / 3.0,    # recombination price of a leftover triangle
    price_outline=1.0,  # ... plus this for one touching an outline run: the core's triangle stays off the outline row
    size_iters=30,      # m02's fill smoothing: "size" sweeps, then "angle" sweeps, loop nodes fixed
    angle_iters=3,
)


# ----------------------------------------------------------------------------------------------------------- chart

class Chart:
    """Stereographic plane at the unit vector c (unit-sphere units)."""

    def __init__(self, c: np.ndarray):
        self.c = c / np.linalg.norm(c)
        e1, e2 = tangent_basis(self.c[None])
        self.e1, self.e2 = e1[0], e2[0]

    def xy(self, P: np.ndarray) -> np.ndarray:
        d = 1.0 + P @ self.c
        return np.column_stack([2.0 * (P @ self.e1) / d, 2.0 * (P @ self.e2) / d])

    def xyz(self, xy: np.ndarray) -> np.ndarray:
        r2 = (xy ** 2).sum(axis=1)
        return (4.0 * xy[:, :1] * self.e1 + 4.0 * xy[:, 1:] * self.e2 + (4.0 - r2)[:, None] * self.c) / (4.0 + r2)[:, None]


def project(x: np.ndarray, V: np.ndarray, tree: cKDTree | None = None, k: int = 6):
    """Distance of points x (N,2) to the open polyline V (K,2) and the arc length of the foot point."""
    K = len(V)
    seg = np.diff(V, axis=0)
    sl = np.einsum("ij,ij->i", seg, seg)
    arc = np.r_[0.0, np.cumsum(np.sqrt(sl))]
    tree = cKDTree(V) if tree is None else tree
    _, vi = tree.query(x, k=min(k, K))
    vi = np.atleast_2d(vi.T).T if vi.ndim == 1 else vi
    cand = np.clip(np.concatenate([vi - 1, vi], axis=1), 0, K - 2)              # (N, 2k) segments
    a = V[cand]
    f = np.clip(np.einsum("nkd,nkd->nk", x[:, None] - a, seg[cand]) / np.maximum(sl[cand], 1e-300), 0.0, 1.0)
    foot = a + f[..., None] * seg[cand]
    d = np.linalg.norm(x[:, None] - foot, axis=2)
    j = np.argmin(d, axis=1)
    r = np.arange(len(x))
    c = cand[r, j]
    return d[r, j], arc[c] + f[r, j] * np.sqrt(sl[c])


# ---------------------------------------------------------------------------------------------- extrapolated maps

def face_maps(xy: np.ndarray, ij: np.ndarray, run: np.ndarray, rad: int = 1) -> np.ndarray:
    """(K,2,2) Jacobians G_k of the outline faces: u - l_k = G_k (x - x_k) by least squares over the block nodes
    within rad lattice steps (index box) of run node k."""
    ij = np.asarray(ij, dtype=np.int64)
    lo = ij.min(axis=0) - rad
    W = int(ij[:, 1].max() - lo[1]) + 2 * rad + 1
    key = (ij[:, 0] - lo[0]) * W + (ij[:, 1] - lo[1])
    order = np.argsort(key)
    sk = key[order]
    XX = np.zeros((len(run), 2, 2))
    UX = np.zeros((len(run), 2, 2))
    for di in range(-rad, rad + 1):
        for dj in range(-rad, rad + 1):
            if di == 0 and dj == 0:
                continue
            q = key[run] + di * W + dj
            p = np.clip(np.searchsorted(sk, q), 0, len(sk) - 1)
            ok = sk[p] == q
            n = order[p]
            X = np.where(ok[:, None], xy[n] - xy[run], 0.0)
            U = np.where(ok[:, None], ij[n] - ij[run], 0.0).astype(float)
            XX += np.einsum("ki,kj->kij", X, X)
            UX += np.einsum("ki,kj->kij", U, X)
    return UX @ np.linalg.inv(XX)


class Side:
    """One block's outline run with its extrapolated map (module docstring)."""

    def __init__(self, chart: Chart, points: np.ndarray, ij: np.ndarray, run: np.ndarray, o: dict,
                 M: np.ndarray | None = None):
        xy = chart.xy(points)
        self.xk = xy[run]
        self.Gk = face_maps(xy, ij, run, o["face_rad"])
        self.lk = np.asarray(ij, dtype=float)[run]
        self.cell = np.sqrt(1.0 / np.abs(np.linalg.det(self.Gk)))
        self.tree = cKDTree(self.xk)
        self.M = np.eye(2) if M is None else M
        self.sig = None

    def turn(self, M: np.ndarray) -> None:
        """Express the indices in another frame: u -> M u."""
        self.lk = self.lk @ M.T
        self.Gk = M[None] @ self.Gk
        self.M = M @ self.M

    def eval(self, x: np.ndarray):
        kn = min(8, len(self.xk))
        d, j = self.tree.query(x, k=kn)
        w = np.exp(-(d / self.sig) ** 2) + 1e-300
        w /= w.sum(axis=1, keepdims=True)
        dx = x[:, None, :] - self.xk[j]
        uk = self.lk[j] + np.einsum("nkab,nkb->nka", self.Gk[j], dx)
        return np.einsum("nk,nka->na", w, uk), np.einsum("nk,nkab->nab", w, self.Gk[j])


_ROT = [np.array([[1, 0], [0, 1]]), np.array([[0, -1], [1, 0]]), np.array([[-1, 0], [0, -1]]),
        np.array([[0, 1], [-1, 0]])]


class Corridor:
    """Two sides, the blended map and its mid-line (module docstring). runs: block node ids in loop order (A from cap
    C0 to C1, B from C1 to C0, the corridor on the left of both)."""

    def __init__(self, gA, runA, gB, runB, o: dict):
        o = dict(DEFAULTS, **(o or {}))
        self.o = o
        P = np.vstack([gA.points[runA], gB.points[runB]])
        self.chart = Chart(P.mean(axis=0))
        self.A = Side(self.chart, gA.points, gA.ij, np.asarray(runA), o)
        self.B = Side(self.chart, gB.points, gB.ij, np.asarray(runB), o)
        GA, GB = self.A.Gk.mean(axis=0), self.B.Gk.mean(axis=0)
        k = int(np.argmin([np.linalg.norm(GA - R @ GB) for R in _ROT]))
        self.B.turn(_ROT[k].astype(float))
        self.hc = float(np.median(np.r_[self.A.cell, self.B.cell]))
        self.A.sig = self.B.sig = o["sigma"] * self.hc
        self.xa, self.xb = self.A.xk, self.B.xk
        self.ta, self.tb = cKDTree(self.xa), cKDTree(self.xb)
        # mid-line: A's run smoothed (its lattice steps averaged out) and resampled at hc / 4, each sample halfway to
        # its foot on B's smoothed run; delta = u_A - M u_B on it, averaged over +-1 cell along it
        ra, rb = _smooth_run(self.xa, o["run_smooth"]), _smooth_run(self.xb, o["run_smooth"])
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ra, axis=0), axis=1))]
        sa = np.linspace(0.0, arc[-1], max(2, int(np.ceil(4.0 * arc[-1] / self.hc)) + 1))
        xs = np.column_stack([np.interp(sa, arc, ra[:, 0]), np.interp(sa, arc, ra[:, 1])])
        _, fb = project(xs, rb)
        arcb = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(rb, axis=0), axis=1))]
        xf = np.column_stack([np.interp(fb, arcb, rb[:, 0]), np.interp(fb, arcb, rb[:, 1])])
        mid = 0.5 * (xs + xf)
        keep = np.r_[True, np.linalg.norm(np.diff(mid, axis=0), axis=1) > 1e-9 * self.hc]
        self.mid = mid[keep]
        self.tm = cKDTree(self.mid)
        self.s_mid = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.mid, axis=0), axis=1))] / self.hc
        uA, _ = self.A.eval(self.mid)
        uB, _ = self.B.eval(self.mid)
        self.delta = _moving_mean(uA - uB, self.s_mid, 1.0)

    # ---- coordinates
    def s(self, x: np.ndarray) -> np.ndarray:
        return project(x, self.mid, self.tm)[1] / self.hc

    def t(self, x: np.ndarray) -> np.ndarray:
        da = project(x, self.xa, self.ta)[0]
        db = project(x, self.xb, self.tb)[0]
        return da / np.maximum(da + db, 1e-300)

    def umap(self, x: np.ndarray, T: np.ndarray):
        t = self.t(x)
        uA, JA = self.A.eval(x)
        uB, JB = self.B.eval(x)
        u = (1.0 - t)[:, None] * uA + t[:, None] * (uB + T)
        J = (1.0 - t)[:, None, None] * JA + t[:, None, None] * JB
        return u, J, t

    def burgers(self) -> np.ndarray:
        """b = delta(end) - delta(start) along the mid-line (index units, A's frame)."""
        return self.delta[-1] - self.delta[0]

    def burgers_split(self) -> tuple[float, float]:
        """(b_par, b_perp): b's components on A's lattice axis nearest to the corridor (A's index change from the
        mid-line's start to its end) and on the other, as stitch_mix.burgers_vector counts them on the runs. From the
        maps, b is the closure failure also where the lattices curve (a conformal ramp's fan turns both lattices
        alike: b = 0, while the runs' along counts differ by the ramp across the gap)."""
        b = self.burgers()
        uA, _ = self.A.eval(self.mid[[0, -1]])
        ax = int(np.argmax(np.abs(uA[1] - uA[0])))
        return float(b[ax]), float(b[1 - ax])

    def mode(self) -> str:
        """The schedule of the corridor rule (stitch_mix.quantise, register B47) on the maps' b and the median width."""
        from meshom.methods.m14.stitch_mix import quantise

        bpar, bperp = self.burgers_split()
        w = float(np.median(project(self.mid, self.xa, self.ta)[0] + project(self.mid, self.xb, self.tb)[0])) / self.hc
        return "none" if quantise((round(bpar), round(bperp)), w) in ("none", "shear") else "even"


def _smooth_run(x: np.ndarray, win: int) -> np.ndarray:
    """Moving average of a polyline's nodes over +-win nodes (shrinking at the ends, the end nodes kept)."""
    n = len(x)
    cs = np.r_[np.zeros((1, 2)), np.cumsum(x, axis=0)]
    i = np.arange(n)
    w = np.minimum(np.minimum(i, n - 1 - i), win)
    return (cs[i + w + 1] - cs[i - w]) / (2 * w + 1)[:, None]


def _moving_mean(y: np.ndarray, s: np.ndarray, half: float) -> np.ndarray:
    """Mean of y over |s' - s| <= half (s increasing)."""
    cs = np.r_[np.zeros((1,) + y.shape[1:]), np.cumsum(y, axis=0)]
    lo = np.searchsorted(s, s - half, side="left")
    hi = np.searchsorted(s, s + half, side="right")
    return (cs[hi] - cs[lo]) / (hi - lo)[:, None]


# --------------------------------------------------------------------------------------------------- the schedule

def schedule(s: np.ndarray, delta: np.ndarray, mode: str = "even", eta: float = 0.15, margin: float = 0.0):
    """Events of T along s: (positions (E,), T before the first event (2,), steps (E,2) unit vectors). "even": the
    re-spaced events within margin cells of an end (B58: off the cap, i.e. the junction or coast beyond it) move to
    s_0 + margin (s_end - margin), those that would stack there 1 cell apart inward; the others stay.
    """
    n = len(s)
    if mode == "none" or n < 2:
        return np.zeros(0), np.round(delta[n // 2]), np.zeros((0, 2))
    T0 = np.round(delta[0])
    ev = []
    for c in (0, 1):
        Tc = T0[c]
        r = delta[:, c]
        for k in range(1, n):
            while r[k] - Tc > 0.5 + (eta if mode == "even" else 0.0):
                ev.append((_crossing(s, r, k, Tc + 0.5), c, 1.0))
                Tc += 1.0
            while r[k] - Tc < -0.5 - (eta if mode == "even" else 0.0):
                ev.append((_crossing(s, r, k, Tc - 0.5), c, -1.0))
                Tc -= 1.0
    ev.sort()
    pos = np.array([e[0] for e in ev])
    steps = np.zeros((len(ev), 2))
    for i, (_, c, sg) in enumerate(ev):
        steps[i, c] = sg
    if mode == "even" and len(ev) > 1:
        acc = np.r_[0.0, np.cumsum(np.abs(np.diff(delta, axis=0)).sum(axis=1))]
        acc = acc + 1e-9 * np.arange(n)                                 # strictly increasing for the inverse
        a = np.interp(pos, s, acc)
        k = np.arange(len(ev))
        a2 = k + np.mean(a - k)
        pos = np.clip(np.interp(a2, acc, s), s[0], s[-1])
    m0, m1 = (margin, margin) if np.isscalar(margin) else margin
    lo, hi = s[0] + m0, s[-1] - m1
    if mode == "even" and len(ev) and hi > lo and (pos[0] < lo or pos[-1] > hi):
        # the events within margin of an end move to it; where that stacks them, they keep 1 cell apart
        d = min(1.0, (hi - lo) / len(pos))
        k = np.arange(len(pos))
        pos = np.minimum(np.maximum(pos, lo + d * k), hi - d * (len(pos) - 1 - k))
    return pos, T0, steps


def _crossing(s, r, k, level):
    """Position of the last crossing of level by r before sample k (linear between samples)."""
    j = k
    while j > 0 and (r[j - 1] - level) * (r[k] - level) > 0:
        j -= 1
    if j == 0:
        return float(s[0])
    f = (level - r[j - 1]) / (r[j] - r[j - 1])
    return float(s[j - 1] + f * (s[j] - s[j - 1]))


def segment_T(pos: np.ndarray, T0: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """(E+1,2) T of each constant segment."""
    return T0[None] + np.r_[np.zeros((1, 2)), np.cumsum(steps, axis=0)]


# ------------------------------------------------------------------------------------------------------- points

def gap_points(cor: Corridor, loop_xy: np.ndarray, sched=None):
    """Integer points of the blended map inside the corridor loop (plane coordinates): (xy (N,2), u labels (N,2),
    v labels (N,2), segment (N,), t (N,)) and the schedule (pos, T0, steps)."""
    o = cor.o
    hc = cor.hc
    mode = cor.mode() if o["schedule"] == "rule" else o["schedule"]
    pos, T0, steps = schedule(cor.s_mid, cor.delta, mode, o["eta"], o["end_margin"]) if sched is None else sched
    Ts = segment_T(pos, T0, steps)
    bnd = np.r_[-np.inf, pos, np.inf]
    sa = cor.s(cor.xa)
    sb = cor.s(cor.xb)
    uB_run = cor.B.lk
    lab, seg, x0 = [], [], []
    for k in range(len(Ts)):
        lo, hi = max(bnd[k], cor.s_mid[0]), min(bnd[k + 1], cor.s_mid[-1])
        ia = (sa >= lo - 1.0 - o["cut_win"]) & (sa <= hi + 1.0 + o["cut_win"])
        ib = (sb >= lo - 1.0 - o["cut_win"]) & (sb <= hi + 1.0 + o["cut_win"])
        if not ia.any() or not ib.any():
            continue
        ends = np.vstack([cor.A.lk[ia], uB_run[ib] + Ts[k]])
        i0, j0 = np.floor(ends.min(axis=0)).astype(int) - 1
        i1, j1 = np.ceil(ends.max(axis=0)).astype(int) + 1
        I, J = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        L = np.column_stack([I.ravel(), J.ravel()]).astype(float)
        sm = 0.5 * (lo + hi)
        xc = np.array([[np.interp(sm, cor.s_mid, cor.mid[:, 0]), np.interp(sm, cor.s_mid, cor.mid[:, 1])]])
        uc, Jc, _ = cor.umap(xc, Ts[k])
        x0.append(xc + np.linalg.solve(Jc[0], (L - uc).T).T)
        lab.append(L)
        seg.append(np.full(len(L), k))
    lab, seg, x = np.vstack(lab), np.concatenate(seg), np.vstack(x0)
    T = Ts[seg]
    for _ in range(o["newton"]):
        u, J, _ = cor.umap(x, T)
        dx = np.linalg.solve(J, (u - lab)[..., None])[..., 0]
        nx = np.linalg.norm(dx, axis=1, keepdims=True)
        x = x - dx * np.minimum(1.0, 0.5 * hc / np.maximum(nx, 1e-300))
    u, J, t = cor.umap(x, T)
    ok = np.abs(u - lab).max(axis=1) < 1e-3
    ok &= MplPath(np.vstack([loop_xy, loop_xy[:1]])).contains_points(x)
    s = cor.s(x)
    depth = np.minimum(s - bnd[seg], bnd[seg + 1] - s)                 # > 0 inside the own segment
    ok &= depth > -o["cut_win"]
    dl = project(x, np.vstack([loop_xy, loop_xy[:1]]))[0]
    ok &= dl > o["d_loop"] * hc
    x, lab, seg, t, T, depth = x[ok], lab[ok], seg[ok], t[ok], T[ok], depth[ok]
    keep = _thin(x, depth, o["min_sep"] * hc)
    x, lab, seg, t, T = x[keep], lab[keep], seg[keep], t[keep], T[keep]
    v = lab - (t > 0.5)[:, None] * T
    return (x, lab, v, seg, t), (pos, T0, steps)


def _thin(x: np.ndarray, prio: np.ndarray, r: float) -> np.ndarray:
    """Greedy thinning by decreasing priority: a point is kept if no kept point lies within r."""
    nb = [[] for _ in range(len(x))]
    for i, j in cKDTree(x).query_pairs(r):
        nb[i].append(j)
        nb[j].append(i)
    keep = np.zeros(len(x), dtype=bool)
    for i in np.argsort(-prio, kind="stable"):
        if not any(keep[j] for j in nb[i]):
            keep[i] = True
    return keep


def run_labels(cor: Corridor, sched):
    """u and v labels of the two runs' nodes: A's own indices; B's in A's frame plus T at their position."""
    pos, T0, steps = sched
    Ts = segment_T(pos, T0, steps)
    kb = np.searchsorted(pos, cor.s(cor.xb), side="right")
    return (cor.A.lk, cor.A.lk), (cor.B.lk + Ts[kb], cor.B.lk)


# ------------------------------------------------------------------------------------------------ recombination

def _is_square(L: np.ndarray) -> np.ndarray:
    """(M,4,2) labels of CCW quads -> (M,) bool: a unit CCW square of the lattice."""
    d = np.roll(L, -1, axis=1) - L
    unit = (np.abs(d).sum(axis=2) == 1.0) & (np.abs(d).max(axis=2) == 1.0)
    rot = (np.roll(d, -1, axis=1)[..., 0] == -d[..., 1]) & (np.roll(d, -1, axis=1)[..., 1] == d[..., 0])
    return (unit & rot).all(axis=1)


def pair_labels(P: np.ndarray, tris: np.ndarray, lu: np.ndarray, lv: np.ndarray):
    """Triangle pairs whose four labels form a unit square (in u or in v), by increasing quad cost: (quads, the
    triangles left)."""
    from meshom.delaunay import adjacency
    from meshom.recombine import _greedy_take, _merge, _quad_cost

    t_n = len(tris)
    nbr = adjacency(tris)
    t_idx, s_idx = np.nonzero(nbr > np.arange(t_n)[:, None])
    ea, eb = t_idx.astype(np.int64), nbr[t_idx, s_idx].astype(np.int64)
    q = _merge(tris[ea], tris[eb])
    ok = _is_square(lu[q]) | _is_square(lv[q])
    ea, eb, q = ea[ok], eb[ok], q[ok]
    cost = _quad_cost(P, q)
    fin = np.isfinite(cost)
    ea, eb, q, cost = ea[fin], eb[fin], q[fin], cost[fin]
    mate = np.full(t_n, -1, dtype=np.int64)
    _greedy_take(np.argsort(cost, kind="stable"), ea, eb, mate)
    lo = np.flatnonzero(mate > np.arange(t_n))
    quads = _merge(tris[lo], tris[mate[lo]]) if len(lo) else np.zeros((0, 4), dtype=np.int64)
    return quads.astype(np.int64), tris[mate < 0]


def lift_triangles(m: Mesh, side: np.ndarray, passes: int = 3) -> Mesh:
    """Each triangle with a node on a side (side: (N,) bool, the outline runs) that shares an edge with a quad: the
    pentagon of the two is re-split into the quad + triangle (an ear of the pentagon) whose triangle is off the sides,
    whose interior nodes' sum |4 - valence| does not grow, with the squarest quad (B58: a core never on the outline
    row; the dislocation count is kept)."""
    from meshom.mesh import build_topology
    from meshom.recombine import _quad_cost

    P = m.points
    tris, quads = m.tris.copy(), m.quads.copy()
    inner = m.node_flag == 0
    for _ in range(passes):
        deg = np.bincount(build_topology(Mesh(P, tris, quads, m.node_flag)).edges.ravel(), minlength=len(P))
        edge_q = {}
        for qi, q in enumerate(quads):
            for k in range(4):
                edge_q[(int(q[k]), int(q[(k + 1) % 4]))] = qi
        moved = 0
        for ti in np.flatnonzero(side[tris].any(axis=1)):
            best = None
            for k in range(3):
                u, v, w = (int(x) for x in np.roll(tris[ti], -k))
                qi = edge_q.get((v, u))
                if qi is None:
                    continue
                q = [int(x) for x in quads[qi]]
                r = q.index(v)
                pent = [u, q[(r + 2) % 4], q[(r + 3) % 4], v, w]
                for e in range(4):                                      # ear e (index 4 = the current triangle)
                    tri = [pent[e - 1], pent[e], pent[(e + 1) % 5]]
                    quad = [pent[(e + 1) % 5], pent[(e + 2) % 5], pent[(e + 3) % 5], pent[e - 1]]
                    if side[tri].any() or np.dot(P[tri[0]], np.cross(P[tri[1]], P[tri[2]])) <= 0:
                        continue
                    c = float(_quad_cost(P, np.array([quad]))[0])
                    if not np.isfinite(c):
                        continue
                    # the diagonal u-v goes, tri[0]-tri[2] comes (ears 1 and 2 share a node with it: net per node)
                    nd, inv_ = np.unique([u, v, tri[0], tri[2]], return_inverse=True)
                    d1 = deg[nd] + np.bincount(inv_, weights=[-1, -1, 1, 1]).astype(np.int64)
                    if (np.abs(4 - d1) * inner[nd]).sum() > (np.abs(4 - deg[nd]) * inner[nd]).sum():
                        continue
                    if best is None or c < best[0]:
                        best = (c, qi, tri, quad, q, (u, v))
            if best is None:
                continue
            _, qi, tri, quad, q, (u, v) = best
            np.add.at(deg, [u, v], -1)
            np.add.at(deg, [tri[0], tri[2]], 1)
            for k in range(4):
                edge_q.pop((q[k], q[(k + 1) % 4]), None)
                edge_q[(quad[k], quad[(k + 1) % 4])] = qi
            tris[ti], quads[qi] = tri, quad
            moved += 1
        if not moved:
            break
    return Mesh(P, tris, quads, m.node_flag)


# ---------------------------------------------------------------------------------------------------------- fill

def cdt_local(pts: np.ndarray, n_loop: int) -> np.ndarray:
    """Constrained Delaunay of a closed loop (the first n_loop points, region on its left) and interior points; one
    antipodal helper point closes the hull (module docstring)."""
    from meshom.delaunay import recover_edges, remove_land, sphere_delaunay
    from meshom.domain import loop_edges

    m = pts[:n_loop].mean(axis=0)
    m /= np.linalg.norm(m)
    allp = np.vstack([pts, -m[None]])
    tris = sphere_delaunay(allp)
    e = loop_edges([pts[:n_loop]])
    tris, e, _ = recover_edges(allp, tris, e, return_edges=True)      # e split at touches (M40)
    tris = remove_land(allp, tris, e)
    return tris[(tris < len(pts)).all(axis=1)]


def corridor_schedule(cor: Corridor):
    """The corridor's schedule (pos, T0, steps) by its rule (gap_points' default)."""
    o = cor.o
    mode = cor.mode() if o["schedule"] == "rule" else o["schedule"]
    return schedule(cor.s_mid, cor.delta, mode, o["eta"], o["end_margin"])


def lattice_cap(cor: Corridor, sched, end: int, variant: int = 0):
    """The lattice cap of a corridor end (B57): the lattice path of the blended map (T of that end) from A's run end
    to B's (end 0: A's first node to B's last, cap C0; end 1: A's last to B's first, C1): half its steps across the
    corridor (along the axis nearer to the A-B chord), then the steps along it on that middle row, then the other
    across steps (so it leaves and meets both outlines square); variant 1: the across and along steps interleaved
    evenly (Bresenham) after one across step and before the last; its interior nodes at those integer points (Newton
    from the chord). Returns (xyz (n-1,3), u labels, v labels) or None (fewer than 2 steps, or no convergence)."""
    Ts = segment_T(*sched)
    T = Ts[0] if end == 0 else Ts[-1]
    ia, ib = (0, -1) if end == 0 else (-1, 0)
    ua, ub = cor.A.lk[ia], cor.B.lk[ib] + T
    D = np.round(ub - ua).astype(np.int64)
    n = int(np.abs(D).sum())
    if n < 2 or np.abs(ub - ua - D).max() > 1e-6:
        return None
    ac = int(np.argmax(np.abs(D)))                                      # the across axis
    e = np.zeros((2, 2), dtype=np.int64)
    e[0, ac], e[1, 1 - ac] = np.sign(D[ac]), np.sign(D[1 - ac])
    na, nl = abs(int(D[ac])), abs(int(D[1 - ac]))
    if variant == 0 or na < 2:
        steps = [e[0]] * (na // 2) + [e[1]] * nl + [e[0]] * (na - na // 2)
    else:
        mid, k = [], np.zeros(2)
        for _ in range(na - 2 + nl):                                    # the lagging kind steps (Bresenham)
            c = 0 if nl == 0 or (k[0] + 0.5) / (na - 2) <= (k[1] + 0.5) / nl else 1
            if na == 2:
                c = 1
            k[c] += 1
            mid.append(e[c])
        steps = [e[0]] + mid + [e[0]]
    lab = (ua + np.cumsum(steps, axis=0))[:-1].astype(float)
    x = cor.xa[ia] + np.linspace(0.0, 1.0, n + 1)[1:-1, None] * (cor.xb[ib] - cor.xa[ia])
    TT = np.repeat(T[None], len(x), axis=0)
    for _ in range(cor.o["newton"]):
        u, J, _ = cor.umap(x, TT)
        dx = np.linalg.solve(J, (u - lab)[..., None])[..., 0]
        nx = np.linalg.norm(dx, axis=1, keepdims=True)
        x = x - dx * np.minimum(1.0, 0.5 * cor.hc / np.maximum(nx, 1e-300))
    u, _, t = cor.umap(x, TT)
    if np.abs(u - lab).max() > 1e-3:
        return None
    return cor.chart.xyz(x), lab, lab - (t > 0.5)[:, None] * T


def fill(loop: np.ndarray, a_loop: np.ndarray, b_loop: np.ndarray, gA, runA, gB, runB, h, o: dict | None = None,
         cor: Corridor | None = None, sched=None, known: dict | None = None):
    """The continuation fill of one corridor: Mesh (the loop's nodes first, flag 1) and info. loop (N,3) closed, the
    region on its left; a_loop, b_loop: positions of the runs' nodes in it; gA, gB: blocks (points, ij); runA, runB:
    their run nodes (block ids, loop order). cor, sched: the corridor and its schedule when already made (the lattice
    caps); known: {loop position: (u label, v label)} of other loop nodes on lattice points (the lattice caps)."""
    from meshom.recombine import recombine
    from meshom.smooth import smooth

    cor = Corridor(gA, runA, gB, runB, o) if cor is None else cor
    o = cor.o
    lxy = cor.chart.xy(loop)
    (x, lu, lv, seg, t), sched = gap_points(cor, lxy, sched)
    (au, av), (bu, bv) = run_labels(cor, sched)
    nl = len(loop)
    LU = np.full((nl + len(x), 2), np.nan)
    LV = np.full((nl + len(x), 2), np.nan)
    LU[a_loop], LV[a_loop] = au, av
    LU[b_loop], LV[b_loop] = bu, bv
    for k, (ku, kv) in (known or {}).items():
        LU[k], LV[k] = ku, kv
    LU[nl:], LV[nl:] = lu, lv
    pts = np.vstack([loop, cor.chart.xyz(x)]) if len(x) else loop
    tris = cdt_local(pts, nl)
    used = np.zeros(len(pts), dtype=bool)
    used[tris.ravel()] = True
    used[:nl] = True
    new = np.cumsum(used) - 1
    pts, tris, LU, LV = pts[used], new[tris], LU[used], LV[used]
    quads, rest = pair_labels(pts, tris, LU, LV)
    flag = np.zeros(len(pts), dtype=np.int8)
    flag[:nl] = 1
    run = np.zeros(len(pts), dtype=bool)
    run[np.r_[a_loop, b_loop]] = True
    price = o["price"] + o["price_outline"] * run[rest].any(axis=1)
    fm = recombine(Mesh(pts, rest, quads, flag), price, "greedy")
    fm = lift_triangles(fm, run)
    fixed = fm.node_flag == 1
    if o["size_iters"]:
        fm = smooth(fm, h, fixed, iters=o["size_iters"], kind="size")
    if o["angle_iters"]:
        fm = smooth(fm, h, fixed, iters=o["angle_iters"], kind="angle")
    pos, T0, steps = sched
    info = {"events": int(len(pos)), "label_quads": int(len(quads)), "points": int(len(x)),
            "b": cor.burgers().tolist(), "frame_turn": int(round(np.degrees(np.arctan2(cor.B.M[1, 0], cor.B.M[0, 0])))),
            "cell_m": cor.hc, "cap_labels": int(len(known or {}))}
    return fm, info
