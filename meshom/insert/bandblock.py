"""The band block of a coastal strip or a current corridor (decision B181): rows offset from a spine, wrapping round
its two ends (a half O-grid), blended into a host edge chain at the outside.

Coordinates (tau, d): d = the distance from the spine (0 on it, D at the host chain); tau = the arc length of the
foot point on the spine (0 at its start P, Ls at its end Q), continued round the ends by the angle: tau = -theta
before P and Ls + theta after Q, theta from 0 (the spine's normal) to pi/2 (the spine's line continued behind P or
Q: the bank, the coast or the axis beyond P and Q). The map

    F(tau, d) = S_d(tau) + d n_d(tau)                  on the spine (0 <= tau <= Ls),
              = S_d(P) + d (cos theta n_d(P) - sin theta t_d(P))       round P (and alike round Q),

with S_d the spine smoothed along its arc by a Gaussian of OFFSET_SIGMA x d (its ends held, so the offset rows stay
apart where the coast is rough), n_d its normal toward the band; the row ends (theta = pi/2) are moved onto the bank
(the move fades in with (theta / (pi/2))^2), and the outer correction U(tau) - F(tau, D) (U = the host chain at the
tau of its nodes' foot points) is added with the smoothstep of d / D. Rows: d_k at equal steps of the integral of
dd / h(d), h = the band target (block.size_target_spine's profile: h_fine over the band, then growing at G_LAYOUT,
capped at the host size); counts N_k = row length / h(d_k), at most max(1, BIRTH_FRAC N) births per row; nodes at
equal arc along the row; neighbouring rows zipped by tau (block.zip_rows: quads where the counts match, a triangle
where a row gains a node). In the middle the cells are the offset rows times the normals; round P and Q they fan.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

from meshom.insert.block import BIRTH_FRAC, G_LAYOUT, zip_rows
from meshom.insert.geom import Polyline

OFFSET_SIGMA = 1.0           # B181: the spine under row d is smoothed with a Gaussian of this x d
STAIR_ROWS = 3.0             # B193: on a fixed host chain the rows within this many outer cells follow its steps
FAN_PTS = 96                 # samples of a row's quarter turn round P or Q
FAN_MIN = 0.35               # B181: a quarter turn spans at least this angle (rad) and at most pi - it
EQUALIZE = 0.35              # B181: per row, a column moves this share of the way to equal spacing along the row ...
EQ_MAX = 0.25                # B181: ... at most this many cells


@dataclass
class BandBlock:
    xy: np.ndarray               # (n,2) chart
    quads: np.ndarray
    tris: np.ndarray
    row: np.ndarray              # (n,) row of each node (0 = the spine, K = the host chain)
    off: np.ndarray              # row offsets
    N: np.ndarray                # cells per row
    d: np.ndarray                # (K+1,) row distances from the spine
    h: np.ndarray                # (K+1,) row size targets
    tau: list                    # per row: tau of its nodes
    info: dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return len(self.N) - 1


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class BandMap:
    """F(tau, d) of the module docstring for one side of the spine (side +1: the spine's left, -1: its right)."""

    def __init__(self, spine: np.ndarray, side: float, D: float, bank_p: Polyline, bank_q: Polyline,
                 step: float):
        pl = Polyline(spine)
        m = max(int(np.ceil(pl.L / step)), 8)
        self.s = np.linspace(0.0, pl.L, m + 1)
        self.sp = pl.at(self.s)
        self.Ls = pl.L
        self.side, self.D, self.step = float(side), float(D), pl.L / m
        self.bank_p, self.bank_q = bank_p, bank_q
        self._cache = {}

    def frame(self, d: float):
        key = round(float(d), 3)
        if key in self._cache:
            return self._cache[key]
        sig = OFFSET_SIGMA * d
        q = self.sp
        if sig > 0 and len(q) > 4:
            g = gaussian_filter1d(q, sig / self.step, axis=0, mode="nearest")
            w = np.clip(self.s / (2 * sig), 0, 1) * np.clip((self.Ls - self.s) / (2 * sig), 0, 1)
            w = w * w * (3 - 2 * w)
            q = q + w[:, None] * (g - q)
        t = _unit(np.gradient(q, axis=0))
        n = self.side * np.c_[-t[:, 1], t[:, 0]]
        fr = (q, t, n)
        self._cache[key] = fr
        return fr

    def raw(self, tau: np.ndarray, d: float) -> np.ndarray:
        q, t, n = self.frame(d)
        tau = np.asarray(tau, dtype=float)
        out = np.empty((len(tau), 2))
        mid = (tau >= 0) & (tau <= self.Ls)
        if mid.any():
            sm = tau[mid]
            out[mid] = (np.c_[np.interp(sm, self.s, q[:, 0]), np.interp(sm, self.s, q[:, 1])]
                        + d * _unit(np.c_[np.interp(sm, self.s, n[:, 0]), np.interp(sm, self.s, n[:, 1])]))
        for sel, th, P, tt, nn, sg, end in ((tau < 0, -tau, q[0], t[0], n[0], -1.0, "p"),
                                            (tau > self.Ls, tau - self.Ls, q[-1], t[-1], n[-1], 1.0, "q")):
            if sel.any():
                f = np.clip(th[sel], 0, 0.5 * np.pi) / (0.5 * np.pi)
                th_e, r_e = self.fan_end(d, end)
                a = f * th_e
                r = d + f * f * (r_e - d)
                out[sel] = P + r[:, None] * (np.cos(a)[:, None] * nn + sg * np.sin(a)[:, None] * tt)
        return out

    def fan_end(self, d: float, end: str) -> tuple:
        """The quarter turn round P (Q) of row d ends on the bank d / D of its length from P (Q): (its angle from the normal,
        its distance from P (Q)); without a bank: (pi/2, d)."""
        bank = self.bank_p if end == "p" else self.bank_q
        q, t, n = self.frame(d)
        if bank is None or d <= 0:
            return 0.5 * np.pi, d
        P, tt, nn, sg = (q[0], t[0], n[0], -1.0) if end == "p" else (q[-1], t[-1], n[-1], 1.0)
        a = min(d / self.D, 1.0) * bank.L                    # row d ends d / D of the way along the bank
        B = bank.at(bank.L - a if end == "p" else a)[0]
        v = B - P
        th = float(np.arctan2(sg * (v @ tt), v @ nn))
        return float(np.clip(th, FAN_MIN, np.pi - FAN_MIN)), float(np.linalg.norm(v))

    def F(self, tau: np.ndarray, d: float) -> np.ndarray:
        """The map of the module docstring (the row ends on the bank by fan_end)."""
        return self.raw(np.asarray(tau, dtype=float), d)

    def tau_of(self, x: np.ndarray) -> np.ndarray:
        """tau of points near row D (the host chain's nodes): the foot on the spine smoothed for D, or the angle
        round P / Q; made strictly increasing along the chain, its ends at -pi/2 and Ls + pi/2."""
        q, t, n = self.frame(self.D)
        pl = Polyline(q)
        arc_q = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        s = np.interp(pl.project(x), arc_q, self.s)          # the foot's tau: the spine's own arc, not S_D's
        tau = s.astype(float).copy()
        for P, tt, nn, sg, base in ((q[0], t[0], n[0], -1.0, 0.0), (q[-1], t[-1], n[-1], 1.0, self.Ls)):
            sel = sg * ((x - P) @ tt) > 0                        # behind the normal at P (Q): round the end
            v = x[sel] - P
            th = np.arctan2(sg * (v @ tt), v @ nn)
            th_e = self.fan_end(self.D, "p" if sg < 0 else "q")[0]
            tau[sel] = base + sg * 0.5 * np.pi * np.clip(th / th_e, 0, 1)
        tau[0], tau[-1] = -0.5 * np.pi, self.Ls + 0.5 * np.pi
        tau = _isotonic(tau)                                     # a zigzag chain may step back a little
        for k in range(1, len(tau)):                             # strictly increasing (tau mixes rad and m:
            tau[k] = max(tau[k], tau[k - 1] + 1e-9)              # an absolute epsilon)
        return tau

    def grid(self) -> np.ndarray:
        fan = np.linspace(0.0, 0.5 * np.pi, FAN_PTS + 1)
        return np.r_[-fan[::-1], self.s[1:-1], self.Ls + fan]


def _isotonic(y: np.ndarray) -> np.ndarray:
    """The non-decreasing least-squares fit of y (pool adjacent violators), each pooled run spread linearly between
    the midpoints to its neighbouring runs (strictly increasing, no plateau for the interpolation of the outer
    correction to jump across)."""
    vals, wts, cnt = [], [], []
    for v in y:
        vals.append(float(v))
        wts.append(1.0)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            c = cnt[-2] + cnt[-1]
            vals[-2:], wts[-2:], cnt[-2:] = [v2], [w], [c]
    out = np.repeat(vals, cnt).astype(float)
    pos = 0
    for r, (v, c) in enumerate(zip(vals, cnt)):
        if c > 1:
            lo = 0.5 * (vals[r - 1] + v) if r > 0 else v
            hi = 0.5 * (v + vals[r + 1]) if r + 1 < len(vals) else v
            out[pos:pos + c] = np.linspace(lo, hi, c + 2)[1:-1] if r > 0 and r + 1 < len(vals) else \
                np.linspace(lo, hi, c)
        pos += c
    return out


def band_levels(D: float, h_f: float, band: float, h_host: float, g: float = G_LAYOUT) -> tuple:
    """(d_k, h_k): rows at equal steps of the integral of dd / h(d) from the spine (d = 0) to the host chain (D)."""
    dd = np.linspace(0.0, D, 2001)
    h = np.minimum(h_f + g * np.maximum(0.0, dd - band), h_host)
    q = np.r_[0, np.cumsum(np.diff(dd) / (0.5 * (h[1:] + h[:-1])))]
    K = max(2, int(round(q[-1])))
    d = np.interp(np.linspace(0, q[-1], K + 1), q, dd)
    return d, np.minimum(h_f + g * np.maximum(0.0, d - band), h_host)


def _counts(raw: np.ndarray, n0: int, nK: int) -> np.ndarray:
    N = np.maximum(2, np.round(raw).astype(int))
    N[0], N[-1] = n0, nK
    for _ in range(60):
        old = N.copy()
        for i in range(1, len(N)):
            lim = max(1, int(BIRTH_FRAC * N[i - 1]))
            N[i] = np.clip(N[i], N[i - 1] - lim, N[i - 1] + lim)
        N[-1] = nK
        for i in range(len(N) - 2, -1, -1):
            lim = max(1, int(BIRTH_FRAC * N[i + 1]))
            N[i] = np.clip(N[i], N[i + 1] - lim, N[i + 1] + lim)
        N[0] = n0
        if (N == old).all():
            break
    return N


def _thin(cur: np.ndarray, n_new: int, phase: list, recent: set) -> tuple[np.ndarray, set]:
    """cur (sorted taus, ends kept) changed to n_new intervals: interior nodes removed (or interval midpoints added)
    at golden-ratio positions, at most max(1, BIRTH_FRAC n) per row, not next to a node removed in the last row."""
    n = len(cur) - 1
    lim = max(1, int(BIRTH_FRAC * n))
    ch = int(np.clip(n_new - n, -lim, lim))
    if ch == 0 or n < 2:
        return cur, set()
    picked, tries = [], 0
    pool = list(range(1, n)) if ch < 0 else list(range(n))
    while len(picked) < abs(ch) and tries < 20 * n + 100:
        u = (0.37 + 0.6180339887 * phase[0]) % 1.0
        phase[0] += 1
        tries += 1
        k = pool[min(int(u * len(pool)), len(pool) - 1)]
        if k in picked or any(abs(k - j) <= 1 for j in picked) or (ch < 0 and k in recent and tries < 10 * n):
            continue
        picked.append(k)
    if ch < 0:
        keep = np.ones(len(cur), bool)
        keep[picked] = False
        out = cur[keep]
        return out, {int(np.searchsorted(out, cur[k])) for k in picked}
    mids = 0.5 * (cur[np.array(picked)] + cur[np.array(picked) + 1])
    return np.sort(np.r_[cur, mids]), set()


def build_side(bm: BandMap, spine_nodes: np.ndarray, chain_xy: np.ndarray, d: np.ndarray, h: np.ndarray,
               scale=None, stair: bool = False) -> BandBlock:
    """One side's band block: row 0 = spine_nodes (tau = their arc), row K = chain_xy (the host chain, exact), rows
    between as in the module docstring. Along the spine the rows share columns (the normals at the spine's nodes):
    a row keeps its inner neighbour's taus and drops (or adds) a few where its size target asks for fewer (more)
    cells, so the cells between are quads; round P and Q a row's nodes lie at equal arc on its quarter turn."""
    K = len(d) - 1
    tU = bm.tau_of(chain_xy)
    FD = bm.F(tU, bm.D)
    corr = gaussian_filter1d(chain_xy - FD, 1.5, axis=0, mode="nearest")   # the outer correction (smoothed over
                                                                            # 1.5 chain nodes: a zigzag chain)
    raw = chain_xy - FD                                     # B193: a fixed staircase chain (ORCA): the outer rows
    hK = float(h[-1])                                       # follow its steps, blending to the smooth correction

    def G(tau, dk):
        c = np.c_[np.interp(tau, tU, corr[:, 0]), np.interp(tau, tU, corr[:, 1])]
        if stair:
            w = float(np.clip(1.0 - (bm.D - dk) / (STAIR_ROWS * hK), 0.0, 1.0))
            if w > 0:
                cr = np.c_[np.interp(tau, tU, raw[:, 0]), np.interp(tau, tU, raw[:, 1])]
                c = (1.0 - w) * c + w * cr
        r = np.clip(dk / bm.D, 0, 1)
        return bm.F(tau, dk) + (r * r * (3 - 2 * r)) * c

    sp_pl = Polyline(spine_nodes)
    t0 = sp_pl.project(spine_nodes)
    t0[0], t0[-1] = 0.0, bm.Ls
    t0 = np.maximum.accumulate(t0)
    fan = np.linspace(0.0, 0.5 * np.pi, FAN_PTS + 1)
    dens = [-fan[::-1], bm.s, bm.Ls + fan]                  # the three segments: round P, along the spine, round Q
    xs, ts = [spine_nodes], [t0]
    segs, recent, phase = [None, t0.copy(), None], [set(), set(), set()], [0]
    for k in range(1, K):
        tau_parts = []
        for g_, td in enumerate(dens):
            c = G(td, d[k])
            sa = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
            n_new = max(1, int(round(sa[-1] / h[k])))
            prev = segs[g_]
            if prev is None:                                # a quarter turn's first row: equal arc
                cur = np.interp(np.linspace(0, sa[-1], n_new + 1), sa, td)
            else:                                           # the columns of the row inside, a few dropped / added
                cur, recent[g_] = _thin(prev, n_new, phase, recent[g_])
                if len(cur) > 2:                            # columns drift toward equal spacing (B181)
                    u = np.interp(cur, td, sa)
                    uu = np.linspace(0, sa[-1], len(cur))
                    st = sa[-1] / (len(cur) - 1)
                    u = u + np.clip(EQUALIZE * (uu - u), -EQ_MAX * st, EQ_MAX * st)
                    cur = np.interp(u, sa, td)
                cur[0], cur[-1] = td[0], td[-1]
            segs[g_] = cur
            tau_parts.append(cur)
        tau = np.r_[tau_parts[0][:-1], tau_parts[1], tau_parts[2][1:]]
        xs.append(G(tau, d[k]))
        ts.append(tau)
    xs.append(chain_xy)
    ts.append(tU)
    N = np.array([len(t) - 1 for t in ts])
    off = np.r_[0, np.cumsum(N + 1)]
    allxy = np.vstack(xs)
    ql, tl = [], []
    for k in range(K):
        qq, tt = zip_rows(ts[k], ts[k + 1], np.arange(off[k], off[k + 1]), np.arange(off[k + 1], off[k + 2]), allxy)
        ql += qq
        tl += tt
    row = np.concatenate([np.full(N[k] + 1, k) for k in range(K + 1)])
    info = {"rows": int(K), "counts": [int(x) for x in N], "d_km": [round(float(x) / 1e3, 3) for x in d],
            "h_km": [round(float(x) / 1e3, 3) for x in h]}
    return BandBlock(np.vstack(xs), np.array(ql, np.int64).reshape(-1, 4), np.array(tl, np.int64).reshape(-1, 3),
                     row, off, N, d, h, ts, info)


UNTANGLE_ROUNDS = 40         # B181: rounds of the untangling of a band block's folded cells ...
UNTANGLE_RINGS = 2           # ... each round: the folded cells' nodes and this many node rings around them ...
UNTANGLE_SWEEPS = 10         # ... move this many Jacobi sweeps (relaxation 0.5) toward their neighbours' mean


def _signed(xy, cells):
    if not len(cells):
        return np.zeros(0)
    p = xy[cells]
    return 0.5 * np.sum(p[:, :, 0] * np.roll(p[:, :, 1], -1, 1) - p[:, :, 1] * np.roll(p[:, :, 0], -1, 1), axis=1)


def _convex(xy, quads):
    if not len(quads):
        return np.zeros(0, bool)
    p = xy[quads]
    a = np.roll(p, -1, 1) - p
    b = np.roll(p, 1, 1) - p
    cr = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return np.all(cr * np.sign(cr.sum(1, keepdims=True)) < 0, axis=1) | np.all(cr * np.sign(cr.sum(1, keepdims=True)) > 0, axis=1)


def untangle(xy: np.ndarray, quads: np.ndarray, tris: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, dict]:
    """B181: the band's explicit rows can fold where the coast is rough (a bay, a fan over a rough bank): the nodes
    of every cell that is inverted (against the cells' majority orientation) or a non-convex quad, with
    UNTANGLE_RINGS node rings round them, are moved toward the mean of their edge neighbours (Jacobi, relaxation
    0.5), the fixed nodes (spine, host chain, row ends on the bank) never; at most UNTANGLE_ROUNDS rounds. Returns
    (xy, record)."""
    xy = xy.copy()
    n = len(xy)
    e = [np.c_[quads[:, k], quads[:, (k + 1) % 4]] for k in range(4)] + \
        [np.c_[tris[:, k], tris[:, (k + 1) % 3]] for k in range(3)]
    E = np.vstack([x for x in e if len(x)])
    E = np.unique(np.sort(E, axis=1), axis=0)
    deg = np.bincount(E.ravel(), minlength=n).astype(float)
    sq = np.sign(np.sum(_signed(xy, quads)) + np.sum(_signed(xy, tris)))
    rec = {"bad_before": 0, "bad_after": 0, "rounds": 0}
    for rnd in range(UNTANGLE_ROUNDS):
        bq = (sq * _signed(xy, quads) <= 0) | ~_convex(xy, quads)
        bt = sq * _signed(xy, tris) <= 0
        nb = int(bq.sum() + bt.sum())
        if rnd == 0:
            rec["bad_before"] = nb
        rec["bad_after"] = nb
        if nb == 0:
            break
        rec["rounds"] = rnd + 1
        mv = np.zeros(n, bool)
        mv[quads[bq].ravel()] = True
        mv[tris[bt].ravel()] = True
        for _ in range(UNTANGLE_RINGS + min(rnd // 10, 3)):
            mv[E[mv[E[:, 0]], 1]] = True
            mv[E[mv[E[:, 1]], 0]] = True
        mv &= ~fixed
        for _ in range(UNTANGLE_SWEEPS):
            acc = np.zeros_like(xy)
            np.add.at(acc, E[:, 0], xy[E[:, 1]])
            np.add.at(acc, E[:, 1], xy[E[:, 0]])
            tgt = acc / np.maximum(deg, 1)[:, None]
            xy[mv] += 0.5 * (tgt[mv] - xy[mv])
    return xy, rec



UNFOLD_ROUNDS = 60           # B186: rounds of the orientation-preserving repair of a band's remaining folds
UNFOLD_STEPS = (1.0, 0.5, 0.25, 0.1, 0.04)   # B186: trial moves of a node, in its shortest incident edge


def orient_all(xy: np.ndarray, quads: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """B186: every cell of a band keeps the block's topological orientation; the whole set is reversed when its
    total signed area (chart) is negative. A cell is never reversed alone (that hides a fold as an overlap)."""
    if np.sum(_signed(xy, quads)) + np.sum(_signed(xy, tris)) < 0:
        return quads[:, ::-1].copy(), tris[:, ::-1].copy()
    return quads, tris


def _corner_min(xy, cells):
    """Per cell: the smallest corner cross product (twice the signed area of the corner's triangle); <= 0 for a
    folded cell or a quad with a reflex corner (cells counter-clockwise)."""
    if not len(cells):
        return np.zeros(0)
    p = xy[cells]
    a = np.roll(p, -1, 1) - p
    b = np.roll(p, 1, 1) - p
    return (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).min(1)


def unfold(xy: np.ndarray, quads: np.ndarray, tris: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, dict]:
    """B186: the band's cells left folded (a corner cross product <= 0, cells counter-clockwise as one surface) are
    repaired by moving their free nodes and those one ring round them, one at a time, to the trial position (UNFOLD_STEPS x the shortest incident
    edge in 8 directions, or the neighbours' mean) that most raises the smallest corner cross product of the node's
    cells, normalised by the squared edge; at most UNFOLD_ROUNDS rounds. Returns (xy, record): what still folds is
    left to the validity gate (an invalid, refused band)."""
    xy = xy.copy()
    n = len(xy)
    cells = [(0, q) for q in quads] + [(1, t) for t in tris]
    inc = [[] for _ in range(n)]
    for k, (_, c) in enumerate(cells):
        for v in c:
            inc[int(v)].append(k)
    nbr = [set() for _ in range(n)]
    for _, c in cells:
        m = len(c)
        for i in range(m):
            nbr[int(c[i])].update((int(c[(i + 1) % m]), int(c[(i - 1) % m])))

    def score(v, pos):
        old = xy[v].copy()
        xy[v] = pos
        worst = np.inf
        for k in inc[v]:
            c = cells[k][1]
            p = xy[c]
            a = np.roll(p, -1, 0) - p
            b = np.roll(p, 1, 0) - p
            cr = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            e2 = max(float(np.min((a ** 2).sum(1))), 1e-12)
            worst = min(worst, float(cr.min()) / e2)
        xy[v] = old
        return worst

    dirs = np.c_[np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)]
    rec = {"folded_before": 0, "folded_after": 0, "rounds": 0}
    for rnd in range(UNFOLD_ROUNDS):
        bad = np.r_[_corner_min(xy, quads) <= 0, _corner_min(xy, tris) <= 0]
        if rnd == 0:
            rec["folded_before"] = int(bad.sum())
        rec["folded_after"] = int(bad.sum())
        if not bad.any():
            break
        rec["rounds"] = rnd + 1
        core = {int(v) for k in np.flatnonzero(bad) for v in cells[k][1]}
        ring = core | {w for v in core for w in nbr[v]}            # the folded cells' nodes and one ring round them
        nodes = sorted(v for v in ring if not fixed[v])
        moved = False
        for v in nodes:
            nb = list(nbr[v])
            if not nb:
                continue
            L = float(np.min(np.linalg.norm(xy[nb] - xy[v], axis=1)))
            cand = [xy[nb].mean(0)] + [xy[v] + f * L * d for f in UNFOLD_STEPS for d in dirs]
            s0 = score(v, xy[v])
            best, bp = s0, None
            for c in cand:
                sc = score(v, c)
                if sc > best + 1e-12:
                    best, bp = sc, c
            if bp is not None:
                xy[v] = bp
                moved = True
        if not moved:
            break
    return xy, rec
