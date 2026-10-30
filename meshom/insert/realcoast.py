"""The real coast of an insert (decisions B164, B166): the run's coast (meshom.coast, the `[coast] source` the run
was built from) cut to the window, smoothed at the insert's own cell size, its bays found and bridged.

The rule: the insert's banks follow the real coast, never the host's cut staircase; the join
and the outside keep the host's coast exactly. Steps (all in the insert's stereographic chart):

1. pieces: every coast loop (water on its left, meshom.domain's convention) cut to the window: closed loops inside
   it, open runs where a loop leaves it;
2. smoothing (B164): a Gaussian G along the arc whose width follows the insert's local cell size,
   sigma(x) = SIGMA_CELLS x h_loc(x), h_loc the size target of the insert near x (the coast filter's idea, C16, at
   the insert's h instead of the run's), applied as "twicing" 2G - G^2: wiggles shorter than sigma go as with G,
   but a cape or a bay wider than sigma keeps its place (G alone pulls a cape back by ~ curvature x sigma^2 and
   widened Bonifacio's neck from 17.9 to 23.7 km). No curvature limit: rounding every bend to a radius of 2.5 cells
   moved the Denmark Strait's neck from 310 to 357 km and closed Dover's funnel; the cells at a tight cape are left
   to the cleaning instead (meshom.insert.network, B166);
3. bays (B166): a pocket of the smoothed coast is a run of it that a chord of at most MOUTH_MAX_H local cells (and
   at most MOUTH_MAX_W of the local width) bridges on the water side, deeper than DEPTH_H local cells and DEPTH_RATIO
   of its mouth. The "main coast" takes the chord; a pocket whose mouth holds at least MOUTH_MIN_H cells is meshed
   as a bay block of its own (meshom.insert.network), a narrower one is dropped (water narrower than two cells
   closes, as the coast filter's opening, C41) and reported.

Capes are not split off: inside a bank a cape is a narrowing that the block's rows carry (the banana), smoothed at
the cell size; at a join it is the corner rule's job.
"""
from dataclasses import dataclass, field

import numba
import numpy as np

from meshom.insert.geom import Polyline

SIGMA_CELLS = 1.5            # B164: the bank smoothing's Gaussian, in local insert cells
STEP_CELLS = 0.25            # B164: resampling step of the smoothed coast, in the smallest local cell
BLEND_H = 2.0                # B164: a bank blends from the join's host end node onto the real coast over this many
                             # join cells
DEPTH_H = 1.0                # B166: a pocket is a bay when deeper than this many local cells ...
DEPTH_RATIO = 0.3            # B166: ... and deeper than this fraction of its mouth
MOUTH_MAX_H = 12.0           # B166: a bay's mouth is at most this many local cells ...
MOUTH_MAX_W = 0.6            # B166: ... and at most this fraction of the channel's local width
MOUTH_MIN_H = 2.0            # B166: a bay narrower than this many local cells is dropped (not meshed)
ARC_MAX_MOUTH = 6.0          # B166: a bay's coast is at most this many times its longest allowed mouth
SIDE_TOL_H = 0.05            # B166: a coast point may lie this far (local cells) on the water side of a mouth chord


@dataclass
class Bay:
    piece: int
    i: int                       # first and last index of the pocket on the smoothed piece (the mouth ends)
    j: int
    arc: np.ndarray              # (K,2) the smoothed coast from mouth end a to mouth end b (water on the left)
    depth_m: float
    mouth_m: float
    h_m: float                   # local cell size at the mouth
    meshed: bool                 # mouth >= MOUTH_MIN_H cells: a bay block; else dropped

    @property
    def a(self):
        return self.arc[0]

    @property
    def b(self):
        return self.arc[-1]

    def record(self) -> dict:
        return {"depth_km": round(self.depth_m / 1e3, 3), "mouth_km": round(self.mouth_m / 1e3, 3),
                "h_km": round(self.h_m / 1e3, 3), "depth_cells": round(self.depth_m / self.h_m, 2),
                "mouth_cells": round(self.mouth_m / self.h_m, 2), "meshed": self.meshed}


@dataclass
class RealCoast:
    pieces: list                 # chart (K,2), water on the left
    closed: list                 # bool per piece
    source: str = ""
    main: list = field(default_factory=list)       # the smoothed pieces with bays bridged
    smooth: list = field(default_factory=list)     # the smoothed pieces
    bays: list = field(default_factory=list)       # Bay
    _pl: list = field(default_factory=list)

    @classmethod
    def from_loops(cls, loops_xyz: list, chart, centre: np.ndarray, radius_m: float, source: str = ""):
        from meshom.geo import R_EARTH_M

        c = centre / np.linalg.norm(centre)
        cosr = np.cos(min(radius_m / R_EARTH_M, np.pi))
        pieces, closed = [], []
        for lp in loops_xyz:
            ins = lp @ c >= cosr
            if not ins.any():
                continue
            if ins.all():
                pieces.append(chart.fwd(lp))
                closed.append(True)
                continue
            k = int(np.flatnonzero(~ins)[0])                       # rotate so the loop starts outside
            lp, ins = np.roll(lp, -k, axis=0), np.roll(ins, -k)
            d = np.diff(np.r_[0, ins.astype(int), 0])
            for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                if b - a >= 2:
                    pieces.append(chart.fwd(lp[a:b]))
                    closed.append(False)
        return cls(pieces, closed, source)

    @staticmethod
    def _segs(pieces, closed) -> np.ndarray:
        out = []
        for p, cl in zip(pieces, closed):
            q = np.vstack([p, p[:1]]) if cl else p
            if len(q) >= 2:
                out.append(np.stack([q[:-1], q[1:]], axis=1))
        return np.concatenate(out) if out else np.zeros((0, 2, 2))

    def segments(self, which: str = "raw") -> np.ndarray:
        pcs = {"raw": self.pieces, "smooth": self.smooth, "main": self.main}[which]
        return self._segs(pcs, self.closed)

    # ------------------------------------------------------------------------------------- smoothing and bays
    def prepare(self, h_loc, w_loc, h_floor: float, sigma_cells: float = SIGMA_CELLS, bays: bool = True,
                bridge: str = "all"):
        """Smooth every piece at sigma_cells x h_loc and find its bays (module docstring). h_loc, w_loc: callables
        (K,2) -> (K,) of the local cell size and the local channel width (inf where not in a channel). bridge:
        "all" (the main coast takes every pocket's chord) or "narrow" (only the pockets whose mouth is below
        MOUTH_MIN_H cells: a coastal strip keeps its wider bays in its spine, B181)."""
        self.smooth, self.main, self.bays = [], [], []
        step = max(STEP_CELLS * h_floor, 1.0)
        for ip, (p, cl) in enumerate(zip(self.pieces, self.closed)):
            q = _resample(p, cl, step)
            if sigma_cells > 0 and len(q) >= 4:
                s = _arc(q, cl)
                sig = sigma_cells * np.asarray(h_loc(q), dtype=float)
                per = float(s[-1] + (np.linalg.norm(q[0] - q[-1]) if cl else 0.0))
                g1 = _smooth_var(q, s, sig, cl, per)
                q = 2.0 * g1 - _smooth_var(g1, s, sig, cl, per)          # twicing (B164): 2G - G^2 keeps capes

            self.smooth.append(q)
            if not bays or len(q) < 4:
                self.main.append(q)
                continue
            hq = np.asarray(h_loc(q), dtype=float)
            wq = np.asarray(w_loc(q), dtype=float)
            lmax = np.minimum(MOUTH_MAX_H * hq, np.where(np.isfinite(wq), MOUTH_MAX_W * wq, np.inf))
            far, dep = _far_chords(q, _arc(q, False), lmax, ARC_MAX_MOUTH * lmax, SIDE_TOL_H * hq, DEPTH_H * hq,
                                   DEPTH_RATIO)
            cand = [(dep[i] ** 2 / np.linalg.norm(q[far[i]] - q[i]), i, int(far[i])) for i in np.flatnonzero(far >= 0)]
            taken = np.zeros(len(q), bool)
            extra = []
            for _, i, j in sorted(cand, reverse=True):       # deepest for their mouth first, no overlaps
                if taken[i:j + 1].any():
                    continue
                taken[i:j + 1] = True
                L = float(np.linalg.norm(q[j] - q[i]))
                h_m = float(0.5 * (hq[i] + hq[j]))
                self.bays.append(Bay(ip, i, j, q[i:j + 1].copy(), float(dep[i]), L, h_m, L >= MOUTH_MIN_H * h_m))
                if bridge == "narrow" and self.bays[-1].meshed:
                    continue
                m = max(int(np.ceil(L / step)), 1)
                tt = np.linspace(0, 1, m + 1)[1:-1, None]
                extra.append((i, j, q[i] + tt * (q[j] - q[i])))
            extra.sort(key=lambda e: e[0])
            if extra:
                parts, last = [], 0
                for i0, j0, pts in extra:
                    parts += [q[last:i0 + 1], pts]
                    last = j0
                parts.append(q[last:])
                q = np.vstack(parts)
            self.main.append(q)
        self._pl = [Polyline(np.vstack([m, m[:1]]) if cl else m) for m, cl in zip(self.main, self.closed)]
        return self

    # ------------------------------------------------------------------------------------------ banks
    def nearest(self, xy: np.ndarray) -> tuple[int, float, float]:
        """(piece, arc length on its main polyline, distance) of the main coast point nearest to xy."""
        best = (-1, 0.0, np.inf)
        for k, pl in enumerate(self._pl):
            if pl.L <= 0:
                continue
            s = float(pl.project(xy[None])[0])
            d = float(np.linalg.norm(pl.at(s)[0] - xy))
            if d < best[2]:
                best = (k, s, d)
        return best

    def walk(self, xy: np.ndarray, dist: float) -> list:
        """The two main-coast points `dist` along the coast either way from the point nearest to xy."""
        k, s, _ = self.nearest(xy)
        if k < 0:
            return []
        pl = self._pl[k]
        if self.closed[k]:
            return [pl.at((s + dist) % pl.L)[0], pl.at((s - dist) % pl.L)[0]]
        return [pl.at(min(s + dist, pl.L))[0], pl.at(max(s - dist, 0.0))[0]]

    def arc(self, xa: np.ndarray, xb: np.ndarray, guide: np.ndarray | None = None) -> tuple[np.ndarray | None, str]:
        """The main coast from the point nearest to xa to the point nearest to xb (both on one piece; on a closed
        piece the way nearer to `guide`, the host's bank), blended at both ends onto xa and xb exactly (BLEND_H
        join cells; the caller passes the join cell sizes through `blend`). Returns (polyline or None, why)."""
        ka, sa, _ = self.nearest(xa)
        kb, sb, _ = self.nearest(xb)
        if ka < 0 or ka != kb:
            return None, "the two ends lie on different coast pieces"
        pl = self._pl[ka]
        if not self.closed[ka]:
            ways = [_sub(pl, sa, sb)]
        else:
            L = pl.L
            fwd = _sub(pl, sa, sb if sb >= sa else sb + L, wrap=L)
            bwd = _sub(pl, sb, sa if sa >= sb else sa + L, wrap=L)[::-1]
            ways = [fwd, bwd]
        if guide is not None and len(ways) > 1:
            gp = Polyline(guide)
            ways.sort(key=lambda w: float(np.mean(Polyline(w).distance(gp.xy))) + float(np.mean(gp.distance(w))))
        return ways[0], ""


def _curvature(q: np.ndarray, closed: bool, half_pts: np.ndarray) -> np.ndarray:
    """|turning| over +-half_pts points divided by the arc between them (1/m)."""
    n = len(q)
    idx = np.arange(n)
    lo, hi = idx - half_pts, idx + half_pts
    if closed:
        lo, hi = lo % n, hi % n
    else:
        lo, hi = np.clip(lo, 0, n - 1), np.clip(hi, 0, n - 1)
    d1 = q - q[lo]
    d2 = q[hi] - q
    a = np.abs(np.arctan2(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0], (d1 * d2).sum(1)))
    L = np.linalg.norm(d1, axis=1) + np.linalg.norm(d2, axis=1)
    return np.where(L > 0, 2.0 * a / np.maximum(L, 1e-9), 0.0)


def blend_ends(arc: np.ndarray, xa: np.ndarray, xb: np.ndarray, la: float, lb: float) -> np.ndarray:
    """arc shifted near its ends so that it starts at xa and ends at xb exactly: the offset tapers to zero over la
    (lb) of arc length with a smoothstep."""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(arc, axis=0), axis=1))]
    L = s[-1]

    def ss(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3 - 2 * t)
    wa = 1.0 - ss(s / max(la, 1e-9))
    wb = 1.0 - ss((L - s) / max(lb, 1e-9))
    out = arc + wa[:, None] * (xa - arc[0]) + wb[:, None] * (xb - arc[-1])
    out[0], out[-1] = xa, xb
    return out


def _sub(pl: Polyline, s0: float, s1: float, wrap: float | None = None) -> np.ndarray:
    """The polyline between arc lengths s0 and s1 (s1 may exceed the length on a closed polyline: wrap)."""
    rev = s1 < s0
    a, b = (s1, s0) if rev else (s0, s1)
    if wrap is not None and b > pl.L:
        inner1 = pl.xy[(pl.s > a) & (pl.s < pl.L)]
        inner2 = pl.xy[(pl.s > 0) & (pl.s < b - pl.L)]
        out = np.vstack([pl.at(a), inner1, inner2, pl.at(b - pl.L)])
    else:
        inner = pl.xy[(pl.s > a) & (pl.s < b)]
        out = np.vstack([pl.at(a), inner, pl.at(b)])
    return out[::-1] if rev else out


def _arc(q: np.ndarray, closed: bool) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]


def _resample(p: np.ndarray, closed: bool, step: float) -> np.ndarray:
    q = np.vstack([p, p[:1]]) if closed else p
    pl = Polyline(q)
    if pl.L <= 0:
        return p
    m = max(int(np.ceil(pl.L / step)), 3)
    t = np.linspace(0.0, pl.L, m + 1)
    out = pl.at(t)
    return out[:-1] if closed else out


@numba.njit(cache=True)
def _smooth_var(q, s, sig, closed, period):
    """Gaussian along the arc with a width per output point; open pieces keep their ends (mode 'nearest')."""
    n = len(q)
    out = np.empty_like(q)
    for i in range(n):
        si = sig[i]
        if si <= 0:
            out[i] = q[i]
            continue
        w0 = 0.0
        x = 0.0
        y = 0.0
        r = 3.0 * si
        for j in range(n):
            d = s[j] - s[i]
            if closed:
                if d > 0.5 * period:
                    d -= period
                elif d < -0.5 * period:
                    d += period
            if abs(d) > r:
                continue
            w = np.exp(-0.5 * (d / si) ** 2)
            w0 += w
            x += w * q[j, 0]
            y += w * q[j, 1]
        if not closed:                          # the missing tail beyond an open end counts as the end point
            ds = (s[n - 1] - s[0]) / max(n - 1, 1)
            for e in range(2):
                ie = 0 if e == 0 else n - 1
                d = abs(s[i] - s[ie])
                if d < r:
                    acc = 0.0
                    hh = (r - d) / 16.0
                    for k in range(16):
                        tt = d + (k + 0.5) * hh
                        acc += np.exp(-0.5 * (tt / si) ** 2) * hh
                    wt = acc / max(ds, 1e-12)
                    w0 += wt
                    x += wt * q[ie, 0]
                    y += wt * q[ie, 1]
        out[i, 0] = x / w0
        out[i, 1] = y / w0
    if not closed:
        out[0] = q[0]
        out[n - 1] = q[n - 1]
    return out


@numba.njit(cache=True)
def _far_chords(q, s, lmax, arcmax, tol, dmin, ratio):
    """For every i: the chord q_i -> q_j (j > i + 1, |q_j - q_i| <= lmax_i, s_j - s_i <= arcmax_i, every q_k between
    on the land side (right) of it within tol_i) whose pocket is deepest for its mouth (score depth^2 / mouth) among
    those deeper than dmin_i and ratio x mouth; (j or -1, depth)."""
    n = len(q)
    far = np.full(n, -1, np.int64)
    dep = np.zeros(n)
    for i in range(n - 2):
        best = 0.0
        for j in range(i + 2, n):
            if s[j] - s[i] > arcmax[i]:
                break
            dx = q[j, 0] - q[i, 0]
            dy = q[j, 1] - q[i, 1]
            L = np.sqrt(dx * dx + dy * dy)
            if L > lmax[i] or L <= 0:
                continue
            ok = True
            d = 0.0
            for k in range(i + 1, j):
                cr = (dx * (q[k, 1] - q[i, 1]) - dy * (q[k, 0] - q[i, 0])) / L
                if cr > tol[i]:
                    ok = False
                    break
                if -cr > d:
                    d = -cr
            if ok and d >= dmin[i] and d >= ratio * L and d * d / L > best:
                best = d * d / L
                far[i] = j
                dep[i] = d
    return far, dep


def islands_in(rc: RealCoast, outline: np.ndarray) -> list:
    """Closed coast pieces lying inside the insert's outline (islands in the strait): [(area km^2, centre xy)]."""
    from meshom.insert.geom import point_in_polygon

    out = []
    for p, cl in zip(rc.pieces, rc.closed):
        if cl and len(p) >= 3 and point_in_polygon(p, outline).all():
            x, y = p[:, 0], p[:, 1]
            a = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
            out.append((a / 1e6, p.mean(0)))
    return out
