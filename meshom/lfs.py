"""Local feature size: the medial axis of the water on the sphere, its width W, and the passages along it (decisions
S22, S31, S32).

Medial axis (`medial_axis`). The coast loops are resampled at `step` (equal steps of the integral of ds/step, corners
kept: domain.resample_loops) and the samples triangulated on the sphere: sphere_delaunay, with ghost points where the
samples leave the sphere empty (Fibonacci points farther than GHOST_M from every sample: a hull of points in one cap
has a degenerate back face; a ghost changes only discs wider than GHOST_M / 2), the coast edges
recovered and the land removed (recover_edges, remove_land). A water triangle whose three corners are coast samples is
a disc touching the coast at its corners and empty of samples: its circumcentre is a Voronoi vertex of the samples
inside the water, i.e. a medial-axis vertex, its circumradius r the distance to the coast there and W = 2 r the width
of the water across it. Vertices are joined across shared Delaunay edges (the Voronoi edges).
Pruning (S32): (1) the discrete lambda-medial axis (Chazal & Lieutier 2005, Chaussard et al. 2011): a vertex stays
when the smallest circle around its three samples has radius >= lam (an acute triangle: r; otherwise half its longest
edge): three samples on one smooth bank give a circle of the sample step and drop, two banks give W / 2; (2) the
potential residue (Ogniewicz & Kubler 1995): a vertex stays when two of its samples lie on different loops or when
the coast between them, the shorter way along their loop, is longer than their chord by at least RESIDUE W (a coastal
concavity, a sampling ripple or a small bay holds little coast behind its vertices); (3) dead
ends go: a branch from a leaf to its first junction that is never wider than OPEN_RATIO times the width at the
junction (a bay, a fjord, a coast bump or a basin corner; a leaf next to a ghost triangle is open water and stays) is
removed, repeated until none is left (a tree of bays goes branch by branch, so a strait's centreline runs on through
the junctions of its side bays); isolated vertices drop.

Passages (`passages`, S31). The axis graph is cut into chains between its terminal vertices (leaves, junctions).
Along a chain: s the arc length, W(s) and dW/ds by central differences over half a width on each side. A passage is
a run of a chain with W <= w_max h, cut where |dW/ds| > n g (n = max(n_min, round(median W/h)) cells across, the
width changing faster than the grading g allows at h = W/n; two rounds, n recomputed on each piece), whose two ends
are both mouths. The run stops at a junction (the centreline branches), where W exceeds w_max h ("wide"), or where
the width changes too fast; such an end is a mouth when wider water lies beyond it: a search of the axis outward
(never back through the run) reaches a vertex with W > OPEN_RATIO W_end or an open vertex (next to a ghost triangle:
a disc wider than GHOST_M / 2) before it runs out of vertices; otherwise it is a dead end (the head of a bay or a
fjord, whose corner branches end in leaves) and the run is no passage. Cycles of the axis without a terminal are not
searched.

`auto_straits` turns the passages of a coast into the size zone of meshom.run.size_field: h = W / n within W / 2 of
the centreline, then linear with the slope g to the base h (decisions S22, S24: a passage whose W / n falls below
h_min anywhere is too narrow and stays closed, i.e. gets no zone; a strait list entry still opens it, C26-C31).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from meshom.delaunay import adjacency, recover_edges, remove_land, sphere_delaunay
from meshom.domain import Domain, loop_edges, resample_loops
from meshom.geo import R_EARTH_M, xyz_to_lonlat
from meshom.size import SizeField

GHOST_M = 2.5e6          # ghost points at least this far from every coast sample (discs up to half of it are exact)
GHOST_N = 4000           # Fibonacci candidates for the ghost points (spacing ~360 km)
LAM = 0.5                # S32: lambda = LAM h (the smallest half-width kept, units of h)
STEP = 0.25              # coast samples every STEP h (and at most lambda / 2)
DEAD_ROUNDS = 200        # S32: rounds of dead-end pruning (each removes every dead leaf branch)
W_MAX = 20.0             # S31: a passage is at most W_MAX h wide
GRADING = 0.1            # S31: |dW/ds| <= n g with g the size grading (the limiter's default grad_max)
SLOPE_BASE = 0.5         # dW/ds over a baseline of SLOPE_BASE W on each side of the station
RESIDUE = 2.0            # S32: potential residue >= RESIDUE W
OPEN_RATIO = 1.02        # S31: a run's end opens when the axis beyond it reaches a width > OPEN_RATIO W_end
OPEN_BUDGET = 20000      # ... searched over at most this many vertices (more: a large network, taken as open)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _arc(chord: np.ndarray) -> np.ndarray:
    return 2.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))


def _fibonacci(m: int) -> np.ndarray:
    i = np.arange(m) + 0.5
    z = 1.0 - 2.0 * i / m
    ph = np.pi * (1.0 + 5**0.5) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(ph), r * np.sin(ph), z])


@dataclass
class MedialAxis:
    points: np.ndarray       # (K,3) medial vertices (circumcentres)
    r: np.ndarray            # (K,) medial radius = distance to the coast, m
    tri: np.ndarray          # (K,3) the coast samples of each vertex's Delaunay triangle (CCW seen from outside)
    nbr: np.ndarray          # (K,3) neighbour vertex across tri edge k (tri[k] -> tri[k+1]), -1 none
    open: np.ndarray         # (K,) the vertex borders a disc wider than GHOST_M / 2 (a ghost triangle): open water
    samples: np.ndarray      # (N,3) coast samples
    loop: np.ndarray         # (N,) loop index of each sample (into the loops the axis was built from)
    info: dict = field(default_factory=dict)

    @property
    def degree(self) -> np.ndarray:
        return (self.nbr >= 0).sum(axis=1)


def medial_axis(loops: list, step: SizeField, lam: SizeField) -> MedialAxis:
    """The pruned lambda-medial axis of the water bounded by `loops` (module docstring). step, lam: callables
    xyz -> metres (sample spacing along the coast; the smallest half-width kept)."""
    info = {}
    keep = [i for i, lp in enumerate(loops) if len(lp) >= 3]
    rl = resample_loops(Domain([loops[i] for i in keep]), step)
    nl = [len(lp) for lp in rl]
    S = np.vstack(rl)
    loop_of = np.repeat(np.asarray(keep, dtype=np.int64), nl)
    tree = cKDTree(S)
    fib = _fibonacci(GHOST_N)
    d, _ = tree.query(fib)
    ghosts = fib[_arc(d) * R_EARTH_M > GHOST_M]
    P = np.vstack([S, ghosts])
    e = loop_edges(rl)
    tris, pe, _ = recover_edges(P, sphere_delaunay(P), e, return_edges=True)      # pe: e split at touches (M40)
    tris = remove_land(P, tris, pe)
    info.update(samples=len(S), ghosts=len(ghosts), water_tris=len(tris))
    ghost_t = (tris >= len(S)).any(axis=1)
    nb0 = adjacency(tris)
    near_ghost = (np.where(nb0 >= 0, ghost_t[np.maximum(nb0, 0)], False)).any(axis=1)[~ghost_t]
    tris = tris[~ghost_t]
    a, b, c = S[tris[:, 0]], S[tris[:, 1]], S[tris[:, 2]]
    cen = _unit(np.cross(b - a, c - a))
    r = _arc(np.linalg.norm(cen - a, axis=1)) * R_EARTH_M
    # lambda test: the smallest circle around the three samples (acute: the circumcircle; else the longest edge's)
    l2 = np.column_stack([((b - a) ** 2).sum(1), ((c - b) ** 2).sum(1), ((a - c) ** 2).sum(1)])
    lmax = l2.max(axis=1)
    obtuse = 2.0 * lmax > l2.sum(axis=1)
    rho = np.where(obtuse, 0.5 * _arc(np.sqrt(lmax)) * R_EARTH_M, r)
    ok = rho >= lam(cen)
    info["lambda_kept"] = int(ok.sum())
    # potential residue: per pair of samples, the coast between them (the shorter way) minus the chord
    pos = np.concatenate([np.r_[0.0, np.cumsum(R_EARTH_M * _arc(np.linalg.norm(np.diff(lp, axis=0), axis=1)))]
                          for lp in rl])
    per = np.concatenate([np.full(len(lp), R_EARTH_M * float(_arc(np.linalg.norm(np.diff(np.vstack([lp, lp[:1]]),
                                                                                       axis=0), axis=1)).sum()))
                          for lp in rl])
    res = np.zeros(len(tris))
    for i, j in ((0, 1), (1, 2), (2, 0)):
        u, v = tris[:, i], tris[:, j]
        d = np.abs(pos[u] - pos[v])
        d = np.minimum(d, per[u] - d) - R_EARTH_M * _arc(np.linalg.norm(S[u] - S[v], axis=1))
        res = np.maximum(res, np.where(loop_of[u] != loop_of[v], np.inf, d))
    ok &= res >= RESIDUE * 2.0 * r
    info["residue_kept"] = int(ok.sum())
    nbr_all = adjacency(tris)
    idx = np.full(len(tris), -1, dtype=np.int64)
    idx[ok] = np.arange(int(ok.sum()))
    nbr = np.where(nbr_all[ok] >= 0, idx[np.maximum(nbr_all[ok], 0)], -1)
    ma = MedialAxis(cen[ok], r[ok], tris[ok], nbr, near_ghost[ok], S, loop_of, info)
    _prune_dead(ma)
    return ma


def _drop(ma: MedialAxis, alive: np.ndarray) -> None:
    idx = np.full(len(alive), -1, dtype=np.int64)
    idx[alive] = np.arange(int(alive.sum()))
    nb = ma.nbr[alive]
    ma.nbr = np.where(nb >= 0, idx[np.maximum(nb, 0)], -1)
    ma.points, ma.r, ma.tri, ma.open = ma.points[alive], ma.r[alive], ma.tri[alive], ma.open[alive]


def _prune_dead(ma: MedialAxis) -> None:
    """S32 (2): dead-end branches go, round by round, until none is left; isolated vertices go."""
    removed, rounds = 0, 0
    for rounds in range(1, DEAD_ROUNDS + 1):
        nbr, W = ma.nbr, 2.0 * ma.r
        deg = (nbr >= 0).sum(axis=1)
        alive = deg > 0
        for lf in np.flatnonzero((deg == 1) & ~ma.open):
            path, prev, v, wmax, junction = [lf], -1, lf, W[lf], -1
            while True:
                nx = [int(u) for u in nbr[v] if u >= 0 and u != prev and alive[u]]
                if not nx:
                    break
                u = nx[0]
                if deg[u] >= 3:
                    junction = u
                    break
                if deg[u] == 1:                              # an isolated chain: kept
                    break
                path.append(u)
                wmax = max(wmax, W[u])
                prev, v = v, u
            if junction >= 0 and wmax <= OPEN_RATIO * W[junction]:
                alive[path] = False
                deg[junction] -= 1
        alive &= deg > 0
        if alive.all():
            break
        removed += int((~alive).sum())
        _drop(ma, alive)
    ma.info.update(dead_removed=removed, dead_rounds=rounds, vertices=len(ma.r))


# --------------------------------------------------------------------------------------------------- passages

@dataclass
class Passage:
    ids: np.ndarray          # (S,) medial vertex ids along the centreline
    points: np.ndarray       # (S,3) centreline stations
    W: np.ndarray            # (S,) width, m
    s: np.ndarray            # (S,) arc length from the first station, m
    dWds: np.ndarray         # (S,)
    n: int                   # cells across
    ends: tuple              # (kind at station 0, kind at station S-1): junction | wide | slope | open

    @property
    def length_m(self) -> float:
        return float(self.s[-1] - self.s[0])

    def record(self, h_m: np.ndarray | None = None) -> dict:
        lon, lat = xyz_to_lonlat(self.points[[len(self.W) // 2]])
        k = int(np.argmin(self.W))
        lo, la = xyz_to_lonlat(self.points[[k]])
        out = {"centre_lonlat": [round(float(lon[0]), 4), round(float(lat[0]), 4)],
               "narrowest_lonlat": [round(float(lo[0]), 4), round(float(la[0]), 4)],
               "W_min_m": round(float(self.W.min()), 1), "W_median_m": round(float(np.median(self.W)), 1),
               "W_max_m": round(float(self.W.max()), 1), "length_m": round(self.length_m, 1), "n": int(self.n),
               "stations": int(len(self.W)), "ends": list(self.ends)}
        k = np.unique(np.linspace(0, len(self.W) - 1, min(len(self.W), 16)).round().astype(np.int64))
        lo, la = xyz_to_lonlat(self.points[k])                              # the centreline, at most 16 stations
        out["path_lonlat"] = np.round(np.column_stack([lo, la]), 4).tolist()
        if h_m is not None:
            out["h_zone_min_m"] = round(float(self.W.min() / self.n), 1)
        return out


def chains(ma: MedialAxis) -> list[np.ndarray]:
    """Vertex sequences between terminal vertices (degree != 2), each once; cycles without a terminal are skipped."""
    nbr = ma.nbr
    deg = (nbr >= 0).sum(axis=1)
    term = deg != 2
    seen = set()
    out = []
    for t in np.flatnonzero(term & (deg > 0)):
        for u in nbr[t]:
            if u < 0 or (t, u) in seen:
                continue
            seq, prev, v = [t], t, int(u)
            while True:
                seq.append(v)
                if term[v]:
                    break
                nx = [x for x in nbr[v] if x >= 0 and x != prev]
                if not nx:
                    break
                prev, v = v, int(nx[0])
            seen.add((seq[-1], seq[-2]))
            seen.add((t, int(u)))
            out.append(np.asarray(seq, dtype=np.int64))
    return out


def _slope(s: np.ndarray, W: np.ndarray) -> np.ndarray:
    """dW/ds by central differences over SLOPE_BASE W on each side (clamped at the chain ends)."""
    lo = np.searchsorted(s, s - SLOPE_BASE * W, side="left")
    hi = np.minimum(np.searchsorted(s, s + SLOPE_BASE * W, side="right") - 1, len(s) - 1)
    lo = np.minimum(lo, np.arange(len(s)))
    hi = np.maximum(hi, np.arange(len(s)))
    ds = s[hi] - s[lo]
    return np.where(ds > 0, (W[hi] - W[lo]) / np.where(ds > 0, ds, 1.0), 0.0)


def _intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [a, b] index intervals where mask holds."""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def _opens(ma: MedialAxis, v0: int, run: np.ndarray, w_end: float) -> bool:
    """Wider water beyond the run's end vertex v0: a search of the axis from v0 away from the run (module docstring)."""
    seen = set(run.tolist())
    stack = [int(u) for u in ma.nbr[v0] if u >= 0 and u not in seen]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        if ma.open[u] or 2.0 * ma.r[u] > OPEN_RATIO * w_end or len(seen) > OPEN_BUDGET:
            return True
        stack += [int(x) for x in ma.nbr[u] if x >= 0 and x not in seen]
    return False


def passages(ma: MedialAxis, h: SizeField, n_min: int, w_max: float = W_MAX, g: float = GRADING,
             min_stations: int = 3) -> list[Passage]:
    """The passages of the axis (module docstring, S31). h: the size field the widths are measured against."""
    deg = ma.degree
    out = []
    for ch in chains(ma):
        if len(ch) < min_stations:
            continue
        P = ma.points[ch]
        s = np.r_[0.0, np.cumsum(_arc(np.linalg.norm(np.diff(P, axis=0), axis=1)))] * R_EARTH_M
        W = 2.0 * ma.r[ch]
        hv = h(P)
        dW = _slope(s, W)
        okw = W <= w_max * hv
        pieces = []
        for a, b in _intervals(okw):
            cur = [(a, b)]
            for _ in range(2):                              # n from the piece, cut where |dW/ds| > n g, twice
                nxt = []
                for a1, b1 in cur:
                    n = max(n_min, int(round(float(np.median(W[a1:b1 + 1] / hv[a1:b1 + 1])))))
                    for a2, b2 in _intervals(np.abs(dW[a1:b1 + 1]) <= n * g):
                        nxt.append((a1 + a2, a1 + b2))
                cur = nxt
            pieces += cur
        last = len(ch) - 1
        for a, b in pieces:
            if b - a + 1 < min_stations:
                continue
            ends = []
            for e, o in ((a, a - 1), (b, b + 1)):
                kind = "junction" if e in (0, last) and deg[ch[e]] >= 3 else \
                    "open" if e in (0, last) else "wide" if not okw[o] else "slope"
                if kind in ("junction", "slope") and not _opens(ma, int(ch[e]), ch[a:b + 1], float(W[e])):
                    kind = "dead"
                if kind == "open" and not ma.open[ch[e]]:
                    kind = "dead"
                ends.append(kind)
            if "dead" in ends:
                continue
            n = max(n_min, int(round(float(np.median(W[a:b + 1] / hv[a:b + 1])))))
            out.append(Passage(ch[a:b + 1], P[a:b + 1], W[a:b + 1], s[a:b + 1] - s[a], dW[a:b + 1], n,
                               tuple(ends)))
    return out


# ------------------------------------------------------------------------------------------- size zone (run)

class PassageField:
    """min over the passages of W(s)/n within W(s)/2 of the centreline (s = the nearest station), linear with the
    slope g up to h_out_m beyond; h_out_m elsewhere (decisions S22, S24). One k-d tree and one bounding cap per
    passage, built once; a point is looked up only in the passages whose cap holds it."""

    def __init__(self, stations: list, hz: list, half: list, h_out_m: float, grad: float):
        self.stations, self.hz, self.half = tuple(stations), tuple(hz), tuple(half)
        self.h_out_m, self.grad = float(h_out_m), float(grad)
        self.trees = [cKDTree(P) for P in self.stations]
        self.centres = np.vstack([_unit(P.mean(axis=0)) for P in self.stations])
        cap = [float(np.arccos(np.clip((P @ c).min(), -1.0, 1.0))) + float(hh.max() + (h_out_m - z.min()) / grad)
               / R_EARTH_M for P, c, z, hh in zip(self.stations, self.centres, self.hz, self.half)]
        self.cos_cap = np.cos(np.minimum(cap, np.pi))

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "trees"}

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.trees = [cKDTree(P) for P in self.stations]

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        out = np.full(len(xyz), self.h_out_m)
        step = max(1000, 20_000_000 // len(self.stations))              # the cap test matrix stays <= 2e7 entries
        for a in range(0, len(xyz), step):
            x = xyz[a:a + step]
            pi, ki = np.nonzero(x @ self.centres.T >= self.cos_cap[None, :])
            o = np.argsort(ki, kind="stable")
            pi, ki = pi[o], ki[o]
            cuts = np.flatnonzero(np.diff(np.r_[-1, ki, len(self.stations)]))
            for c0, c1 in zip(cuts[:-1], cuts[1:]):
                k, pts = ki[c0], pi[c0:c1]
                d, j = self.trees[k].query(x[pts])
                v = self.hz[k][j] + self.grad * np.maximum(_arc(d) * R_EARTH_M - self.half[k][j], 0.0)
                np.minimum.at(out, a + pts, v)
        return out


def auto_straits(loops: list, h: SizeField, n_min: int, g: float = GRADING, h_min_m: float = 0.0,
                 h_out_m: float = np.inf) -> tuple[PassageField | None, dict]:
    """The automatic strait zones of a coast (S22, S24): the medial axis at lambda = LAM min(h, n_min h_min) (a
    passage S24 keeps open, W >= n_min h_min, is never pruned), the passages measured against h, h = W / n on the
    open ones (min W / n >= h_min) where it is finer than h. Returns (field or None, record)."""
    def lam(x):
        hv = h(x)
        return LAM * (np.minimum(hv, n_min * h_min_m) if h_min_m > 0 else hv)

    def step(x):
        return np.minimum(STEP * h(x), 0.5 * lam(x))

    ma = medial_axis(loops, step, lam)
    ps = passages(ma, h, n_min, W_MAX, g)
    st, hz, half, recs = [], [], [], []
    n_open = n_closed = 0
    for p in ps:
        z = p.W / p.n
        rec = p.record(z)
        rec["open"] = bool(z.min() >= h_min_m)
        rec["refines"] = bool(np.any(z < h(p.points)))
        if rec["open"]:
            n_open += 1
            if rec["refines"]:
                st.append(p.points)
                hz.append(z)
                half.append(0.5 * p.W)
        else:
            n_closed += 1
        recs.append(rec)
    info = {"axis": ma.info, "n_passages": len(ps), "n_open": n_open, "n_closed": n_closed, "n_zones": len(st),
            "h_zone_min_m": float(min(z.min() for z in hz)) if hz else None, "passages": recs}
    fld = PassageField(st, hz, half, h_out_m, g) if st else None
    return fld, info
