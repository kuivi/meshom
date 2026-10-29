"""Domains for the fill methods: the closed sphere or the sphere minus land, bounded by closed coast loops.

Conventions (binding for all methods):
- A loop is a (K,3) array of unit vectors, a closed spherical polyline: vertex K-1 joins back to vertex 0, the first
  vertex is not repeated. Consecutive vertices are joined by great-circle arcs.
- Water is on the LEFT of every loop, seen from outside the sphere (a land piece is walked clockwise).
- The water region is connected: it is the intersection of the left sides of all loops (land pieces and islands
  as holes, or one outer basin loop with islands inside).
- Lengths in metres (R_EARTH_M); a size field h gives the target edge length in metres.
"""
from dataclasses import dataclass, field

import numpy as np

from meshom.geo import R_EARTH_M, arc_length, lonlat_to_xyz, tangent_basis
from meshom.size import SizeField

# continent case geometry: a 30° cap at 20°E 0°N cut by a 150 km channel, an island of 300 km
# radius at 60°E 10°N. The channel runs through the cap centre at azimuth 45° (not along lat-lon lines).
CONTINENT_LONLAT = (20.0, 0.0)
CONTINENT_RADIUS_DEG = 30.0
CHANNEL_WIDTH_M = 150e3
CHANNEL_AZIMUTH_DEG = 45.0
ISLAND_LONLAT = (60.0, 10.0)
ISLAND_RADIUS_M = 300e3
LOOP_SPACING_M = 5e3          # vertex spacing of the analytic loops (resample_loops works at any h >= 15 km)


@dataclass
class Domain:
    loops: list[np.ndarray] = field(default_factory=list)   # closed boundary polylines (K,3), water on the left
    name: str = "sphere"
    exact: list[bool] = field(default_factory=list)          # exact[i]: resample_loops keeps loop i as given (its
                                                             # vertices are mesh nodes, e.g. a block outline); missing
                                                             # entries = False


def closed_sphere() -> Domain:
    return Domain([], "sphere")


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def continent_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (c, t, n): cap centre c, channel direction t at c (azimuth from north), n = c x t.

    The channel centreline is the great circle through c along t; channel water is |x . n| < sin(W / 2R).
    """
    lon, lat = CONTINENT_LONLAT
    c = lonlat_to_xyz(np.array([lon]), np.array([lat]))[0]
    lo, la = np.radians(lon), np.radians(lat)
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    az = np.radians(CHANNEL_AZIMUTH_DEG)
    t = np.cos(az) * north + np.sin(az) * east
    return c, t, np.cross(c, t)


def _samples(arc_len_m: float) -> int:
    return max(2, int(np.ceil(arc_len_m / LOOP_SPACING_M)))


def _half_cap(c: np.ndarray, t: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Boundary of {x . c >= cos rho, x . n >= s}: the channel edge arc, then the cap arc (orientation fixed later)."""
    s = np.sin(0.5 * CHANNEL_WIDTH_M / R_EARTH_M)
    rho = np.radians(CONTINENT_RADIUS_DEG)
    th_m = np.arccos(np.cos(rho) / np.sqrt(1.0 - s * s))
    ph_m = np.arccos(s / np.sin(rho))
    k1 = _samples(R_EARTH_M * np.sqrt(1.0 - s * s) * 2.0 * th_m)
    k2 = _samples(R_EARTH_M * np.sin(rho) * 2.0 * ph_m)
    th = np.linspace(-th_m, th_m, k1 + 1)[:-1]
    ph = np.linspace(ph_m, -ph_m, k2 + 1)[:-1]
    edge = s * n + np.sqrt(1.0 - s * s) * (np.cos(th)[:, None] * c + np.sin(th)[:, None] * t)
    cap = np.cos(rho) * c + np.sin(rho) * (np.cos(ph)[:, None] * n + np.sin(ph)[:, None] * t)
    return _unit(np.vstack([edge, cap]))


def _cap_loop(center: np.ndarray, radius_m: float) -> np.ndarray:
    rho = radius_m / R_EARTH_M
    e1, e2 = tangent_basis(center[None, :])
    k = _samples(2.0 * np.pi * R_EARTH_M * np.sin(rho))
    ph = np.linspace(0.0, 2.0 * np.pi, k + 1)[:-1]
    return _unit(np.cos(rho) * center + np.sin(rho) * (np.cos(ph)[:, None] * e1[0] + np.sin(ph)[:, None] * e2[0]))


def _land_on_right(loop: np.ndarray) -> np.ndarray:
    """Reverse a loop around a small land piece if it runs CCW (land on the left)."""
    return loop[::-1].copy() if loop_signed_area(loop) > 0 else loop


def continent_case() -> Domain:
    """Sphere minus a 30° continent cap (cut in two by a 150 km channel through its centre) and a 300 km island.

    Three loops, each walked with water on the left; vertex spacing LOOP_SPACING_M; the four channel mouths are
    ~90° corners kept by resample_loops.
    """
    c, t, n = continent_frame()
    island = lonlat_to_xyz(np.array([ISLAND_LONLAT[0]]), np.array([ISLAND_LONLAT[1]]))[0]
    loops = [_land_on_right(_half_cap(c, t, n)), _land_on_right(_half_cap(c, -t, -n)),
             _land_on_right(_cap_loop(island, ISLAND_RADIUS_M))]
    return Domain(loops, "continent")


def loop_signed_area(loop: np.ndarray) -> float:
    """Signed spherical area (m^2) of the fan of the loop about its normalised vertex mean.

    > 0 when the loop runs CCW (seen from outside) around its mean, i.e. the small side is on its left. A loop
    around land with water on the left is therefore < 0. Valid for loops that fit in a hemisphere.
    """
    m = _unit(loop.mean(axis=0))
    a, b = loop, np.roll(loop, -1, axis=0)
    det = np.einsum("ij,ij->i", np.cross(a, b), np.broadcast_to(m, a.shape))
    den = 1.0 + np.einsum("ij,ij->i", a, b) + a @ m + b @ m
    return float(2.0 * np.sum(np.arctan2(det, den)) * R_EARTH_M**2)


def loop_edges(boundary: list[np.ndarray]) -> np.ndarray:
    """(B,2) int64 directed edges (water on the left) into np.vstack(boundary), loop by loop."""
    out, off = [], 0
    for lp in boundary:
        i = np.arange(len(lp), dtype=np.int64) + off
        out.append(np.column_stack([i, np.roll(i, -1)]))
        off += len(lp)
    return np.vstack(out) if out else np.zeros((0, 2), dtype=np.int64)


def in_water(domain: Domain, xyz: np.ndarray) -> np.ndarray:
    """(N,) bool: the point is in the water of the domain.

    Each loop is mapped by the stereographic projection from the antipode of its vertex mean; the planar polygon's
    orientation tells whether water is its inside (a *basin*: a closed sea such as the Mediterranean of a cut mesh)
    or its outside (a *hole*: a continent or an island). Water = the main region (the water side of every top-level
    hole; it exists only when some hole is not nested in a basin, or when there are no loops) plus every basin minus
    the holes nested in it. Points on a loop: undefined side. (Before 2026-10-22 water was the AND over all loops,
    which is empty as soon as two disjoint basins exist: the 150 km global mesh with the Mediterranean kept.)
    """
    from matplotlib.path import Path

    n = len(xyz)
    if not domain.loops:
        return np.ones(n, dtype=bool)
    insides, ccws, polys = [], [], []
    for lp in domain.loops:
        m = _unit(lp.mean(axis=0))
        e1, e2 = tangent_basis(m[None, :])
        e1, e2 = e1[0], e2[0]

        def proj(p, m=m, e1=e1, e2=e2):
            d = np.maximum(1.0 + p @ m, 1e-12)
            return np.column_stack([p @ e1 / d, p @ e2 / d])

        poly = proj(lp)
        x, y = poly[:, 0], poly[:, 1]
        ccw = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0     # water = inside the planar polygon: a basin
        near = xyz @ m >= (lp @ m).min() - 1e-12                       # outside this cap = outside the polygon
        inside = np.zeros(n, dtype=bool)
        path = Path(poly, closed=False)
        if near.any():
            inside[near] = path.contains_points(proj(xyz[near]))
        insides.append(inside); ccws.append(ccw); polys.append((path, proj, m, lp))
    basins = [k for k, c in enumerate(ccws) if c]
    holes = [k for k, c in enumerate(ccws) if not c]
    parent = {}                                                       # hole -> the basin it lies in, if any
    for hk in holes:
        v = domain.loops[hk][:1]
        for b in basins:
            path, proj, m, lp = polys[b]
            if (v @ m >= (lp @ m).min() - 1e-12) and path.contains_points(proj(v))[0]:
                parent[hk] = b
                break
    top = [hk for hk in holes if hk not in parent]
    water = np.zeros(n, dtype=bool)
    if top or not basins:                                              # the main region exists
        water[:] = True
        for hk in top:
            water &= ~insides[hk]
    for b in basins:
        w = insides[b].copy()
        for hk in holes:
            if parent.get(hk) == b:
                w &= ~insides[hk]
        water |= w
    return water


def _turn_angles(lp: np.ndarray) -> np.ndarray:
    """(K,) turning angle (rad, unsigned) at each vertex of a closed loop."""
    prv, nxt = np.roll(lp, 1, axis=0), np.roll(lp, -1, axis=0)
    t_out = _unit(nxt - np.einsum("ij,ij->i", nxt, lp)[:, None] * lp)
    t_in = _unit(lp - prv - np.einsum("ij,ij->i", lp - prv, lp)[:, None] * lp)    # arriving direction at lp
    return np.arccos(np.clip(np.einsum("ij,ij->i", t_in, t_out), -1.0, 1.0))


def _place_jf(w: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Segment j and fraction f of the points at cumulative weights u along a polyline with segment weights w (k,)."""
    cum = np.r_[0.0, np.cumsum(w)]
    j = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(w) - 1)
    return j, np.clip((u - cum[j]) / np.where(w[j] > 0, w[j], 1.0), 0.0, 1.0)


def _place(lp: np.ndarray, w: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Points at cumulative weights u along the open polyline lp (k+1 vertices) with segment weights w (k,)."""
    j, f = _place_jf(w, u)
    f = f[:, None]
    return _unit((1.0 - f) * lp[j] + f * lp[j + 1])


def _at(lp: np.ndarray, j: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Points at (vertex j, fraction f towards vertex j + 1) of the closed loop lp."""
    f = f[:, None]
    return _unit((1.0 - f) * lp[j] + f * lp[(j + 1) % len(lp)])


def crossing_chords(pts: np.ndarray, ia: np.ndarray, ib: np.ndarray, test: np.ndarray | None = None) -> np.ndarray:
    """(C,) bool: chord pts[ia] -> pts[ib] strictly crosses another chord (great-circle arcs; a shared node id or a
    touch is no crossing). test: the chords to look from (default all; a crossing between two others is missed)."""
    from scipy.spatial import cKDTree

    a, b = pts[ia], pts[ib]
    mid = 0.5 * (a + b)
    r = 0.5 * np.linalg.norm(a - b, axis=1) * 1.01 + 1e-12          # the arc lies in this ball around mid
    test = np.arange(len(ia)) if test is None else np.asarray(test, dtype=np.int64)
    hit = np.zeros(len(ia), dtype=bool)
    if len(ia) < 2 or len(test) == 0:
        return hit
    cls = np.floor(np.log2(r / r.min())).astype(np.int64)           # length classes: a ball query per class
    ii, jj = [], []
    for c in np.unique(cls):
        sel = np.flatnonzero(cls == c)
        lists = cKDTree(mid[sel]).query_ball_point(mid[test], r[test] + r[sel].max())
        cnt = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
        if cnt.sum():
            ii.append(np.repeat(test, cnt))
            jj.append(sel[np.fromiter((k for x in lists for k in x), dtype=np.int64, count=int(cnt.sum()))])
    if not ii:
        return hit
    i, j = np.concatenate(ii), np.concatenate(jj)
    k = (ia[i] != ia[j]) & (ia[i] != ib[j]) & (ib[i] != ia[j]) & (ib[i] != ib[j])
    i, j = i[k], j[k]
    nab, ncd = np.cross(a[i], b[i]), np.cross(a[j], b[j])
    s1 = np.einsum("ij,ij->i", nab, a[j]) * np.einsum("ij,ij->i", nab, b[j])
    s2 = np.einsum("ij,ij->i", ncd, a[i]) * np.einsum("ij,ij->i", ncd, b[i])
    x = (s1 < 0) & (s2 < 0)
    hit[i[x]] = True
    hit[j[x]] = True
    return hit


def _uncross(src: list, js: list, fs: list, out: list) -> None:
    """Refine resampled loops in place until no two chords cross (decisions C38). src[i]: the loop resampled into
    out[i] (None: exact, never changed), js[i] / fs[i]: out[i]'s points as (vertex, fraction) along src[i]. A crossing
    chord gets the middle source vertex it skips; a chord within one source edge is never split (two of those do not
    cross on a simple coast), so the refinement ends at the source loops at the latest."""
    if all(s is None for s in src):                                     # no loop, or only exact ones
        return
    test = None
    while True:
        size = np.array([len(x) for x in out], dtype=np.int64)
        off = np.r_[0, np.cumsum(size)]
        ia = np.arange(off[-1], dtype=np.int64)
        ib = ia + 1
        ib[off[1:] - 1] = off[:-1]
        hit = crossing_chords(np.vstack(out), ia, ib, test)
        fresh, moved = [], False
        for li in range(len(out)):
            h_l = hit[off[li]:off[li + 1]]
            if src[li] is None or not h_l.any():
                fresh.append(np.zeros(size[li], dtype=bool))
                continue
            k = len(src[li])
            u = js[li] + fs[li]
            u = u + k * np.r_[0, np.cumsum(np.diff(u) < 0)]                # unwrapped along the loop
            nxt = np.r_[u[1:], u[0] + k]
            lo, hi = np.floor(u).astype(np.int64) + 1, np.ceil(nxt).astype(np.int64) - 1
            split = np.flatnonzero(h_l & (lo <= hi))
            if len(split) == 0:
                fresh.append(np.zeros(size[li], dtype=bool))
                continue
            moved = True
            v = (lo[split] + hi[split]) // 2 % k
            pos = np.r_[np.arange(size[li]), split + 0.5]
            o = np.argsort(pos, kind="stable")
            js[li] = np.r_[js[li], v][o]
            fs[li] = np.r_[fs[li], np.zeros(len(split))][o]
            out[li] = _at(src[li], js[li], fs[li])
            inv = np.empty_like(o)
            inv[o] = np.arange(len(o))
            fr = np.zeros(len(o), dtype=bool)
            fr[inv[np.r_[split, np.arange(size[li], len(o))]]] = True       # both halves of each split chord
            fresh.append(fr)
        if not moved:
            return
        test = np.flatnonzero(np.concatenate(fresh))


def resample_loops(domain: Domain, h: SizeField, corner_deg: float = 30.0) -> list[np.ndarray]:
    """Each loop resampled with spacing h along the curve (equal steps of the integral of ds/h).

    Vertices turning by more than corner_deg are corners and are kept; between corners the step count is
    round(integral ds/h), at least 1, and a loop gets at least 3 points (the step counts with the longest steps grow).
    Chords that cross another loop's or their own are refined at the source vertices (_uncross, decisions C38).
    Orientation is kept. A loop marked exact (domain.exact) is returned as given.
    """
    out, src, js, fs = resample_jf(domain, h, corner_deg)
    _uncross(src, js, fs, out)
    return out


def resample_jf(domain: Domain, h: SizeField, corner_deg: float = 30.0) -> tuple[list, list, list, list]:
    """resample_loops before _uncross: (points, source loops, vertex, fraction) per loop (None for an exact loop)."""
    out, src, js, fs = [], [], [], []
    for li, lp in enumerate(domain.loops):
        if li < len(domain.exact) and domain.exact[li]:
            out.append(lp)
            src.append(None)
            js.append(None)
            fs.append(None)
            continue
        hv = h(lp)
        nxt = np.roll(lp, -1, axis=0)
        seg = R_EARTH_M * arc_length(lp, nxt)
        w = seg * 0.5 * (1.0 / hv + 1.0 / np.roll(hv, -1))           # integral ds/h per segment (trapezoid)
        corners = np.flatnonzero(_turn_angles(lp) > np.radians(corner_deg))
        k = len(lp)
        if len(corners) == 0:
            n = max(3, int(round(w.sum())))
            j, f = _place_jf(w, np.arange(n) * (w.sum() / n))
        else:
            jl, fl = [], []
            ends = list(zip(corners, np.r_[corners[1:], corners[0] + k]))
            wp = np.array([w[np.arange(a, b) % k].sum() for a, b in ends])
            npc = np.maximum(1, np.rint(wp).astype(np.int64))
            while npc.sum() < 3:                                        # a loop gets >= 3 points (two corners of a
                npc[np.argmax(wp / npc)] += 1                           # thin island gave a 2-point loop: no area)
            for (a, b), n in zip(ends, npc):
                idx = np.arange(a, b + 1) % k
                ww = w[idx[:-1]]
                j, f = _place_jf(ww, np.arange(n) * (ww.sum() / n))
                jl.append(idx[j])
                fl.append(f)
            j, f = np.concatenate(jl), np.concatenate(fl)
        out.append(_at(lp, j, f))
        src.append(lp)
        js.append(j)
        fs.append(f)
    return out, src, js, fs
