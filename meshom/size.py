"""Size fields: the analytic fields of the method competition, the config layers, and the gradient limiter.

A size field maps (N,3) unit vectors to (N,) target edge lengths in metres (h = target edge length).

- Competition fields (Uniform, Disc, Band): piecewise linear in a geodesic distance, so |grad h| is exactly the
  stated g on their ramps and 0 elsewhere; their kinks put delta lines into Laplace(ln h) (compete.rho_c_integrals).
- Config layers (`[[size.layers]]`, meshom.config): Uniform (kind "constant"), Latitude, CoastDistance, Zone (mode
  "min"), SetZone (mode "set"), SmoothZone (mode "smooth", bound to the running h by `smoothed`); the run combines
  them in file order with `layered` (Minimum when every layer is a min layer, else Layered: decisions S34), clips them
  with Clipped ([size] h_min_m / h_max_m) and limits the result with `limit_gradient` ([size] grad_max). Distances
  are geodesic, in metres on the sphere of radius R_EARTH_M.
- `limit_gradient`: the largest field <= h with |grad h| <= g along the edges of a cubed-sphere grid
  (Dijkstra-style relaxation h_j <= h_i + g d_ij), returned as a bilinear
  interpolant on that grid (`Limited`). Its slope over two cells (h_floor, the shortest mesh edge) is <= g plus the
  stencil's metrication; within one cell of a kink of h the bilinear cells reach up to 2 g (decisions S11).
"""
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M, arc_length, lonlat_to_xyz, tangent_basis


class SizeField(Protocol):
    def __call__(self, xyz: np.ndarray) -> np.ndarray: ...   # (N,3) unit vectors -> (N,) metres


@dataclass(frozen=True)
class Uniform:
    h_m: float

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        return np.full(len(xyz), float(self.h_m))


@dataclass(frozen=True)
class Disc:
    """h_in within geodesic distance r_in of the centre, linear in the distance to h_out at r_out, h_out beyond."""
    center: tuple[float, float, float]
    r_in_m: float
    h_in_m: float
    r_out_m: float
    h_out_m: float

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        r = R_EARTH_M * arc_length(xyz, np.asarray(self.center)[None, :])
        t = np.clip((r - self.r_in_m) / (self.r_out_m - self.r_in_m), 0.0, 1.0)
        return self.h_in_m + t * (self.h_out_m - self.h_in_m)


@dataclass(frozen=True)
class Band:
    """h_south south of the band, h_north north of it, linear in the meridional distance across a band centred on
    lat_c of width |h_north - h_south| / g, so |grad h| = g inside it."""
    lat_c: float
    h_south_m: float
    h_north_m: float
    g: float

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        lat = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))
        y = R_EARTH_M * (lat - np.radians(self.lat_c))                 # metres north of lat_c along a meridian
        lo, hi = sorted((self.h_south_m, self.h_north_m))
        s = 1.0 if self.h_north_m >= self.h_south_m else -1.0
        return np.clip(0.5 * (self.h_south_m + self.h_north_m) + s * self.g * y, lo, hi)


@dataclass(frozen=True)
class Scaled:
    h: SizeField
    s: float

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        return self.s * self.h(xyz)


@dataclass(frozen=True)
class Minimum:
    """Pointwise minimum of several fields."""
    fields: tuple

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        return np.min([f(xyz) for f in self.fields], axis=0)


def uniform(h_m: float) -> SizeField:
    return Uniform(float(h_m))


def disc(lon0: float, lat0: float, r_in_m: float, h_in_m: float, r_out_m: float, h_out_m: float) -> SizeField:
    """Refinement disc centred at (lon0, lat0) degrees; |grad h| = (h_out - h_in) / (r_out - r_in) on the ramp."""
    c = lonlat_to_xyz(np.array([lon0]), np.array([lat0]))[0]
    return Disc(tuple(float(v) for v in c), float(r_in_m), float(h_in_m), float(r_out_m), float(h_out_m))


def band(lat_c: float, h_south_m: float, h_north_m: float, g: float) -> SizeField:
    return Band(float(lat_c), float(h_south_m), float(h_north_m), float(g))


def scaled(h: SizeField, s: float) -> SizeField:
    """h * s: the same case at a coarser (s > 1) or finer (s < 1) scale; rho_c does not change."""
    return Scaled(h, float(s))


# ------------------------------------------------------------------------------------------- config layers

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _chord_to_m(chord: np.ndarray) -> np.ndarray:
    return 2.0 * R_EARTH_M * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))


def _m_to_chord(d_m: float) -> float:
    return 2.0 * np.sin(0.5 * min(d_m / R_EARTH_M, np.pi)) if np.isfinite(d_m) else np.inf


def arc_distance_m(xyz: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(N,) geodesic distance (m) of unit points to the great-circle arc p -> q (unit vectors, arc < 180°)."""
    n = _unit(np.cross(p, q))
    z = xyz @ n
    w = xyz - z[:, None] * n
    between = (np.cross(p, w) @ n >= 0) & (np.cross(w, q) @ n >= 0)
    d_line = np.arcsin(np.clip(np.abs(z), 0.0, 1.0))
    d_end = np.minimum(arc_length(xyz, p[None, :]), arc_length(xyz, q[None, :]))
    return R_EARTH_M * np.where(between, d_line, d_end)


def polygon_centre(poly: np.ndarray) -> np.ndarray:
    """(3,) a centre m of an open hemisphere holding the polygon poly (K,3): every vertex has poly @ m > 1e-9 (the
    polygon fits in a hemisphere: decisions S19). The normalised vertex mean when it is one (every zone of a usual
    shape: unchanged), else the centre of the smallest cap holding the vertices (a long thin zone, whose far vertex
    can lie beyond 90 degrees of the mean): m = x / |x| for the shortest x with poly @ x >= 1, a least-distance
    programme solved exactly by one NNLS (Lawson & Hanson 1974, ch. 23); min poly @ m = 1 / |x|."""
    from scipy.optimize import nnls

    m = _unit(poly.mean(axis=0))
    if (poly @ m).min() > 1e-9:
        return m
    E, f = np.vstack([poly.T, np.ones(len(poly))]), np.array([0.0, 0.0, 0.0, 1.0])
    r = E @ nnls(E, f)[0] - f
    return _unit(-r[:3] / r[3])


def polygon_inside(xyz: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """(N,) bool: inside the spherical polygon poly (K,3) with great-circle edges (either orientation).

    Exact in the gnomonic plane about m = polygon_centre(poly), where great circles are straight lines; the polygon
    must lie in an open hemisphere (a zone, not a continent: S19). Points in the far hemisphere of m are outside.
    """
    from matplotlib.path import Path as MplPath

    m = polygon_centre(poly)
    e1, e2 = (b[0] for b in tangent_basis(m[None, :]))
    front = xyz @ m > 1e-9
    inside = np.zeros(len(xyz), dtype=bool)
    if front.any():
        g = xyz[front] / (xyz[front] @ m)[:, None]
        pv = poly / (poly @ m)[:, None]
        inside[front] = MplPath(np.column_stack([pv @ e1, pv @ e2])).contains_points(np.column_stack([g @ e1, g @ e2]))
    return inside


def polygon_distance_m(xyz: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """(N,) 0 inside the spherical polygon poly (K,3), else the geodesic distance (m) to its nearest edge."""
    d = np.full(len(xyz), np.inf)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        d = np.minimum(d, arc_distance_m(xyz, a, b))
    return np.where(polygon_inside(xyz, poly), 0.0, d)


@dataclass(frozen=True)
class Latitude:
    """h = max(h_m cos(lat), h_min_m): h_m at the equator, shrinking with the meridian convergence (kind "latitude")."""
    h_m: float
    h_min_m: float

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        return np.maximum(self.h_m * np.hypot(xyz[:, 0], xyz[:, 1]), self.h_min_m)      # |xy| = cos(lat)


@dataclass(frozen=True, eq=False)
class CoastDistance:
    """h = min(h_m + grad max(d - plateau_m, 0), h_cap_m), d = geodesic distance to the nearest coast point
    (kind "coast_distance").

    The coast is a point set (KD-tree of 3-D unit vectors, built once per domain: `coast_distance`); points should be
    spaced well below h_m (meshom.coast.Coast.points densifies the loops). The search stops at the distance where h
    reaches h_cap_m (the layer cannot go above it), as in the UI preview; h_cap_m = inf searches without bound.
    """
    tree: cKDTree
    h_m: float
    plateau_m: float
    grad: float
    h_cap_m: float = np.inf

    def distance_m(self, xyz: np.ndarray, d_max_m: float = np.inf) -> np.ndarray:
        """(N,) geodesic distance (m) to the nearest coast point; points beyond d_max_m get pi R."""
        chord, _ = self.tree.query(xyz, distance_upper_bound=_m_to_chord(d_max_m), workers=-1)
        return _chord_to_m(chord)

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        d_max = self.plateau_m + (self.h_cap_m - self.h_m) / self.grad
        d = self.distance_m(xyz, d_max)
        return np.minimum(self.h_m + self.grad * np.maximum(d - self.plateau_m, 0.0), self.h_cap_m)


def coast_distance(points: np.ndarray, h_m: float, plateau_m: float, grad: float,
                   h_cap_m: float = np.inf) -> CoastDistance:
    """CoastDistance over coast points (P,3) unit vectors (the KD-tree is built here, once)."""
    return CoastDistance(cKDTree(points), float(h_m), float(plateau_m), float(grad), float(h_cap_m))


def _zone_near(polygon: np.ndarray, reach_m: float, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(idx, d): the points that can lie within reach_m of the polygon (the cap around polygon_centre that holds the
    polygon, widened by reach_m) and their geodesic distance to it (m, 0 inside)."""
    m = polygon_centre(polygon)
    reach = np.arccos(np.clip((polygon @ m).min(), -1.0, 1.0)) + reach_m / R_EARTH_M
    near = np.flatnonzero(xyz @ m >= np.cos(min(reach, np.pi)))
    return near, (polygon_distance_m(xyz[near], polygon) if len(near) else np.zeros(0))


def _ramp(d: np.ndarray, transition_m: float) -> np.ndarray:
    """0 in the polygon (d = 0), linear in d to 1 at transition_m, 1 beyond; transition_m = 0: a step."""
    return np.clip(d / transition_m, 0.0, 1.0) if transition_m > 0 else (d > 0).astype(np.float64)


@dataclass(frozen=True, eq=False)
class Zone:
    """h_m inside a spherical polygon (great-circle edges), linear in the geodesic distance to it up to h_out_m at
    transition_m, h_out_m beyond (kind "zone", mode "min"). transition_m = 0 is a step (the limiter grades it)."""
    polygon: np.ndarray           # (K,3) unit vectors, either orientation, inside an open hemisphere
    h_m: float
    transition_m: float
    h_out_m: float

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        near, d = _zone_near(self.polygon, self.transition_m, xyz)      # only points the zone can reach
        t = np.ones(len(xyz))
        if len(near):
            t[near] = _ramp(d, self.transition_m)
        return self.h_m + (self.h_out_m - self.h_m) * t


# Zone modes "set" and "smooth" act on the running h of the layers before them (Layered), so they are steps
# (`apply(xyz, h) -> h`), not fields of their own.

@dataclass(frozen=True, eq=False)
class SetZone:
    """Zone mode "set": h_m inside the polygon whatever the running h, linear in the geodesic distance from h_m to
    the running h over transition_m (0: a step), the running h beyond."""
    polygon: np.ndarray
    h_m: float
    transition_m: float

    def apply(self, xyz: np.ndarray, h: np.ndarray) -> np.ndarray:
        near, d = _zone_near(self.polygon, self.transition_m, xyz)
        out = np.array(h, dtype=np.float64)
        if len(near):
            t = _ramp(d, self.transition_m)
            with np.errstate(invalid="ignore"):                       # t = 0 with h = inf: the where takes h_m
                out[near] = np.where(t > 0, self.h_m + t * (out[near] - self.h_m), self.h_m)
        return out


BLEND_SAMPLE_M = 1000.0         # S37: the blend's polygon edges sampled at most this far apart (distance error <= 0.5 km)
BLEND_EDGE_SMOOTH_M = 50_000.0  # S37: the running h along the outer edge, ln h averaged over +- this along the edge


@dataclass(frozen=True, eq=False)
class BlendZone:
    """Zone mode "blend" (S37): h_m inside the polygon, the running h outside the outer polygon, and between them
    ln h = (1 - t) ln h_m + t ln h_e with t = d_in / (d_in + d_out) (d_in, d_out: the geodesic distances to the
    polygon and to the outer polygon's edge) and h_e the running h at the nearest point of the outer edge (bound by
    `layered`: the running h at the edge samples, ln h averaged over +- BLEND_EDGE_SMOOTH_M along the edge), so h
    rises from h_m to the running h of the edge over whatever distance lies between the two edges, gently where they
    are far apart, whatever the running h does between them. Distances by KD-trees on the edges sampled every
    BLEND_SAMPLE_M; both polygons in an open hemisphere, the polygon inside the outer one.

    With `h_factor` (B110) in place of h_m: h_factor times the running h inside the polygon, and between the two
    edges ln h = ln h_run + (1 - t) ln h_factor (the running h at the point itself), so the running h is scaled down
    by h_factor over the band (an ORCA grid refined k times inside: h_factor = 1 / k)."""
    polygon: np.ndarray
    outer: np.ndarray
    h_m: float
    tree_in: cKDTree
    tree_out: cKDTree
    edge: np.ndarray                  # (S,3) the outer edge's samples, in order along it
    h_edge: np.ndarray | None = None  # (S,) the bound running h there; None = not bound (the running h at the point)
    h_factor: float | None = None     # B110: h_factor x the running h inside (h_m not used), scaled over the band

    def bind(self, h: SizeField) -> "BlendZone":
        if self.h_factor is not None:                                 # B110: the running h at the point, no edge
            return replace(self, h_edge=np.zeros(0))
        lnh = np.log(evaluate(h, self.edge))
        w = max(1, int(round(BLEND_EDGE_SMOOTH_M / BLEND_SAMPLE_M)))
        k = np.ones(2 * w + 1) / (2 * w + 1)
        sm = np.convolve(np.r_[lnh[-w:], lnh, lnh[:w]], k, mode="valid")      # the edge is a closed loop
        return replace(self, h_edge=np.exp(sm))

    def apply(self, xyz: np.ndarray, h: np.ndarray) -> np.ndarray:
        out = np.array(h, dtype=np.float64)
        m = polygon_centre(self.outer)
        near = np.flatnonzero(xyz @ m >= (self.outer @ m).min() - 1e-9)
        if not len(near):
            return out
        sel = near[polygon_inside(xyz[near], self.outer)]
        if not len(sel):
            return out
        inner = polygon_inside(xyz[sel], self.polygon)
        mid = sel[~inner]
        if self.h_factor is not None:                                 # B110: the running h scaled
            out[sel[inner]] *= self.h_factor
            if len(mid):
                d_in = _chord_to_m(self.tree_in.query(xyz[mid], workers=-1)[0])
                d_out = _chord_to_m(self.tree_out.query(xyz[mid], workers=-1)[0])
                t = d_in / np.maximum(d_in + d_out, 1e-9)
                out[mid] *= np.exp((1.0 - t) * np.log(self.h_factor))
            return out
        out[sel[inner]] = self.h_m
        if len(mid):
            d_in = _chord_to_m(self.tree_in.query(xyz[mid], workers=-1)[0])
            c_out, j = self.tree_out.query(xyz[mid], workers=-1)
            d_out = _chord_to_m(c_out)
            he = out[mid] if self.h_edge is None else self.h_edge[j]
            t = d_in / np.maximum(d_in + d_out, 1e-9)
            out[mid] = np.exp((1.0 - t) * np.log(self.h_m) + t * np.log(he))
        return out


def _edge_samples(poly: np.ndarray, step_m: float) -> np.ndarray:
    """Points along the great-circle edges of poly (K,3) in order, at most step_m apart, the vertices included."""
    out = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ang = float(arc_length(a[None], b[None])[0])
        if ang < 1e-12:
            continue
        n = max(1, int(np.ceil(ang * R_EARTH_M / step_m)))
        s = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        out.append((np.sin((1 - s) * ang) * a + np.sin(s * ang) * b) / np.sin(ang))
    return _unit(np.vstack(out))


def blend_zone(polygon: np.ndarray, outer: np.ndarray, h_m: float | None, h_factor: float | None = None) -> BlendZone:
    """BlendZone of two polygons (K,3) unit vectors (unbound: `layered` binds it to the running h before it); h_factor
    (B110) in place of h_m: the running h scaled by it inside."""
    edge = _edge_samples(outer, BLEND_SAMPLE_M)
    return BlendZone(polygon, outer, float(h_m) if h_m is not None else float("nan"),
                     cKDTree(_edge_samples(polygon, BLEND_SAMPLE_M)), cKDTree(edge), edge,
                     h_factor=None if h_factor is None else float(h_factor))


@dataclass(frozen=True, eq=False)
class SmoothZone:
    """Zone mode "smooth" as given (unbound): inside the polygon the running h is replaced by its smoothed value
    (ln h averaged over a geodesic disc of radius smooth_m), blended linearly in the distance to the unsmoothed h
    over transition_m. `layered` binds it to the running h before it (`smoothed`)."""
    polygon: np.ndarray
    smooth_m: float
    transition_m: float


SMOOTH_CELLS = 12                # the smoothed h is held on a grid of spacing smooth_m / SMOOTH_CELLS (S36)
SMOOTH_MAX_K = 20_000            # most source nodes per disc (the source grid coarsens to stay below it; S36)


def _cube_area_density(cc: np.ndarray, n: int) -> np.ndarray:
    """(U,) area per unit of equiangular coordinates du dv at lattice nodes cc (U,3) of cube_grid(n):
    (1 + x^2)(1 + y^2) / (1 + x^2 + y^2)^1.5, x, y = tan of the node's two face coordinates (same on every face of
    a seam node)."""
    c = np.asarray(cc, dtype=np.float64)
    k = np.argmax(np.abs(c), axis=1)
    r = np.arange(len(c))
    x = np.tan(0.25 * np.pi * c[r, (k + 1) % 3] / n)
    y = np.tan(0.25 * np.pi * c[r, (k + 2) % 3] / n)
    return (1.0 + x * x) * (1.0 + y * y) / (1.0 + x * x + y * y) ** 1.5


@dataclass(frozen=True, eq=False)
class Smoothed:
    """A SmoothZone bound to the running h before it: s = the smoothed h on a cubed-sphere grid (Limited, the nodes
    within transition_m + 2 cells of the polygon; NaN elsewhere, never read), applied as (1 - w) h + w s with
    w = 1 - _ramp(d, transition_m)."""
    zone: SmoothZone
    s: "Limited"                  # defined with the limiter below

    def apply(self, xyz: np.ndarray, h: np.ndarray) -> np.ndarray:
        near, d = _zone_near(self.zone.polygon, self.zone.transition_m, xyz)
        out = np.array(h, dtype=np.float64)
        if len(near):
            w = 1.0 - _ramp(d, self.zone.transition_m)
            sel = w > 0
            idx, w = near[sel], w[sel]
            if len(idx):
                out[idx] = (1.0 - w) * out[idx] + w * self.s(xyz[idx])
        return out


def smoothed(zone: SmoothZone, h: SizeField, spacing_m: float | None = None,
             max_nodes: int | None = None) -> Smoothed:
    """Bind a SmoothZone to the running h: s_i = exp(sum_j w_ij ln h_j / sum_j w_ij) at the target nodes i, over the
    source nodes j within smooth_m of i, w_ij = (1 - (r_ij / smooth_m)^2)^2 (r geodesic; the biweight: no jump when
    a node enters the disc) x the node's area (_cube_area_density). Deterministic.

    Grids (equiangular cubed spheres, cube_grid; S36): targets at the nominal spacing smooth_m / SMOOTH_CELLS (s is
    smooth on that scale; bilinear between them), only the nodes within transition_m + 2 cells of the polygon;
    sources at spacing_m (the run: half the smallest h of the layers, as the limiter, so steps of h narrower than
    smooth_m are integrated, not aliased; default: the target spacing), capped at max_nodes (default
    LIMITER_MAX_NODES) and coarsened so a disc holds at most SMOOTH_MAX_K nodes; only the nodes within
    transition_m + smooth_m + 2 target cells are evaluated. Neighbours by a KD-tree k-nearest query, k = the most
    source nodes a disc can hold (cells >= 0.70 x nominal) + 10 %."""
    r = float(zone.smooth_m)
    max_nodes = LIMITER_MAX_NODES if max_nodes is None else max_nodes
    n_cap = int(np.sqrt((max_nodes - 2) / 6.0))
    q_r = 0.5 * np.pi * R_EARTH_M / r                                 # cube edge / smooth_m
    n_t = min(max(2, int(np.ceil(q_r * SMOOTH_CELLS))), n_cap)
    n_k = int(np.sqrt(SMOOTH_MAX_K / (1.1 * np.pi)) * 0.70 * q_r)     # finest source grid within SMOOTH_MAX_K
    n_s = n_t if spacing_m is None else min(max(n_t, int(np.ceil(0.5 * np.pi * R_EARTH_M / spacing_m))), n_cap,
                                            max(n_t, n_k))
    node_id, cc_t, gt = cube_grid(n_t)
    sp_t = 0.5 * np.pi * R_EARTH_M / n_t
    t_reach = zone.transition_m + 2.0 * sp_t
    near, d = _zone_near(zone.polygon, t_reach, gt)
    tgt = near[d <= t_reach]
    _, cc_s, gs = (node_id, cc_t, gt) if n_s == n_t else cube_grid(n_s)
    sp_s = 0.5 * np.pi * R_EARTH_M / n_s
    near, d = _zone_near(zone.polygon, t_reach + r, gs)
    src = near[d <= t_reach + r]
    lnh = np.log(evaluate(h, gs[src]))
    area = _cube_area_density(cc_s[src], n_s)
    tree = cKDTree(gs[src])
    k = int(min(len(src), np.ceil(1.1 * np.pi * (r / (0.70 * sp_s)) ** 2) + 8))
    s = np.full(len(gt), np.nan)
    chunk = max(1, 4_000_000 // k)
    for a in range(0, len(tgt), chunk):
        q = gt[tgt[a:a + chunk]]
        dist, j = tree.query(q, k=k, distance_upper_bound=_m_to_chord(r), workers=-1)
        dist, j = dist.reshape(len(q), -1), j.reshape(len(q), -1)
        ok = np.isfinite(dist)
        j = np.where(ok, j, 0)
        rr = _chord_to_m(np.where(ok, dist, 0.0)) / r
        w = np.where(ok, (1.0 - np.minimum(rr, 1.0) ** 2) ** 2, 0.0) * area[j]
        s[tgt[a:a + chunk]] = np.exp((w * lnh[j]).sum(axis=1) / w.sum(axis=1))
    info = {"grid": "cubed_sphere_equiangular", "n": n_t, "spacing_m": sp_t, "n_source": n_s,
            "spacing_source_m": sp_s, "smooth_m": r, "k": k, "nodes_source": int(len(src)),
            "nodes_target": int(len(tgt))}
    return Smoothed(zone, Limited(n_t, node_id, s, info))


@dataclass(frozen=True, eq=False)
class Layered:
    """The layers applied in order to a running h that starts at `start` (the base h): a field (mode "min") gives
    h = min(h, field) where the field is below the base and leaves h alone where it is not (a min layer at the base
    is neutral, so it does not undo a set zone's raise above the base outside its own reach), a SetZone or Smoothed
    step h = step.apply(xyz, h). Built by `layered`."""
    steps: tuple
    start: float

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        return self.trace(xyz)[0]

    def trace(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(h, step): h as __call__, and per point the index of the step that last changed h (-1: none, the start);
        the UI's "which layer sets h"."""
        h = np.full(len(xyz), self.start)
        last = np.full(len(xyz), -1, dtype=np.int64)
        for k, s in enumerate(self.steps):
            if hasattr(s, "apply"):
                new = s.apply(xyz, h)
            else:
                v = s(xyz)
                new = np.minimum(h, np.where(v < self.start, v, np.inf))
            last[new != h] = k
            h = new
        return h, last


def layered(steps, start: float = np.inf, spacing_m: float | None = None) -> SizeField:
    """The size field of the layers in order (fields for mode "min", SetZone, SmoothZone).

    Only "min" layers: the field itself (one) or Minimum (several), as before the zone modes (min commutes, so the
    order does not matter and old configs give the same h bit for bit). Otherwise a Layered field: the running h
    starts at `start` (the run passes the base h; a min layer acts only where it is below it); each SmoothZone is
    bound (`smoothed`, source spacing spacing_m) to the Layered field of the layers before it."""
    steps = list(steps)
    modal = (SetZone, SmoothZone, BlendZone)
    if not any(isinstance(s, modal) for s in steps):
        return steps[0] if len(steps) == 1 else Minimum(tuple(steps))
    start = float(start)
    bound = []
    for s in steps:
        if isinstance(s, SmoothZone):
            s = smoothed(s, Layered(tuple(bound), start), spacing_m)
        elif isinstance(s, BlendZone) and s.h_edge is None:               # S37: the running h along its outer edge
            s = s.bind(Layered(tuple(bound), start))
        bound.append(s)
    return Layered(tuple(bound), start)


def zone(polygon_lonlat, h_m: float, transition_m: float, h_out_m: float) -> Zone:
    """Zone from [[lon, lat], ...] degrees."""
    p = np.asarray(polygon_lonlat, dtype=np.float64)
    return Zone(lonlat_to_xyz(p[:, 0], p[:, 1]), float(h_m), float(transition_m), float(h_out_m))


@dataclass(frozen=True)
class Clipped:
    """clip(h, h_min_m, h_max_m): the global bounds of [size]."""
    h: SizeField
    h_min_m: float = 0.0
    h_max_m: float = np.inf

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        return np.clip(self.h(xyz), self.h_min_m, self.h_max_m)


# ------------------------------------------------------------------------------------------ gradient limiter
#
# Grid: an equiangular cubed sphere with n cells per face edge. Its nodes are the integer points C of the cube
# surface max|C_k| = n with every C_k = n (mod 2); face f = (axis k = f // 2, sign + for even f) holds C_k = +-n,
# C_(k+1)%3 = 2i - n, C_(k+2)%3 = 2j - n for i, j in 0..n, and the node sits at unit(tan(pi/4 C / n)). Seam nodes are
# shared by 2 faces, the 8 cube corners by 3; there are 6 n^2 + 2 of them. Why this grid: no pole and no
# antimeridian (a lon-lat grid wastes pi/2 x the nodes and its
# stencil degenerates at the poles), cell edges within [0.71, 1] x the nominal spacing R (pi/2) / n everywhere, and
# a point is located in O(1) (face = largest |x_k|, two arctangents), so the interpolant is cheap for the methods.

LIMITER_MAX_NODES = 12_000_000     # ~1.1 GB of grid arrays during the solve; nominal spacing >= 7.1 km at the cap

# 16-neighbour stencil: in square cells the graph distance overestimates the geodesic by <= 1/cos(13.3°) - 1 = 2.7 %
# (8 neighbours: 8.2 %), near a cube corner (60° cells, 3 faces at the corner node) by <= 5 % (measured: decisions
# S11), so the limited field is that much less graded than the exact Hopf-Lax solution off the stencil directions.
_STENCIL = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1],
                     [1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]], dtype=np.int64)


def cube_grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(node_id (6, n+1, n+1) int32, C (U,3) int16 lattice coordinates, xyz (U,3) unit vectors), U = 6 n^2 + 2.

    node_id[f, i, j] is the unique node at face f, index (i, j); a seam node belongs to the lowest face index holding
    it, and its id is shared by the other faces.
    """
    ii, jj = np.meshgrid(np.arange(n + 1, dtype=np.int32), np.arange(n + 1, dtype=np.int32), indexing="ij")
    C = np.empty((6, n + 1, n + 1, 3), dtype=np.int32)
    for f in range(6):
        k, s = f // 2, 1 - 2 * (f % 2)
        C[f, :, :, k] = s * n
        C[f, :, :, (k + 1) % 3] = 2 * ii - n
        C[f, :, :, (k + 2) % 3] = 2 * jj - n
    owner = np.full((6, n + 1, n + 1), 6, dtype=np.int8)
    for f in range(5, -1, -1):                                        # the lowest face holding the node wins
        k, s = f // 2, 1 - 2 * (f % 2)
        owner[C[..., k] == s * n] = f
    owned = owner == np.arange(6, dtype=np.int8)[:, None, None]
    node_id = np.full((6, n + 1, n + 1), -1, dtype=np.int32)
    node_id[owned] = np.arange(int(owned.sum()), dtype=np.int32)
    nf, ni, nj = np.nonzero(~owned)
    of = owner[nf, ni, nj].astype(np.int64)
    c = C[nf, ni, nj]
    r = np.arange(len(of))
    node_id[nf, ni, nj] = node_id[of, (c[r, (of // 2 + 1) % 3] + n) // 2, (c[r, (of // 2 + 2) % 3] + n) // 2]
    cu = C[owned]
    return node_id, cu.astype(np.int16), _unit(np.tan(0.25 * np.pi * cu / n))


def cube_locate(xyz: np.ndarray, n: int) -> tuple[np.ndarray, ...]:
    """(face, i0, j0, fu, fv): the grid cell [i0, i0+1] x [j0, j0+1] of face `face` holding each point and the
    bilinear weights of its position in equiangular coordinates."""
    r = np.arange(len(xyz))
    ax = np.abs(xyz)
    k = np.argmax(ax, axis=1)
    ak = ax[r, k]
    f = 2 * k + (xyz[r, k] < 0)
    u = (np.arctan(xyz[r, (k + 1) % 3] / ak) * (4.0 / np.pi) + 1.0) * (0.5 * n)
    v = (np.arctan(xyz[r, (k + 2) % 3] / ak) * (4.0 / np.pi) + 1.0) * (0.5 * n)
    i0 = np.clip(np.floor(u).astype(np.int64), 0, n - 1)
    j0 = np.clip(np.floor(v).astype(np.int64), 0, n - 1)
    return f, i0, j0, u - i0, v - j0


def cube_edges(node_id: np.ndarray) -> np.ndarray:
    """(E,2) int64 unique node pairs joined by the limiter's stencil (for checks; the solver does not store them)."""
    n = node_id.shape[1] - 1
    out = []
    for di, dj in _STENCIL:
        if di < 0 or (di == 0 and dj < 0):
            continue
        a = node_id[:, max(0, -di):n + 1 - max(0, di), max(0, -dj):n + 1 - max(0, dj)]
        b = node_id[:, max(0, di):n + 1 - max(0, -di), max(0, dj):n + 1 - max(0, -dj)]
        out.append(np.column_stack([a.ravel(), b.ravel()]).astype(np.int64))
    e = np.sort(np.vstack(out), axis=1)
    return np.unique(e[e[:, 0] != e[:, 1]], axis=0)


@njit(cache=True)
def _sift_up(heap, pos, h, i):
    v = heap[i]
    hv = h[v]
    while i > 0:
        p = (i - 1) >> 1
        u = heap[p]
        if h[u] <= hv:
            break
        heap[i] = u
        pos[u] = i
        i = p
    heap[i] = v
    pos[v] = i


@njit(cache=True)
def _sift_down(heap, pos, h, i, size):
    v = heap[i]
    hv = h[v]
    while True:
        c = 2 * i + 1
        if c >= size:
            break
        if c + 1 < size and h[heap[c + 1]] < h[heap[c]]:
            c += 1
        u = heap[c]
        if h[u] >= hv:
            break
        heap[i] = u
        pos[u] = i
        i = c
    heap[i] = v
    pos[v] = i


@njit(cache=True)
def _dijkstra(h, xyz, cc, node_id, n, g_r, stencil, order):
    """In place: h_v <- min(h_v, h_u + g_r * angle(u, v)) over the stencil edges, nodes settled in increasing h
    (Dijkstra with an indexed binary heap; `order` = argsort(h) is the initial heap). Returns the relaxation count."""
    nu = len(h)
    heap = order.astype(np.int32)
    pos = np.empty(nu, dtype=np.int32)
    for q in range(nu):
        pos[heap[q]] = q
    size = nu
    relaxed = 0
    while size > 0:
        u = heap[0]
        size -= 1
        if size > 0:
            heap[0] = heap[size]
            pos[heap[0]] = 0
            _sift_down(heap, pos, h, 0, size)
        pos[u] = -1
        hu = h[u]
        for f in range(6):
            k = f // 2
            if cc[u, k] != (1 - 2 * (f % 2)) * n:
                continue
            fi = (cc[u, (k + 1) % 3] + n) // 2
            fj = (cc[u, (k + 2) % 3] + n) // 2
            for t in range(stencil.shape[0]):
                a = fi + stencil[t, 0]
                b = fj + stencil[t, 1]
                if a < 0 or a > n or b < 0 or b > n:
                    continue
                w = node_id[f, a, b]
                if pos[w] < 0:
                    continue
                dx = xyz[u, 0] - xyz[w, 0]
                dy = xyz[u, 1] - xyz[w, 1]
                dz = xyz[u, 2] - xyz[w, 2]
                cand = hu + g_r * 2.0 * np.arcsin(min(0.5 * np.sqrt(dx * dx + dy * dy + dz * dz), 1.0))
                if cand < h[w]:
                    h[w] = cand
                    relaxed += 1
                    _sift_up(heap, pos, h, pos[w])
    return relaxed


@dataclass(frozen=True, eq=False)
class Limited:
    """Bilinear interpolant of node values h_node on the equiangular cubed-sphere grid (cube_grid(n)): exact at the
    nodes, O(1) per point, no pole or antimeridian case. `info` records how the grid was built.

    Its gradient in a cell lies in the box of the cell's two u-edge and two v-edge differences, so with edge slopes
    <= g it is <= g sqrt(2 / (1 - |cos a|)) (a: the cell's angle; sqrt 2 g in square cells, 2 g at 60°): reached
    within one cell of a kink of a limited field (S11), as by any exact-at-node interpolant linear along the edges."""
    n: int
    node_id: np.ndarray
    h_node: np.ndarray
    info: dict

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        f, i0, j0, fu, fv = cube_locate(xyz, self.n)
        nid, h = self.node_id, self.h_node
        h00, h10 = h[nid[f, i0, j0]], h[nid[f, i0 + 1, j0]]
        h01, h11 = h[nid[f, i0, j0 + 1]], h[nid[f, i0 + 1, j0 + 1]]
        return h00 + fu * (h10 - h00) + fv * (h01 - h00) + fu * fv * (h11 - h10 - h01 + h00)   # exact on constants


def _fibonacci(m: int) -> np.ndarray:
    i = np.arange(m) + 0.5
    z = 1.0 - 2.0 * i / m
    ph = np.pi * (1.0 + 5**0.5) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(ph), r * np.sin(ph), z])


def evaluate(h: SizeField, xyz: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    """h(xyz) in chunks (bounds the temporaries of KD-tree and polygon layers on large grids)."""
    return np.concatenate([np.asarray(h(xyz[a:a + chunk]), dtype=np.float64) for a in range(0, len(xyz), chunk)])


def limit_gradient(h: SizeField, domain, grad_max: float, spacing_m: float | None = None,
                   max_nodes: int = LIMITER_MAX_NODES) -> Limited:
    """The largest field <= h with |h_i - h_j| <= grad_max d_ij on every stencil edge of a cubed-sphere grid
    (d_ij geodesic), as a bilinear interpolant (Lipschitz limiting / edge relaxation; Hopf-Lax on the graph).

    - Grid: nominal spacing spacing_m (default: half the minimum of h over 2e5 Fibonacci points; the run passes half
      the smallest h its layers can take), so a size feature of width >= 2 h_min spans several cells; capped at
      max_nodes nodes (the spacing then grows; info["capped"]).
    - Solver: Dijkstra-style relaxation h_j <- min(h_j, h_i + grad_max d_ij) over a 16-neighbour stencil (Numba,
      indexed heap, O(U log U)), deterministic. Along every stencil edge the result satisfies the bound exactly; off
      the stencil directions the graph metric is <= 2.7 % (square cells) to ~5 % (cube corners, measured) longer
      than the geodesic (metrication).
    - Continuous slope (S11): over two cells <= g (1 + metrication), measured <= 1.050 g; within one cell of a kink
      of h (a ramp's start at a step, a cone apex, a ridge) the bilinear cells reach 1.4 g (square cells) to 1.7 g
      (cube corners; bound 2 g, `Limited`).
    - Nodes where h was already graded keep h exactly; between nodes the interpolant is bilinear, so it differs from
      h by the interpolation error only (exact where h is constant over a cell).
    - Limiting runs over the whole sphere, across land as fesom2_mesh_gen did; `domain` is not used yet (a "water
      plus a land margin" variant needs a land mask on the grid). Across land the field is only ever smaller,
      never steeper.
    """
    if spacing_m is None:
        spacing_m = 0.5 * float(np.min(h(_fibonacci(200_000))))
    n_cap = int(np.sqrt((max_nodes - 2) / 6.0))
    n = max(2, int(np.ceil(0.5 * np.pi * R_EARTH_M / spacing_m)))
    capped = n > n_cap
    n = min(n, n_cap)
    node_id, cc, xyz = cube_grid(n)
    h0 = evaluate(h, xyz)
    hl = h0.copy()
    relaxed = _dijkstra(hl, xyz, cc, node_id, n, float(grad_max) * R_EARTH_M, _STENCIL,
                        np.argsort(hl, kind="stable"))
    info = {"grid": "cubed_sphere_equiangular", "n": n, "nodes": int(len(hl)),
            "spacing_m": 0.5 * np.pi * R_EARTH_M / n, "spacing_requested_m": float(spacing_m), "capped": bool(capped),
            "grad_max": float(grad_max), "relaxations": int(relaxed), "nodes_lowered": int(np.sum(hl < h0)),
            "h_min_m": float(hl.min()), "h_max_m": float(hl.max())}
    return Limited(n, node_id, hl, info)
