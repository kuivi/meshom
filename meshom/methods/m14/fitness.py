"""m14 step 1: fitness map from h alone.

Background mesh at 1.0 h, the B4 cross field ("cones", coast nodes fixed) and its cones, rho_c per node (dual cell:
(2/pi)(spherical area + integral of Laplace ln h), compete._node_rho), and the distances that make a node admissible
for a block, in units of the local h: >= cone_margin from every cone, >= coast_margin from the coast, >=
complex_margin from a complex coast point (a resampled coast vertex turning by more than complex_turn_deg).

Water under the cut (`land`, B70): a run whose land is cut away afterwards (variant _cut, `meshom run` with treatment
"cut") builds on the closed sphere, so the coast distances are infinite (land does not stop a block); `wet` then
marks the bg nodes in the water of the true coast (domain.in_water), for the placement's water-aware rules.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from meshom.delaunay import adjacency, background_mesh
from meshom.domain import Domain, _turn_angles, resample_loops
from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh
from meshom.size import SizeField


@dataclass
class Fitness:
    bg: Mesh                 # background triangle mesh (coast nodes first, flag 1)
    u: np.ndarray            # (N,3) cross-field direction per bg node
    hv: np.ndarray           # (N,) h at the bg nodes, m
    area: np.ndarray         # (N,) spherical dual area, sr (= integral of K dA on the unit sphere)
    lap: np.ndarray          # (N,) integral of Laplace ln h over the dual cell
    rho: np.ndarray          # (N,) rho_c per dual cell, cone units
    cone_pos: np.ndarray     # (C,3) cone centroids
    cone_idx: np.ndarray     # (C,) cone indices (multiples of 1/4)
    d_cone: np.ndarray       # (N,) distance to the nearest cone, units of h at the node
    d_coast: np.ndarray      # (N,) distance to the coast, units of h
    d_complex: np.ndarray    # (N,) distance to a complex coast point, units of h
    nbr: np.ndarray          # (T,3) triangle adjacency of bg
    wet: np.ndarray | None = None    # (N,) bool: node in the water of the land cut (B70); None = all water


def _arc(chord: np.ndarray) -> np.ndarray:
    return 2.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))


def dist_h(points: np.ndarray, targets: np.ndarray, hv: np.ndarray) -> np.ndarray:
    """Great-circle distance from each point to the nearest target, in units of hv (inf without targets)."""
    if len(targets) == 0:
        return np.full(len(points), np.inf)
    d, _ = cKDTree(targets).query(points, workers=-1)
    return _arc(d) * R_EARTH_M / hv


def fitness(domain: Domain, h: SizeField, o: dict, land: Domain | None = None) -> Fitness:
    """The fitness map of `domain` (m14 step 1); land: the true coast of a land cut that follows (B70), whose
    water marks the bg nodes `wet` (None: no cut, every node water)."""
    from meshom.compete import _node_rho
    from meshom.field import cross_field, singularities

    bg = background_mesh(domain, h, factor=o["bg_factor"])
    bnodes = np.flatnonzero(bg.node_flag == 1) if domain.loops else None
    cf = cross_field(bg, h, n=4, boundary_nodes=bnodes)
    ids, idx = singularities(cf)
    P, T = bg.points, bg.tris
    nbr = adjacency(T)
    t, k = np.nonzero(nbr < 0)
    bedges = np.column_stack([T[t, k], T[t, (k + 1) % 3]])
    hv = h(P)
    area, lap = _node_rho(P, T, bedges, np.log(hv))
    rho = (2.0 / np.pi) * (area + lap)
    cp = P[T[ids]].sum(axis=1)
    cp = cp / np.linalg.norm(cp, axis=1, keepdims=True) if len(ids) else np.zeros((0, 3))
    d_cone = dist_h(P, cp, hv)
    if domain.loops:
        d_coast = dist_h(P, np.vstack(domain.loops), hv)
        rl = resample_loops(domain, h)
        cx = np.vstack([lp[_turn_angles(lp) > np.radians(o["complex_turn_deg"])] for lp in rl])
        d_complex = dist_h(P, cx, hv)
    else:
        d_coast = np.full(len(P), np.inf)
        d_complex = np.full(len(P), np.inf)
    wet = None
    if land is not None and land.loops:
        from meshom.domain import in_water

        wet = in_water(land, P)
    return Fitness(bg, cf.u, hv, area, lap, rho, cp, idx, d_cone, d_coast, d_complex, nbr, wet)
