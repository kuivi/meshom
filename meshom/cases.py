"""Competition test cases at reference scale 1.0: CASES[name] = (Domain, SizeField).

Quick runs use scaled(h, 4); the scale changes the node count, not rho_c. STRAITS[name] is the case's strait corridor
(statistics per strait corridor), a callable xyz -> bool, for the cases that have one.
"""
from dataclasses import dataclass

import numpy as np

from meshom.domain import CHANNEL_WIDTH_M, Domain, closed_sphere, continent_case, continent_frame, CONTINENT_RADIUS_DEG
from meshom.geo import R_EARTH_M, arc_length
from meshom.size import SizeField, band, disc, uniform

CHANNEL_H_M = 15e3        # 150 km channel / 15 km = 10 cells across (>= 7 required)
CHANNEL_PLATEAU_M = 100e3  # h = CHANNEL_H_M within this distance of the channel centreline arc
CONTINENT_H_M = 50e3
CONTINENT_G = 0.1


def _arc_distance_m(xyz: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Great-circle distance (m) from each point to the arc p-q (the nearer end point beyond the arc's span)."""
    n = np.cross(p, q)
    n /= np.linalg.norm(n)
    z = xyz @ n
    w = xyz - z[:, None] * n
    # inside the arc's span: w lies between p and q (same side of the planes through n,p and n,q)
    between = (np.cross(p, w) @ n >= 0) & (np.cross(w, q) @ n >= 0)
    d_line = np.arcsin(np.clip(np.abs(z), 0.0, 1.0))
    d_end = np.minimum(arc_length(xyz, p[None, :]), arc_length(xyz, q[None, :]))
    return R_EARTH_M * np.where(between, d_line, d_end)


@dataclass(frozen=True)
class ArcDistanceField:
    """h_near within d0 of a great-circle arc, linear with slope g to h_far, h_far beyond."""
    p: tuple
    q: tuple
    d0_m: float
    h_near_m: float
    h_far_m: float
    g: float

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        d = _arc_distance_m(xyz, np.asarray(self.p), np.asarray(self.q))
        return np.clip(self.h_near_m + self.g * (d - self.d0_m), self.h_near_m, self.h_far_m)


@dataclass(frozen=True)
class StraitCorridor:
    """A strait corridor: the points within half_width_m of the great-circle arc p-q (its centreline, mouth to
    mouth); callable xyz -> bool. Meant for water elements, so land inside the corridor does not matter."""
    p: tuple
    q: tuple
    half_width_m: float

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        return _arc_distance_m(xyz, np.asarray(self.p), np.asarray(self.q)) <= self.half_width_m


def continent_h() -> SizeField:
    """50 km, 15 km within 100 km of the channel centreline (extended 1° past both mouths), slope 0.1 between."""
    c, t, _ = continent_frame()
    beta = np.radians(CONTINENT_RADIUS_DEG + 1.0)
    p = np.cos(beta) * c - np.sin(beta) * t
    q = np.cos(beta) * c + np.sin(beta) * t
    return ArcDistanceField(tuple(p), tuple(q), CHANNEL_PLATEAU_M, CHANNEL_H_M, CONTINENT_H_M, CONTINENT_G)


def continent_strait() -> StraitCorridor:
    """The continent's channel, the case's narrowest passage (150 km; the island gap is ~900 km): the water within
    half the channel width of its centreline between the two mouths (the channel itself, plus a half-disc of that
    radius beyond each mouth)."""
    c, t, _ = continent_frame()
    beta = np.radians(CONTINENT_RADIUS_DEG)
    p = np.cos(beta) * c - np.sin(beta) * t
    q = np.cos(beta) * c + np.sin(beta) * t
    return StraitCorridor(tuple(p), tuple(q), 0.5 * CHANNEL_WIDTH_M)


CASES: dict[str, tuple[Domain, SizeField]] = {
    "uniform": (closed_sphere(), uniform(100e3)),
    "disc": (closed_sphere(), disc(-30.0, 40.0, 500e3, 10e3, 900e3, 50e3)),
    "band": (closed_sphere(), band(0.0, 25e3, 100e3, 0.1)),
    "continent": (continent_case(), continent_h()),
}

STRAITS: dict = {"continent": continent_strait()}

import meshom.cases_seams  # noqa: E402,F401  registers the seam cases
