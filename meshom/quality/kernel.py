"""Per-element geometry in each element's tangent plane: corner angles, edge lengths, areas, centroids.

Corner k of an element with tangent-plane corners q: a = q[k+1] - q[k], b = q[k-1] - q[k],
angle = atan2(a x b, a . b) mapped to (0, 2 pi); a reflex corner of a non-convex CCW quad is > pi.
Quad corner scaled Jacobian = sin(angle); the element SJ is its minimum over corners (negative when reflex).
"""
from dataclasses import dataclass

import numpy as np

from meshom.geo import R_EARTH_M, arc_length, tangent_coords
from meshom.mesh import Mesh, _shoelace

TWO_PI = 2.0 * np.pi


@dataclass
class ElementGeometry:
    tri_angles: np.ndarray    # (T,3) rad in (0, 2 pi), corner k
    quad_angles: np.ndarray   # (Q,4) rad in (0, 2 pi)
    tri_edges: np.ndarray     # (T,3) m, edge k joins corner k and k+1 (great circle)
    quad_edges: np.ndarray    # (Q,4) m
    tri_area: np.ndarray      # (T,) m^2, signed, tangent plane
    quad_area: np.ndarray     # (Q,) m^2
    centroid: np.ndarray      # (T+Q,3) unit vectors, normalised corner mean


def _angles(q: np.ndarray) -> np.ndarray:
    a = np.roll(q, -1, axis=1) - q
    b = np.roll(q, 1, axis=1) - q
    cr = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dt = a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]
    return np.mod(np.arctan2(cr, dt), TWO_PI)


def _cells(points: np.ndarray, cells: np.ndarray) -> tuple:
    k = cells.shape[1]
    if len(cells) == 0:
        return np.zeros((0, k)), np.zeros((0, k)), np.zeros(0), np.zeros((0, 3))
    p = points[cells]                                                  # (M,k,3)
    q = tangent_coords(p)
    nxt = np.roll(p, -1, axis=1)
    edges = R_EARTH_M * arc_length(p.reshape(-1, 3), nxt.reshape(-1, 3)).reshape(-1, k)
    c = p.mean(axis=1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    return _angles(q), edges, _shoelace(q), c


def element_geometry(mesh: Mesh) -> ElementGeometry:
    ta, te, tA, tc = _cells(mesh.points, mesh.tris)
    qa, qe, qA, qc = _cells(mesh.points, mesh.quads)
    return ElementGeometry(ta, qa, te, qe, tA, qA, np.vstack([tc, qc]))
