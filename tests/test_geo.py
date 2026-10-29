import numpy as np

from helpers import local_polygon, shoelace
from meshom.geo import (
    R_EARTH_M,
    arc_length,
    euler_matrix,
    geo_to_rot,
    lonlat_to_xyz,
    rot_to_geo,
    stereographic_inverse,
    tangent_basis,
    tangent_coords,
    xyz_to_lonlat,
)


def test_lonlat_roundtrip_and_range():
    lon = np.array([-179.9, -40.0, 0.0, 16.0, 299.3988166]); lat = np.array([-89.0, 75.0, 0.0, 37.0, 74.28292396])
    xyz = lonlat_to_xyz(lon, lat)
    assert np.allclose(np.linalg.norm(xyz, axis=1), 1.0, atol=1e-15)
    lo, la = xyz_to_lonlat(xyz)
    assert np.allclose(la, lat, atol=1e-12)
    assert np.allclose(lo, [-179.9, -40.0, 0.0, 16.0, 299.3988166 - 360], atol=1e-9)


def test_pole_lonlat():
    lo, la = xyz_to_lonlat(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))
    assert np.allclose(la, [90.0, -90.0]) and np.all(np.isfinite(lo))


def test_arc_length_quarter_and_one_metre():
    assert np.isclose(arc_length(np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))[0], np.pi / 2)
    d = 1.0 / R_EARTH_M
    q = np.array([[np.cos(d), np.sin(d), 0.0]])
    assert abs(arc_length(np.array([[1.0, 0, 0]]), q)[0] * R_EARTH_M - 1.0) < 1e-9


def test_tangent_basis_orthonormal_including_poles():
    rng = np.random.default_rng(0)
    c = np.vstack([[[0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0]], rng.normal(size=(1000, 3))])
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    e1, e2 = tangent_basis(c)
    assert np.allclose(np.einsum("ij,ij->i", e1, e2), 0, atol=1e-14)
    assert np.allclose(np.einsum("ij,ij->i", e1, c), 0, atol=1e-14)
    assert np.allclose(np.cross(e1, e2), c, atol=1e-14)


def test_tangent_coords_square_10km():
    # 10 km square built CCW in the local east/north frame at (16E, 37N)
    p = local_polygon(16.0, 37.0, [(0, 0), (1e4, 0), (1e4, 1e4), (0, 1e4)])
    q = tangent_coords(p[None])[0]
    assert shoelace(q) > 0
    assert np.allclose(np.linalg.norm(np.roll(q, -1, 0) - q, axis=1), 1e4, atol=0.1)


def test_euler_rotated_poles():
    m = euler_matrix()
    assert np.allclose(m @ m.T, np.eye(3), atol=1e-15) and np.isclose(np.linalg.det(m), 1.0)
    lo, la = xyz_to_lonlat(rot_to_geo(np.array([[0, 0, 1.0], [0, 0, -1.0]]), m))
    assert np.allclose(lo, [-40.0, 140.0], atol=1e-9) and np.allclose(la, [75.0, -75.0], atol=1e-9)
    g = lonlat_to_xyz(np.array([10.0]), np.array([50.0]))
    assert np.allclose(rot_to_geo(geo_to_rot(g, m), m), g, atol=1e-15)


def test_stereographic_inverse():
    p = stereographic_inverse(np.array([0.0]), np.array([0.0]), 16.0, 37.0)
    assert np.allclose(xyz_to_lonlat(p), ([16.0], [37.0]))
    dphi = np.radians(5.0)
    p = stereographic_inverse(np.array([0.0]), np.array([2 * R_EARTH_M * np.tan(dphi / 2)]), 16.0, 37.0)
    assert np.allclose(xyz_to_lonlat(p), ([16.0], [42.0]), atol=1e-9)
