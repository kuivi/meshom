import numpy as np

from helpers import local_polygon, mesh_from_polygons, random_rotation, structured_patch
from meshom.mesh import Mesh
from meshom.quality.kernel import element_geometry


def deg(a):
    return np.degrees(a)


def test_equilateral_triangle():
    s = 1e4
    g = element_geometry(mesh_from_polygons([local_polygon(16, 37, [(0, 0), (s, 0), (s / 2, s * 3**0.5 / 2)])]))
    assert np.allclose(deg(g.tri_angles), 60.0, atol=1e-4) and np.allclose(g.tri_edges, s, atol=1e-2)


def test_square_quad():
    g = element_geometry(mesh_from_polygons([local_polygon(16, 37, [(0, 0), (1e4, 0), (1e4, 1e4), (0, 1e4)])]))
    assert np.allclose(deg(g.quad_angles), 90.0, atol=2e-4)
    assert np.allclose(g.quad_edges, 1e4, atol=0.1) and abs(g.quad_area[0] - 1e8) < 1e8 * 1e-5


def test_quad_with_150_degree_corner():
    c = np.cos(np.radians(30)) * 1e4
    s = np.sin(np.radians(30)) * 1e4
    g = element_geometry(mesh_from_polygons([local_polygon(0, 0, [(0, 0), (2e4, 0), (2e4 - c, s), (c, s)])]))
    assert np.isclose(deg(g.quad_angles).max(), 150.0, atol=1e-3)


def test_nonconvex_quad():
    g = element_geometry(mesh_from_polygons([local_polygon(0, 0, [(0, 0), (1e4, 5e3), (0, 1e4), (3e3, 5e3)])]))
    a = deg(g.quad_angles[0])
    assert a.max() > 180.0 and np.isclose(a.sum(), 360.0, atol=1e-4)
    assert np.sin(g.quad_angles[0]).min() < 0


def test_element_at_north_pole():
    sq = [(-5e3, -5e3), (5e3, -5e3), (5e3, 5e3), (-5e3, 5e3)]
    g = element_geometry(mesh_from_polygons([local_polygon(0, 90, sq)]))
    assert np.allclose(deg(g.quad_angles), 90.0, atol=1e-4)


def test_element_across_antimeridian():
    sq = [(-5e3, -5e3), (5e3, -5e3), (5e3, 5e3), (-5e3, 5e3)]
    a = element_geometry(mesh_from_polygons([local_polygon(180, 0, sq)]))
    b = element_geometry(mesh_from_polygons([local_polygon(0, 0, sq)]))
    assert np.allclose(a.quad_angles, b.quad_angles, atol=1e-12) and np.allclose(a.quad_edges, b.quad_edges, atol=1e-6)


def test_rotation_invariance():
    m = structured_patch(4, 3, 5e3, lon0=30, lat0=-60)
    r = random_rotation(np.random.default_rng(1))
    g1 = element_geometry(m)
    g2 = element_geometry(Mesh(m.points @ r.T, m.tris, m.quads, m.node_flag))
    assert np.allclose(g1.quad_angles, g2.quad_angles, atol=1e-9) and np.allclose(g1.quad_edges, g2.quad_edges, atol=1e-6)
