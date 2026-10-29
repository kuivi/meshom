"""smoothing of mixed meshes on the sphere."""
import numpy as np

from helpers import local_polygon, mesh_from_polygons, split_quads, structured_patch
from meshom.cases import CASES
from meshom.delaunay import background_mesh
from meshom.geo import R_EARTH_M, arc_length
from meshom.mesh import Mesh, build_topology, signed_area
from meshom.quality.kernel import element_geometry
from meshom.size import scaled, uniform
from meshom.smooth import size_scale, smooth


def _perturbed_patch(amp=0.25, seed=0):
    """10 x 10 quads of 10 km; interior nodes moved by up to amp * 10 km in a random tangent direction."""
    m = structured_patch(10, 10, 10e3, lon0=10.0, lat0=40.0)
    fixed = build_topology(m).boundary_node
    rng = np.random.default_rng(seed)
    p = m.points.copy()
    t = rng.normal(size=p.shape)
    t -= np.sum(t * p, axis=1, keepdims=True) * p
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    r = amp * 10e3 / R_EARTH_M * rng.uniform(0, 1, len(p))
    p[~fixed] += (r[:, None] * t)[~fixed]
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return Mesh(p, m.tris, m.quads, m.node_flag), fixed


def _quad_dev(m):
    return np.degrees(np.abs(element_geometry(m).quad_angles - np.pi / 2)).max()


def test_angle_undoes_perturbation_of_structured_patch():
    m, fixed = _perturbed_patch()
    assert _quad_dev(m) > 20.0
    s = smooth(m, uniform(10e3), fixed, iters=100, kind="angle")
    assert _quad_dev(s) < 1.0
    assert np.array_equal(s.points[fixed], m.points[fixed])
    assert np.allclose(np.linalg.norm(s.points, axis=1), 1.0, atol=1e-15)
    s2 = smooth(m, uniform(10e3), np.flatnonzero(fixed), iters=100, kind="angle")   # fixed as an index array
    assert np.array_equal(s2.points, s.points)


def test_angle_on_mixed_patch():
    """Eight quads split into triangles: no element inverts, and angle-based beats Laplace on the quads."""
    m, fixed = _perturbed_patch()
    m = split_quads(m, [22, 26, 44, 47, 63, 66, 75, 78])
    a = smooth(m, None, fixed, iters=50, kind="angle")
    lp = smooth(m, None, fixed, iters=50, kind="laplace")
    assert (signed_area(a) > 0).all() and (signed_area(lp) > 0).all()
    assert _quad_dev(a) < 20.0 < _quad_dev(lp)


def test_smart_guard_blocks_inversion():
    """Arrow-shaped star: the neighbour mean lies between the prongs, outside the star's kernel."""
    ring = local_polygon(0.0, 0.0, 1e4 * np.array([(-1, -1), (5, -0.2), (-0.5, 0), (5, 0.2), (-1, 1)]))
    centre = local_polygon(0.0, 0.0, [(-0.8e4, 0.0)])[0]
    m = mesh_from_polygons([[centre, ring[i], ring[(i + 1) % 5]] for i in range(5)])
    assert (signed_area(m) > 0).all()
    fixed = build_topology(m).boundary_node
    assert (signed_area(smooth(m, None, fixed, iters=1, kind="laplace", smart=False)) < 0).any()
    assert (signed_area(smooth(m, None, fixed, iters=1, kind="laplace")) > 0).all()


def _edge_ratio(m, h, centre):
    """L / h per triangle edge (h at the midpoint) and the midpoint distance from the disc centre, metres."""
    e = np.unique(np.sort(np.vstack([m.tris[:, [0, 1]], m.tris[:, [1, 2]], m.tris[:, [2, 0]]]), axis=1), axis=0)
    a, b = m.points[e[:, 0]], m.points[e[:, 1]]
    mid = (a + b) / np.linalg.norm(a + b, axis=1, keepdims=True)
    return R_EARTH_M * arc_length(a, b) / h(mid), R_EARTH_M * arc_length(mid, centre[None, :])


def test_size_fits_h_on_disc():
    """'disc' at scale 4 (h 40 km inside 500 km, ramp to 200 km at 900 km), node count matched to h, then the fine
    disc squeezed radially (r -> 0.85 r inside 500 km, back to r at 1300 km): `size` restores edge p50 to h."""
    dom, h = CASES["disc"]
    h = scaled(h, 4.0)
    lam = size_scale(background_mesh(dom, h, factor=1.0), h)    # ~1.10: the factor-1 count gives edges of 1.10 h
    m = background_mesh(dom, h, factor=1.0 / lam)
    c = np.asarray(h.h.center)
    r = R_EARTH_M * arc_length(m.points, c[None, :])
    s = 1.0 - 0.15 * np.clip((1300e3 - r) / 800e3, 0.0, 1.0)
    t = m.points - (m.points @ c)[:, None] * c
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    rn = (r * s / R_EARTH_M)[:, None]
    m = Mesh(np.cos(rn) * c + np.sin(rn) * t, m.tris, m.quads, m.node_flag)
    q, d = _edge_ratio(m, h, c)
    assert np.median(q[d < 500e3]) < 0.92
    out = smooth(m, h, np.zeros(m.n_nodes, dtype=bool), iters=100, kind="size")
    q, d = _edge_ratio(out, h, c)
    assert abs(np.median(q) - 1.0) < 0.05
    assert abs(np.median(q[d < 500e3]) - 1.0) < 0.05


def _star(xy, n_ring):
    """Centre 0 (the only movable node) and a ring 1..n_ring-1: quad (0, 1, 2, 3), triangles (0, k, k+1) after it."""
    tris = np.array([[0, k + 1, 1 + (k + 1) % n_ring] for k in range(2, n_ring)], dtype=np.int64)
    m = Mesh(local_polygon(0.0, 0.0, 100.0 * np.asarray(xy, dtype=float)), tris,
             np.array([[0, 1, 2, 3]], dtype=np.int64), np.zeros(n_ring + 1, dtype=np.int8))
    fixed = np.ones(n_ring + 1, dtype=bool)
    fixed[0] = False
    return m, fixed


def test_smart_guard_keeps_quads_convex():
    """The star's worst deviation is a poor triangle (107 / 115 deg); a move that improves
    it made the convex quad reflex (185 / 187 deg corner). No accepted move may invert an element or make a quad
    non-convex, whatever the rest of the star gains."""
    cases = [  # (ring + centre in hm, kind)
        ([[0, 0], [58, 19], [444, 398], [-31, 3], [5, -198], [173, -228], [82, -82], [198, -137]], "angle"),
        ([[0, 0], [-16, 35], [-43, 77], [-32, 8], [-35, -40], [-138, -191], [-119, -196], [-215, -393], [68, -84]],
         "size"),
    ]
    for xy, kind in cases:
        m, fixed = _star(xy, len(xy) - 1)
        assert (signed_area(m) > 0).all() and element_geometry(m).quad_angles.max() < np.pi
        for iters in (1, 10):
            out = smooth(m, uniform(10e3), fixed, iters=iters, kind=kind)
            assert (signed_area(out) > 0).all()
            assert element_geometry(out).quad_angles.max() < np.pi, (kind, iters)
            assert np.array_equal(out.points[fixed], m.points[fixed])


def test_smooth_keeps_tri_zone():
    """Smoothing moves nodes only: the declared triangle zone stays."""
    m = split_quads(structured_patch(8, 8, 10e3), [27, 28, 35, 36])
    m.tri_zone = np.ones(m.n_tris, dtype=bool)
    out = smooth(m, None, build_topology(m).boundary_node, iters=1)
    assert out.tri_zone is not None and np.array_equal(out.tri_zone, m.tri_zone)
