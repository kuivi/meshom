"""m04: Lp-CVT points -> sphere Delaunay -> recombination; exact energy gradient; valid meshes at scale 4."""
import numpy as np
import pytest

from meshom.cases import CASES
from meshom.compete import irregular_counts
from meshom.delaunay import sphere_delaunay
from meshom.domain import resample_loops
from meshom.geo import R_EARTH_M, tangent_basis
from meshom.mesh import build_topology, signed_area
from meshom.methods import METHODS
from meshom.methods.m04.energy import delaunay_update, lp_energy, unit_square_integral
from meshom.quality.metrics import compute_quality
from meshom.size import scaled


def _random_setup(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((n, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    e1, e2 = tangent_basis(p)
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    u = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
    h = 400e3 * rng.uniform(0.8, 1.2, n)
    own = np.ones(n, dtype=bool)
    own[:50] = False
    return p, u, np.cross(p, u), R_EARTH_M / h, (h / 400e3) ** 2 / unit_square_integral(8), own


def test_gradient_matches_finite_differences():
    p, u, v, inv_h, wgt, own = _random_setup()
    tris = sphere_delaunay(p)
    _, g, _ = lp_energy(p, tris, own, u, v, inv_h, wgt, 8)
    for i in (3, 700, 1999):
        for d in (u[i], v[i]):
            eps = 1e-7
            pp, pm = p.copy(), p.copy()
            pp[i] += eps * d
            pm[i] -= eps * d
            fd = (lp_energy(pp, tris, own, u, v, inv_h, wgt, 8)[0] - lp_energy(pm, tris, own, u, v, inv_h, wgt, 8)[0])
            assert abs(fd / (2 * eps) - g[i] @ d) < 1e-5 * (abs(g[i] @ d) + 1.0)


def test_aligned_square_cell_has_unit_energy():
    """A cell of a square lattice of side h, aligned with its frame, contributes 1; rotated by 45 deg about 3.2."""
    rng = np.random.default_rng(0)
    hh, n = 100e3, 11
    xs = (np.arange(n) - n // 2) * hh / R_EARTH_M
    xg, yg = np.meshgrid(xs, xs)
    q = np.column_stack([xg.ravel(), yg.ravel(), np.ones(n * n)])
    q += 1e-9 * rng.standard_normal(q.shape)
    far = rng.standard_normal((300, 3))
    far = far[far[:, 2] < 0.5]
    pts = np.vstack([q, far])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    tris = sphere_delaunay(pts)
    e1, e2 = tangent_basis(pts)
    c = n * n // 2
    own = np.zeros(len(pts), dtype=bool)
    own[c] = True
    one = np.ones(len(pts))
    for ang, want in ((0.0, 1.0), (45.0, 3.2)):
        a = np.radians(ang)
        u = np.cos(a) * e1 + np.sin(a) * e2
        cell = lp_energy(pts, tris, own, u, np.cross(pts, u), one * R_EARTH_M / hh, one / unit_square_integral(8), 8)[2]
        assert abs(cell[c] - want) < 0.01 * want


def test_lawson_update_gives_the_hull():
    p = _random_setup()[0]
    state, full = delaunay_update(p, None)
    assert full
    rng = np.random.default_rng(3)
    q = p + 3e-4 * rng.standard_normal(p.shape)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    state, full = delaunay_update(q, state)
    key = lambda t: np.sort(np.sort(t, axis=1) @ np.array([len(q) ** 2, len(q), 1]))
    assert not full and np.array_equal(key(state[0]), key(sphere_delaunay(q)))


@pytest.mark.parametrize("case", ["uniform", "continent"])
def test_valid_mesh(case):
    dom, h0 = CASES[case]
    h = scaled(h0, 4.0)
    m = METHODS["m04_lpcvt"](dom, h, {})
    topo = build_topology(m)
    assert (signed_area(m) > 0).all()                                     # CCW, no inverted element
    assert (topo.elems_per_node > 0).all()                                # every node used
    ic = irregular_counts(m)
    assert ic["euler_ok"] and ic["euler_sum"] == 4 * (2 - len(dom.loops))   # edge-manifold, right topology
    s = compute_quality(m).summary
    assert s["n_pinch_nodes"] == 0
    loops = resample_loops(dom, h)
    nb = sum(len(lp) for lp in loops)
    assert np.array_equal(m.points[:nb], np.vstack(loops) if loops else m.points[:0])   # coast nodes kept, first
    assert topo.boundary_node.sum() == nb and topo.boundary_node[:nb].all()
    assert m.n_quads > 0.9 * m.n_elems and s["quad_max_angle_deg"]["max"] <= 150.0
    assert ic["n3"] + ic["n5"] < 0.1 * ic["n_interior"]                     # aligned: few irregular nodes
