"""sphere Delaunay, edge recovery, land removal, Poisson-disc points, background mesh."""
import time

import numpy as np
import pytest
from scipy.spatial import cKDTree

from meshom.cases import CASES
from meshom.delaunay import background_mesh, poisson_points, recover_edges, remove_land, sphere_delaunay
from meshom.domain import Domain, in_water, loop_edges, resample_loops
from meshom.geo import R_EARTH_M, arc_length
from meshom.size import scaled


def _edges(tris):
    e = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0, return_counts=True)


def _det(p, t):
    return np.einsum("ij,ij->i", p[t[:, 0]], np.cross(p[t[:, 1]], p[t[:, 2]]))


def _random_sphere(n, seed):
    p = np.random.default_rng(seed).normal(size=(n, 3))
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def test_sphere_delaunay_closed_manifold():
    p = _random_sphere(10_000, 0)
    t = sphere_delaunay(p)
    e, cnt = _edges(t)
    assert (cnt == 2).all()
    assert len(p) - len(e) + len(t) == 2
    assert (_det(p, t) > 0).all()


def test_recover_and_remove_land_continent():
    """Coarse coast (scale 16) against dense random water points: many coast edges are missing from the hull."""
    dom, h = CASES["continent"]
    loops = resample_loops(dom, scaled(h, 16))
    coast = Domain(loops, "resampled")
    r = _random_sphere(100_000, 1)
    p = np.vstack([np.vstack(loops), r[in_water(coast, r)]])
    be = loop_edges(loops)
    t0 = sphere_delaunay(p)
    want = set(map(tuple, np.sort(be, axis=1)))
    assert len(want - set(map(tuple, _edges(t0)[0]))) > 10
    t = recover_edges(p, t0, be)
    e, cnt = _edges(t)
    assert want <= set(map(tuple, e))
    assert (cnt == 2).all() and len(p) - len(e) + len(t) == 2 and (_det(p, t) > 0).all()
    w = remove_land(p, t, be)
    e, cnt = _edges(w)
    assert set(map(tuple, e[cnt == 1])) == want                 # the coast is exactly the mesh boundary
    c = p[w].sum(axis=1)
    assert in_water(coast, c / np.linalg.norm(c, axis=1, keepdims=True)).all()


def test_poisson_points_spacing_disc():
    dom, h = CASES["disc"]
    hs = scaled(h, 4)
    p = poisson_points(dom, hs, [])
    d, _ = cKDTree(p).query(p, k=2)
    nn = R_EARTH_M * d[:, 1] / hs(p)
    lo, hi = np.percentile(nn, [1, 99])
    assert 0.7 <= lo and hi <= 1.4
    e, _ = _edges(sphere_delaunay(p))                          # Delaunay edges: theory [alpha, 2 alpha] h
    m = p[e[:, 0]] + p[e[:, 1]]
    r = R_EARTH_M * arc_length(p[e[:, 0]], p[e[:, 1]]) / hs(m / np.linalg.norm(m, axis=1, keepdims=True))
    assert np.percentile(r, 1) >= 0.7 and np.percentile(r, 99) <= 1.7


def test_background_mesh_continent_scale4():
    dom, h = CASES["continent"]
    t0 = time.perf_counter()
    m = background_mesh(dom, scaled(h, 4))
    wall = time.perf_counter() - t0
    assert wall < 30.0
    c = m.points[m.tris].sum(axis=1)
    assert in_water(dom, c / np.linalg.norm(c, axis=1, keepdims=True)).all()   # no triangle on the true land
    e, cnt = _edges(m.tris)
    assert cnt.max() == 2
    assert m.n_nodes - len(e) + m.n_tris == 2 - len(dom.loops)                 # sphere minus 3 discs
    assert len(np.unique(m.tris)) == m.n_nodes and (_det(m.points, m.tris) > 0).all()
    assert (m.node_flag[e[cnt == 1].ravel()] == 1).all()


def test_sphere_delaunay_cap_is_the_stereographic_delaunay():
    """Points in a 0.3 rad cap (every point within one hemisphere): the far-cap helpers close the hull and are dropped,
    so the result is an open CCW disc whose triangles are the plane Delaunay triangles of the stereographic image
    (projected from the antipode of the mean: the circle-preserving map) except a few hull slivers."""
    from scipy.spatial import Delaunay

    from meshom.geo import tangent_basis

    p = _random_sphere(40_000, 2)
    p = p[p[:, 2] > np.cos(0.3)]
    t = sphere_delaunay(p)
    assert t.max() < len(p) and (_det(p, t) > 0).all()
    e, cnt = _edges(t)
    assert cnt.max() == 2 and len(p) - len(e) + len(t) == 1                     # a disc: no double cover
    m = p.sum(axis=0) / np.linalg.norm(p.sum(axis=0))
    e1, e2 = tangent_basis(m[None])
    xy = np.column_stack([p @ e1[0], p @ e2[0]]) / (1.0 + p @ m)[:, None]
    ref = set(map(tuple, np.sort(Delaunay(xy).simplices, axis=1)))
    ours = set(map(tuple, np.sort(t, axis=1)))
    assert ours <= ref and len(ours) >= 0.98 * len(ref)


@pytest.mark.parametrize("theta, k, jitter", [(0, 10, 2.0), (15, 0, 2.0), (0, 10, 0.0), (15, 0, 0.0)])
def test_background_mesh_lake_alone(monkeypatch, theta, k, jitter):
    """The seam lakes without their ring and inlet (chi -15 / -13 before the far-cap helpers), also
    with exactly great-circle coasts (jitter 0: the edge recovery hung before): one disc covering the lake once."""
    from meshom import cases_seams
    from meshom.compete import irregular_counts
    from meshom.domain import loop_signed_area
    from meshom.quality.kernel import element_geometry

    monkeypatch.setattr(cases_seams, "JITTER_M", jitter)
    dom, h, _ = cases_seams.seam_rot(theta, k, ring=False)
    m = background_mesh(dom, h, factor=1.0)
    ic = irregular_counts(m)
    assert ic["chi"] == 1 and ic["euler_sum"] == 4 and ic["components"] == 1
    assert np.abs(element_geometry(m).tri_area).sum() == pytest.approx(loop_signed_area(dom.loops[0]), rel=2e-3)
    e, cnt = _edges(m.tris)
    assert cnt.max() == 2 and (_det(m.points, m.tris) > 0).all()
    assert set(map(tuple, e[cnt == 1])) == set(map(tuple, np.sort(loop_edges([m.points[m.node_flag == 1]]), axis=1)))


@pytest.mark.parametrize("method", ["m02_linf_frontal", "m14_blocks", "m14_blocks_mix"])
def test_methods_lake_alone(monkeypatch, method):
    """m02 and m14 on a lake alone with exactly great-circle coasts: one disc (Euler sum 4 chi, chi = 1;
    -28 / -26 before), covering the lake once. m14_blocks_mix: one block, so the partition has no
    separator chain (stitch_mix._resample_try raised ValueError on the empty concatenation before)."""
    from meshom import cases_seams
    from meshom.compete import irregular_counts
    from meshom.domain import loop_signed_area
    from meshom.methods import METHODS
    from meshom.quality.kernel import element_geometry

    monkeypatch.setattr(cases_seams, "JITTER_M", 0.0)
    dom, h, _ = cases_seams.seam_rot(15, 0, ring=False)
    m = METHODS[method](dom, h, {})
    ic = irregular_counts(m)
    assert ic["chi"] == 1 and ic["euler_sum"] == 4 and ic["euler_ok"]
    g = element_geometry(m)
    assert np.abs(g.tri_area).sum() + np.abs(g.quad_area).sum() == pytest.approx(loop_signed_area(dom.loops[0]),
                                                                                  rel=2e-3)


def test_recover_edges_duplicate_point_terminates():
    """A hang (m14 _mix gaps layout): a constrained node that duplicates another point is left out of the
    hull (vt = -1) and the fan walk from it never returned; now a RuntimeError names the constraint and the twin."""
    p = _random_sphere(200, 3)
    t0 = sphere_delaunay(p)
    a, b = t0[0, 0], t0[0, 1]
    q = np.vstack([p, p[a]])                                    # node 200 coincides with node a
    n = len(p)
    cons = np.array([[n, b], [a, t0[0, 2]]])                    # one constraint at each copy: one copy is dropped
    with pytest.raises(RuntimeError, match=rf"constraint 0 \({n}, {b}\): node {n} is in no triangle; it coincides "
                                           rf"with node {a} \(0 m apart\)|constraint 1 .* node {a} is in no triangle"):
        sphere_delaunay(q, cons)


@pytest.mark.parametrize("row", [False, True])
def test_self_touching_coast_split_at_the_touch(row):
    """A self-touching coast (two triangular land lobes meeting at one point, L15) has a
    segment through another coast vertex. It was left unrecovered (orient tie) and remove_land kept both lobes as
    water; now it is split at that vertex: the boundary is the 6 pieces, no land triangle is kept. row=False: the
    hull has the edges to the touching vertex (the fan search of the start node meets it); row=True: a row of water
    points 3 km below the segment removes them (the walk along the segment meets it)."""
    import warnings

    from meshom.delaunay import _fibonacci

    def xyz(xy):
        p = np.column_stack([np.ones(len(xy)), np.asarray(xy) * 1000 / R_EARTH_M])
        return p / np.linalg.norm(p, axis=1, keepdims=True)

    p = xyz(np.array([[-200., 0.], [200., 0.], [200., 200.], [0., 0.], [-200., 200.]])[::-1])
    dom = Domain([p], "self-touching", [True])
    r = _fibonacci(1500)
    P = np.vstack([p, r[in_water(dom, r)]])
    if row:
        P = np.vstack([P, xyz(np.column_stack([np.arange(-195., 200., 10.), np.full(40, -3.)]))])
    be = loop_edges([p])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = sphere_delaunay(P, be)
    e, cnt = _edges(t)
    assert cnt.max() == 2 and (_det(P, t) > 0).all()
    assert set(map(tuple, e[cnt == 1])) == {(0, 1), (1, 2), (2, 3), (1, 3), (1, 4), (0, 4)}
    c = P[t].sum(axis=1)
    assert in_water(dom, c / np.linalg.norm(c, axis=1, keepdims=True)).all()
    assert len(t) == 2 * len(P) - 4 - 2
    k = len(p)                                                  # an extra constraint, the coast given apart
    t2 = sphere_delaunay(P, np.vstack([be, [[k, k + 1]]]), be)
    e2, cnt2 = _edges(t2)
    assert len(t2) == len(t) and set(map(tuple, e2[cnt2 == 1])) == set(map(tuple, e[cnt == 1]))


@pytest.mark.parametrize("lon, lat, n_tri", [([0., 0.1, 0.], [0., 0., 0.1], 1),
                                             ([-0.1, 0.1, 0.1, -0.1], [-0.1, -0.1, 0.1, 0.1], 2)])
def test_small_contour_alone(lon, lat, n_tri):
    """A water contour of 3 points, or of 4 co-circular points, has no 3-d hull (Qhull
    QH6214 / QH6013); the far-cap helpers (M36) close it instead. Also popctl deleting the only interior node."""
    from meshom.geo import lonlat_to_xyz
    from meshom.mesh import Mesh
    from meshom.popctl import popctl
    from meshom.size import Uniform

    b = lonlat_to_xyz(np.array(lon), np.array(lat))
    be = loop_edges([b])
    t = sphere_delaunay(b, be)
    assert len(t) == n_tri and (_det(b, t) > 0).all()
    c = b.mean(axis=0)
    p = np.vstack([b, c / np.linalg.norm(c)])
    m = Mesh(p, sphere_delaunay(p, be), np.zeros((0, 4), dtype=np.int64), np.r_[np.ones(len(b), np.int8), 0])
    out, _ = popctl(m, Uniform(20e3), bedges=be, rounds=1, sweeps=0)
    assert out.n_nodes == len(b) and out.n_tris == n_tri


def _touching_ring():
    """Two triangular land lobes meeting at node 1, clockwise (water outside): segment 3 -> 4 runs through node 1."""
    xy = np.array([[-200., 0.], [200., 0.], [200., 200.], [0., 0.], [-200., 200.]])[::-1] * 1000
    p = np.column_stack([np.ones(len(xy)), xy / R_EARTH_M])
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def test_medial_axis_self_touching_coast():
    """M40 at lfs.medial_axis (it recovers and removes land itself): with the coast samples = the 5 ring nodes (a step
    longer than the ring), the touching segment was not recovered and both lobes stayed water (water_tris = the whole
    sphere, 2 N - 4); now they go."""
    import warnings

    from meshom.lfs import medial_axis
    from meshom.size import Uniform

    p = _touching_ring()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ma = medial_axis([p], Uniform(1e7), Uniform(1.0))
    assert ma.info["samples"] == 5
    assert ma.info["water_tris"] == 2 * (5 + ma.info["ghosts"]) - 4 - 2


def test_cdt_local_self_touching_loop():
    """M40 at m14 continuation.cdt_local (it recovers and removes land itself): no triangle of a lobe is kept."""
    import warnings

    from meshom.delaunay import _fibonacci
    from meshom.methods.m14.continuation import cdt_local

    p = _touching_ring()
    dom = Domain([p], "self-touching", [True])
    r = _fibonacci(1500)
    P = np.vstack([p, r[in_water(dom, r)]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        t = cdt_local(P, len(p))
    c = P[t].sum(axis=1)
    assert in_water(dom, c / np.linalg.norm(c, axis=1, keepdims=True)).all()
    assert (_det(P, t) > 0).all()
