"""B64: a coast quad whose largest angle sits at a coast corner is split through that corner (m14_blocks_rep)."""
import numpy as np

from meshom.mesh import Mesh
from meshom.methods.m14.coastfix import _deg, coast_fix, split_coast_quads


def _sphere(xy_km: np.ndarray) -> np.ndarray:
    lon, lat = xy_km[:, 0] / 6371.0, xy_km[:, 1] / 6371.0
    return np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]


def _kite(corner_deg: float) -> Mesh:
    """One quad c, a, o, b (CCW): two coast edges of 100 km meeting at c with the angle corner_deg, o opposite."""
    h = np.radians(corner_deg) / 2
    a, b = 100 * np.array([np.cos(-h), np.sin(-h)]), 100 * np.array([np.cos(h), np.sin(h)])
    xy = np.array([[0, 0], a, [150, 0], b], dtype=float)
    return Mesh(_sphere(xy), np.zeros((0, 3), np.int64), np.array([[0, 1, 2, 3]]), np.ones(4, np.int8))


def test_corner_quad_split_into_two_good_triangles():
    m = _kite(147.0)
    assert _deg(m.points, m.quads).max() > 145
    out, n = split_coast_quads(m)
    assert n == 1 and out.n_quads == 0 and out.n_tris == 2
    ta = _deg(out.points, out.tris)
    assert ta.max() < 120 and ta.min() >= 30
    assert sorted(out.tris[:, 0].tolist()) == [0, 0]          # both triangles share the corner c (the corner halves)


def test_quad_below_target_kept_and_no_cluster_made():
    out, n = split_coast_quads(_kite(120.0))
    assert n == 0 and out.n_quads == 1
    m = _kite(147.0)                                          # a triangle on the far side: splitting makes a cluster
    p = np.vstack([m.points, _sphere(np.array([[200.0, 60.0]]))])
    m2 = Mesh(p, np.array([[2, 4, 3]]), m.quads, np.ones(5, np.int8))
    out, n = split_coast_quads(m2)
    assert out.n_tris - m2.n_tris <= 1                        # never two new triangles beside an old one


def _quad(corner_deg: float, la: float, lb: float, o_km) -> Mesh:
    """One quad c, a, o, b (CCW): coast edges of la and lb km meeting at c with the angle corner_deg, o at o_km."""
    h = np.radians(corner_deg) / 2
    xy = np.array([[0, 0], la * np.array([np.cos(-h), np.sin(-h)]), o_km, lb * np.array([np.cos(h), np.sin(h)])])
    return Mesh(_sphere(xy), np.zeros((0, 3), np.int64), np.array([[0, 1, 2, 3]]), np.ones(4, np.int8))


def test_split_limits_target_and_triangle_minimum():
    """A kite of 140 deg (between the 135 deg target and 145) is split; a skewed 140 deg quad whose split leaves a
    triangle angle of 29 deg (< TRI_MIN_DEG 30, still above the 20 deg gate) is kept."""
    out, n = split_coast_quads(_kite(140.0))
    assert n == 1 and out.n_tris == 2 and _deg(out.points, out.tris).min() >= 30
    m = _quad(140.0, 40.0, 100.0, [60.0, 30.0])
    assert abs(_deg(m.points, m.quads).max() - 140.0) < 0.5
    q = m.quads[0]
    ta = _deg(m.points, np.array([[q[0], q[1], q[2]], [q[0], q[2], q[3]]]))
    assert 20 < ta.min() < 30 and ta.max() < 120                # the split the limit refuses
    out, n = split_coast_quads(m)
    assert n == 0 and out.n_quads == 1


def _fan_in_ring() -> Mesh:
    """Three triangles of a pentagon fan surrounded by five quads near the pole and the
    antimeridian; every angle within the hard gates."""
    from tests.helpers import local_polygon

    xy = np.array([[-25.52588, 52.24088], [-71.68019, 9.16002], [-33.17094, -52.19362], [26.86649, -62.38890],
                   [60.10123, -5.22315]]) * 1000
    edge = np.roll(xy, -1, axis=0) - xy
    nr = np.c_[edge[:, 1], -edge[:, 0]] / np.linalg.norm(edge, axis=1)[:, None]
    prev = np.roll(nr, 1, axis=0)
    outer = xy + 25e3 * (prev + nr) / (1 + (prev * nr).sum(axis=1))[:, None]
    q = np.array([[i, 5 + i, 5 + (i + 1) % 5, (i + 1) % 5] for i in range(5)])
    return Mesh(local_polygon(179.9, 89.8, np.vstack([xy, outer])), np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]]), q,
                np.r_[np.zeros(5, np.int8), np.ones(5, np.int8)])


def _hard_failures(m: Mesh) -> list:
    from meshom.config import DEFAULT_GATES
    from meshom.quality.metrics import check_gates, compute_quality

    g = {k: DEFAULT_GATES[k] for k in ("tri_max_angle_hard_deg", "tri_min_angle_hard_deg", "quad_max_angle_hard_deg",
                                       "quad_sj_hard")}
    return [r["gate"] for r in check_gates(compute_quality(m).summary, g) if not r["passed"]]


def test_coast_fix_cluster_break_keeps_the_hard_limits():
    """coast_fix after the final fit (every node fixed) breaks the cluster of 3 only with elements inside the hard
    gates: the pentagon re-split chose a 131.75 deg triangle before."""
    from meshom.mesh import build_topology

    m = _fan_in_ring()
    assert not _hard_failures(m)
    out, fixed, rec = coast_fix(m, np.ones(m.n_nodes, dtype=bool))
    assert not _hard_failures(out), _deg(out.points, out.tris)
    assert rec["clusters_broken"] == 1 and out.n_tris == 1 and fixed.all()
    assert np.array_equal(out.points, m.points)
    bnd = lambda x: {tuple(e) for e in (lambda t: t.edges[t.edge_elems[:, 1] < 0])(build_topology(x))}
    assert bnd(out) == bnd(m)


def test_cluster_break_triangle_limits():
    """break_clusters with tri_lim: no new triangle outside [min, max]; coast_fix's limits are the hard gates."""
    from meshom.config import DEFAULT_GATES
    from meshom.methods.m14 import coastfix
    from meshom.methods.m14.stitch_mix import break_clusters

    assert (coastfix.TRI_HARD_MIN_DEG, coastfix.TRI_MAX_DEG) == (DEFAULT_GATES["tri_min_angle_hard_deg"],
                                                                 DEFAULT_GATES["tri_max_angle_hard_deg"])
    m = _fan_in_ring()
    z3, z5 = np.zeros(3, np.int64), np.zeros(5, np.int64)
    t, _, _, _, n, _ = break_clusters(m.points, m.tris, m.quads, z3, z5, quad_max=135.0, free_from=m.n_nodes,
                                      tri_lim=(20.0, 120.0))
    assert n == 1 and len(t) == 1 and 33.0 < _deg(m.points, t).min() < 34.0
    t, _, _, _, n, _ = break_clusters(m.points, m.tris, m.quads, z3, z5, quad_max=135.0, free_from=m.n_nodes,
                                      tri_lim=(34.0, 120.0))
    assert n == 0 or _deg(m.points, t).min() >= 34.0
