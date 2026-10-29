"""Competitor A of the mesh fixing (meshom.meshfix.geom, decisions B122-B125): the energy's gradient, the
optimiser's guarantees (fixed nodes kept, no inverted element, the no-worse guard, coast nodes stay on the coast) and
the valence-3 floor report."""
import numpy as np

from meshom.geo import stereographic_inverse
from meshom.mesh import Mesh, build_topology, orient_ccw, signed_area
from meshom.meshfix.geom.core import Problem, _elem_angles, optimize_geometry
from meshom.meshfix.geom.floor import floor_report
from meshom.size import Uniform

L = 10_000.0      # m


def _grid(n: int = 8, shear: float = 0.3, seed: int = 0, tri: bool = True) -> Mesh:
    """n x n quad grid of spacing L near 60 N, interior nodes sheared (x += shear L y-row parity) and jittered; the
    last quad of the first row split into two triangles."""
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    x, y = ii.ravel().astype(float) * L, jj.ravel().astype(float) * L
    inner = (ii.ravel() > 0) & (ii.ravel() < n) & (jj.ravel() > 0) & (jj.ravel() < n)
    x[inner] += shear * L * (jj.ravel()[inner] % 2) + rng.uniform(-0.1, 0.1, inner.sum()) * L
    y[inner] += rng.uniform(-0.1, 0.1, inner.sum()) * L
    pts = stereographic_inverse(x, y, 0.0, 60.0)
    idx = lambda i, j: i * (n + 1) + j                      # noqa: E731
    quads = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                      for i in range(n) for j in range(n)])
    tris = np.zeros((0, 3), np.int64)
    if tri:
        q = quads[-1]
        quads = quads[:-1]
        tris = np.array([[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
    return orient_ccw(Mesh(pts, tris, quads, np.zeros(len(pts), np.int8)))


def test_gradient_matches_finite_differences():
    m = _grid()
    pb = Problem(m, Uniform(L), opts={"wall": 0.05, "guard_hard": 90.0})   # tight wall: its terms are active
    fg = pb.round(m.points.copy())
    rng = np.random.default_rng(1)
    v = rng.normal(0.0, 0.02, 2 * len(pb.free))
    f, g, ok = fg(v)
    g = g.copy()
    assert ok
    parts = pb.fg(v, parts=True)
    assert parts["edges"] > 0 and parts["coast"] > 0 and parts["corners"] > 0
    for j in rng.choice(len(v), 20, replace=False):
        e = 1e-6
        vp, vm = v.copy(), v.copy()
        vp[j] += e
        vm[j] -= e
        fd = (fg(vp)[0] - fg(vm)[0]) / (2 * e)
        assert abs(fd - g[j]) <= 1e-4 * max(1.0, abs(fd)), (j, g[j], fd)


def test_optimise_improves_and_keeps_rules():
    m = _grid()
    topo = build_topology(m)
    fixed = np.zeros(m.n_nodes, bool)
    fixed[np.flatnonzero(topo.boundary_node)[::3]] = True            # some boundary nodes fixed, the rest slide
    hi0, _ = _elem_angles(m)
    out, info = optimize_geometry(m, Uniform(L), fixed=fixed, opts={"wall": 0.3}, log=lambda *a: None)
    hi1, _ = _elem_angles(out)
    q0, q1 = hi0[m.n_tris:], hi1[m.n_tris:]
    assert q1.max() < q0.max() - 5.0                                  # the sheared grid straightens
    assert (signed_area(out) > 0).all()
    assert np.array_equal(out.points[fixed], m.points[fixed])
    guard = np.r_[np.maximum(hi0[:m.n_tris], 90.0), np.maximum(q0, 110.0)]
    assert (hi1 <= guard + info["opts"]["guard_tol"]).all()          # no element worse than the guard allows
    # slide nodes stay on the input's boundary polyline (here the straight sides of the grid: within 0.1 L)
    bnd = topo.boundary_node & ~fixed
    d = np.linalg.norm(out.points[bnd] - m.points[bnd], axis=1) * 6.371e6
    assert d.max() < 0.3 * L                                          # coast_move 0.25 h (+ soft overshoot)


def test_floor_report_counts_valence3():
    assert floor_report(_grid(tri=False))["n3_nodes"] == 0
    # three kites around one node c (corners at c 120 deg each): c is an interior pure-quad valence-3 node
    ang = np.radians([90.0, 210.0, 330.0])
    xy = [(0.0, 0.0)]
    for a in ang:
        xy.append((L * np.cos(a), L * np.sin(a)))
        b = a + np.radians(60.0)
        xy.append((L / np.cos(np.radians(60.0)) * np.cos(b), L / np.cos(np.radians(60.0)) * np.sin(b)))
    P = np.array(xy)
    pts = stereographic_inverse(P[:, 0], P[:, 1], 0.0, 0.0)
    quads = np.array([[0, 1, 2, 3], [0, 3, 4, 5], [0, 5, 6, 1]])
    star = orient_ccw(Mesh(pts, np.zeros((0, 3), np.int64), quads, np.zeros(len(pts), np.int8)))
    r = floor_report(star)
    assert r["n3_nodes"] == 1
    assert abs(r["n3_max_corner_max"] - 120.0) < 0.1


def test_lbfgs_freezes_blocked_variables():
    """Two variables, minimum at (2, 2), a hard limit x0 <= 1 (ok = False beyond): without freezing the line search
    stalls at the limit; freezing x0 there lets x1 reach its minimum."""
    from meshom.meshfix.geom.core import lbfgs

    frozen = np.zeros(2, bool)

    def fg(v):
        g = 2.0 * (v - 2.0)
        g[frozen] = 0.0
        return float(((v - 2.0) ** 2).sum()), g, bool(v[0] <= 1.0)

    def freeze(v):
        new = int(v[0] > 1.0 - 0.05 and not frozen[0])
        frozen[0] |= bool(new)
        return new

    x, f, it, reason, used = lbfgs(fg, 2, 200, 5, 0.5, 0.5, window=5, wtol=1e-12, freeze=freeze)
    assert frozen[0] and used >= 1
    assert x[0] <= 1.0 and abs(x[1] - 2.0) < 1e-6


def test_reference_coast_wall_gradient_matches_finite_differences():
    """B146: the backward wall of the reference vertices (active: coast_move 0) has the right gradient."""
    from meshom.meshfix.cases import _segments, boundary_edges

    m = _grid()
    ref = _segments(m, boundary_edges(m))
    pb = Problem(m, Uniform(L), opts={"wall": 0.05, "guard_hard": 90.0, "coast_move": 0.0, "ref_margin": 0.0,
                                      "ref_margin2": 0.0}, ref=ref)
    fg = pb.round(m.points.copy())
    rng = np.random.default_rng(1)
    v = rng.normal(0.0, 0.05, 2 * len(pb.free))
    f, g, ok = fg(v)
    g = g.copy()
    assert ok and pb.fg(v, parts=True)["ref"] > 0 and len(pb.V) > 0
    for j in rng.choice(len(v), 30, replace=False):
        e = 1e-6
        vp, vm = v.copy(), v.copy()
        vp[j] += e
        vm[j] -= e
        fd = (fg(vp)[0] - fg(vm)[0]) / (2 * e)
        assert abs(fd - g[j]) <= 1e-4 * max(1.0, abs(fd)), (j, g[j], fd)


def test_a_keeps_the_case_coast_whatever_moved_it_before():
    """B146: A after a stage that moved the coast stacked its own 0.25 h on it
    (three A runs in a row without ref: 0.245, 0.286, 0.307 h on this grid). With ref = the case's coast every
    reference vertex stays within coast_move h of the coast edges and every coast node within coast_move h of the
    reference, also with the soft walls switched off (the hard check after each round puts nodes back)."""
    from meshom.meshfix.cases import _segments, boundary_edges, segment_distance_km

    m = _grid(shear=0.4)
    ref = _segments(m, boundary_edges(m))
    V = np.unique(ref.reshape(-1, 3), axis=0)
    bnd = build_topology(m).boundary_node
    for opts in ({"wall": 0.3}, {"wall": 0.3, "ref_margin": -1.0, "ref_margin2": -1.0}):
        cur, rev = m, 0
        for _ in range(3):
            cur, info = optimize_geometry(cur, Uniform(L), opts=opts, log=lambda *a: None, ref=ref)
            rev += sum(r["ref_reverted"] for r in info["rounds"])
            back = segment_distance_km(V, _segments(cur, boundary_edges(cur))) * 1e3 / L
            fwd = segment_distance_km(cur.points[bnd], ref) * 1e3 / L
            assert back.max() <= 0.25 + 1e-9 and fwd.max() <= 0.25 + 1e-9, (opts, back.max(), fwd.max())
        if "ref_margin" in opts:
            assert rev > 0                                    # the hard check did the work
