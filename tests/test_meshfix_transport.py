"""Core topology and transaction checks for competitor B."""
import numpy as np
import pytest

from meshom.mesh import build_topology, signed_area
from meshom.meshfix.transport import classify, splits, trace_chord
from meshom.optimize import _Work, _convex_ok, boundary_loop
from test_meshfix import _fan, _grid


def test_chord_to_boundary():
    work = _Work(_grid(5, 3))
    path, kind, terminal = trace_chord(work, 1, (1, 2))
    assert path == [1, 4, 7, 10, 13]
    assert kind == 'boundary' and terminal == -1


def test_recast_parity_and_undo():
    mesh = _fan()
    work = _Work(mesh)
    star = work.at(0)
    loop = boundary_loop([work.cells(e) for e in star])
    assert len(loop) == 6
    choices = splits(loop, 2)
    assert {sum(len(c) == 3 for c in p) for p in choices} == {0, 2}
    accepted = [p for p in choices if work.patch_ok(star, p) and _convex_ok(work.P, p)]
    assert accepted
    # A complete recast is a disk with the same directed boundary; removing the
    # centre is allowed, removing any coast node is not.
    for p in accepted:
        old_inc = work.ninc.copy()
        before = [work.cells(e) for e in star]
        rec = work.apply(star, p)
        candidate, _, _ = work.to_mesh()
        topo = build_topology(candidate)
        assert signed_area(candidate).min() > 0
        assert topo.boundary_node.sum() == build_topology(mesh).boundary_node.sum()
        assert candidate.n_nodes == mesh.n_nodes - 1
        work.undo(rec)
        assert np.array_equal(work.ninc, old_inc)
        assert [work.cells(e) for e in star] == before
        assert work.alive.all()


def test_classifier_triangle_separate():
    result = classify(_fan())
    assert result['n3'] == 1 and result['n5'] == 0
    assert sum(result['triangle_patterns'].values()) == 1


def _hybrid_star():
    from meshom.geo import stereographic_inverse
    from meshom.mesh import Mesh
    angles = np.radians([0, 60, 135, 210, 285])
    xy = [(0., 0.)]
    for k, a in enumerate(angles):
        xy.append((np.cos(a), np.sin(a)))
        if k > 1:
            b = (angles[k - 1] + a) / 2
            r = 1 / np.cos((a - angles[k - 1]) / 2)
            xy.insert(len(xy) - 1, (r * np.cos(b), r * np.sin(b)))
    b = np.radians(322.5)
    r = 1 / np.cos(np.radians(37.5))
    xy.append((r * np.cos(b), r * np.sin(b)))
    xy = np.asarray(xy) * 10000
    pts = stereographic_inverse(xy[:, 0], xy[:, 1], 0., 0.)
    # a=1, b=2, c=4, d=6, e=8; bridge nodes 3,5,7,9.
    quads = np.array([[0, 2, 3, 4], [0, 4, 5, 6], [0, 6, 7, 8], [0, 8, 9, 1]])
    return Mesh(pts, np.array([[0, 1, 2]]), quads, np.r_[np.int8(0), np.ones(9, np.int8)])


@pytest.mark.parametrize("axis_roll", [0, 1, 2])
def test_hybrid_ve_ev_inverse(axis_roll):
    from meshom.meshfix.transport.operators import Work, split_triangle_tip, collapse_triangle_edge
    mesh = _hybrid_star()
    mesh.points = np.roll(mesh.points, axis_roll, axis=1)
    work = Work(mesh)
    fresh = work.new_node(work.P[0])
    rm, new, moves = split_triangle_tip(work, 0, 0, 6, fresh)
    assert work.patch_ok(rm, new)
    rec = work.apply(rm, new, moves)
    assert _convex_ok(work.P, new)
    t = next(e for e in rec[1] if work.nv[e] == 3)
    rm2, new2, moves2 = collapse_triangle_edge(work, t, 0)
    assert work.patch_ok(rm2, new2)
    rec2 = work.apply(rm2, new2, moves2)
    assert _convex_ok(work.P, new2)
    result, old, _ = work.to_mesh()
    assert result.n_nodes == mesh.n_nodes
    assert result.n_tris == 1 and result.n_quads == 4
    assert {frozenset(old[t]) for t in result.tris} == {frozenset(t) for t in mesh.tris}
    assert {frozenset(old[q]) for q in result.quads} == {frozenset(q) for q in mesh.quads}
    # Both complete operations are individually reversible, including positions.
    work.undo(rec2)
    work.undo(rec)
    work.discard_node(fresh)
    assert np.array_equal(work.P[:mesh.n_nodes], mesh.points)
    assert np.array_equal(work.ninc[:mesh.n_nodes], build_topology(mesh).elems_per_node)


def test_ve_only_splits_the_tip_into_two_regular_nodes():
    """A triangle tip has four quad wedges: a VE partitions them 2+2.

    A 3+1 cut hides an extra 3/5 pair at triangle corners instead of moving
    the single dislocation through an otherwise regular grid.
    """
    from meshom.meshfix.transport.operators import Work, split_triangle_tip
    work = Work(_hybrid_star())
    fresh = work.new_node(work.P[0])
    assert split_triangle_tip(work, 0, 0, 8, fresh) is None
    rm, new, moves = split_triangle_tip(work, 0, 0, 6, fresh)
    work.apply(rm, new, moves)
    assert work.ninc[0] == work.ninc[fresh] == 4


@pytest.mark.parametrize("polygon", ["hex", "pent"])
def test_artificial_boundary_cannot_absorb_incidence(polygon):
    """A real coast can change valence/composition; a cut-out port cannot."""
    from meshom.geo import stereographic_inverse
    from meshom.mesh import Mesh
    from meshom.meshfix.cases import whole_case
    from meshom.meshfix.transport import DEFAULTS, _trial
    from meshom.meshfix.transport.operators import Work
    from meshom.size import Uniform
    n = 6 if polygon == "hex" else 5
    a = np.arange(n) * 2 * np.pi / n
    p = stereographic_inverse(10000 * np.cos(a), 10000 * np.sin(a), 0., 0.)
    if polygon == "hex":
        tris = np.empty((0, 3), np.int64)
        quads = np.array([[0, 1, 2, 3], [3, 4, 5, 0]])
        new = [(1, 2, 3, 4), (4, 5, 0, 1)]
    else:
        tris = np.array([[0, 1, 2]])
        quads = np.array([[0, 2, 3, 4]])
        new = [(0, 3, 4), (0, 1, 2, 3)]
    mesh = Mesh(p, tris, quads, np.ones(n, np.int8))
    case = whole_case(polygon, mesh, Uniform(10000), .2, {})
    case.kind[[0, 1]] = 2
    case.art_edges = np.array([[0, 1]])
    w = Work(mesh, case.fixed)
    old = [w.cells(e) for e in (0, 1)]
    assert _trial(w, [0, 1], new, case, DEFAULTS, set(range(n)), {}, fit=False) is None
    assert np.array_equal(w.P[:n], mesh.points)
    assert [w.cells(e) for e in (0, 1)] == old
    case.kind[:] = 1
    case.art_edges = np.empty((0, 2), np.int64)
    w = Work(mesh, case.fixed)
    assert _trial(w, [0, 1], new, case, DEFAULTS, set(range(n)), {}, fit=False) is not None


# --------------------------------------------------------------------------------------------- the fast B (B150-B152)

def test_nfit_equals_smooth_fit():
    """B151: B's relaxation in one Numba call gives smooth_fit's round to the last bit (mixed patch, fixed nodes)."""
    from meshom.mesh import Mesh
    from meshom.meshfix.transport.nfit import fit_sub
    from meshom.size import Uniform
    from meshom.size_fit import smooth_fit
    from test_meshfix import L

    rng = np.random.default_rng(1)
    fo = dict(max_iter=100, w_angle=2.0, w_wall=200.0, w_tol=50.0, ftol=1e-9)
    moved = 0
    for k in range(6):
        m = _grid(6, 5)
        P = m.points + rng.normal(scale=0.12 * L / 6.371e6, size=m.points.shape)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        q, tris = m.quads, np.zeros((0, 3), np.int64)
        if k % 2:
            tris = np.array([[q[7, 0], q[7, 1], q[7, 2]], [q[7, 0], q[7, 2], q[7, 3]]])
            q = np.delete(q, 7, axis=0)
        fixed = np.zeros(len(P), bool)
        fixed[rng.choice(len(P), 4, replace=False)] = True
        h = Uniform(L * (0.8 + 0.4 * rng.random()))
        a, _ = smooth_fit(Mesh(P, tris, q, np.zeros(len(P), np.int8)), h, 0.27, fixed, rounds=1, **fo)
        b, info = fit_sub(P, tris, q, fixed, h, 0.27, **fo)
        assert np.array_equal(a.points, b)
        moved += info[2] > 1
    assert moved >= 3


def test_undo_restores_the_node_rows():
    """B150: apply + undo leaves the node -> element rows in their order (evaluations are pure)."""
    from meshom.meshfix.transport.operators import Work

    work = Work(_grid(4, 4))
    inc, ninc = work.inc.copy(), work.ninc.copy()
    tried = 0
    for e in range(work.m):
        for f in work.nbrs(e):
            loop = boundary_loop([work.cells(e), work.cells(f)])
            for p in splits(loop, 2):
                if work.patch_ok([e, f], p):
                    work.undo(work.apply([e, f], p))
                    tried += 1
                    assert np.array_equal(work.inc, inc) and np.array_equal(work.ninc, ninc)
    assert tried > 20


def _b_case():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_meshfix_coast import _coast_case
    from test_meshfix_fast import _defect_mesh

    return _coast_case(_defect_mesh(seed=4, jitter=0.2))


def test_fast_b_cache_and_workers_give_the_serial_result():
    """B150: the cache and the worker processes change the work, not the result (bitwise)."""
    from meshom.meshfix.transport import fix

    case = _b_case()
    ref, r0 = fix(case, dict(workers=1, cache=False, passes=4))
    acc = sum(p.get(k, 0) for p in r0["passes"] for k in ("recast", "pair", "glide", "EV", "VE"))
    assert acc >= 2
    for o in (dict(workers=1, cache=True), dict(workers=2, inflight=2), dict(workers=3, window=2)):
        out, r = fix(case, dict(passes=4, **o))
        assert [{k: v for k, v in p.items() if k != "wall_s"} for p in r["passes"]] == \
            [{k: v for k, v in p.items() if k != "wall_s"} for p in r0["passes"]], o
        assert np.array_equal(out.quads, ref.quads) and np.array_equal(out.tris, ref.tris), o
        assert np.array_equal(out.points, ref.points), o
    assert r["stats"]["evaluations"] >= 1


def test_fast_b_screen_and_numba_fit_keep_the_answer():
    """B151, B152: the topology screen skips only fits _allowed rejects; the Numba fit equals smooth_fit's."""
    from meshom.meshfix.transport import fix

    case = _b_case()
    ref, r0 = fix(case, dict(workers=1, passes=2))
    for o in (dict(screen=False), dict(fit="smooth_fit")):
        out, r = fix(case, dict(workers=1, passes=2, **o))
        assert np.array_equal(out.quads, ref.quads) and np.array_equal(out.tris, ref.tris), o
        assert np.array_equal(out.points, ref.points), o
