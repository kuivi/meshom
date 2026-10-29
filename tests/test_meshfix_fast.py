"""Fast D (Numba relaxation, cavity kernels, batched engine) and the mesh-fixing pipeline (decisions B138-B141)."""
import numpy as np

from meshom.geo import lonlat_to_xyz
from meshom.mesh import Mesh


def _zone(seed=0, nx=6, ny=5, d=0.1, jitter=0.25):
    rng = np.random.default_rng(seed)
    lon, lat = np.meshgrid(np.arange(nx + 1) * d, np.arange(ny + 1) * d)
    lon = lon + rng.uniform(-jitter, jitter, lon.shape) * d
    lat = lat + rng.uniform(-jitter, jitter, lat.shape) * d
    P = lonlat_to_xyz(lon.ravel(), lat.ravel())
    idx = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    # split two quads into triangles so both element kinds are present
    t = np.array([[q[3, 0], q[3, 1], q[3, 2]], [q[3, 0], q[3, 2], q[3, 3]]])
    q = np.delete(q, 3, axis=0)
    free = np.zeros(len(P), bool)
    free[idx[1:-1, 1:-1].ravel()] = True
    return P, q, t, free


def _h(x):
    return np.full(len(x), 0.1 * np.pi / 180 * 6.371e6)


def test_numba_energy_and_gradient_match_jax():
    import jax.numpy as jnp

    from meshom.meshfix.cavity import nrelax, relax

    P, q, t, free = _zone()
    k = nrelax.prepare(P, free, q, t, _h, 0.2)
    f = nrelax.fg_of(k)
    rng = np.random.default_rng(1)
    v = rng.normal(0, 0.15, 2 * len(k["fidx"]))
    e, g = f(v)
    # the JAX energy on the same zone (relax.relax's padded set-up, evaluated through its jitted value_and_grad)
    captured = {}
    orig = relax.minimize

    def grab(fun, v0, **kw):
        captured["fun"] = fun
        captured["n"] = len(v0)
        return orig(fun, v0, **{**kw, "options": dict(maxiter=0)})

    relax.minimize = grab
    try:
        relax.relax(P, free, q, t, _h, 0.2, bias=k["bias"])
    finally:
        relax.minimize = orig
    vp = np.zeros(captured["n"])
    vp[: len(v)] = v
    ej, gj = captured["fun"](vp)
    assert abs(e - ej) < 1e-9 * max(1.0, abs(ej))
    assert np.allclose(g, gj[: len(v)], rtol=1e-7, atol=1e-9)
    # and a finite-difference check of the Numba gradient itself
    i = 3
    hstep = 1e-6
    vp, vm = v.copy(), v.copy()
    vp[i] += hstep
    vm[i] -= hstep
    assert abs((f(vp)[0] - f(vm)[0]) / (2 * hstep) - g[i]) < 1e-5 * max(1.0, abs(g[i]))


def test_numba_relax_matches_jax_relax():
    from meshom.meshfix.cavity import nrelax, relax

    P, q, t, free = _zone(seed=2, jitter=0.3)
    Xa, ia = relax.relax(P, free, q, t, _h, 0.2)
    Xb, ib = nrelax.relax(P, free, q, t, _h, 0.2)
    assert ib["e1"] < ib["e0"]
    assert abs(ia["e1"] - ib["e1"]) < 1e-6 * max(1.0, ia["e1"])
    assert np.abs(Xa - Xb).max() < 1e-6


def _defect_mesh(seed=3, n=20, d=0.1, jitter=0.18):
    P, q, t, _ = _zone(seed=seed, nx=n, ny=n, d=d, jitter=jitter)
    # a second triangle pair in another place
    k = 5 * n + 7
    t = np.vstack([t, [[q[k, 0], q[k, 1], q[k, 2]], [q[k, 0], q[k, 2], q[k, 3]]]])
    q = np.delete(q, k, axis=0)
    return Mesh(P, t.astype(np.int64), q.astype(np.int64), np.zeros(len(P), np.int8))


def _run_engine(mesh, **o):
    from meshom.meshfix.cavity.engine import Engine
    from meshom.size import Uniform

    h = Uniform(0.1 * np.pi / 180 * 6.371e6)
    eng = Engine(mesh, h, 0.2, np.zeros(mesh.n_nodes, bool), dict(passes=3, **o))
    out = eng.run()
    return out, eng.stats


def test_fast_engine_cache_and_workers_give_the_serial_result():
    """B140: the cache and the worker processes change the work, not the result."""
    mesh = _defect_mesh()
    ref, s0 = _run_engine(mesh, cache=False)
    assert s0["accepted"] > 0
    st = []
    for o in (dict(cache=True), dict(cache=True, workers=2, inflight=3)):
        out, s = _run_engine(mesh, **o)
        st.append(s)
        assert s["accepted"] == s0["accepted"], o
        assert out.n_nodes == ref.n_nodes and np.array_equal(out.quads, ref.quads), o
        assert np.array_equal(out.tris, ref.tris), o
        assert np.abs(out.points - ref.points).max() < 1e-12, o
    assert st[0]["seeds_cached"] > 0 and st[0]["evaluations"] < s0["evaluations"]


def _case(mesh):
    from meshom.mesh import build_topology
    from meshom.meshfix.cases import Case, boundary_edges
    from meshom.size import Uniform

    h = Uniform(0.1 * np.pi / 180 * 6.371e6)
    kind = build_topology(mesh).boundary_node.astype(np.int8) * 2        # the whole outline artificial (a cut-out)
    be = boundary_edges(mesh)
    z = np.zeros((0, 2, 3))
    return Case("synthetic", mesh, kind, h, h(mesh.points), z, z, mesh.points[:1], be, 0.2, {}, {}, {"h_pkl": "-"})


def test_pipeline_rebase_keeps_the_artificial_boundary_between_stages():
    """B141: after a topology stage the kinds are rebuilt (kind 2 by position, art_edges remapped, h_node new)."""
    from scipy.spatial import cKDTree

    import meshom.meshfix.pipeline as pl
    from meshom.meshfix.cases import boundary_edges
    from meshom.meshfix.pipeline import rebase, run_stages

    case = _case(_defect_mesh())
    art = case.mesh.points[case.kind == 2]
    seen = []
    fix0 = pl.stage_fix

    def spy(s):
        f = fix0(s)

        def g(c, o):
            seen.append(cKDTree(c.mesh.points[c.kind == 2]).query(art)[0].max())     # the stage's kind 2
            return f(c, o)
        return g

    pl.stage_fix = spy
    try:
        mesh, recs = run_stages(case, ["d", "a", "d"], {"d": dict(workers=1, passes=2)}, verbose=False)
    finally:
        pl.stage_fix = fix0
    assert seen == [0.0, 0.0, 0.0]                  # every stage gets the outline as kind 2
    assert [r["stage"] for r in recs] == ["d", "a", "d"]
    assert recs[0]["nodes"] != case.mesh.n_nodes or recs[0]["quads"] != case.mesh.n_quads
    c2 = rebase(case, mesh)
    assert (c2.kind == 2).sum() == (case.kind == 2).sum() and (c2.kind == 1).sum() == 0
    assert cKDTree(mesh.points[c2.kind == 2]).query(art)[0].max() == 0.0     # the outline did not move
    assert len(c2.h_node) == mesh.n_nodes and len(c2.kind) == mesh.n_nodes
    # the remapped artificial edges join kind-2 nodes and are boundary edges of the new mesh
    be = {tuple(sorted(e)) for e in boundary_edges(mesh).tolist()}
    assert all(tuple(sorted(e)) in be for e in c2.art_edges.tolist())


def test_numba_convexify_matches_python():
    """B139: the kernel gives convexify_py's cavity, stop reason and read set on every seed."""
    from meshom.meshfix.cavity import cavity as cv
    from meshom.meshfix.cavity.work import Work

    mesh = _defect_mesh(jitter=0.1)
    W = Work(mesh)
    W.art = np.zeros(len(W.P), bool)
    W.art[np.flatnonzero(W.bnode)[::3]] = True
    n_ok = 0
    for e in range(0, mesh.n_quads + mesh.n_tris, 3):
        for coast, me in ((False, 40), (True, 60), (True, 300)):
            r0, r1 = set(), set()
            W.last_stop = None
            a = cv.convexify_py(W, {e}, me, coast, r0)
            s0 = W.last_stop
            W.last_stop = None
            b = cv.convexify(W, {e}, me, coast, r1)
            assert a == b and W.last_stop == s0 and r0 <= r1, (e, coast, me)
            n_ok += a is not None
    assert n_ok > 10


def test_select_applies_the_budget_at_the_current_totals():
    """B138: a contender rejected by the global budget at evaluation may be accepted later, and the other way."""
    from meshom.meshfix.cavity.engine import Engine
    from meshom.size import Uniform

    mesh = _defect_mesh()
    eng = Engine(mesh, Uniform(1.1e4), 0.2, np.zeros(mesh.n_nodes, bool), {})
    eng.budget = {"bad": 10, "jump": 10}
    old = dict(inv=0, bad=2, jump=0, lnsh_max=0.1, ratio_max=1.1, n120=1, tri_worst=0, cost=2, worst=125, J=20.0)

    def c(J, bad):
        return dict(old=old, met=dict(old, bad=bad, n120=0, worst=110, J=J), name="x")

    a, b = c(10.0, 6), c(12.0, 2)
    eng.tot = {"bad": 5, "jump": 0}
    assert eng.select([a, b]) is a                  # 5 + 6 - 2 = 9 <= 10: the lower J wins
    eng.tot = {"bad": 8, "jump": 0}
    assert eng.select([a, b]) is b                  # 8 + 4 > 10: a is over the budget now
    eng.tot = {"bad": 11, "jump": 0}
    assert eng.select([a, b]) is None
