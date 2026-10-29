"""Core operations of the coast method e_coast (meshom/meshfix/coast; decisions B142-B145)."""
import numpy as np

from meshom.geo import lonlat_to_xyz
from meshom.mesh import Mesh, build_topology, signed_area
from meshom.meshfix.cavity import patterns as pt
from meshom.meshfix.coast import ops
from meshom.meshfix.coast.engine import CWork
from meshom.meshfix.coast.refcoast import RefCoast


def ladder_mesh(a=6, c=5, R=4, d=0.1):
    """A rectangle of R rows, a edges at the bottom and c at the top: one column-end triangle per a - c."""
    def line(lon0, lat0, lon1, lat1, n):
        t = np.linspace(0, 1, n + 1)
        return lonlat_to_xyz(lon0 + t * (lon1 - lon0), lat0 + t * (lat1 - lat0))

    W_, H_ = a * d, R * d
    B, T = line(0, 0, W_, 0, a), line(0, H_, W_, H_, c)
    L, Rr = line(0, 0, 0, H_, R), line(W_, 0, W_, H_, R)
    P = np.vstack([B, T[1:-1], L[1:-1], Rr[1:-1]])
    ib = list(range(a + 1))                               # node ids: bottom 0..a, then the inner nodes of the sides
    nB = a + 1
    top_mid = list(range(nB, nB + c - 1))
    left_mid = list(range(nB + c - 1, nB + c - 1 + R - 1))
    right_mid = list(range(nB + c - 1 + R - 1, nB + c - 1 + 2 * (R - 1)))
    tl, tr = len(P), len(P) + 1
    P = np.vstack([P, lonlat_to_xyz(np.array([0.0, W_]), np.array([H_, H_]))])
    bottom, top = ib, [tl] + top_mid + [tr]
    left, right = [0] + left_mid + [tl], [a] + right_mid + [tr]
    F = pt.Fill(P)
    name, nrow, bases = pt.patch_plans(F, bottom, right, top, left)[0]
    cells = pt.build_patch(F, bottom, right, top, left, nrow, bases)
    X = np.vstack([P, np.array(F.X).reshape(-1, 3)])
    n0 = len(P)
    cells = [tuple(v if v >= 0 else n0 + (-1 - v) for v in cc) for cc in cells]
    tris = np.array([cc for cc in cells if len(cc) == 3], np.int64).reshape(-1, 3)
    quads = np.array([cc for cc in cells if len(cc) == 4], np.int64).reshape(-1, 4)
    return Mesh(X, tris, quads, np.zeros(len(X), np.int8)), set(bottom), set(top)


def _work(m):
    W = CWork(m)
    W.art = np.zeros(len(W.P), bool)
    return W


def _apply(W, r):
    rm, X, cells, cn, moves, _ = r
    rec = W.replace(rm, X, cells, cn)
    assert rec is not None
    for v, x in moves.items():
        W.P[v] = x
    for j, i in enumerate(rec[1]):
        W.P[i] = X[j]
    return W.to_mesh()[0]


def _interior_counts(m):
    b = build_topology(m).boundary_node
    nq = np.bincount(m.quads.ravel(), minlength=m.n_nodes)
    nt = np.bincount(m.tris.ravel(), minlength=m.n_nodes) if len(m.tris) else np.zeros(m.n_nodes, int)
    return nq[~b], nt[~b]


def test_ladder_has_one_triangle():
    m, _, _ = ladder_mesh()
    assert m.n_tris == 1
    assert (signed_area(m) > 0).all()


def test_collapse_carries_triangle_to_the_coast():
    m, bottom, _ = ladder_mesh()
    W = _work(m)
    coast = lambda a, b: a in bottom and b in bottom and W.is_bedge(a, b)
    t = int(np.flatnonzero(W.nv[: W.m] == 3)[0])
    res = [ops.collapse(W, t, i, coast, 10) for i in range(3)]
    ok = [r for r in res if r is not None]
    assert len(ok) == 1                                   # only the base faces the bottom coast
    m2 = _apply(W, ok[0])
    assert m2.n_tris == 0
    nq, nt = _interior_counts(m2)
    assert (nq == 4).all() and (nt == 0).all()
    assert (signed_area(m2) > 0).all()
    be = build_topology(m2).edge_elems[:, 1] < 0
    assert be.sum() == build_topology(m).edge_elems[:, 1].__lt__(0).sum() - 1   # one coast edge fewer


def test_split_carries_triangle_to_the_coast():
    m, _, top = ladder_mesh()
    W = _work(m)
    coast = lambda a, b: a in top and b in top and W.is_bedge(a, b)
    t = int(np.flatnonzero(W.nv[: W.m] == 3)[0])
    ok = [r for r in (ops.split(W, t, i, coast, 10) for i in range(3)) if r is not None]
    assert len(ok) >= 1
    m2 = _apply(W, ok[0])
    assert m2.n_tris == 0
    nq, nt = _interior_counts(m2)
    assert (nq == 4).all() and (nt == 0).all()
    assert (signed_area(m2) > 0).all()


def test_refcoast_projection_stays_on_own_shore():
    m, bottom, _ = ladder_mesh()
    ref = RefCoast(m)
    x = m.points[2] + 0.3 * (m.points[3] - m.points[2])
    x = x / np.linalg.norm(x)
    segs = ref.allowed(m.points[2], 0.15 * 6.371e6 * np.pi / 180)
    p = ref.project(x[None] * 1.0 + 1e-4, segs[None])[0]
    assert np.linalg.norm(p - x) < 1e-3


def test_coast_fill_respaces_the_coast_side():
    """The whole ladder as a cavity with the bottom as coast: a regular fill re-spaces the bottom from 6 to 5 edges,
    every new coast node on the reference coast, no triangle left."""
    from meshom.meshfix.cavity import cavity as cv
    from meshom.meshfix.coast import fills as fl

    m, bottom, _ = ladder_mesh()
    W = _work(m)
    ref = RefCoast(m)
    coast = lambda a, b: a in bottom and b in bottom and W.is_bedge(a, b)
    S = set(range(W.m))
    lp = cv.loop_of(W, S)
    n = len(lp)
    cr = [i for i, v in enumerate(lp) if v in (0, 6) or W.ninc[v] == 1]   # the 4 rectangle corners
    assert len(cr) == 4
    h = lambda x: np.full(len(np.atleast_2d(x)), 0.1 * np.pi / 180 * 6.371e6)
    cands = fl.coast_fills(W.P, lp, cr, h, coast, ref, adj_lnmax=0.3)
    reg = [c for c in cands if "regular" in c[0] and c[3]]
    assert reg, [c[0] for c in cands]
    name, X, cells, cn = reg[0]
    assert len(cn) == 4                                    # 5 bottom edges: 4 new interior coast nodes
    from meshom.meshfix.coast.refcoast import seg_point
    _, d = seg_point(X[[-1 - j for j in cn]][:, None, :], ref.A[None], ref.B[None])
    assert d.min(1).max() < 1e-5                      # on the arc: only the chord sagitta (m) remains
    assert all(len(c) == 4 for c in cells)


def _flipped_grid(nx=8, ny=8, d=0.1):
    """A regular grid with one 2x1 block re-split along the other diagonal of its hexagon: two 3-5 pairs."""
    lon, lat = np.meshgrid(np.arange(nx + 1) * d, np.arange(ny + 1) * d)
    P = lonlat_to_xyz(lon.ravel(), lat.ravel())
    idx = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    q = [[idx[j, i], idx[j, i + 1], idx[j + 1, i + 1], idx[j + 1, i]] for j in range(ny) for i in range(nx)]
    j, i = 3, 3                                           # quads (i, j) and (i + 1, j): hexagon p0..p5 (CCW)
    hexa = [idx[j, i], idx[j, i + 1], idx[j, i + 2], idx[j + 1, i + 2], idx[j + 1, i + 1], idx[j + 1, i]]
    q = [c for c in q if not (c[0] in (idx[j, i], idx[j, i + 1]) and c[1] in (idx[j, i + 1], idx[j, i + 2]))]
    q += [[hexa[0], hexa[1], hexa[2], hexa[5]], [hexa[2], hexa[3], hexa[4], hexa[5]]]
    m = Mesh(P, np.zeros((0, 3), np.int64), np.array(q, np.int64), np.zeros(len(P), np.int8))
    return m, set(idx[0].tolist())


def test_pair_collapse_turns_a_35_pair_into_one_triangle():
    m, bottom = _flipped_grid()
    assert (signed_area(m) > 0).all()
    nq0, _ = _interior_counts(m)
    assert sorted(nq0[nq0 != 4].tolist()) == [3, 3, 5, 5]
    W = _work(m)
    coast = lambda a, b: a in bottom and b in bottom and W.is_bedge(a, b)
    done = None
    nq_all = np.bincount(m.quads.ravel(), minlength=m.n_nodes)
    for e in range(W.m):
        c = W.cells(e)
        for k in range(4):
            a, b = c[k], c[(k + 1) % 4]
            if {nq_all[a], nq_all[b]} == {3, 5}:
                r = ops.collapse(W, e, k, coast, 10)
                if r is not None and r[3]:
                    done = r
                    break
        if done:
            break
    assert done is not None
    m2 = _apply(W, done)
    assert m2.n_tris == 1
    nq, nt = _interior_counts(m2)
    assert ((nq != 4) & (nt == 0)).sum() <= 2              # this pair is gone (the other one may touch the triangle)
    assert ((nq == 4) & (nt == 1)).any()                   # the merged node: the (4,4,5) tip, 4 quads + the triangle
    assert (signed_area(m2) > 0).all()


def test_a_guard_holds_coast_nodes_where_e_moved_the_coast():
    """Before A: the coast nodes near a reference vertex already > A_GUARD_H h off the current coast are held."""
    from types import SimpleNamespace

    from meshom.meshfix.coast import A_GUARD_H, a_slide_mask

    m, bottom, _ = ladder_mesh()
    ref = RefCoast(m)
    kind = build_topology(m).boundary_node.astype(np.int8)
    hval = 0.1 * np.pi / 180 * 6.371e6                    # the edge length: h = one edge
    case = SimpleNamespace(h=lambda x: np.full(len(np.atleast_2d(x)), hval))
    assert a_slide_mask(case, m, kind, ref).sum() == (kind == 1).sum()   # on the reference coast: all slide
    P = m.points.copy()
    v = 3                                                  # a bottom node pulled into the water by 0.3 h
    P[v] = lonlat_to_xyz(np.array([0.3]), np.array([0.3 * 0.1]))[0]
    m2 = Mesh(P, m.tris, m.quads, m.node_flag)
    s = a_slide_mask(case, m2, kind, ref)
    assert not s[v] and not s[2] and not s[4]              # the moved node and its coast neighbours are held
    assert s[[i for i in range(len(P)) if kind[i] == 1 and i not in range(7)]].all()   # 2 rings: 0..6 at most
    assert A_GUARD_H < 0.25


# ------------------------------------------------------------------------------- B146-B149

def _coast_zone(seed=0, jitter=0.2):
    """The ladder with its bottom as the coast: every node but the top row free, the bottom nodes slide along the
    bottom line (their allowed segments: the whole bottom), positions jittered (the coast nodes off the line too)."""
    m, bottom, top = ladder_mesh()
    rng = np.random.default_rng(seed)
    d = 0.1 * np.pi / 180
    P = m.points + rng.normal(0, jitter * d / 3, m.points.shape)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    free = np.ones(len(P), bool)
    free[list(top)] = False
    slide = np.array(sorted(bottom), np.int64)
    ref = RefCoast(m)
    segs = np.array([ref.allowed(m.points[v], 3 * d * 6.371e6, 8) for v in slide])
    return P, m.quads, m.tris, free, slide, ref.A[segs], ref.B[segs]


def _h_coast(x):
    return np.full(len(np.atleast_2d(x)), 0.1 * np.pi / 180 * 6.371e6)


def test_numba_coast_energy_and_gradient_match_jax():
    """B147: the Numba coast energy (cavity/ncoast.py) equals coast/relax.py's JAX energy and gradient."""
    from meshom.meshfix.cavity import ncoast
    from meshom.meshfix.coast import relax as cr

    for name in ("W_CAP", "CAP", "W_RWALL", "RWALL", "W_COAST", "WALL_MARGIN"):
        assert getattr(ncoast, name) == getattr(cr, name), name
    P, q, t, free, slide, SA, SB = _coast_zone()
    k = ncoast.prepare(P, free, q, t, _h_coast, 0.2, slide, SA, SB, w_wall=3000.0)
    f = ncoast.fg_of(k)
    rng = np.random.default_rng(1)
    v = rng.normal(0, 0.2, 2 * len(k["fidx"]))
    e, g = f(v)
    captured = {}
    orig = cr.minimize

    def grab(fun, v0, **kw):
        captured["fun"] = fun
        captured["n"] = len(v0)
        return orig(fun, v0, **{**kw, "options": dict(maxiter=0)})

    cr.minimize = grab
    try:
        cr.relax(P, free, q, t, _h_coast, 0.2, slide, SA, SB, bias=k["bias"], w_wall=3000.0)
    finally:
        cr.minimize = orig
    vp = np.zeros(captured["n"])
    vp[: len(v)] = v
    ej, gj = captured["fun"](vp)
    assert abs(e - ej) < 1e-9 * max(1.0, abs(ej)), (e, ej)
    assert np.allclose(g, gj[: len(v)], rtol=1e-7, atol=1e-9 * max(1.0, np.abs(gj).max()))
    for i in (0, 5, 2 * len(slide) - 1):                # finite differences of the Numba gradient itself
        hs = 1e-6
        vp, vm = v.copy(), v.copy()
        vp[i] += hs
        vm[i] -= hs
        assert abs((f(vp)[0] - f(vm)[0]) / (2 * hs) - g[i]) < 1e-5 * max(1.0, abs(g[i]))


def test_numba_coast_relax_matches_jax_relax():
    """B147: the Numba relaxation reaches the JAX one's minimum, the sliding nodes end on the reference coast."""
    from meshom.meshfix.cavity import ncoast
    from meshom.meshfix.coast import relax as cr
    from meshom.meshfix.coast.refcoast import seg_point

    P, q, t, free, slide, SA, SB = _coast_zone(seed=2)
    Xa, ia = cr.relax(P, free, q, t, _h_coast, 0.2, slide, SA, SB, w_wall=3000.0)
    Xb, ib = ncoast.relax(P, free, q, t, _h_coast, 0.2, slide, SA, SB, w_wall=3000.0)
    assert ib["e1"] < ib["e0"]
    # both stop at max_iter on this stiff zone (walls 3000 / 1e4 / 1e3): the paths part by rounding (scipy's
    # L-BFGS-B on the Numba energy parts from it as much), so the minima agree to ~1e-5 relative only
    assert abs(ia["e1"] - ib["e1"]) < 1e-4 * max(1.0, ia["e1"])
    assert np.abs(Xa - Xb).max() < 1e-4
    _, d = seg_point(Xb[slide][:, None, :], SA, SB)
    assert d.min(1).max() < 1e-6                         # on the arc: the chord sagitta of a 0.1 deg segment


def test_refcoast_from_segments_equals_the_mesh_graph():
    """B148: the reference coast rebuilt from Case.coast_seg (a later stage's view) has the same segments, the same
    neighbours along the coast and the same allowed lists as the one built from the input mesh."""
    from meshom.meshfix.cases import _segments, boundary_edges

    m, bottom, _ = ladder_mesh()
    r0 = RefCoast(m)
    r1 = RefCoast.from_segments(_segments(m, boundary_edges(m)))
    assert np.array_equal(r0.A, r1.A) and np.array_equal(r0.B, r1.B)
    assert np.array_equal(np.sort(r0.V, axis=0), np.sort(r1.V, axis=0))
    for v in range(m.n_nodes):
        if v in bottom:
            assert np.array_equal(r0.allowed(m.points[v], 3e4, 8), r1.allowed(m.points[v], 3e4, 8))


def _coast_case(mesh):
    """A cut-out case of mesh: its bottom boundary row is real coast (kind 1), the rest of the outline artificial."""
    from meshom.meshfix.cases import Case, _segments, boundary_edges
    from meshom.size import Uniform

    h = Uniform(0.1 * np.pi / 180 * 6.371e6)
    b = build_topology(mesh).boundary_node
    lat = np.degrees(np.arcsin(mesh.points[:, 2]))
    kind = b.astype(np.int8) * 2
    kind[b & (lat < 0.03)] = 1
    be = boundary_edges(mesh)
    art_e = be[(kind[be] == 2).all(1)]
    seg = _segments(mesh, be[~(kind[be] == 2).all(1)])
    return Case("synthetic_coast", mesh, kind, h, h(mesh.points), seg, seg, mesh.points[:1], art_e, 0.2, {}, {},
                {"h_pkl": "-"})


def _e_run(case, **o):
    from meshom.meshfix.coast import ref_of
    from meshom.meshfix.coast.engine import CoastEngine

    eng = CoastEngine(case.mesh, case.h, 0.2, case.kind == 2, ref_of(case), dict(passes=4, **o))
    return eng.run(), eng.stats


def test_fast_e_cache_and_workers_give_the_serial_result():
    """B148: E on D's engine: the cache and the worker processes change the work, not the result."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_meshfix_fast import _defect_mesh

    case = _coast_case(_defect_mesh(seed=4, jitter=0.2))
    ref, s0 = _e_run(case, cache=False)
    assert s0["accepted"] >= 2
    for o in (dict(cache=True), dict(cache=True, workers=2, inflight=2)):
        out, s = _e_run(case, **o)
        assert s["accepted"] == s0["accepted"], o
        assert out.n_nodes == ref.n_nodes and np.array_equal(out.quads, ref.quads), o
        assert np.array_equal(out.tris, ref.tris), o
        assert np.abs(out.points - ref.points).max() < 1e-12, o


def _regional_ladder(art_side: str, bottom_rows: list, bases: list):
    """A rectangular cut-out whose side
    art_side ("top" or "bottom") is artificial and the other three real coast, filled by build_patch with one
    column-end triangle next to the artificial side."""
    from meshom.meshfix.cases import Case, boundary_edges
    from meshom.meshfix.cavity.patterns import Fill, build_patch

    a, c, R, d = 6, 5, 4, 0.1
    coords = []

    def line(x0, y0, x1, y1, n):
        ids = []
        for f in np.linspace(0, 1, n + 1):
            xy = (round(x0 + f * (x1 - x0), 12), round(y0 + f * (y1 - y0), 12))
            if xy not in coords:
                coords.append(xy)
            ids.append(coords.index(xy))
        return ids

    b, t = line(0, 0, a * d, 0, a), line(0, R * d, a * d, R * d, c)
    lft, rgt = line(0, 0, 0, R * d, R), line(a * d, 0, a * d, R * d, R)
    P = lonlat_to_xyz(*np.array(coords).T)
    F = Fill(P)
    cells = build_patch(F, b, rgt, t, lft, bottom_rows, bases)
    n0 = len(P)
    cells = [tuple(v if v >= 0 else n0 - 1 - v for v in cc) for cc in cells]
    P = np.vstack([P, np.array(F.X)])
    mesh = Mesh(P, np.array([cc for cc in cells if len(cc) == 3], np.int64).reshape(-1, 3),
                np.array([cc for cc in cells if len(cc) == 4], np.int64).reshape(-1, 4), np.zeros(len(P), np.int8))
    art = np.zeros(len(P), bool)
    art[t if art_side == "top" else b] = True
    be = boundary_edges(mesh)
    ae, coast = be[art[be].all(1)], be[~art[be].all(1)]
    kind = build_topology(mesh).boundary_node.astype(np.int8)
    kind[art] = 2
    h = lambda X: np.full(len(np.atleast_2d(X)), 0.11 * np.pi / 180 * 6.371e6)   # noqa: E731
    return Case("regional_ladder", mesh, kind, h, h(P), P[coast], P[coast], np.zeros((0, 3)), ae, 0.2)


def _art_counts(mesh, art_pts):
    from scipy.spatial import cKDTree

    _, idx = cKDTree(mesh.points).query(art_pts)
    nq = np.bincount(mesh.quads.ravel(), minlength=mesh.n_nodes)
    nt = np.bincount(mesh.tris.ravel(), minlength=mesh.n_nodes) if mesh.n_tris else np.zeros(mesh.n_nodes, int)
    return nq[idx], nt[idx]


def test_e_operators_keep_the_artificial_counts():
    """B121: a collapse whose remaining tip is artificial and a split that turns a
    triangle with artificial base corners into a quad were accepted (artificial_incidence_changed 1 and 2); every
    candidate must keep the quad and the triangle count of each artificial node."""
    from meshom.meshfix.coast import ref_of
    from meshom.meshfix.coast.engine import CoastEngine

    for side, rows, bases in (("top", [6, 6, 6, 6, 5], [set(), set(), set(), {2}]),
                              ("bottom", [6, 5, 5, 5, 5], [{2}, set(), set(), set()])):
        case = _regional_ladder(side, rows, bases)
        assert case.mesh.n_tris == 1
        art_pts = case.mesh.points[case.kind == 2]
        q0, t0 = _art_counts(case.mesh, art_pts)
        eng = CoastEngine(case.mesh, case.h, 0.2, case.kind == 2, ref_of(case), dict(cavity=False, passes=1))
        out = eng.run()
        q1, t1 = _art_counts(out, art_pts)
        assert np.array_equal(q0, q1) and np.array_equal(t0, t1), (side, eng.stats["reject"])
        assert sum(d.get("art", 0) for d in eng.stats["reject"].values()) > 0, side   # the operator was built


def test_pipeline_runs_stage_e_against_the_input_coast():
    """B148: stage e in the one pipeline; every stage after the first gets the input's coast_seg as its reference
    and the current mesh's kinds; a,e,d,a keeps the artificial boundary (e after a: a moved the coast)."""
    import sys

    from scipy.spatial import cKDTree

    import meshom.meshfix.pipeline as pl

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_meshfix_fast import _defect_mesh

    case = _coast_case(_defect_mesh(seed=4, jitter=0.2))
    art = case.mesh.points[case.kind == 2]
    seen = []
    fix0 = pl.stage_fix

    def spy(s):
        f = fix0(s)

        def g(c, o):
            seen.append((s, c.coast_seg is case.coast_seg, len(c.kind) == c.mesh.n_nodes,
                         cKDTree(c.mesh.points[c.kind == 2]).query(art)[0].max()))
            return f(c, o)
        return g

    pl.stage_fix = spy
    try:
        mesh, recs = pl.run_stages(case, ["a", "e", "d", "a"], {"d": dict(workers=1, passes=2),
                                                                "e": dict(workers=1, passes=3)}, verbose=False)
    finally:
        pl.stage_fix = fix0
    assert [x[0] for x in seen] == ["a", "e", "d", "a"] and all(x[1] and x[2] and x[3] == 0.0 for x in seen)
    assert recs[1]["method"] == "e_coast" and recs[1]["stats"]["seeds"] >= 1
    assert cKDTree(mesh.points).query(art)[0].max() == 0.0


def test_e_evaluation_of_a_removed_seed_is_empty():
    """B148: a re-evaluation dispatched after a commit removed the seed's element (an IndexError in the
    collapse of a dead triangle, d,b,e,a on glob100) returns no candidate; the seed is skipped at retire."""
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_meshfix_fast import _defect_mesh

    from meshom.meshfix.coast import ref_of
    from meshom.meshfix.coast.engine import CoastEngine

    case = _coast_case(_defect_mesh(seed=4, jitter=0.2))
    eng = CoastEngine(case.mesh, case.h, 0.2, case.kind == 2, ref_of(case), dict(passes=1))
    sd = [x for x in eng.seeds() if x[1] == "tri"]
    seed = sd[0][2]
    W = eng.W
    t = seed[1][0]
    W._remove(t)                                          # the triangle gone, as after another seed's commit
    r = eng.evaluate(seed, dict(eng.tot))
    assert r["cands"] == [] and r["keys"] == [] and eng._skip(seed)
