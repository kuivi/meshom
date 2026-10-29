"""The optimisation stage (meshom.optimize, decisions Z1-Z13): each operation's validity rule on small meshes."""
import json

import numpy as np
import pytest

from helpers import _mesh, local_polygon, structured_patch
from meshom.compete import irregular_counts
from meshom.mesh import Mesh, build_topology, signed_area
from meshom.optimize import (DEFAULTS, _Engine, _Work, flip_triangles, optimize, poly_splits, summary)
from meshom.size import Uniform

DX = 10e3
H = Uniform(DX)


def _patch_xy(nx, ny):
    """Node (j, i) of structured_patch(nx, ny, DX) in metres (centred)."""
    xs, ys = np.arange(nx + 1) * DX, np.arange(ny + 1) * DX
    X, Y = np.meshgrid(xs - 0.5 * xs[-1], ys - 0.5 * ys[-1])
    return np.column_stack([X.ravel(), Y.ravel()])


def _with(nx, ny, xy, tris, quads_keep_mask, extra_quads=()):
    base = structured_patch(nx, ny, DX)
    q = base.quads[quads_keep_mask]
    if len(extra_quads):
        q = np.vstack([q, np.asarray(extra_quads, dtype=np.int64)])
    return _mesh(local_polygon(0.0, 0.0, xy), tris, q)


def _node(nx, j, i):
    return j * (nx + 1) + i


def _boundary(m: Mesh) -> set:
    t = build_topology(m)
    return {tuple(e) for e in t.edges[t.edge_elems[:, 1] < 0].tolist()}


def _fan(fixed_centre=False):
    """4 x 4 quads, the cell (1, 1) replaced by a fan of 4 triangles round a new centre node (a cluster of 4 whose
    pair merges all have a 180 deg corner)."""
    nx = ny = 4
    xy = _patch_xy(nx, ny)
    p = [_node(nx, 1, 1), _node(nx, 1, 2), _node(nx, 2, 2), _node(nx, 2, 1)]
    c = len(xy)
    xy = np.vstack([xy, xy[p].mean(axis=0)])
    keep = np.ones(nx * ny, dtype=bool)
    keep[1 * nx + 1] = False
    tris = [(c, p[k], p[(k + 1) % 4]) for k in range(4)]
    m = _with(nx, ny, xy, tris, keep)
    if fixed_centre:
        m.node_flag[c] = 1
    return m, c


def test_poly_splits_counts_and_cover():
    assert len(poly_splits((0, 1, 2, 3, 4, 5), 0)) == 3
    assert len(poly_splits(tuple(range(8)), 0)) == 12
    five = poly_splits((0, 1, 2, 3, 4), 1)
    assert len(five) == 5 and all(sorted(len(c) for c in s) == [3, 4] for s in five)
    for s in poly_splits(tuple(range(7)), 1):                  # every split covers the polygon once, CCW
        he = {(c[i], c[(i + 1) % len(c)]) for c in s for i in range(len(c))}
        outer = {e for e in he if (e[1], e[0]) not in he}
        assert outer == {(i, (i + 1) % 7) for i in range(7)} and sum(len(c) == 3 for c in s) == 1


def test_flip_lowers_the_largest_angle():
    nx = ny = 4
    xy = _patch_xy(nx, ny)
    p0, p1, p2, p3 = _node(nx, 1, 1), _node(nx, 1, 2), _node(nx, 2, 2), _node(nx, 2, 1)
    cen = xy[[p0, p2]].mean(axis=0)
    xy[p1] = cen + 0.6 * (xy[p1] - cen)                        # p1 and p3 pulled in: p1-p3 is the short diagonal
    xy[p3] = cen + 0.6 * (xy[p3] - cen)
    keep = np.ones(nx * ny, dtype=bool)
    keep[nx + 1] = False
    m = _with(nx, ny, xy, [(p0, p1, p2), (p0, p2, p3)], keep)
    out, n = flip_triangles(m)
    assert n == 1 and all({p1, p3} <= set(t) for t in out.tris.tolist())
    from meshom.optimize import _minmax
    before = max(_minmax(m.points, *t)[1] for t in m.tris)
    after = max(_minmax(out.points, *t)[1] for t in out.tris)
    assert after < before and (signed_area(out) > 0).all() and _boundary(out) == _boundary(m)
    out2, n2 = flip_triangles(out)                             # a fixed point: no flip back
    assert n2 == 0


def test_flip_never_gives_a_triangle_two_boundary_edges():
    nx = ny = 3
    xy = _patch_xy(nx, ny)
    a, b, c, d = _node(nx, 0, 0), _node(nx, 0, 1), _node(nx, 1, 1), _node(nx, 1, 0)
    xy[c] = xy[c] + 0.3 * (xy[c] - xy[a])                      # the corner cell's diagonal a-c made long
    keep = np.ones(nx * ny, dtype=bool)
    keep[0] = False
    m = _with(nx, ny, xy, [(a, b, c), (a, c, d)], keep)
    from meshom.optimize import _minmax
    old = max(_minmax(m.points, *t)[1] for t in m.tris)
    new = max(_minmax(m.points, b, c, d)[1], _minmax(m.points, d, a, b)[1])
    assert new < old                                           # the flip would lower the largest angle ...
    out, n = flip_triangles(m)
    assert n == 0                                              # ... but (d, a, b) would hold two boundary edges


def test_merge_only_into_a_good_quad():
    nx = ny = 3
    xy = _patch_xy(nx, ny)
    p0, p1, p2, p3 = _node(nx, 1, 1), _node(nx, 1, 2), _node(nx, 2, 2), _node(nx, 2, 1)
    keep = np.ones(nx * ny, dtype=bool)
    keep[nx + 1] = False
    m = _with(nx, ny, xy, [(p0, p1, p2), (p0, p2, p3)], keep)
    out, info = optimize(m, H, None, {"steps": ("merge",)})
    assert out.n_tris == 0 and out.n_quads == nx * ny and info["checks"]["euler_kept"]
    xy2 = xy.copy()
    xy2[p3] = xy[p3] + np.array([0.3, -0.3]) * DX               # the merged quad would have 136 deg at p3
    m2 = _with(nx, ny, xy2, [(p0, p1, p2), (p0, p2, p3)], keep)
    from meshom.recombine import quad_cost
    assert 90 + 90 * quad_cost(m2.points, m2.tris[0], m2.tris[1])[0] > 135
    out2, _ = optimize(m2, H, None, {"steps": ("merge",)})
    assert out2.n_tris == 2


def test_cluster_collapse_of_a_free_node_only():
    m, c = _fan()
    assert summary(m)["tri_cluster_max"] == 4
    out, info = optimize(m, H, None, {"steps": ("merge", "cluster")})
    s = summary(out)
    assert s["tri_cluster_max"] <= 2 and out.n_nodes == m.n_nodes - 1     # the centre removed: the cell is a quad
    assert info["checks"] == {"euler_kept": True, "boundary_kept": True, "inverted_before": 0, "inverted_after": 0}
    mf, cf = _fan(fixed_centre=True)
    outf, infof = optimize(mf, H, None, {"steps": ("merge", "cluster")})
    kept = np.flatnonzero(outf.node_flag == 1)
    assert len(kept) == 1                                      # a fixed node is never removed ...
    assert np.array_equal(outf.points[kept[0]], mf.points[cf])   # ... nor moved
    assert infof["checks"]["euler_kept"] and infof["checks"]["boundary_kept"]
    assert summary(outf)["tri_cluster_max"] <= 4


def test_patch_rules():
    m, c = _fan()
    W = _Work(m)
    star = W.at(c)
    from meshom.optimize import boundary_loop
    lp = boundary_loop([W.cells(e) for e in star])
    assert W.patch_ok(star, [lp])                              # the square: valid (centre removed)
    assert not W.patch_ok(star, [lp[::-1]])                    # reversed: the boundary half-edges differ
    assert not W.patch_ok(star[:2], [lp])                      # not the patch's boundary
    assert not W.patch_ok(star, [(lp[0], lp[1], lp[2])])       # a hole: boundary and Euler characteristic differ
    Wf = _Work(_fan(fixed_centre=True)[0])
    assert not Wf.patch_ok(Wf.at(c), [lp])                     # a fixed node is never removed


def test_gross_change_from_topology_is_exact():
    m, c = _fan()
    W = _Work(m)
    eng = _Engine(W, H, 0.15, dict(DEFAULTS), m)
    n = 0
    for t in W.at(c):
        for e in W.nbrs(t):
            for rm, new, Z, M in eng._resplits(sorted({t, e})) + eng._resplits(sorted(W.at(c))):
                if not W.patch_ok(rm, new):
                    continue
                old = W.measure(Z, H, eng.mo)
                rec = W.apply(rm, new)
                nm = W.measure(Z, H, eng.mo)
                W.undo(rec)
                assert eng.dgross(rm, new) == nm["gross"] - old["gross"]
                n += 1
    assert n > 10


def _split_node():
    """6 x 6 quads with the interior node (3, 3) split into two valence-3 nodes v, w joined by a new quad
    (v, S, w, N): a 3-3-5-5 configuration that the diagonal collapse removes."""
    nx = ny = 6
    xy = _patch_xy(nx, ny)
    m_ = _node(nx, 3, 3)
    N, S, E, Wn = _node(nx, 4, 3), _node(nx, 2, 3), _node(nx, 3, 4), _node(nx, 3, 2)
    v, w = m_, len(xy)
    xy = np.vstack([xy, xy[m_] + [0.45 * DX, 0.0]])
    xy[v] = xy[m_] - [0.45 * DX, 0.0]
    base = structured_patch(nx, ny, DX).quads.tolist()
    quads = []
    for q in base:
        if m_ in q:
            east = any(x in (E, _node(nx, 4, 4), _node(nx, 2, 4)) for x in q)
            q = [w if (x == m_ and east) else x for x in q]
        quads.append(q)
    quads.append([v, S, w, N])
    return _mesh(local_polygon(0.0, 0.0, xy), [], quads), v, w


def test_valence_collapse_cancels_a_33_pair():
    m, v, w = _split_node()
    ic = irregular_counts(m)
    assert (ic["n3"], ic["n5"]) == (2, 2) and (signed_area(m) > 0).all()
    out, info = optimize(m, H, None, {"steps": ("valence",)})
    ic2 = irregular_counts(out)
    assert (ic2["n3"], ic2["n5"]) == (0, 0) and out.n_nodes == m.n_nodes - 1 and out.n_quads == m.n_quads - 1
    assert info["checks"]["euler_kept"] and info["checks"]["boundary_kept"] and info["checks"]["inverted_after"] == 0


def test_fit_moves_only_free_nodes_near_the_change():
    m, c = _fan()
    out, info = optimize(m, H, 0.2, {"steps": ("merge", "cluster", "fit"), "fit_rings": 1})
    assert info["steps"]["fit"]["free_nodes"] > 0
    old = np.delete(np.arange(m.n_nodes), c)                   # the centre removed, the other nodes in order
    moved = np.linalg.norm(out.points - m.points[old], axis=1) > 1e-12
    assert moved.any() and not (moved & build_topology(m).boundary_node[old]).any()
    dist = 6371e3 * np.linalg.norm(m.points[old] - m.points[c], axis=1)
    assert not moved[dist > 2.5 * DX].any()                    # beyond fit_rings of the touched nodes: unmoved


@pytest.mark.parametrize("method", ["m01_isotropic"])
def test_run_stage_and_cli(tmp_path, method):
    pytest.importorskip("tomli_w")
    from meshom import config
    from meshom.cli import main

    text = f"""
[mesh]
method = "{method}"
optimize = true
[coast]
source = "none"
[size]
grad_max = 0.1
[[size.layers]]
kind = "constant"
h_m = 800000.0
[run]
timeout = 300.0
"""
    cfg = config.loads(text, "opt")
    assert config.loads(config.dumps(cfg), "opt") == cfg and cfg.mesh.optimize
    from meshom.run import run

    d = tmp_path / "opt"
    q = run(cfg, d)
    r = q["optimized"]
    assert r["status"] == "ok" and r["irregular"]["euler_ok"] and r["optimize"]["checks"]["euler_kept"]
    assert (d / "optimized.npz").exists()
    stages = [s["stage"] for s in json.loads((d / "stages.json").read_text())]
    assert "optimize" in stages and stages.index("optimize") < stages.index("write")
    assert int((d / "fesom/nod2d.out").read_text().split()[0]) == r["n_nodes"]
    main(["optimize", str(d), "--out", "opt2"])
    rec = json.loads((d / "opt2.json").read_text())
    assert rec["after"]["irregular"]["euler_ok"] and "opt2" in json.loads((d / "quality.json").read_text())


def test_local_fit_on_the_submesh_equals_the_whole_mesh_fit():
    from meshom.optimize import _fit_sub
    from meshom.size_fit import smooth_fit

    nx = ny = 8
    rng = np.random.default_rng(3)
    xy = _patch_xy(nx, ny) + rng.normal(0.0, 0.08 * DX, (81, 2))
    m = _mesh(local_polygon(0.0, 0.0, xy), [], structured_patch(nx, ny, DX).quads)
    free = np.zeros(m.n_nodes, dtype=bool)
    free[[_node(nx, j, i) for j in range(3, 6) for i in range(3, 6)]] = True
    a, _ = smooth_fit(m, H, 0.2, ~free)
    b, _ = _fit_sub(m, H, 0.2, free, {})
    assert np.abs(b.points - m.points)[~free].max() == 0.0
    d = 6371e3 * np.linalg.norm(a.points - b.points, axis=1).max()
    assert d < 1e-3 * DX and 6371e3 * np.linalg.norm(a.points - m.points, axis=1).max() > 1e-2 * DX


def test_flipped_triangles_are_relaxed_by_the_local_fit():
    nx = ny = 4
    xy = _patch_xy(nx, ny)
    p0, p1, p2, p3 = _node(nx, 1, 1), _node(nx, 1, 2), _node(nx, 2, 2), _node(nx, 2, 1)
    cen = xy[[p0, p2]].mean(axis=0)
    xy[p1] = cen + 0.6 * (xy[p1] - cen)
    xy[p3] = cen + 0.6 * (xy[p3] - cen)
    keep = np.ones(nx * ny, dtype=bool)
    keep[nx + 1] = False
    m = _with(nx, ny, xy, [(p0, p1, p2), (p0, p2, p3)], keep)
    out, info = optimize(m, H, 0.2, {"steps": ("flip", "fit"), "fit_rings": 1})
    assert info["steps"]["flip"]["flips"] == 1 and info["steps"]["fit"]["free_nodes"] > 0


def test_m14_methods_own_the_stage():
    """Z1: every m14 method runs the stage as its own last stage by default; the others have
    none (m15_tri, the m02 _rep variant)."""
    from meshom.methods import METHODS, defaults, own_optimize

    m14 = [k for k in METHODS if k.startswith("m14_blocks")]
    assert "m14_blocks_rep" in m14 and len(m14) >= 6
    assert all(own_optimize(k) is True for k in m14)
    assert all(defaults(k)["optimize"] is True for k in m14 if k != "m14_blocks_rep")
    assert own_optimize("m15_tri") is None and own_optimize("m02_linf_frontal_rep") is None
    assert "optimize" not in defaults("m15_tri")


@pytest.mark.parametrize("method", ["m14_blocks", "m14_blocks_rep", "m15_tri"])
def test_method_stage_default_and_switch(tmp_path, monkeypatch, method):
    """The default reaches m14_blocks and m14_blocks_rep (their record optimize.json, the stage's checks kept),
    optimize = 0 switches it off; m15_tri has no stage."""
    from meshom import compete
    from meshom.domain import closed_sphere
    from meshom.methods import METHODS, m14_blocks
    from meshom.size import uniform

    h = uniform(500e3)
    for opts, on in [({}, method != "m15_tri"), ({"optimize": 0}, False)]:
        d = tmp_path / str(bool(opts))
        d.mkdir()
        monkeypatch.setenv("MESHOM_RUN_DIR", str(d))
        if method == "m14_blocks":
            mesh, info = m14_blocks.run(closed_sphere(), h, opts)
            assert ("optimize" in info) == on
        else:
            mesh = METHODS[method](closed_sphere(), h, opts)
        compete.DEFERRED.clear()
        f = d / "optimize.json"
        assert f.exists() == on, (method, opts)
        if on:
            rec = json.loads(f.read_text())
            assert rec["checks"]["euler_kept"] and rec["checks"]["boundary_kept"] and rec["tol"] == 0.2
            assert rec["after"]["n_nodes"] == mesh.n_nodes


def test_run_no_double_stage(tmp_path):
    """meshom run with an m14 method: [mesh] optimize unset = the method's own stage (optimize.json, no "optimized"
    record, no second run); false = off."""
    pytest.importorskip("tomli_w")
    from meshom import config
    from meshom.run import run

    base = """
[mesh]
method = "m14_blocks"
{extra}
[coast]
source = "none"
[size]
grad_max = 0.1
[[size.layers]]
kind = "constant"
h_m = 500000.0
[run]
timeout = 300.0
"""
    for extra, on in [("", True), ("optimize = false", False)]:
        d = tmp_path / str(on)
        q = run(config.loads(base.format(extra=extra), "o"), d)
        assert q["raw"]["status"] == "ok" and "optimized" not in q
        assert (d / "optimize.json").exists() == on and not (d / "optimized.npz").exists()
        stages = [s["stage"] for s in json.loads((d / "stages.json").read_text())]
        assert "optimize" not in stages
