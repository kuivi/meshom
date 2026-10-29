"""m14 adaptive gaps (decisions B63, meshom/methods/m14/gaps.py): the trim removes exactly the k cell rows along the
run that faces the corridor and nothing else; the keep rule (the pair improves, its blocks' corridors together too);
the rounds on a synthetic corridor keep the trim that widens a bad gap and undo the one that makes a corridor worse,
the undone side restored exactly; on uniform scale 4 the variant's mesh is valid and holds the returned blocks."""
import numpy as np
import pytest

from meshom.geo import stereographic_inverse
from meshom.methods.m14 import gaps as G
from meshom.methods.m14.blockmap import BlockGrid, outline_loop

HM = 50e3


def _block(i0: int, j0: int, ni: int, nj: int) -> BlockGrid:
    """An ni x nj cell lattice block at lattice offset (i0, j0), cell size HM, near (0, 0) (i east, j north)."""
    ii, jj = np.meshgrid(np.arange(i0, i0 + ni + 1), np.arange(j0, j0 + nj + 1), indexing="ij")
    ij = np.column_stack([ii.ravel(), jj.ravel()]).astype(np.int64)
    pts = stereographic_inverse(ij[:, 0] * HM, ij[:, 1] * HM, 0.0, 0.0)
    nid = np.arange(len(ij)).reshape(ni + 1, nj + 1)
    ci, cj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    ci, cj = ci.ravel(), cj.ravel()
    quads = np.column_stack([nid[ci, cj], nid[ci + 1, cj], nid[ci + 1, cj + 1], nid[ci, cj + 1]])
    loop = outline_loop(np.ones((ni, nj), dtype=bool), nid)
    assert len(loop) == 1
    return BlockGrid(pts, quads, ij, loop[0], {"n_cells": ni * nj})


def _side(g: BlockGrid, axis: int, top: bool) -> np.ndarray:
    """Local outline edge ids of the block side where ij[:, axis] is extreme (max when top)."""
    v = g.ij[:, axis]
    x = v.max() if top else v.min()
    a, b = g.outline, np.roll(g.outline, -1)
    return np.flatnonzero((v[a] == x) & (v[b] == x))


O = {"min_cells": 100.0, "min_inradius": 5.0, "gap_rows": 1, "gap_rounds": 2, "gap_gates": None}


def test_trim_removes_k_rows_along_the_run_only():
    g = _block(0, 0, 20, 20)
    top = _side(g, 1, True)
    assert len(top) == 20
    cells = G.trim_cells(g, top, 2)
    assert len(cells) == 40 and set(cells[:, 1]) == {18, 19}
    t = G.rebuild(g, cells, O)
    assert len(t.quads) == 360 and t.info["gap_cells_removed"] == 40
    assert t.ij[:, 1].max() == 18 and t.ij[:, 1].min() == 0 and t.ij[:, 0].min() == 0 and t.ij[:, 0].max() == 20
    # positions kept: every node of the trimmed block is a node of the old one
    key = {tuple(x) for x in g.ij}
    assert all(tuple(x) in key for x in t.ij)
    old = {tuple(x): p for x, p in zip(g.ij, g.points)}
    assert max(np.abs(old[tuple(x)] - p).max() for x, p in zip(t.ij, t.points)) == 0.0
    # one outline loop, the fill on its left as before (the block's cells on its right): the same turning sense
    assert len(t.outline) == 2 * (20 + 18)
    # a part of a side: only those columns lose their rows (the gap widens there only)
    part = top[:8]
    c2 = G.trim_cells(g, part, 1)
    assert len(c2) == 8 and set(c2[:, 1]) == {19}
    t2 = G.rebuild(g, c2, O)
    assert len(t2.quads) == 392
    # a trim that leaves the block below min_inradius is not made
    assert G.rebuild(g, G.trim_cells(g, top, 12), O) is None


def _s(excess: float, failed=("ln_sh_p99",)) -> dict:
    return {"excess": excess, "failed": list(failed) if excess > 0 else []}


def test_keep_rule():
    p, q, r = (0, 1), (0, 2), (1, 3)
    before = {p: _s(0.6), q: _s(0.2), r: _s(0.1)}
    # p improves, its neighbourhood (q shares block 0, r block 1) too
    assert G.keep(p, before, {p: _s(0.3), q: _s(0.25), r: _s(0.1)})
    # p improves but a corridor of its blocks gets much worse: undone
    assert not G.keep(p, before, {p: _s(0.3), q: _s(0.7), r: _s(0.1)})
    # p's exceedance falls but a gate it passed fails now: undone
    assert not G.keep(p, before, {p: _s(0.3, ("ln_sh_p99", "ratio_p99")), q: _s(0.2), r: _s(0.1)})
    # the corridor is gone (e.g. an open region now): not improved
    assert not G.keep(p, before, {q: _s(0.2), r: _s(0.1)})
    # gate evaluation: exceedance = sum of value / limit - 1 over the failed gates (ratio: its excess over 1)
    f, x = G.gate_eval({"defects_per10": 3.0, "ln_sh_p99": 0.18, "ratio_p99": 1.30, "quad_p99_deg": 100.0,
                        "tri_cluster": 3})
    assert f == ["defects_per10", "ratio_p99", "tri_cluster"] and x == pytest.approx(0.5 + 1.0 + 0.5)


def test_rounds_keep_the_widening_that_helps_and_undo_the_other(monkeypatch):
    """A (0) with B (1) 6 cells above it and C (2) 6 cells to its right. The fake corridor A-B gets better as it
    widens (|ln s/h| p99 0.30 - 0.02 per extra cell: passes at 12 cells), A-C worse (0.20 + 0.02 per cell). One trim
    per block and round (A-B and A-C share A): A-B, the worst, is trimmed in rounds 1-3 (1 row per block and round:
    12 cells wide, passing), then A-C is tried in round 4, undone and frozen, A's right side restored exactly."""
    import meshom.methods.m14.stitch as S

    grids = [_block(0, 0, 20, 20), _block(0, 26, 20, 20), _block(26, 0, 20, 20)]
    calls = []

    def fake_stitch(domain, h, gs, o, fixed):
        calls.append([len(g.quads) for g in gs])
        return None, {"grids": gs}

    def fake_score(mesh, h, st, o):
        a, b, c = st["grids"]
        w_ab = int(b.ij[:, 1].min() - a.ij[:, 1].max())
        w_ac = int(c.ij[:, 0].min() - a.ij[:, 0].max())
        out = {}
        for p, ln, runs in (((0, 1), 0.30 - 0.02 * (w_ab - 6), {0: _side(a, 1, True), 1: _side(b, 1, False)}),
                            ((0, 2), 0.20 + 0.02 * (w_ac - 6), {0: _side(a, 0, True), 2: _side(c, 0, False)})):
            m = {"ln_sh_p99": ln, "runs": runs}
            m["failed"], m["excess"] = G.gate_eval(m)
            out[p] = m
        return out

    monkeypatch.setattr(S, "stitch", fake_stitch)
    monkeypatch.setattr(G, "score", fake_score)
    _, st, out, rec = G.adapt(None, None, grids, dict(O, gap_rounds=6), [], None, {"grids": grids})
    a, b, c = out
    assert int(b.ij[:, 1].min() - a.ij[:, 1].max()) == 12           # A-B: 6 + 3 rounds x (1 + 1) rows
    assert a.ij[:, 0].max() == 20 and c.ij[:, 0].min() == 26          # A-C: undone, A's right side as it was
    assert len(a.quads) == 20 * 17 and len(b.quads) == 20 * 17 and len(c.quads) == 400
    assert rec["trimmed"] == {"0-1": 3}
    assert [r["tried"] for r in rec["rounds"]] == [["0-1"], ["0-1"], ["0-1"], ["0-2"]]
    assert [r["kept"] for r in rec["rounds"]] == [["0-1"], ["0-1"], ["0-1"], []]
    assert rec["rounds"][3]["undone"] == ["0-2"] and rec["failed_last"] == 1
    assert len(calls) == 4                                             # one stitch per round, none to verify
    assert [len(g.quads) for g in st["grids"]] == [len(g.quads) for g in out]   # the mesh is the returned grids'


def test_a_kept_trim_is_verified_without_the_undone_ones(monkeypatch):
    """Two disjoint fake corridors A-B and C-D in one round: A-B looks better only while C-D is trimmed too, C-D
    never improves. Stitch 1 undoes C-D; stitch 2 (A-B's trim alone) shows A-B no better: undone too, nothing kept,
    the blocks as they were."""
    import meshom.methods.m14.stitch as S

    grids = [_block(0, 0, 20, 20), _block(0, 26, 20, 20), _block(60, 0, 20, 20), _block(60, 26, 20, 20)]
    n = []

    def fake_stitch(domain, h, gs, o, fixed):
        n.append(1)
        return None, {"grids": gs}

    def fake_score(mesh, h, st, o):
        a, b, c, d = st["grids"]
        cd_trimmed = c.ij[:, 1].max() < 20
        out = {}
        for p, x, y, ln in (((0, 1), a, b, 0.25 if cd_trimmed else 0.30), ((2, 3), c, d, 0.30)):
            m = {"ln_sh_p99": ln, "runs": {p[0]: _side(x, 1, True), p[1]: _side(y, 1, False)}}
            m["failed"], m["excess"] = G.gate_eval(m)
            out[p] = m
        return out

    monkeypatch.setattr(S, "stitch", fake_stitch)
    monkeypatch.setattr(G, "score", fake_score)
    _, _, out, rec = G.adapt(None, None, grids, O, [], None, {"grids": grids})
    r = rec["rounds"][0]
    assert r["tried"] == ["0-1", "2-3"] and r["undone"] == ["2-3", "0-1"] and r["kept"] == []
    assert len(n) == 2 and len(rec["rounds"]) == 1 and rec["trimmed"] == {}
    assert all(g is h for g, h in zip(out, grids))


@pytest.mark.parametrize("judge", ["raw", "post"])
def test_post_judge_keeps_the_trim_the_later_stages_fix(monkeypatch, judge):
    """B63, gap_judge "post": A (0) with B (1) 6 cells above it. On the raw stitch the fake corridor A-B gets worse as
    it widens (|ln s/h| p99 0.20 + 0.02 per extra cell); on the mesh the variant's later stages make of it (`post`,
    here a marker) it gets better (0.30 - 0.02 per cell: passes at 12 cells). The raw judge undoes the first trim; the
    post judge keeps three trims (rows 1 per block and round), scores every stitch on the post mesh, its elements
    located in the stitch's partition (locate_all)."""
    import meshom.methods.m14.stitch as S

    grids = [_block(0, 0, 20, 20), _block(0, 26, 20, 20)]
    seen = []

    def fake_stitch(domain, h, gs, o, fixed):
        return "raw", {"grids": gs}

    def fake_post(mesh, st):
        assert mesh == "raw"
        return "post"

    def fake_score(mesh, h, st, o, locate_all=False):
        seen.append((mesh, locate_all))
        a, b = st["grids"]
        w = int(b.ij[:, 1].min() - a.ij[:, 1].max())
        ln = 0.30 - 0.02 * (w - 6) if mesh == "post" else 0.20 + 0.02 * (w - 6)
        m = {"ln_sh_p99": ln, "runs": {0: _side(a, 1, True), 1: _side(b, 1, False)}}
        m["failed"], m["excess"] = G.gate_eval(m)
        return {(0, 1): m}

    monkeypatch.setattr(S, "stitch", fake_stitch)
    monkeypatch.setattr(G, "score", fake_score)
    _, _, out, rec = G.adapt(None, None, grids, dict(O, gap_rounds=6), [], "raw", {"grids": grids},
                             post=fake_post if judge == "post" else None)
    a, b = out
    if judge == "raw":
        assert rec["trimmed"] == {} and int(b.ij[:, 1].min() - a.ij[:, 1].max()) == 6
        assert set(seen) == {("raw", False)}
        return
    assert rec["trimmed"] == {"0-1": 3} and int(b.ij[:, 1].min() - a.ij[:, 1].max()) == 12
    assert set(seen) == {("post", True)}


def test_mix_on_uniform_is_valid_with_the_returned_blocks():
    from meshom.cases import CASES
    from meshom.compete import irregular_counts
    from meshom.mesh import signed_area
    from meshom.methods import m14_blocks as M
    from meshom.size import scaled

    dom, h0 = CASES["uniform"]
    mesh, info = M.run(dom, scaled(h0, 4.0), {"fill_method": "mix", "gap_adapt": True})
    g = info["gaps"]
    assert (signed_area(mesh) > 0).all() and irregular_counts(mesh)["euler_ok"]
    assert sum(b["n_cells"] for b in (info["blocks"][k] for k in info["blocks_kept"])) == \
        info["stitch"]["n_block_quads"]
    assert set(g["trimmed"]) <= {k for k, v in g["first"].items() if v["failed"]}
    assert g["cells_removed"] == sum(info["blocks"][k].get("gap_cells_removed", 0) for k in info["blocks_kept"])
    assert (g["cells_removed"] > 0) == bool(g["trimmed"])


def test_a_lake_whose_corridors_hold_no_triangle():
    """seam_rot0_k0 with its layout hint, the seeds grown one after the other (front off, B67): the _mix corridor is
    an all-quad ladder (no triangle to cluster); scoring it raised IndexError before. (With the shared front this
    lake's corridor holds 3 triangles.)"""
    from meshom.cases import CASES
    from meshom.cases_seams import SEAM_HINTS
    from meshom.methods import m14_blocks as M

    dom, h = CASES["seam_rot0_k0"]
    mesh, info = M.run(dom, h, {**SEAM_HINTS["seam_rot0_k0"], "fill_method": "mix", "gap_adapt": True,
                                "front": False})
    g = info["gaps"]
    assert g["pairs_first"] >= 1 and all(v["tri_cluster"] == 0 for v in g["first"].values())

