"""meshom.insert: a curvilinear strait insert fitted into a host mesh (decisions B158-B163)."""
import numpy as np
import pytest

from meshom.config import loads, dumps
from meshom.insert import block, synth
from meshom.insert.__main__ import parse_inline
from meshom.insert.core import Spec, insert_window
from meshom.insert.geom import Polyline, smooth_bank
from meshom.insert.host import cut_window, splice
from meshom.insert.measure import validity
from meshom.mesh import Mesh, build_topology
from meshom.quality.kernel import element_geometry
from meshom.size import Uniform

H = 10e3


@pytest.fixture(scope="module")
def banana():
    win, ch, path = synth.banana(H=H, L=400e3)
    res = insert_window(win, Spec("banana", path, cells=10, stages="", bank_cap_h=10.0), Uniform(H),
                        log=lambda *a, **k: None)
    return win, res


def test_block_of_a_rectangle_is_a_regular_grid():
    xa = np.c_[np.zeros(9), np.linspace(0, 80e3, 9)]
    xb = np.c_[np.full(9, 200e3), np.linspace(0, 80e3, 9)]
    bS = Polyline(np.c_[np.linspace(0, 200e3, 50), np.zeros(50)])
    bN = Polyline(np.c_[np.linspace(0, 200e3, 50), np.full(50, 80e3)])
    b = block.build(xa, xb, bS, bN, 8, fixed_count=True)
    assert len(b.tris) == 0 and (b.N == 8).all()
    assert np.allclose(b.xy[b.off[0]:b.off[0] + 9], xa) and np.allclose(b.xy[b.off[b.M]:b.off[b.M] + 9], xb)
    m = Mesh(np.c_[b.xy / 6.371e6, np.ones(len(b.xy))], np.zeros((0, 3), np.int64), b.quads, np.zeros(len(b.xy)))
    m.points = m.points / np.linalg.norm(m.points, axis=1, keepdims=True)
    a = np.degrees(element_geometry(m).quad_angles)
    assert 88.0 < a.min() and a.max() < 92.0                  # square cells: rows by the aspect iteration


def test_y_splits_change_the_count_and_stagger():
    ids, quads, tris, vert = block.topology(np.array([10, 10, 11, 12, 12, 13, 14, 14]))
    assert len(tris) == 4
    n = ids[-1][-1] + 1
    used = np.bincount(np.r_[quads.ravel(), tris.ravel()], minlength=n)
    assert (used > 0).all()


def test_smoothed_staircase_bank_stays_within_the_cap():
    k = np.arange(12)
    stair = np.c_[np.repeat(k, 2)[1:] * H, np.repeat(k, 2)[:-1] * H]   # a 45 deg staircase of H steps
    out = smooth_bank(stair, 1.5 * H, 0.2 * H, 0.1 * H)
    fwd = Polyline(stair).distance(out)
    back = Polyline(out).distance(stair)
    assert fwd.max() <= 0.2 * H + 1e-6 and back.max() <= 0.2 * H * 1.001
    raw_turn = np.abs(np.diff(np.arctan2(*np.diff(stair, axis=0).T[::-1])))
    turn = np.abs(np.diff(np.unwrap(np.arctan2(*np.diff(out, axis=0).T[::-1]))))
    assert turn.max() < raw_turn.max()                          # smoother than the staircase


def test_banana_insert_conforms_and_keeps_the_coast(banana):
    win, res = banana
    assert res.ok, res.record.get("refused")
    jA, jB = res.joins
    assert not jA.fails and not jB.fails
    m = res.window
    for j in (jA, jB):                                          # join nodes are the host's, unmoved
        d = np.min(np.linalg.norm(m.points[None, :, :] - win.mesh.points[j.nodes][:, None, :], axis=2), axis=1)
        assert d.max() < 1e-12
    v = validity(m)
    assert v["inverted"] == 0 and v["nonconvex"] == 0 and v["nonmanifold_edges"] == 0
    assert v["water_components"] == 1
    sc = res.record["scores"]["final"]
    assert sc["coast"]["shift_h_max"] <= 0.5                    # smooth banks: 1.5 cells of Gaussian, no cap
    assert sc["core"]["invalid"] == 0 and sc["core"]["quads"] > 300
    assert sc["core"]["qmax"] < 120.0                           # the rules passed: a clean core
    assert res.record["block"]["y_splits"] == sc["core"]["tris"] + sc["ring"]["tris"]


def test_banana_zoom_follows_the_neck(banana):
    _, res = banana
    b = res.record["block"]
    assert b["counts"][0] == res.joins[0].n and b["counts"][-1] == res.joins[1].n
    assert min(b["counts"]) == 10                               # the neck carries the requested cells
    assert b["h_min_m"] < 0.3 * H                               # zoom about x5 from the host


def test_coarse_host_is_refused_with_the_rule_and_a_fix():
    win, ch, path = synth.banana(H=40e3, L=400e3)
    res = insert_window(win, Spec("coarse", path, cells=10, stages="", reach_m=150e3), Uniform(40e3),
                        log=lambda *a, **k: None)
    assert not res.ok
    msg = " ".join(res.record["refused"])
    assert "room" in msg and "fix:" in msg


def test_coast_rule_against_smooth_banks_trade_off(banana):
    """On a staircase host the bank held within 0.2 h of the coast (default) keeps the coast rule but leaves the
    staircase in the bank (worse core angles) than the smooth bank (no cap)."""
    win, ch, path = synth.banana(H=H, L=400e3)
    keep = insert_window(win, Spec("keep", path, cells=10, stages=""), Uniform(H), log=lambda *a, **k: None)
    assert keep.ok
    k, s = keep.record["scores"]["final"], banana[1].record["scores"]["final"]
    assert k["coast"]["shift_h_max"] <= 0.25 < s["coast"]["shift_h_max"]
    assert k["core"]["qmax"] > s["core"]["qmax"]


def test_forced_build_with_failing_rules_records_them():
    win, ch, path = synth.butterfly(H=H)
    res = insert_window(win, Spec("bf", path, cells=16, stages="", force=True), Uniform(H), log=lambda *a, **k: None)
    assert res.ok
    assert any(j.fails for j in res.joins)                      # the rules' verdict is kept in the record
    assert any(j["fails"] for j in res.record["joins"])


def test_ring_cleaning_keeps_the_core_and_improves_the_ring():
    win, ch, path = synth.banana(H=H, L=400e3)
    res = insert_window(win, Spec("banana", path, cells=10, stages="d,b,e,a", force=True), Uniform(H),
                        log=lambda *a, **k: None)
    sc = res.record["scores"]
    assert sc["final"]["ring"]["q120"] < sc["merged"]["ring"]["q120"]
    if res.record.get("released_cells", 0) == 0:                # the core is cut out of the case: untouched
        assert sc["final"]["core"] == sc["merged"]["core"]
    else:                                                        # B166: its worst cells were released to the cleaning
        assert sc["final"]["core"]["q120"] < sc["merged"]["core"]["q120"]
    assert sc["final"]["coast"]["shift_h_max"] <= 0.25          # ref = the host coast (B162): no stacked shift
    assert validity(res.window)["inverted"] == 0


def test_window_splice_round_trip_keeps_ids():
    win, ch, path = synth.banana(H=H, L=400e3)
    host = win.mesh
    w = cut_window(host, host.points.mean(0), 80e3)
    assert (w.kind == 2).any() and (w.kind == 1).any()
    out, info = splice(host, w, w.mesh)
    assert info["appended_nodes"] == 0 and out.n_nodes == host.n_nodes
    assert np.array_equal(out.points, host.points)
    assert out.n_elems == host.n_elems


def test_inserts_table_and_inline_spec():
    cfg = loads('[mesh]\nmethod = "m00_background"\n[[inserts]]\nname = "gib"\nstrait = "Gibraltar"\ncells = 12\n'
                'mouth_a = [[-6.0, 35.8], [-6.0, 36.2]]\n')
    assert cfg.inserts[0].cells == 12 and cfg.inserts[0].mouth_a[1] == [-6.0, 36.2]
    assert loads(dumps(cfg)).inserts == cfg.inserts
    d = parse_inline("strait=Gibraltar; cells=10; path=[[-31.0, 64.5], [-22.0, 67.5]]; force=true")
    assert d == {"strait": "Gibraltar", "cells": 10, "path": [[-31.0, 64.5], [-22.0, 67.5]], "force": True}


def test_insert_h_layer_is_the_exact_cone_minimum():
    # the 16 nearest samples alone missed a farther, smaller one (h jumped, overestimated by 33 %)
    from meshom.insert.assemble import InsertH

    rng = np.random.default_rng(3)

    def patch(n):                                # points within ~100 km of (1, 0, 0), like a block's samples
        p = np.c_[np.ones(n), rng.uniform(-0.008, 0.008, (n, 2))]
        return p / np.linalg.norm(p, axis=1)[:, None]

    pts = patch(400)
    hk = rng.uniform(3e3, 5e3, 400)
    hk[:3] = 10.0                                # a few small samples: they win beyond the 16 nearest
    q = patch(300)
    lay = InsertH(lambda x: np.full(len(x), 1e9), pts, hk, 0.12)
    from meshom.geo import R_EARTH_M

    G = 0.12 * R_EARTH_M
    full = np.min(hk[None, :] + G * np.linalg.norm(q[:, None, :] - pts[None, :, :], axis=2), axis=1)
    assert np.allclose(lay(q), full, rtol=0, atol=1e-6)
    capped = InsertH(lambda x: np.full(len(x), 2e5), pts, hk, 0.12)
    assert np.allclose(capped(q), np.minimum(full, 2e5), rtol=0, atol=1e-6)


def test_mouth_hints_do_not_skip_the_rules():
    # hinted joins bypassed every refusal rule even with force off
    from meshom.insert import synth
    from meshom.size import Uniform

    win, ch = synth.lattice_host(lambda p: np.abs(p[:, 1]) <= 20000, (-150000, 150000, -70000, 70000), 10000)
    path = ch.inv(np.array([[-30000.0, 0.0], [30000.0, 0.0]]))
    kw = dict(cells=10, stages="", reach_m=80000, bank_sigma_h=0.0)
    quiet = lambda *a, **k: None                 # noqa: E731
    auto = insert_window(win, Spec("rect", path, **kw), Uniform(10000), log=quiet)
    assert not auto.ok
    cs = [min((j for j in auto.candidates if j.side == s and j.fails == ["room"]), key=lambda j: abs(j.d_m - 50000))
          for s in "AB"]
    hinted = Spec("rect_hinted", path, mouth_a=win.mesh.points[cs[0].nodes], mouth_b=win.mesh.points[cs[1].nodes],
                  **kw)
    r = insert_window(win, hinted, Uniform(10000), log=quiet)
    assert not r.ok and r.record.get("refused")


# ------------------------------------------------------------------ step 2: real coast, size floor, bays, joins

@pytest.fixture(scope="module")
def banana_real():
    win, ch, path, loops = synth.strait_real(H=H, L=400e3)
    quiet = lambda *a, **k: None                 # noqa: E731
    real = insert_window(win, Spec("real", path, cells=10, stages=""), Uniform(H), real=loops, log=quiet)
    host = insert_window(win, Spec("host", path, cells=10, stages="", coast="host"), Uniform(H), real=loops,
                         log=quiet)
    return win, real, host


def test_real_coast_banks_follow_the_real_coast_and_clean_the_core(banana_real):
    """B164: banks on the real coast (smoothed at the insert's cells), not on the host's staircase: the core is
    clean where the staircase bank held at 0.2 h leaves bad cells, and the coast shift is measured against the real
    coast."""
    win, real, host = banana_real
    assert real.ok and host.ok
    assert real.record["banks"] == ["real", "real"]
    r, h = real.record["scores"]["final"], host.record["scores"]["final"]
    assert r["coast_real"]["shift"]["h_ins_max"] <= 0.35     # the banks blend from the host join ends (B164)
    assert r["core"]["qmax"] < 110.0 < h["core"]["qmax"]
    assert r["core"]["invalid"] == 0
    # the join ends are the host's nodes, unmoved; the banks start on them exactly
    for j in real.joins:
        d = np.min(np.linalg.norm(real.window.points[None] - win.mesh.points[j.nodes][:, None], axis=2), axis=1)
        assert d.max() < 1e-12


def test_size_floor_caps_the_cells_across_and_says_so():
    win, ch, path, loops = synth.strait_real(H=H, L=400e3)
    quiet = lambda *a, **k: None                 # noqa: E731
    r = insert_window(win, Spec("floor", path, cells=10, stages="", h_min_m=3300.0), Uniform(H), real=loops,
                      log=quiet)
    p = r.record["profile"]
    W = p["neck_width_km"] * 1e3
    assert p["cells"] == int(W // 3300.0) and p["cells"] < 10
    assert "size floor" in p["floor"]
    if r.ok:
        assert r.record["block"]["h_min_m"] >= 3300.0 * 0.999
    from meshom.insert.region import InsertError

    with pytest.raises(InsertError, match="fewer than 2 cells"):
        insert_window(win, Spec("floor2", path, cells=10, stages="", h_min_m=0.6 * W), Uniform(H), real=loops,
                      log=quiet)


def test_a_bay_becomes_a_block_of_its_own():
    """B166: a deep bay on a bank is bridged by the main block and meshed as its own block (mouth = the main
    block's bank nodes, -1 corners at the mouth ends)."""
    win, ch, path, loops = synth.strait_real(H=H, L=400e3, bend=0.0, w_neck=40e3, bay=(10e3, 45e3, 40e3))
    r = insert_window(win, Spec("bay", path, cells=10, stages=""), Uniform(H), real=loops, log=lambda *a, **k: None)
    assert r.ok
    bays = [b for b in r.record["bays"] if b["status"] == "meshed"]
    assert len(bays) == 1 and bays[0]["side"] == "N" and bays[0]["cells"] >= 2
    sc = r.record["scores"]["final"]
    assert sc["core"]["invalid"] == 0
    m = r.window
    topo = build_topology(m)
    nq = np.bincount(m.quads.ravel(), minlength=m.n_nodes)
    assert ((nq == 3) & topo.boundary_node).sum() >= 2           # the two mouth ends: 3 quads at a coast node


def test_joins_prefer_host_grid_lines():
    """B167: the chain finder pays for turns, so a slightly oblique shape still gives the host column (tier 1)."""
    from meshom.insert.geom import Polyline
    from meshom.insert.host import coast_neighbours, graph
    from meshom.insert.region import Chains

    win, ch = synth.lattice_host(lambda p: np.abs(p[:, 1]) <= 40000, (-100000, 100000, -60000, 60000), 10000)
    xy = ch.fwd(win.mesh.points)
    g = graph(win.mesh)
    c = Chains(g, xy, coast_neighbours(win.coast_edges, win.mesh.n_nodes), win.mesh.quads)
    cn = np.unique(win.coast_edges)
    a = int(cn[np.argmin(np.linalg.norm(xy[cn] - [0, -40000], axis=1))])
    b = int(cn[np.argmin(np.linalg.norm(xy[cn] - [0, 40000], axis=1))])
    shape = Polyline(np.array([[-4000.0, -40000.0], [4000.0, 40000.0]]))
    p = c.path(a, b, shape)
    assert c.zig(p) == 0.0 and c.tier(p) == 1 and len(p) == 9


# ------------------------------------------------------------------ the floor and the held coast

def _made_cells_sizes(host, mesh):
    """Mean great-circle edge length of every cell of mesh that is not a host cell at the host's exact positions (the
    measure of core.host_min_cell), computed here without the module's own check."""
    from scipy.spatial import cKDTree

    d, j = cKDTree(host.points).query(mesh.points)
    hid = np.where(d < 1e-12, j, -1)
    keys = {tuple(r) for c in (host.tris, host.quads) for r in np.sort(c, axis=1).tolist()}
    g = element_geometry(mesh)
    me = np.r_[g.tri_edges.mean(1) if mesh.n_tris else np.zeros(0), g.quad_edges.mean(1)]
    made = np.array([min(r) < 0 or tuple(r) not in keys for c in (mesh.tris, mesh.quads) if len(c)
                     for r in np.sort(hid[c], axis=1).tolist()])
    return me[made]


def test_size_floor_holds_on_every_cell_the_insert_makes():
    """B165: h_min_m bounds the cells, not only the size target: the harmonic map squeezes
    the cells at the banana's inner bank to ~0.8 of a target held at the floor; the layout raises the target there."""
    win, ch, path, loops = synth.strait_real(H=H, L=400e3)
    r = insert_window(win, Spec("floor_cells", path, cells=10, stages="", h_min_m=2500.0), Uniform(H), real=loops,
                      log=lambda *a, **k: None)
    assert r.ok, r.record.get("refused")
    sizes = _made_cells_sizes(win.mesh, r.window)
    assert sizes.min() >= 2500.0                                 # no tolerance
    f = r.record["floor"]
    assert f["below"] == 0 and abs(f["smallest_m"] - sizes.min()) < 1e-6


def test_size_floor_refuses_a_build_it_cannot_lay_out(monkeypatch):
    """B165: a cell still below the floor after the layout rounds refuses the build (nothing written) and says which
    h_min_m or cells across would pass."""
    monkeypatch.setattr(block, "FLOOR_ROUNDS", 1, raising=False)  # no round may raise the target
    win, ch, path, loops = synth.strait_real(H=H, L=400e3)
    r = insert_window(win, Spec("floor_refused", path, cells=10, stages="", h_min_m=2500.0), Uniform(H), real=loops,
                      log=lambda *a, **k: None)
    assert not r.ok
    msg = " ".join(r.record["refused"])
    assert "size floor" in msg and "fix: h_min_m <=" in msg and "cells <=" in msg
    assert r.record["floor"]["below"] > 0


def test_cleaning_holds_the_joins_coast_ends_and_the_host_coast():
    """B162: with real-coast banks the cleaning slid the joins' coast end nodes (180-918 m)
    and 16 host coast nodes outside the replaced region (the old coast up to 1.3 km from the new). The coast end nodes
    and the host's coast outside the insert stay exactly in place; the inner join nodes may move (the cleaning repairs
    the join across them) and the insert's banks slide."""
    from scipy.spatial import cKDTree

    from meshom import straits as stm
    from meshom.insert import assemble

    win, ch, path, loops = synth.strait_real(H=H, L=400e3)
    r = insert_window(win, Spec("held", path, cells=10, stages="a"), Uniform(H), real=loops, log=lambda *a, **k: None)
    assert r.ok and r.record["valid"]
    tree = cKDTree(r.window.points)
    ends = [r.joins[0].nodes[0], r.joins[0].nodes[-1], r.joins[1].nodes[0], r.joins[1].nodes[-1]]
    assert tree.query(win.mesh.points[ends])[0].max() < 1e-12             # the joins' coast end nodes, exactly
    seed = int(stm.locate(win.mesh, r.chart.inv(r.profile.xy[r.profile.i0][None]), stm._elem_tree(win.mesh))[0])
    reg = assemble.region_between(win.mesh, seed, [j.nodes for j in r.joins], win.kind)
    T = win.mesh.n_tris
    inreg = np.unique(np.r_[win.mesh.tris[reg[:T]].ravel(), win.mesh.quads[reg[T:]].ravel()])
    coast = np.unique(win.coast_edges)
    outside = coast[~np.isin(coast, inreg)]
    assert tree.query(win.mesh.points[outside])[0].max() < 1e-12          # the host's coast outside, exactly
    assert r.record["clean"]["ring_sliding_nodes"] > 0                     # the insert's own banks slide
    assert r.record["host_coast"]["h_run_max"] == 0.0 and r.record["join_ends_moved"] == 0


def test_floor_rounds_never_make_the_block_coarser_than_the_host():
    """B165: the floor rounds raise the target at most to the host's size at the joins (step 2's cap);
    uncapped, Denmark's default floor (0.86 of the host's cell) raised it to 57 km, 2.2 host cells, the layout gave 3
    rows for 17 -> 14 -> 17 -> 21 cells and a quad folded. The layout keeps the rows its birth rule needs."""
    xa = np.c_[np.zeros(9), np.linspace(0, 80e3, 9)]             # host cells of 10 km on both joins
    xb = np.c_[np.full(9, 200e3), np.linspace(0, 80e3, 9)]
    bS = Polyline(np.c_[np.linspace(0, 200e3, 50), np.zeros(50)])
    bN = Polyline(np.c_[np.linspace(0, 200e3, 50), np.full(50, 80e3)])
    b = block.build(xa, xb, bS, bN, 8, h_min=0.0, cell_floor=15e3)  # a floor above the host's cell
    assert b.fine_h.max() <= 10e3 * (1 + 1e-9)
    assert b.info["cell_min_m"] < 15e3                           # the caller refuses it (core.insert_window)
    assert block.birth_rows(17, 14) + block.birth_rows(14, 21) == 4
