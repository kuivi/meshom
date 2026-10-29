"""m14_blocks_rep leaves its fixed blocks alone (decisions R24): the repair, the coast stage, the coast fix, the fit
and the optimisation stage never change a frozen cell nor move its nodes; the cells next to them are still worked on."""
import numpy as np
import pytest

from meshom.cases import CASES
from meshom.mesh import Mesh
from meshom.size import scaled


def cells_on(old: Mesh, cells: np.ndarray, new: Mesh) -> np.ndarray:
    """Element mask of `new`: the cells of `old` found in `new` on nodes at the same positions (independent of
    rep_variants.cells_on)."""
    from scipy.spatial import cKDTree

    def el(m):
        return np.r_[np.c_[m.tris, np.full(m.n_tris, -1)], m.quads]
    d, j = cKDTree(new.points).query(old.points)
    nmap = np.where(d < 1e-9, j, -2)
    nmap = np.r_[nmap, -1]                               # index -1 (triangle padding) -> -1
    want = {tuple(sorted(r)) for r in nmap[el(old)[cells]].tolist() if -2 not in r}
    return np.array([tuple(sorted(r)) in want for r in el(new).tolist()], dtype=bool)


def _same_nodes(old: Mesh, cells: np.ndarray, new: Mesh) -> bool:
    """Every node of the cells of `old` is a node of `new` at the same position."""
    from scipy.spatial import cKDTree

    el = np.r_[np.c_[old.tris, np.full(old.n_tris, -1)], old.quads][cells]
    nodes = np.unique(el[el >= 0])
    return bool((cKDTree(new.points).query(old.points[nodes])[0] < 1e-9).all())


def test_frozen_cells_by_kind():
    from meshom.methods.rep_variants import frozen_cells

    rec = {"fixed_id": np.array([-1, 0, 0, 1, 2, -1]), "fixed_kind": ["hand", "strait", "hand"]}
    assert frozen_cells(rec, ("hand",)).tolist() == [False, True, True, False, True, False]
    assert frozen_cells(rec, ("hand", "strait")).tolist() == [False, True, True, True, True, False]
    assert frozen_cells(rec, ("band",)) is None and frozen_cells(rec, ()) is None and frozen_cells(None, ("hand",)) is None


def test_repair_keeps_frozen_cells_and_repairs_next_to_them():
    """A lattice with a rotated edge (two 3-5 pairs): the plain repair changes 6 of the 16 cells two rows away; frozen,
    all 16 stay with their nodes, the mask comes back, and the defect beside them is still removed."""
    from meshom.compete import irregular_counts
    from meshom.geo import xyz_to_lonlat
    from meshom.repair import repair
    from tests.test_repair import DOM, H, _lattice_with_rotated_edge

    m = _lattice_with_rotated_edge()
    la = xyz_to_lonlat(m.points[m.quads].mean(axis=1))[1]
    fz = np.r_[np.zeros(m.n_tris, bool), la > 40.18]
    cells = np.flatnonzero(fz)
    out0, _ = repair(m, H, DOM)
    assert cells_on(m, cells, out0).sum() < len(cells)                # what the freeze prevents
    out, info = repair(m, H, DOM, frozen=fz)
    kept = cells_on(m, cells, out)
    assert kept.sum() == len(cells) and _same_nodes(m, cells, out)
    assert np.array_equal(info["frozen_after"], kept) and info["frozen_cells"] == len(cells)
    ic = irregular_counts(out)
    assert sum(info["accepted"].values()) >= 1 and ic["n3"] == 0 and ic["n5"] == 0


def test_optimize_keeps_frozen_cells():
    """A fan of 4 triangles in a quad patch: the stage collapses its centre either way; the 8 quads round it change
    without the freeze (re-split or moved by the fit), frozen they stay."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_optimize import H, _fan

    from meshom.optimize import optimize, summary

    m, c = _fan()
    fz = np.r_[np.zeros(m.n_tris, bool), np.isin(m.quads, np.unique(m.tris)).any(axis=1)]
    cells = np.flatnonzero(fz)
    out0, _ = optimize(m, H, None, None)
    assert cells_on(m, cells, out0).sum() == 0
    out, info = optimize(m, H, None, None, frozen=fz)
    assert cells_on(m, cells, out).sum() == len(cells) and _same_nodes(m, cells, out)
    assert out.n_nodes == m.n_nodes - 1 and summary(out)["tri_cluster_max"] == 0
    assert info["checks"]["euler_kept"] and info["checks"]["boundary_kept"]


def test_coast_fix_never_splits_a_frozen_quad():
    from meshom.methods.m14.coastfix import coast_fix
    from tests.test_m14_coastfix import _kite

    m = _kite(147.0)
    assert coast_fix(m, m.node_flag != 0)[0].n_quads == 0                  # split without the freeze
    out, keep, rec = coast_fix(m, m.node_flag != 0, frozen=np.array([True]))
    assert out.n_quads == 1 and rec["quads_split"] == 0 and np.array_equal(out.points, m.points)


def _channel(scale: float):
    from meshom.methods.m14.hand import resolve
    from tests.test_hand_blocks import _channel_corners

    dom, h0 = CASES["continent"]
    h = scaled(h0, scale)
    return dom, h, resolve({"name": "channel", "corners": _channel_corners(500e3, 0.0), "cols": 3}, dom.loops, h)


def _hand_cells(spy: dict) -> tuple[Mesh, np.ndarray]:
    st = spy["st"]
    i = st.get("fixed_kind", ["hand"]).index("hand")     # (before R24 no kinds: the hand block comes first)
    return spy["mesh"], np.flatnonzero(st["fixed_id"] == i)


@pytest.fixture
def stitch_spy(monkeypatch):
    """The mesh and record of m14's last stitch (the mesh the variant's later stages start from)."""
    import meshom.methods.m14.stitch as ms

    seen, orig = {}, ms.stitch

    def spy(*a, **k):
        mesh, st = orig(*a, **k)
        seen.update(mesh=mesh, st=st)
        return mesh, st
    monkeypatch.setattr(ms, "stitch", spy)
    return seen


def test_rep_cut_keeps_the_hand_block(stitch_spy):
    """meshom run's land cut (the closed sphere, the coast as `land`), with the variant's optimisation stage: every
    cell of the hand block is kept, its nodes unmoved, and the land cut protects them all (before R24: 0 of 24 kept
    at scale 8, the nodes moved by 16-92 km)."""
    from meshom.domain import closed_sphere
    from meshom.methods import METHODS, m14_blocks
    from meshom.run import cut_land

    dom, h, spec = _channel(8.0)
    m = METHODS["m14_blocks_rep"](closed_sphere(), h, {"blocks": [spec], "land": dom, "optimize": True})
    old, cells = _hand_cells(stitch_spy)
    assert len(cells) > 0
    kept = cells_on(old, cells, m)
    assert kept.sum() == len(cells) and _same_nodes(old, cells, m)
    rec = m14_blocks.CUT_FIXED
    assert np.array_equal(rec["protect"], kept)
    out, info = cut_land(m, dom, fixed=rec)
    assert info["n_block_protected"] == len(cells)


def test_rep_conforming_keeps_the_hand_block(stitch_spy):
    """Conforming (the coast as the domain): repair, coast stage, coast fix, fit and the optimisation stage, the hand
    block kept whole; freeze_blocks = () gives the old behaviour (the block's cells changed)."""
    from meshom.methods import METHODS

    dom, h, spec = _channel(8.0)
    opts = {"blocks": [spec], "strait_blocks": False}
    m = METHODS["m14_blocks_rep"](dom, h, opts)
    old, cells = _hand_cells(stitch_spy)
    assert cells_on(old, cells, m).sum() == len(cells) and _same_nodes(old, cells, m)
    m0 = METHODS["m14_blocks_rep"](dom, h, {**opts, "freeze_blocks": ()})
    old0, cells0 = _hand_cells(stitch_spy)
    assert cells_on(old0, cells0, m0).sum() < len(cells0)
