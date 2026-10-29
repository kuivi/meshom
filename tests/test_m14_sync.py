"""m14 frame and count synchronisation (m14/sync.py; decisions B54-B56):
- the L1 LP turns the blocks of a 3-block toy graph so that every corridor's rotation vanishes where the cycle allows,
  puts a cycle's residual on the shortest corridor (the cheapest wall), and turns a block beside the coast to the
  coast's fixed frame;
- the count shifts stay within |sigma_b| <= s_b = f_max - F_b and equalise the along counts where the bound allows;
- a block re-extracted from its map turned by psi has its lattice turned by psi (CCW seen from outside), and scaled by
  e^sigma at the bound s_b its cells stay inside the size budget F <= 0.08."""
import numpy as np
import pytest

from meshom.cases import CASES
from meshom.geo import R_EARTH_M, arc_length, tangent_basis
from meshom.methods.m14 import sync as S
from meshom.methods.m14_blocks import DEFAULTS, build_blocks
from meshom.size import scaled


def _cor(a, b, phi, n):
    return S.Corridor(a, b, np.full(n, float(phi)), float(n), None, None)


def _rot(cs, psi):
    return [S._wrap90(c.phi + (psi[c.b] if c.b != S.COAST else 0.0) - psi[c.a]) for c in cs]


def test_lp_turns_three_blocks():
    th = np.array([0.0, 20.0, -10.0])                                  # each block's lattice angle
    cs = [_cor(0, 1, th[1] - th[0], 40), _cor(1, 2, th[2] - th[1], 40), _cor(0, 2, th[2] - th[0], 10)]
    psi, log = S.solve_frames(cs, 3)
    assert max(np.abs(r).max() for r in _rot(cs, psi)) < 1e-6          # a consistent cycle: every rotation removed
    assert np.allclose(S._wrap90(psi - psi[0]), S._wrap90(-(th - th[0])), atol=1e-6)
    assert abs(np.degrees(np.angle(np.exp(4j * np.radians(psi)).sum())) / 4.0) < 1e-6   # no turn beyond the need
    # a cycle residual of 6 deg (holonomy) goes to the shortest corridor, not spread over the three
    cs[2] = _cor(0, 2, th[2] - th[0] + 6.0, 10)
    psi, log = S.solve_frames(cs, 3)
    r = _rot(cs, psi)
    assert np.abs(r[0]).max() < 1e-6 and np.abs(r[1]).max() < 1e-6 and np.allclose(np.abs(r[2]), 6.0)
    assert log["cost_lp"] == pytest.approx(60.0) and log["cost_lp"] <= log["cost_zero"]
    # a 90-deg branch: lattices 88 deg apart are 2 deg apart (4-fold)
    psi, _ = S.solve_frames([_cor(0, 1, 88.0, 20)], 2)
    assert abs(S._wrap90(88.0 + psi[1] - psi[0])) < 1e-6 and abs(psi[1] - psi[0]) < 2.0 + 1e-6
    psi, _ = S.solve_frames(cs, 4)                                     # block 3 has no corridor: no turn
    assert psi[3] == 0.0


def test_coast_is_a_fixed_frame():
    cs = [_cor(0, S.COAST, 15.0, 30), _cor(0, 1, -20.0, 30)]           # the coast 15 deg from block 0's lattice
    psi, _ = S.solve_frames(cs, 2)
    assert psi[0] == pytest.approx(15.0) and psi[1] == pytest.approx(35.0)   # block 0 to the coast, block 1 to block 0
    assert all(np.abs(r).max() < 1e-6 for r in _rot(cs, psi))


def test_count_shift_bound():
    s_max = np.array([0.03, 0.05, 0.08 - 0.075])                       # s_b = f_max - F_b
    cs = [S.Corridor(0, 1, np.zeros(10), 100.0, 120, 100), S.Corridor(1, 2, np.zeros(10), 100.0, 100, 101)]
    sig = S.solve_counts(cs, s_max)
    assert (np.abs(sig) <= s_max + 1e-12).all()
    assert sig[0] == pytest.approx(0.03) and sig[1] == pytest.approx(-0.05)   # ln(120/100) = 0.18: both at the bound
    # a mismatch the budget can absorb is absorbed (up to the ridge)
    cs = [S.Corridor(0, 1, np.zeros(10), 100.0, 102, 100)]
    sig = S.solve_counts(cs, np.array([0.05, 0.05]))
    assert abs(np.log(102 / 100) - sig[0] + sig[1]) < 2e-3
    # a block that already carries a shift (a turn retried with its cells e^-s): the total stays within the bound
    cs = [S.Corridor(0, 1, np.zeros(10), 100.0, 100, 100)]
    sig = S.solve_counts(cs, np.array([0.05, 0.05]), sig0=np.array([-0.05, 0.0]))
    assert (np.abs(sig) <= 0.05 + 1e-12).all() and abs(sig[0] - sig[1] + 0.05) < 2e-3   # counts taken at -0.05
    # coast corridors carry no count
    assert not S.solve_counts([S.Corridor(0, S.COAST, np.zeros(5), 5.0, 50, None)], np.array([0.05])).any()


@pytest.fixture(scope="module")
def blocks():
    dom, h0 = CASES["uniform"]
    h = scaled(h0, 4.0)
    o = dict(DEFAULTS)
    fit, regions, grids, kept, plog, t = build_blocks(dom, h, o)
    return [grids[k] for k in kept], h, o


def _lattice_angle(g):
    """4-fold mean direction (deg) of the block's i-edges in the tangent plane at the block's centre."""
    c = g.points.mean(axis=0)
    c /= np.linalg.norm(c)
    e1, e2 = tangent_basis(c[None])
    d = g.points[g.quads[:, 1]] - g.points[g.quads[:, 0]]
    a = np.arctan2(d @ e2[0], d @ e1[0])
    return np.degrees(np.angle(np.exp(4j * a).mean())) / 4.0


def _ln_sh(g, h):
    e = np.r_[g.quads[:, [0, 1]], g.quads[:, [1, 2]]]
    p, q = g.points[e[:, 0]], g.points[e[:, 1]]
    m = (p + q) / np.linalg.norm(p + q, axis=1, keepdims=True)
    return np.log(R_EARTH_M * arc_length(p, q) / h(m))


def test_reextracted_block_turns_by_psi(blocks):
    gs, h, o = blocks
    g0 = max(gs, key=lambda g: g.info["n_cells"])
    assert g0.map is not None
    for psi in (12.0, -30.0):
        g = S.extract_ok(g0, h, o, psi, 0.0)
        assert g is not None and g.info["sync_psi_deg"] == pytest.approx(psi)
        assert abs(S._wrap90(_lattice_angle(g) - _lattice_angle(g0) - psi)) < 1.0   # turned by psi, CCW from outside


def test_reextracted_block_scaled_within_the_budget(blocks):
    gs, h, o = blocks
    g0 = max((g for g in gs if g.info["n_cells"] >= 150), key=lambda g: o["f_max"] - g.info["F_map"])
    s = o["f_max"] - g0.info["F_map"]                                   # s_b = f_max - F_b
    assert s > 0.04
    ln0 = _ln_sh(g0, h)
    assert np.percentile(np.abs(ln0), 99) <= g0.info["F_map"] + 0.01   # a block's own noise above F: <= 0.01
    for sig in (s, -s):
        g = S.extract_ok(g0, h, o, 0.0, sig)
        assert g is not None and g.info["sync_sigma"] == sig
        ln = _ln_sh(g, h)
        assert abs(np.median(ln) - np.median(ln0) - sig) < 0.01          # cells e^sigma larger
        assert np.percentile(np.abs(ln), 99) <= o["f_max"] + 0.01       # inside F + |sigma| <= 0.08 (+ that noise)
        assert np.abs(ln).max() <= o["trim"]                            # the trim still holds


def test_sync_on_continent_lowers_the_rotation():
    """End to end on continent s4 (conforming: the coast term on): the realised rotational dislocations of the
    block-block corridors after the turn are below the first pass's, every kept block carries psi, sigma and its
    bound, sigma within it, and the record lists the corridors (block-block and coast)."""
    from meshom.domain import resample_loops

    dom, h0 = CASES["continent"]
    h = scaled(h0, 4.0)
    o = dict(DEFAULTS)
    fit, regions, grids, kept, plog, t = build_blocks(dom, h, o)
    kg = [grids[k] for k in kept]
    rec = S.sync(resample_loops(dom, h), kg, h, o, kept)
    before, turned = rec["pass1"]["before"]["blocks"], rec["pass2"]["turned"]["blocks"]
    assert rec["coast_term"] and turned["rot_per10"] < before["rot_per10"]
    assert {c["b"] for c in rec["corridors"]} >= {"coast"} and len(rec["corridors"]) >= before["corridors"]
    for g in kg:
        assert -45.0 < g.info["sync_psi_deg"] <= 45.0 and abs(g.info["sync_sigma"]) <= g.info["sync_s_max"] + 1e-12
