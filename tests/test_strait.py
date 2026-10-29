"""m14 strait blocks (meshom/methods/m14/strait.py) and the re-assembled coast loops of the stitch (decisions S31,
B59, C35)."""
import numpy as np
import pytest

from meshom.cases import CASES
from meshom.domain import CHANNEL_WIDTH_M, Domain, continent_frame, loop_signed_area
from meshom.geo import R_EARTH_M
from meshom.methods.m14.stitch import merged_loops
from meshom.methods.m14.strait import strait_blocks
from meshom.methods.m14_blocks import DEFAULTS
from meshom.mesh import Mesh, signed_area
from meshom.size import scaled


@pytest.fixture(scope="module")
def continent_blocks():
    dom, h = CASES["continent"]
    return dom, h, strait_blocks(dom, h, dict(DEFAULTS))


def test_continent_channel_block(continent_blocks):
    dom, h, (blocks, info) = continent_blocks
    assert info["n_passages"] == 2 and len(blocks) == 1      # the island gap is too short (m = 12 < ... n = 19)
    fb = blocks[0]
    r = fb.record
    assert (r["n"], r["m"]) == (10, 444) and r["status"] == "accepted"
    assert r["lnsh_max"] <= 0.01 and r["quad_max_deg"] <= 91.0 and r["inverted"] == 0
    g = fb.grid
    assert len(g.quads) == 4440 and fb.coast.sum() == 2 * 445
    assert (signed_area(Mesh(g.points, np.zeros((0, 3), np.int64), g.quads, np.zeros(len(g.points), np.int8)))
            > 0).all()
    assert [m["cells"] for m in r["mouths"]] == [10, 10]
    assert {s[0] for _, s, e in fb.runs} == {0, 1} and {e[0] for _, s, e in fb.runs} == {0, 1}


def test_merged_loops(continent_blocks):
    dom, h, (blocks, _) = continent_blocks
    loops, ids, touched = merged_loops(dom, h, blocks)
    assert touched == {0, 1} and len(loops) == 1             # the two half caps and the block: one hole
    i = ids[0]
    assert (i >= 0).sum() == 2 * 11                          # the two mouths' nodes are the block's own
    np.testing.assert_array_equal(loops[0][i >= 0], blocks[0].grid.points[i[i >= 0]])


def test_no_block_where_h_does_not_resolve_the_strait():
    """S31: at scale 4 (h = 60 km in the 150 km channel) a 5-cell block would be 30 km cells: no block."""
    dom, h = CASES["continent"]
    blocks, info = strait_blocks(dom, scaled(h, 4.0), dict(DEFAULTS))
    ch = [r for r in info["tried"] if r["W_median_m"] < 2 * CHANNEL_WIDTH_M]
    assert [r["status"] for r in ch] == ["h_mismatch"] and ch[0]["ln_hb_h_median"] == pytest.approx(np.log(2), 0.01)
    assert all(b.record["W_median_m"] > 2 * CHANNEL_WIDTH_M for b in blocks)


def test_island_inside_leaves_the_strait_to_the_fill():
    """A 2 km island 5 km off a bank in mid-channel: the axis passes it without a junction, the block would cover
    it, so the channel gets no block."""
    dom, h = CASES["continent"]
    c, t, n = continent_frame()
    off = (0.5 * CHANNEL_WIDTH_M - 5e3) / R_EARTH_M
    ctr = np.cos(off) * c + np.sin(off) * n
    e1, e2 = np.cross(ctr, n), n - (n @ ctr) * ctr
    e1, e2 = e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)
    ph = np.linspace(0.0, 2 * np.pi, 9)[:-1]
    r = 2e3 / R_EARTH_M
    isl = np.cos(r) * ctr + np.sin(r) * (np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2)
    isl = isl if loop_signed_area(isl) < 0 else isl[::-1]                  # land on the right
    blocks, info = strait_blocks(Domain(dom.loops + [isl], "continent_island"), h, dict(DEFAULTS))
    assert blocks == [] and "island" in {r["status"] for r in info["tried"]}


def test_no_block_where_h_b_is_off_h_by_0p2():
    """B22 / S31 at TRIM = 0.10: at scale 2.45 the channel's n = 5 cells give h_b = W/5 off h by |ln| = 0.20 (as on
    Gibraltar's narrows): no block (h_mismatch), where an acceptance at 0.5 would try one."""
    dom, h = CASES["continent"]
    _, info = strait_blocks(dom, scaled(h, 2.45), dict(DEFAULTS))
    ch = [r for r in info["tried"] if r["W_median_m"] < 2 * CHANNEL_WIDTH_M]
    assert [r["status"] for r in ch] == ["h_mismatch"] and 0.15 < ch[0]["ln_hb_h_median"] < 0.25
