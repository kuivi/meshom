"""The bank edges of a fixed block one cell wide (decisions B59, B60): its mouths join two
coast-side nodes, as its banks do, and stay mouths: open in the land cut (m14/cut.py fixed_banks), seams in the
outline and in the report (m14/stitch.py, m14/report.py fixed_seams)."""
import pytest

from meshom.cases import CASES
from meshom.methods.m14_blocks import run
from meshom.size import scaled
from tests.test_hand_blocks import _channel_corners


@pytest.mark.parametrize("cut", [True, False])
def test_one_cell_wide_hand_block_keeps_its_mouths(cut):
    """A hand block one cell across the continent's channel (scale 4, cols = 1; sides coast, cut, coast, cut by the
    rule): in cut mode (variant A) no cell of it is removed and the channel stays open (3 boundary loops, as with
    cols = 2 or 3; before the fix the cells across both mouths became land, the block and the channel's water were
    dropped: 17 cells, 2 loops); in both modes its two mouths are seams of one edge each (before: 0)."""
    dom, h0 = CASES["continent"]
    spec = {"name": "channel", "corners": _channel_corners(500e3, 0.0), "cols": 1}
    m, info = run(dom, scaled(h0, 4.0), {"blocks": [spec], "strait_blocks": False, "optimize": False, "cut": cut})
    fb = info["fixed_blocks"]
    assert len(fb) == 1 and fb[0]["status"] == "built" and fb[0]["n"] == 1
    assert fb[0]["sides"] == ["coast", "cut", "coast", "cut"]
    if cut:
        assert info["cut"]["n_fixed_cells_removed"] == 0 and info["cut"]["n_loops"] == 3
    assert fb[0]["seam"]["mouth_edges"] == 2
