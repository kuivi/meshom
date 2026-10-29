"""The config's [[blocks]] (hand-placed blocks, decisions U6, B60): the TOML round trip and the UI's JSON round trip."""
import json

import pytest

from meshom import config

pytest.importorskip("tomli_w")

TEXT = """
[mesh]
method = "m14_blocks_cut"
[[size.layers]]
kind = "constant"
h_m = 100000.0
[[blocks]]
name = "Atlantic"
corners = [[-42, -20], [15.0, -20.0], [-7.0, 40.0], [-76.0, 40.0]]
rows = 64
cols = 70.0
h_m = 100000
sides = ["lat", "cut", "lat", "cut"]
snap_m = 0
filter_m = 50000.0
[[blocks]]
corners = [[-6.0, 35.8], [-5.4, 35.8], [-5.4, 36.1], [-6.0, 36.1]]
"""


def test_blocks_round_trip():
    cfg = config.loads(TEXT)
    a, b = cfg.blocks
    assert a.corners[0] == [-42.0, -20.0] and isinstance(a.corners[0][0], float) and a.cols == 70
    assert isinstance(a.h_m, float) and a.snap_m == 0.0 and a.sides == ["lat", "cut", "lat", "cut"]
    assert (b.name, b.rows, b.cols, b.h_m, b.sides, b.snap_m, b.filter_m) == ("", None, None, None, None, None, None)
    out = config.dumps(cfg)
    assert config.loads(out) == cfg and out.count("[[blocks]]") == 2 and "sides" not in out.split("[[blocks]]")[2]
    assert config.loads(TEXT.split("[[blocks]]")[0] + "[blocks]\n").blocks == []      # the old empty [blocks] table
    with pytest.raises(config.ConfigError, match=r"blocks\[1\]\.colz: unknown key"):
        config.loads(TEXT + "colz = 3\n")


def test_blocks_ui_round_trip():
    """load -> the page's JSON (JavaScript turns integral floats into integers) -> from_json: the same config."""
    from ui.common import scenario

    def jsify(x):
        if isinstance(x, dict):
            return {k: jsify(v) for k, v in x.items()}
        if isinstance(x, list):
            return [jsify(v) for v in x]
        return int(x) if isinstance(x, float) and x.is_integer() else x

    cfg = config.loads(TEXT)
    page = jsify(json.loads(json.dumps(scenario.to_json(cfg))))
    back = scenario.from_json(page, cfg)
    assert back == cfg and config.dumps(back) == config.dumps(cfg)
