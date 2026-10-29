import json

import numpy as np

from helpers import mixed_cube
from meshom.bench import append_jsonl, stage
from meshom.cli import main
from meshom.io.fesom import write_fesomc


def test_cli_quality(tmp_path):
    m = mixed_cube()
    write_fesomc(m, tmp_path / "mesh", depth_m=np.ones(m.n_nodes))
    main(["quality", str(tmp_path / "mesh"), "--out", str(tmp_path / "q"), "--bench", str(tmp_path / "b.jsonl")])
    for f in ["report.md", "summary.json", "hist_quad_dev.png", "hist_tri_angles.png", "hist_size_ratio.png",
              "map_flagged.png"]:
        assert (tmp_path / "q" / f).stat().st_size > 0
    assert json.loads((tmp_path / "q/summary.json").read_text())["n_quads"] == 5
    recs = [json.loads(line) for line in (tmp_path / "b.jsonl").read_text().splitlines()]
    assert [r["stage"] for r in recs] == ["read", "quality"] and all(r["n_nodes"] == 8 for r in recs)


def test_stage_record():
    rec = []
    with stage("x", rec, n=5):
        sum(range(10**6))
    assert rec[0]["wall_s"] > 0 and rec[0]["peak_rss_mb"] > 1 and rec[0]["n"] == 5


def test_bench_table(tmp_path):
    rec = []
    with stage("read", rec, mesh="a", n_nodes=1):
        pass
    append_jsonl(rec, tmp_path / "b.jsonl")
    main(["bench-table", "--in", str(tmp_path / "b.jsonl"), "--out", str(tmp_path / "b.md")])
    assert "| read |" in (tmp_path / "b.md").read_text()


def test_ui_help(capsys):
    import pytest

    with pytest.raises(SystemExit) as e:
        main(["ui", "--help"])
    out = capsys.readouterr().out
    assert e.value.code == 0 and "--allow-submit" in out and "--refs" in out and "meshom ui" in out
