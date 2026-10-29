import json

import numpy as np

from helpers import cube_sphere, structured_patch
from meshom import compete
from meshom.cli import main
from meshom.quality.metrics import size_fidelity
from meshom.size import uniform


def test_irregular_counts():
    ic = compete.irregular_counts(cube_sphere())                       # 8 valence-3 nodes, 6 quads
    assert (ic["n3"], ic["n5"], ic["T"], ic["euler_sum"], ic["chi"], ic["euler_ok"]) == (8, 0, 0, 8, 2, True)
    ic = compete.irregular_counts(structured_patch(6, 5, 1e4))          # a disc: each corner node carries +1
    assert (ic["n_interior"], ic["n_other"], ic["euler_sum"], ic["chi"], ic["euler_ok"]) == (20, 0, 4, 1, True)


def test_size_fidelity():
    m = structured_patch(6, 5, 1e4)
    f = size_fidelity(m, uniform(1e4))
    assert f["size_fidelity"]["p99"] < 1e-3 and f["size_fidelity_coast"]["max"] < 1e-3
    f = size_fidelity(m, uniform(2e4))["size_fidelity"]
    assert np.isclose(f["median"], np.log(2), atol=1e-3) and np.isclose(f["bias"], -np.log(2), atol=1e-3)


def test_compete_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(compete, "WORK_DIR", tmp_path / "work")
    j, md = tmp_path / "c.jsonl", tmp_path / "t.md"
    md.write_text("# head\n\nnotes\n")
    main(["compete", "--methods", "m00", "--scale", "8", "--jsonl", str(j), "--out", str(md)])
    recs = {r["case"]: r for r in map(json.loads, j.read_text().splitlines())}
    assert set(recs) == {"uniform", "disc", "band", "continent"}
    assert all(r["status"] == "ok" and r["gates_total"] > 0 for r in recs.values())
    for c in ("uniform", "disc", "band"):                               # closed sphere: sum(4 - deg) + T = 8
        assert recs[c]["irregular"]["euler_sum"] == 8 and recs[c]["irregular"]["euler_ok"]
    ic = recs["continent"]["irregular"]
    assert ic["chi_domain"] == -1 and ic["euler_sum"] == -4 and ic["euler_ok"]      # three coast loops
    assert recs["disc"]["rho_c"]["plus"] > 20 and recs["continent"]["quality"]["size_fidelity_coast"]["p99"] > 0
    main(["compete", "--table-only", "--jsonl", str(j), "--out", str(md)])         # block replaced, notes kept
    text = md.read_text()
    assert text.startswith("# head\n\nnotes\n") and text.count(compete.TABLE_BEGIN) == 1
    assert text.count("| m00_background |") == 4


def test_timeout_and_smooth_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(compete, "WORK_DIR", tmp_path / "work")
    j = tmp_path / "c.jsonl"
    r = compete.run_case("m00_background", "uniform", 8, timeout=0.01, out=j)
    assert r["status"] == "timeout" and "quality" not in r and "| timeout |" in compete.table(j)
    r = compete.run_case("m00_background", "uniform", 8, smooth=True, out=j)
    if compete.smooth_available():
        assert r["after_smoothing"]["smoothed"] and r["after_smoothing"]["status"] == "ok"
    else:
        assert "after_smoothing" not in r and "--smooth skipped" in capsys.readouterr().out
