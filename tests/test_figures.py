import json

from meshom import compete
from meshom.cli import main

CLOSED = ["overview", "zoom_angle", "zoom_size", "zoom_irregular", "zoom_seam"]
COAST = CLOSED + ["zoom_coast", "zoom_strait"]


def _check(d, names):
    idx = json.loads((d / "index.json").read_text())
    assert [f["name"] for f in idx["figures"]] == names
    for f in idx["figures"]:
        assert (d / f["file"]).stat().st_size > 10_000
        if f["name"] != "overview":
            assert len(f["center_lonlat"]) == 2 and f["radius_km"] > 0 and f["numbers"]["n_quads"] > 0
    assert sorted(p.stem for p in d.glob("*.png")) == sorted(names)


def test_compete_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(compete, "WORK_DIR", tmp_path / "work")
    main(["compete", "--methods", "m01_isotropic", "--cases", "uniform", "continent", "--scale", "4", "--figures",
          "--jsonl", str(tmp_path / "c.jsonl")])
    _check(tmp_path / "work/m01_isotropic_uniform_s4/figures/raw", CLOSED)
    run = tmp_path / "work/m01_isotropic_continent_s4"
    _check(run / "figures/raw", COAST)
    strait = json.loads((run / "figures/raw/index.json").read_text())["figures"][-1]
    assert abs(strait["numbers"]["target_width_km"] - 150.0) < 10.0          # the continent channel
    main(["figures", str(run / "raw.npz"), "--case", "continent", "--out", str(tmp_path / "again")])
    _check(tmp_path / "again", COAST)
