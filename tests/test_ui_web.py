"""Web UI: the server starts, every endpoint answers on a coarse copy of the example config (the gradient-limited
preview and a window with a polygon across 180° included), the TOML round trip through the page's JSON is exact, the
indexed binary mesh payload decodes to the same arrays, a `meshom run` folder and a reference mesh are listed and
loaded with their quality.json records, Run writes a run folder with the real command (dry run). The requests go
through the handler in this process (ui/web/inproc.py): no test opens a port."""
import json
import shutil
import struct
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tomli_w")
pytest.importorskip("shapefile")

from ui_helpers import tiny_mesh, write_tiny_mesh  # noqa: E402

from meshom import compete, config  # noqa: E402
from meshom.domain import closed_sphere  # noqa: E402
from meshom.size import uniform  # noqa: E402
from ui.common import meshview, scenario  # noqa: E402
from ui.web import inproc, payload  # noqa: E402
from ui.web.server import STATIC, make_handler  # noqa: E402

SPHERE = """
[mesh]
method = "m01_isotropic"
[coast]
source = "none"
[size]
grad_max = 0.1
[[size.layers]]
kind = "constant"
h_m = 2000000.0
"""


def coarse(cfg: config.Config) -> config.Config:
    """The example config with coarse layers: the limiter grid has ~2.7e4 nodes."""
    L = [replace(cfg.size.layers[0], h_m=1000e3), replace(cfg.size.layers[1], h_m=400e3, plateau_m=0.0),
         replace(cfg.size.layers[2], h_m=300e3, transition_m=1000e3)]
    return replace(cfg, size=replace(cfg.size, layers=L))


def scored(d: Path, record: dict) -> None:
    """d/raw.npz = the tiny mesh, d/quality.json = {"raw": its _score record (uniform 2000 km) + record}."""
    m = tiny_mesh()
    d.mkdir(parents=True)
    np.savez(d / "raw.npz", points=m.points, tris=m.tris, quads=m.quads, node_flag=m.node_flag)
    rec = dict(status="ok", date="2026-10-22T12:00:00",
               **compete._score(m, closed_sphere(), uniform(2000e3), {}, config.DEFAULT_GATES), **record)
    (d / "quality.json").write_text(json.dumps({"raw": compete._round(rec)}))


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    if not scenario.resolve(scenario.load().coast.source).exists():
        pytest.skip("coastline not found")
    d = tmp_path_factory.mktemp("web")
    cfg = d / "c.toml"
    scenario.save(coarse(scenario.load()), cfg)
    mesh = write_tiny_mesh(d / "mesh")
    runs = d / "runs"
    run = runs / "m14_blocks_uniform_s4"                     # a compete-style run folder: h known from the name
    run.mkdir(parents=True)
    shutil.copy(mesh / "mesh.npz", run / "raw.npz")
    (run / "stages.json").write_text(json.dumps([{"stage": "generate", "wall_s": 1.5, "peak_rss_mb": 300}]))
    (run / "slurm-1.log").write_text("== start\nline 2\n== rc 0\n")
    (run / "m14.json").write_text(json.dumps({"method": "m14_blocks", "n_blocks": 1, "blocks_kept": [0],
                                              "coverage_area": 0.5, "blocks": [{
                                                  "seed_lonlat": [10.0, 5.0], "n_cells": 120, "B": 0.1,
                                                  "outline_lonlat": [[0, 0], [20, 0], [20, 10], [0, 10]],
                                                  "outline_corners": [0, 1, 2, 3]}]}))
    core = runs / "sphere"                                   # what `meshom run` writes (docs/config.md, "Running")
    scored(core, {"method": "m01_isotropic", "config": "sphere", "smoothed": False, "wall_s": 2.0})
    c = config.loads(SPHERE, "sphere")
    config.save(replace(c, mesh=replace(c.mesh, out=str(core))), core / "config.resolved.toml")
    (core / "stages.json").write_text(json.dumps([{"stage": "coast", "wall_s": 0.0}, {"stage": "size", "wall_s": 0.1}]))
    refs = d / "refs"
    scored(refs / "grid_x", {"family": "NEMO", "config": "test grid", "res": "20°"})
    shutil.move(refs / "grid_x" / "raw.npz", refs / "grid_x" / "mesh.npz")
    HANDLER[0] = make_handler(cfg, mesh, run_roots=[runs], ref_roots=[refs])
    return {"url": "", "dir": d, "mesh": mesh, "run": run, "cfg": cfg, "core": core, "ref": refs / "grid_x"}


HANDLER: list = [None]            # the srv fixture's handler: get() and post() send their requests to it


def _call(method, path, body=b""):
    s, _, b = inproc.call(HANDLER[0], method, path, body)
    assert s < 400, f"{s} {path}: {b.decode(errors='replace')}"      # the server's error text
    return s, b


def get(path):
    return _call("GET", path)


def post(path, obj):
    return json.loads(_call("POST", path, json.dumps(obj).encode())[1])


def jsify(x):
    """What JSON through JavaScript does to the config: integral floats come back as integers."""
    if isinstance(x, dict):
        return {k: jsify(v) for k, v in x.items()}
    if isinstance(x, list):
        return [jsify(v) for v in x]
    return int(x) if isinstance(x, float) and x.is_integer() else x


def test_page_and_static(srv):
    s, body = get(srv["url"] + "/")
    assert s == 200 and b"deck.gl@9.4.0" in body and b"static/app.js" in body
    for f in STATIC.glob("*.js"):
        assert get(f"{srv['url']}/static/{f.name}")[0] == 200


def test_every_endpoint_answers(srv):
    u = srv["url"]
    c = json.loads(get(u + "/api/config")[1])
    cfg = c["config"]
    assert "[[size.layers]]" in c["toml"] and "--out {dir}" in c["run_cmd"]
    for path in ("/api/coast/sources", "/api/methods", "/api/mesh/info", "/api/mesh/hist", "/api/mesh/worst.json?n=5",
                 "/api/runs", "/api/refs", f"/api/runs/log?path={srv['run']}", f"/api/runs/blocks?path={srv['run']}"):
        s, b = get(u + path)
        assert s == 200 and "error" not in json.loads(b), path
    for path in ("/api/coast.bin?spacing_m=50000", "/api/mesh/overview.png?proj=north", "/api/mesh/mesh.bin",
                 "/api/mesh/mesh.bin?sel=worst&n=10", "/api/mesh/mesh.bin?sel=window&lon=180&lat=0&r_km=2000",
                 "/api/mesh/mesh.bin?sel=tris", "/api/mesh/irregular.bin"):
        assert get(u + path)[0] == 200, path
    r0 = post(u + "/api/preview", {"config": cfg, "step": 4.0, "polar": ["north"]})
    r = post(u + "/api/preview", {"config": cfg, "step": 4.0, "polar": ["north"], "limited": True})
    assert r["png"] and r["png_winner"] and r["png_north"] and r["n_nodes_est"] > 0 and r["limited_applied"] is True
    assert r["limiter"]["limiter"]["nodes"] > 0 and r["h_max"] <= r0["h_max"] + 1e-6
    assert post(u + "/api/preview", {"config": cfg, "step": 4.0, "limited": True})["limiter"]["cached"] is True
    r = post(u + "/api/preview/window", {"config": cfg, "bounds": [170, 200, 55, 72], "step": 1.0, "polar": ["north"],
                                         "polygon": [[172, 58], [-165, 58], [-165, 70], [172, 70]]})
    assert r["png"] and r["png_north"] and len(r["bounds_north"]) == 4 and r["shape"] == [17, 30]
    r = post(u + "/api/rho", {"config": cfg, "step": 4.0, "limited": True})
    assert r["png"] and r["curv"] > 0
    assert post(u + "/api/toml", {"text": "[blocks]\na = 1\n"})["value"] == {"blocks": {"a": 1}}
    assert "[blocks]" in post(u + "/api/toml", {"config": {"blocks": {"a": 1}}})["toml"]
    runs = {x["name"]: x for x in json.loads(get(u + "/api/runs")[1])["runs"]}
    x = runs["m14_blocks_uniform_s4"]
    assert x["state"] == "done" and x["meshes"] == ["raw.npz"] and x["m14"]
    x = runs["sphere"]
    assert x["state"] == "done" and x["quality"]["raw"]["n_quads"] == 18 * 8 and x["line"].startswith("sphere ")
    b = json.loads(get(u + f"/api/runs/blocks?path={srv['run']}")[1])
    assert b["blocks"][0]["outline"] and b["blocks"][0]["corners"] == [0, 1, 2, 3] and b["note"] is None
    refs = json.loads(get(u + "/api/refs")[1])["refs"]
    assert [x["name"] for x in refs] == ["grid_x"] and refs[0]["quality"]["family"] == "NEMO"
    assert post(u + "/api/mesh/load", {"path": str(srv["run"])})["mesh"] == str(srv["run"])
    info = json.loads(get(u + "/api/mesh/info")[1])
    assert info["m14"] and info["has_ln_sh"] and info["quality"] is None      # h from the folder name (uniform, s4)
    post(u + "/api/mesh/load", {"path": str(srv["core"])})
    info = json.loads(get(u + "/api/mesh/info")[1])
    assert info["has_ln_sh"] and info["quality"]["gates_total"] > 0           # h from config.resolved.toml
    hist = {h["name"]: h for h in json.loads(get(u + "/api/mesh/hist")[1])["histograms"]}
    assert hist["quad θmax"]["record"]["max"] == pytest.approx(hist["quad θmax"]["max"], rel=1e-5)
    assert hist["|ln s/h| per edge"]["record"]["p99"] is not None
    post(u + "/api/mesh/load", {"path": str(srv["ref"])})
    info = json.loads(get(u + "/api/mesh/info")[1])
    assert info["quality"]["family"] == "NEMO" and not info["has_ln_sh"]
    hist = {h["name"]: h for h in json.loads(get(u + "/api/mesh/hist")[1])["histograms"]}
    assert hist["|ln s/h| per edge"]["counts"] == [] and hist["|ln s/h| per edge"]["record"]["p99"] is not None
    post(u + "/api/mesh/load", {"path": str(srv["mesh"])})


def test_methods_have_defaults(srv):
    m = {x["name"]: x for x in json.loads(get(srv["url"] + "/api/methods")[1])["methods"]}
    assert "m13_squadgen" in m and m["m02_linf_frontal"]["defaults"]["alpha"] == 0.75


def test_toml_round_trip_through_page_json(srv, tmp_path):
    u = srv["url"]
    cfg = scenario.load()
    cfg.mesh.method = "m02_linf_frontal"
    cfg.mesh.opts = {"alpha": 1.0, "verbose": True}                 # a float option that JSON turns into 1
    cfg.size.layers[1].enabled = False
    cfg.coast.smoothing_m = 5000.0
    cfg.bathymetry = replace(cfg.bathymetry, source="/data/GEBCO_2024.nc", min_depth_m=20.0,   # the Coast panel's keys
                             levels=[0.0, 10.0, 25.0, 4000.0])
    scenario.save(cfg, tmp_path / "a.toml")
    cfg = scenario.load(tmp_path / "a.toml")
    r = post(u + "/api/config", {"config": jsify(scenario.to_json(cfg)), "path": str(tmp_path / "b.toml")})
    back = scenario.load(tmp_path / "b.toml")
    text = (tmp_path / "a.toml").read_text()
    assert config.to_dict(back) == config.to_dict(cfg) and (tmp_path / "b.toml").read_text() == text == r["toml"]
    assert isinstance(back.mesh.opts["alpha"], float) and back.size.layers[1].enabled is False
    assert "[mesh.opts]" in text and "enabled = false" in text and text.count("enabled") == 1
    assert back.bathymetry.levels == [0.0, 10.0, 25.0, 4000.0] and isinstance(back.bathymetry.min_depth_m, float)


def test_mesh_payload_decodes_to_the_same_arrays(srv):
    u = srv["url"]
    mv = meshview.load(str(srv["mesh"]))
    m = payload.decode_mesh_bin(get(u + "/api/mesh/mesh.bin")[1])
    assert np.array_equal(m["lonlat"], mv.lonlat) and np.array_equal(m["tris"], mv.tris)
    assert np.array_equal(m["quads"], mv.quads) and np.array_equal(m["angle"], mv.fields["angle"])
    assert np.array_equal(m["size_ratio"], mv.fields["size_ratio"])
    w = payload.decode_mesh_bin(get(u + "/api/mesh/mesh.bin?sel=window&lon=180&lat=0&r_km=2000")[1])
    T = len(mv.tris)
    e = w["elem_id"].astype(np.int64)
    assert len(e) == len(mv.window(180.0, 0.0, 2000.0)) > 0
    nt = w["header"]["n_tris"]
    assert np.array_equal(w["lonlat"][w["tris"]], mv.lonlat[mv.tris[e[:nt]]])
    assert np.array_equal(w["lonlat"][w["quads"]], mv.lonlat[mv.quads[e[nt:] - T]])
    assert np.array_equal(w["angle"], mv.fields["angle"][e])
    b = get(u + "/api/mesh/irregular.bin")[1]
    ll, cls = payload.decode_points_bin(b)
    assert len(ll) == int((mv.node_class > 0).sum()) and set(cls.tolist()) <= {1, 2, 3}
    s, b = get(u + "/api/coast.bin")
    assert struct.unpack("<I", b[:4])[0] > 100


def test_run_writes_a_run_folder(srv):
    cfg = json.loads(get(srv["url"] + "/api/config")[1])["config"]
    r = post(srv["url"] + "/api/run", {"config": jsify(cfg), "resources": {"cpus": 4, "mem": "3G"}})
    d = Path(r["dir"])
    assert r["submitted"] is False and r["command"].endswith(f"-m meshom.cli run {d / 'config.toml'} --out {d}")
    assert r["command"].startswith(f"PYTHONPATH={scenario.REPO} ")
    c = config.load(d / "config.toml")
    assert c.mesh.out == str(d) and Path(c.coast.source).is_absolute() and Path(c.coast.source).exists()
    sb = (d / "run.sbatch").read_text()
    assert "--cpus-per-task=4" in sb and "--mem=3G" in sb and "--partition=shared" in sb and r["command"] in sb
    assert "--account=ab0995" in sb and f"--output={d}/slurm-%j.log" in sb
    shutil.rmtree(d)


def test_token_guards_the_api(tmp_path):
    """With a token every request but /static/* needs it (query string, the cookie set by the first page load, or
    the X-Meshom-Token header): without it 403, with it 200; a second handler in the process keeps its own state."""
    cfg = tmp_path / "c.toml"
    scenario.save(scenario.load(), cfg)
    tok = "0123456789abcdef0123456789abcdef"
    h = make_handler(cfg, write_tiny_mesh(tmp_path / "mesh"), run_roots=[tmp_path / "runs"], ref_roots=[], token=tok)
    h.state.cookie = "meshom_token_8765"                     # make_server names it after the port

    def status(path, headers=None, data=None):
        return inproc.call(h, "POST" if data else "GET", path, data or b"", headers)
    for path in ("/", "/api/config", "/api/runs", f"/api/config?token={'0' * 32}"):
        code, _, body = status(path)
        assert code == 403 and body.startswith(b"meshom ui: 403"), path
    assert status("/api/toml", data=b'{"text": "a = 1"}')[0] == 403
    assert status("/static/app.js")[0] == 200                            # static files stay public
    code, hdr, _ = status(f"/?token={tok}")
    cookie = hdr["Set-Cookie"].split(";")[0]
    assert code == 200 and cookie == f"meshom_token_8765={tok}"
    assert status("/api/config", {"Cookie": cookie})[0] == 200
    assert status(f"/api/config?token={tok}")[0] == 200
    assert status("/api/runs", {"X-Meshom-Token": tok})[0] == 200
    assert status("/api/toml", {"Cookie": cookie}, b'{"text": "a = 1"}')[0] == 200


def test_action_log_has_the_requests_and_the_page_events(tmp_path):
    """With a log file every API request is written when it starts and when it ends (status, ms; never the token),
    and the page's events posted to /api/log follow as they came; /api/log itself and static files are not logged;
    no log file: /api/log answers and writes nothing."""
    cfg = tmp_path / "c.toml"
    scenario.save(scenario.load(), cfg)
    tok, log = "f" * 32, tmp_path / "logs" / "ui.jsonl"
    h = make_handler(cfg, write_tiny_mesh(tmp_path / "mesh"), run_roots=[tmp_path / "runs"], ref_roots=[], token=tok,
                     log_path=log)
    assert inproc.call(h, "GET", f"/api/config?token={tok}")[0] == 200
    assert inproc.call(h, "GET", "/static/app.js")[0] == 200
    ev = {"page": "p1", "events": [{"t": 1, "ev": "click", "on": "Zoom window"}, {"t": 2, "ev": "beat", "renders": 0}]}
    assert inproc.call(h, "POST", "/api/log", json.dumps(ev).encode(), {"X-Meshom-Token": tok})[0] == 200
    lines = [json.loads(s) for s in log.read_text().splitlines()]
    assert tok not in log.read_text()
    req = [x for x in lines if x["ev"] in ("req", "done")]
    assert [(x["ev"], x["path"]) for x in req] == [("req", "/api/config"), ("done", "/api/config")]
    assert req[1]["status"] == 200 and req[1]["ms"] >= 0 and req[0]["id"] == req[1]["id"]
    ui = [x for x in lines if x.get("page") == "p1"]
    assert [(x["ev"], x["t"]) for x in ui] == [("click", 1), ("beat", 2)] and ui[0]["on"] == "Zoom window"
    h2 = make_handler(cfg, write_tiny_mesh(tmp_path / "mesh2"), run_roots=[tmp_path / "runs"], ref_roots=[])
    assert inproc.call(h2, "POST", "/api/log", json.dumps(ev).encode())[0] == 200


def test_host_beyond_loopback_needs_the_token(capsys):
    from ui.web.server import loopback, main as serve

    for host in ("0.0.0.0", "l40051", "10.1.2.3"):
        with pytest.raises(SystemExit) as e:
            serve(["--host", host, "--no-token"])
        assert e.value.code == 2 and "needs the token" in capsys.readouterr().err
    assert loopback("127.0.0.1") and loopback("localhost") and loopback("::1") and not loopback("0.0.0.0")


def test_save_keeps_the_example_config(tmp_path, monkeypatch):
    """The repository's example config is never overwritten: a page started on it saves to scenario.SAVE_DEFAULT
    and keeps saving there."""
    monkeypatch.setattr(scenario, "SAVE_DEFAULT", tmp_path / "config.toml", raising=False)
    before = scenario.EXAMPLE.read_bytes()
    h = make_handler(scenario.EXAMPLE, write_tiny_mesh(tmp_path / "mesh"), run_roots=[tmp_path / "runs"], ref_roots=[])
    cfg = json.loads(inproc.call(h, "GET", "/api/config")[2])["config"]
    try:
        r = json.loads(inproc.call(h, "POST", "/api/config", json.dumps({"config": jsify(cfg)}).encode())[2])
        after = scenario.EXAMPLE.read_bytes()
    finally:
        scenario.EXAMPLE.write_bytes(before)
    assert after == before and r["saved"] == str(tmp_path / "config.toml")
    assert json.loads(inproc.call(h, "GET", "/api/config")[2])["path"] == str(tmp_path / "config.toml")


def test_mesh_info_has_gates_in_words_extent_cell_size_and_its_coast(srv):
    """info of a `meshom run` folder: every applicable gate with words, limit and comparison (failed first), the mesh's
    extent and typical cell size, the coast of the config it was built with; coast.bin?of=mesh draws that coast."""
    post("/api/mesh/load", {"path": str(srv["core"])})
    info = json.loads(get("/api/mesh/info")[1])
    g = info["gates"]
    assert len(g) == json.loads((srv["core"] / "quality.json").read_text())["raw"]["gates_total"]
    assert all({"what", "where", "value", "limit", "op", "passed", "group"} <= set(x) for x in g)
    assert [x["passed"] for x in g] == sorted(x["passed"] for x in g)                  # failed rows first
    e = info["extent"]
    assert e["global"] and e["south"] < -79 and e["north"] > 79 and 1500 < info["cell_km"] < 2500   # 20° cells
    assert info["coast"] is None and info["config"].endswith("config.resolved.toml")    # the sphere config: no coast
    assert struct.unpack("<I", get("/api/coast.bin?of=mesh")[1][:4])[0] == 0
    assert struct.unpack("<I", get("/api/coast.bin")[1][:4])[0] > 100                    # the page's own coast
    post("/api/mesh/load", {"path": str(srv["mesh"])})


def test_preview_sends_the_readout_grid(srv):
    """The global preview carries h (km, NaN on land) and the winning layer on its grid, deflated, for the cursor."""
    import base64
    import zlib
    cfg = json.loads(get("/api/config")[1])["config"]
    r = post("/api/preview", {"config": cfg, "step": 4.0})
    g = r["grid"]
    h = np.frombuffer(zlib.decompress(base64.b64decode(g["h"])), np.float32).reshape(g["ny"], g["nx"])
    w = np.frombuffer(zlib.decompress(base64.b64decode(g["winner"])), np.uint8).reshape(g["ny"], g["nx"])
    assert (g["nx"], g["ny"], g["step"], g["lon0"], g["lat0"]) == (90, 45, 4.0, -180.0, -90.0)
    water = np.isfinite(h)
    assert water.any() and (~water).any() and np.array_equal(water, w != 255)
    assert abs(np.nanmin(h) - r["h_min"] / 1000) < 1e-3 and abs(np.nanmax(h) - r["h_max"] / 1000) < 1e-3


def test_open_a_runs_config(srv, tmp_path, monkeypatch):
    """A run's config becomes the page's config; later saves go to SAVE_DEFAULT, never into the run's folder."""
    monkeypatch.setattr(scenario, "SAVE_DEFAULT", tmp_path / "config.toml")
    before = (srv["core"] / "config.resolved.toml").read_bytes()
    r = post("/api/config/open", {"path": str(srv["core"])})
    c = json.loads(get("/api/config")[1])
    assert r["opened"].endswith("config.resolved.toml") and c["config"]["mesh"]["method"] == "m01_isotropic"
    post("/api/config", {"config": jsify(c["config"])})
    assert (tmp_path / "config.toml").exists() and (srv["core"] / "config.resolved.toml").read_bytes() == before
    post("/api/config/open", {"path": str(srv["cfg"])})                                 # back for the other tests


def test_opened_run_config_forgets_the_runs_out_and_says_where_it_came_from(srv, tmp_path, monkeypatch):
    """A config opened from a run drops [mesh] out (a later `meshom run FILE` must not write into that run's folder)
    and the page learns which run it came from; the gate table carries units and marks counts."""
    monkeypatch.setattr(scenario, "SAVE_DEFAULT", tmp_path / "config.toml")
    post("/api/config/open", {"path": str(srv["core"])})
    c = json.loads(get("/api/config")[1])
    assert c["opened_from"] == str(srv["core"]) and not c["config"]["mesh"].get("out")
    post("/api/config", {"config": jsify(c["config"])})
    assert config.load(tmp_path / "config.toml").mesh.out is None
    post("/api/config/open", {"path": str(srv["cfg"])})
    post("/api/mesh/load", {"path": str(srv["core"])})
    g = {x["gate"]: x for x in json.loads(get("/api/mesh/info")[1])["gates"]}
    assert g["quad_max_angle_hard_deg"]["unit"] == "°" and g["elems_per_node_max"]["count"] is True
    post("/api/mesh/load", {"path": str(srv["mesh"])})


def test_method_variants_show_their_own_defaults(srv):
    """A variant's form shows the module's DEFAULTS with the variant's overrides on top (meshom.methods.defaults), the
    overridden keys named; a composed variant (another method plus a stage) says its form is the base method's."""
    m = {x["name"]: x for x in json.loads(get("/api/methods")[1])["methods"]}
    t = m["m14_blocks_tri"]
    assert t["defaults"]["fill_method"] == "tri" and t["overrides"] == {"fill_method": "tri"} and not t["composed"]
    assert m["m14_blocks"]["overrides"] == {} and m["m14_blocks"]["defaults"]["fill_method"] != "tri"
    assert m["m01_triangles"]["composed"] and m["m14_blocks_rep"]["composed"] and not m["m02_linf_frontal"]["composed"]


def test_a_repaired_mesh_opens_with_its_before_and_after(srv):
    """A mesh repaired by `meshom repair` (repaired-NN.npz + .json in a run folder) is a mesh of the run; its gates come
    from the record's "after" and its info carries the before / after rows of the zone and of the whole mesh."""
    core = srv["core"]
    poly = [[-30.0, -20.0], [30.0, -20.0], [30.0, 20.0], [-30.0, 20.0]]
    rec = json.loads((core / "quality.json").read_text())["raw"]
    zone = {"n_elems": 12, "n_tris": 0, "n_quads": 12, "n3": 1, "n5": 1, "n_other": 0, "gates_passed": 3, "gates_total": 4,
            "quality": {"quad_max_angle_deg": {"max": 120.0}, "elem_size_ratio": {"p99": 1.1}}}
    shutil.copy(core / "raw.npz", core / "repaired-01.npz")
    (core / "repaired-01.json").write_text(json.dumps({"input": "raw.npz", "polygon": poly, "opts": {}, "before": rec,
        "after": rec, "repair": {"accepted": True, "forced": False, "kind": "ops"},
        "region": {"before": zone, "after": dict(zone, n3=0, n5=0)}}))
    runs = {x["name"]: x for x in json.loads(get("/api/runs")[1])["runs"]}
    assert "repaired-01.npz" in runs["sphere"]["meshes"]
    post("/api/mesh/load", {"path": str(core / "repaired-01.npz")})
    info = json.loads(get("/api/mesh/info")[1])
    rp = info["repair"]
    assert rp["accepted"] and rp["kind"] == "ops" and rp["input"] == "raw.npz" and rp["polygon"] == poly
    rows = {x["what"]: x for x in rp["rows"]}
    assert rows["irregular nodes n3"]["zone"] == [1, 0] and rows["gates passed"]["whole"][0] == rec["gates_passed"]
    assert len(info["gates"]) == rec["gates_total"]
    for f in ("repaired-01.npz", "repaired-01.json"):
        (core / f).unlink()
    post("/api/mesh/load", {"path": str(srv["mesh"])})


def test_compare_reads_the_quality_records_side_by_side(srv):
    """/api/compare: one column per mesh from its quality record (a run's raw, a reference mesh, a run without one),
    without opening a mesh; the best value of a row is marked; gates only on the gates every mesh has."""
    core, ref = srv["core"], srv["ref"]
    rec = json.loads((core / "quality.json").read_text())["raw"]
    before = HANDLER[0].state.mesh_path
    r = post("/api/compare", {"paths": [str(core / "raw.npz"), str(ref), str(srv["run"] / "raw.npz")]})
    assert HANDLER[0].state.mesh_path == before                          # nothing is opened
    cols = r["meshes"]
    assert [c["name"] for c in cols] == ["sphere / raw", "grid_x", "m14_blocks_uniform_s4 / raw"]
    assert [c["has_record"] for c in cols] == [True, True, False]
    rows = {x["what"]: x for x in r["rows"]}
    q = rec["quality"]
    assert rows["nodes"]["values"][:2] == [rec["n_nodes"]] * 2 and rows["nodes"]["values"][2] is None
    assert rows["quads"]["values"][0] == q["n_quads"] and rows["triangles"]["values"][0] == q["n_tris"]
    assert rows["triangle share"]["values"][0] == pytest.approx(q["n_tris"] / q["n_elems"])
    assert rows["n3 per 1000 nodes"]["values"][0] == pytest.approx(1000 * rec["irregular"]["n3"] / rec["n_nodes"])
    assert rows["largest quad angle, max"]["values"][0] == q["quad_max_angle_deg"]["max"]
    assert rows["largest quad angle, p99.9"]["values"][0] == q["quad_max_angle_deg"]["p99_9"]
    assert rows["largest triangle angle, p99"]["values"][0] == q["tri_max_angle_deg"]["p99"]
    assert rows["size jump across an edge, p99"]["values"][0] == q["elem_size_ratio"]["p99"]
    assert rows["|ln(edge / h)|, p99"]["values"][0] == q["size_fidelity"]["p99"]
    assert rows["nodes"]["best"] == [] and rows["largest quad angle, max"]["best"] == [0, 1]   # the same mesh: a tie
    g = r["gates"]
    assert g["names"] and len(g["names"]) <= rec["gates_total"]
    assert g["columns"][0]["total"] == len(g["names"]) and g["columns"][2] is None
    assert g["columns"][0]["passed"] + len(g["columns"][0]["failed"]) == g["columns"][0]["total"]


def test_depth_overlay_from_a_runs_depth_npy(srv):
    """A run folder with depth.npy (the core's bathymetry stage: depth per node, m positive down) gives the colour
    "depth": per element the mean depth of its nodes, in mesh.bin (all and subsets) and the overview raster; info says
    whether the mesh has it and the legend's range. A mesh without depth.npy has neither."""
    core = srv["core"]
    with np.load(core / "raw.npz") as z:
        tris, quads, n = z["tris"], z["quads"], len(z["points"])
    depth = np.linspace(10.0, 5000.0, n)
    np.save(core / "depth.npy", depth)
    try:
        post("/api/mesh/load", {"path": str(core)})
        info = json.loads(get("/api/mesh/info")[1])
        assert info["has_depth"] and info["depth"]["lo"] == 0 and info["depth"]["hi"] >= 4500
        want = np.r_[depth[tris].mean(axis=1), depth[quads].mean(axis=1)].astype(np.float32)
        m = payload.decode_mesh_bin(get("/api/mesh/mesh.bin")[1])
        assert "depth" in m["header"]["fields"] and np.allclose(m["depth"], want, rtol=1e-6)
        w = payload.decode_mesh_bin(get("/api/mesh/mesh.bin?sel=window&lon=180&lat=0&r_km=2000")[1])
        assert np.allclose(w["depth"], want[w["elem_id"].astype(np.int64)], rtol=1e-6)
        assert get("/api/mesh/overview.png?field=depth")[0] == 200
        assert get("/api/mesh/overview.png?field=depth&proj=south")[0] == 200
    finally:
        (core / "depth.npy").unlink()
    post("/api/mesh/load", {"path": str(core)})
    info = json.loads(get("/api/mesh/info")[1])
    assert not info["has_depth"] and info["depth"] is None
    assert "depth" not in payload.decode_mesh_bin(get("/api/mesh/mesh.bin")[1])["header"]["fields"]
    post("/api/mesh/load", {"path": str(srv["mesh"])})


def test_depth_of_a_written_mesh(srv):
    """A mesh written as model files (`meshom write` / the page's Write: <run>/fesom-<mesh stem>/ with its quality.json)
    shows the depth of that write: depth.out for fesom-c (its aux3d.out is a placeholder), aux3d.out for fesom2 /
    fesom3; info names the source; a newer write replaces the depth shown. The run's depth.npy comes second (it is
    the run's final mesh's, written before any later edit)."""
    from meshom.bathymetry import levels
    from meshom.io.fesom import write_fesom3, write_fesomc
    from meshom.mesh import Mesh

    core = srv["core"]
    with np.load(core / "raw.npz") as z:
        mesh = Mesh(z["points"], z["tris"], z["quads"], np.zeros(len(z["points"]), np.int8))
    o = core / "fesom-raw"
    o.mkdir()
    d1 = np.linspace(20.0, 3000.0, mesh.n_nodes)
    elem = lambda d: np.r_[d[mesh.tris].mean(axis=1), d[mesh.quads].mean(axis=1)].astype(np.float32)  # noqa: E731
    try:
        write_fesomc(mesh, o, d1)
        (o / "quality.json").write_text(json.dumps({"raw": {"format": "fesom-c", "date": "2026-10-27T10:00:00"}}))
        post("/api/mesh/load", {"path": str(core / "raw.npz")})
        info = json.loads(get("/api/mesh/info")[1])
        assert info["has_depth"] and "fesom-raw" in info["depth_source"] and "fesom-c" in info["depth_source"]
        m = payload.decode_mesh_bin(get("/api/mesh/mesh.bin")[1])
        assert np.allclose(m["depth"], elem(d1), rtol=1e-6)
        d2 = d1[::-1].copy()
        write_fesom3(mesh, o, d2, levels("core2"))
        (o / "quality.json").write_text(json.dumps({"raw": {"format": "fesom3", "date": "2026-10-27T11:00:00"}}))
        post("/api/mesh/load", {"path": str(core / "raw.npz")})
        info = json.loads(get("/api/mesh/info")[1])
        assert "fesom3" in info["depth_source"]
        m = payload.decode_mesh_bin(get("/api/mesh/mesh.bin")[1])
        assert np.allclose(m["depth"], elem(d2), rtol=1e-5)
    finally:
        shutil.rmtree(o)
        post("/api/mesh/load", {"path": str(srv["mesh"])})


def test_depth_shown_is_the_writes_base_with_the_current_node_edits(srv):
    """The depth shown is the last write's depth_base.npy (the depth before the node edits, C3) with the mesh's current
    node edits (<run>/<mesh stem>.depth_edits.json, C2) applied as the core applies them at the next write
    (bathymetry.apply_node_edits at the run's floor in the write's format: fesom3, CORE2 levels: 30 m); a change of
    the edits file changes the depth shown at once; without depth_base.npy the written aux3d.out stands."""
    from meshom.bathymetry import apply_node_edits
    from meshom.io.fesom import write_fesom3
    from meshom.mesh import Mesh
    from meshom.bathymetry import levels

    core = srv["core"]
    with np.load(core / "raw.npz") as z:
        mesh = Mesh(z["points"], z["tris"], z["quads"], np.zeros(len(z["points"]), np.int8))
    o, ef = core / "fesom-raw", core / "raw.depth_edits.json"
    o.mkdir()
    base = np.linspace(100.0, 3000.0, mesh.n_nodes)
    elem = lambda d: np.r_[d[mesh.tris].mean(axis=1), d[mesh.quads].mean(axis=1)].astype(np.float32)  # noqa: E731
    try:
        write_fesom3(mesh, o, base + 1.0, levels("core2"))           # the written files: an older depth
        np.save(o / "depth_base.npy", base)
        (o / "quality.json").write_text(json.dumps({"raw": {"format": "fesom3", "date": "2026-10-27T12:00:00"}}))
        ef.write_text(json.dumps({"mesh": "raw.npz", "n_nodes": mesh.n_nodes, "edits": [
            {"op": "set", "depth_m": 500.0, "nodes": [0, 1, 2]}, {"op": "min", "depth_m": 4000.0, "nodes": [5]},
            {"op": "set", "depth_m": 5.0, "nodes": [7]}]}))
        want, rec = apply_node_edits(base, ef, 30.0)
        assert want[7] == 30.0 and rec["n_clamped_to_floor"] == 1
        post("/api/mesh/load", {"path": str(core / "raw.npz")})
        info = json.loads(get("/api/mesh/info")[1])
        assert "depth_base.npy" in info["depth_source"] and "3 node edits" in info["depth_source"]
        m = payload.decode_mesh_bin(get("/api/mesh/mesh.bin")[1])
        assert np.allclose(m["depth"], elem(want), rtol=1e-6)
        ef.unlink()
        m = payload.decode_mesh_bin(get("/api/mesh/mesh.bin")[1])
        assert np.allclose(m["depth"], elem(base), rtol=1e-6)
        (o / "depth_base.npy").unlink()
        m = payload.decode_mesh_bin(get("/api/mesh/mesh.bin")[1])
        assert np.allclose(m["depth"], elem(base + 1.0), rtol=1e-5)
    finally:
        shutil.rmtree(o)
        ef.unlink(missing_ok=True)
        post("/api/mesh/load", {"path": str(srv["mesh"])})


def test_node_edits_from_the_page(srv):
    """The node tool's edits (Tools, "Mesh depth"): "set to" / "at least" / "at most" D on the nodes of a box or a
    polygon of the mesh shown go to <run>/<mesh stem>.depth_edits.json in the core's format (C2: mesh, n_nodes,
    edits [{op, depth_m, nodes, outline, date}]), listed, undone (the last) or removed (any); the depth shown follows
    at once; a mesh outside a `meshom run` folder has nowhere its edits would be applied."""
    core = srv["core"]
    mf, ef = core / "raw.npz", core / "raw.depth_edits.json"
    n = len(np.load(mf)["points"])
    np.save(core / "depth.npy", np.full(n, 1000.0))
    box = [0.0, 40.0, -10.0, 30.0]
    ids = sorted(j * 18 + i for j in (4, 5) for i in (9, 10, 11))
    try:
        post("/api/mesh/load", {"path": str(mf)})
        r = post("/api/mesh/depth/edit", {"path": str(mf), "op": "set", "depth_m": 500.0, "box": box})
        e = json.loads(ef.read_text())
        assert e["mesh"] == "raw.npz" and e["n_nodes"] == n and len(e["edits"]) == 1
        ed = e["edits"][0]
        assert ed["op"] == "set" and ed["depth_m"] == 500.0 and sorted(ed["nodes"]) == ids and ed["date"]
        assert len(ed["outline"]) == 4 and r["edits"][0]["n"] == 6
        assert r["area"]["depth"] == pytest.approx({"min": 500.0, "median": 500.0, "mean": 500.0, "max": 500.0})
        post("/api/mesh/depth/edit", {"path": str(mf), "op": "min", "depth_m": 4000.0,
                                      "polygon": [[-5.0, -5.0], [45.0, -5.0], [45.0, 25.0], [-5.0, 25.0]]})
        r = json.loads(get(f"/api/mesh/depth/edits?path={mf}")[1])
        assert [(x["op"], x["depth_m"], x["n"]) for x in r["edits"]] == [("set", 500.0, 6), ("min", 4000.0, 6)]
        t = json.loads(get(f"/api/mesh/write?path={mf}")[1])["meshes"][0]
        assert t["name"] == "raw.npz" and t["edits"] == 2 and t["edits_newer"] is True        # never written
        m = payload.decode_mesh_bin(get("/api/mesh/mesh.bin")[1])
        assert m["depth"].max() == pytest.approx(4000.0)
        r = post("/api/mesh/depth/undo", {"path": str(mf)})
        assert [x["op"] for x in r["edits"]] == ["set"]
        r = post("/api/mesh/depth/remove", {"path": str(mf), "index": 0})
        assert r["edits"] == [] and not ef.exists()
        m = payload.decode_mesh_bin(get("/api/mesh/mesh.bin")[1])
        assert np.allclose(m["depth"], 1000.0)
        post("/api/mesh/load", {"path": str(srv["mesh"])})
        s_, _, b = inproc.call(HANDLER[0], "POST", "/api/mesh/depth/edit", json.dumps(
            {"path": str(srv["mesh"] / "mesh.npz"), "op": "set", "depth_m": 50.0, "box": box}).encode())
        assert s_ == 400 and "meshom run" in json.loads(b)["error"]
    finally:
        (core / "depth.npy").unlink()
        ef.unlink(missing_ok=True)
        post("/api/mesh/load", {"path": str(srv["mesh"])})


def test_cell_size_colour_in_km(srv):
    """The colour "cell size": per element the mean great-circle length of its edges, km (an ORCA grid's cells shrink
    toward the poles while its angles and neighbour ratios stay near ideal, so only this field shows its resolution),
    on a log scale between the 1st and 99th percentiles rounded out to two significant digits (ORCA1: 27-110 km); in
    mesh.bin (all and subsets), the overview raster and info."""
    core = srv["core"]
    with np.load(core / "raw.npz") as z:
        p, tris, quads = z["points"].astype(np.float64), z["tris"], z["quads"]

    def mean_edge_km(el):
        a, b = p[el], p[np.roll(el, -1, axis=1)]
        return (2 * np.arcsin(np.clip(np.linalg.norm(a - b, axis=2) / 2, 0, 1))).mean(axis=1) * 6371.0
    want = np.r_[mean_edge_km(tris), mean_edge_km(quads)]
    try:
        post("/api/mesh/load", {"path": str(core)})
        s = json.loads(get("/api/mesh/info")[1])["size_km"]
        assert 0 < s["lo"] <= np.percentile(want, 1) and s["hi"] >= np.percentile(want, 99) and s["hi"] > s["lo"]
        for v in (s["lo"], s["hi"]):
            k = v / 10 ** (np.floor(np.log10(v)) - 1)
            assert np.isclose(k, round(k))
        assert s["lo"] > 0.9 * np.percentile(want, 1) and s["hi"] < 1.1 * np.percentile(want, 99)
        m = payload.decode_mesh_bin(get("/api/mesh/mesh.bin")[1])
        assert "size_km" in m["header"]["fields"] and np.allclose(m["size_km"], want, rtol=1e-5)
        assert m["size_km"].max() > 1.5 * m["size_km"].min()            # the lon-lat sphere: smaller near the poles
        w = payload.decode_mesh_bin(get("/api/mesh/mesh.bin?sel=window&lon=180&lat=0&r_km=2000")[1])
        assert np.allclose(w["size_km"], want[w["elem_id"].astype(np.int64)], rtol=1e-5)
        assert get("/api/mesh/overview.png?field=size_km")[0] == 200
        assert get("/api/mesh/overview.png?field=size_km&proj=north")[0] == 200
    finally:
        post("/api/mesh/load", {"path": str(srv["mesh"])})


# ------------------------------------------------------------------------------------------- edit a mesh by hand

def _edit_run(srv, name: str) -> Path:
    """A copy of the scored run folder `sphere` (raw.npz, quality.json, config.resolved.toml) whose raw.npz marks every
    7th node as coast (node_flag 1), loaded as the current mesh."""
    d = srv["core"].parent / name
    shutil.copytree(srv["core"], d)
    with np.load(d / "raw.npz") as z:
        a = {k: z[k] for k in z.files}
    a["node_flag"] = a["node_flag"].copy()
    a["node_flag"][::7] = 1
    np.savez(d / "raw.npz", **a)
    post("/api/mesh/load", {"path": str(d / "raw.npz")})
    return d


def _ccw(p: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """det(p_a, p_b, p_c) of each cell's first three corners: > 0 when CCW seen from outside."""
    a, b, c = p[cells[:, 0]], p[cells[:, 1]], p[cells[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c))


def _ll(xyz) -> list[float]:
    from meshom.geo import xyz_to_lonlat
    return [float(x) for x in xyz_to_lonlat(np.asarray(xyz) / np.linalg.norm(xyz))]


def test_window_payload_for_editing_carries_global_node_ids_and_flags(srv):
    d = _edit_run(srv, "edit_payload")
    try:
        mv = meshview.load(str(d / "raw.npz"))
        w = payload.decode_mesh_bin(get("/api/mesh/mesh.bin?sel=window&lon=10&lat=0&r_km=3000&edit=1")[1])
        assert w["header"]["node_fields"] == ["node_id", "node_flag"]
        nid = w["node_id"].astype(np.int64)
        assert len(nid) == w["header"]["n_nodes"] and np.array_equal(w["lonlat"], mv.lonlat[nid])
        with np.load(d / "raw.npz") as z:
            assert np.array_equal(w["node_flag"], z["node_flag"][nid]) and w["node_flag"].any()
        plain = payload.decode_mesh_bin(get("/api/mesh/mesh.bin?sel=window&lon=10&lat=0&r_km=3000")[1])
        assert "node_fields" not in plain["header"] and "node_id" not in plain
    finally:
        post("/api/mesh/load", {"path": str(srv["mesh"])})
        shutil.rmtree(d)


def test_save_edits_moves_nodes_and_splits_elements(srv):
    """The page's edits replayed on the full mesh: moved nodes (old and added), a triangle split into 3, a quad into
    4, a new triangle split again; written as edited-01.npz in raw.npz's layout, raw.npz untouched."""
    from meshom.geo import lonlat_to_xyz
    d = _edit_run(srv, "edit_save")
    try:
        raw = (d / "raw.npz").read_bytes()
        with np.load(d / "raw.npz") as z:
            P, T3, Q4, F = z["points"], z["tris"], z["quads"], z["node_flag"]
        N, T, Q = len(P), len(T3), len(Q4)
        E = T + Q
        k = int(Q4[30][0])
        lon_k, lat_k = _ll(P[k])
        a, b, c, dd = (int(x) for x in Q4[30])
        c0, c1 = _ll(P[T3[0]].mean(axis=0)), _ll(P[Q4[30]].mean(axis=0))
        p1 = lonlat_to_xyz(c1[0] + 0.5, c1[1])                                    # node N+1 after its move
        c2 = _ll(P[a] + P[b] + p1)                                                  # inside triangle (a, b, N+1)
        ops = [{"op": "move", "node": k, "lon": lon_k + 1.5, "lat": lat_k + 1.0},
               {"op": "split", "elem": 0, "lon": c0[0], "lat": c0[1]},            # node N, triangles E, E+1, E+2
               {"op": "split", "elem": T + 30, "lon": c1[0], "lat": c1[1]},       # node N+1, triangles E+3 .. E+6
               {"op": "move", "node": N + 1, "lon": c1[0] + 0.5, "lat": c1[1]},
               {"op": "split", "elem": E + 3, "lon": c2[0], "lat": c2[1]}]        # node N+2, triangles E+7 .. E+9
        r = post("/api/mesh/edit", {"ops": ops})
        f = Path(r["saved"])
        assert f == d / "edited-01.npz" and r["n_nodes"] == N + 3
        with np.load(f) as z:
            assert sorted(z.files) == ["node_flag", "points", "quads", "tris"]
            p, t, q, fl = z["points"], z["tris"], z["quads"], z["node_flag"]
        assert p.dtype == P.dtype and t.dtype == T3.dtype and q.dtype == Q4.dtype and fl.dtype == F.dtype
        assert len(p) == N + 3 and len(t) == T - 1 + 3 + 4 - 1 + 3 and len(q) == Q - 1
        assert np.allclose(p[k], lonlat_to_xyz(lon_k + 1.5, lat_k + 1.0))
        assert np.allclose(p[N], lonlat_to_xyz(*c0)) and np.allclose(p[N + 1], p1)
        keep = np.ones(N, bool)
        keep[k] = False
        assert np.array_equal(p[:N][keep], P[keep]) and np.array_equal(fl, np.r_[F, np.zeros(3, F.dtype)])
        assert np.array_equal(t[:T - 1], T3[1:]) and np.array_equal(q, np.delete(Q4, 30, axis=0))
        new = t[T - 1:]
        assert [int(c[2]) for c in new] == [N] * 3 + [N + 1] * 3 + [N + 2] * 3       # each around its new node
        assert [tuple(x) for x in new[3:6]] == [(b, c, N + 1), (c, dd, N + 1), (dd, a, N + 1)]   # (a, b, N+1) split
        assert [tuple(x) for x in new[6:]] == [(a, b, N + 2), (b, N + 1, N + 2), (N + 1, a, N + 2)]
        assert (_ccw(p, t) > 0).all() and (_ccw(p, q) > 0).all()
        assert (d / "raw.npz").read_bytes() == raw
        log = json.loads((d / "edited-01.json").read_text())
        assert log["input"] == "raw.npz" and log["ops"] == ops and log["moves"] == 2 and log["splits"] == 3
    finally:
        post("/api/mesh/load", {"path": str(srv["mesh"])})
        shutil.rmtree(d)


def test_saved_edits_are_numbered_listed_and_open_like_raw(srv):
    d = _edit_run(srv, "edit_names")
    try:
        with np.load(d / "raw.npz") as z:
            T = len(z["tris"])
            lon, lat = _ll(z["points"][z["quads"][40]].mean(axis=0))
        split = [{"op": "split", "elem": T + 40, "lon": lon, "lat": lat}]
        (d / "smooth.npz").write_bytes((d / "raw.npz").read_bytes())
        before = {n: (d / n).read_bytes() for n in ("raw.npz", "smooth.npz")}
        assert Path(post("/api/mesh/edit", {"ops": []})["saved"]).name == "edited-01.npz"
        one = (d / "edited-01.npz").read_bytes()
        assert Path(post("/api/mesh/edit", {"ops": split})["saved"]).name == "edited-02.npz"
        assert (d / "edited-01.npz").read_bytes() == one
        assert {n: (d / n).read_bytes() for n in ("raw.npz", "smooth.npz")} == before
        runs = {x["name"]: x for x in json.loads(get("/api/runs")[1])["runs"]}
        assert runs["edit_names"]["meshes"] == ["raw.npz", "smooth.npz", "edited-01.npz", "edited-02.npz"]
        post("/api/mesh/load", {"path": str(d / "edited-02.npz")})
        info = json.loads(get("/api/mesh/info")[1])
        assert info["path"] == str(d / "edited-02.npz") and info["n_tris"] == T + 4
        assert info["quality"] is None and info["gates"] == []           # quality.json scores raw / smooth, not this
        assert Path(post("/api/mesh/edit", {"ops": split})["saved"]).name == "edited-03.npz"   # an edit of an edit
        assert json.loads((d / "edited-03.json").read_text())["input"] == "edited-02.npz"
    finally:
        post("/api/mesh/load", {"path": str(srv["mesh"])})
        shutil.rmtree(d)


def test_reference_meshes_are_not_edited(srv):
    post("/api/mesh/load", {"path": str(srv["ref"])})
    try:
        s, _, b = inproc.call(HANDLER[0], "POST", "/api/mesh/edit", json.dumps({"ops": []}).encode())
        assert s == 400 and "reference" in json.loads(b)["error"]
        assert not list(srv["ref"].glob("edited-*"))
    finally:
        post("/api/mesh/load", {"path": str(srv["mesh"])})


# ------------------------------------------------------------------------------- write a saved mesh as model files

def _wait_write(timeout: float = 120.0) -> dict:
    """The write job (GET /api/mesh/write/job) once it no longer runs."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        j = json.loads(get("/api/mesh/write/job")[1])
        if j["state"] != "running":
            return j
        time.sleep(0.05)
    raise AssertionError("the write did not end")


def test_a_saved_edit_is_written_as_model_files_with_its_checks_and_gates(srv):
    """"Write model files" (Mesh panel, the hand editor): the run's meshes listed with the run's format and the folder
    each is written to; the write (meshom.run.write_mesh) runs off the request thread and ends with the folder, the
    format's checks (the STOP items in words), the gates with value and limit; opened again, the record is read back
    from the folder's quality.json; the run's own files are untouched."""
    d = _edit_run(srv, "write_edit")
    try:
        post("/api/mesh/edit", {"ops": [{"op": "move", "node": 5, "lon": -79.0, "lat": -79.5}]})    # edited-01
        e = d / "edited-01.npz"
        t = json.loads(get(f"/api/mesh/write?path={e}")[1])
        assert t["run_dir"] == str(d) and t["mesh"] == "edited-01.npz" and t["format"] == "fesom-c"
        assert [m["name"] for m in t["meshes"]] == ["raw.npz", "edited-01.npz"]
        assert t["meshes"][1]["out"] == str(d / "fesom-edited-01") and t["meshes"][1]["written"] is None
        assert t["formats"] == ["fesom-c", "fesom2", "fesom3"] and t["fesom3_ini"] == "" and t["job"]["state"] == "idle"
        raw = (d / "raw.npz").read_bytes()
        r = post("/api/mesh/write", {"path": str(e), "format": "fesom3"})
        assert r["state"] == "running" and r["mesh"] == str(e) and r["format"] == "fesom3"
        j = _wait_write()
        assert j["state"] == "done", j.get("error")
        s, o = j["summary"], d / "fesom-edited-01"
        assert s["out"] == str(o) and s["format"] == "fesom3" and s["mesh"] == "edited-01.npz"
        assert (o / "elem2d.out").exists() and (o / "quality.json").exists() and s["n_nodes"] == 9 * 18 + 2
        c = s["checks"]
        assert c["name"] == "fesom3_checks" and c["ok"] is False
        stop = {x["key"]: x for x in c["stop"]}        # the lon-lat test mesh has an element at each rotated pole
        assert stop["elems_containing_rotated_north_pole"]["value"] >= 1 and stop["elems_containing_rotated_north_pole"]["what"]
        assert {x["key"] for x in c["items"]} >= {"elems_not_clockwise", "quads_nonconvex_sphere"} and "stop" not in {x["key"] for x in c["items"]}
        g = s["gates"]
        assert s["gates_total"] == len(g) and s["gates_passed"] == sum(x["passed"] for x in g)
        assert all({"what", "value", "limit", "op"} <= set(x) for x in g) and not g[0]["passed"]       # failed first
        assert s["straits"] is None and s["fesom_ini"] is None and s["wall_s"] > 0
        again = json.loads(get(f"/api/mesh/write?path={e}")[1])
        assert again["meshes"][1]["written"] == s and again["meshes"][0]["written"] is None
        assert again["job"]["state"] == "done"
        assert (d / "raw.npz").read_bytes() == raw and not (d / "fesom").exists()
    finally:
        post("/api/mesh/load", {"path": str(srv["mesh"])})
        shutil.rmtree(d)


def test_a_hand_edit_keeps_the_meshs_depth_edits(srv):
    """A hand edit saved as edited-NN keeps the node ids of its input (moved nodes keep theirs, added nodes come
    last), so the input's depth edits (<stem>.depth_edits.json) are copied to edited-NN.depth_edits.json with the new
    file's name and node count, as the core checks them: depth edit, move a node, write keeps the depth edit."""
    d = _edit_run(srv, "edit_depth")
    try:
        n = len(np.load(d / "raw.npz")["points"])
        (d / "raw.depth_edits.json").write_text(json.dumps({"mesh": "raw.npz", "n_nodes": n, "edits": [
            {"op": "set", "depth_m": 50.0, "nodes": [1, 2], "outline": [], "date": "2026-10-27T12:00:00"}]}))
        post("/api/mesh/edit", {"ops": [{"op": "move", "node": 5, "lon": -79.0, "lat": -79.5}]})
        e = json.loads((d / "edited-01.depth_edits.json").read_text())
        assert e["mesh"] == "edited-01.npz" and e["n_nodes"] == len(np.load(d / "edited-01.npz")["points"]) == n
        assert e["edits"] == json.loads((d / "raw.depth_edits.json").read_text())["edits"]
    finally:
        post("/api/mesh/load", {"path": str(srv["mesh"])})
        shutil.rmtree(d)


def test_one_write_at_a_time_and_only_from_a_run_folder(srv):
    """A second write while one runs is refused (the server has a few GB: the write recomputes the depth and scores
    the mesh); an error of the write is the job's; a folder without config.resolved.toml has nothing to write with."""
    st = HANDLER[0].state
    go = threading.Event()

    def slow(run_dir, mesh_name, fmt, progress=None, fesom_ini=False):
        go.wait(30)
        raise RuntimeError("stopped by the test")
    st.writer.fn = slow
    try:
        assert post("/api/mesh/write", {"path": str(srv["core"] / "raw.npz"), "format": "fesom-c"})["state"] == "running"
        s, _, b = inproc.call(HANDLER[0], "POST", "/api/mesh/write",
                              json.dumps({"path": str(srv["core"] / "raw.npz"), "format": "fesom-c"}).encode())
        assert s == 409 and "already" in json.loads(b)["error"]
        go.set()
        j = _wait_write()
        assert j["state"] == "error" and "stopped by the test" in j["error"]
    finally:
        go.set()
        st.writer.fn = None
    s, _, b = inproc.call(HANDLER[0], "GET", f"/api/mesh/write?path={srv['run'] / 'raw.npz'}")
    assert s == 400 and "config.resolved.toml" in json.loads(b)["error"]
    assert not list(srv["core"].glob("fesom*"))


def test_fesom_ini_runs_after_the_write_when_ticked(srv, tmp_path):
    """With [mesh] fesom3_ini and the format fesom3 the window offers fesom_ini; ticked, the write runs it on the
    written folder (write_mesh(fesom_ini=True), its log in <run>/fesom_ini-<mesh>/) and its record is shown and kept in
    the folder's quality.json with the write's."""
    d = srv["core"].parent / "write_ini"
    shutil.copytree(srv["core"], d)
    ini = tmp_path / "fake_fesom_ini"                 # writes the derived files fesom_ini makes, as fesom_ini would
    ini.write_text("#!/bin/sh\nd=$(sed -n 's/^MeshPath=\"\\(.*\\)\\/\"$/\\1/p' namelist.config)\n"
                   "for f in edgenum.out edges.out edge_tri.out nlvls.out; do : > \"$d/$f\"; done\necho ' Total time 0.1'\n")
    ini.chmod(0o755)
    c = config.load(d / "config.resolved.toml")
    config.save(replace(c, mesh=replace(c.mesh, format="fesom3", fesom3_ini=str(ini), out=str(d))), d / "config.resolved.toml")
    try:
        t = json.loads(get(f"/api/mesh/write?path={d / 'raw.npz'}")[1])
        assert t["format"] == "fesom3" and t["fesom3_ini"] == str(ini)
        post("/api/mesh/write", {"path": str(d / "raw.npz"), "format": "fesom3", "fesom_ini": True})
        j = _wait_write()
        assert j["state"] == "done", j.get("error")
        f = j["summary"]["fesom_ini"]
        assert f["ok"] is True and f["derived_files"] is True and f["log"] == str(d / "fesom_ini-raw" / "fesom_ini.out")
        assert (d / "fesom-raw" / "edges.out").exists()
        rec = json.loads((d / "fesom-raw" / "quality.json").read_text())["raw"]
        assert rec["fesom_ini"]["log"] == f["log"] and "fesom_ini" in [x["stage"] for x in rec["stages"]]
        assert not (d / "fesom_ini-raw" / "fesom_ini.json").exists()          # the core's record, not the page's own
        again = json.loads(get(f"/api/mesh/write?path={d / 'raw.npz'}")[1])
        assert again["meshes"][0]["written"]["fesom_ini"] == f
    finally:
        shutil.rmtree(d)


def test_the_running_write_tells_its_stage(srv):
    """While a write runs the job tells the stage write_mesh is in (its progress callback: size, score, bathymetry,
    write, straits, passages, fesom_ini), for the window's busy line; the tick reaches write_mesh as fesom_ini only
    with the format fesom3 and a [mesh] fesom3_ini binary."""
    st = HANDLER[0].state
    steps = [threading.Event() for _ in range(2)]
    seen = {}

    def staged(run_dir, mesh_name, fmt, progress=None, fesom_ini=False):
        seen["fesom_ini"] = fesom_ini
        progress("size")
        steps[0].wait(30)
        progress("bathymetry")
        steps[1].wait(30)
        raise RuntimeError("stopped by the test")
    st.writer.fn = staged

    def stage_is(name):
        end = time.monotonic() + 10
        while time.monotonic() < end:
            j = json.loads(get("/api/mesh/write/job")[1])
            if j.get("stage") == name:
                return j
            time.sleep(0.02)
        raise AssertionError(f"no stage {name}: {j}")
    try:
        post("/api/mesh/write", {"path": str(srv["core"] / "raw.npz"), "format": "fesom-c", "fesom_ini": True})
        assert stage_is("size")["state"] == "running"
        steps[0].set()
        assert stage_is("bathymetry")["state"] == "running"
        steps[1].set()
        j = _wait_write()
        assert j["state"] == "error" and seen["fesom_ini"] is False          # fesom-c: no fesom_ini
    finally:
        for e in steps:
            e.set()
        st.writer.fn = None


def test_write_summary_has_the_straits_and_passages():
    """The straits of a write record in the page's words: per strait connected, cells across against the target, the
    narrowest width and depth, where; the water connectivity; the narrow passages when the record has them; a format
    without checks (fesom-c) has none."""
    from ui.web import write

    rec = dict(status="ok", format="fesom-c", mesh="/x/run/edited-01.npz", out="/x/run/fesom-edited-01",
               stages=[{"stage": "size", "wall_s": 1.5}, {"stage": "write", "wall_s": 0.5}],
               **compete._score(tiny_mesh(), closed_sphere(), uniform(2000e3), {}, config.DEFAULT_GATES))
    rec["straits"] = [{"name": "gibraltar", "ok": False, "connected": True, "cells_min": 1.5, "cells_target": 3,
                       "closed_sections": 0, "width_min_m": 14000.0, "depth_min_m": 280.0, "at": [-5.6, 35.9],
                       "profile": "widen", "n_sections": 21}]
    rec["water"] = {"connectivity_ok": True, "n_components": 1, "components": [{"id": 0}], "not_connected": []}
    rec["passages"] = {"cells": 2, "n_narrow": 1, "narrow": [{"kind": "island", "cells_across": 1, "at": [12.0, 55.0]}]}
    s = write.summary(rec, config.DEFAULT_GATES)
    assert s["checks"] is None and s["wall_s"] == 2.0 and s["mesh"] == "edited-01.npz"
    st = s["straits"]
    assert st["connectivity_ok"] is True and st["n_components"] == 1 and st["not_connected"] == []
    assert st["rows"] == [{"name": "gibraltar", "ok": False, "connected": True, "cells_min": 1.5, "cells_target": 3,
                           "closed_sections": 0, "width_min_m": 14000.0, "depth_min_m": 280.0, "at": [-5.6, 35.9]}]
    assert s["passages"] == rec["passages"]
    assert s["regions"] is None and s["depth_edits"] is None
    reg = [{"name": "Gibraltar", "n_nodes": 12, "n_changed": 5, "n_clamped_to_floor": 0}]
    de = {"file": "/x/run/edited-01.depth_edits.json", "n_edits": 2, "n_nodes_changed": 7, "n_clamped_to_floor": 1,
          "floor_m": 30.0}
    rec["stages"].insert(1, {"stage": "bathymetry", "wall_s": 0.0, "regions": reg, "depth_edits": de})
    rec["depth_edits"] = de
    s = write.summary(rec, config.DEFAULT_GATES)
    assert s["regions"] == reg and s["depth_edits"] == de


def test_run_list_and_refs_carry_date_resolution_and_scale(srv):
    """The sortable run and reference tables (floating windows) sort by date and resolution: a run gives its date (the
    final quality record's, else the folder's), its config's h as [finest, base] km after the [size] clip and the
    [mesh] scale (config.resolved.toml, else the UI's config.toml) and the scale; a reference mesh its typical element
    size in km (the quality record's median sqrt(area); a quad mesh here)."""
    runs = {x["name"]: x for x in json.loads(get("/api/runs")[1])["runs"]}
    x = runs["sphere"]
    assert x["date"] == "2026-10-22T12:00:00" and x["h_km"] == [2000.0, 2000.0] and x["scale"] == 1.0
    y = runs["m14_blocks_uniform_s4"]                       # no config, no quality record: the folder's date
    assert y["h_km"] is None and y["scale"] is None and y["date"][:2] == "20" and "T" in y["date"]
    d = srv["dir"] / "runs" / "zoned"                        # what Run writes before the job starts: config.toml
    d.mkdir()
    try:
        c = config.loads(SPHERE + """
[[size.layers]]
kind = "coast_distance"
h_m = 50000.0
grad = 0.1
[[size.layers]]
kind = "zone"
h_m = 20000.0
transition_m = 100000.0
polygon = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
""", "zoned")
        c = replace(c, mesh=replace(c.mesh, scale=0.5), size=replace(c.size, h_min_m=30e3, h_max_m=150e3))
        config.save(c, d / "config.toml")
        z = {x["name"]: x for x in json.loads(get("/api/runs")[1])["runs"]}["zoned"]
        assert z["h_km"] == [15.0, 75.0] and z["scale"] == 0.5     # finest max(20, 30) km, base min(2000, 150) km, x 0.5
        config.save(config.loads(SPHERE + """
[[size.layers]]
kind = "orca"
orca = "ORCA025"
h_min_m = 10000.0
[[size.layers]]
kind = "zone"
mode = "smooth"
smooth_m = 300000.0
transition_m = 100000.0
polygon = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
[[size.layers]]
kind = "zone"
mode = "blend"
h_m = 4500.0
transition_m = 0.0
polygon = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
outer = [[-5.0, -5.0], [15.0, -5.0], [15.0, 15.0]]
""", "orca"), d / "config.toml")
        z = {x["name"]: x for x in json.loads(get("/api/runs")[1])["runs"]}["zoned"]
        assert z["h_km"] == [4.5, 2000.0]              # the orca layer's h_min, a smooth zone none (run.h_floor)
    finally:
        shutil.rmtree(d)
    refs = json.loads(get("/api/refs")[1])["refs"]
    q = json.loads((srv["ref"] / "quality.json").read_text())["raw"]["quality"]
    assert refs[0]["res_km"] == pytest.approx(q["sqrt_area_m"]["median"] / 1000)


def test_run_list_searches_subfolders_and_folders_added_from_the_page(tmp_path):
    """A run root lists the run folders right in it; with "subfolders" on (the page's tick, per root) also those at
    any depth below (through hidden folders such as .worktrees and .tmp), not inside another run folder nor in
    __pycache__ / .git. Each run carries the folder it sits in, relative to its root's parent. A folder added from the
    page (POST /api/runs/roots) is searched too, until removed; the first root (where new runs go) cannot be removed.
    An unchanged list answers {"same": true} to the page's tag. /api/dirs lists a folder's subfolders for the page's
    folder browser, marking run folders, and with no path offers the run roots as places. A new run folder shows on
    the next list."""
    cfg = tmp_path / "c.toml"
    scenario.save(scenario.load(), cfg)
    mesh = write_tiny_mesh(tmp_path / "mesh") / "mesh.npz"
    first, extra = tmp_path / "runs", tmp_path / "meshom"
    first.mkdir()

    def run(d, name="raw.npz"):
        d.mkdir(parents=True)
        shutil.copy(mesh, d / name)
    run(extra / "r0")                                                                           # right in the root
    run(extra / ".worktrees" / "coastf" / ".tmp" / "coastf" / "runs" / "glob100")
    run(extra / ".worktrees" / "coastf" / ".tmp" / "coastf" / "runs" / "glob100" / "fesom")    # inside a run folder
    run(extra / ".tmp" / "compete" / "m14_blocks_uniform_s4", "smooth.npz")
    run(extra / "__pycache__" / "x")
    run(extra / ".git" / "y")
    (extra / "docs").mkdir()
    h = make_handler(cfg, mesh.parent, run_roots=[first], ref_roots=[])

    def call(method, path, obj=None):
        s, _, b = inproc.call(h, method, path, json.dumps(obj).encode() if obj is not None else b"")
        return s, json.loads(b)

    def listed():
        r = call("GET", "/api/runs")[1]
        return r["roots"], {(x["folder"], x["name"]) for x in r["runs"]}
    assert listed() == ([str(first)], set())
    s, r = call("POST", "/api/runs/roots", {"add": str(extra)})
    assert s == 200 and r["roots"] == [str(first), str(extra)] and r["path"] == str(extra) and r["deep"] == []
    assert listed()[1] == {("meshom", "r0")}
    s, r = call("POST", "/api/runs/roots", {"add": str(extra), "deep": True})          # the tick: subfolders too
    assert s == 200 and r["roots"] == [str(first), str(extra)] and r["deep"] == [str(extra)]
    roots, found = listed()
    assert found == {("meshom", "r0"), ("meshom/.worktrees/coastf/.tmp/coastf/runs", "glob100"),
                     ("meshom/.tmp/compete", "m14_blocks_uniform_s4")}
    s, hdr, _ = inproc.call(h, "GET", "/api/runs")                     # unchanged: the page is told so, not sent it all
    assert call("GET", f"/api/runs?known={hdr['X-Meshom-Tag']}")[1] == {"same": True, "tag": hdr["X-Meshom-Tag"]}
    run(extra / ".tmp" / "compete" / "m15_tri_band_s0.3")
    assert ("meshom/.tmp/compete", "m15_tri_band_s0.3") in listed()[1]
    s, r = call("GET", f"/api/dirs?path={extra / '.tmp' / 'compete'}")
    assert s == 200 and r["path"] == str(extra / ".tmp" / "compete") and r["parent"] == str(extra / ".tmp")
    assert [(x["name"], x["run"]) for x in r["dirs"]] == [("m14_blocks_uniform_s4", True), ("m15_tri_band_s0.3", True)]
    assert {x["name"] for x in call("GET", f"/api/dirs?path={extra}")[1]["dirs"]} >= {".tmp", ".worktrees", "docs"}
    assert str(extra) in [p["path"] for p in call("GET", "/api/dirs")[1]["places"]]
    assert call("POST", "/api/runs/roots", {"add": str(tmp_path / "nowhere")})[0] == 400
    assert call("POST", "/api/runs/roots", {"remove": str(first)})[0] == 400
    assert call("POST", "/api/runs/roots", {"remove": str(extra)})[1] == {"roots": [str(first)], "deep": [],
                                                                          "path": str(extra)}
    assert listed() == ([str(first)], set())


def test_a_runs_optimized_mesh_is_its_final_mesh(tmp_path):
    """`meshom run` writes optimized.npz (record "optimized" of quality.json) when it applies the optimisation stage:
    the run lists it as a mesh, takes its state and summary from it, opens it by default, and its info carries the
    optimized record (not raw's)."""
    from ui.web import runs as runsmod
    d = tmp_path / "runs" / "meshB"
    scored(d, {"method": "m14_blocks", "config": "meshB"})
    q = json.loads((d / "quality.json").read_text())
    q["optimized"] = {**q["raw"], "n_nodes": q["raw"]["n_nodes"] - 1, "status": "ok"}
    (d / "quality.json").write_text(json.dumps(q))
    shutil.copy(d / "raw.npz", d / "optimized.npz")
    x = runsmod.describe(d)
    assert x["meshes"] == ["raw.npz", "optimized.npz"] and x["state"] == "done"
    assert set(x["quality"]) == {"raw", "optimized"}
    assert meshview.mesh_file(d).name == "optimized.npz"
    cfg = tmp_path / "c.toml"
    scenario.save(scenario.load(), cfg)
    h = make_handler(cfg, d, run_roots=[tmp_path / "runs"], ref_roots=[])
    info = json.loads(inproc.call(h, "GET", "/api/mesh/info")[2])
    assert info["path"].endswith("optimized.npz") and info["quality"]["n_nodes"] == q["raw"]["n_nodes"] - 1


def test_hand_blocks_from_the_page_are_saved_as_blocks_tables(tmp_path, monkeypatch):
    """The Blocks panel edits the page config's `blocks` (a list of the set keys of [[blocks]] tables): saved, they are
    [[blocks]] tables the core reads back (corners counter-clockwise, sides, rows, h_m); unset keys stay unset (the
    core's defaults: sides by its rule, counts and h from the run's h), also when the config had no blocks before."""
    monkeypatch.setattr(scenario, "SAVE_DEFAULT", tmp_path / "config.toml")
    h = make_handler(scenario.EXAMPLE, write_tiny_mesh(tmp_path / "mesh"), run_roots=[tmp_path / "runs"], ref_roots=[])
    cfg = json.loads(inproc.call(h, "GET", "/api/config")[2])["config"]
    assert not cfg.get("blocks")
    cfg["blocks"] = [{"name": "Gibraltar", "corners": [[-6.2, 35.6], [-5.2, 35.8], [-5.3, 36.2], [-6.1, 36.3]],
                      "sides": ["cut", "coast", "cut", "coast"], "rows": 12, "h_m": 4000.0},
                     {"name": "block 2", "corners": [[-30.0, 0.0], [-20.0, 0.0], [-20.0, 10.0], [-30.0, 10.0]]}]
    r = json.loads(inproc.call(h, "POST", "/api/config", json.dumps({"config": jsify(cfg)}).encode())[2])
    b = config.load(Path(r["saved"])).blocks
    assert [x.name for x in b] == ["Gibraltar", "block 2"]
    assert b[0].corners[1] == [-5.2, 35.8] and b[0].sides == ["cut", "coast", "cut", "coast"] and b[0].rows == 12
    assert b[0].h_m == 4000.0 and b[0].cols is None
    assert b[1].sides is None and b[1].rows is None and b[1].h_m is None


def test_h_variants_save_load_and_a_config_that_refers_to_one(tmp_path):
    """h variants (S35): the page's [size] saved as a size-only file (meshom.config.dumps_size; zone modes and coast
    classes kept), listed and loaded back as the page's size; a config saved with size_from writes [size] from = the
    variant, relative to the config's folder, and no layers of its own, so the core reads the variant's layers
    (Config.size_from = the file); GET /api/config says which variant the open config uses."""
    cfgp = tmp_path / "c.toml"
    scenario.save(scenario.load(), cfgp)
    h = make_handler(cfgp, write_tiny_mesh(tmp_path / "mesh"), run_roots=[tmp_path / "runs"], ref_roots=[])

    def call(method, path, obj=None):
        s, _, b = inproc.call(h, method, path, json.dumps(jsify(obj)).encode() if obj is not None else b"")
        return s, json.loads(b)
    cfg = call("GET", "/api/config")[1]["config"]
    cfg["size"]["layers"] = [{"kind": "constant", "h_m": 100000.0},
                             {"kind": "coast_distance", "h_m": 20000.0, "plateau_m": 30000.0, "grad": 0.05,
                              "coasts": "continents", "island_max_km2": 50000.0},
                             {"kind": "zone", "name": "calm", "mode": "smooth", "smooth_m": 400000.0,
                              "transition_m": 100000.0, "polygon": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]}]
    s, r = call("POST", "/api/size/save", {"config": cfg, "path": str(tmp_path / "h-continents.toml")})
    assert s == 200 and r["saved"] == str(tmp_path / "h-continents.toml")
    text = (tmp_path / "h-continents.toml").read_text()
    sz = config.loads_size(text)
    assert [ly.kind for ly in sz.layers] == ["constant", "coast_distance", "zone"] and "[mesh]" not in text
    assert sz.layers[1].coasts == "continents" and sz.layers[2].mode == "smooth" and sz.layers[2].smooth_m == 400000.0
    assert str(tmp_path / "h-continents.toml") in [v["path"] for v in call("GET", "/api/size/variants")[1]["variants"]]
    s, r = call("POST", "/api/size/load", {"path": str(tmp_path / "h-continents.toml")})
    assert s == 200 and [ly["kind"] for ly in r["size"]["layers"]] == ["constant", "coast_distance", "zone"]
    cfg["size"] = r["size"]
    s, r = call("POST", "/api/config", {"config": cfg, "size_from": str(tmp_path / "h-continents.toml")})
    saved = Path(r["saved"]).read_text()
    assert 'from = "h-continents.toml"' in saved and "[[size.layers]]" not in saved
    c = config.load(Path(r["saved"]))
    assert c.size_from == str(tmp_path / "h-continents.toml") and c.size.layers[2].mode == "smooth"
    assert call("GET", "/api/config")[1]["size_from"] == str(tmp_path / "h-continents.toml")


def test_profile_along_a_line_gives_distance_h_and_the_bottom(tmp_path):
    """POST /api/profile (the Tools window): n points along the line's great-circle legs with their distance (km),
    the page config's h there (km) and the bottom of its [bathymetry] grid (m, positive up), land where the coast has
    it (none here: no coast)."""
    pytest.importorskip("netCDF4")
    from test_ui_common import write_grid
    grid = write_grid(tmp_path / "z.nc")
    c = config.loads(SPHERE + f'\n[bathymetry]\nsource = "{grid}"\n', "sphere")
    cfgp = tmp_path / "c.toml"
    config.save(c, cfgp)
    h = make_handler(cfgp, write_tiny_mesh(tmp_path / "mesh"), run_roots=[tmp_path / "runs"], ref_roots=[])
    s, _, b = inproc.call(h, "POST", "/api/profile", json.dumps({"line": [[0.0, 0.0], [10.0, 0.0]], "n": 11}).encode())
    r = json.loads(b)
    assert s == 200 and r["d_km"][-1] == pytest.approx(1111.95, abs=0.01) and len(r["d_km"]) == 11
    assert r["h_km"] == [2000.0] * 11 and r["elev_m"] == [-float(k) for k in range(11)] and not any(r["land"])
    assert r["bathy"] == str(grid) and r["note"] is None


def test_profile_shows_the_nodes_of_the_elements_the_line_crosses(tmp_path):
    """With the mesh shown (its path in the request), the profile also gives the nodes of the elements the line
    crosses: their distance along the line and offset from it (km), their mesh depth (meshview.depth_of) and the
    bottom grid's value at each node (a point sample, as the line's), sorted along the line. The 20 degree test
    sphere: a line at 10 N from 5 E to 75 E crosses the four quads of the 0-20 N band from 0 to 80 E."""
    pytest.importorskip("netCDF4")
    from test_ui_common import write_grid
    grid = write_grid(tmp_path / "z.nc")
    c = config.loads(SPHERE + f'\n[bathymetry]\nsource = "{grid}"\n', "sphere")
    cfgp = tmp_path / "c.toml"
    config.save(c, cfgp)
    md = write_tiny_mesh(tmp_path / "mesh")
    n = len(np.load(md / "mesh.npz")["points"])
    depth = 10.0 * np.arange(n) + 5.0
    np.save(md / "depth.npy", depth)
    h = make_handler(cfgp, md, run_roots=[tmp_path / "runs"], ref_roots=[])
    body = {"line": [[5.0, 10.0], [75.0, 10.0]], "n": 50, "mesh": str(md / "mesh.npz")}
    s, _, b = inproc.call(h, "POST", "/api/profile", json.dumps(body).encode())
    nd = json.loads(b)["nodes"]
    want = {j * 18 + i for j in (4, 5) for i in range(9, 14)}             # lat 0 and 20 N, lon 0 to 80 E
    assert s == 200 and set(nd["id"]) == want and len(nd["id"]) == 10
    assert nd["d_km"] == sorted(nd["d_km"]) and 0 <= nd["d_km"][0] and nd["d_km"][-1] <= 7800
    assert all(0 < o < 2300 for o in nd["off_km"])
    assert nd["depth_m"] == [depth[i] for i in nd["id"]]
    assert nd["elev_m"] == [100 * round(la) - round(lo) for lo, la in zip(nd["lon"], nd["lat"])]
    s, _, b = inproc.call(h, "POST", "/api/profile", json.dumps({**body, "mesh": None}).encode())
    assert json.loads(b)["nodes"] is None                                    # no mesh shown: no nodes


def test_node_selection_gives_depth_statistics(tmp_path):
    """The node tool (Tools, "Mesh depth"): the nodes of the mesh shown inside a box (west, east, south, north as the
    map draws it; east > 180 across the antimeridian) or a polygon (great-circle edges), their mesh depth (min,
    median, mean, max), the bottom grid at the nodes (point samples) and their positions for the map."""
    pytest.importorskip("netCDF4")
    from test_ui_common import write_grid
    grid = write_grid(tmp_path / "z.nc")
    c = config.loads(SPHERE + f'\n[bathymetry]\nsource = "{grid}"\n', "sphere")
    cfgp = tmp_path / "c.toml"
    config.save(c, cfgp)
    md = write_tiny_mesh(tmp_path / "mesh")
    n = len(np.load(md / "mesh.npz")["points"])
    depth = 10.0 * np.arange(n) + 5.0
    np.save(md / "depth.npy", depth)
    h = make_handler(cfgp, md, run_roots=[tmp_path / "runs"], ref_roots=[])

    def select(**area):
        s, _, b = inproc.call(h, "POST", "/api/mesh/depth/select", json.dumps({"path": str(md / "mesh.npz"), **area}).encode())
        assert s == 200, b
        return json.loads(b)
    r = select(box=[0.0, 40.0, -10.0, 30.0])
    ids = sorted(j * 18 + i for j in (4, 5) for i in (9, 10, 11))           # lon 0, 20, 40 E x lat 0, 20 N
    assert r["n"] == 6 and sorted(r["ids"]) == ids
    d = depth[ids]
    assert r["depth"] == pytest.approx({"min": d.min(), "median": float(np.median(d)), "mean": d.mean(), "max": d.max()})
    z = [100 * la - lo for la in (0, 20) for lo in (0, 20, 40)]
    assert r["bottom"] == pytest.approx({"min": min(z), "median": float(np.median(z)), "max": max(z)})
    assert len(r["lon"]) == len(r["lat"]) == 6 and r["depth_source"]
    across = select(box=[150.0, 190.0, -10.0, 10.0])                         # 160 E, 180, 170 W ... at the equator
    assert sorted(round(x) for x in across["lon"]) == [-180, 160]
    poly = select(polygon=[[-5.0, -5.0], [45.0, -5.0], [45.0, 25.0], [-5.0, 25.0]])
    assert sorted(poly["ids"]) == ids
    assert select(box=[1.0, 2.0, 1.0, 2.0])["n"] == 0
