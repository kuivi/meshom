"""Smoothness first within a size tolerance (decisions G38, M38).

smooth_fit leaves a smooth mesh alone while its edges are within the tolerance of h (a fit to h would move it), pulls
the edges back within the tolerance where they are not, and never inverts; the size-fidelity gate is judged at the
tolerance a mesh was built for ([mesh] size_tolerance, opts["size_tolerance"] in meshom compete), the element-ratio
gates are not.
"""
import json

import numpy as np
import pytest

from meshom import compete, config
from meshom.geo import R_EARTH_M, arc_length
from meshom.mesh import Mesh, build_topology
from meshom.quality.kernel import element_geometry
from meshom.size import disc
from meshom.size_fit import edge_ln_sh, ln_ratio, size_fit, smooth_fit
from tests.helpers import local_polygon

H = 20e3
N = 24                         # cells per side of the square patch
# h = 1.5 H within 3 H of the centre, down to H at 9 H: the regular grid (spacing H) is smooth, but its edges are
# up to ln 1.5 = 0.405 off h in the middle
HB = disc(20.0, 30.0, 3 * H, 1.5 * H, 9 * H, H)


def _grid() -> Mesh:
    """N x N quads of side H about (20E, 30N), CCW; the rim nodes are the fixed boundary (flag 1)."""
    i, j = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing="ij")
    xy = (np.c_[i.ravel(), j.ravel()] - N / 2) * H
    k = lambda a, b: a * (N + 1) + b                                     # noqa: E731
    q = np.array([[k(a, b), k(a + 1, b), k(a + 1, b + 1), k(a, b + 1)] for a in range(N) for b in range(N)])
    rim = ((i == 0) | (i == N) | (j == 0) | (j == N)).ravel()
    return Mesh(local_polygon(20.0, 30.0, xy), np.zeros((0, 3), dtype=np.int64), q, rim.astype(np.int8))


def _moved(a: Mesh, b: Mesh) -> float:
    return float(np.max(R_EARTH_M * arc_length(a.points, b.points)) / H)


def test_within_the_tolerance_smoothness_wins():
    """Every edge is within 0.5 of h: the smooth grid stays (a size fit to h moves it by ~half a cell)."""
    g = _grid()
    r0 = np.abs(edge_ln_sh(g.points, build_topology(g).edges, HB))
    assert 0.35 < r0.max() < 0.5
    out, info = smooth_fit(g, HB, 0.5)
    assert _moved(g, out) < 0.02 and info["after"]["ratio_p99"] < 1.01
    fit, _ = size_fit(g, HB)
    assert _moved(g, fit) > 0.3


def test_the_tolerance_is_a_wall():
    """At tolerance 0.15 the bump's edges (up to 0.405 off h) come back within it: |ln s/h| p99 <= tol, max <= tol +
    0.03, while the element ratio stays small and every quad keeps its corners near 60-120 deg (the soft angle wall)."""
    g = _grid()
    tol = 0.15
    r0 = np.abs(edge_ln_sh(g.points, build_topology(g).edges, HB))
    assert np.mean(r0 > tol) > 0.15
    out, info = smooth_fit(g, HB, tol)
    r = np.abs(edge_ln_sh(out.points, build_topology(out).edges, HB))
    assert np.percentile(r, 99) <= tol and r.max() <= tol + 0.03, (np.percentile(r, 99), r.max())
    assert np.exp(ln_ratio(out).max()) < 1.15
    ang = np.degrees(element_geometry(out).quad_angles)
    assert ang.min() > 55.0 and ang.max() < 125.0
    assert info["after"]["share_over_tol"] == pytest.approx(float(np.mean(r > tol)))


def test_size_gate_reads_the_tolerance():
    base = '[mesh]\nmethod = "m00_background"\n'
    assert config.loads(base).gates_all()["size_fidelity_p99"] == config.DEFAULT_GATES["size_fidelity_p99"]
    cfg = config.loads(base + "size_tolerance = 0.3\n")
    g = cfg.gates_all()
    assert g["size_fidelity_p99"] == 0.3 and g["elem_size_ratio_p99"] == config.DEFAULT_GATES["elem_size_ratio_p99"]
    assert config.loads(base + "size_tolerance = 0.3\n[gates]\nsize_fidelity_p99 = 0.2\n").gates_all()[
        "size_fidelity_p99"] == 0.2                                     # an explicit [gates] value wins
    assert config.loads(base + 'opts = {size_tolerance = 0.25}\n').gates_all()["size_fidelity_p99"] == 0.25
    assert config.loads(config.dumps(cfg)) == cfg


def test_compete_judges_a_mesh_at_its_tolerance(tmp_path, monkeypatch):
    """m00_background (edges ~0.3 off h at p99) fails the default size gate and passes it at tolerance 1."""
    monkeypatch.setattr(compete, "WORK_DIR", tmp_path / "work")
    j = tmp_path / "c.jsonl"
    r0 = compete.run_case("m00_background", "uniform", 8, out=j)
    r1 = compete.run_case("m00_background", "uniform", 8, opts={"size_tolerance": 1.0}, out=j)
    assert r0["quality"]["size_fidelity"]["p99"] > config.DEFAULT_GATES["size_fidelity_p99"]
    assert "size_fidelity_p99" in r0["gates_failed"] and "size_fidelity_p99" not in r1["gates_failed"]
    assert r1["quality"]["size_fidelity"] == r0["quality"]["size_fidelity"]           # the same mesh, other limit
    assert r1["opts"] == {"size_tolerance": 1.0}


def test_run_passes_the_tolerance_to_the_method_and_the_gate(tmp_path):
    """meshom run: [mesh] size_tolerance reaches the method's opts (the record's opts are those the method ran with)
    and the size gate of the score."""
    from meshom.run import run

    text = """
[mesh]
method = "m00_background"
size_tolerance = 1.0
[coast]
source = "none"
[size]
grad_max = 0.1
[[size.layers]]
kind = "constant"
h_m = 800000.0
[run]
timeout = 300.0
"""
    r = run(config.loads(text, "tol"), tmp_path / "tol")["raw"]
    assert r["status"] == "ok", r.get("error")
    assert r["opts"] == {"size_tolerance": 1.0}
    assert r["quality"]["size_fidelity"]["p99"] > 0.15 and "size_fidelity_p99" not in r["gates_failed"]


@pytest.mark.parametrize("method", ["m15_tri", "m14_blocks", "m14_blocks_mix"])
def test_methods_end_with_the_smooth_fit(method):
    """With size_tolerance the last stage of m15_tri and of m14's stitch is smooth_fit (its record), and the size
    error stays within the tolerance at p99."""
    from meshom.domain import closed_sphere
    from meshom.methods import OVERRIDES
    from meshom.methods import m14_blocks, m15_tri
    from meshom.size import uniform

    h = uniform(500e3)
    if method == "m15_tri":
        mesh, info = m15_tri.run(closed_sphere(), h, {"size_tolerance": 0.2})
        rec = info["tol_fit"]
        assert "fit" not in info
    else:
        mesh, info = m14_blocks.run(closed_sphere(), h, dict(OVERRIDES[method], size_tolerance=0.2))
        rec = info["stitch"]["tol_fit"]
    assert rec["tol"] == 0.2 and rec["after"]["ln_sh_p99"] <= 0.2
    r = np.abs(edge_ln_sh(mesh.points, build_topology(mesh).edges, h))
    assert np.percentile(r, 99) <= 0.2


def test_rep_ends_with_the_smooth_fit(tmp_path, monkeypatch):
    """m14_blocks_rep with size_tolerance: the repair counts an edge as bad beyond max(t, 0.15), then smooth_fit runs
    once more on the repaired mesh (its record tol_fit_final.json), and the result is within t at p99."""
    from meshom import compete as cp
    from meshom.domain import closed_sphere
    from meshom.methods import METHODS
    from meshom.size import uniform

    monkeypatch.setenv("MESHOM_RUN_DIR", str(tmp_path))
    h = uniform(500e3)
    mesh = METHODS["m14_blocks_rep"](closed_sphere(), h, {"size_tolerance": 0.1})
    cp.DEFERRED.clear()
    rec = json.loads((tmp_path / "tol_fit_final.json").read_text())
    rep = json.loads((tmp_path / "repair.json").read_text())
    assert rec["tol"] == 0.1 and rep["options"]["bad_edge"] == 0.15
    r = np.abs(edge_ln_sh(mesh.points, build_topology(mesh).edges, h))
    assert np.percentile(r, 99) <= 0.1 and rec["after"]["ln_sh_p99"] == pytest.approx(np.percentile(r, 99))


def test_m14_default_tolerance_and_the_switch():
    """G38: every m14 method runs at 0.20 when the caller gives none, m15_tri at none; 0 is off."""
    from meshom.methods import METHODS, defaults, effective_tolerance

    m14 = [k for k in METHODS if k.startswith("m14_blocks")]
    assert "m14_blocks_rep" in m14 and len(m14) >= 6
    for k in m14:
        assert effective_tolerance(k, {}) == 0.2, k
        assert k == "m14_blocks_rep" or defaults(k)["size_tolerance"] == 0.2, k
        assert effective_tolerance(k, {"size_tolerance": 0}) is None
        assert effective_tolerance(k, {"size_tolerance": 0.3}) == 0.3
    assert effective_tolerance("m15_tri", {}) is None and effective_tolerance("m00_background", None) is None
    base = '[mesh]\nmethod = "m14_blocks_rep"\n'
    assert config.loads(base).gates_all()["size_fidelity_p99"] == 0.2                  # built at 0.20, judged at 0.20
    assert config.loads(base + "size_tolerance = 0\n").gates_all()["size_fidelity_p99"] == 0.15     # off: G9
    assert config.loads(base + "opts = {size_tolerance = 0}\n").size_tolerance() is None
    assert config.loads('[mesh]\nmethod = "m15_tri"\n').gates_all()["size_fidelity_p99"] == 0.15


@pytest.mark.parametrize("opts, tol", [({}, 0.2), ({"size_tolerance": 0}, None), ({"size_tolerance": 0.3}, 0.3)])
def test_compete_gate_follows_the_default(tmp_path, monkeypatch, opts, tol):
    """meshom compete judges an m14 run at the tolerance it ran with (the method's default when opts give none)."""
    seen = []
    monkeypatch.setattr(compete, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(compete, "_spawn", lambda *a: ("error", "stub"))
    monkeypatch.setattr(compete, "tolerance_gates", lambda g, t, *a: seen.append(t) or g)
    compete.run_case("m14_blocks", "uniform", 8, opts=opts, out=tmp_path / "c.jsonl")
    assert seen == [tol]


def test_m14_default_reaches_the_stitch_and_zero_switches_it_off():
    from meshom.domain import closed_sphere
    from meshom.methods import m14_blocks
    from meshom.size import uniform

    h = uniform(500e3)
    _, info = m14_blocks.run(closed_sphere(), h, {})
    assert info["stitch"]["tol_fit"]["tol"] == 0.2
    _, info = m14_blocks.run(closed_sphere(), h, {"size_tolerance": 0})
    assert "tol_fit" not in info["stitch"]


@pytest.mark.parametrize("opts, tol", [({}, 0.2), ({"size_tolerance": 0}, None)])
def test_rep_default_and_off(tmp_path, monkeypatch, opts, tol):
    """m14_blocks_rep: the default 0.20 gives the repair bad edge 0.20 and the final fit at 0.20; 0 = as before."""
    from meshom import compete as cp
    from meshom.domain import closed_sphere
    from meshom.methods import METHODS
    from meshom.size import uniform

    monkeypatch.setenv("MESHOM_RUN_DIR", str(tmp_path))
    METHODS["m14_blocks_rep"](closed_sphere(), uniform(500e3), opts)
    cp.DEFERRED.clear()
    rep = json.loads((tmp_path / "repair.json").read_text())
    assert rep["options"]["bad_edge"] == (tol or 0.15)
    f = tmp_path / "tol_fit_final.json"
    assert (json.loads(f.read_text())["tol"] == tol) if tol else not f.exists()


def test_run_passes_zero_to_the_method(tmp_path):
    """meshom run with [mesh] size_tolerance = 0: the method gets 0 (an m14 method then skips its 0.20), the size
    gate is G9's 0.15."""
    from meshom.run import run

    cfg = config.loads('[mesh]\nmethod = "m00_background"\nsize_tolerance = 0.0\n[coast]\nsource = "none"\n[size]\n'
                       'grad_max = 0.1\n[[size.layers]]\nkind = "constant"\nh_m = 800000.0\n[run]\ntimeout = 300.0\n',
                       "tol0")
    assert cfg.gates_all()["size_fidelity_p99"] == 0.15
    r = run(cfg, tmp_path / "tol0")["raw"]
    assert r["status"] == "ok", r.get("error")
    assert r["opts"] == {"size_tolerance": 0.0}


def test_compete_keeps_an_explicit_size_gate(tmp_path, monkeypatch):
    """A size-fidelity gate the caller gives run_case is not replaced by the tolerance (as Config.gates_all)."""
    seen = []
    monkeypatch.setattr(compete, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(compete, "_spawn", lambda *a: ("error", "stub"))
    real = compete.tolerance_gates
    monkeypatch.setattr(compete, "tolerance_gates", lambda *a: seen.append(real(*a)) or seen[-1])
    compete.run_case("m14_blocks", "uniform", 8, gates={"size_fidelity_p99": 0.05}, out=tmp_path / "c.jsonl")
    compete.run_case("m14_blocks", "uniform", 8, out=tmp_path / "c.jsonl")
    assert [g["size_fidelity_p99"] for g in seen] == [0.05, 0.2]


def test_seam_size_gate_follows_the_tolerance():
    """The seam size gate (G33, 0.18) of a mesh built at tolerance t is max(0.18, t) (as G38 on the whole mesh)."""
    from meshom.quality.seams import SEAM_GATES, seam_gates, tolerance_seam_gates

    b = {"ln_sh_p99": 0.19}
    assert "seam_ln_sh_p99_max" in seam_gates(b) and tolerance_seam_gates(None) == SEAM_GATES
    assert seam_gates(b, tolerance_seam_gates(0.2)) == {}
    assert tolerance_seam_gates(0.1)["seam_ln_sh_p99_max"] == 0.18


def test_compete_judges_the_seam_size_at_the_tolerance(tmp_path, monkeypatch):
    """run_case hands the run's tolerance to the seam block's gates (the default, given, the m14 default)."""
    from pathlib import Path

    from meshom.quality import seams
    from tests.helpers import structured_patch

    def spawn(code, spec, timeout):
        d = Path(json.loads(spec)["dir"])
        compete._save(structured_patch(4, 4, 50e3), d / "raw.npz")
        (d / "stages.json").write_text(json.dumps([{"stage": "generate", "wall_s": 0.0, "peak_rss_mb": 0.0}]))
        return "ok", None

    seen = []
    monkeypatch.setattr(compete, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(compete, "_spawn", spawn)
    monkeypatch.setattr(seams, "seam_block", lambda m, h, spec, gates=None: seen.append(
        (gates or seams.SEAM_GATES)["seam_ln_sh_p99_max"]) or {"gates_failed": []})
    for m, o in (("m00_background", None), ("m00_background", {"size_tolerance": 1.0}), ("m14_blocks", None)):
        compete.run_case(m, "seam_rot0_k0", 8, opts=o, out=tmp_path / "c.jsonl")
    assert seen == [0.18, 1.0, 0.2]
