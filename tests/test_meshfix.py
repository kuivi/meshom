"""The mesh-fixing bench (meshom.meshfix, decisions B120-B121): the scorer on a tiny mesh with known answers, and a
cut-out that keeps its artificial-boundary marking."""
import numpy as np
import pytest

from meshom.config import DEFAULT_GATES
from meshom.geo import stereographic_inverse
from meshom.mesh import Mesh, orient_ccw
from meshom.meshfix import cases as mc
from meshom.meshfix.score import analyse
from meshom.size import Uniform

L = 10_000.0      # m


def _fan(move_v0_m: float = 0.0) -> Mesh:
    """Three kites around one valence-3 node c (corner angles at c 125, 117, 118 deg; right angles at the spoke
    ends, so each kite's largest angle is the one at c) and an equilateral triangle on the outer edge v1-v2; every
    other edge is coast. move_v0_m moves the spoke end v0 outward (its two coast edges are collinear)."""
    phis = [125.0, 117.0, 118.0]
    spokes = np.radians([90.0, 215.0, 332.0])
    xy = {"c": (0.0, 0.0)}
    for k in range(3):
        a = spokes[k]
        xy[f"v{2 * k}"] = (L * np.cos(a), L * np.sin(a))
        b = a + np.radians(phis[k]) / 2
        r = L / np.cos(np.radians(phis[k]) / 2)
        xy[f"v{2 * k + 1}"] = (r * np.cos(b), r * np.sin(b))
    v1, v2 = np.array(xy["v1"]), np.array(xy["v2"])
    m = (v1 + v2) / 2
    d = v2 - v1
    out = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    out *= np.sign(out @ m)
    xy["t"] = tuple(m + out * np.sqrt(3) / 2 * np.linalg.norm(v2 - v1))
    xy["v0"] = (xy["v0"][0], xy["v0"][1] + move_v0_m)
    names = ["c", "v0", "v1", "v2", "v3", "v4", "v5", "t"]
    ix = {n: i for i, n in enumerate(names)}
    P = np.array([xy[n] for n in names])
    pts = stereographic_inverse(P[:, 0], P[:, 1], 0.0, 0.0)
    quads = np.array([[ix[a] for a in q] for q in (("c", "v0", "v1", "v2"), ("c", "v2", "v3", "v4"),
                                                   ("c", "v4", "v5", "v0"))])
    tris = np.array([[ix["v2"], ix["v1"], ix["t"]]])
    flag = np.ones(len(names), dtype=np.int8)
    flag[0] = 0
    return orient_ccw(Mesh(pts, tris, quads, flag))


def _case(h_m: float) -> mc.Case:
    return mc.finish(mc.whole_case("fan", _fan(), Uniform(h_m), 0.2, dict(DEFAULT_GATES, size_fidelity_p99=0.2)))


def test_scorer_known_answers():
    case = _case(4000.0)                     # c lies 10 km = 2.5 h from the coast: open ocean
    r, _ = analyse(case, case.mesh)
    q, t, irr = r["quad_angle"], r["triangles"], r["irregular"]
    assert q["n"] == 3 and q["n_gt110"] == 3 and q["n_gt120"] == 1
    assert q["max"] == pytest.approx(125.0, abs=0.01)
    assert q["by_region"]["coastal"]["gt120"] == 1    # the 125-deg kite's centroid is 6.4 km from the coast
    assert t["n"] == 1 and t["max_angle_max"] == pytest.approx(60.0, abs=0.01) and t["clusters_ge3"] == 0
    assert t["by_region"]["coastal"]["n"] == 1
    assert irr["open"]["n3"] == 1 and irr["all"]["n5"] == 0 and irr["all"]["tri_nodes"] == 0
    assert irr["defect_units_open"] == 1 and irr["defect_units_all"] == 2
    v = r["validity"]
    assert v["ok"] and v["euler"]["euler_sum"] == 4 * v["euler"]["chi"] == 4
    c = r["coast"]
    assert c["shift_km_max"] == pytest.approx(0.0, abs=1e-6) and c["area_change_pct"] == pytest.approx(0.0, abs=1e-9)
    assert c["water_bodies"] == 1 and c["basins_lost"] == 0
    # the triangle has one neighbour: the input fails elem_neighbours_min, so it is no new failure
    assert "elem_neighbours_min" in r["gates"]["failed"] and r["gates"]["new_failed"] == []
    # spokes are 10 km: |ln(10/4)| is the smallest edge error; p99 over every edge
    assert r["size"]["n_edges"] == 11

    coastal = _case(6000.0)                  # 10 km < 2 h: the valence-3 node is coastal
    r2, _ = analyse(coastal, coastal.mesh)
    assert r2["irregular"]["open"]["n3"] == 0 and r2["irregular"]["coastal"]["n3"] == 1


def test_scorer_coast_shift_and_area():
    case = _case(4000.0)
    r, _ = analyse(case, _fan(move_v0_m=1000.0))
    c = r["coast"]
    assert c["forward_km_max"] == pytest.approx(1.0, abs=0.01)
    assert c["shift_km_max"] == pytest.approx(1.0, abs=0.01)
    # added area: the triangle v5 v0' v1 of height 1 km over the straight coast v5-v1
    base_km = 2 * L / 1e3 * (np.tan(np.radians(59.0)) + np.tan(np.radians(62.5))) / 2
    assert c["area_change_pct"] == pytest.approx(100 * 0.5 * base_km / case.meta["input_area_km2"], rel=0.01)
    assert r["validity"]["ok"]


def _grid(nx: int, ny: int) -> Mesh:
    x, y = np.meshgrid(np.arange(nx + 1) * L, np.arange(ny + 1) * L, indexing="ij")
    pts = stereographic_inverse(x.ravel(), y.ravel(), 0.0, 0.0)
    nid = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    quads = np.array([[nid[i, j], nid[i + 1, j], nid[i + 1, j + 1], nid[i, j + 1]] for i in range(nx)
                      for j in range(ny)])
    flag = np.zeros(len(pts), dtype=np.int8)
    flag[np.r_[nid[0], nid[-1], nid[:, 0], nid[:, -1]]] = 1
    return orient_ccw(Mesh(pts, np.zeros((0, 3), np.int64), quads, flag))


def test_cutout_keeps_artificial_marking(tmp_path):
    full = _grid(10, 6)
    parent = mc.whole_case("grid", full, Uniform(L), 0.2, dict(DEFAULT_GATES, size_fidelity_p99=0.2))
    c = parent.mesh.points[parent.mesh.quads].mean(axis=1)
    keep = c[:, 1] < c[:, 1].min() + 4.5 * L / 6.371e6       # the western 5 of the 10 columns (x < 5 L)
    case, old = mc.cutout_case("grid_west", parent, keep, {"h_pkl": "h.pkl"}, min_elems=1)
    case = mc.finish(case)
    i, j = np.divmod(old, 7)                                  # parent node (i, j) = id // 7, id % 7
    assert (case.kind == 2).sum() == 7 and np.all(i[case.kind == 2] == 5)
    assert np.all((case.kind == 1) == ((i < 5) & ((i == 0) | (j == 0) | (j == 6))))
    assert len(case.art_edges) == 6 and len(case.coast_seg) == 5 + 5 + 6
    mc.save(case, tmp_path, write_h=True)
    back = mc.load("grid_west", tmp_path)
    assert np.array_equal(back.kind, case.kind) and np.array_equal(back.art_edges, case.art_edges)
    assert np.array_equal(back.fixed, case.kind == 2)
    r, _ = analyse(back, back.mesh)
    assert r["validity"]["ok"] and r["validity"]["artificial_nodes_missing"] == 0
    assert r["coast"]["shift_km_max"] == pytest.approx(0.0, abs=1e-6)
    # a method that moves a fixed node breaks the artificial boundary
    P = back.mesh.points.copy()
    k = np.flatnonzero(back.kind == 2)[3]
    P[k] += 1e-6 * np.cross(P[k], [0.0, 0.0, 1.0])
    P[k] /= np.linalg.norm(P[k])
    r2, _ = analyse(back, Mesh(P, back.mesh.tris, back.mesh.quads, back.mesh.node_flag))
    assert r2["validity"]["artificial_nodes_missing"] == 1 and not r2["validity"]["ok"]
    # a method that keeps a fixed node in place but changes its quads / triangles uses the artificial boundary as a
    # defect sink: split a quad at an artificial node into two triangles (positions and edges of the boundary kept)
    Q = back.mesh.quads
    iq = int(np.flatnonzero(np.isin(Q, np.flatnonzero(back.kind == 2)).any(axis=1))[0])
    a, b, c_, d = Q[iq]
    split = Mesh(back.mesh.points, np.array([[a, b, c_], [a, c_, d]]), np.delete(Q, iq, axis=0),
                 back.mesh.node_flag)
    r3, _ = analyse(back, split)
    assert r3["validity"]["artificial_incidence_changed"] == 2 and not r3["validity"]["ok"]
    assert r3["validity"]["artificial_nodes_missing"] == 0 and r3["validity"]["artificial_edges_missing"] == 0


def test_worker_h_file_is_never_found_by_the_case_name(tmp_path, monkeypatch):
    # a case built in memory (pipeline --run) whose run folder shares the name of a bench case must not hand the
    # bench case's h pickle to the engines' workers; a loaded case points at its own pickle
    monkeypatch.setenv("MESHFIX_DIR", str(tmp_path))
    bench = _case(100e3)
    mc.save(bench, mc.case_dir("fan"), write_h=True)
    built = _case(90e3)
    assert mc.h_file(built) is None
    loaded = mc.load("fan")
    assert mc.h_file(loaded) == (mc.case_dir("fan") / "h.pkl").resolve()
    assert "h_file" not in (mc.case_dir("fan") / "case.json").read_text()
