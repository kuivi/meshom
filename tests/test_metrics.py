import numpy as np

from helpers import local_polygon, mesh_from_polygons, split_quads, structured_patch
from meshom.config import DEFAULT_GATES, load_gates
from meshom.quality.metrics import check_gates, compute_quality


def test_uniform_patch():
    s = compute_quality(structured_patch(6, 5, 1e4)).summary
    assert s["quad_dev_deg"]["max"] < 0.01 and s["elem_size_ratio"]["max"] < 1.001
    assert s["node_size_ratio"]["max"] < 1.001 and s["n_tri_quad_edges"] == 0
    assert s["elems_per_node_hist"] == {1: 4, 2: 18, 4: 20}
    assert s["n_pinch_nodes"] == 0 and s["n_elems_ge2_boundary_edges"] == 4      # the 4 corner quads
    assert s["tri_min_angle_deg"]["min"] is None and s["quad_sj_interior"]["min"] > 0.999


def test_column_jump_ratio():
    xs = np.array([0, 1, 2, 3, 5, 7, 9]) * 1e4            # 10 km then 20 km columns, rows 10 km
    s = compute_quality(structured_patch(6, 4, 1e4, xs_m=xs)).summary
    assert np.isclose(s["elem_size_ratio"]["max"], 1.5, atol=1e-3)     # mean edge 10 km vs 15 km
    assert np.isclose(s["share_elem_size_ratio_gt_1_2"], 4 / 38, atol=1e-9)
    assert np.isclose(s["elem_sqrt_area_ratio"]["max"], 2**0.5, atol=1e-3)


def test_size_ratio_tri_quad_equal_edges():
    sq = local_polygon(0, 0, [(0, 0), (1e4, 0), (1e4, 1e4), (0, 1e4)])
    top = local_polygon(0, 0, [(0, 1e4), (1e4, 1e4), (5e3, 1e4 + 5e3 * 3**0.5)])
    s = compute_quality(mesh_from_polygons([sq, top])).summary   # shared nodes merged by the helper
    assert np.isclose(s["elem_size_ratio"]["max"], 1.0, atol=1e-4) and s["n_tri_quad_edges"] == 1


def test_triangle_clusters():
    m = split_quads(structured_patch(8, 4, 1e4), [0, 1, 2, 20])   # helpers: split listed quads along a diagonal
    c = compute_quality(m).summary["tri_clusters"]
    assert c == {"n": 2, "max_size": 6, "n_ge3": 1}


def test_pinch_node():
    a = local_polygon(0, 0, [(0, 0), (1e4, 0), (1e4, 1e4), (0, 1e4)])
    b = local_polygon(0, 0, [(1e4, 1e4), (2e4, 1e4), (2e4, 2e4), (1e4, 2e4)])     # touches a at one corner
    s = compute_quality(mesh_from_polygons([a, b])).summary
    assert s["n_pinch_nodes"] == 1 and s["n_elems_ge2_boundary_edges"] == 2


def test_gates():
    s = compute_quality(structured_patch(4, 4, 1e4)).summary
    s["quad_max_angle_deg"]["max"] = 151.0
    s["quad_sj_coast"]["min"] = 0.2
    r = {g["gate"]: g["passed"] for g in check_gates(s, DEFAULT_GATES)}
    assert r["quad_max_angle_hard_deg"] is False and r["elem_size_ratio_p99"] is True
    assert r["quad_max_angle_hard_deg_interior"] is True and r["elem_size_ratio_max_coast"] is True
    assert r["quad_sj_hard_coast"] is False and r["quad_sj_hard"] is True       # gates per region
    assert r["tri_max_angle_hard_deg"] is True and r["tri_min_angle_hard_deg_coast"] is True   # no triangles: None
    assert "size_fidelity_p99" not in r and "node_size_ratio_p99" not in r      # no h given; node ratio not gated
    assert r["elems_per_node_max"] is True and "elems_per_node_max_coast" not in r


def test_load_gates_override(tmp_path):
    (tmp_path / "g.toml").write_text("[gates]\nquad_max_angle_hard_deg = 140.0\n")
    g = load_gates(tmp_path / "g.toml")
    assert g["quad_max_angle_hard_deg"] == 140.0 and g["elems_per_node_max"] == 32
