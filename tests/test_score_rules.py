"""The measures of the competition table (config.DEFAULT_GATES) pinned through
compete._score on hand-built meshes with known answers: |ln s/h| per edge with s on the sphere and h at the edge
midpoint, the element-size ratio across edges (not across nodes), the p99 / p99.9 statistics the gates read, the quad
corner angle in the element's tangent plane, n3 / n5 and the Euler sum with "other" valences, the count of applicable
gates. Patches sit at 60-75 N, across the antimeridian and over the pole, where a lon-lat measure would break.
Also the later gates: the triangle-cluster gate with triangle zones, the boundary rule
("tri" / "all" / "off"), the pinch and Euler gates, the coast size set without the coast edges, the first three
coastal rows and the strait corridor, failed gate families, the child's own peak RSS, and the re-score of saved runs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers import local_polygon, mesh_from_polygons, split_quads, structured_patch
from meshom import compete
from meshom.cases import CASES, STRAITS
from meshom.config import load_gates
from meshom.domain import Domain
from meshom.geo import R_EARTH_M, arc_length, lonlat_to_xyz
from meshom.mesh import Mesh, build_topology, orient_ccw
from meshom.size import uniform

GATES = load_gates()
_RHO_C = compete._rho_c


@pytest.fixture(autouse=True)
def _no_rho(monkeypatch):
    monkeypatch.setattr(compete, "_rho_c", lambda case: {})     # rho_c belongs to the case, not to the mesh


def grid(xs, ys, lon0=0.0, lat0=0.0, rot_deg=0.0, move=None) -> Mesh:
    """Quads on the east/north offsets xs x ys (m), turned by rot_deg, inverse gnomonic at (lon0, lat0); move =
    (node, dx, dy) shifts one node in the plane first."""
    X, Y = np.meshgrid(np.asarray(xs, float), np.asarray(ys, float))
    xy = np.column_stack([X.ravel(), Y.ravel()])
    if move is not None:
        xy[move[0]] += move[1:]
    a = np.radians(rot_deg)
    xy = xy @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    nx = len(xs) - 1
    j, i = np.divmod(np.arange(nx * (len(ys) - 1)), nx)
    n = j * (nx + 1) + i
    quads = np.column_stack([n, n + 1, n + nx + 2, n + nx + 1])
    return Mesh(local_polygon(lon0, lat0, xy), np.zeros((0, 3), np.int64), quads, np.zeros(len(xy), np.int8))


def score(mesh: Mesh, h, loops: int = 1, gates: dict = GATES, strait=None) -> dict:
    """_score with a domain of `loops` coast loops (only their number enters: chi_domain = 2 - loops)."""
    return compete._score(mesh, Domain([np.zeros((0, 3))] * loops), h, "uniform", gates, strait)


@pytest.mark.parametrize("lon0, lat0, rot", [(10.0, 60.0, 0.0), (180.0, 0.0, 0.0), (0.0, 90.0, 0.0),
                                             (-170.0, 75.0, 30.0)])
def test_square_patch_on_the_sphere(lon0, lat0, rot):
    """8 x 6 cells of 10 km (great-circle edges) at 60 N, across the antimeridian, over the pole, turned at 75 N."""
    m = grid(np.arange(9) * 1e4 - 4e4, np.arange(7) * 1e4 - 3e4, lon0, lat0, rot)
    r = score(m, uniform(1e4))
    q = r["quality"]
    assert q["size_fidelity"]["max"] < 1e-3 and abs(q["size_fidelity"]["bias"]) < 1e-3
    assert abs(q["quad_max_angle_deg"]["max"] - 90.0) < 0.01 and q["elem_size_ratio"]["max"] < 1.001
    ic = r["irregular"]
    assert (ic["n3"], ic["n5"], ic["n_other"], ic["T"], ic["euler_sum"], ic["euler_ok"]) == (0, 0, 0, 0, 4, True)
    # quads, no triangle gate; 6 region gates x (whole, coast, interior, first 3 coastal rows) + neighbours,
    # elements per node, triangle clusters (0), triangle share, pinch, Euler; the boundary rule "tri" has no triangle to check
    assert r["gates_failed"] == {} and r["gates_total"] == 30 and r["gates_failed_families"] == []
    f = score(m, uniform(2e4))["quality"]["size_fidelity"]           # every edge half of h
    assert abs(f["p99"] - np.log(2)) < 1e-3 and abs(f["bias"] + np.log(2)) < 1e-3


def test_size_fidelity_takes_h_at_the_edge_midpoint():
    """One 10 km square at 65 N; h = 10 km at the edge midpoints (5 km from the centre), 20 km at the corners."""
    L = 1e4
    sq = local_polygon(30, 65, [(-L / 2, -L / 2), (L / 2, -L / 2), (L / 2, L / 2), (-L / 2, L / 2)])
    c = local_polygon(30, 65, [(0.0, 0.0)])

    def h(x):
        d = R_EARTH_M * arc_length(x, np.broadcast_to(c, x.shape))
        return L * (1.0 + (d - L / 2) / (L / np.sqrt(2) - L / 2))

    f = score(mesh_from_polygons([sq]), h)["quality"]["size_fidelity"]
    assert f["max"] < 1e-3                                           # h at an end node would give ln 2


def test_element_ratio_across_edges_not_nodes():
    """2 x 2 cells, widths 10 / 20 km, heights 10 / 20 km, at 70 N turned by 30 deg: mean edges 10, 15, 15, 20 km.
    Edge neighbours: 1.5, 1.5, 4/3, 4/3. Node-only neighbours (diagonal) would add 2.0; sqrt(area) would give 1.41."""
    m = grid([0.0, 1e4, 3e4], [0.0, 1e4, 3e4], 100.0, 70.0, 30.0)
    r = score(m, uniform(1.5e4))
    er = r["quality"]["elem_size_ratio"]
    assert abs(er["max"] - 1.5) < 1e-3 and abs(er["median"] - (1.5 + 4 / 3) / 2) < 1e-3
    assert "elem_size_ratio_max" not in r["gates_failed"]           # 1.5 <= 2
    assert abs(r["gates_failed"]["elem_size_ratio_p99"] - er["p99"]) < 1e-12 and er["p99"] > 1.15


def test_element_ratio_p99_is_the_99th_percentile():
    """2 rows x 40 columns at 60 N, 20 columns of 10 km then 20 of 15 km (rows 10 km): 118 interior edges, 2 of them
    with ratio 12.5 / 10 = 1.25. Linear p99 = 1 + 0.83 * 0.25 = 1.2075 fails the 1.15 gate; p95 would be 1.0."""
    xs = np.r_[np.arange(21) * 1e4, 2e5 + np.arange(1, 21) * 1.5e4] - 2.5e5
    r = score(grid(xs, [0.0, 1e4, 2e4], -30.0, 60.0), uniform(1e4))
    er = r["quality"]["elem_size_ratio"]
    assert r["quality"]["n_edges_interior"] == 118
    assert abs(er["p99"] - 1.2075) < 1e-3 and abs(er["max"] - 1.25) < 1e-3 and abs(er["p95"] - 1.0) < 1e-3
    assert abs(r["gates_failed"]["elem_size_ratio_p99"] - 1.2075) < 1e-3


def test_size_fidelity_p99_is_the_99th_percentile():
    """2 rows x 30 columns of 10 km at 60 N, one column 15 km wide: 152 edges, 3 of them 15 km (|ln s/h| = ln 1.5).
    p99 sits on the 3 long edges (0.405, gate fails), p95 would be ~0."""
    xs = np.r_[np.arange(16) * 1e4, 1.65e5 + np.arange(15) * 1e4] - 1.525e5
    r = score(grid(xs, [0.0, 1e4, 2e4], 45.0, 60.0), uniform(1e4))
    f = r["quality"]["size_fidelity"]
    assert r["quality"]["n_edges"] == 152
    assert abs(f["p99"] - np.log(1.5)) < 1e-3 and abs(f["max"] - np.log(1.5)) < 1e-3 and f["median"] < 1e-3
    assert abs(r["gates_failed"]["size_fidelity_p99"] - np.log(1.5)) < 1e-3


def _planar_max_angle(xy: np.ndarray) -> float:
    a = np.roll(xy, -1, axis=0) - xy
    b = np.roll(xy, 1, axis=0) - xy
    return float(np.degrees(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(axis=1))).max())


@pytest.mark.parametrize("corner, passed", [(150.0, None), (150.1, False), (149.9, True)])
def test_quad_corner_angle_in_the_tangent_plane(corner, passed):
    """One quad with a known largest corner at 70 N, turned by 30 deg (a lon-lat angle would be off by tens of deg)."""
    s = np.radians(180.0 - corner)
    xy = np.array([(0, 0), (2e4, 0), (2e4 - 1e4 * np.cos(s), 1e4 * np.sin(s)), (1e4 * np.cos(s), 1e4 * np.sin(s))])
    a = np.radians(30.0)
    xy = xy @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    r = score(mesh_from_polygons([local_polygon(-60, 70, xy)]), uniform(1e4))
    assert abs(r["quality"]["quad_max_angle_deg"]["max"] - corner) < 1e-3
    assert abs(r["quality"]["quad_max_angle_deg_coast"]["max"] - corner) < 1e-3      # every node is a boundary node
    if passed is not None:                                           # the 150 deg gate on either side
        assert ("quad_max_angle_hard_deg" not in r["gates_failed"]) == passed


def test_quad_angle_p99_9():
    """40 x 30 cells of 1 km at 72 N, the centre node moved by (0.4, 0.3) km: 4 distorted quads out of 1200. Linear
    p99.9 lies between the 2nd and 3rd largest of them; p99 would be 90 deg."""
    xs, ys = np.arange(41) * 1e3 - 2e4, np.arange(31) * 1e3 - 1.5e4
    centre = 15 * 41 + 20
    r = score(grid(xs, ys, 5.0, 72.0, move=(centre, 400.0, 300.0)), uniform(1e3))
    X, Y = np.meshgrid(xs, ys)
    xy = np.column_stack([X.ravel(), Y.ravel()])
    xy[centre] += (400.0, 300.0)
    worst = []
    for n in (centre - 42, centre - 41, centre - 1, centre):        # lower-left node of the 4 quads at the centre
        worst.append(_planar_max_angle(xy[[n, n + 1, n + 42, n + 41]]))
    want = np.percentile(np.r_[np.full(1196, 90.0), worst], 99.9)
    q = r["quality"]["quad_max_angle_deg"]
    assert abs(q["p99_9"] - want) < 1e-2 and q["p99_9"] > 100.0 and abs(q["max"] - max(worst)) < 1e-2


def test_n3_n5_pair_across_the_antimeridian():
    """A 5-fan of quads around A with three more quads closing a 3-fan around the rim corner r0, at 180 E: interior
    n5 = 1 (A), n3 = 1 (r0); boundary: r1..r4, w, v in one element (+1 each), s0, s1 in three (-1 each): sum 4."""
    k = np.arange(5)
    s = np.column_stack([np.cos(np.radians(72 * k)), np.sin(np.radians(72 * k))])
    rr = 1.4 * np.column_stack([np.cos(np.radians(72 * k + 36)), np.sin(np.radians(72 * k + 36))])
    w, y = np.array([2.0, 0.0]), 2.4 * np.array([np.cos(np.radians(36)), np.sin(np.radians(36))])
    v = 2.0 * np.array([np.cos(np.radians(72)), np.sin(np.radians(72))])
    polys = [[(0.0, 0.0), s[i], rr[i], s[(i + 1) % 5]] for i in range(5)]
    polys += [[s[0], w, y, rr[0]], [rr[0], y, v, s[1]]]
    m = mesh_from_polygons([local_polygon(180, 10, 1e4 * np.array(p)) for p in polys])
    ic = score(m, uniform(1e4))["irregular"]
    assert (ic["n3"], ic["n5"], ic["n_other"], ic["n_interior"]) == (1, 1, 0, 2)
    assert (ic["euler_sum"], ic["chi"], ic["chi_domain"], ic["euler_ok"]) == (4, 1, 1, True)


def test_euler_sum_counts_other_valences():
    """Closed mixed mesh: 6 triangles around each pole (valence 6: 'other'), 6 quads between rings at +-30 deg.
    sum(4 - deg) = 2 * (-2), T = 12: Euler sum 8 = 4 chi. Mixed and closed: 18 applicable gates."""
    lon = np.arange(6) * 60.0
    pts = np.vstack([lonlat_to_xyz(np.array([0.0]), np.array([90.0])), lonlat_to_xyz(lon, np.full(6, 30.0)),
                     lonlat_to_xyz(lon, np.full(6, -30.0)), lonlat_to_xyz(np.array([0.0]), np.array([-90.0]))])
    i, j = np.arange(6), (np.arange(6) + 1) % 6
    tris = np.vstack([np.column_stack([np.zeros(6, int), 1 + i, 1 + j]), np.column_stack([np.full(6, 13), 7 + j, 7 + i])])
    quads = np.column_stack([1 + i, 7 + i, 7 + j, 1 + j])
    m = orient_ccw(Mesh(pts, tris, quads, np.zeros(14, np.int8)))
    r = score(m, uniform(3e6), loops=0)
    ic = r["irregular"]
    assert (ic["n3"], ic["n5"], ic["n_other"], ic["T"]) == (0, 0, 2, 12)
    assert (ic["euler_sum"], ic["chi"], ic["chi_domain"], ic["euler_ok"]) == (8, 2, 2, True)
    # 8 region gates x (whole, interior) + neighbours, elements per node, triangle clusters, Euler; no boundary
    assert r["gates_total"] == 21 and r["quality"]["tri_cluster_max"] == 6        # a polar fan of 6 triangles
    assert "tri_cluster_max" in r["gates_failed"] and "euler_ok" not in r["gates_failed"]


def test_runner_smoothing_keeps_boundary_nodes(tmp_path, monkeypatch):
    """Column "smoothed": the common smoothing runs with the boundary nodes fixed; the smoothed record reuses the
    method's generate_s and adds the smoothing time to wall_s."""
    monkeypatch.setattr(compete, "WORK_DIR", tmp_path / "work")
    r = compete.run_case("m00_background", "continent", 8, smooth=True, out=tmp_path / "c.jsonl")
    d = tmp_path / "work" / "m00_background_continent_s8"
    raw, sm = compete._load(d / "raw.npz"), compete._load(d / "smooth.npz")
    b = build_topology(raw).boundary_node
    assert b.sum() > 0 and np.array_equal(raw.points[b], sm.points[b]) and not np.array_equal(raw.points, sm.points)
    s = r["after_smoothing"]
    assert s["generate_s"] == r["generate_s"] == r["wall_s"] and s["wall_s"] > r["wall_s"]


# ------------------------------------------------------------------------------------- later gates

def _strip_with_fan(n_tri: int, lat0: float = 65.0) -> Mesh:
    """4 quads of 10 km in a row, then n_tri = 2 (a square cut along its diagonal) or 3 (a pentagon fanned from one
    corner) edge-connected triangles at the east end."""
    L = 1e4
    polys = [[(i * L, 0), ((i + 1) * L, 0), ((i + 1) * L, L), (i * L, L)] for i in range(4)]
    x = 4 * L
    if n_tri == 3:
        polys += [[(x, 0), (x + L, 0), (x + 1.5 * L, 0.5 * L)], [(x, 0), (x + 1.5 * L, 0.5 * L), (x + L, L)],
                  [(x, 0), (x + L, L), (x, L)]]
    else:
        polys += [[(x, 0), (x + L, 0), (x + L, L)], [(x, 0), (x + L, L), (x, L)]]
    return mesh_from_polygons([local_polygon(20.0, lat0, np.array(p, float)) for p in polys])


@pytest.mark.parametrize("n_tri, fails", [(3, True), (2, False)])
def test_triangle_cluster_gate(n_tri, fails):
    """No edge-connected cluster of >= 3 triangles outside triangle zones (gate tri_cluster_max = 2)."""
    m = _strip_with_fan(n_tri)
    r = score(m, uniform(1e4))
    assert r["quality"]["tri_cluster_max"] == n_tri and ("tri_cluster_max" in r["gates_failed"]) == fails
    m.tri_zone = np.ones(m.n_tris, dtype=bool)                       # declared triangle zone: not a defect
    r = score(m, uniform(1e4))
    assert r["quality"]["tri_cluster_max"] == 0 and r["quality"]["n_tri_zone"] == n_tri
    assert "tri_cluster_max" not in r["gates_failed"]


def test_triangle_cluster_gate_not_applicable_without_quads():
    m = split_quads(structured_patch(4, 3, 1e4, 0.0, 60.0), range(12))
    r = score(m, uniform(1e4))
    assert r["quality"]["tri_cluster_max"] is None
    assert not any(x.startswith("tri_cluster") for x in r["gates_failed"])
    gates = compete.gate_results(r["quality"], r["irregular"], GATES)
    assert "tri_cluster_max" not in {x["gate"] for x in gates}


@pytest.mark.parametrize("rule, fails", [("all", True), ("tri", False), ("off", None)])
def test_boundary_rule_on_staircase_quads(rule, fails):
    """A quad patch: the 4 corner quads have 2 boundary edges each (a staircase corner) and keep 2 neighbours.
    "all" (the strict form) fails them, "tri" (FESOM2: triangles only, the default) passes, "off" has no gate."""
    m = grid(np.arange(6) * 1e4, np.arange(5) * 1e4, 15.0, 62.0)
    r = score(m, uniform(1e4), gates={**GATES, "boundary_rule": rule})
    assert r["quality"]["n_elems_ge2_boundary_edges"] == 4 and r["quality"]["boundary"]["edges_per_elem_max"] == 2
    assert r["quality"]["elem_neighbours_min"] == 2 and "elem_neighbours_min" not in r["gates_failed"]
    names = {x["gate"] for x in compete.gate_results(r["quality"], r["irregular"], {**GATES, "boundary_rule": rule})}
    assert ("boundary_edges_max" in names) == (rule == "all")      # "tri": no triangle in the mesh, N/A
    if fails is not None:
        assert ("boundary_edges_max" in r["gates_failed"]) == fails
    assert GATES["boundary_rule"] == "tri"


def test_boundary_rule_tri_matches_the_neighbour_gate_for_triangles():
    """Under "tri" a triangle with 2 boundary edges fails, as elem_neighbours_min = 2 does (it has 1 neighbour): the
    end triangle (x, 0), (x + L, 0), (x + L, L) of the strip."""
    r = score(_strip_with_fan(2), uniform(1e4))
    q = r["quality"]
    assert q["n_tris_ge2_boundary_edges"] == 1 and q["boundary"]["edges_per_tri_max"] == 2
    assert "boundary_edges_max" in r["gates_failed"] and "elem_neighbours_min" in r["gates_failed"]


def test_euler_gate():
    """euler_ok is a gate: the same patch scored as if the domain had 2 coast loops (4 chi = 0, sum 4) fails it."""
    m = grid(np.arange(9) * 1e4, np.arange(7) * 1e4, 10.0, 60.0)
    assert "euler_ok" not in score(m, uniform(1e4))["gates_failed"]
    r = score(m, uniform(1e4), loops=2)
    assert r["gates_failed"]["euler_ok"] is False and "euler_ok" in r["gates_failed_families"]


def test_pinch_gate():
    a = local_polygon(0, 50, [(0, 0), (1e4, 0), (1e4, 1e4), (0, 1e4)])
    b = local_polygon(0, 50, [(1e4, 1e4), (2e4, 1e4), (2e4, 2e4), (1e4, 2e4)])    # touches a at one corner
    r = score(mesh_from_polygons([a, b]), uniform(1e4))
    assert r["gates_failed"]["pinch_nodes_max"] == 1


def test_coast_size_set_leaves_out_the_coast_edges():
    """h = 20 km at the midpoints of the boundary edges (the resampled coast), 10 km elsewhere on a 10 km patch:
    the coast subsets see only the fill rows (|ln s/h| = 0), the coast edges are reported apart (ln 2)."""
    m = grid(np.arange(9) * 1e4, np.arange(7) * 1e4, -40.0, 70.0)
    topo = build_topology(m)
    b = topo.edges[topo.edge_elems[:, 1] < 0]
    mid = m.points[b[:, 0]] + m.points[b[:, 1]]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)

    def h(x):
        d = R_EARTH_M * arc_length(x[:, None, :], mid[None, :, :]).min(axis=1)
        return np.where(d < 1e3, 2e4, 1e4)

    r = score(m, h)
    q = r["quality"]
    assert q["size_fidelity_coast"]["max"] < 1e-3 and q["size_fidelity_coast3"]["max"] < 1e-3
    assert abs(q["size_fidelity_boundary"]["median"] - np.log(2)) < 1e-3 and q["size_fidelity"]["max"] > 0.69
    assert "size_fidelity_p99_coast" not in r["gates_failed"] and "size_fidelity_p99" in r["gates_failed"]


@pytest.mark.parametrize("node, in_coast3", [((2, 6), True), ((6, 6), False)])
def test_first_three_coastal_rows(node, in_coast3):
    """12 x 12 cells of 10 km; one node moved by 3 km. At 2 edge hops from the boundary its 4 quads are rows 2-3
    (in _coast3, not in _coast); at 6 hops they are rows 6-7 (interior only)."""
    i, j = node
    m = grid(np.arange(13) * 1e4, np.arange(13) * 1e4, 120.0, 61.0, move=(j * 13 + i, 3e3, 2e3))
    q = score(m, uniform(1e4))["quality"]
    assert q["quad_max_angle_deg_coast"]["max"] < 90.1 and q["quad_max_angle_deg_interior"]["max"] > 100   # 120 km
    # patch: the inverse gnomonic map bends its corners by 0.02 deg
    assert (q["quad_max_angle_deg_coast3"]["max"] > 100) == in_coast3 and q["quad_max_angle_deg_coast3"]["max"] > 89


def test_strait_subset():
    """The continent's strait corridor holds the channel, not the open sea or the island gap; on a mesh, the
    `_strait` statistics cover the elements whose centroid lies in the corridor."""
    st = STRAITS["continent"]
    from meshom.domain import continent_frame
    c, t, n = continent_frame()
    off = lambda a: (c * np.cos(a) + n * np.sin(a))[None, :]          # noqa: E731
    assert st(c[None, :])[0] and st(off(70e3 / R_EARTH_M))[0] and not st(off(80e3 / R_EARTH_M))[0]
    far = np.cos(0.6) * c + np.sin(0.6) * t                            # 34 deg along the channel: open sea
    assert not st(far[None, :])[0] and "continent" in STRAITS and set(STRAITS) <= set(CASES)
    m = grid(np.arange(13) * 1e4, np.arange(13) * 1e4, 120.0, 61.0, move=(6 * 13 + 2, 3e3, 2e3))
    lon = np.degrees(np.arctan2(m.points[:, 1], m.points[:, 0]))
    lon_mid = 0.5 * (lon.min() + lon.max())                            # the moved node lies in the western half

    def west(x):
        return np.degrees(np.arctan2(x[:, 1], x[:, 0])) < lon_mid

    q = score(m, uniform(1e4), strait=west)["quality"]
    assert q["quad_max_angle_deg_strait"]["max"] > 100 and q["size_fidelity_strait"]["max"] > 0.1
    q = score(m, uniform(1e4), strait=lambda x: ~west(x))["quality"]
    assert q["quad_max_angle_deg_strait"]["max"] < 90.1 and q["size_fidelity_strait"]["max"] < 1e-3
    q = score(m, uniform(1e4))["quality"]
    assert q["quad_max_angle_deg_strait"]["max"] is None               # no corridor: N/A


def test_failed_families():
    assert compete.failed_families({"size_fidelity_p99": 0.2, "size_fidelity_p99_coast": 0.3,
                                    "size_fidelity_p99_coast3": 0.3, "elem_size_ratio_p99_strait": 1.2,
                                    "euler_ok": False}) == ["elem_size_ratio_p99", "euler_ok", "size_fidelity_p99"]


def test_peak_rss_is_the_childs_own():
    """A runner holding 1.5 GB starts a child whose own peak is ~0.1 GB: the child's stage record says ~0.1 GB
    (ru_maxrss would report the inherited 1.5 GB on Linux)."""
    if not sys.platform.startswith("linux"):
        pytest.skip("the inherited ru_maxrss is a Linux effect")
    hold = np.ones(int(1.5 * 2**30 / 8))                              # touched: resident
    code = ("import json; from meshom import bench; rec = []\n"
            "with bench.stage('x', rec): pass\n"
            "print(json.dumps(rec[0]['peak_rss_mb']))")
    env = dict(os.environ, PYTHONPATH=str(Path(compete.__file__).resolve().parents[1]))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    peak = json.loads(out.stdout.strip().splitlines()[-1])
    assert hold.sum() > 0 and 10.0 < peak < 400.0


def test_method_diagnostics_run_outside_the_timed_stage(tmp_path, monkeypatch):
    """m14's own measurement runs in the runner's "diagnostics" stage after the timed generate stage:
    the record has diagnostics_s, m14.json still gets the measurement, the generate time excludes it."""
    monkeypatch.setattr(compete, "WORK_DIR", tmp_path / "work")
    r = compete.run_case("m14_blocks", "uniform", 4, out=tmp_path / "c.jsonl")
    d = tmp_path / "work" / "m14_blocks_uniform_s4"
    stages = [x["stage"] for x in json.loads((d / "stages.json").read_text())]
    info = json.loads((d / "m14.json").read_text())
    assert stages == ["generate", "diagnostics"] and r["diagnostics_s"] > 0 and r["status"] == "ok"
    assert "seam_band" in info and info["timings"]["measure_s"] > 0 and not compete.DEFERRED
    assert info["timings"]["total_s"] <= r["generate_s"]


# ------------------------------------------------------------------------- stated rules

def test_triangle_cluster_is_edge_connected():
    """The gate counts edge-connected clusters: two split quads that touch at one corner node are 2 clusters of 2
    (the gate passes); a node-connected count would see one cluster of 4."""
    nx = 8
    m = split_quads(grid(np.arange(nx + 1) * 1e4, np.arange(7) * 1e4, 30.0, 63.0), [2 * nx + 2, 3 * nx + 3])
    r = score(m, uniform(1e4))
    assert r["quality"]["tri_cluster_max"] == 2 and r["quality"]["tri_clusters"]["n"] == 2
    assert "tri_cluster_max" not in r["gates_failed"]


def test_triangle_zone_splits_a_cluster():
    """Three split quads in a row are one edge-connected chain of 6 triangles (fails); declaring the middle quad's
    two triangles a zone leaves two free clusters of 2 (passes): zone triangles do not link free ones."""
    nx = 8
    m = split_quads(grid(np.arange(nx + 1) * 1e4, np.arange(7) * 1e4, -60.0, 64.0), [2 * nx + 2, 2 * nx + 3,
                                                                                       2 * nx + 4])
    r = score(m, uniform(1e4))
    assert r["quality"]["tri_cluster_max"] == 6 and "tri_cluster_max" in r["gates_failed"]
    m.tri_zone = np.zeros(m.n_tris, dtype=bool)
    m.tri_zone[[1, 4]] = True                                          # split_quads: A triangles first, then B
    r = score(m, uniform(1e4))
    assert r["quality"]["tri_cluster_max"] == 2 and "tri_cluster_max" not in r["gates_failed"]


def test_tri_zone_kept_by_save_orient_and_runner_smoothing(tmp_path, monkeypatch):
    """Mesh.tri_zone survives the npz round trip (raw.npz and smooth.npz), orient_ccw, and the runner's smoothing
    (compete._generate with smooth=True); a mesh without a zone loads back with tri_zone None."""
    from meshom.methods import METHODS

    nx = 8
    m = split_quads(grid(np.arange(nx + 1) * 1e4, np.arange(7) * 1e4, 100.0, 62.0, move=(3 * (nx + 1) + 5, 3e3, 2e3)),
                    [2 * nx + 2, 2 * nx + 3, 4 * nx + 6])
    zone = np.array([True, False, True, False, True, False])
    m.tri_zone = zone.copy()
    compete._save(m, tmp_path / "z.npz")
    assert np.array_equal(compete._load(tmp_path / "z.npz").tri_zone, zone)
    flipped = Mesh(m.points, m.tris[:, ::-1].copy(), m.quads[:, ::-1].copy(), m.node_flag, m.tri_zone)
    o = orient_ccw(flipped)
    assert not np.array_equal(o.tris, flipped.tris) and np.array_equal(o.tri_zone, zone)
    compete._save(Mesh(m.points, m.tris, m.quads, m.node_flag), tmp_path / "n.npz")
    assert compete._load(tmp_path / "n.npz").tri_zone is None

    monkeypatch.setitem(METHODS, "_zone_probe", lambda domain, h, opts: m)
    monkeypatch.setenv("MESHOM_RUN_DIR", str(tmp_path))               # _generate sets it; restored after the test
    d = tmp_path / "run"
    d.mkdir()
    compete._generate("_zone_probe", Domain([np.zeros((0, 3))]), uniform(1e4), {}, True, d)
    raw, sm = compete._load(d / "raw.npz"), compete._load(d / "smooth.npz")
    assert not np.array_equal(raw.points, sm.points)                  # the smoothing ran
    assert np.array_equal(raw.tri_zone, zone) and np.array_equal(sm.tri_zone, zone)
    assert score(sm, uniform(1e4))["quality"]["n_tri_zone"] == 3


@pytest.mark.parametrize("node, in_coast3", [((3, 6), True), ((4, 6), False)])
def test_third_coastal_row_is_the_last(node, in_coast3):
    """The edge of the `_coast3` subset: a node 3 edge hops from the boundary distorts quads of rows 3 and 4 (row 3
    is in _coast3); a node 4 hops away distorts rows 4 and 5 only."""
    i, j = node
    m = grid(np.arange(13) * 1e4, np.arange(13) * 1e4, -150.0, 66.0, move=(j * 13 + i, 3e3, 2e3))
    q = score(m, uniform(1e4))["quality"]
    assert (q["quad_max_angle_deg_coast3"]["max"] > 100) == in_coast3 and q["quad_max_angle_deg_coast"]["max"] < 90.1


def test_tri_share_gate(tmp_path):
    """The allowed triangle level is the gate tri_share_max; triangle-only meshes are N/A."""
    from meshom.quality.metrics import check_gates
    g = {"tri_share_max": 0.35}
    assert check_gates({"tri_share": 0.30}, g)[0]["passed"]
    assert not check_gates({"tri_share": 0.40}, g)[0]["passed"]
    assert check_gates({"tri_share": None}, g)[0]["passed"]           # triangle-only: not applicable
    assert check_gates({"tri_share": 0.40}, {"tri_share_max": 0.5})[0]["passed"]


def test_euler_check_with_two_water_components():
    """Two disjoint quad patches (a closed sea kept by the land cut): chi_domain = 2 K - loops, the check holds."""
    import numpy as np
    from meshom.compete import irregular_counts, _score
    from meshom.config import DEFAULT_GATES
    from meshom.domain import Domain
    from meshom.geo import R_EARTH_M
    from meshom.mesh import Mesh
    from meshom.size import uniform

    def patch(lon0, n):
        lon = np.deg2rad(lon0 + np.arange(n + 1)); lat = np.deg2rad(np.arange(n + 1))
        L, B = np.meshgrid(lon, lat, indexing="ij")
        pts = np.stack([np.cos(B) * np.cos(L), np.cos(B) * np.sin(L), np.sin(B)], axis=-1).reshape(-1, 3) * R_EARTH_M
        idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
        q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)
        return pts, q
    p1, q1 = patch(0.0, 5); p2, q2 = patch(40.0, 3)
    pts = np.vstack([p1, p2]); quads = np.vstack([q1, q2 + len(p1)])
    flag = np.zeros(len(pts), dtype=np.int8)
    m = Mesh(pts, np.zeros((0, 3), dtype=np.int64), quads, flag)
    ic = irregular_counts(m)
    assert ic["components"] == 2 and ic["euler_ok"] and ic["euler_sum"] == 4 * ic["chi"]
    from meshom.methods.m14.cut import boundary_loops
    dom = Domain([m.points[lp] for lp in boundary_loops(m)], "two", [True, True])
    r = _score(m, dom, uniform(1.1e5), {"curv": 0.0, "lap_plus": 0.0, "lap_minus": 0.0}, dict(DEFAULT_GATES))
    assert r["irregular"]["chi_domain"] == 2 * 2 - 2 and r["irregular"]["euler_ok"]
