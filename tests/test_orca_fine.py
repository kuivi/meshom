"""m16_orca's refined part around the fold (B108), the ORCA frame of the fill (B109) and the blend by a factor (B110),
on ORCA2 and its own map refined 3 times (a test preset, as ORCA12 is ORCA025 x 3): every third line of the fine
grid is a line of the coarse one, the fine part stays unchanged and joins the fill conformingly on its one edge loop,
the fill's quads follow ORCA's directions."""
import numpy as np
import pytest

from meshom.domain import Domain
from meshom.geo import lonlat_to_xyz


@pytest.fixture(scope="module")
def presets():
    from dataclasses import replace

    from meshom.orca.presets import PRESETS

    PRESETS["ORCA2X3"] = replace(PRESETS["ORCA2"], name="ORCA2X3", q=3, pole_row=3 * 147, rows=0, ref=None,
                                 verified=False)
    yield PRESETS
    PRESETS.pop("ORCA2X3", None)


@pytest.fixture(scope="module")
def fine_mesh(presets):
    from meshom.methods.m16_orca import _STATE, run
    from meshom.orca.sizes import orca_size
    from meshom.size import blend_zone, layered

    lo = np.arange(-180.0, 180.0, 1.0)
    h = layered([orca_size("ORCA2", 1000.0),
                 blend_zone(lonlat_to_xyz(lo, np.full(len(lo), 62.0)), lonlat_to_xyz(lo, np.full(len(lo), 40.0)),
                            None, 1.0 / 3.0)], np.inf)
    mesh, info = run(Domain([], "sphere"), h,
                     {"orca": "ORCA2", "lat_north": 40.0, "fine_orca": "ORCA2X3", "fine_lat": 62.0, "mask": "all",
                      "cap_method": "m02_linf_frontal", "cap_opts": {"frame": "ORCA2"}, "size_tolerance": None})
    return mesh, info, h, _STATE["belt"]


def test_nesting_every_third_line(presets):
    """B108: ORCA2's map on the 1/3 lattice: F_f(3 r + b_j, 3 c + b_i) = F_c(r, c) (the same map, so to rounding)."""
    from meshom.orca import build
    from meshom.orca.mesh import nesting

    n = nesting(build("ORCA2"), build("ORCA2X3"))
    assert n["k"] == 3 and n["max_m"] < 1e-3 and n["points"] > 10_000, n


def test_fine_part_unchanged_and_joined(fine_mesh):
    """B108: the fine part's cells north of fine_lat per column from the fold, kept as they are (0 m, every element);
    both edges of the fill conforming, every loop node an ORCA node; the mesh closed except ORCA2's south row."""
    from meshom.compete import irregular_counts
    from meshom.geo import xyz_to_lonlat
    from meshom.mesh import signed_area
    from meshom.methods.m16_orca import belt_unchanged

    mesh, info, _, (pts, quads, tris) = fine_mesh
    f = info["fine"]
    assert f["nesting"]["k"] == 3 and f["orca_quads"] > 0
    j = info["joints"]["north"]
    assert j["conforming"] and j["matched"] == j["row_nodes"] == j["coarse_edge_nodes"] + j["fine_edge_nodes"]
    assert j["boundary_edges_after"] == info["cols"] + sum(f["holes"])   # ORCA2's south row, the grid poles' holes
    assert len(f["holes"]) <= 2 and max(f["holes"], default=0) < 200      # the pole line's dropped slivers
    u = belt_unchanged(mesh, pts, quads, tris)
    assert u["max_displacement_m"] == 0.0 and u["belt_elems_kept"] == u["belt_elems_total"]
    fine_pts = pts[info["coarse_nodes"]:]
    _, lat = xyz_to_lonlat(fine_pts)
    assert lat.min() > 61.0                        # the fine part lies north of its line (its edge half a cell south)
    assert (signed_area(mesh) > 0).all() and irregular_counts(mesh)["euler_ok"]


def test_fill_follows_orca_directions(fine_mesh):
    """B109: the frame at a T point is ORCA's i-direction; the fill's quad edges follow the frame (mod 90 deg); B110:
    the fill's h is the coarse cell size at the coarse edge and a third of it inside the fine line."""
    from meshom.geo import tangent_basis
    from meshom.methods.m16_orca import _ruled
    from meshom.orca.frame import orca_frame
    from meshom.orca.sizes import orca_size, preset_cell_sizes

    fr = orca_frame("ORCA2")
    t, _ = preset_cell_sizes("ORCA2")
    sel = np.arange(0, len(t), 97)
    c = np.einsum("ij,ij->i", fr(t[sel]), fr.d[sel])
    assert np.allclose(np.abs(c) * np.sqrt(np.maximum(1.0 - c * c, 0.0)), 0.0, atol=1e-6)   # i or j (a cross)
    mesh, info, h, (pts, quads, tris) = fine_mesh
    cap = ~_ruled(mesh, pts, quads, tris)[mesh.n_tris:]
    q = mesh.quads[cap]
    a, b = mesh.points[q].reshape(-1, 3), mesh.points[np.roll(q, -1, axis=1)].reshape(-1, 3)
    m = (a + b) / np.linalg.norm(a + b, axis=1, keepdims=True)
    e1, e2 = tangent_basis(m)
    d = b - a
    ang_e = np.arctan2(np.einsum("ij,ij->i", d, e2), np.einsum("ij,ij->i", d, e1))
    u = fr(m)
    ang_u = np.arctan2(np.einsum("ij,ij->i", u, e2), np.einsum("ij,ij->i", u, e1))
    dev = np.degrees(np.abs(np.angle(np.exp(4j * (ang_e - ang_u))) / 4))
    assert np.median(dev) < 5.0 and np.percentile(dev, 90) < 15.0, (np.median(dev), np.percentile(dev, 90))
    hc = orca_size("ORCA2", 1000.0)
    x = lonlat_to_xyz(np.array([-30.0, -30.0]), np.array([30.0, 80.0]))
    assert np.allclose(h(x), hc(x) * np.array([1.0, 1.0 / 3.0]))


def test_one_front_from_the_seed_loops():
    """B112: with seed_loops only those loops' nodes grow fronts; the other loops' nodes stay points the front stops
    at; the loop nodes come first in loop order, as without the option, and the mesh is valid."""
    from meshom.compete import irregular_counts
    from meshom.domain import resample_loops
    from meshom.mesh import signed_area
    from meshom.methods.m02_linf_frontal import points, triangle_mesh
    from meshom.size import Uniform

    lo = np.linspace(-180.0, 180.0, 120, endpoint=False)
    south = lonlat_to_xyz(lo, np.full(len(lo), 20.0))                        # the band on the left of both loops
    north = lonlat_to_xyz(lo[::-1], np.full(len(lo), 60.0))
    dom = Domain([south, north], "band", [True, True])
    h = Uniform(300e3)
    nb = sum(len(lp) for lp in resample_loops(dom, h))
    p_all, nb_all, _ = points(dom, h, {})
    p_one, nb_one, info = points(dom, h, {"seed_loops": [1]})
    assert nb_all == nb_one == nb and info["seed_loops"] == [1]
    assert np.array_equal(p_one[:nb], np.vstack(resample_loops(dom, h)))
    assert len(p_one) != len(p_all) or not np.allclose(p_one, p_all)
    mesh, _ = triangle_mesh(dom, h, {"seed_loops": [1]})
    assert (signed_area(mesh) > 0).all() and irregular_counts(mesh)["euler_ok"]


@pytest.fixture(scope="module")
def ladder_mesh(presets):
    from meshom.methods.m16_orca import _STATE, run
    from meshom.size import Uniform

    mesh, info = run(Domain([], "sphere"), Uniform(1e5),
                     {"orca": "ORCA2", "fine_orca": "ORCA2X3", "transition": "ladder", "ladder_lat": [40.0, 62.0],
                      "mask": "all", "size_tolerance": None})
    return mesh, info, _STATE["belt"]


def test_ladder_conforming_and_edges_exact(ladder_mesh):
    """B114: the ladder joins the two ORCA parts node for node: its boundary nodes are exactly the coarse F row R_c
    and the fine F row R_f (all matched at 0 m), both parts unchanged, the mesh valid and closed but for ORCA2's south
    row and the grid poles' holes."""
    from meshom.compete import irregular_counts
    from meshom.mesh import signed_area
    from meshom.methods.m16_orca import belt_unchanged
    from meshom.orca import build

    mesh, info, (pts, quads, tris) = ladder_mesh
    lad, j = info["ladder"], info["joints"]["north"]
    nc, nf = info["cols"], info["fine"]["cols"]
    assert j["conforming"] and j["matched"] == j["cap_boundary_nodes"] == nc + nf and j["max_match_rad"] == 0.0
    assert j["coarse_edge_nodes"] == nc and j["fine_edge_nodes"] == nf
    gc, gf = build("ORCA2"), build("ORCA2X3")
    edge = np.vstack([lonlat_to_xyz(gc.lon["f"][lad["R_coarse"], :-2], gc.lat["f"][lad["R_coarse"], :-2]),
                      lonlat_to_xyz(gf.lon["f"][lad["R_fine"], :-2], gf.lat["f"][lad["R_fine"], :-2])])
    from scipy.spatial import cKDTree
    d, _ = cKDTree(mesh.points).query(edge)
    assert d.max() < 1e-12                                   # every edge F point is a node of the mesh
    assert info["t_rows"][1] == lad["R_coarse"] and info["fine"]["first_t_rows"] == [lad["R_fine"] + 1] * 2
    u = belt_unchanged(mesh, pts, quads, tris)
    assert u["max_displacement_m"] == 0.0 and u["belt_elems_kept"] == u["belt_elems_total"]
    assert j["boundary_edges_after"] == nc + sum(info["fine"]["holes"])
    assert (signed_area(mesh) > 0).all() and irregular_counts(mesh)["euler_ok"]


def test_ladder_insertions_equal_column_difference(ladder_mesh):
    """B115: 2 Y-splits per coarse column: as many triangles and valence-5 nodes as columns inserted (nf - nc), no
    valence-3 node, every other strip node regular; the rows grade as B110 (first row near 3 fine rows, last near 1)."""
    from meshom.compete import irregular_counts
    from meshom.methods.m16_orca import _ruled
    from meshom.orca.ladder import rows

    mesh, info, (pts, quads, tris) = ladder_mesh
    lad = info["ladder"]
    assert lad["insertions"] == lad["tris"] == lad["column_difference"] == info["fine"]["cols"] - info["cols"]
    assert mesh.n_tris - len(tris) == lad["tris"]
    ic = irregular_counts(mesh)
    assert ic["n5"] == lad["insertions"] and ic["n3"] == 0 and ic["n_other"] == 0
    assert not _ruled(mesh, pts, quads, tris)[:mesh.n_tris][-lad["tris"]:].any()   # the triangles are the strip's
    y = rows(0.0, 120.0)
    h, d = np.diff(y), 0.5 * (y[1:] + y[:-1])
    assert abs(h[0] - 3.0) < 0.1 and abs(h[-1] - 1.0) < 0.05 and np.all(np.diff(h) < 0)
    expect = 3.0 * 3.0 ** (-d / 120.0)                          # B110's law: ln h linear in the distance
    assert np.abs(np.log(h / expect)).max() < 0.01


def test_ladder_quality_in_index_space():
    """B115-B116: in index space every quad's largest angle is 90 + theta at most (the arm's tilt), the triangles'
    smallest 2 theta (less where an arm's full offset is under h tan theta), no quad wider than 1.55 x its height
    (before the relaxation), and neighbouring columns never split on the same row."""
    from meshom.orca.ladder import THETA_DEG, index_quality, rows, topology

    t = topology(1440, rows(0.0, 119.0))
    q = index_quality(t)
    assert q["quad_max"] <= 90.0 + THETA_DEG + 1e-6 and q["tri_min"] >= 2 * THETA_DEG - 1.0   # capped arms: 35.7
    assert q["aspect_max"] < 1.55 and q["tris"] == 2 * 1440
    assert np.all(np.abs(np.diff(np.r_[t.k1, t.k1[0]])) >= 2) and np.all(np.abs(np.diff(np.r_[t.k2, t.k2[0]])) >= 2)


def test_ladder_relaxation_spreads_the_width_jumps():
    """B115: the relaxation keeps the rows (y), the edge rows and the topology, and lowers the element size ratio
    across inner edges (a split's width jump spread over its neighbours) at a small cost in angle."""
    from dataclasses import replace

    from meshom.orca.ladder import RELAX, index_quality, relax, rows, topology

    t = topology(1440, rows(0.0, 119.0))
    x = relax(t, *RELAX)
    n = len(t.y_rows) - 1
    edge = (t.row == 0) | (t.row == n)
    assert np.array_equal(x[edge], t.x[edge])
    q0, q1 = index_quality(t), index_quality(replace(t, x=x))
    assert q1["ratio_p99"] < 1.12 < q0["ratio_p99"]
    assert q1["quad_max"] < 112.0 and q1["tri_min"] > 30.0 and q1["aspect_max"] < 1.5
