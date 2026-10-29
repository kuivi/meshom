"""m16_orca's ORCA part bounded by a line (B100) and the size blend between two edges (S37): the cut gives one
staircase loop without pinches, joined conformingly to the cap; the ORCA part stays unchanged through the land cut;
the blend is h_m inside, the edge's running h at the outer edge, monotone between."""
import numpy as np
import pytest

from meshom.domain import Domain
from meshom.mesh import signed_area
from meshom.size import Uniform
from tests.helpers import land_loop

# south of 40 N, plus an Atlantic tongue to 60 N with a step (the cut is per column, contiguous from the south)
REGION = [[-181, -91], [-181, 40], [-60, 40], [-60, 60], [-30, 60], [-30, 50], [0, 50], [0, 40], [181, 40],
          [181, -91]]


@pytest.fixture(scope="module")
def region_mesh():
    from meshom.methods.m16_orca import run

    mesh, info = run(Domain([], "sphere"), Uniform(250e3),
                     {"orca": "ORCA2", "region": REGION, "mask": "all", "cap_method": "m15_tri",
                      "size_tolerance": None})
    return mesh, info


def test_region_one_loop_no_pinch(region_mesh):
    """B100: the ORCA part's cells per column contiguous from the south up to the line: its boundary is the south
    row and one staircase loop, every node on it once (no pinch); the cap matches every loop node."""
    from meshom.compete import irregular_counts
    from meshom.geo import xyz_to_lonlat
    from meshom.methods.m14.cut import boundary_loops
    from meshom.methods.m16_orca import _STATE
    from meshom.mesh import Mesh

    mesh, info = region_mesh
    pts, quads, tris = _STATE["belt"]
    belt = Mesh(pts, tris, quads, np.zeros(len(pts), np.int8))
    loops = boundary_loops(belt)
    assert len(loops) == 2
    north = max(loops, key=lambda lp: pts[lp, 2].mean())
    assert len(np.unique(north)) == len(north)
    lon, lat = xyz_to_lonlat(pts[north])
    assert lat.max() > 55 and lat[(lon > 30) & (lon < 170)].max() < 42      # the tongue in, 40 N elsewhere
    j = info["joints"]["north"]
    assert j["conforming"] and j["matched"] == j["row_nodes"] == len(north)
    assert j["boundary_edges_after"] == info["cols"]                       # only the grid's south row is open
    assert (signed_area(mesh) > 0).all() and irregular_counts(mesh)["euler_ok"]


def test_region_orca_unchanged_after_cut(region_mesh):
    """B100 with B85: the ORCA part bounded by the line keeps its nodes and elements through the run's land cut."""
    from meshom.methods.m16_orca import _STATE, belt_unchanged, cut_wet
    from meshom.run import cut_land

    mesh, info = region_mesh
    pts, quads, tris = _STATE["belt"]
    land = Domain([land_loop(-40.0, 55.0, [[-600, -600], [600, -600], [600, 600], [-600, 600]]),
                   land_loop(90.0, 70.0, [[-800, -800], [800, -800], [800, 800], [-800, 800]])], "land")
    cut, cinfo = cut_land(mesh, land, wet_rule=cut_wet)
    assert cinfo["n_ruled_removed"] == 0
    u = belt_unchanged(cut, pts, quads, tris)
    assert u["max_displacement_m"] == 0.0 and u["nodes_within_1km_not_on_belt"] == 0
    assert u["belt_elems_kept"] == u["belt_elems_total"]
    # the cap's triangles in the staircase's steps have all three nodes on the loop: counted as "altered" by
    # belt_unchanged, before the cut as after; no ORCA element changed
    assert u["belt_elems_altered"] <= belt_unchanged(mesh, pts, quads, tris)["belt_elems_altered"]
    assert cut.n_elems < mesh.n_elems


def test_blend_between_two_edges():
    """S37: h_m inside the polygon, the running h of the outer edge at the edge and the running h beyond, ln h
    rising monotonically between."""
    from meshom.geo import lonlat_to_xyz
    from meshom.size import Latitude, blend_zone, layered

    lo = np.arange(-180.0, 180.0, 2.0)
    inner = lonlat_to_xyz(lo, np.full(len(lo), 75.0))
    outer = lonlat_to_xyz(lo, np.full(len(lo), 55.0))
    h = layered([Latitude(30e3, 1e3), blend_zone(inner, outer, 4e3)], 40e3)
    la = np.linspace(40.0, 89.0, 200)
    v = h(lonlat_to_xyz(np.full(len(la), 10.0), la))
    run = 30e3 * np.cos(np.radians(la))
    assert np.allclose(v[la >= 75.2], 4e3)
    assert np.allclose(v[la < 54.8], run[la < 54.8])
    mid = (la > 55.2) & (la < 74.8)
    assert (np.diff(v[mid]) < 0).all()
    e = h(lonlat_to_xyz(np.array([10.0, 10.0]), np.array([55.0 + 1e-3, 55.0 - 1e-3])))
    assert abs(e[0] / e[1] - 1) < 0.01                                     # continuous at the outer edge
