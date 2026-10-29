"""m14: curvilinear blocks placed from h, stitched by m02; valid meshes on `uniform` and `continent` at scale 4, blocks
cover more than half of the uniform sphere and hold no irregular node. Variant _cut (coast treatment A): with no land
it is the conforming mesh exactly; on `continent` the cut mesh is manifold, CCW, Euler-consistent, its new boundary
first with flag 1 and within 1 h of the true coast."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from meshom.cases import CASES
from meshom.compete import irregular_counts
from meshom.domain import resample_loops
from meshom.geo import R_EARTH_M, arc_length
from meshom.mesh import build_topology, signed_area
from meshom.methods import METHODS
from meshom.methods.m14_blocks import run
from meshom.quality.metrics import compute_quality
from meshom.size import scaled


def _check_valid(m, dom, h):
    topo = build_topology(m)
    assert (signed_area(m) > 0).all()                                     # CCW, no inverted element
    assert (topo.elems_per_node > 0).all()                                # every node used
    ic = irregular_counts(m)
    assert ic["euler_ok"] and ic["euler_sum"] == 4 * (2 - len(dom.loops))   # edge-manifold, no hole, right topology
    s = compute_quality(m).summary
    assert s["n_pinch_nodes"] == 0 and s["quad_max_angle_deg"]["max"] <= 150.0
    loops = resample_loops(dom, h)
    nb = sum(len(lp) for lp in loops)
    assert np.array_equal(m.points[:nb], np.vstack(loops) if loops else m.points[:0])   # coast nodes kept, first
    assert topo.boundary_node.sum() == nb and topo.boundary_node[:nb].all()


@pytest.mark.parametrize("case", ["uniform", "continent"])
def test_valid_mesh(case):
    dom, h0 = CASES[case]
    h = scaled(h0, 4.0)
    m, info = run(dom, h, {})
    _check_valid(m, dom, h)
    assert info["n_blocks"] >= 1 and info["irregular_in_blocks"] == 0
    if case == "uniform":
        assert info["coverage_area"] > 0.5                                # blocks cover more than half of the sphere
    for k in info["blocks_kept"]:                                         # outlines in lon-lat for the UI (panel 4)
        b = info["blocks"][k]
        assert len(b["outline_lonlat"]) == b["outline_nodes"] and len(b["outline_corners"]) >= 4


def test_registered():
    assert {"m14_blocks", "m14_blocks_igm", "m14_blocks_tri", "m14_blocks_cut"} <= set(METHODS)


def test_cut_without_land_is_the_conforming_mesh():
    dom, h0 = CASES["uniform"]
    h = scaled(h0, 4.0)
    a, b = METHODS["m14_blocks_cut"](dom, h, {}), METHODS["m14_blocks"](dom, h, {})
    for k in ("points", "tris", "quads", "node_flag"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k


def test_cut_continent():
    dom, h0 = CASES["continent"]
    h = scaled(h0, 4.0)
    m, info = run(dom, h, {"cut": True})
    topo = build_topology(m)
    assert (signed_area(m) > 0).all() and (topo.elems_per_node > 0).all()
    ic = irregular_counts(m)
    assert ic["euler_ok"] and ic["euler_sum"] == 4 * (2 - len(dom.loops))   # manifold, 3 loops, connected
    s = compute_quality(m).summary
    assert s["n_pinch_nodes"] == 0 and s["elem_neighbours_min"] >= 2
    nb = int(topo.boundary_node.sum())
    assert topo.boundary_node[:nb].all() and np.array_equal(m.node_flag == 1, topo.boundary_node)
    b = m.points[:nb]
    _, j = cKDTree(np.vstack(dom.loops)).query(b)                          # coast vertices 5 km apart
    d = R_EARTH_M * arc_length(b, np.vstack(dom.loops)[j])
    assert (d <= h(b)).all()                                                # the staircase stays within 1 h
    c = info["cut"]
    assert c["n_removed_mask"] > 0 and c["n_loops"] == 3 and info["irregular_in_blocks"] == 0
