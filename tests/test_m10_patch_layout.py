"""m10: patch layout (separatrices of the cone field) -> MILP quantisation -> Coons fill; valid meshes on `uniform`
and `continent` at scale 4."""
import numpy as np
import pytest

from meshom.cases import CASES
from meshom.compete import irregular_counts
from meshom.domain import resample_loops
from meshom.mesh import build_topology, signed_area
from meshom.methods import METHODS
from meshom.methods.m10_patch_layout import LAST_INFO
from meshom.quality.metrics import compute_quality
from meshom.size import scaled


@pytest.mark.parametrize("case", ["uniform", "continent"])
def test_valid_mesh(case):
    dom, h0 = CASES[case]
    h = scaled(h0, 4.0)
    m = METHODS["m10_patch_layout"](dom, h, {})
    topo = build_topology(m)
    assert (signed_area(m) > 0).all()                                     # CCW, no inverted element
    assert (topo.elems_per_node > 0).all()                                # every node used
    ic = irregular_counts(m)
    assert ic["euler_ok"] and ic["euler_sum"] == 4 * (2 - len(dom.loops))   # edge-manifold, right topology
    assert LAST_INFO["euler"] == 2                                        # the layout itself is a planar graph
    s = compute_quality(m).summary
    assert s["n_pinch_nodes"] == 0 and s["quad_max_angle_deg"]["max"] < 150.0
    loops = resample_loops(dom, h)
    nb = sum(len(lp) for lp in loops)
    assert np.array_equal(m.points[:nb], np.vstack(loops) if loops else m.points[:0])   # coast nodes kept, first
    assert topo.boundary_node.sum() == nb and topo.boundary_node[:nb].all()
    if case == "uniform":
        # the cube layout: 8 valence-3 cones, all quads, no other irregular node
        assert m.n_tris == 0 and ic["n3"] == 8 and ic["n5"] == 0 and ic["n_other"] == 0
    else:
        assert m.n_quads > 0.95 * m.n_elems and s["tri_max_angle_deg"]["max"] < 120.0
