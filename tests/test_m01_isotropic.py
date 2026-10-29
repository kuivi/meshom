"""m01: isotropic points -> sphere Delaunay -> recombination; valid meshes on `uniform` and `continent` at scale 4."""
import numpy as np
import pytest

from meshom.cases import CASES
from meshom.compete import irregular_counts
from meshom.domain import resample_loops
from meshom.mesh import build_topology, signed_area
from meshom.methods import METHODS
from meshom.quality.metrics import compute_quality
from meshom.size import scaled


@pytest.mark.parametrize("case", ["uniform", "continent"])
@pytest.mark.parametrize("method", ["m01_isotropic", "m01_triangles"])
def test_valid_mesh(case, method):
    dom, h0 = CASES[case]
    h = scaled(h0, 4.0)
    m = METHODS[method](dom, h, {})
    topo = build_topology(m)
    assert (signed_area(m) > 0).all()                                     # CCW, no inverted element
    assert (topo.elems_per_node > 0).all()                                # every node used
    ic = irregular_counts(m)
    assert ic["euler_ok"] and ic["euler_sum"] == 4 * (2 - len(dom.loops))   # edge-manifold, right topology
    s = compute_quality(m).summary
    assert s["n_pinch_nodes"] == 0 and s["tri_max_angle_deg"]["max"] < 120.0
    loops = resample_loops(dom, h)
    nb = sum(len(lp) for lp in loops)
    assert np.array_equal(m.points[:nb], np.vstack(loops) if loops else m.points[:0])   # coast nodes kept, first
    assert topo.boundary_node.sum() == nb and topo.boundary_node[:nb].all()
    if method == "m01_isotropic":
        assert m.n_quads > 0.8 * m.n_elems and s["quad_max_angle_deg"]["max"] <= 150.0   # price 1/3: hard gate
    else:
        assert m.n_quads == 0 and s["tri_max_angle_deg"]["p99"] <= 90.0
