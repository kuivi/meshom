"""m02: L-inf frontal packing along the cross field -> Delaunay -> recombination; valid meshes on `uniform` and
`continent` at scale 4."""
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
@pytest.mark.parametrize("method", ["m02_linf_frontal", "m02_linf_frontal_tri"])
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
    deg = np.bincount(topo.edges.ravel(), minlength=m.n_nodes)[~topo.boundary_node]
    if method == "m02_linf_frontal":
        assert m.n_quads > 0.95 * m.n_elems and s["quad_max_angle_deg"]["max"] <= 150.0
        # field-aligned packing: far fewer irregular interior nodes than recombined isotropic points (~27 %)
        assert np.mean(deg != 4) < 0.06
    else:                                                                  # asterisk field: near-regular triangles
        assert m.n_quads == 0 and s["tri_max_angle_deg"]["p99"] <= 90.0 and s["tri_min_angle_deg"]["min"] >= 20.0
        assert np.mean(deg != 6) < 0.10
