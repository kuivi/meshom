"""m03: packing of parallelograms -> Delaunay -> recombination; valid meshes on `uniform` and `continent` at scale 4."""
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
def test_valid_mesh(case):
    dom, h0 = CASES[case]
    h = scaled(h0, 4.0)
    m = METHODS["m03_parallelograms"](dom, h, {})
    topo = build_topology(m)
    assert (signed_area(m) > 0).all()                                     # CCW, no inverted element
    assert (topo.elems_per_node > 0).all()                                # every node used
    ic = irregular_counts(m)
    assert ic["euler_ok"] and ic["euler_sum"] == 4 * (2 - len(dom.loops))   # edge-manifold, right topology
    s = compute_quality(m).summary
    assert s["n_pinch_nodes"] == 0
    loops = resample_loops(dom, h)
    nb = sum(len(lp) for lp in loops)
    assert np.array_equal(m.points[:nb], np.vstack(loops) if loops else m.points[:0])   # coast nodes kept, first
    assert topo.boundary_node.sum() == nb and topo.boundary_node[:nb].all()
    # the packing makes a lattice: few triangles and few irregular nodes (m01_isotropic: ~10 % and ~27 %)
    assert m.n_tris < 0.06 * m.n_elems and ic["n3"] + ic["n5"] < 0.06 * ic["n_interior"]
    assert s["quad_max_angle_deg"]["max"] <= 150.0
