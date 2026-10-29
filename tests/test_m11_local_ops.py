import numpy as np
import pytest

from meshom.cases import CASES
from meshom.compete import irregular_counts
from meshom.domain import resample_loops
from meshom.mesh import build_topology, signed_area
from meshom.methods import METHODS
from meshom.methods.m11.ops import edge_ratios
from meshom.size import scaled


@pytest.mark.parametrize("case", ["uniform", "continent"])
def test_valid_mesh(case):
    domain, h0 = CASES[case]
    h = scaled(h0, 4)
    tri = METHODS["m11_local_ops_tri"](domain, h, {})
    r = edge_ratios(tri, h)
    assert np.mean((r >= 0.7) & (r <= 1.4)) > 0.99                       # the operators reached the window
    mixed = METHODS["m11_local_ops"](domain, h, {})
    assert mixed.n_quads > 0.8 * mixed.n_elems
    for m in (tri, mixed):
        assert (signed_area(m) > 0).all()                                # CCW, nothing inverted
        a = np.r_[m.tris.ravel(), m.quads.ravel()]
        b = np.r_[np.roll(m.tris, -1, axis=1).ravel(), np.roll(m.quads, -1, axis=1).ravel()]
        assert len(np.unique(a * m.n_nodes + b)) == len(a)              # manifold: each directed edge once
        topo = build_topology(m)
        ic = irregular_counts(m)
        assert ic["euler_sum"] == 4 * (2 - len(domain.loops)) and ic["euler_ok"]
        if domain.loops:                                       # the resampled coast nodes, all of them on the boundary
            coast = np.vstack(resample_loops(domain, h))
            assert np.array_equal(m.points[: len(coast)], coast)
            assert np.array_equal(topo.boundary_node, m.node_flag != 0) and (m.node_flag != 0).sum() == len(coast)
