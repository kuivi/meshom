"""ref_jigsaw (reference baseline, JIGSAW executable of the meshom-jigsaw env): valid closed mesh, the two variants
agree without a coast, the continent cut keeps the boundary within 1 h of the coast. Skipped without JIGSAW."""
from pathlib import Path

import numpy as np
import pytest

from meshom.cases import CASES
from meshom.compete import irregular_counts
from meshom.mesh import build_topology, signed_area
from meshom.methods.ref_jigsaw import JIGSAW_BIN, run
from meshom.size import scaled

pytestmark = pytest.mark.skipif(not Path(JIGSAW_BIN).exists(), reason=f"no JIGSAW executable at {JIGSAW_BIN}")


def _valid(m, n_loops):
    topo = build_topology(m)
    assert np.allclose(np.linalg.norm(m.points, axis=1), 1.0, atol=1e-14)    # on the unit sphere
    assert m.n_quads == 0 and (signed_area(m) > 0).all()                      # CCW triangles
    assert (topo.elems_per_node > 0).all()                                    # every node used
    ic = irregular_counts(m)
    assert ic["euler_ok"] and ic["euler_sum"] == 4 * (2 - n_loops)          # edge-manifold, right topology
    assert np.array_equal(m.node_flag == 1, topo.boundary_node)
    nb = int(topo.boundary_node.sum())
    assert topo.boundary_node[:nb].all()                                      # boundary nodes first
    return topo


def test_valid_closed_uniform():
    dom, h0 = CASES["uniform"]
    m, info = run(dom, scaled(h0, 4.0))
    topo = _valid(m, 0)
    assert (topo.edge_elems[:, 1] >= 0).all()                                 # closed: every edge has two triangles
    assert info["jigsaw_reported"]["total_s"] > 0 and info["timings"]["jigsaw_s"] >= info["jigsaw_reported"]["total_s"]


@pytest.mark.parametrize("case", ["uniform", "disc", "band"])
def test_variants_agree_without_coast(case):
    dom, h0 = CASES[case]
    h = scaled(h0, 4.0)
    a, _ = run(dom, h, fit=False)
    b, _ = run(dom, h, fit=True)
    assert np.array_equal(a.points, b.points) and np.array_equal(a.tris, b.tris)


def test_continent_coast():
    dom, h0 = CASES["continent"]
    h = scaled(h0, 4.0)
    m, info = run(dom, h, fit=False)
    _valid(m, len(dom.loops))
    assert info["coast_fit"]["boundary_to_coast_h"]["max"] <= 1.0              # every boundary node within 1 h
    assert info["coast_fit"]["n_loops"] == len(dom.loops)                      # the channel stays open
    f, fi = run(dom, h, fit=True)
    _valid(f, len(dom.loops))
    assert fi["cut"]["boundary_edges_on_features"] == fi["cut"]["n_boundary_edges"]   # conforming: coast = JIGSAW edges
    assert fi["coast_fit"]["boundary_to_coast_h"]["max"] <= 0.05
