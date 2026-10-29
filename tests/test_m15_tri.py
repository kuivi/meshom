"""m15_tri (seeded packing -> Delaunay + coast -> population control -> size fit): valid meshes on `uniform`
and `continent` at scale 4, deterministic, every gate on `uniform` at scale 4; the fit variants are registered."""
import numpy as np
import pytest

from meshom.cases import CASES
from meshom.compete import _score, irregular_counts
from meshom.config import load_gates
from meshom.domain import resample_loops
from meshom.mesh import build_topology, signed_area
from meshom.methods import METHODS
from meshom.size import scaled


def test_registered():
    assert {"m15_tri", "m01_triangles_fit", "m02_linf_frontal_tri_fit"} <= set(METHODS)


@pytest.mark.parametrize("case", ["uniform", "continent"])
def test_valid_mesh(case):
    dom, h0 = CASES[case]
    h = scaled(h0, 4.0)
    m = METHODS["m15_tri"](dom, h, {})
    topo = build_topology(m)
    assert m.n_quads == 0 and (signed_area(m) > 0).all()                 # CCW, no inverted triangle
    assert (topo.elems_per_node > 0).all()
    ic = irregular_counts(m)
    assert ic["euler_ok"] and ic["euler_sum"] == 4 * (2 - len(dom.loops))   # manifold, right topology
    loops = resample_loops(dom, h)
    nb = sum(len(lp) for lp in loops)
    assert np.array_equal(m.points[:nb], np.vstack(loops) if loops else m.points[:0])   # coast nodes kept, first
    assert topo.boundary_node.sum() == nb and topo.boundary_node[:nb].all()
    if case == "uniform":
        sc = _score(m, dom, h, {"plus": 0.0, "minus": 0.0}, load_gates())
        assert sc["gates_passed"] == sc["gates_total"] >= 14, sc["gates_failed"]
        deg = np.bincount(topo.edges.ravel(), minlength=m.n_nodes)
        assert np.mean(deg != 6) < 0.08                                   # 3.6 k nodes: 12 valence-5 nodes + seams


def test_deterministic():
    dom, h0 = CASES["uniform"]
    h = scaled(h0, 4.0)
    a, b = METHODS["m15_tri"](dom, h, {}), METHODS["m15_tri"](dom, h, {})
    assert np.array_equal(a.points, b.points) and np.array_equal(a.tris, b.tris)
