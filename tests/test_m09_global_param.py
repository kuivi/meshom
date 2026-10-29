"""m09: seamless integer-grid map -> quads; valid meshes at scale 4, the cube-like uniform sphere."""
import numpy as np
import pytest

from meshom.cases import CASES
from meshom.compete import irregular_counts
from meshom.domain import resample_loops
from meshom.mesh import build_topology, signed_area
from meshom.methods import METHODS
from meshom.methods.m09.param import _gauss_jordan
from meshom.quality.metrics import compute_quality
from meshom.size import scaled


def test_gauss_jordan_keeps_translations_integer():
    """A regular branch node: s2 = i s1 + t1, s3 = i s2 + t2, s1 = -s3 + t3 (rotations compose to 1). The continuous
    sectors are eliminated first, then one translation, as a unit (Gaussian-integer) combination of the others."""
    kind = np.array([0, 0, 0, 1, 1, 1])                                    # s1 s2 s3 continuous, t1 t2 t3 translations
    rows = [({1: 1, 0: -1j, 3: -1}, 0j), ({2: 1, 1: -1j, 4: -1}, 0j), ({0: 1, 2: 1, 5: -1}, 0j)]
    rows = [({k: complex(v) for k, v in d.items()}, c) for d, c in rows]
    piv, rows, n_red, bad, n_nonunit = _gauss_jordan(rows, kind)
    assert n_nonunit == 0 and n_red == 0
    assert sorted(kind[list(piv)]) == [0, 0, 1]                            # two sectors and one translation
    t = [p for p in piv if kind[p] == 1][0]
    d, _ = rows[piv[t]]
    assert all(kind[v] == 1 for v in d) and all(abs(abs(x) - 1) < 1e-12 for x in d.values())


@pytest.mark.parametrize("case", ["uniform", "continent"])
def test_valid_mesh(case):
    dom, h0 = CASES[case]
    h = scaled(h0, 4.0)
    m = METHODS["m09_global_param"](dom, h, {})
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
    assert m.n_quads > 0.98 * m.n_elems
    assert ic["n3"] + ic["n5"] + ic["n_other"] < 0.01 * ic["n_interior"]   # irregular nodes ~ the field's cones
    if case == "uniform":                                                  # the cube: 8 valence-3 nodes, nothing else
        assert (ic["n3"], ic["n5"], ic["n_other"], m.n_tris) == (8, 0, 0, 0)
