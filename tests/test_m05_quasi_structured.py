"""m05: m02 + cavity remeshing with quad / mixed patterns; valid meshes on `uniform` and `continent` at scale 4, fewer
irregular spots than m02."""
import numpy as np
import pytest

from meshom.cases import CASES
from meshom.compete import irregular_counts
from meshom.domain import resample_loops
from meshom.mesh import build_topology, signed_area
from meshom.methods import METHODS
from meshom.methods.m05.patterns import STEP2, STEP4, fine_mesh, match
from meshom.quality.metrics import compute_quality
from meshom.size import scaled


@pytest.mark.parametrize("case", ["uniform", "continent"])
def test_valid_mesh(case):
    dom, h0 = CASES[case]
    h = scaled(h0, 4.0)
    m = METHODS["m05_quasi_structured"](dom, h, {})
    topo = build_topology(m)
    assert (signed_area(m) > 0).all()                                     # CCW, no inverted element
    assert (topo.elems_per_node > 0).all()
    ic = irregular_counts(m)
    assert ic["euler_ok"] and ic["euler_sum"] == 4 * (2 - len(dom.loops))
    s = compute_quality(m).summary
    assert s["n_pinch_nodes"] == 0 and s["quad_max_angle_deg"]["max"] <= 150.0
    loops = resample_loops(dom, h)
    nb = sum(len(lp) for lp in loops)
    assert np.array_equal(m.points[:nb], np.vstack(loops) if loops else m.points[:0])   # coast nodes kept, first
    assert topo.boundary_node.sum() == nb and topo.boundary_node[:nb].all()
    m2 = METHODS["m02_linf_frontal"](dom, h, {})
    i2 = irregular_counts(m2)
    assert ic["n3"] + ic["n5"] + ic["T"] < 0.8 * (i2["n3"] + i2["n5"] + i2["T"])


def test_patterns_fill_the_loop():
    """Every matched pattern gives a disk (Euler 1) whose boundary loop has the cavity's side lengths."""
    for pats, N in [(STEP4, [3, 2, 2, 2]), (STEP4, [4, 3, 4, 3]), (STEP4, [3, 4, 3, 5]), (STEP2, [3, 3, 3]),
                    (STEP2, [4, 4, 4]), (STEP2, [2, 2, 2, 2, 2])]:
        p, r, s, _ = match(pats, np.array(N))
        n, el, loop, inner = fine_mesh(p, s, r)
        edges, half = set(), set()
        for e in el:
            k = 3 if e[3] < 0 else 4
            for i in range(k):
                edges.add((min(e[i], e[(i + 1) % k]), max(e[i], e[(i + 1) % k])))
                half.add((e[i], e[(i + 1) % k]))
        assert n - len(edges) + len(el) == 1
        assert len(loop) == sum(N) and all((a, b) in half for a, b in zip(loop, np.roll(loop, -1)))
