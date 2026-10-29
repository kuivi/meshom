"""m13 (cubed sphere + SQuadGen LOWCONN refinement tiles): valid meshes on uniform, disc and continent at scale 4."""
import numpy as np

from meshom.cases import CASES
from meshom.domain import Domain, in_water, resample_loops
from meshom.mesh import build_topology, signed_area
from meshom.methods import METHODS
from meshom.methods.m13_squadgen import build
from meshom.size import scaled


def _valid(m):
    topo = build_topology(m)
    a = np.concatenate([m.tris.ravel(), m.quads.ravel()])
    b = np.concatenate([np.roll(m.tris, -1, axis=1).ravel(), np.roll(m.quads, -1, axis=1).ravel()])
    assert len(np.unique(a * m.n_nodes + b)) == len(a)          # each directed edge once: manifold, consistent CCW
    assert (signed_area(m) > 0).all()
    assert np.isin(np.arange(m.n_nodes), a).all()
    return topo


def _closed_all_quads(m):
    topo = _valid(m)
    assert m.n_tris == 0
    assert (topo.edge_elems[:, 1] >= 0).all()                    # closed, conforming (no hanging nodes)
    assert m.n_nodes - len(topo.edges) + m.n_elems == 2
    return topo


def test_uniform_scale4_is_the_cubed_sphere():
    dom, h = CASES["uniform"]
    topo = _closed_all_quads(METHODS["m13_squadgen"](dom, scaled(h, 4), {}))
    assert np.bincount(topo.elems_per_node)[3] == 8               # the 8 cube corners, everything else valence 4
    assert (topo.elems_per_node >= 3).all() and (topo.elems_per_node <= 4).all()


def test_disc_scale4_two_levels_of_tiles():
    dom, h = CASES["disc"]
    m, info = build(dom, scaled(h, 4), {"size_iters": 0, "spring_iters": 0})
    _closed_all_quads(m)
    assert info["levels"] == 2 and set(np.unique(info["tag"])) == {0, 1, 2}


def test_continent_scale4_valid_coast():
    dom, h = CASES["continent"]
    hs = scaled(h, 4)
    m = METHODS["m13_squadgen"](dom, hs, {})
    topo = _valid(m)
    bed = topo.edges[topo.edge_elems[:, 1] < 0]
    nbe = np.bincount(bed.ravel(), minlength=m.n_nodes)
    assert set(np.unique(nbe[topo.boundary_node])) == {2}        # boundary nodes on exactly 2 boundary edges
    coast = np.vstack(resample_loops(dom, hs))
    assert np.allclose(m.points[: len(coast)], coast)             # every resampled coast node kept, first
    assert (m.node_flag[: len(coast)] == 1).all() and topo.boundary_node[: len(coast)].all()
    c = np.vstack([m.points[m.tris].sum(1), m.points[m.quads].sum(1)])
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    assert in_water(Domain(resample_loops(dom, hs)), c).all()
    chi = m.n_nodes - len(topo.edges) + m.n_elems
    assert chi == 2 - len(dom.loops)                              # channel open: sphere minus 3 discs
