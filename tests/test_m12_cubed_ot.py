"""m12 (cubed sphere moved by optimal transport): valid meshes on uniform and continent at scale 4."""
import numpy as np

from meshom.cases import CASES
from meshom.domain import Domain, in_water, resample_loops
from meshom.mesh import build_topology, signed_area
from meshom.methods import METHODS
from meshom.size import scaled


def _valid(m):
    topo = build_topology(m)
    a = np.concatenate([m.tris.ravel(), m.quads.ravel()])
    b = np.concatenate([np.roll(m.tris, -1, axis=1).ravel(), np.roll(m.quads, -1, axis=1).ravel()])
    assert len(np.unique(a * m.n_nodes + b)) == len(a)          # each directed edge once: manifold, consistent CCW
    assert (signed_area(m) > 0).all()
    assert np.isin(np.arange(m.n_nodes), a).all()
    return topo


def test_uniform_scale4_all_quads_closed():
    dom, h = CASES["uniform"]
    m = METHODS["m12_cubed_ot"](dom, scaled(h, 4), {})
    topo = _valid(m)
    assert m.n_tris == 0
    assert (topo.edge_elems[:, 1] >= 0).all()                    # closed surface
    assert m.n_nodes - len(topo.edges) + m.n_elems == 2
    assert np.bincount(topo.elems_per_node)[3] == 8               # the 8 cube corners, everything else valence 4
    assert (topo.elems_per_node >= 3).all() and (topo.elems_per_node <= 4).all()


def test_continent_scale4_valid_coast():
    dom, h = CASES["continent"]
    hs = scaled(h, 4)
    m = METHODS["m12_cubed_ot"](dom, hs, {})
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
    assert m.n_tris < 0.05 * m.n_elems
