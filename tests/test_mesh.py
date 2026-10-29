import numpy as np

from helpers import cube_sphere, structured_patch
from meshom.mesh import build_topology, orient_ccw, signed_area


def test_cube_sphere_topology():
    m = cube_sphere(); t = build_topology(m)
    assert len(t.edges) == 12 and np.all(t.edge_elems[:, 1] >= 0)
    assert np.all(t.elems_per_node == 3) and not t.boundary_node.any()
    assert np.all(t.elem_neighbours == 4)
    assert m.n_nodes - len(t.edges) + m.n_elems == 2


def test_patch_topology():
    m = structured_patch(3, 2, 1e4)      # nx x ny quads of dx metres at (0E, 0N)
    t = build_topology(m)
    assert len(t.edges) == 17 and (t.edge_elems[:, 1] < 0).sum() == 10
    assert t.boundary_node.sum() == 10 and t.elem_neighbours.min() == 2


def test_orient_ccw():
    m = cube_sphere(); m.quads[[0, 2, 4]] = m.quads[[0, 2, 4], ::-1]
    assert (signed_area(m) < 0).sum() == 3
    m2 = orient_ccw(m)
    assert np.all(signed_area(m2) > 0)
    assert np.array_equal(orient_ccw(m2).quads, m2.quads)
