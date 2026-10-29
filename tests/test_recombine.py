import numpy as np
import pytest

from helpers import same_elements, split_quads, structured_patch
from meshom.cases import CASES
from meshom.delaunay import background_mesh
from meshom.recombine import quad_cost, recombine
from meshom.size import scaled


@pytest.fixture(scope="module")
def random_mesh():
    dom, h = CASES["uniform"]
    return background_mesh(dom, scaled(h, 20.0))           # ~10^3 isotropic triangles on the closed sphere


@pytest.mark.parametrize("method", ["greedy", "exact"])
def test_right_triangle_split_recombines_to_original_quads(method):
    patch = structured_patch(6, 5, 10e3, lon0=20.0, lat0=50.0)
    tris = split_quads(patch, range(patch.n_quads))
    assert quad_cost(tris.points, tris.tris[: patch.n_quads], tris.tris[patch.n_quads:]).max() < 1e-3
    out = recombine(tris, 1.0, method)
    assert out.n_tris == 0
    assert same_elements(out, patch)


@pytest.mark.parametrize("price", [1.0, 0.4])
def test_greedy_close_to_exact_on_random_mesh(random_mesh, price):
    g = recombine(random_mesh, price, "greedy")
    e = recombine(random_mesh, price, "exact")
    assert 2 * g.n_quads + g.n_tris == random_mesh.n_tris
    assert g.n_tris <= 1.2 * e.n_tris


@pytest.mark.parametrize("method", ["greedy", "exact"])
def test_price_zero_leaves_mesh_unchanged(random_mesh, method):
    out = recombine(random_mesh, 0.0, method)
    assert same_elements(out, random_mesh)
    assert np.array_equal(out.node_flag, random_mesh.node_flag)
