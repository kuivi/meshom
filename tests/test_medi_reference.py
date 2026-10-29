"""The reference quality numbers of the medi meshes (FESOM-C, Gmsh) are reproduced.

REF: an earlier independent measurement of these meshes (and their edgenum.out). Topology numbers are exact. Geometric
ones carry tolerances: REF was measured in the ellipsoidal stereographic plane of nod2d.out.cart, we measure on the
sphere with geodetic latitude read as spherical (a quad's max |angle - 90| moves by at most 0.286 deg on these meshes).
"""
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from meshom.geo import arc_length, lonlat_to_xyz
from meshom.io.fesom import read_fesom
from meshom.quality.metrics import Quality, compute_quality

REF = {
 "mesh_v5_1": dict(n_nodes=639670, n_elems=658682, n_tris=50041, n_quads=608641, n_tri_quad_edges=148975,
                   epn={3: 6743, 4: 568681, 5: 53029, 6: 574}, tri_min=(2.7, 39.7, 50.4), tri_max=124,
                   dev=(5.0, 32.8, 44.3, 80.7), dev45=322, sj=(0.72, 0.16), sj03=11, er=(1.59, 14.4),
                   sr=(1.02, 1.15, 1.23, 2.39), gt12=0.017),
 "mesh_v9_1": dict(n_nodes=135764, n_elems=132472, n_tris=18, n_quads=132454, n_tri_quad_edges=23,
                   n_edges=268322, n_edges_interior=261548,
                   epn={3: 8683, 4: 113684, 5: 7428, 6: 104}, tri_min=(26.2, 26.3, 36.6), tri_max=113,
                   dev=(6.4, 41.7, 67.7, 88.5), dev45=554, sj=(0.38, 0.03), sj03=94, er=(1.84, 5.7),
                   sr=(1.03, 1.22, 1.34, 2.30), gt12=0.069),
 "mesh_v11_1": dict(n_nodes=2623192, n_elems=2606311, n_tris=32, n_quads=2606279, n_tri_quad_edges=37,
                    n_edges=5229650, n_edges_interior=5195562,
                    epn={3: 75957, 4: 2441755, 5: 72352, 6: 740}, tri_min=(15.6, 15.6, 34.0), tri_max=116,
                    dev=(4.3, 37.3, 43.9, 89.0), dev45=1294, sj=(0.72, 0.02), sj03=310, er=(1.76, 4.7),
                    sr=(1.01, 1.18, 1.27, 2.05), gt12=0.035),
}
MESHES = list(REF)
COUNTS = ("n_nodes", "n_elems", "n_tris", "n_quads", "n_tri_quad_edges", "n_edges", "n_edges_interior")


@cache
def _quality(path: str) -> Quality:
    return compute_quality(read_fesom(Path(path)))


@pytest.fixture
def q(medi_dir, request) -> Quality:
    return _quality(str(medi_dir / request.param))


def _close(got: dict, keys: tuple, ref: tuple, tol: float) -> None:
    for k, r in zip(keys, ref):
        assert abs(got[k] - r) <= tol, (k, got[k], r, tol)


@pytest.mark.data
@pytest.mark.parametrize("q", MESHES, indirect=True)
def test_counts_and_valence(q, request):
    ref, s = REF[request.node.callspec.params["q"]], q.summary
    for k in COUNTS:
        if k in ref:
            assert s[k] == ref[k], k
    assert {v: s["elems_per_node_hist"].get(v, 0) for v in ref["epn"]} == ref["epn"]


@pytest.mark.data
@pytest.mark.parametrize("q", MESHES, indirect=True)
def test_triangle_angles(q, request):
    ref, s = REF[request.node.callspec.params["q"]], q.summary
    _close(s["tri_min_angle_deg"], ("min", "p1", "median"), ref["tri_min"], 0.5)
    assert abs(s["tri_max_angle_deg"]["max"] - ref["tri_max"]) <= 1.0     # REF rounds to integers


@pytest.mark.data
@pytest.mark.parametrize("q", MESHES, indirect=True)
def test_quad_shape(q, request):
    ref, s = REF[request.node.callspec.params["q"]], q.summary
    _close(s["quad_dev_deg"], ("median", "p99", "p99_9", "max"), ref["dev"], 0.5)
    _close(s["quad_sj"], ("p0_1", "min"), ref["sj"], 0.01)
    assert abs(s["quad_edge_ratio"]["p99"] - ref["er"][0]) <= 0.02
    assert abs(s["quad_edge_ratio"]["max"] / ref["er"][1] - 1.0) <= 0.02
    assert abs(s["n_quads_sj_lt03"] / ref["sj03"] - 1.0) <= 0.03
    assert s["n_quads_sj_lt03_boundary"] == s["n_quads_sj_lt03"]      # REF: every SJ < 0.3 quad touches the boundary


@pytest.mark.data
@pytest.mark.parametrize("q", MESHES, indirect=True)
def test_quads_over_45_within_angle_noise(q, request):
    """Plane vs sphere moves a quad's deviation by < 0.3 deg, so only quads within 0.3 deg of 45 can cross over."""
    ref, d = REF[request.node.callspec.params["q"]], q.quad_dev_deg
    band = int((np.abs(d - 45.0) <= 0.3).sum())
    assert abs(q.summary["n_quads_dev_gt45"] - ref["dev45"]) <= band, (q.summary["n_quads_dev_gt45"], band)


@pytest.mark.data
@pytest.mark.parametrize("q", MESHES, indirect=True)
def test_neighbour_size_ratio(q, request):
    ref, s = REF[request.node.callspec.params["q"]], q.summary
    _close(s["elem_size_ratio"], ("median", "p95", "p99"), ref["sr"][:3], 0.01)
    assert abs(s["elem_size_ratio"]["max"] - ref["sr"][3]) <= 0.02
    assert abs(s["share_elem_size_ratio_gt_1_2"] - ref["gt12"]) <= 0.002


@pytest.mark.data
@pytest.mark.parametrize("q", ["mesh_v11_1"], indirect=True)
def test_v11_worst_quad_in_venice_lagoon(q):
    venice = lonlat_to_xyz(np.array([12.19]), np.array([45.41]))
    T = q.touches_boundary.size - q.quad_sj.size                        # quads follow the triangles
    worst = q.centroid[T + np.argsort(q.quad_sj, kind="stable")[:10]]
    dist_deg = np.degrees(arc_length(worst, np.repeat(venice, len(worst), axis=0)))
    assert dist_deg.min() <= 0.1, dist_deg
