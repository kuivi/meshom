"""size_fit: a strained lattice patch with a fixed rim converges to |ln s/h| < 0.05; a strongly strained
start never yields an inverted triangle."""
import numpy as np
from scipy.spatial import Delaunay

from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh, build_topology, signed_area
from meshom.size import uniform
from meshom.size_fit import edge_ln_sh, size_fit
from tests.helpers import local_polygon

H = 10e3


def _patch(rings: int = 6):
    """Hexagonal lattice (spacing H) within `rings` rings of the origin at (20E, 30N): points, CCW triangles, rim."""
    xy = []
    for i in range(-rings, rings + 1):
        for j in range(-rings, rings + 1):
            if abs(i) <= rings and abs(j) <= rings and abs(i + j) <= rings:
                xy.append((i + 0.5 * j, j * np.sqrt(3) / 2))
    xy = np.array(xy) * H
    tri = Delaunay(xy).simplices.astype(np.int64)
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    cr = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    tri[cr < 0] = tri[cr < 0][:, [0, 2, 1]]
    ii = np.round(xy[:, 1] / (H * np.sqrt(3) / 2))
    rim = np.isclose(np.max(np.abs(np.c_[xy[:, 0] / H - 0.5 * ii, ii, xy[:, 0] / H - 0.5 * ii + ii]), axis=1), rings)
    return xy, tri, rim


def _mesh(xy, tri, rim):
    flag = rim.astype(np.int8)
    return Mesh(local_polygon(20.0, 30.0, xy), tri, np.zeros((0, 4), dtype=np.int64), flag)


def test_strained_patch_converges():
    xy, tri, rim = _patch()
    rng = np.random.default_rng(1)
    d = rng.uniform(-0.2, 0.2, xy.shape) * H                     # interior jitter up to 0.2 h per axis
    d[rim] = 0.0
    m0 = _mesh(xy + d, tri, rim)
    assert (signed_area(m0) > 0).all()
    e = build_topology(m0).edges
    assert np.abs(edge_ln_sh(m0.points, e, uniform(H))).max() > 0.2
    m, info = size_fit(m0, uniform(H), None, p=6, max_iter=300, rounds=3, flips=False)
    r = edge_ln_sh(m.points, e, uniform(H))
    assert np.abs(r).max() < 0.05, info
    assert np.array_equal(m.points[rim], m0.points[rim])           # the rim never moves
    assert (signed_area(m) > 0).all()


def test_never_inverts_from_a_valid_start():
    xy, tri, rim = _patch()
    # squeeze the interior toward a line (thin, nearly flat triangles, all still CCW) and ask for 1.5 h inside:
    # the rim fixes the count, so the fit must compress something; nothing may fold
    sq = xy.copy()
    sq[~rim, 1] *= 0.25
    m0 = _mesh(sq, tri, rim)
    assert (signed_area(m0) > 0).all()
    h = lambda x: np.where(np.linalg.norm(x - m0.points[len(xy) // 2], axis=1) * R_EARTH_M < 3.5 * H, 1.5 * H, H)
    m, info = size_fit(m0, h, None, p=6, max_iter=200, rounds=2)
    assert (signed_area(m) > 0).all(), info
    assert info["rounds"][0]["E"] < info["rounds"][0]["E0"]
