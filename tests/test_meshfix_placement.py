"""Core operations of c_placement (meshom/meshfix/placement, decisions B130-B133) on small synthetic meshes."""
import numpy as np

from meshom.geo import R_EARTH_M, lonlat_to_xyz
from meshom.mesh import Mesh
from meshom.meshfix.placement import extract as X
from meshom.meshfix.placement import fields as F
from meshom.meshfix.placement.relax import locate


def _grid(nx=12, ny=8, d=0.1):
    """A regular lon-lat quad grid near the equator (d degrees), CCW quads."""
    lon, lat = np.meshgrid(np.arange(nx) * d, np.arange(ny) * d)
    P = lonlat_to_xyz(lon.ravel(), lat.ravel())
    idx = np.arange(nx * ny).reshape(ny, nx)
    q = np.column_stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()])
    return Mesh(P, np.zeros((0, 3), np.int64), q, np.zeros(len(P), np.int8)), lon.ravel(), lat.ravel()


def _local_edges(tris, n):
    E = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    return np.unique(np.sort(E, 1), axis=0)


def test_row_polyline_and_nodes_on_a_regular_grid():
    """v = row index, psi = exp(i 2 pi column): one polyline per interior level, one node per column."""
    m, lon, lat = _grid()
    tr, _ = F.tri_split(m)
    E = _local_edges(tr, m.n_nodes)
    v = np.round(lat / 0.1)
    col = lon / 0.1
    psi = np.exp(2j * np.pi * col)
    h = np.full(m.n_nodes, 0.1 * np.pi / 180 * R_EARTH_M)
    ndir = np.tile([0.0, 0.0, 1.0], (m.n_nodes, 1))
    ndir -= np.einsum("ij,ij->i", ndir, m.points)[:, None] * m.points
    ndir /= np.linalg.norm(ndir, axis=1, keepdims=True)
    edir = np.cross(ndir, m.points)
    om = 2 * np.pi * (col[E[:, 1]] - col[E[:, 0]])
    lines = X.row_polylines(m.points, tr, E, X.edge_lookup(E, m.n_nodes), v, 3 - 1e-6)
    assert len(lines) == 1
    Y, k, _ = X.row_nodes(m.points, E, lines[0][0], lines[0][1], psi, om, h, edir)
    assert len(Y) == 12                                   # every column of row 3
    assert np.all(np.diff(k) == 1)


def test_dp_quads_and_one_triangle():
    """Equal rows give quads only; a row with one node more gives exactly one triangle."""
    o = X.DEFAULTS
    hh = np.full(10, 0.1 * np.pi / 180)
    e = np.tile([0.0, 0.0, 1.0], (10, 1))
    a = lonlat_to_xyz(np.zeros(10), np.arange(10) * 0.1)          # points along +y in xyz near lon 0: rows
    b = lonlat_to_xyz(np.full(10, 0.1), np.arange(10) * 0.1)
    r = X._dp(a, b, hh, hh, e, e, o["tilt_w"], o["tri_cost"], o["max_rung"])
    steps = np.diff(r, axis=0)
    assert len(r) == 10 and np.all(steps == 1)
    b2 = lonlat_to_xyz(np.full(11, 0.1), np.arange(11) * 0.09)
    hb = np.full(11, 0.1 * np.pi / 180)
    eb = np.tile([0.0, 0.0, 1.0], (11, 1))
    r2 = X._dp(a, b2, hh, hb, e, eb, o["tilt_w"], o["tri_cost"], o["max_rung"])
    steps2 = np.diff(r2, axis=0)
    assert (steps2.sum(1) == 1).sum() == 1                       # one triangle
    assert r2[0].tolist() == [0, 0] and r2[-1].tolist() == [9, 10]


def test_winding_finds_a_vortex():
    """psi = (x + i y) / |x + i y| around the grid's centre: exactly one triangle with winding +1."""
    m, lon, lat = _grid(9, 9)
    tr, _ = F.tri_split(m)
    E = _local_edges(tr, m.n_nodes)
    z = (lon - 0.43) + 1j * (lat - 0.37)
    psi = z / np.abs(z)
    w = F.winding(tr, E, np.zeros(len(E)), psi)
    assert (w != 0).sum() == 1 and w.sum() == 1


def test_ginzburg_landau_keeps_a_consistent_field():
    """Without frustration (om = the lattice's own phase steps) the GL minimiser stays vortex free and |psi| ~ 1."""
    m, lon, lat = _grid()
    tr, _ = F.tri_split(m)
    E = _local_edges(tr, m.n_nodes)
    col = lon / 0.1
    om = 2 * np.pi * (col[E[:, 1]] - col[E[:, 0]]) * 0.97
    fixed = (lat < 0.05) | (lat > 0.65)
    L = F.connection_laplacian(E, om, m.n_nodes)
    p0 = np.exp(2j * np.pi * col)
    p0[~fixed] = 1.0
    p0[fixed] = 1.0
    psi, _ = F.ginzburg_landau(L, fixed, p0, kappa=1.0, iters=2000)
    assert np.all(F.winding(tr, E, om, psi) == 0)
    assert np.abs(psi[~fixed]).min() > 0.8


def test_locate_points_in_elements():
    m, lon, lat = _grid()
    tr, owner = F.tri_split(m)
    c = m.points[m.quads].mean(1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    assert np.array_equal(locate(m, tr, owner, c), np.arange(m.n_quads))
    far = lonlat_to_xyz(np.array([50.0]), np.array([50.0]))
    assert locate(m, tr, owner, far)[0] == -1


def test_banded_dp_equals_full_dp():
    o = X.DEFAULTS
    a = lonlat_to_xyz(np.zeros(40), np.arange(40) * 0.1)
    b = lonlat_to_xyz(np.full(43, 0.1), np.arange(43) * 0.093)
    ha, hb = np.full(40, 0.1 * np.pi / 180), np.full(43, 0.1 * np.pi / 180)
    ea, eb = np.tile([0.0, 0.0, 1.0], (40, 1)), np.tile([0.0, 0.0, 1.0], (43, 1))
    full = X._dp(a, b, ha, hb, ea, eb, o["tilt_w"], o["tri_cost"], o["max_rung"])
    from scipy.spatial import cKDTree

    W = 16
    _, jn = cKDTree(b).query(a)
    jlo = np.maximum.accumulate(np.clip(jn - W // 2, 0, 43 - W)).astype(np.int64)
    band = X._dp_band(a, b, ha, hb, ea, eb, o["tilt_w"], o["tri_cost"], o["max_rung"], jlo, W)
    assert np.array_equal(full, band)
    assert (np.diff(band, axis=0).sum(1) == 1).sum() == 3      # three triangles for three extra nodes


def test_split_bad_quads_splits_a_flat_quad():
    """A quad with a 174 deg corner becomes two triangles through that corner (no triangle with two boundary
    edges); alone (four boundary edges) it stays."""
    P = lonlat_to_xyz(np.array([0.0, 0.1, 0.2, 0.1, 0.1]), np.array([0.0, -0.005, 0.0, 0.1, -0.1]))
    q = np.array([[0, 1, 2, 3]])
    t = np.array([[0, 4, 1], [1, 4, 2]])
    m = Mesh(P, t, q, np.zeros(len(P), np.int8))
    out, k = X.split_bad_quads(m, 145.0)
    assert k == 1 and out.n_quads == 0 and out.n_tris == 4
    assert all(1 in tt for tt in out.tris[2:].tolist())            # both new triangles through the flat corner
    alone = Mesh(P, np.zeros((0, 3), np.int64), q, np.zeros(len(P), np.int8))
    assert X.split_bad_quads(alone, 145.0)[1] == 0
