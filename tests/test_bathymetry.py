"""Depth rules of the bathymetry stage (meshom.bathymetry.depth_policy): the floor from the levels, volume-conserving
smoothing, the slope limiter and the tapered carving of strait depth floors, on small synthetic meshes."""
import numpy as np
import pytest

from meshom.bathymetry import (CORE2_LEVELS_M, depth_policy, limit_slope, min_depth_of, node_area_m2,
                               node_spacing_m, smooth_depth)
from meshom.geo import R_EARTH_M
from meshom.straits import Strait, depth_floor
from tests.helpers import local_polygon, split_quads, structured_patch


def _tri_patch(nx: int, ny: int, dx: float):
    m = structured_patch(nx, ny, dx)
    return split_quads(m, range(m.n_quads))


def _local_xy(mesh):
    """(x, y) metres of the nodes in the tangent plane at (0, 0) (the patches are centred there)."""
    p = mesh.points
    return R_EARTH_M * np.arctan2(p[:, 1], p[:, 0]), R_EARTH_M * np.arcsin(p[:, 2])


def test_min_depth_follows_the_levels():
    """The floor is the bottom of the min_layers-th layer: CORE2 zbar[4] = 30 m (FESOM's zbar(5)); no levels or
    min_layers 0 leave min_depth_m; a larger min_depth_m wins."""
    z = np.asarray(CORE2_LEVELS_M)
    assert min_depth_of(10.0, z, 4) == 30.0 and min_depth_of(10.0, z, 0) == 10.0
    assert min_depth_of(10.0, None, 4) == 10.0 and min_depth_of(40.0, z, 4) == 40.0


def test_smoothing_conserves_volume_and_range():
    """Random depths on a 10 km triangle patch: 5 passes keep sum A d to 1e-12, stay within the input range and
    shrink the spread; restricted to a node set, the other nodes do not move and the set's volume is kept."""
    mesh = _tri_patch(20, 16, 10e3)
    _, e, _ = node_spacing_m(mesh)
    a = node_area_m2(mesh)
    d0 = np.random.default_rng(1).uniform(30.0, 3000.0, mesh.n_nodes)
    d = smooth_depth(d0, e, a, 5)
    assert abs(a @ d / (a @ d0) - 1.0) < 1e-12
    assert d.min() >= d0.min() and d.max() <= d0.max() and d.std() < 0.5 * d0.std()
    x, _ = _local_xy(mesh)
    sel = x < 0
    ds = smooth_depth(d0, e, a, 5, sel)
    assert np.array_equal(ds[~sel], d0[~sel]) and abs(a[sel] @ ds[sel] / (a[sel] @ d0[sel]) - 1.0) < 1e-12


def test_slope_limiter_reaches_its_slope_and_conserves_volume():
    """A 30 m | 3000 m step across the patch: after the limiter no edge is steeper than 0.02 (+ tol), the volume is
    kept, the range is kept, and nodes far from the step (beyond the spread the volume needs) do not move."""
    mesh = _tri_patch(60, 10, 10e3)
    _, e, L = node_spacing_m(mesh)
    a = node_area_m2(mesh)
    x, _ = _local_xy(mesh)
    d0 = np.where(x < 0, 30.0, 3000.0)
    d, rec = limit_slope(d0, e, L, a, 0.02)
    s = np.abs(d[e[:, 0]] - d[e[:, 1]]) / L
    assert s.max() <= 0.02 * (1 + 1e-3) + 1e-12 and rec["edges_steep_left"] == 0
    assert abs(a @ d / (a @ d0) - 1.0) < 1e-12
    assert d.min() >= 30.0 - 1e-9 and d.max() <= 3000.0 + 1e-9
    far = np.abs(x) > 250e3          # the 2 970 m step at 200 m per 10 km spreads over about +-75 km
    assert np.allclose(d[far], d0[far])


def test_slope_limiter_converges_on_a_long_ramp():
    """A 30 m | 3000 m step on a 2 km quad strip at 0.02 needs a ramp of 74 cells (148.5 km): the limiter runs until
    no edge is steeper than the limit (decisions O30), with the volume and the range kept."""
    mesh = structured_patch(240, 2, 2e3)
    _, e, L = node_spacing_m(mesh)
    a = node_area_m2(mesh)
    x, _ = _local_xy(mesh)
    d0 = np.where(x < 0, 30.0, 3000.0)
    d, rec = limit_slope(d0, e, L, a, 0.02)
    s = np.abs(d[e[:, 0]] - d[e[:, 1]]) / L
    assert rec["edges_steep_left"] == 0 and s.max() <= 0.02 * (1 + 1e-3) + 1e-12
    assert abs(a @ d / (a @ d0) - 1.0) < 1e-12 and d.min() >= 30.0 - 1e-9 and d.max() <= 3000.0 + 1e-9


def test_slope_limiter_fails_loudly_when_steep_edges_are_left():
    """Stopped before convergence, the limiter raises instead of returning a depth above the limit."""
    mesh = structured_patch(240, 2, 2e3)
    _, e, L = node_spacing_m(mesh)
    x, _ = _local_xy(mesh)
    with pytest.raises(RuntimeError, match="steep"):
        limit_slope(np.where(x < 0, 30.0, 3000.0), e, L, node_area_m2(mesh), 0.02, max_iter=3)


def test_carving_reaches_its_floor_with_a_cosine_taper():
    """A 30 m flat patch, a strait along the x axis (the equator) with a 100 m floor (h 10 km): nodes within h of
    the path are at 100 m, beyond h + T (T = 1 x h) at 30 m, in between at max(30, 100 (1 + cos(pi x / T)) / 2),
    x = distance - h (to 3 m: the distance is to path points 3.6 km apart)."""
    mesh = _tri_patch(40, 40, 2.5e3)
    x, y = _local_xy(mesh)
    s = Strait("ch", local_polygon(0.0, 0.0, [[-200e3, 0.0], [0.0, 0.0], [200e3, 0.0]]), width_m=5e3, h_m=10e3,
               cells=2, depth_m=100.0)
    for f in (1.0, 2.0):                                         # the taper length T = taper h_m
        half, T = s.h_m, f * s.h_m
        d, rec = depth_floor(mesh, np.full(mesh.n_nodes, 30.0), [s], taper=f)
        r = np.abs(y)
        assert np.all(d[r <= half - 200.0] == 100.0) and np.all(d[r >= half + T + 200.0] == 30.0)
        xt = np.clip(r - half, 0.0, T)
        want = np.maximum(30.0, 100.0 * 0.5 * (1.0 + np.cos(np.pi * xt / T)))
        taper = (r > half) & (r < half + T)
        assert taper.sum() >= 3 * 41 and np.allclose(d, want, atol=3.0)
        assert rec["n_raised"] == int((d > 30.0).sum())


def test_depth_policy_keeps_the_floor_and_reports_moves():
    """The whole chain on a patch with a deep trench next to a strait: depth >= floor everywhere, the strait floor
    holds on the path after smoothing and limiting, and the record counts the moved nodes in and out of corridors."""
    mesh = _tri_patch(40, 20, 5e3)
    x, y = _local_xy(mesh)
    d0 = np.where(y > 20e3, 2000.0, 30.0)
    s = Strait("ch", local_polygon(0.0, 0.0, [[-100e3, 0.0], [100e3, 0.0]]), width_m=2e3, h_m=5e3, depth_m=60.0)
    d, rec = depth_policy(mesh, d0, 30.0, 2, 0.05, [s], 1.0)
    assert d.min() >= 30.0 and np.all(d[np.abs(y) < 1e3] >= 60.0)
    assert rec["corridors"]["n_nodes"] > 0 and rec["elsewhere"]["n_moved_1m"] > 0 and rec["slope_max_after"] < 0.06


def test_box_cache_gives_the_full_depth_bit_for_bit(tmp_path):
    """node_depth through the box-sum cache (`meshom write`, decisions O33) equals node_depth without it bit for bit:
    cold, after a hand edit (4 nodes moved: their boxes and their neighbours' change) with only the new boxes read,
    and on the first mesh again with nothing read."""
    netCDF4 = pytest.importorskip("netCDF4")
    from meshom.bathymetry import node_depth
    from meshom.delaunay import background_mesh
    from meshom.domain import closed_sphere
    from meshom.mesh import Mesh, build_topology
    from meshom.size import Uniform

    lat, lon = np.arange(-89.5, 90.0, 1.0), np.arange(-179.5, 180.0, 1.0)
    z = np.random.default_rng(3).integers(-5000, 400, size=(len(lat), len(lon))).astype(np.int16)
    nc = tmp_path / "grid.nc"
    with netCDF4.Dataset(nc, "w") as ds:
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        ds.createVariable("elevation", "i2", ("lat", "lon"))[:] = z
    m = background_mesh(closed_sphere(), Uniform(600e3))
    cache = tmp_path / "cache" / "depth.npz"
    d0, i0 = node_depth(m, nc, min_depth_m=10.0)
    d1, i1 = node_depth(m, nc, min_depth_m=10.0, cache=cache)
    assert np.array_equal(d0, d1) and cache.exists() and 0 < i1["n_boxes_read"] <= m.n_nodes
    assert "n_boxes_read" not in i0
    P = m.points.copy()
    e = build_topology(m).edges
    moved = [10, 200, 500, 900]
    for i in moved:
        j = e[e[:, 0] == i, 1][0]
        p = P[i] + 0.2 * (P[j] - P[i])
        P[i] = p / np.linalg.norm(p)
    m2 = Mesh(P, m.tris, m.quads, m.node_flag)
    e0, _ = node_depth(m2, nc, min_depth_m=10.0)
    e1, i2 = node_depth(m2, nc, min_depth_m=10.0, cache=cache)
    near = np.unique(e[np.isin(e, moved).any(axis=1)])                   # the moved nodes and their neighbours
    assert np.array_equal(e0, e1) and not np.array_equal(e0, d0)
    assert 0 < i2["n_boxes_read"] <= len(near)
    d2, i3 = node_depth(m, nc, min_depth_m=10.0, cache=cache)
    assert np.array_equal(d0, d2) and i3["n_boxes_read"] == 0


def test_cells_deepest_node_is_the_deepest_of_its_cells(tmp_path):
    """B103: depth_at "cells_deepest": an element's depth is the mean of the grid's ocean cells in a box of its area
    around its centroid, a node the deepest of its elements; so no element has a node shallower than its own water
    (a coastal node cannot block a strait), and the node depths are those maxima (above the floor)."""
    netCDF4 = pytest.importorskip("netCDF4")
    from meshom.bathymetry import box_means, cells_deepest_depth, node_depth
    from meshom.geo import xyz_to_lonlat

    m = structured_patch(12, 8, 5000.0)
    lat, lon = np.arange(-1.0, 1.0, 0.005) + 0.0025, np.arange(-1.0, 1.0, 0.005) + 0.0025
    z = np.random.default_rng(5).integers(-400, 50, size=(len(lat), len(lon))).astype(np.int16)
    z[np.abs(lat) < 0.03, :] = -300                                     # a deep channel along the equator
    nc = tmp_path / "grid.nc"
    with netCDF4.Dataset(nc, "w") as ds:
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        ds.createVariable("elevation", "i2", ("lat", "lon"))[:] = z
    d, info = cells_deepest_depth(m, nc, min_depth_m=10.0)
    P = m.points
    q = P[m.quads]
    c = q.mean(axis=1)
    a = 0.5 * (np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=1)
               + np.linalg.norm(np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0]), axis=1)) * R_EARTH_M ** 2
    lo, la = xyz_to_lonlat(c / np.linalg.norm(c, axis=1, keepdims=True))
    s, n = box_means(lo, la, 0.5 * np.sqrt(a), nc)
    de = -s / n
    assert (n > 0).all() and info["n_without_ocean_cell"] == 0
    assert (d[m.quads].min(axis=1) >= de - 1e-9).all()                   # no node shallower than its element
    exp = np.full(m.n_nodes, -np.inf)
    np.maximum.at(exp, m.quads.ravel(), np.repeat(de, 4))
    assert np.allclose(d, np.maximum(exp, 10.0))
    dn, _ = node_depth(m, nc, min_depth_m=10.0)
    assert d.mean() > dn.mean()                                          # deeper than the box means


# ------------------------------------------------------------------------- depth regions and node edits (C1, C2)

def _points(lonlat):
    """A Mesh of points only (the region rules look at the nodes alone)."""
    from meshom.geo import lonlat_to_xyz
    from meshom.mesh import Mesh

    p = np.asarray(lonlat, dtype=np.float64)
    return Mesh(lonlat_to_xyz(p[:, 0], p[:, 1]), np.zeros((0, 3), np.int64), np.zeros((0, 4), np.int64),
                np.zeros(len(p), np.int8))


def _region(**kw):
    from meshom.config import DepthRegionCfg

    return DepthRegionCfg(**kw)


def test_region_box_polygon_and_the_antimeridian():
    """A box [lon_min, lat_min, lon_max, lat_max] holds its edges; lon_min > lon_max (or lon_max > 180) crosses 180;
    a polygon is a spherical one (great-circle edges) and crosses 180 as it is."""
    from meshom.bathymetry import region_mask

    m = _points([[-5.6, 36.0], [-4.0, 36.0], [-6.0, 35.9], [179.5, 0.0], [-179.5, 0.0], [178.0, 0.0], [0.0, 0.0],
                 [-5.0, 36.5]])
    box = region_mask(m.points, _region(box=[-6.0, 35.8, -5.0, 36.2]))
    assert box.tolist() == [True, False, True, False, False, False, False, False]
    assert region_mask(m.points, _region(box=[-5.0, 36.5, -5.0, 36.5]))[7]                 # edges included
    across = [False, False, False, True, True, False, False, False]
    assert region_mask(m.points, _region(box=[179.0, -1.0, -179.0, 1.0])).tolist() == across
    assert region_mask(m.points, _region(box=[179.0, -1.0, 181.0, 1.0])).tolist() == across
    assert region_mask(m.points, _region(box=[-180.0, -1.0, 180.0, 1.0])).tolist() == [False] * 3 + [True] * 4 + [False]
    poly = region_mask(m.points, _region(polygon=[[179.0, -1.0], [-179.0, -1.0], [-179.0, 1.0], [179.0, 1.0]]))
    assert poly.tolist() == across
    tri = region_mask(m.points, _region(polygon=[[-7.0, 35.0], [-5.0, 35.0], [-5.0, 37.0]]))
    assert tri.tolist() == [True, False, True, False, False, False, False, False]


def test_regions_in_file_order_with_the_floor():
    """depth_m sets, then min_depth_m, then max_depth_m; a later region overrides an earlier one on common nodes;
    a rule below the floor gives the floor (decisions O36), counted per region; nodes outside are untouched."""
    from meshom.bathymetry import apply_regions

    m = _points([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    d0 = np.array([30.0, 100.0, 400.0, 5.0])
    d, rec = apply_regions(m, d0, [_region(name="a", box=[-0.5, -1, 2.5, 1], min_depth_m=50.0, max_depth_m=200.0),
                                   _region(name="b", box=[1.5, -1, 2.5, 1], depth_m=300.0),
                                   _region(name="c", box=[0.5, -1, 1.5, 1], depth_m=500.0, max_depth_m=20.0)],
                           floor_m=30.0)
    assert d.tolist() == [50.0, 30.0, 300.0, 5.0]           # node 3: outside every region, below the floor, kept
    assert [r["name"] for r in rec] == ["a", "b", "c"]
    assert [r["n_nodes"] for r in rec] == [3, 1, 1] and [r["n_changed"] for r in rec] == [2, 1, 1]
    assert [r["n_clamped_to_floor"] for r in rec] == [0, 0, 1]
    assert d0.tolist() == [30.0, 100.0, 400.0, 5.0]         # the input is not changed


def test_node_edits_in_order_with_the_record_and_the_floor(tmp_path):
    """set / min / max in file order on the listed nodes; the record counts the edits and the changed nodes; an
    edit below the floor gives the floor (counted); an edits file of another node count is an error."""
    import json

    from meshom.bathymetry import apply_node_edits

    d0 = np.array([100.0, 100.0, 100.0, 100.0, 10.0])
    f = tmp_path / "edited-01.depth_edits.json"
    edits = [{"op": "set", "depth_m": 300.0, "nodes": [0, 1], "outline": [], "date": "2026-10-27"},
             {"op": "min", "depth_m": 200.0, "nodes": [1, 2], "outline": [], "date": "2026-10-27"},
             {"op": "max", "depth_m": 250.0, "nodes": [0, 2], "outline": [], "date": "2026-10-27"},
             {"op": "max", "depth_m": 20.0, "nodes": [3], "outline": [], "date": "2026-10-27"}]
    f.write_text(json.dumps({"mesh": "edited-01.npz", "n_nodes": 5, "edits": edits}))
    d, rec = apply_node_edits(d0, f, 30.0)
    assert d.tolist() == [250.0, 300.0, 200.0, 30.0, 10.0]   # node 4 not edited: left below the floor
    assert rec == {"file": str(f), "n_edits": 4, "n_nodes_changed": 4, "n_clamped_to_floor": 1, "floor_m": 30.0}
    f.write_text(json.dumps({"mesh": "edited-01.npz", "n_nodes": 5, "edits": edits[::-1]}))
    assert apply_node_edits(d0, f, 30.0)[0].tolist() == [300.0, 300.0, 200.0, 30.0, 10.0]   # the order matters
    f.write_text(json.dumps({"mesh": "edited-01.npz", "n_nodes": 6, "edits": edits}))
    with pytest.raises(ValueError, match="another mesh"):
        apply_node_edits(d0, f, 30.0)
