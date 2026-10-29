"""UI shared part: the config through the core's schema, the layer fields and the limited h of the preview, the
preview grid (a window polygon across 180°), the mesh view at the antimeridian and the poles."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("tomli_w")
shapefile = pytest.importorskip("shapefile")

from ui_helpers import write_tiny_mesh  # noqa: E402

from meshom import config  # noqa: E402
from meshom.geo import lonlat_to_xyz, xyz_to_lonlat  # noqa: E402
from ui.common import coast, meshview, scenario  # noqa: E402

LAYERS = """
[mesh]
method = "m01_isotropic"
[size]
[[size.layers]]
kind = "constant"
h_m = 100000.0
[[size.layers]]
kind = "latitude"
enabled = false
h_m = 100000.0
h_min_m = 1000.0
[[size.layers]]
kind = "zone"
h_m = 10000.0
transition_m = 500000.0
polygon = [[170.0, -5.0], [-170.0, -5.0], [-170.0, 5.0], [170.0, 5.0]]
[[size.layers]]
kind = "zone"
name = "being drawn"
h_m = 1000.0
transition_m = 0.0
polygon = [[0.0, 0.0], [10.0, 0.0]]
"""


def test_config_round_trip(tmp_path):
    cfg = scenario.load()
    scenario.save(cfg, tmp_path / "c.toml")
    assert config.to_dict(scenario.load(tmp_path / "c.toml")) == config.to_dict(cfg)
    assert [ly.kind for ly in cfg.size.layers] == ["constant", "coast_distance", "zone"]


def test_layer_fields_skip_disabled_and_unfinished():
    cfg = config.loads(LAYERS)
    names, index, fields = scenario.layer_fields(cfg)
    assert names == ["constant", "zone"] and index == [0, 2]
    h = scenario.h_field(cfg)
    pts = lonlat_to_xyz(np.array([180.0, 0.0, 0.0]), np.array([0.0, 0.0, 80.0]))
    assert h(pts) == pytest.approx([10e3, 100e3, 100e3])        # the disabled latitude layer (1 km) does not win
    assert scenario.layer_fields(cfg)[2][1] is fields[1]          # cached


STEP = """
[mesh]
method = "m01_isotropic"
[size]
grad_max = 0.1
[[size.layers]]
kind = "constant"
h_m = 400000.0
[[size.layers]]
kind = "zone"
h_m = 200000.0
transition_m = 0.0
polygon = [[170.0, -5.0], [-170.0, -5.0], [-170.0, 5.0], [170.0, 5.0]]
"""


def test_limited_preview_and_cache():
    """A 200 -> 400 km step: the limited preview is the core's limiter (graded to |grad h| = 0.1), built once."""
    cfg = config.loads(STEP)
    p0 = scenario.preview(cfg, None, step_deg=2.0)
    p1 = scenario.preview(cfg, None, step_deg=2.0, limited=True)
    p2 = scenario.preview(cfg, None, step_deg=2.0, limited=True)
    assert p1.limiter["cached"] is False and p2.limiter["cached"] is True and p1.limiter["limiter"]["nodes_lowered"] > 0
    dx = p1.limiter["limiter"]["spacing_m"]                       # bilinear between limiter nodes: <= g dx above
    assert np.all(p1.h <= p0.h + 0.1 * dx) and p1.h.min() == pytest.approx(200e3) and p0.grad_max > 0.5
    assert 0.05 < p1.grad_max < 0.12                              # 2° differences across the graded ramp
    big = replace(cfg, mesh=replace(cfg.mesh, scale=4.0))
    assert scenario.preview(big, None, step_deg=2.0).h.max() == pytest.approx(1600e3)


@pytest.fixture(scope="module")
def gshhs():
    p = scenario.resolve(scenario.load().coast.source)
    if not p.exists():
        pytest.skip(f"coastline not found: {p}")
    return coast.load(str(p), 200.0)


def test_preview_coarse(gshhs):
    p = scenario.preview(scenario.load(), gshhs, step_deg=2.0)
    assert p.h.shape == (90, 180)
    assert 0.2 < p.water.mean() < 0.9
    assert p.h[p.water].min() == pytest.approx(8e3, rel=0.5) and p.h.max() == pytest.approx(100e3)
    assert 1e4 < p.n_nodes_est < 1e7 and p.grad_max > 0


def test_preview_window_polygon_across_180(gshhs):
    """The window polygon is lon-lat (the core's polygon tests take unit vectors): the 500 of /api/preview/window."""
    p = scenario.preview(scenario.load(), gshhs, 1.0, (170.0, 200.0, 55.0, 72.0),
                         [[172, 58], [-165, 58], [-165, 70], [172, 70]])
    assert p.h.shape == (17, 30) and 0.3 < p.inside.mean() < 0.9
    assert p.inside[5, 15] and not p.inside[0, 0] and not p.water[~p.inside].any()


def test_coast_polylines_skip_antimeridian_cut(gshhs):
    for pl in gshhs.polylines():
        cut = (np.abs(pl[:-1, 0]) == 180) & (np.abs(pl[1:, 0]) == 180)
        assert not cut.any()


def test_meshview_unwrapped_antimeridian_and_poles(tmp_path):
    mv = meshview.load(str(write_tiny_mesh(tmp_path / "m")))
    ll, starts, eid = mv.unwrapped()
    for a, b in zip(starts[:-1], starts[1:]):
        lon = ll[a:b, 0]
        assert lon.max() - lon.min() <= 40.0 + 1e-6                      # no element spans the map
    ov = mv.overview(step_deg=30.0)                                    # pixels no smaller than the 20° elements
    assert ov.shape == (6, 12) and np.isfinite(ov).all()               # no empty pixel, polar rows included
    assert len(mv.window(180.0, 0.0, 2000.0)) > 0      # centroids at ±10°N, ±170°


def test_overview_fills_pixels_inside_elements_larger_than_a_pixel(tmp_path):
    """A pixel inside an element shows that element's value even when no centroid falls into it (the 0.5° overview of
    a 150 km mesh left 63-69 % of the open ocean empty); a pixel in a hole of the mesh stays empty."""
    mv = meshview.load(str(write_tiny_mesh(tmp_path / "m")))                 # 20° elements, a closed sphere
    assert np.isfinite(mv.overview()).all()
    with np.load(tmp_path / "m" / "mesh.npz") as z:
        pts, tris, quads = z["points"], z["tris"], z["quads"]
    lon, lat = xyz_to_lonlat(pts[quads].mean(axis=1))
    keep = ~((lon > 0) & (lon < 60) & (lat > 0) & (lat < 40))               # a 60° x 40° hole
    (tmp_path / "h").mkdir()
    np.savez(tmp_path / "h" / "mesh.npz", points=pts, tris=tris, quads=quads[keep], node_flag=np.zeros(len(pts), np.int8))
    ov = meshview.load(str(tmp_path / "h")).overview()                       # rows south to north, columns from -180°
    px = lambda lo, la: ov[int((la + 90) / 0.5), int((lo + 180) / 0.5)]      # noqa: E731
    assert np.isnan(px(30.0, 20.0)) and np.isfinite(px(-100.0, 20.0)) and np.isfinite(px(30.0, -30.0))


def test_meshview_load_from_threads_builds_the_overlay_once(tmp_path, monkeypatch):
    """The page asks for a new mesh's info, overview and elements at once (three server threads): the mesh is read and
    its overlay computed once, the others wait and get the same MeshView (three overlays of a 3.9 M node mesh at once
    took the server past its 7 GB)."""
    import shutil
    import threading
    import time
    from ui.common import prepare_mesh
    src = write_tiny_mesh(tmp_path / "src") / "mesh.npz"
    run = tmp_path / "run"
    run.mkdir()
    shutil.copy(src, run / "raw.npz")                        # raw.npz: its overlay goes to the cache, not next to it
    monkeypatch.setattr(meshview, "CACHE", tmp_path / "cache")
    calls, real = [], prepare_mesh.write_overlay

    def slow(*a, **k):
        calls.append(1)
        time.sleep(0.3)
        return real(*a, **k)
    monkeypatch.setattr(prepare_mesh, "write_overlay", slow)
    got = []
    ts = [threading.Thread(target=lambda: got.append(meshview.load(run))) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(calls) == 1 and len(got) == 3 and got[0] is got[1] is got[2]


def test_preview_with_every_layer_off_says_so():
    """Every [[size.layers]] switched off ("use this layer"): no h to show; the page gets that in words, not numpy's
    "need at least one array to stack"."""
    c = scenario.load()
    c = replace(c, size=replace(c.size, layers=[replace(ly, enabled=False) for ly in c.size.layers]))
    with pytest.raises(ValueError, match="every h layer is off"):
        scenario.preview(c, None, 2.0)


def test_natural_earth_countries_are_not_a_coast_source(tmp_path):
    """A Natural Earth countries file (admin_0) is not a land mask: every border between two countries would be a
    coast, and meshom.coast cannot join its pieces at 180° (Russia).
    It is listed, not usable; Natural Earth land polygons are."""
    import shapefile
    for name in ("ne_50m_admin_0_countries", "ne_50m_land"):
        with shapefile.Writer(str(tmp_path / name), shapeType=shapefile.POLYGON) as w:
            w.field("name", "C")
            w.poly([[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]])
            w.record("x")
    by = {d["name"]: d for d in coast.sources(tmp_path / "ne_50m_land.shp")}
    assert by["ne_50m_admin_0_countries.shp"]["usable"] is False
    assert "border" in by["ne_50m_admin_0_countries.shp"]["note"]
    assert by["ne_50m_land.shp"]["usable"] is True


MODES = """
[mesh]
method = "m01_isotropic"
[size]
grad_max = 0.1
[[size.layers]]
kind = "constant"
h_m = 400000.0
[[size.layers]]
kind = "zone"
name = "fine"
h_m = 200000.0
transition_m = 0.0
polygon = [[-100.0, -10.0], [-80.0, -10.0], [-80.0, 10.0], [-100.0, 10.0]]
[[size.layers]]
kind = "zone"
name = "smoothed"
mode = "smooth"
smooth_m = 1500000.0
transition_m = 0.0
polygon = [[-110.0, -30.0], [-70.0, -30.0], [-70.0, 30.0], [-110.0, 30.0]]
[[size.layers]]
kind = "zone"
name = "coarse box"
mode = "set"
h_m = 800000.0
transition_m = 0.0
polygon = [[0.0, -20.0], [40.0, -20.0], [40.0, 20.0], [0.0, 20.0]]
"""


SET_FIRST = """
[mesh]
method = "m01_isotropic"
[size]
grad_max = 0.1
[[size.layers]]
kind = "constant"
h_m = 400000.0
[[size.layers]]
kind = "zone"
name = "coarse box"
mode = "set"
h_m = 800000.0
transition_m = 0.0
polygon = [[0.0, -20.0], [40.0, -20.0], [40.0, 20.0], [0.0, 20.0]]
[[size.layers]]
kind = "zone"
name = "fine"
h_m = 200000.0
transition_m = 0.0
polygon = [[-100.0, -10.0], [-80.0, -10.0], [-80.0, 10.0], [-100.0, 10.0]]
"""


def test_preview_applies_zone_modes_in_order():
    """Zone modes "set" and "smooth" (S34) act on the running h in file order: the preview is meshom.size.layered's h
    (a set zone, last, raises h above the constant layer inside; the smooth zone softens the fine box's step), the map of
    "which layer sets h" names the layer that changed h last (the base layer where none did: Layered.trace), and
    h_field agrees with the preview; a set zone placed before min zones keeps its h (S34)."""
    cfg = config.loads(MODES)
    p = scenario.preview(cfg, None, step_deg=2.0)
    at = lambda lon, lat: (np.abs(p.lat - lat).argmin(), np.abs(p.lon - lon).argmin())   # noqa: E731
    assert p.h[at(20, 0)] == pytest.approx(800e3) and p.names[p.winner[at(20, 0)]] == "coarse box"
    assert p.h[at(150, 0)] == pytest.approx(400e3) and p.names[p.winner[at(150, 0)]] == "constant"
    inner = p.h[at(-91, 1)]
    assert 200e3 < inner < 400e3 and p.names[p.winner[at(-91, 1)]] == "smoothed"   # the step averaged over 1500 km
    lon, lat = p.lon[at(-91, 1)[1]], p.lat[at(-91, 1)[0]]
    assert scenario.h_field(cfg)(lonlat_to_xyz(np.array([lon]), np.array([lat])))[0] == pytest.approx(inner)
    first = config.loads(SET_FIRST)
    q = scenario.preview(first, None, step_deg=2.0)                  # the set zone before the min zones: kept (S34)
    assert q.h[at(20, 0)] == pytest.approx(800e3) and q.names[q.winner[at(20, 0)]] == "coarse box"
    assert q.names[q.winner[at(150, 0)]] == "constant"                # the base h, set by no step


def write_grid(path):
    """A small regular lon-lat elevation grid (cell centres, 1°): elevation = 100 lat - lon (positive up)."""
    import netCDF4
    lat, lon = np.arange(-90.0, 90.5, 1.0), np.arange(-180.0, 180.0, 1.0)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        ds.createVariable("elevation", "i2", ("lat", "lon"))[:] = (100 * lat[:, None] - lon[None, :]).astype(np.int16)
    return path


def test_profile_points_along_great_circle_legs_and_the_grid_under_them(tmp_path):
    """The Tools window's profile: n points evenly spaced along the polyline's great-circle legs (a quarter of the
    equator is pi R / 2 long; the second leg runs up a meridian), and the bathymetry grid's nearest cell under each
    point, the longitude wrapping at 180°."""
    from meshom.geo import R_EARTH_M
    from ui.common import profile
    lon, lat, d = profile.sample_line([[0.0, 0.0], [90.0, 0.0], [90.0, 90.0]], 181)
    assert d[-1] == pytest.approx(np.pi * R_EARTH_M) and np.allclose(np.diff(d), d[-1] / 180)
    assert np.allclose(lon[:91], np.arange(91)) and np.allclose(lat[:91], 0, atol=1e-9)
    assert np.allclose(lat[90:], np.arange(91), atol=1e-6) and np.allclose(lon[90:-1], 90)
    g = write_grid(tmp_path / "z.nc")
    z = profile.elevation_at(g, np.array([10.2, -30.0, 179.8]), np.array([20.4, -5.0, 0.0]))
    assert z.tolist() == [100 * 20 - 10, 100 * -5 + 30, 0 + 180]
    assert not profile.land_at(None, lon, lat).any()


def test_a_strait_narrower_than_a_preview_cell_stays_water(tmp_path):
    """The h preview's land: a cell is land only when it holds no water at all (the coast on a grid PREVIEW_SUB times
    finer): a 0.1° channel between two land masses shows in a 0.5° preview (Gibraltar, 14 km, was drawn land at
    0.5°); the node estimate counts the water fraction of each cell."""
    from scipy.spatial import cKDTree
    from meshom.geo import lonlat_to_xyz
    west = np.array([[-5.0, -5.0], [0.0, -5.0], [0.0, 5.0], [-5.0, 5.0], [-5.0, -5.0]])
    east = np.array([[0.1, -5.0], [5.0, -5.0], [5.0, 5.0], [0.1, 5.0], [0.1, -5.0]])
    pts = np.vstack([west, east])
    c = coast.Coast([west, east], np.array([1e5, 1e5]), cKDTree(lonlat_to_xyz(pts[:, 0], pts[:, 1])), len(pts),
                    [np.ones(4, bool), np.ones(4, bool)])
    cfg = config.loads(STEP)
    p = scenario.preview(cfg, c, step_deg=0.5, bounds=(-2.0, 2.0, -2.0, 2.0))
    col = np.abs(p.lon - 0.25).argmin()                       # the cell [0, 0.5] holds the channel [0, 0.1]
    assert p.water[:, col].all() and not p.water[:, np.abs(p.lon - 1.25).argmin()].any()
    full = scenario.preview(cfg, None, step_deg=0.5, bounds=(-2.0, 2.0, -2.0, 2.0))
    assert p.n_nodes_est < 0.1 * full.n_nodes_est              # mostly land: only the channel's water counts


def test_a_blend_zone_counts_once_both_polygons_are_drawn():
    """Zone mode "blend" (S37) needs its polygon and its outer polygon: while the page draws either (fewer than 3
    vertices) the preview leaves the zone out, as it leaves out a zone being drawn; the orca layer (B101) is a layer
    like the others."""
    sq = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    big = [[-10.0, -10.0], [20.0, -10.0], [20.0, 20.0], [-10.0, 20.0]]
    cfg = config.loads("""
[mesh]
method = "m01_isotropic"
[size]
[[size.layers]]
kind = "constant"
h_m = 100000.0
[[size.layers]]
kind = "orca"
orca = "ORCA2"
h_min_m = 50000.0
""")
    zone = config.ZoneLayer(transition_m=0.0, h_m=20000.0, polygon=sq, mode="blend", outer=[])
    c = replace(cfg, size=replace(cfg.size, layers=cfg.size.layers + [zone]))
    assert [i for i, _ in scenario.active(c)] == [0, 1]
    zone.outer = big
    assert [i for i, _ in scenario.active(c)] == [0, 1, 2]
