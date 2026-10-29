"""Straits (meshom.straits): the forced corridor of the land cut, the strait report, the config and the built-in list."""
import numpy as np

from meshom import config
from meshom.domain import Domain
from meshom.mesh import Mesh
from meshom.straits import Strait, strait_report, straits_of
from tests.helpers import local_polygon, split_quads, structured_patch


def _tri_patch(nx: int, ny: int, dx: float) -> Mesh:
    m = structured_patch(nx, ny, dx)
    return split_quads(m, range(m.n_quads))


def _path(x0_m: float, x1_m: float) -> np.ndarray:
    return local_polygon(0.0, 0.0, [[x0_m, 0.0], [0.5 * (x0_m + x1_m), 0.0], [x1_m, 0.0]])


def test_corridor_keeps_a_strait_open():
    """A 5 km gap in a 40 km land wall across a 10 km mesh: the cut closes it (two basins, the smaller one dropped);
    with the strait (width 5 km, h 10 km, 2 cells: a 2 x 1.15 x 10 + 10 / sqrt 3 = 28.8 km corridor) the gap's
    elements are water, both sides stay one basin and every section of the inner path is wet."""
    from meshom.run import cut_land

    mesh = _tri_patch(60, 40, 10e3)
    wall = [[[-20e3, -250e3], [-20e3, -2.5e3], [20e3, -2.5e3], [20e3, -250e3]],      # clockwise: water on the left
            [[-20e3, 2.5e3], [-20e3, 250e3], [20e3, 250e3], [20e3, 2.5e3]]]
    land = Domain([local_polygon(0.0, 0.0, w) for w in wall], "coast")
    s = Strait("gap", _path(-100e3, 100e3), width_m=5e3, h_m=10e3, cells=2)
    assert abs(s.corridor_m - (23e3 + 1e4 / np.sqrt(3))) < 1e-6

    closed, info0 = cut_land(mesh, land)
    r0 = strait_report(closed, [s])[0]
    # the two basins are mirror images (2240 elements, 111 812 km^2 each): which one is the largest by area is decided
    # by round-off, so only one end stays wet
    assert r0["closed_sections"] > 0 and not r0["connected"] and sorted(r0["ends_wet"]) == [False, True]

    opened, info = cut_land(mesh, land, None, [s])
    r = strait_report(opened, [s])[0]
    assert info["n_forced_wet"] > 0 and info["n_corridor_kept"] >= 0.9 * info["n_corridor"]   # final pass: ragged edge
    assert r["closed_sections"] == 0 and r["connected"] and r["cells_min"] >= 1.5, r
    assert opened.n_elems > 1.9 * closed.n_elems


def _split_case():
    """A 500 x 400 km patch of 10 km triangles: basin W | strait A (5 km gap in a 40 km wall at x = -140..-100 km) |
    middle basin | strait B (the same at x = 100..140 km) | basin E; the middle basin is split by a wall at x = 10..40
    km whose only gap (8 km at y = 100 km) the cut closes. Each strait path stays wet end to end, but B's water is cut
    off from the main water (W + the western middle)."""
    mesh = _tri_patch(50, 40, 10e3)

    def wall(x0, x1, y0, y1):
        return [[x0, -250e3], [x0, y0], [x1, y0], [x1, -250e3]], [[x0, y1], [x0, 250e3], [x1, 250e3], [x1, y1]]

    walls = [*wall(-140e3, -100e3, -2.5e3, 2.5e3), *wall(100e3, 140e3, -2.5e3, 2.5e3), *wall(10e3, 40e3, 96e3, 104e3)]
    land = Domain([local_polygon(0.0, 0.0, w) for w in walls], "coast")
    a = Strait("A", _path(-180e3, -60e3), width_m=5e3, h_m=10e3)
    b = Strait("B", _path(60e3, 180e3), width_m=5e3, h_m=10e3)
    neck = Strait("neck", local_polygon(0.0, 0.0, [[-20e3, 100e3], [25e3, 100e3], [70e3, 100e3]]), width_m=8e3,
                  h_m=10e3)
    return mesh, land, a, b, neck


def test_report_flags_a_basin_split_between_two_straits():
    """Basin to basin: B's path is wet and in one component end to end (the old per-path check said "connected"),
    but that component is not the main water: B is not connected. A strait across the neck (the fix of the
    Marmara split, decisions C27) makes the water one component and both straits connected."""
    from meshom.run import cut_land

    mesh, land, a, b, neck = _split_case()
    split, info = cut_land(mesh, land, 1000.0, [a, b])
    ra, rb = strait_report(split, [a, b])
    assert rb["ends_wet"] == [True, True] and rb["closed_sections"] == 0, rb
    assert ra["connected"] and not rb["connected"] and not rb["ok"], (ra, rb)    # the per-path check: both connected
    assert ra["ends_component"] == [0, 0] and rb["ends_component"] == [1, 1] and info["n_components"] == 2

    joined, info = cut_land(mesh, land, 1000.0, [a, b, neck])
    rs = strait_report(joined, [a, b, neck])
    assert info["n_components"] == 1
    assert all(r["connected"] and r["ends_component"] == [0, 0] for r in rs), rs


def test_water_components_lists_the_split():
    """water_components: label 0 = the largest area; per component elements, nodes, area, box; the split case has
    W + western middle (220 km x 400 km of water less the strait corridor's share) and eastern middle + E."""
    from meshom.run import cut_land
    from meshom.straits import water_components

    mesh, land, a, b, _ = _split_case()
    split, _ = cut_land(mesh, land, 1000.0, [a, b])
    # the elements reversed: the raw first component is the smaller eastern one, yet component 0 is the largest
    rev = Mesh(split.points, split.tris[::-1].copy(), split.quads[::-1].copy(), split.node_flag,
               None if split.tri_zone is None else split.tri_zone[::-1].copy())
    lab_r, comps_r = water_components(rev)
    assert comps_r[0]["area_km2"] > comps_r[1]["area_km2"] and comps_r[0]["box"][0] < -2.2
    assert lab_r[-1] == 0 and lab_r[0] == 1                     # the first triangle is in the east (rev order)
    lab, comps = water_components(split)
    assert len(comps) == 2 and [c["id"] for c in comps] == [0, 1]
    assert comps[0]["area_km2"] > comps[1]["area_km2"] > 0.5 * comps[0]["area_km2"]
    assert sum(c["n_elems"] for c in comps) == split.n_elems == len(lab)
    assert sum(c["n_nodes"] for c in comps) == split.n_nodes          # no node is shared across components
    w0, e0, s0, n0 = comps[0]["box"]
    w1, e1, s1, n1 = comps[1]["box"]
    assert w0 < -2.2 and e0 < 0.4 and w1 > 0.3 and e1 > 2.2 and s0 < -1.7 and n1 > 1.7, comps
    _, one = water_components(cut_land(mesh, land, None, [a, b])[0])
    assert len(one) == 1
    from meshom.straits import _lon_box
    assert _lon_box(np.array([170.0, 175.0, -175.0, -170.0])) == (170.0, -170.0)      # across 180: west > east
    assert _lon_box(np.linspace(-180.0, 179.0, 360)) == (-180.0, 179.0)              # all around: min, max


def test_report_counts_cells_across_a_channel():
    """A channel of 4 rows of 5 km squares split into right triangles: wet width 20 km, mean edge 5 (2 + sqrt 2) / 3
    km, so 3.51 cells; both ends in one component; the smallest element depth on the inner path."""
    mesh = _tri_patch(40, 4, 5e3)
    s = Strait("channel", _path(-90e3, 90e3), width_m=20e3, h_m=5e3)
    x = np.linspace(-100e3, 100e3, 41)
    depth = np.tile(100.0 + x / 1e3, 5)                    # node (j, i): row-major, 100 m + 1 m per km eastwards
    r = strait_report(mesh, [s], depth)[0]
    assert abs(r["width_min_m"] - 20e3) <= 1e3
    assert abs(r["cells_min"] - 20.0 / (5.0 * (2 + np.sqrt(2)) / 3)) < 0.2, r
    assert r["connected"] and r["ends_wet"] == [True, True] and r["closed_sections"] == 0
    assert 100.0 - 72.0 - 5.0 <= r["depth_min_m"] <= 100.0 - 72.0 + 5.0      # inner 80 % of 180 km starts at -72 km
    assert r["ok"] and r["cells_target"] == 2


def test_report_on_a_mixed_mesh():
    """C34 / C44 on a mixed mesh: the same channel of 4 rows of 5 km squares kept as quads (one split):
    the report locates path points in the quads too (it searched triangles only: 40 closed sections, ends in
    components [0, -1], not connected); wet width 20 km (19 km sampled), mean edge 5 km, so 4 cells; element depths
    of quads."""
    mesh = split_quads(structured_patch(40, 4, 5e3), [0])
    s = Strait("channel", _path(-90e3, 90e3), width_m=20e3, h_m=5e3)
    x = np.linspace(-100e3, 100e3, 41)
    r = strait_report(mesh, [s], np.tile(100.0 + x / 1e3, 5))[0]
    assert r["closed_sections"] == 0 and r["ends_component"] == [0, 0] and r["connected"], r
    assert abs(r["width_min_m"] - 20e3) <= 1e3 and abs(r["cells_min"] - 4.0) < 0.25 and r["ok"], r
    assert 100.0 - 72.0 - 5.0 <= r["depth_min_m"] <= 100.0 - 72.0 + 5.0


def test_config_loads_the_builtin_list():
    """[coast] straits = "fesom2_mesh_gen" loads the 90 straits (with Marmara_West, decisions C27); a [[straits]] entry
    overrides fields by name, drops one (enabled = false) or adds one; strait_h_min_m raises h_m; the config
    round-trips."""
    text = """
[mesh]
method = "m15_tri"
[coast]
straits = "fesom2_mesh_gen"
strait_h_min_m = 10000.0
[[size.layers]]
kind = "constant"
h_m = 100000.0
[[straits]]
name = "Gibraltar"
h_m = 20000.0
cells = 3
[[straits]]
name = "Bosphorus"
enabled = false
[[straits]]
name = "test"
path = [[0.0, 0.0], [1.0, 0.0]]
width_m = 5000.0
h_m = 1000.0
"""
    cfg = config.loads(text, "s")
    assert config.loads(config.dumps(cfg), "s") == cfg
    st = {s.name: s for s in straits_of(cfg)}
    assert len(st) == 90 and "Bosphorus" not in st and "test" in st and "Great_Belt" in st
    g = st["Gibraltar"]
    assert g.h_m == 20000.0 and g.cells == 3 and g.width_m == 14450.0 and g.depth_m == 322.0 and len(g.path) == 62
    assert st["Dardanelles"].h_m == 10000.0 and st["Dardanelles"].depth_m == 55.0 and st["test"].h_m == 10000.0
    mw = st["Marmara_West"]
    assert mw.width_m == 8150.0 and mw.h_m == 10000.0 and mw.depth_m is None and len(mw.path) == 65
    assert np.abs(mw.path[-1] - st["Dardanelles"].path[-1]).max() < 1e-3          # it continues the Dardanelles
    assert st["Oresund"].cells == 2 and abs(st["Oresund"].corridor_m - (23000.0 + 1e4 / np.sqrt(3))) < 1e-6
    assert straits_of(config.loads('[mesh]\nmethod = "m15_tri"\n')) == []


# ------------------------------------------------------------------------------------------- profiles (C45-C47)

def _gap_land():
    wall = [[[-20e3, -250e3], [-20e3, -2.5e3], [20e3, -2.5e3], [20e3, -250e3]],
            [[-20e3, 2.5e3], [-20e3, 250e3], [20e3, 250e3], [20e3, 2.5e3]]]
    return Domain([local_polygon(0.0, 0.0, w) for w in wall], "coast")


_GAP = """
[mesh]
method = "m15_tri"
[coast]
strait_profile = "widen"
strait_widen_cells = {cells}
[[straits]]
name = "gap"
path = [[-0.9, 0.0], [0.0, 0.0], [0.9, 0.0]]
width_m = 5000.0
"""


def test_widen_opens_cells_of_the_background_h():
    """widen (C45): no h_m given, h_m = the largest background h over the inner 80 % of the path (8 -> 10 km along
    lon -0.9..0.9: 9.8 km at lon 0.72), corridor max(width, (cells 1.15 + 1/sqrt 3) h); on a 10 km mesh with a 5 km
    gap the cut keeps at least `cells` cells of the background h across, and 3 cells give a wider channel than 2."""
    from meshom.geo import xyz_to_lonlat
    from meshom.run import cut_land

    def ramp(xyz):
        return 8e3 + 2e3 * (xyz_to_lonlat(xyz)[0] + 0.9) / 1.8

    s = straits_of(config.loads(_GAP.format(cells=2)), ramp)[0]
    assert s.profile == "widen" and s.cells == 2 and abs(s.h_m - 9.8e3) < 20.0, s.h_m
    assert abs(s.corridor_m - (2 * 1.15 + 1 / np.sqrt(3)) * s.h_m) < 1e-6
    mesh, land = _tri_patch(60, 40, 10e3), _gap_land()
    width = {}
    for cells in (2, 3):
        s = straits_of(config.loads(_GAP.format(cells=cells)), lambda xyz: np.full(len(xyz), 10e3))[0]
        assert s.h_m == 10e3 and s.cells == cells
        r = strait_report(cut_land(mesh, land, None, [s])[0], [s])[0]
        assert r["closed_sections"] == 0 and r["connected"] and r["width_min_m"] >= (cells - 0.05) * s.h_m, r
        width[cells] = r["width_min_m"]
    assert width[3] > width[2] + 5e3, width


_REGIONS = """
[mesh]
method = "m15_tri"
[coast]
straits = "fesom2_mesh_gen"
strait_h_min_m = 10000.0
[[strait_regions]]
name = "Baltic"
profile = "off"
box = [9.0, 31.0, 53.0, 66.0]
[[strait_regions]]
name = "Arctic"
profile = "resolve"
box = [-180.0, 180.0, 60.0, 90.0]
"""


def test_regions_select_straits_and_profiles():
    """[[strait_regions]] (C46): the first region holding the path's middle point gives the profile, "off" and no
    region leave the strait out; a box with west > east crosses 180; the polygon form."""
    bg = lambda xyz: np.full(len(xyz), 30e3)          # noqa: E731
    cfg = config.loads(_REGIONS, "r")
    assert config.loads(config.dumps(cfg), "r") == cfg
    st = {s.name: s for s in straits_of(cfg, bg)}
    assert "Oresund" not in st and "Great_Belt" not in st and "Gibraltar" not in st and "Irbe_Strait" not in st
    assert st["Nares"].profile == "resolve" and st["Nares"].h_m == 10000.0 and "Hinchinbrook" in st
    world = '\n[[strait_regions]]\nprofile = "widen"\nbox = [-180.0, 180.0, -90.0, 90.0]\n'
    st = {s.name: s for s in straits_of(config.loads(_REGIONS + world), bg)}
    assert len(st) == 90 - 4 and "Oresund" not in st                    # 4 Baltic straits: Oresund, belts, Irbe
    assert st["Gibraltar"].profile == "widen" and st["Gibraltar"].h_m == 30e3 and st["Nares"].profile == "resolve"
    dateline = """
[mesh]
method = "m15_tri"
[coast]
straits = "fesom2_mesh_gen"
[[strait_regions]]
profile = "widen"
box = [170.0, -170.0, 60.0, 75.0]
[[strait_regions]]
profile = "resolve"
polygon = [[-8.0, 33.0], [-2.0, 33.0], [-2.0, 38.0], [-8.0, 38.0]]
"""
    st = {s.name: s for s in straits_of(config.loads(dateline), bg)}
    assert set(st) == {"Long_Strait", "Gibraltar"}, set(st)
    assert st["Long_Strait"].profile == "widen" and st["Gibraltar"].profile == "resolve"


def test_per_strait_profile_wins():
    """A [[straits]] entry's own profile (C46) wins over the region and the default, and opens the strait outside
    every region; enabled = false still drops it."""
    bg = lambda xyz: np.full(len(xyz), 30e3)          # noqa: E731
    own = """
[[straits]]
name = "Gibraltar"
profile = "widen"
cells = 3
[[straits]]
name = "Nares"
profile = "widen"
[[straits]]
name = "Bosphorus"
profile = "resolve"
enabled = false
"""
    st = {s.name: s for s in straits_of(config.loads(_REGIONS + own), bg)}
    g, n = st["Gibraltar"], st["Nares"]
    assert g.profile == "widen" and g.cells == 3 and g.h_m == 30e3 and n.profile == "widen" and n.cells == 2
    assert "Bosphorus" not in st and st["Penny_Strait"].profile == "resolve"


def test_widen_gets_no_size_zone():
    """size_field (C45): a widened strait leaves h at the background, a resolved one puts its zone h_m there."""
    from meshom.geo import lonlat_to_xyz
    from meshom.run import size_field

    base = """
[mesh]
method = "m15_tri"
[coast]
straits = "fesom2_mesh_gen"
strait_h_min_m = 20000.0
strait_profile = "{p}"
[size]
grad_max = inf
[[size.layers]]
kind = "constant"
h_m = 100000.0
"""
    at = lonlat_to_xyz(np.array([-5.58]), np.array([35.94]))              # Gibraltar's middle
    info = {}
    hw = size_field(config.loads(base.format(p="widen")), None, info)(at)[0]
    hr = size_field(config.loads(base.format(p="resolve")), None)(at)[0]
    assert hw == 100000.0 and hr == 20000.0 and info["strait_profiles"] == {"resolve": 0, "widen": 90}


def test_widen_takes_a_strait_shorter_than_the_sample_step():
    """widen on a 556 m channel (shorter than BG_STEP_M): the inner 80 % is sampled at its own ends, no empty sample."""
    from meshom.size import uniform

    cfg = config.from_dict({"mesh": {"method": "m15_tri"}, "coast": {"strait_profile": "widen"},
                            "straits": [{"name": "short", "path": [[0.0, 0.0], [0.005, 0.0]], "width_m": 100.0}]})
    s = straits_of(cfg, uniform(1000.0))[0]
    assert s.h_m == 1000.0


def test_midpoint_is_half_the_arc_length_on_the_sphere():
    """The region test's middle point (C46) is the point at half the arc length, also on an unevenly sampled path
    and in either direction; a region whose west edge lies 5.6 m west of it selects the strait."""
    from meshom.geo import lonlat_to_xyz, xyz_to_lonlat
    from meshom.straits import midpoint, profile_of

    p = lonlat_to_xyz(np.array([0.0, 1.0, 4.0]), np.zeros(3))
    for q in (p, p[::-1]):
        assert abs(float(xyz_to_lonlat(midpoint(q)[None])[0][0]) - 2.0) < 1e-9
    a, b = lonlat_to_xyz(np.array([-60.0, 40.0]), np.array([70.0, 75.0]))   # one great circle, cut at 10 % / 90 %
    w = np.arccos(a @ b)
    gc = lambda f: (np.sin((1 - f) * w) * a + np.sin(f * w) * b) / np.sin(w)   # noqa: E731
    q = np.array([a, gc(0.1), gc(0.9), b])
    assert np.linalg.norm(midpoint(q) - gc(0.5)) < 1e-12 and np.linalg.norm(midpoint(q[::-1]) - gc(0.5)) < 1e-12
    r = config.StraitRegionCfg("resolve", box=[1.99995, 3.0, -1.0, 1.0])
    assert profile_of(p, None, config.CoastCfg(), [r]) == "resolve"


def test_widen_keeps_the_cells_across_on_oblique_quads():
    """On a mixed method's quads (m14 at its tolerance 0.20: squares of 12 km where h = 10 km, every edge within the
    tolerance) at 0 and 45 degrees and three lattice offsets, the widened corridor keeps `cells` cells of the edge
    length and of h across (C45; before: 1.44 cells at 45 degrees)."""
    from meshom.geo import R_EARTH_M
    from meshom.run import cut_land
    from meshom.size import uniform

    cfg = config.from_dict({"mesh": {"method": "m14_blocks"}, "coast": {"strait_profile": "widen"},
                            "straits": [{"name": "gap", "path": [[-0.9, 0.0], [0.0, 0.0], [0.9, 0.0]],
                                         "width_m": 5000.0}]})
    s = straits_of(cfg, uniform(10e3))[0]
    from meshom.straits import BAND_SAFETY, QUAD_MARGIN
    assert s.quads and abs(s.size_tol - 0.2) < 1e-12          # the corridor of the longest permitted edge e^t h
    assert abs(s.corridor_m - (s.cells * BAND_SAFETY + QUAD_MARGIN) * np.exp(0.2) * s.h_m) < 1e-6
    base = structured_patch(40, 40, 12e3)
    xy = base.points[:, 1:] / base.points[:, :1] * R_EARTH_M
    for a in (0.0, np.pi / 4):
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        for shift in (0.0, 2e3, 4e3):
            mesh = Mesh(local_polygon(0.0, 0.0, xy @ rot.T + [2e3, shift]), base.tris, base.quads, base.node_flag)
            r = strait_report(cut_land(mesh, _gap_land(), None, [s])[0], [s])[0]
            assert r["connected"] and r["closed_sections"] == 0, r
            assert r["cells_min"] >= s.cells and r["width_min_m"] >= s.cells * s.h_m, (a, shift, r)


def test_global_examples_keep_only_the_ocean_connected_water():
    """Rule C48: a global mesh keeps only the water connected to the world ocean, so no global example sets
    keep_basins_km2 (the default keeps the largest basin only)."""
    import tomllib
    from pathlib import Path

    ex = sorted((Path(__file__).parents[1] / "docs" / "examples").glob("*.toml"))
    assert ex and all("keep_basins_km2" not in tomllib.loads(f.read_text()).get("coast", {}) for f in ex)


def test_gibraltar_is_mandatory():
    """C49: with a real coast Gibraltar is opened even when no strait list is chosen (widened, at the background h),
    widened when the regions would leave it closed; an explicit enabled = false closes it; no coast source, no mandatory strait."""
    import numpy as np

    from meshom.config import loads
    from meshom.size import Uniform
    from meshom.straits import straits_of

    h = Uniform(100e3)
    base = '[mesh]\nmethod = "m15_tri"\n[coast]\nsource = "x.shp"\n'
    names = lambda cfg: {s.name: s.profile for s in straits_of(cfg, h)}
    assert names(loads(base)) == {"Gibraltar": "widen"}     # no list chosen: widened, no 900 m zone
    arctic = base + '[[strait_regions]]\nprofile = "resolve"\nbox = [-180.0, 180.0, 60.0, 90.0]\n'
    assert names(loads(arctic)) == {"Gibraltar": "widen"}
    assert names(loads(base + '[[straits]]\nname = "Gibraltar"\nenabled = false\n')) == {}
    assert names(loads('[mesh]\nmethod = "m15_tri"\n')) == {}
    assert np.isfinite(straits_of(loads(arctic), h)[0].h_m)
    # a chosen list (Gibraltar without its own profile) and regions that miss Gibraltar: widened, not closed;
    # profile = "off" on Gibraltar itself closes it
    listed = base + 'straits = "fesom2_mesh_gen"\n[[strait_regions]]\nprofile = "resolve"\nbox = [-180.0, 180.0, 60.0, 90.0]\n'
    got = names(loads(listed))
    assert got["Gibraltar"] == "widen" and all(v == "resolve" for k, v in got.items() if k != "Gibraltar")
    assert "Gibraltar" not in names(loads(listed + '[[straits]]\nname = "Gibraltar"\nprofile = "off"\n'))


def test_every_written_mesh_gets_the_water_check(tmp_path):
    """C52: write_files records the water components of every mesh it writes, in every format; a mesh of two pieces
    is not ok and names the extra piece (elements, nodes, area, box); fesom2 / fesom3 add it to the checks' stop."""
    from meshom.run import write_files, write_line

    m = structured_patch(10, 4, 10e3)
    j, i = np.divmod(np.arange(m.n_quads), 10)
    two = Mesh(m.points, m.tris, m.quads[i != 4], m.node_flag)        # column 4 dropped: 16 | 20 quads
    used = np.unique(two.quads)
    remap = np.full(m.n_nodes, -1)
    remap[used] = np.arange(len(used))
    two = Mesh(m.points[used], m.tris, remap[two.quads], m.node_flag[used])
    lev = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]
    for fmt in ("fesom-c", "fesom3"):
        rec = write_files(two, fmt, tmp_path / fmt, np.full(two.n_nodes, 25.0), lev)
        w = rec["water"]
        assert not w["connectivity_ok"] and w["n_components"] == 2, w
        c = w["components"][1]
        assert c["id"] == 1 and c["n_elems"] == 16 and c["n_nodes"] == 25 and c["box"][0] < 0.0 < c["box"][1] + 0.4
    assert "water_components" in rec["fesom3_checks"]["stop"] and rec["fesom3_checks"]["water_components"] == 2
    one = write_files(m, "fesom3", tmp_path / "one", np.full(m.n_nodes, 25.0), lev)
    assert one["water"]["connectivity_ok"] and one["water"]["n_components"] == 1
    assert "water_components" not in one["fesom3_checks"]["stop"]
    line = write_line({"mesh": "raw.npz", "out": "o", "format": "fesom3", "quality": {"n_nodes": 1, "n_tris": 0,
                       "n_quads": 1}, "irregular": {"euler_sum": 0, "euler_ok": True}, "gates_passed": 1,
                       "gates_total": 1, "gates_failed": [], **rec})
    assert "WATER NOT CONNECTED: 2 water components" in line and "16 elements" in line, line
