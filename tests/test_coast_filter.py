"""The coast filter (decisions C16, C40-C42): opening of the water at h and closing at h/2, strait corridors
protected, the closing never splits land or joins waters, cut-off basins dropped (`meshom run` with the filter:
test_coast_filter_run.py); the signed distance's side is exact and a neck thinner than the point spacing keeps the land
in one piece (C43, C40)."""
import numpy as np

from meshom.cases import ArcDistanceField
from meshom.coast import Coast
from meshom.coast_filter import fields, filter_coast
from meshom.domain import Domain, crossing_chords, in_water, loop_edges
from meshom.geo import R_EARTH_M, arc_length, lonlat_to_xyz, xyz_to_lonlat
from meshom.size import Uniform
from meshom.straits import Strait
from tests.helpers import land_loop, local_polygon

LON0, LAT0 = 20.0, 10.0


def land(xy_km) -> np.ndarray:
    return land_loop(LON0, LAT0, xy_km)


def disc(cx, cy, r, n=64) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return land(np.c_[cx + r * np.cos(t), cy + r * np.sin(t)])


def at(x_km, y_km) -> np.ndarray:
    return local_polygon(LON0, LAT0, np.array([[x_km * 1e3, y_km * 1e3]]))


def wet(coast: Coast, *xy) -> list[bool]:
    dom = Domain(coast.loops)
    return [bool(in_water(dom, at(x, y))[0]) for x, y in xy]


def features() -> list[np.ndarray]:
    """Land x in [-1000, 1000] km, y in [-500, 0] km; on its north coast a bay 30 km wide (x = -600) and one 150 km
    wide (x = -450), both 200 km deep, a bay 300 km wide (x = -300..0) and a cape 20 km wide, 200 km long (x = 400);
    islands of 40 km and 400 km diameter to the north."""
    poly = [(-1000, -500), (1000, -500), (1000, 0), (410, 0), (410, 200), (390, 200), (390, 0), (0, 0), (0, -200),
            (-300, -200), (-300, 0), (-375, 0), (-375, -200), (-525, -200), (-525, 0), (-585, 0), (-585, -200),
            (-615, -200), (-615, 0), (-1000, 0)]
    return [land(poly), disc(-500, 400, 20), disc(600, 600, 200)]


def two_blocks(gap_km=None, neck_km=None, neck_len_km=300.0) -> list[np.ndarray]:
    """Two land blocks of 600 x 600 km, gap_km apart (a strait along y) or joined by a neck neck_km wide."""
    if gap_km is not None:
        g = gap_km / 2
        return [land([(-600 - g, -300), (-g, -300), (-g, 300), (-600 - g, 300)]),
                land([(g, -300), (600 + g, -300), (600 + g, 300), (g, 300)])]
    L, w = neck_len_km / 2, neck_km / 2
    return [land([(-600 - L, -300), (-L, -300), (-L, -w), (L, -w), (L, -300), (600 + L, -300), (600 + L, 300),
                  (L, 300), (L, w), (-L, w), (-L, 300), (-600 - L, 300)])]


def valid(c: Coast, h) -> None:
    """The filtered loops resampled at h: >= 3 points each, no chord crossing another, no edge below h/2."""
    lp = c.loops_at(h)
    assert all(len(x) >= 3 for x in lp)
    e = loop_edges(lp)
    P = np.vstack(lp)
    assert not crossing_chords(P, e[:, 0], e[:, 1]).any()
    s = R_EARTH_M * arc_length(P[e[:, 0]], P[e[:, 1]]) / h(P[e[:, 0]])
    assert s.min() > 0.5, s.min()


def test_bays_capes_and_islands_at_the_filter_scale():
    """h = 100 km: water narrower than 2 h closes (bays of 30 and 150 km), wider stays (300 km); land narrower than
    h goes (a 20 km cape, a 40 km island), wider stays (a 400 km island); the rest of the coast is unchanged."""
    h = Uniform(100e3)
    c, info = filter_coast(Coast(features(), np.zeros(3)), h)
    assert wet(c, (-600, -100), (-450, -100), (-150, -100), (400, 100), (-500, 400), (600, 600)) == \
        [False, False, True, True, True, False]
    assert wet(c, (0, -400), (0, 300), (800, -60), (800, 60)) == [False, True, False, True]
    assert info["n_loops"] == 2                        # the block and the big island
    valid(c, h)


def test_land_between_h_and_2h_stays():
    """C41: the closing radius is h/2, so land narrower than h goes and an island of 150 km (between h and 2 h at
    h = 100 km) stays; a closing at h would remove it."""
    c, info = filter_coast(Coast(features() + [disc(-300, 700, 75)], np.zeros(4)), Uniform(100e3))
    assert wet(c, (-300, 700), (-300, 640), (-500, 400)) == [False, False, True]
    assert info["n_loops"] == 3                        # the block and the islands of 400 and 150 km


def test_a_strait_is_kept_open_by_its_corridor():
    """A 20 km strait at h = 100 km closes (S24), unless it is a listed strait: its corridor stays water."""
    h = Uniform(100e3)
    blocks = Coast(two_blocks(gap_km=20.0), np.zeros(2))
    probes = [(0, -250), (0, 0), (0, 250)]
    closed, _ = filter_coast(blocks, h)
    assert wet(closed, *probes) == [False, False, False]
    path = local_polygon(LON0, LAT0, np.array([[0.0, -400e3], [0.0, 400e3]]))
    opened, info = filter_coast(blocks, h, [Strait("test", path, 20e3, 10e3)])
    assert wet(opened, *probes) == [True, True, True]
    assert info["n_loops"] == 2
    valid(opened, h)


def test_a_passage_resolved_by_h_stays_open():
    """A 60 km strait closes at h = 100 km, and stays open where h is refined to 12 km along it (an automatic strait
    zone, W / 5), without a corridor."""
    blocks = Coast(two_blocks(gap_km=60.0), np.zeros(2))
    probes = [(0, -250), (0, 0), (0, 250)]
    assert wet(filter_coast(blocks, Uniform(100e3))[0], *probes) == [False, False, False]
    p, q = local_polygon(LON0, LAT0, np.array([[0.0, -500e3], [0.0, 500e3]]))
    h = ArcDistanceField(tuple(p), tuple(q), 30e3, 12e3, 100e3, 0.1)
    c, _ = filter_coast(blocks, h)
    assert wet(c, *probes) == [True, True, True]
    valid(c, h)


def test_the_closing_keeps_an_isthmus():
    """A 40 km neck between two land masses at h = 100 km is thinner than the closing removes (h), but removing it
    would split the land (Panama, Kra): it stays (C40); a cape of the same width goes (the first test)."""
    c, info = filter_coast(Coast(two_blocks(neck_km=40.0), np.zeros(1)), Uniform(100e3))
    assert wet(c, (0, 0), (-120, 0), (120, 0), (0, 150), (0, -150)) == [False, False, False, True, True]
    assert info["n_kept_pieces"] >= 1 and info["n_loops"] == 1


def test_a_neck_thinner_than_the_point_spacing_keeps_the_land_whole():
    """C40 against the source: necks of 5, 2 and 1 km (300 km long) at h = 100 km, thinner than the point spacing
    (about 12 km): no chain of land points runs along them, so the sampled land fell in two before the closing's
    guard and the filter split the land mass (2 loops). The two blocks' land is joined by a bridge along the neck,
    which the closing then keeps (C40): one land mass, water on both sides of the neck."""
    for w in (5.0, 2.0, 1.0):
        c, info = filter_coast(Coast(two_blocks(neck_km=w), np.zeros(1)), Uniform(100e3))
        assert info["n_loops"] == 1 and info["n_bridged_pieces"] >= 1, (w, info["n_loops"])
        assert wet(c, (-400, 0), (400, 0), (0, 150), (0, -150)) == [False, False, True, True]
        valid(c, Uniform(100e3))


def test_a_dense_islet_does_not_flip_the_inland_side():
    """C43: the side of the signed distance is the source polygon's at every point. A mainland whose straight shore
    has vertices 5 km apart and a 40 m islet of 100 vertices 1 m off it, h = 100 km: the islet's
    samples took all six nearest-sample slots of points 100-170 km inland, whose side then came from the islet:
    333 land points became water and a point 131 km inland was water in the filtered coast."""
    xy = np.r_[np.c_[np.arange(-500, 501, 5), np.zeros(201)], [[500, -500], [-500, -500]]] * 1e3
    main = local_polygon(0.0, 0.0, xy)
    h = Uniform(100e3)
    f0, _ = fields(Coast([main], np.zeros(1)), h)
    P, s0 = f0["P"], f0["s0"]
    lo, la = xyz_to_lonlat(P)
    cand = np.flatnonzero((s0 < -105e3) & (s0 > -170e3) & (np.abs(lo) < 2) & (la < 0))
    x = R_EARTH_M * np.tan(np.radians(lo[cand]))
    j = cand[np.argmax(np.abs((x + 2500) % 5000 - 2500))]        # the farthest from a shore vertex along the shore
    x0 = R_EARTH_M * np.tan(np.radians(lo[j]))
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    islet = local_polygon(0.0, 0.0, np.c_[x0 + 20 * np.cos(t), 21 + 20 * np.sin(t)][::-1])
    raw = Coast([main, islet], np.zeros(2))
    f, _ = fields(raw, h)
    np.testing.assert_array_equal(f["P"], P)
    far = np.abs(f["s0"]) > 1e3              # in_water maps the long straight edges to chords: off by < 1 km there
    assert ((f["s0"] > 0) == in_water(Domain(raw.loops), P))[far].all()
    c, _ = filter_coast(raw, h)
    assert not in_water(Domain(c.loops), P[j:j + 1])[0]
    assert not in_water(Domain(c.loops_at(h)), P[j:j + 1])[0]


def test_a_basin_cut_off_by_the_opening_is_dropped_unless_kept():
    """A 1000 km inland sea behind a 20 km mouth at h = 100 km: the opening closes the mouth; the sea becomes land
    unless keep_basins_km2 keeps it (the cut's rule)."""
    ring = [(-800, -800), (800, -800), (800, 800), (10, 800), (10, 500), (500, 500), (500, -500), (-500, -500),
            (-500, 500), (-10, 500), (-10, 800), (-800, 800)]
    c0 = Coast([land(ring)], np.zeros(1))
    dropped, info = filter_coast(c0, Uniform(100e3))
    assert wet(dropped, (0, 0), (0, 1000)) == [False, True] and info["n_basins_dropped"] == 1
    kept, info = filter_coast(c0, Uniform(100e3), keep_basins_km2=5e5)
    assert wet(kept, (0, 0), (0, 1000)) == [True, True] and len(info["basins_km2"]) == 2


def test_the_antimeridian_and_the_pole_need_no_special_case():
    """An island across 180 deg and an Antarctica-like cap around the South Pole keep their shape through the filter."""
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    isl = np.c_[180.0 + 3.0 * np.cos(t), 40.0 + 2.0 * np.sin(t)][::-1]        # clockwise: land on the right
    cap = np.c_[np.linspace(-180, 180, 720, endpoint=False), np.full(720, -70.0)]   # eastward: water (north) left
    c, _ = filter_coast(Coast([lonlat_to_xyz(isl[:, 0], isl[:, 1]), lonlat_to_xyz(cap[:, 0], cap[:, 1])],
                              np.zeros(2)), Uniform(50e3))
    dom = Domain(c.loops)
    q = lonlat_to_xyz(np.array([180.0, 179.0, -179.0, 0.0, 0.0, 100.0]),
                      np.array([40.0, 40.0, 40.0, -89.0, -60.0, -71.0]))
    assert in_water(dom, q).tolist() == [False, False, False, False, True, False]
    lat = xyz_to_lonlat(max(c.loops, key=len))[1]
    assert np.abs(lat + 70.0).max() < 0.05                                  # the cap's edge: within 5 km
