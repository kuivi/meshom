"""h editing: coast classes of the coast_distance layer,
the zone modes min / set / smooth on the running h, h variants ([size] from, loads_size / dumps_size)."""
from pathlib import Path

import numpy as np
import pytest

from meshom import config
from meshom.coast import Coast, read_coast
from meshom.geo import R_EARTH_M, arc_length, lonlat_to_xyz, tangent_basis
from meshom.run import background_h, layer_fields, size_field
from meshom.size import Latitude, Minimum, Uniform, zone

pytest.importorskip("tomli_w")

GSHHS = Path("/work/ab0995/a270126/projects/mesh_generation/oceanmesh/tests/GSHHS_i_L1.shp")
if not GSHHS.exists():
    GSHHS = Path(__file__).resolve().parents[2] / "oceanmesh/tests/GSHHS_i_L1.shp"

HEAD = '[mesh]\nmethod = "m01_isotropic"\n[size]\ngrad_max = inf\n'


def _ll(lon, lat):
    return lonlat_to_xyz(np.atleast_1d(np.asarray(lon, float)), np.atleast_1d(np.asarray(lat, float)))


def _layers(*tables: str) -> str:
    return HEAD + "".join("[[size.layers]]\n" + t.strip() + "\n" for t in tables)


def _cap(lon0, lat0, radius_deg, n=720, land_right=True):
    """A small-circle loop of n vertices around (lon0, lat0); land on the right (water on the left) by default."""
    c = _ll(lon0, lat0)[0]
    e1, e2 = (b[0] for b in tangent_basis(c[None, :]))
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False) * (-1.0 if land_right else 1.0)
    r = np.radians(radius_deg)
    return np.cos(r) * c + np.sin(r) * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2)


CONST = 'kind = "constant"\nh_m = 100000.0'
BOX20 = 'kind = "zone"\nh_m = 20000.0\ntransition_m = 0.0\npolygon = [[0.0, -2.0], [4.0, -2.0], [4.0, 2.0], [0.0, 2.0]]'


def test_min_layers_bit_identical_in_any_order():
    """Only "min" layers: layered returns the old Minimum, and the Layered evaluation in any order gives the same h
    bit for bit (min commutes), so old configs keep their h."""
    from meshom.size import Layered, layered

    fs = [Uniform(100e3), Latitude(110e3, 25e3), zone([[0, -2], [4, -2], [4, 2], [0, 2]], 20e3, 3e5, 100e3)]
    xyz = _ll(np.linspace(-20, 30, 4001), np.linspace(-80, 80, 4001))
    old = Minimum(tuple(fs))(xyz)
    assert type(layered(fs)) is Minimum and layered(fs[:1]) is fs[0]
    for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
        assert np.array_equal(Layered(tuple(fs[i] for i in order), np.inf)(xyz), old)


def test_coast_classes_select_the_right_loops():
    """Areas on the sphere; a loop below island_max_km2 is an island, the loop around the South Pole a continent
    whatever its area; coasts = islands / continents measure the distance to that class only."""
    from meshom.coast import loop_land_km2

    cont = _cap(0.0, 30.0, 20.0)                      # 1.5e7 km^2
    isl = _cap(60.0, 0.0, 1.0)                        # 1.2e4 km^2
    ant = _cap(0.0, -90.0, 2.0)                       # 4.9e4 km^2, around the South Pole
    cap_km2 = lambda deg: 2 * np.pi * (R_EARTH_M / 1e3) ** 2 * (1 - np.cos(np.radians(deg)))
    assert loop_land_km2(cont) == pytest.approx(cap_km2(20.0), rel=1e-4)
    assert loop_land_km2(isl) == pytest.approx(cap_km2(1.0), rel=1e-4)
    assert loop_land_km2(ant) == pytest.approx(cap_km2(2.0), rel=1e-4)
    coast = Coast([cont, isl, ant], np.zeros(3))
    assert coast.islands(1e5).tolist() == [False, True, False]
    assert coast.islands(1e3).tolist() == [False, False, False]
    near_isl, near_cont, near_ant = _ll(60.0, 1.2), _ll(0.0, 9.5), _ll(0.0, -87.5)
    pts = np.vstack([near_isl, near_cont, near_ant])
    cd = 'kind = "coast_distance"\nh_m = 20000.0\ngrad = 0.1\ncoasts = "{}"'
    h = {c: layer_fields(config.loads(_layers(CONST, cd.format(c))), coast)[1][1](pts)
         for c in ("all", "islands", "continents")}
    assert h["islands"][0] < 50e3 and h["islands"][1] == h["islands"][2] == 100e3
    assert h["continents"][0] == 100e3 and h["continents"][1] < 50e3 and h["continents"][2] < 50e3
    assert np.array_equal(h["all"], np.minimum(h["islands"], h["continents"]))


@pytest.mark.skipif(not GSHHS.exists(), reason="no GSHHS i L1 shapefile")
def test_coast_classes_gshhs():
    """GSHHS i (min_area 3000 km^2): Iceland (1.02e5 km^2 on the sphere) and Great Britain are continents at the
    default 1e5 km^2, Ireland (8.3e4) is an island; their areas match the shapefile's within 1 %."""
    from meshom.coast import loop_land_km2

    c = read_coast(str(GSHHS), 3000.0, "none")
    isl = c.islands(config.CoastDistanceLayer(h_m=1.0, grad=0.1).island_max_km2)
    m = np.array([lp.mean(axis=0) for lp in c.loops])
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    ice, gb, ire = (int(np.argmin(arc_length(m, _ll(lon, lat)))) for lon, lat in ((-19, 65), (-2, 54), (-8, 53.3)))
    assert [bool(isl[i]) for i in (ice, gb, ire)] == [False, False, True]
    for i in (ice, gb, ire):
        assert loop_land_km2(c.loops[i]) == pytest.approx(c.area_km2[i], rel=0.01)


def test_zone_set_inside_transition_outside():
    """set: h_m inside whatever the layers before give (a finer zone and a coarser constant), linear in the distance
    to the running h over transition_m, the running h beyond; a later layer still applies on top."""
    st = ('kind = "zone"\nmode = "set"\nh_m = 50000.0\ntransition_m = 500000.0\n'
          'polygon = [[-2.0, -5.0], [10.0, -5.0], [10.0, 5.0], [-2.0, 5.0]]')
    after = 'kind = "zone"\nh_m = 30000.0\ntransition_m = 0.0\npolygon = [[6.0, -1.0], [8.0, -1.0], [8.0, 1.0], [6.0, 1.0]]'
    cfg = config.loads(_layers(CONST, BOX20, st, after))
    h = background_h(cfg, None)
    d = np.radians(np.array([2.0, 4.0])) * R_EARTH_M                    # beyond the east edge at 10°E on the equator
    got = h(_ll([2.0, -1.0, 7.0, 12.0, 14.0, 30.0], [0.0] * 6))
    assert got[:2].tolist() == [50e3, 50e3] and got[2] == 30e3      # BOX20 overridden; the later zone on top
    assert got[3:5] == pytest.approx(50e3 + 50e3 * d / 500e3, rel=1e-9) and got[5] == 100e3
    hl = size_field(cfg, None)                                          # the UI's path (no limiter: grad_max inf)
    assert hl(_ll([2.0, 12.0], [0.0, 0.0])) == pytest.approx(got[[0, 3]], rel=1e-12)   # BLAS: batch-size ulps



def test_min_layer_after_a_raising_set_zone_acts_only_below_the_base():
    """A set zone coarser than the base (200 km over a 100 km constant) keeps its raise where a later min zone (and a
    later layer at the base) do not refine: a min layer at the base is neutral; inside the later zone it refines."""
    st = ('kind = "zone"\nmode = "set"\nh_m = 200000.0\ntransition_m = 0.0\n'
          'polygon = [[0.0, -20.0], [40.0, -20.0], [40.0, 20.0], [0.0, 20.0]]')
    later = 'kind = "zone"\nh_m = 30000.0\ntransition_m = 0.0\npolygon = [[30.0, -1.0], [32.0, -1.0], [32.0, 1.0], [30.0, 1.0]]'
    h = background_h(config.loads(_layers(CONST, st, later, CONST)), None)
    got = h(_ll([20.0, 31.0, 60.0], [0.0, 0.0, 0.0]))
    assert got.tolist() == [200e3, 30e3, 100e3]
    h_, step = h.trace(_ll([20.0, 31.0, 60.0], [0.0, 0.0, 0.0]))    # the UI's "which layer sets h"
    assert h_.tolist() == got.tolist() and step.tolist() == [1, 2, -1]

def _disc_mean(h, c, r, n=400):
    """exp of the biweight mean of ln h over the geodesic disc of radius r around c, by a fine polar quadrature."""
    e1, e2 = (b[0] for b in tangent_basis(c[None, :]))
    rho = (np.arange(n) + 0.5) / n * r
    phi = (np.arange(4 * n) + 0.5) / (4 * n) * 2 * np.pi
    R, P = np.meshgrid(rho, phi, indexing="ij")
    a = (R / R_EARTH_M).ravel()
    x = np.cos(a)[:, None] * c + np.sin(a)[:, None] * (np.cos(P.ravel())[:, None] * e1 + np.sin(P.ravel())[:, None] * e2)
    w = (1 - (R.ravel() / r) ** 2) ** 2 * np.sin(a)
    return np.exp(np.sum(w * np.log(h(x))) / np.sum(w))


def test_zone_smooth_inside_transition_outside():
    """smooth: inside the polygon the disc mean of ln h (within 2 % of a fine quadrature on a 20 -> 100 km step:
    the grids' quadrature and interpolation), (1 - w) h + w s in the
    transition, the running h exactly beyond; the 20 km step is gone."""
    sm = ('kind = "zone"\nmode = "smooth"\nsmooth_m = 300000.0\ntransition_m = 200000.0\n'
          'polygon = [[-5.0, -5.0], [9.0, -5.0], [9.0, 5.0], [-5.0, 5.0]]')
    from meshom.size import Smoothed

    cfg = config.loads(_layers(CONST, BOX20, sm))
    h = background_h(cfg, None)
    raw = Minimum((Uniform(100e3), zone([[0, -2], [4, -2], [4, 2], [0, 2]], 20e3, 0.0, 100e3)))
    st = h.steps[-1]
    assert isinstance(st, Smoothed)
    for lon in (2.0, 0.0, -1.5, 5.0):
        c = _ll(lon, 0.0)
        assert h(c)[0] == pytest.approx(_disc_mean(raw, c[0], 300e3), rel=0.02)
    xt = _ll(10.0, 0.0)                                                 # 1° = 111 km beyond the east edge
    w = 1 - np.radians(1.0) * R_EARTH_M / 200e3
    assert h(xt)[0] == pytest.approx((1 - w) * raw(xt)[0] + w * st.s(xt)[0], rel=1e-12)
    far = _ll([12.0, 40.0, 2.0], [0.0, 0.0, 30.0])
    assert np.array_equal(h(far), raw(far))
    line = _ll(np.linspace(-1.0, 5.0, 61), np.zeros(61))
    assert np.abs(np.diff(np.log(h(line)))).max() < 0.25 < np.abs(np.diff(np.log(raw(line)))).max()


def test_size_from_merge_and_round_trip(tmp_path):
    """[size] from: the file's keys and layers first, the config's keys over them, its layers after them; a variant
    file holds only [size]; dumps_size round-trips; dumps inlines the variant (the run's config.resolved.toml)."""
    (tmp_path / "h").mkdir()
    (tmp_path / "h/var.toml").write_text(
        '[size]\ngrad_max = 0.05\nh_max_m = 100000.0\n'
        '[[size.layers]]\nkind = "constant"\nh_m = 90000.0\n'
        '[[size.layers]]\nkind = "coast_distance"\nh_m = 30000.0\ngrad = 0.1\ncoasts = "continents"\n')
    (tmp_path / "cfg.toml").write_text(
        '[mesh]\nmethod = "m01_isotropic"\n[size]\nfrom = "h/var.toml"\nh_max_m = 200000.0\n' + "[[size.layers]]\n"
        + BOX20 + "\n")
    cfg = config.load(tmp_path / "cfg.toml")
    s = cfg.size
    assert (s.grad_max, s.h_max_m) == (0.05, 200000.0)
    assert [ly.kind for ly in s.layers] == ["constant", "coast_distance", "zone"]
    assert s.layers[1].coasts == "continents" and cfg.size_from == str((tmp_path / "h/var.toml").resolve())
    assert config.loads_size(config.dumps_size(s)) == s
    text = config.dumps(cfg)
    assert "from" not in text and config.loads(text, cfg.name) == cfg
    with pytest.raises(config.ConfigError, match="only"):
        config.loads_size('[mesh]\nmethod = "m01_isotropic"\n')
