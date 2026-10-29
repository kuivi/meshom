"""size fields, domains, cases, rho_c integrals."""
import numpy as np
import pytest

from meshom.cases import CASES
from meshom.compete import rho_c_integrals
from meshom.domain import in_water, loop_signed_area, resample_loops
from meshom.geo import R_EARTH_M, arc_length, lonlat_to_xyz
from meshom.size import scaled


def test_size_fields():
    _, disc = CASES["disc"]
    _, band = CASES["band"]
    p = lonlat_to_xyz(np.array([-30.0, -30.0, -30.0, 150.0]), np.array([40.0, 40.0 + np.degrees(700e3 / R_EARTH_M),
                                                                       40.0 + np.degrees(2e6 / R_EARTH_M), 0.0]))
    assert np.allclose(disc(p), [10e3, 30e3, 50e3, 50e3])
    q = lonlat_to_xyz(np.zeros(3), np.array([-10.0, 0.0, 10.0]))
    assert np.allclose(band(q), [25e3, 62.5e3, 100e3])
    assert np.allclose(scaled(disc, 4)(p), 4 * disc(p))


def test_continent_loops_closed_water_left():
    dom, h = CASES["continent"]
    assert len(dom.loops) == 3
    for lp in dom.loops:
        seg = R_EARTH_M * arc_length(lp, np.roll(lp, -1, axis=0))           # includes the closing segment
        assert seg.max() <= 10e3 and seg.min() > 0
        assert loop_signed_area(lp) < 0                                     # land on the right, water on the left
        # a point just left of each segment midpoint is water, just right is land
        a, b = lp, np.roll(lp, -1, axis=0)
        mid = (a + b) / np.linalg.norm(a + b, axis=1, keepdims=True)
        left = np.cross(mid, b - a)
        left /= np.linalg.norm(left, axis=1, keepdims=True)
        off = 1e3 / R_EARTH_M
        assert in_water(dom, mid + off * left).all()
        assert not in_water(dom, mid - off * left).any()
    # the channel is water, the island centre and the continent body are land
    p = lonlat_to_xyz(np.array([20.0, 60.0, 20.0, -100.0]), np.array([0.0, 10.0, 20.0, 0.0]))
    assert in_water(dom, p).tolist() == [True, False, False, True]
    assert h(p[:1])[0] == 15e3


@pytest.mark.parametrize("s", [1.0, 4.0])
def test_resample_loops_spacing_and_corners(s):
    dom, h = CASES["continent"]
    hs = scaled(h, s)
    loops = resample_loops(dom, hs)
    for lp, orig in zip(loops, dom.loops):
        seg = R_EARTH_M * arc_length(lp, np.roll(lp, -1, axis=0))
        mid = lp + np.roll(lp, -1, axis=0)
        r = seg / hs(mid / np.linalg.norm(mid, axis=1, keepdims=True))
        assert 0.7 < r.min() and r.max() < 1.4
        assert loop_signed_area(lp) < 0
    # the four channel mouths (corners) are kept exactly
    for lp, orig in zip(loops[:2], dom.loops[:2]):
        from meshom.domain import _turn_angles
        corners = orig[_turn_angles(orig) > np.radians(30)]
        assert len(corners) == 2
        assert all(np.min(np.linalg.norm(lp - c, axis=1)) < 1e-12 for c in corners)


def test_rho_c_disc():
    dom, h = CASES["disc"]
    r = rho_c_integrals(dom, h)
    analytic = 4.0 * R_EARTH_M * np.sin(500e3 / R_EARTH_M) * 0.1 / 10e3     # (2/pi) 2 pi r_in g / h_in ~ 20
    assert abs(r["lap_plus"] - analytic) < 0.2 * analytic
    assert abs(r["lap_minus"] + analytic) < 0.2 * analytic
    assert r["curv"] == pytest.approx(8.0, abs=1e-6)
    assert r["total"] == pytest.approx(8.0, abs=0.5)


def test_rho_c_uniform():
    dom, h = CASES["uniform"]
    r = rho_c_integrals(dom, h)
    assert r["plus"] == pytest.approx(8.0, abs=1e-6)
    assert r["minus"] == 0.0
