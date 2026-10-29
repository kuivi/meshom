"""Medial axis, passages and the automatic strait zone (meshom/lfs.py; decisions S22, S24, S31, S32)."""
import numpy as np
import pytest

from meshom import lfs
from meshom.cases import CASES
from meshom.cases_seams import H0_M, Plane, _lake_loop
from meshom.domain import CHANNEL_WIDTH_M, continent_frame
from meshom.geo import R_EARTH_M
from meshom.size import scaled, uniform


def _axis(loops, h):
    return lfs.medial_axis(loops, scaled(h, lfs.STEP), scaled(h, lfs.LAM))


def test_continent_channel_is_one_passage():
    dom, h = CASES["continent"]
    ps = lfs.passages(_axis(dom.loops, h), h, 5)
    ch = [p for p in ps if p.W.max() < 2 * CHANNEL_WIDTH_M]
    assert len(ps) == 2 and len(ch) == 1                     # the channel, and the 900 km island gap (n = 19)
    p = ch[0]
    assert p.n == 10 and p.ends == ("open", "open")          # 150 km / 15 km; both ends open into the ocean
    assert np.median(p.W) == pytest.approx(CHANNEL_WIDTH_M, rel=1e-3)
    assert p.length_m == pytest.approx(6.67e6, rel=5e-3)     # mouth to mouth, 60 degrees of arc
    _, _, n = continent_frame()
    assert np.abs(p.points @ n).max() * R_EARTH_M < 1e3      # on the centreline great circle


def test_dead_end_arm_is_no_passage():
    """A lake with a 6 H0 wide, 30 H0 long dead-end arm: the arm is never a passage (only the lake's inlet is)."""
    pl = Plane()
    loop, _ = _lake_loop(pl, [(-20, -20), (20, -20), (20, 20), (3, 20), (3, 50), (-3, 50), (-3, 20), (-20, 20)], 0)
    h = uniform(H0_M)
    ps = lfs.passages(_axis([loop], h), h, 5)
    c, ex, ey = pl.frame()
    for p in ps:
        y = (p.points - c) @ ey * R_EARTH_M / H0_M
        assert y.max() < 20.0                                # nothing in the arm (y > 20)
    assert len(ps) == 1 and ps[0].n == 8                     # the inlet, 8 H0 wide


def test_auto_zone_h_is_w_over_n():
    """Base h 50 km: the 150 km channel gets n = n_min = 5, h = W/5 = 30 km inside, graded with g beyond."""
    dom, _ = CASES["continent"]
    zone, info = lfs.auto_straits(dom.loops, uniform(50e3), 5, 0.1, 10e3, 50e3)
    assert zone is not None and info["n_open"] == info["n_passages"]
    c, t, n = continent_frame()
    assert zone(c[None])[0] == pytest.approx(30e3, rel=5e-3)   # the Voronoi radius: + step^2 / 4W at step h/4
    off = 0.5 * CHANNEL_WIDTH_M + 50e3                       # 50 km beyond the bank (over land): 30 + 0.1 * 50 km
    x = np.cos(off / R_EARTH_M) * c + np.sin(off / R_EARTH_M) * n
    assert zone(x[None])[0] == pytest.approx(35e3, rel=1e-2)
    assert zone(-c[None])[0] == 50e3                         # far away: the base h
    closed, info2 = lfs.auto_straits(dom.loops, uniform(50e3), 5, 0.1, 40e3, 50e3)   # S24: W/n < h_min -> closed
    assert info2["n_closed"] >= 1 and all(not r["open"] for r in info2["passages"] if r["W_min_m"] < 2e5)
