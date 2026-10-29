"""meshom.coast: Antarctica from GSHHS L5 / L6 next to an L1 source (one loop around the South Pole, the pole is land,
the ice-front area), and the UI coast showing it without artificial edges."""
from pathlib import Path

import numpy as np
import pytest

from meshom.coast import read_coast
from meshom.domain import Domain, in_water
from meshom.geo import R_EARTH_M, lonlat_to_xyz

L1 = Path("/work/ab0995/a270126/projects/mesh_generation/coastlines/GSHHS_shp/i/GSHHS_i_L1.shp")
pytestmark = pytest.mark.skipif(not L1.with_name("GSHHS_i_L5.shp").exists(), reason="GSHHG 2.3.7 not on disk")


def test_antarctica_ice_front_is_one_loop_around_the_pole():
    none = read_coast(str(L1), 1e6, "none")                        # only the continents (> 1e6 km^2)
    ice = read_coast(str(L1), 1e6, "ice_front")
    assert len(ice.loops) == len(none.loops) + 1 and ice.antarctica.startswith("ice_front: GSHHS_i_L5.shp")
    ant = ice.loops[-1]
    lat = np.degrees(np.arcsin(ant[:, 2]))
    assert lat.max() < -60 and lat.min() > -86                      # the coast only: no pole-row vertex left
    pts = lonlat_to_xyz(np.array([0.0, 123.0, -60.0, 0.0, 20.0]), np.array([-90.0, -85.0, -55.0, 0.0, 10.0]))
    assert in_water(Domain([ant]), pts).tolist() == [False, False, True, True, True]
    assert in_water(ice.domain(), pts).tolist() == [False, False, True, True, False]      # (20, 10): Africa
    q = np.random.default_rng(0).normal(size=(400_000, 3))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    area = np.mean(~in_water(Domain([ant]), q)) * 4 * np.pi * (R_EARTH_M / 1e3) ** 2
    assert area == pytest.approx(13.92e6, rel=0.02)                 # GSHHS: 13 919 141 km^2 (Monte Carlo, 4e5 points)
    grounding = read_coast(str(L1), 1e6, "grounding")
    assert len(grounding.loops) == len(none.loops) + 1 and grounding.antarctica.startswith("grounding: GSHHS_i_L6")


def test_no_antarctica_without_l5_next_to_the_source():
    old = Path("/work/ab0995/a270126/projects/mesh_generation/oceanmesh/tests/GSHHS_i_L1.shp")
    if not old.exists():
        pytest.skip("oceanmesh copy not found")
    c = read_coast(str(old), 1e6)
    assert c.antarctica.startswith("none (no GSHHS L5 file")
    assert len(c.loops) == len(read_coast(str(L1), 1e6, "none").loops)


def test_ui_coast_draws_antarctica_without_cut_edges():
    pytest.importorskip("shapefile")
    from ui.common import coast

    c = coast.load(str(L1), 1e6)
    land = c.land_mask(np.arange(-179.75, 180, 0.5), np.arange(-89.75, 90, 0.5))
    assert land[0].all() and land[19, 560] and not land[180, 360]  # the pole row, (100E, 80S) land; (0, 0) water
    for pl in c.polylines():
        assert pl[:, 1].min() > -89.0                               # no edge along the pole row or down to it
        same = (pl[:-1, 0] == pl[1:, 0]) & (np.abs(pl[:-1, 0]) % 180 == 0)
        assert not same.any()                                       # no meridian cut edge at 0° or ±180°
