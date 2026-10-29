"""Narrow through-passages (meshom.passages, [coast] passage_min_cells, decisions C51): found on the mesh as built,
dead-end inlets left alone, and widened by the land cut (meshom.run.cut_land)."""
import numpy as np
import pytest

from meshom.domain import Domain
from meshom.geo import lonlat_to_xyz
from meshom.mesh import Mesh
from meshom.passages import narrow_passages
from meshom.run import cut_land
from meshom.size import Uniform

STEP = 0.1                                   # grid step (degrees): 11.1 km cells near the equator


def grid_mesh(mask: np.ndarray, quads: bool = False) -> Mesh:
    """The squares of `mask` (rows = y, columns = x) as quads or two triangles each, near (0, 0)."""
    ny, nx = mask.shape
    I, J = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
    P = lonlat_to_xyz((I * STEP).ravel().astype(float), (J * STEP).ravel().astype(float))
    T, Q = [], []
    for j, i in zip(*np.nonzero(mask)):
        a, b, c, d = j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i
        if quads:
            Q.append([a, b, c, d])
        else:
            T += [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
    used = np.unique(np.r_[np.ravel(T), np.ravel(Q)]).astype(np.int64)
    m = np.full(len(P), -1, dtype=np.int64)
    m[used] = np.arange(len(used))
    T = m[np.asarray(T, dtype=np.int64).reshape(-1, 3)]
    Q = m[np.asarray(Q, dtype=np.int64).reshape(-1, 4)]
    return Mesh(P[used], T, Q, np.zeros(len(used), dtype=np.int8))


def basins(chan: int) -> np.ndarray:
    """Two 10 x 10 basins joined by a channel `chan` cells wide and 4 long; a one-cell inlet 5 long off the first."""
    w = np.zeros((16, 24), dtype=bool)
    w[0:10, 0:10] = w[0:10, 14:24] = True
    w[5:5 + chan, 10:14] = True
    w[10:15, 3] = True
    return w


@pytest.mark.parametrize("quads", [False, True])
def test_one_cell_channel_is_a_narrow_passage_and_the_inlet_is_not(quads):
    r = narrow_passages(grid_mesh(basins(1), quads), 2)
    assert len(r["passages"]) == 1
    p = r["passages"][0]
    assert p["kind"] == "pools" and p["cells_across"] == 1
    assert 1.0 < p["at"][0] < 1.4 and 0.5 < p["at"][1] < 0.6          # in the channel (x 10-14, y 5-6 cells)
    assert narrow_passages(grid_mesh(basins(2), quads), 2)["passages"] == []      # 2 cells: fine; the inlet: a dead end


@pytest.mark.parametrize("quads", [False, True])
def test_three_cells_level(quads):
    """K = 3: a 2-cell channel is narrow (2 cells across), a 3-cell one is not."""
    r = narrow_passages(grid_mesh(basins(2), quads), 3)
    assert [(p["kind"], p["cells_across"]) for p in r["passages"]] == [("pools", 2)]
    assert narrow_passages(grid_mesh(basins(3), quads), 3)["passages"] == []


def test_island_one_cell_off_the_coast_and_a_small_basin_behind_a_mouth():
    w = np.zeros((14, 24), dtype=bool)
    w[0:10, :] = True
    w[1:5, 6:12] = False                                               # an island, one cell off the south coast
    w[10, 18] = True
    w[11:13, 17:20] = True                                             # a 6-cell basin behind a one-cell mouth
    r = narrow_passages(grid_mesh(w), 2)
    assert [p["kind"] for p in r["passages"]] == ["island"]
    assert r["n_inlet"] == 1                                           # < BASIN_MIN_H2 h^2: an inlet, left alone
    r = narrow_passages(grid_mesh(w), 2, basin_min_h2=1.0)
    assert sorted(p["kind"] for p in r["passages"]) == ["island", "pools"]


def _rect(x0, x1, y0, y1) -> np.ndarray:
    """A land rectangle in grid cells, clockwise (water on the left, the domain convention)."""
    lon = np.array([x0, x0, x1, x1]) * STEP
    lat = np.array([y0, y1, y1, y0]) * STEP
    return lonlat_to_xyz(lon, lat)


@pytest.mark.parametrize("quads", [False, True])
def test_land_cut_widens_a_one_cell_passage(quads):
    """The cut leaves a one-cell channel (y 7-8) between two land blocks; with passage_min_cells = 2 the cut widens it
    to >= 2 cells into the land on both sides and leaves the one-cell inlet (x 3, y 13-19) alone."""
    mesh = grid_mesh(np.ones((20, 30), dtype=bool), quads)
    land = Domain([_rect(10, 20, 7.8, 20), _rect(10, 20, 0, 7.2), _rect(0, 3.2, 12.7, 20), _rect(3.8, 10, 12.7, 20)],
                  "land")
    out, info = cut_land(mesh, land)
    r = narrow_passages(out, 2)
    assert len(r["passages"]) == 1 and "passages" not in info
    h = Uniform(STEP * np.pi / 180 * 6.371e6)
    out2, info2 = cut_land(mesh, land, passages={"cells": 2, "h_bg": h, "quads": quads, "tol": 0.0})
    assert narrow_passages(out2, 2)["passages"] == []
    rec = info2["passages"]
    assert rec["n_widened"] == 1 and rec["n_left"] == 0 and rec["cut_km2"] > 0
    assert rec["n_land_touched"] == 2 and rec["n_land_removed"] == 0
    new = set(map(tuple, np.round(out2.points, 9))) - set(map(tuple, np.round(out.points, 9)))
    lon = np.degrees(np.arctan2([p[1] for p in new], [p[0] for p in new]))
    lat = np.degrees(np.arcsin([p[2] for p in new]))
    assert len(new) and lon.min() > 0.8 and lon.max() < 2.2 and 0.5 < lat.min() and lat.max() < 1.05  # near the channel
