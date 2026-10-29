"""Resampled coast loops never cross (decisions C38; m14's merged coast walks too), and m02's packing ends on a NaN
field (M37): the West Africa
`meshom run` with treatment "loops" hung there."""
import subprocess
import sys

import numpy as np
import pytest

from meshom.delaunay import background_mesh
from meshom.domain import Domain, crossing_chords, loop_edges, resample_loops
from meshom.field import _boundary_halfedges, cross_field
from meshom.geo import lonlat_to_xyz
from meshom.size import Uniform
from tests.helpers import local_polygon


def wavy_strait(width=0.09, amp=0.4, lam=1.5, length=10.0, depth=3.0) -> Domain:
    """Two land pieces (clockwise) on both sides of a 10 km wide strait that meanders with amplitude 44 km and wave
    length 167 km: chords of 200 km on the two banks cross each other (as at Suez in GSHHS i)."""
    x = np.arange(0.0, length + 1e-9, 0.01)
    y = amp * np.sin(2 * np.pi * x / lam)
    side = np.arange(0.05, depth, 0.05)
    xb = np.arange(length - 0.05, 0.0, -0.05)
    ys, yn = y - width / 2, y + width / 2
    s = np.r_[np.c_[x, ys], np.c_[np.full(len(side), length), ys[-1] - side],
              np.c_[xb, np.full(len(xb), ys[0] - depth)], np.c_[np.zeros(len(side)), ys[0] - depth + side]]
    n = np.r_[np.c_[x[::-1], yn[::-1]], np.c_[np.zeros(len(side)), yn[0] + side],
              np.c_[xb[::-1], np.full(len(xb), yn[0] + depth)], np.c_[np.full(len(side), length), yn[-1] + depth - side]]
    return Domain([lonlat_to_xyz(s[:, 0], s[:, 1]), lonlat_to_xyz(n[:, 0], n[:, 1])], "wavy_strait")


def test_resampled_loops_do_not_cross_and_the_field_is_finite():
    h = Uniform(200e3)
    dom = Domain(resample_loops(wavy_strait(), h, corner_deg=180.0))      # as Coast.loops_at (treatment "loops")
    bg = background_mesh(dom, h, factor=1.0)                              # as m14's fitness and m02
    a, b = _boundary_halfedges(bg.tris, bg.n_nodes)
    bnodes = np.flatnonzero(bg.node_flag == 1)
    nbh = np.bincount(np.r_[a, b], minlength=bg.n_nodes)[bnodes]
    assert (nbh == 2).all(), f"{int((nbh == 0).sum())} coast nodes inside the water (the land leaked in)"
    cf = cross_field(bg, h, n=4, boundary_nodes=bnodes)
    assert np.isfinite(cf.u).all()
    lp = resample_loops(dom, h)
    e = loop_edges(lp)
    assert not crossing_chords(np.vstack(lp), e[:, 0], e[:, 1]).any()


def thin_island() -> Domain:
    """A 120 km long, 2 km wide island (clockwise, a hole): its two tips are the only corners (> 30 deg) and each side
    is 1.2 h long at h = 100 km, so the corner rule placed one point per side, a 2-point loop without area (as Cebu
    and Melville Island after Coast.loops_at)."""
    x = np.linspace(-0.54, 0.54, 25)
    y = 0.009 * (1.0 - np.abs(x) / 0.54)
    ll = np.r_[np.c_[x, y], np.c_[x[-2:0:-1], -y[-2:0:-1]]]             # clockwise seen from outside
    return Domain([lonlat_to_xyz(30.0 + ll[:, 0], 10.0 + ll[:, 1])], "thin_island")


def test_a_thin_island_keeps_an_area():
    h = Uniform(100e3)
    dom = thin_island()
    assert len(resample_loops(dom, h)[0]) >= 3
    bg = background_mesh(dom, h, factor=1.0)
    a, b = _boundary_halfedges(bg.tris, bg.n_nodes)
    bnodes = np.flatnonzero(bg.node_flag == 1)
    assert (np.bincount(np.r_[a, b], minlength=bg.n_nodes)[bnodes] == 2).all()
    assert np.isfinite(cross_field(bg, h, n=4, boundary_nodes=bnodes).u).all()


def test_crossing_chords():
    p = lonlat_to_xyz(np.array([0.0, 1.0, 0.5, 0.5, 2.0, 3.0]), np.array([0.0, 0.0, -0.5, 0.5, 0.0, 0.0]))
    ia, ib = np.array([0, 2, 4, 1]), np.array([1, 3, 5, 4])
    assert crossing_chords(p, ia, ib).tolist() == [True, True, False, False]   # a shared node is no crossing


def _bay_and_block():
    """Land x in [-600, 600] km, y in [-500, 0] with a smooth bay 15 km deep, 50 km wide at x = 50 km on its north
    coast (1 km vertices, no corner), a fixed block on the coast at x = -400..-300 (its mouth run only: what
    merged_loops reads) and a 2 km islet (100 vertices) across the bay's mouth."""
    from meshom.methods.m14.blockmap import BlockGrid
    from meshom.methods.m14.strait import FixedBlock

    x = np.arange(-600.0, 600.0, 1.0)
    side, back = np.arange(0.0, 500.0, 5.0), np.arange(600.0, -600.0, -5.0)
    xy = np.r_[np.c_[x, -15.0 * np.exp(-((x - 50.0) ** 2) / 128.0)], np.c_[np.full(100, 600.0), -side],
               np.c_[back, np.full(240, -500.0)], np.c_[np.full(100, -600.0), side - 500.0]]
    i1, i2 = int(np.flatnonzero(x == -400.0)[0]), int(np.flatnonzero(x == -300.0)[0])
    run = local_polygon(0.0, 0.0, np.array([[-400, 0], [-400, 50], [-350, 50], [-300, 50], [-300, 0]]) * 1e3)
    fb = FixedBlock(BlockGrid(run, np.zeros((0, 4), np.int64), np.zeros((5, 2), np.int64), np.arange(5)),
                    np.zeros(5, bool), [(np.arange(5), (0, i1, 0.0), (0, i2, 0.0))])
    t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    islet = local_polygon(0.0, 0.0, np.c_[50.0 + 2.0 * np.cos(t), 2.0 * np.sin(t)][::-1] * 1e3)
    return local_polygon(0.0, 0.0, xy * 1e3), fb, islet


@pytest.mark.parametrize("islet_as", ["island", "exact outline"])
def test_a_merged_coast_walk_does_not_cross_an_islet(islet_as):
    """C38 for m14's merged coast walks (stitch.merged_loops): the walk from the block's mouth along the coast,
    resampled at h = 100 km, cuts across the 50 km bay with one chord, which crosses the islet (an untouched loop
    resampled at h, or an exact block outline of the fill); the chord is refined at the bay's vertices until it
    crosses nothing (3 crossings without the refinement)."""
    from meshom.methods.m14.stitch import merged_loops

    h = Uniform(100e3)
    land, fb, islet = _bay_and_block()
    if islet_as == "island":
        ml, _, touched = merged_loops(Domain([land, islet]), h, [fb])
        out = resample_loops(Domain(ml + [islet], "bay", [True] * len(ml) + [False]), h)     # as the stitch
    else:
        ml, _, touched = merged_loops(Domain([land]), h, [fb], [islet])
        out = ml + [islet]
    assert touched == {0} and len(ml) == 1
    e = loop_edges(out)
    assert not crossing_chords(np.vstack(out), e[:, 0], e[:, 1]).any()


_NAN_PACK = """
import numpy as np
import meshom.field as F
from meshom.cases import CASES
from meshom.methods import m02_linf_frontal as m02
from meshom.size import scaled
cf0 = F.cross_field
def nan_field(*a, **k):
    cf = cf0(*a, **k)
    cf.u[cf.mesh.tris[0]] = np.nan
    return cf
F.cross_field = nan_field
dom, h0 = CASES["uniform"]
try:
    m02.points(dom, scaled(h0, 4.0), {})
except RuntimeError as e:
    print("ERR", e)
"""


def test_pack_ends_on_a_nan_field():
    r = subprocess.run([sys.executable, "-c", _NAN_PACK], capture_output=True, text=True, timeout=120)
    assert "ERR m02 pack" in r.stdout, r.stdout + r.stderr
