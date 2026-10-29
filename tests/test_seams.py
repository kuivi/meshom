"""Seam test cases and seam measures."""
import numpy as np
import pytest

from meshom.cases import CASES, STRAITS
from meshom.cases_seams import H0_M, SEAM_CASES, SEAMS, Plane
from meshom.domain import in_water, loop_signed_area
from meshom.mesh import Mesh
from meshom.quality.seams import dislocations, seam_block


def test_cases_build_and_register():
    assert len(SEAM_CASES) == 15 and all(k in CASES for k in SEAM_CASES)
    assert "seam_strait" in STRAITS
    for name, (dom, h) in SEAM_CASES.items():
        assert len(dom.loops) == 1 and loop_signed_area(dom.loops[0]) < 0      # lake + inlet + ring: one coast
        for c in SEAMS[name].corridors:
            sm = 0.5 * sum(c.span())
            mid = c.point(sm, 0.0)[None, :]
            assert in_water(dom, mid)[0]
            assert np.isfinite(h(mid)).all()
    _, h = CASES["seam_rot30_k10"]
    c = SEAMS["seam_rot30_k10"].corridors[0]
    sm = 0.5 * sum(c.span())
    ha, hb = (h(c.point(sm, d)[None, :])[0] for d in (-c.half_width_m, c.half_width_m))
    assert ha == pytest.approx(H0_M) and hb == pytest.approx(1.1 * H0_M)              # plateaus beyond the corridor


def test_reference_counts():
    L, h = 40 * H0_M, H0_M
    assert dislocations(L, 0.0, (h, h), 0.0, (h, h)) == pytest.approx(0.0, abs=1e-9)
    assert dislocations(L, 0.0, (h, h), 0.0, (1.1 * h, 1.1 * h)) == pytest.approx(40 * (1 - 1 / 1.1))   # |N - M|
    for th in (15.0, 30.0, 45.0):                                                   # symmetric tilt: 2 (L/h) sin(th/2)
        assert dislocations(L, th / 2, (h, h), -th / 2, (h, h)) == pytest.approx(80 * np.sin(np.radians(th / 2)))
    assert dislocations(L, 0.0, (h, h), 90.0, (h, h)) == pytest.approx(0.0, abs=1e-9)   # a quarter turn is no seam
    assert dislocations(L, -60.0, (h, h), 60.0, (h, h)) == pytest.approx(40 * (np.sqrt(3) - 1))   # the Y's walls


def _lake_lattice(name: str, plane: Plane, box: tuple) -> Mesh:
    """A perfect lattice of the lake: unit cells of the plane grid whose centre is water (the lake's vertices lie on
    the grid, so its boundary is the lake's coast)."""
    dom, _ = CASES[name]
    x0, x1, y0, y1 = box
    xs, ys = np.arange(x0, x1 + 1), np.arange(y0, y1 + 1)
    nx = len(xs)
    gx, gy = np.meshgrid(xs, ys)
    pts = plane.to_sphere(np.column_stack([gx.ravel(), gy.ravel()]).astype(float))
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(len(ys) - 1))
    i, j = i.ravel(), j.ravel()
    quads = np.column_stack([j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i])
    cen = plane.to_sphere(np.column_stack([xs[i] + 0.5, ys[j] + 0.5]))
    quads = quads[in_water(dom, cen)]
    used = np.unique(quads)
    new = np.full(len(pts), -1)
    new[used] = np.arange(len(used))
    return Mesh(pts[used], np.zeros((0, 3), dtype=np.int64), new[quads], np.zeros(len(used), dtype=np.int8))


@pytest.mark.parametrize("name, plane, box, elems", [
    ("seam_rot0_k0", Plane(), (-43, 43, -20, 20), 6 * 34),
    ("seam_coast", Plane(x_az=60.0), (-28, 28, 0, 56), 8 * 40),
    ("seam_strait", Plane(x_az=60.0), (-30, 30, -70, 70), 12 * 54)])
def test_perfect_lattice_scores_zero(name, plane, box, elems):
    mesh = _lake_lattice(name, plane, box)
    _, h = CASES[name]
    b = seam_block(mesh, h, SEAMS[name])
    c = b["corridors"][0]
    assert c["elems"] == elems                                     # the corridor mask covers the right cells
    assert c["irregular"] == 0 and c["gross"] == 0 and c["T"] == 0
    assert c["disl_min_case"] == 0 and abs(c["disl_min_real"]) < 0.5 and c["excess_per10"] <= 0.0
    assert c["quad_p99_9_deg"] < 92.0 and c["ln_sh_p99"] < 0.03 and not c["gates_failed"]
    for s in c["sides"].values():
        if s is not None:
            assert abs(s["phi_deg"]) < 1.0 and s["coherence"] > 0.99
    if name == "seam_strait":
        assert c["cells_across"]["min"] == c["cells_across"]["max"] == 12


def test_score_carries_the_seam_block():
    """compete._score adds the seam block on a seam case only; the ranking fields do not change (decisions P29)."""
    from meshom.compete import _score
    from meshom.config import DEFAULT_GATES

    name = "seam_rot0_k0"
    mesh = _lake_lattice(name, Plane(), (-43, 43, -20, 20))
    dom, h = CASES[name]
    rho = {"plus": 0.0, "minus": 0.0}
    with_seam, plain = (_score(mesh, dom, h, rho, dict(DEFAULT_GATES), None, s) for s in (SEAMS[name], None))
    assert "seam" not in plain and with_seam["seam"]["corridors"][0]["elems"] == 6 * 34
    assert not with_seam["seam"]["gates_failed"]
    assert with_seam["gates_failed"] == plain["gates_failed"] and set(with_seam) - set(plain) == {"seam"}


def test_layout_hint():
    """SEAM_HINTS (decisions P28): one seed per region, in the water and outside every corridor, grow limit 20 h,
    align; merged under the caller's opts for m14's methods only."""
    from meshom.cases_seams import SEAM_HINTS, seam_opts
    from meshom.geo import lonlat_to_xyz

    n = {"seam_junction": 3, "seam_coast": 1}
    for name, (dom, _) in SEAM_CASES.items():
        hint = SEAM_HINTS[name]
        sd = np.asarray(hint["seeds"])
        x = lonlat_to_xyz(sd[:, 0], sd[:, 1])
        assert len(sd) == n.get(name, 2) and hint["grow_limit"] == 20.0 and hint["align"] is True
        assert in_water(dom, x).all()
        for c in SEAMS[name].corridors:                                     # >= 10 H0 off the measured strip
            s, d = c.coords(x)
            assert not ((s >= c.span()[0]) & (s <= c.span()[1]) & (np.abs(d) <= c.half_width_m + 10 * H0_M)).any()
    o = seam_opts("m14_blocks_mix", "seam_rot30_k3", {"grow_limit": 12.0, "verbose": True})
    assert o["grow_limit"] == 12.0 and o["verbose"] and o["seeds"] == SEAM_HINTS["seam_rot30_k3"]["seeds"]
    assert seam_opts("m02_linf_frontal", "seam_rot30_k3", None) is None
    assert seam_opts("m14_blocks", "seam_rot30_k3", None, hint=False) is None
    assert seam_opts("m14_blocks_igm", "seam_rot30_k3", None) is None
