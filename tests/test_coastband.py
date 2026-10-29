"""m14 thin coastal bands (meshom/methods/m14/coastband.py; decisions C36, C37, C25): the row count rule
n = min(n_max, k*, plateau / h, LFS / (2 h)) and the turning rule, a bay where the normals converge drops rows, the
coast row fixed in the sweeps, and the bands through m14's fixed-block path."""
import numpy as np
from scipy.spatial import cKDTree

from meshom.domain import Domain
from meshom.geo import R_EARTH_M, lonlat_to_xyz, tangent_basis
from meshom.methods.m14 import coastband as CB
from meshom.methods.m14_blocks import DEFAULTS
from meshom.size import coast_distance, uniform

H = 50e3
O = dict(DEFAULTS, coast_bands=True)


def _plane(lon=10.0, lat=-20.0):
    c = lonlat_to_xyz(np.array([lon]), np.array([lat]))[0]
    e1, e2 = tangent_basis(c[None])
    return c, e1[0], e2[0]


def _poly(xy_km: np.ndarray, step_km: float = 5.0) -> np.ndarray:
    """Closed polygon in the gnomonic plane (km), sampled every step_km, on the sphere."""
    c, e1, e2 = _plane()
    out = []
    for a, b in zip(xy_km, np.roll(xy_km, -1, axis=0)):
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / step_km)))
        out.append(a + (b - a) * (np.arange(k)[:, None] / k))
    xy = np.vstack(out) * 1e3 / R_EARTH_M
    p = c + xy[:, :1] * e1 + xy[:, 1:] * e2
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def _island(xy_km) -> np.ndarray:
    """A land polygon given counter-clockwise in the plane, walked clockwise (water on the left)."""
    return _poly(np.asarray(xy_km, float))[::-1].copy()


def _circle_km(r_km: float, n: int = 3000) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return r_km * np.column_stack([np.cos(a), np.sin(a)])


def _depths(bands):
    return sorted({int(b.record["rows"][2]) for b in bands})


def test_k_star_on_a_wavy_coast():
    """k* = floor(0.32 rho_med / h) binds where the coast waves (its net turning cancels, so the turning rule passes):
    a sine of 1000 km wavelength and 80 km amplitude on the north side of a large land block, h = 50 km."""
    x = np.linspace(1500.0, -1500.0, 601)
    top = np.column_stack([x, 1000.0 + 80.0 * np.sin(2 * np.pi * x / 1000.0)])        # east to west
    land = np.vstack([[[-1500.0, -1000.0], [1500.0, -1000.0]], top])
    bands, _ = CB.coast_bands(Domain([_island(land)], "wave"), uniform(H), O)
    wavy = [b for b in bands if b.record["rho_med_m"] is not None and b.record["rho_med_m"] < 2e6]
    assert len(wavy) == 1 and wavy[0].record["m"] >= 50
    r = wavy[0].record
    assert r["k_star"] == int(np.floor(CB.K_STAR * r["rho_med_m"] / r["h_med_m"])) == 3
    assert r["rows"][2] == 3


def test_turning_rule_on_round_islands():
    """A round island of radius 1000 km (rho = R tan(r / R) = 1008 km) at h = 50 km: k* and n_max allow 4 rows, but
    the outer row of a band D deep is (1 + D / rho) longer: 4 rows 0.18, 3 rows 0.14, 2 rows 0.095 <= TURN_TOL, so 2.
    At radius 600 km even 2 rows fail (0.155): no band, as the turning cannot be cut away on uniform curvature."""
    bands, _ = CB.coast_bands(Domain([_island(_circle_km(1000.0))], "isl"), uniform(H), O)
    assert len(bands) == 1 and _depths(bands) == [2] and bands[0].record["k_star"] == 4
    bands, info = CB.coast_bands(Domain([_island(_circle_km(600.0))], "isl"), uniform(H), O)
    assert bands == [] and info["stretches"] == 1


def test_row_rule_plateau():
    """plateau / h: h flat for 100 km from the coast, then + 0.1 per metre: h(150 km) = 55 km (ln 1.1 > 0.05), so
    the band has 2 rows (the turning rule allows 3 at this radius)."""
    loop = _island(_circle_km(1500.0))
    bands, _ = CB.coast_bands(Domain([loop], "isl"), coast_distance(loop, H, 100e3, 0.1), O)
    assert len(bands) == 1 and _depths(bands) == [2] and bands[0].record["plateau_min"] == 2
    bands, _ = CB.coast_bands(Domain([loop], "isl"), uniform(H), O)
    assert _depths(bands) == [3]


def test_row_rule_lfs():
    """LFS / (2 h): a lake 8 h wide gets 2 rows along its long sides (the opposite coast 8 h away)."""
    w, L = 8 * H / 1e3, 80 * H / 1e3
    lake = _poly(np.array([[0, 0], [L, 0], [L, w], [0, w]], float))           # water inside: counter-clockwise
    bands, _ = CB.coast_bands(Domain([lake], "lake"), uniform(H), O)
    long_bands = [b for b in bands if b.record["m"] >= 40]
    assert len(long_bands) == 2 and _depths(long_bands) == [2]
    assert all(b.record["lfs_rows_min"] == 2 for b in long_bands)


def test_bay_drops_rows():
    """Converging normals (a bay): on a straight coast of 41 nodes 50 km apart, the normals of nodes 15-25 turned
    toward a point 3 h off the coast: two of them meet at the point, d_x = 3-5 h from the coast, so those cells get
    at most d_x / (BAY h) = 1-2 rows, the middle ones fewer than ROWS_MIN: the band ends at the bay; the straight
    coast keeps n_max = 4."""
    c, e1, e2 = _plane()
    n = 41
    x = (np.arange(n) - 20) * H / R_EARTH_M
    F = c + x[:, None] * e1                                       # along e1; the water (left) is +e2
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    nrm = np.repeat(e2[None], n, axis=0)
    focus = c + (3 * H / R_EARTH_M) * e2
    bay = np.abs(np.arange(n) - 20) <= 5
    d = focus - F[bay]
    nrm[bay] = d / np.linalg.norm(d, axis=1, keepdims=True)
    up = np.repeat(np.arange(n), CB.SUB)[: CB.SUB * (n - 1) + 1]  # samples: the nodes repeated (only nodes are read)
    sg = dict(F=F[up], NRM=nrm[up], N=n - 1)
    L = CB._limits(sg, uniform(H), 4, cKDTree(F))
    assert L["bay"][15:25].max() <= 2 and L["bay"][18:22].max() <= 1
    assert L["bay"][:14].min() > 100 and L["bay"][26:].min() > 100                # parallel: never meet
    plan, nc, _ = CB._plan(0, n - 1, L, 4, np.full(n - 1, 4))
    assert nc[:10].min() == 4 and nc[18:22].max() < CB.ROWS_MIN
    assert plan and all(pb <= 18 or pa >= 22 for pa, pb in plan)


def test_coast_row_fixed_in_the_sweeps(monkeypatch):
    """SWEEPS "size" sweeps with the coast row (and the end cuts) fixed: the band's coast nodes are fixed, unmoved
    nodes of the swept grid; its inner nodes moved."""
    import meshom.smooth as S
    calls = []
    orig = S.smooth

    def spy(mesh, h, fixed, iters=10, kind="angle", **kw):
        out = orig(mesh, h, fixed, iters=iters, kind=kind, **kw)
        calls.append((mesh.points.copy(), np.asarray(fixed).copy(), out.points.copy(), iters, kind))
        return out

    monkeypatch.setattr(S, "smooth", spy)
    bands, _ = CB.coast_bands(Domain([_island(_circle_km(1000.0))], "isl"), uniform(H), O)
    assert len(bands) == 1 and len(calls) == 1
    before, fixed, after, iters, kind = calls[0]
    assert (iters, kind) == (CB.SWEEPS, "size")
    np.testing.assert_array_equal(after[fixed], before[fixed])
    coast = bands[0].grid.points[bands[0].coast]
    d, _ = cKDTree(before[fixed]).query(coast)
    assert d.max() == 0.0                                        # every coast node is a fixed node, unmoved
    inner = bands[0].grid.points[~bands[0].coast]
    d, _ = cKDTree(before).query(inner)
    assert d.max() > 1e-6                                        # the inner nodes moved


def test_a_band_over_an_islet_is_dropped():
    """C39: a 2 km islet 15 km off the coast of the 1000 km island (h = 50 km), midway between two band nodes: the
    LFS samples, h apart along the normals, miss it, so the band would cover it (the fill then leaks into its land,
    the West Africa run's NaN field): the band is dropped and counted (1 band built without the islet)."""
    c, e1, e2 = _plane()
    big = _island(_circle_km(1000.0))
    bands, _ = CB.coast_bands(Domain([big], "isl"), uniform(H), O)
    assert len(bands) == 1
    p = bands[0].grid.points[bands[0].coast]
    q = p / (p @ c)[:, None] - c
    xy = np.c_[q @ e1, q @ e2] * R_EARTH_M / 1e3                  # the band's coast nodes in the plane (km)
    a = np.sort(np.arctan2(xy[:, 1], xy[:, 0]))
    am, rm = 0.5 * (a[len(a) // 2] + a[len(a) // 2 + 1]), float(np.hypot(*xy.T).mean()) + 15.0
    cx, cy = rm * np.cos(am), rm * np.sin(am)
    islet = _island([[cx - 1, cy - 1], [cx + 1, cy - 1], [cx + 1, cy + 1], [cx - 1, cy + 1]])
    bands, info = CB.coast_bands(Domain([big, islet], "isl"), uniform(H), O)
    assert bands == [] and info["islands"] == 1


def test_bands_through_m14():
    """continent at scale 4 with coast bands: the bands are fixed blocks (records of kind "band"), their coast nodes
    are boundary nodes of the mesh, nothing is inverted."""
    from meshom.cases import CASES
    from meshom.mesh import build_topology, signed_area
    from meshom.methods.m14_blocks import run
    from meshom.size import scaled

    dom, h = CASES["continent"]
    mesh, info = run(dom, scaled(h, 4.0), {"coast_bands": True})
    bands = [r for r in info["fixed_blocks"] if r.get("kind") == "band"]
    assert bands and info["coast_bands"]["bands"] == len(bands)
    assert (signed_area(mesh) > 0).all()
    topo = build_topology(mesh)
    assert topo.boundary_node.sum() == (mesh.node_flag == 1).sum()
    assert all(b["seam"]["mouth_edges"] > 0 for b in bands)


def test_count_row_follows_h_offshore(monkeypatch):
    """C50: the along count follows h on the row at COUNT_ROW x the band's depth. A round island of radius 1500 km at
    h = 50 km takes 3 rows; counted on the coast row (0) the offshore row is (1 + 3 h / rho) longer (along p99 0.094,
    181 cells); counted on the offshore row (COUNT_ROW = 1) that row meets the fill at h (0.002) with 198 cells, and the
    coast row takes the error instead, within the turning rule's TURN_TOL (0.095)."""
    loop = _island(_circle_km(1500.0))
    assert CB.COUNT_ROW == 1.0
    rec = {}
    for cr in (1.0, 0.0):
        monkeypatch.setattr(CB, "COUNT_ROW", cr)
        bands, _ = CB.coast_bands(Domain([loop], "isl"), uniform(H), O)
        assert len(bands) == 1 and _depths(bands) == [3]
        rec[cr] = bands[0].record
    assert rec[1.0]["m"] > 1.05 * rec[0.0]["m"]
    a0 = [r["along_p99"] for r in rec[0.0]["row_stats"]]
    a1 = [r["along_p99"] for r in rec[1.0]["row_stats"]]
    assert a0[0] < 0.01 and a0[-1] > 0.08                          # coast row at h, offshore row long
    assert a1[-1] < 0.01 and 0.08 < a1[0] <= CB.TURN_TOL           # offshore row at h, coast row short


def test_coast_row_ends_on_the_raw_coast():
    """C50: the coast row is blended onto the raw coast over END_RAW nodes at each end. A straight raw coast and a
    filtered coast 0.3 h off it (toward the water): the band's two end nodes lie on the raw coast, its middle nodes on
    the filtered one (0.3 h off), so the walk to the next block starts where the band ends."""
    c, e1, e2 = _plane()
    n = 41
    x = (np.arange(CB.SUB * n + 1) / CB.SUB - n / 2) * H / R_EARTH_M
    R = c + x[:, None] * e1
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    F = R + 0.3 * H / R_EARTH_M * e2
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    k = np.arange(len(R))
    sg = dict(R=R, F=F, NRM=np.repeat(e2[None], len(R), axis=0), seg=k, t=np.zeros(len(R)), loop=0, N=n)
    res, bad, _ = CB._build(sg, 5, 35, np.full(30, 3), uniform(H))
    assert bad is None
    grid, coast, (run, s, e) = res
    cn = grid.points[coast]
    off = lambda P: (P @ e2) * R_EARTH_M / H                       # noqa: E731  distance off the raw coast, in h
    end = cn[[np.argmin(cn @ e1), np.argmax(cn @ e1)]]
    np.testing.assert_allclose(off(end), 0.0, atol=1e-9)
    mid = cn[np.abs(cn @ e1) * R_EARTH_M < 5 * H]
    np.testing.assert_allclose(off(mid), 0.3, atol=1e-3)
    assert (s[1], e[1]) == (5 * CB.SUB, 35 * CB.SUB)             # the run ends located at the same raw samples


def test_band_gap_of_the_placement():
    """C50: the automatic placement keeps band_gap h (4.5) from a coastal band's nodes and block_gap (6) from the other
    fixed blocks' nodes (B59): nodes 0.25 h apart on a great circle, a strait block's node at 0 and a band's at 40 h."""
    from meshom.methods.m14.place import clear_of_fixed

    c, e1, _ = _plane()

    def at(s):
        ang = (np.atleast_1d(s) * H / R_EARTH_M)[:, None]
        return np.cos(ang) * c + np.sin(ang) * e1

    s = np.arange(160) * 0.25 + 0.125                              # units of h
    P, hv = at(s), np.full(len(s), H)
    o = dict(DEFAULTS)
    assert (o["block_gap"], o["band_gap"]) == (6.0, 4.5)
    free = s[clear_of_fixed(P, hv, o, at(0.0), at(40.0))]
    assert free.min() == 6.125 and free.max() == 35.375            # > 6 h from the strait, > 4.5 h from the band
    free = s[clear_of_fixed(P, hv, dict(o, band_gap=None), at(0.0), at(40.0))]
    assert free.max() == 33.875                                    # None: block_gap from the band too


def _corner_coast(theta_deg: float, L: float = 1500.0) -> np.ndarray:
    """A straight coast that turns by theta toward the water at (0, 0) (a concave corner seen from the water): walked
    with the water on the left, (-L, 0) -> (0, 0) -> L (cos theta, sin theta), closed around the land below."""
    t = np.radians(theta_deg)
    p2 = L * np.array([np.cos(t), np.sin(t)])
    return _poly(np.array([[-L, 0.0], [0.0, 0.0], p2, [p2[0], -L], [-L, -L]]))


def _band_crossings(dom, h, bands):
    """Crossing chords of the coast loops the stitch re-assembles with the bands' outlines (stitch.merged_loops)."""
    from meshom.domain import crossing_chords
    from meshom.methods.m14.stitch import merged_loops

    ml, _, _ = merged_loops(dom, h, bands, [])
    P = np.vstack(ml)
    off = np.r_[0, np.cumsum([len(x) for x in ml])]
    ia = np.arange(off[-1])
    ib = ia + 1
    ib[off[1:] - 1] = off[:-1]
    return int(crossing_chords(P, ia, ib).sum())


def test_bands_at_a_concave_corner_do_not_cross():
    """The 50 km crash (the Tanga corner near the Pemba channel, the Ronne front, the Arica bend): at a concave corner
    the two segments' bands end GAP nodes from it, 2 h apart, and their end cuts converge; 4 rows deep they crossed at
    turns of 40-50 deg (they meet about cot(theta / 2) h out; sharper, the LFS rule stops them), the re-assembled coast
    loop crossed itself, the stitch fill's cross field was NaN there and m02's pack overflowed. The corner wedge rule
    (C36) bounds the end cells on both sides by the rays' meeting distance d_x / (BAY h): no crossing, no band dropped,
    the bands ramp up from 2 rows at the corner and keep their 4 rows on the straight coast. (Below about 40 deg the
    filtered coast does not confirm the corner: one band bends round it under the bay rule.)"""
    h = uniform(H)
    for theta in (35.0, 45.0, 60.0, 90.0):
        dom = Domain([_corner_coast(theta)], "corner")
        bands, info = CB.coast_bands(dom, h, O)
        assert info["crossed"] == 0 and max(_depths(bands)) == 4
        assert _band_crossings(dom, h, bands) == 0, theta
        if theta >= 45.0:                                            # the two bands at the concave corner
            a, b = bands[0].record, bands[1].record
            assert a["wedge"][1] <= 1 and b["wedge"][0] <= 1
            assert a["row_stats"] and a["rows"] == [2, 4.0, 4] and b["rows"][0] == 2


def test_crossing_bands_are_dropped(monkeypatch):
    """C36: a band whose outline still crosses a chord of the re-assembled coast loops (the fill needs simple loops) is
    dropped. Without the corner wedge rule the two bands at a 45 deg concave corner cross: both go, the rest stay."""
    h = uniform(H)
    dom = Domain([_corner_coast(45.0)], "corner")
    monkeypatch.setattr(CB, "_wedge", lambda A, B, h: np.inf)
    bands, info = CB.coast_bands(dom, h, O)
    assert info["crossed"] == 2 and info["bands"] == len(bands) == 3
    assert _band_crossings(dom, h, bands) == 0
    assert [b.record["fixed_index"] for b in bands] == [0, 1, 2]
    monkeypatch.setattr(CB, "_crossing", lambda *a: set())             # the old source: the loop crosses itself
    bands, info = CB.coast_bands(dom, h, O)
    assert info["crossed"] == 0 and len(bands) == 5 and _band_crossings(dom, h, bands) > 0
