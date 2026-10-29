"""Hand-placed blocks (m14/hand.py, decisions B60, U6): the sides' default rule and the snap, one block through m14 in
conforming mode, and a cut-mode block over land and sea cut by the mask."""
import numpy as np
import pytest

from meshom.cases import CASES
from meshom.compete import irregular_counts
from meshom.domain import CHANNEL_WIDTH_M, continent_frame
from meshom.geo import R_EARTH_M, xyz_to_lonlat
from meshom.mesh import build_topology, signed_area
from meshom.methods.m14.hand import hand_blocks, resolve
from meshom.methods.m14_blocks import DEFAULTS, run
from meshom.size import scaled


def _channel_corners(half_len_m: float, off_m: float) -> list:
    """Four corners on the continent's channel banks (off_m into the water), +-half_len_m along it from the cap
    centre, counter-clockwise: right bank forward, then across, the left bank back, across."""
    c, t, n = continent_frame()
    w = (0.5 * CHANNEL_WIDTH_M - off_m) / R_EARTH_M
    out = []
    for s, side in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        a = s * half_len_m / R_EARTH_M
        p = np.cos(w) * (np.cos(a) * c + np.sin(a) * t) + np.sin(w) * side * n
        lo, la = xyz_to_lonlat(p[None])
        out.append([float(lo[0]), float(la[0])])
    return out


def test_default_sides_and_snap():
    """Corners 2 km off the banks snap onto them (snap 1 h = 15 km); the banks become coast sides (forward path along
    their loop = the chord), the cross sides cut; snap_m = 0 keeps the corners where they are."""
    dom, h = CASES["continent"]
    r = resolve({"corners": _channel_corners(500e3, 2e3)}, dom.loops, h)
    assert r["kinds"] == ["coast", "cut", "coast", "cut"] and r["snapped"] == [True] * 4
    assert r["loops"][0] != r["loops"][2] and r["loops"][1] == r["loops"][3] == -1
    r0 = resolve({"corners": _channel_corners(500e3, 2e3), "snap_m": 0.0}, dom.loops, h)
    assert r0["snapped"] == [False] * 4 and r0["kinds"] == ["coast", "cut", "coast", "cut"]
    r1 = resolve({"corners": _channel_corners(500e3, 2e3), "sides": ["lat", "cut", "lat", "cut"]}, dom.loops, h)
    assert r1["kinds"] == ["lat", "cut", "lat", "cut"]


def test_offset_side_of_a_coastal_band():
    """[coast, cut, offset, cut] on the cap's outer rim (azimuth 115 -> 155 deg from the cap centre, forward along the
    rim), 300 km wide: the offset side is the coast side extruded 300 km along its normals, its azimuth monotone (no
    hook at the corners)."""
    dom, h = CASES["continent"]
    c, t, n = continent_frame()
    north, east = (t + n) / np.sqrt(2.0), (t - n) / np.sqrt(2.0)           # t = (north + east) / sqrt 2, n = c x t
    rho = np.radians(30.0)

    def at(az, d):
        a = np.radians(az)
        lo, la = xyz_to_lonlat((np.cos(d) * c + np.sin(d) * (np.cos(a) * north + np.sin(a) * east))[None])
        return [float(lo[0]), float(la[0])]
    D = 300e3 / R_EARTH_M
    r = resolve({"corners": [at(115, rho), at(155, rho), at(155, rho + D), at(115, rho + D)],
                 "sides": ["coast", "cut", "offset", "cut"]}, dom.loops, h)
    assert r["kinds"] == ["coast", "cut", "offset", "cut"]
    q = r["sides_xyz"][2]
    dist = R_EARTH_M * np.arccos(np.clip(q @ c, -1, 1)) - R_EARTH_M * rho
    assert np.abs(dist - 300e3).max() < 2e3
    az = np.degrees(np.arctan2(q @ east, q @ north))
    assert (np.diff(az) < 0).all()                                            # from 155 back to 115


def test_hand_block_in_the_channel_through_m14():
    """A hand block over 1000 km of the channel (scale 4: 60 km in the channel; cols = 3 across) instead of the strait
    block: built first, the coast loops re-assembled with its mouths, its banks the mesh boundary. Without m14's own
    optimisation stage, as meshom run calls it before its land cut (Z1): CUT_FIXED indexes the stitch's elements, which
    the stage may merge (with B68's layout it removed 3 here)."""
    dom, h0 = CASES["continent"]
    h = scaled(h0, 4.0)
    spec = {"name": "channel", "corners": _channel_corners(500e3, 0.0), "cols": 3}
    m, info = run(dom, h, {"blocks": [spec], "strait_blocks": False, "optimize": False})
    topo = build_topology(m)
    assert (signed_area(m) > 0).all() and (topo.elems_per_node > 0).all()
    ic = irregular_counts(m)
    assert ic["euler_ok"] and ic["euler_sum"] == 4 * (2 - len(dom.loops))
    fb = info["fixed_blocks"]
    assert len(fb) == 1 and fb[0]["kind"] == "hand" and fb[0]["status"] == "built" and fb[0]["n"] == 3
    assert fb[0]["sides"] == ["coast", "cut", "coast", "cut"] and [x["cells"] for x in fb[0]["mouths"]] == [3, 3]
    assert fb[0]["inverted"] == 0 and info["stitch"]["n_fixed_blocks"] == 1
    nb = int(topo.boundary_node.sum())
    assert np.array_equal(m.node_flag == 1, topo.boundary_node) and nb == int((m.node_flag == 1).sum())
    from meshom.methods import m14_blocks                # the record meshom run's land cut takes (CUT_FIXED)
    rec = m14_blocks.CUT_FIXED
    assert rec["protect"].shape == (m.n_elems,) and rec["protect"][m.n_tris:].sum() == rec["protect"].sum() > 0
    assert topo.boundary_node[rec["coast_nodes"]].all() and len(rec["coast_nodes"]) > 0


def test_cut_mode_block_over_land_is_cut_by_the_mask():
    """A lat / cut block across the cap's rim (its west half on land, corners on land and at sea), variant A: built
    on the closed sphere, its cells on land removed by the mask (not protected), the cut mesh valid."""
    dom, h0 = CASES["continent"]
    h = scaled(h0, 4.0)
    spec = {"name": "rim", "corners": [[40.0, -15.0], [60.0, -15.0], [60.0, 5.0], [40.0, 5.0]],
            "sides": ["lat", "cut", "lat", "cut"], "h_m": 200e3}
    fixed, recs = hand_blocks([spec], dom, h, dict(DEFAULTS, cut=True))
    assert recs[0]["b22"] and not fixed[0].coast.any() and fixed[0].runs == []
    m, info = run(dom, h, {"cut": True, "blocks": [spec]})
    assert (signed_area(m) > 0).all() and irregular_counts(m)["euler_ok"]
    assert info["cut"]["n_fixed_cells_removed"] > 0 and info["fixed_blocks"][0]["status"] == "built"


def test_run_land_cut_makes_the_fitted_banks_the_coast():
    """meshom run's land cut (run.cut_land) with a built coast-sided block (the record m14_blocks.run leaves in
    CUT_FIXED): its cells are water and protected, the element across each fitted bank edge is land, as m14's own
    cut (m14/cut.py). The case: 10 km quads, the fitted bank 0.75 h offshore of
    the raw coast, where the mask alone keeps the fill across the bank wet."""
    from meshom.domain import Domain
    from meshom.geo import lonlat_to_xyz
    from meshom.mesh import Mesh
    from meshom.methods.m14.cut import cut
    from meshom.run import cut_land
    from meshom.size import Uniform

    x, y = np.arange(-4, 5.0) * 10e3, (np.arange(-4, 5.0) + 0.75) * 10e3
    xx, yy = np.meshgrid(x, y, indexing="ij")
    pts = lonlat_to_xyz(np.degrees(xx.ravel() / R_EARTH_M), np.degrees(yy.ravel() / R_EARTH_M))
    nid = np.arange(81).reshape(9, 9)
    q = np.array([[nid[i, j], nid[i + 1, j], nid[i + 1, j + 1], nid[i, j + 1]] for i in range(8) for j in range(8)])
    m = Mesh(pts, np.empty((0, 3), np.int64), q, np.zeros(81, np.int8))
    bx, by = np.array([-100e3, 100e3, 100e3, -100e3]), np.array([0.0, 0.0, 100e3, 100e3])
    land = Domain([lonlat_to_xyz(np.degrees(bx / R_EARTH_M), np.degrees(by / R_EARTH_M))], "basin")
    fx = np.array([0 if 2 <= i < 6 and 4 <= j < 7 else -1 for i in range(8) for j in range(8)])
    bank = nid[2:7, 4]
    bk = np.column_stack([bank[:-1], bank[1:]])
    st = {"fixed_id": fx, "fixed_protect": np.array([True]), "fixed_coast_nodes": bank, "fixed_bank_edges": bk,
          "is_block": fx >= 0,
          "outline_edges": np.empty((0, 2), np.int64), "block_nodes": np.unique(q[fx >= 0]), "info": {}}
    a, _, _ = cut(m, st, land, Uniform(10e3))
    b, info = cut_land(m, land, fixed={"protect": fx >= 0, "coast_nodes": bank, "bank_edges": bk})

    def bank_edges(mesh: Mesh) -> int:
        t = build_topology(mesh)
        e = mesh.points[t.edges[t.edge_elems[:, 1] < 0]]
        hit = [np.any(np.all(np.isclose(e, [p, r], atol=1e-12), axis=(1, 2)) |
                      np.all(np.isclose(e, [r, p], atol=1e-12), axis=(1, 2)))
               for p, r in zip(pts[bank[:-1]], pts[bank[1:]])]
        return int(np.sum(hit))

    assert bank_edges(a) == 4 and bank_edges(b) == 4 and b.n_elems == a.n_elems == 36
    assert info["n_block_protected"] == 12


def test_nearest_shore_finds_a_long_segment_past_dense_vertices():
    """B60 snaps a corner onto its nearest coast point. A straight bank of one 50 km segment, 1 km from the corner,
    and a 2 km islet of 64 vertices 7.6 km away: the islet's vertices are the nearest ones, the bank's segment is the
    nearest coast. Shore.nearest returns the bank, and resolve() moves the corner 1 km onto it (before:
    the 16 nearest vertices held only the islet, the corner snapped 7.6 km onto it and the stitch crashed)."""
    from meshom.geo import arc_length, lonlat_to_xyz
    from meshom.methods.m14.hand import Shore
    from meshom.size import uniform

    def xyz(x, y):
        return lonlat_to_xyz(np.degrees(np.asarray(x) / R_EARTH_M), np.degrees(np.asarray(y) / R_EARTH_M))

    t = -2 * np.pi * np.arange(64) / 64
    loops = [xyz(np.array([-25e3, 25e3, 25e3, -25e3]), np.array([0, 0, 60e3, 60e3])),
             xyz(-5e3 + 1e3 * np.cos(t), 8e3 + 1e3 * np.sin(t))]
    p = xyz(0.0, 1e3)
    L, x = Shore(loops).nearest(p)
    assert L == 0 and x[3] == pytest.approx(1e3, rel=1e-6)
    lo, la = xyz_to_lonlat(xyz(np.array([0, 15e3, 15e3, 0]), np.array([1e3, 1e3, 35e3, 35e3])))
    r = resolve(dict(corners=np.column_stack([lo, la]).tolist(), sides=["coast", "cut", "cut", "cut"]), loops,
                uniform(10e3))
    assert r["loops"][0] == 0
    assert R_EARTH_M * float(arc_length(r["corners"][0][None], p[None])[0]) == pytest.approx(1e3, rel=1e-6)



def test_rep_cut_record_follows_the_repair():
    """m14_blocks_rep as meshom run calls it under a land cut (the closed sphere, the coast as opts["land"], a hand
    block resolved on the true coast, no optimisation stage): the repair and the fit renumber nodes and elements, so
    the record m14_blocks.CUT_FIXED is carried onto the returned mesh (before: it indexed m14's
    mesh, 14056 elements against 13810 at scale 4) and run.cut_land takes it. The block is frozen in those stages
    (decisions R24; before it the repair remeshed the whole block, nothing stayed protected), see
    test_frozen_blocks.py."""
    from meshom.domain import closed_sphere
    from meshom.methods import METHODS, m14_blocks
    from meshom.run import cut_land

    dom, h0 = CASES["continent"]
    h = scaled(h0, 8.0)
    spec = resolve({"name": "channel", "corners": _channel_corners(500e3, 0.0), "cols": 3}, dom.loops, h)
    m = METHODS["m14_blocks_rep"](closed_sphere(), h, {"blocks": [spec], "land": dom, "optimize": False})
    rec = m14_blocks.CUT_FIXED
    assert rec["protect"].shape == (m.n_elems,)
    out, info = cut_land(m, dom, fixed=rec)
    assert info["n_block_protected"] == int(rec["protect"].sum()) and out.n_elems > 0


def test_carry_cut_fixed_by_node_position():
    """rep_variants.carry_cut_fixed: the record of a mesh carried onto a renumbered copy (nodes and elements
    permuted): the same cells protected; a protected cell split into two triangles and a moved bank node drop out."""
    from meshom.geo import lonlat_to_xyz
    from meshom.mesh import Mesh
    from meshom.methods.rep_variants import carry_cut_fixed

    xx, yy = np.meshgrid(np.arange(5.0), np.arange(4.0), indexing="ij")
    pts = lonlat_to_xyz(xx.ravel(), yy.ravel())
    nid = np.arange(20).reshape(5, 4)
    q = np.array([[nid[i, j], nid[i + 1, j], nid[i + 1, j + 1], nid[i, j + 1]] for i in range(4) for j in range(3)])
    old = Mesh(pts, np.empty((0, 3), np.int64), q, np.zeros(20, np.int8))
    protect = np.zeros(12, dtype=bool)
    protect[[0, 1, 2, 3]] = True                        # the cells of column i = 0 and the first of i = 1
    bank = nid[0, :]
    rec = {"protect": protect, "coast_nodes": bank, "bank_edges": np.column_stack([bank[:-1], bank[1:]])}
    rng = np.random.default_rng(0)
    perm = rng.permutation(20)                          # new id of old node k: inv[k]
    inv = np.argsort(perm)
    P = pts[perm].copy()
    eperm = rng.permutation(12)
    new = Mesh(P, np.empty((0, 3), np.int64), inv[q][eperm], np.zeros(20, np.int8))
    r = carry_cut_fixed(rec, old, new)
    assert np.array_equal(np.sort(np.flatnonzero(r["protect"])), np.sort(np.flatnonzero(protect[eperm])))
    assert np.array_equal(r["coast_nodes"], inv[bank]) and np.array_equal(r["bank_edges"], inv[rec["bank_edges"]])
    c = inv[q[3]]                                        # cell 3 split into two triangles, bank node 3 moved
    keep = eperm != 3
    P2 = P.copy()
    P2[inv[bank[3]]] = lonlat_to_xyz(np.array([-0.5]), np.array([3.0]))[0]
    new2 = Mesh(P2, np.array([[c[0], c[1], c[2]], [c[0], c[2], c[3]]]), inv[q][eperm][keep], np.zeros(20, np.int8))
    r2 = carry_cut_fixed(rec, old, new2)
    assert int(r2["protect"].sum()) == 2 and not r2["protect"][:2].any()   # cells 0, 1 (cell 2 has bank node 3)
    assert np.array_equal(r2["coast_nodes"], inv[bank[:3]]) and len(r2["bank_edges"]) == 2
