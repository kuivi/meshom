"""m14 kernels pinned to their stated rules (docs/methods/m14_blocks.md):
- placement accepts a region iff F = (max lambda - min lambda)/2 <= 0.08 and max over the background triangles of
  h |grad lambda| <= 0.05, grad = the P1 gradient on each triangle, not the slope along its edges (a synthetic lambda,
  0 on the outline, run through the real Dirichlet solve, just below / just above each bound);
- stitching shares every block-outline node between the block and the fill by node id (each outline edge has one
  block and one fill element, no duplicate node) and never moves a coast node in the seam-band smoothing;
- the cut (FESOM2 clean-up) leaves one basin, no pinched node (the elements at every node form one fan), no spur,
  and Euler sum = 4 (2 - boundary loops)."""
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

import meshom.smooth as smooth_mod
from meshom.cases import CASES
from meshom.compete import irregular_counts
from meshom.delaunay import adjacency, background_mesh
from meshom.domain import closed_sphere
from meshom.geo import R_EARTH_M, arc_length
from meshom.mesh import Mesh, build_topology, orient_ccw, signed_area
from meshom.methods.m14 import place as place_mod
from meshom.methods.m14.cut import clean, extract, half_edges
from meshom.methods.m14.fitness import Fitness
from meshom.methods.m14_blocks import DEFAULTS, build_blocks, run
from meshom.methods.m14.stitch import stitch
from meshom.size import scaled, uniform


# ------------------------------------------------------------------------------------------------ acceptance rule

def _flat_fitness() -> Fitness:
    """Closed sphere, h = 400 km, everything admissible, rho_c = 0 (growth limited by a_max only)."""
    h = uniform(4e5)
    bg = background_mesh(closed_sphere(), h, factor=1.0)
    n = bg.n_nodes
    inf, zero = np.full(n, np.inf), np.zeros(n)
    return Fitness(bg, np.zeros((n, 3)), h(bg.points), zero, zero, zero, np.zeros((0, 3)), np.zeros(0), inf, inf,
                   inf, adjacency(bg.tris))


def _p1_grad(P, T, lam, hv):
    """h |grad lam| per triangle, computed apart from place.tri_grad: a 2 x 2 solve in an orthonormal basis of the
    chord triangle's plane, h = the mean of its nodes."""
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    e1 = (b - a) / np.linalg.norm(b - a, axis=1, keepdims=True)
    nrm = np.cross(b - a, c - a)
    e2 = np.cross(nrm / np.linalg.norm(nrm, axis=1, keepdims=True), e1)
    A = np.stack([np.stack([((b - a) * e1).sum(1), ((b - a) * e2).sum(1)], 1),
                  np.stack([((c - a) * e1).sum(1), ((c - a) * e2).sum(1)], 1)], 1) * R_EARTH_M
    rhs = np.stack([lam[T[:, 1]] - lam[T[:, 0]], lam[T[:, 2]] - lam[T[:, 0]]], 1)
    return np.linalg.norm(np.linalg.solve(A, rhs[..., None])[..., 0], axis=1) * hv[T].mean(axis=1)


def _synthetic_lambda(F=None, G=None):
    """region_lambda replacement: lambda_true = 0 on the region outline, growing with the ring distance from it,
    scaled to half-range F or to the gradient bound G (max over the triangles of h |grad lambda|, _p1_grad);
    rhs = L lambda_true, so the real Dirichlet solve (place.lambda_solve) must return lambda_true and its F / grad.
    The edge-slope maximum of the same lambda is at most G (a directional derivative), the old rule's value."""
    def region_lambda(fit, tris):
        nodes, Tl = np.unique(fit.bg.tris[tris], return_inverse=True)
        Tl = Tl.reshape(-1, 3)
        P, hv = fit.bg.points[nodes], fit.hv[nodes]
        n = len(nodes)
        E = place_mod.edge_list(Tl, n)
        g = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
        d = dijkstra(g, directed=False, indices=np.flatnonzero(place_mod.boundary_nodes(Tl, n)), min_only=True)
        s = d / d.max()
        if F is not None:
            lam_true = 2.0 * F * s
        else:
            lam_true = s * G / _p1_grad(P, Tl, s, hv).max()
            ln = R_EARTH_M * arc_length(P[E[:, 0]], P[E[:, 1]])
            edge = np.max(np.abs(lam_true[E[:, 0]] - lam_true[E[:, 1]]) * 0.5 * (hv[E[:, 0]] + hv[E[:, 1]]) / ln)
            assert edge <= 1.001 * G                                    # an edge slope is at most the gradient
        lam, Fs, grad = place_mod.lambda_solve(P, Tl, place_mod.cot_laplacian(P, Tl) @ lam_true, hv)
        assert np.abs(lam - lam_true).max() < 1e-9
        if F is not None:
            assert abs(Fs - F) < 1e-9                                   # F = half the range of lambda
        else:
            assert abs(grad / G - 1.0) < 1e-9                           # the P1 gradient, max over the triangles
        return nodes, lam, Fs, grad
    return region_lambda


@pytest.mark.parametrize("F, G, accepted", [(0.0799, None, True), (0.0801, None, False),
                                            (None, 0.0499, True), (None, 0.0501, False)])
def test_acceptance_rule(monkeypatch, F, G, accepted):
    monkeypatch.setattr(place_mod, "region_lambda", _synthetic_lambda(F, G))
    o = dict(DEFAULTS, grow="metric", a_max=400.0, max_blocks=1, bisect=2)
    if F is not None:
        o["grad_max"] = 1e9                                             # F bound alone
    else:
        o["f_max"] = 1e9                                                # gradient bound alone
    assert (o["f_max"], o["grad_max"]) in ((0.08, 1e9), (1e9, 0.05))   # the stated thresholds are the defaults
    regions, log = place_mod.place(_flat_fitness(), o)
    assert (len(regions) == 1) == accepted
    if accepted:
        r = regions[0]
        assert r.cells >= o["min_cells"] and (r.F <= 0.08 if F is not None else r.grad <= 0.05)
    else:
        assert log["rejected"] >= 1 and log["lambda_solves"] >= 1


@pytest.mark.parametrize("front, peer", [(False, None), (True, None), ("seeds", None), ("seeds", 6.0)])
def test_shared_front_centres_the_gap(front, peer):
    """B67: two given seeds 70 deg apart on the equator of a flat sphere (h = 400 km, rho_c = 0, everything
    admissible; discs up to a_max = 400 cells, radius 11.3 h, so a disc grown alone crosses the seeds' bisector, the
    plane through the centre normal to s1 - s0). Without the shared front the first region crosses it by 2.5 h (the
    second keeps block_gap from the first); with it (True, or "seeds", the default) each region stays in its own half,
    more than block_gap / 2 from where the growth fronts meet, and both come within one bg edge (h) of that bound: the
    gap is centred on the bisector. The peer front (front_peer 6 h, B69) finds the two seeds peers (the first region
    crosses the bisector by 2.5 h) and centres the gap the same way."""
    assert DEFAULTS["front"] == "seeds"                                 # on by default for given seeds
    o = dict(DEFAULTS, grow="metric", a_max=400.0, max_blocks=2, bisect=2, front=front,
             seeds=[[-35.0, 0.0], [35.0, 0.0]], front_peer=peer)
    fit = _flat_fitness()
    regions, log = place_mod.place(fit, o)
    assert len(regions) == 2
    P, hv = fit.bg.points, fit.hv
    nrm = P[regions[1].seed] - P[regions[0].seed]
    nrm /= np.linalg.norm(nrm)
    d = [R_EARTH_M * np.arcsin(P[r.nodes] @ nrm) / hv[r.nodes] for r in regions]    # signed distance (h) from it
    gap = o["block_gap"]
    if not front:
        assert d[0].max() > 2.0                                         # the first region crosses the bisector
        return
    near = [-d[0].max(), d[1].min()]                                    # each region's nearest node to the bisector
    assert all(0.5 * gap - 0.25 < x < 0.5 * gap + 1.0 for x in near), near
    assert abs(near[0] - near[1]) < 1.0
    assert log["front"]["dropped"] == 0
    if peer is not None:
        assert log["front"]["peers"] == [[0, 1]]


def _seeds_of(fit, regions) -> list:
    from meshom.geo import xyz_to_lonlat

    lon, lat = xyz_to_lonlat(fit.bg.points[[r.seed for r in regions]])
    return np.column_stack([lon, lat]).tolist()


def test_peer_front_keeps_the_greedy_layout_of_its_own_seeds():
    """B69: on the flat sphere with a_max = 2400 cells the greedy places a 2400-cell disc and a 285-cell region in the
    space left beside it. Its two seeds given back: the Voronoi front (front_peer None) cuts the large disc at the
    seeds' bisector (both regions about 1210 cells, coverage 0.84 -> 0.76); the peer front sees that the gap lies far
    from the bisector (no peers) and grows them one after the other: the greedy's regions exactly."""
    assert DEFAULTS["front_peer"] == 6.0                                # on by default
    o = dict(DEFAULTS, grow="metric", a_max=2400.0, bisect=2, drop_small=None)
    fit = _flat_fitness()
    rg, _ = place_mod.place(fit, o)
    assert [round(r.cells) for r in rg] == [2400, 285]
    seeds = _seeds_of(fit, rg)
    rf, _ = place_mod.place(fit, dict(o, seeds=seeds, front_peer=None))
    assert max(r.cells for r in rf) < 1300                              # the Voronoi cells cut the large disc
    rp, log = place_mod.place(fit, dict(o, seeds=seeds, front_peer=6.0))
    assert log["front"]["peers"] == []
    assert len(rp) == 2 and all(np.array_equal(a.nodes, b.nodes) for a, b in zip(rp, rg))


def test_drop_small_leaves_the_small_region_to_the_fill():
    """B68: the flat sphere with a_max = 400 cells holds four 400-cell discs and a 121-cell region in the space left.
    drop_small = 0.5 drops the regions below half the median (400): the 121-cell one; the greedy reruns without
    seeding there and places the four discs exactly as before."""
    assert DEFAULTS["drop_small"] == 0.25                               # on by default (121 > 0.25 x 400 stays)
    o = dict(DEFAULTS, grow="metric", a_max=400.0, bisect=2, drop_small=None)
    fit = _flat_fitness()
    rg, _ = place_mod.place(fit, o)
    assert [round(r.cells) for r in rg] == [399, 400, 400, 400, 121]
    rd, log = place_mod.place(fit, dict(o, drop_small=0.5))
    assert [round(x) for x in log["drop_small"]["dropped"]] == [121] and log["drop_small"]["reruns"] == 1
    assert len(rd) == 4 and all(np.array_equal(a.nodes, b.nodes) for a, b in zip(rd, rg[:4]))


def test_water_place_seeds_in_the_water_and_counts_water_cells():
    """B70: under a land cut (fit.wet) the greedy seeds only in the water and counts a region's cells in the water.
    The flat sphere (every node equally clear: the land-blind greedy seeds at node 0) with a land cap of 2000 km
    around node 0: the land-blind greedy seeds its first region on the land; the water-aware one seeds every region
    in the water, and a region's value is its triangles' cells times their wet-node share."""
    import dataclasses

    assert DEFAULTS["water_place"] is True                              # on by default

    fit = _flat_fitness()
    P = fit.bg.points
    wet = P @ P[0] < np.cos(2.0e6 / R_EARTH_M)
    fit = dataclasses.replace(fit, wet=wet)
    o = dict(DEFAULTS, grow="metric", a_max=400.0, bisect=2, drop_small=None)
    blind, _ = place_mod.place(fit, dict(o, water_place=False))
    assert not wet[blind[0].seed] and all(r.water is None and r.value == r.cells for r in blind)
    rw, _ = place_mod.place(fit, o)
    assert len(rw) >= 3 and all(wet[r.seed] for r in rw)
    _, a = place_mod.tri_capacity(fit)
    aw = a * wet[fit.bg.tris].mean(axis=1)
    for r in rw:
        assert r.value == r.water == pytest.approx(float(aw[r.tris].sum()))
    assert any(r.water < r.cells - 10.0 for r in rw)                   # a region grown over the land (not stopped)


def test_place_region_keeps_every_block_inside_the_polygon():
    """B106: with `place_region` (a spherical polygon, [[lon, lat], ...]) the blocks are seeded and grown only on bg
    nodes inside it: on the flat sphere every region's nodes lie inside a 60 x 40 degree box, while the placement
    without it puts nodes outside the box."""
    from meshom.geo import lonlat_to_xyz
    from meshom.size import polygon_inside

    assert DEFAULTS["place_region"] is None                             # off by default
    fit = _flat_fitness()
    box = [[-30.0, -20.0], [30.0, -20.0], [30.0, 20.0], [-30.0, 20.0]]
    poly = lonlat_to_xyz(*np.asarray(box).T)
    o = dict(DEFAULTS, grow="metric", a_max=100.0, min_cells=20.0, block_gap=1.0, min_clear=1.0, bisect=2,
             drop_small=None, max_blocks=None)
    free, _ = place_mod.place(fit, o)
    assert any(not polygon_inside(fit.bg.points[r.nodes], poly).all() for r in free)
    inside, _ = place_mod.place(fit, dict(o, place_region=box))
    assert len(inside) >= 1
    for r in inside:
        assert polygon_inside(fit.bg.points[r.nodes], poly).all()


def test_max_blocks_none_places_until_no_seed_is_clear():
    """B8: max_blocks None lifts the cap (the default of m14_blocks_rep; 32 for the other m14 variants): the greedy
    places the capped run's regions first, then goes on while a seed is min_clear from the non-admissible nodes."""
    from meshom.methods.rep_variants import REP_MAX_BLOCKS

    assert DEFAULTS["max_blocks"] == 32 and REP_MAX_BLOCKS is None      # B8
    fit = _flat_fitness()
    o = dict(DEFAULTS, grow="metric", a_max=100.0, min_cells=20.0, block_gap=1.0, min_clear=1.0, bisect=2,
             drop_small=None)
    r4, _ = place_mod.place(fit, dict(o, max_blocks=4))
    rn, _ = place_mod.place(fit, dict(o, max_blocks=None))
    assert len(r4) == 4 and len(rn) > 8
    assert all(np.array_equal(a.nodes, b.nodes) for a, b in zip(rn, r4))


@pytest.mark.parametrize("method", ["m14_blocks", "m14_blocks_rep"])
def test_meshom_run_cut_hands_the_land_to_the_placement(tmp_path, monkeypatch, method):
    """B70: meshom run with treatment "cut" gives the method the closed sphere and cuts the land afterwards; its worker
    hands the true coast to the m14 methods and to the variants composed on them (m14_blocks_rep; opts `land`), so the
    placement seeds only in the water (m14.json seeds_on_land = 0, the regions' water coverage recorded)."""
    import json
    import pickle

    from meshom.domain import Domain, in_water
    from meshom.geo import lonlat_to_xyz
    from meshom.run import _worker

    lon = np.linspace(360.0, 0.0, 90, endpoint=False)                 # westward: the land (north of 10N) on the right
    land = Domain([np.asarray(lonlat_to_xyz(lon, np.full_like(lon, 10.0)))], "land")
    assert not in_water(land, np.array([[0.0, 0.0, 1.0]]))[0] and in_water(land, np.array([[0.0, 0.0, -1.0]]))[0]
    with open(tmp_path / "inputs.pkl", "wb") as f:
        pickle.dump((closed_sphere(), uniform(4e5), land, [], [], None), f)      # inputs.pkl: + passages (C51)
    monkeypatch.setenv("MESHOM_RUN_DIR", str(tmp_path))               # restored after the test (_generate sets it)
    _worker(json.dumps({"method": method, "opts": {"optimize": False, "size_tolerance": 0.0}, "smooth": False,
                        "dir": str(tmp_path), "inputs": str(tmp_path / "inputs.pkl"), "keep_basins_km2": None}))
    rec = json.loads((tmp_path / "m14.json").read_text())
    assert rec["n_regions"] >= 2 and rec["seeds_on_land"] == 0 and 0.0 < rec["region_water_coverage"] <= 1.0
    assert all(b["seed_wet"] for b in rec["blocks"])
    assert rec["params"]["max_blocks"] == (None if method == "m14_blocks_rep" else 32)      # B8: _rep without a cap


def test_overlap_retry_recomputes_the_retained_region(monkeypatch):
    """A block whose map overlaps itself is retried on 80 % of its growth prefix; lambda, F, the gradient, B and the
    cell estimate of the retained region are recomputed from its own triangles, not carried over."""
    import dataclasses

    from meshom.methods.m14 import blockmap, fitness as fit_mod
    from meshom.methods.m14.blockmap import BlockGrid

    fit = _flat_fitness()
    fit = dataclasses.replace(fit, area=np.full(fit.bg.n_nodes, 4 * np.pi / fit.bg.n_nodes))   # curvature: F > 0
    monkeypatch.setattr(fit_mod, "fitness", lambda dom, h, o, land=None: fit)
    before = []
    orig_place = place_mod.place

    def place(f, o):
        regs, log = orig_place(f, o)
        before.extend((r.F, len(r.tris), r.cells) for r in regs)
        return regs, log

    def grid(P, T, lam, h, o, in_r=None, u_ref=None):                 # overlaps on the first try only
        e = np.zeros(0, np.int64)
        info = {"overlap": not calls, "n_cells": 0 if not calls else 10**6, "inradius_cells": 0.0 if not calls else 99.0}
        calls.append(len(T))
        return BlockGrid(P[:0], e.reshape(0, 4), e.reshape(0, 2), e, info)

    calls: list = []
    monkeypatch.setattr(place_mod, "place", place)
    monkeypatch.setattr(blockmap, "block_grid", grid)
    o = dict(DEFAULTS, grow="metric", a_max=400.0, max_blocks=1, bisect=2, f_max=1e9, grad_max=1e9)
    _, regions, _, kept, plog, _ = build_blocks(closed_sphere(), uniform(4e5), o)
    r = regions[0]
    assert plog["overlap_retries"] == 1 and len(calls) == 2 and kept == [0]
    assert len(r.tris) == int(0.8 * before[0][1]) and r.F < before[0][0] and r.cells < before[0][2]
    nodes, lam, F, grad = place_mod.region_lambda(fit, r.tris)
    assert np.array_equal(r.nodes, nodes) and np.allclose(r.lam, lam) and (r.F, r.grad) == (F, grad)
    q, a = place_mod.tri_capacity(fit)
    assert np.isclose(r.cells, a[r.tris].sum()) and np.isclose(r.B, q[r.tris].sum())


# ------------------------------------------------------------------------------------------------ stitch

def test_stitch_shares_outlines_and_keeps_the_coast(monkeypatch):
    dom, h0 = CASES["continent"]
    h = scaled(h0, 4.0)
    o = dict(DEFAULTS, band_fill=8, size_tolerance=None)                # a wide band reaches the coast; the band
    fit, regions, grids, kept, plog, t = build_blocks(dom, h, o)        # smoothing last (no G38 fit after it)
    grids = [grids[k] for k in kept]
    assert grids
    calls = []
    real = smooth_mod.smooth

    def spy(mesh, hh, fixed, *a, **kw):
        out = real(mesh, hh, fixed, *a, **kw)
        calls.append((mesh, np.asarray(fixed), kw.get("kind"), out))
        return out

    monkeypatch.setattr(smooth_mod, "smooth", spy)
    mesh, st = stitch(dom, h, grids, o)
    before, after = calls[-2][0], calls[-1][3]                          # the joint band smoothing: size, then angle
    assert [c[2] for c in calls[-2:]] == ["size", "angle"] and after is mesh
    coast = mesh.node_flag == 1
    assert coast.sum() > 0 and all(c[1][coast].all() for c in calls[-2:])
    assert np.array_equal(before.points[coast], mesh.points[coast])    # coast nodes never move in the seam band
    # every outline node is one node of both the block and the fill: each outline edge has a block and a fill element
    topo = build_topology(mesh)
    n = mesh.n_nodes
    ol = np.sort(st["outline_edges"], axis=1)
    pos = np.searchsorted(topo.edges[:, 0] * n + topo.edges[:, 1], ol[:, 0] * n + ol[:, 1])
    assert np.array_equal(topo.edges[pos], ol)
    ee = topo.edge_elems[pos]
    assert (ee >= 0).all() and (st["is_block"][ee[:, 0]] != st["is_block"][ee[:, 1]]).all()
    assert len(np.unique(mesh.points, axis=0)) == n                     # no duplicated outline node
    offs = int(coast.sum()) + np.r_[0, np.cumsum([len(g.points) for g in grids])]
    for b, g in enumerate(grids):                                        # outline nodes keep the block's ids and points
        assert np.array_equal(before.points[offs[b] + g.outline], g.points[g.outline])
    assert np.array_equal(np.unique(st["outline_edges"]), np.unique(np.concatenate([offs[b] + g.outline
                                                                                   for b, g in enumerate(grids)])))
    fm = calls[-3][3]                                                   # the fill as m02 left it (last fill smoothing)
    nqb = int(st["is_block"].sum())
    assert (fm.n_tris, fm.n_quads) == (before.n_tris, before.n_quads - nqb)
    assert np.array_equal(before.points[before.tris], fm.points[fm.tris])          # merged by node id, nothing moved
    assert np.array_equal(before.points[before.quads[nqb:]], fm.points[fm.quads])
    assert (signed_area(mesh) > 0).all()


# ------------------------------------------------------------------------------------------------ cut

def _fans_per_node(mesh: Mesh) -> np.ndarray:
    """(N,) number of fans at each node: its elements joined across the edges at that node."""
    topo = build_topology(mesh)
    M = mesh.n_elems
    node = np.r_[mesh.tris.ravel(), mesh.quads.ravel()]
    elem = np.r_[np.repeat(np.arange(mesh.n_tris), 3), np.repeat(np.arange(mesh.n_quads) + mesh.n_tris, 4)]
    key = np.unique(node * M + elem)
    inner = topo.edge_elems[:, 1] >= 0
    u = np.r_[topo.edges[inner, 0], topo.edges[inner, 1]]
    e0, e1 = np.tile(topo.edge_elems[inner, 0], 2), np.tile(topo.edge_elems[inner, 1], 2)
    a, b = np.searchsorted(key, u * M + e0), np.searchsorted(key, u * M + e1)
    _, lab = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(key),) * 2), directed=False)
    pairs = np.unique((key // M) * len(key) + lab)                     # distinct (node, fan) pairs
    return np.bincount(pairs // len(key), minlength=mesh.n_nodes)


def _check_cut_result(m: Mesh) -> int:
    """One basin, one fan per node, >= 2 edge neighbours per element, Euler = 4 (2 - loops); returns loops."""
    topo = build_topology(m)
    inner = topo.edge_elems[:, 1] >= 0
    ncomp, _ = connected_components(coo_matrix((np.ones(inner.sum()), (topo.edge_elems[inner, 0],
                                                topo.edge_elems[inner, 1])), shape=(m.n_elems,) * 2), directed=False)
    assert ncomp == 1
    assert (_fans_per_node(m) == 1).all()                               # no pinched node
    assert topo.elem_neighbours.min() >= 2                              # no spur
    be = topo.edges[~inner]
    bn = np.unique(be)
    assert (np.bincount(be.ravel(), minlength=m.n_nodes)[bn] == 2).all()
    nl, _ = connected_components(coo_matrix((np.ones(len(be)), (np.searchsorted(bn, be[:, 0]),
                                             np.searchsorted(bn, be[:, 1]))), shape=(len(bn),) * 2), directed=False)
    nl = nl if len(bn) else 0
    assert irregular_counts(m)["euler_sum"] == 4 * (2 - nl)
    return nl


def _cube_grid(n: int) -> Mesh:
    """Closed quad sphere: each cube face an n x n grid; face f = 2 axis + (sign > 0), cell (i, j) -> f n^2 + j n + i."""
    t = np.linspace(-1.0, 1.0, n + 1)
    U, V = np.meshgrid(t, t)
    pts, quads, cell = [], [], {}
    j, i = np.divmod(np.arange(n * n), n)
    loc = j * (n + 1) + i
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for s in (-1.0, 1.0):
            P = np.zeros((n + 1, n + 1, 3))
            P[..., a], P[..., b], P[..., c] = s, U, V
            off = sum(len(p) for p in pts)
            pts.append(P.reshape(-1, 3))
            quads.append(off + np.column_stack([loc, loc + 1, loc + n + 2, loc + n + 1]))
    P = np.vstack(pts)
    key, inv = np.unique(np.round(P, 12), axis=0, return_inverse=True)
    key = key / np.linalg.norm(key, axis=1, keepdims=True)
    m = orient_ccw(Mesh(key, np.zeros((0, 3), np.int64), inv.ravel()[np.vstack(quads)], np.zeros(len(key), np.int8)))
    return m


def test_clean_removes_pinches_spurs_and_stray_basins():
    n = 12
    m = _cube_grid(n)

    def cell(face, i, j):
        return face * n * n + j * n + i

    wet = np.ones(m.n_elems, dtype=bool)
    for i in range(2, 5):                                               # a 3 x 3 island on face 1
        for j in range(2, 5):
            wet[cell(1, i, j)] = False
    wet[cell(1, 8, 7)] = wet[cell(1, 7, 8)] = False                     # two wet cells touching at one node: a pinch
    for i in range(0, 4):                                               # a land block with a one-cell inlet: a spur
        for j in range(9, 12):
            wet[cell(1, i, j)] = False
    wet[cell(1, 3, 10)] = True
    for i in range(3, 7):                                               # a land ring around a 2 x 2 lake on face 0
        for j in range(3, 7):
            wet[cell(0, i, j)] = 4 <= i <= 5 and 4 <= j <= 5
    keep, cnt = clean(half_edges(m), wet, m.n_nodes)
    assert cnt["pinched"] >= 1 and cnt["spurs"] >= 1 and cnt["components"] == 4   # every rule fired
    out, nmap, loops = extract(m, keep)
    assert _check_cut_result(out) == len(loops) == 4                  # island, inlet block, pinch L, lake ring


def test_cut_continent_fans_and_euler():
    dom, h0 = CASES["continent"]
    m, info = run(dom, scaled(h0, 4.0), {"cut": True})
    assert _check_cut_result(m) == info["cut"]["n_loops"] == len(dom.loops)


# ------------------------------------------------------------------------- the seam edge rows

@pytest.mark.parametrize("bad_row, in_seam", [(3, True), (4, False)])
def test_seam_band_edges_are_rows_0_to_3(bad_row, in_seam):
    """report.measure: the seam band's edges are those of the fill rows 0-3 (as its elements and nodes), not 0-5.
    A strip of 14 x 3 quads of 10 km; block = columns 0-1, outline = the edges at x = 2; fill column c has row c - 2.
    h = 20 km at the 4 horizontal edges of the fill row `bad_row` (|ln s/h| = ln 2 there), 10 km elsewhere."""
    from helpers import structured_patch
    from meshom.methods.m14.report import measure

    nx, ny, dx = 14, 3, 1e4
    m = structured_patch(nx, ny, dx, 40.0, 55.0)
    j, i = np.divmod(np.arange(nx * ny), nx)
    is_block = i < 2
    node = lambda jj, ii: jj * (nx + 1) + ii                           # noqa: E731
    ol = np.array([[node(jj, 2), node(jj + 1, 2)] for jj in range(ny)])
    c = bad_row + 2
    bad = np.array([[node(jj, c), node(jj, c + 1)] for jj in range(ny + 1)])
    mid = m.points[bad[:, 0]] + m.points[bad[:, 1]]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)

    def h(x):
        d = R_EARTH_M * arc_length(x[:, None, :], mid[None, :, :]).min(axis=1)
        return np.where(d < 1e3, 2e4, 1e4)

    st = {"is_block": is_block, "outline_edges": ol, "block_nodes": np.unique(m.quads[is_block])}
    out = measure(m, h, st)
    assert out["seam_band"]["elems"] == 4 * ny                         # fill rows 0-3
    p99 = out["seam_band"]["ln_sh_p99"]
    assert (abs(p99 - np.log(2)) < 1e-3) if in_seam else (p99 < 1e-3)


def test_overlap_retry_must_pass_the_acceptance_rule_again(monkeypatch):
    """The retained 80 % region of an overlapping block is accepted only if its own F and gradient pass the rule
    (m14_blocks.py, build_blocks): here its recomputed F is made to exceed f_max, so the block is dropped and logged
    as overlap_retry_rejected; the block map is not tried again."""
    import dataclasses

    from meshom.methods.m14 import blockmap, fitness as fit_mod
    from meshom.methods.m14.blockmap import BlockGrid

    fit = _flat_fitness()
    fit = dataclasses.replace(fit, area=np.full(fit.bg.n_nodes, 4 * np.pi / fit.bg.n_nodes))
    monkeypatch.setattr(fit_mod, "fitness", lambda dom, h, o, land=None: fit)
    placed = []
    orig_place, orig_rl = place_mod.place, place_mod.region_lambda

    def place(f, o):
        regs, log = orig_place(f, o)
        placed.append(True)
        return regs, log

    def region_lambda(f, tris):                                        # after placement: F above f_max
        nodes, lam, F, grad = orig_rl(f, tris)
        return nodes, lam, (2e3 if placed else F), grad

    def grid(P, T, lam, h, o, in_r=None, u_ref=None):                 # every map overlaps
        e = np.zeros(0, np.int64)
        calls.append(len(T))
        return BlockGrid(P[:0], e.reshape(0, 4), e.reshape(0, 2), e, {"overlap": True, "n_cells": 0,
                                                                      "inradius_cells": 0.0})

    calls: list = []
    monkeypatch.setattr(place_mod, "place", place)
    monkeypatch.setattr(place_mod, "region_lambda", region_lambda)
    monkeypatch.setattr(blockmap, "block_grid", grid)
    o = dict(DEFAULTS, grow="metric", a_max=400.0, max_blocks=1, bisect=2, f_max=1e3, grad_max=1e3)
    _, regions, grids, kept, plog, _ = build_blocks(closed_sphere(), uniform(4e5), o)
    assert len(regions) == 1 and plog["overlap_retries"] == 1 and plog["overlap_retry_rejected"] == 1
    assert kept == [] and grids == [None] and len(calls) == 1 and regions[0].F == 2e3


def test_cut_keeps_basins_above_the_area():
    """Two water basins joined by nothing: FESOM2's rule keeps the largest only; keep_basins_km2 keeps both when
    both are large enough and drops the small one when it is not."""
    import numpy as np
    from meshom.geo import R_EARTH_M
    from meshom.mesh import Mesh
    from meshom.methods.m14.cut import clean, element_areas, half_edges

    def patch(lon0, n):                               # n x n quads of 1 degree starting at lon0, on the equator
        lon = np.deg2rad(lon0 + np.arange(n + 1)); lat = np.deg2rad(np.arange(n + 1))
        L, B = np.meshgrid(lon, lat, indexing="ij")
        pts = np.stack([np.cos(B) * np.cos(L), np.cos(B) * np.sin(L), np.sin(B)], axis=-1).reshape(-1, 3) * R_EARTH_M
        idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
        q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)
        return pts, q
    p1, q1 = patch(0.0, 6); p2, q2 = patch(20.0, 2)
    pts = np.vstack([p1, p2]); quads = np.vstack([q1, q2 + len(p1)])
    m = Mesh(pts, np.zeros((0, 3), dtype=np.int64), quads, np.zeros(len(pts), dtype=np.int8))
    he = half_edges(m); keep = np.ones(m.n_elems, dtype=bool); areas = element_areas(m)
    k_fesom, _ = clean(he, keep, m.n_nodes)
    assert k_fesom.sum() == 36                                              # the largest basin only
    small = areas[36:].sum() / 1e6                                          # the 2 x 2 patch, km^2
    k_both, _ = clean(he, keep, m.n_nodes, areas, 0.5 * small * 1e6)
    assert k_both.sum() == 40
    k_big, _ = clean(he, keep, m.n_nodes, areas, 2.0 * small * 1e6)
    assert k_big.sum() == 36


def _coarse_and_fine_basins():
    """Two disjoint water patches: 3 x 3 quads of 100 km (9 elements, 90 000 km^2) and 6 x 6 quads of 10 km (36
    elements, 3600 km^2), and a small land loop away from both (nothing is land)."""
    from meshom.domain import Domain
    from tests.helpers import local_polygon, structured_patch

    a = structured_patch(3, 3, 100e3)
    b = structured_patch(6, 6, 10e3, lon0=10.0)
    m = Mesh(np.vstack([a.points, b.points]), np.zeros((0, 3), np.int64), np.vstack([a.quads, b.quads + a.n_nodes]),
             np.zeros(a.n_nodes + b.n_nodes, np.int8))
    land = Domain([local_polygon(0.0, 30.0, [[-5e3, -5e3], [-5e3, 5e3], [5e3, 5e3], [5e3, -5e3]])])
    return m, land


def test_cut_keeps_the_largest_basin_by_area():
    """C48: the basin kept by default is the largest by area (the ocean), not the one with the most elements."""
    from meshom.run import cut_land

    m, land = _coarse_and_fine_basins()
    out, info = cut_land(m, land)
    assert out.n_elems == 9 and out.n_quads == 9


def test_keep_basins_km2_always_keeps_the_largest_basin():
    """`keep_basins_km2` keeps basins of at least that area besides the largest one (C0, C42): the largest stays
    even when it is below the threshold."""
    from meshom.run import cut_land

    m, land = _coarse_and_fine_basins()
    assert cut_land(m, land, keep_basins_km2=1e6)[0].n_elems == 9          # both below: the largest only
    assert cut_land(m, land, keep_basins_km2=1e4)[0].n_elems == 9          # the coarse one above, the fine below
    assert cut_land(m, land, keep_basins_km2=1e3)[0].n_elems == 45         # both above


def test_in_water_with_disjoint_basins():
    """A continent (hole), a closed sea inside it (basin) with an island (hole nested in the basin): the ocean, the
    sea and the land each test right; a basin-only domain has no main region."""
    import numpy as np
    from meshom.domain import Domain, in_water
    from meshom.geo import lonlat_to_xyz

    def ring(lon0, lat0, r_deg, n=64, ccw=True):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        if not ccw:
            t = t[::-1]
        return lonlat_to_xyz(lon0 + r_deg * np.cos(t) / np.cos(np.radians(lat0)), lat0 + r_deg * np.sin(t))
    continent = ring(20.0, 30.0, 25.0, ccw=False)          # water outside (clockwise): a hole
    sea = ring(20.0, 30.0, 10.0, ccw=True)                 # water inside: a basin (a closed sea on the continent)
    islet = ring(20.0, 30.0, 2.0, ccw=False)               # an island in the sea: a hole nested in the basin
    dom = Domain([continent, sea, islet], "t")
    pts = lonlat_to_xyz(np.array([-100.0, 20.0, 20.0, 20.0]), np.array([0.0, 30.0, 45.0, 20.0]))
    #            ocean      islet centre  continent land  inside the sea (lat 20 is 10 deg from the centre? use 35)
    pts = lonlat_to_xyz(np.array([-100.0, 20.0, 20.0, 20.0]), np.array([0.0, 30.0, 50.0, 36.0]))
    assert in_water(dom, pts).tolist() == [True, False, False, True]
    lake_only = Domain([sea, islet], "lake")
    assert in_water(lake_only, pts).tolist() == [False, False, False, True]
    assert in_water(Domain([continent], "c"), pts).tolist() == [True, False, False, False]


def test_igm_regions_keep_water_numeric_under_a_land_cut():
    """B70 / Region: place_igm's regions carry their cone count in `info` and `water` stays None or a number, so the
    report of a land-cut run (fit.wet set) is written (before: the info dict went positionally into
    `water` and block_records crashed on round(dict))."""
    from meshom.domain import Domain
    from meshom.geo import lonlat_to_xyz
    from meshom.methods.m14.fitness import fitness
    from meshom.methods.m14.igm import place_igm
    from meshom.methods.m14.report import block_records

    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)[::-1]
    coast = Domain([lonlat_to_xyz(20 + 10 * np.cos(t), 30 + 10 * np.sin(t))], "island")
    fit = fitness(closed_sphere(), uniform(4e5), dict(DEFAULTS, igm=True), land=coast)
    assert fit.wet is not None
    regs, _ = place_igm(fit, dict(DEFAULTS, igm=True))
    assert regs and all(r.water is None or isinstance(r.water, float) for r in regs)
    assert all(isinstance(r.info.get("n_cones"), int) for r in regs)
    recs = block_records(regs, [None] * len(regs), fit)
    assert all("n_cones" in rec for rec in recs)
