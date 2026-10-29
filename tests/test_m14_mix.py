"""m14_blocks_mix: the partition covers the fill domain exactly once; the ladder strip between two straight block
sides is all quads with every interior node regular, and a row-count mismatch of 2 gives exactly 2 triangles; the
mix mesh on uniform scale 4 is valid (manifold, CCW, Euler)."""
import numpy as np

from meshom.cases import CASES
from meshom.compete import irregular_counts
from meshom.domain import Domain, resample_loops
from meshom.geo import R_EARTH_M, stereographic_inverse
from meshom.mesh import Mesh, build_topology, signed_area
from meshom.methods import METHODS
from meshom.methods import m14_blocks as M
from meshom.methods.m14 import stitch_mix as S
from meshom.methods.m14.blockmap import BlockGrid, outline_loop
from meshom.quality.metrics import compute_quality
from meshom.size import scaled, uniform


def _sphere_area(P, T):
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    return 2.0 * np.arctan2(det, 1.0 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c)
                            + np.einsum("ij,ij->i", c, a))


def test_partition_covers_fill_once_and_mix_is_valid():
    dom, h0 = CASES["uniform"]
    h = scaled(h0, 4.0)
    o = dict(M.DEFAULTS, fill_method="mix")
    _, _, grids, kept, _, _ = M.build_blocks(dom, h, o)
    grids = [grids[k] for k in kept]
    part = S.partition(resample_loops(dom, h), [g.points[g.outline] for g in grids], h, o)
    reg = part.reg
    assert len(reg) == len(part.bg.tris) and reg.min() >= 0 and reg.max() == len(part.regions) - 1
    # every background triangle in one region; their areas sum to the sphere minus the block outlines' interiors
    area = _sphere_area(part.bg.points, part.bg.tris)
    per = np.bincount(reg, area, minlength=len(part.regions))
    blocks = sum(_sphere_area(g.points, np.r_[g.quads[:, [0, 1, 2]], g.quads[:, [0, 2, 3]]]).sum() for g in grids)
    assert np.isclose(per.sum(), area.sum()) and abs(per.sum() + blocks - 4 * np.pi) < 1e-6 * 4 * np.pi
    # the mix mesh: valid, every fill element in exactly one region, block quads in none
    mesh, info = M.run(dom, h, {"fill_method": "mix", "optimize": False})    # the records index the stitch's mesh
    assert (signed_area(mesh) > 0).all() and (build_topology(mesh).elems_per_node > 0).all()
    ic = irregular_counts(mesh)
    assert ic["euler_ok"] and ic["euler_sum"] == 8
    s = compute_quality(mesh).summary
    assert s["n_pinch_nodes"] == 0 and s["quad_max_angle_deg"]["max"] <= 150.0
    assert info["irregular_in_blocks"] == 0 and len(info["regions"]) == info["stitch"]["n_parts"]
    n_elems = sum(r["elems"] for r in info["regions"])
    assert n_elems == mesh.n_elems - info["stitch"]["n_block_quads"]


def _corridor(nA: int, nB: int, gap: int = 6):
    """Two straight block sides gap cells apart (h = 100 km at the equator), A running east with the corridor on its
    left, B running west; the caps are straight."""
    hm = 100e3
    xa = stereographic_inverse(np.linspace(0, nA, nA + 1) * hm, np.zeros(nA + 1), 0.0, 0.0)
    xb = stereographic_inverse(np.linspace(nA, 0, nB + 1) * hm, np.full(nB + 1, gap * hm), 0.0, 0.0)
    return xa, xb, uniform(hm)


def _strip(nA, nB):
    xa, xb, h = _corridor(nA, nB)
    ca, cb = S._edge_classes(xa), S._edge_classes(xb)
    plan = S._strip_plan(ca, cb[::-1], xa, xb[::-1], h)
    assert not isinstance(plan, str), plan
    m0, m1 = plan["m0"], plan["m1"]
    c0 = S._unit(xa[0] + np.linspace(0, 1, m0 + 1)[1:-1, None] * (xb[-1] - xa[0]))
    c1 = S._unit(xa[-1] + np.linspace(0, 1, m1 + 1)[1:-1, None] * (xb[0] - xa[-1]))
    xyz = np.vstack([xa, xb, c0, c1])
    A = np.arange(nA + 1)
    Bs = nA + 1 + np.arange(nB + 1)
    C0 = np.r_[A[0], 2 * nA + 2 + np.arange(m0 - 1) + (nB - nA), Bs[-1]]
    C1 = np.r_[A[-1], 2 * nA + 2 + (nB - nA) + m0 - 1 + np.arange(m1 - 1), Bs[0]]
    new, q, t = S.strip_patch(xyz, A, Bs[::-1], C0, C1, plan, len(xyz))
    pts = np.vstack([xyz, new])
    mesh = Mesh(pts, t, q, np.zeros(len(pts), dtype=np.int8))
    assert (signed_area(mesh) > 0).all()
    topo = build_topology(mesh)
    deg = np.bincount(topo.edges.ravel(), minlength=len(pts))
    interior = ~topo.boundary_node & (topo.elems_per_node > 0)
    return plan, mesh, deg[interior]


def test_strip_matching_counts_is_all_quads_and_regular():
    plan, mesh, deg = _strip(20, 20)
    assert plan["m0"] == 6 and plan["row_terms"] == 0 and plan["col_terms"] == 0
    assert mesh.n_tris == 0 and mesh.n_quads == 20 * 6 and (deg == 4).all()


def test_strip_mismatch_two_gives_two_triangles():
    plan, mesh, deg = _strip(20, 22)
    assert plan["col_terms"] == 2 and plan["row_terms"] == 0
    assert mesh.n_tris == 2 and (deg != 4).sum() == 2


def test_registered():
    assert "m14_blocks_mix" in METHODS


def test_coincident_chains_b66():
    """B66: a separator node within SEED_COINCIDE of another seed marks its chains (the hang layout: two lattice caps
    of one continuation corridor through the same lattice points); two coincident loop nodes alone mark none."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x[3] = x[2]                                         # two loop nodes (0..3) coincide: no separator involved
    chains = [{"ids": np.array([0, 4, 5, 1])}, {"ids": np.array([2, 6, 7, 3])}, {"ids": np.array([1, 8, 9, 0])}]
    assert S._coincident_chains(x, chains, 4) == []
    x[7] = x[5] + 0.3 * S.SEED_COINCIDE                 # separator node 7 on 5: chains 0 and 1
    assert S._coincident_chains(x, chains, 4) == [0, 1]
    x[7] = x[5] + 3.0 * S.SEED_COINCIDE
    assert S._coincident_chains(x, chains, 4) == []


def test_tri_regions_keys_do_not_alias_fill_nodes():
    """_tri_regions finds the known half-edges (seed ids < part.N) among the triangulation's half-edges, whose nodes
    run past part.N: keyed by a * part.N + b, (4, 1) and (0, 25) were one key at N = 6 and a separator vote went to
    the wrong component (a false B52 leak [[0, 1]]). Two regions split by the separator 1-4 at x = 0."""
    from types import SimpleNamespace

    from meshom.delaunay import sphere_delaunay

    base = np.array([[-2, -2], [0, -2], [2, -2], [2, 2], [0, 2], [-2, 2]], float)
    le = np.column_stack([np.arange(6), np.roll(np.arange(6), -1)])
    cons = np.vstack([le, [1, 4]])
    he = np.vstack([le, [1, 4], [4, 1]])
    rng = np.random.default_rng(12)
    for _ in range(24):                                                 # the 24th draw has the aliasing edges
        inner = rng.uniform(-1.9, 1.9, (30, 2))
    xy = np.vstack([base, inner])
    P = stereographic_inverse(xy[:, 0] * 50e3, xy[:, 1] * 50e3, 0.0, 0.0)
    tris = sphere_delaunay(P, cons, le)
    N = len(base)
    a, b = tris.ravel(), np.roll(tris, -1, axis=1).ravel()
    assert np.isin(a * N + b, he[:, 0] * N + he[:, 1])[np.maximum(a, b) >= N].any()   # the input aliases
    part = SimpleNamespace(cons=cons, N=N, he_keys=he[:, 0] * N + he[:, 1], he_reg=np.array([0, 1, 1, 1, 0, 0, 0, 1]),
                           he_sep=np.r_[np.zeros(6, bool), True, True], regions=[{}, {}])
    reg, conflicts = S._tri_regions(P, tris, part)
    assert conflicts == []
    assert np.array_equal(reg, (xy[tris].mean(axis=1)[:, 0] > 0).astype(int))


def test_mix_stitch_without_blocks():
    """A water channel 500 x 100 km at h = 50 km takes no block: the mixed stitch fills it all (it crashed on the
    empty outline-edge and block-node lists)."""
    xy = np.array([[-250, -50], [250, -50], [250, 50], [-250, 50]]) * 1e3
    dom = Domain([stereographic_inverse(xy[:, 0], xy[:, 1], 0.0, 0.0)], "channel")
    h = uniform(50e3)
    o = dict(M.DEFAULTS, fill_method="mix", optimize=False)
    _, _, _, kept, _, _ = M.build_blocks(dom, h, o)
    assert len(kept) == 0
    mesh, st = S.stitch_mix(dom, h, [], o)
    assert mesh.n_elems > 0 and (signed_area(mesh) > 0).all() and (build_topology(mesh).elems_per_node > 0).all()
    assert len(st["block_nodes"]) == 0 and st["outline_edges"].shape == (0, 2)


def test_tri_fill_hook_on_merged_regions():
    """tri_fill hook with merge_same_fill (B50): every region "tri", so all five merge into fill region 0; the hook
    fills that region once (absorbed regions have no triangles of their own: _region_loops crashed on them), the
    mesh keeps the coast and the block nodes, and every fill element is measured in one of the five regions."""
    hm = 50e3

    def block(i0):
        ii, jj = np.meshgrid(np.arange(i0, i0 + 11), np.arange(11), indexing="ij")
        ij = np.column_stack([ii.ravel(), jj.ravel()])
        P = stereographic_inverse(ij[:, 0] * hm, ij[:, 1] * hm, 0.0, 0.0)
        nid = np.arange(len(P)).reshape(11, 11)
        q = np.column_stack([nid[:-1, :-1].ravel(), nid[1:, :-1].ravel(), nid[1:, 1:].ravel(), nid[:-1, 1:].ravel()])
        return BlockGrid(P, q, ij, outline_loop(np.ones((10, 10), bool), nid)[0], {"n_cells": 100})

    xy = np.array([[-6, -6], [32, -6], [32, 16], [-6, 16]], float) * hm
    dom = Domain([stereographic_inverse(xy[:, 0], xy[:, 1], 0.0, 0.0)], "two-block-channel")
    gs = [block(0), block(16)]
    o = dict(M.DEFAULTS, fill_method="mix", size_tolerance=None, optimize=False, band_size_iters=0,
             band_angle_iters=0, fill={"size_iters": 0, "angle_iters": 0}, break_clusters=False, force_fill="tri",
             coast_fill="tri", open_fill="tri", junction_fill="tri",
             tri_fill="meshom.methods.m14.stitch_mix:fill_tri_local")
    mesh, st = S.stitch_mix(dom, uniform(hm), gs, o)
    assert st["info"]["n_tri_hooked"] == 1 and len(np.unique(st["partition"].group)) == 1
    assert (signed_area(mesh) > 0).all() and (build_topology(mesh).elems_per_node > 0).all()
    nc = len(resample_loops(dom, uniform(hm))[0])
    assert np.array_equal(mesh.points[nc:nc + 242], np.vstack([g.points for g in gs]))
    er = st["elem_region"]
    assert (er[~st["is_block"]] >= 0).all() and len(np.unique(er[~st["is_block"]])) == len(st["region_records"])
