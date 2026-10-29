"""m14 direct join (`merge`, m14/sync.py merge; decisions B35, B61):
- the candidate rule: a corridor with b = 0 is tried whatever its drift, one with b != 0 only from merge_drift on;
- the union test on a toy pair of blocks (uniform sphere, scale 2, two seeds 67 deg apart on the equator): merged when
  the union with its corridor passes F <= 0.08 (blocks of B = 0.6), not merged when it does not (B = 1.0: F 0.12);
- end to end on that pair: one block, and 0 irregular nodes across the merged seam (the stitched seam of the same pair
  without the merge holds some)."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from meshom.cases import CASES
from meshom.geo import lonlat_to_xyz
from meshom.mesh import build_topology
from meshom.methods.m14 import sync as S
from meshom.methods.m14.fitness import dist_h, fitness
from meshom.methods.m14.place import boundary_nodes, place, union_region
from meshom.methods.m14_blocks import DEFAULTS, run
from meshom.size import scaled

SEEDS = [[0.0, 0.0], [67.0, 0.0]]


def test_candidate_rule():
    o = dict(DEFAULTS, merge_drift=1.0)
    flat = np.zeros(40)
    ramp = np.linspace(-20.0, 20.0, 40)                                # a drift of 40 deg along the corridor
    assert S.drift_per10(S.Corridor(0, 1, flat + 7.0, 40.0, 40, 40, bvec=(0, 0))) == pytest.approx(0.0, abs=1e-9)
    assert 1.5 < S.drift_per10(S.Corridor(0, 1, ramp, 40.0, 40, 40)) < 2.0
    assert S.is_candidate(S.Corridor(0, 1, flat, 40.0, 40, 40, bvec=(0, 0)), o)      # b = 0: tried
    assert not S.is_candidate(S.Corridor(0, 1, flat, 40.0, 42, 40, bvec=(2, 0)), o)  # cheap
    assert S.is_candidate(S.Corridor(0, 1, ramp, 40.0, 42, 40, bvec=(2, 5)), o)      # costly drift
    assert not S.is_candidate(S.Corridor(0, S.COAST, ramp, 40.0, 40, None), o)       # never with the coast
    assert S.is_candidate(S.Corridor(0, 1, flat, 40.0, 42, 40, bvec=(2, 0)), dict(o, merge_drift=0.0))


@pytest.fixture(scope="module")
def toy():
    dom, h0 = CASES["uniform"]
    h = scaled(h0, 2.0)
    o = dict(DEFAULTS, seeds=SEEDS, max_blocks=2, front=False)       # the pair grown one after the other (B67 off):
    return dom, h, o, fitness(dom, h, o)                             # the layout these union tests were set on


def test_union_test_on_a_toy_pair(toy):
    dom, h, o, fit = toy
    for b_cap, merged in ((0.6, True), (1.0, False)):
        oo = dict(o, b_cap=b_cap)
        ra, rb = place(fit, oo)[0]
        assert ra.F <= oo["f_max"] and rb.F <= oo["f_max"]
        u, info = union_region(fit, ra.tris, rb.tris, [], oo)
        assert info["corridor_tris"] > 0                               # the pair shares a corridor
        if merged:
            assert u is not None and info["F"] <= oo["f_max"] and info["grad"] <= oo["grad_max"]
            assert np.isin(ra.tris, u.tris).all() and np.isin(rb.tris, u.tris).all()
            assert u.cells > ra.cells + rb.cells and u.B > ra.B + rb.B
        else:
            assert u is None and info["why"] == "lambda" and info["F"] > oo["f_max"]


def _seam_nodes(fit, o, mesh, h):
    """Mesh nodes inside the toy pair's corridor (the union's triangles outside both regions) and >= 1 h from the
    union's outline; and the valence of every mesh node."""
    ra, rb = place(fit, o)[0]
    u, _ = union_region(fit, ra.tris, rb.tris, [], o)
    T, P = fit.bg.tris, fit.bg.points
    corr = np.setdiff1d(u.tris, np.r_[ra.tris, rb.tris])
    _, j = cKDTree(P[T].mean(axis=1)).query(mesh.points)              # the bg triangle holding each node (nearly)
    d = dist_h(mesh.points, P[boundary_nodes(T[u.tris], len(P))], h(mesh.points))
    deg = np.bincount(build_topology(mesh).edges.ravel(), minlength=mesh.n_nodes)
    return np.isin(j, corr) & (d >= 1.0), deg


def test_merged_seam_is_regular(toy):
    dom, h, o, fit = toy
    oo = dict(o, b_cap=0.6)
    m1, i1 = run(dom, h, dict(oo, merge=True))
    m0, i0 = run(dom, h, oo)
    assert i0["n_blocks"] == 2 and i1["n_blocks"] == 1
    assert i1["merge"]["n_merged"] == 1 and i1["merge"]["candidates"][0]["result"] == "merged"
    assert i1["irregular_in_blocks"] == 0
    deep1, deg1 = _seam_nodes(fit, oo, m1, h)
    deep0, deg0 = _seam_nodes(fit, oo, m0, h)
    assert deep1.sum() > 100 and (deg1[deep1] == 4).all()                # 0 irregular of 150 across the merged seam
    assert (deg0[deep0] != 4).sum() > 5                                 # the stitched seam: 14 of 169
    assert i1["coverage_area"] > i0["coverage_area"]


MID = [[31.7, -1.8], [35.3, -1.8], [35.3, 1.8], [31.7, 1.8]]       # a 400 km square in the pair's corridor


def test_union_keeps_the_fixed_blocks_gap(toy):
    """A fixed block's nodes (B59) in the toy pair's corridor: the regions placed around them keep block_gap, and so
    must the union (without them it came within 0.04 h, its grid 0.44 h)."""
    dom, h, o, fit = toy
    oo = dict(o, b_cap=0.6)
    lo, la = np.meshgrid([31.7, 33.5, 35.3], [-1.8, 0.0, 1.8])
    fixed = lonlat_to_xyz(lo.ravel(), la.ravel())
    ra, rb = place(fit, oo, fixed_pts=fixed)[0]
    P, hv = fit.bg.points, fit.hv
    assert min(dist_h(P[r.nodes], fixed, hv[r.nodes]).min() for r in (ra, rb)) > oo["block_gap"]
    u0, _ = union_region(fit, ra.tris, rb.tris, [], oo)                # blind to the fixed block: runs across it
    assert u0 is not None and dist_h(P[u0.nodes], fixed, hv[u0.nodes]).min() < 1.0
    u, info = union_region(fit, ra.tris, rb.tris, [], oo, fixed_pts=fixed)
    assert u is None or dist_h(P[u.nodes], fixed, hv[u.nodes]).min() > oo["block_gap"], info


def test_merge_keeps_a_hand_blocks_gap(toy):
    """End to end: a hand-placed block (B60, cut sides) in that corridor; merge is passed the fixed blocks, so the
    merged automatic block (if any) keeps block_gap from the hand block's nodes."""
    dom, h, o, fit = toy
    oo = dict(o, b_cap=0.6, merge=True, blocks=[{"name": "mid", "corners": MID, "sides": ["cut"] * 4}])
    m, info = run(dom, h, oo)
    assert info["fixed_blocks"][0]["status"] == "built" and info["n_blocks"] == 2
    res = [c["result"] for c in info["merge"]["candidates"]]
    assert len(res) > 0 and "merged" not in res, res                  # the only corridor runs across the hand block
