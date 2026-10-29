import numpy as np
import pytest

from meshom.cases import CASES
from meshom.compete import irregular_counts
from meshom.domain import resample_loops
from meshom.mesh import build_topology, signed_area
from meshom.methods import METHODS
from meshom.methods.m07.hierarchy import build_hierarchy
from meshom.methods.m07_instant_meshes import instant_mesh
from meshom.size import scaled


@pytest.mark.parametrize("case", ["uniform", "continent"])
def test_valid_mesh(case):
    domain, h0 = CASES[case]
    h = scaled(h0, 4)
    m, st = instant_mesh(domain, h, {})
    assert st["levels"] < 30                                              # the hierarchy halves per level
    assert (signed_area(m) > 0).all()                                     # CCW, nothing inverted
    a = np.r_[m.tris.ravel(), m.quads.ravel()]
    b = np.r_[np.roll(m.tris, -1, axis=1).ravel(), np.roll(m.quads, -1, axis=1).ravel()]
    assert len(np.unique(a * m.n_nodes + b)) == len(a)                    # manifold: each directed edge once
    ic = irregular_counts(m)
    assert ic["euler_sum"] == 4 * (2 - len(domain.loops)) and ic["euler_ok"]
    assert m.n_tris < 0.06 * m.n_elems                                     # quad-dominant
    assert ic["n3"] + ic["n5"] + ic["n_other"] < 0.06 * ic["n_interior"]   # few irregular nodes (m01: ~27 %)
    if domain.loops:                                          # the resampled coast nodes, all of them on the boundary
        coast = np.vstack(resample_loops(domain, h))
        topo = build_topology(m)
        assert np.array_equal(m.points[: len(coast)], coast)
        assert np.array_equal(topo.boundary_node, m.node_flag != 0) and (m.node_flag != 0).sum() == len(coast)


def test_registered():
    assert "m07_instant_meshes" in METHODS and "m07_instant_meshes_im" in METHODS


def test_hierarchy_levels():
    from meshom.delaunay import background_mesh

    domain, h0 = CASES["uniform"]
    bg = background_mesh(domain, scaled(h0, 4), 0.5)
    lev = build_hierarchy(bg.points, bg.tris)
    sizes = [x.n for x in lev]
    assert sizes[-1] == 1 and all(b <= 0.6 * a for a, b in zip(sizes[:-1], sizes[1:]) if a >= 10)
    for f, c in zip(lev[:-1], lev[1:]):                                   # parent / children are consistent
        assert (c.children[f.parent[np.arange(f.n)], 0] == np.arange(f.n)).sum() + \
            (c.children[f.parent[np.arange(f.n)], 1] == np.arange(f.n)).sum() == f.n


def test_edge_frames_match_jumps():
    """The frame indices m08 needs: the jump diff of an edge is expressed in q_i turned by ki quarter turns, and that
    frame matches q_j turned by kj (the turned vectors are within 45 deg)."""
    from meshom.delaunay import background_mesh
    from meshom.field import cross_field
    from meshom.methods.m07.fields import edge_frames

    domain, h0 = CASES["uniform"]
    h = scaled(h0, 8)
    bg = background_mesh(domain, h, 0.5)
    lev = build_hierarchy(bg.points, bg.tris)[0]
    q = cross_field(bg, h).u
    k = edge_frames(lev, q)
    i, j = lev.edges[:, 0], lev.edges[:, 1]

    def turn(v, n, m):
        out = v.copy()
        for _ in range(3):
            nxt = np.cross(n, out)
            out = np.where((m > 0)[:, None], nxt, out)
            m = m - 1
        return out

    a = turn(q[i], bg.points[i], k[:, 0].copy())
    b = turn(q[j], bg.points[j], k[:, 1].copy())
    assert (np.einsum("ij,ij->i", a, b) > np.cos(np.radians(45))).all()
