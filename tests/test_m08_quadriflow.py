import numpy as np
import pytest

from meshom.cases import CASES
from meshom.compete import irregular_counts
from meshom.domain import resample_loops
from meshom.mesh import build_topology, signed_area
from meshom.methods import METHODS
from meshom.methods.m08_quadriflow import quadriflow
from meshom.size import scaled


@pytest.mark.parametrize("case", ["uniform", "continent"])
def test_valid_mesh(case):
    domain, h0 = CASES[case]
    h = scaled(h0, 4)
    m, st = quadriflow(domain, h, {})
    # the flow removes the position singularities (continent: 1.7 % of 383 on the Mac/arm64, 5.7 % on Levante/x86:
    # the Poisson points differ between platforms, so the threshold is a loose statistical bound)
    assert st["pos_sing_after"] <= 0.10 * st["pos_sing_before"]
    assert (signed_area(m) > 0).all()                                     # CCW, nothing inverted
    a = np.r_[m.tris.ravel(), m.quads.ravel()]
    b = np.r_[np.roll(m.tris, -1, axis=1).ravel(), np.roll(m.quads, -1, axis=1).ravel()]
    assert len(np.unique(a * m.n_nodes + b)) == len(a)                    # manifold: each directed edge once
    ic = irregular_counts(m)
    assert ic["euler_sum"] == 4 * (2 - len(domain.loops)) and ic["euler_ok"]
    assert m.n_tris < 0.05 * m.n_elems                                     # quad-dominant
    if domain.loops:                                          # the resampled coast nodes, all of them on the boundary
        coast = np.vstack(resample_loops(domain, h))
        topo = build_topology(m)
        assert np.array_equal(m.points[: len(coast)], coast)
        assert np.array_equal(topo.boundary_node, m.node_flag != 0) and (m.node_flag != 0).sum() == len(coast)


def test_registered():
    assert {"m08_quadriflow", "m08_quadriflow_im", "m08_quadriflow_keep"} <= set(METHODS)


def test_flow_is_min_cost():
    """The successive-shortest-path solver against networkx's network simplex on a small network (L1 cost, keep
    arcs to the ground node)."""
    nx = pytest.importorskip("networkx")
    import meshom.methods.m08.flow as F
    from meshom.delaunay import background_mesh
    from meshom.field import cross_field
    from meshom.geo import R_EARTH_M
    from meshom.methods.m07.fields import edge_frames, edge_jumps, position_field, restrict_constraints, \
        restrict_orientation
    from meshom.methods.m07.hierarchy import build_hierarchy

    domain, h0 = CASES["uniform"]
    h = scaled(h0, 12)
    bg = background_mesh(domain, h, 0.5)
    levels = build_hierarchy(bg.points, bg.tris)
    n = bg.n_nodes
    cons = restrict_constraints(levels, np.zeros((n, 3)), np.zeros(n), np.zeros((n, 3)), np.zeros(n))
    Q = restrict_orientation(levels, cross_field(bg, h).u)
    rho = [h(lev.V) / R_EARTH_M for lev in levels]
    O = position_field(levels, Q, rho, cons)
    diff, _ = edge_jumps(levels[0], Q[0], O, rho[0])
    D, r = F.own_frame(diff, edge_frames(levels[0], Q[0]))
    keep = 20.0
    net = {}
    orig = F._ssp

    def hook(G, ptr, adj, vm, vp, x, *rest):
        net.update(G=G, adj=adj.copy(), vm=vm.copy(), vp=vp.copy(), d0=rest[0].copy(), lb=rest[2].copy(),
                   ub=rest[3].copy(), ex=rest[6].copy())
        return orig(G, ptr, adj, vm, vp, x, *rest)

    F._ssp = hook
    try:
        Dn, st = F.remove_position_singularities(bg.tris, levels[0].edges, n, D, r, bg.points, Q[0], cost="l1",
                                                 keep=keep)
    finally:
        F._ssp = orig
    assert st["failed"] == 0 and st["pos_sing_before"] > 0
    G, ex = net["G"], net["ex"]
    g = nx.DiGraph()
    for v in range(G):
        g.add_node(v, demand=-int(ex[v]))
    g.add_node(G, demand=int(ex[:G].sum()))
    for k in np.unique(net["adj"]):
        a, b, d0 = int(net["vm"][k]), int(net["vp"][k]), int(net["d0"][k])
        up, dn = int(net["ub"][k]) - d0, d0 - int(net["lb"][k])
        if up > 0:
            g.add_edge(a, ("f", k), weight=1, capacity=up)
            g.add_edge(("f", k), b, weight=0, capacity=up)
        if dn > 0:
            g.add_edge(b, ("b", k), weight=1, capacity=dn)
            g.add_edge(("b", k), a, weight=0, capacity=dn)
    for v in range(G):
        g.add_edge(v, ("k+", v), weight=int(keep), capacity=10**6)
        g.add_edge(("k+", v), G, weight=0, capacity=10**6)
        g.add_edge(G, ("k-", v), weight=int(keep), capacity=10**6)
        g.add_edge(("k-", v), v, weight=0, capacity=10**6)
    assert st["cost"] == nx.min_cost_flow_cost(g)
