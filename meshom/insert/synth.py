"""Synthetic hosts for the insert's tests: a strait between two banks cut out of a uniform quad lattice (a "cut" run's
staircase coast), in a stereographic chart at (lon0, lat0).

- banana: a curved channel (centre line y = bend cos^2(pi x / L)), narrowing to the neck in the middle, banks
  meeting the ends at right angles;
- butterfly: a short neck between two wide wings (the banks open at `open_deg` on both sides, then run parallel).
"""
import numpy as np

from meshom.insert.geom import Chart
from meshom.insert.host import Window
from meshom.mesh import Mesh, build_topology, orient_ccw
from meshom.geo import lonlat_to_xyz


def lattice_host(water, box, H, rot_deg=0.0, lon0=0.0, lat0=0.0):
    """Quads of a lattice of spacing H (rotated by rot_deg) covering box (x0, x1, y0, y1) m, kept where water(xy) holds
    at the centroid; the largest water piece. Returns (Window, chart): kind 2 on the lattice's outer edge."""
    ch = Chart(lonlat_to_xyz(np.array(lon0), np.array(lat0)))
    x0, x1, y0, y1 = box
    R = max(abs(x0), abs(x1), abs(y0), abs(y1)) * 1.5
    n = int(np.ceil(2 * R / H))
    u = np.arange(n + 1) * H - R
    U, V = np.meshgrid(u, u, indexing="ij")
    a = np.radians(rot_deg)
    X, Y = np.cos(a) * U - np.sin(a) * V, np.sin(a) * U + np.cos(a) * V
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    q = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    P = np.c_[X.ravel(), Y.ravel()]
    cen = P[q].mean(1)
    inbox = (cen[:, 0] > x0) & (cen[:, 0] < x1) & (cen[:, 1] > y0) & (cen[:, 1] < y1)
    keep = inbox & water(cen)
    q = q[keep]
    used = np.unique(q)
    loc = np.full(len(P), -1)
    loc[used] = np.arange(len(used))
    m = Mesh(ch.inv(P[used]), np.zeros((0, 3), np.int64), loc[q], np.zeros(len(used), np.int8))
    m = orient_ccw(m)
    # largest edge-connected piece, no pinch nodes
    from meshom.meshfix.cases import components
    for _ in range(20):
        lab = components(m)
        big = lab == np.bincount(lab).argmax()
        topo = build_topology(m)
        be = topo.edges[topo.edge_elems[:, 1] < 0]
        pinch = np.bincount(be.ravel(), minlength=m.n_nodes) > 2
        bad = pinch[m.quads].any(1) | ~big
        if not bad.any():
            break
        qq = m.quads[~bad]
        used = np.unique(qq)
        loc = np.full(m.n_nodes, -1)
        loc[used] = np.arange(len(used))
        m = Mesh(m.points[used], m.tris, loc[qq], np.zeros(len(used), np.int8))
    topo = build_topology(m)
    be = topo.edges[topo.edge_elems[:, 1] < 0]
    xy = ch.fwd(m.points)
    c = 0.5 * (xy[be[:, 0]] + xy[be[:, 1]])
    edge = (c[:, 0] < x0 + 1.01 * H) | (c[:, 0] > x1 - 1.01 * H) | (c[:, 1] < y0 + 1.01 * H) | (c[:, 1] > y1 - 1.01 * H)
    kind = np.zeros(m.n_nodes, np.int8)
    kind[be[~edge].ravel()] = 1
    kind[be[edge].ravel()] = 2
    m.node_flag = topo.boundary_node.astype(np.int8)
    win = Window(m, np.arange(m.n_nodes), np.arange(m.n_elems), kind, be[~edge], be[edge])
    return win, ch


def banana(H=20e3, L=300e3, w_mouth=200e3, w_neck=20e3, bend=40e3, rot_deg=0.0):
    """(Window, chart, path xyz) of a curved strait between two basins (module docstring)."""
    def yc(x):
        return bend * np.cos(np.pi * np.clip(x / L, -0.5, 0.5)) ** 2

    def w(x):
        a = np.clip(np.abs(x) / (L / 2), 0, 1)
        return w_neck + (w_mouth - w_neck) * (3 * a ** 2 - 2 * a ** 3)

    def water(p):
        x, y = p[:, 0], p[:, 1]
        inch = (np.abs(x) <= L / 2) & (np.abs(y - yc(x)) <= w(x) / 2)
        basin = (np.abs(x) > L / 2) & (np.abs(y) <= w_mouth / 2 + 3 * H) & (np.abs(x) < L / 2 + 6 * H)
        return inch | basin
    box = (-L / 2 - 6 * H, L / 2 + 6 * H, -w_mouth / 2 - 4 * H, w_mouth / 2 + bend + 4 * H)
    win, ch = lattice_host(water, box, H, rot_deg)
    x = np.linspace(-0.45 * L, 0.45 * L, 41)
    return win, ch, ch.inv(np.c_[x, yc(x)])


def butterfly(H=10e3, L=160e3, w_neck=12e3, open_deg=60.0, w_wing=180e3, rot_deg=0.0):
    """A neck in the middle, banks opening at open_deg (full angle) on both sides up to w_wing, then wings."""
    t = np.tan(np.radians(open_deg / 2))

    def water(p):
        x, y = np.abs(p[:, 0]), np.abs(p[:, 1])
        return y <= np.minimum(w_neck / 2 + t * x, w_wing / 2)
    box = (-L, L, -w_wing / 2 - 3 * H, w_wing / 2 + 3 * H)
    win, ch = lattice_host(water, box, H, rot_deg)
    x = np.linspace(-0.4 * L, 0.4 * L, 33)
    return win, ch, ch.inv(np.c_[x, 0 * x])


def _land_loops(ch, xs, y_s, y_n, far):
    """Two land polygons (unit vectors, water on the left) below the bank y_s(x) and above y_n(x), x over xs."""
    s = np.c_[xs, y_s]
    south = np.vstack([s, [[xs[-1], -far], [xs[0], -far]]])                 # east along the bank, back below
    n = np.c_[xs[::-1], y_n[::-1]]
    north = np.vstack([n, [[xs[0], far], [xs[-1], far]]])                  # west along the bank, back above
    return [ch.inv(south), ch.inv(north)]


def strait_real(H=10e3, L=400e3, w_mouth=200e3, w_neck=20e3, bend=40e3, bay=None, rot_deg=0.0):
    """(Window, chart, path xyz, real coast loops) of the banana (bend > 0) or a straight strait (bend = 0) whose
    real coast is the analytic bank; the host is the lattice cut to it (a staircase). bay = (x0, x1, depth): a
    rectangular bay in the north bank between x0 and x1 (m), `depth` m deep."""
    def yc(x):
        return bend * np.cos(np.pi * np.clip(x / L, -0.5, 0.5)) ** 2

    def w(x):
        a = np.clip(np.abs(x) / (L / 2), 0, 1)
        return w_neck + (w_mouth - w_neck) * (3 * a ** 2 - 2 * a ** 3)

    def bay_top(x):
        if bay is None:
            return np.zeros_like(x)
        x0, x1, dep = bay
        return np.where((x >= x0) & (x <= x1), dep, 0.0)

    ext = L / 2 + 6 * H
    yb = w_mouth / 2 + 3 * H

    def ys(x):
        return np.where(np.abs(x) <= L / 2, yc(x) - w(x) / 2, -yb)

    def yn(x):
        return np.where(np.abs(x) <= L / 2, yc(x) + w(x) / 2 + bay_top(x), yb)

    def water(p):
        x, y = p[:, 0], p[:, 1]
        return (np.abs(x) < ext) & (y >= ys(x)) & (y <= yn(x))
    box = (-ext, ext, -yb - 4 * H, yb + bend + 4 * H + (bay[2] if bay else 0.0))
    win, ch = lattice_host(water, box, H, rot_deg)
    xs = np.linspace(-ext - 4 * H, ext + 4 * H, 4001)
    if bay is not None:                                   # the bay's walls as vertical steps
        xs = np.unique(np.r_[xs, bay[0] - 1e-3, bay[0], bay[1], bay[1] + 1e-3])
    loops = _land_loops(ch, xs, ys(xs), yn(xs) + 0.0 * (np.isclose(xs, 0)), 4 * yb + bend)
    x = np.linspace(-0.45 * L, 0.45 * L, 41)
    return win, ch, ch.inv(np.c_[x, yc(x)]), loops
