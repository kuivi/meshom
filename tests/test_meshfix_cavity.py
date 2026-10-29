"""Core operations of the cavity remeshing method d_cavity (meshom/meshfix/cavity; decisions B134-B137)."""
import numpy as np

from meshom.geo import lonlat_to_xyz
from meshom.mesh import Mesh, build_topology, signed_area
from meshom.meshfix.cavity import cavity as cv
from meshom.meshfix.cavity import patterns as pt
from meshom.meshfix.cavity.work import Work


def grid(nx: int, ny: int, d: float = 0.1) -> Mesh:
    lon, lat = np.meshgrid(np.arange(nx + 1) * d, np.arange(ny + 1) * d)
    P = lonlat_to_xyz(lon.ravel(), lat.ravel())
    idx = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
    q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    return Mesh(P, np.zeros((0, 3), np.int64), q, np.zeros(len(P), np.int8))


def _work(m: Mesh) -> Work:
    W = Work(m)
    W.art = np.zeros(len(W.P), bool)
    return W


def _loop_patch(nx, ny, x0, y0, a, b):
    """A grid and the elements of an a x b block of it (rows y0.., cols x0..)."""
    m = grid(nx, ny)
    S = {(y0 + j) * nx + x0 + i for j in range(b) for i in range(a)}
    return m, S


def _check_fill(W, S, lp, X, cells):
    """The fill closes the loop: every new interior edge in two cells, the loop edges once, CCW cells."""
    r = W.replace(sorted(S), X, cells)
    assert r is not None
    _, ids, new = r[:3]
    he = {}
    for c in new:
        for i in range(len(c)):
            k = (c[i], c[(i + 1) % len(c)])
            assert k not in he
            he[k] = 1
    loop_he = {(lp[i], lp[(i + 1) % len(lp)]) for i in range(len(lp))}
    bnd = {k for k in he if (k[1], k[0]) not in he}
    assert bnd == loop_he
    m, _ = W.to_mesh()
    assert (signed_area(m) > 0).all()
    return r, m


def test_grid_cavity_is_its_own_rectangle():
    m, S = _loop_patch(12, 10, 3, 3, 4, 3)
    W = _work(m)
    C = cv.convexify(W, S, 100)
    assert C == S
    lp = cv.loop_of(W, C)
    cr = cv.corners(W, C, lp)
    assert len(cr) == 4
    e = sorted(len(s) - 1 for s in pt.sides(lp, cr))
    assert e == [3, 3, 4, 4]


def test_convexify_fills_concave_block():
    m, S = _loop_patch(12, 10, 3, 3, 4, 3)
    S.discard(3 * 12 + 3)                          # an L-shape: the missing corner quad is concave
    W = _work(m)
    C = cv.convexify(W, S, 100)
    assert C == S | {3 * 12 + 3}


def test_regular_fill_reproduces_grid_counts():
    m, S = _loop_patch(12, 10, 3, 3, 5, 4)
    W = _work(m)
    lp = cv.loop_of(W, S)
    cr = cv.corners(W, S, lp)
    cands = pt.fills(W.P, lp, cr)
    reg = [c for c in cands if c[0].startswith("regular")]
    assert len(reg) == 1
    _, X, cells, _ = reg[0]
    assert len(cells) == 20 and all(len(c) == 4 for c in cells) and len(X) == 4 * 3
    _check_fill(W, S, lp, X, cells)


def test_ladder_triangles_and_valences():
    """A cavity whose top has 2 edges fewer than its bottom: 2 column-end triangles; every new node has 4 elements
    except the tips (4 quads + 1 triangle); parity t == boundary edges (mod 2)."""
    m, S = _loop_patch(14, 12, 3, 3, 7, 5)
    W = _work(m)
    lp = cv.loop_of(W, S)
    cr = cv.corners(W, S, lp)
    sd = pt.sides(lp, cr)                          # sd[2] is the top (loop order): drop 2 of its interior nodes
    top = sd[2]
    keep = [top[0]] + top[2:-2:1][:len(top) - 4] + [top[-1]]
    assert len(keep) == len(top) - 2
    # build the ladder directly from the side lists (the old top nodes simply stay outside the fill)
    bottom, right, left = sd[0], sd[1], sd[3][::-1]
    F = pt.Fill(W.P)
    plans = pt.patch_plans(F, bottom, right, keep[::-1], left)
    assert plans
    for name, nrow, bases in plans:
        F = pt.Fill(W.P)
        cells = pt.build_patch(F, bottom, right, keep[::-1], left, nrow, bases)
        t = sum(len(c) == 3 for c in cells)
        assert t == 2
        nb = (len(bottom) - 1) + (len(right) - 1) + (len(keep) - 1) + (len(left) - 1)
        assert t % 2 == nb % 2
        cnt_q, cnt_t = {}, {}
        for c in cells:
            for v in c:
                (cnt_q if len(c) == 4 else cnt_t)[v] = (cnt_q if len(c) == 4 else cnt_t).get(v, 0) + 1
        new = [v for v in set(cnt_q) | set(cnt_t) if v < 0]
        comp = sorted((cnt_q.get(v, 0), cnt_t.get(v, 0)) for v in new)
        assert all(x in ((4, 0), (3, 1), (4, 1)) for x in comp)
        assert sum(x == (4, 1) for x in comp) == 2          # the two tips


def test_split_fill_closes_turn_cavity():
    """A cavity with a != c and right != left (a chord turn) gets the split pattern: column ends + row ends."""
    m, S = _loop_patch(16, 14, 3, 3, 8, 6)
    W = _work(m)
    lp = cv.loop_of(W, S)
    cr = cv.corners(W, S, lp)
    sd = pt.sides(lp, cr)
    bottom, right, top, left = sd[0], sd[1], sd[2][::-1], sd[3][::-1]
    top2 = [top[0]] + top[1:-1][1:] + [top[-1]]               # top one edge shorter
    left2 = [left[0]] + left[1:-1][1:] + [left[-1]]           # left one edge shorter
    out = pt._split(pt.Fill(W.P), bottom, right, top2, left2, None, None, 0)
    assert out
    for name, F, cells in out:
        assert sum(len(c) == 3 for c in cells) == 2
        he = set()
        for c in cells:
            for i in range(len(c)):
                k = (c[i], c[(i + 1) % len(c)])
                assert k not in he
                he.add(k)


def test_replace_and_undo_restore_the_mesh():
    m, S = _loop_patch(12, 10, 3, 3, 5, 4)
    W = _work(m)
    lp = cv.loop_of(W, S)
    cr = cv.corners(W, S, lp)
    name, X, cells, cn = pt.fills(W.P, lp, cr)[0]
    m0, _ = W.to_mesh()
    r, _ = _check_fill(W, S, lp, X, cells)
    W.undo_replace(r)
    m1, _ = W.to_mesh()
    assert np.array_equal(np.sort(np.sort(m0.quads, 1), 0), np.sort(np.sort(m1.quads, 1), 0))
    assert np.allclose(m0.points, m1.points)


def test_coast_adjust_merges_and_splits_edges():
    m = grid(10, 4)
    W = _work(m)
    side = list(range(0, 8))                                   # the straight bottom row: collinear nodes
    h = lambda x: np.full(len(x), 11e3)
    F = pt.Fill(W.P)
    s2 = pt.adjust(F, side, 5, h)
    assert len(s2) == 6 and s2[0] == 0 and s2[-1] == 7 and not F.X
    F = pt.Fill(W.P)
    s3 = pt.adjust(F, side, 9, h)
    assert len(s3) == 10 and len(F.coast) == 2


def test_coast_run_is_one_side():
    m = grid(12, 8)
    W = _work(m)
    W.art = np.zeros(len(W.P), bool)
    S = {j * 12 + i for j in range(0, 3) for i in range(3, 8)}     # a block on the bottom (coast) row
    C = cv.convexify(W, S, 100, coast=True)
    lp = cv.loop_of(W, C)
    cr = cv.corners(W, C, lp, coast=True)
    assert cr is not None and len(cr) == 4
    assert cv.convexify(W, S, 100, coast=False) is None
    topo = build_topology(m)
    assert topo.boundary_node[list(lp)].any()
