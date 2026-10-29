"""Stage t of the mesh fixing (meshom.meshfix.tri, decisions B154-B157): the flip undo is exact, the stage lowers the
irregular count of an amorphous disc without moving or reconnecting its boundary, the guard keeps the input when a
measure gets worse, the estimated h keeps the triangle count, the scorer's triangle block."""
import numpy as np

from meshom.delaunay import adjacency, sphere_delaunay
from meshom.geo import R_EARTH_M, stereographic_inverse
from meshom.mesh import Mesh, build_topology, orient_ccw, signed_area
from meshom.size import Uniform

L = 10_000.0      # m


def _disc(n_ring: int = 60, r_km: float = 120.0, seed: int = 0, jitter: float = 0.35) -> Mesh:
    """A disc near 60 N: a boundary ring of n_ring nodes and a jittered hexagonal lattice (spacing L) inside,
    Delaunay-triangulated: an amorphous mesh with many valence defects."""
    rng = np.random.default_rng(seed)
    r = r_km * 1e3
    th = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    bx, by = r * np.cos(th), r * np.sin(th)
    xs, ys = [], []
    for j in range(-20, 21):
        for i in range(-20, 21):
            x = (i + 0.5 * (j % 2)) * L
            y = j * L * np.sqrt(3) / 2
            if np.hypot(x, y) < r - 0.7 * L:
                xs.append(x + rng.uniform(-jitter, jitter) * L)
                ys.append(y + rng.uniform(-jitter, jitter) * L)
    x, y = np.r_[bx, xs], np.r_[by, ys]
    pts = stereographic_inverse(x, y, 0.0, 60.0)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    tris = sphere_delaunay(pts)
    m = orient_ccw(Mesh(pts, tris, np.zeros((0, 4), np.int64), np.zeros(len(pts), np.int8)))
    return m


def _kind(m: Mesh) -> np.ndarray:
    return build_topology(m).boundary_node.astype(np.int8)


def _irregular(m: Mesh) -> int:
    t = build_topology(m)
    deg = np.bincount(t.edges.ravel(), minlength=m.n_nodes)
    return int(((deg != 6) & ~t.boundary_node).sum())


def _bedges(m: Mesh) -> set:
    t = build_topology(m)
    return set(map(tuple, np.sort(t.edges[t.edge_elems[:, 1] < 0], axis=1).tolist()))


def test_flip_undo_restores_the_rows_exactly():
    """The undo of a rejected flip (kernels._try_flip: flip back, swap the rows, rotate them) gives back tris, nbr
    and a valid vt for every interior edge."""
    from meshom.delaunay import _flip
    from meshom.meshfix.tri import kernels as K

    m = _disc()
    tris0 = m.tris.copy()
    nbr0 = adjacency(tris0)
    n = m.n_nodes
    for t in range(0, len(tris0), 7):
        for k in range(3):
            u = nbr0[t, k]
            if u < 0:
                continue
            tris, nbr = tris0.copy(), nbr0.copy()
            vt = np.full(n, -1, np.int64)
            vt[tris.ravel()] = np.repeat(np.arange(len(tris)), 3)
            tf, uf = tris[t, 0], tris[u, 0]
            _flip(tris, nbr, vt, t, k)
            _flip(tris, nbr, vt, t, 2)
            K._swap(tris, nbr, vt, t, u)
            K._rotate_to(tris, nbr, t, tf)
            K._rotate_to(tris, nbr, u, uf)
            assert np.array_equal(tris, tris0) and np.array_equal(nbr, nbr0)
            assert all(v in tris[vt[v]] for v in tris[[t, u]].ravel())


def test_stage_t_lowers_irregular_nodes_and_keeps_the_boundary():
    from meshom.meshfix.tri import improve

    m = _disc()
    out, info = improve(m, Uniform(L), _kind(m), {"rounds": 6}, log=lambda *a: None)
    assert _irregular(out) < 0.8 * _irregular(m)
    assert (signed_area(out) > 0).all()
    b = build_topology(m).boundary_node
    assert np.array_equal(out.points[b], m.points[b])
    assert _bedges(out) == _bedges(m)
    assert out.n_tris == m.n_tris and out.n_nodes == m.n_nodes


def test_artificial_nodes_keep_their_incidence():
    from meshom.meshfix.tri import improve

    m = _disc()
    kind = _kind(m)
    kind[kind == 1] = 2                               # the whole boundary artificial
    inner = np.flatnonzero(kind == 0)
    kind[inner[::7]] = 2                              # and some interior nodes locked too
    before = np.bincount(m.tris.ravel(), minlength=m.n_nodes)
    out, _ = improve(m, Uniform(L), kind, {"rounds": 4}, log=lambda *a: None)
    after = np.bincount(out.tris.ravel(), minlength=m.n_nodes)
    lock = kind == 2
    assert np.array_equal(before[lock], after[lock])
    assert np.array_equal(out.points[lock], m.points[lock])


def test_guard_keeps_the_input_when_a_measure_gets_worse():
    from meshom.meshfix.tri import guard_summary, guard_worse

    m = _disc()
    b = guard_summary(m, Uniform(L))
    a = dict(b, obtuse=b["obtuse"] + 1)
    assert guard_worse(b, a) == ["obtuse"]
    assert guard_worse(b, b) == []
    assert guard_worse(b, dict(b, amin=b["amin"] - 1.0)) == ["amin"]


def test_estimated_h_keeps_the_triangle_count():
    from meshom.meshfix.tri.hest import EstimatedH

    m = _disc()
    h = EstimatedH(m)
    P, T = m.points, m.tris
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1) * R_EARTH_M ** 2
    cen = A + B + C
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    count = np.sum(area / (np.sqrt(3) / 4 * h(cen) ** 2))
    assert abs(count / len(T) - 1.0) < 1e-3        # weights follow the scaled h: not exact
    assert 0.7 * L < np.median(h(P)) < 1.3 * L
    import pickle

    h2 = pickle.loads(pickle.dumps(h))
    assert np.allclose(h2(P[:10]), h(P[:10]))


def test_scorer_triangle_block():
    from meshom.meshfix.cases import whole_case
    from meshom.meshfix.score import analyse, table

    m = _disc()
    case = whole_case("disc", m, Uniform(L), None, {}, {}, {})
    case.meta["input_area_km2"] = 1.0
    res, arr = analyse(case, m)
    t = res["tri"]
    assert t["irregular"] == _irregular(m)
    assert sum(t["valence"]["all"].values()) == t["interior"]
    assert t["coast_shift_h_max"] == 0.0
    assert "irregular (share)" in table([dict(res, method="input")])
