"""m06: Q-Morph (Owen, Staten, Canann & Saigal 1999): quad fronts advance over an isotropic triangle mesh and turn
its triangles into quads by edge swaps, splits and merging. Kernel: meshom/methods/m06/qmorph.py.

1. Triangle mesh: m02_linf_frontal_tri (default; hexagonal lattice along the asterisk field, ~98 % valence 6) or
   m01_triangles (DistMesh springs), both with the resampled coast nodes first.
2. Front: the coast edges (level 0). Closed sphere: a seed of 3 quads around one valence-6 node next to a +1/4 cone of
   the cross field (a cube-corner node); the 6 edges of its star are the first front.
3. Q-Morph advance (kernel docstring): side edges by the front-angle rule (< 3 pi / 4: reuse the neighbour front edge;
   else bisector, or the cross-field direction nearest to it in the _field variant), swap / split recovery, new top
   nodes moved to distance h along the side, top-edge recovery by swaps; the front gains an edge (quad + triangle)
   where the top edge would exceed expand h and loses one (a triangle) below contract h.
4. Closure: groups of leftover triangles with inner nodes and a 3-6 node boundary loop are re-split without them
   (close_pockets); the other leftover triangles stay, get Delaunay flips among themselves, and adjacent pairs are
   recombined (price 1/3); then meshom.smooth "size" sweeps and a short "angle" run, coast fixed, and one more
   close_pockets + short smoothing for the clusters smoothing exposed.
"""
import sys
import time

import numpy as np

from meshom.delaunay import adjacency
from meshom.domain import Domain
from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh
from meshom.methods import OVERRIDES, register
from meshom.size import SizeField

DEFAULTS = dict(
    tri="m02",              # input triangle mesh: "m02" (m02_linf_frontal_tri) | "m01" (m01_triangles)
    field=False,            # side directions from the cross field (nearest to the bisector) instead of the bisector
    field_method="cones",   # cross_field method
    field_tol=45.0,         # deg: the field is used at an edge only if within this of the bisector at both ends
    state1=135.0,           # deg: front angle below which the neighbour front edge is reused as side (Owen 3 pi / 4)
    side_eps=30.0,          # deg: an existing edge within this of the side direction is used as side (Owen pi / 6)
    split_long=1.6,         # side edges longer than this * h are split at h
    side_len=1.0,           # top nodes are moved to side_len * h from the base node
    expand=1.7,             # top edge > expand * h: quad + triangle (front gains an edge)
    contract=0.6,           # top edge < contract * h: triangle (front loses an edge)
    qmin=30.0, qmax=165.0,  # deg: quad corner bounds at formation
    qlong=2.0,              # quads with an edge longer than qlong * h are refused
    tmin=20.0, tmax=140.0,  # deg: triangle corner bounds at formation
    move=True,              # move new top nodes to their targets
    max_fail=3,             # refusals per front edge before it is dropped
    snap=1.4,               # another front closer than snap * h along the side direction: join it (seam row)
    gap2=2.5,               # another front closer than gap2 * h: the next row takes half of the gap
    price=1.0 / 3.0,        # recombination of leftover triangle pairs (None: off)
    size_iters=30,          # meshom.smooth "size" sweeps (coast fixed)
    angle_iters=3,          # then "angle" sweeps
    verbose=False,
)
TQ_TOL = 1e-3


def _opts(opts: dict | None) -> dict:
    o = dict(DEFAULTS)
    o.update(opts or {})
    return o


def _log(o: dict, msg: str) -> None:
    if o["verbose"]:
        print(f"[m06] {msg}", file=sys.stderr, flush=True)


def _triangles(domain: Domain, h: SizeField, o: dict) -> Mesh:
    if o["tri"] == "m01":
        from meshom.methods.m01_isotropic import triangle_mesh

        return triangle_mesh(domain, h, {})
    from meshom.methods.m02_linf_frontal import generate_triangles

    return generate_triangles(domain, h, {})


def _seed(mesh: Mesh, h: SizeField, cf) -> tuple[np.ndarray, np.ndarray]:
    """Closed sphere: 3 quads around a valence-6 node near a +1/4 cone; returns (quads (3,4), dead triangle ids)."""
    tris = mesh.tris
    n = mesh.n_nodes
    val = np.bincount(tris.ravel(), minlength=n)
    cand = np.flatnonzero(val == 6)
    target = mesh.points[0]
    if cf is not None:
        from meshom.field import singularities

        ids, ind = singularities(cf)
        pos = ids[ind > 0]
        if len(pos):
            c = cf.mesh.points[cf.mesh.tris[pos[0]]].sum(axis=0)
            target = c / np.linalg.norm(c)
    v = int(cand[np.argmax(mesh.points[cand] @ target)])
    star = np.flatnonzero((tris == v).any(axis=1))
    nbr = adjacency(tris)
    # CCW ring around v: start at one star triangle, cross the edge (c -> v)
    t = int(star[0])
    ring, tl = [], []
    for _ in range(6):
        i = int(np.flatnonzero(tris[t] == v)[0])
        ring.append(int(tris[t, (i + 1) % 3]))
        tl.append(t)
        t = int(nbr[t, (i + 2) % 3])
    quads = np.array([[v, ring[0], ring[1], ring[2]], [v, ring[2], ring[3], ring[4]],
                      [v, ring[4], ring[5], ring[0]]], dtype=np.int64)
    return quads, np.array(tl, dtype=np.int64)


def _corner_angles(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    from meshom.geo import tangent_coords
    from meshom.quality.kernel import _angles

    return np.degrees(_angles(tangent_coords(points[cells])))


def _splits(L: np.ndarray) -> list:
    """Ways to split the CCW loop L (3-6 nodes) into quads and triangles without inner nodes: (tris, quads)."""
    n = len(L)
    r6 = lambda *k: L[[i % n for i in k]]
    out = [([r6(r, r + k, r + k + 1) for k in range(1, n - 1)], []) for r in range(n)]   # triangle fans
    if n == 4:
        out.append(([], [L]))
    elif n == 5:
        out += [([r6(r + 3, r + 4, r)], [r6(r, r + 1, r + 2, r + 3)]) for r in range(5)]
    elif n == 6:
        out += [([], [r6(r, r + 1, r + 2, r + 3), r6(r + 3, r + 4, r + 5, r)]) for r in range(3)]
        out += [([r6(r + 3, r + 4, r + 5), r6(r + 5, r, r + 3)], [r6(r, r + 1, r + 2, r + 3)]) for r in range(6)]
        out += [([r6(r + 4, r + 5, r), r6(r, r + 1, r + 4)], [r6(r + 1, r + 2, r + 3, r + 4)]) for r in range(6)]
    return out


def close_pockets(points: np.ndarray, left: np.ndarray, qmax: float = 160.0, tmax: float = 130.0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Final closure of the regions the fronts left (Q-Morph's last loops): each connected group of leftover
    triangles with interior nodes whose boundary is one simple loop of 3-6 nodes is re-split without them: the valid split
    (corners <= qmax / tmax) with the fewest triangles, then the squarest. Returns (new triangles, new quads,
    leftover triangles kept)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if len(left) == 0:
        return np.zeros((0, 3), np.int64), np.zeros((0, 4), np.int64), left
    nbr = adjacency(left)
    t_of, k_of = np.nonzero(nbr >= 0)
    g = coo_matrix((np.ones(len(t_of)), (t_of, nbr[t_of, k_of])), shape=(len(left), len(left)))
    nc, lab = connected_components(g, directed=False)
    order = np.argsort(lab, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(lab, minlength=nc))]
    new_t, new_q, keep = [], [], np.ones(len(left), dtype=bool)
    for c in range(nc):
        ids = order[starts[c]:starts[c + 1]]
        tr = left[ids]
        bt, bk = np.nonzero(nbr[ids] < 0)
        a, b = tr[bt, bk], tr[bt, (bk + 1) % 3]
        if len(a) > 6 or len(np.unique(a)) != len(a):
            continue
        nxt = dict(zip(a.tolist(), b.tolist()))
        loop = [int(a[0])]
        while len(loop) <= len(a) and nxt[loop[-1]] != loop[0]:
            loop.append(nxt[loop[-1]])
        if len(loop) != len(a):
            continue
        n = len(loop)
        if len(ids) == n - 2:                       # no interior node: left to the recombination
            continue
        L = np.array(loop, dtype=np.int64)
        best, bw = None, (np.inf, np.inf)
        for ts, qs in _splits(L):
            w = 0.0
            for t in ts:
                ang = _corner_angles(points, t[None])[0]
                w = max(w, np.abs(ang - 60.0).max() / 60.0 if ang.max() <= tmax else np.inf)
            for q in qs:
                ang = _corner_angles(points, q[None])[0]
                w = max(w, np.abs(ang - 90.0).max() / 90.0 if ang.max() <= qmax else np.inf)
            if w < np.inf and (len(ts), w) < bw:
                best, bw = (ts, qs), (len(ts), w)
        if best is None:
            continue
        keep[ids] = False
        new_t += best[0]
        new_q += best[1]
    nt = np.array(new_t, dtype=np.int64).reshape(-1, 3)
    nq = np.array(new_q, dtype=np.int64).reshape(-1, 4)
    return nt, nq, left[keep]


def qmorph_mesh(domain: Domain, h: SizeField, opts: dict | None = None) -> tuple[Mesh, dict]:
    """Q-Morph over the input triangle mesh, before recombination of leftovers and smoothing."""
    from meshom.methods.m06 import qmorph as K

    o = _opts(opts)
    info: dict = {}
    t0 = time.perf_counter()
    tm = _triangles(domain, h, o)
    t1 = time.perf_counter()
    info.update(tri_nodes=tm.n_nodes, tri_s=t1 - t0)
    coast = np.flatnonzero(tm.node_flag == 1)
    cf = None
    if o["field"] or not domain.loops:
        from meshom.field import cross_field

        cf = cross_field(tm, h, n=4, boundary_nodes=coast if len(coast) else None, method=o["field_method"])
    t2 = time.perf_counter()
    info["field_s"] = t2 - t1

    n0, t_n = tm.n_nodes, tm.n_tris
    extra = n0 // 2 + 1000
    cap_n, cap_t = n0 + extra, t_n + 2 * extra
    P = np.zeros((cap_n, 3))
    P[:n0] = tm.points
    hv = np.zeros(cap_n)
    hv[:n0] = h(tm.points) / R_EARTH_M
    U = np.zeros((cap_n, 3))
    U[:, 0] = 1.0
    if cf is not None:
        U[:n0] = cf.u
    fixed = np.zeros(cap_n, dtype=np.bool_)
    fixed[coast] = True
    level = np.full(cap_n, K.BIG, dtype=np.int64)
    level[coast] = 0
    tris = np.zeros((cap_t, 3), dtype=np.int64)
    tris[:t_n] = tm.tris
    nbr = np.full((cap_t, 3), -1, dtype=np.int64)
    nbr[:t_n] = adjacency(tm.tris)
    talive = np.zeros(cap_t, dtype=np.bool_)
    talive[:t_n] = True
    quads = np.zeros((cap_t // 2 + 16, 4), dtype=np.int64)
    nq = 0
    if not domain.loops:
        sq, st = _seed(tm, h, cf)
        quads[:3] = sq
        nq = 3
        talive[st] = False
        nbr[np.isin(nbr, st)] = -1
        level[np.unique(sq)] = 0
    prm = np.zeros(K.N_PRM)
    prm[K.P_STATE1] = np.radians(o["state1"])
    prm[K.P_EPS] = np.radians(o["side_eps"])
    prm[K.P_SPLIT_LONG] = o["split_long"]
    prm[K.P_SIDE_LEN] = o["side_len"]
    prm[K.P_EXPAND] = o["expand"]
    prm[K.P_CONTRACT] = o["contract"]
    prm[K.P_QMIN], prm[K.P_QMAX] = np.radians(o["qmin"]), np.radians(o["qmax"])
    prm[K.P_TMIN], prm[K.P_TMAX] = np.radians(o["tmin"]), np.radians(o["tmax"])
    prm[K.P_FIELD_TOL] = np.radians(o["field_tol"])
    prm[K.P_MOVE] = 1.0 if o["move"] else 0.0
    prm[K.P_USE_FIELD] = 1.0 if o["field"] else 0.0
    prm[K.P_MAX_FAIL] = o["max_fail"]
    prm[K.P_TQ_TOL] = TQ_TOL
    prm[K.P_SNAP] = o["snap"]
    prm[K.P_GAP2] = o["gap2"]
    prm[K.P_QLONG] = o["qlong"]
    n_nodes, n_tris, nq, otris, okind, olev, qlev, dead, st = K.advance(P, hv, U, fixed, level, tris, nbr, talive, n0, t_n, quads, nq,
                                                     prm)
    t3 = time.perf_counter()
    names = ("flips", "splits", "quads", "tri_contract", "tri_expand", "tri_isolated", "refusals", "moves", "pops",
             "stale", "dropped", "quad_refused", "snaps", "r_side", "r_contract", "r_top", "r_expand", "r_qangle",
             "r_qregion", "r_isolated", "r_7")
    info.update({k: int(v) for k, v in zip(names, st)})
    left = tris[:n_tris][talive[:n_tris]]
    pt, pq, left = close_pockets(P, left)
    otris = np.vstack([otris.reshape(-1, 3), pt])
    okind = np.r_[okind, np.zeros(len(pt), dtype=np.int64)]
    quads[nq:nq + len(pq)] = pq
    qlev = np.r_[qlev, np.full(len(pq), -1)]
    nq += len(pq)
    info.update(pocket_tris=len(pt), pocket_quads=len(pq))
    info.update(advance_s=t3 - t2, n_leftover=len(left), n_out_tris=len(otris),
                kinds=np.bincount(okind, minlength=5)[1:].tolist(), max_level=int(qlev.max()) if nq else 0)
    info["_okind"] = np.r_[okind, np.zeros(len(left), dtype=np.int64)]
    info["_tlev"] = np.r_[olev, np.full(len(left), -1)]
    info["_qlev"] = np.r_[np.full(nq - len(qlev), -1), qlev] if len(qlev) < nq else qlev
    elems_t = np.vstack([otris.reshape(-1, 3), left]).astype(np.int64)
    elems_q = quads[:nq]
    used = np.zeros(n_nodes, dtype=bool)
    used[elems_t.ravel()] = True
    used[elems_q.ravel()] = True
    used[coast] = True
    new = np.cumsum(used) - 1
    flag = np.zeros(int(used.sum()), dtype=np.int8)
    flag[new[coast]] = 1
    mesh = Mesh(P[:n_nodes][used].copy(), new[elems_t], new[elems_q], flag)
    _log(o, " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in info.items()
                     if not k.startswith("_")))
    return mesh, info


def _compact(mesh: Mesh) -> Mesh:
    """Drop nodes in no element (coast nodes stay, first)."""
    used = np.zeros(mesh.n_nodes, dtype=bool)
    used[mesh.tris.ravel()] = True
    used[mesh.quads.ravel()] = True
    used |= mesh.node_flag == 1
    new = np.cumsum(used) - 1
    return Mesh(mesh.points[used], new[mesh.tris], new[mesh.quads], mesh.node_flag[used])


def _smooth(mesh: Mesh, h: SizeField, o: dict, scale: float = 1.0) -> Mesh:
    from meshom.smooth import smooth

    fixed = mesh.node_flag == 1
    if o["size_iters"]:
        mesh = smooth(mesh, h, fixed, iters=max(1, int(o["size_iters"] * scale)), kind="size")
    if o["angle_iters"]:
        mesh = smooth(mesh, h, fixed, iters=o["angle_iters"], kind="angle")
    return mesh


def _finish(mesh: Mesh, h: SizeField, o: dict) -> Mesh:
    """Delaunay flips among the triangles, recombination of leftover pairs, smoothing; then triangle clusters with
    inner nodes that smoothing left are closed (close_pockets) and smoothed once more."""
    if mesh.n_tris:
        from meshom.methods.m11.ops import flip_edges

        mesh, _ = flip_edges(mesh, kind="delaunay")
    if o["price"] is not None and mesh.n_tris:
        from meshom.recombine import recombine

        mesh = recombine(mesh, o["price"])
    mesh = _smooth(mesh, h, o)
    pt, pq, keep = close_pockets(mesh.points, mesh.tris)
    if len(pt) or len(pq):
        mesh = _compact(Mesh(mesh.points, np.vstack([keep, pt]), np.vstack([mesh.quads, pq]), mesh.node_flag))
        mesh = _smooth(mesh, h, o, 0.5)
    return mesh


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """Q-Morph with bisector side directions over the m02 triangle mesh."""
    o = _opts(opts)
    mesh, _ = qmorph_mesh(domain, h, o)
    return _finish(mesh, h, o)


@register("m06_qmorph_field", overrides={"field": True})
def generate_field(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """Q-Morph with side directions from the cross field (nearest to the bisector)."""
    o = _opts(dict(OVERRIDES["m06_qmorph_field"], **(opts or {})))
    mesh, _ = qmorph_mesh(domain, h, o)
    return _finish(mesh, h, o)
