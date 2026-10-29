"""Narrow through-passages of a mesh's water and their widening (`[coast] passage_min_cells`, decisions C51).

A passage one cell wide is a dead end for the model: every node across it is on the coast, so nothing flows through.
`narrow_passages` finds, on a mesh as built (the final one: the land cut's staircase narrows passages after any
coast-based widening), every through-passage with fewer than K = `passage_min_cells` cells across its narrowest
section, listed strait or not; `passage_straits` turns them into `widen` corridors (meshom.straits, decisions C45:
W = max(0, (K 1.15 + 1/sqrt 3) h) on triangles, (K 1.15 + sqrt 2) e^t h on quads, h the background h at its largest
along the path) that meshom.run.cut_land forces wet, widening the passage into the land on both sides.

The measure is topological (cells, not metres):
- ring distance d(v): element rings from the boundary nodes (two nodes sharing an element are one ring apart);
- quads are split into two triangles along the diagonal with fewer core corners (below), so a quad that touches both
  coasts counts as one cell;
- K cells need r = K // 2 rings: the core nodes have d >= r; two triangles sharing an edge are core-linked when the
  edge has at least one core end (K even) or two (K odd). Across a channel of k cells (k + 1 node rows, ring distances
  0, 1, ..., 1, 0) the core links run along the channel iff k >= K: at k = K - 1 there is no core node (K even) or
  only one row of them, whose edges link the two sides pairwise but not along (K odd);
- pools: the components of the core links with at least two triangles (water at least K cells across); every other
  triangle is rim. A neck is a component of rim triangles joined by the remaining (chord) links, or a single chord
  link between two pool triangles; its ends are its chord links into pools;
- a neck is a through-passage when its ends reach two pools, or one pool from two ends or more with the coast along it
  on two boundary loops or more (water around an island: a section between two boundary loops never splits the
  water, and a neck whose two ends are joined through one pool always lies between two loops). A neck with one end
  (a one-cell fjord, a staircase corner) is a dead end. A neck that is the only link to a set of pools (removing it
  from the pool graph splits it) whose area is below BASIN_MIN_H2 h^2 (h: the neck's mean edge) is a dead-end inlet;
  the side holding the largest pool is never one. Dead ends are counted and left alone.
- cells across a through-passage: the largest k < K for which the place is no neck at the k-cell level (1 when
  K = 2).

The path through a passage (its corridor's centreline) runs from a triangle of one pool through the neck's rim
triangles (fewest triangles) to a triangle of the other (or, around an island, between the two ends farthest apart),
smoothed, and extended by EXT_H h into the pools at both ends.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from meshom.geo import R_EARTH_M, xyz_to_lonlat
from meshom.mesh import Mesh, build_topology

BASIN_MIN_H2 = 8.0       # a basin behind a single neck counts when its pools hold at least this many h^2 (else an inlet)
EXT_H = 1.0              # the corridor path is extended this many h into the pools at both ends
SMOOTH_PASSES = 2        # 3-point averages of the path's inner points (the rim centroids zigzag by about h/3)
MAX_ROUNDS = 3           # meshom.run.cut_land: detect -> widen -> cut again, at most this many rounds


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _arc_m(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2.0 * np.arcsin(np.clip(0.5 * np.linalg.norm(a - b, axis=-1), 0.0, 1.0)) * R_EARTH_M


def ring_distance(mesh: Mesh, boundary: np.ndarray, r_max: int) -> np.ndarray:
    """(N,) element rings from the boundary nodes, capped at r_max (nodes sharing an element are one ring apart)."""
    n = mesh.n_nodes
    d = np.where(boundary, 0, r_max).astype(np.int64)
    if r_max <= 1:
        return d
    p = np.vstack([mesh.tris[:, [i, j]] for i, j in ((0, 1), (1, 2), (0, 2))]
                  + [mesh.quads[:, [i, j]] for i in range(4) for j in range(i + 1, 4)])
    A = coo_matrix((np.ones(2 * len(p), dtype=np.int32), (np.r_[p[:, 0], p[:, 1]], np.r_[p[:, 1], p[:, 0]])),
                   shape=(n, n)).tocsr()
    reached, front = boundary.copy(), boundary.astype(np.int32)
    for k in range(1, r_max):
        nxt = (A @ front > 0) & ~reached
        d[nxt] = k
        reached |= nxt
        front = nxt.astype(np.int32)
    return d


def _split(mesh: Mesh, core: np.ndarray) -> np.ndarray:
    """(T + 2Q, 3) the triangles and each quad split along its diagonal with fewer core corners (ties: 0-2)."""
    q = mesh.quads
    c02 = core[q[:, 0]].astype(np.int64) + core[q[:, 2]]
    c13 = core[q[:, 1]].astype(np.int64) + core[q[:, 3]]
    u = (c13 < c02)[:, None]
    a = np.where(u, q[:, [0, 1, 3]], q[:, [0, 1, 2]])
    b = np.where(u, q[:, [1, 2, 3]], q[:, [0, 2, 3]])
    return np.vstack([mesh.tris, a, b]).astype(np.int64)


def _links(T: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interior edges of the triangles: (L,2) nodes, (L,) and (L,) the two triangles."""
    e = np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    t = np.tile(np.arange(len(T), dtype=np.int64), 3)
    key = e[:, 0] * n + e[:, 1]
    o = np.argsort(key, kind="stable")
    same = key[o][1:] == key[o][:-1]
    i, j = o[:-1][same], o[1:][same]
    return e[i], t[i], t[j]


def _labels(m: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(m, m))
    return connected_components(g, directed=False)[1]


def _tri_area(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1) * R_EARTH_M**2


def _loops(mesh: Mesh) -> np.ndarray:
    """(N,) boundary loop of every boundary node (components of the boundary edges), -1 elsewhere."""
    topo = build_topology(mesh)
    be = topo.edges[topo.edge_elems[:, 1] < 0]
    lab = _labels(mesh.n_nodes, be[:, 0], be[:, 1])
    out = np.full(mesh.n_nodes, -1, dtype=np.int64)
    out[topo.boundary_node] = lab[topo.boundary_node]
    return out


class _Level:
    """The pools and necks of one cells-across level K (module docstring)."""

    def __init__(self, mesh: Mesh, K: int, boundary: np.ndarray, d: np.ndarray):
        n = mesh.n_nodes
        core = d >= K // 2
        need = 1 if K % 2 == 0 else 2
        T = _split(mesh, core)
        e, ta, tb = _links(T, n)
        corel = core[e[:, 0]].astype(np.int64) + core[e[:, 1]] >= need
        reg = _labels(len(T), ta[corel], tb[corel])
        # a pool holds a triangle with a core node (K even) or with three (K odd: a strip two core rows wide)
        strong = core[T].sum(axis=1) >= (1 if K % 2 == 0 else 3)
        has = np.zeros(reg.max() + 1, dtype=bool)
        has[reg[strong]] = True
        pool_t = has[reg]
        self.T, self.reg, self.pool_t = T, reg, pool_t
        ch = ~(corel & pool_t[ta])                                       # every link that leaves or skips the pools
        self.ce, self.ca, self.cb = e[ch], ta[ch], tb[ch]                   # chord links (and links among rim)
        rim = np.flatnonzero(~pool_t)
        idx = np.full(len(T), -1, dtype=np.int64)
        idx[rim] = np.arange(len(rim))
        both = (~pool_t[self.ca]) & (~pool_t[self.cb])
        rc = _labels(len(rim), idx[self.ca[both]], idx[self.cb[both]]) if len(rim) else np.zeros(0, np.int64)
        self.rim, self.rim_idx, self.rim_comp = rim, idx, rc
        self.n_rim_comp = int(rc.max()) + 1 if len(rc) else 0
        self.both_rim = both

    def neck_of_tri(self) -> np.ndarray:
        """(T',) neck id of every rim triangle (rim component), -1 for pool triangles."""
        out = np.full(len(self.T), -1, dtype=np.int64)
        out[self.rim] = self.rim_comp
        return out


def _path_bfs(adj_a: np.ndarray, adj_b: np.ndarray, nodes: np.ndarray, src: list, dst: list) -> list:
    """Fewest-step path (as global ids of `nodes`) from any of `src` to any of `dst` in the graph on `nodes`."""
    loc = {int(v): i for i, v in enumerate(nodes)}
    m = len(nodes) + 1                                                  # + a super source
    a = [loc[int(x)] for x in adj_a]
    b = [loc[int(x)] for x in adj_b]
    s = [m - 1] * len(src)
    g = coo_matrix((np.ones(len(a) + len(s)), (np.r_[a, s], np.r_[b, [loc[int(x)] for x in src]])), shape=(m, m))
    dist, pred = shortest_path(g, directed=False, unweighted=True, indices=m - 1, return_predecessors=True)
    dl = [loc[int(x)] for x in dst]
    j = dl[int(np.argmin(dist[dl]))]
    if not np.isfinite(dist[j]):
        return []
    out = [j]
    while pred[out[-1]] != m - 1 and pred[out[-1]] >= 0:
        out.append(int(pred[out[-1]]))
    return [int(nodes[i]) for i in out[::-1]]


def _extend(p: np.ndarray, ext_m: float) -> np.ndarray:
    """The path with its ends extended by ext_m along the end directions (great circles); smoothed inner points."""
    p = _unit(p)
    if len(p) >= 3:
        for _ in range(SMOOTH_PASSES):
            p = np.r_[p[:1], _unit(p[:-2] + p[1:-1] + p[2:]), p[-1:]]
    if len(p) < 2 or ext_m <= 0:
        return p
    out = [p]
    for a, b, where in ((p[1], p[0], 0), (p[-2], p[-1], 1)):          # direction b - a, beyond b
        t = b - a - np.dot(b - a, b) * b
        if np.linalg.norm(t) < 1e-15:
            continue
        t = t / np.linalg.norm(t)
        ang = ext_m / R_EARTH_M
        q = np.cos(ang) * b + np.sin(ang) * t
        out = [q[None, :], out[0]] if where == 0 else [out[0], q[None, :]]
        out = [np.vstack(out)]
    return out[0]


def narrow_passages(mesh: Mesh, cells: int, basin_min_h2: float = BASIN_MIN_H2, straits: list | None = None,
                    ext_h: float = EXT_H, skip: np.ndarray | None = None) -> dict:
    """The through-passages of `mesh` with fewer than `cells` cells across (module docstring). Returns {"cells",
    "n_necks" (all rim components and pool-pool chords), "n_dead_end", "n_inlet" (dead-end basins below
    basin_min_h2 h^2), "n_pools", "n_skipped", "passages": [{"kind": "pools" | "island", "cells_across",
    "at": [lon, lat], "h_m", "pools", "basin_km2", "loops", "n_tris", "near" (a listed strait whose corridor holds
    `at`, else None), "path": (k, 3) unit vectors}]}. `skip` (n_elems,) bool: a passage whose neck lies only in these
    elements is left out (counted in n_skipped): m16_orca's ORCA cells keep NEMO's straits (B95)."""
    K = int(cells)
    rec = {"cells": K, "n_necks": 0, "n_dead_end": 0, "n_inlet": 0, "n_pools": 0, "n_skipped": 0, "passages": []}
    if K < 2 or mesh.n_elems == 0:
        return rec
    P = mesh.points
    boundary = build_topology(mesh).boundary_node
    loop = _loops(mesh)
    d = ring_distance(mesh, boundary, max(K // 2, 1))
    L = _Level(mesh, K, boundary, d)
    T, reg, pool_t = L.T, L.reg, L.pool_t
    area = _tri_area(P, T)
    preg = np.unique(reg[pool_t])
    pid = np.full(reg.max() + 1, -1, dtype=np.int64)
    pid[preg] = np.arange(len(preg))
    pool_of_t = np.where(pool_t, pid[reg], -1)
    parea = np.bincount(pool_of_t[pool_t], weights=area[pool_t], minlength=len(preg))
    main = int(np.argmax(parea)) if len(parea) else -1
    rec["n_pools"] = len(preg)
    cent = _unit(P[T].sum(axis=1))
    elen = np.mean([_arc_m(P[T[:, i]], P[T[:, (i + 1) % 3]]) for i in range(3)], axis=0)

    # necks: rim components (ids 0..R-1) and pool-pool chord links (ids R..)
    ca, cb, ce = L.ca, L.cb, L.ce
    R = L.n_rim_comp
    pp = np.flatnonzero(pool_t[ca] & pool_t[cb])
    rp = np.flatnonzero(pool_t[ca] ^ pool_t[cb])                       # rim <-> pool: the ends of rim necks
    r_tri = np.where(pool_t[ca[rp]], cb[rp], ca[rp])
    p_tri = np.where(pool_t[ca[rp]], ca[rp], cb[rp])
    r_neck = L.rim_comp[L.rim_idx[r_tri]]
    ends_neck = np.r_[r_neck, R + np.arange(len(pp)), R + np.arange(len(pp))]
    ends_pool = np.r_[pool_of_t[p_tri], pool_of_t[ca[pp]], pool_of_t[cb[pp]]]
    ends_rtri = np.r_[r_tri, cb[pp], ca[pp]]                           # the neck-side triangle of each end
    ends_ptri = np.r_[p_tri, ca[pp], cb[pp]]
    n_neck = R + len(pp)
    rec["n_necks"] = n_neck
    if n_neck == 0:
        return rec
    # distinct pools and ends per neck; boundary loops along each neck
    n_ends = np.bincount(ends_neck, minlength=n_neck)
    up = np.unique(ends_neck * (len(preg) + 1) + ends_pool)
    n_pools = np.bincount(up // (len(preg) + 1), minlength=n_neck)
    rim_nodes_neck = np.repeat(L.rim_comp, 3)
    rim_nodes = T[L.rim].ravel()
    nk = np.r_[rim_nodes_neck, R + np.arange(len(pp)), R + np.arange(len(pp))]
    nd = np.r_[rim_nodes, ce[pp, 0], ce[pp, 1]]
    nl = loop[nd]
    ok = nl >= 0
    ul = np.unique(nk[ok] * (loop.max() + 2) + nl[ok])
    n_loops = np.bincount(ul // (loop.max() + 2), minlength=n_neck)
    pools_neck = np.flatnonzero(n_pools >= 2)
    island_neck = np.flatnonzero((n_pools == 1) & (n_ends >= 2) & (n_loops >= 2))
    rec["n_dead_end"] = int(n_neck - len(pools_neck) - len(island_neck))

    # pool graph: every pools neck links its pools (a star through the neck's own vertex)
    order = np.argsort(ends_neck, kind="stable")
    starts = np.searchsorted(ends_neck[order], np.arange(n_neck + 1))

    def ends_of(k):
        return order[starts[k]:starts[k + 1]]

    npool = len(preg)
    ga, gb, gk = [], [], []
    for i, k in enumerate(pools_neck):
        for p in np.unique(ends_pool[ends_of(k)]):
            ga.append(npool + i)
            gb.append(int(p))
            gk.append(int(k))
    ga, gb, gk = np.asarray(ga, np.int64), np.asarray(gb, np.int64), np.asarray(gk, np.int64)
    m = npool + len(pools_neck)
    full_area = np.r_[parea, np.zeros(len(pools_neck))]

    tri_neck = L.neck_of_tri()
    out = []
    for kind, ks in (("pools", pools_neck), ("island", island_neck)):
        for k in ks:
            k = int(k)
            ee = ends_of(k)
            if k < R:
                tris = L.rim[L.rim_comp == k]
            else:
                j = pp[k - R]
                tris = np.array([ca[j], cb[j]])
            h = float(elen[tris].mean())
            groups = []
            if kind == "pools":
                keep = gk != k
                lab = _labels(m, ga[keep], gb[keep])
                ps = np.unique(ends_pool[ee])
                comps = {}
                for p in ps:
                    comps.setdefault(int(lab[p]), []).append(int(p))
                if len(comps) > 1:                                      # a bridge: each side must be a basin
                    carea = np.bincount(lab, weights=full_area)
                    mside = int(lab[main]) if int(lab[main]) in comps else max(comps, key=lambda c: carea[c])
                    groups = [comps[mside]] + [g for c, g in comps.items()
                                               if c != mside and carea[c] >= basin_min_h2 * h * h]
                    if len(groups) < 2:
                        rec["n_inlet"] += 1
                        continue
                    basin = float(min(carea[c] for c in comps if c != mside)) / 1e6
                else:
                    groups = [[int(p)] for p in ps]
                    basin = None
            else:
                basin = None
            # the path: from one end group through the neck's rim triangles to another
            paths = []
            if k < R:
                link = L.both_rim & (tri_neck[ca] == k)
                if kind == "pools":
                    g0 = groups[0]
                    src = [int(x) for x in ee if ends_pool[x] in g0]
                    for g in groups[1:]:
                        dst = [int(x) for x in ee if ends_pool[x] in g]
                        paths.append(_end_path(L, ca[link], cb[link], tris, ends_rtri, ends_ptri, src, dst))
                else:
                    paths.append(_far_path(L, ca[link], cb[link], tris, ends_rtri, ends_ptri, [int(x) for x in ee]))
            else:
                paths.append([int(ends_ptri[ee[0]]), int(ends_ptri[ee[1]])])
            if skip is not None and all(skip[x] for x in _owners(mesh, tris)):
                rec["n_skipped"] += 1
                continue
            for pt in paths:
                if len(pt) < 2:
                    continue
                path = _extend(cent[pt], ext_h * h)
                mid = cent[pt[len(pt) // 2]]
                lon, lat = xyz_to_lonlat(mid[None, :])
                near = None
                if straits:
                    dd = [(s.distance_m(mid[None, :], 0.5 * s.corridor_m + 2.0 * h)[0], s.name) for s in straits]
                    dd = [x for x in dd if np.isfinite(x[0])]
                    near = min(dd)[1] if dd else None
                out.append({"kind": kind, "at": [round(float(lon[0]), 3), round(float(lat[0]), 3)],
                            "h_m": round(h, 1), "pools": sorted({int(ends_pool[x]) for x in ee}),
                            "basin_km2": None if basin is None else round(basin, 1),
                            "loops": int(n_loops[k]), "n_tris": int(len(tris)), "near": near, "path": path,
                            "_tris": tris})
    # cells across: the largest level k < K at which the place is no neck
    for p in out:
        p["cells_across"] = 1
    if K > 2 and out:
        for k2 in range(2, K):
            Lk = _Level(mesh, k2, boundary, ring_distance(mesh, boundary, max(k2 // 2, 1)))
            narrow_k = set(_neck_tris(Lk, mesh))
            for p in out:
                if p["cells_across"] == k2 - 1 and not narrow_k.intersection(_owners(mesh, p["_tris"])):
                    p["cells_across"] = k2
    for p in out:
        del p["_tris"]
    rec["passages"] = out
    return rec


def _owners(mesh: Mesh, tris) -> set:
    """Mesh elements of split triangles (tris first, then each quad's two halves)."""
    t = np.asarray(tris, dtype=np.int64)
    nt, nq = mesh.n_tris, mesh.n_quads
    return {int(x) if x < nt else int(nt + (x - nt) % nq) for x in t}


def _neck_tris(L: _Level, mesh: Mesh) -> list:
    """Mesh elements of every neck of a level (rim triangles and the two sides of pool-pool chords) with two ends
    or more (a coarser test than narrow_passages: used to tell 1 from 2 cells across)."""
    pp = np.flatnonzero(L.pool_t[L.ca] & L.pool_t[L.cb])
    rp = L.pool_t[L.ca] ^ L.pool_t[L.cb]
    r_tri = np.where(L.pool_t[L.ca[rp]], L.cb[rp], L.ca[rp])
    n_ends = np.bincount(L.rim_comp[L.rim_idx[r_tri]], minlength=L.n_rim_comp)
    multi = n_ends >= 2
    tris = list(L.rim[multi[L.rim_comp]]) + list(L.ca[pp]) + list(L.cb[pp])
    return list(_owners(mesh, tris))


def _end_path(L, la, lb, tris, ends_rtri, ends_ptri, src, dst) -> list:
    """Triangles from a pool triangle of `src` ends through the rim to one of `dst` (fewest rim triangles)."""
    s_r = [int(ends_rtri[x]) for x in src]
    d_r = [int(ends_rtri[x]) for x in dst]
    mid = _path_bfs(la, lb, tris, s_r, d_r)
    if not mid:
        return []
    a = next(int(ends_ptri[x]) for x in src if int(ends_rtri[x]) == mid[0])
    b = next(int(ends_ptri[x]) for x in dst if int(ends_rtri[x]) == mid[-1])
    return [a] + mid + [b]


def _far_path(L, la, lb, tris, ends_rtri, ends_ptri, ee) -> list:
    """Around an island: the path between the two ends farthest apart in rim steps (two sweeps)."""
    loc = {int(v): i for i, v in enumerate(tris)}
    g = coo_matrix((np.ones(len(la)), ([loc[int(x)] for x in la], [loc[int(x)] for x in lb])),
                   shape=(len(tris), len(tris)))
    er = np.array([loc[int(ends_rtri[x])] for x in ee])
    d0 = shortest_path(g, directed=False, unweighted=True, indices=int(er[0]))
    a = int(np.argmax(np.where(np.isfinite(d0[er]), d0[er], -1)))
    da = shortest_path(g, directed=False, unweighted=True, indices=int(er[a]))
    b = int(np.argmax(np.where(np.isfinite(da[er]), da[er], -1)))
    return _end_path(L, la, lb, tris, ends_rtri, ends_ptri, [ee[a]], [ee[b]])


def passage_straits(passages: list, h_bg, cells: int, quads: bool, size_tol: float, start: int = 0) -> list:
    """`widen` straits (meshom.straits.Strait, C45) along the passages' paths: h_m = the background h at its largest
    over the path, `cells` cells across, no width or depth floor; named passage_<n>."""
    from meshom.straits import Strait

    out = []
    for i, p in enumerate(passages):
        path = p["path"]
        hm = float(np.max(h_bg(path)))
        out.append(Strait(f"passage_{start + i}", path, 0.0, hm, int(cells), None, "widen", quads, size_tol))
    return out


def cells_for(found: list, previous: list, cells: int) -> list[int]:
    """The cells across each found passage is widened to: `cells`, or one more than the earlier passage corridors
    (`previous`, Strait objects) that hold its middle point (`at`): the widening there did not suffice."""
    from meshom.geo import lonlat_to_xyz

    out = []
    for p in found:
        q = lonlat_to_xyz(np.array([p["at"][0]]), np.array([p["at"][1]]))
        prev = [s.cells for s in previous if s.distance_m(q, 0.5 * s.corridor_m)[0] <= 0.5 * s.corridor_m]
        out.append(max(prev) + 1 if prev else int(cells))
    return out
