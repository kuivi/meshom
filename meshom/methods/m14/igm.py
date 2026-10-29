"""m14_blocks_igm: blocks that may hold B4 cones.

Placement: a cone is "good" when no other cone lies within igm_cone_gap h; the others keep their cone_margin exclusion.
Each good cone's disc (triangles within igm_cone_edge h of it) joins a region whole or not at all: its triangles get
the disc's smallest growth priority, so the growth floods it in one go, and only growth prefixes that hold complete
discs (and at most igm_max_cones of them) are candidates. Admission: the residual pre-screen, |B - sum of the
cone charges| <= min(b_cap, b_slope L) on the largest valid prefix (the cone charge, +-1 per cone, is spread over the
nodes within 2 h of it); the lambda test is not used, since the log peaks of the cones dominate F. The growth
runs without the pre-screen stop and leaves at least igm_min_fill cells of water to the fill; one block covers at
most igm_max_share of the water (a fill inside one open hemisphere makes the sphere Delaunay of the fill points a
convex hull with one huge, nearly planar face, where the outline edges are not recovered).

Block map: m09's parametrisation with cuts and integers inside the block (cones at integer points, integer seam
translations), free outline, target size h, stiffening as m09; the global B4 field restricted to the block is the
frame. Cells: m09.extract's unit cells (walks across the seams; cells near folds dropped), its fill triangles dropped;
cleaning on the quad graph: boundary cells with an edge |ln s/h| > igm_trim peeled (looser than trim: cells near a
cone are 0.2-0.3 off h in any integer-grid map, and peeling through them empties the block), the largest edge-connected component,
to a fixed point cells with >= 3 boundary edges and cells at pinch nodes removed; a block with more than one outline
loop is dropped. Then `smooth` kind "size" with the outline fixed.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M, arc_length
from meshom.mesh import Mesh, signed_area
from meshom.methods.m14.blockmap import BlockGrid
from meshom.methods.m14.fitness import Fitness, dist_h
from meshom.methods.m14.place import Region, _grow, _tri_excess, border, boundary_nodes, edge_cost, edge_list


def _good_cones(fit: Fitness, o: dict) -> np.ndarray:
    cp = fit.cone_pos
    if len(cp) < 2:
        return np.ones(len(cp), dtype=bool)
    hc = fit.hv[cKDTree(fit.bg.points).query(cp)[1]]
    d, _ = cKDTree(cp).query(cp, k=2)
    return 2.0 * np.arcsin(np.clip(0.5 * d[:, 1], 0, 1)) * R_EARTH_M / hc >= o["igm_cone_gap"]


def place_igm(fit: Fitness, o: dict) -> tuple[list[Region], dict]:
    P, T, hv = fit.bg.points, fit.bg.tris, fit.hv
    n = len(P)
    cp, cidx = fit.cone_pos, fit.cone_idx
    good = _good_cones(fit, o)
    base = (fit.d_coast >= o["coast_margin"]) & (fit.d_complex >= o["complex_margin"])
    cen = P[T].sum(axis=1)
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    h_t = hv[T].mean(axis=1)
    for _ in range(3):                                   # demote good cones whose disc is not admissible
        adm = base & (dist_h(P, cp[~good], hv) >= o["cone_margin"])
        gi = np.flatnonzero(good)
        disc = np.full(len(T), -1, dtype=np.int64)
        if len(gi):
            d, k = cKDTree(cp[gi]).query(cen)
            near = 2.0 * np.arcsin(np.clip(0.5 * d, 0, 1)) * R_EARTH_M / h_t <= o["igm_cone_edge"]
            disc[near] = k[near]
        adm_t = adm[T].all(axis=1)
        bad = np.unique(disc[(disc >= 0) & ~adm_t])
        if len(bad) == 0:
            break
        good[gi[bad]] = False
    gi = np.flatnonzero(good)
    # cone charges (rho units, +-1 per cone) spread over the nodes within 2 h, then onto the triangles
    exc = _tri_excess(P, T)
    q = ((fit.rho / np.maximum(fit.area, 1e-300))[T] * (exc / 3.0)[:, None]).sum(axis=1)
    if len(gi):
        tree = cKDTree(P)
        for c in gi:
            hc = hv[tree.query(cp[c])[1]]
            nodes = tree.query_ball_point(cp[c], 2.0 * np.sin(hc / R_EARTH_M))     # chord of an arc of 2 h
            nodes = np.asarray(nodes, dtype=np.int64)
            if len(nodes) == 0:
                nodes = np.array([tree.query(cp[c])[1]])
            w = np.zeros(n)
            w[nodes] = fit.area[nodes] / fit.area[nodes].sum()
            q -= 4.0 * cidx[c] * ((w / np.maximum(fit.area, 1e-300))[T] * (exc / 3.0)[:, None]).sum(axis=1)
    a = exc * R_EARTH_M ** 2 / h_t ** 2
    a_tot = float(a.sum())
    a_cap = max(min(a_tot - o["igm_min_fill"], o["igm_max_share"] * a_tot), o["min_cells"])
    E = edge_list(T, n)
    G = coo_matrix((edge_cost(P, E, fit.u, hv, o["grow"]), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    vt = np.full(n, -1, dtype=np.int64)
    vt[T.ravel()] = np.repeat(np.arange(len(T)), 3)
    discs = [np.flatnonzero(disc == j) for j in range(len(gi))]
    tried = np.zeros(n, dtype=bool)
    regions: list[Region] = []
    log = {"rejected": 0, "good_cones": int(len(gi)), "cones": int(len(cp))}
    mb = o["max_blocks"] or 10 ** 6                       # None / 0: no cap (B70)
    for _ in range(mb * 4):
        if len(regions) >= mb:
            break
        cand = np.flatnonzero(adm & ~tried)
        if len(cand) == 0:
            break
        clear = dist_h(P[cand], P[border(E, adm)], hv[cand])
        if np.isinf(clear).all():                       # nothing excluded: start far from the good cones
            clear = dist_h(P[cand], cp[gi], hv[cand])
        k = int(np.argmax(clear))
        if clear[k] < o["min_clear"]:
            break
        seed = int(cand[k])
        dist = dijkstra(G, directed=False, indices=seed)
        prio = np.where(np.isfinite(dist[T]).all(axis=1), dist[T].max(axis=1), np.inf)
        for D in discs:
            if len(D):
                prio[D] = prio[D].min()
        adm_tt = adm[T].all(axis=1) & np.isfinite(prio)
        order, cum_b, cum_a = _grow(int(vt[seed]), prio, adm_tt, fit.nbr, T, n, q, a, np.inf, np.inf, a_cap,
                                    o["compact"], 0.25 * o["min_cells"])
        m = len(order)
        valid = np.ones(m, dtype=bool)
        ncone = np.zeros(m + 1, dtype=np.int64)
        pos = np.full(len(T), -1, dtype=np.int64)
        pos[order] = np.arange(m)
        for D in discs:
            p = pos[D]
            if not (p >= 0).any():
                continue
            first = int(p[p >= 0].min())
            if (p >= 0).all():
                last = int(p.max())
                valid[first:last] = False
                ncone[last] += 1
            else:
                valid[first:] = False
        ncone = np.cumsum(ncone)[:m]
        L = 2.0 * np.sqrt(np.maximum(cum_a, 0.0) / np.pi)
        ok = valid & (ncone <= o["igm_max_cones"]) & (cum_a >= o["min_cells"]) & \
            (np.abs(cum_b) <= np.minimum(o["b_cap"], o["b_slope"] * L))
        if not ok.any():
            log["rejected"] += 1
            tried |= dist_h(P, P[[seed]], hv) <= o["min_clear"]
            continue
        j = int(np.flatnonzero(ok)[-1])
        tris = order[: j + 1]
        nodes = np.unique(T[tris])
        regions.append(Region(tris, nodes, np.zeros(len(nodes)), 0.0, 0.0, float(cum_b[j]), float(cum_a[j]), seed,
                              info={"n_cones": int(ncone[j])}))
        adm[nodes] = False
        adm &= dist_h(P, P[boundary_nodes(T[tris], n)], hv) > o["block_gap"]
    log["n_regions"] = len(regions)
    return regions, log


def _boundary_halfedges(quads: np.ndarray, n: int):
    a = quads.ravel()
    b = np.roll(quads, -1, axis=1).ravel()
    key, rkey = a * n + b, b * n + a
    sk = np.sort(key)
    pos = np.minimum(np.searchsorted(sk, rkey), len(sk) - 1)
    bnd = (sk[pos] != rkey).reshape(quads.shape)
    return bnd


def clean_quads(pts: np.ndarray, quads: np.ndarray, h, trim: float, max_iter: int = 100) -> tuple:
    """Cleaning on the quad graph (module docstring): (kept quads, outline loop with the fill on its left) or None."""
    n = len(pts)
    q = quads
    # edge |ln s/h| per quad edge
    a, b = pts[q], pts[np.roll(q, -1, axis=1)]
    mid = a + b
    mid /= np.linalg.norm(mid, axis=2, keepdims=True)
    badq = (np.abs(np.log(R_EARTH_M * arc_length(a.reshape(-1, 3), b.reshape(-1, 3))
                          / h(mid.reshape(-1, 3)))).reshape(q.shape) > trim).any(axis=1)
    keep = np.ones(len(q), dtype=bool)
    for _ in range(max_iter):
        old = keep.copy()
        idx = np.flatnonzero(keep)
        bnd = _boundary_halfedges(q[idx], n)
        nb = bnd.sum(axis=1)
        keep[idx[(nb > 0) & badq[idx]]] = False                       # peel trimmed cells from the outside
        keep[idx[nb >= 3]] = False                                     # spurs
        idx = np.flatnonzero(keep)
        if len(idx) == 0:
            return None
        bnd = _boundary_halfedges(q[idx], n)
        outdeg = np.bincount(q[idx][bnd], minlength=n)
        pinch = np.flatnonzero(outdeg > 1)
        if len(pinch):
            hit = np.isin(q[idx], pinch).any(axis=1)
            keep[idx[hit & (bnd.sum(axis=1) >= 1)]] = False
        # largest edge-connected component
        idx = np.flatnonzero(keep)
        qq = q[idx]
        ea = np.sort(np.stack([qq, np.roll(qq, -1, axis=1)], axis=-1).reshape(-1, 2), axis=1)
        key = ea[:, 0] * n + ea[:, 1]
        owner = np.repeat(np.arange(len(idx)), 4)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        same = np.flatnonzero(ks[1:] == ks[:-1])
        g = coo_matrix((np.ones(len(same)), (owner[order[same]], owner[order[same + 1]])),
                       shape=(len(idx), len(idx)))
        _, lab = connected_components(g, directed=False)
        big = np.argmax(np.bincount(lab))
        keep[idx[lab != big]] = False
        if np.array_equal(keep, old):
            break
    idx = np.flatnonzero(keep)
    qq = q[idx]
    bnd = _boundary_halfedges(qq, n)
    fa = np.roll(qq, -1, axis=1)[bnd]                                  # reversed: fill on the left
    fb = qq[bnd]
    nxt = dict(zip(fa.tolist(), fb.tolist()))
    loops, seen = [], set()
    for s in fa.tolist():
        if s in seen:
            continue
        lp, x = [s], nxt[s]
        seen.add(s)
        while x != s and len(lp) <= len(fa):
            lp.append(x)
            seen.add(x)
            x = nxt[x]
        if x != s:                                                     # a pinch survived: no simple outline
            return None
        loops.append(lp)
    if len(loops) != 1:
        return None
    return qq, np.asarray(loops[0], dtype=np.int64)


def block_grid_igm(P, T, lam, in_r, r: Region, h, o, u_ref) -> BlockGrid | None:
    from meshom.methods.m09.extract import extract
    from meshom.methods.m09.param import parametrize
    from meshom.smooth import smooth

    info = {}
    prm = parametrize(P, T, u_ref, h(P), "greedy", 0.0, o["igm_stiffen"], o["igm_stiffen"], 1.0,
                      free_boundary=True)
    info.update(n_cones_map=prm.info["n_cones"], n_flipped=int(prm.flipped.sum()),
                map_log_scale_p99=prm.info["log_scale_p99"], map_aniso_p99=prm.info["aniso_p99"],
                int_nonint=prm.info.get("int_nonint", 0))
    pts, _, quads, xinfo = extract(P, T, prm, np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int64))
    info["n_cells_lattice"] = int(len(quads))
    res = clean_quads(pts, quads, h, o["igm_trim"])
    if res is None:
        return None
    quads, loop = res
    used = np.unique(quads)
    new = np.full(len(pts), -1, dtype=np.int64)
    new[used] = np.arange(len(used))
    quads, loop, pts = new[quads], new[loop], pts[used]
    from meshom.methods.m14.blockmap import self_crossings

    if self_crossings(pts[loop]):
        return None
    info["n_cells"] = int(len(quads))
    # inradius in cells: graph distance (quad rings) from the outline
    on = np.zeros(len(pts), dtype=bool)
    on[loop] = True
    ring, k = on.copy(), 0
    while not ring.all() and k < 10000:
        touch = ring[quads].any(axis=1)
        new_ring = ring.copy()
        new_ring[quads[touch].ravel()] = True
        if np.array_equal(new_ring, ring):
            break
        ring, k = new_ring, k + 1
    info["inradius_cells"] = float(k)
    mesh = Mesh(pts, np.zeros((0, 3), dtype=np.int64), quads, np.zeros(len(pts), dtype=np.int8))
    fixed = np.zeros(len(pts), dtype=bool)
    fixed[loop] = True
    if o["block_size_iters"]:
        mesh = smooth(mesh, h, fixed, iters=o["block_size_iters"], kind="size")
        mesh = smooth(mesh, h, fixed, iters=3, kind="angle")
    info["n_inverted"] = int((signed_area(mesh) <= 0).sum())
    return BlockGrid(mesh.points, quads, np.zeros((len(pts), 2), dtype=np.int64), loop, info)
