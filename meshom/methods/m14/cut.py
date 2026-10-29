"""m14 variant A (`m14_blocks_cut`): the mesh of the closed sphere cut by the land mask, as FESOM2 cuts its JIGSAW
mesh. The blocks were placed and filled as if there were no land; here whole elements are
removed, nothing moves, so the coast becomes a staircase through the block cells (as on every curvilinear ocean grid).

Rules after D. Sein (AWI), fesom2_mesh_gen:
- wet test per element: wet when the wet fraction of the element is at least 0.5, the fraction from 9 sub-triangle
  samples of the mask. Here: the element's centroid against the coast loops of the original domain
  (`meshom.domain.in_water`), i.e. one sample.
- clean-up loop, repeated until the wet count stops changing:
  spurs: elements with >= 2 boundary edges, repeated (a triangle hanging by one edge; FESOM2 rejects triangles with
  a single neighbour). For a mixed mesh: an element with fewer than 2 edge neighbours (a triangle with >= 2, a quad
  with >= 3 boundary edges); the gate `elem_neighbours_min` = 2.
  pinched nodes: at a node with > 2 boundary edges the wet elements form several fans (joined across the edges at
  the node); the largest fan stays, the others go.
  components: only the largest edge-connected basin stays (here the largest by spherical area, decisions C0).
- extract: kept elements, used nodes renumbered (here: the boundary loops first, in loop order, water on the left,
  node_flag 1; the other nodes after them in their old order).
Not reproduced: the strait edits, the closing of thin channels and narrow necks, the final unprotected pass (nothing
is protected here, so the main loop already ends in that state).
"""
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from meshom.domain import Domain, in_water, loop_signed_area
from meshom.geo import R_EARTH_M, tangent_basis
from meshom.mesh import Mesh
from meshom.methods.m12.coast import Coast
from meshom.size import SizeField


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _arc(chord: np.ndarray) -> np.ndarray:
    return 2.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))


def half_edges(mesh: Mesh) -> tuple:
    """(a, b, e, twin): directed half-edges a -> b of every element (element e on the left), the twin half-edge
    index or -1."""
    a, b, e = [], [], []
    for cells, off in ((mesh.tris, 0), (mesh.quads, mesh.n_tris)):
        a.append(cells.ravel())
        b.append(np.roll(cells, -1, axis=1).ravel())
        e.append(np.repeat(np.arange(len(cells), dtype=np.int64) + off, cells.shape[1]))
    a, b, e = (np.concatenate(x).astype(np.int64) for x in (a, b, e))
    n = mesh.n_nodes
    key = a * n + b
    order = np.argsort(key)
    pos = np.clip(np.searchsorted(key[order], b * n + a), 0, len(key) - 1)
    twin = np.where(key[order][pos] == b * n + a, order[pos], -1)
    return a, b, e, twin


def _spurs(he: tuple, keep: np.ndarray) -> np.ndarray:
    """Kept elements with fewer than 2 kept edge neighbours."""
    a, b, e, tw = he
    inner = keep[e] & (tw >= 0) & keep[e[np.maximum(tw, 0)]]
    nbrs = np.bincount(e[inner], minlength=len(keep))
    return keep & (nbrs < 2)


def _pinch_drop(he: tuple, keep: np.ndarray, n: int) -> np.ndarray:
    """At every node with more than one outgoing boundary half-edge: the kept elements outside its largest fan."""
    a, b, e, tw = he
    M = len(keep)
    hk = keep[e]
    inner = hk & (tw >= 0) & keep[e[np.maximum(tw, 0)]]
    cand = np.bincount(a[hk & ~inner], minlength=n) > 1
    drop = np.zeros(M, dtype=bool)
    if not cand.any():
        return drop
    corner = hk & cand[a]                                   # one half-edge per element corner at a candidate node
    key = np.unique(a[corner] * M + e[corner])              # (node, element) pairs
    lk = inner & cand[a]                                    # every edge at a candidate node, once from that node
    u = np.searchsorted(key, a[lk] * M + e[lk])
    v = np.searchsorted(key, a[lk] * M + e[tw[lk]])
    g = coo_matrix((np.ones(len(u)), (u, v)), shape=(len(key), len(key)))
    _, fan = connected_components(g, directed=False)
    node = key // M
    size = np.bincount(fan)
    # per node: the largest fan (ties: the smallest fan label) stays
    order = np.lexsort((fan, -size[fan], node))
    first = np.r_[True, node[order][1:] != node[order][:-1]]
    best = np.zeros(n, dtype=np.int64)
    best[node[order][first]] = fan[order][first]
    drop[(key % M)[fan != best[node]]] = True
    return drop


def _largest_component(he: tuple, keep: np.ndarray, areas: np.ndarray | None = None,
                       min_area_m2: float | None = None) -> np.ndarray:
    a, b, e, tw = he
    M = len(keep)
    inner = keep[e] & (tw >= 0) & keep[e[np.maximum(tw, 0)]]
    g = coo_matrix((np.ones(inner.sum()), (e[inner], e[tw[inner]])), shape=(M, M))
    _, lab = connected_components(g, directed=False)
    lab = np.where(keep, lab, -1)
    tot = np.bincount(lab[keep], weights=None if areas is None else areas[keep])   # basin areas (element counts)
    big = keep & (lab == np.argmax(tot))                       # the largest basin always stays (C48)
    if min_area_m2 is not None:                                # and every other basin of at least min_area_m2
        return big | (keep & (tot[np.maximum(lab, 0)] >= min_area_m2))
    return big


def clean(he: tuple, keep: np.ndarray, n: int, areas: np.ndarray | None = None,
          min_basin_m2: float | None = None, protect: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """FESOM2's clean-up loop on the element mask (spurs repeated, pinched nodes, components) until stable.
    Components: FESOM2 keeps the largest basin only, here the largest by `areas` (m^2 per element; by element count
    without them: under variable resolution a refined cut-off sea can outnumber the ocean, C48); with `min_basin_m2`
    every other edge-connected basin of at least that area stays too (the 150 km m14
    mesh lost the Mediterranean and the Red Sea when Gibraltar did not connect).
    `protect` (forced strait corridors, meshom.straits): never removed as spurs or pinch fans in the main loop (only
    with their basin), then one more loop without protection, as fesom2_mesh_gen's final unprotected pass
    (FESOM2 rejects an element with one neighbour); its removals are counted in "final_pass"."""
    keep = keep.copy()
    free = np.ones(len(keep), dtype=bool) if protect is None else ~protect
    cnt = {"spurs": 0, "pinched": 0, "components": 0, "iterations": 0}
    while True:
        k0 = int(keep.sum())
        cnt["iterations"] += 1
        while True:
            s = _spurs(he, keep) & free
            if not s.any():
                break
            keep &= ~s
            cnt["spurs"] += int(s.sum())
        p = _pinch_drop(he, keep, n) & free
        keep &= ~p
        cnt["pinched"] += int(p.sum())
        c = _largest_component(he, keep, areas, min_basin_m2)
        cnt["components"] += int((keep & ~c).sum())
        keep = c
        if int(keep.sum()) == k0:
            break
    if protect is not None:
        k0 = int(keep.sum())
        keep, fin = clean(he, keep, n, areas, min_basin_m2)
        cnt["final_pass"] = k0 - int(keep.sum())
    return keep, cnt


def boundary_loops(mesh: Mesh, he: tuple | None = None) -> list[np.ndarray]:
    """Closed node loops of the mesh boundary, water (the elements) on the left; requires no pinch node."""
    a, b, e, tw = half_edges(mesh) if he is None else he
    bnd = tw < 0
    nxt = np.full(mesh.n_nodes, -1, dtype=np.int64)
    nxt[a[bnd]] = b[bnd]
    seen = np.zeros(mesh.n_nodes, dtype=bool)
    loops = []
    for s in a[bnd]:                                         # O(boundary nodes) walk
        if seen[s]:
            continue
        lp = [s]
        seen[s] = True
        v = nxt[s]
        while v != s:
            lp.append(v)
            seen[v] = True
            v = nxt[v]
        loops.append(np.array(lp, dtype=np.int64))
    return loops


def extract(mesh: Mesh, keep: np.ndarray) -> tuple[Mesh, np.ndarray, list[np.ndarray]]:
    """Sub-mesh of the kept elements: boundary loop nodes first (loop order, flag 1), then the other used nodes in
    their old order. Returns (mesh, old -> new node map with -1 for dropped nodes, boundary loops in new ids)."""
    T = mesh.n_tris
    sub = Mesh(mesh.points, mesh.tris[keep[:T]], mesh.quads[keep[T:]], mesh.node_flag)
    loops = boundary_loops(sub)
    used = np.zeros(mesh.n_nodes, dtype=bool)
    used[sub.tris.ravel()] = True
    used[sub.quads.ravel()] = True
    bn = np.concatenate(loops) if loops else np.zeros(0, dtype=np.int64)
    rest = np.flatnonzero(used)
    rest = rest[~np.isin(rest, bn)]
    old = np.concatenate([bn, rest])
    nmap = np.full(mesh.n_nodes, -1, dtype=np.int64)
    nmap[old] = np.arange(len(old))
    flag = np.zeros(len(old), dtype=np.int8)
    flag[:len(bn)] = 1
    out = Mesh(mesh.points[old], nmap[sub.tris], nmap[sub.quads], flag)
    return out, nmap, [nmap[lp] for lp in loops]


# ------------------------------------------------------------------------------------------------- coast fit

def _excess(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    return 2.0 * np.arctan2(det, 1.0 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c)
                            + np.einsum("ij,ij->i", c, a))


def element_areas(mesh: Mesh) -> np.ndarray:
    """(T+Q,) spherical areas (m^2; quads as two triangles 012 + 023)."""
    P = mesh.points
    t, q = mesh.tris, mesh.quads
    at = _excess(P[t[:, 0]], P[t[:, 1]], P[t[:, 2]])
    aq = _excess(P[q[:, 0]], P[q[:, 1]], P[q[:, 2]]) + _excess(P[q[:, 0]], P[q[:, 2]], P[q[:, 3]])
    return np.r_[at, aq] * R_EARTH_M**2


def _samples(mesh: Mesh, elems: np.ndarray, k: int) -> np.ndarray:
    """(len(elems), k*k, 3) sample points per element: sub-triangle centroids of a k-split triangle, cell centres of
    a k x k bilinear quad grid."""
    T = mesh.n_tris
    P = mesh.points
    out = np.empty((len(elems), k * k, 3))
    it, iq = elems < T, elems >= T
    bary = []
    for i in range(k):
        for j in range(k - i):
            bary.append((i + 1 / 3, j + 1 / 3, k - i - j - 2 / 3))
    for i in range(k - 1):
        for j in range(k - 1 - i):
            bary.append((i + 2 / 3, j + 2 / 3, k - i - j - 4 / 3))
    bary = np.array(bary) / k                                               # (k*k, 3)
    if it.any():
        out[it] = np.einsum("sk,tkd->tsd", bary, P[mesh.tris[elems[it]]])
    if iq.any():
        s = (np.arange(k) + 0.5) / k
        u, v = np.meshgrid(s, s)
        u, v = u.ravel(), v.ravel()
        w = np.column_stack([(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v])   # (k*k, 4)
        out[iq] = np.einsum("sk,tkd->tsd", w, P[mesh.quads[elems[iq] - T]])
    return _unit(out)


def coast_fit(mesh: Mesh, domain: Domain, h: SizeField, nsub: int = 6) -> dict:
    """How the mesh boundary represents the true coast (domain.loops as given, not resampled).

    - boundary -> coast: great-circle distance of every boundary node to the coast polyline, units of h at the node
      (max, p99, median); coast -> boundary: of every coast vertex to the boundary polyline (catches closed straits);
    - water area: the true water (sphere minus the land loops) vs the mesh: gained = mesh area on land, lost = true
      water outside the mesh (elements near the coast sampled by nsub^2 points), net = mesh / true - 1, shares of the
      true water;
    - boundary shape: length / true coast length, the share of boundary nodes turning by more than 45 degrees and the
      "coast turn" share (boundary nodes with 1 or >= 3 elements), boundary loops, boundary nodes and elements.
    """
    he = half_edges(mesh)
    a, b, e, tw = he
    loops = boundary_loops(mesh, he)
    P = mesh.points
    bn = np.concatenate(loops)
    coast = Coast(list(domain.loops))
    _, _, ch = coast.project(P[bn])
    d_b = _arc(ch) * R_EARTH_M / h(P[bn])
    bl = Coast([P[lp] for lp in loops])
    cv = np.vstack(domain.loops)
    _, _, ch = bl.project(cv)
    d_c = _arc(ch) * R_EARTH_M / h(cv)

    area = element_areas(mesh)
    water_true = 4.0 * np.pi * R_EARTH_M**2 + sum(loop_signed_area(lp) for lp in domain.loops)
    cent = _unit(np.r_[P[mesh.tris].sum(axis=1), P[mesh.quads].sum(axis=1)])
    rad = np.r_[np.linalg.norm(P[mesh.tris] - cent[:mesh.n_tris, None], axis=2).max(axis=1, initial=0.0),
                np.linalg.norm(P[mesh.quads] - cent[mesh.n_tris:, None], axis=2).max(axis=1, initial=0.0)]
    _, _, ch = coast.project(cent)
    near = np.flatnonzero(ch < rad + 1e-6)
    s = _samples(mesh, near, nsub)
    land = ~in_water(domain, s.reshape(-1, 3)).reshape(len(near), -1)
    gained = float((area[near] * land.mean(axis=1)).sum())
    mesh_area = float(area.sum())

    # boundary shape: turning angle at each loop node in its tangent plane, lengths
    turn, blen = [], 0.0
    for lp in loops:
        p, q, r = P[np.roll(lp, 1)], P[lp], P[np.roll(lp, -1)]
        e1, e2 = tangent_basis(q)
        din = np.column_stack([np.einsum("ij,ij->i", q - p, e1), np.einsum("ij,ij->i", q - p, e2)])
        dout = np.column_stack([np.einsum("ij,ij->i", r - q, e1), np.einsum("ij,ij->i", r - q, e2)])
        turn.append(np.degrees(np.abs(np.arctan2(din[:, 0] * dout[:, 1] - din[:, 1] * dout[:, 0],
                                                 np.einsum("ij,ij->i", din, dout)))))
        blen += float(_arc(np.linalg.norm(r - q, axis=1)).sum())
    turn = np.concatenate(turn)
    clen = sum(float(_arc(np.linalg.norm(np.roll(lp, -1, axis=0) - lp, axis=1)).sum()) for lp in domain.loops)
    epn = np.bincount(np.r_[mesh.tris.ravel(), mesh.quads.ravel()], minlength=mesh.n_nodes)[bn]
    bel = np.zeros(mesh.n_elems, dtype=bool)
    bel[e[tw < 0]] = True
    bnode = np.zeros(mesh.n_nodes, dtype=bool)
    bnode[bn] = True
    touch = np.r_[bnode[mesh.tris].any(axis=1), bnode[mesh.quads].any(axis=1)]

    def st(x):
        return {"max": float(x.max()), "p99": float(np.percentile(x, 99)), "median": float(np.median(x))}

    return {"boundary_to_coast_h": st(d_b), "coast_to_boundary_h": st(d_c),
            "water_true_m2": water_true, "mesh_area_m2": mesh_area,
            "area_gained_share": gained / water_true, "area_lost_share": (water_true - mesh_area + gained) / water_true,
            "area_net_share": mesh_area / water_true - 1.0,
            "boundary_length_ratio": blen / clen, "share_turn_gt45": float(np.mean(turn > 45.0)),
            "turn_deg_p50": float(np.median(turn)), "share_coast_turn_07c": float(np.mean(epn != 2)),
            "n_loops": len(loops), "n_boundary_nodes": int(len(bn)), "n_boundary_edge_elems": int(bel.sum()),
            "n_boundary_node_elems": int(touch.sum()), "n_sampled_elems": int(len(near))}


# ------------------------------------------------------------------------------------------------------- cut

def fixed_banks(he: tuple, wet: np.ndarray, protect: np.ndarray, bank_edges: np.ndarray, n_nodes: int) -> None:
    """In place on the element mask `wet`: the protected cells (a fixed block with a coast side) are water and the
    element across each of their coast-side (bank) edges is land, so the cut coast follows the fitted banks (B59).
    bank_edges (E, 2): the blocks' bank edges (FixedBlock.bank), not every edge between two coast-side nodes: the
    mouths of a block one cell wide join two coast-side nodes too and stay open. Shared by cut (m14's variant A) and
    meshom.run.cut_land."""
    wet |= protect
    a, b, e, tw = he
    key = np.minimum(bank_edges[:, 0], bank_edges[:, 1]) * n_nodes + np.maximum(bank_edges[:, 0], bank_edges[:, 1])
    across = protect[e] & np.isin(np.minimum(a, b) * n_nodes + np.maximum(a, b), key) & (tw >= 0)   # a bank edge
    wet[e[tw[across]][~protect[e[tw[across]]]]] = False                 # the element across it is land


def cut(mesh: Mesh, st: dict, domain: Domain, h: SizeField) -> tuple[Mesh, dict, dict]:
    """Remove the elements whose centroid is on land, clean (FESOM2 rules), renumber; map stitch's records (is_block,
    outline_edges, block_nodes; fixed_id, fixed_coast_nodes, fixed_bank_edges) to the new ids. Returns (mesh, st,
    info). Without loops nothing changes. Fixed blocks with a coast side (st["fixed_id"] >= 0 and st["fixed_protect"]:
    the strait blocks of m14/strait.py, hand-placed blocks with a coast side, m14/hand.py) are water and protected in
    the clean-up's main loop (as strait corridors, C32), and the fill elements across their coast sides (the edges
    st["fixed_bank_edges"]) are land, so the coast follows the fitted banks there; hand-placed blocks without a coast side are cut by the mask as the rest (ORCA style, B60)."""
    info = {"n_elems_before": mesh.n_elems, "n_nodes_before": mesh.n_nodes}
    if not domain.loops:
        info.update(n_removed_mask=0, n_removed_clean=0, n_elems=mesh.n_elems, n_nodes=mesh.n_nodes)
        return mesh, st, info
    t0 = time.perf_counter()
    P = mesh.points
    cent = _unit(np.r_[P[mesh.tris].sum(axis=1), P[mesh.quads].sum(axis=1)])
    wet = in_water(domain, cent)
    he = half_edges(mesh)
    fx = st.get("fixed_id")
    protect = None
    if fx is not None:
        protect = (fx >= 0) & st["fixed_protect"][np.maximum(fx, 0)]
        fixed_banks(he, wet, protect, st["fixed_bank_edges"], mesh.n_nodes)
    keep, cnt = clean(he, wet, mesh.n_nodes, element_areas(mesh), protect=protect)
    out, nmap, loops = extract(mesh, keep)
    blk = st["is_block"]
    ol = nmap[st["outline_edges"]] if len(st["outline_edges"]) else st["outline_edges"]
    ol = ol[(ol >= 0).all(axis=1)] if len(ol) else ol
    bnodes = nmap[st["block_nodes"]]
    st = dict(st, is_block=blk[keep], outline_edges=ol, block_nodes=bnodes[bnodes >= 0])
    if fx is not None:
        fc = nmap[st["fixed_coast_nodes"]]
        fb = nmap[st["fixed_bank_edges"]]
        st.update(fixed_id=fx[keep], fixed_coast_nodes=fc[fc >= 0], fixed_bank_edges=fb[(fb >= 0).all(axis=1)])
        info["n_fixed_cells_removed"] = int(((fx >= 0) & ~keep).sum())
    info.update(n_removed_mask=int((~wet).sum()), n_removed_clean=int((wet & ~keep).sum()), removed_clean_by=cnt,
                n_block_cells_removed=int((blk & ~keep).sum()), n_block_cells_kept=int((blk & keep).sum()),
                n_elems=out.n_elems, n_nodes=out.n_nodes, n_tris=out.n_tris, n_quads=out.n_quads,
                n_loops=len(loops), n_loops_domain=len(domain.loops), cut_s=time.perf_counter() - t0)
    return out, st, info
