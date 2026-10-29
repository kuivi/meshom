"""Mesh extraction from the position field (Jakob 2015 §3.8), made robust on the sphere with a fixed coast.

1. Edge classes from the integer jumps of the background edges (fields.edge_jumps): jump 0 = both ends refer to the
   same output vertex ("collapse"), a unit jump = one output edge ("unit"), anything else (diagonals, longer) ignored
   (as the released code).
2. Collapse: union-find over the collapse edges in increasing order of their matching error; a merge is skipped when
   the two clusters are already joined by a unit edge (conflicting integer data near singularities; paper §3.8).
3. Output vertex = cluster; position = mean of the members' representative positions o_i with weights
   exp(-9 |o_i - v_i|^2 / rho_i^2) (beta = rho / 3), on the sphere. Clusters with fewer than `spurious` x the mean
   member count are dropped (paper: 1/10). Clusters closer than `merge` x h are merged (the snapping step of the
   released code merges vertices closer than 0.3 rho).
4. Coast (meshom-specific): the output coast nodes are exactly resample_loops(domain, h). A cluster within `snap` x h of
   the resampled coast polyline is snapped onto its nearest coast node (several clusters may land on one node, they
   merge; their unit edges are kept, so the first row still gets lattice quads); clusters on land are dropped.
5. Faces (deviation from the paper's oriented-edge walk, which can give non-manifold output, paper §3.8): the sphere
   Delaunay triangulation of the output vertices with the coast edges recovered and the land removed (always a valid
   CCW manifold). A triangle pair across a non-lattice edge whose other four edges are all lattice edges (unit edges or
   coast edges) is a face bounded by four unit edges: it becomes a quad (if its largest-angle cost passes the same
   price test as recombination). Faces of other sizes (triangles, pentagons and larger at position singularities)
   stay triangulated; meshom.recombine then pairs the leftover triangles geometrically (price), so a pentagon becomes
   a quad + a triangle.
6. resplit_pentagons: each leftover triangle and the quad across its longest edge are re-split into the best of the 5
   quad + triangle splits of their pentagon (worst angle relative to the hard gates); the recombination chose which
   triangle is left over without looking at its shape.
"""
import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from meshom.delaunay import adjacency, sphere_delaunay
from meshom.domain import Domain, in_water, loop_edges, resample_loops
from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh
from meshom.recombine import quad_cost, recombine
from meshom.size import SizeField


@njit(cache=True)
def _find(parent, i):
    r = i
    while parent[r] != r:
        r = parent[r]
    while parent[i] != r:
        nxt = parent[i]
        parent[i] = r
        i = nxt
    return r


@njit(cache=True)
def _collapse(n, ci, cj, order, ui, uj):
    """Union-find over collapse edges (ci, cj) in `order`, skipping a merge of two clusters joined by a unit edge
    (ui, uj). Unit half-edges are kept in one linked list per cluster root. Returns (root per vertex, #conflicts)."""
    parent = np.arange(n)
    size = np.ones(n, dtype=np.int64)
    m = len(ui)
    head = np.full(n, -1, dtype=np.int64)
    tail = np.full(n, -1, dtype=np.int64)
    nxt = np.full(2 * m, -1, dtype=np.int64)
    oth = np.empty(2 * m, dtype=np.int64)
    for e in range(m):
        for side in range(2):
            a = ui[e] if side == 0 else uj[e]
            b = uj[e] if side == 0 else ui[e]
            slot = 2 * e + side
            oth[slot] = b
            if head[a] < 0:
                head[a] = slot
            else:
                nxt[tail[a]] = slot
            tail[a] = slot
    conflicts = 0
    for k in order:
        ra, rb = _find(parent, ci[k]), _find(parent, cj[k])
        if ra == rb:
            continue
        if size[ra] > size[rb]:
            ra, rb = rb, ra                                   # scan the smaller cluster's list
        bad = False
        s = head[ra]
        while s >= 0:
            if _find(parent, oth[s]) == rb:
                bad = True
                break
            s = nxt[s]
        if bad:
            conflicts += 1
            continue
        parent[ra] = rb
        size[rb] += size[ra]
        if head[ra] >= 0:
            if head[rb] < 0:
                head[rb] = head[ra]
            else:
                nxt[tail[rb]] = head[ra]
            tail[rb] = tail[ra]
    roots = np.empty(n, dtype=np.int64)
    for i in range(n):
        roots[i] = _find(parent, i)
    return roots, conflicts


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _relabel(lab: np.ndarray) -> tuple[np.ndarray, int]:
    u, inv = np.unique(lab, return_inverse=True)
    return inv.astype(np.int64), len(u)


def _cluster_points(lab: np.ndarray, nc: int, O: np.ndarray, wgt: np.ndarray) -> np.ndarray:
    s = np.column_stack([np.bincount(lab, wgt * O[:, k], minlength=nc) for k in range(3)])
    return _unit(s)


def _arc_distance(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle from x to the great-circle segment a-b (row-wise)."""
    n = _unit(np.cross(a, b))
    z = np.einsum("ij,ij->i", x, n)
    f = x - z[:, None] * n
    inside = (np.einsum("ij,ij->i", np.cross(a, f), n) >= 0) & (np.einsum("ij,ij->i", np.cross(f, b), n) >= 0)
    d_line = np.arcsin(np.clip(np.abs(z), 0.0, 1.0))
    d_end = np.minimum(np.arccos(np.clip(np.einsum("ij,ij->i", x, a), -1, 1)),
                       np.arccos(np.clip(np.einsum("ij,ij->i", x, b), -1, 1)))
    return np.where(inside, d_line, d_end)


def _merge_pairs(tris: np.ndarray, t: np.ndarray, k: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Quads (a, x, b, c) from CCW triangle t = (a, b, c) rotated so its edge k is a -> b, x = apex of u across it."""
    a = tris[t, k]
    b = tris[t, (k + 1) % 3]
    c = tris[t, (k + 2) % 3]
    tu = tris[u]
    x = tu[(tu != a[:, None]) & (tu != b[:, None])]
    return np.column_stack([a, x, b, c]).astype(np.int64)


def _corner_angles(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """(M,k) interior corner angles (rad) in each cell's tangent plane; a reflex/inverted corner gets > pi."""
    from meshom.geo import tangent_coords

    xy = tangent_coords(points[cells])
    a = np.roll(xy, 1, axis=1) - xy
    b = np.roll(xy, -1, axis=1) - xy
    ang = np.arctan2(b[..., 0] * a[..., 1] - b[..., 1] * a[..., 0], np.einsum("mkd,mkd->mk", a, b))
    return np.where(ang < 0, ang + 2.0 * np.pi, ang)        # CCW cell: angle from b (to next) to a (to prev), CCW


def _badness(points: np.ndarray, tri: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Worst angle relative to the hard gates (1 = on a gate): quad max |theta - 90| / 60 (150 / 30 deg), triangle
    max((theta_max - 60) / 60, (60 - theta_min) / 40) (120 / 20 deg); inf when a cell is not convex."""
    qa, ta = _corner_angles(points, quad), _corner_angles(points, tri)
    qb = np.abs(qa - 0.5 * np.pi).max(axis=1) / (np.pi / 3)
    tb = np.maximum((ta.max(axis=1) - np.pi / 3) / (np.pi / 3), (np.pi / 3 - ta.min(axis=1)) / np.radians(40.0))
    bad = (qa >= np.pi).any(axis=1) | (ta >= np.pi).any(axis=1)
    return np.where(bad, np.inf, np.maximum(qb, tb))


def resplit_pentagons(mesh: Mesh, rounds: int = 2) -> tuple[Mesh, int]:
    """Each triangle and the quad across its longest edge form a pentagon (a -> x -> y -> b -> c); it is re-split into
    the best of its 5 quad + triangle splits (_badness), on disjoint pentagons, worst triangles first. The pentagon's
    outline does not change, so coast nodes and neighbours are untouched. Returns the mesh and the number of changes."""
    tris, quads = mesh.tris.copy(), mesh.quads.copy()
    n = mesh.n_nodes
    changed = 0
    for _ in range(rounds):
        if len(tris) == 0 or len(quads) == 0:
            break
        p = mesh.points
        ta = _corner_angles(p, tris)
        k = (np.argmax(ta, axis=1) + 1) % 3                            # longest edge is opposite the largest angle
        r = np.arange(len(tris))
        a, b, c = tris[r, k], tris[r, (k + 1) % 3], tris[r, (k + 2) % 3]
        qa, qb = quads.ravel(), np.roll(quads, -1, axis=1).ravel()
        qkey = qa * n + qb
        order = np.argsort(qkey)
        pos = np.minimum(np.searchsorted(qkey[order], b * n + a), len(qkey) - 1)
        hit = qkey[order][pos] == b * n + a
        qid, qk = order[pos] // 4, order[pos] % 4                       # quad half-edge b -> a at corner qk
        t_ok = np.flatnonzero(hit)
        q, j = qid[t_ok], qk[t_ok]
        x = quads[q, (j + 2) % 4]
        y = quads[q, (j + 3) % 4]
        P = np.column_stack([a[t_ok], x, y, b[t_ok], c[t_ok]])        # CCW pentagon
        cur = _badness(p, tris[t_ok], quads[q])
        best, arg = cur.copy(), np.full(len(t_ok), -1)
        for s in range(5):
            idx = (np.arange(5) + s) % 5
            tri_s, quad_s = P[:, idx[:3]], P[:, [idx[2], idx[3], idx[4], idx[0]]]
            v = _badness(p, tri_s, quad_s)
            better = v < best - 1e-9
            best, arg = np.where(better, v, best), np.where(better, s, arg)
        cand = np.flatnonzero(arg >= 0)
        cand = cand[np.argsort(-cur[cand], kind="stable")]
        used_q = np.zeros(len(quads), dtype=bool)
        done = 0
        for i in cand:                                                  # disjoint pentagons (few: leftover triangles)
            if used_q[q[i]]:
                continue
            used_q[q[i]] = True
            idx = (np.arange(5) + arg[i]) % 5
            tris[t_ok[i]] = P[i, idx[:3]]
            quads[q[i]] = P[i, [idx[2], idx[3], idx[4], idx[0]]]
            done += 1
        changed += done
        if done == 0:
            break
    return Mesh(mesh.points, tris, quads, mesh.node_flag), changed


def extract(domain: Domain, h: SizeField, V: np.ndarray, edges: np.ndarray, O: np.ndarray, rho: np.ndarray,
            diff: np.ndarray, err: np.ndarray, price: float = 1.0 / 3.0, snap: float = 0.5, merge: float = 0.35,
            spurious: float = 0.1) -> tuple[Mesh, dict]:
    """Mixed mesh from the background vertices V (unit), their edges, representative positions O, scales rho (unit
    sphere) and the integer jumps (diff, err) of the edges. Returns the mesh (coast nodes first, node_flag 1) and
    counts for the notes."""
    n = len(V)
    ad = np.abs(diff)
    col = ad.sum(axis=1) == 0
    unit = (ad.sum(axis=1) == 1)
    st = {"bg_nodes": n, "edges": len(edges), "collapse_edges": int(col.sum()), "unit_edges": int(unit.sum())}
    ce, ue = edges[col], edges[unit]
    order = np.argsort(err[col], kind="stable")
    roots, conflicts = _collapse(n, ce[:, 0], ce[:, 1], order, ue[:, 0], ue[:, 1])
    st["conflicts"] = int(conflicts)
    lab, nc = _relabel(roots)
    wgt = np.exp(-9.0 * np.sum((O - V) ** 2, axis=1) / rho**2)
    cnt = np.bincount(lab, minlength=nc)
    keep = cnt >= spurious * cnt.mean()
    st["clusters"], st["spurious"] = nc, int((~keep).sum())

    # merge clusters closer than merge * h (their mean h)
    X = _cluster_points(lab, nc, O, wgt + 1e-300)
    hx = h(X) / R_EARTH_M
    pairs = cKDTree(X).query_pairs(merge * hx.max(), output_type="ndarray")
    if len(pairs):
        d = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
        pairs = pairs[d < merge * 0.5 * (hx[pairs[:, 0]] + hx[pairs[:, 1]])]
        pairs = pairs[keep[pairs[:, 0]] & keep[pairs[:, 1]]]
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])) if len(pairs) else ([], ([], [])),
                   shape=(nc, nc))
    nc2, comp = connected_components(g, directed=False)
    st["merged"] = int(nc - nc2)
    lab = comp[lab]
    keep = np.bincount(comp, keep, minlength=nc2) > 0
    X = _cluster_points(lab, nc2, O, wgt + 1e-300)

    # coast: snap near clusters onto the resampled coast nodes, drop land
    node = np.full(nc2, -1, dtype=np.int64)
    if domain.loops:
        loops = resample_loops(domain, h)
        cpts = np.vstack(loops)
        ced = loop_edges(loops)
        B = len(cpts)
        nxt = np.empty(B, dtype=np.int64)
        prv = np.empty(B, dtype=np.int64)
        nxt[ced[:, 0]] = ced[:, 1]
        prv[ced[:, 1]] = ced[:, 0]
        _, c = cKDTree(cpts).query(X)
        dist = np.minimum(_arc_distance(X, cpts[prv[c]], cpts[c]), _arc_distance(X, cpts[c], cpts[nxt[c]]))
        snapped = keep & (dist < snap * h(X) / R_EARTH_M)
        node[snapped] = c[snapped]
        wet = in_water(Domain(loops, domain.name), X)
        inner = keep & ~snapped & wet
        st["snapped"], st["land"] = int(snapped.sum()), int((keep & ~snapped & ~wet).sum())
    else:
        cpts, ced, B = np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int64), 0
        inner = keep
    node[inner] = B + np.arange(int(inner.sum()))
    points = np.vstack([cpts, X[inner]])
    N = len(points)

    # lattice edges between output nodes
    a, b = node[lab[ue[:, 0]]], node[lab[ue[:, 1]]]
    ok = (a >= 0) & (b >= 0) & (a != b)
    lat = np.unique(np.minimum(a[ok], b[ok]) * N + np.maximum(a[ok], b[ok]))
    cky = np.minimum(ced[:, 0], ced[:, 1]) * N + np.maximum(ced[:, 0], ced[:, 1])
    lat = np.union1d(lat, cky)

    tris = sphere_delaunay(points, ced if B else None)
    ta, tb = tris, np.roll(tris, -1, axis=1)
    key = np.minimum(ta, tb) * N + np.maximum(ta, tb)
    is_lat = np.isin(key, lat)
    st["lattice_edges"] = len(lat)
    st["lattice_in_tri"] = int(np.isin(lat, key).sum())
    nbr = adjacency(tris)
    nonlat = ~is_lat & (nbr >= 0)
    cnt1 = nonlat.sum(axis=1) == 1
    t = np.flatnonzero(cnt1)
    k = np.argmax(nonlat[t], axis=1)
    u = nbr[t, k]
    pick = cnt1[u] & (t < u)
    t, k, u = t[pick], k[pick], u[pick]
    good = quad_cost(points, tris[t], tris[u]) < 2.0 * price
    t, k, u = t[good], k[good], u[good]
    quads = _merge_pairs(tris, t, k, u)
    st["lattice_quads"] = len(quads)
    rest = np.ones(len(tris), dtype=bool)
    rest[t] = False
    rest[u] = False
    flag = np.zeros(N, dtype=np.int8)
    flag[:B] = 1
    mesh = recombine(Mesh(points, tris[rest], quads, flag), price)
    st["recombined_quads"] = mesh.n_quads - len(quads)
    mesh, st["resplit"] = resplit_pentagons(mesh)

    # drop nodes in no element (clusters that fell into removed land triangles)
    used = np.zeros(N, dtype=bool)
    used[mesh.tris.ravel()] = True
    used[mesh.quads.ravel()] = True
    used[:B] = True
    st["orphans"] = int((~used).sum())
    if not used.all():
        new = np.cumsum(used) - 1
        mesh = Mesh(points[used], new[mesh.tris], new[mesh.quads], flag[used])
    st["nodes"], st["tris"], st["quads"] = mesh.n_nodes, mesh.n_tris, mesh.n_quads
    return mesh, st
