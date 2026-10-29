"""QuadriFlow's removal of position singularities (Huang, Zhou, Nießner, Shewchuk, Guibas 2018, §3.2-3.3) as a
minimum-cost flow on the integer jumps of the Instant Meshes position field, with an optional price for keeping one.

Integer data (m07 fields.edge_jumps / edge_frames): per background edge e = (i, j), i < j, the jump D_e in Z^2 = the
offset from i's lattice point to j's in i's own frame (q_i, n_i x q_i), and r_e in Z_4 = the quarter turns that map
j-frame coordinates to i-frame coordinates (paper Def. 3.1: d = t_uv - R(k_uv - k_vu) t_vu).

Faces: the sum of the jumps around a background triangle, each turned into the frame of the triangle's first vertex,
is zero unless the triangle holds a position singularity (paper eq. 3). Triangles whose frame turns round by a
quarter (holonomy != 0) hold an orientation singularity (cone); they keep their equation (as QuadriFlow's code does),
the flow decides what happens there.

Flow (paper §3.3): every scalar component of a jump appears in two face equations (one per side of its edge). After
each face equation is turned by a gauge g_f in Z_4, a component is "balanced" when it has opposite signs in its two
equations; it is then an arc of the network between the two equation nodes (2 per face: the x and y rows). The gauge
comes from a union-find with Z_4 offsets over the dual edges in decreasing distance from the cones (a maximum spanning
tree on the dual graph), so the components that cannot be balanced lie on short seams between cones (QuadriFlow:
union-find in edge order, singular faces last). Unbalanced components are fixed (paper). The coast: the component of a
coast edge's jump along the coast is an arc to the ground node G (a row of the lattice may end on the coast, so the
coast absorbs Burgers vectors along it); the other component is fixed. The excess of an equation node is its current
residual; the flow moves excess along arcs (one unit on an arc = one jump component changed by one) until every node
is zero, at minimum total change (paper eq. 7, L1 cost, unit weight per component: the background edges are ~factor
x h long, so the cost is a distance in units of the local h). Capacity: |d| <= cap.

Optional price `keep` (meshom variant): every equation node also has an arc to G at cost keep / 2, so a
position singularity is left where it is when the cheapest partner (an opposite singularity, the coast) is more than
`keep` away; keep = inf is QuadriFlow (all removed where the network allows it).

Solver: successive shortest paths (primal-dual) with node potentials: one unit at a time, Dijkstra from a node with
excess to the nearest node with deficit (or G) in the residual network with reduced costs, then the potentials of the
settled nodes are lowered by (D - dist) - the exact min-cost flow (no negative cycle remains), and every search is
local (it stops at the first deficit). Then the remaining deficits, searched backwards (towards G, or a reversed keep).
Not LEMON's network simplex nor QuadriFlow's Boykov-Kolmogorov max flow (which drops the cost at fine levels).
"""
import numpy as np
from numba import njit

INF = np.inf


# ------------------------------------------------------------------------------------------------ integer data

def rot90(v: np.ndarray, k: np.ndarray) -> np.ndarray:
    """(M,2) integer vectors turned by k (M,) quarter turns counter-clockwise: (x, y) -> (-y, x) per turn."""
    k = np.asarray(k) % 4
    x, y = v[:, 0], v[:, 1]
    out = np.empty_like(v)
    out[:, 0] = np.select([k == 0, k == 1, k == 2], [x, -y, -x], y)
    out[:, 1] = np.select([k == 0, k == 1, k == 2], [y, x, -y], -x)
    return out


def own_frame(diff: np.ndarray, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """m07's jump diff (in q_i turned by ki) -> (D in i's own frame, r = (ki - kj) mod 4: j-frame -> i-frame)."""
    return rot90(diff, frames[:, 0]), (frames[:, 0] - frames[:, 1]) % 4


def face_system(tris: np.ndarray, edges: np.ndarray, n: int, r: np.ndarray) -> tuple:
    """Per face side s (vs -> vs+1): edge id, coefficient turn o in Z_4 (sign included as 2 quarter turns) in the
    frame of the face's first vertex; and the holonomy of each face (0 = no cone)."""
    key = edges[:, 0] * n + edges[:, 1]

    def eid(a, b):
        return np.searchsorted(key, np.minimum(a, b) * n + np.maximum(a, b))

    def turn_to(a, b):                                   # quarter turns b-frame -> a-frame across edge (a, b)
        e = eid(a, b)
        return np.where(a < b, r[e], -r[e]) % 4

    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    hol = (turn_to(v0, v1) + turn_to(v1, v2) + turn_to(v2, v0)) % 4
    se = np.empty_like(tris)
    so = np.empty_like(tris)
    for s in range(3):
        a, b = tris[:, s], tris[:, (s + 1) % 3]
        m = np.minimum(a, b)
        se[:, s] = eid(a, b)
        safe = np.where(m == v0, v1, m)                  # turn_to(v0, v0) is not an edge; masked below
        so[:, s] = (np.where(m == v0, 0, turn_to(v0, safe)) + np.where(a < b, 0, 2)) % 4
    return se, so, hol


def residuals(se: np.ndarray, so: np.ndarray, D: np.ndarray) -> np.ndarray:
    """(T,2) sum of the jumps around each face in its first vertex's frame."""
    res = np.zeros((len(se), 2), dtype=np.int64)
    for s in range(3):
        res += rot90(D[se[:, s]], so[:, s])
    return res


# ------------------------------------------------------------------------------------------------ gauge

@njit(cache=True)
def _find(par, off, f):
    """Root of f and the offset g_f - g_root (mod 4), with path compression."""
    r = f
    acc = 0
    while par[r] != r:
        acc += off[r]
        r = par[r]
    # compress
    cur, cacc = f, acc
    while par[cur] != cur:
        nxt = par[cur]
        o = off[cur]
        par[cur] = r
        off[cur] = cacc % 4
        cacc -= o
        cur = nxt
    return r, acc % 4


@njit(cache=True)
def _gauge(nf, order, f0, s0, f1, s1, so):
    """Union-find with Z_4 offsets: dual edge k (faces f0, f1 at sides s0, s1) wants g1 - g0 = o0 - o1 - 2."""
    par = np.arange(nf)
    off = np.zeros(nf, dtype=np.int64)
    for k in order:
        a, b = f0[k], f1[k]
        need = (so[a, s0[k]] - so[b, s1[k]] - 2) % 4
        ra, oa = _find(par, off, a)
        rb, ob = _find(par, off, b)
        if ra == rb:
            continue
        par[rb] = ra                                     # g_rb - g_ra = need - ob + oa
        off[rb] = (need - ob + oa) % 4
    g = np.empty(nf, dtype=np.int64)
    for f in range(nf):
        g[f] = _find(par, off, f)[1]
    return g


@njit(cache=True)
def _face_bfs(nbr, seeds):
    nf = len(nbr)
    d = np.full(nf, -1, dtype=np.int64)
    q = np.empty(nf, dtype=np.int64)
    h, t = 0, 0
    for s in seeds:
        d[s] = 0
        q[t] = s
        t += 1
    while h < t:
        f = q[h]
        h += 1
        for k in range(3):
            g = nbr[f, k]
            if g >= 0 and d[g] < 0:
                d[g] = d[f] + 1
                q[t] = g
                t += 1
    return d


# ------------------------------------------------------------------------------------------------ solver

@njit(cache=True)
def _push(hk, hv, n, key, val):
    i = n
    hk[i] = key
    hv[i] = val
    while i > 0:
        p = (i - 1) >> 1
        if hk[p] <= hk[i]:
            break
        hk[p], hk[i] = hk[i], hk[p]
        hv[p], hv[i] = hv[i], hv[p]
        i = p
    return n + 1


@njit(cache=True)
def _pop(hk, hv, n):
    key, val = hk[0], hv[0]
    n -= 1
    hk[0], hv[0] = hk[n], hv[n]
    i = 0
    while True:
        a = 2 * i + 1
        if a >= n:
            break
        b = a + 1
        c = b if (b < n and hk[b] < hk[a]) else a
        if hk[i] <= hk[c]:
            break
        hk[c], hk[i] = hk[i], hk[c]
        hv[c], hv[i] = hv[i], hv[c]
        i = c
    return key, val, n


@njit(cache=True, inline="always")
def _cost(x, d0, u, a, eps, dd):
    """Marginal cost of moving x by dd (+-1): a (x - u)^2 + eps |x - d0| (u = continuous offset, d0 = start)."""
    if dd > 0:
        return a * (2.0 * (x - u) + 1.0) + (eps if x >= d0 else -eps)
    return a * (1.0 - 2.0 * (x - u)) + (eps if x <= d0 else -eps)


@njit(cache=True)
def _ssp(G, ptr, adj, vm, vp, x, d0, uc, lb, ub, a, eps, excess, keep, max_settle, ground):
    """Successive shortest paths with potentials (see module doc). Nodes 0..G-1 are equation rows, G is the ground
    (free source / sink, never expanded). Variable k is an arc vm[k] -> vp[k] raising x[k] by one (and back, lowering
    it). keep = cost of an equation node <-> G arc (inf: none). Returns (augmentations, failed searches, total cost)."""
    nn = G + 1
    pi = np.zeros(nn)
    dist = np.zeros(nn)
    mark = np.full(nn, -1, dtype=np.int32)
    done = np.full(nn, -1, dtype=np.int32)
    pk = np.empty(nn, dtype=np.int32)          # arc into / out of the node on the search tree (-1: keep arc)
    pd = np.empty(nn, dtype=np.int8)
    pn = np.empty(nn, dtype=np.int32)
    kflow = np.zeros(nn, dtype=np.int32)       # net units node -> G on the keep arcs
    cap = 1 << 16
    hk = np.empty(cap)
    hv = np.empty(cap, dtype=np.int64)
    settled = np.empty(max_settle + 2, dtype=np.int64)
    dead = np.zeros(nn, dtype=np.bool_)        # settled by a search that found no target: no target reachable
    stamp = 0
    n_aug, n_fail, total, n_settle = 0, 0, 0.0, 0
    has_keep = keep < np.inf
    for phase in range(2):                     # 0: excess -> deficit / G, 1: G -> remaining deficits (backwards)
        if phase == 1 and not ground:
            break
        dead[:] = False
        for s in range(G):
            while (excess[s] > 0) if phase == 0 else (excess[s] < 0):
                if dead[s]:
                    n_fail += 1
                    break
                stamp += 1
                hn = 0
                ns = 0
                dist[s] = 0.0
                mark[s] = stamp
                hn = _push(hk, hv, hn, 0.0, s)
                tgt = -1
                D = 0.0
                while hn > 0:
                    d, u, hn = _pop(hk, hv, hn)
                    if done[u] == stamp or d > dist[u]:
                        continue
                    done[u] = stamp
                    settled[ns] = u
                    ns += 1
                    n_settle += 1
                    if u == G or (u != s and ((excess[u] < 0) if phase == 0 else (excess[u] > 0))):
                        tgt, D = u, d
                        break
                    if ns > max_settle:
                        break
                    for ia in range(ptr[u], ptr[u + 1]):
                        k = adj[ia]
                        if phase == 0:                  # arcs out of u
                            if vm[k] == u:
                                v, dd = vp[k], 1
                            else:
                                v, dd = vm[k], -1
                            if (dd > 0 and x[k] >= ub[k]) or (dd < 0 and x[k] <= lb[k]):
                                continue
                            c = _cost(x[k], d0[k], uc[k], a, eps, dd) + pi[u] - pi[v]
                        else:                           # arcs into u (searched backwards)
                            if vp[k] == u:
                                v, dd = vm[k], 1
                            else:
                                v, dd = vp[k], -1
                            if (dd > 0 and x[k] >= ub[k]) or (dd < 0 and x[k] <= lb[k]):
                                continue
                            c = _cost(x[k], d0[k], uc[k], a, eps, dd) + pi[v] - pi[u]
                        nd = d + c
                        if done[v] != stamp and (mark[v] != stamp or nd < dist[v]):
                            dist[v] = nd
                            mark[v] = stamp
                            pk[v], pd[v], pn[v] = k, dd, u
                            if hn == len(hk):
                                hk2 = np.empty(2 * len(hk))
                                hv2 = np.empty(2 * len(hk), dtype=np.int64)
                                hk2[:hn] = hk
                                hv2[:hn] = hv
                                hk, hv = hk2, hv2
                            hn = _push(hk, hv, hn, nd, v)
                    if has_keep:
                        if phase == 0:
                            c = (keep if kflow[u] >= 0 else -keep) + pi[u] - pi[G]
                        else:
                            c = (keep if kflow[u] <= 0 else -keep) + pi[G] - pi[u]
                        nd = d + c
                        if done[G] != stamp and (mark[G] != stamp or nd < dist[G]):
                            dist[G] = nd
                            mark[G] = stamp
                            pk[G], pd[G], pn[G] = -1, 0, u
                            if hn == len(hk):
                                hk2 = np.empty(2 * len(hk))
                                hv2 = np.empty(2 * len(hk), dtype=np.int64)
                                hk2[:hn] = hk
                                hv2[:hn] = hv
                                hk, hv = hk2, hv2
                            hn = _push(hk, hv, hn, nd, G)
                if tgt < 0:
                    n_fail += 1
                    if hn == 0:                         # exhausted (not capped): nothing reachable has a target
                        for m in range(ns):
                            dead[settled[m]] = True
                    break                               # this node keeps its excess
                v = tgt
                while v != s:                            # augment one unit along the tree path
                    k = pk[v]
                    if k >= 0:
                        x[k] += pd[v]
                        total += _cost(x[k] - pd[v], d0[k], uc[k], a, eps, pd[v])
                    else:
                        u = pn[v] if v == G else v
                        sgn = 1 if phase == 0 else -1
                        total += (keep if kflow[u] * sgn >= 0 else -keep)
                        kflow[u] += sgn
                    v = pn[v]
                if phase == 0:
                    excess[s] -= 1
                    if tgt != G:
                        excess[tgt] += 1
                else:
                    excess[s] += 1
                    if tgt != G:
                        excess[tgt] -= 1
                for m in range(ns):                      # potentials (lazy: only settled nodes move)
                    u = settled[m]
                    if phase == 0:
                        pi[u] -= D - dist[u]
                    else:
                        pi[u] += D - dist[u]
                n_aug += 1
    return n_aug, n_fail, total, n_settle


# ------------------------------------------------------------------------------------------------ driver

def remove_position_singularities(tris: np.ndarray, edges: np.ndarray, n: int, D: np.ndarray, r: np.ndarray,
                                  V: np.ndarray, Q: np.ndarray, cap: int = 1, keep: float = np.inf,
                                  coast: bool = True, max_settle: int = 400_000, cost: str = "l2",
                                  eps: float = 0.25, U: np.ndarray | None = None,
                                  seed: int = 0) -> tuple[np.ndarray, dict]:
    """New jumps D (E,2) with the position singularities removed (or kept at price `keep`), and stats.
    cost "l1": |d - d*| (QuadriFlow); "l2": (d - u)^2 + eps |d - d*| with u = U (E,2), the continuous offset
    of the position field in lattice units (continuous_offsets), clipped to d* +- 1/2."""
    from meshom.delaunay import adjacency

    se, so, hol = face_system(tris, edges, n, r)
    nf = len(tris)
    res0 = residuals(se, so, D)
    sing = hol != 0
    st = {"faces": nf, "cones": int(sing.sum()), "cone_index": int(np.where(hol == 1, 1, 0).sum()
                                                                   - np.where(hol == 3, 1, 0).sum()),
          "pos_sing_before": int(((res0 != 0).any(1) & ~sing).sum())}

    # dual edges: the two face sides of every interior edge
    fs = np.repeat(np.arange(nf), 3)
    ss = np.tile(np.arange(3), nf)
    ee = se.ravel()
    order = np.argsort(ee, kind="stable")
    ee, fs, ss = ee[order], fs[order], ss[order]
    first = np.r_[True, ee[1:] != ee[:-1]]
    cnt = np.bincount(ee, minlength=len(edges))
    start = np.flatnonzero(first)
    two = cnt[ee[start]] == 2
    a0 = start[two]
    df0, ds0, df1, ds1 = fs[a0], ss[a0], fs[a0 + 1], ss[a0 + 1]
    de = ee[a0]
    # gauge: maximum spanning tree of the dual graph, weight = hop distance from the cones
    nbr = adjacency(tris)
    seeds = np.flatnonzero(sing)
    if len(seeds):
        fd = _face_bfs(nbr, seeds)
    else:
        fd = np.zeros(nf, dtype=np.int64)
    wdual = np.minimum(fd[df0], fd[df1])
    gorder = np.argsort(-wdual, kind="stable")
    g = _gauge(nf, gorder, df0, ds0, df1, ds1, so)
    og = (so + g[:, None]) % 4

    # scalar variables (e, c): row = (og + c) % 2 of face f, sign + when (og + c) % 4 < 2
    G = 2 * nf
    m = len(edges)
    vm = np.full(2 * m, -1, dtype=np.int32)
    vp = np.full(2 * m, -1, dtype=np.int32)
    excess = np.zeros(G + 1, dtype=np.int64)
    x = D.ravel().astype(np.int16)
    for c in range(2):
        kk = (og + c) % 4
        node = 2 * np.arange(nf)[:, None] + kk % 2
        sgn = np.where(kk < 2, 1, -1)
        np.add.at(excess, node.ravel(), sgn.ravel() * x[2 * se.ravel() + c])
    # balanced components of interior edges: opposite signs in the two faces
    unb = []
    for c in range(2):
        k0 = (og[df0, ds0] + c) % 4
        k1 = (og[df1, ds1] + c) % 4
        n0, n1 = 2 * df0 + k0 % 2, 2 * df1 + k1 % 2
        p0 = k0 < 2
        bal = p0 != (k1 < 2)
        var = 2 * de[bal] + c
        vp[var] = np.where(p0[bal], n0[bal], n1[bal])
        vm[var] = np.where(p0[bal], n1[bal], n0[bal])
        unb.append(np.column_stack([2 * de[~bal] + c, n0[~bal], n1[~bal], np.where(p0[~bal], 1, -1)]))
    st["unbalanced"] = int(2 * len(de) - (vm[2 * de] >= 0).sum() - (vm[2 * de + 1] >= 0).sum())
    # coast edges: the component along the coast is an arc to G
    if coast:
        b1 = start[~two]
        bf, bs, be = fs[b1], ss[b1], ee[b1]
        i, j = edges[be, 0], edges[be, 1]
        t = V[j] - V[i]
        qi = Q[i]
        ti = np.cross(V[i], qi)
        c = (np.abs(np.einsum("ij,ij->i", t, ti)) > np.abs(np.einsum("ij,ij->i", t, qi))).astype(np.int64)
        kk = (og[bf, bs] + c) % 4
        node = 2 * bf + kk % 2
        pos = kk < 2
        var = 2 * be + c
        vp[var] = np.where(pos, node, G)
        vm[var] = np.where(pos, G, node)
        st["coast_arcs"] = len(var)
    del fs, ss, ee, order, first, cnt, start, df0, ds0, df1, ds1, de, nbr, wdual, gorder
    free = vm >= 0
    st["arcs"] = int(free.sum())
    # CSR adjacency over equation rows (G is never expanded)
    kv = np.flatnonzero(free)
    ends = np.r_[vm[kv], vp[kv]]
    kk = np.r_[kv, kv]
    keepm = ends < G
    ends, kk = ends[keepm], kk[keepm]
    o = np.argsort(ends, kind="stable")
    adj = kk[o].astype(np.int32)
    ptr = np.zeros(G + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(ends, minlength=G))
    del kv, ends, kk, keepm, o
    d0 = x.copy()
    lb = np.minimum(-cap, d0).astype(np.int16)
    ub = np.maximum(cap, d0).astype(np.int16)
    st["excess_units"] = int(np.abs(excess[:G]).sum())
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    fr = np.flatnonzero(free)
    ncomp, lab = connected_components(coo_matrix((np.ones(len(fr)), (vm[fr], vp[fr])), shape=(G + 1, G + 1)),
                                      directed=False)
    tot = np.bincount(lab[:G], excess[:G], minlength=ncomp)
    if (vm[fr] == G).any() or (vp[fr] == G).any():
        tot[lab[G]] = 0                                  # the coast (ground) balances its component
    st["components"], st["imbalance"] = int(ncomp), int(np.abs(tot).sum())
    if keep == np.inf and (tot != 0).any():
        # QuadriFlow (BuildIntegerConstraints): a component of the network can only reach full flow when its
        # excess sums to zero; change fixed (unbalanced) components, which enter both their rows with the same sign,
        # in random order until it does
        u = np.vstack(unb)
        u = u[np.random.default_rng(seed).permutation(len(u))]
        for var, a0, a1, sg in u:
            ca, cb = lab[a0], lab[a1]
            ta, tb = tot[ca], tot[cb]
            if ta == 0 or (ca != cb and np.sign(tb) != np.sign(ta)) or (ca == cb and abs(ta) < 2):
                continue
            dd = -int(np.sign(ta)) * sg
            if abs(int(x[var]) + dd) > cap:
                continue
            x[var] += dd
            excess[a0] += sg * dd
            excess[a1] += sg * dd
            tot[ca] += sg * dd
            tot[cb] += sg * dd
            if not tot.any():
                break
        st["imbalance_left"] = int(np.abs(tot).sum())
    vm[~free] = 0
    vp[~free] = 0
    del free
    ground = keep < np.inf or (coast and st.get("coast_arcs", 0) > 0)
    if cost == "l2":
        uc = np.clip(U.ravel(), d0 - 0.5, d0 + 0.5).astype(np.float32)
        a = 1.0
    else:
        uc, a, eps = d0.astype(np.float32), 0.0, 1.0
    n_aug, n_fail, total, n_settle = _ssp(G, ptr, adj, vm, vp, x, d0, uc, lb, ub, a, eps, excess, keep, max_settle,
                                ground)
    Dn = x.reshape(-1, 2).astype(np.int64)
    res1 = residuals(se, so, Dn)
    det0, det1 = inverted(se, so, D), inverted(se, so, Dn)
    st.update(inverted_before=int((det0 < 0).sum()), inverted_after=int((det1 < 0).sum()))
    st.update(settled=int(n_settle), augment=int(n_aug), failed=int(n_fail), cost=float(total),
              changed=int((Dn != D).any(1).sum()),
              pos_sing_after=int(((res1 != 0).any(1) & ~sing).sum()),
              cone_faces_res=int(((res1 != 0).any(1) & sing).sum()))
    return Dn, st


def inverted(se: np.ndarray, so: np.ndarray, D: np.ndarray) -> np.ndarray:
    """(T,) det of the first two side jumps of each face in its first vertex's frame (paper eq. 4: < 0 = inverted)."""
    a = rot90(D[se[:, 0]], so[:, 0])
    b = -rot90(D[se[:, 2]], so[:, 2])                    # side 2 runs v2 -> v0; v0 -> v2 is its negative
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def continuous_offsets(V: np.ndarray, edges: np.ndarray, Q: np.ndarray, O: np.ndarray, rho: np.ndarray,
                       D: np.ndarray) -> np.ndarray:
    """(E,2) continuous offset from i to j in i's frame, lattice units: D + ((v_j - o_j) - (v_i - o_i)) / rho_ij
    projected on (q_i, n_i x q_i) - the integer jump plus the change of the fractional lattice position."""
    i, j = edges[:, 0], edges[:, 1]
    q = Q[i] / np.linalg.norm(Q[i], axis=1, keepdims=True)
    t = np.cross(V[i], q)
    d = ((V[j] - O[j]) - (V[i] - O[i])) / (0.5 * (rho[i] + rho[j]))[:, None]
    return D + np.column_stack([np.einsum("ij,ij->i", d, q), np.einsum("ij,ij->i", d, t)])
