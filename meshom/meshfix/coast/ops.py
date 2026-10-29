"""Half-chord operators that carry a triangle to the coast along a quad row (decisions B145).

A triangle (4,4,5) is a column end: the column of quads on the far side of its base ends in it. Two ways to move it
to the coast, both along one quad row (a corridor one element wide), both leaving every interior node of the row as
regular as it was:

- collapse (base toward the coast): the strip of quads from the triangle's base (p0, q0) to a coast edge (pk, qk)
  is removed with the triangle and every rung (pi, qi) merges into one node: interior merged nodes keep 4 quads
  when pi and qi had 4, the tip loses the triangle (5 -> 4 elements), the coast loses one edge (pk, qk merge).
- split (tip toward the coast): the straight grid line from the tip t = v0 to a coast node vk is cut open: each vi
  is duplicated, the elements on the right of the line take the copy, a new quad fills each segment, the triangle
  becomes a quad (p, q, t', t); the tip goes from 5 to 4 elements, the line's nodes keep 4, the coast gains one
  edge (vk splits into two coast nodes).

The operators only build the replacement (elements removed, new nodes, new cells, new coast nodes, moved nodes);
the engine relaxes it with a collar and accepts or rejects (engine.py). Positions: merged nodes at the midpoint
(coast: projected onto the reference coast), split copies offset by a quarter edge to each side of the line.
"""
import numpy as np


def fan(W, v: int) -> list[tuple[int, int, int]]:
    """(element, next node, prev node) for the elements at v: the element spans CCW from edge (v, next) to edge
    (v, prev)."""
    out = []
    for e in W.at(v):
        c = W.cells(e)
        i = c.index(v)
        out.append((e, c[(i + 1) % len(c)], c[i - 1]))
    return out


def ccw_from(W, v: int, start: int, stop: int | None) -> list[int] | None:
    """Elements at v going CCW from edge (v, start) until edge (v, stop) (or the boundary when stop is None);
    None when the walk does not close as asked."""
    f = {nx: (e, pv) for e, nx, pv in fan(W, v)}
    out, cur = [], start
    for _ in range(len(f) + 1):
        if stop is not None and cur == stop:
            return out
        if cur not in f:
            return out if stop is None else None
        e, pv = f[cur]
        out.append(e)
        cur = pv
    return None


def _check_cells(cells) -> bool:
    he = set()
    for c in cells:
        if len(set(c)) != len(c):
            return False
        for i in range(len(c)):
            k = (c[i], c[(i + 1) % len(c)])
            if k in he:
                return False
            he.add(k)
    return True


def collapse(W, t_elem: int, base: int, is_coast_edge, max_len: int, pair: bool = True, reads: set | None = None):
    """Collapse the strip from the element's edge `base` (edge c[base] -> c[base+1]) to the coast, or (pair=True) to
    a second triangle whose edge closes the strip (the two column ends cancel). t_elem a triangle: it goes with the
    strip. t_elem a quad whose edge joins a 3-5 pair: the quad is kept and becomes a (4,4,5) triangle (the pair's
    merged node is its tip with 4 quads), i.e. the pair turns into one column end and the coast takes the other half.
    Returns (rm, X, cells, coast_new, moves, core) or None. reads (optional): every node whose star or flags the
    walk looked at is added (the read set of the fast engine, B148: the engine adds the nodes of their stars)."""
    reads = set() if reads is None else reads
    c = W.cells(t_elem)
    reads.update(c)
    n = len(c)
    p, q = c[base], c[(base + 1) % n]
    if n == 3:
        tips = [c[(base + 2) % 3]]
        strip = [t_elem]
    else:                                                 # a quad at a 3-5 pair edge: kept, it becomes a triangle
        tips = []
        strip = []
    rungs = [(p, q)]
    prev = t_elem
    end = None
    for _ in range(max_len):
        a, b = rungs[-1]                                  # a -> b is the edge in the previous element's order
        reads.update((a, b))
        if is_coast_edge(a, b):
            end = "coast"
            break
        if W.bnode[a] or W.bnode[b]:
            return None
        nb = [e for e in W.edge_elems(a, b) if e != prev]
        if len(nb) != 1:
            return None
        if W.nv[nb[0]] == 3:
            if not pair:
                return None
            cc = W.cells(nb[0])
            reads.update(cc)
            tips.append([v for v in cc if v not in (a, b)][0])
            strip.append(nb[0])
            end = "pair"
            break
        if W.nv[nb[0]] != 4:
            return None
        e = nb[0]
        cc = W.cells(e)
        reads.update(cc)
        i = cc.index(b)
        if cc[(i + 1) % 4] != a:
            return None
        a1, b1 = cc[(i + 2) % 4], cc[(i + 3) % 4]      # cc = (b, a, a1, b1): a1 next to a, b1 next to b
        rungs.append((a1, b1))
        strip.append(e)
        prev = e
    if end is None:
        return None
    nodes = [v for r in rungs for v in r]
    if len(set(nodes)) != len(nodes) or any(t in nodes for t in tips) or len(set(tips)) != len(tips) or \
            any(W.art[v] for v in nodes):
        return None
    ref = {}
    for i, (a, b) in enumerate(rungs):
        ref[a] = -1 - i
        ref[b] = -1 - i
    rm = sorted({e for v in nodes for e in W.at(v)} | set(strip) | {t_elem})
    S = set(strip)
    cells = []
    for e in rm:
        if e in S:
            continue
        cc = [ref.get(v, v) for v in W.cells(e)]
        cc = tuple(v for i, v in enumerate(cc) if v != cc[i - 1])   # the kept quad loses one corner
        if len(cc) < 3 or (len(cc) < W.nv[e] and e != t_elem):
            return None
        cells.append(cc)
    if not _check_cells(cells):
        return None
    X = np.array([W.P[a] + W.P[b] for a, b in rungs])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    k = len(rungs) - 1
    return rm, X, cells, ([-1 - k] if end == "coast" else []), {}, tips + nodes


def split(W, t_elem: int, tip_i: int, is_coast_edge, max_len: int, pair: bool = True, reads: set | None = None):
    """Split the straight grid line from the triangle's corner tip_i to the coast, or (pair=True) to the tip of a
    second triangle that the line meets from the side opposite it (both column ends cancel). Returns (rm, X, cells,
    coast_new, moves, core) or None. reads: as collapse."""
    reads = set() if reads is None else reads
    c = W.cells(t_elem)
    reads.update(c)
    if len(c) != 3:
        return None
    t = c[tip_i]
    p, q = c[(tip_i + 1) % 3], c[(tip_i + 2) % 3]        # T = (t, p, q) CCW: base p -> q, tip t
    if W.bnode[t] or W.art[t]:
        return None
    # at the tip the forward edge leaves the fan of the quads opposite the triangle: the triangle spans (t,p)->(t,q)
    # CCW, so the quads go CCW from (t, q) back to (t, p); the straight line is the middle edge of that arc
    arc = ccw_from(W, t, q, p)
    if arc is None or len(arc) % 2 or any(W.nv[e] != 4 for e in arc):
        return None
    f = {e: (nx, pv) for e, nx, pv in fan(W, t)}
    v1 = f[arc[len(arc) // 2 - 1]][1]                    # the prev edge of the middle-left quad
    path = [t, v1]
    end, T2 = None, None
    for _ in range(max_len):
        v, u = path[-1], path[-2]
        reads.add(v)
        if W.bnode[v]:
            end = "coast"
            break
        fv = fan(W, v)
        tri = [e for e, _, _ in fv if W.nv[e] == 3]
        if pair and len(tri) == 1 and len(fv) == 5 and tri[0] != t_elem:
            # the tip of a second triangle met from the side opposite it: both column ends cancel
            c2 = W.cells(tri[0])
            reads.update(c2)
            j = c2.index(v)
            p2, q2 = c2[(j + 1) % 3], c2[(j + 2) % 3]
            arc2 = ccw_from(W, v, q2, p2)
            f2 = {e: (nx, pv) for e, nx, pv in fv}
            if arc2 is not None and len(arc2) == 4 and f2[arc2[1]][1] == u:
                end, T2 = "pair", (tri[0], p2, q2)
                break
            return None
        if len(fv) != 4 or any(W.nv[e] != 4 for e, _, _ in fv):
            return None
        # straight on at a regular node: two quads CCW from (v, u)
        g = {e: pv for e, nx, pv in fv}
        nxt = {nx: e for e, nx, pv in fv}
        e1 = nxt[u]
        w = g[e1]
        e2 = nxt[w]
        path.append(g[e2])
    if end is None:
        return None
    vk = path[-1]
    if W.art[vk] or len(set(path)) != len(path) or p in path or q in path:
        return None
    if T2 is not None and (T2[1] in path or T2[2] in path):
        return None
    # the elements on the right of the path take the copies
    right = {}
    # tip: CCW from the forward edge (t, v1) to (t, q)... the right side is adjacent to q
    r0 = ccw_from(W, t, q, v1)
    if r0 is None:
        return None
    right[t] = [e for e in r0 if e != t_elem]
    for i in range(1, len(path) - 1):
        r = ccw_from(W, path[i], path[i - 1], path[i + 1])
        if r is None:
            return None
        right[path[i]] = r
    if end == "coast":
        rk = ccw_from(W, vk, path[-2], None)
    else:                                                 # the second tip: CCW from the backward edge to T2's start
        rk = ccw_from(W, vk, path[-2], T2[1])
    if rk is None:
        return None
    right[vk] = rk
    ref = {v: -1 - i for i, v in enumerate(path)}
    tset = {t_elem} | ({T2[0]} if T2 else set())
    rm = sorted({e for v in path for e in right[v]} | tset)
    cells = []
    for e in rm:
        if e in tset:
            continue
        cc = W.cells(e)
        cells.append(tuple(ref[v] if (v in ref and e in right[v]) else v for v in cc))
    cells.append((t, p, q, ref[t]))                       # T = (t, p, q) -> quad (t, p, q, t')
    if T2:
        cells.append((ref[vk], T2[1], T2[2], vk))         # T2 = (vk, p2, q2) -> quad (vk', p2, q2, vk)
    for i in range(len(path) - 1):
        cells.append((ref[path[i]], ref[path[i + 1]], path[i + 1], path[i]))
    # orientation: the new quads must be CCW; with right = CCW from backward to forward edge the right copies are
    # on the clockwise side of the path, so (v_i', v_i+1', v_i+1, v_i) is CCW
    if not _check_cells(cells):
        return None
    # positions: offsets of a quarter of the local edge to each side (tangent plane), coast copies along the coast
    P = W.P
    X, moves = [], {}
    for i, v in enumerate(path):
        x = P[v]
        a = P[path[min(i + 1, len(path) - 1)]] - P[path[max(i - 1, 0)]]
        a -= (a @ x) * x
        n = np.cross(x, a)
        n /= max(np.linalg.norm(n), 1e-300)
        L = np.linalg.norm(P[path[1]] - P[path[0]]) if i == 0 else np.linalg.norm(P[v] - P[path[i - 1]])
        d = 0.25 * L * n                                  # n points left of the path
        xl, xr = x + d, x - d
        X.append(xr / np.linalg.norm(xr))
        moves[v] = xl / np.linalg.norm(xl)
    return rm, np.array(X), cells, ([-1 - (len(path) - 1)] if end == "coast" else []), moves, list(path) + [p, q]
