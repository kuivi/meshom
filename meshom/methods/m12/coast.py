"""m12 coast step: cut the moved closed-sphere quad mesh to the water and fit its outer row to the resampled coast.

Steps: keep cells (strip: every node in water and >= gap*h from the coast, no coast vertex inside; snap: centroid in
water) -> clean (drop cells with >= 3 boundary edges and cells at pinch nodes, keep the largest edge-connected
piece) -> staircase loops (water on the left) -> closest-point projection of the staircase nodes onto the resampled
coast, made monotone per run on one coast loop -> either
- `strip` (default): one row of elements between the staircase and the coast, built by a merge walk along both
  polylines (quad where both advance, triangle where one advances); every resampled coast node is kept; or
- `snap` (the literal variant): staircase nodes moved onto their coast images; resampled coast nodes are not kept;
-> guarded Laplacian sweeps on the rows next to the coast (coast nodes fixed).
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from meshom.domain import Domain, in_water, resample_loops
from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh
from meshom.size import SizeField


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------------------------------- projection

class Coast:
    """Resampled coast loops as one polyline set: closest-point projection and points at a loop parameter."""

    def __init__(self, loops: list[np.ndarray]):
        self.loops = loops
        self.off = np.r_[0, np.cumsum([len(lp) for lp in loops])].astype(np.int64)
        self.a = np.vstack(loops)
        self.b = np.vstack([np.roll(lp, -1, axis=0) for lp in loops])
        self.seg_loop = np.repeat(np.arange(len(loops)), [len(lp) for lp in loops])
        self.tree = cKDTree(_unit(self.a + self.b))

    def project(self, p: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(loop id, parameter t in [0, K_loop), chord distance) of the closest coast point to each p."""
        k = min(k, len(self.a))
        _, j = self.tree.query(p, k=k)
        j = j.reshape(len(p), k)
        a, b = self.a[j], self.b[j]                                # (N,k,3)
        d = b - a
        f = np.clip(np.einsum("nki,nki->nk", p[:, None, :] - a, d) / np.einsum("nki,nki->nk", d, d), 0.0, 1.0)
        dist = np.linalg.norm(a + f[..., None] * d - p[:, None, :], axis=2)
        m = np.argmin(dist, axis=1)
        r = np.arange(len(p))
        seg = j[r, m]
        lid = self.seg_loop[seg]
        return lid, (seg - self.off[lid]) + f[r, m], dist[r, m]

    def point(self, lid: int, t: np.ndarray) -> np.ndarray:
        lp = self.loops[lid]
        k = len(lp)
        t = np.mod(t, k)
        i = np.minimum(np.floor(t).astype(np.int64), k - 1)
        f = (t - i)[:, None]
        return _unit((1.0 - f) * lp[i] + f * lp[(i + 1) % k])


# ---------------------------------------------------------------------------------------------------- cell set

def _half_edges(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return cells.ravel(), np.roll(cells, -1, axis=1).ravel()


def _boundary_mask(cells: np.ndarray, n: int) -> np.ndarray:
    """(C,4) bool: half-edge k of cell c has no twin."""
    a, b = _half_edges(cells)
    return ~np.isin(b * n + a, a * n + b).reshape(cells.shape)


def _largest_piece(cells: np.ndarray, n: int) -> np.ndarray:
    a, b = _half_edges(cells)
    key = a * n + b
    order = np.argsort(key)
    rk = b * n + a
    pos = np.clip(np.searchsorted(key[order], rk), 0, len(key) - 1)
    hit = key[order][pos] == rk
    c_of = np.repeat(np.arange(len(cells)), cells.shape[1])
    g = coo_matrix((np.ones(hit.sum()), (c_of[hit], c_of[order[pos[hit]]])), shape=(len(cells),) * 2)
    _, lab = connected_components(g, directed=False)
    return lab == np.argmax(np.bincount(lab))


def clean_cells(cells: np.ndarray, n: int) -> np.ndarray:
    """Drop cells with >= 3 boundary edges and, at each pinch node (> 1 outgoing boundary half-edge), the attached
    cells with the most boundary edges; keep the largest edge-connected piece; repeat until stable."""
    while True:
        bm = _boundary_mask(cells, n)
        nb = bm.sum(axis=1)
        drop = nb >= 3
        pinch = np.bincount(cells[bm], minlength=n) > 1
        if pinch.any():
            worst = np.zeros(n, dtype=np.int64)
            np.maximum.at(worst, cells.ravel(), np.repeat(nb, 4))
            drop |= (pinch[cells] & (nb[:, None] >= worst[cells])).any(axis=1)
        if drop.any():
            cells = cells[~drop]
            continue
        piece = _largest_piece(cells, n)
        if piece.all():
            return cells
        cells = cells[piece]


def boundary_loops(cells: np.ndarray, n: int) -> list[np.ndarray]:
    """Closed node loops of the cell set's boundary, cells on the left."""
    bm = _boundary_mask(cells, n)
    a = cells[bm]
    nxt = np.full(n, -1, dtype=np.int64)
    nxt[a] = np.roll(cells, -1, axis=1)[bm]
    seen = np.zeros(n, dtype=bool)
    loops = []
    for s in a:                                                       # O(boundary) walk
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


def _cells_holding(x: np.ndarray, cells: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(C,) bool: the cell contains one of pts (checked in the cells around the 2 nearest nodes of each point)."""
    n = len(x)
    node_cells = coo_matrix((np.ones(cells.size), (cells.ravel(), np.repeat(np.arange(len(cells)), 4))),
                            shape=(n, len(cells))).tocsr()
    _, near = cKDTree(x).query(pts, k=2)
    hit = np.zeros(len(cells), dtype=bool)
    for col in range(2):
        rows = node_cells[near[:, col]].tocoo()
        c, p = rows.col, pts[rows.row]
        q = x[cells[c]]
        inside = np.ones(len(c), dtype=bool)
        for k in range(4):
            inside &= np.einsum("ij,ij->i", np.cross(q[:, k], q[:, (k + 1) % 4]), p) > 0
        hit[c[inside]] = True
    return hit


# ---------------------------------------------------------------------------------------------------- runs

def _wrap(d: np.ndarray, k: int) -> np.ndarray:
    return d - k * np.round(d / k)


def _runs(x: np.ndarray, sl: np.ndarray, coast: Coast) -> list[tuple]:
    """Split a staircase loop into runs on one coast loop each: (positions in sl, loop id, closed, u), u the
    unwrapped non-decreasing projection parameters. A closed run starts at the node nearest to a coast node."""
    lid, t, _ = coast.project(x[sl])
    m = len(sl)
    if (lid == lid[0]).all():
        k = len(coast.loops[lid[0]])
        r = int(np.argmin(np.abs(t - np.round(t))))
        idx = np.roll(np.arange(m), -r)
        tt = t[idx]
        u = tt[0] + np.r_[0.0, np.cumsum(_wrap(np.diff(tt), k))]
        return [(idx, int(lid[0]), True, np.maximum.accumulate(u))]
    start = np.flatnonzero(lid != np.roll(lid, 1))
    idx = np.roll(np.arange(m), -start[0])
    ll = lid[idx]
    ch = np.flatnonzero(ll != np.roll(ll, 1))
    out = []
    for a, b in zip(ch, np.r_[ch[1:], m]):
        k = len(coast.loops[ll[a]])
        tt = t[idx[a:b]]
        u = tt[0] + np.r_[0.0, np.cumsum(_wrap(np.diff(tt), k))]
        out.append((idx[a:b], int(ll[a]), False, np.maximum.accumulate(u)))
    return out


def _walk(s: list, c: list, u: np.ndarray, elems: list) -> None:
    """Merge walk from (s[0], c[0]) to (s[-1], c[-1]); u[i] = coast parameter of s[i], coast node c[j] sits at
    parameter u[0]-relative j (c spacing 1). Quad where both advance, triangle where one does. Ids are encoded by the
    caller (staircase >= 0, coast < 0)."""
    i, j = 0, 0
    ms, mc = len(s) - 1, len(c) - 1
    j0 = np.round(u[0])
    while i < ms or j < mc:
        if i == ms:
            elems.append((s[i], c[j], c[j + 1]))
            j += 1
            continue
        if j == mc:
            elems.append((s[i + 1], s[i], c[j]))
            i += 1
            continue
        d = u[i + 1] - (j0 + j + 1)
        if abs(d) <= 0.5:
            elems.append((s[i + 1], s[i], c[j], c[j + 1]))
            i += 1
            j += 1
        elif d < 0:
            elems.append((s[i + 1], s[i], c[j]))
            i += 1
        else:
            elems.append((s[i], c[j], c[j + 1]))
            j += 1


def strip_elements(x: np.ndarray, sloops: list[np.ndarray], coast: Coast) -> tuple[list, int]:
    """Elements of the coastal row (tuples of 3 or 4 ids: staircase node id >= 0, coast node -1 - global id) and the
    number of bridge elements (between runs on different coast loops, their coast-side edge is not on the coast)."""
    elems, bridges = [], 0
    for sl in sloops:
        runs = _runs(x, sl, coast)
        if len(runs) == 1 and runs[0][2]:
            idx, li, _, u = runs[0]
            k = len(coast.loops[li])
            j0 = int(np.round(u[0]))
            s = [int(v) for v in sl[idx]] + [int(sl[idx[0]])]
            uu = np.r_[u, u[0] + k]
            c = [-1 - int(coast.off[li] + (j0 + j) % k) for j in range(k + 1)]
            _walk(s, c, uu, elems)
            continue
        prev = None                                                   # (staircase id, coast id) at the end of a run
        first = None
        for idx, li, _, u in runs:
            k = len(coast.loops[li])
            js = np.arange(np.ceil(u[0] - 0.5), np.floor(u[-1] + 0.5) + 1).astype(np.int64)
            s = [int(v) for v in sl[idx]]
            if len(js) == 0:                                          # no coast node: staircase-only triangles
                for v in s:
                    if prev is not None:
                        elems.append((v, prev[0], prev[1]))
                        prev = (v, prev[1])
                continue
            c = [-1 - int(coast.off[li] + j % k) for j in js]
            if prev is None:
                first = (s[0], c[0])
            else:
                bridges += 1
                elems.append((s[0], prev[0], prev[1], c[0]) if prev[1] != c[0] else (s[0], prev[0], prev[1]))
            _walk(s, c, u - (np.round(u[0]) - js[0]), elems)
            prev = (s[-1], c[-1])
        if first is not None and prev is not None:
            bridges += 1
            elems.append((first[0], prev[0], prev[1], first[1]) if prev[1] != first[1] else
                         (first[0], prev[0], prev[1]))
    return elems, bridges


def snap_images(x: np.ndarray, sloops: list[np.ndarray], coast: Coast) -> dict:
    """Staircase node -> its coast image (closest point, made strictly monotone along each run)."""
    img = {}
    for sl in sloops:
        for idx, li, closed, u in _runs(x, sl, coast):
            m = len(u)
            k = len(coast.loops[li])
            top = u[0] + k if closed else u[-1] + 0.5
            u = np.maximum(u, u[0] + 0.02 * np.arange(m) * min(1.0, (top - u[0]) / max(m, 1)))
            for v, p in zip(sl[idx], coast.point(li, u)):
                img[int(v)] = p
    return img


# ---------------------------------------------------------------------------------------------------- smoothing

def _ok(x: np.ndarray, tris: np.ndarray, quads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangles with positive orientation, quads convex (every corner turns left)."""
    t_ok = np.einsum("ij,ij->i", x[tris[:, 0]], np.cross(x[tris[:, 1]], x[tris[:, 2]])) > 0
    q_ok = np.ones(len(quads), dtype=bool)
    p = [x[quads[:, k]] for k in range(4)]
    for k in range(4):
        a, b, c = p[k - 1], p[k], p[(k + 1) % 4]
        q_ok &= np.einsum("ij,ij->i", np.cross(c - b, a - b), b) > 0
    return t_ok, q_ok


def smooth_near_coast(x: np.ndarray, tris: np.ndarray, quads: np.ndarray, fixed: np.ndarray, rows: int,
                      iters: int) -> np.ndarray:
    """Jacobi Laplacian sweeps (neighbour mean, re-normalised) on free nodes within `rows` edge steps of a fixed
    node; moves that turn a valid element invalid are undone."""
    n = len(x)
    e = np.vstack([np.column_stack([c, np.roll(c, -1, axis=1)]).reshape(-1, 2) for c in (tris, quads) if len(c)])
    e = np.unique(np.sort(e, axis=1), axis=0)
    A = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    near = np.zeros(n, dtype=bool)
    near[fixed] = True
    for _ in range(rows):
        near |= (A @ near.astype(float)) > 0
    move = near.copy()
    move[fixed] = False
    x = x.copy()
    for _ in range(iters if move.any() else 0):
        t0, q0 = _ok(x, tris, quads)
        new = x.copy()
        new[move] = _unit((A @ x)[move] / deg[move, None])
        for _ in range(20):
            t1, q1 = _ok(new, tris, quads)
            bt, bq = t0 & ~t1, q0 & ~q1
            if not (bt.any() or bq.any()):
                break
            revert = np.zeros(n, dtype=bool)
            revert[tris[bt].ravel()] = True
            revert[quads[bq].ravel()] = True
            new[revert] = x[revert]
        x = new
    return x


# ---------------------------------------------------------------------------------------------------- driver

def cut_to_coast(x: np.ndarray, quads: np.ndarray, domain: Domain, h: SizeField, o: dict) -> Mesh:
    loops = resample_loops(domain, h)
    coast = Coast(loops)
    bpts = np.vstack(loops)
    nb = len(bpts)
    n = len(x)
    wet_dom = Domain(loops, domain.name)
    if o["coast"] == "strip":
        _, _, dist = coast.project(x)
        ok = in_water(wet_dom, x) & (R_EARTH_M * dist >= o["coast_gap"] * h(x))
        keep = ok[quads].all(axis=1)
        keep[keep] &= ~_cells_holding(x, quads[keep], bpts)
    else:
        keep = in_water(wet_dom, _unit(x[quads].sum(axis=1)))
    cells = clean_cells(quads[keep], n)
    sloops = boundary_loops(cells, n)

    if o["coast"] == "strip":
        elems, bridges = strip_elements(x, sloops, coast)
        inner = np.unique(cells)
        cuse = np.zeros(nb, dtype=bool)
        for el in elems:
            for v in el:
                if v < 0:
                    cuse[-1 - v] = True
        cpts = bpts
    else:
        img = snap_images(x, sloops, coast)
        stair = np.fromiter(img.keys(), dtype=np.int64)
        cpts = np.array([img[s] for s in stair]).reshape(-1, 3)
        cuse = np.ones(len(cpts), dtype=bool)
        inner = np.setdiff1d(np.unique(cells), stair)
        elems, bridges = [], 0
    # numbering: coast nodes first (node_flag 1), then interior nodes
    cmap = np.full(len(cpts), -1, dtype=np.int64)
    nc = int(cuse.sum())
    cmap[cuse] = np.arange(nc)
    imap = np.full(n, -1, dtype=np.int64)
    imap[inner] = nc + np.arange(len(inner))
    points = np.vstack([cpts[cuse], x[inner]])
    if o["coast"] == "strip":
        def enc(v):
            return int(cmap[-1 - v]) if v < 0 else int(imap[v])
        tl = [[enc(v) for v in el] for el in elems if len(el) == 3]
        ql = [[enc(v) for v in el] for el in elems if len(el) == 4]
        tris = np.array(tl, dtype=np.int64).reshape(-1, 3)
        quads_out = np.vstack([imap[cells], np.array(ql, dtype=np.int64).reshape(-1, 4)])
    else:
        smap = np.full(n, -1, dtype=np.int64)
        smap[stair] = np.arange(len(stair))
        quads_out = np.where(smap[cells] >= 0, smap[cells], imap[cells])
        tris = np.zeros((0, 3), dtype=np.int64)
    flag = np.zeros(len(points), dtype=np.int8)
    flag[:nc] = 1
    points = smooth_near_coast(points, tris, quads_out, np.arange(nc), o["smooth_rows"], o["smooth_iters"])
    if o["verbose"]:
        print(f"m12 coast ({o['coast']}): kept cells {len(cells)}, staircase loops {[len(s) for s in sloops]}, "
              f"coast loops {[len(lp) for lp in loops]}, coast nodes kept {nc if o['coast'] == 'strip' else 0}/{nb},"
              f" coastal-row quads {len(quads_out) - len(cells)}, triangles {len(tris)}, bridges {bridges}",
              flush=True)
    return Mesh(points, tris, quads_out.astype(np.int64), flag)
