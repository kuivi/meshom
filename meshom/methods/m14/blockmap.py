"""m14 steps 3-4: the block map and the block grid.

Map: on the region's bg triangles, the cross field with zero cones (`cross_field(..., cones=False)`, the trivial
connection grounded through the free boundary: parallel in the flat metric g / (h e^lambda)^2), and m09's least-squares
map dw = (R / (h e^(lambda - c))) e^(-i Theta) dx with a free boundary (no cuts, no integers, one solve);
c = (max lambda + min lambda) / 2. Grid: nodes at the integer points (+ the best of 4 x 4 lattice offsets) of the
map, barycentric on the chord triangle, normalised; cells (i, j)..(i+1, j+1) whose centre lies in the region (a
corner up to half a cell outside it is extrapolated with the affine map of the face holding the centre); cells with an
edge |ln s/h| > trim are dropped; then in cell space: opening and closing with a 3 x 3 square (within the valid
cells), and to a fixed point: the largest 4-connected component, holes filled (with any region cell, so a block keeps
one outline), one-cell spurs removed, one-cell notches filled, diagonal pinches removed; with shrink = k the cleaned set
is eroded by k cells (3 x 3 square) and cleaned again. Then `smooth` kind "size" with the outline fixed. The outline loop runs with
the fill on its left (the block is land for the fill).

Frame and count synchronisation (m14/sync.py): the map is conformal, so a block turned by psi (its lattice CCW seen
from outside) and scaled by e^sigma (cells e^sigma larger, the size target h e^(lambda - c + sigma)) is the same map
times e^(-i psi - sigma); `extract` re-extracts the lattice from the kept map (`BlockGrid.map`) with the same steps
(4 x 4 offsets, trim, clean, the B17 overlap check, the size sweeps), no new solve.
"""
from dataclasses import dataclass, field

import numpy as np
from numba import njit
from scipy import ndimage

from meshom.geo import R_EARTH_M, arc_length
from meshom.mesh import Mesh, signed_area
from meshom.size import SizeField


@dataclass
class BlockGrid:
    points: np.ndarray       # (Nb,3) block nodes
    quads: np.ndarray        # (Q,4) CCW, into points
    ij: np.ndarray           # (Nb,2) lattice index of each node
    outline: np.ndarray      # (K,) node ids of the outline loop, fill on the left
    info: dict = field(default_factory=dict)
    map: object = None       # BlockMap the grid was extracted from (re-extraction by m14/sync.py)


@dataclass
class BlockMap:
    P: np.ndarray            # (N,3) map region nodes
    T: np.ndarray            # (F,3) map region triangles
    W: np.ndarray            # (F,3) complex map per face corner (cells at the integer points)
    ok: np.ndarray           # (F,) faces that carry lattice nodes (not folded, 2 rings off folds)
    in_r: np.ndarray         # (F,) faces of the accepted region (a cell centre must lie in one)
    info: dict = field(default_factory=dict)


@njit(cache=True)
def _lattice_pass(Ur, Ui, ok, oi, oj, of, ob, fill):
    """Integer points inside each positively oriented face (closed, tiny tolerance): count, or fill."""
    cnt = 0
    for f in range(Ur.shape[0]):
        if not ok[f]:
            continue
        u0, v0 = Ur[f, 0], Ui[f, 0]
        u1, v1 = Ur[f, 1], Ui[f, 1]
        u2, v2 = Ur[f, 2], Ui[f, 2]
        det = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)
        if det <= 0.0:
            continue
        for i in range(int(np.ceil(min(u0, min(u1, u2)))), int(np.floor(max(u0, max(u1, u2)))) + 1):
            for j in range(int(np.ceil(min(v0, min(v1, v2)))), int(np.floor(max(v0, max(v1, v2)))) + 1):
                l1 = ((i - u0) * (v2 - v0) - (u2 - u0) * (j - v0)) / det
                l2 = ((u1 - u0) * (j - v0) - (i - u0) * (v1 - v0)) / det
                l0 = 1.0 - l1 - l2
                if l0 >= -1e-12 and l1 >= -1e-12 and l2 >= -1e-12:
                    if fill:
                        oi[cnt] = i
                        oj[cnt] = j
                        of[cnt] = f
                        ob[cnt, 0] = l0
                        ob[cnt, 1] = l1
                        ob[cnt, 2] = l2
                    cnt += 1
    return cnt


def lattice(P: np.ndarray, T: np.ndarray, W: np.ndarray, ok: np.ndarray, overlap: list | None = None):
    """Integer points of the map W (F,3) on faces ok: (ij (M,2), xyz (M,3), face (M,)), one per lattice index.
    overlap (a list) receives the largest chord between two points with the same index (> 0: the map's image
    overlaps itself, e.g. a curved tail of the region)."""
    Ur, Ui = np.ascontiguousarray(W.real), np.ascontiguousarray(W.imag)
    z = np.zeros(1, dtype=np.int64)
    n = _lattice_pass(Ur, Ui, ok, z, z, z, np.zeros((1, 3)), False)
    oi, oj, of = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64)
    ob = np.empty((n, 3))
    _lattice_pass(Ur, Ui, ok, oi, oj, of, ob, True)
    if n == 0:
        return np.zeros((0, 2), np.int64), np.zeros((0, 3)), np.zeros(0, np.int64)
    key = (oi - oi.min()) * (oj.max() - oj.min() + 1) + (oj - oj.min())
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    if overlap is not None:
        xa = np.einsum("mk,mkd->md", ob, P[T[of]])
        overlap.append(float(np.max(np.linalg.norm(xa - xa[first][inv.ravel()], axis=1))))
    oi, oj, of, ob = oi[first], oj[first], of[first], ob[first]
    x = np.einsum("mk,mkd->md", ob, P[T[of]])
    return np.column_stack([oi, oj]), x / np.linalg.norm(x, axis=1, keepdims=True), of


def _grid(ij: np.ndarray) -> tuple:
    """Node-id array over the lattice bounding box: (nid (ni, nj), i0, j0); -1 = no node."""
    i0, j0 = ij.min(axis=0)
    ni, nj = ij.max(axis=0) - (i0, j0) + 1
    nid = np.full((ni, nj), -1, dtype=np.int64)
    nid[ij[:, 0] - i0, ij[:, 1] - j0] = np.arange(len(ij))
    return nid, i0, j0


def _neighbours4(c: np.ndarray) -> np.ndarray:
    p = np.pad(c, 1).astype(np.int64)
    return p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]


def clean_cells(cells: np.ndarray, valid: np.ndarray, fillable: np.ndarray | None = None,
                max_iter: int = 50) -> np.ndarray:
    """Cell-set cleaning (module docstring); returns the cleaned boolean mask. Holes are filled with the fillable
    cells (default: valid), so trimmed interior cells come back and the block keeps one outline."""
    fillable = valid if fillable is None else fillable
    sq = np.ones((3, 3), dtype=bool)
    c = cells & valid
    c = ndimage.binary_opening(c, sq) & valid
    c = ndimage.binary_closing(c, sq, border_value=0) & valid | c
    for _ in range(max_iter):
        old = c.copy()
        lab, nl = ndimage.label(c)                                       # 4-connected
        if nl == 0:
            return c
        if nl > 1:
            c = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        holes = ndimage.binary_fill_holes(c) & ~c
        c |= holes & fillable
        nb = _neighbours4(c)
        c &= ~(c & (nb <= 1))                                            # spurs
        nb = _neighbours4(c)
        c |= valid & ~c & (nb >= 3)                                      # notches
        # diagonal pinch at node (i+1, j+1): cells (i,j) & (i+1,j+1) without (i+1,j), (i,j+1), or the mirror
        a, b, cc, d = c[:-1, :-1], c[1:, 1:], c[1:, :-1], c[:-1, 1:]
        p1 = a & b & ~cc & ~d
        p2 = cc & d & ~a & ~b
        if p1.any() or p2.any():
            nb = _neighbours4(c)
            drop = np.zeros_like(c)
            ii, jj = np.nonzero(p1)                                      # drop the cell with fewer neighbours
            pick_a = nb[ii, jj] <= nb[ii + 1, jj + 1]
            drop[ii[pick_a], jj[pick_a]] = True
            drop[ii[~pick_a] + 1, jj[~pick_a] + 1] = True
            ii, jj = np.nonzero(p2)
            pick_c = nb[ii + 1, jj] <= nb[ii, jj + 1]
            drop[ii[pick_c] + 1, jj[pick_c]] = True
            drop[ii[~pick_c], jj[~pick_c] + 1] = True
            c &= ~drop
        if np.array_equal(c, old):
            break
    return c


def self_crossings(lp: np.ndarray) -> int:
    """Number of pairs of non-adjacent edges of the closed spherical polyline lp that cross (local k-d tree test)."""
    from scipy.spatial import cKDTree

    n = len(lp)
    A, B = lp, np.roll(lp, -1, axis=0)
    pairs = cKDTree(A + B).query_pairs(4.0 * np.linalg.norm(B - A, axis=1).max(), output_type="ndarray")
    if len(pairs) == 0:
        return 0
    i, j = pairs[:, 0], pairs[:, 1]
    far = (np.abs(i - j) != 1) & (np.abs(i - j) != n - 1)
    i, j = i[far], j[far]

    def orient(a, b, c):
        return np.einsum("ij,ij->i", np.cross(b - a, c - a), a)

    c1 = orient(A[i], B[i], A[j]) * orient(A[i], B[i], B[j]) < 0
    c2 = orient(A[j], B[j], A[i]) * orient(A[j], B[j], B[i]) < 0
    return int((c1 & c2).sum())


def outline_loop(cells: np.ndarray, nid: np.ndarray) -> list[np.ndarray]:
    """Boundary loops of the cell set as node-id sequences, fill (outside) on the left."""
    ci, cj = np.nonzero(cells)
    pad = np.pad(cells, 1)
    a_list, b_list = [], []
    # the CCW cell edges (block on the left) without a neighbour cell, reversed: (neighbour offset, from, to)
    edges_def = [((0, -1), (1, 0), (0, 0)), ((1, 0), (1, 1), (1, 0)), ((0, 1), (0, 1), (1, 1)),
                 ((-1, 0), (0, 0), (0, 1))]
    for (di, dj), (ai, aj), (bi, bj) in edges_def:
        free = ~pad[ci + 1 + di, cj + 1 + dj]
        a_list.append(nid[ci[free] + ai, cj[free] + aj])
        b_list.append(nid[ci[free] + bi, cj[free] + bj])
    a, b = np.concatenate(a_list), np.concatenate(b_list)
    nxt = {int(x): int(y) for x, y in zip(a, b)}
    loops, seen = [], set()
    for s in a:
        s = int(s)
        if s in seen:
            continue
        lp = [s]
        seen.add(s)
        x = nxt[s]
        while x != s and len(lp) <= len(a):
            lp.append(x)
            seen.add(x)
            x = nxt[x]
        if x != s:                                                      # a pinch survived: no simple outline
            return []
        loops.append(np.asarray(lp, dtype=np.int64))
    return loops


def lattice_cells(P: np.ndarray, T: np.ndarray, W: np.ndarray, ok: np.ndarray, centre_ok: np.ndarray):
    """Cells of the map W whose centre (i+1/2, j+1/2) lies in a face marked centre_ok, with their four corners: a corner
    inside an ok face is placed there (barycentric on the chord triangle); a corner outside the faces (the cell sticks
    out of the region by up to half a cell) is extrapolated with the affine map of the face holding the centre.
    Returns (cell ij (C,2), node ij (N,2), node xyz (N,3), corner node ids (C,4) CCW, overlap chord)."""
    ov = []
    cij, _, cf = lattice(P, T, W - (0.5 + 0.5j), ok & centre_ok, ov)
    if len(cij) == 0:
        return cij, np.zeros((0, 2), np.int64), np.zeros((0, 3)), np.zeros((0, 4), np.int64), 0.0
    nij, nxyz, _ = lattice(P, T, W, ok, ov)
    off = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    want = (cij[:, None, :] + off[None]).reshape(-1, 2)                  # (4C,2) corners, CCW per cell
    face = np.repeat(cf, 4)
    lo = np.minimum(want.min(axis=0), nij.min(axis=0) if len(nij) else want.min(axis=0))
    span = np.maximum(want.max(axis=0), nij.max(axis=0) if len(nij) else want.max(axis=0)) - lo + 1
    kw = (want[:, 0] - lo[0]) * span[1] + (want[:, 1] - lo[1])
    kn = (nij[:, 0] - lo[0]) * span[1] + (nij[:, 1] - lo[1])
    have = np.isin(kw, kn)
    miss_k, first = np.unique(kw[~have], return_index=True)
    mf = face[~have][first]
    mij = want[~have][first]
    # extrapolate: barycentric of the corner in the centre's face (some weights negative)
    w0, w1, w2 = W[mf, 0], W[mf, 1], W[mf, 2]
    q = mij[:, 0] + 1j * mij[:, 1]
    det = ((w1 - w0).conjugate() * (w2 - w0)).imag
    l1 = ((q - w0).conjugate() * (w2 - w0)).imag / det
    l2 = ((w1 - w0).conjugate() * (q - w0)).imag / det
    bc = np.column_stack([1.0 - l1 - l2, l1, l2])
    x = np.einsum("mk,mkd->md", bc, P[T[mf]])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    allk = np.r_[kn, miss_k]
    allij = np.vstack([nij, mij]) if len(mij) else nij
    allx = np.vstack([nxyz, x]) if len(mij) else nxyz
    order = np.argsort(allk)
    pos = np.searchsorted(allk[order], kw)
    corners = order[pos].reshape(-1, 4)
    return cij, allij, allx, corners, max(ov)


def align_frame(P: np.ndarray, u: np.ndarray, u_ref: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    """u turned about each node's normal by the one constant angle that best matches the cross field u_ref (circular
    mean of the 4th powers, weights w); returns (turned u, angle in degrees)."""
    phi = np.arctan2(np.einsum("ij,ij->i", np.cross(u, u_ref), P), np.einsum("ij,ij->i", u, u_ref))
    a = np.angle(np.sum(w * np.exp(4j * phi))) / 4.0
    return np.cos(a) * u + np.sin(a) * np.cross(P, u), float(np.degrees(a))


def block_map(P: np.ndarray, T: np.ndarray, lam: np.ndarray, h: SizeField, o: dict,
              in_r: np.ndarray | None = None, u_ref: np.ndarray | None = None) -> BlockMap:
    """The block map of the map region (P, T) (local bg nodes and triangles: the accepted region plus a margin) with its
    size correction lam (per node): zero-cone frame (turned to u_ref when `align`), m09's least-squares map with the
    target h e^(lam - c), the folded faces and 2 rings around them marked. in_r (F,) marks the faces of the accepted
    region (default: all faces)."""
    from meshom.field import cross_field, singularities
    from meshom.methods.m09.param import parametrize

    info = {}
    region = Mesh(P, T, np.zeros((0, 4), dtype=np.int64), np.zeros(len(P), dtype=np.int8))
    cf = cross_field(region, h, n=4, boundary_nodes=None, cones=False)
    info["frame_cones"] = int(len(singularities(cf)[0]))
    u = cf.u
    if u_ref is not None and o["align"]:                                # lattice along the global B4 field
        u, info["frame_turn_deg"] = align_frame(P, u, u_ref, np.ones(len(P)))
    c = 0.5 * (lam.max() + lam.min())
    info["F_map"] = 0.5 * float(lam.max() - lam.min())                  # the size budget the map uses (B12)
    hv = h(P) * (np.exp(lam - c) if o["target"] == "conformal" else 1.0)
    prm = parametrize(P, T, u, hv, free_boundary=True)
    info.update(n_cones_map=prm.info["n_cones"], n_flipped=int(prm.flipped.sum()),
                map_log_scale_p99=prm.info["log_scale_p99"], map_aniso_p99=prm.info["aniso_p99"])
    ok = ~prm.flipped
    for _ in range(2 if prm.flipped.any() else 0):                      # folded faces + 2 rings carry no node
        nb = prm.nbr[~ok].ravel()
        ok[nb[nb >= 0]] = False
    in_r = np.ones(len(T), dtype=bool) if in_r is None else in_r
    return BlockMap(P, T, prm.W, ok, in_r, info)


def block_grid(P: np.ndarray, T: np.ndarray, lam: np.ndarray, h: SizeField, o: dict,
               in_r: np.ndarray | None = None, u_ref: np.ndarray | None = None, psi: float = 0.0,
               sigma: float = 0.0) -> BlockGrid | None:
    """Block grid of the map region (block_map), turned by psi (rad) and scaled by e^sigma (extract)."""
    return extract(block_map(P, T, lam, h, o, in_r, u_ref), h, o, psi, sigma)


def region_grid(fit, tris: np.ndarray, h: SizeField, o: dict) -> BlockGrid | None:
    """Block grid of the region made of fit's bg triangles `tris` (steps 3-4 as m14_blocks.build_blocks runs them): the
    map region (plus margin_rings), lambda on it, block_grid. The direct join (B61) builds the one map over a union with
    it, so the lattice runs through the old corridor without a seam."""
    from meshom.methods.m14.place import lambda_solve, with_margin

    mt = with_margin(fit.bg.tris, tris, o["margin_rings"])
    nodes, Tl = np.unique(fit.bg.tris[mt], return_inverse=True)
    Tl = Tl.reshape(-1, 3)
    lam, _, _ = lambda_solve(fit.bg.points[nodes], Tl, (fit.area + fit.lap)[nodes], fit.hv[nodes])
    return block_grid(fit.bg.points[nodes], Tl, lam, h, o, np.isin(mt, tris), fit.u[nodes])


def extract(bm: BlockMap, h: SizeField, o: dict, psi: float = 0.0, sigma: float = 0.0) -> BlockGrid | None:
    """Lattice of the map bm times e^(-i psi - sigma): the block's lattice turned by psi (CCW seen from outside), its
    cells e^sigma larger; 4 x 4 offsets, trim, clean, the B17 overlap check, the size sweeps (module docstring)."""
    from meshom.smooth import smooth

    P, T, ok, in_r = bm.P, bm.T, bm.ok, bm.in_r
    info = dict(bm.info)
    if psi or sigma:
        info.update(sync_psi_deg=float(np.degrees(psi)), sync_sigma=float(sigma))
    Wt = bm.W * np.exp(-1j * psi - sigma)
    best = None
    for du in (0.0, 0.25, 0.5, 0.75):
        for dv in (0.0, 0.25, 0.5, 0.75):
            W = Wt - (du + 1j * dv)
            cij, nij, nxyz, corners, ov = lattice_cells(P, T, W, ok, in_r)
            if best is None or len(cij) > len(best[0]):
                best = (cij, nij, nxyz, corners, (du, dv), ov)
    cij, ij, xyz, corners, off, ov = best
    if len(cij) == 0:
        return None
    if ov * R_EARTH_M > 0.5 * float(h(P).min()):                       # the map overlaps itself: no block
        return BlockGrid(np.zeros((0, 3)), np.zeros((0, 4), np.int64), np.zeros((0, 2), np.int64),
                         np.zeros(0, np.int64), dict(info, overlap=True, n_cells=0, inradius_cells=0.0))
    info["lattice_offset"] = off
    nid, i0, j0 = _grid(ij)
    present = np.zeros((nid.shape[0] - 1, nid.shape[1] - 1), dtype=bool)
    present[cij[:, 0] - i0, cij[:, 1] - j0] = True

    def edge_bad(a, b):                                                 # trim: an edge with |ln s/h| > trim
        m = (a >= 0) & (b >= 0)
        out = np.zeros(a.shape, dtype=bool)
        pa, pb = xyz[a[m]], xyz[b[m]]
        mid = pa + pb
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        out[m] = np.abs(np.log(R_EARTH_M * arc_length(pa, pb) / h(mid))) > o["trim"]
        return out

    bu = edge_bad(nid[:-1, :], nid[1:, :])                               # (ni-1, nj): edge (i,j)-(i+1,j)
    bv = edge_bad(nid[:, :-1], nid[:, 1:])                               # (ni, nj-1): edge (i,j)-(i,j+1)
    valid = present & ~bu[:, :-1] & ~bu[:, 1:] & ~bv[:-1, :] & ~bv[1:, :]
    info["n_cells_lattice"] = int(present.sum())
    info["n_cells_trimmed"] = int((present & ~valid).sum())
    cells = clean_cells(valid, valid, present)
    if o["shrink"] and cells.any():                                    # erode k cells, then clean again
        info["n_cells_before_shrink"] = int(cells.sum())
        full = cells
        cells = clean_cells(ndimage.binary_erosion(full, np.ones((3, 3), dtype=bool), iterations=int(o["shrink"])),
                            full, full)
    if not cells.any():
        return None
    info["n_cells"] = int(cells.sum())
    info["inradius_cells"] = float(ndimage.distance_transform_cdt(np.pad(cells, 1), metric="taxicab").max())
    # nodes of the kept cells, renumbered
    ci, cj = np.nonzero(cells)
    q = np.column_stack([nid[ci, cj], nid[ci + 1, cj], nid[ci + 1, cj + 1], nid[ci, cj + 1]])
    used = np.unique(q)
    new = np.full(len(xyz), -1, dtype=np.int64)
    new[used] = np.arange(len(used))
    quads = new[q]
    pts = xyz[used]
    loops = outline_loop(cells, nid)
    info["n_outline_loops"] = len(loops)
    if len(loops) != 1:
        return None
    loop = new[loops[0]]
    if self_crossings(pts[loop]):                                       # the map's image overlaps itself: no block
        return BlockGrid(np.zeros((0, 3)), np.zeros((0, 4), np.int64), np.zeros((0, 2), np.int64),
                         np.zeros(0, np.int64), dict(info, overlap=True, n_cells=0, inradius_cells=0.0))
    mesh = Mesh(pts, np.zeros((0, 3), dtype=np.int64), quads, np.zeros(len(pts), dtype=np.int8))
    sa = signed_area(mesh)
    if np.median(sa) < 0:                                               # (u, v) orientation: should not happen
        quads = quads[:, ::-1]
        loop = loop[::-1]
        mesh = Mesh(pts, mesh.tris, quads, mesh.node_flag)
        info["flipped_orientation"] = True
    fixed = np.zeros(len(pts), dtype=bool)
    fixed[loop] = True
    if o["block_size_iters"]:
        mesh = smooth(mesh, h, fixed, iters=o["block_size_iters"], kind="size")
    info["n_inverted"] = int((signed_area(mesh) <= 0).sum())
    return BlockGrid(mesh.points, quads, ij[used], loop, info, bm)
