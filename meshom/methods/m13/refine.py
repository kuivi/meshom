"""m13 refinement: a cubed sphere of element blocks refined level by level with SQuadGen's LOWCONN transition tiles.

After SQuadGen (P. A. Ullrich, github.com/ClimateGlobalChange/squadgen) and Guba et al. (2014).

Element blocks (as SQuadGen's): the base cubed sphere has nb x nb blocks per panel; a block that is
never refined ends as 2 x 2 elements. At step k the blocks of the level-k region (resolution nb 2^k per panel) are
split into
- R: blocks marked for level k+1 -> quartered, their children are the blocks of step k+1;
- ring: unmarked blocks next to R -> paved with a LOWCONN tile (standard: one R edge; inner corner: two adjacent R
  edges; outer corner: R only at one corner), plus one "connecting" quad across every side edge shared by two tiles,
  and the inner-corner node merges (SQuadGen merges the two nodes at 0.8 next to the far corner unless the corner is
  a cube corner or a node is claimed twice);
- the rest -> quartered once into final elements.
The tile edge on R carries 4 elements (3 cuts at 1/4, 1/2, 3/4), the far edge 2 (bisected), the side edges 3 (cuts at
0.4 and 0.8 from the R end). After the last step the remaining blocks are quartered into elements. Everything is
conforming and all-quad; the ring is one coarse block (2 coarse = 4 fine elements) wide.

Marks come from a level map on the per-panel block grids (as SQuadGen's refinement map): required level at a point
= smallest l >= 0 with (level-0 element size) / 2^l <= tol * h; a block needs level l if any point in it does. The
map is normalised top-down as in CSRefinementMap::Normalize: on each level the 4 fill rules (cross-over corner,
exterior-corner intersection, opposite sides, checkerboard diagonals) until stable, then every 8-neighbour of a
level-L block gets its parent marked for L-1 (the next ring sits inside the coarser refined region).
"""
import numpy as np
from scipy.sparse import coo_matrix

from meshom.geo import R_EARTH_M
from meshom.methods.m12_cubed_ot import cubed_sphere

BIG = np.int64(1) << 31                       # edge key lo * BIG + hi (node ids < 2^31)
FACES = [(0, 1, 2, 1), (0, 2, 1, -1), (1, 2, 0, 1), (1, 0, 2, -1), (2, 0, 1, 1), (2, 1, 0, -1)]  # = m12 cubed_sphere


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------------------------------- topology

def adjacency(quads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(C,4) edge neighbours E (across edge k = corner k -> k+1) and (C,4) diagonal neighbours D (at corner k, the
    cell sharing only that corner; -1 at valence-3 corners, open boundaries, or no such cell)."""
    C = len(quads)
    n = np.int64(quads.max()) + 1
    a = quads.ravel().astype(np.int64)
    b = np.roll(quads, -1, axis=1).ravel().astype(np.int64)
    key = a * n + b
    order = np.argsort(key)
    ks = key[order]
    tw = b * n + a
    pos = np.minimum(np.searchsorted(ks, tw), len(ks) - 1)
    owner = np.repeat(np.arange(C, dtype=np.int64), 4)
    E = np.where(ks[pos] == tw, owner[order[pos]], -1).reshape(C, 4)
    D = np.full((C, 4), -1, dtype=np.int64)
    ar = np.arange(C)
    for k in range(4):
        n1 = E[:, k]
        ok = n1 >= 0
        n1c = np.maximum(n1, 0)
        j = np.argmax(quads[n1c] == quads[:, k][:, None], axis=1)
        d = E[n1c, j]
        ok &= (d >= 0) & (d != E[:, (k - 1) % 4]) & (d != ar)
        D[:, k] = np.where(ok, d, -1)
    return E, D


def normalize(ref: np.ndarray, E: np.ndarray, D: np.ndarray) -> np.ndarray:
    """CSRefinementMap::Normalize fill rules on one level, until stable (only adds cells)."""
    ref = ref.copy()
    dx = D >= 0
    while True:
        e = (E >= 0) & ref[np.maximum(E, 0)]
        d = dx & ref[np.maximum(D, 0)]
        em1, ep1, ep2 = np.roll(e, 1, axis=1), np.roll(e, -1, axis=1), np.roll(e, -2, axis=1)
        r1 = (em1 & e & dx & ~d).any(axis=1)                                   # two R edges, corner cell not R
        r2 = (d & ((~em1 & ep2) | (~e & ep1))).any(axis=1)                     # R diagonal + R edge on the far side
        r3 = (e & ep2).any(axis=1)                                             # R on opposite sides
        r4 = dx.all(axis=1) & (d & np.roll(d, -2, axis=1) & ~np.roll(d, -1, axis=1)
                               & ~np.roll(d, -3, axis=1)).any(axis=1)          # checkerboard diagonals
        new = ~ref & (r1 | r2 | r3 | r4)
        if not new.any():
            return ref
        ref |= new


def _parent(cell: np.ndarray, n: int) -> np.ndarray:
    f, r = np.divmod(cell, n * n)
    i, j = np.divmod(r, n)
    m = n // 2
    return f * m * m + (i // 2) * m + j // 2


# ---------------------------------------------------------------------------------------------------- level map

def face_grid(n: int, f: int) -> np.ndarray:
    """(n+1, n+1, 3) equiangular nodes of panel f in m12.cubed_sphere's (i, j) layout."""
    t = np.tan(np.linspace(-np.pi / 4, np.pi / 4, n + 1))
    U, V = np.meshgrid(t, t, indexing="ij")
    k, a, b, s = FACES[f]
    e = np.eye(3)
    return _unit(s * e[k] + U[..., None] * e[a] + V[..., None] * e[b])


def _grid_area(g: np.ndarray) -> np.ndarray:
    """(n, n) spherical-quad areas (sr, two triangles) of a face_grid."""
    p0, p1, p2, p3 = g[:-1, :-1], g[1:, :-1], g[1:, 1:], g[:-1, 1:]

    def tri(a, b, c):
        det = np.einsum("...i,...i", a, np.cross(b, c))
        den = 1 + np.einsum("...i,...i", a, b) + np.einsum("...i,...i", b, c) + np.einsum("...i,...i", c, a)
        return 2 * np.arctan2(det, den)
    return np.abs(tri(p0, p1, p2) + tri(p0, p2, p3))


def level_maps(h, nb: int, h_min: float, h_max: float, tol: float, verbose: bool = False, cap=None):
    """Normalised required-level maps. Returns (L, req) with req[k] the (6 n_k^2,) level map at n_k = nb 2^k,
    k = 0..L-1; block c at step k is refined iff req[k][c] >= k + 1. cap(points) -> (N,) largest allowed element
    size (m, inf = none) is an extra criterion (narrow water)."""
    n0 = 2 * nb                                                         # level-0 elements per panel edge
    lg = max(1, int(np.ceil(np.log2(1.3 * h_max / (tol * h_min)))))
    while True:
        nf = nb * 2 ** (lg - 1)                                         # finest block grid of the guess
        lev = np.empty((6, nf, nf), dtype=np.int64)
        for f in range(6):                                              # per face: h on a 2x finer node grid
            g = face_grid(2 * nf, f).reshape(-1, 3)
            hh = tol * h(g)
            if cap is not None:
                hh = np.minimum(hh, cap(g))
            hh = hh.reshape(2 * nf + 1, 2 * nf + 1)
            hmin = np.full((nf, nf), np.inf)
            for di in range(3):
                for dj in range(3):
                    hmin = np.minimum(hmin, hh[di:di + 2 * nf:2, dj:dj + 2 * nf:2])
            s0 = R_EARTH_M * np.sqrt(_grid_area(face_grid(nf, f))) * nf / n0   # level-0 element size here
            lev[f] = np.maximum(0, np.ceil(np.log2(s0 / hmin) - 1e-9)).astype(np.int64)
        L = int(lev.max())
        if L <= lg:
            break
        lg = L
    if L == 0:
        return 0, []
    req = []
    for k in range(L):                                                  # max-pool to n_k = nb 2^k
        r = 2 ** (lg - 1 - k)
        nk = nb * 2 ** k
        req.append(lev.reshape(6, nk, r, nk, r).max(axis=(2, 4)).reshape(-1))
    for lv in range(L, 0, -1):                                          # Normalize, top-down
        k = lv - 1
        nk = nb * 2 ** k
        E, D = adjacency(cubed_sphere(nk)[1])
        ref = normalize(req[k] >= lv, E, D)
        if verbose:
            print(f"m13 level {lv}: {int((req[k] >= lv).sum())} blocks -> {int(ref.sum())} after normalize "
                  f"(grid {nk})", flush=True)
        req[k][ref] = np.maximum(req[k][ref], lv)
        if lv >= 2:
            c = np.flatnonzero(ref)
            nbh = np.concatenate([c, E[c].ravel(), D[c].ravel()])
            par = _parent(np.unique(nbh[nbh >= 0]), nk)
            req[k - 1][par] = np.maximum(req[k - 1][par], lv - 1)
    return L, req


# ---------------------------------------------------------------------------------------------------- tiles

def _edge_xy(e: int, t: float) -> tuple[float, float]:
    return [(t, 0.0), (1.0, t), (1.0 - t, 1.0), (0.0, 1.0 - t)][e]


class Tile:
    """A transition tile in local coordinates: c0 (0,0), c1 (1,0), c2 (1,1), c3 (0,1), CCW; edge e = c_e -> c_e+1.

    edge: name -> (local edge, t from c_e); inner: name -> (x, y); faces: tuples of names (oriented here);
    side: local edge -> (node at 0.4 from the R end, node at 0.8, connecting node); quarter: local edges cut in 4
    (the R edges); merge: pair of names merged unless the far corner c3 is a cube corner."""

    def __init__(self, edge, inner, faces, side, quarter, merge=None):
        self.edge, self.inner, self.side, self.quarter, self.merge = edge, inner, side, quarter, merge
        # R end of each side edge (local corner index): the end nearer to the 0.4 node
        self.r_end = {e: (e if edge[v[0]][1] < 0.5 else (e + 1) % 4) for e, v in side.items()}
        self.bisect = [e for e in range(4) if e not in side and e not in quarter]
        self.inner_xy = np.array([inner[k] for k in inner], dtype=np.float64)
        xy = {f"c{i}": p for i, p in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)])}
        xy.update({k: _edge_xy(*v) for k, v in edge.items()})
        xy.update(inner)
        self.xy = xy
        out = []
        for f in faces:
            p = np.array([xy[v] for v in f])
            a = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
            out.append(tuple(f) if a > 0 else tuple(f[::-1]))
        self.faces = out


STANDARD = Tile(
    edge={"S0": (0, .25), "S1": (0, .5), "S2": (0, .75), "L0": (1, .4), "L1": (1, .8),
          "O0": (2, .5), "R0": (3, .6), "R1": (3, .2)},
    inner={"C": (.5, .4), "n0": (.25, .2), "n1": (.75, .2), "n2": (.75, .65), "n3": (.25, .65)},
    faces=[("S2", "c1", "L0", "n1"), ("S1", "S2", "n1", "C"), ("S0", "S1", "C", "n0"), ("c0", "S0", "n0", "R0"),
           ("C", "n1", "L0", "n2"), ("C", "n2", "O0", "n3"), ("C", "n3", "R0", "n0"), ("n2", "L1", "c2", "O0"),
           ("n3", "O0", "c3", "R1")],
    side={1: ("L0", "L1", "n2"), 3: ("R0", "R1", "n3")},
    quarter={0: ("S0", "S1", "S2")})

INNER = Tile(
    edge={"S0": (0, .25), "S1": (0, .5), "S2": (0, .75), "L0": (1, .25), "L1": (1, .5), "L2": (1, .75),
          "O0": (2, .4), "O1": (2, .8), "R0": (3, .6), "R1": (3, .2)},
    inner={"C": (.6, .4), "n0": (.775, .225), "n1": (.7, .725), "n2": (.4, .6), "n3": (.275, .3),
           "n03": (.5125, .13125), "n01": (.86875, .4875)},
    faces=[("n3", "R0", "c0", "S0"), ("S0", "S1", "n03", "n3"), ("S1", "S2", "n0", "n03"), ("S2", "c1", "L0", "n0"),
           ("L0", "L1", "n01", "n0"), ("L1", "L2", "n1", "n01"), ("L2", "c2", "O0", "n1"), ("n03", "n0", "C", "n3"),
           ("n01", "n1", "C", "n0"), ("n1", "O0", "n2", "C"), ("n3", "C", "n2", "R0"), ("n2", "O1", "c3", "R1")],
    side={2: ("O0", "O1", "n2"), 3: ("R0", "R1", "n2")},
    quarter={0: ("S0", "S1", "S2"), 1: ("L0", "L1", "L2")},
    merge=("R1", "O1"))

OUTER = Tile(                                                           # R touches only corner c1
    edge={"S0": (0, .6), "S1": (0, .2), "L0": (1, .4), "L1": (1, .8), "O0": (2, .5), "Rm": (3, .5)},
    inner={"C": (.5, .5), "n0": (.375, .25), "n1": (.75, .625)},
    faces=[("C", "c3", "Rm", "n0"), ("c0", "S1", "n0", "Rm"), ("S0", "c1", "C", "n0"), ("c1", "L0", "n1", "C"),
           ("L1", "c2", "O0", "n1"), ("C", "n1", "O0", "c3")],
    side={0: ("S0", "S1", "n0"), 1: ("L0", "L1", "n1")},
    quarter={})


# ---------------------------------------------------------------------------------------------------- node store

class Nodes:
    """Growing node store: add() appends without copying; array() concatenates once per call after additions."""

    def __init__(self, x: np.ndarray):
        self.parts = [np.asarray(x, dtype=np.float64)]
        self.n = len(x)

    def add(self, p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p)
        ids = np.arange(self.n, self.n + len(p), dtype=np.int64)
        self.parts.append(p)
        self.n += len(p)
        return ids

    def array(self) -> np.ndarray:
        if len(self.parts) > 1:
            self.parts = [np.vstack(self.parts)]
        return self.parts[0]


class EdgeNodes:
    """Nodes on block edges, keyed by (lo, hi, round(1000 t)) with t measured from lo; the cut pattern of each edge is
    recorded and a second, different pattern raises (a template mismatch would make the mesh non-conforming)."""

    def __init__(self):
        self.node = {}
        self.pattern = {}

    def get(self, X: Nodes, a: int, b: int, pa: np.ndarray, pb: np.ndarray, t: float) -> int:
        """Node at fraction t from a on edge (a, b) with end coordinates pa, pb (created on first use)."""
        key = (a, b, int(round(1000 * t))) if a < b else (b, a, int(round(1000 * (1.0 - t))))
        v = self.node.get(key)
        if v is None:
            v = int(X.add(_unit((1 - t) * pa + t * pb))[0])
            self.node[key] = v
        return v

    def claim(self, a: int, b: int, kind: str, r_end: int | None = None) -> None:
        lo, hi = (a, b) if a < b else (b, a)
        pat = kind if r_end is None else f"{kind}@{r_end}"
        old = self.pattern.setdefault((lo, hi), pat)
        if old != pat:
            raise RuntimeError(f"m13: edge ({lo},{hi}) cut as {old} and {pat}")

    def midpoints(self) -> tuple[np.ndarray, np.ndarray]:
        k = [(lo * BIG + hi, v) for (lo, hi, t), v in self.node.items() if t == 500]
        if not k:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        a = np.array(k, dtype=np.int64)
        o = np.argsort(a[:, 0])
        return a[o, 0], a[o, 1]

    def trisected(self) -> np.ndarray:
        return np.array([lo * BIG + hi for (lo, hi), p in self.pattern.items() if p.startswith("trisect")],
                        dtype=np.int64)


# ---------------------------------------------------------------------------------------------------- one step

def quarter(X: Nodes, quads: np.ndarray, en: EdgeNodes) -> np.ndarray:
    """(4m,4) children of m blocks: child k sits at corner k with corner 0 at the block-grid (2i+di, 2j+dj) corner;
    edge midpoints already in `en` (tile cuts, split midpoints) are reused."""
    if len(quads) == 0:
        return np.zeros((0, 4), dtype=np.int64)
    a, b = quads, np.roll(quads, -1, axis=1)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = (lo * BIG + hi).ravel()
    uk, inv = np.unique(key, return_inverse=True)
    if np.isin(uk, en.trisected()).any():
        raise RuntimeError("m13: a quartered block shares a trisected tile edge")
    mid = np.full(len(uk), -1, dtype=np.int64)
    ek, ev = en.midpoints()
    if len(ek):
        pos = np.minimum(np.searchsorted(ek, uk), len(ek) - 1)
        hit = ek[pos] == uk
        mid[hit] = ev[pos[hit]]
    new = mid < 0
    P = X.array()
    ul, uh = uk[new] // BIG, uk[new] % BIG
    mid[new] = X.add(_unit(P[ul] + P[uh]))
    m = mid[inv].reshape(-1, 4)
    ctr = X.add(_unit(X.array()[quads].sum(axis=1)))
    c = quads
    ch = np.stack([np.column_stack([c[:, 0], m[:, 0], ctr, m[:, 3]]),
                   np.column_stack([m[:, 0], c[:, 1], m[:, 1], ctr]),
                   np.column_stack([ctr, m[:, 1], c[:, 2], m[:, 2]]),
                   np.column_stack([m[:, 3], ctr, m[:, 2], c[:, 3]])], axis=1)   # (m,4,4)
    return ch.reshape(-1, 4)


def _child_cells(cells: np.ndarray, n: int) -> np.ndarray:
    f, r = np.divmod(cells, n * n)
    i, j = np.divmod(r, n)
    m = 2 * n
    out = [f * m * m + (2 * i + di) * m + (2 * j + dj) for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1))]
    return np.stack(out, axis=1).reshape(-1)


def step(X: Nodes, blocks: np.ndarray, cells: np.ndarray, n: int, marks: np.ndarray, en: EdgeNodes,
         corner: set) -> tuple:
    """One refinement step. Returns (new blocks, their cells at 2n, final quads, tile quads, stats)."""
    E, D = adjacency(blocks)
    eR = (E >= 0) & marks[np.maximum(E, 0)]
    dR = (D >= 0) & marks[np.maximum(D, 0)]
    ne = eR.sum(axis=1)
    U = ~marks
    adj2 = eR & np.roll(eR, -1, axis=1)
    std = U & (ne == 1)
    inn = U & (ne == 2) & adj2.any(axis=1)
    out = U & (ne == 0) & dR.any(axis=1)
    bad = U & ~(std | inn | out) & ((ne > 0) | dR.any(axis=1))
    if bad.any() or (out & (dR.sum(axis=1) != 1)).any():
        raise RuntimeError(f"m13: {int(bad.sum())} ring blocks without a tile (normalisation incomplete)")
    tiles = std | inn | out
    faces, conn, merges = [], {}, []
    P = X.array()
    for b in np.flatnonzero(tiles):
        if std[b]:
            r, T = int(np.argmax(eR[b])), STANDARD
        elif inn[b]:
            r, T = int(np.argmax(adj2[b])), INNER
        else:
            r, T = (int(np.argmax(dR[b])) - 1) % 4, OUTER
        c = [int(v) for v in np.roll(blocks[b], -r)]
        q = P[c]
        ids = {f"c{i}": c[i] for i in range(4)}
        for e in T.quarter:
            en.claim(c[e], c[(e + 1) % 4], "quarter")
        for e in T.bisect:
            en.claim(c[e], c[(e + 1) % 4], "bisect")
        for e in T.side:
            en.claim(c[e], c[(e + 1) % 4], "trisect", c[T.r_end[e]])
        for name, (e, t) in T.edge.items():
            ids[name] = en.get(X, c[e], c[(e + 1) % 4], q[e], q[(e + 1) % 4], t)
        xy = T.inner_xy
        pts = _unit(((1 - xy[:, 0]) * (1 - xy[:, 1]))[:, None] * q[0] + (xy[:, 0] * (1 - xy[:, 1]))[:, None] * q[1]
                    + (xy[:, 0] * xy[:, 1])[:, None] * q[2] + ((1 - xy[:, 0]) * xy[:, 1])[:, None] * q[3])
        for k, v in zip(T.inner, X.add(pts)):
            ids[k] = int(v)
        for e, (q1, q2, q3) in T.quarter.items():                        # split midpoints for the R children
            a_, b_, m_ = int(c[e]), int(c[(e + 1) % 4]), ids[q2]
            for (u, w), mm in (((a_, m_), ids[q1]), ((m_, b_), ids[q3])):
                lo, hi = min(u, w), max(u, w)
                en.node[(lo, hi, 500)] = mm
        faces += [tuple(ids[v] for v in f) for f in T.faces]
        for e, (p04, p08, cn) in T.side.items():
            lo, hi = sorted((int(c[e]), int(c[(e + 1) % 4])))
            conn.setdefault((lo, hi), []).append((ids[p04], ids[p08], ids[cn]))
        if T.merge is not None and c[3] not in corner:
            merges.append((ids[T.merge[0]], ids[T.merge[1]]))
    P = X.array()
    for key, lst in conn.items():                                        # connecting quads across side edges
        if len(lst) != 2 or lst[0][:2] != lst[1][:2]:
            raise RuntimeError(f"m13: side edge {key} has {len(lst)} tiles")
        p04, p08, ca = lst[0]
        cb = lst[1][2]
        f = (p04, ca, p08, cb)
        nrm = np.cross(P[p08] - P[p04], P[cb] - P[ca])
        faces.append(f if nrm @ (P[p04] + P[p08]) > 0 else f[::-1])
    tf = np.array(faces, dtype=np.int64).reshape(-1, 4)
    n_merged = 0
    if merges:                                                           # inner-corner merges, none claimed twice
        mm = np.array(merges, dtype=np.int64)
        cnt = np.bincount(mm.ravel(), minlength=X.n)
        mm = mm[(cnt[mm] == 1).all(axis=1)]
        if len(mm):
            P = X.array()
            P[mm[:, 0]] = _unit(P[mm[:, 0]] + P[mm[:, 1]])
            remap = np.arange(X.n)
            remap[mm[:, 1]] = mm[:, 0]
            tf = remap[tf]
            s = np.sort(tf, axis=1)
            tf = tf[(s[:, 1:] != s[:, :-1]).all(axis=1)]
            n_merged = len(mm)
    rest = U & ~tiles
    ch = quarter(X, np.vstack([blocks[marks], blocks[rest]]), en)
    nR = 4 * int(marks.sum())
    stats = {"R": int(marks.sum()), "standard": int(std.sum()), "inner": int(inn.sum()), "outer": int(out.sum()),
             "connecting": len(conn), "merged": n_merged}
    return ch[:nR], _child_cells(cells[marks], n), ch[nR:], tf, stats


# ---------------------------------------------------------------------------------------------------- driver

def refine(h, nb: int, h_min: float, h_max: float, tol: float = np.sqrt(2.0), verbose: bool = False, cap=None):
    """All-quad conforming refined cubed sphere. Returns (points (N,3), quads (Q,4), tag (Q,) int: 0 = quartered
    block, k+1 = transition tile or connecting quad of step k)."""
    x0, q0 = cubed_sphere(nb)
    X = Nodes(x0)
    corner = set(int(v) for v in np.flatnonzero(np.bincount(q0.ravel()) == 3))     # the 8 cube corners
    L, req = level_maps(h, nb, h_min, h_max, tol, verbose, cap)
    en = EdgeNodes()
    blocks, cells, n = q0.astype(np.int64), np.arange(len(q0), dtype=np.int64), nb
    final, tags = [], []
    for k in range(L):
        marks = req[k][cells] >= k + 1
        blocks, cells, done, tf, st = step(X, blocks, cells, n, marks, en, corner)
        final += [done, tf]
        tags += [np.zeros(len(done), dtype=np.int64), np.full(len(tf), k + 1, dtype=np.int64)]
        n *= 2
        if verbose:
            print(f"m13 step {k}: {st}", flush=True)
    done = quarter(X, blocks, en)
    final.append(done)
    tags.append(np.zeros(len(done), dtype=np.int64))
    quads = np.vstack(final)
    tag = np.concatenate(tags)
    used = np.zeros(X.n, dtype=bool)
    used[quads.ravel()] = True
    remap = np.cumsum(used) - 1
    return X.array()[used], remap[quads], tag


# ---------------------------------------------------------------------------------------------------- springs

def spring_smooth(x: np.ndarray, quads: np.ndarray, mobile: np.ndarray, dist: int, iters: int,
                  k: float = 0.1) -> np.ndarray:
    """SQuadGen's spring dynamics: zero-rest-length springs (constant k) on every mesh edge, Jacobi steps
    x += k sum_j (x_j - x), renormalised; mobile = the given nodes grown by dist - 1 edge rings (dist <= 0: all)."""
    n = len(x)
    e = np.column_stack([quads.ravel(), np.roll(quads, -1, axis=1).ravel()])
    e = np.unique(np.sort(e, axis=1), axis=0)
    A = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    mob = np.ones(n, dtype=bool) if dist < 0 else mobile.copy()
    for _ in range(max(dist - 1, 0)):
        mob |= (A @ mob.astype(float)) > 0
    x = x.copy()
    for _ in range(iters):
        d = k * (A @ x - deg[:, None] * x)
        x[mob] = _unit(x[mob] + d[mob])
    return x
