"""Quad and mixed patterns for cavity remeshing (Reberol, Georgiadis, Remacle 2021 §6.1; the integer system of
Takayama 2014).

A pattern is a coarse mesh of a disk (quads and triangles, CCW). Its boundary loop is split at the convex corners
(boundary vertices in one coarse element) into m sides. Chords are the classes of coarse edges under "opposite edges
of a quad"; chord j gets s_j >= 1 subdivisions and a coarse quad with chords (j, k) becomes an s_j x s_k grid. A coarse
triangle is never subdivided: the chords through its edges are fixed to 1 (one row of quads ends at it). The fine
mesh fits a cavity with side lengths N (edges, CCW from a corner) when W s = N, W[i, j] = number of chord-j coarse
edges on side i. Among the positive integer solutions the most balanced one (least sum of s^2) is taken.

Patterns (quad topologies of the paper's Fig. 4, written here from their description, plus mixed ones for meshom):
- regular: one quad (4 corners, no irregular vertex) — removes opposite +-1 pairs (step 1);
- triangle: 3 quads around a valence-3 vertex (3 corners); pentagon: 5 quads around a valence-5 vertex (5 corners)
  — cavities around one singularity that absorb +-1 pairs (step 2);
- tri_cone (mixed): a triangle element in the middle of a 3-corner patch, the cone as a triangle instead of a
  valence-3 vertex (60 deg corners instead of 120 deg);
- diag35 / aligned35: 4 corners with one valence-3 and one valence-5 vertex (a dislocation: one row more on two
  sides) — size-transition patterns (step 4); pent_tri (mixed): the same dislocation as a valence-5 vertex with a
  triangle (m02's own dislocation shape). Asymmetric patterns are kept with their mirror images.
"""
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np


@dataclass
class Pattern:
    name: str
    elems: list                       # coarse elements (tuples of 3 or 4 vertices), CCW
    loop: np.ndarray                  # coarse boundary loop, CCW, starts at a corner
    corner_pos: np.ndarray            # (m,) positions of the corners in loop (corner_pos[0] = 0)
    chord: dict                       # (min, max) coarse edge -> chord id
    W: np.ndarray                     # (m, c) side x chord counts
    fixed: np.ndarray                 # chord ids fixed to 1 (through a triangle edge)
    irr: int                          # sum over interior coarse vertices of (valence - 4)^2, plus the triangles
    _elim: tuple = field(default=None, repr=False)

    @property
    def m(self) -> int:
        return len(self.corner_pos)


def _key(a: int, b: int) -> tuple:
    return (min(a, b), max(a, b))


def _build(name: str, elems) -> Pattern:
    used = sorted({int(v) for e in elems for v in e})
    ren = {v: i for i, v in enumerate(used)}                  # contiguous vertex ids
    elems = [tuple(ren[int(v)] for v in e) for e in elems]
    nvert = len(used)
    half = set()
    for e in elems:
        for k in range(len(e)):
            half.add((e[k], e[(k + 1) % len(e)]))
    nxt = {a: b for (a, b) in half if (b, a) not in half}
    count = np.bincount([v for e in elems for v in e], minlength=nvert)
    start = min(v for v in nxt if count[v] == 1)
    loop = [start]
    while nxt[loop[-1]] != start:
        loop.append(nxt[loop[-1]])
    assert len(loop) == len(nxt), name
    loop = np.array(loop, dtype=np.int64)
    corner_pos = np.flatnonzero(count[loop] == 1)
    assert all(count[v] == 2 for v in loop if count[v] != 1), f"{name}: not convex"
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in elems:
        for k in range(len(e)):
            find(_key(e[k], e[(k + 1) % len(e)]))
        if len(e) == 4:
            for k in range(2):
                a, b = find(_key(e[k], e[k + 1])), find(_key(e[k + 2], e[(k + 3) % 4]))
                parent[a] = b
    roots = sorted({find(x) for x in list(parent)})
    rid = {r: i for i, r in enumerate(roots)}
    chord = {x: rid[find(x)] for x in list(parent)}
    fixed = sorted({chord[_key(e[k], e[(k + 1) % 3])] for e in elems if len(e) == 3 for k in range(3)})
    m = len(corner_pos)
    W = np.zeros((m, len(roots)), dtype=np.int64)
    for i in range(m):
        a, b = corner_pos[i], (corner_pos[i + 1] if i + 1 < m else len(loop))
        for p in range(a, b):
            W[i, chord[_key(loop[p], loop[(p + 1) % len(loop)])]] += 1
    bset = set(loop.tolist())
    irr = int(sum((count[v] - 4) ** 2 for v in range(nvert) if v not in bset)) + sum(len(e) == 3 for e in elems)
    return Pattern(name, elems, loop, corner_pos, chord, W, np.array(fixed, dtype=np.int64), irr)


def _mirror(p: Pattern) -> Pattern:
    return _build(p.name + "_m", [e[::-1] for e in p.elems])


def _elim(W: np.ndarray):
    """Exact reduced row echelon form of [W | I]: (R, M, pivots, free) with R = M W."""
    m, c = W.shape
    A = [[Fraction(int(x)) for x in row] + [Fraction(int(i == j)) for j in range(m)] for i, row in enumerate(W)]
    piv, r = [], 0
    for col in range(c):
        p = next((i for i in range(r, m) if A[i][col] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        A[r] = [x / A[r][col] for x in A[r]]
        for i in range(m):
            if i != r and A[i][col] != 0:
                f = A[i][col]
                A[i] = [x - f * y for x, y in zip(A[i], A[r])]
        piv.append(col)
        r += 1
    R = np.array([[float(x) for x in row[:c]] for row in A])
    M = np.array([[float(x) for x in row[c:]] for row in A])
    free = [j for j in range(c) if j not in piv]
    return R, M, piv, free


def solve(p: Pattern, N: np.ndarray):
    """Positive integer s with W s = N, s = 1 on the fixed chords, and least sum s^2; or None."""
    c = p.W.shape[1]
    Wf = np.vstack([p.W, np.eye(c, dtype=np.int64)[p.fixed]])
    Nf = np.r_[N, np.ones(len(p.fixed), dtype=np.int64)]
    if p._elim is None:
        p._elim = _elim(Wf)
    R, M, piv, free = p._elim
    b = M @ Nf.astype(np.float64)
    r = len(piv)
    if np.any(np.abs(b[r:]) > 1e-9):
        return None
    hi = int(N.max())
    if free:
        grids = np.meshgrid(*[np.arange(1, hi + 1)] * len(free), indexing="ij")
        F = np.stack([g.ravel() for g in grids], axis=1).astype(np.float64)       # (K, nf)
    else:
        F = np.zeros((1, 0))
    S = np.zeros((len(F), c))
    S[:, free] = F
    S[:, piv] = b[:r][None, :] - F @ R[:r][:, free].T
    Si = np.rint(S)
    ok = np.all(np.abs(S - Si) < 1e-6, axis=1) & np.all(Si >= 1, axis=1)
    if not ok.any():
        return None
    Si = Si[ok].astype(np.int64)
    Si = Si[np.all(Si @ Wf.T == Nf[None, :], axis=1)]
    if not len(Si):
        return None
    return Si[np.argmin((Si ** 2).sum(axis=1))]


_MEMO: dict = {}


def match(patterns: list, N: np.ndarray):
    """Best (pattern, rotation r, s, fine element area in quads) for a cavity with side lengths N (CCW from its corner
    0): cavity side i maps to pattern side (i + r) % m. Least pattern irregularity first, then the most balanced
    subdivision (memoised)."""
    mk = (tuple(id(p) for p in patterns), tuple(int(x) for x in N))
    if mk not in _MEMO:
        _MEMO[mk] = _match(patterns, np.asarray(N, dtype=np.int64))
    return _MEMO[mk]


def _match(patterns: list, N: np.ndarray):
    best, key = None, None
    for p in patterns:
        if p.m != len(N):
            continue
        for r in range(p.m):
            s = solve(p, np.roll(N, r))
            if s is None:
                continue
            k = (p.irr, int((s ** 2).sum()))
            if key is None or k < key:
                area = sum(int(s[p.chord[_key(e[0], e[1])]] * s[p.chord[_key(e[1], e[2])]]) if len(e) == 4 else 0.5
                           for e in p.elems)
                best, key = (p, r, s, area), k
    return best


def fine_mesh(p: Pattern, s: np.ndarray, r: int):
    """Fine elements of pattern p with chord subdivisions s.

    Returns (n_nodes, elems (F,4) CCW with -1 in the 4th column of triangles, loop (B,) fine boundary loop CCW
    starting at pattern corner r, interior (I,) fine node ids). Fine ids 0..nc-1 are the coarse vertices."""
    nc = max(max(e) for e in p.elems) + 1
    nid = [nc]
    edge_nodes = {}

    def enodes(a: int, b: int) -> list:
        k = _key(a, b)
        if k not in edge_nodes:
            n = int(s[p.chord[k]])
            edge_nodes[k] = [k[0]] + list(range(nid[0], nid[0] + n - 1)) + [k[1]]
            nid[0] += n - 1
        e = edge_nodes[k]
        return e if a == k[0] else e[::-1]

    out = []
    for e in p.elems:
        if len(e) == 3:
            out.append((e[0], e[1], e[2], -1))
            continue
        q0, q1, q2, q3 = e
        na, nb = int(s[p.chord[_key(q0, q1)]]), int(s[p.chord[_key(q1, q2)]])
        G = np.empty((na + 1, nb + 1), dtype=np.int64)
        G[:, 0] = enodes(q0, q1)
        G[na, :] = enodes(q1, q2)
        G[:, nb] = enodes(q3, q2)
        G[0, :] = enodes(q0, q3)
        k = (na - 1) * (nb - 1)
        if k > 0:
            G[1:na, 1:nb] = np.arange(nid[0], nid[0] + k).reshape(na - 1, nb - 1)
            nid[0] += k
        for i in range(na):
            for j in range(nb):
                out.append((G[i, j], G[i + 1, j], G[i + 1, j + 1], G[i, j + 1]))
    loop, cpos = [], []
    L = len(p.loop)
    cset = set(p.corner_pos.tolist())
    for i in range(L):
        if i in cset:
            cpos.append(len(loop))
        loop.extend(enodes(int(p.loop[i]), int(p.loop[(i + 1) % L]))[:-1])
    loop = np.roll(np.array(loop, dtype=np.int64), -cpos[r])
    inner = np.setdiff1d(np.arange(nid[0]), loop)
    return nid[0], np.array(out, dtype=np.int64), loop, inner


_PENT = [(2 * i, 2 * i + 1, (2 * i + 2) % 10, 10) for i in range(5)]
REGULAR = _build("regular", [(0, 1, 2, 3)])
TRIANGLE = _build("triangle", [(0, 1, 6, 5), (1, 2, 3, 6), (3, 4, 5, 6)])
PENTAGON = _build("pentagon", _PENT)
# central triangle (0,1,2), a quad strip on each of its edges, a quad at each of its corners
TRI_CONE = _build("tri_cone", [(0, 1, 2), (1, 0, 3, 4), (2, 1, 5, 6), (0, 2, 7, 8),
                               (0, 8, 9, 3), (1, 4, 10, 5), (2, 6, 11, 7)])
# pentagon + one quad in the notch between two corners (b9, b1 around side vertex b0): b0 becomes valence 3
DIAG35 = _build("diag35", _PENT + [(1, 0, 9, 11)])
# pentagon + a layer around corner b1: b1 becomes valence 3 (diagonal to the centre), b0 and b2 regular
ALIGNED35 = _build("aligned35", _PENT + [(1, 0, 11, 12), (2, 1, 12, 13), (0, 9, 14, 11), (3, 2, 13, 15)])
# pentagon with one quad replaced by a triangle: valence-5 centre + triangle, 4 corners
PENT_TRI = _build("pent_tri", _PENT[1:] + [(0, 2, 10)])

STEP1 = [REGULAR]
STEP2 = [TRIANGLE, PENTAGON, TRI_CONE]
STEP4 = [REGULAR, DIAG35, _mirror(DIAG35), ALIGNED35, _mirror(ALIGNED35), PENT_TRI, _mirror(PENT_TRI)]
