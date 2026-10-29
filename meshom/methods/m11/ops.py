"""Local operators on triangle meshes on the sphere, driven by a size field h (MMG / Omega_h pattern).

Each operator is one pass over an independent set (as in Omega_h): the candidates are scored, a greedy maximal
independent set is taken in priority order (no two picked operations touch the same triangle) and applied. On the CPU
the set is applied in one Numba sweep; on a device the same selection runs as Luby rounds (random priorities,
segment_max over the triangle incidence).

- split_edges: bisect every picked edge longer than hi (in units of h) at its metric midpoint on the arc, 1:2 on
  both sides. A boundary edge (coast chord) is split by the coast node half way between its ends along the resampled
  loop (Coast), whatever its length: the coast may start coarse and is filled in by splits.
- collapse_edges: merge the ends of a picked edge shorter than lo; the removed node is a free one (never a fixed node).
  Rejected when the link condition fails, a triangle inverts, an angle exceeds max_angle_deg (and grows), or a new
  edge is longer than hi (so split and collapse do not undo each other).
- flip_edges: Lawson flips to the spherical Delaunay triangulation (empty circumcircle = convex hull), or valence
  flips towards 6 edges per interior node (isotropic remeshing, Botsch & Kobbelt 2004), in rounds of independent
  edges; boundary edges are never flipped, no node drops below 3 edges.

Edge length in units of h ("metric length"): l = L / logmean(h_a, h_b), the integral of ds / h along the arc when h is
linear along it (UGAWG, Ibanez 2017 §4). Fixed nodes (node_flag != 0, the coast) never move and are never removed.
All operators take and return a Mesh with triangles only; nodes may be unreferenced (coast nodes not inserted yet).
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

from meshom.delaunay import adjacency
from meshom.geo import R_EARTH_M, arc_length
from meshom.mesh import Mesh

_TOL = 1e-3        # minimum sin of a corner angle for a new triangle (orientation with margin, ~0.06 deg)


# ---------------------------------------------------------------------------------------------------- lengths

def logmean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(y - x) / ln(y / x), = x when x = y."""
    r = y / x
    near = np.abs(r - 1.0) < 1e-6
    return np.where(near, 0.5 * (x + y), (y - x) / np.log(np.where(near, 2.0, r)))


def metric_length(points: np.ndarray, a: np.ndarray, b: np.ndarray, hn: np.ndarray) -> np.ndarray:
    """Length of edges (a, b) in units of h: arc length / logmean(h_a, h_b); hn = h at the nodes (metres)."""
    return R_EARTH_M * arc_length(points[a], points[b]) / logmean(hn[a], hn[b])


def unique_edges(tris: np.ndarray) -> np.ndarray:
    """(E,2) int64 edges of a triangle mesh, n0 < n1."""
    n = int(tris.max()) + 1 if len(tris) else 0
    a, b = tris.ravel(), np.roll(tris, -1, axis=1).ravel()
    key = np.unique(np.minimum(a, b) * n + np.maximum(a, b))
    return np.column_stack([key // n, key % n])


def edge_ratios(mesh: Mesh, h) -> np.ndarray:
    """Metric length of every edge of the triangles."""
    e = unique_edges(mesh.tris)
    return metric_length(mesh.points, e[:, 0], e[:, 1], h(mesh.points))


def _edges_once(tris: np.ndarray, nbr: np.ndarray, boundary: bool) -> tuple[np.ndarray, np.ndarray]:
    """Half-edges (t, k), one per edge: nbr[t,k] > t, plus the boundary ones (nbr < 0) when asked."""
    sel = nbr > np.arange(len(tris))[:, None]
    if boundary:
        sel |= nbr < 0
    t, k = np.nonzero(sel)
    return t.astype(np.int64), k.astype(np.int64)


# ---------------------------------------------------------------------------------------------------- coast

@dataclass
class Coast:
    """The resampled coast loops as mesh nodes 0..B-1, loop by loop in loop order (water on the left)."""
    start: np.ndarray    # (L,) first node of each loop
    size: np.ndarray     # (L,) nodes per loop
    loop: np.ndarray     # (B,) loop of each coast node

    @classmethod
    def from_loops(cls, loops: list[np.ndarray]) -> "Coast":
        size = np.array([len(lp) for lp in loops], dtype=np.int64)
        start = np.r_[0, np.cumsum(size)[:-1]].astype(np.int64)
        return cls(start, size, np.repeat(np.arange(len(loops), dtype=np.int64), size))

    @property
    def n(self) -> int:
        return len(self.loop)

    def between(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Coast node half way from a to b along their loop (a -> b in loop direction), -1 if they are neighbours."""
        lp = self.loop[a]
        s, n = self.start[lp], self.size[lp]
        pa = a - s
        gap = (b - s - pa) % n
        return np.where(gap > 1, s + (pa + gap // 2) % n, -1)


# ---------------------------------------------------------------------------------------------------- predicates

@njit(cache=True, inline="always")
def _orient_ok(p, q, r):
    """((q-p) x (r-p)) . p > _TOL |q-p| |r-p|: CCW seen from outside, corner at p not degenerate."""
    ux, uy, uz = q[0] - p[0], q[1] - p[1], q[2] - p[2]
    vx, vy, vz = r[0] - p[0], r[1] - p[1], r[2] - p[2]
    d = (uy * vz - uz * vy) * p[0] + (uz * vx - ux * vz) * p[1] + (ux * vy - uy * vx) * p[2]
    return d > _TOL * np.sqrt((ux * ux + uy * uy + uz * uz) * (vx * vx + vy * vy + vz * vz))


@njit(cache=True, inline="always")
def _min_cos(p, q, r):
    """Smallest corner cosine of the chord triangle (p, q, r): cos of its largest angle."""
    out = 1.0
    for k in range(3):
        if k == 0:
            a, b, c = p, q, r
        elif k == 1:
            a, b, c = q, r, p
        else:
            a, b, c = r, p, q
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        cs = (ux * vx + uy * vy + uz * vz) / np.sqrt((ux * ux + uy * uy + uz * uz) * (vx * vx + vy * vy + vz * vz))
        out = min(out, cs)
    return out


@njit(cache=True, inline="always")
def _max_cos(p, q, r):
    """Largest corner cosine of the chord triangle (p, q, r): cos of its smallest angle."""
    out = -1.0
    for k in range(3):
        if k == 0:
            a, b, c = p, q, r
        elif k == 1:
            a, b, c = q, r, p
        else:
            a, b, c = r, p, q
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        cs = (ux * vx + uy * vy + uz * vz) / np.sqrt((ux * ux + uy * uy + uz * uz) * (vx * vx + vy * vy + vz * vz))
        out = max(out, cs)
    return out


@njit(cache=True, inline="always")
def _incircle(P, a, b, c, d):
    """d strictly inside the circumcircle of CCW (a, b, c) on the sphere (above the plane of a, b, c)."""
    ux, uy, uz = P[b, 0] - P[a, 0], P[b, 1] - P[a, 1], P[b, 2] - P[a, 2]
    vx, vy, vz = P[c, 0] - P[a, 0], P[c, 1] - P[a, 1], P[c, 2] - P[a, 2]
    wx, wy, wz = P[d, 0] - P[a, 0], P[d, 1] - P[a, 1], P[d, 2] - P[a, 2]
    s = wx * (uy * vz - uz * vy) + wy * (uz * vx - ux * vz) + wz * (ux * vy - uy * vx)
    tol = 1e-12 * np.sqrt((ux * ux + uy * uy + uz * uz) * (vx * vx + vy * vy + vz * vz) * (wx * wx + wy * wy + wz * wz))
    return s > tol


@njit(cache=True, inline="always")
def _index(tris, t, v):
    for k in range(3):
        if tris[t, k] == v:
            return k
    return -1


@njit(cache=True, inline="always")
def _mlen(P, hr, a, b):
    """Metric length of edge (a, b); hr = h / R at the nodes."""
    cx = P[a, 1] * P[b, 2] - P[a, 2] * P[b, 1]
    cy = P[a, 2] * P[b, 0] - P[a, 0] * P[b, 2]
    cz = P[a, 0] * P[b, 1] - P[a, 1] * P[b, 0]
    L = np.arctan2(np.sqrt(cx * cx + cy * cy + cz * cz), P[a, 0] * P[b, 0] + P[a, 1] * P[b, 1] + P[a, 2] * P[b, 2])
    x, y = hr[a], hr[b]
    r = y / x
    hb = 0.5 * (x + y) if abs(r - 1.0) < 1e-6 else (y - x) / np.log(r)
    return L / hb


# ---------------------------------------------------------------------------------------------------- split

@njit(cache=True)
def _split(P, tris, nbr, ct, ck, cm, cpos, n_nodes, out, take):
    """Greedy independent set of the candidate half-edges (in priority order), each split 1:2 on both sides.

    cm[i] >= 0: coast split by the existing node cm[i]; else a new node at cpos[i] (id n_nodes + running count).
    out: (T + 2C, 3) with out[:T] = tris. take[i] set for the new nodes in creation order. Returns (#tris, #splits).
    """
    t_n = len(tris)
    mark = np.zeros(t_n, dtype=np.bool_)
    n_out = t_n
    n_new = 0
    n_split = 0
    for i in range(len(ct)):
        t1, k = ct[i], ck[i]
        t2 = nbr[t1, k]
        if mark[t1] or (t2 >= 0 and mark[t2]):
            continue
        a, b, c = tris[t1, k], tris[t1, (k + 1) % 3], tris[t1, (k + 2) % 3]
        m_pos = cpos[i]
        if not (_orient_ok(P[a], m_pos, P[c]) and _orient_ok(m_pos, P[b], P[c])):
            continue
        d = -1
        if t2 >= 0:
            j = 0
            for kk in range(3):
                if tris[t2, kk] == b and tris[t2, (kk + 1) % 3] == a:
                    j = kk
            d = tris[t2, (j + 2) % 3]
            if not (_orient_ok(P[b], m_pos, P[d]) and _orient_ok(m_pos, P[a], P[d])):
                continue
        if cm[i] >= 0:
            m = cm[i]
        else:
            m = n_nodes + n_new
            n_new += 1
            take[i] = True
        out[t1, 0], out[t1, 1], out[t1, 2] = a, m, c
        out[n_out, 0], out[n_out, 1], out[n_out, 2] = m, b, c
        n_out += 1
        mark[t1] = True
        if t2 >= 0:
            out[t2, 0], out[t2, 1], out[t2, 2] = b, m, d
            out[n_out, 0], out[n_out, 1], out[n_out, 2] = m, a, d
            n_out += 1
            mark[t2] = True
        n_split += 1
    return n_out, n_split


def split_edges(mesh: Mesh, h, hi: float = 1.4, coast: Coast | None = None) -> tuple[Mesh, int]:
    """One pass: split the edges longer than hi (units of h), longest first, and the coast chords that skip coast
    nodes (first of all). Returns the new mesh and the number of splits."""
    P, tris = mesh.points, mesh.tris
    nbr = adjacency(tris)
    hn = h(P)
    t, k = _edges_once(tris, nbr, boundary=coast is not None)
    a, b = tris[t, k], tris[t, (k + 1) % 3]
    ell = metric_length(P, a, b, hn)
    bnd = nbr[t, k] < 0
    cm = np.full(len(t), -1, dtype=np.int64)
    if bnd.any():
        cm[bnd] = coast.between(a[bnd], b[bnd])
    cand = np.flatnonzero((~bnd & (ell > hi)) | (cm >= 0))
    if len(cand) == 0:
        return mesh, 0
    prio = ell[cand] + np.where(cm[cand] >= 0, 1e6, 0.0)
    cand = cand[np.argsort(-prio, kind="stable")]
    t, k, a, b, cm = t[cand], k[cand], a[cand], b[cand], cm[cand]
    # metric midpoint on the arc: h linear along the edge splits the integral of ds/h in two at h = sqrt(h_a h_b)
    ha, hb = hn[a], hn[b]
    dh = hb - ha
    f = np.where(np.abs(dh) > 1e-9 * ha, (np.sqrt(ha * hb) - ha) / np.where(dh == 0, 1.0, dh), 0.5)
    pos = (1.0 - f)[:, None] * P[a] + f[:, None] * P[b]
    pos[cm >= 0] = P[cm[cm >= 0]]
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    out = np.empty((len(tris) + 2 * len(cand), 3), dtype=np.int64)
    out[: len(tris)] = tris
    take = np.zeros(len(cand), dtype=np.bool_)
    n_out, n_split = _split(P, tris, nbr, t, k, cm, pos, len(P), out, take)
    pts = np.vstack([P, pos[take]])
    flag = np.r_[mesh.node_flag, np.zeros(int(take.sum()), dtype=mesh.node_flag.dtype)]
    return Mesh(pts, out[:n_out], mesh.quads, flag), int(n_split)


# ---------------------------------------------------------------------------------------------------- collapse

@njit(cache=True)
def _fan(tris, nbr, vt, v, touched, out):
    """Triangles around v into out (both ways round at a boundary). Returns the count, or -1 when a touched
    triangle is met (the node is not independent of an operation of this pass) or out is too small."""
    t0 = vt[v]
    if t0 < 0 or touched[t0]:
        return -1
    n = 0
    t = t0
    while True:
        if touched[t] or n >= len(out):
            return -1
        out[n] = t
        n += 1
        k = _index(tris, t, v)
        if k < 0:
            return -1
        t = nbr[t, (k + 2) % 3]
        if t < 0 or t == t0:
            break
    if t < 0:
        t = t0
        while True:
            k = _index(tris, t, v)
            t = nbr[t, k]
            if t < 0:
                break
            if touched[t] or n >= len(out):
                return -1
            out[n] = t
            n += 1
    return n


@njit(cache=True)
def _ring(tris, fan, nf, v, out):
    """Distinct nodes of the triangles in fan, except v."""
    n = 0
    for i in range(nf):
        t = fan[i]
        for kk in range(3):
            x = tris[t, kk]
            if x == v:
                continue
            seen = False
            for j in range(n):
                if out[j] == x:
                    seen = True
                    break
            if not seen and n < len(out):
                out[n] = x
                n += 1
    return n


@njit(cache=True)
def _collapse_score(P, hr, tris, fan, nf, r, keep, hi, cos_max):
    """Moving r onto keep: smallest new max-angle cosine (higher is better), or -9 when the collapse is rejected."""
    worst = 1.0
    for i in range(nf):
        t = fan[i]
        if tris[t, 0] == keep or tris[t, 1] == keep or tris[t, 2] == keep:
            continue
        k = _index(tris, t, r)
        x, y = tris[t, (k + 1) % 3], tris[t, (k + 2) % 3]
        if not (_orient_ok(P[keep], P[x], P[y]) and _orient_ok(P[x], P[y], P[keep])
                and _orient_ok(P[y], P[keep], P[x])):
            return -9.0
        c_new = _min_cos(P[keep], P[x], P[y])
        if c_new < cos_max and c_new < _min_cos(P[r], P[x], P[y]):
            return -9.0
        if _mlen(P, hr, keep, x) > hi or _mlen(P, hr, keep, y) > hi:
            return -9.0
        worst = min(worst, c_new)
    return worst


@njit(cache=True)
def _collapse(P, hr, tris, nbr, vt, fixed, ca, cb, hi, cos_max, removed):
    """Greedy independent collapses of candidate edges (ca[i], cb[i]) in priority order. Deleted triangles get -1
    rows; removed[v] marks the nodes taken out. Returns the number of collapses."""
    touched = np.zeros(len(tris), dtype=np.bool_)
    fa = np.empty(64, dtype=np.int64)
    fb = np.empty(64, dtype=np.int64)
    ra = np.empty(64, dtype=np.int64)
    rb = np.empty(64, dtype=np.int64)
    n_col = 0
    for i in range(len(ca)):
        a, b = ca[i], cb[i]
        if removed[a] or removed[b] or (fixed[a] and fixed[b]):
            continue
        na = _fan(tris, nbr, vt, a, touched, fa)
        if na < 0:
            continue
        nb = _fan(tris, nbr, vt, b, touched, fb)
        if nb < 0:
            continue
        # link condition: a and b share exactly the two opposite nodes
        nra = _ring(tris, fa, na, a, ra)
        nrb = _ring(tris, fb, nb, b, rb)
        common = 0
        for p in range(nra):
            for q in range(nrb):
                if ra[p] == rb[q]:
                    common += 1
        if common != 2:
            continue
        s_ab = -9.0 if fixed[a] else _collapse_score(P, hr, tris, fa, na, a, b, hi, cos_max)
        s_ba = -9.0 if fixed[b] else _collapse_score(P, hr, tris, fb, nb, b, a, hi, cos_max)
        if s_ab < -1.5 and s_ba < -1.5:
            continue
        if s_ab >= s_ba:
            r, keep, fan, nf = a, b, fa, na
        else:
            r, keep, fan, nf = b, a, fb, nb
        for j in range(nf):
            t = fan[j]
            touched[t] = True
            if tris[t, 0] == keep or tris[t, 1] == keep or tris[t, 2] == keep:
                tris[t, 0], tris[t, 1], tris[t, 2] = -1, -1, -1
            else:
                tris[t, _index(tris, t, r)] = keep
        removed[r] = True
        n_col += 1
    return n_col


def compact(mesh: Mesh, keep_fixed: bool = True) -> Mesh:
    """Drop unreferenced nodes (fixed ones kept when keep_fixed) and renumber; node order is kept."""
    used = np.zeros(mesh.n_nodes, dtype=bool)
    used[mesh.tris.ravel()] = True
    if mesh.n_quads:
        used[mesh.quads.ravel()] = True
    if keep_fixed:
        used |= mesh.node_flag != 0
    if used.all():
        return mesh
    new = np.cumsum(used) - 1
    return Mesh(mesh.points[used], new[mesh.tris], new[mesh.quads], mesh.node_flag[used])


def collapse_edges(mesh: Mesh, h, lo: float = 0.7, hi: float = 1.4, max_angle_deg: float = 150.0
                   ) -> tuple[Mesh, int]:
    """One pass: collapse the edges shorter than lo (units of h), shortest first. Returns the compacted mesh (free
    nodes renumbered, fixed nodes keep their ids) and the number of collapses."""
    P, tris = mesh.points, mesh.tris
    nbr = adjacency(tris)
    hn = h(P)
    t, k = _edges_once(tris, nbr, boundary=False)
    a, b = tris[t, k], tris[t, (k + 1) % 3]
    ell = metric_length(P, a, b, hn)
    fixed = mesh.node_flag != 0
    cand = np.flatnonzero((ell < lo) & ~(fixed[a] & fixed[b]))
    if len(cand) == 0:
        return mesh, 0
    cand = cand[np.argsort(ell[cand], kind="stable")]
    vt = np.full(len(P), -1, dtype=np.int64)
    vt[tris.ravel()] = np.repeat(np.arange(len(tris), dtype=np.int64), 3)
    tris = tris.copy()
    removed = np.zeros(len(P), dtype=np.bool_)
    n = _collapse(P, hn / R_EARTH_M, tris, nbr, vt, fixed, a[cand], b[cand], hi,
                  np.cos(np.radians(max_angle_deg)), removed)
    return compact(Mesh(P, tris[tris[:, 0] >= 0], mesh.quads, mesh.node_flag)), int(n)


# ---------------------------------------------------------------------------------------------------- flip

@njit(cache=True, inline="always")
def _set_nbr(tris, nbr, t, a, b, new):
    """In triangle t, the neighbour across edge (a -> b) becomes new."""
    if t < 0:
        return
    for k in range(3):
        if tris[t, k] == a and tris[t, (k + 1) % 3] == b:
            nbr[t, k] = new
            return


@njit(cache=True)
def _flip_rounds(P, tris, nbr, deg, opt, valence, cos_max, cos_min, max_rounds):
    """Rounds of independent flips (each triangle in at most one flip per round) until none is left.
    valence False: Delaunay (d inside the circumcircle of a, b, c). True: the flip lowers the sum of (deg - opt)^2
    over the four nodes, and the new triangles keep their largest angle below acos(cos_max) and their smallest angle
    above acos(cos_min), or no worse than the old pair."""
    t_n = len(tris)
    stamp = np.full(t_n, -1, dtype=np.int64)
    total = 0
    for rnd in range(max_rounds):
        nf = 0
        for t in range(t_n):
            if stamp[t] == rnd:
                continue
            for i in range(3):
                u = nbr[t, i]
                if u < t or stamp[u] == rnd:
                    continue
                a, b, c = tris[t, i], tris[t, (i + 1) % 3], tris[t, (i + 2) % 3]
                j = 0
                for kk in range(3):
                    if tris[u, kk] == b and tris[u, (kk + 1) % 3] == a:
                        j = kk
                d = tris[u, (j + 2) % 3]
                if deg[a] < 4 or deg[b] < 4 or c == d:
                    continue
                if valence:
                    if (deg[a] - opt[a]) + (deg[b] - opt[b]) - (deg[c] - opt[c]) - (deg[d] - opt[d]) <= 2:
                        continue
                    c_new = min(_min_cos(P[c], P[a], P[d]), _min_cos(P[d], P[b], P[c]))
                    if c_new < cos_max and c_new < min(_min_cos(P[a], P[b], P[c]), _min_cos(P[b], P[a], P[d])):
                        continue
                    c_new = max(_max_cos(P[c], P[a], P[d]), _max_cos(P[d], P[b], P[c]))
                    if c_new > cos_min and c_new > max(_max_cos(P[a], P[b], P[c]), _max_cos(P[b], P[a], P[d])):
                        continue
                elif not _incircle(P, a, b, c, d):
                    continue
                if not (_orient_ok(P[c], P[a], P[d]) and _orient_ok(P[d], P[b], P[c])):
                    continue
                n_bc, n_ca = nbr[t, (i + 1) % 3], nbr[t, (i + 2) % 3]
                n_ad, n_db = nbr[u, (j + 1) % 3], nbr[u, (j + 2) % 3]
                tris[t, 0], tris[t, 1], tris[t, 2] = c, a, d
                nbr[t, 0], nbr[t, 1], nbr[t, 2] = n_ca, n_ad, u
                tris[u, 0], tris[u, 1], tris[u, 2] = d, b, c
                nbr[u, 0], nbr[u, 1], nbr[u, 2] = n_db, n_bc, t
                _set_nbr(tris, nbr, n_ad, d, a, t)
                _set_nbr(tris, nbr, n_bc, c, b, u)
                deg[a] -= 1
                deg[b] -= 1
                deg[c] += 1
                deg[d] += 1
                stamp[t] = rnd
                stamp[u] = rnd
                nf += 1
                break
        total += nf
        if nf == 0:
            break
    return total


def flip_edges(mesh: Mesh, kind: str = "delaunay", max_angle_deg: float = 110.0, min_angle_deg: float = 30.0,
               max_rounds: int = 200) -> tuple[Mesh, int]:
    """Flips in rounds of independent edges until none is left; kind "delaunay" (empty circumcircle on the sphere) or
    "valence" (Botsch & Kobbelt 2004: towards 6 edges at interior nodes, 4 at boundary nodes, no new angle above
    max_angle_deg or below min_angle_deg unless the old pair was already that bad). Returns the mesh and the number
    of flips."""
    tris = np.ascontiguousarray(mesh.tris, dtype=np.int64).copy()
    nbr = adjacency(tris)
    t, k = np.nonzero(nbr < 0)                       # boundary half-edges: one leaves each boundary node
    deg = (np.bincount(tris.ravel(), minlength=mesh.n_nodes) + np.bincount(tris[t, k], minlength=mesh.n_nodes))
    deg = deg.astype(np.int64)                        # edges at a node = triangles (+ 1 on the boundary)
    opt = np.full(mesh.n_nodes, 6, dtype=np.int64)
    opt[tris[t, k]] = 4
    n = _flip_rounds(np.ascontiguousarray(mesh.points), tris, nbr, deg, opt, kind == "valence",
                     np.cos(np.radians(max_angle_deg)), np.cos(np.radians(min_angle_deg)), max_rounds)
    return Mesh(mesh.points, tris, mesh.quads, mesh.node_flag), int(n)
