"""Cavity remeshing of a mixed quad-dominant mesh with quad patterns (Reberol, Georgiadis, Remacle 2021 §6.2).

The input mesh (m02) is quad-dominant with leftover triangles; it is never subdivided. A cavity is a disk of elements
(quads and triangles) grown from an irregular vertex; it is replaced by a fine quad pattern whose boundary is the
cavity's boundary loop (so its triangles go only in pairs: an all-quad fill needs an even boundary). The replacement
is kept when the irregularity sum over the cavity's vertices of (valence - 4)^2 plus its triangles falls, and after a
local smoothing (cavity + one ring of elements, ring's outer nodes fixed) no element is inverted, the worst corner
deviation does not exceed max(before, tol_angle) and the worst edge |ln(s/h)| does not exceed max(before, tol_size).

Passes (paper §6.2.2), repeated while something changes: 1 regular quads in maximal cavities; 2 triangle / pentagon
patterns (one valence-3 / -5 vertex) in minimal cavities; 3 = 1; 4 regular + 3-5 transition patterns in minimal
cavities.
"""
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from meshom.geo import R_EARTH_M, arc_length, tangent_basis
from meshom.mesh import Mesh
from meshom.methods.m05 import cavity as cv
from meshom.methods.m05.patterns import STEP1, STEP2, STEP4, fine_mesh, match


class MutableMesh:
    def __init__(self, mesh: Mesh, K: int = 12):
        T, Q, N = mesh.n_tris, mesh.n_quads, mesh.n_nodes
        ne = T + Q
        ce, cn = int(1.5 * ne) + 4096, int(1.5 * N) + 4096
        self.EL = np.full((ce, 4), -1, dtype=np.int64)
        self.EL[:T, :3] = mesh.tris
        self.EL[T:ne] = mesh.quads
        self.NV = np.zeros(ce, dtype=np.int64)
        self.NV[:T], self.NV[T:ne] = 3, 4
        self.alive = np.zeros(ce, dtype=bool)
        self.alive[:ne] = True
        self.ne = ne
        self.P = np.zeros((cn, 3))
        self.P[:N] = mesh.points
        self.flag = np.zeros(cn, dtype=np.int8)
        self.flag[:N] = mesh.node_flag
        self.fixed = np.zeros(cn, dtype=bool)
        self.fixed[:N] = mesh.node_flag != 0
        self.nalive = np.zeros(cn, dtype=bool)
        self.nalive[:N] = True
        self.nn = N
        self.N2E = np.full((cn, K), -1, dtype=np.int64)
        self.ncnt = np.zeros(cn, dtype=np.int64)
        cv.build_n2e(self.EL, self.NV, self.alive, self.ne, self.N2E, self.ncnt)
        self.inc = np.zeros(ce, dtype=bool)
        self.vnext = np.full(cn, -1, dtype=np.int64)
        self.vmark = np.zeros(cn, dtype=np.int64)
        self.protect = np.zeros(cn, dtype=bool)
        self.fails = np.zeros((cn, 3), dtype=np.int64)    # failed seeds per pass kind (regulator, paper §6.2.2)
        self.stamp = 0

    def reserve(self, de: int, dn: int) -> None:
        if self.ne + de > len(self.NV):
            c = 2 * (self.ne + de)
            self.EL = np.vstack([self.EL, np.full((c - len(self.EL), 4), -1, dtype=np.int64)])
            for a in ("NV", "alive", "inc"):
                x = getattr(self, a)
                setattr(self, a, np.concatenate([x, np.zeros(c - len(x), dtype=x.dtype)]))
        if self.nn + dn > len(self.P):
            c = 2 * (self.nn + dn)
            self.P = np.vstack([self.P, np.zeros((c - len(self.P), 3))])
            self.N2E = np.vstack([self.N2E, np.full((c - len(self.N2E), self.N2E.shape[1]), -1, dtype=np.int64)])
            for a in ("flag", "fixed", "nalive", "ncnt", "vnext", "vmark", "protect"):
                x = getattr(self, a)
                setattr(self, a, np.concatenate([x, np.zeros(c - len(x), dtype=x.dtype)]))
            self.fails = np.vstack([self.fails, np.zeros((c - len(self.fails), 3), dtype=np.int64)])

    def to_mesh(self) -> Mesh:
        alive = np.flatnonzero(self.alive[: self.ne])
        used = np.zeros(self.nn, dtype=bool)
        el, nv = self.EL[alive], self.NV[alive]
        for k in range(4):
            used[el[nv > k, k]] = True
        used |= self.fixed[: self.nn] & self.nalive[: self.nn]
        new = np.cumsum(used) - 1
        tris = new[el[nv == 3, :3]]
        quads = new[el[nv == 4]]
        return Mesh(self.P[: self.nn][used].copy(), tris, quads, self.flag[: self.nn][used].copy())


def _tangent(P, p, a):
    d = P[a] - P[p]
    d -= np.einsum("ij,ij->i", d, P[p])[:, None] * P[p]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def region_quality(P, EL, NV, ids, h) -> tuple[float, float, float]:
    """(worst |corner angle - ideal| deg, largest corner angle deg, worst edge |ln(s/h)|) over elements ids."""
    el, nv = EL[ids], NV[ids]
    bad, big = 0.0, 0.0
    ln_max = 0.0
    k = np.arange(4)[None, :]
    nxt = (k + 1) % nv[:, None]
    prv = (k - 1 + nv[:, None]) % nv[:, None]
    valid = k < nv[:, None]
    p = el[valid]
    a = np.take_along_axis(el, nxt, axis=1)[valid]
    b = np.take_along_axis(el, prv, axis=1)[valid]
    u, v = _tangent(P, p, a), _tangent(P, p, b)
    cr = np.einsum("ij,ij->i", np.cross(u, v), P[p])
    ang = np.degrees(np.mod(np.arctan2(cr, np.einsum("ij,ij->i", u, v)), 2 * np.pi))
    ideal = np.where(np.repeat(nv, nv) == 3, 60.0, 90.0)
    bad, big = float(np.max(np.abs(ang - ideal))), float(np.max(ang))
    mid = P[p] + P[a]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    ln_max = float(np.max(np.abs(np.log(R_EARTH_M * arc_length(P[p], P[a]) / h(mid)))))
    return bad, big, ln_max


def _tutte(P, loop, n_f, fq, floop, finner, h, rounds: int = 2):
    """Positions of the interior fine nodes: harmonic map with the boundary loop fixed, solved in the gnomonic plane
    at the loop centroid; edge weights 1 / h at the edge midpoints of the previous solve (first solve uniform), so
    that the spacing follows h along the chords."""
    if len(finner) == 0:
        return np.zeros((0, 3))
    c = P[loop].mean(axis=0)
    c /= np.linalg.norm(c)
    e1, e2 = tangent_basis(c[None, :])
    e1, e2 = e1[0], e2[0]
    g = P[loop] / (P[loop] @ c)[:, None]
    X = np.zeros((n_f, 2))
    X[floop] = np.column_stack([g @ e1, g @ e2])
    loc = np.full(n_f, -1, dtype=np.int64)
    loc[finner] = np.arange(len(finner))
    tri = fq[:, 3] < 0
    last = np.where(tri, fq[:, 2], fq[:, 3])
    e = np.vstack([fq[:, [0, 1]], fq[:, [1, 2]], np.column_stack([fq[:, 2], fq[:, 3]])[~tri],
                   np.column_stack([last, fq[:, 0]])])
    e = np.unique(np.sort(e, axis=1), axis=0)
    i, j = np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])
    m = loc[i] >= 0
    i, j = i[m], j[m]
    n = len(finner)
    inner_j = loc[j] >= 0
    w = np.ones(len(i))
    for it in range(rounds):
        A = coo_matrix((np.r_[np.bincount(loc[i], w, minlength=n), -w[inner_j]],
                        (np.r_[np.arange(n), loc[i][inner_j]], np.r_[np.arange(n), loc[j][inner_j]])),
                       shape=(n, n)).tocsc()
        rhs = np.zeros((n, 2))
        np.add.at(rhs, loc[i][~inner_j], w[~inner_j, None] * X[j[~inner_j]])
        X[finner] = np.column_stack([spsolve(A, rhs[:, 0]), spsolve(A, rhs[:, 1])]).reshape(n, 2)
        if it + 1 < rounds:
            mid = 0.5 * (X[i] + X[j])
            q = c[None, :] + mid[:, :1] * e1[None, :] + mid[:, 1:] * e2[None, :]
            w = 1.0 / h(q / np.linalg.norm(q, axis=1, keepdims=True))
            w /= w.mean()
    q = c[None, :] + X[finner, :1] * e1[None, :] + X[finner, 1:] * e2[None, :]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _smooth_region(mm: MutableMesh, ids, movable_nodes, h, iters: int, kind: str) -> None:
    from meshom.smooth import smooth

    el, nv = mm.EL[ids], mm.NV[ids]
    nodes = np.unique(el[el >= 0])
    loc = np.full(mm.nn, -1, dtype=np.int64)
    loc[nodes] = np.arange(len(nodes))
    tris = loc[el[nv == 3, :3]]
    quads = loc[el[nv == 4]]
    sub = Mesh(mm.P[nodes].copy(), tris, quads, np.zeros(len(nodes), dtype=np.int8))
    fixed = np.ones(len(nodes), dtype=bool)
    fixed[loc[movable_nodes]] = False
    sub = smooth(sub, h, fixed, iters=iters, kind=kind)
    mm.P[nodes] = sub.points


def _rings(mm: MutableMesh, cel: np.ndarray, loop: np.ndarray, k: int) -> np.ndarray:
    """Alive elements within k element layers around the cavity (vertex-adjacent), cavity excluded, and never an
    element touching a fixed (coast) node: the coastal row is not moved by the local smoothing."""
    mm.inc[cel] = True
    out = []
    front = loop
    for _ in range(k):
        e = mm.N2E[front]
        e = np.unique(e[e >= 0])
        e = e[~mm.inc[e]]
        f = mm.EL[e]
        e = e[~np.any((f >= 0) & mm.fixed[np.maximum(f, 0)], axis=1)]
        mm.inc[e] = True
        out.append(e)
        f = mm.EL[e]
        front = np.unique(f[f >= 0])
    mm.inc[cel] = False
    r = np.concatenate(out) if out else np.zeros(0, dtype=np.int64)
    mm.inc[r] = False
    return r


def _inner_nodes(mm: MutableMesh, region: np.ndarray) -> np.ndarray:
    """Nodes all of whose elements are in region, not fixed."""
    mm.inc[region] = True
    el = mm.EL[region]
    nodes = np.unique(el[el >= 0])
    E = mm.N2E[nodes]
    ok = np.all((E < 0) | mm.inc[np.maximum(E, 0)], axis=1) & ~mm.fixed[nodes]
    mm.inc[region] = False
    return nodes[ok]


def try_cavity(mm: MutableMesh, cel: np.ndarray, best, h, o: dict, st: dict) -> bool:
    """Replace the cavity cel by the matched pattern best = (pattern, rotation, s); keep it if the checks pass."""
    mm.inc[cel] = True
    loop = np.zeros(4 * len(cel) + 8, dtype=np.int64)
    nin = np.zeros_like(loop)
    mm.stamp += 1
    L = cv.boundary(mm.EL, mm.NV, mm.N2E, mm.ncnt, mm.inc, cel, len(cel), mm.vnext, mm.vmark, mm.stamp, loop, nin)
    mm.inc[cel] = False
    if L == 0:
        return False
    loop, nin = loop[:L].copy(), nin[:L]
    p, r, s = best[:3]
    n_f, fq, floop, finner = fine_mesh(p, s, r)
    if len(floop) != L:
        return False
    # old state
    el_old = mm.EL[cel]
    v_old = np.unique(el_old[el_old >= 0])
    inner_old = np.setdiff1d(v_old, loop)
    ring = _rings(mm, cel, loop, int(o["rings"]))
    before = region_quality(mm.P, mm.EL, mm.NV, np.r_[cel, ring], h)
    region_nodes = np.unique(np.r_[v_old, mm.EL[ring][mm.EL[ring] >= 0]])
    P_save = mm.P[region_nodes].copy()
    # new nodes and elements
    mm.reserve(len(fq), len(finner))
    gid = np.empty(n_f, dtype=np.int64)
    gid[floop] = loop
    nn0, ne0 = mm.nn, mm.ne
    gid[finner] = nn0 + np.arange(len(finner))
    mm.P[nn0: nn0 + len(finner)] = _tutte(mm.P, loop, n_f, fq, floop, finner, h)
    mm.nalive[nn0: nn0 + len(finner)] = True
    mm.flag[nn0: nn0 + len(finner)] = 0
    mm.fixed[nn0: nn0 + len(finner)] = False
    mm.ncnt[nn0: nn0 + len(finner)] = 0
    mm.nn += len(finner)
    new = ne0 + np.arange(len(fq))
    mm.EL[new] = np.where(fq >= 0, gid[np.maximum(fq, 0)], -1)
    mm.NV[new] = np.where(fq[:, 3] < 0, 3, 4)
    mm.ne += len(fq)
    cv.remove_elems(mm.EL, mm.NV, mm.alive, mm.N2E, mm.ncnt, cel)
    ok = cv.add_elems(mm.EL, mm.NV, mm.alive, mm.N2E, mm.ncnt, new)
    if ok:
        mm.nalive[inner_old] = False
        region = np.r_[new, ring]
        movable = _inner_nodes(mm, region)
        _smooth_region(mm, region, movable, h, o["local_iters"], "size")
        after = region_quality(mm.P, mm.EL, mm.NV, region, h)
        ok = (after[1] < 179.0 and after[0] <= max(before[0] + o["eps_angle"], o["tol_angle"])
              and after[2] <= max(before[2] + o["eps_size"], o["tol_size"]))
        st["rejected_quality"] += not ok
        if "log" in st:
            st["log"].append((len(cel), len(fq), p.name, before, after, ok))
    else:
        st["rejected_valence"] += 1
    if ok:
        mm.fails[region_nodes] = 0
        mm.fails[nn0: mm.nn] = 0
        return True
    # revert
    if mm.alive[new].any():
        cv.remove_elems(mm.EL, mm.NV, mm.alive, mm.N2E, mm.ncnt, new)
    mm.ne, mm.nn = ne0, nn0
    mm.nalive[inner_old] = True
    cv.add_elems(mm.EL, mm.NV, mm.alive, mm.N2E, mm.ncnt, cel)
    mm.P[region_nodes] = P_save
    return False


def _irregular_seeds(mm: MutableMesh) -> np.ndarray:
    n = mm.nn
    deg = mm.ncnt[:n]
    irr = mm.nalive[:n] & ~mm.fixed[:n] & (deg != 4)
    tri = np.flatnonzero(mm.alive[: mm.ne] & (mm.NV[: mm.ne] == 3))
    irr[mm.EL[tri, :3].ravel()] = True
    irr &= mm.nalive[:n] & ~mm.fixed[:n]
    v = np.flatnonzero(irr)
    return v[np.argsort(-((deg[v] - 4) ** 2), kind="stable")]


def run_pass(mm: MutableMesh, patterns: list, policy: str, h, o: dict, st: dict, kind: int = 0) -> int:
    """One sweep over the irregular vertices: grow a cavity from each, remesh the best snapshot. Returns the count of
    remeshed cavities."""
    max_el = int(o["max_cavity"])
    cel = np.zeros(max_el + 256, dtype=np.int64)
    loop = np.zeros(8 * max_el + 64, dtype=np.int64)
    nin = np.zeros_like(loop)
    ms = int(o["max_steps"])
    s_n = np.zeros(ms, dtype=np.int64)
    s_nc = np.zeros(ms, dtype=np.int64)
    s_sides = np.zeros((ms, cv.MAXS), dtype=np.int64)
    s_before = np.zeros(ms, dtype=np.int64)
    s_after = np.zeros(ms, dtype=np.int64)
    s_tri = np.zeros(ms, dtype=np.int64)
    ncorner = {p.m for p in patterns}
    min_irr = {m: min(p.irr for p in patterns if p.m == m) for m in ncorner}
    only_regular = all(p.name == "regular" for p in patterns)
    done = 0
    for v in _irregular_seeds(mm):
        if not mm.nalive[v] or mm.ncnt[v] == 0:
            continue
        deg = mm.ncnt[v]
        seeds = mm.N2E[v, :deg].copy()
        if deg == 4 and not (mm.NV[seeds] == 3).any():
            continue
        if mm.fails[v, kind] >= o["max_fails"]:
            continue
        ns, mm.stamp = cv.grow(mm.EL, mm.NV, mm.alive, mm.N2E, mm.ncnt, mm.fixed, mm.protect, seeds, max_el, mm.inc,
                               mm.vnext, mm.vmark, mm.stamp, cel, loop, nin, s_n, s_nc, s_sides, s_before, s_after, s_tri)
        st["grown"] += 1
        cand = []
        for k in range(ns):
            nc = s_nc[k]
            if nc not in ncorner or nc > cv.MAXS:
                continue
            if s_before[k] - s_after[k] - min_irr[nc] <= 0:
                continue
            sd = s_sides[k, :nc]
            if only_regular and not (sd[0] == sd[2] and sd[1] == sd[3]):
                continue
            m = match(patterns, sd.copy())
            if m is None or s_before[k] - s_after[k] - m[0].irr <= 0:
                continue
            dl = 0.5 * np.log((s_n[k] - 0.5 * s_tri[k]) / m[3])
            if abs(dl) > o["max_dln"]:
                continue
            cand.append((k, m, dl))
        if policy == "max":
            cand = cand[::-1]
        hit = False
        for k, m, dl in cand[: int(o["tries"])]:
            st["tried"] += 1
            if try_cavity(mm, cel[: s_n[k]].copy(), m, h, o, st):
                done += 1
                st["gain"] += int(s_before[k] - s_after[k] - m[0].irr)
                hit = True
                break
        if not hit:
            mm.fails[v, kind] += 1
    return done


def improve(mesh: Mesh, h, o: dict) -> tuple[Mesh, dict]:
    import numba

    nt = numba.get_num_threads()
    numba.set_num_threads(1)                 # the local smoothing runs on small regions: no thread start-up
    try:
        return _improve(mesh, h, o)
    finally:
        numba.set_num_threads(nt)


def _improve(mesh: Mesh, h, o: dict) -> tuple[Mesh, dict]:
    t0 = time.perf_counter()
    mm = MutableMesh(mesh)
    st = dict(grown=0, tried=0, rejected_quality=0, rejected_valence=0, gain=0, remeshed=[])
    small = o.get("policy", "min")
    steps = [(STEP1, "max", 0), (STEP2, small, 1), (STEP1, "max", 0), (STEP4, small, 2)]
    for rnd in range(int(o["rounds"])):
        total = 0
        for pats, pol, kind in steps[: int(o["steps"])]:
            n = run_pass(mm, pats, pol, h, o, st, kind)
            st["remeshed"].append(n)
            total += n
        if total == 0:
            break
    st["cavity_s"] = time.perf_counter() - t0
    return mm.to_mesh(), st
