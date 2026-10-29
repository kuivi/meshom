"""Numba kernels of stage t (meshom/meshfix/tri/__init__.py; decisions B155, B156): local operators on a pure-triangle
mesh on the sphere, each accepted only after a local relaxation when the local measure does not get worse.

State: P (N,3) unit vectors; tris (T,3) CCW, nbr (T,3) the triangle across edge (k, k+1) (-1 on the boundary), vt (N)
one triangle per node (meshom.delaunay's layout, whose _flip is used); val (N) edges per node; mov (N) bool, the nodes
that may move (interior nodes; coast and artificial nodes are held: A slides the coast afterwards); lock (N) bool, the
nodes whose incidence must not change (the artificial boundary, B121); tgt (N) the target valence, w (N) the weight
of a node's valence term; hr (N) h at the nodes over R (unit sphere).

Local measure of a patch (a set of triangles S; B155), measure():
  amin, amax   smallest / largest corner angle over S (each corner in its vertex's tangent plane, optimize._corner)
  sz           largest |ln(l / sqrt(h_a h_b))| over the edges of S (l the great-circle length)
  rt           largest |ln(M_s / M_u)| over the pairs of S's triangles s and their neighbours u (M = mean edge length:
               the scorer's element size ratio)
  energy       sum over S of the corner terms (theta - 60 deg)^2 + w_size / 2 sum of its edges' (ln l/h)^2 + w_ratio
               sum over its neighbours of (ln M_s / M_u)^2 (halved inside S: every edge and pair counts once)
  ok           every triangle of S strictly CCW
A change is accepted (verdict() == 0) when ok, amax_new <= max(amax_old, AMAX_FREE), amin_new >= min(amin_old,
AMIN_FREE), sz_new <= max(sz_old, SZ_FREE), rt_new <= max(rt_old, RT_FREE) and energy_new <= energy_old (glide flips:
<= energy_old (1 + EA_SLACK)). The free values (by default the input's own whole-mesh tails) let triangles move inside
the input's envelope; nothing beyond them may get worse than the patch's worst before.

Relaxation of a node (relax_node): lower the star energy (the energy above over its fan): first the mean of the
equilateral apexes over the opposite edges of its fan, (y1, y2) of each triangle (x, y1, y2): apex = m + sqrt(3)/2
|y2 - y1| n, m the edge midpoint, n the unit tangent normal left of y1 -> y2 (x's side), x <- normalise(x + relax
(target - x)); then one step down the forward-difference gradient of the star energy; each kept only when the star
energy falls.

Parallel drivers (flip_blocks, smooth_blocks; blocks.py): the blocks of one colour run at once; an operator whose
patch or the neighbours of its triangles leave its block is not tried there (deferred), so blocks never share a
triangle they read or write.
"""
import numpy as np
from numba import njit, prange

from meshom.delaunay import _find_edge, _flip, _orient
from meshom.optimize import _corner

MAXF = 64           # B156: the largest fan a kernel handles (FESOM2 stops above 32 elements per node)
SQ3_2 = 0.8660254037844386
THIRD_PI = np.pi / 3.0

# prm indices
I_AMAX, I_AMIN, I_SZ, I_RT, I_EAS, I_RELAX, I_SWEEPS, I_RINGS, I_VMIN, I_VMAX, I_WS, I_WR = range(12)


@njit(cache=True, inline="always")
def _index(tris, t, v):
    for k in range(3):
        if tris[t, k] == v:
            return k
    return -1


@njit(cache=True)
def fan(tris, nbr, vt, v, out):
    """The triangles around v into out; their count, or -1 when the fan is larger than out (or broken)."""
    t0 = vt[v]
    if t0 < 0:
        return -1
    n = 0
    t = t0
    closed = False
    while True:
        if n >= len(out):
            return -1
        out[n] = t
        n += 1
        k = _index(tris, t, v)
        if k < 0:
            return -1
        t = nbr[t, (k + 2) % 3]               # next CCW around v
        if t == t0:
            closed = True
            break
        if t < 0:
            break
    if not closed:
        t = t0
        while True:
            k = _index(tris, t, v)
            t = nbr[t, k]                     # next CW around v
            if t < 0:
                break
            if n >= len(out):
                return -1
            out[n] = t
            n += 1
    return n


@njit(cache=True, inline="always")
def _arc(P, a, b):
    cx = P[a, 1] * P[b, 2] - P[a, 2] * P[b, 1]
    cy = P[a, 2] * P[b, 0] - P[a, 0] * P[b, 2]
    cz = P[a, 0] * P[b, 1] - P[a, 1] * P[b, 0]
    return np.arctan2(np.sqrt(cx * cx + cy * cy + cz * cz), P[a, 0] * P[b, 0] + P[a, 1] * P[b, 1] + P[a, 2] * P[b, 2])


@njit(cache=True, inline="always")
def _mean_edge(P, tris, t):
    a, b, c = tris[t, 0], tris[t, 1], tris[t, 2]
    return (_arc(P, a, b) + _arc(P, b, c) + _arc(P, c, a)) / 3.0


@njit(cache=True, inline="always")
def _inset(S, ns, t):
    for i in range(ns):
        if S[i] == t:
            return True
    return False


@njit(cache=True)
def measure(P, tris, nbr, hr, S, ns, ws, wr, out):
    """The local measure of the triangles S[:ns] into out = (amin, amax, sz, rt, energy, ok). energy = sum over S of
    the corner terms (theta - 60 deg)^2 + ws / 2 sum of its edges' (ln l / h_e)^2 + wr sum over its neighbours u of
    (ln M_t / M_u)^2 (halved when u is in S too, so every edge and pair counts once)."""
    amin, amax, sz, rt, en = 10.0, 0.0, 0.0, 0.0, 0.0
    ok = 1.0
    for i in range(ns):
        t = S[i]
        a, b, c = tris[t, 0], tris[t, 1], tris[t, 2]
        if _orient(P, a, b, c) <= 0:
            ok = 0.0
        for k in range(3):
            p, q, r = tris[t, k], tris[t, (k + 1) % 3], tris[t, (k + 2) % 3]
            th = _corner(P, p, q, r)
            amin = min(amin, th)
            amax = max(amax, th)
            en += (th - THIRD_PI) ** 2
            ln = _arc(P, p, q)
            s = np.log(ln / np.sqrt(hr[p] * hr[q]))
            en += 0.5 * ws * s * s
            sz = max(sz, abs(s))
        ms = _mean_edge(P, tris, t)
        for k in range(3):
            u = nbr[t, k]
            if u >= 0:
                x = np.log(ms / _mean_edge(P, tris, u))
                rt = max(rt, abs(x))
                en += (0.5 if _inset(S, ns, u) else 1.0) * wr * x * x
    out[0], out[1], out[2], out[3], out[4], out[5] = amin, amax, sz, rt, en, ok


@njit(cache=True, inline="always")
def verdict(o0, o1, prm, slack=0.0):
    """The acceptance rule of the module docstring, o1 (after) against o0 (before): 0 accepted, else the first rule
    broken (1 inverted, 2 max angle, 3 min angle, 4 size, 5 ratio, 6 energy; the energy may rise by the relative
    `slack`, which only glide flips get: B155)."""
    if o1[5] < 0.5:
        return 1
    if o1[1] > max(o0[1], prm[I_AMAX]) + 1e-12:
        return 2
    if o1[0] < min(o0[0], prm[I_AMIN]) - 1e-12:
        return 3
    if o1[2] > max(o0[2], prm[I_SZ]) + 1e-12:
        return 4
    if o1[3] > max(o0[3], prm[I_RT]) + 1e-12:
        return 5
    if o1[4] > o0[4] * (1.0 + slack) + 1e-12:
        return 6
    return 0


@njit(cache=True, inline="always")
def better(o0, o1, prm):
    return verdict(o0, o1, prm) == 0


@njit(cache=True)
def _outer(P, tris, nbr, F, nf, OM):
    """OM[i, k] = mean edge of the neighbour of F[i] across edge k when that neighbour is outside the fan, -1 when it
    is in the fan, 0 on the boundary (constant while only the fan's centre moves)."""
    for i in range(nf):
        t = F[i]
        for k in range(3):
            u = nbr[t, k]
            if u < 0:
                OM[i, k] = 0.0
            elif _inset(F, nf, u):
                OM[i, k] = -1.0
            else:
                OM[i, k] = _mean_edge(P, tris, u)


@njit(cache=True)
def _star_energy(P, tris, nbr, hr, F, nf, OM, ws, wr, MS):
    """The energy of measure() over the fan F[:nf] (inf when a triangle of it is not strictly CCW); OM from _outer;
    MS (nf,) work array."""
    en = 0.0
    for i in range(nf):
        t = F[i]
        if _orient(P, tris[t, 0], tris[t, 1], tris[t, 2]) <= 0:
            return np.inf
        ms = 0.0
        for k in range(3):
            p, q, r = tris[t, k], tris[t, (k + 1) % 3], tris[t, (k + 2) % 3]
            en += (_corner(P, p, q, r) - THIRD_PI) ** 2
            ln = _arc(P, p, q)
            ms += ln
            s = np.log(ln / np.sqrt(hr[p] * hr[q]))
            en += 0.5 * ws * s * s
        MS[i] = ms / 3.0
    for i in range(nf):
        t = F[i]
        for k in range(3):
            om = OM[i, k]
            if om > 0.0:
                x = np.log(MS[i] / om)
                en += wr * x * x
            elif om < 0.0:
                u = nbr[t, k]
                for j in range(nf):
                    if F[j] == u:
                        x = np.log(MS[i] / MS[j])
                        en += 0.5 * wr * x * x
                        break
    return en


@njit(cache=True, inline="always")
def _put(P, x, px, py, pz):
    r = np.sqrt(px * px + py * py + pz * pz)
    P[x, 0], P[x, 1], P[x, 2] = px / r, py / r, pz / r


@njit(cache=True)
def relax_node(P, tris, nbr, vt, hr, x, prm, F):
    """Lower the star energy of node x (module docstring): try the apex target, then a step down the forward-difference
    gradient in x's tangent plane (steps 0.3, 0.1, 0.03 of the star's mean edge); a trial is kept only when the star
    energy drops. Returns True when x moved."""
    nf = fan(tris, nbr, vt, x, F)
    if nf <= 0:
        return False
    ws, wr, relax = prm[I_WS], prm[I_WR], prm[I_RELAX]
    OM = np.empty((nf, 3))
    MS = np.empty(nf)
    _outer(P, tris, nbr, F, nf, OM)
    x0, y0, z0 = P[x, 0], P[x, 1], P[x, 2]
    e0 = _star_energy(P, tris, nbr, hr, F, nf, OM, ws, wr, MS)
    moved = False
    # apex target
    sx, sy, sz, sl = 0.0, 0.0, 0.0, 0.0
    for i in range(nf):
        t = F[i]
        k = _index(tris, t, x)
        y1, y2 = tris[t, (k + 1) % 3], tris[t, (k + 2) % 3]
        mx, my, mz = 0.5 * (P[y1, 0] + P[y2, 0]), 0.5 * (P[y1, 1] + P[y2, 1]), 0.5 * (P[y1, 2] + P[y2, 2])
        dx, dy, dz = P[y2, 0] - P[y1, 0], P[y2, 1] - P[y1, 1], P[y2, 2] - P[y1, 2]
        nx, ny, nz = my * dz - mz * dy, mz * dx - mx * dz, mx * dy - my * dx       # m x d: left of y1 -> y2
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)
        if nn == 0.0:
            return False
        L = np.sqrt(dx * dx + dy * dy + dz * dz)
        sl += L
        f = SQ3_2 * L / nn
        sx += mx + f * nx
        sy += my + f * ny
        sz += mz + f * nz
    sx /= nf
    sy /= nf
    sz /= nf
    sl /= nf
    _put(P, x, x0 + relax * (sx - x0), y0 + relax * (sy - y0), z0 + relax * (sz - z0))
    e1 = _star_energy(P, tris, nbr, hr, F, nf, OM, ws, wr, MS)
    if e1 < e0:
        e0 = e1
        moved = True
        x0, y0, z0 = P[x, 0], P[x, 1], P[x, 2]
    else:
        P[x, 0], P[x, 1], P[x, 2] = x0, y0, z0
    # gradient step: tangent basis at x
    ax, ay, az = (1.0, 0.0, 0.0) if abs(x0) < 0.6 else (0.0, 1.0, 0.0)
    d = ax * x0 + ay * y0 + az * z0
    e1x, e1y, e1z = ax - d * x0, ay - d * y0, az - d * z0
    nn = np.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    e1x, e1y, e1z = e1x / nn, e1y / nn, e1z / nn
    e2x, e2y, e2z = y0 * e1z - z0 * e1y, z0 * e1x - x0 * e1z, x0 * e1y - y0 * e1x
    h = 1e-3 * sl
    _put(P, x, x0 + h * e1x, y0 + h * e1y, z0 + h * e1z)
    fp = _star_energy(P, tris, nbr, hr, F, nf, OM, ws, wr, MS)
    _put(P, x, x0 + h * e2x, y0 + h * e2y, z0 + h * e2z)
    gp = _star_energy(P, tris, nbr, hr, F, nf, OM, ws, wr, MS)
    P[x, 0], P[x, 1], P[x, 2] = x0, y0, z0
    if not (np.isfinite(fp) and np.isfinite(gp)):
        return moved
    g1, g2 = fp - e0, gp - e0
    gn = np.sqrt(g1 * g1 + g2 * g2)
    if gn == 0.0:
        return moved
    g1, g2 = -g1 / gn, -g2 / gn
    for step in (0.3, 0.1, 0.03):
        al = step * sl
        _put(P, x, x0 + al * (g1 * e1x + g2 * e2x), y0 + al * (g1 * e1y + g2 * e2y), z0 + al * (g1 * e1z + g2 * e2z))
        e1 = _star_energy(P, tris, nbr, hr, F, nf, OM, ws, wr, MS)
        if e1 < e0:
            return True
    P[x, 0], P[x, 1], P[x, 2] = x0, y0, z0
    return moved


@njit(cache=True)
def _add(arr, n, v):
    for i in range(n):
        if arr[i] == v:
            return n
    if n < len(arr):
        arr[n] = v
        return n + 1
    return -1


@njit(cache=True)
def _patch(tris, nbr, vt, seeds, nseeds, rings, mov, M, S, F):
    """M = the movable nodes among seeds and (rings = 1) their edge neighbours; S = every triangle around a seed or a
    node of M. Returns (nm, ns), ns = -1 on overflow."""
    nm = 0
    ns = 0
    # nodes: seeds plus rings
    R = np.empty(len(M), np.int64)
    nr = 0
    for i in range(nseeds):
        nr = _add(R, nr, seeds[i])
    if rings > 0:
        n0 = nr
        for i in range(n0):
            nf = fan(tris, nbr, vt, R[i], F)
            if nf < 0:
                return 0, -1
            for j in range(nf):
                t = F[j]
                for k in range(3):
                    nr = _add(R, nr, tris[t, k])
                    if nr < 0:
                        return 0, -1
    for i in range(nr):
        v = R[i]
        if mov[v]:
            M[nm] = v
            nm += 1
        nf = fan(tris, nbr, vt, v, F)
        if nf < 0:
            return 0, -1
        if mov[v] or i < nseeds:
            for j in range(nf):
                ns = _add(S, ns, F[j])
                if ns < 0:
                    return 0, -1
    return nm, ns


@njit(cache=True)
def _swap(tris, nbr, vt, t, u):
    """Exchange the ids of the adjacent triangles t and u (rows, the neighbours' references and vt)."""
    for k in range(3):
        tris[t, k], tris[u, k] = tris[u, k], tris[t, k]
        nbr[t, k], nbr[u, k] = nbr[u, k], nbr[t, k]
    for x in (t, u):
        for k in range(3):
            if nbr[x, k] == t:
                nbr[x, k] = u
            elif nbr[x, k] == u:
                nbr[x, k] = t
    for x in (t, u):
        for k in range(3):
            y = nbr[x, k]
            vt[tris[x, k]] = x
            if y >= 0 and y != t and y != u:
                p, q = tris[x, k], tris[x, (k + 1) % 3]
                for kk in range(3):
                    if tris[y, kk] == q and tris[y, (kk + 1) % 3] == p:
                        nbr[y, kk] = x


@njit(cache=True)
def _rotate_to(tris, nbr, t, v):
    """Rotate row t (corners and neighbours together) until tris[t, 0] == v."""
    for _ in range(3):
        if tris[t, 0] == v:
            return
        a, na = tris[t, 0], nbr[t, 0]
        tris[t, 0], nbr[t, 0] = tris[t, 1], nbr[t, 1]
        tris[t, 1], nbr[t, 1] = tris[t, 2], nbr[t, 2]
        tris[t, 2], nbr[t, 2] = a, na


@njit(cache=True)
def _opp(tris, nbr, t, k):
    u = nbr[t, k]
    a, b = tris[t, k], tris[t, (k + 1) % 3]
    for kk in range(3):
        if tris[u, kk] == b and tris[u, (kk + 1) % 3] == a:
            return tris[u, (kk + 2) % 3]
    return -1


@njit(cache=True)
def _confined(tris, nbr, S, ns, blk, b):
    """Every node of the triangles S[:ns] and of their neighbours lies in block b (b < 0: no blocks)."""
    if b < 0:
        return True
    for i in range(ns):
        t = S[i]
        for k in range(3):
            if blk[tris[t, k]] != b:
                return False
            u = nbr[t, k]
            if u >= 0:
                for kk in range(3):
                    if blk[tris[u, kk]] != b:
                        return False
    return True


@njit(cache=True)
def _try_flip(P, tris, nbr, vt, val, mov, lock, bnode, tgt, w, hr, prm, cand, touched, stats, t, k, blk, b,
              M, S, F, seeds, SAVE, o0, o1):
    """The flip of edge k of t (flip_pass); 1 when accepted, 0 when not tried or rejected, -1 when its patch leaves
    block b (left for a later sweep)."""
    vmin, vmax = int(prm[I_VMIN]), int(prm[I_VMAX])
    u = nbr[t, k]
    if u < 0 or u < t:
        return 0
    a, bb, c = tris[t, k], tris[t, (k + 1) % 3], tris[t, (k + 2) % 3]
    d = _opp(tris, nbr, t, k)
    if d < 0:
        return 0
    ea, eb, ec, ed = val[a] - tgt[a], val[bb] - tgt[bb], val[c] - tgt[c], val[d] - tgt[d]
    before = w[a] * ea * ea + w[bb] * eb * eb + w[c] * ec * ec + w[d] * ed * ed
    after = w[a] * (ea - 1) ** 2 + w[bb] * (eb - 1) ** 2 + w[c] * (ec + 1) ** 2 + w[d] * (ed + 1) ** 2
    if after >= before - 1e-9:
        return 0
    if not (cand[a] or cand[bb] or cand[c] or cand[d]):
        return 0
    if b >= 0 and (blk[a] != b or blk[bb] != b or blk[c] != b or blk[d] != b):
        return -1
    if lock[a] or lock[bb] or lock[c] or lock[d] \
            or (not bnode[a] and val[a] - 1 < vmin) or (not bnode[bb] and val[bb] - 1 < vmin) \
            or (bnode[a] and val[a] - 1 < 2) or (bnode[bb] and val[bb] - 1 < 2) \
            or val[c] + 1 > vmax or val[d] + 1 > vmax:
        stats[0] += 1
        stats[3] += 1
        return 0
    seeds[0], seeds[1], seeds[2], seeds[3] = a, bb, c, d
    nm, ns = _patch(tris, nbr, vt, seeds, 4, int(prm[I_RINGS]), mov, M, S, F)
    if ns < 0:
        stats[0] += 1
        stats[3] += 1
        return 0
    if not _confined(tris, nbr, S, ns, blk, b):
        return -1
    stats[0] += 1
    tt, kk = _find_edge(tris, nbr, vt, c, d)
    if tt < 0:
        tt, kk = _find_edge(tris, nbr, vt, d, c)
    if tt >= 0 or ((_orient(P, c, a, d) <= 0 or _orient(P, d, bb, c) <= 0)
                   and not (mov[a] or mov[bb] or mov[c] or mov[d])):
        stats[3] += 1
        return 0
    measure(P, tris, nbr, hr, S, ns, prm[I_WS], prm[I_WR], o0)
    for i in range(nm):
        SAVE[i, 0], SAVE[i, 1], SAVE[i, 2] = P[M[i], 0], P[M[i], 1], P[M[i], 2]
    t_first, u_first = tris[t, 0], tris[u, 0]
    _flip(tris, nbr, vt, t, k)
    for _ in range(int(prm[I_SWEEPS])):
        for i in range(nm):
            relax_node(P, tris, nbr, vt, hr, M[i], prm, F)
    measure(P, tris, nbr, hr, S, ns, prm[I_WS], prm[I_WR], o1)
    glide = (ea * ea + eb * eb + ec * ec + ed * ed) == ((ea - 1) ** 2 + (eb - 1) ** 2 + (ec + 1) ** 2 + (ed + 1) ** 2)
    vd = verdict(o0, o1, prm, prm[I_EAS] if glide else 0.0)
    stats[4 + vd] += 1
    if vd == 0:
        val[a] -= 1
        val[bb] -= 1
        val[c] += 1
        val[d] += 1
        stats[1] += 1
        for i in range(nm):
            touched[M[i]] = 1
        touched[a] = 1
        touched[bb] = 1
        touched[c] = 1
        touched[d] = 1
        return 1
    # undo: flip the new edge (c, d) back (t = (c, a, d): edge (d, c) is index 2), which leaves t and u swapped, so
    # swap their rows back and rotate them as they were; restore the positions
    _flip(tris, nbr, vt, t, 2)
    _swap(tris, nbr, vt, t, u)
    _rotate_to(tris, nbr, t, t_first)
    _rotate_to(tris, nbr, u, u_first)
    for i in range(nm):
        P[M[i], 0], P[M[i], 1], P[M[i], 2] = SAVE[i, 0], SAVE[i, 1], SAVE[i, 2]
    stats[2] += 1
    return 0


@njit(cache=True)
def _flip_tris(P, tris, nbr, vt, val, mov, lock, bnode, tgt, w, hr, prm, cand, touched, stats, tlist, blk, b,
               deferred):
    """The flips of the triangles tlist in order (one block, or all); deferred[v] = 1 for the nodes of edges whose
    patch left the block. Returns accepted."""
    M = np.empty(4 * MAXF, np.int64)
    S = np.empty(8 * MAXF, np.int64)
    F = np.empty(MAXF, np.int64)
    seeds = np.empty(4, np.int64)
    SAVE = np.empty((4 * MAXF, 3))
    o0 = np.empty(6)
    o1 = np.empty(6)
    acc = 0
    for i in range(len(tlist)):
        t = tlist[i]
        for k in range(3):
            r = _try_flip(P, tris, nbr, vt, val, mov, lock, bnode, tgt, w, hr, prm, cand, touched, stats, t, k,
                          blk, b, M, S, F, seeds, SAVE, o0, o1)
            if r == 1:
                acc += 1
                break                           # t changed: go to the next triangle
            if r < 0:
                deferred[tris[t, k]] = 1
                deferred[tris[t, (k + 1) % 3]] = 1
    return acc


@njit(cache=True)
def flip_pass(P, tris, nbr, vt, val, mov, lock, bnode, tgt, w, hr, prm, cand, touched, stats):
    """One greedy pass of valence flips over every interior edge, in triangle order (module docstring): an edge
    (a, b) with opposite nodes c, d is flipped when the weighted valence energy sum w (val - tgt)^2 over a, b, c, d
    drops, no node of the four is locked, a and b keep valence >= VMIN (interior) / 2 (boundary), c and d stay <=
    VMAX, the new edge does not exist, and after the flip and `sweeps` Gauss-Seidel relaxation sweeps of the patch's
    movable nodes the patch measure is no worse (verdict() == 0); otherwise the flip and the positions are undone.
    Only edges with a node in cand are tried (the caller passes the nodes near the last round's changes: a rejected
    flip whose surroundings did not change is rejected again). touched[v] = 1 for every node of an accepted patch.
    stats (>= 11): tried, accepted, rejected by measure, rejected by topology, then the verdict counts 0-6. Returns
    accepted."""
    blk = np.zeros(1, np.int64)
    deferred = np.zeros(1, np.int8)
    return _flip_tris(P, tris, nbr, vt, val, mov, lock, bnode, tgt, w, hr, prm, cand, touched, stats,
                      np.arange(len(tris)), blk, -1, deferred)


@njit(cache=True, parallel=True)
def flip_blocks(P, tris, nbr, vt, val, mov, lock, bnode, tgt, w, hr, prm, cand, touched, bstats, blk, blocks, bptr,
                btri, deferred):
    """flip_pass over the blocks `blocks` (all of one colour: no two share a node or a triangle with a node in the
    other) in parallel, each block's triangles btri[bptr[b]:bptr[b+1]] in order; an edge whose patch (or a neighbour
    of it) leaves its block is not tried and its nodes are marked in deferred. bstats (len(blocks), 11)."""
    acc = np.zeros(len(blocks), np.int64)
    for i in prange(len(blocks)):
        b = blocks[i]
        acc[i] = _flip_tris(P, tris, nbr, vt, val, mov, lock, bnode, tgt, w, hr, prm, cand, touched, bstats[i],
                            btri[bptr[b]:bptr[b + 1]], blk, b, deferred)
    return acc.sum()


@njit(cache=True)
def _smooth_nodes(P, tris, nbr, vt, mov, active, hr, prm, touched, stats, nodes, blk, b, deferred):
    F = np.empty(MAXF, np.int64)
    G = np.empty(MAXF, np.int64)
    o0 = np.empty(6)
    o1 = np.empty(6)
    acc = 0
    for j in range(len(nodes)):
        x = nodes[j]
        if not (mov[x] and active[x]):
            continue
        nf = fan(tris, nbr, vt, x, F)
        if nf <= 0:
            continue
        for i in range(nf):
            G[i] = F[i]
        if not _confined(tris, nbr, G, nf, blk, b):
            deferred[x] = 1
            continue
        stats[0] += 1
        measure(P, tris, nbr, hr, G, nf, prm[I_WS], prm[I_WR], o0)
        px, py, pz = P[x, 0], P[x, 1], P[x, 2]
        if not relax_node(P, tris, nbr, vt, hr, x, prm, F):
            continue
        measure(P, tris, nbr, hr, G, nf, prm[I_WS], prm[I_WR], o1)
        if better(o0, o1, prm) and o1[4] < o0[4]:
            acc += 1
            stats[1] += 1
            touched[x] = 1
        else:
            P[x, 0], P[x, 1], P[x, 2] = px, py, pz
    return acc


@njit(cache=True)
def smooth_pass(P, tris, nbr, vt, mov, active, hr, prm, touched, stats):
    """One Gauss-Seidel sweep over the active movable nodes in node order: each relaxed (relax_node), kept when its
    star (the fan, and its neighbours for the ratio) measures no worse (verdict() == 0) with a lower energy; else put
    back. stats: tried, accepted. Returns accepted."""
    return _smooth_nodes(P, tris, nbr, vt, mov, active, hr, prm, touched, stats, np.arange(len(P)),
                         np.zeros(1, np.int64), -1, np.zeros(1, np.int8))


@njit(cache=True, parallel=True)
def smooth_blocks(P, tris, nbr, vt, mov, active, hr, prm, touched, bstats, blk, blocks, nptr, bnodes, deferred):
    """smooth_pass over the blocks `blocks` of one colour in parallel (each block's nodes bnodes[nptr[b]:nptr[b+1]]
    in order); a node whose star leaves its block is marked in deferred."""
    acc = np.zeros(len(blocks), np.int64)
    for i in prange(len(blocks)):
        b = blocks[i]
        acc[i] = _smooth_nodes(P, tris, nbr, vt, mov, active, hr, prm, touched, bstats[i],
                               bnodes[nptr[b]:nptr[b + 1]], blk, b, deferred)
    return acc.sum()
