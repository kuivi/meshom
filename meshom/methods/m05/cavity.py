"""Numba kernels of the m05 cavity remeshing on a mutable mixed mesh.

Mutable mesh: EL (cap_e,4) corners (triangles padded with -1), NV (cap_e,) corner counts, alive (cap_e,) flags,
N2E (cap_n,K) node -> alive incident elements (first ncnt[v] entries). Interior nodes of the whole mesh have as many
edges as elements, so ncnt is the valence.

A cavity is a set of alive elements (marked in inc) whose union is a topological disk with one boundary loop. Its
boundary vertices carry n_in (cavity elements at the vertex) and n_out = ncnt - n_in; n_in = 1 is a convex corner,
n_out = 1 a concave one (Reberol 2021 §6). Growth (§6.2.1): add the outside elements along the longest side, then
convexify (add the outside element at every concave vertex) until no vertex is concave; elements touching a fixed
(coast) node are never added and protected vertices never become interior.
"""
import numpy as np
from numba import njit

MAXS = 8          # sides recorded per snapshot


@njit(cache=True)
def build_n2e(EL, NV, alive, ne, N2E, ncnt):
    ncnt[:] = 0
    for e in range(ne):
        if alive[e]:
            for k in range(NV[e]):
                v = EL[e, k]
                N2E[v, ncnt[v]] = e
                ncnt[v] += 1


@njit(cache=True)
def remove_elems(EL, NV, alive, N2E, ncnt, ids):
    for e in ids:
        alive[e] = False
        for k in range(NV[e]):
            v = EL[e, k]
            for t in range(ncnt[v]):
                if N2E[v, t] == e:
                    N2E[v, t] = N2E[v, ncnt[v] - 1]
                    N2E[v, ncnt[v] - 1] = -1           # slots past ncnt stay -1
                    ncnt[v] -= 1
                    break


@njit(cache=True)
def add_elems(EL, NV, alive, N2E, ncnt, ids):
    """Mark ids alive and register them; False (nothing changed) if a node would exceed the N2E width."""
    K = N2E.shape[1]
    for e in ids:
        for k in range(NV[e]):
            ncnt[EL[e, k]] += 1
    over = False
    for e in ids:
        for k in range(NV[e]):
            if ncnt[EL[e, k]] > K:
                over = True
    for e in ids:
        for k in range(NV[e]):
            ncnt[EL[e, k]] -= 1
    if over:
        return False
    for e in ids:
        alive[e] = True
        for k in range(NV[e]):
            v = EL[e, k]
            N2E[v, ncnt[v]] = e
            ncnt[v] += 1
    return True


@njit(cache=True)
def _across(EL, NV, N2E, ncnt, a, b):
    """The element with the directed edge b -> a (the other side of a -> b), or -1."""
    for t in range(ncnt[b]):
        f = N2E[b, t]
        mf = NV[f]
        for j in range(mf):
            if EL[f, j] == b and EL[f, (j + 1) % mf] == a:
                return f
    return -1


@njit(cache=True)
def boundary(EL, NV, N2E, ncnt, inc, cel, nce, vnext, vmark, stamp, loop, nin):
    """Boundary loop of the cavity cel[:nce] (members marked in inc), CCW, starting at a convex corner when there is
    one; nin[i] = cavity elements at loop[i]. Returns the loop length, or 0 if the cavity is not a disk with one
    boundary loop. vmark / vnext are node scratch arrays, stamp a fresh value."""
    nb = 0
    H = 0
    V = 0
    first = -1
    for i in range(nce):
        e = cel[i]
        m = NV[e]
        H += m
        for k in range(m):
            v = EL[e, k]
            if vmark[v] != stamp:
                vmark[v] = stamp
                vnext[v] = -1
                V += 1
    for i in range(nce):
        e = cel[i]
        m = NV[e]
        for k in range(m):
            a, b = EL[e, k], EL[e, (k + 1) % m]
            f = _across(EL, NV, N2E, ncnt, a, b)
            if f < 0 or not inc[f]:
                if vnext[a] >= 0:
                    return 0                                  # pinched: a vertex twice on the boundary
                vnext[a] = b
                nb += 1
                first = a
    if nb == 0 or (H + nb) % 2 != 0:
        return 0
    E = (H + nb) // 2
    if V - E + nce != 1:
        return 0
    L = 0
    v = first
    while True:
        if L >= nb or L >= len(loop):
            return 0
        loop[L] = v
        L += 1
        v = vnext[v]
        if v == first:
            break
    if L != nb:
        return 0
    start = -1
    for i in range(L):
        v = loop[i]
        c = 0
        for t in range(ncnt[v]):
            if inc[N2E[v, t]]:
                c += 1
        nin[i] = c
        if c == 1 and start < 0:
            start = i
    if start > 0:
        tmp = loop[:L].copy()
        tn = nin[:L].copy()
        for i in range(L):
            loop[i] = tmp[(i + start) % L]
            nin[i] = tn[(i + start) % L]
    return L


@njit(cache=True)
def _forbidden(EL, NV, fixed, e):
    for k in range(NV[e]):
        if fixed[EL[e, k]]:
            return True
    return False


@njit(cache=True)
def _add(EL, NV, fixed, inc, cel, nce, e):
    """Add e to the cavity; returns the new count, or -1 if e is forbidden."""
    if inc[e]:
        return nce
    if _forbidden(EL, NV, fixed, e) or nce >= len(cel):
        return -1
    inc[e] = True
    cel[nce] = e
    return nce + 1


@njit(cache=True)
def _revert(inc, cel, nce, keep):
    for i in range(keep, nce):
        inc[cel[i]] = False
    return keep


@njit(cache=True)
def _convexify(EL, NV, N2E, ncnt, fixed, protect, inc, cel, nce, max_el, vnext, vmark, stamp, loop, nin):
    """Add outside elements at concave vertices until none is left (a protected vertex must not become interior).
    Returns (nce, L, stamp); L = 0 on failure (the added elements stay in cel[:nce] for the caller to revert)."""
    for _ in range(1000):
        stamp += 1
        L = boundary(EL, NV, N2E, ncnt, inc, cel, nce, vnext, vmark, stamp, loop, nin)
        if L == 0:
            return nce, 0, stamp
        added = 0
        for i in range(L):
            v = loop[i]
            nout = ncnt[v] - nin[i]
            if nout == 1:
                if protect[v]:
                    return nce, 0, stamp
                for t in range(ncnt[v]):
                    f = N2E[v, t]
                    if not inc[f]:
                        n2 = _add(EL, NV, fixed, inc, cel, nce, f)
                        if n2 < 0:
                            return nce, 0, stamp
                        nce = n2
                        added += 1
        if added == 0:
            return nce, L, stamp
        if nce > max_el:
            return nce, 0, stamp
    return nce, 0, stamp


@njit(cache=True)
def _record(EL, NV, N2E, ncnt, cel, nce, loop, nin, L, vmark, stamp, s_n, s_nc, s_sides, s_before, s_after,
            s_tri, k):
    """Snapshot k: element count, corners, side lengths, irregularity before (cavity vertices sum (deg-4)^2 +
    triangles) and after on the boundary (n_out + new n_in, new n_in = 1 at corners, 2 elsewhere)."""
    before = 0
    ntri = 0
    for i in range(nce):
        e = cel[i]
        if NV[e] == 3:
            before += 1
            ntri += 1
        for j in range(NV[e]):
            v = EL[e, j]
            if vmark[v] != stamp:
                vmark[v] = stamp
                d = ncnt[v] - 4
                before += d * d
    after = 0
    nc = 0
    last = 0
    for i in range(L):
        nout = ncnt[loop[i]] - nin[i]
        d = nout + (1 if nin[i] == 1 else 2) - 4
        after += d * d
        if nin[i] == 1:
            if nc > 0 and nc <= MAXS:
                s_sides[k, nc - 1] = i - last
            last = i
            nc += 1
    if nc > 0 and nc <= MAXS:
        s_sides[k, nc - 1] = L - last
    s_n[k] = nce
    s_nc[k] = nc
    s_before[k] = before
    s_after[k] = after
    s_tri[k] = ntri


@njit(cache=True)
def grow(EL, NV, alive, N2E, ncnt, fixed, protect, seeds, max_el, inc, vnext, vmark, stamp, cel, loop, nin,
         s_n, s_nc, s_sides, s_before, s_after, s_tri):
    """Grow a convex cavity from the elements `seeds`, one side at a time (longest first), recording a snapshot after
    every step (the cavity of snapshot k is cel[:s_n[k]]). Returns (number of snapshots, stamp). inc is clean on
    return."""
    nce = 0
    for e in seeds:
        if alive[e]:
            n2 = _add(EL, NV, fixed, inc, cel, nce, e)
            if n2 < 0:
                _revert(inc, cel, nce, 0)
                return 0, stamp
            nce = n2
    nce, L, stamp = _convexify(EL, NV, N2E, ncnt, fixed, protect, inc, cel, nce, max_el, vnext, vmark, stamp,
                               loop, nin)
    ns = 0
    while L > 0 and nce <= max_el and ns < len(s_n):
        stamp += 1
        _record(EL, NV, N2E, ncnt, cel, nce, loop, nin, L, vmark, stamp, s_n, s_nc, s_sides, s_before, s_after,
                s_tri, ns)
        ns += 1
        corners = np.flatnonzero(nin[:L] == 1)
        nc = len(corners)
        if nc == 0:
            break
        slen = np.empty(nc, dtype=np.int64)
        for i in range(nc):
            slen[i] = (corners[(i + 1) % nc] - corners[i]) % L if nc > 1 else L
        order = np.argsort(-slen, kind="mergesort")
        loop0 = loop[:L].copy()
        grown = False
        for oi in range(nc):
            si = order[oi]
            p0, ln = corners[si], slen[si]
            ok = True
            for j in range(1, ln):
                if protect[loop0[(p0 + j) % L]]:
                    ok = False
            if not ok:
                continue
            keep = nce
            cnt = nce
            if ln == 1:
                f = _across(EL, NV, N2E, ncnt, loop0[p0], loop0[(p0 + 1) % L])
                if f >= 0:
                    cnt = _add(EL, NV, fixed, inc, cel, cnt, f)
            else:
                for j in range(1, ln):
                    v = loop0[(p0 + j) % L]
                    for t in range(ncnt[v]):
                        f = N2E[v, t]
                        if not inc[f]:
                            c2 = _add(EL, NV, fixed, inc, cel, cnt, f)
                            if c2 < 0:
                                cnt = -1 - cnt
                                break
                            cnt = c2
                    if cnt < 0:
                        break
            if cnt < 0:
                _revert(inc, cel, -1 - cnt, keep)
                continue
            if cnt == keep:
                continue
            nce, L2, stamp = _convexify(EL, NV, N2E, ncnt, fixed, protect, inc, cel, cnt, max_el, vnext, vmark,
                                        stamp, loop, nin)
            if L2 == 0 or nce > max_el:
                nce = _revert(inc, cel, nce, keep)
                continue
            L = L2
            grown = True
            break
        if not grown:
            break
    _revert(inc, cel, nce, 0)
    return ns, stamp
