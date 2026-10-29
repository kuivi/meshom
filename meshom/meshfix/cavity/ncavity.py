"""Numba kernel of the cavity convexification (cavity.convexify; decisions B134, B139).

The same rule as cavity.convexify_py: every node of S whose outside elements leave a fill angle other than 90 or 180
takes them into S, repeated until none is left; stops at max_elems (status 1) or at a mesh-boundary node (status 2;
with coast=True only an artificial one). Element and node marks live in scratch arrays kept on the Work (reset on
exit), so a call costs only the cavity it visits.
"""
import numpy as np
from numba import njit
from numba.typed import List


@njit(cache=True)
def convexify_kernel(el, nv, inc, ninc, bnode, art, seed, max_elems, coast, emark, nstamp, stamp):
    """(status 0 ok / 1 size / 2 boundary, S as an array (the elements looked at, = S), last stamp)."""
    S = List()
    for e in seed:
        if emark[e] == 0:
            emark[e] = 1
            S.append(e)
    status = 0
    add = List()
    add.append(np.int64(0))
    add.pop()
    for _ in range(10_000):
        if len(S) > max_elems:
            status = 1
            break
        stamp += 1
        add = List()
        add.append(np.int64(0))
        add.pop()
        for e in S:
            for k in range(nv[e]):
                v = el[e, k]
                if nstamp[v] == stamp:
                    continue
                nstamp[v] = stamp
                if bnode[v]:
                    if (not coast) or art[v]:
                        status = 2
                        break
                    continue
                ang = 0
                nout = 0
                for t in range(ninc[v]):
                    f = inc[v, t]
                    if emark[f] != 1:
                        ang += 90 if nv[f] == 4 else 60
                        nout += 1
                if nout == 0:
                    continue
                fill = 360 - ang
                if fill == 90 or fill == 180:
                    continue
                for t in range(ninc[v]):
                    f = inc[v, t]
                    if emark[f] == 0:
                        emark[f] = 2
                        add.append(f)
            if status:
                break
        if status or len(add) == 0:
            break
        for f in add:
            emark[f] = 1
            S.append(f)
        add = List()
        add.append(np.int64(0))
        add.pop()
    out = np.empty(len(S), dtype=np.int64)
    for i in range(len(S)):
        out[i] = S[i]
        emark[S[i]] = 0
    for f in add:
        emark[f] = 0
    return status, out, stamp


def scratch(W):
    """(element marks, node stamps) of W, grown with its arrays."""
    c = getattr(W, "_cvx", None)
    if c is None or len(c[0]) < len(W.nv) or len(c[1]) < len(W.P):
        c = [np.zeros(len(W.nv), dtype=np.int8), np.zeros(len(W.P), dtype=np.int64), 0]
        W._cvx = c
    return c


@njit(cache=True)
def worst_at(quads, qmax, n):
    """(n,) the largest quad corner (deg) over the quads at each node (np.maximum.at, B139)."""
    out = np.zeros(n)
    for k in range(len(quads)):
        a = qmax[k]
        for i in range(4):
            v = quads[k, i]
            if a > out[v]:
                out[v] = a
    return out
