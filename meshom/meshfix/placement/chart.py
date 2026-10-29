"""Combinatorial helpers of c_placement: element incidence, defect nodes, the integer lattice charts of the regular
parts (B130).

A lattice chart labels the nodes of a pure-quad, valence-4 region with integer (i, j) by walking across the quads:
for a CCW quad (a, b, c, d) with a, b labelled and e = L(b) - L(a) a unit step, L(d) = L(a) + rot90(e) and
L(c) = L(b) + rot90(e). In a regular region the labels are consistent; a dislocation shows as a conflict.
"""
import numpy as np
from numba import njit

from meshom.mesh import Mesh, build_topology


def node_incidence(mesh: Mesh):
    """CSR node -> elements (element ids: tris first, then quads) as (ptr, idx)."""
    n = mesh.n_nodes
    cells = [mesh.tris, mesh.quads]
    eid, nid = [], []
    off = 0
    for c in cells:
        if len(c):
            eid.append(np.repeat(np.arange(len(c)) + off, c.shape[1]))
            nid.append(c.ravel())
        off += len(c)
    eid, nid = np.concatenate(eid), np.concatenate(nid)
    o = np.argsort(nid, kind="stable")
    ptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(nid, minlength=n), out=ptr[1:])
    return ptr, eid[o]


def node_counts(mesh: Mesh):
    """(nq, nt): quads and triangles per node."""
    n = mesh.n_nodes
    nq = np.bincount(mesh.quads.ravel(), minlength=n) if len(mesh.quads) else np.zeros(n, np.int64)
    nt = np.bincount(mesh.tris.ravel(), minlength=n) if len(mesh.tris) else np.zeros(n, np.int64)
    return nq, nt


def defect_nodes(mesh: Mesh, boundary: np.ndarray) -> np.ndarray:
    """(N,) bool: interior nodes that are not 4 quads (irregular pure-quad nodes and triangle corners)."""
    nq, nt = node_counts(mesh)
    return ~boundary & ((nt > 0) | (nq != 4))


@njit(cache=True)
def _label_bfs(quads, qadj, allowed, seed_q, li, lj, lab, conflicts):
    """BFS over allowed quads from seed_q (its corner 0 at (0,0), edge 0 -> 1 = +i). qadj (Q,4) neighbour quad
    across edge k (k -> k+1) or -1. Fills li, lj, lab (node labelled flag); counts label conflicts."""
    nq = quads.shape[0]
    seen = np.zeros(nq, np.bool_)
    queue = np.empty(nq, np.int64)
    a, b = quads[seed_q, 0], quads[seed_q, 1]
    if not lab[a]:
        li[a], lj[a], lab[a] = 0, 0, True
    if not lab[b]:
        li[b], lj[b], lab[b] = li[a] + 1, lj[a], True
    head, tail = 0, 0
    queue[tail] = seed_q
    tail += 1
    seen[seed_q] = True
    while head < tail:
        q = queue[head]
        head += 1
        # find a labelled consecutive pair (k, k+1)
        kk = -1
        for k in range(4):
            u, v = quads[q, k], quads[q, (k + 1) % 4]
            if lab[u] and lab[v]:
                di, dj = li[v] - li[u], lj[v] - lj[u]
                if abs(di) + abs(dj) == 1:
                    kk = k
                    break
        if kk < 0:
            continue
        u, v = quads[q, kk], quads[q, (kk + 1) % 4]
        ei, ej = li[v] - li[u], lj[v] - lj[u]
        ri, rj = -ej, ei                      # rot90 (CCW)
        w = quads[q, (kk + 2) % 4]
        x = quads[q, (kk + 3) % 4]
        for node, ti, tj in ((w, li[v] + ri, lj[v] + rj), (x, li[u] + ri, lj[u] + rj)):
            if lab[node]:
                if li[node] != ti or lj[node] != tj:
                    conflicts[0] += 1
            else:
                li[node], lj[node], lab[node] = ti, tj, True
        for k in range(4):
            nb = qadj[q, k]
            if nb >= 0 and allowed[nb] and not seen[nb]:
                seen[nb] = True
                queue[tail] = nb
                tail += 1
    return seen


def quad_adjacency(mesh: Mesh) -> np.ndarray:
    """(Q,4): the quad across edge k (quads[:, k] -> quads[:, k+1]) or -1 (boundary or a triangle)."""
    q = mesh.quads
    nq = len(q)
    n = mesh.n_nodes
    a = q.ravel()
    b = np.roll(q, -1, axis=1).ravel()
    key = np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)
    o = np.argsort(key, kind="stable")
    ks = key[o]
    adj = -np.ones(nq * 4, np.int64)
    same = np.flatnonzero(ks[1:] == ks[:-1])
    i, j = o[same], o[same + 1]
    adj[i] = j // 4
    adj[j] = i // 4
    return adj.reshape(nq, 4)


def lattice_labels(mesh: Mesh, allowed: np.ndarray, seed_q: int, qadj=None):
    """Integer labels (li, lj, labelled, quads reached, conflicts) of the quads `allowed` reachable from seed_q."""
    qadj = quad_adjacency(mesh) if qadj is None else qadj
    n = mesh.n_nodes
    li = np.zeros(n, np.int64)
    lj = np.zeros(n, np.int64)
    lab = np.zeros(n, np.bool_)
    conf = np.zeros(1, np.int64)
    seen = _label_bfs(mesh.quads.astype(np.int64), qadj, allowed.astype(np.bool_), int(seed_q), li, lj, lab, conf)
    return li, lj, lab, seen, int(conf[0])

