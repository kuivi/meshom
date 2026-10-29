"""Blocks for the parallel passes of stage t (decision B156).

partition(P, tris, block_nodes, rotate): equal-count blocks by a k-d split of the node positions (after a fixed
rotation, so a second partition has other borders), and a colouring in which two blocks of one colour are more than
HALO_RINGS triangle rings apart. A parallel pass runs the blocks of one colour at once; an operator whose patch (or a
neighbour of it) leaves its block is deferred to the next partition and, last, to a sequential sweep. Patches reach
two rings (the four flip nodes, their fans, the neighbours of those triangles), so blocks of one colour never read or
write the same triangle. Deterministic: the partition does not depend on the thread count.
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, identity

HALO_RINGS = 3       # B156: blocks of one colour are more than this many rings apart
BLOCK_NODES = 2000   # B156: nodes per block


def _kd(idx: np.ndarray, X: np.ndarray, leaf: int, out: np.ndarray, start: int) -> int:
    if len(idx) <= leaf:
        out[idx] = start
        return start + 1
    ax = int(np.argmax(X[idx].max(0) - X[idx].min(0)))
    order = idx[np.argsort(X[idx, ax], kind="stable")]
    h = len(order) // 2
    start = _kd(order[:h], X, leaf, out, start)
    return _kd(order[h:], X, leaf, out, start)


def _rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q


def partition(P: np.ndarray, tris: np.ndarray, block_nodes: int = BLOCK_NODES, rotate: int = 0):
    """(blk (N,) block per node, colours: list of block-id arrays)."""
    n = len(P)
    X = P @ _rotation(rotate).T if rotate else P
    blk = np.empty(n, np.int64)
    nb = _kd(np.arange(n), X, max(int(block_nodes), 16), blk, 0)
    # node adjacency through triangles, I + A, and the blocks within HALO_RINGS rings of each node
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    A = coo_matrix((np.ones(2 * len(e), np.int8), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                   shape=(n, n)).tocsr()
    A.data[:] = 1
    A = (A + identity(n, dtype=np.int8, format="csr")).astype(bool).astype(np.int32)
    B = csr_matrix((np.ones(n, np.int32), (np.arange(n), blk)), shape=(n, nb))
    R = B
    for _ in range(HALO_RINGS):
        R = (A @ R).astype(bool).astype(np.int32)
    C = (B.T @ R).tocoo()                                    # block i reaches a node of block j within the halo
    nbrs = [[] for _ in range(nb)]
    for i, j in zip(C.row.tolist(), C.col.tolist()):
        if i != j:
            nbrs[i].append(j)
            nbrs[j].append(i)
    col = np.full(nb, -1, np.int64)
    for b in np.argsort([-len(x) for x in nbrs], kind="stable"):  # greedy, most-constrained first
        used = {col[j] for j in nbrs[b] if col[j] >= 0}
        c = 0
        while c in used:
            c += 1
        col[b] = c
    colours = [np.flatnonzero(col == c) for c in range(int(col.max()) + 1)]
    return blk, colours


def csr(keys: np.ndarray, nb: int):
    """(ptr, items): the items 0..len(keys)-1 grouped by key (stable)."""
    order = np.argsort(keys, kind="stable")
    ptr = np.searchsorted(keys[order], np.arange(nb + 1))
    return ptr.astype(np.int64), order.astype(np.int64)
