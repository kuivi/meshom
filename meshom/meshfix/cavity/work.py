"""Mutable mixed mesh for cavity remeshing: optimize._Work plus new nodes (decisions B134).

optimize._Work keeps the node -> element table, replaces element sets and undoes them, but a patch may only use the
nodes it removes. A cavity fill creates its own interior nodes, so Work adds `add_nodes` (the node arrays grow by
doubling) and `replace` (remove the cavity's elements, append the fill's nodes and cells; the undo record also kills
the new nodes).
"""
import numpy as np

from meshom.mesh import Mesh
from meshom.optimize import _Work


class Work(_Work):
    def __init__(self, mesh: Mesh, fixed: np.ndarray | None = None):
        super().__init__(mesh, fixed)
        self.n = mesh.n_nodes                               # nodes in use (P[: self.n])
        self.node_flag = np.asarray(mesh.node_flag).copy()

    def _grow_nodes(self, k: int) -> None:
        if self.n + k <= len(self.P):
            return
        cap = 2 * len(self.P) + k

        def g(a, fill):
            out = np.full((cap,) + a.shape[1:], fill, dtype=a.dtype)
            out[: len(a)] = a
            return out

        self.P = g(self.P, 0.0)
        self.inc = g(self.inc, -1)
        self.ninc = g(self.ninc, 0)
        self.alive = g(self.alive, False)
        self.fixed = g(self.fixed, False)
        self.bnode = g(self.bnode, False)
        self.touched = g(self.touched, False)
        self._loc = g(self._loc, -1)
        self.node_flag = g(self.node_flag, 0)
        if getattr(self, "art", None) is not None:          # the artificial mask grows with the nodes (new: False)
            self.art = g(self.art, False)

    def add_nodes(self, X: np.ndarray) -> np.ndarray:
        """Append the unit vectors X (k,3) as free interior nodes; their ids."""
        k = len(X)
        self._grow_nodes(k)
        ids = np.arange(self.n, self.n + k, dtype=np.int64)
        self.P[ids] = X
        self.alive[ids] = True
        self.n += k
        return ids

    def replace(self, rm: list, X: np.ndarray, cells: list, coast_new=()):
        """Remove the elements rm, add nodes X (cells refer to them as -1 - j for X[j]), add the cells; coast_new: the
        refs (-1 - j) of new nodes on the mesh boundary (marked boundary and fixed). The mesh-boundary edge set
        follows (a coast cavity may merge or split coast edges). Returns the undo record or None (a node would exceed
        the node -> element table width)."""
        old_b = set()
        for e in rm:
            c = self.cells(e)
            for i in range(len(c)):
                k = (min(c[i], c[(i + 1) % len(c)]), max(c[i], c[(i + 1) % len(c)]))
                if k in self.bedges:
                    old_b.add(k)
        ids = self.add_nodes(X) if len(X) else np.zeros(0, dtype=np.int64)
        new = [tuple(int(ids[-1 - v]) if v < 0 else int(v) for v in c) for c in cells]
        rec = self.apply(rm, new)
        if rec is None:
            self.alive[ids] = False
            self.n -= len(ids)
            return None
        cn = np.array([ids[-1 - v] for v in coast_new], dtype=np.int64)
        self.bnode[cn] = True
        self.fixed[cn] = True
        self.node_flag[cn] = 1
        new_b = set()
        if old_b:
            cnt = {}
            for c in new:
                for i in range(len(c)):
                    k = (min(c[i], c[(i + 1) % len(c)]), max(c[i], c[(i + 1) % len(c)]))
                    cnt[k] = cnt.get(k, 0) + 1
            new_b = {k for k, m in cnt.items() if m == 1 and len(self.edge_elems(*k)) == 1}
            self.bedges.difference_update(old_b)
            self.bedges.update(new_b)
        return rec, ids, new, old_b, new_b, cn

    def undo_replace(self, r) -> None:
        rec, ids, _, old_b, new_b, cn = r
        self.undo(rec)
        self.alive[ids] = False
        self.bnode[cn] = False
        self.fixed[cn] = False
        self.node_flag[cn] = 0
        self.bedges.difference_update(new_b)
        self.bedges.update(old_b)
        if len(ids) and ids[-1] == self.n - 1:
            self.n = int(ids[0])

    def alive_elems(self) -> np.ndarray:
        return np.flatnonzero(self.nv[: self.m] > 0)

    def to_mesh(self):
        """(compact mesh, old id per new node); node_flag of new nodes is 0."""
        live = self.nv[: self.m] > 0
        el, nv = self.el[: self.m][live], self.nv[: self.m][live]
        used = np.zeros(len(self.P), dtype=bool)
        used[el[el >= 0]] = True
        nid = np.cumsum(used) - 1
        tris = nid[el[nv == 3, :3]].reshape(-1, 3)
        quads = nid[el[nv == 4]].reshape(-1, 4)
        old = np.flatnonzero(used)
        return Mesh(self.P[old].copy(), tris.astype(np.int64), quads.astype(np.int64),
                    self.node_flag[old].astype(np.int8), None), old
