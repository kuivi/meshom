"""Hybrid Q-zip EV/VE templates written from Feng et al. (2021), Fig. 14.

Templates return complete disk replacements, with initial spherical positions.
They do not commit intermediate edge collapses/splits to a mesh.
"""
import numpy as np

from meshom.optimize import _Work, _half_edges


class Work(_Work):
    """Reserve node slots once; new-node trials do not copy a global mesh."""

    def __init__(self, mesh, fixed=None, reserve=1024):
        super().__init__(mesh, fixed)
        self.next_node = mesh.n_nodes
        self.capacity = mesh.n_nodes + reserve
        self.P = np.vstack([self.P, np.zeros((reserve, 3))])
        self.inc = np.vstack([self.inc, np.full((reserve, self.K), -1, np.int64)])
        self.ninc = np.r_[self.ninc, np.zeros(reserve, np.int64)]
        self.node_flag = np.r_[self.node_flag, np.zeros(reserve, np.int8)]
        self.fixed = np.r_[self.fixed, np.zeros(reserve, bool)]
        self.bnode = np.r_[self.bnode, np.zeros(reserve, bool)]
        self.alive = np.r_[self.alive, np.zeros(reserve, bool)]
        self.touched = np.r_[self.touched, np.zeros(reserve, bool)]
        self._loc = np.r_[self._loc, np.full(reserve, -1, np.int64)]

    def new_node(self, xyz):
        # Reserve is only a storage chunk, never a numerical rejection limit.
        if self.next_node == self.capacity:
            extra = self.capacity
            self.P = np.vstack([self.P, np.zeros((extra, 3))])
            self.inc = np.vstack([self.inc, np.full((extra, self.K), -1, np.int64)])
            self.ninc = np.r_[self.ninc, np.zeros(extra, np.int64)]
            self.node_flag = np.r_[self.node_flag, np.zeros(extra, np.int8)]
            self.fixed = np.r_[self.fixed, np.zeros(extra, bool)]
            self.bnode = np.r_[self.bnode, np.zeros(extra, bool)]
            self.alive = np.r_[self.alive, np.zeros(extra, bool)]
            self.touched = np.r_[self.touched, np.zeros(extra, bool)]
            self._loc = np.r_[self._loc, np.full(extra, -1, np.int64)]
            self.capacity += extra
        v = self.next_node
        self.next_node += 1
        self.P[v] = xyz
        self.alive[v] = True
        return v

    def discard_node(self, v):
        assert self.ninc[v] == 0 and v == self.next_node - 1
        self.alive[v] = False
        self.next_node -= 1

    def apply(self, rm, new, moves=None, zone=False):
        """optimize._Work.apply; the undo record also keeps the node -> element rows of every node it changes, so undo
        restores them in their order (B150: an evaluation leaves the mesh exactly as it was, so the answer of a
        later evaluation does not depend on which trials ran before)."""
        nodes = {v for e in rm for v in self.el[e, : self.nv[e]].tolist()}
        nodes.update(v for c in new for v in c)
        rows = [(v, self.inc[v, : self.ninc[v]].copy()) for v in nodes]
        rec = super().apply(rm, new, moves, zone)
        return None if rec is None else (*rec, rows)

    def undo(self, rec):
        super().undo(rec[:4])
        for v, row in rec[4]:
            self.inc[v, : len(row)] = row

    def mark(self, rec):
        super().mark(rec[:4])

    def fit(self, M, h, tol, fit_opts, kind="numba"):
        """optimize._Work.fit (size_fit.smooth_fit with only the free nodes M moving, on the elements at the nodes of
        the elements at M); kind "numba": the same round in one Numba call (nfit.fit_sub, B151), "smooth_fit": the
        original."""
        if kind != "numba":
            return super().fit(M, h, tol, fit_opts)
        from .nfit import fit_sub

        M = sorted(v for v in M if self.alive[v] and not self.fixed[v])
        if not M:
            return {}
        act = sorted({e for v in M for e in self.at(v)})
        ring = {v for e in act for v in self.cells(e)}
        E = sorted({e for v in ring for e in self.at(v)})
        nodes = np.unique(self.el[E][self.el[E] >= 0])
        self._loc[nodes] = np.arange(len(nodes))
        el, nv = self.el[E], self.nv[E]
        tl, ql = self._loc[el[nv == 3, :3]], self._loc[el[nv == 4]]
        mloc = self._loc[np.asarray(M)]
        self._loc[nodes] = -1
        fixed = np.ones(len(nodes), dtype=bool)
        fixed[mloc] = False
        o = {k: v for k, v in fit_opts.items() if k != "rounds"}
        assert fit_opts.get("rounds", 1) == 1
        out, _ = fit_sub(self.P[nodes], tl.reshape(-1, 3), ql.reshape(-1, 4), fixed, h, tol, **o)
        res = {}
        for v, i in zip(M, mloc.tolist()):
            res[v] = out[i].copy()
            self.P[v] = out[i]
        return res

    def patch_ok(self, rm, new):
        old = [self.cells(e) for e in rm]
        if any(len(c) not in (3, 4) or len(set(c)) != len(c) for c in new):
            return False
        ho, hn = _half_edges(old), _half_edges(new)
        so, sn = set(ho), set(hn)
        if len(sn) != len(hn):
            return False
        if {e for e in ho if e[::-1] not in so} != {e for e in hn if e[::-1] not in sn}:
            return False
        vo, vn = {v for c in old for v in c}, {v for c in new for v in c}
        if any(self.ninc[v] != 0 or self.fixed[v] for v in vn - vo):
            return False
        rms = set(rm)
        for v in vo - vn:
            if self.fixed[v] or self.bnode[v] or any(e not in rms for e in self.at(v)):
                return False
        eo = {tuple(sorted(e)) for e in ho}
        en = {tuple(sorted(e)) for e in hn}
        if len(vo) - len(eo) + len(old) != len(vn) - len(en) + len(new):
            return False
        for a, b in en - eo:
            if any(e not in rms for e in self.edge_elems(a, b)):
                return False
        return True


def collapse_triangle_edge(work, t, k):
    """EV: zip a triangle base into a vertex; the opposite quad becomes a tri."""
    tri = work.cells(t)
    a, b = tri[k], tri[(k + 1) % 3]
    if work.fixed[a] or work.fixed[b] or work.bnode[a] or work.bnode[b]:
        return None
    across = [e for e in work.edge_elems(a, b) if e != t]
    if len(across) != 1 or work.nv[across[0]] != 4:
        return None
    rm = sorted(set(work.at(a)) | set(work.at(b)))
    if any(work.zone[e] for e in rm):
        return None
    new = []
    for e in rm:
        if e == t:
            continue
        c = tuple(b if v == a else v for v in work.cells(e))
        c = tuple(c[i] for i in range(len(c)) if c[i] != c[i - 1])
        if len(c) not in (3, 4) or len(set(c)) != len(c):
            return None
        new.append(c)
    xyz = work.P[a] + work.P[b]
    xyz /= np.linalg.norm(xyz)
    return rm, tuple(new), {b: xyz}


def split_triangle_tip(work, t, tip, across, fresh):
    """VE: unzip a v5 triangle tip along an opposite edge (tip, across).

    fresh is an unused reserved node id. The star is partitioned by traversing
    its two oriented edge fans; no compass derived from longitude is needed.
    """
    tri = work.cells(t)
    if tip not in tri or work.bnode[tip] or work.fixed[tip] or work.ninc[tip] != 5:
        return None
    star = work.at(tip)
    if sum(work.nv[e] == 3 for e in star) != 1 or any(work.zone[e] for e in star):
        return None
    k = tri.index(tip)
    a, b = tri[(k + 1) % 3], tri[(k + 2) % 3]
    # At the tip, traverse face wedges from b counterclockwise to a. Splitting
    # at the opposite wedge edge leaves exactly two quads on either side.
    wedges = {}
    for e in star:
        if e == t:
            continue
        c = work.cells(e)
        j = c.index(tip)
        wedges[c[(j + 1) % len(c)]] = (c[j - 1], e)
    cur, left, seen = b, [], set()
    while cur != across:
        if cur not in wedges or cur in seen:
            return None
        seen.add(cur)
        cur, e = wedges[cur]
        left.append(e)
    if len(left) != 2:
        return None
    new = [(a, b, tip, fresh), (fresh, tip, across)]
    for e in star:
        if e == t:
            continue
        new.append(tuple(fresh if v == tip and e not in left else v for v in work.cells(e)))
    # Separate the split ends across the old triangle base by half its width.
    d = (work.P[b] - work.P[a]) * 0.5
    xleft, xright = work.P[tip] - d, work.P[tip] + d
    xleft /= np.linalg.norm(xleft)
    xright /= np.linalg.norm(xright)
    return sorted(star), tuple(new), {fresh: xleft, tip: xright}
