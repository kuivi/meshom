"""The coast loop of e_coast (decisions B142-B145): carry defects near the coast to it and absorb them there.

For every defect seed within `reach_h` h of the reference coast (a triangle, an interior pure-quad node with 3 or 5+
quads, a quad with a corner above SEED_DEG), worst first, the candidates are:

1. operators (ops.py, B145): per triangle the half-chord collapse (the quad strip from one of its edges to a coast
   edge, or to a second triangle, removed, its rungs merged) and split (the grid line from one of its corners to a
   coast node, or to a second triangle's tip, cut open and filled with a quad column); per interior 3-5 pair the
   collapse from the pair's edge (the pair becomes one triangle, the coast takes the other half); each relaxed with a
   collar of `collar_op` rings;
2. coast cavities (D's cavity.grow with coast=True; only loops holding a coast edge), grown from the seed and from
   `n_corr` corridors (chord walks of <= `corr_len` quads to the coast); per cavity the coast fills (fills.py, B143:
   D's patterns with the coast run of a side re-spaced to the opposite side's or the h-ideal count);
3. every candidate relaxed (relax.py, B144: D's energy plus sliding coast nodes, projected onto the reference coast
   at the end) and measured on its zone;
4. acceptance (B145) = D's rule (B137: nothing inverted, size budgets and caps, quads > 120 not more, the defect
   cost and the worst quad angle not up, J down by >= MIN_GAIN_J) plus: budgets over the gates' _coast region too,
   every reference coast vertex near the zone within COAST_SHIFT_H h of the new coast edges (or no farther than
   before), no triangle cluster >= 3. The best accepted candidate (lowest J) is applied.

The defect cost counts interior nodes only (a defect that reaches the coast is gone; any coast valence is allowed);
a triangle with a coast node costs w_coast_tri. No candidate changes the quad or triangle count of an artificial node
(B121, art_counts_ok).

The fast engine (B148) is D's (cavity/engine.py, B138-B140): read sets, contenders chosen when the seed is retired,
a cache of unchanged seeds, spawned workers evaluating seeds ahead and results retired in seed order (the serial
answer for every worker count); the relaxation is the Numba coast energy (cavity/ncoast.py, B147).
"""
import os
import pickle
import shutil
import time

import numpy as np

from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh
from meshom.meshfix.cavity import cavity as cv
from meshom.meshfix.cavity.engine import QUAD_WORST_MIN
from meshom.meshfix.cavity.engine import Engine as DEngine
from meshom.meshfix.cavity.engine import _Pool as _DPool
from meshom.meshfix.cavity.nrelax import edges_of, elem_pairs
from meshom.meshfix.cavity.work import Work
from meshom.meshfix.coast import fills as fl
from meshom.meshfix.coast.refcoast import dist_to_edges
from meshom.quality.kernel import element_geometry

SEED_DEG = 110.0
RATIO_JUMP = 1.15
TRI_MAX_DEG = 100.0
MIN_GAIN_J = 1.0
LNSH_CAP = 0.30
RATIO_CAP = 1.6
COAST_SHIFT_H = 0.25     # B145: the coast rule (a coast node may leave the coast by 0.25 h)
SLIDE_REACH_H = 1.5      # B144: a sliding node may use the reference segments within this arc length (in h)

DEFAULTS = dict(reach_h=6.0, corr_len=10, n_corr=2, max_elems=300, max_steps=10, max_cav=3, collar=2, passes=8,
                max_iter=300, max_seeds=None, verbose=False, budget_share=0.004, budget_share_coast=0.006, recount=20,
                layouts=None, split=True, max_adj=6, adj_lnmax=0.2, same=True, workers=1, K=16, w_coast_tri=0.5,
                ops=True, pair_ops=True, cavity=True, op_len=12, collar_op=3, w_wall=3000.0,
                lnsh_budget=None, worst_floor=0.0, relax="numba", cache=True, window=None, inflight=1)


class CWork(Work):
    """D's Work (its artificial mask grows with the node arrays)."""


# ------------------------------------------------------------------------------------- local evaluation (workers)

_G = {}


def init_worker(h_path, h, tol: float, max_iter: int, w_ct: float = 0.5, w_wall: float = 200.0,
                thr: float | None = None) -> None:
    import os
    import pickle

    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    if h is None:
        with open(h_path, "rb") as f:
            h = pickle.load(f)
    _G.update(h=h, tol=tol, max_iter=max_iter, w_ct=w_ct, w_wall=w_wall, thr=thr)


def measure_local(X, quads, tris, free, bnd, h, tol, w_coast_tri: float = 0.5, thr: float | None = None) -> dict:
    """D's zone measures (cavity/engine.py measure_local, B137; copied): the defect cost counts the free interior
    nodes (coast nodes never count); a triangle with a coast node costs w_coast_tri instead of 1 (B145: a triangle
    at the coast does less harm than one in the open)."""
    m = Mesh(X, tris, quads, np.zeros(len(X), np.int8))
    g = element_geometry(m)
    qa = np.degrees(g.quad_angles)
    qmax = qa.max(1) if len(quads) else np.zeros(0)
    ta = np.degrees(g.tri_angles)
    inv = int((g.quad_area <= 0).sum() + (g.tri_area <= 0).sum() + (qa >= 180).any(1).sum())
    Ed = edges_of(quads, tris)
    L = R_EARTH_M * 2 * np.arcsin(np.minimum(np.linalg.norm(X[Ed[:, 0]] - X[Ed[:, 1]], axis=1) / 2, 1))
    mid = X[Ed[:, 0]] + X[Ed[:, 1]]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    r_e = np.abs(np.log(L / h(mid)))
    thr = tol if thr is None else thr
    bad = int((r_e > thr).sum())
    me = np.concatenate([g.quad_edges.mean(1), g.tri_edges.mean(1)]) if len(tris) else g.quad_edges.mean(1)
    pr = elem_pairs(quads, tris, len(X))
    rat = np.maximum(me[pr[:, 0]], me[pr[:, 1]]) / np.minimum(me[pr[:, 0]], me[pr[:, 1]]) if len(pr) else \
        np.ones(1)
    jump = int((rat > RATIO_JUMP).sum())
    ce = np.r_[bnd[quads].any(1), bnd[tris].any(1)] if len(tris) else bnd[quads].any(1)   # coast elements
    bad_c = int(((r_e > thr) & _coast_edge_mask(quads, tris, ce, Ed, len(X), free)).sum())
    jump_c = int(((rat > RATIO_JUMP) & (ce[pr[:, 0]] | ce[pr[:, 1]])).sum()) if len(pr) else 0
    nq = np.bincount(quads.ravel(), minlength=len(X)) if len(quads) else np.zeros(len(X), np.int64)
    nt = np.bincount(tris.ravel(), minlength=len(X)) if len(tris) else np.zeros(len(X), np.int64)
    sel = free & ~bnd & (nt == 0)
    tc = bnd[tris].any(1) if len(tris) else np.zeros(0, bool)
    cost = float((~tc).sum() + w_coast_tri * tc.sum() + np.abs(nq[sel] - 4).sum())
    worst = float(qmax.max()) if len(qmax) else 0.0
    r = dict(n110=int((qmax > 110).sum()), n120=int((qmax > 120).sum()), worst=worst,
             tri_worst=float(ta.max()) if len(tris) else 0.0, cost=cost, bad=bad, jump=jump, inv=inv,
             bad_c=bad_c, jump_c=jump_c,
             n_tri=len(tris), lnsh_max=float(r_e.max()), ratio_max=float(rat.max()))
    r["J"] = 2 * cost + r["n120"] + 0.25 * r["n110"] + 0.1 * worst + 0.5 * (bad + jump)
    return r


def _coast_edge_mask(quads, tris, ce, Ed, n, free=None) -> np.ndarray:
    """(len(Ed),) edges of the coast elements ce (the gates' _coast region: elements with a boundary node) that are
    no boundary edges (size_fidelity's coast subset leaves the boundary edges out): edges in one element only are
    left out (in a zone: those with a free end, the zone's coast edges; free None: all of them)."""
    allc = [c for c in (quads, tris) if len(c)]
    ea = np.sort(np.concatenate([np.stack([c, np.roll(c, -1, 1)], -1).reshape(-1, 2) for c in allc]), axis=1)
    ka, cnt = np.unique(ea[:, 0] * n + ea[:, 1], return_counts=True)
    kE = Ed[:, 0] * n + Ed[:, 1]
    single = np.isin(kE, ka[cnt == 1])
    if free is not None:
        single &= free[Ed[:, 0]] | free[Ed[:, 1]]
    cells = [(quads, ce[: len(quads)])] + ([(tris, ce[len(quads):])] if len(tris) else [])
    keys = []
    for c, m in cells:
        c = c[m]
        if len(c):
            e = np.sort(np.stack([c, np.roll(c, -1, 1)], -1).reshape(-1, 2), axis=1)
            keys.append(e[:, 0] * n + e[:, 1])
    if not keys:
        return np.zeros(len(Ed), bool)
    return np.isin(kE, np.concatenate(keys)) & ~single


def zone_bias(X, quads, tris, h) -> float:
    """Median ln(s/h) of a zone's edges (the mesh's own offset from h there)."""
    Ed = edges_of(quads, tris)
    mid = X[Ed[:, 0]] + X[Ed[:, 1]]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    return float(np.median(np.log(np.linalg.norm(X[Ed[:, 0]] - X[Ed[:, 1]], axis=1) * R_EARTH_M / h(mid))))


def coast_edges_local(quads, tris, free) -> np.ndarray:
    """Edges of a zone with one element in it and a free end: the mesh's coast edges there (every element at a free
    node is in the zone)."""
    e = [np.stack([c, np.roll(c, -1, 1)], -1).reshape(-1, 2) for c in (quads, tris) if len(c)]
    e = np.sort(np.concatenate(e), axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    u = u[cnt == 1]
    return u[free[u[:, 0]] | free[u[:, 1]]]


def _relax_fn(kind: str):
    """B147: "numba" the Numba coast energy and L-BFGS (cavity/ncoast.py, default), "scipy" the Numba energy with
    scipy's L-BFGS-B, "jax" relax.py (the original)."""
    if kind == "jax":
        from meshom.meshfix.coast.relax import relax
        return relax
    from functools import partial

    from meshom.meshfix.cavity.ncoast import relax
    return partial(relax, solver="scipy") if kind == "scipy" else relax


def eval_job(job):
    """Relax (coast sliding) and measure one candidate zone; -> (metrics, positions)."""
    X0, quads, tris, free, bnd, slide, SA, SB = job
    h, tol = _G["h"], _G["tol"]
    Xr, _ = _relax_fn(_G.get("relax", "jax"))(X0, free, quads, tris, h, tol, slide, SA, SB, max_iter=_G["max_iter"],
                                              w_wall=_G["w_wall"])
    return measure_local(Xr, quads, tris, free, bnd, h, tol, _G.get("w_ct", 0.5), _G.get("thr")), Xr


def screen(quads, tris, free, bnd, old, w_coast_tri: float) -> str:
    """B147 (D's B139 screen with E's cost): reasons a candidate fails acceptance from its topology alone: the
    defect cost above the zone's ("screen_cost"), or J cannot fall by MIN_GAIN_J even at the best geometry (no quad
    above 110 / 120, no bad edge or jump, the worst quad at 90 deg) ("screen_gain"). Exact: reject() would refuse
    it whatever the relaxation gives (both reasons are not budget reasons, so it is no contender either)."""
    n = len(free)
    nq = np.bincount(quads.ravel(), minlength=n) if len(quads) else np.zeros(n, np.int64)
    nt = np.bincount(tris.ravel(), minlength=n) if len(tris) else np.zeros(n, np.int64)
    sel = free & ~bnd & (nt == 0)
    tc = bnd[tris].any(1) if len(tris) else np.zeros(0, bool)
    cost = float((~tc).sum() + w_coast_tri * tc.sum() + np.abs(nq[sel] - 4).sum())
    if cost > old["cost"]:
        return "screen_cost"
    if 2 * cost + (0.1 * QUAD_WORST_MIN if len(quads) else 0.0) >= old["J"] - MIN_GAIN_J:
        return "screen_gain"
    return ""


def art_counts_ok(W, rm, cells) -> bool:
    """B121 / B148: replacing the elements rm by cells keeps the quad count and the triangle count of every
    artificial node (the artificial boundary is no defect sink). cells refer to new nodes as -1 - j (never
    artificial)."""
    art = W.art
    d = {}
    for e in rm:
        k = 0 if W.nv[e] == 4 else 1
        for v in W.cells(e):
            if art[v]:
                d.setdefault(v, [0, 0])[k] -= 1
    for c in cells:
        k = 0 if len(c) == 4 else 1
        for v in c:
            if v >= 0 and art[v]:
                d.setdefault(v, [0, 0])[k] += 1
    return all(x == [0, 0] for x in d.values())


# ------------------------------------------------------------------------------------------------- worker process

def _worker_main(wid: int, inbox, outbox, init: dict) -> None:
    """A worker: its own CoastEngine (copy of the working mesh); ("op", bytes) applies a committed change,
    ("eval", i, seed, totals) evaluates a seed, ("stop",) ends (cavity.engine._worker_main's protocol)."""
    from meshom.compete import _load
    from meshom.meshfix.coast.refcoast import RefCoast

    try:
        with open(init["h_path"], "rb") as f:
            h = pickle.load(f)
        mesh = _load(init["mesh"])
        art = np.load(init["art"])
        ref = RefCoast.from_segments(np.load(init["seg"]))
        eng = CoastEngine(mesh, h, init["tol"], art, ref, init["o"], None, budget=init["budget"])
        mesh = None
        import gc

        gc.collect()
        outbox.put(("ready", wid, None, None))
        while True:
            msg = inbox.get()
            if msg[0] == "op":
                eng.apply_op(pickle.loads(msg[1]))
            elif msg[0] == "eval":
                _, i, seed, tot = msg
                outbox.put(("res", wid, i, eng.evaluate(seed, tot)))
            else:
                return
    except BaseException as e:
        import traceback

        outbox.put(("error", wid, None, traceback.format_exc()))
        raise e


class _Pool(_DPool):
    """D's worker pool (cavity.engine._Pool) with the coast worker as the process target."""

    def __init__(self, n: int, init: dict):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.out = ctx.Queue()
        self.inb = [ctx.Queue() for _ in range(n)]
        env = {k: os.environ.get(k) for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                               "OPENBLAS_NUM_THREADS")}
        for k in env:
            os.environ[k] = "1"
        try:
            self.procs = [ctx.Process(target=_worker_main, args=(w, self.inb[w], self.out, init), daemon=True)
                          for w in range(n)]
            for p in self.procs:
                p.start()
        finally:
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.busy = [0] * n
        for _ in range(n):
            m = self.get()
            assert m[0] == "ready", m


_SKIP = "skip"
_PH = ("t_ops", "t_grow", "t_fill", "t_job", "t_relax", "t_measure", "t_shift")


class CoastEngine(DEngine):
    """The coast loop (module docstring) on D's fast engine (B148): evaluate() returns the contenders of a seed
    (every candidate that passes all acceptance rules but the global budgets) with its read set; the main process
    retires the seeds in order and commits the best contender that fits the budgets at that moment and makes no
    triangle cluster; seeds are evaluated ahead by spawned workers, and a seed whose read set did not change keeps
    its evaluation (cache), so every worker count gives the serial answer.

    The per-pass set of tried candidates (a candidate already tried by an earlier seed of the pass is not tried
    again) is applied when a seed is retired: its contenders that an earlier retired seed of the pass produced are
    dropped, then its own candidate keys are added."""

    def __init__(self, mesh: Mesh, h, tol: float, art: np.ndarray, ref, o: dict, h_path=None, budget=None):
        self.mesh0 = mesh
        self.W = CWork(mesh)
        self.W.art = np.zeros(len(self.W.P), dtype=bool)
        self.W.art[: len(art)] = art
        self.art0 = np.asarray(art, dtype=bool)
        self.h, self.tol, self.o, self.ref = h, tol, {**DEFAULTS, **o}, ref
        o = self.o
        self.h_path = h_path
        self.hV = h(ref.V)
        self.stats = dict(seeds=0, seeds_with_cavity=0, cavities=0, cavities_coast=0, candidates=0, accepted=0,
                          by_kind={}, acc=[], t_eval=0.0, seeds_skipped=0, seeds_cached=0, evaluations=0, redo=0,
                          why={}, reject={}, corridors=0, recounts=0, t_recount=0.0, t_seeds=0.0, t_commit=0.0,
                          t_wait=0.0, t_start=0.0, **{k: 0.0 for k in _PH})
        if budget is None:
            self._global_counts(mesh)
        else:
            self.tot, self.budget = dict(budget[0]), dict(budget[1])
        init_worker(None, h, tol, o["max_iter"], o["w_coast_tri"], o["w_wall"], o["lnsh_budget"])
        _G["relax"] = o["relax"]
        self._relax = _relax_fn(o["relax"])
        self.pool = None
        self.version = 0
        self.node_ver = np.zeros(len(self.W.P), dtype=np.int64)
        self.cache = {}
        self.tried = set()

    def _global_counts(self, mesh):
        t = time.perf_counter()
        X = mesh.points
        Ed = edges_of(mesh.quads, mesh.tris)
        L = R_EARTH_M * np.linalg.norm(X[Ed[:, 0]] - X[Ed[:, 1]], axis=1)
        mid = X[Ed[:, 0]] + X[Ed[:, 1]]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        thr = self.o["lnsh_budget"] or self.tol
        r_all = np.abs(np.log(L / self.h(mid)))
        bad = int((r_all > thr).sum())
        g = element_geometry(mesh)
        me = np.concatenate([g.quad_edges.mean(1), g.tri_edges.mean(1)]) if mesh.n_tris else g.quad_edges.mean(1)
        pr = elem_pairs(mesh.quads, mesh.tris, len(X))
        jmask = np.maximum(me[pr[:, 0]], me[pr[:, 1]]) / np.minimum(me[pr[:, 0]], me[pr[:, 1]]) > RATIO_JUMP
        jump = int(jmask.sum())
        from meshom.mesh import build_topology

        bn = build_topology(mesh).boundary_node
        ce = np.r_[bn[mesh.quads].any(1), bn[mesh.tris].any(1)] if mesh.n_tris else bn[mesh.quads].any(1)
        cm = _coast_edge_mask(mesh.quads, mesh.tris, ce, Ed, len(X))
        bad_c = int(((r_all > thr) & cm).sum())
        jump_c = int((jmask & (ce[pr[:, 0]] | ce[pr[:, 1]])).sum())
        self.tot = {"bad": bad, "jump": jump, "bad_c": bad_c, "jump_c": jump_c}
        self.stats["t_recount"] += time.perf_counter() - t
        if hasattr(self, "budget"):                       # a recount (B145): the running totals drift, since the
            self.stats["recounts"] += 1                   # zones measured before and after an operation differ
            return
        sh, shc = self.o["budget_share"], self.o["budget_share_coast"]
        self.budget = {"bad": max(bad, int(sh * len(Ed))), "jump": max(jump, int(sh * len(pr))),
                       "bad_c": max(bad_c, int(shc * cm.sum())),
                       "jump_c": max(jump_c, int(shc * (ce[pr[:, 0]] | ce[pr[:, 1]]).sum()))}
        self.stats.update(input_bad=bad, input_jump=jump, input_bad_c=bad_c, input_jump_c=jump_c,
                          budget=dict(self.budget), n_edges=int(len(Ed)))

    # ------------------------------------------------------------------------------------------- zone
    def is_coast_edge(self, a: int, b: int) -> bool:
        W = self.W
        return W.is_bedge(a, b) and not (W.art[a] or W.art[b])

    def zone(self, free_nodes):
        """(node ids, local quads, tris, free mask, boundary mask, local slide ids, SA, SB): the elements at the
        free nodes; every live non-artificial node is free, coast nodes slide."""
        W = self.W
        F = sorted(v for v in free_nodes if W.alive[v] and not W.art[v] and W.ninc[v] > 0)
        E = sorted({e for v in F for e in W.at(v)})
        nodes = np.unique(W.el[E][W.el[E] >= 0])
        W._loc[nodes] = np.arange(len(nodes))
        el, nv = W.el[E], W.nv[E]
        quads = W._loc[el[nv == 4]].reshape(-1, 4)
        tris = W._loc[el[nv == 3, :3]].reshape(-1, 3)
        free = np.zeros(len(nodes), dtype=bool)
        free[W._loc[np.asarray(F, dtype=np.int64)]] = True
        W._loc[nodes] = -1
        bnd = W.bnode[nodes].copy()
        slide = np.flatnonzero(free & bnd)
        X = W.P[nodes]
        K = int(self.o["K"])
        if len(slide):
            hs = self.h(X[slide])
            segs = self.ref.allowed_many(X[slide], SLIDE_REACH_H * hs, K)
            SA, SB = self.ref.A[segs], self.ref.B[segs]
        else:
            SA = SB = np.zeros((0, K, 3))
        return nodes, quads, tris, free, bnd, slide, SA, SB

    def job(self, S, lp, X=None, cells=None, cn=()):
        """The zone of a cavity candidate (X None: the cavity as it is), prepared and undone: (job, node ids)."""
        W = self.W
        if X is None:
            inner = {v for e in S for v in W.cells(e)} - set(lp)
            z = self.zone(self.collar_nodes(lp, inner))
            return (W.P[z[0]].copy(),) + z[1:], z[0]
        r = W.replace(sorted(S), X, cells, cn)
        if r is None:
            return None, None
        try:
            z = self.zone(self.collar_nodes(lp, r[1]))
            return (W.P[z[0]].copy(),) + z[1:], z[0]
        finally:
            W.undo_replace(r)

    def job_op(self, rm, X, cells, cn, moves):
        """Zone of an operator's replacement (collar rings around its cells), with the moved nodes' positions."""
        W = self.W
        r = W.replace(sorted(rm), X, cells, cn)
        if r is None:
            return None, None
        try:
            core = {v for c in r[2] for v in c}
            z = self.zone(W.rings(core, self.o["collar_op"]))
            X0 = W.P[z[0]].copy()
            for v, x in moves.items():
                k = int(np.searchsorted(z[0], v))
                if k < len(z[0]) and z[0][k] == v:
                    X0[k] = x
            return (X0,) + z[1:], z[0]
        finally:
            W.undo_replace(r)

    # ------------------------------------------------------------------------------------------- acceptance
    def reject(self, old, new, tot: dict | None = None, budget: bool = True) -> str:
        """The first reason (B145 order) to reject new against old; budget=False skips the global budgets (the only
        rules that depend on the running totals)."""
        tot = self.tot if tot is None else tot
        if new["inv"] > 0:
            return "inv"
        if budget:
            for k in ("bad", "jump", "bad_c", "jump_c"):
                if tot[k] + new[k] - old[k] > self.budget[k]:
                    return k
        if new["lnsh_max"] > max(LNSH_CAP, old["lnsh_max"]):
            return "lnsh_cap"
        if new["ratio_max"] > max(RATIO_CAP, old["ratio_max"]):
            return "ratio_cap"
        if new["n120"] > old["n120"]:
            return "n120"
        if new["tri_worst"] > max(old["tri_worst"], TRI_MAX_DEG):
            return "tri_worst"
        if new["cost"] > old["cost"]:
            return "cost"
        if new["worst"] > max(old["worst"], self.o["worst_floor"]):
            return "worst"
        if new["J"] < old["J"] - MIN_GAIN_J:
            return ""
        return "gain"

    def coast_shift_ok(self, jold, Xr, jnew) -> tuple[bool, float]:
        """Reference vertices near the zones' coast nodes: within COAST_SHIFT_H h of the new zone's coast edges, or
        no farther than from the old zone's (B145); also the largest shift in h."""
        Xo, qo, to, fo = jold[0], jold[1], jold[2], jold[3]
        qn, tn, fn = jnew[1], jnew[2], jnew[3]
        Eo = coast_edges_local(qo, to, fo)
        En = coast_edges_local(qn, tn, fn)
        if len(En) == 0 and len(Eo) == 0:
            return True, 0.0
        pts = np.vstack([Xo[np.unique(Eo)] if len(Eo) else np.zeros((0, 3)),
                         Xr[np.unique(En)] if len(En) else np.zeros((0, 3))])
        hp = self.h(pts) / R_EARTH_M
        vid = self.ref.vertices_near(pts, 0.75 * hp)
        if len(vid) == 0:
            return True, 0.0
        V = self.ref.V[vid]
        hv = self.hV[vid] / R_EARTH_M
        dn = dist_to_edges(V, Xr, En)
        do = dist_to_edges(V, Xo, Eo)
        lim = np.maximum(COAST_SHIFT_H * hv, do + 1.0 / R_EARTH_M)
        return bool((dn <= lim).all()), float((dn / hv).max())

    # ------------------------------------------------------------------------------------------- seeds
    def coast_dist_h(self, X: np.ndarray) -> np.ndarray:
        """Distance of X to the reference coast in units of h there."""
        _, d = self.ref.nearest(X)
        return d * R_EARTH_M / self.h(X)

    def seeds(self):
        """(severity, kind, seed) worst first, seed = (kind, seed elements, coast distance in h, seed id): the
        triangles, interior pure-quad nodes != 4 and quads > SEED_DEG within reach_h h of the reference coast."""
        W = self.W
        out = []
        E = W.alive_elems()
        el, nv = W.el[E], W.nv[E]
        q = E[nv == 4]
        m = Mesh(W.P[: W.n], np.zeros((0, 3), np.int64), W.el[q], np.zeros(W.n, np.int8))
        qmax = np.degrees(element_geometry(m).quad_angles).max(1)
        worst_at = np.zeros(W.n)
        np.maximum.at(worst_at, W.el[q].ravel(), np.repeat(qmax, 4))
        cnt_q = np.bincount(el[nv == 4].ravel(), minlength=W.n)[: W.n]
        cnt_t = np.bincount(el[nv == 3, :3].ravel(), minlength=W.n)[: W.n]
        tri = E[nv == 3]
        irr = np.flatnonzero((cnt_t == 0) & (cnt_q > 0) & (cnt_q != 4) & ~W.bnode[: W.n] & W.alive[: W.n])
        bq = q[qmax > SEED_DEG]
        cand = [("tri", int(e), W.P[W.el[e, :3]].mean(0), float(worst_at[W.el[e, :3]].max())) for e in tri]
        cand += [("node", int(v), W.P[v], float(worst_at[v])) for v in irr]
        cand += [("quad", int(e), W.P[W.el[e]].mean(0), float(a)) for e, a in zip(bq, qmax[qmax > SEED_DEG])]
        if not cand:
            return out
        X = np.array([c[2] for c in cand])
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        dh = self.coast_dist_h(X)
        for (k, i, _, sev), d in zip(cand, dh):
            if d > self.o["reach_h"]:
                continue
            es = tuple(sorted(W.at(i))) if k == "node" else (i,)
            out.append((sev, k, (k, es, float(d), i)))
        out.sort(key=lambda x: -x[0])
        return out

    # ------------------------------------------------------------------------------------------- candidates
    def has_coast_edge(self, e: int) -> bool:
        c = self.W.cells(e)
        return any(self.is_coast_edge(c[i], c[(i + 1) % len(c)]) for i in range(len(c)))

    def corridors(self, seed_elems, reads: set) -> list[list[int]]:
        """Straight quad rows (chord walks) from the seed's edges to an element with a coast edge, shortest first;
        reads collects the elements looked at."""
        W, L = self.W, int(self.o["corr_len"])
        walks = []
        seed = set(seed_elems)
        for e0 in seed_elems:
            c0 = W.cells(e0)
            for i in range(len(c0)):
                a, b = c0[i], c0[(i + 1) % len(c0)]
                path, prev, edge = [], e0, (a, b)
                for _ in range(L):
                    nb = [x for x in W.edge_elems(*edge) if x != prev]
                    reads.update(nb)
                    if len(nb) != 1:
                        break
                    e = nb[0]
                    if e in seed:
                        break
                    path.append(e)
                    if self.has_coast_edge(e):
                        walks.append(path)
                        break
                    c = W.cells(e)
                    if len(c) != 4:
                        break
                    j = c.index(edge[0])
                    k = c.index(edge[1])
                    if (j + 1) % 4 == k:
                        edge = (c[(j + 3) % 4], c[(j + 2) % 4])
                    else:
                        edge = (c[(j + 1) % 4], c[(j + 2) % 4])
                    prev = e
        walks.sort(key=len)
        return walks[: int(self.o["n_corr"])]

    def cavities(self, seed_elems, st: dict, reads: set):
        W, o = self.W, self.o
        starts = [set(seed_elems)]
        for p in self.corridors(seed_elems, reads):
            st["corridors"] += 1
            starts.append(set(seed_elems) | set(p))
        seen = set()
        for s0 in starts:
            n = 0
            for S, lp, cr in cv.grow(W, s0, o["max_elems"], o["max_steps"], st["why"], True, reads):
                st["cavities"] += 1
                key = frozenset(S)
                if key in seen:
                    continue
                seen.add(key)
                if not any(self.is_coast_edge(lp[i], lp[(i + 1) % len(lp)]) for i in range(len(lp))):
                    continue
                st["cavities_coast"] += 1
                yield S, lp, cr
                n += 1
                if n >= o["max_cav"]:
                    break

    def _nq(self, v: int) -> int:
        return sum(1 for e in self.W.at(v) if self.W.nv[e] == 4)

    def _irregular(self, v: int) -> bool:
        W = self.W
        return all(W.nv[e] == 4 for e in W.at(v)) and W.ninc[v] != 4

    def op_candidates(self, kind, seed_elems, sid, st: dict, vis: set, Rn: set, keys: list) -> list:
        """The half-chord operators (ops.py) of a triangle seed (collapse each edge, split from each corner) or of a
        3-5 pair at a node seed; each candidate as (rm, old, name, X, cells, cn, job, nodes, old job, key)."""
        from meshom.meshfix.coast import ops

        W, o = self.W, self.o
        tries = []
        if kind == "tri":
            t = seed_elems[0]
            tries = [(name, f, t, i) for i in range(3) for name, f in (("collapse", ops.collapse),
                                                                        ("split", ops.split))]
        elif kind == "node" and o["pair_ops"]:
            v = sid
            vis.add(v)
            for e in sorted(W.at(v)):
                c = W.cells(e)
                if len(c) != 4:
                    continue
                i = c.index(v)
                for j in (i, (i - 1) % 4):                # the edges (v, next) and (prev, v) of this quad
                    a, b = c[j], c[(j + 1) % 4]
                    u = b if a == v else a
                    vis.add(u)
                    if W.bnode[u] or not self._irregular(u) or self._nq(u) + self._nq(v) != 8:
                        continue
                    tries.append(("pairtri", ops.collapse, e, j))
        out = []
        for name, f, t, i in tries:
            r = f(W, t, i, self.is_coast_edge, int(o["op_len"]), reads=vis)
            if r is None:
                continue
            rm, X, cells, cn, moves, _ = r
            key = ("op", name, tuple(rm))
            if key in keys:
                continue
            keys.append(key)
            if not art_counts_ok(W, rm, cells):           # B121: no defect into the artificial boundary
                kd = st["reject"].setdefault("op" + name, {})
                kd["art"] = kd.get("art", 0) + 1
                continue
            jn, nd = self.job_op(rm, X, cells, cn, moves)
            if jn is None:
                continue
            Rn.update(int(v) for v in nd if v < W.n)
            core = {v for e in rm for v in W.cells(e)}
            z = self.zone(W.rings(core, o["collar_op"]))
            Rn.update(z[0].tolist())
            jo = (W.P[z[0]].copy(),) + z[1:]
            old = measure_local(jo[0], jo[1], jo[2], jo[3], jo[4], self.h, self.tol, o["w_coast_tri"],
                               o["lnsh_budget"])
            nm = name + ("pair" if not cn and name != "pairtri" else "")
            out.append((sorted(rm), old, f"op{nm}_L{len(X) - 1}", X, cells, cn, jn, nd, jo, key))
        return out

    # ------------------------------------------------------------------------------------------- evaluation
    def evaluate(self, seed, tot: dict) -> dict:
        """Every candidate of one seed at the current mesh (operators, coast cavities and their fills), relaxed and
        measured, the mesh left as it was. Returns {cands: the contenders (candidates that pass every rule but the
        global budgets, the coast-shift rule included) in candidate order, each with its op (apply_op), measures
        and key; keys: every candidate key the seed produced (the pass's tried set); R: the read set; st}. tot only
        labels the reject statistics."""
        kind, es, dist_h, sid = seed
        W, o = self.W, self.o
        st = dict(cavities=0, cavities_coast=0, candidates=0, seeds_with_cavity=0, corridors=0, why={}, reject={},
                  **{k: 0.0 for k in _PH})
        reads, vis, Rn, keys = set(), set(), set(), []
        n0 = W.n
        if any(W.nv[e] == 0 for e in es):                   # a re-evaluation dispatched after a commit removed the
            return dict(cands=[], keys=[], R=np.zeros(0, np.int64), n=int(n0), st=st)   # seed: skipped at retire
        cavs = []
        t0 = time.perf_counter()
        for e in es:
            vis.update(W.cells(e))
        if o["ops"]:
            cavs += self.op_candidates(kind, list(es), sid, st, vis, Rn, keys)
        t1 = time.perf_counter()
        st["t_ops"] += t1 - t0
        if o["cavity"]:
            inner_t = 0.0
            for S, lp, cr in self.cavities(list(es), st, reads):
                ta = time.perf_counter()
                key = frozenset(S)
                if key in keys:
                    continue
                keys.append(key)
                jb, nodes = self.job(S, lp)
                Rn.update(nodes.tolist())
                old = measure_local(jb[0], jb[1], jb[2], jb[3], jb[4], self.h, self.tol, o["w_coast_tri"],
                                   o["lnsh_budget"])
                tb = time.perf_counter()
                cands = fl.coast_fills(W.P, lp, cr, self.h, self.is_coast_edge, self.ref, layouts=o.get("layouts"),
                                       split=o["split"], max_adj=o["max_adj"], adj_lnmax=o["adj_lnmax"],
                                       same=o["same"], bias=zone_bias(jb[0], jb[1], jb[2], self.h))
                tc = time.perf_counter()
                for name, X, cells, cn in cands:
                    if not art_counts_ok(W, S, cells):
                        continue
                    j, nd = self.job(S, lp, X, cells, cn)
                    if j is not None:
                        Rn.update(int(v) for v in nd if v < n0)
                        cavs.append((sorted(S), old, name, X, cells, cn, j, nd, jb, key))
                td = time.perf_counter()
                st["t_job"] += (tb - ta) + (td - tc)
                st["t_fill"] += tc - tb
                inner_t += td - ta
            st["t_grow"] += time.perf_counter() - t1 - inner_t
        if reads:
            E = np.fromiter(reads, dtype=np.int64)
            Rn.update(W.el[E][W.el[E] >= 0].tolist())
        for v in vis:                                       # the stars the operators looked at
            Rn.add(v)
            for e in W.at(v):
                Rn.update(W.cells(e))
        R = np.fromiter((v for v in Rn if v < n0), dtype=np.int64)
        out = dict(cands=[], keys=keys, R=R, n=int(n0), st=st)
        if not cavs:
            return out
        st["seeds_with_cavity"] = 1
        st["candidates"] = len(cavs)
        for c in cavs:
            rm, old, name, X, cells, cn, j, nd, jb, key = c
            kd = st["reject"].setdefault(name.split("_")[0], {})
            scr = screen(j[1], j[2], j[3], j[4], old, o["w_coast_tri"])
            if scr:
                kd[scr] = kd.get(scr, 0) + 1
                continue
            ta = time.perf_counter()
            Xr, _ = self._relax(j[0], j[3], j[1], j[2], self.h, self.tol, j[5], j[6], j[7], max_iter=o["max_iter"],
                                w_wall=o["w_wall"])
            tb = time.perf_counter()
            met = measure_local(Xr, j[1], j[2], j[3], j[4], self.h, self.tol, o["w_coast_tri"], o["lnsh_budget"])
            tc = time.perf_counter()
            st["t_relax"] += tb - ta
            st["t_measure"] += tc - tb
            why = self.reject(old, met, tot)
            if not why or why in ("bad", "jump", "bad_c", "jump_c"):
                if why and self.reject(old, met, budget=False):
                    why = self.reject(old, met, budget=False)
                else:
                    ok, sh = self.coast_shift_ok(jb, Xr, j)
                    met["shift_h"] = sh
                    if not ok:
                        why = "coast_shift"
                st["t_shift"] += time.perf_counter() - tc
            kd[why or "ok"] = kd.get(why or "ok", 0) + 1
            if why and why not in ("bad", "jump", "bad_c", "jump_c"):
                continue
            fi = np.flatnonzero(j[3])
            nf = nd[fi]
            rel = np.where(nf >= n0, -1 - (nf - n0), nf)
            out["cands"].append(dict(op=(list(rm), X, cells, list(cn), rel, Xr[fi]), name=name, met=met, old=old,
                                     key=key, n_elems=len(rm), dist_h=dist_h))
        return out

    # ------------------------------------------------------------------------------------------- retire
    def _retire(self, key, r) -> int:
        """Commit the best contender of a valid result at the current totals (1), or remember the seed (0). The
        contenders an earlier retired seed of this pass produced are dropped first (the tried set)."""
        st = self.stats
        cands = [c for c in r["cands"] if c["key"] not in self.tried]
        self.tried.update(r["keys"])
        order = sorted(range(len(cands)), key=lambda i: (cands[i]["met"]["J"], i))
        W = self.W
        for i in order:
            c = cands[i]
            why = self.reject(c["old"], c["met"])
            if why:
                kd = st["reject"].setdefault(c["name"].split("_")[0], {})
                kd["retire_" + why] = kd.get("retire_" + why, 0) + 1
                continue
            rm, X, cells, cn, _, _ = c["op"]
            rr = W.replace(rm, X, cells, cn)
            if rr is None:
                continue
            tri_new = [e for e in rr[0][1] if W.nv[e] == 3]
            bad = bool(tri_new) and W.cluster_excess(tri_new) > 0
            W.undo_replace(rr)
            if bad:
                kd = st["reject"].setdefault(c["name"].split("_")[0], {})
                kd["cluster"] = kd.get("cluster", 0) + 1
                continue
            t = time.perf_counter()
            self._commit(c)
            st["t_commit"] += time.perf_counter() - t
            return 1
        if self.o["cache"] and key is not None:
            self.cache[key] = (r["R"], self.version, r["cands"], r["keys"])
        return 0

    def _commit(self, best) -> None:
        st, o = self.stats, self.o
        op = best["op"]
        self.apply_op(op)
        if self.pool is not None:
            self.pool.broadcast(pickle.dumps(op, protocol=pickle.HIGHEST_PROTOCOL))
        met, old, name = best["met"], best["old"], best["name"]
        for k in ("bad", "jump", "bad_c", "jump_c"):
            self.tot[k] += met[k] - old[k]
        kind = name.split("_")[0]
        st["accepted"] += 1
        st["by_kind"][kind] = st["by_kind"].get(kind, 0) + 1
        from meshom.geo import xyz_to_lonlat

        W = self.W
        nodes = np.array(op[4], dtype=np.int64)
        nodes = nodes[nodes >= 0]
        cpos = W.P[nodes].mean(0) if len(nodes) else W.P[W.cells(W.alive_elems()[0])[0]]
        lon, lat = xyz_to_lonlat(cpos[None] / np.linalg.norm(cpos))
        st["acc"].append(dict(name=name, dist_h=best["dist_h"], lon=float(lon[0]), lat=float(lat[0]),
                              elems=best["n_elems"], worst0=old["worst"], worst1=met["worst"], cost0=old["cost"],
                              cost1=met["cost"], n120_0=old["n120"], n120_1=met["n120"], n_tri0=old["n_tri"],
                              n_tri1=met["n_tri"], shift_h=met.get("shift_h", 0.0)))
        if o["verbose"] > 1:
            print(f"  accept {name}: |rm| {best['n_elems']} worst {old['worst']:.1f} -> {met['worst']:.1f} cost "
                  f"{old['cost']:.0f} -> {met['cost']:.0f} n120 {old['n120']} -> {met['n120']} tri "
                  f"{old['n_tri']} -> {met['n_tri']}", flush=True)
        if st["accepted"] % o["recount"] == 0:
            self._global_counts(W.to_mesh()[0])

    def _merge(self, s: dict) -> None:
        st = self.stats
        st["evaluations"] += 1
        for k in ("cavities", "cavities_coast", "candidates", "seeds_with_cavity", "corridors"):
            st[k] += s[k]
        for k, v in s["why"].items():
            st["why"][k] = st["why"].get(k, 0) + v
        for kind, d in s["reject"].items():
            kd = st["reject"].setdefault(kind, {})
            for k, v in d.items():
                kd[k] = kd.get(k, 0) + v
        for k in _PH:
            st[k] += s[k]
        st["t_eval"] += s["t_relax"] + s["t_measure"]

    @staticmethod
    def _key(seed):
        kind, es, _, sid = seed
        return (kind, sid, tuple(es))

    def _skip(self, seed) -> bool:
        W = self.W
        es = seed[1]
        return any(W.nv[e] == 0 for e in es) or any(W.touched[v] for e in es for v in W.cells(e))

    def _cached(self, key):
        c = self.cache.get(key)
        if c is None:
            return None
        if len(c[0]) and int(self.node_ver[c[0]].max()) > c[1]:
            del self.cache[key]
            return None
        return c

    # ------------------------------------------------------------------------------------------- passes
    def _pass_serial(self, sd) -> int:
        st, acc = self.stats, 0
        for _, _, seed in sd:
            if self._skip(seed):
                st["seeds_skipped"] += 1
                continue
            st["seeds"] += 1
            key = self._key(seed)
            c = self._cached(key) if self.o["cache"] else None
            if c is not None:
                st["seeds_cached"] += 1
                r = dict(cands=c[2], keys=c[3], R=c[0], version=c[1])
            else:
                r = self.evaluate(seed, dict(self.tot))
                self._merge(r["st"])
                r.update(version=self.version)
            acc += self._retire(key, r)
        return acc

    def _pass_parallel(self, sd) -> int:
        """cavity.engine.Engine._pass_parallel for coast seeds: seeds evaluated ahead (at most `window`), retired
        in seed order, a result made stale by a commit evaluated again at once (the nearest first)."""
        import heapq

        st, o, P = self.stats, self.o, self.pool
        n = len(sd)
        window = int(o.get("window") or 8 * len(P.procs))
        keys = [self._key(x[2]) for x in sd]
        res = [None] * n
        fly = [False] * n
        ver = [0] * n
        evald = [False] * n
        ahead = 0
        redo = []
        nxt = ret = acc = 0

        def check(i):
            r = res[i]
            if r is None or r is _SKIP:
                return
            if self._skip(sd[i][2]):
                res[i] = _SKIP
            elif self._stale(r):
                st["redo"] += 1
                res[i] = None
                heapq.heappush(redo, i)

        while ret < n:
            while True:
                w = P.free_worker(o["inflight"])
                if w is None:
                    break
                if redo:
                    i = heapq.heappop(redo)
                else:
                    while nxt < n and ahead < window:
                        if self._skip(sd[nxt][2]):
                            res[nxt] = _SKIP
                        elif o["cache"] and (c := self._cached(keys[nxt])) is not None:
                            res[nxt] = dict(cands=c[2], keys=c[3], R=c[0], version=c[1], cached=True)
                        else:
                            break
                        nxt += 1
                    if nxt >= n or ahead >= window:
                        break
                    i = nxt
                    nxt += 1
                    evald[i] = True
                    ahead += 1
                ver[i] = self.version
                fly[i] = True
                P.send(w, ("eval", i, sd[i][2], dict(self.tot)))
            while ret < n and res[ret] is not None:
                r = res[ret]
                if r is _SKIP or self._skip(sd[ret][2]):
                    st["seeds_skipped"] += 1
                    ahead -= evald[ret]
                    ret += 1
                    continue
                if self._stale(r):
                    check(ret)
                    break
                st["seeds"] += 1
                if r.get("cached"):
                    st["seeds_cached"] += 1
                a = self._retire(keys[ret], r)
                acc += a
                ahead -= evald[ret]
                ret += 1
                if a:
                    for k in range(ret, nxt):
                        check(k)
            if ret >= n:
                break
            if res[ret] is None and not fly[ret] and P.free_worker(o["inflight"]) is not None:
                continue
            if res[ret] is None:
                t = time.perf_counter()
                m = P.get()
                st["t_wait"] += time.perf_counter() - t
                _, w, i, r = m
                P.busy[w] -= 1
                fly[i] = False
                self._merge(r["st"])
                r.update(version=ver[i])
                res[i] = r
                check(i)
        return acc

    def _start_pool(self) -> None:
        from meshom.compete import _save
        from meshom.meshfix.cases import meshfix_dir

        t = time.perf_counter()
        d = meshfix_dir() / "tmp" / f"e_{os.getpid()}_{int(time.time() * 1e3)}"
        d.mkdir(parents=True, exist_ok=True)
        self._tmp = d
        _save(self.mesh0, d / "mesh.npz")
        np.save(d / "art.npy", self.art0)
        np.save(d / "seg.npy", np.stack([self.ref.A, self.ref.B], axis=1))
        h_path = self.h_path
        if h_path is None:
            h_path = d / "h.pkl"
            with open(h_path, "wb") as f:
                pickle.dump(self.h, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._warm()
        o = dict(self.o, workers=1, verbose=False)
        self.pool = _Pool(int(self.o["workers"]), dict(h_path=str(h_path), mesh=str(d / "mesh.npz"),
                                                       art=str(d / "art.npy"), seg=str(d / "seg.npy"),
                                                       tol=self.tol, o=o,
                                                       budget=(dict(self.tot), dict(self.budget))))
        self.stats["t_start"] += time.perf_counter() - t

    def _warm(self) -> None:
        """Compile the Numba kernels once (cache=True) before the workers load them."""
        from meshom.meshfix.cavity import ncoast

        P = np.array([[1.0, 0, 0], [0.999, 0.01, 0], [0.999, 0.01, 0.01], [0.999, 0, 0.01]])
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        SA, SB = P[None, None, 0], P[None, None, 1]
        ncoast.relax(P, np.array([True, True, False, False]), np.array([[0, 1, 2, 3]]), np.zeros((0, 3), int),
                     lambda x: np.full(len(x), 1e4), 0.2, np.array([0]), SA, SB, max_iter=2)

    def run(self):
        o, st = self.o, self.stats
        st["passes"] = []
        try:
            if o["workers"] > 1:
                self._start_pool()
            for p in range(o["passes"]):
                W = self.W
                W.touched[:] = False
                self.tried = set()
                t = time.perf_counter()
                if p:
                    self._global_counts(W.to_mesh()[0])
                sd = self.seeds()
                if o["max_seeds"]:
                    sd = sd[: o["max_seeds"]]
                if not o["cavity"]:
                    sd = [x for x in sd if x[1] == "tri" or (x[1] == "node" and o["pair_ops"])]
                st["t_seeds"] += time.perf_counter() - t
                e0, c0 = st["evaluations"], st["seeds_cached"]
                acc = self._pass_parallel(sd) if self.pool is not None else self._pass_serial(sd)
                st["passes"].append(dict(seeds=len(sd), accepted=acc, evaluated=st["evaluations"] - e0,
                                         cached=st["seeds_cached"] - c0, s=time.perf_counter() - t))
                if o["verbose"]:
                    print(f"pass {p}: {len(sd)} seeds, {st['evaluations'] - e0} evaluated, "
                          f"{st['seeds_cached'] - c0} cached, {acc} accepted, {time.perf_counter() - t:.1f} s, "
                          f"reject {st['reject']}", flush=True)
                if acc == 0:
                    break
        finally:
            if self.pool is not None:
                self.pool.close()
                self.pool = None
                shutil.rmtree(self._tmp, ignore_errors=True)
        return self.W.to_mesh()[0]
