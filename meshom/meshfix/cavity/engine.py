"""The cavity remeshing loop (method d_cavity; decisions B134-B137, the fast engine B138-B140).

For every defect seed (an interior pure-quad node with 3 or 5+ quads, a triangle, a quad with a corner above
SEED_DEG), worst first:

1. grow grid cavities around it (cavity.grow: convexified disks with 4 corners, up to max_elems elements, at most
   max_cav of them);
2. per cavity, every pattern (patterns.fills: the regular grid, a ladder with its column-end triangles in several
   layouts, the split of a turn cavity, coast-side adjustments) and, with polish=True, the identity (the cavity's
   own elements), each followed by the relaxation (relax.py; nrelax.py the same energy in Numba, B139) of the fill
   and a collar of `collar` element rings (free nodes only: the coast and the artificial boundary stay);
3. the zone of a candidate = the elements at its free nodes; measured: quads > 110 / > 120 deg, the worst quad
   angle, the triangles' worst angle, the defect cost (sum over the zone's free nodes of |quads - 4| at pure-quad
   nodes, plus 1 per triangle), edges with |ln s/h| > tol, element pairs with mean-edge ratio > RATIO_JUMP,
   inverted / reflex elements;
4. accept the best candidate (lowest J = 2 cost + n120 + 0.25 n110 + 0.1 worst + 0.5 (bad + jump)) when, against
   the zone as it is (B137): nothing inverted; the whole mesh's bad edges and size jumps stay within their budgets
   (budget_share of the edges / element pairs, or the input's count if higher: the gates are p99); no zone edge
   above LNSH_CAP and no pair above RATIO_CAP (unless the input already had it); quads > 120 deg not more; the
   triangles' worst angle not above max(input's, TRI_MAX_DEG); the defect cost and the worst quad angle not up; and
   J lower by >= MIN_GAIN_J. The identity is allowed only with polish=True.

Passes repeat while something is accepted (<= passes).

The fast engine (B140) computes the same result as the serial loop above, faster:

- read sets: the evaluation of a seed (steps 1-3) reads only the mesh near it: the elements the cavity search looked
  at (cavity.grow's `reads`), the collar and the zone nodes. Every committed change bumps a version stamp on the
  nodes it touches (the nodes of the removed and the new cells, the moved nodes). An evaluation is valid at a later
  state when no node of its read set changed since;
- contenders: the evaluation returns every candidate that passes all rules of step 4 but the global budget (the
  only rule that depends on the rest of the mesh, through the running totals); the accepted one is chosen when the
  seed is retired, at the totals of that moment (select);
- screen: a candidate whose topology alone fails step 4 (defect cost up, or J cannot fall by MIN_GAIN_J even at the
  best geometry) is not relaxed;
- cache (`cache`): a seed whose evaluation accepted nothing keeps its contenders and is not evaluated again in a later
  pass while its read set is unchanged (it would give the same answer);
- workers (`workers` > 1): processes, each with its own copy of the working mesh, evaluate seeds ahead of the
  current one (at most `inflight` seeds per worker); the main process retires the results strictly in seed order
  (worst first), as the serial loop would: a skipped seed (touched in this pass) is skipped, a result whose read set
  changed since it was evaluated is evaluated again, an accepted candidate is committed and its change (`op`) is
  sent to every worker, which applies it before its next evaluation. Element and node ids stay equal in every copy
  (the same changes in the same order).
"""
import os
import pickle
import shutil
import time

import numpy as np

from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh
from meshom.meshfix.cavity import cavity as cv
from meshom.meshfix.cavity import patterns as pt
from meshom.meshfix.cavity.nrelax import edges_of, elem_pairs
from meshom.meshfix.cavity.work import Work
from meshom.quality.kernel import element_geometry

SEED_DEG = 110.0
RATIO_JUMP = 1.15
TRI_MAX_DEG = 100.0
MIN_GAIN_J = 1.0        # B137: the zone's J must fall by at least this

LNSH_CAP = 0.30          # B137: no zone edge beyond this |ln s/h| (the reported limit for p99)
RATIO_CAP = 1.6          # B137: no zone element pair beyond this mean-edge ratio (the gate's max is 2.0)

DEFAULTS = dict(max_elems=300, max_steps=12, max_cav=5, collar=2, passes=10, polish=False, max_iter=300,
                max_seeds=None, verbose=False, budget_share=0.004, layouts=None, split=True,
                coast=True, max_adj=4, workers=1, final=None, cache=True, relax="numba", inflight=1,
                window=None)

_PHASES = ("t_grow", "t_fill", "t_job", "t_relax", "t_measure")

# ------------------------------------------------------------------------------------- local evaluation (workers)

_G = {}


def init_worker(h_path, h, tol: float, max_iter: int) -> None:
    """Process globals of the evaluation: h (unpickled from h_path in a worker), tol, max_iter."""
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    if h is None:
        with open(h_path, "rb") as f:
            h = pickle.load(f)
    _G.update(h=h, tol=tol, max_iter=max_iter)


def measure_local(X, quads, tris, free, bnd, h, tol) -> dict:
    """The zone measures of the module docstring on a local mesh (free / bnd: node masks); the defect cost counts
    the free non-boundary nodes, whose whole star is in the zone."""
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
    bad = int((r_e > tol).sum())
    me = np.concatenate([g.quad_edges.mean(1), g.tri_edges.mean(1)]) if len(tris) else g.quad_edges.mean(1)
    pr = elem_pairs(quads, tris, len(X))
    rat = np.maximum(me[pr[:, 0]], me[pr[:, 1]]) / np.minimum(me[pr[:, 0]], me[pr[:, 1]]) if len(pr) else \
        np.ones(1)
    jump = int((rat > RATIO_JUMP).sum())
    nq = np.bincount(quads.ravel(), minlength=len(X)) if len(quads) else np.zeros(len(X), np.int64)
    nt = np.bincount(tris.ravel(), minlength=len(X)) if len(tris) else np.zeros(len(X), np.int64)
    sel = free & ~bnd & (nt == 0)
    cost = float(len(tris) + np.abs(nq[sel] - 4).sum())
    worst = float(qmax.max()) if len(qmax) else 0.0
    r = dict(n110=int((qmax > 110).sum()), n120=int((qmax > 120).sum()), worst=worst,
             tri_worst=float(ta.max()) if len(tris) else 0.0, cost=cost, bad=bad, jump=jump, inv=inv,
             n_tri=len(tris), lnsh_max=float(r_e.max()), ratio_max=float(rat.max()))
    r["J"] = 2 * cost + r["n120"] + 0.25 * r["n110"] + 0.1 * worst + 0.5 * (bad + jump)
    return r


def zone_cost(quads, tris, free, bnd) -> float:
    """measure_local's defect cost (topology only)."""
    n = len(free)
    nq = np.bincount(quads.ravel(), minlength=n) if len(quads) else np.zeros(n, np.int64)
    nt = np.bincount(tris.ravel(), minlength=n) if len(tris) else np.zeros(n, np.int64)
    sel = free & ~bnd & (nt == 0)
    return float(len(tris) + np.abs(nq[sel] - 4).sum())


QUAD_WORST_MIN = 89.9   # B139: the worst corner of a (convex, spherical) quad is >= 90 deg (corner sum >= 360)


def screen(quads, tris, free, bnd, old) -> str:
    """B139: reasons a candidate fails acceptance before any relaxation, from its topology alone: its defect cost
    is above the zone's ("cost"), or even the best possible geometry (no quad above 110 / 120 deg, no bad edge or
    size jump, the worst quad at 90 deg) cannot lower J by MIN_GAIN_J ("gain"); "" otherwise. Exact: such a
    candidate is rejected by Engine.reject whatever the relaxation gives."""
    cost = zone_cost(quads, tris, free, bnd)
    if cost > old["cost"]:
        return "screen_cost"
    if 2 * cost + (0.1 * QUAD_WORST_MIN if len(quads) else 0.0) >= old["J"] - MIN_GAIN_J:
        return "screen_gain"
    return ""


def _relax_fn(kind: str):
    """The relaxation (B139): "numba" the Numba energy and L-BFGS (default), "scipy" the Numba energy with
    scipy's L-BFGS-B, "jax" relax.py (the original)."""
    if kind == "jax":
        from meshom.meshfix.cavity.relax import relax
        return relax
    from functools import partial

    from meshom.meshfix.cavity.nrelax import relax
    return partial(relax, solver="scipy") if kind == "scipy" else relax


def eval_job(job):
    """Relax and measure one candidate zone: job = (X0, quads, tris, free, bnd); -> (metrics, positions)."""
    X0, quads, tris, free, bnd = job
    h, tol = _G["h"], _G["tol"]
    Xr, _ = _relax_fn(_G.get("relax", "jax"))(X0, free, quads, tris, h, tol, max_iter=_G["max_iter"])
    return measure_local(Xr, quads, tris, free, bnd, h, tol), Xr


# ------------------------------------------------------------------------------------------------- worker process

def _worker_main(wid: int, inbox, outbox, init: dict) -> None:
    """A worker: its own Engine (copy of the working mesh); messages ("op", bytes) apply a committed change,
    ("eval", i, seed elements, totals) evaluate a seed, ("stop",) ends."""
    from meshom.compete import _load

    try:
        with open(init["h_path"], "rb") as f:
            h = pickle.load(f)
        mesh = _load(init["mesh"])
        art = np.load(init["art"])
        eng = Engine(mesh, h, init["tol"], art, init["o"], None, budget=init["budget"])
        eng.mesh0 = mesh = None                               # the copy lives in eng.W only
        import gc

        gc.collect()
        outbox.put(("ready", wid, None, None))
        while True:
            msg = inbox.get()
            if msg[0] == "op":
                eng.apply_op(pickle.loads(msg[1]))
            elif msg[0] == "eval":
                _, i, es, tot = msg
                outbox.put(("res", wid, i, eng.evaluate(es, tot)))
            else:
                return
    except BaseException as e:                                # report and die (the main process raises)
        import traceback

        outbox.put(("error", wid, None, traceback.format_exc()))
        raise e


class _Pool:
    """Worker processes with one inbox each (ops are broadcast, evaluations go to one worker) and one outbox."""

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

    def get(self):
        import queue

        while True:
            try:
                m = self.out.get(timeout=30)
            except queue.Empty:
                dead = [w for w, p in enumerate(self.procs) if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"cavity workers died: {dead}")
                continue
            if m[0] == "error":
                raise RuntimeError(f"cavity worker {m[1]} failed:\n{m[3]}")
            return m

    def broadcast(self, payload: bytes) -> None:
        for q in self.inb:
            q.put(("op", payload))

    def send(self, w: int, msg) -> None:
        self.busy[w] += 1
        self.inb[w].put(msg)

    def free_worker(self, limit: int):
        w = int(np.argmin(self.busy))
        return w if self.busy[w] < limit else None

    def close(self) -> None:
        for q in self.inb:
            q.put(("stop",))
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


_SKIP = "skip"


class Engine:
    def __init__(self, mesh: Mesh, h, tol: float, art: np.ndarray, o: dict, h_path=None, budget=None):
        self.mesh0 = mesh
        self.W = Work(mesh)
        self.W.art = np.zeros(len(self.W.P), dtype=bool)
        self.W.art[: len(art)] = art
        self.art0 = np.asarray(art, dtype=bool)
        self.h, self.tol, self.o = h, tol, {**DEFAULTS, **o}
        self.h_path = h_path
        self.stats = dict(seeds=0, seeds_with_cavity=0, cavities=0, candidates=0, accepted=0, by_kind={},
                          acc_cav_elems=[], acc_cav_k=[], acc_gain_deg=[], acc_dcost=[], t_eval=0.0, t_grow=0.0,
                          seeds_skipped=0, seeds_cached=0, evaluations=0, redo=0, why={}, reject={},
                          **{k: 0.0 for k in _PHASES[1:]}, t_seeds=0.0, t_commit=0.0, t_wait=0.0, t_start=0.0)
        if budget is None:
            self._global_counts(mesh)
        else:
            self.tot, self.budget = dict(budget[0]), dict(budget[1])
        init_worker(None, h, tol, self.o["max_iter"])
        _G["relax"] = self.o["relax"]
        self._relax = _relax_fn(self.o["relax"])
        self.pool = None
        self.version = 0                                    # committed changes so far
        self.node_ver = np.zeros(len(self.W.P), dtype=np.int64)   # the change that last touched each node
        self.cache = {}                                     # seed key -> (read set, version it is valid at)

    def _global_counts(self, mesh):
        """Edges with |ln s/h| > tol and element pairs with ratio > RATIO_JUMP over the whole mesh (the budgets of
        B137: the gates are p99, so an operation may add such edges while the totals stay below budget_share)."""
        X = mesh.points
        Ed = edges_of(mesh.quads, mesh.tris)
        L = R_EARTH_M * np.linalg.norm(X[Ed[:, 0]] - X[Ed[:, 1]], axis=1)
        mid = X[Ed[:, 0]] + X[Ed[:, 1]]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        bad = int((np.abs(np.log(L / self.h(mid))) > self.tol).sum())
        g = element_geometry(mesh)
        me = np.concatenate([g.quad_edges.mean(1), g.tri_edges.mean(1)]) if mesh.n_tris else g.quad_edges.mean(1)
        pr = elem_pairs(mesh.quads, mesh.tris, len(X))
        jump = int((np.maximum(me[pr[:, 0]], me[pr[:, 1]]) / np.minimum(me[pr[:, 0]], me[pr[:, 1]])
                    > RATIO_JUMP).sum())
        sh = self.o["budget_share"]
        self.tot = {"bad": bad, "jump": jump}
        self.budget = {"bad": max(bad, int(sh * len(Ed))), "jump": max(jump, int(sh * len(pr)))}
        self.stats.update(input_bad=bad, input_jump=jump, budget=dict(self.budget), n_edges=int(len(Ed)))

    # ------------------------------------------------------------------------------------------- zone
    def zone(self, free_nodes):
        """(node ids, local quads, local tris, free mask, boundary mask) of the elements at the free nodes."""
        W = self.W
        F = [v for v in free_nodes if W.alive[v] and not W.fixed[v]]
        E = sorted({e for v in F for e in W.at(v)})
        nodes = np.unique(W.el[E][W.el[E] >= 0])
        W._loc[nodes] = np.arange(len(nodes))
        el, nv = W.el[E], W.nv[E]
        quads = W._loc[el[nv == 4]].reshape(-1, 4)
        tris = W._loc[el[nv == 3, :3]].reshape(-1, 3)
        free = np.zeros(len(nodes), dtype=bool)
        free[W._loc[np.asarray(F, dtype=np.int64)]] = True
        W._loc[nodes] = -1
        return nodes, quads, tris, free, W.bnode[nodes].copy()

    def collar_nodes(self, lp, inner):
        return set(self.W.rings(lp, self.o["collar"])) | set(lp) | set(int(v) for v in inner)

    def job(self, S, lp, X=None, cells=None, cn=()):
        """The zone of a candidate (X None: the cavity as it is), prepared and undone: (job, node ids)."""
        W = self.W
        if X is None:
            inner = {v for e in S for v in W.cells(e)} - set(lp)
            nodes, q, t, free, bnd = self.zone(self.collar_nodes(lp, inner))
            return (W.P[nodes].copy(), q, t, free, bnd), nodes
        r = W.replace(sorted(S), X, cells, cn)
        if r is None:
            return None, None
        try:
            nodes, q, t, free, bnd = self.zone(self.collar_nodes(lp, r[1]))
            return (W.P[nodes].copy(), q, t, free, bnd), nodes
        finally:
            W.undo_replace(r)

    # ------------------------------------------------------------------------------------------- acceptance
    def ok(self, old, new, name: str = "polish") -> bool:
        why = self.reject(old, new)
        kd = self.stats["reject"].setdefault(name.split("_")[0], {})
        kd[why or "ok"] = kd.get(why or "ok", 0) + 1
        return not why

    def reject(self, old, new, tot: dict | None = None, budget: bool = True) -> str:
        """The first reason (B137 order) to reject new against old; budget=False skips the global budget check
        (the only one that depends on the running totals tot)."""
        tot = self.tot if tot is None else tot
        if new["inv"] > 0:
            return "inv"
        if budget:
            for k in ("bad", "jump"):
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
        if new["worst"] > old["worst"]:
            return "worst"
        if new["J"] < old["J"] - MIN_GAIN_J:
            return ""
        return "gain"

    # ------------------------------------------------------------------------------------------- seeds
    def seeds(self):
        """(severity, kind, seed elements), worst first: triangles, irregular interior pure-quad nodes, quads with a
        corner above SEED_DEG."""
        W = self.W
        out = []
        E = W.alive_elems()
        el, nv = W.el[E], W.nv[E]
        q = E[nv == 4]
        from meshom.geo import tangent_coords
        from meshom.quality.kernel import _angles

        qmax = np.degrees(_angles(tangent_coords(W.P[W.el[q]]))).max(1)     # element_geometry's quad angles
        from meshom.meshfix.cavity.ncavity import worst_at as _worst_at

        worst_at = _worst_at(np.ascontiguousarray(W.el[q]), qmax, W.n)
        for e in E[nv == 3]:
            out.append((float(worst_at[W.el[e, :3]].max()), "tri", [int(e)]))
        cnt_q = np.bincount(el[nv == 4].ravel(), minlength=W.n)[: W.n]
        cnt_t = np.bincount(el[nv == 3, :3].ravel(), minlength=W.n)[: W.n]
        irr = np.flatnonzero((cnt_t == 0) & (cnt_q > 0) & (cnt_q != 4) & ~W.bnode[: W.n] & W.alive[: W.n])
        for v in irr:
            out.append((float(worst_at[v]), "node", W.at(int(v))))
        for e, a in zip(q[qmax > SEED_DEG], qmax[qmax > SEED_DEG]):
            out.append((float(a), "quad", [int(e)]))
        out.sort(key=lambda x: -x[0])
        return out

    # ------------------------------------------------------------------------------------------- evaluation
    def evaluate(self, seed_elems, tot: dict) -> dict:
        """Steps 1-4 of the module docstring for one seed at the current mesh, the mesh left as it was. Returns
        {cands: the contenders, i.e. the candidates that pass every acceptance rule but the global budget, in
        candidate order, each as an op (apply_op) plus its measures (select() picks the accepted one at the totals
        of the moment); R: the read set (node ids); n: nodes in use; st: the evaluation's counts and phase
        times}. tot only labels the reject statistics."""
        W, o = self.W, self.o
        st = dict(cavities=0, candidates=0, seeds_with_cavity=0, why={}, reject={}, **{k: 0.0 for k in _PHASES})
        t0 = time.perf_counter()
        reads, Rn = set(), set()
        cavs = []
        ncav = 0
        inner_t = 0.0
        for S, lp, cr in cv.grow(W, set(seed_elems), o["max_elems"], o["max_steps"], st["why"], o["coast"], reads):
            ta = time.perf_counter()
            st["cavities"] += 1
            ncav += 1
            inner = {v for e in S for v in W.cells(e)} - set(lp)
            col = self.collar_nodes(lp, inner)
            nodes, q, t, free, bnd = self.zone(col)
            Rn.update(col)
            Rn.update(nodes.tolist())
            tb = time.perf_counter()
            old = measure_local(W.P[nodes].copy(), q, t, free, bnd, self.h, self.tol)
            tc = time.perf_counter()
            cands = pt.fills(W.P, lp, cr, h=self.h, layouts=o.get("layouts"), split=o["split"],
                             is_bedge=W.is_bedge if o["coast"] else None, max_adj=o["max_adj"])
            td = time.perf_counter()
            if not cands:
                e = [len(x) - 1 for x in pt.sides(lp, cr)]
                k = "no_pattern_turn" if e[0] != e[2] and e[1] != e[3] else "no_pattern_other"
                st["why"][k] = st["why"].get(k, 0) + 1
            if o["polish"]:
                cands = [("polish", None, None, ())] + cands
            for name, X, cells, cn in cands:
                j, nd = self.job(S, lp, X, cells, cn)
                if j is not None:
                    cavs.append((S, lp, old, name, X, cells, cn, j, nd))
            te = time.perf_counter()
            st["t_job"] += (tb - ta) + (te - td)
            st["t_measure"] += tc - tb
            st["t_fill"] += td - tc
            inner_t += te - ta
            if ncav >= o["max_cav"]:
                break
        st["t_grow"] += time.perf_counter() - t0 - inner_t
        if reads:
            E = np.fromiter(reads, dtype=np.int64)
            Rn.update(W.el[E][W.el[E] >= 0].tolist())
        R = np.fromiter(Rn, dtype=np.int64, count=len(Rn))
        out = dict(cands=[], R=R, n=int(W.n), st=st)
        if not cavs:
            return out
        st["seeds_with_cavity"] = 1
        st["candidates"] = len(cavs)
        for c in cavs:
            S, lp, old, name, X, cells, cn, j, nd = c
            scr = screen(j[1], j[2], j[3], j[4], old)
            if scr:                                         # rejected whatever the relaxation does (B139)
                kd = st["reject"].setdefault(name.split("_")[0], {})
                kd[scr] = kd.get(scr, 0) + 1
                continue
            ta = time.perf_counter()
            Xr, _ = self._relax(j[0], j[3], j[1], j[2], self.h, self.tol, max_iter=o["max_iter"])
            tb = time.perf_counter()
            met = measure_local(Xr, j[1], j[2], j[3], j[4], self.h, self.tol)
            st["t_relax"] += tb - ta
            st["t_measure"] += time.perf_counter() - tb
            why = self.reject(old, met, tot)
            kd = st["reject"].setdefault(name.split("_")[0], {})
            kd[why or "ok"] = kd.get(why or "ok", 0) + 1
            if why and (why not in ("bad", "jump") or self.reject(old, met, budget=False)):
                continue
            nodes = nd
            fi = np.flatnonzero(j[3])
            nf = nodes[fi]
            n0 = W.n
            rel = np.where(nf >= n0, -1 - (nf - n0), nf)            # new nodes as -1 - j (their place in X)
            out["cands"].append(dict(op=(sorted(S), X, cells, list(cn), rel, Xr[fi]), name=name, met=met,
                                     old=old, n_elems=len(S)))
        return out

    def select(self, cands: list):
        """The accepted contender at the current totals: the lowest J among those within the budget (the first of
        equal ones), or None."""
        best = None
        for c in cands:
            if not self.reject(c["old"], c["met"]) and (best is None or c["met"]["J"] < best["met"]["J"]):
                best = c
        return best

    # ------------------------------------------------------------------------------------------- commit
    def apply_op(self, op) -> np.ndarray:
        """Commit a change: replace the cavity S by the fill (X None: positions only) and move the free nodes; the
        version stamps of the touched nodes advance. Returns the touched nodes."""
        W = self.W
        S, X, cells, cn, rel, Xf = op
        ch = []
        ids = np.zeros(0, dtype=np.int64)
        if X is not None:
            r = W.replace(S, X, cells, cn)
            assert r is not None, "a committed fill no longer fits (node table width)"
            W.mark(r[0])
            ids = r[1]
            for _, c, _ in r[0][0]:
                ch.extend(c)
            for c in r[2]:
                ch.extend(c)
        nodes = np.array(rel, dtype=np.int64)
        neg = nodes < 0
        nodes[neg] = ids[-1 - nodes[neg]]
        W.P[nodes] = Xf
        W.touched[nodes] = True
        self.version += 1
        if len(self.node_ver) < len(W.P):
            nv = np.zeros(len(W.P), dtype=np.int64)
            nv[: len(self.node_ver)] = self.node_ver
            self.node_ver = nv
        touched = np.unique(np.concatenate([np.asarray(ch, dtype=np.int64), nodes, ids]))
        self.node_ver[touched] = self.version
        return touched

    def _commit(self, best) -> None:
        st, o = self.stats, self.o
        op = best["op"]
        self.apply_op(op)
        if self.pool is not None:
            self.pool.broadcast(pickle.dumps(op, protocol=pickle.HIGHEST_PROTOCOL))
        met, old, name = best["met"], best["old"], best["name"]
        for k in ("bad", "jump"):
            self.tot[k] += met[k] - old[k]
        kind = name.split("_")[0]
        st["accepted"] += 1
        st["by_kind"][kind] = st["by_kind"].get(kind, 0) + 1
        st["acc_cav_elems"].append(best["n_elems"])
        st["acc_cav_k"].append(name)
        st["acc_gain_deg"].append(old["worst"] - met["worst"])
        st["acc_dcost"].append(met["cost"] - old["cost"])
        if o["verbose"] > 1:
            print(f"  accept {name}: |S| {best['n_elems']} worst {old['worst']:.1f} -> {met['worst']:.1f} cost "
                  f"{old['cost']:.0f} -> {met['cost']:.0f} n120 {old['n120']} -> {met['n120']}", flush=True)

    def _merge(self, s: dict) -> None:
        st = self.stats
        st["evaluations"] += 1
        for k in ("cavities", "candidates", "seeds_with_cavity"):
            st[k] += s[k]
        for k, v in s["why"].items():
            st["why"][k] = st["why"].get(k, 0) + v
        for kind, d in s["reject"].items():
            kd = st["reject"].setdefault(kind, {})
            for k, v in d.items():
                kd[k] = kd.get(k, 0) + v
        for k in _PHASES:
            st[k] += s[k]
        st["t_eval"] += s["t_relax"] + s["t_measure"]

    def _stale(self, r: dict) -> bool:
        """The result was computed on a state that differs from the current one where it read."""
        R = r["R"]
        return bool(len(R)) and int(self.node_ver[R].max()) > r["version"]

    def _skip(self, es) -> bool:
        W = self.W
        return any(W.nv[e] == 0 for e in es) or any(W.touched[v] for e in es for v in W.cells(e))

    def _cached(self, key):
        c = self.cache.get(key)
        if c is None:
            return None
        R, ver, _ = c
        if len(R) and int(self.node_ver[R].max()) > ver:
            del self.cache[key]
            return None
        return c

    def _retire(self, key, r) -> int:
        """Retire a valid result: commit its accepted contender at the current totals (1) or remember the seed and
        its contenders (0)."""
        best = self.select(r["cands"])
        if best is not None:
            t = time.perf_counter()
            self._commit(best)
            self.stats["t_commit"] += time.perf_counter() - t
            return 1
        if self.o["cache"] and key is not None:
            self.cache[key] = (r["R"], self.version, r["cands"])
        return 0

    def run_seed(self, seed_elems) -> bool:
        """Evaluate one seed at the current mesh and commit its best candidate (the serial step)."""
        r = self.evaluate(seed_elems, dict(self.tot))
        self._merge(r["st"])
        r.update(version=self.version)
        return bool(self._retire(None, r))

    # ------------------------------------------------------------------------------------------- passes
    def _pass_serial(self, sd) -> int:
        st, acc = self.stats, 0
        for sev, kind, es in sd:
            if self._skip(es):
                st["seeds_skipped"] += 1
                continue
            st["seeds"] += 1
            key = (kind, tuple(sorted(es)))
            c = self._cached(key) if self.o["cache"] else None
            if c is not None:
                st["seeds_cached"] += 1
                r = dict(cands=c[2], R=c[0], version=c[1])
            else:
                r = self.evaluate(es, dict(self.tot))
                self._merge(r["st"])
                r.update(version=self.version)
            acc += self._retire(key, r)
        return acc

    def _pass_parallel(self, sd) -> int:
        """One pass with the workers (module docstring): seeds are evaluated ahead of the retired one (at most
        `window` evaluated seeds ahead), results retired in seed order; a result made stale by a commit (its read set
        changed) is evaluated again at once, the nearest first."""
        import heapq

        st, o, P = self.stats, self.o, self.pool
        n = len(sd)
        window = int(o.get("window") or 8 * len(P.procs))
        keys = [(k, tuple(sorted(es))) for _, k, es in sd]
        res = [None] * n                                     # None: to evaluate or in flight; _SKIP; a result
        fly = [False] * n
        ver = [0] * n
        tots = [None] * n
        evald = [False] * n                                  # dispatched for evaluation at least once
        ahead = 0                                            # seeds dispatched and not yet retired
        redo = []                                            # heap of seed indices to evaluate again
        nxt = ret = acc = 0

        def check(i):                                       # a finished result still valid? else skip / redo
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
            while True:                                      # dispatch
                w = P.free_worker(o["inflight"])
                if w is None:
                    break
                if redo:
                    i = heapq.heappop(redo)
                else:
                    while nxt < n and ahead < window:
                        es = sd[nxt][2]
                        if self._skip(es):
                            res[nxt] = _SKIP
                        elif o["cache"] and (c := self._cached(keys[nxt])) is not None:
                            res[nxt] = dict(cands=c[2], R=c[0], version=c[1], cached=True)
                        else:
                            break
                        nxt += 1
                    if nxt >= n or ahead >= window:
                        break
                    i = nxt
                    nxt += 1
                    evald[i] = True
                    ahead += 1
                ver[i], tots[i] = self.version, dict(self.tot)
                fly[i] = True
                P.send(w, ("eval", i, sd[i][2], tots[i]))
            while ret < n and res[ret] is not None:          # retire in seed order
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
                if a:                                        # the commit may invalidate results ahead
                    for k in range(ret, nxt):
                        check(k)
            if ret >= n:
                break
            if res[ret] is None and not fly[ret] and P.free_worker(o["inflight"]) is not None:
                continue                                     # the head is not in flight: dispatch it first
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
        d = meshfix_dir() / "tmp" / f"d_{os.getpid()}_{int(time.time() * 1e3)}"
        d.mkdir(parents=True, exist_ok=True)
        self._tmp = d
        _save(self.mesh0, d / "mesh.npz")
        np.save(d / "art.npy", self.art0)
        h_path = self.h_path
        if h_path is None:
            h_path = d / "h.pkl"
            with open(h_path, "wb") as f:
                pickle.dump(self.h, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._warm()
        o = dict(self.o, workers=1, verbose=False)
        self.pool = _Pool(int(self.o["workers"]), dict(h_path=str(h_path), mesh=str(d / "mesh.npz"),
                                                       art=str(d / "art.npy"), tol=self.tol, o=o,
                                                       budget=(dict(self.tot), dict(self.budget))))
        self.stats["t_start"] += time.perf_counter() - t

    def _warm(self) -> None:
        """Compile the Numba kernels once (cache=True) before the workers load them."""
        from meshom.meshfix.cavity import nrelax

        P = np.array([[1.0, 0, 0], [0.999, 0.01, 0], [0.999, 0.01, 0.01], [0.999, 0, 0.01]])
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        nrelax.relax(P, np.array([True, False, False, False]), np.array([[0, 1, 2, 3]]), np.zeros((0, 3), int),
                     lambda x: np.full(len(x), 1e4), 0.2, max_iter=2)

    def run(self):
        o, st = self.o, self.stats
        st["passes"] = []
        try:
            if o["workers"] > 1:
                self._start_pool()
            for p in range(o["passes"]):
                W = self.W
                W.touched[:] = False
                t = time.perf_counter()
                sd = self.seeds()
                if o["max_seeds"]:
                    sd = sd[: o["max_seeds"]]
                st["t_seeds"] += time.perf_counter() - t
                e0, c0 = st["evaluations"], st["seeds_cached"]
                acc = self._pass_parallel(sd) if self.pool is not None else self._pass_serial(sd)
                st["passes"].append(dict(seeds=len(sd), accepted=acc, evaluated=st["evaluations"] - e0,
                                         cached=st["seeds_cached"] - c0, s=time.perf_counter() - t))
                if o["verbose"]:
                    print(f"pass {p}: {len(sd)} seeds, {st['evaluations'] - e0} evaluated, "
                          f"{st['seeds_cached'] - c0} cached, {acc} accepted, {time.perf_counter() - t:.1f} s",
                          flush=True)
                if acc == 0:
                    break
        finally:
            if self.pool is not None:
                self.pool.close()
                self.pool = None
                shutil.rmtree(self._tmp, ignore_errors=True)
        return self.W.to_mesh()[0]
