"""The fast B engine (decision B150): the serial sweep of transport/__init__.py, computed exactly, faster.

- Pure evaluations: transport.evaluate tries every rewrite of a seed against the mesh as it is and leaves the mesh
  exactly as it was (Work.undo restores the node -> element rows in their order), so its answer is a function of the
  committed mesh only.
- Read sets: an evaluation returns the nodes whose incidence or position it may have read (READ_RINGS element rings
  around every patch it tried, plus the triangles its cluster searches visited). Every commit stamps the nodes of the
  removed and the new cells and the moved nodes with a version; a result stays valid while no node of its read set
  has a newer stamp (the argument of D's engine, B138).
- Cache (`cache`): a seed that accepted nothing keeps (read set, version, tried counts) and is not evaluated again,
  in this sweep or a later one, while its read set is unchanged.
- Workers (`workers` > 1): spawned processes, each with its own Work built from the input mesh; every commit is sent
  to every worker in commit order, so element and node ids stay equal (a VE's fresh node gets the next node id in
  every copy). Seeds of a phase are evaluated ahead of the retired one (at most `window` ahead, `inflight` per
  worker); results are retired strictly in seed order: a seed the serial loop would skip is skipped, a stale result
  is evaluated again, an accepted rewrite is committed and broadcast. A phase's seed list is made when the phase
  starts, after every earlier commit, as in the serial loop.
The result equals the serial loop's, bitwise (tests).
"""
import os
import pickle
import shutil
import time
from collections import Counter

import numpy as np

from . import PHASES, Work, commit, evaluate, phase_items, skip, sweep_seeds

_SKIP = "skip"


def _worker_main(wid, inbox, outbox, init):
    """A worker: its own Engine (copy of the working mesh); ("op", res) applies a committed rewrite, ("eval", i,
    kind, item) evaluates a seed, ("stop",) ends."""
    try:
        with open(init["case"], "rb") as f:
            case = pickle.load(f)
        eng = Engine(case, init["opts"], worker=True)
        outbox.put(("ready", wid, None, None))
        while True:
            msg = inbox.get()
            if msg[0] == "op":
                eng.apply(pickle.loads(msg[1]))
            elif msg[0] == "eval":
                _, i, kind, item = msg
                t = time.perf_counter()
                r = evaluate(eng.work, kind, item, eng.case, eng.opts)
                r["t"] = time.perf_counter() - t
                outbox.put(("res", wid, i, r))
            else:
                return
    except BaseException as e:
        import traceback

        outbox.put(("error", wid, None, traceback.format_exc()))
        raise e


class _Pool:
    """Worker processes with one inbox each (commits are broadcast, evaluations go to one worker) and one outbox
    (D's cavity.engine._Pool, copied here: D's pool starts D's worker)."""

    def __init__(self, n, init):
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
                    raise RuntimeError(f"transport workers died: {dead}")
                continue
            if m[0] == "error":
                raise RuntimeError(f"transport worker {m[1]} failed:\n{m[3]}")
            return m

    def broadcast(self, payload):
        for q in self.inb:
            q.put(("op", payload))

    def send(self, w, msg):
        self.busy[w] += 1
        self.inb[w].put(msg)

    def free_worker(self, limit):
        w = int(np.argmin(self.busy))
        return w if self.busy[w] < limit else None

    def close(self):
        for q in self.inb:
            q.put(("stop",))
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()


class Engine:
    def __init__(self, case, opts, worker=False):
        self.case, self.opts = case, opts
        self.work = Work(case.mesh, case.fixed)
        self.version = 0
        self.node_ver = np.zeros(len(self.work.P), dtype=np.int64)
        self.cache = {}
        self.pool = None
        self.stats = dict(evaluations=0, cached=0, skipped=0, redo=0, committed=0, t_eval=0.0, t_wait=0.0,
                          t_commit=0.0, t_list=0.0, t_start=0.0, workers=int(opts.get("workers") or 1),
                          phases={k: [] for k in PHASES})
        if not worker and opts.get("workers", 1) > 1:
            self._start_pool()

    # --------------------------------------------------------------------------------------------- commits
    def apply(self, res):
        """Commit an accepted rewrite and stamp the nodes it touched; returns its blocked nodes."""
        _, touched, nodes = commit(self.work, res)
        self.version += 1
        if len(self.node_ver) < len(self.work.P):
            nv = np.zeros(len(self.work.P), dtype=np.int64)
            nv[: len(self.node_ver)] = self.node_ver
            self.node_ver = nv
        self.node_ver[np.fromiter(touched, dtype=np.int64, count=len(touched))] = self.version
        return nodes

    def _stale(self, R, ver):
        return bool(len(R)) and int(self.node_ver[R].max()) > ver

    def _cached(self, key):
        c = self.cache.get(key)
        if c is None:
            return None
        if self._stale(c["R"], c["version"]):
            del self.cache[key]
            return None
        return c

    def _retire(self, key, r, blocked, counts, ops):
        """Retire a valid result: count its trials, commit its rewrite (1) or cache the seed (0)."""
        counts.update(r["tried"])
        res = r["res"]
        if res is None:
            if self.opts["cache"]:
                self.cache[key] = dict(res=None, tried=r["tried"], R=r["R"], version=r["version"])
            return 0
        t = time.perf_counter()
        kind = key[0] if key[0] != "zip" else res["kind"]
        nodes = self.apply(res)
        if self.pool is not None:
            self.pool.broadcast(pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL))
        blocked |= nodes
        counts[kind] += 1
        entry = dict(kind=kind)
        if kind == "recast":
            entry["node"] = res["node"]
        entry.update(before=res["before"], after=res["after"], tris_delta=res["tris_delta"])
        ops.append(entry)
        self.stats["committed"] += 1
        self.stats["t_commit"] += time.perf_counter() - t
        return 1

    # --------------------------------------------------------------------------------------------- passes
    def sweep(self, ops):
        """One sweep (the serial loop's four phases); returns its counts."""
        blocked, counts = set(), Counter()
        t = time.perf_counter()
        seeds = sweep_seeds(self.work)
        self.stats["t_list"] += time.perf_counter() - t
        for kind in PHASES:
            if not self.opts[kind]:
                continue
            t = time.perf_counter()
            items = phase_items(self.work, kind, seeds, blocked)
            self.stats["t_list"] += time.perf_counter() - t
            e0, c0 = self.stats["evaluations"], self.stats["cached"]
            if self.pool is None:
                acc = self._phase_serial(kind, items, blocked, counts, ops)
            else:
                acc = self._phase_parallel(kind, items, blocked, counts, ops)
            self.stats["phases"][kind].append(dict(items=len(items), evaluated=self.stats["evaluations"] - e0,
                                                   cached=self.stats["cached"] - c0, accepted=acc,
                                                   s=time.perf_counter() - t))
        return dict(counts)

    def _phase_serial(self, kind, items, blocked, counts, ops):
        acc = 0
        for item in items:
            if skip(self.work, kind, item, blocked):
                self.stats["skipped"] += 1
                continue
            key = (kind, item)
            c = self._cached(key) if self.opts["cache"] else None
            if c is not None:
                self.stats["cached"] += 1
                r = c
            else:
                t = time.perf_counter()
                r = evaluate(self.work, kind, item, self.case, self.opts, read_set=self.opts["cache"])
                self.stats["t_eval"] += time.perf_counter() - t
                self.stats["evaluations"] += 1
                r["version"] = self.version
            acc += self._retire(key, r, blocked, counts, ops)
        return acc

    def _phase_parallel(self, kind, items, blocked, counts, ops):
        """One phase with the workers (module docstring)."""
        import heapq

        st, o, P = self.stats, self.opts, self.pool
        n = len(items)
        window = int(o.get("window") or 8 * len(P.procs))
        res = [None] * n
        fly = [False] * n
        ver = [0] * n
        evald = [False] * n
        ahead = 0
        redo = []
        nxt = ret = acc = 0
        W = self.work

        def check(i):
            r = res[i]
            if r is None or r is _SKIP:
                return
            if skip(W, kind, items[i], blocked):
                res[i] = _SKIP
            elif self._stale(r["R"], r["version"]):
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
                    if fly[i] or res[i] is not None:
                        continue
                else:
                    while nxt < n and ahead < window:
                        if skip(W, kind, items[nxt], blocked):
                            res[nxt] = _SKIP
                        elif o["cache"] and (c := self._cached((kind, items[nxt]))) is not None:
                            res[nxt] = dict(c, cached=True)
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
                P.send(w, ("eval", i, kind, items[i]))
            while ret < n and res[ret] is not None:          # retire in seed order
                r = res[ret]
                if r is _SKIP or skip(W, kind, items[ret], blocked):
                    st["skipped"] += 1
                    ahead -= evald[ret]
                    ret += 1
                    continue
                if self._stale(r["R"], r["version"]):
                    check(ret)
                    break
                if r.get("cached"):
                    st["cached"] += 1
                a = self._retire((kind, items[ret]), r, blocked, counts, ops)
                acc += a
                ahead -= evald[ret]
                ret += 1
                if a:
                    for k in range(ret, nxt):
                        check(k)
            if ret >= n:
                break
            if res[ret] is None and not fly[ret] and P.free_worker(o["inflight"]) is not None:
                if not redo or redo[0] != ret:
                    heapq.heappush(redo, ret)
                continue
            if res[ret] is None:
                t = time.perf_counter()
                m = P.get()
                st["t_wait"] += time.perf_counter() - t
                _, w, i, r = m
                P.busy[w] -= 1
                fly[i] = False
                st["evaluations"] += 1
                st["t_eval"] += r.pop("t")
                r["version"] = ver[i]
                res[i] = r
                check(i)
        return acc

    # --------------------------------------------------------------------------------------------- pool
    def _start_pool(self):
        from meshom.meshfix.cases import meshfix_dir

        t = time.perf_counter()
        d = meshfix_dir() / "tmp" / f"b_{os.getpid()}_{int(time.time() * 1e3)}"
        d.mkdir(parents=True, exist_ok=True)
        self._tmp = d
        with open(d / "case.pkl", "wb") as f:
            pickle.dump(self.case, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._warm()
        o = dict(self.opts, workers=1)
        self.pool = _Pool(int(self.opts["workers"]), dict(case=str(d / "case.pkl"), opts=o))
        self.stats["t_start"] += time.perf_counter() - t

    def _warm(self):
        """Compile the Numba fit once (cache=True) before the workers load it."""
        from .nfit import fit_sub

        g = np.array([(x, y) for x in range(3) for y in range(3)], dtype=float) * 1e-3
        P = np.c_[np.ones(9), g]
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        q = np.array([[0, 3, 4, 1], [1, 4, 5, 2], [3, 6, 7, 4]])
        t = np.array([[4, 7, 8], [4, 8, 5]])
        fixed = np.ones(9, dtype=bool)
        fixed[4] = False
        fit_sub(P, t, q, fixed, lambda x: np.full(len(x), 6.4e3), 0.2, max_iter=2)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool = None
            shutil.rmtree(self._tmp, ignore_errors=True)
