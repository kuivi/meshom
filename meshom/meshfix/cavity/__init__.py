"""Competitor D of the mesh-fixing competition: cavity (disk) remeshing of bad zones with chosen patterns.

    python -m meshom.meshfix.cavity --case NAME [--opt key=value ...] [--method NAME] [--no-score]

fix(case, opts) -> (Mesh, record) (the bench contract, meshom/meshfix/__init__.py). The loop is engine.py, the cavity
search cavity.py, the fills patterns.py, the relaxation nrelax.py (Numba; relax.py the JAX reference, opt relax=jax),
the mutable mesh work.py (decisions B134-B137). The fast engine (B138-B140): read sets and a cache of unchanged seeds,
worker processes (opt workers, default one per CPU up to MAX_WORKERS) evaluating seeds ahead, results retired in
seed order, so the result is the serial one.
"""
import argparse
import json
import time

import numpy as np

METHOD = "d_cavity"


MAX_WORKERS = 128            # B140: at most this many worker processes
WORKER_BYTES_PER_NODE = 700  # B140: a worker's memory per mesh node (v4: 192 GB for 129 processes, 1.75 M nodes)
WORKER_BYTES_BASE = 3e8      # B140: and per worker (Python, numba, scipy, h)
MEM_SHARE = 0.8              # B140: the workers may use this share of the job's memory


def default_workers(n_nodes: int | None = None) -> int:
    """B140: one worker per CPU of the job, at most MAX_WORKERS and as many as fit in MEM_SHARE of its memory (each
    holds a copy of the mesh and h; the job's memory: SLURM_MEM_PER_NODE, else MemAvailable)."""
    import os

    w = min(len(os.sched_getaffinity(0)), MAX_WORKERS)
    if n_nodes:
        mem = float(os.environ.get("SLURM_MEM_PER_NODE", "0") or 0) * 2 ** 20
        if mem <= 0:
            with open("/proc/meminfo") as f:
                mem = next(float(x.split()[1]) * 1024 for x in f if x.startswith("MemAvailable"))
        w = min(w, int(MEM_SHARE * mem / (WORKER_BYTES_PER_NODE * n_nodes + WORKER_BYTES_BASE)))
    return max(1, w)


def fix(case, opts: dict | None = None):
    from meshom.meshfix.cavity.engine import DEFAULTS, Engine

    o = {**DEFAULTS, **(opts or {})}
    if "workers" not in (opts or {}):
        o["workers"] = default_workers(case.mesh.n_nodes)
    t = time.perf_counter()
    from meshom.meshfix.cases import h_file

    eng = Engine(case.mesh, case.h, case.tol or 0.20, case.kind == 2, o, h_file(case))
    mesh = eng.run()
    t_cav = time.perf_counter() - t
    final = o.get("final")
    if final == "smooth_fit":                                # combination row: then the smooth_fit baseline
        from meshom.size_fit import smooth_fit

        mesh, _ = smooth_fit(mesh, case.h, case.tol)
    wall = time.perf_counter() - t
    st = eng.stats
    sizes = np.array(st["acc_cav_elems"]) if st["acc_cav_elems"] else np.zeros(0)
    rec = {"method": o.get("method", METHOD), "case": case.name, "wall_s": wall, "cavity_s": t_cav, "opts": o,
           "stats": {k: v for k, v in st.items() if k not in ("acc_cav_elems", "acc_cav_k", "acc_gain_deg",
                                                              "acc_dcost")},
           "accepted_cavity_elems": {"n": int(len(sizes)),
                                     **({"min": int(sizes.min()), "median": float(np.median(sizes)),
                                         "p90": float(np.percentile(sizes, 90)), "max": int(sizes.max())}
                                        if len(sizes) else {})},
           "accepted": [dict(name=a, elems=int(b), gain_deg=float(c), dcost=float(d)) for a, b, c, d in
                        zip(st["acc_cav_k"], st["acc_cav_elems"], st["acc_gain_deg"], st["acc_dcost"])]}
    return mesh, rec


def _val(s: str):
    for f in (int, float):
        try:
            return f(s)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None}.get(s.lower(), s)


def main(argv=None):
    from meshom.compete import _save
    from meshom.meshfix import cases
    from meshom.meshfix.score import score_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--method", default=METHOD)
    ap.add_argument("--no-score", action="store_true")
    a = ap.parse_args(argv)
    opts = {k: _val(v) for k, v in (x.split("=", 1) for x in a.opt)}
    opts["method"] = a.method
    case = cases.load(a.case)
    mesh, rec = fix(case, opts)
    d = cases.meshfix_dir() / "results" / a.method
    d.mkdir(parents=True, exist_ok=True)
    _save(mesh, d / f"{a.case}.npz")
    (d / f"{a.case}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: rec[k] for k in ("wall_s", "stats", "accepted_cavity_elems")}, indent=1, default=float),
          flush=True)
    if not a.no_score:
        from meshom.meshfix.score import table

        r = score_file(a.case, d / f"{a.case}.npz")
        print(table([r]), end="", flush=True)
