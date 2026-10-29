"""Carry defects to the coast and absorb them there (method `e_coast`; decisions
B142-B145, the fast engine B147-B148). The loop is engine.py, the coast fills fills.py, the relaxation with sliding
coast nodes cavity/ncoast.py (relax.py the JAX reference), the reference coast refcoast.py.

fix(case, opts) -> (Mesh, record) (the bench contract) is stage `e` of the one pipeline (meshom/meshfix/pipeline.py):

    python -m meshom.meshfix.pipeline --case NAME --stages d,e,a [--opt e.workers=64 ...]

The reference coast is always the case's (Case.coast_seg, rebuilt into a segment graph by position), so a stage e
after another stage slides the coast along the input's coast, not along the coast the stages before left.
`python -m meshom.meshfix.coast --case NAME [--stages e|d,e,a] [--opt key=value]` is a thin wrapper of the pipeline
(the options go to stage e).
"""
import time

import numpy as np

METHOD = "e_coast"


def _h_path(case):
    from meshom.meshfix.cases import h_file

    return h_file(case)


def ref_of(case):
    """The case's reference coast as a segment graph (refcoast.RefCoast.from_segments(case.coast_seg))."""
    from meshom.meshfix.coast.refcoast import RefCoast

    return RefCoast.from_segments(case.coast_seg)


def fix(case, opts: dict | None = None):
    """e_coast on case.mesh: fixed = kind 2, the reference coast = case.coast_seg; workers default as D's (one per
    CPU of the job, B140)."""
    from meshom.meshfix.cavity import default_workers
    from meshom.meshfix.coast.engine import DEFAULTS, CoastEngine

    o = {**DEFAULTS, **(opts or {})}
    if "workers" not in (opts or {}):
        o["workers"] = default_workers(case.mesh.n_nodes)
    t = time.perf_counter()
    eng = CoastEngine(case.mesh, case.h, case.tol or 0.20, case.kind == 2, ref_of(case), o, _h_path(case))
    out = eng.run()
    st = eng.stats
    rec = {"method": o.get("method", METHOD), "case": case.name, "wall_s": time.perf_counter() - t,
           "opts": {k: v for k, v in o.items()}, "stats": {k: v for k, v in st.items() if k != "acc"},
           "accepted": st["acc"]}
    return out, rec


A_GUARD_H = 0.10   # B145: before A, coast nodes near a reference vertex already this far (in h) off the coast stay put
A_GUARD_RINGS = 2  # ... with this many coast neighbours on each side


def a_slide_mask(case, mesh, kind, ref) -> np.ndarray:
    """Coast nodes A may slide: A's reference coast is its input's boundary (E's re-spaced coast), so A's own
    0.25 h would stack on E's shift; the coast nodes at reference vertices that are already > A_GUARD_H h off the
    current coast (and A_GUARD_RINGS coast neighbours each way) are kept in place."""
    from scipy.spatial import cKDTree

    from meshom.geo import R_EARTH_M
    from meshom.meshfix.cases import boundary_edges
    from meshom.meshfix.coast.refcoast import seg_point

    slide = kind == 1
    be = boundary_edges(mesh)
    be = be[(kind[be[:, 0]] != 2) & (kind[be[:, 1]] != 2)]
    if len(be) == 0:
        return slide
    A, B = mesh.points[be[:, 0]], mesh.points[be[:, 1]]
    mid = A + B
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    k = min(8, len(be))
    _, idx = cKDTree(mid).query(ref.V, k=k)
    idx = idx.reshape(len(ref.V), k)
    _, d = seg_point(ref.V[:, None, :], A[idx], B[idx])
    j = idx[np.arange(len(ref.V)), d.argmin(1)]
    far = d.min(1) * R_EARTH_M > A_GUARD_H * case.h(ref.V)
    hold = set(be[j[far]].ravel().tolist())
    nb = {}
    for a, b in be.tolist():
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    front = set(hold)
    for _ in range(A_GUARD_RINGS):
        front = {w for v in front for w in nb.get(v, ())} - hold
        hold |= front
    slide = slide.copy()
    slide[list(hold)] = False
    return slide


def main(argv=None):
    import argparse

    from meshom.meshfix import pipeline

    ap = argparse.ArgumentParser(prog="python -m meshom.meshfix.coast")
    ap.add_argument("--case", required=True)
    ap.add_argument("--stages", default="e")
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--method", default=None)
    ap.add_argument("--no-score", action="store_true")
    a = ap.parse_args(argv)
    stages = a.stages.split(",")
    method = a.method or (METHOD if stages == ["e"] else "e_coast_" + "".join(stages))
    args = ["--case", a.case, "--stages", a.stages, "--name", method] + \
        sum((["--opt", "e." + x] for x in a.opt), []) + (["--no-score"] if a.no_score else [])
    pipeline.main(args)
