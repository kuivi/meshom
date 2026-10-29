"""Competitor A of the mesh-fixing competition: free-node geometric optimisation on the sphere (`a_geom`; decisions
B122-B125). The engine is core.optimize_geometry (callable on any Mesh with a fixed-node mask: the final stage of
the topology methods); this module wraps it as a bench method.

    python -m meshom.meshfix.geom --case NAME [NAME ...] [--tag TAG] [--set key=value ...] [--no-score]

saves <MESHFIX_DIR>/results/a_geom[_TAG]/<case>.npz + .json and scores it.
"""
import json
import time

import numpy as np

from meshom.meshfix.geom.core import DEFAULTS, optimize_geometry

METHOD = "a_geom"


def fix(case, opts: dict | None = None):
    """Bench contract: (Mesh, record). Fixed = the case's artificial boundary (kind 2); coast nodes (kind 1) slide
    unless opts["coast"] == "fixed". The slide is measured against the case's reference coast (Case.coast_seg, B146:
    in a pipeline that is the input's coast, not the coast the stages before moved) unless opts["ref"] is False."""
    opts = dict(opts or {})
    coast = opts.pop("coast", "slide")
    use_ref = bool(opts.pop("ref", True))
    envelope = {}
    if opts.pop("tri_envelope", True) and case.mesh.n_quads == 0 and case.mesh.n_tris:
        envelope = tri_envelope(case.mesh, case.h, opts)
        opts = {**envelope, **opts}
    fixed = case.kind == 2
    slide = (case.kind == 1) if coast == "slide" else np.zeros(case.mesh.n_nodes, bool)
    ref = case.coast_seg if use_ref and case.coast_seg is not None and len(case.coast_seg) else None
    t0 = time.perf_counter()
    mesh, info = optimize_geometry(case.mesh, case.h, fixed=fixed, slide=slide, opts=opts, ref=ref)
    rec = {"method": METHOD, "case": case.name, "wall_s": time.perf_counter() - t0, "coast": coast,
           "ref": ref is not None, "tri_envelope": envelope, "info": info}
    return mesh, rec


def tri_envelope(mesh, h, opts: dict) -> dict:
    """A on a pure-triangle mesh (B157): the triangle guards and the size / ratio walls tightened to the
    input's own tails, never loosened: lo_t = max(35, min-angle p1), wall = min(0.20, |ln s/h| p99 + margin),
    ratio_wall = min(1.12, element ratio p99); each only when not set in opts (guard_t stays 90: tightened to the
    max-angle p99 it held farc's max angle at 105 deg against 97 deg). On meshom's good triangle mesh
    (core2_A2_tri_v2) A with its defaults lowered the min angle 42.8 -> 40.2 deg and raised |ln s/h| p99 0.117 ->
    0.177 (4 new size gate failures); with the envelope min angle 42.8, p99 0.118, 29/29 gates."""
    from meshom.meshfix.tri import guard_summary

    g = guard_summary(mesh, h)
    out = {"lo_t": max(DEFAULTS["lo_t"], g["amin_p1"]),
           "wall": min(DEFAULTS["wall"], g["lnsh_p99"] + DEFAULTS["margin"]),
           "ratio_wall": min(DEFAULTS["ratio_wall"], g["ratio_p99"]),
           # speed (B157): fewer rings around the seeds and iterations per round (farc t,a: A 503 s of 580 s with
           # the defaults 8 rings / 1500 iterations, round 0 stopped at max_iter)
           "active_rings": TRI_ACTIVE_RINGS, "max_iter": TRI_MAX_ITER, "rounds": TRI_ROUNDS}
    return {k: v for k, v in out.items() if k not in opts}


TRI_ACTIVE_RINGS = 4   # B157
TRI_MAX_ITER = 500     # B157
TRI_ROUNDS = 2         # B157


def run(case_name: str, opts: dict | None = None, tag: str = "", score: bool = True) -> dict:
    from meshom.compete import _save
    from meshom.meshfix.cases import load, meshfix_dir
    from meshom.meshfix.score import score_file

    case = load(case_name)
    mesh, rec = fix(case, opts)
    method = METHOD + (f"_{tag}" if tag else "")
    rec["method"] = method
    d = meshfix_dir() / "results" / method
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{case_name}.npz"
    _save(mesh, f)
    f.with_suffix(".json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"{case_name} {method}: {rec['wall_s']:.1f} s -> {f}", flush=True)
    if score:
        t1 = time.perf_counter()
        r = score_file(case_name, f, case=case, label=method)
        print(f"score {time.perf_counter() - t1:.0f} s", flush=True)
        return r
    return rec


def _parse(kv: list[str]) -> dict:
    out = {}
    for s in kv:
        k, v = s.split("=", 1)
        if k == "coast":
            out[k] = v
            continue
        if k not in DEFAULTS:
            raise KeyError(k)
        out[k] = type(DEFAULTS[k])(float(v)) if not isinstance(DEFAULTS[k], str) else v
    return out
