"""One mesh-fixing pipeline over every method (decisions B141, B148, B149, B157): any sequence of a, b, c, d, e, t on a
case or a finished run.

    python -m meshom.meshfix.pipeline --case NAME [--stages d,b,e,a] [--name NAME] [--opt d.workers=64 ...] [--no-score]
    python -m meshom.meshfix.pipeline --run RUN_DIR [--stages d,b,e,a] [...]

Default stages d,b,e,a (the best order on every bench case, B149). B runs its seeds on worker processes (B150; each
worker and the main process with one Numba thread).

Stages (each the method's bench contract fix(case, opts) -> (Mesh, record)):
  a  meshom.meshfix.geom       (A: free-node geometry, coast slide)
  b  meshom.meshfix.transport  (B: local defect transport)
  c  meshom.meshfix.placement  (C: global placement, re-extraction)
  d  meshom.meshfix.cavity     (D: cavity remeshing with patterns)
  e  meshom.meshfix.coast      (E: defects carried to the coast and absorbed there)
  t  meshom.meshfix.tri        (T: valence flips and smoothing of a pure-triangle mesh, guarded; order t,a; B155-B157)

A pure-triangle input (a meshom m15_tri run, a FESOM2 case) takes --stages t,a (the quad stages d, b, e assume quads).

Between two stages the case is rebuilt for the new mesh (rebase): node kind 2 (the artificial boundary, never moved)
by exact position against the input's kind-2 nodes, kind 1 = every other boundary node of the new mesh, kind 0 the
rest; art_edges remapped; h_node = h at the new nodes. The geography (coast_seg,
dist_seg, basin_pts), tol and gates stay those of the input, so the final score is against the input. Stage e takes
its reference coast from coast_seg (the input's coast) and A measures its coast slide against it too (B146), so no
stage stacks a coast shift on an earlier one.

--case: the bench case; result <MESHFIX_DIR>/results/<name>/<case>.npz + .json (name default `p_<stages>`, e.g.
p_d_a), scored once at the end (.score.json and maps next to it).
--run: a finished run folder (raw.npz, config.resolved.toml, quality.json); the case is built on the fly as
cases.build_whole does (the run's h from cache/size.pkl or the config), nothing is written into cases/; the result
is <run>/fixed_<stages>.npz + .json + .score.json.

--opt STAGE.KEY=VALUE sets an option of one stage (d.workers=64, a.coast=fixed, ...). a.guard=true applies E's
former A guard (coast.a_slide_mask: the coast nodes near reference vertices already > A_GUARD_H h off the current coast
held, as kind 2, during stage a only; kept for the before/after comparison, B146); a.ref=false lets A measure its slide
against its own input's boundary (the former behaviour).
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import numba
import numpy as np
from scipy.spatial import cKDTree

from meshom.mesh import Mesh, build_topology

SERIAL_STAGES = ("b",)   # B141, B150: one Numba thread in the main process (B's fits are tiny; its workers are processes)
STAGES = {"a": "meshom.meshfix.geom", "b": "meshom.meshfix.transport", "c": "meshom.meshfix.placement",
          "d": "meshom.meshfix.cavity", "e": "meshom.meshfix.coast", "t": "meshom.meshfix.tri"}


def stage_fix(s: str):
    import importlib

    return importlib.import_module(STAGES[s]).fix


def rebase(case, mesh: Mesh):
    """The input case with `mesh` in place of its mesh (module docstring): kind 2 = the input's kind-2 nodes, found by
    position; kind 1 = the other boundary nodes of mesh; art_edges remapped; h_node = h at the nodes of mesh."""
    kind = build_topology(mesh).boundary_node.astype(np.int8)
    art_edges = np.zeros((0, 2), np.int64)
    art = np.flatnonzero(case.kind == 2)
    if len(art):
        d, idx = cKDTree(mesh.points).query(case.mesh.points[art])
        assert d.max() < 1e-12, "an artificial-boundary node moved or vanished"
        kind[idx] = 2
        if len(case.art_edges):
            m = np.full(case.mesh.n_nodes, -1, np.int64)
            m[art] = idx
            art_edges = m[case.art_edges]
    return dataclasses.replace(case, mesh=mesh, kind=kind, h_node=np.asarray(case.h(mesh.points)),
                               art_edges=art_edges, _dist_tree=case._dist_tree)


def run_stages(case, stages: list[str], opts: dict | None = None, verbose: bool = True):
    """Run the stages in sequence on the case (each stage on the input case rebased to the current mesh); (final
    mesh, per-stage records)."""
    opts = opts or {}
    mesh, recs = case.mesh, []
    for i, s in enumerate(stages):
        t = time.perf_counter()
        cur = rebase(case, mesh) if i else case
        t_rebase = time.perf_counter() - t
        nt = numba.get_num_threads()
        if s in SERIAL_STAGES:                              # B141: tiny patches, thread start-up dominates
            numba.set_num_threads(1)
        o = dict(opts.get(s, {}))
        held = 0
        if s == "a" and o.pop("guard", False):              # E's former A guard (comparison only)
            from meshom.meshfix.coast import a_slide_mask, ref_of

            sl = a_slide_mask(cur, cur.mesh, cur.kind, ref_of(case))
            kind = cur.kind.copy()
            kind[(cur.kind == 1) & ~sl] = 2
            held = int(((cur.kind == 1) & ~sl).sum())
            cur = dataclasses.replace(cur, kind=kind)
        try:
            mesh, rec = stage_fix(s)(cur, o)
        finally:
            numba.set_num_threads(nt)
        dt = time.perf_counter() - t
        recs.append({"stage": s, "method": rec.get("method"), "wall_s": dt, "rebase_s": t_rebase,
                     "method_wall_s": rec.get("wall_s"), "nodes": int(mesh.n_nodes), "quads": int(mesh.n_quads),
                     "tris": int(mesh.n_tris), **_angles(mesh), **({"a_guard_held": held} if held else {}),
                     "stats": rec.get("stats", rec.get("info"))})
        if verbose:
            r = recs[-1]
            if mesh.n_quads:
                print(f"stage {s} ({r['method']}): {dt:.0f} s, {r['nodes']} nodes, quads >110 / >120 "
                      f"{r['q110']} / {r['q120']}, max {r['qmax']:.1f}", flush=True)
            else:
                print(f"stage {s} ({r['method']}): {dt:.0f} s, {r['nodes']} nodes, triangles {r['tris']}, "
                      f"obtuse {r['t90']}, max angle {r['tmax']:.1f}", flush=True)
    return mesh, recs


def _angles(mesh: Mesh) -> dict:
    from meshom.quality.kernel import element_geometry

    g = element_geometry(mesh)
    qa = np.degrees(g.quad_angles).max(1) if mesh.n_quads else np.zeros(1)
    ta = np.degrees(g.tri_angles).max(1) if mesh.n_tris else np.zeros(1)
    return {"q110": int((qa > 110).sum()), "q120": int((qa > 120).sum()), "qmax": float(qa.max()),
            "t90": int((ta > 90).sum()), "tmax": float(ta.max())}


def run_case(run_dir: Path):
    """A whole case of a finished run, built as cases.build_whole does but nothing written (finish() without d)."""
    from meshom.compete import _load
    from meshom.config import load as load_cfg
    from meshom.meshfix import cases

    run_dir = Path(run_dir).resolve()
    cfg = load_cfg(run_dir / "config.resolved.toml")
    mesh = _load(run_dir / "raw.npz")
    h, h_src = cases._run_h(run_dir, cfg)
    q = json.loads((run_dir / "quality.json").read_text())
    rec = q.get("optimized", q.get("smooth", q["raw"]))
    case = cases.whole_case(run_dir.name, mesh, h, cfg.size_tolerance(), cfg.gates_all(), rec.get("rho_c", {}),
                            {"source": str(run_dir / "raw.npz"), "config": str(run_dir / "config.resolved.toml"),
                             "h_source": f"{run_dir.name}: {h_src}", "windows": [cases.WIN_GLOBAL]})
    return cases.finish(case)


def _val(s: str):
    for f in (int, float):
        try:
            return f(s)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None}.get(s.lower(), s)


def parse_opts(items: list[str]) -> dict:
    out = {}
    for x in items:
        k, v = x.split("=", 1)
        s, key = k.split(".", 1)
        out.setdefault(s, {})[key] = _val(v)
    return out


def main(argv=None):
    from meshom.compete import _save
    from meshom.meshfix import cases
    from meshom.meshfix.score import score_file, table

    ap = argparse.ArgumentParser(prog="python -m meshom.meshfix.pipeline", description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--case")
    g.add_argument("--run", type=Path)
    ap.add_argument("--stages", default="d,b,e,a")
    ap.add_argument("--name", default=None)
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args(argv)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    assert all(s in STAGES for s in stages), f"stages must be among {sorted(STAGES)}"
    tag = "_".join(stages)
    t0 = time.perf_counter()
    case = cases.load(a.case) if a.case else run_case(a.run)
    t_load = time.perf_counter() - t0
    name = a.name or f"p_{tag}"
    if a.case:
        d = cases.meshfix_dir() / "results" / name
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{a.case}.npz"
    else:
        f = Path(a.run).resolve() / f"fixed_{tag}.npz"
    t = time.perf_counter()
    mesh, recs = run_stages(case, stages, parse_opts(a.opt))
    wall = time.perf_counter() - t
    _save(mesh, f)
    rec = {"method": name, "case": case.name, "stages": stages, "wall_s": wall, "load_s": t_load,
           "per_stage": recs}
    f.with_suffix(".json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"{case.name} {name}: {wall:.0f} s -> {f}", flush=True)
    if not a.no_score:
        r = score_file(case.name, f, case=case, figures=not a.no_figure, label=name)
        print(table([r]), end="", flush=True)


if __name__ == "__main__":
    main()
