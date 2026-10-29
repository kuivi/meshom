"""Baselines of the mesh-fixing competition: the trivial methods that exist already, scored like every competitor.

    python -m meshom.meshfix.baselines --case NAME [NAME ...] --method M [M ...]   run, save, score
    python -m meshom.meshfix.baselines --table                                     baselines/table.md

Methods:
  input       the case's mesh unchanged
  optimize    meshom.optimize.optimize(mesh, h, tol) with its defaults (`meshom optimize`; decisions Z1-Z13): flips,
              merges, cluster breaking, valence pairs, the local fit; boundary nodes fixed (the artificial boundary
              of a cut-out is boundary)
  smooth_fit  meshom.size_fit.smooth_fit(mesh, h, tol) with its defaults over every interior node (node_flag is
              the boundary flag, so ORCA nodes are free too; boundary nodes fixed)

Results in <MESHFIX_DIR>/baselines/<method>/<case>.npz, .json (the record, wall_s), .score.json, .<window>.png.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from meshom.compete import _save
from meshom.meshfix.cases import CASE_SPECS, load, meshfix_dir
from meshom.meshfix.score import score_file, table

METHODS = ("input", "optimize", "smooth_fit")


def baselines_dir() -> Path:
    return meshfix_dir() / "baselines"


def run(case_name: str, method: str) -> dict:
    case = load(case_name)
    rec = {"method": method, "case": case_name, "tol": case.tol}
    t0 = time.perf_counter()
    if method == "input":
        mesh = case.mesh
    elif method == "optimize":
        from meshom.optimize import optimize, record

        mesh, info = optimize(case.mesh, case.h, case.tol)
        rec["optimize"] = record(info)
    elif method == "smooth_fit":
        from meshom.size_fit import smooth_fit

        mesh, info = smooth_fit(case.mesh, case.h, case.tol)
        mesh.tri_zone = case.mesh.tri_zone
        rec["smooth_fit"] = {k: info[k] for k in ("free_nodes", "before", "after", "total_s")}
        rec["smooth_fit"]["rounds"] = [{k: r[k] for k in ("iters", "stop", "s", "ratio_p99", "ln_sh_p99")}
                                       for r in info["rounds"]]
    else:
        raise KeyError(method)
    rec["wall_s"] = time.perf_counter() - t0
    d = baselines_dir() / method
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{case_name}.npz"
    _save(mesh, f)
    f.with_suffix(".json").write_text(json.dumps(rec, indent=1, default=float))
    t1 = time.perf_counter()
    r = score_file(case_name, f, case=case, label=f"baseline {method}")
    print(f"{case_name} {method}: method {rec['wall_s']:.0f} s, score {time.perf_counter() - t1:.0f} s", flush=True)
    return r


def write_table() -> Path:
    rows = []
    for c in CASE_SPECS:
        for m in METHODS:
            f = baselines_dir() / m / f"{c}.score.json"
            if f.exists():
                rows.append(json.loads(f.read_text()))
    out = baselines_dir() / "table.md"
    head = ("# Mesh-fixing baselines\n\nScorer: `meshom.meshfix.score` (decisions B121). Counts over the whole case; "
            "'open' = farther than 2 h from the case's coast. Irregular nodes: interior pure-quad nodes with 3 / 5 "
            "/ >= 6 quads; tri-nodes = interior nodes touching a triangle. Open defect units = open irregular "
            "pure-quad nodes + open triangles.\n\n")
    out.write_text(head + table(rows))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m meshom.meshfix.baselines")
    ap.add_argument("--case", nargs="*", default=[])
    ap.add_argument("--method", nargs="*", default=list(METHODS))
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args(argv)
    for c in a.case:
        for m in a.method:
            run(c, m)
    if a.table or a.case:
        print(write_table().read_text())


if __name__ == "__main__":
    main(sys.argv[1:])
