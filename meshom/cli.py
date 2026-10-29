"""meshom command line: `meshom quality`, `meshom bench-table`, `meshom compete`, `meshom figures`, `meshom run`,
`meshom runs`, `meshom ui`, `meshom repair`, `meshom write`, `meshom optimize`."""
import argparse
import json
import re
from pathlib import Path

from meshom import bench
from meshom.config import load_gates


def _quality(a: argparse.Namespace) -> None:
    from meshom.io.fesom import read_fesom
    from meshom.io.gmsh import read_gmsh41
    from meshom.quality.metrics import check_gates, compute_quality
    from meshom.quality.report import write_report

    path = Path(a.mesh)
    name = path.name if path.is_dir() else path.stem
    out = Path(a.out) if a.out else Path(".tmp/quality") / name
    rec: list[dict] = []
    with bench.stage("read", rec, mesh=name) as meta:
        mesh = read_fesom(path) if a.format == "fesom" else read_gmsh41(path, *a.center)
        meta.update(n_nodes=mesh.n_nodes, n_elems=mesh.n_elems)
    with bench.stage("quality", rec, mesh=name, n_nodes=mesh.n_nodes, n_elems=mesh.n_elems):
        q = compute_quality(mesh)
    gates = check_gates(q.summary, load_gates(a.gates))
    write_report(q, gates, mesh, out, name)
    bench.append_jsonl(rec, Path(a.bench))

    s = q.summary
    print(f"{name}: {s['n_nodes']} nodes, {s['n_tris']} tris, {s['n_quads']} quads; "
          + ", ".join(f"{r['stage']} {r['wall_s']:.2f} s" for r in rec) + f", peak {rec[-1]['peak_rss_mb']:.0f} MB")
    for g in gates:
        print(f"  {'ok  ' if g['passed'] else 'FAIL'} {g['gate']}: {g['value']} (limit {g['limit']})")
    print(f"  report: {out / 'report.md'}")


def _bench_table(a: argparse.Namespace) -> None:
    Path(a.out).write_text(bench.render_markdown(Path(a.inp)))


def _compete(a: argparse.Namespace) -> None:
    from meshom import compete
    from meshom.methods import METHODS, matching

    jsonl = Path(a.jsonl)
    if not a.table_only:
        smooth = a.smooth
        if smooth and not compete.smooth_available():
            print(compete.NO_SMOOTH)
            smooth = False
        names = list(dict.fromkeys(m for n in (a.methods or list(METHODS)) for m in matching(n)))
        for m in names:
            for c in a.cases:
                for s in a.scale:
                    r = compete.run_case(m, c, s, smooth=smooth, opts=a.opts, timeout=a.timeout, out=jsonl,
                                         figures=a.figures)
                    print(compete.one_line(r))
                    if "after_smoothing" in r:
                        print(compete.one_line(r["after_smoothing"]))
    text = compete.table(jsonl)
    if a.out:
        compete.write_table(Path(a.out), text)
        print(f"table: {a.out}")
    else:
        print(text)


def _figures(a: argparse.Namespace) -> None:
    from meshom.cases import CASES
    from meshom.compete import _load
    from meshom.figures import render_run
    from meshom.size import scaled

    path = Path(a.path)
    run = path if path.is_dir() else path.parent
    if a.scale is None:                                  # from the run directory name <method>_<case>_s<scale>
        m = re.search(r"_s(\d+(?:\.\d+)?)$", run.name)
        a.scale = float(m.group(1)) if m else 1.0
    domain, h0 = CASES[a.case]
    h = scaled(h0, a.scale)
    if path.is_dir():
        out = Path(a.out) if a.out else path / "figures"
        jobs = [(path / f"{k}.npz", out / k) for k in ("raw", "smooth") if (path / f"{k}.npz").exists()]
    else:
        jobs = [(path, Path(a.out) if a.out else path.parent / "figures" / path.stem)]
    for mesh_path, out in jobs:
        files = render_run(_load(mesh_path), domain, h, out,
                           title=f"{run.name if path.is_dir() else mesh_path.name} | {a.case} | scale {a.scale:g} | "
                                 f"{mesh_path.stem}")
        print(f"{mesh_path}: {len(files) - 1} figures in {out}")


def _run(a: argparse.Namespace) -> None:
    from meshom import compete
    from meshom.config import load
    from meshom.run import run, water_text

    cfg = load(Path(a.config))
    res = run(cfg, Path(a.out) if a.out else None)
    out = Path(a.out) if a.out else cfg.out_dir()
    for r in res.values():
        print(compete.one_line(dict(r, case=cfg.name)))
    w = next((r["water"] for r in res.values() if "water" in r), None)
    if w is not None and not w["connectivity_ok"]:
        print(water_text(w))
    pa = next((r["passages"] for r in res.values() if "passages" in r), None)
    if pa is not None and pa["n_narrow"]:                               # C51: a through-passage below the cells
        print(f"NARROW PASSAGES: {pa['n_narrow']} through-passages below {pa['cells']} cells across: "
              + "; ".join(f"{p['kind']} at {p['at']}" + (f" ({p['near']})" if p.get("near") else "")
                          for p in pa["narrow"][:20]))
    print(f"run directory: {out}")


def _runs(a: argparse.Namespace) -> None:
    from meshom.run import list_runs

    lines = list_runs(Path(a.dir))
    print("\n".join(lines) if lines else f"no runs under {a.dir}")


def _ui(a: argparse.Namespace, rest: list[str]) -> None:
    """The web UI server (ui/web/server.py): the `ui` package sits next to `meshom/` in the repository."""
    import sys

    repo = str(Path(__file__).resolve().parents[1])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from ui.web.server import main as serve

    serve(rest, prog="meshom ui")


def _repair(a: argparse.Namespace) -> None:
    """Repair a saved mesh (meshom.repair): RUN_DIR/<--mesh> -> RUN_DIR/<--out>.npz + <--out>.json; with --polygon
    only that zone (repair_region). Domain and h: the competition case (--case, --scale) or, without --case, the run
    directory's config (config.resolved.toml, else config.toml) built as `meshom run` builds them (coast_of,
    size_field; the cut mesh's own boundary as the domain under [coast] treatment = "cut"). The block mask (m14) is
    the npz's is_block (written by an earlier repair, so repairs chain), else m14.json's block-quad prefix for the
    method's own raw.npz / smooth.npz (not after a land cut, which renumbers the elements)."""
    import numpy as np

    from meshom import compete
    from meshom.repair import region_summary, repair, repair_region

    run = Path(a.run_dir)
    src, stem = run / a.mesh, run / a.out
    stem.parent.mkdir(parents=True, exist_ok=True)
    mesh = compete._load(src)
    with np.load(src) as z:
        blk = z["is_block"] if "is_block" in z else None
    if a.case is not None:
        from meshom.cases import CASES, STRAITS
        from meshom.size import scaled

        if a.scale is None:
            m = re.search(r"_s(\d+(?:\.\d+)?)$", run.name)
            a.scale = float(m.group(1)) if m else 1.0
        domain, h0 = CASES[a.case]
        h = scaled(h0, a.scale)
        gates, rho, strait, cut = load_gates(), compete._rho_c(a.case), STRAITS.get(a.case), False
        where, title = {"case": a.case, "scale": a.scale}, f"{a.case} | scale {a.scale:g}"
    else:
        from meshom.config import load
        from meshom.domain import closed_sphere
        from meshom.run import _boundary_domain, coast_domain, coast_of, size_field

        cf = run / "config.resolved.toml"
        cf = cf if cf.exists() else run / "config.toml"
        cfg = load(cf)
        coast = coast_of(cfg)
        h = size_field(cfg, coast)
        cut = coast is not None and cfg.coast.treatment == "cut"
        domain = (closed_sphere() if coast is None else _boundary_domain(mesh) if cut
                  else coast_domain(cfg, coast, h))
        gates, rho, strait = cfg.gates_all(), compete.rho_c_integrals(domain, h), None
        where, title = {"config": str(cf)}, f"{cfg.name} | {cfg.mesh.method} | scale {cfg.mesh.scale:g}"
    if blk is None and (run / "m14.json").exists() and src.name in ("raw.npz", "smooth.npz") and not cut:
        nb = json.loads((run / "m14.json").read_text())["stitch"]["n_block_quads"]   # m14: block quads come first
        blk = np.zeros(mesh.n_elems, dtype=bool)
        blk[mesh.n_tris:mesh.n_tris + nb] = True
    before = compete._score(mesh, domain, h, rho, gates, strait)
    if a.polygon is not None:                          # one explicit cavity (meshom.repair.repair_region)
        zone = region_summary(mesh, h, a.polygon, gates)
        fixed, info = repair_region(mesh, h, domain, a.polygon, a.opts, is_block=blk)
        print(f"region: {info['n_core']} elements inside, cavity {info.get('n_cavity')}, kind {info['kind']}, "
              f"accepted {info['accepted']}, forced {info['forced']}, {info.get('reason', '')}")
        for k in ("old", "new"):
            if k in info:
                print(f"  {k}: " + ", ".join(f"{x} {info[k][x]}" for x in ("gross", "net", "bad_elem", "bad_edge",
                                                                             "bad_ratio", "badness")))
    else:
        fixed, info = repair(mesh, h, domain, a.opts, is_block=blk)
    after = compete._score(fixed, domain, h, rho, gates, strait)
    oblk = info.pop("is_block_after", None)
    compete._save(fixed, Path(f"{stem}.npz"), **({} if blk is None else {"is_block": np.asarray(oblk, dtype=bool)}))
    rec = {"input": a.mesh, "polygon": a.polygon, "opts": info["options"], "before": before, "after": after,
           "repair": info, "run": str(run), **where}
    if a.polygon is not None:
        rec["region"] = {"before": zone, "after": region_summary(fixed, h, a.polygon, gates)}
    rec = dict(compete._round(rec), polygon=a.polygon)  # the polygon as given (not rounded)
    Path(f"{stem}.json").write_text(json.dumps(rec, indent=1))
    for name, r in (("before", before), ("after", after)):
        q, ic = r["quality"], r["irregular"]
        print(f"{name:6s} {r['n_nodes']} nodes, n3/n5/other/T {ic['n3']}/{ic['n5']}/{ic['n_other']}/{ic['T']}, "
              f"Euler {ic['euler_sum']} ({'ok' if ic['euler_ok'] else 'FAIL'}), quad max {q['quad_max_angle_deg']['max']}, "
              f"|ln s/h| p99 {q['size_fidelity']['p99']:.4f}, elem ratio p99 {q['elem_size_ratio']['p99']:.4f}, "
              f"gates {r['gates_passed']}/{r['gates_total']} {sorted(r['gates_failed'])}")
    if a.polygon is None:
        acc = {k: v for k, v in info["accepted"].items() if v}
        print(f"repair {info['time_s']['total']:.1f} s, accepted {acc}, gross {info['before']['gross']} -> "
              f"{info['after']['gross']}, bad edges {info['before']['bad_edges']} -> {info['after']['bad_edges']}; "
              f"{stem}.npz")
    if a.figures:
        from meshom.figures import render_run

        render_run(fixed, domain, h, run / "figures" / stem.name, title=f"{run.name} {stem.name} | {title}")


def _write(a: argparse.Namespace) -> None:
    from meshom.run import write_line, write_mesh

    print(write_line(write_mesh(Path(a.run_dir), a.mesh, a.format, Path(a.out) if a.out else None,
                                fesom_ini=a.fesom_ini)))

def _optimize(a: argparse.Namespace) -> None:
    """Optimise a saved mesh (meshom.optimize, decisions Z1): RUN_DIR/<--mesh> -> RUN_DIR/<--out>.npz and
    <--out>.json (before / after: compete._score of the whole mesh; the optimisation record). Domain, h and the size
    tolerance: the competition case (--case, --scale; the method, for its tolerance, from --method or the run
    directory name <method>_<case>_s<scale>) or, without --case, the run directory's config (config.resolved.toml,
    else config.toml) as `meshom run` builds them, its gates and Config.size_tolerance; --tol overrides the
    tolerance. A run directory with quality.json (`meshom run`) also gets the after-record under the key <--out>."""
    import datetime

    from meshom import compete
    from meshom.optimize import optimize, record

    run = Path(a.run_dir)
    src, stem = run / a.mesh, run / a.out
    mesh = compete._load(src)
    if a.case is not None:
        from meshom.cases import CASES, STRAITS
        from meshom.config import tolerance_gates
        from meshom.methods import effective_tolerance
        from meshom.size import scaled

        if a.scale is None:
            m = re.search(r"_s(\d+(?:\.\d+)?)$", run.name)
            a.scale = float(m.group(1)) if m else 1.0
        method = a.method or re.sub(rf"_{re.escape(a.case)}_s\d+(?:\.\d+)?$", "", run.name)
        domain, h0 = CASES[a.case]
        h = scaled(h0, a.scale)
        tol = effective_tolerance(method, {}) if a.tol is None else (a.tol or None)
        gates, rho, strait = tolerance_gates(load_gates(), tol), compete._rho_c(a.case), STRAITS.get(a.case)
        where = {"case": a.case, "scale": a.scale, "method": method}
        title = f"{method} | {a.case} | scale {a.scale:g}"
    else:
        from meshom.config import load, tolerance_gates
        from meshom.domain import closed_sphere
        from meshom.run import _boundary_domain, coast_domain, coast_of, size_field

        cf = run / "config.resolved.toml"
        cf = cf if cf.exists() else run / "config.toml"
        cfg = load(cf)
        coast = coast_of(cfg)
        h = size_field(cfg, coast)
        cut = coast is not None and cfg.coast.treatment == "cut"
        domain = (closed_sphere() if coast is None else _boundary_domain(mesh) if cut
                  else coast_domain(cfg, coast, h))
        tol = cfg.size_tolerance() if a.tol is None else (a.tol or None)
        gates = cfg.gates_all() if a.tol is None else tolerance_gates({**load_gates(), **cfg.gates}, tol, cfg.gates)
        rho, strait = compete.rho_c_integrals(domain, h), None
        where = {"config": str(cf), "method": cfg.mesh.method}
        title = f"{cfg.name} | {cfg.mesh.method} | scale {cfg.mesh.scale:g}"
    before = compete._score(mesh, domain, h, rho, gates, strait)
    out, info = optimize(mesh, h, tol, a.opts)
    after = compete._score(out, domain, h, rho, gates, strait)
    compete._save(out, Path(f"{stem}.npz"))
    rec = compete._round({"input": a.mesh, "run": str(run), "tol": tol, **where, "before": before, "after": after,
                          "optimize": record(info), "opts": info["opts"]})
    Path(f"{stem}.json").write_text(json.dumps(rec, indent=1))
    qf = run / "quality.json"
    if qf.exists():
        q = json.loads(qf.read_text())
        base = q.get(a.mesh.removesuffix(".npz"), q.get("raw", {}))
        q[a.out] = compete._round(dict({k: v for k, v in base.items() if k in ("config", "method", "scale", "opts")},
                                       date=datetime.datetime.now().isoformat(timespec="seconds"), status="ok",
                                       input=a.mesh, smoothed=a.mesh == "smooth.npz", wall_s=info["time_s"]["total"],
                                       optimize_s=info["time_s"]["total"], **after, optimize=record(info)))
        qf.write_text(json.dumps(q, indent=1))
    for name, r in (("before", before), ("after", after)):
        qq, ic = r["quality"], r["irregular"]
        print(f"{name:6s} {r['n_nodes']} nodes, T {qq['n_tris']}, tri cluster max {qq['tri_cluster_max']}, "
              f"n3/n5 {ic['n3']}/{ic['n5']}, Euler {ic['euler_sum']} ({'ok' if ic['euler_ok'] else 'FAIL'}), "
              f"quad max {qq['quad_max_angle_deg']['max']}, |ln s/h| p99 {qq['size_fidelity']['p99']:.4f}, "
              f"elem ratio p99 {qq['elem_size_ratio']['p99']:.4f}, gates {r['gates_passed']}/{r['gates_total']} "
              f"{sorted(r['gates_failed'])}")
    print(f"optimize {info['time_s']['total']:.1f} s, checks {info['checks']}; {stem}.npz")
    if a.figures:
        from meshom.figures import render_run

        render_run(out, domain, h, run / "figures" / stem.name, title=f"{run.name} {stem.name} | {title}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="meshom")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quality", help="quality metrics, gates and report of a mesh")
    q.add_argument("mesh", help="FESOM mesh directory or Gmsh .msh file")
    q.add_argument("--format", choices=["fesom", "gmsh41"], default="fesom")
    q.add_argument("--center", nargs=2, type=float, default=[16.0, 37.0], metavar=("LON", "LAT"),
                   help="stereographic centre of a Gmsh mesh (deg)")
    q.add_argument("--gates", type=Path, default=None, help="TOML file with a [gates] table")
    q.add_argument("--out", default=None, help="report directory (default .tmp/quality/<mesh name>)")
    q.add_argument("--bench", default="bench/results.jsonl", help="benchmark records (JSON lines, appended)")
    q.set_defaults(func=_quality)
    b = sub.add_parser("bench-table", help="render benchmark records as a Markdown table")
    b.add_argument("--in", dest="inp", default="bench/results.jsonl")
    b.add_argument("--out", default="docs/benchmarks.md")
    b.set_defaults(func=_bench_table)
    c = sub.add_parser("compete", help="run fill methods on the test cases, score them, render the table")
    c.add_argument("--methods", nargs="+", default=None,
                   help="registered names or mNN prefixes (default: every registered method)")
    c.add_argument("--cases", nargs="+", default=["uniform", "disc", "band", "continent"])
    c.add_argument("--scale", nargs="+", type=float, default=[4.0], help="h is multiplied by S (4: ~1e4 nodes)")
    c.add_argument("--smooth", action="store_true", help="also score after the common smoothing (meshom.smooth)")
    c.add_argument("--timeout", type=float, default=1800.0, help="seconds per method run (child process killed)")
    c.add_argument("--jsonl", default="bench/competition.jsonl", help="records (JSON lines, appended)")
    c.add_argument("--out", default=None, help="Markdown file whose table block is replaced (default: print)")
    c.add_argument("--table-only", action="store_true", help="only render the table from the records")
    c.add_argument("--opts", type=json.loads, default=None,
                   help='method options as JSON, passed to every run (e.g. \'{"block_gap": 12}\')')
    c.add_argument("--figures", action="store_true",
                   help="render the standard picture set of each scored mesh into <run dir>/figures/raw|smooth")
    c.set_defaults(func=_compete)
    f = sub.add_parser("figures", help="standard picture set (overview + zooms, index.json) of an existing mesh")
    f.add_argument("path", help="compete run directory (raw.npz, smooth.npz) or one mesh .npz")
    f.add_argument("--case", required=True, help="test case of the mesh (domain and h): uniform, disc, band, ...")
    f.add_argument("--scale", type=float, default=None,
                   help="h x S (default: from the run directory name <method>_<case>_s<S>, else 1)")
    f.add_argument("--out", default=None, help="output directory (default <run dir>/figures/{raw,smooth}, or "
                                               "<mesh dir>/figures/<mesh name> for one file)")
    f.set_defaults(func=_figures)
    r = sub.add_parser("run", help="build the mesh a config describes (coast, size layers, limiter, method, scores)")
    r.add_argument("config", help="meshom TOML config (docs/config.md)")
    r.add_argument("--out", default=None, help="run directory (default: [mesh] out, else .tmp/runs/<config stem>)")
    r.set_defaults(func=_run)
    rs = sub.add_parser("runs", help="list run directories with their status, one line each")
    rs.add_argument("dir", nargs="?", default=".tmp/runs", help="directory to search (default .tmp/runs)")
    rs.set_defaults(func=_runs)
    rp = sub.add_parser("repair", help="local cavity remeshing of the defect clusters of a saved mesh")
    rp.add_argument("run_dir", help="run directory: a `meshom run` folder (config.resolved.toml) or, with --case, a "
                                    "`meshom compete` one (and m14.json for m14 meshes)")
    rp.add_argument("--case", default=None, help="competition case of the mesh (domain and h); without it the run "
                                                 "directory's config.resolved.toml (else config.toml) gives them")
    rp.add_argument("--scale", type=float, default=None, help="with --case: h x S (default: from the run directory "
                                                              "name)")
    rp.add_argument("--mesh", default="raw.npz", help="mesh to repair, relative to RUN_DIR: raw.npz (default), "
                                                      "smooth.npz, an earlier repair's <NAME>.npz (repairs chain)")
    rp.add_argument("--out", default="repaired", help="output stem relative to RUN_DIR: <NAME>.npz and <NAME>.json "
                                                      "(default repaired)")
    rp.add_argument("--opts", type=json.loads, default=None,
                    help="repair options as JSON (meshom.repair.DEFAULTS), e.g. '{\"candidates\": [\"ops\", "
                         "\"coons\", \"m02\", \"m02_tri\"]}' (the fills tried, in order)")
    rp.add_argument("--figures", action="store_true", help="also render the picture set into RUN_DIR/figures/<NAME>")
    rp.add_argument("--polygon", type=json.loads, default=None,
                    help="remesh only this zone: '[[lon, lat], ...]' (great-circle edges; elements with the centroid "
                         "inside plus `rings` rings); --opts '{\"force\": true}' takes the best fill even without gain")
    rp.set_defaults(func=_repair)
    w = sub.add_parser("write", help="write a saved mesh of a run (a hand edit edited-NN, optimized, raw, smooth) "
                                     "as model files")
    w.add_argument("run_dir", help="a `meshom run` folder (config.resolved.toml)")
    w.add_argument("--mesh", default=None, help="raw, smooth, optimized, edited-NN, repaired (files of RUN_DIR) or a "
                                                "path to an .npz (default: the newest edited-NN.npz, else the run's "
                                                "final mesh: optimized, else smooth, else raw)")
    w.add_argument("--format", choices=["fesom-c", "fesom2", "fesom3"], default=None,
                   help="model format (default: the run's [mesh] format)")
    w.add_argument("--out", default=None, help="output folder (default RUN_DIR/fesom-<mesh name>; never RUN_DIR/fesom)")
    w.add_argument("--fesom-ini", action="store_true",
                   help="fesom3 only: run the config's [mesh] fesom3_ini on the written folder (work folder "
                        "RUN_DIR/fesom_ini-<mesh name>)")
    w.set_defaults(func=_write)
    op = sub.add_parser("optimize", help="method-independent optimisation of a saved mesh (flips, merges, triangle "
                                         "clusters, valence, smoothness fit)")
    op.add_argument("run_dir", help="run directory: a `meshom run` folder (config.resolved.toml) or, with --case, a "
                                    "`meshom compete` one")
    op.add_argument("--case", default=None, help="competition case of the mesh (domain and h); without it the run "
                                                 "directory's config gives them")
    op.add_argument("--scale", type=float, default=None, help="with --case: h x S (default: from the directory name)")
    op.add_argument("--method", default=None, help="with --case: the method, for its size tolerance (default: from "
                                                   "the directory name <method>_<case>_s<scale>)")
    op.add_argument("--tol", type=float, default=None, help="size tolerance (0 = none: the size gate 0.15); default "
                                                            "the run's (methods.effective_tolerance)")
    op.add_argument("--mesh", default="raw.npz", help="mesh to optimise, relative to RUN_DIR (default raw.npz)")
    op.add_argument("--out", default="optimized", help="output stem relative to RUN_DIR: <NAME>.npz and <NAME>.json "
                                                       "(default optimized)")
    op.add_argument("--opts", type=json.loads, default=None,
                    help="options as JSON (meshom.optimize.DEFAULTS), e.g. '{\"steps\": [\"merge\", \"cluster\"]}'")
    op.add_argument("--figures", action="store_true", help="also render the picture set into RUN_DIR/figures/<NAME>")
    op.set_defaults(func=_optimize)
    sub.add_parser("ui", add_help=False,
                   help="web UI: h builder, coast, zoom windows, method, runs, mesh view (options: meshom ui --help: "
                        "--config FILE, --runs DIR, --refs DIR, --port N, --host H, --token T | --no-token, "
                        "--allow-submit)")
    a, rest = p.parse_known_args(argv)
    if a.cmd == "ui":
        return _ui(a, rest)
    if rest:
        p.error(f"unrecognized arguments: {' '.join(rest)}")
    a.func(a)


if __name__ == "__main__":
    main()
