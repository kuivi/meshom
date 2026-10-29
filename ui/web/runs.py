"""Batch runs and reference meshes for the web UI: a run folder per "Run", the sbatch script, the run list, the log
tail, the quality.json records, and the reference meshes other models built.

A run folder (`<runs>/<YYYYmmdd-HHMMSS>-<method>/`) holds config.toml (the page's config: `[mesh] out` = the folder,
`[coast] source` absolute), run.sbatch, job.json (when submitted), slurm-<job>.log, and what `meshom run` writes there
(meshom/run.py, docs/config.md "Running"): config.resolved.toml, stages.json, quality.json ({"raw": record,
"smooth": record, "optimized": record}, records of meshom.compete._score), raw.npz, smooth.npz, optimized.npz (the
final mesh when the run applies the optimisation stage), fesom/, figures/; an m14 method also writes
m14.json; `meshom repair` writes repaired-NN.npz / .json, the page's hand editor edited-NN.npz / .json
(ui/web/edit.py). `meshom compete` folders (.tmp/compete/<method>_<case>_s<scale>: raw.npz, smooth.npz, stages.json)
and folders of `meshom run` started by hand (.tmp/runs/<config stem>) are run folders too. The run list takes the run
folders right in each run root, or with the root's "subfolders" tick those at any depth (find_runs: hidden folders such
as .worktrees and .tmp included, not inside a run folder); folders are rescanned only when their mtime changed, and a
run's description is kept while its files are unchanged.

The batch command (RUN_CMD) is the core's: `PYTHONPATH=<repo> <python> -m meshom.cli run <config> --out <dir>`, in an
sbatch script (account ab0995; partition, CPUs, memory, time from the
Runs panel, partition `shared` by default). Submitting needs the server flag --allow-submit; without it the script is
written and the command returned (dry run).

Reference meshes (REF_ROOTS, `meshom ui --refs DIR`): every folder under a root with mesh.npz in meshom's npz layout
(the layout of a run's raw.npz: points, tris, quads, node_flag), with quality.json ({"raw": record}: the _score-style
record without h, plus where the mesh comes from) and overlay.npz when it has them.
"""
import datetime
import getpass
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from meshom.config import DEFAULT_GATES, Config
from meshom.config import load as load_config
from ui.common import meshview, scenario

RUN_CMD = "PYTHONPATH={repo} {python} -m meshom.cli run {config} --out {dir}"
SBATCH = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --partition={partition}
#SBATCH --account=ab0995
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={dir}/slurm-%j.log
# {what}; written by the meshom web UI (ui/web/runs.py):  sbatch {script}
set -uo pipefail
cd {repo}
export NUMBA_NUM_THREADS={cpus} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "== $(date -Iseconds) host $(hostname) commit $(git rev-parse --short HEAD 2>/dev/null) dir {dir}"
{cmd}
rc=$?
echo "== rc $rc  end $(date -Iseconds)"
exit $rc
"""
# shared: memory is tied to CPUs (940 MB each): 0.9 GB x CPUs
RESOURCES = {"partition": "shared", "cpus": 8, "mem": "7G", "time": "01:00:00"}
MESH_GEN = scenario.main_checkout().parent
REF_ROOTS = [MESH_GEN / "curvilinear_grids", MESH_GEN / "triangle_meshes"]
RUN_MESHES = ("raw.npz", "smooth.npz", "optimized.npz", "mesh.npz")


def read_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def log_file(d: Path) -> Path | None:
    logs = sorted(list(d.glob("slurm-*.log")) + list(d.glob("*.log")) + list(d.glob("slurm-*.out")),
                  key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def tail(p: Path | None, n: int = 40) -> str:
    if p is None:
        return ""
    with open(p, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 64 * 1024))
        return "\n".join(f.read().decode(errors="replace").splitlines()[-n:])


_SQ = {"t": 0.0, "jobs": {}}


def squeue() -> dict:
    """{job id: state} of the user's jobs, cached for 5 s; {} where Slurm is not installed (e.g. a laptop)."""
    if time.time() - _SQ["t"] > 5:
        jobs = {}
        if shutil.which("squeue"):
            r = subprocess.run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%i %T"], capture_output=True,
                               text=True, timeout=20)
            jobs = dict(line.split()[:2] for line in r.stdout.splitlines() if line.strip())
        _SQ.update(t=time.time(), jobs=jobs)
    return _SQ["jobs"]


# ------------------------------------------------------------------------------------------- quality records

SUMMARY_KEYS = ("status", "method", "config", "smoothed", "scale", "wall_s", "generate_s", "peak_rss_mb", "n_nodes",
                "gates_passed", "gates_total", "gates_failed", "irregular", "error", "date",
                "family", "grid", "res", "layout", "source", "licence", "generator")


def record_summary(rec: dict | None) -> dict | None:
    """The fields of a quality.json record the page shows (counts, gates, irregular counts, size fidelity)."""
    if not rec:
        return None
    out = {k: rec[k] for k in SUMMARY_KEYS if k in rec}
    q = rec.get("quality") or {}
    for k in ("n_quads", "n_tris", "n_elems"):
        if k in q:
            out[k] = q[k]
    if "size_fidelity" in q:
        out["size_fidelity"] = q["size_fidelity"]
    return out


# the gates of meshom.config.DEFAULT_GATES in plain words (gate family -> what it bounds), their groups and regions;
# GATE_UNITS: the unit of the value, GATE_COUNTS: gates on a count (shown as integers)
GATE_UNITS = {"quad_max_angle_hard_deg": "°", "quad_max_angle_target_deg": "°", "tri_max_angle_hard_deg": "°",
              "tri_max_angle_target_deg": "°", "tri_min_angle_hard_deg": "°"}
GATE_COUNTS = {"elem_neighbours_min", "elems_per_node_max", "tri_cluster_max", "boundary_edges_max", "pinch_nodes_max"}
GATE_WORDS = {
    "quad_max_angle_hard_deg": ("angles", "largest quad angle, every quad"),
    "quad_max_angle_target_deg": ("angles", "largest quad angle, 99.9 % of quads"),
    "tri_max_angle_hard_deg": ("angles", "largest triangle angle, every triangle"),
    "tri_max_angle_target_deg": ("angles", "largest triangle angle, 99 % of triangles"),
    "tri_min_angle_hard_deg": ("angles", "smallest triangle angle, every triangle"),
    "quad_sj_hard": ("angles", "sine of the sharpest quad corner, every quad"),
    "elem_size_ratio_p99": ("size", "size jump across an edge, 99 % of edges"),
    "elem_size_ratio_max": ("size", "size jump across an edge, every edge"),
    "size_fidelity_p99": ("size", "|ln(edge length / h)|, 99 % of edges"),
    "elem_neighbours_min": ("structure", "neighbours of an element, fewest"),
    "elems_per_node_max": ("structure", "elements at one node, most"),
    "tri_cluster_max": ("structure", "triangles in one cluster outside triangle zones, most"),
    "boundary_edges_max": ("structure", "boundary edges of one element, most"),
    "pinch_nodes_max": ("structure", "pinched boundary nodes"),
    "tri_share_max": ("structure", "share of triangles"),
    "euler_ok": ("structure", "topology (Euler) check"),
}
REGION_WORDS = {"": "whole mesh", "_coast": "coastal cells", "_coast3": "3 cells from the coast", "_interior": "open water",
                "_strait": "straits"}
GROUP_ORDER = ("angles", "size", "structure")


def gate_table(rec: dict | None, gates: dict) -> list[dict]:
    """Every applicable gate of a quality.json record (meshom.quality.metrics.check_gates on its quality summary and
    Euler check): what it bounds in words, where, value, limit, comparison, passed; failed first, then by group."""
    from meshom.quality.metrics import check_gates, gate_family
    if not rec or not rec.get("quality"):
        return []
    irr = rec.get("irregular") or {}
    rows = check_gates({**rec["quality"], **({"euler_ok": irr["euler_ok"]} if "euler_ok" in irr else {})}, gates)
    out = []
    for r in rows:
        if r["value"] is None:
            continue
        fam = gate_family(r["gate"])
        group, what = GATE_WORDS.get(fam, ("structure", fam))
        out.append({"gate": r["gate"], "group": group, "what": what, "where": REGION_WORDS.get(r["gate"][len(fam):], ""),
                    "value": r["value"], "limit": r["limit"], "op": {"le": "≤", "ge": "≥", "eq": "="}[r["op"]],
                    "unit": GATE_UNITS.get(fam, ""), "count": fam in GATE_COUNTS, "passed": r["passed"]})
    return sorted(out, key=lambda x: (x["passed"], GROUP_ORDER.index(x["group"])))


# histogram name (ui.common.meshview.MeshView.histograms) -> (summary metric, statistics) of a quality.json record
RECORD_STATS = {"quad θmax": ("quad_max_angle_deg", ("p99_9", "max")),
                "triangle θmax": ("tri_max_angle_deg", ("p99", "max")),
                meshview.SIZE_RATIO: ("elem_size_ratio", ("median", "p99", "max")),
                "|ln s/h| per edge": ("size_fidelity", ("median", "p99", "max", "bias"))}


def with_record(hists: list[dict], rec: dict | None) -> list[dict]:
    """The overlay's histograms (bins) with the official statistics of the quality.json record attached as "record"
    (meshom.quality.metrics.compute_quality on the same mesh: the numbers the gates use); a |ln s/h| entry without
    bins is added when the record has size fidelity and the overlay does not (h unknown to the UI)."""
    q = (rec or {}).get("quality") or {}
    names = {h["name"] for h in hists}
    if "size_fidelity" in q and "|ln s/h| per edge" not in names:
        hists = hists + [{"name": "|ln s/h| per edge", "unit": "", "lo": 0.0, "hi": 1.0, "step": 0.02, "counts": [],
                          "n": 0, "gates": [["p99", DEFAULT_GATES["size_fidelity_p99"]]], "p99": None,
                          "p99_9": None, "max": None}]
    for h in hists:
        metric, stats = RECORD_STATS.get(h["name"], (None, ()))
        if metric in q and q[metric]:
            h["record"] = {s: q[metric].get(s) for s in stats}
    return hists


# ----------------------------------------------------------------------------------------------------- runs

def run_config(d: Path) -> Config | None:
    """The config of a run folder: config.resolved.toml (what `meshom run` used), else config.toml (what Run wrote
    before the job started); None without one, or for a file of an older schema (old run folders on disk)."""
    for name in ("config.resolved.toml", "config.toml"):
        if (d / name).is_file():
            try:
                return load_config(d / name)
            except ValueError:                       # ConfigError, TOMLDecodeError
                return None
    return None


def h_range_km(cfg: Config) -> list[float] | None:
    """[finest, base] h of a config in km, as the run table shows it: the smallest h its enabled layers can take (the
    core's run.h_floor: a latitude or orca layer's h_min_m, no smooth zone) and Config.base_h, both inside the [size]
    clip h_min_m / h_max_m and times [mesh] scale; None without layers."""
    from meshom.run import h_floor

    base = cfg.base_h()
    if not cfg.layers() or base is None:
        return None
    lo, hi = cfg.size.h_min_m or 0.0, cfg.size.h_max_m or float("inf")
    clip = lambda h: min(max(h, lo), hi)  # noqa: E731
    return [round(clip(h_floor(cfg)) * cfg.mesh.scale / 1000, 6), round(clip(base) * cfg.mesh.scale / 1000, 6)]


def run_meshes(d: Path) -> list[str]:
    """The mesh files of a run folder: raw, smooth, optimized, mesh, then repaired-NN and edited-NN."""
    return [n for n in RUN_MESHES if (d / n).exists()] + sorted(p.name for p in d.glob("repaired-*.npz")) + \
        sorted(p.name for p in d.glob("edited-*.npz"))


def describe(d: Path) -> dict:
    meshes = run_meshes(d)
    stages = read_json(d / "stages.json") or []
    quality = read_json(d / "quality.json")
    job = read_json(d / "job.json")
    log = log_file(d)
    final = ((quality.get("optimized") or quality.get("smooth") or quality.get("raw"))       # the final mesh's record
             if isinstance(quality, dict) else None)
    sq = squeue()
    if job and str(job.get("job_id")) in sq:
        state = sq[str(job["job_id"])].lower()
    elif final is not None:
        state = "done" if final.get("status") == "ok" else final.get("status", "error")
    elif log is not None and re.search(r"Traceback|Error|rc [1-9]", tail(log, 60)):
        state = "failed"
    elif meshes:
        state = "done"
    elif job:
        state = "ended without quality.json"
    elif (d / "config.resolved.toml").exists():
        state = "running?" + (f" (last stage {stages[-1].get('stage')})" if stages else "")
    else:
        state = "not started" if (d / "run.sbatch").exists() else "no mesh"
    line = None
    if (d / "config.resolved.toml").exists():
        from meshom.run import run_line
        line = run_line(d, d.parent)
    cfg = run_config(d)
    date = (final or {}).get("date") or datetime.datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds")
    return {"name": d.name, "path": str(d), "root": str(d.parent), "mtime": d.stat().st_mtime, "state": state,
            "date": date, "h_km": h_range_km(cfg) if cfg else None,
            "scale": cfg.mesh.scale if cfg else (final or {}).get("scale"),
            "meshes": meshes, "job": job, "log": str(log) if log else None, "m14": (d / "m14.json").exists(),
            "config": (d / "config.toml").exists() or (d / "config.resolved.toml").exists(), "line": line,
            "stages": [{"stage": s.get("stage"), "mesh": s.get("mesh"), "wall_s": s.get("wall_s"), "peak_rss_mb": s.get("peak_rss_mb")}
                       for s in stages if isinstance(s, dict)],
            "quality": {k: record_summary(v) for k, v in quality.items() if isinstance(v, dict)}
            if isinstance(quality, dict) else None}


# a run folder holds one of these (or a repaired-NN / edited-NN mesh); a search of subfolders does not enter SKIP_DIRS,
# goes at most WALK_DEPTH folders below a root and stops after WALK_DIRS folders a root (the list says so: "truncated")
RUN_MARKERS = {*RUN_MESHES, "quality.json", "stages.json", "config.toml", "config.resolved.toml", "run.sbatch", "job.json"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints"}
WALK_DEPTH, WALK_DIRS = 8, 20000
_DIRS: dict[str, tuple[int, bool, list[str]]] = {}          # folder -> (mtime ns, a run folder, its subfolders)
_DESC: dict[Path, tuple[tuple, dict]] = {}                  # run folder -> (its files' mtimes, describe())


def scan_dir(p: str) -> tuple[bool, list[str]]:
    """(p is a run folder, its subfolders but SKIP_DIRS, sorted), read again only when p's mtime changed."""
    m = os.stat(p).st_mtime_ns
    hit = _DIRS.get(p)
    if hit and hit[0] == m:
        return hit[1], hit[2]
    names, subs = set(), []
    with os.scandir(p) as it:
        for e in it:
            names.add(e.name)
            if e.name not in SKIP_DIRS and e.is_dir(follow_symlinks=False):
                subs.append(e.name)
    run = bool(names & RUN_MARKERS) or any(n.startswith(("repaired-", "edited-")) and n.endswith(".npz") for n in names)
    _DIRS[p] = (m, run, sorted(subs))
    return run, _DIRS[p][2]


def find_runs(root: Path, deep: bool = False) -> tuple[list[Path], dict]:
    """The run folders right in root, with deep those at any depth below it (root itself when it is one; not those
    inside a run folder), and the search's record {path, deep, dirs: folders read, truncated}."""
    out, stack, seen, truncated = [], [(str(root), 0)], 0, False
    limit = WALK_DEPTH if deep else 1
    while stack:
        p, depth = stack.pop()
        if seen >= WALK_DIRS:
            truncated = True
            break
        try:
            run, subs = scan_dir(p)
        except OSError:                      # vanished or unreadable meanwhile
            continue
        seen += 1
        if run:
            out.append(Path(p))
        elif depth < limit:
            stack += [(os.path.join(p, n), depth + 1) for n in reversed(subs)]
    return out, {"path": str(root), "deep": deep, "dirs": seen, "truncated": truncated}


def _mtime(p: Path) -> int | None:
    try:
        return os.stat(p).st_mtime_ns
    except OSError:
        return None


def describe_cached(d: Path) -> dict:
    """describe(d), kept while the folder and its records are unchanged; never kept for a running run (its state comes
    from squeue or the log)."""
    key = tuple(_mtime(d / n) for n in ("", "quality.json", "stages.json", "job.json", "config.toml",
                                        "config.resolved.toml"))
    hit = _DESC.get(d)
    if hit and hit[0] == key:
        return hit[1]
    x = describe(d)
    if not (x["state"].startswith("running") or (x["job"] and str(x["job"].get("job_id")) in squeue())):
        _DESC[d] = (key, x)
    return x


def scan_runs(roots: list[Path], deep: set[Path] = frozenset()) -> tuple[list[dict], list[dict]]:
    """(every run of the roots (find_runs; the roots in deep with their subfolders), newest first, each with `folder`:
    the folder it sits in, relative to its root's parent; one search record per root). A run under two roots is listed
    once."""
    out, scans, seen = [], [], set()
    for r in map(Path, roots):
        if not r.is_dir():
            scans.append({"path": str(r), "dirs": 0, "truncated": False, "missing": True})
            continue
        ds, info = find_runs(r, r in deep)
        info["runs"] = 0
        for d in ds:
            if d in seen:
                continue
            seen.add(d)
            info["runs"] += 1
            out.append({**describe_cached(d), "folder": str(d.parent.relative_to(r.parent))})
        scans.append(info)
    return sorted(out, key=lambda x: -x["mtime"]), scans


def list_runs(roots: list[Path]) -> list[dict]:
    return scan_runs(roots)[0]


def list_dirs(p: Path, limit: int = 500) -> dict:
    """A folder's subfolders for the page's folder browser: {path, parent, dirs: [{name, run}], more: left out}."""
    _, subs = scan_dir(str(p))
    dirs = []
    for n in subs[:limit]:
        try:
            dirs.append({"name": n, "run": scan_dir(str(p / n))[0]})
        except OSError:
            continue
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "dirs": dirs,
            "more": max(0, len(subs) - limit)}


def submit(cfg: Config, root: Path, repo: Path, python: str, cmd_template: str = RUN_CMD, allow: bool = False,
           resources: dict | None = None) -> dict:
    """A run folder with the config (out = the folder, coast source absolute) and run.sbatch; sbatch when allowed."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = (Path(root) / f"{stamp}-{cfg.mesh.method}").resolve()
    d.mkdir(parents=True)
    src = cfg.coast.source if cfg.coast.source == "none" else str(scenario.resolve(cfg.coast.source))
    cfg = replace(cfg, mesh=replace(cfg.mesh, out=str(d)), coast=replace(cfg.coast, source=src))
    scenario.save(cfg, d / "config.toml")
    cmd = cmd_template.format(repo=repo, python=python, config=d / "config.toml", dir=d)
    res = dict(RESOURCES, **(resources or {}))
    script = d / "run.sbatch"
    script.write_text(SBATCH.format(dir=d, repo=repo, cmd=cmd, job="meshom_run", script=script,
                                    what="`meshom run` of the config in this folder", **res))
    out = {"dir": str(d), "command": cmd, "script": str(script), "submitted": False, "resources": res}
    return _sbatch(out, script, d / "job.json", repo, allow)


def _sbatch(out: dict, script: Path, job_file: Path, repo: Path, allow: bool) -> dict:
    """Submit script when allowed (the job id into job_file), else say how to submit it."""
    if not allow:
        out["note"] = f"dry run: start the server with --allow-submit to submit (sbatch {script})"
        return out
    r = subprocess.run(["sbatch", "--parsable", str(script)], cwd=repo, capture_output=True, text=True, timeout=60)
    if r.returncode:
        out["error"] = r.stderr.strip()
        return out
    job = {"job_id": r.stdout.strip().split(";")[0], "submitted": datetime.datetime.now().isoformat(timespec="seconds")}
    job_file.write_text(json.dumps(job))
    out.update(submitted=True, job_id=job["job_id"])
    return out


# ------------------------------------------------- results of `meshom repair` (repaired-NN.npz / .json)

def _get(d: dict, *keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def repair_summary(rec: dict) -> dict:
    """The page's view of a repaired-NN.json (meshom repair): what was asked and what came out, and rows of before /
    after for the zone (its elements only, without the rings) and the whole mesh."""
    r, reg = rec.get("repair") or {}, rec.get("region") or {}
    acc = r.get("accepted")
    acc = acc if isinstance(acc, bool) else bool(acc and any(acc.values()))   # the automatic repair: counts per fill
    zb, za, wb, wa = reg.get("before") or {}, reg.get("after") or {}, rec.get("before") or {}, rec.get("after") or {}
    rows = []
    for what, zone_key, whole_key in (
            ("elements", ("n_elems",), ("quality", "n_elems")),
            ("triangles", ("n_tris",), ("quality", "n_tris")),
            ("irregular nodes n3", ("n3",), ("irregular", "n3")),
            ("irregular nodes n5", ("n5",), ("irregular", "n5")),
            ("largest quad angle, °", ("quality", "quad_max_angle_deg", "max"), ("quality", "quad_max_angle_deg", "max")),
            ("size jump across an edge, p99", ("quality", "elem_size_ratio", "p99"), ("quality", "elem_size_ratio", "p99")),
            ("|ln(edge / h)|, p99", ("quality", "size_fidelity", "p99"), ("quality", "size_fidelity", "p99")),
            ("gates passed", ("gates_passed",), ("gates_passed",))):
        rows.append({"what": what, "zone": [_get(zb, *zone_key), _get(za, *zone_key)] if reg else None,
                     "whole": [_get(wb, *whole_key), _get(wa, *whole_key)],
                     "higher_is_better": what == "gates passed"})
    return {"input": rec.get("input"), "polygon": rec.get("polygon"), "accepted": acc, "forced": r.get("forced"),
            "kind": r.get("kind"), "reason": r.get("reason"), "n_core": r.get("n_core"), "n_cavity": r.get("n_cavity"),
            "gates_total": [wb.get("gates_total"), wa.get("gates_total")], "rows": rows}


# --------------------------------------------------------------------------- comparison of 2-4 quality records

def _per_1000(rec: dict, k: str):
    n, irr = rec.get("n_nodes"), rec.get("irregular") or {}
    return 1000.0 * irr[k] / n if n and irr.get(k) is not None else None


def _tri_share(q: dict):
    return q["n_tris"] / q["n_elems"] if q.get("n_elems") and q.get("n_tris") is not None else None


# what, value of (record, its quality summary), better ("low", "high"; None: no best marked)
COMPARE_ROWS = (
    ("nodes", lambda r, q: r.get("n_nodes", q.get("n_nodes")), None),
    ("quads", lambda r, q: q.get("n_quads"), None),
    ("triangles", lambda r, q: q.get("n_tris"), None),
    ("triangle share", lambda r, q: _tri_share(q), "low"),
    ("n3 per 1000 nodes", lambda r, q: _per_1000(r, "n3"), None),   # not ranked: a triangle mesh counts differently
    ("n5 per 1000 nodes", lambda r, q: _per_1000(r, "n5"), None),   # not ranked: a triangle mesh counts differently
    ("largest quad angle, max", lambda r, q: _get(q, "quad_max_angle_deg", "max"), "low"),
    ("largest quad angle, p99.9", lambda r, q: _get(q, "quad_max_angle_deg", "p99_9"), "low"),
    ("largest triangle angle, p99", lambda r, q: _get(q, "tri_max_angle_deg", "p99"), "low"),
    ("largest triangle angle, max", lambda r, q: _get(q, "tri_max_angle_deg", "max"), "low"),
    ("size jump across an edge, p99", lambda r, q: _get(q, "elem_size_ratio", "p99"), "low"),
    ("|ln(edge / h)|, p99", lambda r, q: _get(q, "size_fidelity", "p99"), "low"),
)


def _best(values: list, better: str | None) -> list[int]:
    """Indices of the best value (ties included); none when fewer than two meshes have the number."""
    xs = [(i, v) for i, v in enumerate(values) if v is not None]
    if better is None or len(xs) < 2:
        return []
    b = (min if better == "low" else max)(v for _, v in xs)
    return [i for i, v in xs if abs(v - b) <= 1e-9 * max(1.0, abs(b))]


def compare(cols: list[dict]) -> dict:
    """The quality records of several meshes side by side. cols: [{path, file, rec (quality record or None), gates
    (the limits of the mesh's config)}]. Returns a column per mesh (name, path), rows of numbers with the best marked,
    and the gates every mesh with a record has (gate_table rows): passed of them, the failed ones by name."""
    meshes = []
    for c in cols:
        f = Path(c["file"])
        meshes.append({"path": c["path"], "file": str(f), "has_record": bool(c["rec"]),
                       "name": f.parent.name if f.name == "mesh.npz" else f"{f.parent.name} / {f.stem}",
                       "status": (c["rec"] or {}).get("status")})
    rows = []
    for what, f, better in COMPARE_ROWS:
        vals = [f(c["rec"], c["rec"].get("quality") or {}) if c["rec"] else None for c in cols]
        rows.append({"what": what, "values": vals, "best": _best(vals, better)})
    tables = [{g["gate"]: g for g in gate_table(c["rec"], c["gates"])} if c["rec"] else None for c in cols]
    have = [t for t in tables if t]
    names = [g for g in have[0] if all(g in t for t in have[1:])] if have else []
    gcols = [None if t is None else {"total": len(names), "passed": sum(t[g]["passed"] for g in names),
                                     "failed": [{"gate": g, "what": t[g]["what"], "where": t[g]["where"]}
                                                for g in names if not t[g]["passed"]]} for t in tables]
    return {"meshes": meshes, "rows": rows, "gates": {
        "names": names, "columns": gcols, "best": _best([c and c["passed"] for c in gcols], "high")}}


# ------------------------------------------------------------------------------------------ reference meshes

EQUILATERAL_EDGE = (4 / 3 ** 0.5) ** 0.5       # edge / sqrt(area) of an equilateral triangle (1.52)


def typical_size_km(rec: dict | None) -> float | None:
    """A mesh's typical element size in km, to sort meshes by resolution: the median sqrt(area) of its elements
    (quality sqrt_area_m), for a mesh of mostly triangles the edge of an equilateral triangle of that area."""
    q = (rec or {}).get("quality") or {}
    med = (q.get("sqrt_area_m") or {}).get("median")
    if not med:
        return None
    return med / 1000 * (EQUILATERAL_EDGE if q.get("n_tris", 0) > q.get("n_quads", 0) else 1.0)


def list_refs(roots: list[Path]) -> list[dict]:
    """Reference meshes: folders with mesh.npz under the roots, with their quality.json summary and typical element
    size (res_km)."""
    out = []
    for r in map(Path, roots):
        if not r.is_dir():
            continue
        for d in sorted(r.iterdir()):
            if not (d / "mesh.npz").is_file():
                continue
            q = read_json(d / "quality.json")
            rec = q.get("raw") if isinstance(q, dict) else None
            out.append({"name": d.name, "path": str(d), "root": str(r), "quality": record_summary(rec),
                        "res_km": typical_size_km(rec), "mb": round((d / "mesh.npz").stat().st_size / 1e6, 1)})
    return out


# ---------------------------------------------------------------------------------------------------- blocks

def blocks(d: Path) -> dict:
    """The m14 block records of a run (m14.json): seeds, sizes, outlines (lon-lat) and their corners."""
    m = read_json(d / "m14.json")
    if not m:
        return {"blocks": [], "note": "no m14.json in this run"}
    recs = []
    for i, b in enumerate(m.get("blocks", [])):
        recs.append({"i": i, "kept": i in m.get("blocks_kept", []), "seed": b.get("seed_lonlat"),
                     "cells": b.get("n_cells", b.get("region_cells")), "B": b.get("B"),
                     "outline": b.get("outline_lonlat"), "corners": b.get("outline_corners")})
    has = any(r["outline"] for r in recs)
    return {"method": m.get("method"), "n_blocks": m.get("n_blocks"), "coverage_area": m.get("coverage_area"),
            "blocks": recs, "note": None if has else
            "this m14.json records no outlines (written before outline_lonlat existed): seeds are shown"}
