"""Model files of a saved mesh from the page ("Write model files" in the Mesh panel and in the hand editor): the core's
`meshom write` (meshom.run.write_mesh) on a mesh of a `meshom run` folder (raw, smooth, optimized, repaired-NN,
edited-NN), in the run's format or another, into <run>/fesom-<mesh stem>/ with the format's checks, the depth from the
run's [bathymetry] and the mesh scored with the run's gates. It runs off the request thread and one at a time (the
first write of a run computes its size field and depth: 288 s for a 281 k-node Arctic mesh on 8 CPUs, 3.1 GB peak with
the mesh shown; later writes read them from <run>/cache/, ~30 s after a small edit, decisions O33); the job tells the
stage write_mesh is in (its progress callback). With the tick, write_mesh runs FESOM3's fesom_ini on the written
folder ([mesh] fesom3_ini, format fesom3; its log in <run>/fesom_ini-<mesh stem>/, its record in the write's). The
page shows summary(): the folder, the checks (the STOP items in words), the gates (value and limit), the straits and
the narrow passages.

    GET  /api/mesh/write?path=MESH   targets(): the run's meshes, the folder each is written to and its last record
    POST /api/mesh/write             {path, format, fesom_ini} -> Writer.start (409 while a write runs)
    GET  /api/mesh/write/job         Writer.job(): idle | running | done (summary) | error
"""
import threading
import time
from pathlib import Path

from meshom import config as mconfig
from ui.common import meshview
from ui.web import runs

FORMATS = ("fesom-c", "fesom2", "fesom3")

# the checks of meshom.io.fesom (fesom2_checks, fesom3_checks) that stop the model, in words
CHECK_WORDS = {
    "elems_lt2_neighbours": "elements with fewer than 2 neighbours",
    "elems_per_node_max": "elements at one node, most (≤ 32)",
    "node_neighbours_max": "neighbour nodes of one node, quad diagonals included, most (≤ 32)",
    "quads_bowtie_rotated": "bow-tie quads in rotated lon-lat (fesom_ini)",
    "quads_nonconvex_rotated": "non-convex quads in rotated lon-lat (fesom_ini)",
    "quads_nonconvex_sphere": "quads not convex on the sphere",
    "elems_not_clockwise": "elements not clockwise on the sphere",
    "nl": "vertical levels",
    "aux3d_depths": "depths in aux3d.out (one per node)",
    "depth_max_m": "deepest node, m (≤ 11000)",
    "nodes_in_rotated_pole_box": "nodes within ±2.5° of the rotated North Pole",
    "elems_containing_rotated_north_pole": "elements containing the rotated North Pole",
    "elems_containing_rotated_south_pole": "elements containing the rotated South Pole",
}
STRAIT_KEYS = ("name", "ok", "connected", "cells_min", "cells_target", "closed_sections", "width_min_m",
               "depth_min_m", "at")


def run_of(mf: Path) -> tuple[Path, mconfig.Config]:
    """(run folder, its config) of a mesh file: write_mesh writes with config.resolved.toml, `meshom run`'s."""
    d = Path(mf).parent
    if not (d / "config.resolved.toml").is_file():
        raise ValueError(f"{d.name} has no config.resolved.toml (a `meshom run` folder has one): no format, depth or "
                         "gates to write the mesh with")
    return d, mconfig.load(d / "config.resolved.toml")


def out_of(run_dir: Path, name: str) -> Path:
    """The folder write_mesh writes a mesh file `name` of the run to (its default)."""
    return run_dir / f"fesom-{Path(name).stem}"


def summary(rec: dict, gates: dict) -> dict:
    """What the page shows of a write_mesh record: the folder, the counts, the format's checks (ok, the STOP items with
    their values in words, every other item), the gates (runs.gate_table: failed first), the straits and the water
    connectivity, the narrow passages and fesom_ini's record when the record has them, the depth regions applied
    ([[bathymetry.regions]], from the bathymetry stage) and the node edits (C2), the stage times."""
    key = next((k for k in ("fesom3_checks", "fesom2_checks") if k in rec), None)
    checks = None
    if key:
        c = rec[key]
        checks = {"name": key, "ok": not c["stop"],
                  "stop": [{"key": k, "value": c.get(k), "what": CHECK_WORDS.get(k, k)} for k in c["stop"]],
                  "items": [{"key": k, "value": v} for k, v in c.items() if k != "stop"]}
    straits = None
    if rec.get("straits") is not None:
        w = rec.get("water") or {}
        straits = {"rows": [{k: s.get(k) for k in STRAIT_KEYS} for s in rec["straits"]],
                   "connectivity_ok": w.get("connectivity_ok"), "n_components": w.get("n_components"),
                   "not_connected": w.get("not_connected", [])}
    q, irr = rec.get("quality") or {}, rec.get("irregular") or {}
    bathy = next((st for st in rec.get("stages", []) if st.get("stage") == "bathymetry"), {})
    return {"mesh": Path(rec["mesh"]).name, "out": rec["out"], "format": rec["format"], "date": rec.get("date"),
            "n_nodes": rec.get("n_nodes"), "n_tris": q.get("n_tris"), "n_quads": q.get("n_quads"),
            "euler_ok": irr.get("euler_ok"), "checks": checks,
            "gates": runs.gate_table(rec, gates), "gates_passed": rec.get("gates_passed"),
            "gates_total": rec.get("gates_total"), "straits": straits, "passages": rec.get("passages"),
            "fesom_ini": rec.get("fesom_ini"), "regions": bathy.get("regions"), "depth_edits": rec.get("depth_edits"),
            "wall_s": round(sum(s.get("wall_s") or 0.0 for s in rec.get("stages", [])), 2)}


def written(run_dir: Path, name: str, gates: dict) -> dict | None:
    """summary() of a mesh file's last write, read from <out>/quality.json; None if never."""
    q = runs.read_json(out_of(run_dir, name) / "quality.json")
    rec = q.get(Path(name).stem) if isinstance(q, dict) else None
    if not isinstance(rec, dict):
        return None
    return summary(rec, gates)


def targets(mf: Path, job: dict) -> dict:
    """The page's window for a mesh file: its run folder, the run's meshes with the folder each is written to and its
    last write, the run's format, the formats, the fesom_ini binary of the config ("" = none) and the job."""
    d, cfg = run_of(mf)
    gates = cfg.gates_all()
    return {"run_dir": str(d), "mesh": Path(mf).name, "format": cfg.mesh.format, "formats": list(FORMATS),
            "fesom3_ini": cfg.mesh.fesom3_ini or "", "job": job,
            "meshes": [{"name": n, "out": str(out_of(d, n)), "written": written(d, n, gates), **edits_state(d, n)}
                       for n in runs.run_meshes(d)]}


def edits_state(run_dir: Path, name: str) -> dict:
    """The mesh's depth edits for the window: how many, and whether they changed after its last write (the files
    then lack them until it is written again)."""
    ef = meshview.edits_file(run_dir / name)
    if not ef.exists():
        return {"edits": 0, "edits_newer": False}
    q = out_of(run_dir, name) / "quality.json"
    n = len((runs.read_json(ef) or {}).get("edits", []))
    return {"edits": n, "edits_newer": not q.exists() or ef.stat().st_mtime > q.stat().st_mtime}


class Busy(RuntimeError):
    pass


class Writer:
    """One write at a time, in a thread: start() returns at once, job() tells how it goes (the stage write_mesh is
    in while it runs). fn: write_mesh's stand-in (tests); None = meshom.run.write_mesh."""

    def __init__(self):
        self.lock = threading.Lock()
        self.fn = None
        self.state = {"state": "idle"}

    def job(self) -> dict:
        with self.lock:
            j = dict(self.state)
        if j["state"] == "running":
            j["elapsed_s"] = round(time.monotonic() - j.pop("t0"), 1)
        else:
            j.pop("t0", None)
        return j

    def start(self, mf: Path, fmt: str | None, fesom_ini: bool = False) -> dict:
        mf = Path(mf)
        d, cfg = run_of(mf)
        fmt = fmt or cfg.mesh.format
        ini = bool(fesom_ini and fmt == "fesom3" and cfg.mesh.fesom3_ini)
        with self.lock:
            if self.state["state"] == "running":
                raise Busy(f"a write is already running ({Path(self.state['mesh']).name}): one at a time")
            self.state = {"state": "running", "mesh": str(mf), "format": fmt, "fesom_ini": ini,
                          "out": str(out_of(d, mf.name)), "t0": time.monotonic()}
        threading.Thread(target=self._work, args=(mf, d, cfg.gates_all(), fmt, ini), daemon=True).start()
        return self.job()

    def _stage(self, name: str) -> None:
        with self.lock:
            self.state["stage"] = name

    def _work(self, mf: Path, d: Path, gates: dict, fmt: str, ini: bool) -> None:
        try:
            from meshom.run import write_mesh
            rec = (self.fn or write_mesh)(d, mf.name, fmt, progress=self._stage, fesom_ini=ini)
            done = {"state": "done", "summary": summary(rec, gates)}
        except Exception as e:                       # the job's error, shown in the window
            done = {"state": "error", "error": f"{type(e).__name__}: {e}"}
        with self.lock:
            self.state = {k: v for k, v in self.state.items() if k not in ("t0", "stage")} | done
