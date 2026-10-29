"""Numbers of an m16_orca run (ORCA belt + cap) on its final mesh: the belt unchanged (node displacement, quads kept),
the quality summary and gates of the whole mesh, of the cap alone and of the belt alone, and the joint (the elements
on both sides of the cut row, triangles there, the element size ratio across it).

    python scripts/orca_report.py RUN_DIR [MESH]      # MESH: raw.npz (default); writes RUN_DIR/orca_report.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meshom.compete import _load, _score  # noqa: E402
from meshom.config import load  # noqa: E402
from meshom.geo import R_EARTH_M  # noqa: E402
from meshom.mesh import Mesh, build_topology  # noqa: E402
from meshom.methods.m16_orca import _ruled, belt_unchanged  # noqa: E402
from meshom.run import _boundary_domain, write_inputs  # noqa: E402


def sub(mesh: Mesh, el: np.ndarray) -> Mesh:
    """The elements el (global ids, tris first) as their own mesh."""
    t = el[el < mesh.n_tris]
    q = el[el >= mesh.n_tris] - mesh.n_tris
    used = np.unique(np.r_[mesh.tris[t].ravel(), mesh.quads[q].ravel()])
    new = np.full(mesh.n_nodes, -1, np.int64)
    new[used] = np.arange(len(used))
    return Mesh(mesh.points[used], new[mesh.tris[t]], new[mesh.quads[q]], np.zeros(len(used), np.int8))


def mean_edge(mesh: Mesh) -> np.ndarray:
    out = []
    for cells in (mesh.tris, mesh.quads):
        p = mesh.points[cells]
        d = np.linalg.norm(p - np.roll(p, -1, axis=1), axis=2)
        out.append(d.mean(axis=1))
    return np.r_[out[0], out[1]] * R_EARTH_M


def summary(rec: dict) -> dict:
    q = rec["quality"]
    keep = ("n_tris", "n_quads", "tri_share", "tri_min_angle_deg", "tri_max_angle_deg", "quad_max_angle_deg",
            "quad_sj", "quad_edge_ratio", "elem_size_ratio", "size_fidelity", "tri_clusters")
    return {"n_nodes": rec["n_nodes"], **{k: q[k] for k in keep if k in q}, "gates_passed": rec["gates_passed"],
            "gates_total": rec["gates_total"], "gates_failed": rec["gates_failed"],
            "gates_failed_families": rec["gates_failed_families"], "euler_ok": rec["irregular"]["euler_ok"]}


def main(run_dir: Path, mesh_name: str = "raw.npz") -> dict:
    cfg = load(run_dir / "config.resolved.toml")
    _, h, _, _ = write_inputs(run_dir, cfg)
    gates = cfg.gates_all()
    mesh = _load(run_dir / mesh_name)
    b = np.load(run_dir / "orca_belt.npz")
    out = {"mesh": mesh_name, "belt_unchanged": belt_unchanged(mesh, b["points"], b["quads"], b["tris"])}
    is_belt = _ruled(mesh, b["points"], b["quads"], b["tris"])
    rho = {"plus": 0.0, "minus": 0.0}
    for name, el in (("whole", np.arange(mesh.n_elems)), ("cap", np.flatnonzero(~is_belt)),
                     ("belt", np.flatnonzero(is_belt))):
        if not len(el):
            continue
        m = mesh if name == "whole" else sub(mesh, el)
        out[name] = summary(_score(m, _boundary_domain(m), h, rho, gates))
    top = build_topology(mesh)
    ee = top.edge_elems
    inner = ee[:, 1] >= 0
    cross = inner & (is_belt[ee[:, 0]] != is_belt[np.maximum(ee[:, 1], 0)])
    me = mean_edge(mesh)
    a, c = ee[cross, 0], ee[cross, 1]
    ratio = np.maximum(me[a], me[c]) / np.minimum(me[a], me[c])
    capside = np.where(is_belt[a], c, a)
    out["joint"] = {"edges": int(cross.sum()), "cap_tris_at_joint": int((capside < mesh.n_tris).sum()),
                    "cap_quads_at_joint": int((capside >= mesh.n_tris).sum()),
                    "elem_size_ratio_median": float(np.median(ratio)) if len(ratio) else None,
                    "elem_size_ratio_p99": float(np.percentile(ratio, 99)) if len(ratio) else None,
                    "elem_size_ratio_max": float(ratio.max()) if len(ratio) else None}
    (run_dir / "orca_report.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    r = main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "raw.npz")
    print(json.dumps({k: r[k] for k in ("belt_unchanged", "joint")}, indent=1))
    for k in ("whole", "cap", "belt"):
        if k not in r:
            continue
        x = r[k]
        print(k, x["n_nodes"], "tris", x["n_tris"], "quads", x["n_quads"], "gates", x["gates_passed"], "/",
              x["gates_total"], x["gates_failed"])
