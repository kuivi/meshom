"""Batch step for the UI: build a ~10^6-element mesh with `meshom compete` and write the quality overlay.

Runs on a compute node (scripts/on_node.sh); the UI only reads the result:
    scripts/on_node.sh -c 8 -t 01:00:00 -- python -m ui.common.prepare_mesh --scale 0.45
    ... -m ui.common.prepare_mesh --overlay-only [--case continent --scale 0.45]      # mesh.npz exists already
Output: <out>/mesh.npz (points, tris, quads, node_flag) and <out>/overlay.npz:
  per element (tris first): max_angle_deg, size_ratio (largest mean-edge ratio across its edges), touches_boundary,
                            kind (0 tri / 1 quad), lon / lat of the centroid;
  per node:                 node_class (0 regular or boundary, 1 interior valence regular-1 = n3, 2 regular+1 = n5,
                            3 other), with `regular` = 4 (6 on a mesh with more triangles than quads, as figures.py);
  per edge (if h is known): edge_ln_sh = ln(s / h), s the great-circle length, h at the midpoint.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from meshom.compete import WORK_DIR
from meshom.geo import R_EARTH_M, arc_length, xyz_to_lonlat


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="m13_squadgen")
    p.add_argument("--case", default="continent")
    p.add_argument("--scale", type=float, default=0.45)
    p.add_argument("--out", default=".tmp/ui/mesh")
    p.add_argument("--overlay-only", action="store_true", help="only (re)write overlay.npz of <out>/mesh.npz")
    a = p.parse_args(argv)

    t0 = time.perf_counter()
    out = Path(a.out)
    if not a.overlay_only:
        subprocess.run([sys.executable, "-c", "from meshom.cli import main; main()", "compete", "--methods", a.method,
                        "--cases", a.case, "--scale", f"{a.scale:g}", "--jsonl", ".tmp/ui/compete.jsonl"], check=True)
        run = WORK_DIR / f"{a.method}_{a.case}_s{a.scale:g}"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(run / "raw.npz", out / "mesh.npz")
    t1 = time.perf_counter()

    from meshom.cases import CASES
    from meshom.compete import _load
    from meshom.size import scaled
    mesh = _load(out / "mesh.npz")
    write_overlay(mesh, out, scaled(CASES[a.case][1], a.scale))
    print(f"{out}: {mesh.n_nodes} nodes, {mesh.n_tris} tris, {mesh.n_quads} quads; compete {t1 - t0:.1f} s, "
          f"overlay {time.perf_counter() - t1:.1f} s")


def overlay_arrays(mesh, h=None, q=None) -> dict:
    """The per-element, per-node and per-edge arrays of overlay.npz (module docstring); q: compute_quality(mesh) when
    the caller has it already."""
    from meshom.mesh import build_topology
    from meshom.quality.metrics import compute_quality
    q = compute_quality(mesh) if q is None else q
    lon, lat = xyz_to_lonlat(q.centroid)
    kind = np.r_[np.zeros(mesh.n_tris, np.int8), np.ones(mesh.n_quads, np.int8)]
    topo = build_topology(mesh)
    deg = np.bincount(topo.edges.ravel(), minlength=mesh.n_nodes)
    reg = 4 if mesh.n_quads >= mesh.n_tris else 6
    irr = ~topo.boundary_node & (deg != reg)
    cls = np.zeros(mesh.n_nodes, np.int8)
    cls[irr] = 3
    cls[irr & (deg == reg - 1)] = 1
    cls[irr & (deg == reg + 1)] = 2
    out = dict(max_angle_deg=np.r_[q.tri_max_angle_deg, q.quad_max_angle_deg].astype(np.float32),
               size_ratio=q.elem_size_ratio_max.astype(np.float32), touches_boundary=q.touches_boundary, kind=kind,
               lon=lon.astype(np.float32), lat=lat.astype(np.float32), node_class=cls, regular=np.int64(reg))
    if h is not None:
        a, b = mesh.points[topo.edges[:, 0]], mesh.points[topo.edges[:, 1]]
        mid = a + b
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        out["edge_ln_sh"] = np.log(R_EARTH_M * arc_length(a, b) / h(mid)).astype(np.float32)
    return out


def write_overlay(mesh, out: Path, h=None, name: str = "overlay.npz") -> Path:
    """mesh.npz is written by the batch run; overlay.npz holds the quality the UI colours by."""
    path = Path(out) / name
    np.savez(path, **overlay_arrays(mesh, h))
    return path


if __name__ == "__main__":
    main()
