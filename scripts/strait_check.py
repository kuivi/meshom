"""Strait table of FESOM2 mesh directories: per strait of meshom's built-in list (meshom.straits, 89 straits from
fesom2_mesh_gen), the cells across the narrowest inner section, whether both ends are one water component, and the
smallest element depth on the inner path (meshom.straits.strait_report, read back from nod2d.out / elem2d.out /
aux3d.out).

    python scripts/strait_check.py [NAME=DIR ...] [--out bench/straits.jsonl]

Without NAME=DIR: the reference meshes (FESOM2 CORE2 from the pool with its aux3d.out; three fesom2_mesh_gen meshes
as mesh.npz files in triangle_meshes/ next to the main checkout, which hold no depth) and meshom's core2like before
(the git worktree `fesom2` of the main checkout) and after the strait rules (this checkout). A DIR holds
nod2d.out + elem2d.out (+ aux3d.out for depths) or mesh.npz (points, tris). One JSON line per mesh and strait goes to
--out; a Markdown table of the main straits and the counts over all 89 is printed. Run on a compute node.
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meshom.io.fesom import read_aux3d, read_fesom  # noqa: E402
from meshom.mesh import Mesh, orient_ccw  # noqa: E402
from meshom.run import MAIN_CHECKOUT  # noqa: E402
from meshom.straits import Strait, builtin, strait_report  # noqa: E402
from meshom.geo import lonlat_to_xyz  # noqa: E402

TRI = MAIN_CHECKOUT.parent / "triangle_meshes"
MESHES = {"fesom2_core2": Path("/pool/data/AWICM/FESOM2/MESHES_FESOM2.1/core2"),
          "fmg_flow": TRI / "fmg_flow",
          "fmg_BG3_base": TRI / "fmg_BG3_base",
          "fmg_test_BG3x4": TRI / "fmg_test_BG3x4",
          "meshom_core2like": MAIN_CHECKOUT / ".worktrees/fesom2/.tmp/runs/fesom2_core2like/fesom",
          "meshom_core2like_straits": ROOT / ".tmp/runs/fesom2_core2like_straits/fesom"}
MAIN = ["Gibraltar", "Bosphorus", "Dardanelles", "Kerch", "Messina", "Bonifacio", "Dover", "Oresund", "Great_Belt",
        "Little_Belt", "Bab_el_Mandeb", "Hormuz"]


def straits() -> list[Strait]:
    """The built-in list as given (cells 2; h_m and width only set the transect step)."""
    out = []
    for d in builtin("fesom2_mesh_gen"):
        p = np.asarray(d["path"])
        out.append(Strait(d["name"], lonlat_to_xyz(p[:, 0], p[:, 1]), d["width_m"], d["h_m"], 2, d["depth_m"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("meshes", nargs="*", help="NAME=DIR (a directory with nod2d.out, elem2d.out, aux3d.out)")
    ap.add_argument("--out", default=str(ROOT / "bench/straits.jsonl"))
    a = ap.parse_args()
    meshes = dict(m.split("=", 1) for m in a.meshes) if a.meshes else MESHES
    st = straits()
    table = {}
    date = datetime.datetime.now().isoformat(timespec="seconds")
    with open(a.out, "a") as f:
        for name, d in meshes.items():
            t0 = time.perf_counter()
            d = Path(d)
            if (d / "nod2d.out").exists():
                mesh = read_fesom(d)
                depth = read_aux3d(d / "aux3d.out")[1] if (d / "aux3d.out").exists() else None
            else:
                z = np.load(d / "mesh.npz")
                mesh = orient_ccw(Mesh(z["points"], z["tris"], np.zeros((0, 4), dtype=np.int64),
                                       np.zeros(len(z["points"]), dtype=np.int8)))
                depth = None
            rep = strait_report(mesh, st, depth)
            for r in rep:
                f.write(json.dumps({"date": date, "mesh": name, "source": str(d), "n_nodes": mesh.n_nodes, **r}) + "\n")
            table[name] = {r["name"]: r for r in rep}
            print(f"{name}: {mesh.n_nodes} nodes, {len(rep)} straits, {time.perf_counter() - t0:.1f} s", flush=True)
    names = list(table)
    print("\n| strait | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for s in MAIN:
        if s not in table[names[0]]:
            continue
        cells = []
        for n in names:
            r = table[n][s]
            dep = "–" if r.get("depth_min_m") is None else f"{r['depth_min_m']:.0f} m"
            cells.append(f"{r['cells_min']:.1f} / {'yes' if r['connected'] else 'no'} / {dep}")
        print(f"| {s} | " + " | ".join(cells) + " |")
    for label, key in [(f"connected (of {len(st)})", lambda r: r["connected"]),
                       ("every inner section wet", lambda r: r["closed_sections"] == 0),
                       (">= 2 cells across, connected", lambda r: r["ok"])]:
        print(f"| {label} | " + " | ".join(str(sum(key(r) for r in table[n].values())) for n in names) + " |")


if __name__ == "__main__":
    main()
