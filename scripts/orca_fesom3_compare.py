"""ORCA meshes side by side: meshom's builder (NEMO's mask, GEBCO depths) against FESOM3's rules (meshom.orca.fesom3)
and FESOM3's own ORCA025 mesh: the cells changed and the straits one by one (meshom.orca.straits: open, cells across,
sill, the narrowest section's node depths), a figure per strait group (decision B93).

    python scripts/orca_fesom3_compare.py OUT_DIR GRID_NPZ LABEL=FESOM_DIR [LABEL=FESOM_DIR ...]

GRID_NPZ: the preset's grid (T points as lon_t, lat_t in NEMO's layout: meshom.orca.presets.build) to give every
element its T cell (the nearest T point to its centre). A FESOM_DIR holds nod2d.out, elem2d.out and depth.out (FESOM3's
NEMO depth file: node depths are layer bottoms) or aux3d.out (meshom's: GEBCO node depths). Writes OUT_DIR/compare.json,
OUT_DIR/straits.md and OUT_DIR/<group>.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M, lonlat_to_xyz
from meshom.orca.straits import GROUPS, STRAITS, cell_links, centres, measure


def read_dir(d: Path) -> dict:
    d = Path(d)
    with open(d / "nod2d.out") as f:
        n = int(f.readline())
        nod = np.loadtxt(f, max_rows=n, ndmin=2)
    with open(d / "elem2d.out") as f:
        m = int(f.readline())
        el = np.loadtxt(f, max_rows=m, dtype=np.int64, ndmin=2) - 1
    if el.shape[1] == 3:
        el = np.column_stack([el, el[:, 0]])
    if (d / "depth.out").exists():
        v = np.asarray((d / "depth.out").read_text().split(), float)
        nl = int(v[0])
        dep, src = v[1 + nl:1 + nl + n], "levels"
    else:
        v = np.abs(np.asarray((d / "aux3d.out").read_text().split(), float))
        nl = int(v[0])
        dep, src = v[1 + nl:1 + nl + n], "aux3d"
    return {"lon": nod[:, 1], "lat": nod[:, 2], "elem": el, "depth": dep, "depth_source": src}


def cell_ij(m: dict, tree: cKDTree, ni: int) -> np.ndarray:
    clon, clat = centres(m["lon"], m["lat"], m["elem"])
    _, k = tree.query(lonlat_to_xyz(clon, clat))
    return np.stack([k // ni, k % ni], axis=1), clon, clat


def area_km2(m: dict) -> np.ndarray:
    P = lonlat_to_xyz(m["lon"], m["lat"])[m["elem"]]
    a = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    b = 0.5 * np.linalg.norm(np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]), axis=1)
    return (a + b) * (R_EARTH_M / 1e3) ** 2


def main(out: Path, grid: Path, dirs: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm

    out.mkdir(parents=True, exist_ok=True)
    g = np.load(grid)
    lt, pt = g["lon_t"][:, :-2], g["lat_t"][:, :-2]
    ni = lt.shape[1]
    tree = cKDTree(lonlat_to_xyz(lt.ravel(), pt.ravel()))
    meshes, rec = {}, {"grid": str(grid), "meshes": {}}
    for lab, d in dirs.items():
        m = read_dir(Path(d))
        m["ij"], m["clon"], m["clat"] = cell_ij(m, tree, ni)
        m["links"] = cell_links(m["elem"], len(m["lon"]))
        m["cell_depth"] = m["depth"][m["elem"]].min(axis=1)
        m["area"] = area_km2(m)
        meshes[lab] = m
        rec["meshes"][lab] = {"dir": str(d), "nodes": int(len(m["lon"])), "cells": int(len(m["elem"])),
                              "depth_source": m["depth_source"], "water_km2": round(float(m["area"].sum()), 1)}
    labs = list(meshes)
    keys = {lab: set(map(tuple, meshes[lab]["ij"].tolist())) for lab in labs}
    rec["cells"] = {}
    for i, a in enumerate(labs):
        for b in labs[i + 1:]:
            ka, kb = keys[a], keys[b]
            ma, mb = meshes[a], meshes[b]
            only_a = np.array([tuple(x) not in kb for x in ma["ij"].tolist()])
            only_b = np.array([tuple(x) not in ka for x in mb["ij"].tolist()])
            rec["cells"][f"{a} vs {b}"] = {
                "both": len(ka & kb), f"only {a}": int(only_a.sum()), f"only {b}": int(only_b.sum()),
                f"only {a} km2": round(float(ma["area"][only_a].sum()), 1),
                f"only {b} km2": round(float(mb["area"][only_b].sum()), 1)}
    rec["straits"] = {}
    for name in STRAITS:
        rec["straits"][name] = {}
        for lab in labs:
            m = meshes[lab]

            def nd(cells, m=m):
                return [sorted({round(float(x), 1) for x in m["depth"][m["elem"][c]]}) for c in cells]
            r = measure(name, m["clon"], m["clat"], m["links"], m["cell_depth"], nd)
            rec["straits"][name][lab] = r
    (out / "compare.json").write_text(json.dumps(rec, indent=1))
    lines = ["| strait | " + " | ".join(labs) + " |", "|---" * (len(labs) + 1) + "|"]
    for name, by in rec["straits"].items():
        cells = []
        for lab in labs:
            r = by[lab]
            if not r.get("open"):
                cells.append("closed")
            else:
                cut = r.get("cut_node_depth_m", [])
                deep = max((max(x) for x in cut if x), default=None)
                cells.append(f"open, {r['cells_across']} across, sill {r['sill_m']:.0f} m, section nodes to "
                             f"{deep:.0f} m" if deep is not None else f"open, {r['cells_across']} across")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (out / "straits.md").write_text("\n".join(lines) + "\n")
    for grp, names in GROUPS.items():
        fig, axes = plt.subplots(len(names), len(labs), figsize=(4.2 * len(labs), 3.6 * len(names)), squeeze=False)
        for r_, name in enumerate(names):
            A, B, box = STRAITS[name]
            for c_, lab in enumerate(labs):
                ax = axes[r_, c_]
                m = meshes[lab]
                lo0, lo1, la0, la1 = box
                sel = np.flatnonzero((((m["clon"] - lo0) % 360) <= ((lo1 - lo0) % 360)) & (m["clat"] >= la0)
                                     & (m["clat"] <= la1))
                poly = np.stack([m["lon"][m["elem"][sel]], m["lat"][m["elem"][sel]]], axis=-1)
                poly[..., 0] = lo0 + ((poly[..., 0] - lo0 + 180.0) % 360.0) - 180.0
                others = [o for o in labs if o != lab]
                lone = np.array([any(tuple(x) not in keys[o] for o in others) for x in m["ij"][sel].tolist()],
                                bool)
                pc = PolyCollection(poly, array=np.maximum(m["cell_depth"][sel], 1.0), cmap="viridis_r",
                                    norm=LogNorm(5, 6000), edgecolors=np.where(lone[:, None], [[1, 0, 0, 1]],
                                                                             [[1, 1, 1, 0.6]]),
                                    linewidths=np.where(lone, 1.6, 0.3))
                ax.add_collection(pc)
                s = rec["straits"][name][lab]
                for p in s.get("cut_at", []):
                    ax.plot(*p, "k.", ms=3)
                ax.plot(*A, "r^", ms=5)
                ax.plot(*B, "rv", ms=5)
                ax.set_xlim(lo0, lo0 + (lo1 - lo0) % 360)
                ax.set_ylim(la0, la1)
                ax.set_aspect(1.0 / max(np.cos(np.radians(0.5 * (la0 + la1))), 0.05))
                txt = (f"open, {s['cells_across']} across, sill {s['sill_m']:.0f} m" if s.get("open")
                       else "closed")
                ax.set_title(f"{name} | {lab}\n{txt}", fontsize=8)
                ax.tick_params(labelsize=6)
        fig.colorbar(pc, ax=axes, shrink=0.6, label="cell depth (shallowest node), m")
        fig.suptitle(f"{grp}: cells by depth; red outline = cell not in the other mesh(es);\n"
                     "triangles = the ends; dots = the narrowest section", fontsize=9)
        fig.savefig(out / f"{grp.replace(' ', '_')}.png", dpi=110)
        plt.close(fig)
    print(json.dumps({"cells": rec["cells"], "meshes": rec["meshes"]}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), dict(a.split("=", 1) for a in sys.argv[3:]))
