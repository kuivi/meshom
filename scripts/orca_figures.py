"""Figures of an m16_orca run (an ORCA grid whole, or an ORCA latitude range + caps): the whole mesh, the joint or the
fold at chosen places, the Arctic, the equatorial ORCA band. ORCA cells blue, cap quads orange, triangles red; the cut
row (the joint) black. Views are
orthographic (globe) or gnomonic at their centre (zooms), never raw lon-lat.

    python scripts/orca_figures.py RUN_DIR OUT_DIR [MESH]      # MESH: raw.npz (default)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meshom.compete import _load  # noqa: E402
from meshom.geo import R_EARTH_M, lonlat_to_xyz  # noqa: E402

BELT, CAP, TRI = "#9ecae1", "#fdbe85", "#e6550d"


def frame(lon, lat):
    c = lonlat_to_xyz(np.array([lon]), np.array([lat]))[0]
    lo, la = np.radians(lon), np.radians(lat)
    e = np.array([-np.sin(lo), np.cos(lo), 0.0])
    n = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    return c, e, n


def classify(mesh, belt_pts):
    """Element classes: 0 belt quad, 1 cap quad, 2 triangle (tris first, as Mesh's global ids)."""
    tree = cKDTree(belt_pts)
    d, _ = tree.query(mesh.points)
    onb = d < 1e-9
    cls = np.r_[np.full(mesh.n_tris, 2), np.where(onb[mesh.quads].all(axis=1), 0, 1)]
    return cls, onb


def polys(mesh):
    T = [mesh.points[t] for t in mesh.tris]
    Q = [mesh.points[q] for q in mesh.quads]
    return T + Q


def draw(ax, mesh, cls, centre, radius_deg, kind, lw=0.15, cut=None, title=""):
    c, e, n = frame(*centre)
    P = mesh.points
    cent = np.r_[P[mesh.tris].mean(axis=1), P[mesh.quads].mean(axis=1)] if mesh.n_tris else P[mesh.quads].mean(axis=1)
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    cosr = np.cos(np.radians(min(radius_deg * (1.15 if kind == "ortho" else 1.6), 89.0)))
    sel = np.flatnonzero(cent @ c > max(cosr, 0.02 if kind == "ortho" else 0.2))
    elems = [mesh.tris[i] if i < mesh.n_tris else mesh.quads[i - mesh.n_tris] for i in sel]

    def proj(p):
        if kind == "ortho":
            return np.c_[p @ e, p @ n] * R_EARTH_M / 1e3
        w = p @ c
        return np.c_[(p @ e) / w, (p @ n) / w] * R_EARTH_M / 1e3

    vs = [proj(P[el]) for el in elems]
    col = np.array([BELT, CAP, TRI])[cls[sel]]
    ax.add_collection(PolyCollection(vs, facecolors=col, edgecolors="0.25", linewidths=lw))
    if cut is not None:
        nxt = np.roll(cut, -1)
        ok = (cut >= 0) & (nxt >= 0)
        a, b = P[cut[ok]], P[nxt[ok]]
        vis = (a @ c > 0.05) & (b @ c > 0.05)
        xa, xb = proj(a[vis]), proj(b[vis])
        ax.add_collection(LineCollection(np.stack([xa, xb], axis=1), colors="k", linewidths=1.2))
    rr = R_EARTH_M / 1e3 * (np.sin(np.radians(radius_deg)) if kind == "ortho" else np.tan(np.radians(radius_deg)))
    ax.set_xlim(-rr, rr)
    ax.set_ylim(-rr, rr)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def legend(fig, counts, orca="ORCA"):
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=BELT, label=f"{orca} cells ({counts[0]})"),
                        Patch(color=CAP, label=f"cap quads ({counts[1]})"),
                        Patch(color=TRI, label=f"triangles ({counts[2]})"),
                        plt.Line2D([], [], color="k", lw=1.5, label="the cut row (joint)")],
               loc="lower center", ncol=4, fontsize=9, frameon=False)


def main(run_dir: Path, out: Path, mesh_name: str = "raw.npz") -> None:
    from meshom.methods.m16_orca import _ruled

    out.mkdir(parents=True, exist_ok=True)
    mesh = _load(run_dir / mesh_name)
    b = np.load(run_dir / "orca_belt.npz")
    ruled = _ruled(mesh, b["points"], b["quads"], b["tris"])
    cls = np.where(ruled, 0, np.r_[np.full(mesh.n_tris, 2), np.ones(mesh.n_quads, int)])
    counts = np.bincount(cls, minlength=3)
    info = json.loads((run_dir / "orca.json").read_text())
    tree = cKDTree(mesh.points)
    cut = None
    if info.get("north_cap"):
        dd, cut = tree.query(lonlat_to_xyz(b["lon"][-1], b["lat"][-1]))
        cut = np.where(dd < 1e-9, cut, -1)                # -1: the row node is not in the mesh (land below and above)
    name = run_dir.name
    orca = info["orca"]
    what = (f"{orca} T rows {info['t_rows'][0]}-{info['t_rows'][1]} (last F row mean {info['lat_last_f_row']:.2f})"
            + (", caps: " + ", ".join(info["joints"]) if info["joints"] else ", the whole grid"))
    fig, ax = plt.subplots(2, 3, figsize=(18, 12.5))
    globes = [((-30, 25), "Atlantic"), ((-160, 20), "Pacific"), ((75, 25), "Indian / Eurasia"),
              ((0, 90), "North Pole"), ((0, -90), "South Pole"), ((120, -5), "Maritime continent")]
    for a, (cen, t) in zip(ax.ravel(), globes):
        draw(a, mesh, cls, cen, 90, "ortho", lw=0.05, cut=cut, title=t)
    fig.suptitle(f"{name}: {mesh.n_nodes} nodes, {mesh.n_quads} quads, {mesh.n_tris} triangles; {what}")
    legend(fig, counts, orca)
    fig.savefig(out / "global.png", dpi=110)
    plt.close(fig)
    views = [("equator_band.png", (-140, 0), 10, f"the equatorial {orca} band (140 W)")]
    if cut is not None:
        views += [("joint_atlantic.png", (-35, 40), 9, "joint, North Atlantic (35 W 40 N)"),
                  ("joint_pacific.png", (-165, 40), 9, "joint, North Pacific (165 W 40 N)"),
                  ("joint_mediterranean.png", (18, 39), 7, "joint, the Mediterranean (18 E 39 N)"),
                  ("joint_black_sea.png", (32, 42), 7, "joint, the Bosphorus and the Black Sea (32 E 42 N)"),
                  ("joint_caspian.png", (51, 40), 7, "joint, the Caspian (51 E 40 N)"),
                  ("arctic_fram.png", (0, 79), 6, "Arctic: Fram Strait (0 E 79 N)")]
    else:
        views += [("fold_canada.png", (-100, 72), 14, "the fold: the Canadian pole and the Archipelago"),
                  ("fold_arctic.png", (-20, 85), 14, "the fold across the Arctic (the pole line)"),
                  ("fold_siberia.png", (75, 62), 14, "the fold: the Siberian pole"),
                  ("red_sea.png", (40, 20), 9, "the Red Sea and Bab-el-Mandeb (NEMO's mask)"),
                  ("indonesia.png", (122, -5), 12, "the Indonesian passages (NEMO's mask)")]
    for fn, cen, rad, t in views:
        fig, a = plt.subplots(figsize=(10, 10.5))
        draw(a, mesh, cls, cen, rad, "gnomonic", lw=0.25, cut=cut, title=t)
        legend(fig, counts, orca)
        fig.savefig(out / fn, dpi=110)
        plt.close(fig)
    fig, a = plt.subplots(figsize=(12, 12.5))
    draw(a, mesh, cls, (0, 90), 52, "ortho", lw=0.03, cut=cut,
         title=f"{name}: north of 38 N (orthographic, North Pole)")
    legend(fig, counts, orca)
    fig.savefig(out / "arctic.png", dpi=130)
    plt.close(fig)
    print("figures in", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "raw.npz")
