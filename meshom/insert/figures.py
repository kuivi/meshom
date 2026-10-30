"""Maps of an insert (before / after / neck zoom) and its rule profile along the axis."""
from pathlib import Path

import numpy as np

from meshom.quality.kernel import element_geometry


def _panel(ax, chart, mesh, title, outline, chains, coast, box, real=None, bays=()):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    xy = chart.fwd(mesh.points) / 1e3
    g = element_geometry(mesh)
    c = chart.fwd(g.centroid) / 1e3
    T = mesh.n_tris
    sel = (c[:, 0] >= box[0]) & (c[:, 0] <= box[1]) & (c[:, 1] >= box[2]) & (c[:, 1] <= box[3])
    if mesh.n_quads:
        qa = np.degrees(g.quad_angles).max(1)
        qs = sel[T:]
        pc = PolyCollection(xy[mesh.quads[qs]], array=qa[qs], cmap="turbo", clim=(90, 135), edgecolors="0.25",
                            linewidths=0.25)
        ax.add_collection(pc)
    if T and sel[:T].any():
        ax.add_collection(PolyCollection(xy[mesh.tris[sel[:T]]], facecolors="magenta", edgecolors="0.25",
                                         linewidths=0.25))
    if coast is not None and len(coast):
        ax.add_collection(LineCollection(coast / 1e3, colors="saddlebrown", linewidths=0.8))
    if real is not None and len(real):
        ax.add_collection(LineCollection(real / 1e3, colors="green", linewidths=0.6))
    if outline is not None:
        o = np.vstack([outline, outline[:1]]) / 1e3
        ax.plot(o[:, 0], o[:, 1], "k--", lw=0.6)
    for ch in chains or []:
        ax.plot(ch[:, 0] / 1e3, ch[:, 1] / 1e3, "r-", lw=1.2)
    for arc, meshed in bays:                         # B190: bays kept (blue coast) and bridged (orange mouth)
        if meshed:
            ax.plot(arc[:, 0] / 1e3, arc[:, 1] / 1e3, "-", color="blue", lw=1.0)
        else:
            ax.plot(arc[[0, -1], 0] / 1e3, arc[[0, -1], 1] / 1e3, "-", color="orange", lw=1.4)
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("chart x (km)", fontsize=8)
    return pc if mesh.n_quads else None


def maps(path: Path, chart, panels: list, outline, chains, coast_xy, neck_xy, h_fine, real_xy=None, zooms=(),
         bays=()):
    """panels: [(title, mesh)]; one row: each panel over the outline's box, then a zoom of the last at the neck;
    zooms: more close-ups of the last panel [(title, xy chart, half size m)] in a second row; bays: [(arc chart,
    meshed)] drawn on every panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = outline.min(0) / 1e3, outline.max(0) / 1e3
    pad = 0.15 * (hi - lo).max()
    box = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad)
    r = 25 * h_fine / 1e3
    nb = (neck_xy[0] / 1e3 - r, neck_xy[0] / 1e3 + r, neck_xy[1] / 1e3 - r, neck_xy[1] / 1e3 + r)
    k = len(panels) + 1
    rows = 1 + (len(zooms) + k - 1) // k if zooms else 1
    fig, axs = plt.subplots(rows, k, figsize=(5.2 * k, 5.6 * rows), squeeze=False)
    for j, (title, xy, r_m) in enumerate(zooms):
        ax = axs[1 + j // k, j % k]
        zb = (xy[0] / 1e3 - r_m / 1e3, xy[0] / 1e3 + r_m / 1e3, xy[1] / 1e3 - r_m / 1e3, xy[1] / 1e3 + r_m / 1e3)
        _panel(ax, chart, panels[-1][1], title, outline, chains, coast_xy, zb, real_xy, bays)
    for j in range(len(zooms), (rows - 1) * k):
        axs[1 + j // k, j % k].axis("off")
    axs = axs[0]
    pc = None
    for ax, (title, mesh) in zip(axs[:-1], panels):
        pc = _panel(ax, chart, mesh, title, outline, chains, coast_xy, box, real_xy, bays) or pc
    _panel(axs[-1], chart, panels[-1][1], panels[-1][0] + " (neck)", outline, chains, coast_xy, nb, real_xy, bays)
    if pc is not None:
        cb = fig.colorbar(pc, ax=list(axs), shrink=0.7)
        cb.set_label("quad max angle (deg); triangles magenta; host coast brown, real coast green; bays: kept blue, "
                     "bridged orange", fontsize=7)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def collar(path: Path, chart, before, after, chains, coast_xy, title: str):
    """B192: the cleaned case (the collar, the join and the block's edge rows) before and after the cleaning."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy = chart.fwd(before.points) / 1e3
    lo, hi = xy.min(0), xy.max(0)
    pad = 0.05 * (hi - lo).max()
    box = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad)
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    pc = _panel(axs[0], chart, before, f"{title}: before the cleaning", None, chains, coast_xy, box)
    pc = _panel(axs[1], chart, after, f"{title}: after", None, chains, coast_xy, box) or pc
    if pc is not None:
        fig.colorbar(pc, ax=list(axs), shrink=0.7, label="quad max angle (deg); triangles magenta; joins red")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def profile(path: Path, prof, joins: list, cands: list):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = (prof.s - prof.s0) / 1e3
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    W = np.where(np.isfinite(prof.W), prof.W, np.nan)
    a1.plot(x, W / 1e3 / prof.n_cells, label="W / N (km)")
    a1.plot(x, prof.h_t / 1e3, label="size target h_t (km)")
    for c in cands:
        a1.plot((c.s_m - prof.s0) / 1e3, c.h_join_m / 1e3, "o", ms=3, color="g" if c.ok else "0.6")
    for j in joins:
        if j is not None:
            a1.plot((j.s_m - prof.s0) / 1e3, j.h_join_m / 1e3, "r*", ms=10)
    a1.set_ylabel("km")
    a1.set_yscale("log")
    a1.legend(fontsize=8)
    a1.set_title("width profile, size target and candidate joins (host cell size; green: all rules pass)",
                 fontsize=9)
    a2.plot(x, np.degrees(prof.alpha), label="opening angle alpha (deg)")
    a2.set_ylabel("deg")
    a2.set_xlabel("distance along the axis from the neck (km)")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
