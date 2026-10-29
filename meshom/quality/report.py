"""Quality report: report.md, summary.json and four figures (matplotlib, Agg)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from meshom.geo import xyz_to_lonlat  # noqa: E402
from meshom.mesh import Mesh  # noqa: E402
from meshom.quality.metrics import Quality  # noqa: E402

_COUNTS = ("n_nodes", "n_elems", "n_tris", "n_quads", "n_edges", "n_edges_interior", "n_tri_quad_edges",
           "elem_neighbours_min", "n_pinch_nodes", "n_elems_ge2_boundary_edges")
_TRI = ("tri_min_angle_deg", "tri_max_angle_deg")
_QUAD = ("quad_dev_deg", "quad_max_angle_deg", "quad_sj", "quad_edge_ratio")
_SIZE = ("elem_size_ratio", "elem_sqrt_area_ratio", "node_size_ratio")


def _f(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _stat_table(s: dict, names: tuple) -> list[str]:
    out = ["| metric | stat | all | coast | interior |", "|---|---|---|---|---|"]
    for n in names:
        for k, v in s[n].items():
            out.append(f"| {n} | {k} | {_f(v)} | {_f(s[n + '_coast'][k])} | {_f(s[n + '_interior'][k])} |")
    return out


def _hist(path: Path, series: list[tuple[str, np.ndarray]], xlabel: str, bins) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, x in series:
        ax.hist(x, bins=bins, histtype="step", label=f"{label} (n={x.size})")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_report(q: Quality, gates: list[dict], mesh: Mesh, out_dir: Path, title: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = q.summary
    T = mesh.n_tris
    (out / "summary.json").write_text(json.dumps({**s, "gates": gates}, indent=1))

    lon, lat = xyz_to_lonlat(q.centroid)
    lim = {g["gate"]: g["limit"] for g in gates}
    bad_q = (q.quad_sj < 0.3) | (q.quad_max_angle_deg > lim.get("quad_max_angle_hard_deg", 150.0))
    bad_t = q.tri_max_angle_deg > lim.get("tri_max_angle_hard_deg", 120.0)
    flagged = np.concatenate([bad_t, bad_q])

    md = [f"# Quality: {title}", "", "## Counts", "", "| key | value |", "|---|---|"]
    md += [f"| {k} | {_f(s[k])} |" for k in _COUNTS]
    md += [f"| n_quads_dev_gt45 | {s['n_quads_dev_gt45']} |", f"| n_quads_sj_lt03 | {s['n_quads_sj_lt03']} |",
           f"| n_quads_sj_lt03_boundary | {s['n_quads_sj_lt03_boundary']} |",
           f"| share_elem_size_ratio_gt_1_2 (all / coast / interior) | {_f(s['share_elem_size_ratio_gt_1_2'])} / "
           f"{_f(s['share_elem_size_ratio_gt_1_2_coast'])} / {_f(s['share_elem_size_ratio_gt_1_2_interior'])} |",
           f"| tri_clusters | {s['tri_clusters']} |", f"| elems_per_node_hist | {s['elems_per_node_hist']} |",
           f"| sqrt_area_m | {s['sqrt_area_m']} |", f"| flagged elements (map) | {int(flagged.sum())} |"]
    md += ["", "## Triangles", ""] + _stat_table(s, _TRI)
    md += ["", "## Quads", ""] + _stat_table(s, _QUAD)
    md += ["", "## Size", ""] + _stat_table(s, _SIZE)
    md += ["", "## Gates", "", "| gate | value | limit | passed |", "|---|---|---|---|"]
    md += [f"| {g['gate']} | {_f(g['value'])} | {_f(g['limit'])} | {'yes' if g['passed'] else '**NO**'} |"
           for g in gates]

    md += ["", "## 20 worst quads by SJ", "", "| quad | SJ | max angle | max dev | coast | lon | lat |",
           "|---|---|---|---|---|---|---|"]
    for i in np.argsort(q.quad_sj, kind="stable")[:20]:
        e = T + i
        md.append(f"| {i} | {q.quad_sj[i]:.3f} | {q.quad_max_angle_deg[i]:.1f} | {q.quad_dev_deg[i]:.1f} | "
                  f"{'yes' if q.touches_boundary[e] else 'no'} | {lon[e]:.4f} | {lat[e]:.4f} |")
    md += ["", "## 20 worst triangles by max angle", "", "| tri | max angle | min angle | coast | lon | lat |",
           "|---|---|---|---|---|---|"]
    for i in np.argsort(-q.tri_max_angle_deg, kind="stable")[:20]:
        md.append(f"| {i} | {q.tri_max_angle_deg[i]:.1f} | {q.tri_min_angle_deg[i]:.1f} | "
                  f"{'yes' if q.touches_boundary[i] else 'no'} | {lon[i]:.4f} | {lat[i]:.4f} |")
    (out / "report.md").write_text("\n".join(md) + "\n")

    _hist(out / "hist_quad_dev.png", [("quads", q.quad_dev_deg)], "max |angle − 90°| per quad (deg)",
          np.arange(0, 181, 1.0))
    _hist(out / "hist_tri_angles.png", [("min angle", q.tri_min_angle_deg), ("max angle", q.tri_max_angle_deg)],
          "triangle angle (deg)", np.arange(0, 181, 1.0))
    hi = max(2.5, float(np.max(q.node_size_ratio, initial=1.0)), float(np.max(q.elem_size_ratio, initial=1.0)))
    _hist(out / "hist_size_ratio.png", [("element mean edge, per interior edge", q.elem_size_ratio),
                                        ("node size, per edge", q.node_size_ratio)],
          "neighbour size ratio (≥ 1)", np.linspace(1.0, min(hi, 5.0), 161))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lon, lat, ",", color="0.75", rasterized=True)
    ax.plot(lon[flagged], lat[flagged], "o", color="red", ms=2.5, label=f"flagged ({int(flagged.sum())})")
    ax.set_xlabel("lon (deg)")
    ax.set_ylabel("lat (deg)")
    ax.set_title(f"{title}: hard-gate failures and SJ < 0.3")
    ax.set_aspect(1.0 / max(np.cos(np.radians(np.median(lat))), 0.1) if lat.size else "auto")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "map_flagged.png", dpi=130)
    plt.close(fig)
