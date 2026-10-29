"""Measure existing curvilinear ocean grids with meshom's quality kernel on the sphere.

For each grid: the full grid (land cells included) gives the topology — non-manifold edges, singular nodes (interior
nodes with valence != 4, wet or on land), holes (boundary loops: domain edges, displaced poles), the logical blocks
and how their cells meet (regular, periodic seam, fold, block joins); the ocean mesh (wet cells) gives
compute_quality, irregular_counts, the aspect ratio (mean of opposite edges, long / short) and how the coast is
followed (share of coast nodes where the boundary turns: 1 or >= 3 elements; ~0 for a boundary-fitted grid, large
for a staircase). One JSON line per grid into bench/curvilinear_grids.jsonl, one figure per grid (overview, zoom at
a pole / fold / worst cell, zoom at a coast) into the grid's data folder.

    python scripts/grids/measure_grid.py NAME [NAME ...] | --all     (on a compute node: scripts/on_node.sh)
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_grid import DATA, GRIDS, read, submesh  # noqa: E402

from meshom.compete import _round, irregular_counts  # noqa: E402
from meshom.geo import R_EARTH_M, lonlat_to_xyz, tangent_basis, xyz_to_lonlat  # noqa: E402
from meshom.mesh import build_topology  # noqa: E402
from meshom.quality.kernel import element_geometry  # noqa: E402
from meshom.quality.metrics import compute_quality  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "bench" / "curvilinear_grids.jsonl"
COAST_ZOOM = (10.5, 56.0)        # Danish straits: used when the grid covers it


def _stats(x, keys=(50, 99, 99.9, 100)):
    if x.size == 0:
        return None
    return dict(zip(("median", "p99", "p99_9", "max")[: len(keys)], np.percentile(x, keys).tolist()))


def topology_report(gm):
    """Full grid: manifoldness, singular nodes, holes, block joins."""
    m = gm.mesh
    a = np.concatenate([m.tris.ravel(), m.quads.ravel()])
    b = np.concatenate([np.roll(m.tris, -1, 1).ravel(), np.roll(m.quads, -1, 1).ravel()])
    n = m.n_nodes
    el = np.concatenate([np.repeat(np.arange(m.n_tris), 3), m.n_tris + np.repeat(np.arange(m.n_quads), 4)])
    ewet = gm.wet[el]
    und = np.minimum(a, b) * n + np.maximum(a, b)
    _, inv, cnt = np.unique(und, return_inverse=True, return_counts=True)
    uwet = np.zeros(len(cnt), bool)
    np.logical_or.at(uwet, inv, ewet)
    _, dinv, dcnt = np.unique(a * n + b, return_inverse=True, return_counts=True)
    dwet = np.zeros(len(dcnt), bool)
    np.logical_or.at(dwet, dinv, ewet)
    topo = build_topology(m)
    deg = np.bincount(topo.edges.ravel(), minlength=n)
    elem_wet = gm.wet
    node_wet = np.zeros(n, bool)
    for cells, off in ((m.tris, 0), (m.quads, m.n_tris)):
        if len(cells):
            np.logical_or.at(node_wet, cells, elem_wet[off: off + len(cells)][:, None])
    sing = np.flatnonzero(~topo.boundary_node & (deg != 4))
    lon, lat = xyz_to_lonlat(m.points[sing])
    singular = [dict(lon=float(x), lat=float(y), valence=int(deg[k]), wet=bool(node_wet[k]))
                for k, x, y in zip(sing, lon, lat)]
    # holes: connected components of the boundary edges
    ee = topo.edge_elems
    be = topo.edges[ee[:, 1] < 0]
    holes = []
    if len(be):
        g = coo_matrix((np.ones(len(be)), (be[:, 0], be[:, 1])), shape=(n, n))
        lab = connected_components(g, directed=False)[1]
        bl = lab[be[:, 0]]
        for c in np.unique(bl):
            nodes = np.unique(be[bl == c].ravel())
            cen = m.points[nodes].mean(0)
            norm = np.linalg.norm(cen)
            x, y = xyz_to_lonlat(cen / norm) if norm > 1e-9 else (0.0, 90.0)
            rad = R_EARTH_M / 1e3 * float(np.max(np.arccos(np.clip(m.points[nodes] @ (cen / max(norm, 1e-12)), -1, 1))))
            holes.append(dict(n_edges=int((bl == c).sum()), lon=float(x), lat=float(y), radius_km=rad,
                              wet_share=float(node_wet[nodes].mean())))
        holes.sort(key=lambda h: -h["n_edges"])
    # block joins over the interior edges of the full grid
    inner = ee[:, 1] >= 0
    e0, e1 = ee[inner, 0], ee[inner, 1]
    bk, jj, ii = gm.block, gm.j, gm.i
    same = bk[e0] == bk[e1]
    dj, di = np.abs(jj[e0] - jj[e1]), np.abs(ii[e0] - ii[e1])
    ny = np.array([s[0] for s in gm.shapes])[bk[e0]]
    nx = np.array([s[1] for s in gm.shapes])[bk[e0]]
    regular = same & (dj + di == 1)
    lo, hi = np.minimum(jj[e0], jj[e1]), np.maximum(jj[e0], jj[e1])
    mirror = np.abs((ii[e0] + ii[e1] - (nx - 1) + nx // 2) % nx - nx // 2) <= 2       # i1 ~ nx - 1 - i0 (+halo)
    seam = same & ~regular & (dj <= 1) & (di >= nx - 6)                                  # periodic wrap (+halos)
    fold = same & ~regular & ~seam & ((lo >= ny - 3) | (hi <= 2)) & mirror
    other = same & ~regular & ~seam & ~fold
    joins = {"regular": int(regular.sum()), "seam": int(seam.sum()), "fold": int(fold.sum()),
             "other": int(other.sum())}
    if (~same).any():
        pa, pb = np.minimum(bk[e0], bk[e1])[~same], np.maximum(bk[e0], bk[e1])[~same]
        for (x, y), c in zip(*np.unique(np.c_[pa, pb], axis=0, return_counts=True)):
            joins[f"block{x}-block{y}"] = int(c)
    return dict(edges_gt2=int((cnt > 2).sum()), flipped_edges=int((dcnt > 1).sum()),
                edges_gt2_wet=int(((cnt > 2) & uwet).sum()), flipped_edges_wet=int(((dcnt > 1) & dwet).sum()),
                n_holes_wet=int(sum(h["wet_share"] > 0 for h in holes)),
                n_singular=len(singular), n_singular_wet=int(sum(s["wet"] for s in singular)),
                singular=singular[:40], holes=holes[:12], n_holes=len(holes), n_blocks=len(gm.shapes),
                block_shapes=[list(map(int, s)) for s in gm.shapes], joins=joins,
                euler_full=irregular_counts(m)), node_wet


def ocean_report(gm):
    wm, keep = submesh(gm, gm.wet)
    q = compute_quality(wm)
    ic = irregular_counts(wm)
    ge = element_geometry(wm)
    qe = ge.quad_edges
    ar = np.maximum(qe[:, 0] + qe[:, 2], qe[:, 1] + qe[:, 3]) / np.minimum(qe[:, 0] + qe[:, 2], qe[:, 1] + qe[:, 3])
    topo = build_topology(wm)
    a = np.concatenate([wm.tris.ravel(), wm.quads.ravel()])
    b = np.concatenate([np.roll(wm.tris, -1, 1).ravel(), np.roll(wm.quads, -1, 1).ravel()])
    _, cnt = np.unique(np.minimum(a, b) * wm.n_nodes + np.maximum(a, b), return_counts=True)
    _, dcnt = np.unique(a * wm.n_nodes + b, return_counts=True)
    bn = topo.boundary_node
    eb = topo.elems_per_node[bn]
    hist = {int(k): int(c) for k, c in zip(*np.unique(eb, return_counts=True))}
    coast = dict(boundary_nodes=int(bn.sum()), elems_hist=hist,
                 corner_share=float(np.mean(eb != 2)) if eb.size else None)
    ar_tb = q.touches_boundary[wm.n_tris:]
    return wm, q, dict(n_nodes=wm.n_nodes, n_quads=wm.n_quads, n_tris=wm.n_tris,
                       edges_gt2=int((cnt > 2).sum()), flipped_edges=int((dcnt > 1).sum()),
                       aspect=_stats(ar), aspect_coast=_stats(ar[ar_tb]), aspect_interior=_stats(ar[~ar_tb]),
                       share_aspect_gt_1_5=float(np.mean(ar > 1.5)) if ar.size else None,
                       share_aspect_gt_2=float(np.mean(ar > 2.0)) if ar.size else None,
                       quality=q.summary, irregular=ic, coast=coast), ar


# ------------------------------------------------------------------------------------------------------- figures

def _ortho(p, c):
    """Orthographic projection (km) about the unit vector c; p (..., 3)."""
    e1, e2 = tangent_basis(c[None])
    return R_EARTH_M / 1e3 * np.stack([p @ e1[0], p @ e2[0]], axis=-1)


def _zoom(ax, gm, cen_xyz, radius_km, maxang, title, marks):
    m = gm.mesh
    T = m.n_tris
    cen = np.vstack([m.points[m.tris].mean(1), m.points[m.quads].mean(1)])
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    dot = cen @ cen_xyz
    sel = np.flatnonzero(dot > np.cos(1.5 * radius_km * 1e3 / R_EARTH_M))
    sel = sel[np.argsort(-dot[sel])[:60000]]                     # the nearest 60000 cells
    sel = np.sort(sel)
    st, sq = sel[sel < T], sel[sel >= T] - T
    polys = list(_ortho(m.points[m.tris[st]], cen_xyz)) + list(_ortho(m.points[m.quads[sq]], cen_xyz))
    sel = np.r_[st, sq + T]
    wet = gm.wet[sel]
    val = maxang[sel]
    cmap = plt.get_cmap("YlOrRd")
    fc = np.where(wet[:, None], cmap(np.clip((val - 90.0) / 45.0, 0, 1) * 0.85 + 0.05),
                  np.array([[0.72, 0.72, 0.72, 1.0]]))
    lw = 0.5 if len(sel) < 3000 else 0.25
    ax.add_collection(PolyCollection(polys, facecolors=fc, edgecolors="0.15", linewidths=lw, rasterized=True))
    for xyz, style in marks:
        if xyz @ cen_xyz > np.cos(1.5 * radius_km * 1e3 / R_EARTH_M):
            x, y = _ortho(xyz[None], cen_xyz)[0]
            ax.plot(x, y, style, ms=8, mfc="none", mew=1.8)
    ax.set_xlim(-radius_km, radius_km)
    ax.set_ylim(-radius_km, radius_km)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("km", fontsize=8)
    ax.tick_params(labelsize=7)


def figure(name, gm, topo, node_wet, path):
    m = gm.mesh
    ge = element_geometry(m)
    maxang = np.degrees(np.concatenate([ge.tri_angles.max(1, initial=0), ge.quad_angles.max(1, initial=0)]))
    size_km = np.sqrt(np.abs(np.concatenate([ge.tri_area, ge.quad_area]))) / 1e3
    lon, lat = xyz_to_lonlat(ge.centroid)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.0), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    ax = axs[0]
    logs = np.log10(size_km)
    vmin, vmax = np.percentile(logs[gm.wet], [0.5, 99.5]) if gm.wet.any() else (0, 1)
    if len(gm.wet) <= 300_000:         # every cell as a lon-lat polygon (cells across the antimeridian skipped)
        T = m.n_tris
        corners = [m.points[m.tris], m.points[m.quads]]
        polys, vals, wet = [], [], []
        for k, cc in enumerate(corners):
            if not len(cc):
                continue
            lo, la = xyz_to_lonlat(cc)
            ok = np.ptp(lo, axis=1) < 180.0
            polys += list(np.stack([lo, la], axis=-1)[ok])
            off = 0 if k == 0 else T
            idx = off + np.flatnonzero(ok)
            vals.append(logs[idx])
            wet.append(gm.wet[idx])
        vals, wet = np.concatenate(vals), np.concatenate(wet)
        cmap = plt.get_cmap("turbo")
        fc = np.where(wet[:, None], cmap(np.clip((vals - vmin) / max(vmax - vmin, 1e-9), 0, 1)),
                      np.array([[0.85, 0.85, 0.85, 1.0]]))
        ax.add_collection(PolyCollection(polys, facecolors=fc, edgecolors="none", rasterized=True))
        ax.set_xlim(lon.min(), lon.max())
        ax.set_ylim(lat.min(), lat.max())
    else:                               # raster: each pixel takes the nearest cell (centroid within 1.5 cells)
        x0, x1, y0, y1 = lon.min(), lon.max(), lat.min(), lat.max()
        res = max((x1 - x0) / 1400.0, (y1 - y0) / 700.0)
        px, py = np.meshgrid(np.arange(x0, x1, res) + res / 2, np.arange(y0, y1, res) + res / 2)
        d, k = cKDTree(ge.centroid).query(lonlat_to_xyz(px.ravel(), py.ravel()))
        inside = d * R_EARTH_M / 1e3 < 1.5 * size_km[k]
        img = np.where(inside & gm.wet[k], logs[k], np.nan).reshape(px.shape)
        landimg = np.where(inside & ~gm.wet[k], 1.0, np.nan).reshape(px.shape)
        ext = (x0, x0 + px.shape[1] * res, y0, y0 + px.shape[0] * res)
        ax.imshow(landimg, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=6, aspect="auto",
                  interpolation="nearest")
        ax.imshow(img, origin="lower", extent=ext, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto",
                  interpolation="nearest")
    sc = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("log10 cell size (km)", fontsize=8)
    for s in topo["singular"]:
        ax.plot(s["lon"], s["lat"], "r^" if s["wet"] else "kx", ms=7, mfc="none", mew=1.5)
    for h in [h for h in topo["holes"][1:] if h["n_edges"] > 8]:
        ax.plot(h["lon"], h["lat"], "ks", ms=7, mfc="none", mew=1.2)
    meta = GRIDS[name]
    ax.set_title(f"{name}: {meta['config']}, {len(gm.wet)} cells ({gm.wet.sum()} wet)\n"
                 f"^ singular node in water, x on land, square: hole (displaced pole)", fontsize=9)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.tick_params(labelsize=7)
    # zoom 1: a singular node (wet ones first, then the northernmost: tripolar poles, cap corners), else a hole that
    # is not the largest boundary loop (displaced pole), else the worst wet cell
    marks = [(lonlat_to_xyz(s["lon"], s["lat"]), "r^" if s["wet"] else "kx") for s in topo["singular"]]
    sing = sorted(topo["singular"], key=lambda s: (not s["wet"], -s["lat"]))
    holes = topo["holes"][1:]
    med = float(np.median(size_km[gm.wet]))
    holes = [h for h in holes if h["lat"] > 0 and h["n_edges"] > 8]
    if sing:
        c = lonlat_to_xyz(sing[0]["lon"], sing[0]["lat"])
        t1 = f"singular node, valence {sing[0]['valence']} ({'water' if sing[0]['wet'] else 'land'})"
        rad = 12 * med
    elif holes:
        c = lonlat_to_xyz(holes[0]["lon"], holes[0]["lat"])
        t1 = f"hole: displaced pole ({holes[0]['n_edges']} edges, wet share {holes[0]['wet_share']:.2f})"
        rad = max(12 * med, 1.4 * holes[0]["radius_km"])
    else:
        e = np.flatnonzero(gm.wet)[np.argmax(maxang[gm.wet])]
        c = ge.centroid[e]
        t1 = f"worst wet cell: max angle {maxang[e]:.1f} deg"
        near = np.argsort(np.linalg.norm(ge.centroid - c, axis=1))[:200]
        rad = 12 * float(np.median(size_km[near]))
    _zoom(axs[1], gm, c, rad, maxang, t1, marks)
    # zoom 2: a coast
    wetc = ge.centroid[gm.wet]
    cz = lonlat_to_xyz(*COAST_ZOOM)
    if np.min(np.linalg.norm(wetc - cz, axis=1)) * R_EARTH_M > 100e3:
        # a coast node (shared by wet and land cells; a grid without land cells: its outline), nearest the middle
        node_land = np.zeros(m.n_nodes, bool)
        for cells, off in ((m.tris, 0), (m.quads, m.n_tris)):
            if len(cells):
                np.logical_or.at(node_land, cells, ~gm.wet[off: off + len(cells)][:, None])
        cn = np.flatnonzero(node_wet & node_land)
        if not len(cn):
            cn = np.flatnonzero(build_topology(m).boundary_node)
        mid = wetc.mean(0)
        mid /= np.linalg.norm(mid)
        cz = m.points[cn[np.argmin(np.linalg.norm(m.points[cn] - mid, axis=1))]]
    near = np.argsort(np.linalg.norm(ge.centroid - cz, axis=1))[:200]
    loc = float(np.median(size_km[near]))
    _zoom(axs[2], gm, cz, 14 * loc, maxang, "coast (wet: max angle, white 90 to red 135 deg; land grey)", marks)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def measure(name, fig_dir=None):
    t0 = time.perf_counter()
    gm = read(name)
    t_read = time.perf_counter() - t0
    topo, node_wet = topology_report(gm)
    wm, q, ocean, ar = ocean_report(gm)
    meta = {k: v for k, v in GRIDS[name].items() if k != "read"}
    rec = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), grid=name, **meta,
               cells_read=gm.n_cells_in, cells=int(len(gm.wet)), cells_wet=int(gm.wet.sum()),
               duplicates_dropped=gm.n_duplicates, collapsed_dropped=gm.n_collapsed,
               tris_full=int(gm.mesh.n_tris), merge_tol_m=gm.tol_m, topology=topo, ocean=ocean,
               read_s=t_read)
    path = (Path(fig_dir) if fig_dir else DATA / name) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure(name, gm, topo, node_wet, path)
    rec["figure"] = str(path)
    rec["wall_s"] = time.perf_counter() - t0
    return _round(rec)


def summary_line(r):
    o, t = r["ocean"], r["topology"]
    qs = o["quality"]
    a = o["aspect"] or {}
    return (f"{r['grid']}: {r['cells_wet']}/{r['cells']} wet cells, size med {qs['sqrt_area_m']['median'] / 1e3:.3g} km; "
            f"AR p50/p99/max {a.get('median', 0):.2f}/{a.get('p99', 0):.2f}/{a.get('max', 0):.2f}; "
            f"qmax p99.9/max {qs['quad_max_angle_deg']['p99_9']:.1f}/{qs['quad_max_angle_deg']['max']:.1f}; "
            f"esr p99/max {qs['elem_size_ratio']['p99']:.3f}/{qs['elem_size_ratio']['max']:.2f}; "
            f"singular {t['n_singular']} ({t['n_singular_wet']} wet); holes {t['n_holes']}; "
            f"edges>2/flipped all {t['edges_gt2']}/{t['flipped_edges']} wet {t['edges_gt2_wet']}/"
            f"{t['flipped_edges_wet']}; holes wet {t['n_holes_wet']}; joins {t['joins']}; "
            f"ocean mesh edges>2/flipped {o['edges_gt2']}/{o['flipped_edges']}; "
            f"coast corner share {o['coast']['corner_share']:.2f}; {r['wall_s']:.0f} s")


def _joins(t):
    j = t["joins"]
    parts = [f"{k} {j[k]}" for k in ("seam", "fold", "other") if j.get(k)]
    blocks = sum(v for k, v in j.items() if k.startswith("block"))
    if blocks:
        parts.append(f"{sum(k.startswith('block') for k in j)} block joins ({blocks} edges)")
    return f"{t['n_blocks']}: " + (", ".join(parts) if parts else "-")


def _singular(t):
    by = {}
    for s in t["singular"]:
        by.setdefault((s["valence"], s["wet"]), 0)
        by[(s["valence"], s["wet"])] += 1
    txt = ", ".join(f"{n}x v{v} {'WET' if w else 'land'}" for (v, w), n in sorted(by.items()))
    holes = [h for h in t["holes"][1:] if h["n_edges"] > 8]
    if holes:
        txt += (", " if txt else "") + f"{len(holes)} hole(s) " + "/".join(
            f"{abs(h['lat']):.0f}{'N' if h['lat'] >= 0 else 'S'} {'wet' if h['wet_share'] > 0 else 'land'}"
            for h in holes[:3])
    return txt or "-"


def table(path=OUT) -> str:
    """Markdown summary table of the last record per grid."""
    last = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["grid"]] = r
    head = ("grid", "family", "config", "res", "cells (wet / all)", "source", "licence", "downloaded",
            "ocean nodes", "size km p1/med/max", "AR p50 / p99 / max", "AR > 1.5 %", "θmax p99.9 / max",
            "elem ratio p99 / max", "singular points", "blocks: joins", "coast turn %")
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name in [g for g in GRIDS if g in last]:
        r = last[name]
        o, t, q = r["ocean"], r["topology"], r["ocean"]["quality"]
        a, sa = o["aspect"], q["sqrt_area_m"]
        qa, er = q["quad_max_angle_deg"], q["elem_size_ratio"]
        src = r["version"].split(" (")[0] if r["version"].startswith("doi") else r["url"]
        rows.append("| " + " | ".join([
            name, r["family"], r["config"], r["res"], f"{r['cells_wet']} / {r['cells']}", src, r["licence"],
            r["downloaded"], str(o["n_nodes"]),
            f"{sa['p1'] / 1e3:.3g} / {sa['median'] / 1e3:.3g} / {sa['max'] / 1e3:.3g}",
            f"{a['median']:.2f} / {a['p99']:.2f} / {a['max']:.3g}", f"{100 * o['share_aspect_gt_1_5']:.1f}",
            f"{qa['p99_9']:.1f} / {qa['max']:.1f}", f"{er['p99']:.3f} / {er['max']:.2f}", _singular(t), _joins(t),
            f"{100 * o['coast']['corner_share']:.0f}"]) + " |")
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--table", action="store_true", help="print the Markdown summary table of --out and stop")
    a = ap.parse_args()
    if a.table:
        print(table(a.out), end="")
        sys.exit()
    for name in (list(GRIDS) if a.all else a.names):
        try:
            r = measure(name)
        except Exception as e:  # one broken grid must not stop the batch
            print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(summary_line(r), flush=True)
