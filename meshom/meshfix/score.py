"""The one scorer of the mesh-fixing competition (decisions B121): a pure function of a case and a candidate mesh.

    python -m meshom.meshfix.score --case NAME --mesh FILE.npz [--json OUT] [--no-figure]

Writes OUT (default FILE.score.json) and the overview maps FILE.<window>.png. A method's own record FILE.json next
to the mesh (if any) gives the time ("wall_s"); the scorer measures no time.

Fixed definitions (every number below is computed the same way for every method and case):

- Regions. A point is **open ocean** when its great-circle distance to the case's coast (Case.dist_seg: the input's
  real coast; for a cut-out also the parent's coast just outside the window, never the artificial boundary) is more
  than 2 h(x) (cases.OPEN_H), else **coastal**. Fixed geography: moving the boundary does not move a defect into
  the coastal class. Elements are classified by their centroid, nodes by their position, edges by their midpoint.
- **Quad max angle**: per quad the largest corner angle in the element's tangent plane (quality.kernel); over all
  quads: count and share above 110 and 120 deg, p99, p99.9, max (linear percentiles); the counts also per region.
- **Triangles**: count, share of the elements, max-angle p99 and max, edge-connected clusters of >= 3 (all
  triangles, triangle zones included), count per region.
- **Irregular interior nodes** by element composition. Interior = not on the candidate's boundary. A pure-quad node
  (all elements quads) is irregular when its quad count is not 4: n2 (<= 2), n3, n5, n6p (>= 6). Nodes touching a
  triangle are counted apart (tri_nodes; tri_nodes_unbalanced = 90 q + 60 t != 360 deg, i.e. 3 q + 2 t != 12).
  Each per region. defect_units_open = open n2 + n3 + n5 + n6p + open triangles (the headline count).
- **Size**: element size ratio = quality.metrics' mean-edge ratio across every interior edge, p99 and max; edge
  |ln s/h|: s = great-circle edge length, h = the case's h at the normalised edge midpoint, over **every edge except
  those with both nodes on the artificial boundary** (for a whole mesh: every edge, compete's size_fidelity); p99,
  max, share above the run's size tolerance; p99 also over the open-ocean edges.
- **Coast fidelity**: forward = distance (km) of each candidate coast node (boundary node not on the artificial
  boundary) to the case's reference coast (Case.coast_seg); backward = of each case coast node (kind 1) to the
  candidate's coast edges; shift = max and p99 over both together. Water area = sum of |element tangent-plane areas|,
  change in % of the input's. Water bodies = edge-connected element pieces; basins_lost = input water bodies whose
  point (Case.basin_pts) has no candidate element centroid within 2 h.
- **Validity**: inverted (signed tangent-plane area <= 0), non-convex quads (a corner >= 180 deg), degenerate (an
  edge < 0.01 h or a corner < 1 deg), non-manifold edges (> 2 elements), repeated directed half-edges, unused nodes,
  duplicate nodes (closer than 1 m), artificial-boundary nodes or edges of the case missing from the candidate (the
  fixed nodes matched by position within 1 m), artificial nodes whose quad or triangle count changed (the artificial
  boundary is no defect sink); compete.irregular_counts' Euler bookkeeping (euler_sum, chi,
  components, euler_ok against the candidate's boundary cycles).
- **Triangle cases** (B154: the case's input has no quad): an extra block "tri" with the interior node
  valence (edges at a node; <= 4, 5, 6, 7, >= 8) open and coastal and the irregular share (valence != 6); the
  triangle corner angles (min angle min / p1, max angle p99 / max, obtuse > 90 / > 100 / > 120, slivers < 30 deg);
  the coast shift in units of h (forward: candidate coast node to the reference coast over h at the node; backward:
  case coast node to the candidate's coast edges over h there); the maps are the triangle maps (figure_tri: valence
  dots, and orthographic zooms on the input's worst areas, the same windows for every candidate of the case).
- **Gates**: compete._score on the candidate with the run's gates and size tolerance (Config.gates_all, as `meshom
  run` / `meshom write` score a final mesh); the domain is the candidate's own boundary cycles (a "cut" run). Also
  new_failed = gates failing here but passing on the case's input (a cut-out's artificial boundary fails some gates
  on the input already).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M, arc_length, xyz_to_lonlat
from meshom.mesh import Mesh, build_topology, signed_area
from meshom.meshfix.cases import Case, boundary_edges, components, elem_centroids, load, segment_distance_km
from meshom.quality.kernel import element_geometry

MATCH_CHORD = 1.0 / R_EARTH_M      # 1 m: a fixed node "kept" / a duplicate node
DEGEN_EDGE_H = 0.01                # degenerate: an edge shorter than this times h
DEGEN_ANGLE_DEG = 1.0              # ... or a corner below this


def _pct(x: np.ndarray, q: float):
    return float(np.percentile(x, q)) if x.size else None


def _max(x: np.ndarray):
    return float(x.max()) if x.size else None


def _incidence(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """(quads, triangles) at each node."""
    n = mesh.n_nodes
    q = np.bincount(mesh.quads.ravel(), minlength=n) if len(mesh.quads) else np.zeros(n, np.int64)
    t = np.bincount(mesh.tris.ravel(), minlength=n) if len(mesh.tris) else np.zeros(n, np.int64)
    return q, t


def boundary_cycles(mesh: Mesh, be: np.ndarray) -> list[np.ndarray]:
    """Closed node cycles of the oriented boundary edges be (water on the left); terminates on pinch nodes too
    (each half-edge is walked once; at a pinch node the first unused outgoing edge is taken)."""
    if len(be) == 0:
        return []
    order = np.argsort(be[:, 0], kind="stable")
    src = be[order, 0]
    ptr = np.searchsorted(src, np.arange(mesh.n_nodes + 1))
    used = np.zeros(len(be), dtype=bool)
    cyc = []
    for s in range(len(be)):
        if used[order[s]]:
            continue
        e = order[s]
        nodes = []
        while not used[e]:
            used[e] = True
            nodes.append(be[e, 0])
            b = be[e, 1]
            nxt = -1
            for j in range(ptr[b], ptr[b + 1]):
                if not used[order[j]]:
                    nxt = order[j]
                    break
            if nxt < 0:
                break
            e = nxt
        cyc.append(np.array(nodes, dtype=np.int64))
    return cyc


def standard_score(case: Case, mesh: Mesh, be: np.ndarray | None = None) -> dict:
    """compete._score of the candidate: the run's gates, the domain = the candidate's boundary cycles."""
    from meshom import compete
    from meshom.domain import Domain

    be = boundary_edges(mesh) if be is None else be
    loops = boundary_cycles(mesh, be)
    dom = Domain([mesh.points[c] for c in loops], "cut", [True] * len(loops))
    return compete._score(mesh, dom, case.h, case.rho_c, case.gates)


def gates_block(case: Case, mesh: Mesh, std: dict | None = None) -> dict:
    std = standard_score(case, mesh) if std is None else std
    base = set(case.meta.get("input_gates_failed", []))
    failed = std["gates_failed"]
    return {"passed": std["gates_passed"], "total": std["gates_total"], "failed": failed,
            "new_failed": sorted(set(failed) - base)}


def _region_counts(mask_open: np.ndarray, **sel) -> dict:
    out = {}
    for reg, m in (("open", mask_open), ("coastal", ~mask_open)):
        out[reg] = {k: int((v & m).sum()) for k, v in sel.items()}
    out["all"] = {k: int(v.sum()) for k, v in sel.items()}
    return out


def analyse(case: Case, mesh: Mesh) -> tuple[dict, dict]:
    """(result, arrays): the score dict (module docstring) and per-element / per-node arrays for the figure."""
    T, Q = mesh.n_tris, mesh.n_quads
    topo = build_topology(mesh)
    g = element_geometry(mesh)
    P = mesh.points
    bnd = topo.boundary_node
    be = boundary_edges(mesh, topo)

    # candidate node kinds: artificial = a boundary node at the position of a case artificial node
    art_case = case.mesh.points[case.kind == 2]
    is_art = np.zeros(mesh.n_nodes, dtype=bool)
    missing_art_nodes = 0
    art_inc_changed = 0
    if len(art_case):
        d, j = cKDTree(P).query(art_case, distance_upper_bound=MATCH_CHORD)
        found = np.isfinite(d)
        missing_art_nodes = int((~found).sum())
        is_art[j[found]] = True
        is_art &= bnd
        # no defect sink: a kept artificial node keeps its quad and triangle counts
        (q0, t0), (q1, t1) = _incidence(case.mesh), _incidence(mesh)
        a0, a1 = np.flatnonzero(case.kind == 2)[found], j[found]
        art_inc_changed = int(((q1[a1] != q0[a0]) | (t1[a1] != t0[a0])).sum())
    coast_node = bnd & ~is_art

    h_node = case.h(P)
    cen = elem_centroids(mesh)
    h_cen = case.h(cen)
    node_open = case.is_open(P, h_node)
    elem_open = case.is_open(cen, h_cen)
    q_open, t_open = elem_open[T:], elem_open[:T]

    # quads
    qa = np.degrees(g.quad_angles)
    qmax = qa.max(axis=1) if Q else np.zeros(0)
    quad = {"n": Q, "n_gt110": int((qmax > 110).sum()), "share_gt110": float(np.mean(qmax > 110)) if Q else None,
            "n_gt120": int((qmax > 120).sum()), "share_gt120": float(np.mean(qmax > 120)) if Q else None,
            "p99": _pct(qmax, 99), "p99_9": _pct(qmax, 99.9), "max": _max(qmax),
            "by_region": _region_counts(q_open, n=np.ones(Q, bool), gt110=qmax > 110, gt120=qmax > 120),
            "p99_9_open": _pct(qmax[q_open], 99.9), "max_open": _max(qmax[q_open])}

    # triangles
    ta = np.degrees(g.tri_angles)
    tmax = ta.max(axis=1) if T else np.zeros(0)
    tri = {"n": T, "share": T / max(T + Q, 1), "max_angle_p99": _pct(tmax, 99), "max_angle_max": _max(tmax),
           "by_region": _region_counts(t_open, n=np.ones(T, bool))}
    if T:
        ee = topo.edge_elems
        tt = (ee[:, 1] >= 0) & (ee[:, 0] < T) & (ee[:, 1] < T)
        from meshom.quality.metrics import _components

        size = np.bincount(_components(T, ee[tt, 0], ee[tt, 1]), minlength=T)
        size = size[size > 0]
        tri.update(clusters_ge3=int((size >= 3).sum()), cluster_max=int(size.max()))
    else:
        tri.update(clusters_ge3=0, cluster_max=0)

    # irregular interior nodes by element composition
    tc = np.bincount(mesh.tris.ravel(), minlength=mesh.n_nodes) if T else np.zeros(mesh.n_nodes, np.int64)
    qc = np.bincount(mesh.quads.ravel(), minlength=mesh.n_nodes) if Q else np.zeros(mesh.n_nodes, np.int64)
    used = (tc + qc) > 0
    inner = ~bnd & used
    pure = inner & (tc == 0)
    trin = inner & (tc > 0)
    cls = {"n2": pure & (qc <= 2), "n3": pure & (qc == 3), "n5": pure & (qc == 5), "n6p": pure & (qc >= 6),
           "tri_nodes": trin, "tri_nodes_unbalanced": trin & (3 * qc + 2 * tc != 12)}
    irr = _region_counts(node_open, **cls)
    irr_q = cls["n2"] | cls["n3"] | cls["n5"] | cls["n6p"]
    irr["defect_units_open"] = int((irr_q & node_open).sum()) + int(t_open.sum())
    irr["defect_units_all"] = int(irr_q.sum()) + T

    # size
    E = topo.edges
    a, b = P[E[:, 0]], P[E[:, 1]]
    mid = a + b
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    length = R_EARTH_M * arc_length(a, b)
    h_mid = case.h(mid)
    r = np.abs(np.log(length / h_mid))
    pop = ~(is_art[E[:, 0]] & is_art[E[:, 1]])
    e_open = case.is_open(mid, h_mid)
    tol = case.tol
    std = standard_score(case, mesh, be)
    s = std["quality"]
    size = {"elem_ratio_p99": s["elem_size_ratio"]["p99"], "elem_ratio_max": s["elem_size_ratio"]["max"],
            "ln_sh_p99": _pct(r[pop], 99), "ln_sh_max": _max(r[pop]), "ln_sh_median": _pct(r[pop], 50),
            "ln_sh_share_gt_tol": float(np.mean(r[pop] > tol)) if tol and pop.any() else None,
            "ln_sh_p99_open": _pct(r[pop & e_open], 99), "n_edges": int(pop.sum()), "tol": tol}

    # coast fidelity
    is_art_edge = is_art[be[:, 0]] & is_art[be[:, 1]]
    cand_seg = np.stack([P[be[~is_art_edge, 0]], P[be[~is_art_edge, 1]]], axis=1)
    fwd = segment_distance_km(P[coast_node], case.coast_seg)
    case_coast = case.mesh.points[case.kind == 1]
    bwd = segment_distance_km(case_coast, cand_seg)
    both = np.r_[fwd, bwd]
    both = both[np.isfinite(both)]
    area = np.abs(signed_area(mesh))
    lab = components(mesh, topo) if mesh.n_elems else np.zeros(0, np.int64)
    lost = 0
    if len(case.basin_pts):
        d, _ = cKDTree(cen).query(case.basin_pts)
        lost = int((d * R_EARTH_M > 2.0 * case.h(case.basin_pts)).sum())
    a0 = case.meta.get("input_area_km2")
    coast = {"shift_km_max": _max(both), "shift_km_p99": _pct(both, 99),
             "forward_km_max": _max(fwd), "forward_km_p99": _pct(fwd, 99),
             "backward_km_max": _max(bwd), "backward_km_p99": _pct(bwd, 99),
             "area_km2": float(area.sum()) / 1e6,
             "area_change_pct": (100.0 * (area.sum() / 1e6 - a0) / a0) if a0 else None,
             "water_bodies": int(lab.max() + 1) if len(lab) else 0,
             "water_bodies_input": int(len(case.basin_pts)), "basins_lost": lost}

    # validity
    sa = signed_area(mesh)
    el_edges = np.concatenate([g.tri_edges.min(axis=1) if T else np.zeros(0),
                               g.quad_edges.min(axis=1) if Q else np.zeros(0)])
    el_minang = np.concatenate([ta.min(axis=1) if T else np.zeros(0), qa.min(axis=1) if Q else np.zeros(0)])
    degen = (el_edges < DEGEN_EDGE_H * h_cen) | (el_minang < DEGEN_ANGLE_DEG)
    ha = np.concatenate([c.ravel() for c in (mesh.tris, mesh.quads) if len(c)])
    hb = np.concatenate([np.roll(c, -1, axis=1).ravel() for c in (mesh.tris, mesh.quads) if len(c)])
    n = mesh.n_nodes
    _, dcnt = np.unique(ha * n + hb, return_counts=True)
    _, ucnt = np.unique(np.minimum(ha, hb) * n + np.maximum(ha, hb), return_counts=True)
    dup = len(cKDTree(P).query_pairs(MATCH_CHORD, output_type="ndarray"))
    missing_art_edges = 0
    if len(case.art_edges):
        ap = case.mesh.points[case.art_edges]                       # (A,2,3)
        tr = cKDTree(P)
        da, ia = tr.query(ap[:, 0], distance_upper_bound=MATCH_CHORD)
        db, ib = tr.query(ap[:, 1], distance_upper_bound=MATCH_CHORD)
        ok = np.isfinite(da) & np.isfinite(db)
        bkey = set((np.minimum(be[:, 0], be[:, 1]) * n + np.maximum(be[:, 0], be[:, 1])).tolist())
        k = np.minimum(ia[ok], ib[ok]) * n + np.maximum(ia[ok], ib[ok])
        missing_art_edges = int((~ok).sum()) + int(sum(x not in bkey for x in k.tolist()))
    ic = std["irregular"]
    valid = {"inverted": int((sa <= 0).sum()),
             "nonconvex_quads": int((qa >= 180.0).any(axis=1).sum()) if Q else 0,
             "degenerate": int(degen.sum()), "nonmanifold_edges": int((ucnt > 2).sum()),
             "repeated_half_edges": int((dcnt > 1).sum()), "unused_nodes": int((~used).sum()),
             "duplicate_nodes": int(dup), "artificial_nodes_missing": missing_art_nodes,
             "artificial_edges_missing": missing_art_edges, "artificial_incidence_changed": art_inc_changed,
             "euler": {k: ic[k] for k in ("euler_sum", "chi", "chi_domain", "components", "euler_ok", "n3", "n5",
                                          "n_other", "T")}}
    valid["ok"] = bool(valid["inverted"] == 0 and valid["nonconvex_quads"] == 0 and valid["nonmanifold_edges"] == 0
                       and valid["repeated_half_edges"] == 0 and valid["unused_nodes"] == 0
                       and valid["duplicate_nodes"] == 0 and missing_art_nodes == 0 and missing_art_edges == 0
                       and art_inc_changed == 0 and ic["euler_ok"])

    res = {"case": case.name, "n_nodes": mesh.n_nodes, "n_quads": Q, "n_tris": T, "quad_angle": quad,
           "triangles": tri, "irregular": irr, "size": size, "coast": coast, "validity": valid,
           "gates": gates_block(case, mesh, std),
           "open_rule": case.meta.get("open_rule", "distance to dist_seg > 2 h(x)")}
    dev = np.concatenate([tmax - 60.0, qmax - 90.0])
    arrays = {"dev": dev, "irr_quad": irr_q, "tri_cen": cen[:T]}
    if is_tri_case(case):
        deg = np.bincount(topo.edges.ravel(), minlength=mesh.n_nodes)
        tb, tarr = tri_block(g.tri_angles if T else np.zeros((0, 3)), deg, inner, node_open)
        cs_h = np.r_[fwd * 1e3 / h_node[coast_node], bwd * 1e3 / case.h(case_coast)] if len(both) else np.zeros(0)
        cs_h = cs_h[np.isfinite(cs_h)]
        tb["coast_shift_h_max"] = _max(cs_h)
        tb["coast_shift_h_p99"] = _pct(cs_h, 99)
        res["tri"] = tb
        arrays.update(tarr, deg=deg, inner=inner)
    return res, arrays


def is_tri_case(case: Case) -> bool:
    """A triangle case (B154): its input mesh has triangles and no quad."""
    return case.mesh.n_quads == 0 and case.mesh.n_tris > 0


def tri_block(ang_rad: np.ndarray, deg: np.ndarray, inner: np.ndarray, node_open: np.ndarray) -> tuple[dict, dict]:
    """The triangle measures (module docstring): (block, arrays for the maps). ang_rad (T,3) corner angles; deg the
    edges per node; inner the interior used nodes; node_open the open-ocean nodes."""
    a = np.degrees(ang_rad)
    amin, amax = (a.min(1), a.max(1)) if len(a) else (np.zeros(0), np.zeros(0))
    v = np.where(deg <= 4, 4, np.where(deg >= 8, 8, deg))

    def hist(m):
        return {"v4": int((m & (v == 4)).sum()), "v5": int((m & (v == 5)).sum()), "v6": int((m & (v == 6)).sum()),
                "v7": int((m & (v == 7)).sum()), "v8p": int((m & (v == 8)).sum())}

    irr = inner & (deg != 6)
    n_in = int(inner.sum())
    blk = {"valence": {"open": hist(inner & node_open), "coastal": hist(inner & ~node_open), "all": hist(inner)},
           "irregular": int(irr.sum()), "irregular_open": int((irr & node_open).sum()),
           "irregular_coastal": int((irr & ~node_open).sum()), "interior": n_in,
           "irregular_share": float(irr.sum() / max(n_in, 1)),
           "irregular_share_open": float((irr & node_open).sum() / max(int((inner & node_open).sum()), 1)),
           "min_angle_min": float(amin.min()) if len(amin) else None, "min_angle_p1": _pct(amin, 1),
           "max_angle_p99": _pct(amax, 99), "max_angle_max": _max(amax),
           "obtuse_gt90": int((amax > 90).sum()), "obtuse_gt100": int((amax > 100).sum()),
           "obtuse_gt120": int((amax > 120).sum()), "sliver_lt30": int((amin < 30).sum())}
    dev = np.abs(a - 60.0).max(1) if len(a) else np.zeros(0)
    return blk, {"tri_dev": dev, "tri_amax": amax}


def score(case: Case, mesh: Mesh, record: dict | None = None) -> dict:
    """The score dict of mesh on case (module docstring); record: the method's own record (method, wall_s)."""
    res, _ = analyse(case, mesh)
    if record is not None:
        res = {"method": record.get("method"), "wall_s": record.get("wall_s", record.get("time_s")), **res}
    return res


# ------------------------------------------------------------------------------------------------------- figures

def _ps(lon, lat, lon0):
    r = np.tan(np.radians(90 - np.asarray(lat, float)) / 2)
    a = np.radians(np.asarray(lon, float) - lon0)
    return r * np.sin(a) * 2 * 6371.0, -r * np.cos(a) * 2 * 6371.0


def figure(case: Case, mesh: Mesh, stem: Path, arrays: dict, res: dict, label: str = "") -> list[Path]:
    """Overview maps in the style of the v4 maps: every element coloured by its max-angle deviation (quads: max angle -
    90, triangles: max angle - 60; 0-30 deg), black dots the irregular pure-quad interior nodes, blue triangles the
    triangle centroids; one PNG per case window: <stem>.<window>.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    T = mesh.n_tris
    cells = np.vstack([np.c_[mesh.tris, mesh.tris[:, 2]] if T else np.zeros((0, 4), np.int64), mesh.quads])
    nlon, nlat = xyz_to_lonlat(mesh.points)
    clon, clat = xyz_to_lonlat(elem_centroids(mesh))
    dev = arrays["dev"]
    cmap = plt.get_cmap("magma_r")
    norm = matplotlib.colors.Normalize(0, 30)
    out = []
    q, irr, tr = res["quad_angle"], res["irregular"], res["triangles"]
    for w in case.meta.get("windows", []):
        lw, le = w["lon"]
        ls, ln = w["lat"]
        wrap = lambda x: (x - lw) % 360 <= (le - lw)                        # noqa: E731
        sel = np.flatnonzero(wrap(clon) & (clat >= ls) & (clat <= ln))
        if w["proj"] == "stereo":
            fx = lambda lo, la: _ps(lo, la, w["lon0"])                      # noqa: E731
        else:
            fx = lambda lo, la: (np.asarray(lo, float), np.asarray(la, float))   # noqa: E731
        x, y = fx(nlon[cells[sel]], nlat[cells[sel]])
        if w["proj"] == "plate":                                            # drop cells across the antimeridian
            ok = (x.max(axis=1) - x.min(axis=1)) < 180
            sel, x, y = sel[ok], x[ok], y[ok]
        fig, ax = plt.subplots(figsize=(22, 13) if w["proj"] == "stereo" else (24, 12))
        ax.add_collection(PolyCollection(np.stack([x, y], axis=-1), facecolors=cmap(norm(np.clip(dev[sel], 0, 30))),
                                         edgecolors="none"))
        ns = arrays["irr_quad"] & wrap(nlon) & (nlat >= ls) & (nlat <= ln)
        px, py = fx(nlon[ns], nlat[ns])
        ax.plot(px, py, "k.", ms=1.6, label=f"irregular pure-quad nodes ({int(ns.sum())})")
        if T:
            tlon, tlat = xyz_to_lonlat(arrays["tri_cen"])
            ts = wrap(tlon) & (tlat >= ls) & (tlat <= ln)
            px, py = fx(tlon[ts], tlat[ts])
            ax.plot(px, py, "^", color="tab:blue", ms=2.0, label=f"triangles ({int(ts.sum())})")
        for lat0 in w.get("lats", []):
            gl = np.linspace(lw, le, 400)
            px, py = fx(gl, np.full_like(gl, lat0))
            ax.plot(px, py, "--", color="tab:green", lw=0.8)
        c = np.array([fx(lo, la) for lo in np.linspace(lw, le, 50) for la in (ls, ln)])
        ax.set_xlim(c[:, 0].min(), c[:, 0].max())
        ax.set_ylim(c[:, 1].min(), c[:, 1].max())
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=10)
        g = res["gates"]
        ax.set_title(f"{label} {case.name} / {w['name']}: element max-angle deviation (quads - 90, triangles - 60, "
                     f"0-30 deg). Whole case: quads > 110 / > 120 deg {q['n_gt110']} / {q['n_gt120']}, p99.9 "
                     f"{q['p99_9']:.1f}, max {q['max']:.1f}; open-ocean defect units {irr['defect_units_open']}; "
                     f"triangles {tr['n']}; |ln s/h| p99 {res['size']['ln_sh_p99']:.3f}; gates {g['passed']}/"
                     f"{g['total']}", fontsize=11)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, label="deg")
        f = Path(f"{stem}.{w['name']}.png")
        fig.savefig(f, dpi=110, bbox_inches="tight")
        plt.close(fig)
        out.append(f)
    return out


# ------------------------------------------------------------------------------------------------------- table

def _round(x):
    if isinstance(x, float):
        return float(f"{x:.6g}")
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return [_round(v) for v in x] if isinstance(x, list) else x


_HEAD = ("case", "method", "nodes", "quads >110 / >120", "open >110 / >120", "quad θmax p99 / p99.9 / max",
         "tris / θmax max / clusters≥3", "open n3 / n5 / 6+ / tri-nodes", "coastal n3 / n5 / 6+ / tri-nodes",
         "open defect units", "ratio p99 / max", "\\|ln s/h\\| p99 (open)", "coast shift km max / p99", "area %",
         "water bodies (lost)", "valid", "gates (new fails)", "time s")


def _f(v, nd):
    return "-" if v is None else f"{v:.{nd}f}"


def table(rows: list[dict]) -> str:
    """Markdown table of score dicts (each with "method"); triangle cases (a "tri" block) get table_tri."""
    if rows and all("tri" in r for r in rows):
        return table_tri(rows)
    out = ["| " + " | ".join(_HEAD) + " |", "|" + "---|" * len(_HEAD)]
    for r in rows:
        q, t, i, s, c, v, g = (r["quad_angle"], r["triangles"], r["irregular"], r["size"], r["coast"],
                               r["validity"], r["gates"])
        o, k = i["open"], i["coastal"]
        bad = [x for x in ("inverted", "nonconvex_quads", "degenerate", "nonmanifold_edges", "duplicate_nodes",
                           "artificial_nodes_missing", "artificial_edges_missing", "artificial_incidence_changed")
               if v.get(x)]
        out.append("| " + " | ".join([
            r["case"], str(r.get("method")), str(r["n_nodes"]), f"{q['n_gt110']} / {q['n_gt120']}",
            f"{q['by_region']['open']['gt110']} / {q['by_region']['open']['gt120']}",
            f"{_f(q['p99'], 1)} / {_f(q['p99_9'], 1)} / {_f(q['max'], 1)}",
            f"{t['n']} / {_f(t['max_angle_max'], 1)} / {t['clusters_ge3']}",
            f"{o['n3']} / {o['n5']} / {o['n6p']} / {o['tri_nodes']}",
            f"{k['n3']} / {k['n5']} / {k['n6p']} / {k['tri_nodes']}", str(i["defect_units_open"]),
            f"{_f(s['elem_ratio_p99'], 3)} / {_f(s['elem_ratio_max'], 2)}",
            f"{_f(s['ln_sh_p99'], 3)} ({_f(s['ln_sh_p99_open'], 3)})",
            f"{_f(c['shift_km_max'], 2)} / {_f(c['shift_km_p99'], 2)}", _f(c["area_change_pct"], 3),
            f"{c['water_bodies']} ({c['basins_lost']})",
            ("yes" if v["ok"] else "NO") + (f" ({', '.join(f'{x} {v[x]}' for x in bad)})" if bad else ""),
            f"{g['passed']}/{g['total']}" + (f" ({', '.join(g['new_failed'])})" if g["new_failed"] else ""),
            _f(r.get("wall_s"), 0)]) + " |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------- triangle cases (B154)

ZOOM_N = 3          # B154: zoom maps per triangle case, on the input's worst areas
ZOOM_RADIUS_H = 15  # B154: half-width of a zoom map in units of h at its centre
ZOOM_SEP = 3.0      # B154: zoom centres at least this many half-widths apart


def zoom_windows(case: Case) -> list[dict]:
    """The input's worst areas (the same for every candidate of the case): centres at the input's worst triangles
    (largest |angle - 60|), greedily at least ZOOM_SEP half-widths apart; half-width ZOOM_RADIUS_H h there."""
    m = case.mesh
    a = np.degrees(element_geometry(m).tri_angles)
    dev = np.abs(a - 60.0).max(1)
    cen = elem_centroids(m)[:m.n_tris]
    order = np.argsort(-dev, kind="stable")[:20000]
    hc = case.h(cen[order])
    out = []
    for i, t in enumerate(order):
        r = ZOOM_RADIUS_H * hc[i] / R_EARTH_M
        if all(np.linalg.norm(cen[t] - np.asarray(w["c"])) > ZOOM_SEP * max(r, w["r"]) for w in out):
            out.append({"c": cen[t].tolist(), "r": float(r), "dev": float(dev[t])})
        if len(out) == ZOOM_N:
            break
    return out


def figure_tri(case: Case, mesh: Mesh, stem: Path, arrays: dict, res: dict, label: str = "") -> list[Path]:
    """Triangle maps: every triangle coloured by its worst |angle - 60| (0-30 deg), interior nodes of valence 5
    (blue), 7 (red), <= 4 (cyan), >= 8 (green); one PNG per case window (<stem>.<window>.png) and one per zoom on the
    input's worst areas (<stem>.zoom<k>.png, orthographic about the centre, edges drawn), plus a zoom on the
    candidate's own worst triangle (<stem>.zoomc.png)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from meshom.geo import tangent_basis

    P, Tr = mesh.points, mesh.tris
    dev, deg, inner = arrays["tri_dev"], arrays["deg"], arrays["inner"]
    nlon, nlat = xyz_to_lonlat(P)
    cen = elem_centroids(mesh)[:mesh.n_tris]
    clon, clat = xyz_to_lonlat(cen)
    cmap = plt.get_cmap("magma_r")
    norm = matplotlib.colors.Normalize(0, 30)
    t = res["tri"]
    head = (f"{label} {case.name}: irregular {t['irregular']} ({100 * t['irregular_share']:.1f} %), obtuse "
            f"{t['obtuse_gt90']}, max angle {t['max_angle_max']:.1f}, min angle {t['min_angle_min']:.1f}, ratio p99 "
            f"{res['size']['elem_ratio_p99']:.3f}, |ln s/h| p99 {res['size']['ln_sh_p99']:.3f}")
    dots = ((5, "tab:blue", "valence 5"), (7, "tab:red", "valence 7"), (4, "tab:cyan", "valence <= 4"),
            (8, "tab:green", "valence >= 8"))

    def sel_v(vv):
        return inner & ((deg <= 4) if vv == 4 else (deg >= 8) if vv == 8 else (deg == vv))

    out = []
    for w in case.meta.get("windows", []):
        lw, le = w["lon"]
        ls, ln = w["lat"]
        wrap = lambda x: (x - lw) % 360 <= (le - lw)                        # noqa: E731
        sel = np.flatnonzero(wrap(clon) & (clat >= ls) & (clat <= ln))
        if w["proj"] == "stereo":
            fx = lambda lo, la: _ps(lo, la, w["lon0"])                      # noqa: E731
        else:
            fx = lambda lo, la: (np.asarray(lo, float), np.asarray(la, float))   # noqa: E731
        x, y = fx(nlon[Tr[sel]], nlat[Tr[sel]])
        if w["proj"] == "plate":
            ok = (x.max(axis=1) - x.min(axis=1)) < 180
            sel, x, y = sel[ok], x[ok], y[ok]
        fig, ax = plt.subplots(figsize=(22, 13) if w["proj"] == "stereo" else (24, 12))
        ax.add_collection(PolyCollection(np.stack([x, y], axis=-1), facecolors=cmap(norm(np.clip(dev[sel], 0, 30))),
                                         edgecolors="none"))
        nin = wrap(nlon) & (nlat >= ls) & (nlat <= ln)
        for vv, col, lab in dots:
            m = sel_v(vv) & nin
            px, py = fx(nlon[m], nlat[m])
            ax.plot(px, py, ".", color=col, ms=1.0, label=f"{lab} ({int(m.sum())})")
        c = np.array([fx(lo, la) for lo in np.linspace(lw, le, 50) for la in (ls, ln)])
        ax.set_xlim(c[:, 0].min(), c[:, 0].max())
        ax.set_ylim(c[:, 1].min(), c[:, 1].max())
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=10)
        ax.set_title(f"{head} / {w['name']}", fontsize=11)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, label="worst |angle - 60| deg")
        f = Path(f"{stem}.{w['name']}.png")
        fig.savefig(f, dpi=110, bbox_inches="tight")
        plt.close(fig)
        out.append(f)
    zooms = [(f"zoom{k}", w) for k, w in enumerate(case.meta.get("zooms") or zoom_windows(case))]
    if len(dev):
        tw = int(np.argmax(dev))
        zooms.append(("zoomc", {"c": cen[tw].tolist(), "r": ZOOM_RADIUS_H * float(case.h(cen[tw:tw + 1])[0])
                                / R_EARTH_M}))
    bnd = np.zeros(mesh.n_nodes, bool)
    bnd[boundary_edges(mesh).ravel()] = True
    for nm, w in zooms:
        c = np.asarray(w["c"])
        r = w["r"]
        e1, e2 = tangent_basis(c[None])
        near_t = np.flatnonzero(cen @ c > np.cos(1.5 * r))
        near_n = P @ c > np.cos(1.5 * r)
        X, Y = P @ e1[0] * R_EARTH_M / 1e3, P @ e2[0] * R_EARTH_M / 1e3
        fig, ax = plt.subplots(figsize=(13, 12))
        ax.add_collection(PolyCollection(np.stack([X[Tr[near_t]], Y[Tr[near_t]]], -1),
                                         facecolors=cmap(norm(np.clip(dev[near_t], 0, 30))),
                                         edgecolors=(0.4, 0.4, 0.4, 0.6), linewidths=0.3))
        for vv, col, lab in dots:
            m = sel_v(vv) & near_n
            ax.plot(X[m], Y[m], ".", color=col, ms=5, label=f"{lab} ({int(m.sum())})")
        m = bnd & near_n
        ax.plot(X[m], Y[m], "s", color="tab:green", ms=2.5, mfc="none", label="boundary nodes")
        cx, cy = c @ e1[0] * R_EARTH_M / 1e3, c @ e2[0] * R_EARTH_M / 1e3
        rk = r * R_EARTH_M / 1e3
        ax.set_xlim(cx - rk, cx + rk)
        ax.set_ylim(cy - rk, cy + rk)
        ax.set_aspect(1)
        lo, la = xyz_to_lonlat(c[None])
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"{head}\n{nm} at {float(lo[0]):.2f} E {float(la[0]):.2f} N (km)", fontsize=10)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, label="worst |angle - 60| deg")
        f = Path(f"{stem}.{nm}.png")
        fig.savefig(f, dpi=100, bbox_inches="tight")
        plt.close(fig)
        out.append(f)
    return out


_HEAD_TRI = ("case", "method", "nodes", "irregular (share); open / coastal", "valence 4 / 5 / 7 / 8+",
             "min angle min / p1", "max angle p99 / max", "obtuse >90 / >100 / >120", "ratio p99 / max",
             "\\|ln s/h\\| p99 / max", "coast shift h max (km)", "area %", "water bodies (lost)", "valid",
             "gates (new fails)", "time s")


def table_tri(rows: list[dict]) -> str:
    """Markdown table of triangle-case score dicts."""
    out = ["| " + " | ".join(_HEAD_TRI) + " |", "|" + "---|" * len(_HEAD_TRI)]
    for r in rows:
        t, s, c, v, g = r["tri"], r["size"], r["coast"], r["validity"], r["gates"]
        va = t["valence"]["all"]
        bad = [x for x in ("inverted", "degenerate", "nonmanifold_edges", "duplicate_nodes",
                           "artificial_nodes_missing", "artificial_edges_missing", "artificial_incidence_changed")
               if v.get(x)]
        out.append("| " + " | ".join([
            r["case"], str(r.get("method")), str(r["n_nodes"]),
            f"{t['irregular']} ({100 * t['irregular_share']:.1f} %); {t['irregular_open']} / {t['irregular_coastal']}",
            f"{va['v4']} / {va['v5']} / {va['v7']} / {va['v8p']}",
            f"{_f(t['min_angle_min'], 1)} / {_f(t['min_angle_p1'], 1)}",
            f"{_f(t['max_angle_p99'], 1)} / {_f(t['max_angle_max'], 1)}",
            f"{t['obtuse_gt90']} / {t['obtuse_gt100']} / {t['obtuse_gt120']}",
            f"{_f(s['elem_ratio_p99'], 3)} / {_f(s['elem_ratio_max'], 2)}",
            f"{_f(s['ln_sh_p99'], 3)} / {_f(s['ln_sh_max'], 3)}",
            f"{_f(t.get('coast_shift_h_max'), 3)} ({_f(c['shift_km_max'], 2)})", _f(c["area_change_pct"], 3),
            f"{c['water_bodies']} ({c['basins_lost']})",
            ("yes" if v["ok"] else "NO") + (f" ({', '.join(f'{x} {v[x]}' for x in bad)})" if bad else ""),
            f"{g['passed']}/{g['total']}" + (f" ({', '.join(g['new_failed'])})" if g["new_failed"] else ""),
            _f(r.get("wall_s"), 0)]) + " |")
    return "\n".join(out) + "\n"


def score_file(case_name: str, mesh_path: Path, out: Path | None = None, figures: bool = True,
               case: Case | None = None, label: str | None = None) -> dict:
    """Score the mesh file against the case, write <mesh>.score.json (or out) and the figures; returns the dict."""
    from meshom.compete import _load

    mesh_path = Path(mesh_path)
    case = load(case_name) if case is None else case
    mesh = _load(mesh_path)
    rec_f = mesh_path.with_suffix(".json")
    record = json.loads(rec_f.read_text()) if rec_f.exists() else None
    res, arrays = analyse(case, mesh)
    if record is not None:
        res = {"method": record.get("method"), "wall_s": record.get("wall_s", record.get("time_s")), **res}
    res["mesh"] = str(mesh_path)
    res = _round(res)
    out = Path(out) if out else mesh_path.with_suffix(".score.json")
    out.write_text(json.dumps(res, indent=1))
    if figures:
        stem = mesh_path.with_suffix("")
        (figure_tri if "tri" in res else figure)(case, mesh, stem, arrays, res,
                                                 label if label is not None else str(res.get("method") or ""))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m meshom.meshfix.score", description=__doc__.splitlines()[0])
    ap.add_argument("--case", required=True)
    ap.add_argument("--mesh", required=True, type=Path)
    ap.add_argument("--json", type=Path, default=None, help="output (default <mesh>.score.json)")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args(argv)
    r = score_file(a.case, a.mesh, a.json, not a.no_figure)
    print(table([r]), end="")


if __name__ == "__main__":
    main(sys.argv[1:])
