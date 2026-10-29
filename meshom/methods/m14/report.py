"""m14 step 6: measurement: coverage, blocks, irregular nodes inside blocks and in the seam band, and
the statistics per element row counted from the block outlines (block side and fill side: rows 0, 1, 2, 3-5, > 5).

Row of an element: stitch.element_rows (0 = touches an outline edge). An edge belongs to the row group of its adjacent
element nearest to the outline (outline edges to their own group "outline"); a node to the group of its incident
element nearest to the outline. Irregular = interior node (not on the coast) with valence != 4. The seam band = the
outline nodes and the fill elements of rows 0-3 with their nodes and edges (an edge whose nearest element is a fill
element of rows 0-3, plus the outline edges); "fill_all" = every fill element (the outline
nodes and fill nodes off the coast, the outline and fill edges). Triangles with a corner above 100 degrees are counted
as a share (seam band, whole fill): the angle measure of a triangle fill, whose valence counts against 4 mean nothing;
"fill_all" also counts the fill's own nodes (outline and coast excluded) with valence != 6 (val6_*: the triangle-fill
defects).
"""
import numpy as np

from meshom.geo import R_EARTH_M, arc_length, xyz_to_lonlat
from meshom.mesh import Mesh
from meshom.methods.m14.stitch import element_rows
from meshom.quality.kernel import element_geometry

BUCKETS = ("0", "1", "2", "3-5", ">5")


def _bucket(r: np.ndarray) -> np.ndarray:
    return np.select([r == 0, r == 1, r == 2, (r >= 3) & (r <= 5)], [0, 1, 2, 3], 4)


def _pct(x: np.ndarray, q: float):
    return round(float(np.percentile(x, q)), 4) if len(x) else None


def _mx(x: np.ndarray):
    return round(float(x.max()), 4) if len(x) else None


def measure(mesh: Mesh, h, st: dict) -> dict:
    is_block, ol = st["is_block"], st["outline_edges"]
    n, T, M = mesh.n_nodes, mesh.n_tris, mesh.n_elems
    row, topo, is_ol = element_rows(mesh, is_block, ol)
    g = element_geometry(mesh)
    area = np.abs(np.r_[g.tri_area, g.quad_area])
    out = {"coverage_area": float(area[is_block].sum() / area.sum()),
           "coverage_cells": float(is_block.sum() / M),
           "n_block_cells": int(is_block.sum()), "n_elems": int(M)}
    # element groups: side 0 = block, 1 = fill; bucket by row (unreached = > 5)
    grp = np.where(is_block, 0, 5) + _bucket(np.where(row < 0, 99, row))
    names = [f"block_{b}" for b in BUCKETS] + [f"fill_{b}" for b in BUCKETS]
    maxang = np.r_[np.degrees(g.tri_angles).max(axis=1, initial=0.0), np.degrees(g.quad_angles).max(axis=1, initial=0.0)]
    is_tri = np.arange(M) < T
    mean_edge = np.r_[g.tri_edges.mean(axis=1), g.quad_edges.mean(axis=1)]
    # edges: |ln s/h|, size ratio across interior edges; group = the adjacent element nearest to the outline
    E, ee = topo.edges, topo.edge_elems
    p, q = mesh.points[E[:, 0]], mesh.points[E[:, 1]]
    mid = p + q
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    lnsh = np.abs(np.log(R_EARTH_M * arc_length(p, q) / h(mid)))
    inner = ee[:, 1] >= 0
    e1 = np.where(inner, ee[:, 1], ee[:, 0])
    ratio = np.maximum(mean_edge[ee[:, 0]], mean_edge[e1]) / np.minimum(mean_edge[ee[:, 0]], mean_edge[e1])
    rr = np.where(row < 0, 99, row)
    pick = np.where(rr[ee[:, 0]] <= rr[e1], ee[:, 0], e1)
    egrp = grp[pick]
    # nodes: valence, irregular, group of the nearest incident element
    deg = np.bincount(E.ravel(), minlength=n)
    bnd = topo.boundary_node
    irr = (deg != 4) & ~bnd
    nrow = np.full(n, 99, dtype=np.int64)
    for cells, off in ((mesh.tris, 0), (mesh.quads, T)):
        for k in range(cells.shape[1]):
            np.minimum.at(nrow, cells[:, k], rr[off:off + len(cells)])
    ol_nodes = np.unique(ol)
    block_nodes = st["block_nodes"]
    on_block = np.zeros(n, dtype=bool)
    on_block[block_nodes] = True
    on_block[ol_nodes] = False
    ngrp = np.where(on_block, 0, 5) + _bucket(nrow)
    block_inner = np.setdiff1d(block_nodes, ol_nodes)
    groups = {}
    for gi, name in enumerate(names):
        em = grp == gi
        if not em.any():
            continue
        qm = em & ~is_tri
        ed = (egrp == gi) & ~is_ol
        nd = (ngrp == gi) & ~bnd
        if name.startswith("block"):
            nd &= ~np.isin(np.arange(n), ol_nodes)
        groups[name] = {"elems": int(em.sum()), "tris": int((em & is_tri).sum()),
                        "quad_max_deg": _mx(maxang[qm]), "quad_p99_9_deg": _pct(maxang[qm], 99.9),
                        "quad_p99_deg": _pct(maxang[qm], 99),
                        "ln_sh_p99": _pct(lnsh[ed], 99), "ln_sh_max": _mx(lnsh[ed]),
                        "elem_ratio_p99": _pct(ratio[ed & inner], 99), "elem_ratio_max": _mx(ratio[ed & inner]),
                        "nodes": int(nd.sum()), "irregular": int((nd & irr).sum())}
    out["rows"] = groups
    out["outline_edges"] = {"n": int(is_ol.sum()), "ln_sh_p99": _pct(lnsh[is_ol], 99),
                            "elem_ratio_p99": _pct(ratio[is_ol], 99), "elem_ratio_max": _mx(ratio[is_ol])}
    out["irregular_in_blocks"] = int(irr[block_inner].sum())
    # inside the blocks: angles, aspect (mean of opposite sides, long / short), size; edges between two block cells
    bq = is_block[T:]
    qe = g.quad_edges[bq]
    asp = (qe[:, 0] + qe[:, 2]) / (qe[:, 1] + qe[:, 3])
    asp = np.maximum(asp, 1.0 / asp)
    bmax = maxang[T:][bq]
    be = is_block[ee[:, 0]] & inner & is_block[e1]
    out["blocks_interior"] = {"quads": int(bq.sum()), "quad_max_deg": _mx(bmax), "quad_p99_9_deg": _pct(bmax, 99.9),
                              "quad_p99_deg": _pct(bmax, 99), "share_quad_gt_100deg": float(np.mean(bmax > 100.0))
                              if len(bmax) else None,
                              "aspect_p99": _pct(asp, 99), "aspect_max": _mx(asp),
                              "ln_sh_p99": _pct(lnsh[be], 99), "ln_sh_max": _mx(lnsh[be]),
                              "elem_ratio_p99": _pct(ratio[be], 99), "elem_ratio_max": _mx(ratio[be])}
    seam_el = ~is_block & (row >= 0) & (row <= 3)
    seam_nodes = np.zeros(n, dtype=bool)
    for cells, off in ((mesh.tris, 0), (mesh.quads, T)):
        seam_nodes[cells[seam_el[off:off + len(cells)]].ravel()] = True
    seam_nodes[ol_nodes] = True
    seam_nodes[block_inner] = False
    sm = seam_nodes & ~bnd
    seam_edges = (~is_block[pick] & (rr[pick] <= 3)) | is_ol      # rows 0-3 as for the elements and nodes
    sq = seam_el & ~is_tri
    st_ = seam_el & is_tri
    out["seam_band"] = {"nodes": int(sm.sum()), "irregular": int((sm & irr).sum()),
                        "n3": int((sm & (deg == 3)).sum()), "n5": int((sm & (deg == 5)).sum()),
                        "tris": int((seam_el & is_tri).sum()), "elems": int(seam_el.sum()),
                        "quad_max_deg": _mx(maxang[sq]), "quad_p99_9_deg": _pct(maxang[sq], 99.9),
                        "tri_max_deg": _mx(maxang[st_]),
                        "tri_gt100_share": float(np.mean(maxang[st_] > 100.0)) if st_.any() else None,
                        "ln_sh_p99": _pct(lnsh[seam_edges], 99),
                        "elem_ratio_p99": _pct(ratio[seam_edges & inner], 99),
                        "elem_ratio_max": _mx(ratio[seam_edges & inner])}
    fill_far = ~is_block & ((row < 0) | (row > 3))
    fn = np.zeros(n, dtype=bool)
    for cells, off in ((mesh.tris, 0), (mesh.quads, T)):
        fn[cells[fill_far[off:off + len(cells)]].ravel()] = True
    fn &= ~seam_nodes & ~bnd
    fn[block_nodes] = False
    out["fill_beyond_band"] = {"nodes": int(fn.sum()), "irregular": int((fn & irr).sum()),
                               "tris": int((fill_far & is_tri).sum())}
    # the whole fill (every non-block element; nodes: outline and fill nodes off the coast; edges: outline + fill)
    fe = ~is_block
    fnodes = np.zeros(n, dtype=bool)
    for cells, off in ((mesh.tris, 0), (mesh.quads, T)):
        fnodes[cells[fe[off:off + len(cells)]].ravel()] = True
    fnodes[block_inner] = False
    fnodes &= ~bnd
    fed = (egrp >= 5) | is_ol
    fq, ft = fe & ~is_tri, fe & is_tri
    out["fill_all"] = {"nodes": int(fnodes.sum()), "irregular": int((fnodes & irr).sum()),
                       "n3": int((fnodes & (deg == 3)).sum()), "n5": int((fnodes & (deg == 5)).sum()),
                       "elems": int(fe.sum()), "tris": int(ft.sum()),
                       "quad_p99_9_deg": _pct(maxang[fq], 99.9), "tri_max_deg": _mx(maxang[ft]),
                       "tri_gt100_share": float(np.mean(maxang[ft] > 100.0)) if ft.any() else None,
                       "ln_sh_p99": _pct(lnsh[fed], 99), "elem_ratio_p99": _pct(ratio[fed & inner], 99),
                       "elem_ratio_max": _mx(ratio[fed & inner])}
    # triangle fill: valence against 6 of the fill's own nodes (outline and coast nodes excluded)
    fo = fnodes.copy()
    fo[ol_nodes] = False
    out["fill_all"].update(val6_nodes=int(fo.sum()), val6_irregular=int((fo & (deg != 6)).sum()),
                           val6_n5=int((fo & (deg == 5)).sum()), val6_n7=int((fo & (deg == 7)).sum()),
                           val6_share=float(np.mean(deg[fo] != 6)) if fo.any() else None)
    return out


def fixed_seams(mesh: Mesh, h, st: dict) -> list[dict]:
    """Per fixed block (st["fixed_id"]): measure() with that block alone as the block set and its outline edges off
    the coast (its mouths) as the outline: the seam band at the mouths, the block's interior, its outline edges. Edges
    on the mesh boundary are no seam (a hand-placed block cut by the mask: its staircase, m14/hand.py). The coast
    sides are st["fixed_bank_edges"] (FixedBlock.bank): a block one cell wide has mouths between two bank nodes."""
    fx, bk = st["fixed_id"], st["fixed_bank_edges"]
    T = mesh.n_tris
    bkey = np.r_[bk[:, 0] * mesh.n_nodes + bk[:, 1], bk[:, 1] * mesh.n_nodes + bk[:, 0]]
    allq = np.r_[mesh.tris.ravel(), mesh.quads.ravel()]
    nxt = np.r_[np.roll(mesh.tris, -1, axis=1).ravel(), np.roll(mesh.quads, -1, axis=1).ravel()]
    allkey = allq * mesh.n_nodes + nxt                               # every half-edge of the mesh
    out = []
    for k in range(int(fx.max()) + 1 if len(fx) else 0):
        el = fx == k
        q = mesh.quads[el[T:]]
        a, b = q.ravel(), np.roll(q, -1, axis=1).ravel()
        key = a * mesh.n_nodes + b
        own = ~np.isin(b * mesh.n_nodes + a, key)                     # the block's boundary half-edges
        e = np.column_stack([a[own], b[own]])
        e = e[~np.isin(e[:, 0] * mesh.n_nodes + e[:, 1], bkey)]       # off the coast: the mouths
        e = e[np.isin(e[:, 1] * mesh.n_nodes + e[:, 0], allkey)]      # an element across: not the mesh boundary
        r = measure(mesh, h, {"is_block": el, "outline_edges": e, "block_nodes": np.unique(q)})
        out.append({"cells": int(el.sum()), "mouth_edges": int(len(e)), "seam_band": r["seam_band"],
                    "blocks_interior": r["blocks_interior"], "outline_edges": r["outline_edges"]})
    return out


def outline_record(gr) -> dict:
    """A block's outline for m14.json: outline_lonlat = the outline loop (fill on the left, not closed) in lon-lat
    degrees (4 decimals), outline_corners = indices into it of the corners, the vertices where the loop turns by more
    than 45 degrees in the tangent plane (a block-grid outline turns by about 0 or +-90 degrees at each node, so a
    staircase outline has two corners per step)."""
    P = gr.points[gr.outline]
    if len(P) < 3:
        return {}
    u = P - np.roll(P, 1, axis=0)
    w = np.roll(P, -1, axis=0) - P
    u -= (u * P).sum(axis=1, keepdims=True) * P                     # both onto the tangent plane at the vertex
    w -= (w * P).sum(axis=1, keepdims=True) * P
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (u * w).sum(axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1))
    lon, lat = xyz_to_lonlat(P)
    return {"outline_lonlat": np.round(np.column_stack([lon, lat]), 4).tolist(),
            "outline_corners": np.flatnonzero(cos < np.cos(np.radians(45.0))).tolist()}


def block_records(regions, grids, fit) -> list[dict]:
    """One record per region: seed, cells, B, F, gradient, lambda range and the grid's info (with `sync` on: the frame
    turn sync_psi_deg, the count shift sync_sigma and its bound sync_s_max of m14/sync.py; the corridors' phi and b
    are in m14.json "sync"), the outline."""
    recs = []
    for r, gr in zip(regions, grids):
        lon, lat = xyz_to_lonlat(fit.bg.points[r.seed])
        rec = {"seed_lonlat": [round(float(lon), 4), round(float(lat), 4)], "region_cells": round(r.cells, 1),
               "B": round(r.B, 4), "F": round(r.F, 4), "grad": round(r.grad, 4),
               "lam_min": round(float(r.lam.min()), 4), "lam_max": round(float(r.lam.max()), 4), **r.info}
        if fit.wet is not None:                           # B70: under a land cut
            rec["seed_wet"] = bool(fit.wet[r.seed])
            if r.water is not None:
                rec["region_water_cells"] = round(r.water, 1)
        if gr is not None:
            rec.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in gr.info.items()})
            rec["outline_nodes"] = int(len(gr.outline))
            rec.update(outline_record(gr))
        recs.append(rec)
    return recs
