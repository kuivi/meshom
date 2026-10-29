"""Quality summary of a mesh, size fidelity against h, the gate check.

Statistics are dicts of np.percentile values (linear); over an empty set every value is None. Angle, SJ, edge-ratio
and size-ratio statistics are also given for the subsets (the first 3 coastal rows,
strait corridors; for per-edge ratios: edges with an element of the subset on either side):
  _coast     elements touching a boundary node (the first coastal row)
  _interior  the rest
  _coast3    the first three coastal rows: elements with a node at most 2 edge hops from a boundary node
  _strait    elements whose centroid lies in the case's strait corridor (`strait`, a callable xyz -> bool, e.g.
             meshom.cases.STRAITS[case]); empty without one
Size fidelity leaves the boundary edges themselves (the coast resampled at h) out of its subsets and reports them as
`size_fidelity_boundary`.
"""
from dataclasses import dataclass

import numpy as np

from meshom.geo import R_EARTH_M, arc_length
from meshom.mesh import Mesh, build_topology
from meshom.quality.kernel import element_geometry

_Q = {"min": 0.0, "p0_1": 0.1, "p1": 1.0, "median": 50.0, "p95": 95.0, "p99": 99.0, "p99_9": 99.9, "max": 100.0}


@dataclass
class Quality:
    summary: dict
    quad_sj: np.ndarray                 # (Q,) min corner sin(angle), < 0 at a reflex corner
    quad_max_angle_deg: np.ndarray      # (Q,)
    tri_max_angle_deg: np.ndarray       # (T,)
    tri_min_angle_deg: np.ndarray       # (T,)
    elem_size_ratio_max: np.ndarray     # (T+Q,) largest mean-edge ratio over the element's interior edges (1 if none)
    touches_boundary: np.ndarray        # (T+Q,) bool, element has a boundary node
    quad_dev_deg: np.ndarray            # (Q,) max |angle - 90|
    elem_size_ratio: np.ndarray         # (interior edges,) mean-edge ratio >= 1
    node_size_ratio: np.ndarray         # (E,) node-size ratio >= 1
    centroid: np.ndarray                # (T+Q,3) unit vectors


def _stats(x: np.ndarray, keys: tuple) -> dict:
    if x.size == 0:
        return dict.fromkeys(keys)
    v = np.percentile(x, [_Q[k] for k in keys])
    return {k: float(a) for k, a in zip(keys, v)}


REGIONS = ("", "_coast", "_interior", "_coast3", "_strait")


def _split(s: dict, name: str, x: np.ndarray, masks: dict, keys: tuple) -> None:
    """s[name + region] = statistics of x over each region mask ("" = all of x)."""
    s[name] = _stats(x, keys)
    for reg, m in masks.items():
        s[name + reg] = _stats(x[m], keys)


def element_regions(mesh: Mesh, topo, strait=None, rows: int = 3) -> dict:
    """Per-element masks {"_coast", "_interior", "_coast3", "_strait"} (see the module docstring)."""
    bnode = topo.boundary_node
    tb = np.concatenate([bnode[mesh.tris].any(axis=1), bnode[mesh.quads].any(axis=1)])
    d = np.where(bnode, 0, rows)                       # edge hops from the boundary, capped at `rows`
    E = topo.edges
    for k in range(1, rows):
        near = (d[E[:, 0]] == k - 1) | (d[E[:, 1]] == k - 1)
        hit = np.zeros(len(d), dtype=bool)
        hit[E[near].ravel()] = True
        d = np.where(hit & (d > k), k, d)
    dmin = np.concatenate([d[mesh.tris].min(axis=1, initial=rows), d[mesh.quads].min(axis=1, initial=rows)])
    if strait is None or mesh.n_elems == 0:
        st = np.zeros(mesh.n_elems, dtype=bool)
    else:
        c = np.concatenate([mesh.points[mesh.tris].sum(axis=1), mesh.points[mesh.quads].sum(axis=1)])
        st = np.asarray(strait(c / np.linalg.norm(c, axis=1, keepdims=True)), dtype=bool)
    return {"_coast": tb, "_interior": ~tb, "_coast3": dmin <= rows - 1, "_strait": st}


def _edge_masks(masks: dict, ee: np.ndarray, inner_only: bool) -> dict:
    """Element masks -> edge masks: an element of the region on either side (inner_only: interior edges only)."""
    inner = ee[:, 1] >= 0
    e1 = np.maximum(ee[:, 1], 0)
    out = {}
    for reg, m in masks.items():
        if reg == "_interior":
            continue
        out[reg] = m[ee[:, 0]] | (inner & m[e1])
        if inner_only:
            out[reg] &= inner
    return out


def _share(x: np.ndarray, limit: float):
    return float(np.mean(x > limit)) if x.size else None


def _components(n: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Connected-component label per vertex 0..n-1 of the graph with edges (u, v): min-label hooking + shortcutting."""
    lab = np.arange(n)
    while True:
        lu, lv = lab[u], lab[v]
        m = np.minimum(lu, lv)
        new = lab.copy()
        np.minimum.at(new, lu, m)
        np.minimum.at(new, lv, m)
        while True:
            nn = new[new]
            if np.array_equal(nn, new):
                break
            new = nn
        if np.array_equal(new, lab):
            return lab
        lab = new


def compute_quality(mesh: Mesh, strait=None) -> Quality:
    """Quality summary of the mesh; `strait` (callable xyz -> bool) defines the `_strait` subset."""
    topo = build_topology(mesh)
    g = element_geometry(mesh)
    T, Q, M = mesh.n_tris, mesh.n_quads, mesh.n_elems
    ee = topo.edge_elems
    inner = ee[:, 1] >= 0
    e0, e1 = ee[inner, 0], ee[inner, 1]

    em = element_regions(mesh, topo, strait)
    tb = em["_coast"]
    tm = {k: v[:T] for k, v in em.items()}               # triangle and quad masks
    qm = {k: v[T:] for k, v in em.items()}

    s: dict = {
        "n_nodes": mesh.n_nodes, "n_elems": M, "n_tris": T, "n_quads": Q,
        "n_edges": len(topo.edges), "n_edges_interior": int(inner.sum()),
        "n_tri_quad_edges": int(((e0 < T) != (e1 < T)).sum()),
        "elem_neighbours_min": int(topo.elem_neighbours.min()) if M else None,
    }
    val, cnt = np.unique(topo.elems_per_node, return_counts=True)
    s["elems_per_node_hist"] = {int(k): int(c) for k, c in zip(val, cnt)}
    s["elems_per_node_max"] = int(val.max()) if len(val) else None

    tt = (e0 < T) & (e1 < T)
    if T:
        size = np.bincount(_components(T, e0[tt], e1[tt]), minlength=T)
        size = size[size > 0]
        s["tri_clusters"] = {"n": int(len(size)), "max_size": int(size.max()), "n_ge3": int((size >= 3).sum())}
    else:
        s["tri_clusters"] = {"n": 0, "max_size": None, "n_ge3": 0}
    # the cluster gate: triangles outside the declared triangle zones (Mesh.tri_zone); None without quads
    zone = np.zeros(T, dtype=bool) if mesh.tri_zone is None else np.asarray(mesh.tri_zone, dtype=bool)
    s["n_tri_zone"] = int(zone.sum())
    s["tri_share"] = (T / M) if (Q > 0 and M > 0) else None        # gate tri_share_max; N/A for triangle-only meshes
    if Q == 0:
        s["tri_cluster_max"] = None
    elif (~zone).any():
        free = tt & ~zone[np.minimum(e0, max(T - 1, 0))] & ~zone[np.minimum(e1, max(T - 1, 0))]
        lab = _components(T, e0[free], e1[free])
        size = np.bincount(lab[~zone], minlength=T)
        s["tri_cluster_max"] = int(size.max())
    else:
        s["tri_cluster_max"] = 0

    # boundary structure: pinch nodes (> 2 boundary edges) and elements with >= 2 boundary edges
    bedge = ~inner
    nb = np.bincount(ee[bedge, 0], minlength=M)
    s["n_pinch_nodes"] = int((np.bincount(topo.edges[bedge].ravel(), minlength=mesh.n_nodes) > 2).sum())
    s["n_elems_ge2_boundary_edges"] = int((nb >= 2).sum())
    s["n_tris_ge2_boundary_edges"] = int((nb[:T] >= 2).sum())
    has_b = bool(bedge.any())                            # gates on the boundary: None on a closed mesh
    s["boundary"] = {"pinch_nodes": s["n_pinch_nodes"] if has_b else None,
                     "edges_per_elem_max": int(nb.max()) if has_b and M else None,
                     "edges_per_tri_max": int(nb[:T].max()) if has_b and T else None}

    # triangles
    tri_deg = np.degrees(g.tri_angles)
    tri_min, tri_max = tri_deg.min(axis=1, initial=np.inf), tri_deg.max(axis=1, initial=-np.inf)
    _split(s, "tri_min_angle_deg", tri_min, tm, ("min", "p1", "median"))
    _split(s, "tri_max_angle_deg", tri_max, tm, ("p99", "max"))

    # quads
    quad_deg = np.degrees(g.quad_angles)
    quad_dev = np.abs(quad_deg - 90.0).max(axis=1, initial=-np.inf)
    quad_max = quad_deg.max(axis=1, initial=-np.inf)
    quad_sj = np.sin(g.quad_angles).min(axis=1, initial=np.inf)
    quad_er = g.quad_edges.max(axis=1, initial=-np.inf) / g.quad_edges.min(axis=1, initial=np.inf)
    _split(s, "quad_dev_deg", quad_dev, qm, ("median", "p99", "p99_9", "max"))
    _split(s, "quad_max_angle_deg", quad_max, qm, ("p99_9", "max"))
    s["n_quads_dev_gt45"] = int((quad_dev > 45.0).sum())
    _split(s, "quad_sj", quad_sj, qm, ("p0_1", "min"))
    s["n_quads_sj_lt03"] = int((quad_sj < 0.3).sum())
    s["n_quads_sj_lt03_boundary"] = int(((quad_sj < 0.3) & qm["_coast"]).sum())
    _split(s, "quad_edge_ratio", quad_er, qm, ("p99", "max"))

    # size: element mean edge and sqrt(area) ratios per interior edge, node-size ratio per edge
    mean_edge = np.concatenate([g.tri_edges.mean(axis=1), g.quad_edges.mean(axis=1)])
    sqrt_area = np.sqrt(np.abs(np.concatenate([g.tri_area, g.quad_area])))
    a, b = mean_edge[e0], mean_edge[e1]
    esr = np.maximum(a, b) / np.minimum(a, b)
    a, b = sqrt_area[e0], sqrt_area[e1]
    sar = np.maximum(a, b) / np.minimum(a, b)
    ecoast = tb[e0] | tb[e1]
    xm = {"_coast": ecoast, "_interior": ~ecoast,
          **{k: v[inner] for k, v in _edge_masks(em, ee, True).items() if k != "_coast"}}
    sr_keys = ("median", "p95", "p99", "max")
    _split(s, "elem_size_ratio", esr, xm, sr_keys)
    s["share_elem_size_ratio_gt_1_2"] = _share(esr, 1.2)
    for reg, m in xm.items():
        s["share_elem_size_ratio_gt_1_2" + reg] = _share(esr[m], 1.2)
    _split(s, "elem_sqrt_area_ratio", sar, xm, sr_keys)

    ed = topo.edges
    length = R_EARTH_M * arc_length(mesh.points[ed[:, 0]], mesh.points[ed[:, 1]])
    deg = np.bincount(ed.ravel(), minlength=mesh.n_nodes)
    node_size = np.bincount(ed.ravel(), weights=np.repeat(length, 2), minlength=mesh.n_nodes) / np.maximum(deg, 1)
    a, b = node_size[ed[:, 0]], node_size[ed[:, 1]]
    nsr = np.maximum(a, b) / np.minimum(a, b)
    nm = _edge_masks(em, ee, False)
    nm["_interior"] = ~nm["_coast"]
    _split(s, "node_size_ratio", nsr, nm, ("p99", "max"))
    s["sqrt_area_m"] = _stats(sqrt_area, ("p1", "median", "max"))

    esr_max = np.ones(M)
    np.maximum.at(esr_max, e0, esr)
    np.maximum.at(esr_max, e1, esr)
    return Quality(s, quad_sj, quad_max, tri_max, tri_min, esr_max, tb, quad_dev, esr, nsr, g.centroid)


def _edge_coast(mesh: Mesh, topo) -> np.ndarray:
    """(E,) bool: an element on either side of the edge has a boundary node."""
    bnode = topo.boundary_node
    tb = np.concatenate([bnode[mesh.tris].any(axis=1), bnode[mesh.quads].any(axis=1)])
    ee = topo.edge_elems
    return tb[ee[:, 0]] | ((ee[:, 1] >= 0) & tb[np.maximum(ee[:, 1], 0)])


def size_fidelity(mesh: Mesh, h, strait=None) -> dict:
    """Size fidelity |ln(s/h)| per edge: s = great-circle edge length (m), h at the edge midpoint.

    Returns summary entries, to merge with Quality.summary.update(...) so the size_fidelity_p99 gate sees them:
    "size_fidelity" (every edge) and its subsets "_coast", "_interior", "_coast3", "_strait", each {median, p99,
    max} of |ln(s/h)| plus bias = median of the signed ln(s/h) (> 0: edges longer than h). A subset edge has an
    element of the subset on either side (as for node_size_ratio); the coast subsets leave out the boundary edges
    themselves (the coast loops resampled at h by the method or the domain), which are reported alone as
    "size_fidelity_boundary". "_interior" = edges with no coast element on either side.
    """
    topo = build_topology(mesh)
    p, q = mesh.points[topo.edges[:, 0]], mesh.points[topo.edges[:, 1]]
    mid = p + q
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    r = np.log(R_EARTH_M * arc_length(p, q) / h(mid))
    inner = topo.edge_elems[:, 1] >= 0
    em = _edge_masks(element_regions(mesh, topo, strait), topo.edge_elems, True)
    coast_any = _edge_coast(mesh, topo)

    def fid(x: np.ndarray) -> dict:
        d = _stats(np.abs(x), ("median", "p99", "max"))
        d["bias"] = float(np.median(x)) if x.size else None
        return d

    return {"size_fidelity": fid(r), "size_fidelity_coast": fid(r[em["_coast"]]),
            "size_fidelity_interior": fid(r[~coast_any]), "size_fidelity_coast3": fid(r[em["_coast3"]]),
            "size_fidelity_strait": fid(r[em["_strait"]]), "size_fidelity_boundary": fid(r[~inner])}


# gate name -> (summary metric, statistic or None for a scalar, comparison). A gate on a metric with region variants
# is evaluated once per region present (REGIONS), named gate + region suffix. A gate whose metric is not in the summary
# (size fidelity without h, euler_ok outside compete._score) is skipped; a value of None (empty set, not applicable)
# passes. "boundary_edges_max" reads the statistic named by the option gates["boundary_rule"].
_GATES = {
    "quad_max_angle_hard_deg": ("quad_max_angle_deg", "max", "le"),
    "quad_max_angle_target_deg": ("quad_max_angle_deg", "p99_9", "le"),
    "tri_max_angle_hard_deg": ("tri_max_angle_deg", "max", "le"),
    "tri_max_angle_target_deg": ("tri_max_angle_deg", "p99", "le"),     # only when the mesh has no quads
    "tri_min_angle_hard_deg": ("tri_min_angle_deg", "min", "ge"),
    "quad_sj_hard": ("quad_sj", "min", "ge"),
    "elem_size_ratio_p99": ("elem_size_ratio", "p99", "le"),
    "elem_size_ratio_max": ("elem_size_ratio", "max", "le"),
    "size_fidelity_p99": ("size_fidelity", "p99", "le"),
    "elem_neighbours_min": ("elem_neighbours_min", None, "ge"),
    "elems_per_node_max": ("elems_per_node_max", None, "le"),
    "tri_cluster_max": ("tri_cluster_max", None, "le"),
    "boundary_edges_max": ("boundary", "edges_per_tri_max", "le"),      # statistic set by boundary_rule
    "pinch_nodes_max": ("boundary", "pinch_nodes", "le"),
    "tri_share_max": ("tri_share", None, "le"),
    "euler_ok": ("euler_ok", None, "eq"),
}
_OPTIONS = {"boundary_rule"}
_BOUNDARY_STAT = {"tri": "edges_per_tri_max", "all": "edges_per_elem_max"}


def gate_family(gate: str) -> str:
    """The gate name without its region suffix: the requirement it checks."""
    for reg in ("_coast3", "_coast", "_interior", "_strait"):
        if gate.endswith(reg):
            return gate[: -len(reg)]
    return gate


def check_gates(summary: dict, gates: dict) -> list[dict]:
    out = []
    rule = gates.get("boundary_rule", "tri")
    for name, limit in gates.items():
        if name in _OPTIONS or (name == "boundary_edges_max" and rule == "off"):
            continue
        metric, stat, op = _GATES[name]
        if name == "boundary_edges_max":
            stat = _BOUNDARY_STAT[rule]
        for reg in REGIONS if stat else ("",):
            if metric + reg not in summary:
                continue
            v = summary[metric + reg] if stat is None else summary[metric + reg][stat]
            if name == "tri_max_angle_target_deg" and summary["n_quads"]:
                v = None
            ok = v is None or (v <= limit if op == "le" else v >= limit if op == "ge" else v == limit)
            out.append({"gate": name + reg, "value": v, "limit": limit, "op": op, "passed": bool(ok)})
    return out
