"""Local cavity remeshing of defect clusters: "select a zone and remesh" (SMS style), automated from the defect maps.

repair(mesh, h, domain, opts, is_block=None) -> (Mesh, info)                    the automatic repair (below)
repair_region(mesh, h, domain, polygon_lonlat, opts, is_block=None) -> (Mesh, info)   one zone by hand (at the end)
region_summary(mesh, h, polygon_lonlat, gates=None) -> dict      the zone's counts and quality (compete._score terms)

1. Detect. Defect elements: elements touching an irregular interior node (valence != 4, or != 6 at a node whose
   elements are all triangles), quads with a corner above `bad_quad_deg` (135), triangles of a quad-dominant mesh,
   triangles failing a hard angle gate, elements on either side of an edge with |ln s/h| > `bad_edge` (0.15).
   Elements touching a coast node (the coast row, the metrics' `_coast` subset) are never seeds and never part of a
   cavity: coast nodes are never moved or removed and the coastal size fidelity stays as it was. Triangles of a
   declared triangle zone (Mesh.tri_zone: intentional) are neither defects nor part of a cluster or a cavity; the
   returned mesh carries the zone mask (decisions R23). Frozen cells (`frozen`: m14_blocks_rep's fixed blocks, R24)
   are held the same way, so their nodes never move (a cavity's loop nodes are fixed). Clusters: connected
   components (edge adjacency) of the defect set dilated by one element, cut into windows of at most `window`
   elements; ranked by the excess score
   w_irr (gross - |net|) + w_elem (bad elements) + w_edge (bad edges), then by position (z, y, x of the centroid);
   windows with score <= 0 are left alone.
2. Cavity: the cluster plus `rings` node rings of elements (block elements beyond the first `block_rows` rows from a
   block outline are never taken), closed (pinch nodes get their whole fan, holes are filled), so that it is one
   disc (one outer loop) plus possibly islands. Its boundary loops are exact: nodes fixed and shared with the rest.
3. Index budget. A cavity refill keeps the cavity's Euler characteristic, so with c(v) = 4 - deg(v) at interior
   nodes, c(v) = 2 - elements(v) at coast nodes and +1 per triangle, the net charge
        net = sum over the closure nodes (cavity interior + its loop) of c(v) + (triangles in the cavity)
   cannot change (a refill only moves charge between the interior and the loop nodes, whose outside edges are fixed).
   The gross charge (sum |deg - regular| + triangles of a quad-dominant mesh) is what a fill can lower; it is
   bounded below by |net|. A 3-5 pair has net 0 but carries a Burgers vector (one lattice row): with a fixed regular
   loop it can be moved, not removed; only pairs whose Burgers vectors cancel (quadrupoles, zigzags) can go.
4. Candidates, in a fixed order, each a patch of the cavity with its loops fixed:
   "ops"     the existing patch with local topology operations (quad edge rotation, triangle-quad pentagon
             re-split, triangle pair merge or flip, quad diagonal collapse, doublet removal), greedy while the
             gross charge falls, then size + angle smoothing with the loop fixed (no operation = plain smoothing);
   "coons"   one loop with four corners (turning > `corner_deg`): a transfinite (Coons) quad grid when opposite
             sides match; when one pair of sides differs by k edges, a strip whose rows change count by one per
             transition triangle, the k transitions spread over the middle rows (option `strip`);
   "m02"     m02's fill (L-inf frontal packing along a cross field of the cavity, Delaunay, recombination, smoothing)
             on the cavity alone, its own triangulated elements as the background mesh;
   "m02_tri" the same with the asterisk field (n = 6), triangles only.
   Each patch is measured on the cavity plus its first ring (valences complete at every closure node): gross / net
   charge, bad elements, bad edges (|ln s/h| and element size ratio), hard-gate counts. Accept the best candidate
   (lowest badness, then the order above) only if it strictly lowers the badness, inverts nothing, adds no hard-gate
   violation of any kind (count per gate not above the old one), adds no irregular spot and no bad element (gross
   charge and bad-element count not above the old ones), no size jump next to the coast row, keeps the whole mesh's
   count of size jumps (element size ratio > 1.15) within the p99 gate's slack (`ratio_guard` "budget": below 0.9 % of
   the interior edges when the input passes, no growth when it fails) and keeps the patch topology (boundary = the
   loops, manifold, same Euler characteristic), so the global Euler sum stays.
5. Schedule: in each pass the windows are taken in rank order in waves (independent sets: no two cavities of a wave
   share a first-ring element), a wave's cavities evaluated in parallel (`workers`); a cavity whose first ring meets
   one accepted in this pass waits for the next pass, rejected ones block nothing; the pass applies the accepted
   patches, the next re-detects. A failed window is retried only when one of its nodes changed. Ring stages `rings`, 2x, 4x up to
   `rings_max` (2, 4, 8): every window with a positive score is tried again with the larger cavity (a window that a
   2-ring patch improved can still gain from more room); `grow_failed_only` limits the larger stages to windows that
   failed with fewer rings (faster, less gain: m02 disc s1 |ln s/h| p99 0.170 instead of 0.148).
   `max_rounds` rounds over the whole mesh; stop when a round accepts nothing.

All measures are taken on the sphere (gnomonic tangent planes, great-circle lengths). Node ids stay stable during a
repair (new nodes appended, removed ones dropped at the end), coast nodes keep their ids and come first.
"""
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from meshom.domain import Domain, loop_signed_area
from meshom.geo import R_EARTH_M, arc_length, tangent_basis
from meshom.mesh import Mesh, build_topology
from meshom.quality.kernel import element_geometry

KINDS = ("ops", "coons", "m02", "m02_tri")     # every candidate kind

DEFAULTS = dict(
    rings=2,                # first ring stage: node rings of elements around the cluster
    rings_max=8,            # ring stages rings, 2 rings, 4 rings ... up to this (2, 4, 8)
    max_rounds=3,           # rounds over the whole mesh; stop when a round accepts nothing
    candidates=("ops", "coons", "m02"),   # tried in this order (ties go to the earlier one); "m02_tri" optional
    gross_slack=0,          # an accepted patch may raise the gross charge by at most this (0: never adds spots)
    ratio_guard="budget",   # edges with element size ratio > bad_ratio: "count" = never more in a cavity; "budget" =
                            # the whole mesh keeps below ratio_share of its interior edges (the p99 gate's slack, used
                            # only when the input passes; else no growth), coast-adjacent ones never grow
    ratio_share=0.009,      # budget share (the gate is p99, i.e. 1 %; 10 % margin)
    grow_failed_only=False,  # True: the 4- and 8-ring stages retry only windows that failed with fewer rings
    w_irr=2,                # badness weights (integers, so every accepted step lowers an integer: termination)
    w_elem=2,
    w_edge=1,
    bad_quad_deg=135.0,     # a quad above this corner angle is a bad element (the p99.9 target gate)
    bad_edge=0.15,          # an edge with |ln s/h| above this is a bad edge (the size-fidelity target gate)
    bad_ratio=1.15,         # an interior edge with an element size ratio above this is a bad edge (target gate)
    hard=dict(quad_max_deg=150.0, quad_sj=0.3, tri_max_deg=120.0, tri_min_deg=20.0, ratio_max=2.0,
              elems_per_node=32, neighbours=2),   # hard gates (config.DEFAULT_GATES): per-gate count never grows
    block_rows=2,           # block elements a cavity may take: rows 0 .. block_rows-1 from the block outline
    window=100,             # clusters larger than this (elements) are cut into windows of about this size
    convexify=2,            # sweeps adding the element fans of concave loop nodes (turning right > corner_deg)
    corner_deg=45.0,        # a loop node turning left by more than this is a corner of the Coons patch
    strip=True,             # Coons with one pair of sides unequal: strip with transition triangles
    smooth_size=20,         # "size" sweeps of a patch (loop fixed), then "angle" sweeps
    smooth_angle=3,
    ops_max=100,            # most topology operations per cavity
    lookahead=40,           # zero-cost operations tried as the first step of a two-step move (0: greedy only)
    max_cavity=3000,        # larger cavities are skipped
    workers=0,              # processes evaluating the cavities of a wave (0: the CPUs available, at most 16); the
                            # result does not depend on it
    m02_max=800,            # the m02 fills only for cavities up to this many elements (cost)
    coast_row=False,        # coast-row elements may seed and join cavities (coast nodes stay fixed loop nodes, only
                            # their element counts change: a dislocation next to the coast may end on it; B65):
                            # False = never; "after" = one more repair after the usual one, seeded only by defects in
                            # or next to the coast row; "all" = in every cavity of the usual repair
    coast_rings=6,          # coast_row "after": its seeds are the defects within this many element rings of the
                            # coast row (B65: 1 / 3 / 6 rings -> 9 / 11 / 12 of the 15 seam lakes for m14_blocks_rep)
    verbose=False,
)


# ------------------------------------------------------------------------------------------------ small geometry

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _poly_geom(p: np.ndarray) -> float | None:
    """One small element (k,3): None when its vector area along the mean normal is <= 0 (inverted); else the
    largest |corner angle - ideal| (rad; ideal 60 / 90 deg; a reflex corner counts > 90 deg)."""
    c = p.sum(axis=0)
    c = c / np.sqrt(c @ c)
    q = np.roll(p, -1, axis=0)
    cr = np.column_stack([p[:, 1] * q[:, 2] - p[:, 2] * q[:, 1], p[:, 2] * q[:, 0] - p[:, 0] * q[:, 2],
                          p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]])
    if cr.sum(axis=0) @ c <= 0:
        return None
    a = q - p
    b = np.roll(p, 1, axis=0) - p
    x = np.column_stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1], a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                         a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]]) @ c
    ang = np.mod(np.arctan2(x, np.einsum("ij,ij->i", a, b)), 2.0 * np.pi)
    return float(np.max(np.abs(ang - (np.pi / 3.0 if len(p) == 3 else np.pi / 2.0))))


def _turning_deg(x: np.ndarray) -> np.ndarray:
    """Signed turning angle (deg, > 0 = left) at each vertex of the closed loop x (k,3)."""
    prv, nxt = np.roll(x, 1, axis=0), np.roll(x, -1, axis=0)
    t_in = x - prv
    t_in -= np.einsum("ij,ij->i", t_in, x)[:, None] * x
    t_out = nxt - x
    t_out -= np.einsum("ij,ij->i", t_out, x)[:, None] * x
    s = np.einsum("ij,ij->i", np.cross(t_in, t_out), x)
    return np.degrees(np.arctan2(s, np.einsum("ij,ij->i", t_in, t_out)))


def _elements(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    t, q = mesh.n_tris, mesh.n_quads
    el = np.full((t + q, 4), -1, dtype=np.int64)
    el[:t, :3] = mesh.tris
    el[t:] = mesh.quads
    return el, np.r_[np.full(t, 3), np.full(q, 4)].astype(np.int64)


def _half_edges(el: np.ndarray, nv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directed edges (a, b, element) of padded elements."""
    m = len(el)
    rows = np.repeat(np.arange(m, dtype=np.int64), 4)
    cols = np.tile(np.arange(4, dtype=np.int64), m)
    ok = cols < nv[rows]
    rows, cols = rows[ok], cols[ok]
    return el[rows, cols], el[rows, (cols + 1) % nv[rows]], rows


def _ranges(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    tot = int(lens.sum())
    if tot == 0:
        return np.zeros(0, dtype=np.int64)
    off = np.r_[0, np.cumsum(lens)[:-1]]
    return np.repeat(starts - off, lens) + np.arange(tot)


# ------------------------------------------------------------------------------------------------ measures

def _measure(P: np.ndarray, el: np.ndarray, nv: np.ndarray, is_c: np.ndarray, K: int, coast: np.ndarray, h,
             o: dict, quad_dom: bool) -> dict:
    """Measures of a local configuration: nodes 0..K-1 are the closure nodes (counted), elements with is_c are the
    cavity (their shapes, edges and neighbours counted), the others its first ring (fixed, complete the valences)."""
    n = len(P)
    a, b, e = _half_edges(el, nv)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    uk, inv, cnt = np.unique(lo * n + hi, return_inverse=True, return_counts=True)
    eu = np.column_stack([uk // n, uk % n])
    deg = np.bincount(eu.ravel(), minlength=n)
    nq = np.bincount(el[nv == 4].ravel(), minlength=n)
    nt = np.bincount(el[nv == 3, :3].ravel(), minlength=n)
    cl = np.arange(K)
    cst = coast[:K]
    inter = cl[~cst]
    d = deg[inter]
    reg = np.where(nq[inter] == 0, 6, 4)
    tc = int(np.sum(is_c & (nv == 3)))
    gross = int(np.abs(d - reg).sum()) + (tc if quad_dom else 0)
    net = int((4 - d).sum() + (2 - (nq + nt)[cl[cst]]).sum()) + tc
    # shapes of the cavity elements
    ce, cn = el[is_c], nv[is_c]
    tris, quads = ce[cn == 3, :3], ce[cn == 4]
    g = element_geometry(Mesh(P, tris, quads, np.zeros(n, dtype=np.int8)))
    qa, ta = np.degrees(g.quad_angles), np.degrees(g.tri_angles)
    qmax = qa.max(axis=1, initial=0.0)
    qsj = np.sin(g.quad_angles).min(axis=1, initial=1.0)
    tmax, tmin = ta.max(axis=1, initial=0.0), ta.min(axis=1, initial=180.0)
    hd = o["hard"]
    inverted = int(np.sum(g.tri_area <= 0) + np.sum((g.quad_area <= 0) | (qsj <= 0)))
    bad_elem = int(np.sum(qmax > o["bad_quad_deg"]) + np.sum((tmax > hd["tri_max_deg"]) | (tmin < hd["tri_min_deg"])))
    # edges of the cavity elements: size fidelity; interior edges with a cavity element: element size ratio
    cav_edge = np.zeros(len(uk), dtype=bool)
    cav_edge[inv[is_c[e]]] = True
    p0, p1 = P[eu[:, 0]], P[eu[:, 1]]
    length = R_EARTH_M * arc_length(p0, p1)
    ce_i = np.flatnonzero(cav_edge)
    lnsh = np.log(length[ce_i] / h(_unit(p0[ce_i] + p1[ce_i])))
    he_len = length[inv]
    mean_edge = np.bincount(e, he_len, minlength=len(el)) / nv
    order = np.argsort(inv, kind="stable")
    si = inv[order]
    first = np.flatnonzero(np.r_[True, si[1:] != si[:-1]])
    two = cnt[si[first]] == 2
    ea, eb = e[order[first[two]]], e[order[first[two] + 1]]
    pair_c = is_c[ea] | is_c[eb]
    ma, mb = mean_edge[ea[pair_c]], mean_edge[eb[pair_c]]
    ratio = np.maximum(ma, mb) / np.minimum(ma, mb)
    nbrs = np.bincount(np.r_[ea, eb], minlength=len(el))
    bad_edge = int(np.sum(np.abs(lnsh) > o["bad_edge"]))
    bad_ratio = int(np.sum(ratio > o["bad_ratio"]))
    ecst = np.any((el >= 0) & coast[np.maximum(el, 0)], axis=1)
    pc = (ecst[ea] | ecst[eb])[pair_c]
    bad_ratio_coast = int(np.sum((ratio > o["bad_ratio"]) & pc))
    hard = {"quad_max": int(np.sum(qmax > hd["quad_max_deg"])), "quad_sj": int(np.sum(qsj < hd["quad_sj"])),
            "tri_max": int(np.sum(tmax > hd["tri_max_deg"])), "tri_min": int(np.sum(tmin < hd["tri_min_deg"])),
            "ratio_max": int(np.sum(ratio > hd["ratio_max"])),
            "neighbours": int(np.sum(nbrs[is_c] < hd["neighbours"])),
            "elems_per_node": int(np.sum((nq + nt)[:K] > hd["elems_per_node"])),
            "nonmanifold": int(np.sum(cnt > 2))}
    badness = o["w_irr"] * gross + o["w_elem"] * bad_elem + o["w_edge"] * (bad_edge + bad_ratio)
    return {"gross": gross, "net": net, "n3": int(np.sum(d == 3)), "n5": int(np.sum(d == 5)),
            "n_irr": int(np.sum(d != reg)), "T": tc, "bad_elem": bad_elem, "bad_edge": bad_edge,
            "bad_ratio": bad_ratio, "bad_ratio_coast": bad_ratio_coast, "inverted": inverted, "hard": hard, "badness": int(badness)}


def summary(mesh: Mesh, h, o: dict | None = None) -> dict:
    """Whole-mesh counts the repair reports before and after: n3 / n5 / n_other / T (valence 4 baseline, as the
    competition), gross charge (regular valence 6 at all-triangle nodes) and net charge (= the Euler sum), bad quads,
    bad edges, |ln s/h| p99, element size ratio p99, quad max angle."""
    o = {**DEFAULTS, **(o or {})}
    topo = build_topology(mesh)
    deg = np.bincount(topo.edges.ravel(), minlength=mesh.n_nodes)
    bnd = topo.boundary_node
    used = topo.elems_per_node > 0
    inter = ~bnd & used
    nq = np.bincount(mesh.quads.ravel(), minlength=mesh.n_nodes)
    d = deg[inter]
    reg = np.where(nq[inter] == 0, 6, 4)
    quad_dom = mesh.n_quads >= mesh.n_tris
    g = element_geometry(mesh)
    qmax = np.degrees(g.quad_angles).max(axis=1, initial=0.0)
    p0, p1 = mesh.points[topo.edges[:, 0]], mesh.points[topo.edges[:, 1]]
    lnsh = np.abs(np.log(R_EARTH_M * arc_length(p0, p1) / h(_unit(p0 + p1))))
    ee = topo.edge_elems
    inner = ee[:, 1] >= 0
    me = np.r_[g.tri_edges.mean(axis=1), g.quad_edges.mean(axis=1)]
    a, b = me[ee[inner, 0]], me[ee[inner, 1]]
    ratio = np.maximum(a, b) / np.minimum(a, b)
    return {"n_nodes": int(used.sum()), "n_tris": mesh.n_tris, "n_quads": mesh.n_quads,
            "n3": int(np.sum(d == 3)), "n5": int(np.sum(d == 5)), "n_other": int(np.sum((d < 3) | (d > 5))),
            "T": mesh.n_tris, "n_irr": int(np.sum(d != reg)),
            "gross": int(np.abs(d - reg).sum()) + (mesh.n_tris if quad_dom else 0),
            "net": int((4 - d).sum() + (2 - topo.elems_per_node[bnd & used]).sum()) + mesh.n_tris,
            "bad_quads": int(np.sum(qmax > o["bad_quad_deg"])),
            "quad_max_deg": float(qmax.max()) if len(qmax) else None,
            "bad_edges": int(np.sum(lnsh > o["bad_edge"])), "lnsh_p99": float(np.percentile(lnsh, 99)),
            "bad_ratio_edges": int(np.sum(ratio > o["bad_ratio"])), "n_interior_edges": int(len(ratio)),
            "elem_ratio_p99": float(np.percentile(ratio, 99)) if len(ratio) else None}


# ------------------------------------------------------------------------------------------------ the working mesh

def _deep_blocks(el: np.ndarray, blk: np.ndarray, n: int, rows: int) -> np.ndarray:
    """Block elements beyond the first `rows` node rings from the block outline (row 0 = block elements sharing a
    node with a non-block element)."""
    if not blk.any():
        return np.zeros(len(el), dtype=bool)
    ok = el >= 0
    fill_node = np.zeros(n, dtype=bool)
    fill_node[el[~blk][ok[~blk]]] = True
    front = blk & np.any(ok & fill_node[np.maximum(el, 0)], axis=1)
    near = front.copy() if rows > 0 else np.zeros(len(el), dtype=bool)
    for _ in range(max(0, rows - 1)):
        nn = np.zeros(n, dtype=bool)
        nn[el[front][ok[front]]] = True
        front = blk & ~near & np.any(ok & nn[np.maximum(el, 0)], axis=1)
        near |= front
    return blk & ~near


class _Glob:
    """Global topology and defect maps of the current mesh (rebuilt once per pass)."""

    def __init__(self, mesh: Mesh, blk: np.ndarray, deep: np.ndarray | None, h, o: dict, quad_dom: bool,
                 frozen: np.ndarray | None = None):
        self.mesh, self.blk, self.o, self.quad_dom = mesh, blk, o, quad_dom
        n = mesh.n_nodes
        self.loc = np.full(n, -1, dtype=np.int64)          # scratch: global -> cavity-local node id
        self.el, self.nv = _elements(mesh)
        self.M = len(self.el)
        self.zone = np.zeros(self.M, dtype=bool)          # declared triangle zone (R23): never a defect or in a cavity
        if mesh.tri_zone is not None:
            self.zone[: mesh.n_tris] = np.asarray(mesh.tri_zone, dtype=bool)
        # held elements: the zone and the frozen cells (R24: fixed blocks; never a defect, in a cluster or a cavity)
        self.hold = self.zone if frozen is None else self.zone | np.asarray(frozen, dtype=bool)
        flat = self.el.ravel()
        ok = flat >= 0
        nodes = flat[ok]
        elems = np.repeat(np.arange(self.M, dtype=np.int64), 4)[ok]
        order = np.argsort(nodes, kind="stable")
        self.ptr = np.zeros(n + 1, dtype=np.int64)
        self.ptr[1:] = np.cumsum(np.bincount(nodes, minlength=n))
        self.inc = elems[order]
        topo = build_topology(mesh)
        self.topo = topo
        self.coast = topo.boundary_node
        deg = np.bincount(topo.edges.ravel(), minlength=n)
        nq = np.bincount(mesh.quads.ravel(), minlength=n)
        used = topo.elems_per_node > 0
        reg = np.where(nq == 0, 6, 4)
        self.irr = np.where(~self.coast & used, np.abs(deg - reg), 0)
        self.charge = np.where(self.coast, 2 - topo.elems_per_node, 4 - deg)
        ee = topo.edge_elems
        inner = ee[:, 1] >= 0
        a, b = ee[inner, 0], ee[inner, 1]
        self.adj = coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(self.M, self.M)).tocsr()
        # coast row: elements touching a coast node (the metrics' `_coast` subset); never seeds, never in a cavity
        self.coast_row = np.any((self.el >= 0) & self.coast[np.maximum(self.el, 0)], axis=1)
        # defects
        g = element_geometry(mesh)
        hd = o["hard"]
        qmax = np.degrees(g.quad_angles).max(axis=1, initial=0.0)
        ta = np.degrees(g.tri_angles)
        tbad = (ta.max(axis=1, initial=0.0) > hd["tri_max_deg"]) | (ta.min(axis=1, initial=180.0) < hd["tri_min_deg"])
        self.bad_elem = np.r_[tbad, qmax > o["bad_quad_deg"]]
        self.is_tri = self.nv == 3
        p0, p1 = mesh.points[topo.edges[:, 0]], mesh.points[topo.edges[:, 1]]
        lnsh = np.abs(np.log(R_EARTH_M * arc_length(p0, p1) / h(_unit(p0 + p1))))
        bad_e = lnsh > o["bad_edge"]
        self.bad_edge_elem = np.zeros(self.M, dtype=np.int64)
        np.add.at(self.bad_edge_elem, ee[bad_e, 0], 1)
        be1 = ee[bad_e, 1]
        np.add.at(self.bad_edge_elem, be1[be1 >= 0], 1)
        node_bad = self.irr > 0
        touch = np.zeros(self.M, dtype=bool)
        for k in range(4):
            col = self.el[:, k]
            touch |= (col >= 0) & node_bad[np.maximum(col, 0)]
        self.defect = (touch | self.bad_elem | (self.bad_edge_elem > 0) | (self.is_tri & quad_dom)) & ~self.hold
        # coast_row (B65): the coast row may seed and join cavities; coast nodes stay loop nodes (fixed), so only
        # their element counts change (a dislocation next to the coast can end on it)
        stage = bool(o.get("_coast_stage", False))
        cr = np.zeros(self.M, dtype=bool) if stage or o.get("coast_row") == "all" else self.coast_row
        self.defect &= ~cr
        if stage:                                       # the coast stage: only defects within coast_rings element
            near = self.coast_row.copy()                # rings of the coast row
            for _ in range(int(o["coast_rings"])):
                near |= self.adj @ near.astype(np.int64) > 0
            self.defect &= near
        # cavities never take block elements beyond the first block_rows rows of the original outline
        self.allowed = (~deep if deep is not None else np.ones(self.M, dtype=bool)) & ~cr & ~self.hold

    def elems_at(self, nodes: np.ndarray) -> np.ndarray:
        return self.inc[_ranges(self.ptr[nodes], self.ptr[nodes + 1] - self.ptr[nodes])]

    def nodes_of(self, elems: np.ndarray) -> np.ndarray:
        x = self.el[elems].ravel()
        return np.unique(x[x >= 0])

    def clusters(self) -> list[np.ndarray]:
        """Edge-connected components of the defect set dilated by one element (edge adjacency), cut into windows of
        at most `window` elements (breadth-first balls grown from the worst defect element left), ranked."""
        d = self.defect
        dil = (d | (self.adj @ d.astype(np.float64) > 0)) & ~self.hold
        ids = np.flatnonzero(dil)
        if len(ids) == 0:
            return []
        sub = self.adj[ids][:, ids].tocsr()
        nc, lab = connected_components(sub, directed=False)
        order = np.argsort(lab, kind="stable")
        cuts = np.flatnonzero(np.diff(lab[order])) + 1
        out = []
        o = self.o
        cen = self.mesh.points[np.maximum(self.el, 0)].sum(axis=1)
        wt = (self.irr[np.maximum(self.el, 0)] * (self.el >= 0)).sum(axis=1) + self.bad_elem + self.is_tri
        pos = {int(e): i for i, e in enumerate(ids)}
        pieces = []
        for grp in np.split(ids[order], cuts):
            if len(grp) <= o["window"]:
                pieces.append(grp)
                continue
            left = set(grp.tolist())
            seeds = sorted(grp.tolist(), key=lambda e: (-int(wt[e]), e))
            for s0 in seeds:
                if s0 not in left or not d[s0]:
                    continue
                ball, front = [s0], [s0]
                left.discard(s0)
                while front and len(ball) < o["window"]:
                    nxt = []
                    for x in front:
                        i = pos[x]
                        for y in ids[sub.indices[sub.indptr[i]:sub.indptr[i + 1]]].tolist():
                            if y in left and len(ball) < o["window"]:
                                left.discard(y)
                                ball.append(y)
                                nxt.append(y)
                    front = nxt
                pieces.append(np.array(sorted(ball), dtype=np.int64))
        for grp in pieces:
            nodes = self.nodes_of(grp)
            gross = int(self.irr[nodes].sum()) + (int(self.is_tri[grp].sum()) if self.quad_dom else 0)
            net = int(self.charge[nodes].sum()) + int(self.is_tri[grp].sum())
            score = (o["w_irr"] * (gross - abs(net)) + o["w_elem"] * int(self.bad_elem[grp].sum())
                     + o["w_edge"] * int(self.bad_edge_elem[grp].sum()))
            if score <= 0:
                continue
            c = _unit(cen[grp].sum(axis=0))
            out.append((-score, round(float(c[2]), 9), round(float(c[1]), 9), round(float(c[0]), 9), grp))
        out.sort(key=lambda t: t[:4])
        return [t[4] for t in out]


class _Cav:
    """A cavity: global elements C, first ring R1, local nodes (loops first, then the interior closure nodes, then
    the ring-only nodes), loops as local id arrays (cavity on the left)."""

    def __init__(self, G: _Glob, C: np.ndarray, loops_g: list[np.ndarray]):
        self.C = C
        cl_nodes = G.nodes_of(C)
        ring = G.elems_at(cl_nodes)
        self.R1 = np.setdiff1d(ring, C)
        loop_nodes = np.concatenate(loops_g)
        interior = np.setdiff1d(cl_nodes, loop_nodes)
        ring_only = np.setdiff1d(G.nodes_of(self.R1), cl_nodes)
        self.g = np.r_[loop_nodes, interior, ring_only].astype(np.int64)
        self.n_loop, self.n_cl = len(loop_nodes), len(loop_nodes) + len(interior)
        G.loc[self.g] = np.arange(len(self.g))
        self.loops = []
        k = 0
        for lp in loops_g:
            self.loops.append(np.arange(k, k + len(lp), dtype=np.int64))
            k += len(lp)
        self.pts = G.mesh.points[self.g]
        self.coast = G.coast[self.g]

        def local(elems):
            el = G.el[elems].copy()
            m = el >= 0
            el[m] = G.loc[el[m]]
            return el, G.nv[elems].copy()

        self.cel, self.cnv = local(C)
        self.rel, self.rnv = local(self.R1)
        G.loc[self.g] = -1
        self.loop_he = set()
        for lp in self.loops:
            for i in range(len(lp)):
                self.loop_he.add((int(lp[i]), int(lp[(i + 1) % len(lp)])))
        self.chi = self._chi(self.cel, self.cnv, self.n_cl)

    @staticmethod
    def _chi(el, nv, nn):
        a, b, _ = _half_edges(el, nv)
        ne = len(np.unique(np.minimum(a, b) * (nn + 1) + np.maximum(a, b)))
        return nn - ne + len(el)

    def measure(self, pts_cl: np.ndarray, cel: np.ndarray, cnv: np.ndarray, h, o: dict, quad_dom: bool) -> dict:
        K = len(pts_cl)
        rel = self.rel.copy()
        m = rel >= self.n_cl
        rel[m] = rel[m] - self.n_cl + K
        P = np.vstack([pts_cl, self.pts[self.n_cl:]])
        el = np.vstack([cel, rel])
        nv = np.r_[cnv, self.rnv]
        is_c = np.r_[np.ones(len(cel), dtype=bool), np.zeros(len(rel), dtype=bool)]
        coast = np.r_[np.zeros(K, dtype=bool), self.coast[self.n_cl:]]
        coast[: self.n_loop] = self.coast[: self.n_loop]
        return _measure(P, el, nv, is_c, K, coast, h, o, quad_dom)

    def valid(self, K: int, cel: np.ndarray, cnv: np.ndarray) -> bool:
        """Patch topology: every node used, consistent orientation, boundary half-edges = the loops, same chi."""
        if len(cel) == 0:
            return False
        a, b, _ = _half_edges(cel, cnv)
        if np.bincount(np.r_[a, b], minlength=K).min() == 0 or a.max() >= K:
            return False
        key = a * K + b
        if len(np.unique(key)) != len(key):
            return False
        bnd = ~np.isin(b * K + a, key)
        he = set(zip(a[bnd].tolist(), b[bnd].tolist()))
        if he != self.loop_he:
            return False
        return self._chi(cel, cnv, K) == self.chi


def _build_cavity(G: _Glob, core: np.ndarray, rings: int, o: dict):
    """Cluster + rings node rings, closed (pinches, holes); None when it is not one disc (+ islands) or too big."""
    C = np.unique(core)
    for _ in range(rings):
        E = G.elems_at(G.nodes_of(C))
        C = np.union1d(C, E[G.allowed[E]])
        if len(C) > o["max_cavity"]:
            return None, "too_big"
    conv = 0
    for _ in range(8 + int(o["convexify"])):
        if len(C) > o["max_cavity"]:
            return None, "too_big"
        a, b, _ = _half_edges(G.el[C], G.nv[C])
        n = G.mesh.n_nodes
        key = a * n + b
        bnd = ~np.isin(b * n + a, key)
        ba, bb = a[bnd], b[bnd]
        outd = np.bincount(ba, minlength=n)
        pinch = np.flatnonzero(outd > 1)
        if len(pinch):
            C = np.union1d(C, G.elems_at(pinch))
            continue
        nxt = dict(zip(ba.tolist(), bb.tolist()))
        loops, seen = [], set()
        for s in sorted(nxt):
            if s in seen:
                continue
            lp, v = [], s
            while v not in seen:
                seen.add(v)
                lp.append(v)
                v = nxt[v]
            if v != s:
                return None, "loop"
            loops.append(np.array(lp, dtype=np.int64))
        areas = [loop_signed_area(G.mesh.points[lp]) for lp in loops]
        if conv < o["convexify"]:
            conv += 1
            cc = []
            for lp, ar in zip(loops, areas):
                if ar > 0:
                    t = _turning_deg(G.mesh.points[lp])
                    cc += lp[(t < -o["corner_deg"]) & ~G.coast[lp]].tolist()
            if cc:
                E = G.elems_at(np.array(sorted(cc), dtype=np.int64))
                E = E[G.allowed[E]]
                if len(np.setdiff1d(E, C)):
                    C = np.union1d(C, E)
                    continue
        holes = []
        for lp, ar in zip(loops, areas):
            if ar < 0 and not G.coast[lp].all():
                holes.append(lp)
            elif ar < 0:
                # an island loop: every edge must be a coast edge
                pass
        if holes:
            add = _fill_holes(G, C, holes, o)
            if add is None:
                return None, "hole"
            C = np.union1d(C, add)
            continue
        if sum(ar > 0 for ar in areas) != 1:
            return None, "loops"
        if not G.allowed[C].all():                      # closing took a coast-row or deep block element
            return None, "coast_or_block"
        outer = [lp for lp, ar in zip(loops, areas) if ar > 0]
        inner = [lp for lp, ar in zip(loops, areas) if ar <= 0]
        return (C, outer + inner), "ok"
    return None, "close"


def _fill_holes(G: _Glob, C: np.ndarray, holes: list[np.ndarray], o: dict):
    """Elements enclosed by C behind the hole loops (BFS over non-cavity elements, capped)."""
    inC = set(C.tolist())
    n = G.mesh.n_nodes
    topo = G.topo
    ekey = topo.edges[:, 0] * n + topo.edges[:, 1]
    sorter = np.argsort(ekey)
    add = set()
    cap = 4 * len(C) + 100
    for lp in holes:
        a, b = lp, np.roll(lp, -1)
        k = np.minimum(a, b) * n + np.maximum(a, b)
        idx = sorter[np.searchsorted(ekey, k, sorter=sorter)]
        start = [int(x) for x in topo.edge_elems[idx].ravel() if x >= 0 and int(x) not in inC]
        stack, seen = list(start), set(start)
        while stack:
            if len(seen) > cap:
                return None
            x = stack.pop()
            for y in G.adj.indices[G.adj.indptr[x]:G.adj.indptr[x + 1]].tolist():
                if y not in inC and y not in seen:
                    seen.add(y)
                    stack.append(y)
        add |= seen
    return np.array(sorted(add), dtype=np.int64)


# ------------------------------------------------------------------------------------------------ candidate: ops

class _Ops:
    """Topology operations on the cavity elements (local closure ids 0..K-1, loop nodes 0..n_loop-1 fixed)."""

    def __init__(self, cav: _Cav, quad_dom: bool, o: dict):
        self.K, self.nl = cav.n_cl, cav.n_loop
        self.P = cav.pts[: cav.n_cl].copy()
        self.coast = cav.coast[: cav.n_cl]
        self.quad_dom = quad_dom
        self.o = o
        self.elems: dict[int, tuple] = {}
        self.at = [set() for _ in range(self.K)]
        self.onb = [set() for _ in range(self.K)]       # neighbours through the first-ring elements (fixed)
        self.oq = [0] * self.K                          # first-ring quads at the node
        self.oe = set()                                 # first-ring edges at closure nodes
        for i in range(len(cav.rel)):
            r = [int(x) for x in cav.rel[i, : cav.rnv[i]]]
            k = len(r)
            for j in range(k):
                a, b = r[j], r[(j + 1) % k]
                if a < self.K or b < self.K:
                    self.oe.add((min(a, b), max(a, b)))
                if a < self.K:
                    self.onb[a].add(b)
                    if k == 4:
                        self.oq[a] += 1
                if b < self.K:
                    self.onb[b].add(a)
        self.deg = [len(x) for x in self.onb]           # edges at the node (kept up to date)
        self.cq = [0] * self.K                          # cavity quads at the node
        self.ecnt: dict = {}                            # cavity elements per edge
        self.nid = 0
        for i in range(len(cav.cel)):
            self._add(tuple(int(x) for x in cav.cel[i, : cav.cnv[i]]))
        self.n_ops = 0

    def _add(self, t: tuple) -> int:
        i = self.nid
        self.nid += 1
        self.elems[i] = t
        k = len(t)
        for j in range(k):
            a, b = t[j], t[(j + 1) % k]
            self.at[a].add(i)
            if k == 4:
                self.cq[a] += 1
            e = (a, b) if a < b else (b, a)
            c = self.ecnt.get(e, 0)
            self.ecnt[e] = c + 1
            if c == 0 and e not in self.oe:
                self.deg[a] += 1
                self.deg[b] += 1
        return i

    def _rm(self, i: int) -> tuple:
        t = self.elems.pop(i)
        k = len(t)
        for j in range(k):
            a, b = t[j], t[(j + 1) % k]
            self.at[a].discard(i)
            if k == 4:
                self.cq[a] -= 1
            e = (a, b) if a < b else (b, a)
            c = self.ecnt[e] - 1
            if c:
                self.ecnt[e] = c
            else:
                del self.ecnt[e]
                if e not in self.oe:
                    self.deg[a] -= 1
                    self.deg[b] -= 1
        return t

    def _restore(self, i: int, t: tuple) -> None:
        nid = self.nid
        self.nid = i
        self._add(t)
        self.nid = nid

    def nb(self, v: int) -> set:
        s = set(self.onb[v])
        for i in self.at[v]:
            t = self.elems[i]
            k = t.index(v)
            s.add(t[(k + 1) % len(t)])
            s.add(t[k - 1])
        return s

    def cost(self, v: int) -> int:
        if self.coast[v] or not self.at[v]:
            return 0
        return abs(self.deg[v] - (6 if self.oq[v] + self.cq[v] == 0 else 4))

    def delta(self, rm: list, add: list, gone: tuple = ()) -> int:
        """Change of w_irr x (gross charge) if elements rm are replaced by add (nodes in gone vanish), computed from
        the edge-count and quad-count changes without touching the mesh."""
        de: dict = {}
        dq: dict = {}
        nodes = set()
        dt = 0
        for sign, ts in ((-1, [self.elems[i] for i in rm]), (1, add)):
            for t in ts:
                k = len(t)
                dt += sign * (k == 3)
                for j in range(k):
                    a, b = t[j], t[(j + 1) % k]
                    nodes.add(a)
                    e = (a, b) if a < b else (b, a)
                    de[e] = de.get(e, 0) + sign
                    if k == 4:
                        dq[a] = dq.get(a, 0) + sign
        ddeg: dict = {}
        for e, d in de.items():
            if d == 0 or e in self.oe:
                continue
            c = self.ecnt.get(e, 0)
            if (c > 0) != (c + d > 0):
                s = 1 if c + d > 0 else -1
                ddeg[e[0]] = ddeg.get(e[0], 0) + s
                ddeg[e[1]] = ddeg.get(e[1], 0) + s
        before = after = 0
        for v in nodes:
            if self.coast[v]:
                continue
            before += self.cost(v)
            if v in gone:
                continue
            nq = self.oq[v] + self.cq[v] + dq.get(v, 0)
            after += abs(self.deg[v] + ddeg.get(v, 0) - (6 if nq == 0 else 4))
        return self.o["w_irr"] * ((after - before) + (dt if self.quad_dom else 0))

    def _geom(self, add: list, P: np.ndarray) -> float | None:
        """None if a new element is inverted at the current positions; else the worst angle deviation."""
        worst = 0.0
        for t in add:
            g = _poly_geom(P[list(t)])
            if g is None:
                return None
            worst = max(worst, g)
        return worst

    @staticmethod
    def _merge(t1: tuple, t2: tuple, u: int, v: int) -> list:
        """CCW polygon of two elements sharing edge u -> v (in t1) / v -> u (in t2)."""
        k1, k2 = t1.index(v), t2.index(u)
        r1 = t1[k1:] + t1[:k1]          # v ... u
        r2 = t2[k2:] + t2[:k2]          # u ... v
        return list(r1[:-1] + r2[:-1])

    def candidates(self, near: set | None = None):
        """(rm ids, new elements, removed node or None, moved (node, pos) or None, sort key); near: only operations
        on elements with a node in this set."""
        if near is None:
            ids = sorted(self.elems)
            bad = {v for v in range(self.K) if self.cost(v) > 0}
        else:
            ids = sorted({i for v in near for i in self.at[v]})
            nodes = {v for i in ids for v in self.elems[i]}
            nodes |= {v for i in ids for x in self.elems[i] for j in self.at[x] for v in self.elems[j]}
            bad = {v for v in nodes if self.cost(v) > 0}
        tri = {i for i, t in self.elems.items() if len(t) == 3}
        seen = set()
        for i in ids:
            t1 = self.elems[i]
            for k in range(len(t1)):
                u, v = t1[k], t1[(k + 1) % len(t1)]
                js = [j for j in self.at[u] & self.at[v] if j != i]
                if len(js) != 1:
                    continue
                j = js[0]
                if (min(i, j), max(i, j)) in seen:
                    continue
                seen.add((min(i, j), max(i, j)))
                t2 = self.elems[j]
                if i not in tri and j not in tri and not (bad & (set(t1) | set(t2))):
                    continue
                if near is not None and not (near & (set(t1) | set(t2))):
                    continue
                poly = self._merge(t1, t2, u, v)
                if len(set(poly)) != len(poly):
                    continue
                m = len(poly)
                cur = {frozenset(t1), frozenset(t2)}
                outs = []
                if m == 6:
                    for s in range(3):
                        outs.append([tuple(poly[s:s + 4]), tuple(poly[s + 3:] + poly[:s + 1])])
                elif m == 5:
                    for s in range(5):
                        r = poly[s:] + poly[:s]
                        outs.append([tuple(r[:3]), tuple([r[2], r[3], r[4], r[0]])])
                elif m == 4:
                    outs.append([tuple(poly)])
                    outs.append([(poly[1], poly[2], poly[3]), (poly[3], poly[0], poly[1])])
                for new in outs:
                    if {frozenset(x) for x in new} == cur:
                        continue
                    if any(self._exists(x) for x in new):
                        continue
                    yield [i, j], new, None, None, ("e", tuple(sorted(poly)), tuple(sorted(new)))
        for i in ids:
            t = self.elems[i]
            if len(t) != 4 or not (bad & set(t)) or (near is not None and not (near & set(t))):
                continue
            for s in range(2):
                a, c = t[s], t[s + 2]
                b, d = t[s + 1], t[(s + 3) % 4]
                if a >= self.nl and c >= self.nl:
                    keep, gone = (a, c) if a < c else (c, a)
                    pos = _unit(self.P[a] + self.P[c])
                elif c >= self.nl:
                    keep, gone, pos = a, c, None
                elif a >= self.nl:
                    keep, gone, pos = c, a, None
                else:
                    continue
                if self.coast[gone]:
                    continue
                na, nc = self.nb(keep), self.nb(gone)
                if gone in na or (na & nc) != {b, d}:
                    continue
                rm, new = [i], []
                for j in sorted(self.at[gone]):
                    if j == i:
                        continue
                    tj = self.elems[j]
                    rm.append(j)
                    new.append(tuple(keep if x == gone else x for x in tj))
                yield rm, new, gone, (keep, pos), ("c", t, s)
        for v in (range(self.nl, self.K) if near is None else sorted(near)):
            if v < self.nl or len(self.at[v]) != 2 or self.coast[v]:
                continue
            i, j = sorted(self.at[v])
            t1, t2 = self.elems[i], self.elems[j]
            if len(t1) != 4 or len(t2) != 4:
                continue
            k1 = t1.index(v)
            r1 = t1[k1:] + t1[:k1]               # v x y z
            k2 = t2.index(v)
            r2 = t2[k2:] + t2[:k2]               # v z w x
            if r2[1] != r1[3] or r2[3] != r1[1]:
                continue
            new = [(r1[1], r1[2], r1[3], r2[2])]
            if self._exists(new[0]):
                continue
            yield [i, j], new, v, None, ("d", v)

    def _exists(self, t: tuple) -> bool:
        s = frozenset(t)
        common = set.intersection(*(self.at[v] for v in t))
        return any(frozenset(self.elems[i]) == s for i in common)

    def _score(self, c) -> tuple | None:
        """(delta, worst angle deviation) of operation c, None when it degenerates or inverts an element."""
        rm, new, gone, move, key = c
        if any(len(set(t)) != len(t) for t in new):
            return None
        d = self.delta(rm, new, (gone,) if gone is not None else ())
        if d > 0:
            return None
        P = self.P
        if move is not None and move[1] is not None:
            P = self.P.copy()
            P[move[0]] = move[1]
        gq = self._geom(new, P)
        return None if gq is None else (d, round(gq, 9))

    def _apply(self, c):
        rm, new, gone, move, key = c
        saved = [(i, self._rm(i)) for i in rm]
        added = [self._add(t) for t in new]
        old = None
        if move is not None and move[1] is not None:
            old = (move[0], self.P[move[0]].copy())
            self.P[move[0]] = move[1]
        return saved, added, old

    def _undo(self, rec) -> None:
        saved, added, old = rec
        for i in added:
            self._rm(i)
        for i, t in saved:
            self._restore(i, t)
        if old is not None:
            self.P[old[0]] = old[1]

    def _best(self, near: set | None = None):
        best, zero = None, []
        for c in self.candidates(near):
            sc = self._score(c)
            if sc is None:
                continue
            if sc[0] < 0:
                k = (sc[0], sc[1], c[4])
                if best is None or k < best[0]:
                    best = (k, c)
            else:
                zero.append((sc[1], c[4], c))
        return best, zero

    def run(self) -> int:
        """Greedy: the best operation that lowers the gross charge; when there is none, a zero-cost operation
        (a glide) followed by a lowering one next to it (depth-2 lookahead, the first such pair in order)."""
        for _ in range(self.o["ops_max"]):
            best, zero = self._best()
            if best is not None:
                self._apply(best[1])
                self.n_ops += 1
                continue
            if not self.o["lookahead"] or not zero:
                break
            zero.sort(key=lambda z: (z[0], z[1]))
            done = False
            for _, _, c in zero[: self.o["lookahead"]]:
                rec = self._apply(c)
                near = {v for t in c[1] for v in t}
                b2, _ = self._best(near)
                if b2 is not None:
                    self._apply(b2[1])
                    self.n_ops += 2
                    done = True
                    break
                self._undo(rec)
            if not done:
                break
        return self.n_ops

    def patch(self):
        """(points (K',3), old local id per node, padded elements, corner counts) with the loop nodes first."""
        used = sorted({v for t in self.elems.values() for v in t})
        if used[: self.nl] != list(range(self.nl)):
            return None
        idx = {v: i for i, v in enumerate(used)}
        ids = sorted(self.elems)
        el = np.full((len(ids), 4), -1, dtype=np.int64)
        nv = np.zeros(len(ids), dtype=np.int64)
        for r, i in enumerate(ids):
            t = self.elems[i]
            el[r, : len(t)] = [idx[v] for v in t]
            nv[r] = len(t)
        return self.P[used], np.array(used, dtype=np.int64), el, nv


# ------------------------------------------------------------------------------------------------ candidate: Coons

def _coons(cav: _Cav, o: dict):
    """Transfinite quad grid (or a strip with transition triangles) of a one-loop cavity with four corners."""
    if len(cav.loops) != 1:
        return None
    lp = cav.loops[0]
    L = len(lp)
    X = cav.pts[lp]
    turn = _turning_deg(X)
    corners = np.flatnonzero(turn > o["corner_deg"])
    if len(corners) != 4 or np.any(turn < -o["corner_deg"]):
        return None
    n = [int((corners[(k + 1) % 4] - corners[k]) % L) for k in range(4)]
    if n[0] == n[2] and n[1] == n[3]:
        rot = 0
    elif n[1] == n[3]:
        rot = 0
    elif n[0] == n[2]:
        rot = 1
    else:
        return None
    if rot:
        corners = np.roll(corners, -1)
        n = n[1:] + n[:1]
    if n[0] != n[2] and not o["strip"]:
        return None
    side = [[int(lp[(corners[k] + i) % L]) for i in range(n[k] + 1)] for k in range(4)]
    bottom, right, top, left = side[0], side[1], side[2][::-1], side[3][::-1]
    m = n[1]
    c = _unit(X.mean(axis=0))
    e1, e2 = tangent_basis(c[None, :])
    e1, e2 = e1[0], e2[0]

    def xy(ids):
        p = cav.pts[ids]
        g = p / (p @ c)[:, None]
        return np.column_stack([g @ e1, g @ e2])

    def curve(ids):
        q = xy(ids)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        return q, s / s[-1]

    def at(cq, s):
        q, t = cq
        return np.column_stack([np.interp(s, t, q[:, 0]), np.interp(s, t, q[:, 1])])

    B, T, Lc, R = curve(bottom), curve(top), curve(left), curve(right)
    cnt = [n[0]] * (m + 1)
    dlt = n[0] - n[2]
    if dlt:
        if m < 2:
            return None
        pairs = sorted(range(m), key=lambda j: (abs(j - (m - 1) / 2.0), j))
        ch = np.zeros(m, dtype=np.int64)
        for q in range(abs(dlt)):
            ch[pairs[q % m]] += 1
        for j in range(m):
            cnt[j + 1] = cnt[j] - int(np.sign(dlt)) * int(ch[j])
        if min(cnt) < 1:
            return None
    rows = [list(range(len(bottom)))]          # row node lists as patch-local ids
    loop_pos = {v: i for i, v in enumerate(lp.tolist())}
    rows = [[loop_pos[v] for v in bottom]]
    new_pts = []
    nid = cav.n_loop
    p00, p10, p01, p11 = B[0][0], B[0][-1], T[0][0], T[0][-1]
    for j in range(1, m):
        tl, tr = Lc[1][j], R[1][j]
        cj = cnt[j]
        row = [loop_pos[left[j]]]
        for i in range(1, cj):
            if dlt == 0:
                v = j / m
                s = (1 - v) * B[1][i] + v * T[1][i]
            else:
                s = i / cj
            t = (1 - s) * tl + s * tr
            q = ((1 - t) * at(B, [s])[0] + t * at(T, [s])[0] + (1 - s) * at(Lc, [t])[0] + s * at(R, [t])[0]
                 - ((1 - s) * (1 - t) * p00 + s * (1 - t) * p10 + (1 - s) * t * p01 + s * t * p11))
            new_pts.append(_unit(c + q[0] * e1 + q[1] * e2))
            row.append(nid)
            nid += 1
        row.append(loop_pos[right[j]])
        rows.append(row)
    rows.append([loop_pos[v] for v in top])
    els = []
    for j in range(m):
        A, Bn = rows[j], rows[j + 1]
        ca, cb = len(A) - 1, len(Bn) - 1
        if ca == cb:
            els += [(A[i], A[i + 1], Bn[i + 1], Bn[i]) for i in range(ca)]
        elif cb < ca:
            k = ca - cb
            cols = set(np.unique(np.floor((np.arange(k) + 0.5) * ca / k).astype(int)).tolist())
            if len(cols) != k:
                return None
            i2 = 0
            for col in range(ca):
                if col in cols:
                    els.append((A[col], A[col + 1], Bn[i2]))
                else:
                    els.append((A[col], A[col + 1], Bn[i2 + 1], Bn[i2]))
                    i2 += 1
        else:
            k = cb - ca
            cols = set(np.unique(np.floor((np.arange(k) + 0.5) * cb / k).astype(int)).tolist())
            if len(cols) != k:
                return None
            i1 = 0
            for col in range(cb):
                if col in cols:
                    els.append((A[i1], Bn[col + 1], Bn[col]))
                else:
                    els.append((A[i1], A[i1 + 1], Bn[col + 1], Bn[col]))
                    i1 += 1
    P = np.vstack([cav.pts[: cav.n_loop]] + ([np.array(new_pts)] if new_pts else []))
    el = np.full((len(els), 4), -1, dtype=np.int64)
    nv = np.array([len(t) for t in els], dtype=np.int64)
    for r, t in enumerate(els):
        el[r, : len(t)] = t
    gid = np.r_[np.arange(cav.n_loop), np.full(len(new_pts), -1)].astype(np.int64)
    return P, gid, el, nv


# ------------------------------------------------------------------------------------------------ candidate: m02

def _cavity_triangles(cav: _Cav) -> np.ndarray:
    """The cavity's own elements as triangles (quads split along their shorter diagonal), closure-local ids."""
    el, nv = cav.cel, cav.cnv
    tris = [el[nv == 3, :3]]
    q = el[nv == 4]
    if len(q):
        p = cav.pts
        d02 = np.linalg.norm(p[q[:, 0]] - p[q[:, 2]], axis=1)
        d13 = np.linalg.norm(p[q[:, 1]] - p[q[:, 3]], axis=1)
        a = d02 <= d13
        tris += [q[a][:, [0, 1, 2]], q[a][:, [0, 2, 3]], q[~a][:, [0, 1, 3]], q[~a][:, [1, 2, 3]]]
    return np.vstack(tris).astype(np.int64)


def _m02_fill(cav: _Cav, h, nd: int):
    """m02 on the cavity alone: cross field of the cavity's triangulated elements (loop tangent fixed), L-inf
    frontal packing seeded with the loop nodes, sphere Delaunay (an antipodal dummy closes the hull), loop edges
    recovered, outside removed, recombination (nd = 4), m02's smoothing with the loop fixed."""
    from meshom.delaunay import adjacency, recover_edges, remove_land, sphere_delaunay
    from meshom.field import cross_field
    from meshom.methods.m02.packing import pack
    from meshom.methods.m02_linf_frontal import DEFAULTS as M02, _locate, _vertex_triangle
    from meshom.recombine import recombine
    from meshom.smooth import smooth
    from scipy.spatial import cKDTree

    K, nl = cav.n_cl, cav.n_loop
    Pb = np.ascontiguousarray(cav.pts[:K])
    Tb = np.ascontiguousarray(_cavity_triangles(cav))
    flag = np.zeros(K, dtype=np.int8)
    flag[:nl] = 1
    bg = Mesh(Pb, Tb, np.zeros((0, 4), dtype=np.int64), flag)
    cf = cross_field(bg, h, n=nd, boundary_nodes=np.arange(nl), smooth_steps=0, method=M02["field"])
    nbr = adjacency(Tb)
    vt = _vertex_triangle(Tb, K)
    hb = h(Pb) / R_EARTH_M
    lhb = np.log(hb)
    ub = np.ascontiguousarray(cf.u)
    cen = _unit(Pb[Tb].sum(axis=1))
    rad = np.max(np.linalg.norm(Pb[Tb] - cen[:, None, :], axis=2), axis=1)
    e = np.vstack([Tb[:, [0, 1]], Tb[:, [1, 2]], Tb[:, [2, 0]]])
    g = float(np.max(np.abs(hb[e[:, 0]] - hb[e[:, 1]]) / np.linalg.norm(Pb[e[:, 0]] - Pb[e[:, 1]], axis=1)))
    alpha = float(M02["alpha"])
    cc = np.sqrt(2.0) if nd == 4 else 2.0 / np.sqrt(3.0)
    rfac = cc * alpha / max(1.0 - 0.5 * cc * alpha * g, 0.25)
    tree = cKDTree(Pb)
    seeds = Pb[:nl]
    stri = _locate(Pb, Tb, nbr, vt, tree, seeds)
    sh = h(seeds) / R_EARTH_M
    fill = np.arange(nl, K, dtype=np.int64)
    fill_tri = vt[fill]
    area = 0.5 * np.linalg.norm(np.cross(Pb[Tb[:, 1]] - Pb[Tb[:, 0]], Pb[Tb[:, 2]] - Pb[Tb[:, 0]]), axis=1)
    n_est = float(np.sum(area / np.exp(lhb[Tb]).mean(axis=1) ** 2)) * (1.0 if nd == 4 else 2.0 / np.sqrt(3.0))
    cap = int(2.0 * n_est) + nl + 100
    while True:
        P = np.zeros((cap, 3))
        ph = np.zeros(cap)
        ptri = np.zeros(cap, dtype=np.int64)
        P[:nl], ph[:nl], ptri[:nl] = seeds, sh, stri
        npt, full, _ = pack(Pb, Tb, nbr, ub, lhb, cen, rad, P, ph, ptri, nl, fill, fill_tri, alpha, rfac,
                            bool(M02["midpoint"]), float(M02["seam"]), float(M02["alpha_seam"]), nd)
        if not full:
            break
        cap *= 2
    pts = P[:npt]
    c = _unit(pts.mean(axis=0))
    allp = np.vstack([pts, -c])
    le = np.vstack([np.column_stack([lp, np.roll(lp, -1)]) for lp in cav.loops])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tris, le, _ = recover_edges(allp, sphere_delaunay(allp), le, return_edges=True)   # split at touches (M40)
    tris = remove_land(allp, tris, le)
    tris = tris[(tris < npt).all(axis=1)]
    used = np.zeros(npt, dtype=bool)
    used[tris.ravel()] = True
    if not used[:nl].all():
        return None
    new = np.cumsum(used) - 1
    pts, tris = pts[used], new[tris]
    m = Mesh(pts, tris, np.zeros((0, 4), dtype=np.int64), np.zeros(len(pts), dtype=np.int8))
    if nd == 4:
        m = recombine(m, M02["price"], M02["recombine"])
    fixed = np.zeros(len(pts), dtype=bool)
    fixed[:nl] = True
    if M02["size_iters"]:
        m = smooth(m, h, fixed, iters=M02["size_iters"], kind="size")
    if M02["angle_iters"]:
        m = smooth(m, h, fixed, iters=M02["angle_iters"], kind="angle")
    el, nv = _elements(m)
    gid = np.r_[np.arange(nl), np.full(len(pts) - nl, -1)].astype(np.int64)
    return m.points, gid, el, nv


# ------------------------------------------------------------------------------------------------ driver

def _smooth_patch(P: np.ndarray, el: np.ndarray, nv: np.ndarray, nl: int, h, o: dict) -> np.ndarray:
    from meshom.smooth import smooth

    if len(P) == nl or not (o["smooth_size"] or o["smooth_angle"]):
        return P
    m = Mesh(P, el[nv == 3, :3], el[nv == 4], np.zeros(len(P), dtype=np.int8))
    fixed = np.zeros(len(P), dtype=bool)
    fixed[:nl] = True
    if o["smooth_size"]:
        m = smooth(m, h, fixed, iters=o["smooth_size"], kind="size")
    if o["smooth_angle"]:
        m = smooth(m, h, fixed, iters=o["smooth_angle"], kind="angle")
    return m.points


def _candidate(kind: str, cav: _Cav, h, o: dict, quad_dom: bool):
    if kind == "ops":
        ops = _Ops(cav, quad_dom, o)
        ops.run()
        res = ops.patch()
        if res is None:
            return None, 0
        P, gid, el, nv = res
        return (_smooth_patch(P, el, nv, cav.n_loop, h, o), gid, el, nv), ops.n_ops
    if kind == "coons":
        res = _coons(cav, o)
        if res is None:
            return None, 0
        P, gid, el, nv = res
        return (_smooth_patch(P, el, nv, cav.n_loop, h, o), gid, el, nv), int((nv == 3).any())   # 1 = a strip
    if kind in ("m02", "m02_tri"):
        if len(cav.C) > o["m02_max"]:
            return None, 0
        try:
            res = _m02_fill(cav, h, 4 if kind == "m02" else 6)
        except Exception:                                   # a degenerate cavity for the field / packing
            return None, 0
        return res, 0
    raise KeyError(kind)


def _accept(old: dict, new: dict, o: dict) -> bool:
    """Strictly lower badness, nothing inverted, no hard gate with more violations, neither the gross charge nor the
    count of bad elements (quads > bad_quad_deg, triangles failing an angle gate) nor the count of edges with an
    element size ratio > bad_ratio above the old one (a repair adds no irregular spot, no bad element, no size jump),
    the net charge kept (the index budget; a check of the patch)."""
    return (new["inverted"] == 0 and new["badness"] < old["badness"]
            and new["gross"] <= old["gross"] + o["gross_slack"] and new["bad_elem"] <= old["bad_elem"]
            and (new["bad_ratio"] <= old["bad_ratio"] if o["ratio_guard"] == "count" else True)
            and new["bad_ratio_coast"] <= old["bad_ratio_coast"]
            and new["net"] == old["net"]
            and all(new["hard"][k] <= old["hard"][k] for k in old["hard"]))


def _stage_list(o: dict) -> list[int]:
    out, r = [], int(o["rings"])
    while r <= int(o["rings_max"]):
        out.append(r)
        r *= 2
    return out or [int(o["rings"])]


def _evaluate(cav: _Cav, h, o: dict, quad_dom: bool, kinds: list, info: dict, times: dict):
    """Best accepted candidate of a cavity: (kind, patch, number of operations) or None."""
    t0 = time.perf_counter()
    old = cav.measure(cav.pts[: cav.n_cl], cav.cel, cav.cnv, h, o, quad_dom)
    times["measure"] += time.perf_counter() - t0
    b = str(old["net"])
    info["budget_hist"][b] = info["budget_hist"].get(b, 0) + 1
    best = None
    for ki, kind in enumerate(kinds):
        t0 = time.perf_counter()
        res, nops = _candidate(kind, cav, h, o, quad_dom)
        times[f"fill_{kind}"] += time.perf_counter() - t0
        info["tried"][kind] += 1
        if res is None:
            continue
        Pp, gid, pel, pnv = res
        if not cav.valid(len(Pp), pel, pnv):
            continue
        info["valid"][kind] += 1
        t0 = time.perf_counter()
        new = cav.measure(Pp, pel, pnv, h, o, quad_dom)
        times["measure"] += time.perf_counter() - t0
        if not _accept(old, new, o):
            continue
        info["improving"][kind] += 1
        k = (new["badness"], ki)
        if best is None or k < best[0]:
            best = (k, kind, res, nops, old, new)
    if best is None:
        return None
    _, kind, res, nops, old, new = best
    return kind, res, nops, new["bad_ratio"] - old["bad_ratio"], {k: old[k] - new[k] for k in
                                                                  ("gross", "bad_elem", "bad_edge", "bad_ratio")}


_W: dict = {}          # worker context (h, options) of the evaluation pool


def _parts(kinds: list) -> tuple[dict, dict]:
    info = {"tried": {k: 0 for k in kinds}, "valid": {k: 0 for k in kinds}, "improving": {k: 0 for k in kinds},
            "budget_hist": {}, "gain": {"gross": 0, "bad_elem": 0, "bad_edge": 0, "bad_ratio": 0}}
    return info, {"measure": 0.0, **{f"fill_{k}": 0.0 for k in kinds}}


def _merge(info: dict, times: dict, ip: dict, tp: dict) -> None:
    for key in ("tried", "valid", "improving", "gain"):
        for k, v in ip[key].items():
            info[key][k] += v
    for k, v in ip["budget_hist"].items():
        info["budget_hist"][k] = info["budget_hist"].get(k, 0) + v
    for k, v in tp.items():
        times[k] += v


def _worker_init(h, o: dict, quad_dom: bool, kinds: list) -> None:
    import numba

    numba.set_num_threads(1)
    _W.update(h=h, o=o, quad_dom=quad_dom, kinds=kinds)


def _eval_task(cav: "_Cav"):
    ip, tp = _parts(_W["kinds"])
    best = _evaluate(cav, _W["h"], _W["o"], _W["quad_dom"], _W["kinds"], ip, tp)
    return best, ip, tp


def _workers(o: dict) -> int:
    import os

    if o["workers"]:
        return int(o["workers"])
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(16, n))


def repair(mesh: Mesh, h, domain: Domain | None = None, opts: dict | None = None,
           is_block: np.ndarray | None = None, frozen: np.ndarray | None = None) -> tuple[Mesh, dict]:
    """Remesh the defect clusters of `mesh` cavity by cavity (module docstring). is_block (T+Q,) marks block
    elements (m14): cavities take only their first `block_rows` rows. frozen (T+Q,) marks frozen cells (R24, the
    fixed blocks of m14_blocks_rep): never a defect, in a cluster or a cavity, so their nodes never move; the
    returned record has their mask on the returned mesh ("frozen_after"). Returns the repaired mesh (coast nodes first
    and unchanged, unreferenced nodes dropped) and the info record."""
    o = {**DEFAULTS, **(opts or {})}
    o["hard"] = {**DEFAULTS["hard"], **(o.get("hard") or {})}
    t_all = time.perf_counter()
    quad_dom = mesh.n_quads >= mesh.n_tris
    times = {k: 0.0 for k in ("detect", "cavity", "measure", "apply", "summary")}
    times.update({f"fill_{k}": 0.0 for k in o["candidates"]})
    t0 = time.perf_counter()
    before = summary(mesh, h, o)
    times["summary"] += time.perf_counter() - t0
    P = mesh.points.copy()
    el, nv = _elements(mesh)
    blk = np.zeros(len(el), dtype=bool) if is_block is None else np.asarray(is_block, dtype=bool).copy()
    deep = _deep_blocks(el, blk, mesh.n_nodes, int(o["block_rows"]))
    flag = mesh.node_flag.copy()
    has_zone = mesh.tri_zone is not None                # R23: the declared triangle zone is kept, its mask returned
    zone = np.zeros(len(el), dtype=bool)
    if has_zone:
        zone[: mesh.n_tris] = np.asarray(mesh.tri_zone, dtype=bool)
    frz = None if frozen is None else np.asarray(frozen, dtype=bool).copy()   # R24: kept in element order
    kinds = list(o["candidates"])
    stages = _stage_list(o)
    info = {"options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in o.items()},
            "domain": getattr(domain, "name", None), "quad_dominant": bool(quad_dom), "before": before,
            "stages": [], "tried": {k: 0 for k in kinds}, "valid": {k: 0 for k in kinds},
            "improving": {k: 0 for k in kinds}, "accepted": {k: 0 for k in kinds}, "accepted_ops_count": 0,
            "accepted_smooth_only": 0, "accepted_strip": 0, "pool_cavities": 0, "skipped": {}, "budget_hist": {}, "rounds": 0,
            "by_rings": {str(r): {"tried": 0, "accepted": 0} for r in stages},
            "gain": {"gross": 0, "bad_elem": 0, "bad_edge": 0, "bad_ratio": 0}, "block_elems_replaced": 0}

    def to_mesh():
        tm = nv == 3
        return (Mesh(P, el[tm, :3], el[~tm], flag, zone[tm] if has_zone else None), np.r_[blk[tm], blk[~tm]],
                np.r_[deep[tm], deep[~tm]], None if frz is None else np.r_[frz[tm], frz[~tm]])

    # failures: (rings, cluster-core key) -> (epoch, nodes); still valid while none of the nodes changed since
    failed: dict = {}
    stamp = np.zeros(len(P), dtype=np.int64)     # epoch of the last change at each node
    epoch = 0
    nw = _workers(o)
    info["workers"] = nw
    n_int = int(before["n_interior_edges"])
    ratio_now = int(before["bad_ratio_edges"])
    ratio_cap = max(ratio_now, int(o["ratio_share"] * n_int)) if ratio_now <= 0.01 * n_int else ratio_now
    info["ratio_budget"] = {"start": ratio_now, "cap": ratio_cap, "edges": n_int, "rejected": 0}
    pool = None
    for rnd in range(int(o["max_rounds"])):
        acc_round = 0
        for si, rings in enumerate(stages):
            prev = stages[si - 1] if si else None
            st = {"round": rnd + 1, "rings": rings, "passes": 0, "clusters": None, "tried": 0, "accepted": 0,
                  "accepted_by": {k: 0 for k in kinds}}
            while True:
                t0 = time.perf_counter()
                cur, cblk, cdeep, cfrz = to_mesh()
                G = _Glob(cur, cblk, cdeep, h, o, quad_dom, cfrz)
                clusters = G.clusters()
                times["detect"] += time.perf_counter() - t0
                if st["clusters"] is None:
                    st["clusters"] = len(clusters)
                st["passes"] += 1
                changed = np.zeros(G.M, dtype=bool)     # cavities + first rings accepted in this pass
                repl = []
                built: dict = {}
                pending = list(range(len(clusters)))
                while pending:                          # waves: an independent set of the pending clusters
                    taken = changed.copy()
                    wave, rest = [], []
                    for ci in pending:
                        core = clusters[ci]
                        if changed[core].any():
                            continue
                        if ci not in built:
                            cnodes = G.nodes_of(core)
                            chash = hash(tuple(cnodes.tolist()))
                            ckey = (rings, chash)
                            f = failed.get(ckey)
                            if f is not None and stamp[f[1]].max() <= f[0]:
                                built[ci] = None
                                continue
                            if prev is not None and o["grow_failed_only"]:   # grow only what failed before
                                fp = failed.get((prev, chash))
                                if fp is None or stamp[fp[1]].max() > fp[0]:
                                    built[ci] = None
                                    continue
                            t0 = time.perf_counter()
                            res, why = _build_cavity(G, core, rings, o)
                            if res is None:
                                if f is None:
                                    info["skipped"][why] = info["skipped"].get(why, 0) + 1
                                failed[ckey] = (epoch, cnodes)
                                built[ci] = None
                            else:
                                cav = _Cav(G, *res)
                                built[ci] = (cav, ckey, np.r_[cav.C, cav.R1])
                            times["cavity"] += time.perf_counter() - t0
                        if built[ci] is None:
                            continue
                        cav, ckey, region = built[ci]
                        if changed[region].any():
                            continue
                        if taken[region].any():
                            rest.append(ci)
                            continue
                        taken[region] = True
                        wave.append(ci)
                    if not wave:
                        break
                    cavs = [built[ci][0] for ci in wave]
                    if pool is None and nw > 1 and len(cavs) >= 4:
                        import multiprocessing as mp
                        from concurrent.futures import ProcessPoolExecutor

                        pool = ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("spawn"),
                                                   initializer=_worker_init, initargs=(h, o, quad_dom, kinds))
                    if pool is not None and len(cavs) >= 4:
                        out = list(pool.map(_eval_task, cavs, chunksize=max(1, len(cavs) // (4 * nw))))
                        info["pool_cavities"] += len(cavs)
                    else:
                        out = []
                        for cav in cavs:
                            ip, tp = _parts(kinds)
                            out.append((_evaluate(cav, h, o, quad_dom, kinds, ip, tp), ip, tp))
                    for ci, (best, ip, tp) in zip(wave, out):
                        _merge(info, times, ip, tp)
                        cav, ckey, region = built[ci]
                        st["tried"] += 1
                        info["by_rings"][str(rings)]["tried"] += 1
                        if best is None:
                            failed[ckey] = (epoch, cav.g)
                            continue
                        kind, res, nops, dr, gain = best
                        if o["ratio_guard"] == "budget" and dr > 0 and ratio_now + dr > ratio_cap:
                            info["ratio_budget"]["rejected"] += 1
                            failed[ckey] = (epoch, cav.g)
                            continue
                        ratio_now += dr
                        for key, v in gain.items():
                            info["gain"][key] += v
                        changed[region] = True
                        info["accepted"][kind] += 1
                        st["accepted_by"][kind] += 1
                        info["by_rings"][str(rings)]["accepted"] += 1
                        info["block_elems_replaced"] += int(G.blk[cav.C].sum())
                        if kind == "ops":
                            info["accepted_ops_count"] += nops
                            info["accepted_smooth_only"] += int(nops == 0)
                        if kind == "coons":
                            info["accepted_strip"] += nops
                        repl.append((cav, res))
                    pending = rest
                st["accepted"] += len(repl)
                if not repl:
                    break
                t0 = time.perf_counter()
                epoch += 1
                keep = np.ones(G.M, dtype=bool)
                new_el, new_nv, new_pts = [], [], []
                n_next = len(P)
                for cav, (Pp, gid, pel, pnv) in repl:
                    stamp[cav.g] = epoch
                    keep[cav.C] = False
                    gmap = np.empty(len(Pp), dtype=np.int64)
                    old_ids = gid >= 0
                    gmap[old_ids] = cav.g[gid[old_ids]]
                    k_new = int((~old_ids).sum())
                    gmap[~old_ids] = n_next + np.arange(k_new)
                    n_next += k_new
                    new_pts.append(Pp[~old_ids])
                    mv = old_ids & (gid >= cav.n_loop)
                    P[gmap[mv]] = Pp[mv]
                    e2 = pel.copy()
                    m = e2 >= 0
                    e2[m] = gmap[e2[m]]
                    new_el.append(e2)
                    new_nv.append(pnv)
                P = np.vstack([P] + new_pts)
                stamp = np.r_[stamp, np.full(n_next - len(stamp), epoch, dtype=np.int64)]
                flag = np.r_[flag, np.zeros(n_next - len(flag), dtype=flag.dtype)]
                el = np.vstack([G.el[keep]] + new_el)
                nv = np.r_[G.nv[keep], np.concatenate(new_nv)]
                blk = np.r_[G.blk[keep], np.zeros(len(el) - int(keep.sum()), dtype=bool)]
                deep = np.r_[cdeep[keep], np.zeros(len(el) - int(keep.sum()), dtype=bool)]
                zone = np.r_[G.zone[keep], np.zeros(len(el) - int(keep.sum()), dtype=bool)]
                if frz is not None:
                    frz = np.r_[cfrz[keep], np.zeros(len(el) - int(keep.sum()), dtype=bool)]
                times["apply"] += time.perf_counter() - t0
            acc_round += st["accepted"]
            t0 = time.perf_counter()
            cur = to_mesh()[0]
            s = summary(cur, h, o)
            times["summary"] += time.perf_counter() - t0
            st["after"] = {k: s[k] for k in ("gross", "net", "n_irr", "n3", "n5", "n_other", "T", "bad_quads",
                                             "bad_edges", "lnsh_p99")}
            info["stages"].append(st)
            if o["verbose"]:
                print(f"[repair] round {rnd + 1} rings {rings}: {st}", flush=True)
        info["rounds"] = rnd + 1
        if acc_round == 0:
            break
    if pool is not None:
        pool.shutdown()
    # drop unreferenced nodes (coast nodes are always referenced and stay first)
    out, oblk, _, ofrz = to_mesh()
    used = np.zeros(len(P), dtype=bool)
    used[out.tris.ravel()] = True
    used[out.quads.ravel()] = True
    used |= flag != 0
    new = np.cumsum(used) - 1
    out = Mesh(P[used], new[out.tris], new[out.quads], flag[used], out.tri_zone)
    t0 = time.perf_counter()
    info["after"] = summary(out, h, o)
    times["summary"] += time.perf_counter() - t0
    info["is_block_after"] = oblk
    if ofrz is not None:
        info["frozen_after"] = ofrz
        info["frozen_cells"] = int(ofrz.sum())
    times["total"] = time.perf_counter() - t_all
    info["time_s"] = {k: round(v, 3) for k, v in times.items()}
    if o["coast_row"] == "after":                       # B65: the coast stage on the repaired mesh
        out, ci = repair(out, h, domain, {**(opts or {}), "coast_row": False, "_coast_stage": True}, is_block=oblk,
                         frozen=ofrz)
        info["coast_stage"] = {k: ci[k] for k in ("accepted", "rounds", "after", "time_s")}
        info["after"], info["is_block_after"] = ci["after"], ci["is_block_after"]
        if ofrz is not None:
            info["frozen_after"] = ci["frozen_after"]
    return out, info


# ------------------------------------------------------------------------------------------------ explicit region

def repair_region(mesh: Mesh, h, domain: Domain | None, polygon_lonlat, opts: dict | None = None,
                  is_block: np.ndarray | None = None) -> tuple[Mesh, dict]:
    """Remesh one cavity given explicitly ("select a zone and remesh", the UI's request): the elements whose centroid
    lies inside `polygon_lonlat` ([[lon, lat], ...] in degrees, great-circle edges, either orientation: the smaller
    side is the inside) plus `rings` node rings, closed as in `repair` (pinches, holes, concave corners). Elements of
    the coast row may be taken (`region_coast_row`, default True): coast nodes are always loop nodes, so they stay
    fixed; deep block rows are never taken. The same candidates and the same acceptance rule as `repair`; with
    opts["force"] the best valid candidate (patch topology right, nothing inverted, net charge kept) is taken even
    if it does not improve. info: the cavity, the candidates' measures, the chosen kind, before / after (local and
    whole-mesh), is_block_after (the block mask of the returned mesh's elements, as in `repair`)."""
    o = {**DEFAULTS, "ratio_guard": "count", **(opts or {})}    # one cavity: no running budget, the count guard
    o["hard"] = {**DEFAULTS["hard"], **(o.get("hard") or {})}
    t_all = time.perf_counter()
    quad_dom = mesh.n_quads >= mesh.n_tris
    kinds = list(o["candidates"])
    el, nv = _elements(mesh)
    blk = np.zeros(len(el), dtype=bool) if is_block is None else np.asarray(is_block, dtype=bool)
    deep = _deep_blocks(el, blk, mesh.n_nodes, int(o["block_rows"]))
    G = _Glob(mesh, blk, deep, h, o, quad_dom)
    if o.get("region_coast_row", True):
        G.allowed = ~deep & ~G.zone
    pl = np.asarray(polygon_lonlat, dtype=np.float64)
    ok = G.el >= 0
    cen = _unit((mesh.points[np.maximum(G.el, 0)] * ok[:, :, None]).sum(axis=1))
    core = np.flatnonzero(polygon_inside(pl)(cen) & G.allowed)
    info = {"options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in o.items()},
            "polygon": pl.tolist(), "n_core": int(len(core)), "accepted": False, "kind": None, "forced": False,
            "is_block_after": blk}
    if len(core) == 0:
        info["reason"] = "no element centroid inside the polygon"
        return mesh, info
    res, why = _build_cavity(G, core, int(o["rings"]), o)
    if res is None:
        info["reason"] = f"cavity not built: {why}"
        return mesh, info
    cav = _Cav(G, *res)
    info.update(n_cavity=int(len(cav.C)), n_loops=len(cav.loops), n_loop_nodes=int(cav.n_loop))
    old = cav.measure(cav.pts[: cav.n_cl], cav.cel, cav.cnv, h, o, quad_dom)
    info["old"] = old
    info["candidates"] = {}
    best = forced = None
    for ki, kind in enumerate(kinds):
        r, nops = _candidate(kind, cav, h, o, quad_dom)
        if r is None or not cav.valid(len(r[0]), r[2], r[3]):
            info["candidates"][kind] = None
            continue
        new = cav.measure(r[0], r[2], r[3], h, o, quad_dom)
        ok = _accept(old, new, o)
        info["candidates"][kind] = {"accepted": bool(ok), "operations": nops, **new}
        k = (new["badness"], ki)
        if ok and (best is None or k < best[0]):
            best = (k, kind, r)
        if new["inverted"] == 0 and new["net"] == old["net"] and (forced is None or k < forced[0]):
            forced = (k, kind, r)
    if best is None and o.get("force") and forced is not None:
        best = forced
        info["forced"] = True
    info["before"] = summary(mesh, h, o)
    if best is None:
        info["reason"] = "no candidate improves the cavity"
        info["after"] = info["before"]
        info["time_s"] = round(time.perf_counter() - t_all, 3)
        return mesh, info
    _, kind, (Pp, gid, pel, pnv) = best
    keep = np.ones(G.M, dtype=bool)
    keep[cav.C] = False
    gmap = np.empty(len(Pp), dtype=np.int64)
    old_ids = gid >= 0
    gmap[old_ids] = cav.g[gid[old_ids]]
    gmap[~old_ids] = mesh.n_nodes + np.arange(int((~old_ids).sum()))
    P = np.vstack([mesh.points, Pp[~old_ids]])
    mv = old_ids & (gid >= cav.n_loop)
    P[gmap[mv]] = Pp[mv]
    e2 = pel.copy()
    m = e2 >= 0
    e2[m] = gmap[e2[m]]
    el2 = np.vstack([G.el[keep], e2])
    nv2 = np.r_[G.nv[keep], pnv]
    flag = np.r_[mesh.node_flag, np.zeros(int((~old_ids).sum()), dtype=mesh.node_flag.dtype)]
    tm = nv2 == 3
    tris, quads = el2[tm, :3], el2[~tm]
    blk2 = np.r_[blk[keep], np.zeros(len(pnv), dtype=bool)]
    zone2 = np.r_[G.zone[keep], np.zeros(len(pnv), dtype=bool)]
    used = np.zeros(len(P), dtype=bool)
    used[tris.ravel()] = True
    used[quads.ravel()] = True
    used |= flag != 0
    new_id = np.cumsum(used) - 1
    out = Mesh(P[used], new_id[tris], new_id[quads], flag[used], None if mesh.tri_zone is None else zone2[tm])
    info.update(accepted=True, kind=kind, new=info["candidates"][kind], after=summary(out, h, o),
                is_block_after=np.r_[blk2[tm], blk2[~tm]], time_s=round(time.perf_counter() - t_all, 3))
    return out, info


def polygon_inside(polygon_lonlat):
    """xyz (N, 3) -> (N,) bool: inside the polygon [[lon, lat], ...] (degrees, great-circle edges, either orientation:
    the smaller side is the inside), the zone test of repair_region."""
    from meshom.domain import in_water
    from meshom.geo import lonlat_to_xyz

    pl = np.asarray(polygon_lonlat, dtype=np.float64)
    loop = lonlat_to_xyz(pl[:, 0], pl[:, 1])
    if loop_signed_area(loop) < 0:
        loop = loop[::-1].copy()
    dom = Domain([loop], "region")
    return lambda xyz: in_water(dom, xyz)


def region_summary(mesh: Mesh, h, polygon_lonlat, gates: dict | None = None) -> dict:
    """The zone of a polygon in compete._score's terms, for a before / after record of repair_region: the elements
    whose centroid lies inside (repair_region's core, without its rings) and the interior nodes inside. Counts
    (elements, triangles, quads, interior nodes and their n3 / n5 / n_other), "quality": the statistics of
    compute_quality and size_fidelity over those elements (their `_strait` subset with the polygon as the corridor,
    keys without the suffix) and, with `gates`, the gates checked on them (applicable ones: gates_passed /
    gates_total / gates_failed)."""
    from meshom.quality.metrics import check_gates, compute_quality, size_fidelity

    inside = polygon_inside(polygon_lonlat)
    q = compute_quality(mesh, inside)
    s = {**q.summary, **size_fidelity(mesh, h, inside)}
    reg = {k[: -len("_strait")]: v for k, v in s.items() if k.endswith("_strait")}
    ine = inside(_unit(np.r_[mesh.points[mesh.tris].sum(axis=1), mesh.points[mesh.quads].sum(axis=1)]))
    topo = build_topology(mesh)
    deg = np.bincount(topo.edges.ravel(), minlength=mesh.n_nodes)
    d = deg[inside(mesh.points) & ~topo.boundary_node & (topo.elems_per_node > 0)]
    T = mesh.n_tris
    out = {"n_elems": int(ine.sum()), "n_tris": int(ine[:T].sum()), "n_quads": int(ine[T:].sum()),
           "n_interior_nodes": int(d.size), "n3": int(np.sum(d == 3)), "n5": int(np.sum(d == 5)),
           "n_other": int(np.sum((d < 3) | (d > 5))), "quality": reg}
    if gates is not None:
        # n_quads of the whole mesh: the triangle target gate applies only to a mesh without quads, as in _score
        g = [x for x in check_gates({**reg, "n_quads": mesh.n_quads}, gates) if x["value"] is not None]
        out.update(gates_passed=sum(x["passed"] for x in g), gates_total=len(g),
                   gates_failed={x["gate"]: x["value"] for x in g if not x["passed"]})
    return out
