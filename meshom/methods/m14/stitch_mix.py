"""m14_blocks_mix: the fill domain partitioned into regions, one fill per region class.

Partition (`partition`). The fill domain F = water minus blocks gets a background mesh (`background_mesh` at mix_bg h;
the coast loops resampled at h and every loop taken as given, so its boundary nodes are the final coast and outline
nodes). Objects: each block outline, each coast loop. On the dual graph of the background triangles (centroid hops,
length / h) a Dijkstra with up to 3 labels per triangle gives the distances d1 <= d2 <= d3 (units of h, measured
inside F) to the three nearest distinct objects o1, o2, o3. A triangle is
  open      if d1 + d2 > w_open (the gap is wider than w_open h; also when only one object is reachable),
  junction  else if d3 - d2 < j_margin (a third object about as near as the second: the Y where three blocks meet,
            with arms along the lines where the second-nearest object changes, which end at the block corners),
  corridor  else, labelled by the pair {o1, o2}; a pair with a coast loop is a coast corridor (class "coast", also
            the channel between two coast loops).
Regions = edge-connected components of equal labels; components of < min_region triangles, and corridors with < 2
edges on one of their two objects, are merged into the neighbour sharing most dual edges (a junction or open one
first). A junction or open region that touches the coast keeps its class with coast = True. Every background triangle
gets exactly one region.

Shared boundaries (`resample_separators`). Separators = background edges between two regions, chained between branch
nodes (coast / outline nodes, nodes where the separator degree != 2); each chain is smoothed (sep_smooth Laplacian
passes, ends fixed) and resampled at h (count round(integral ds/h) >= 1); interior branch nodes keep their position. The
caps of a strip-eligible corridor (below) are instead straight chords between the ends of its two outline runs, with
the strip's row counts; a cap made of several separators becomes one chord whose interior branch nodes move onto it
(the other separators ending there follow). The resampled separator nodes with the coast and outline nodes are the
shared boundary nodes of every region: they enter every fill as given (the exact-loop mechanism), as constraint edges
of one triangulation, and each region is meshed with them fixed.

Corridor geometry and classes (`corridor_geometry`, `classify`). A block-block corridor's two sides are its outline
runs (one contiguous run per block, the region on their left). Measured: nA, nB = outline edges per side; rotation =
4-fold circular mean of the angle between each A edge and the nearest B edge in the tangent plane at their midpoint
(the relative rotation of the two lattices mod 90 deg); obliquity of each side = 4-fold mean angle of its edges to its
own smoothed tangent (0: lattice rows along the gap, 45: a diagonal staircase); width = median distance from side A's
nodes to side B in h. Every run edge is classified against the local lattice axis nearest to the run's chord: along,
a step toward the region or away (a lattice path); nA_along, nB_along and the mismatch k = |nA_along - nB_along| (per
10 cells of length). A corridor is strip-eligible when both sides are lattice paths, both obliquities < obl_strip_deg
(above it the strip's cells shear: max quad 143-169 deg measured), its separators form two caps joining the run ends,
end steps that would run along a cap are handed to the region beyond it (the cap moves to the next node), and the
ladder plan below exists. Rule: eligible, rotation < rot_quad_deg and mismatch <= rows_tol -> "strip"; rotation >=
rot_tri_deg or mismatch > rows_tri -> "tri"; else "m02". Junctions -> junction_fill, open -> open_fill, coast corridors
and coast-touching regions -> coast_fill; force_fill sets the fill of every block-block corridor ("strip" where
eligible, else "m02"). The defaults are the measured rule (strip below 25 deg rotation where eligible, m02 everywhere
else, no triangles). With merge_same_fill, adjacent non-strip regions with the same fill are then meshed as one fill
region (their separators dropped: they only add colliding fronts); the regions stay the unit of measurement.

The quantised join (join = "quantised", B47). A corridor whose runs are lattice paths gets its Burgers vector
b = (b_par, b_perp) (`burgers_vector`: column ends and rows ending, the minimum dislocation count |b|_1 of any fill) and
its quantisation (`quantise`): b = 0 "none", b_perp = 0 and |b_par| <= width "shear" (the column ends at the corridor
ends), else "even" (|b|_1 dislocations evenly spaced on the middle row, off the outline rows and end_rows cells off
either cap: B58). Join by the obliquity alone: below obl_strip_deg the ladder where eligible, above it the lattice
continuation (m14/continuation.py, `cont_fill`), else m02; rot_quad_deg and rows_tol do not decide. With cont_fallback
the continuation also takes the corridors below obl_strip_deg whose ladder is impossible or invalid (an invalid ladder
patch makes the attempt start again). A continuation corridor is filled after the one triangulation of everything (it
gets no packed points; its CDT triangles give its boundary loop) from its two runs in block node ids; one that fails
(not one loop, a run in pieces, a fold) is m02 in the next attempt. With lattice_caps its two caps are lattice paths of
the blended map (`_lattice_caps`, continuation.lattice_cap: half the across steps, the along steps, the other half;
a run end edge that would run up its cap goes to the region beyond; the cap's interior branch nodes move onto the path,
the separators ending there follow), their nodes keep their labels for the fill's label quads; a cap that would cross
a constraint, or turns a neighbour inside out, tries the interleaved path, then keeps its separators.

Triangle clusters (break_clusters, B62; both joins): after the fill smoothing every edge-connected cluster of >= 3
triangles outside the "tri" regions is broken (`break_clusters`: a pair merged, the cluster's pentagon / hexagon
re-split, or an interior node removed and its star re-split; quads < 150 deg), then 10 "size" + 3 "angle" sweeps.

The ladder strip (`_strip_plan`, `strip_patch`), the simplest correct strip between two lattice-path sides: columns pair
one along edge of each side (where the counts differ, k evenly spaced columns pair two edges of the longer side: a
column termination); column line s runs from side A's topmost node there to side B's lowest; its free part has e_s
edges (e_0, e_S = the caps), chosen by dynamic programming to minimise the number of row terminations sum |R_s - L_s|
(R_s, L_s = the edges of strip s's right and left side, the steps of A and B at a line belonging to one side or the
other) + strip_lam * sum (e_s - w_s)^2, w_s = the length of the free part in h, at most 2 per column. Each column is
filled from its A edge(s) to its B edge(s) by quads; its row terminations are triangles in the middle rows (one
triangle and one valence-5 node each) and a column termination is an extra line from the middle node up to the middle
row closed by one quad and one triangle. Straight parallel sides with equal counts: all quads, every node regular; a
mismatch of 2: exactly 2 triangles (tests/test_m14_mix.py). A step of a side is a row ending at the outline (its corner
quads absorb it, no defect).

Fills, all inside one constrained triangulation (`stitch_mix`): m02's packing (n = 4 and / or n = 6, m02's cross field
on the background mesh) seeded with every coast, outline and separator node, so fronts also start from the
separators; the packed points are assigned to regions by a constrained triangulation cut at the separators, and only the
points of "m02" regions (n = 4) and "tri" regions (n = 6) are kept. One sphere Delaunay of all fixed nodes + kept
points, every loop and separator edge recovered, land and blocks removed, triangles labelled by region (components cut
at the separators, named by the known region left of each separator / loop half-edge; a leak raises). "m02" regions:
m02's recombination, pairs never across a separator (nodes duplicated per region for the matching). "tri": triangles
as they are, or the `tri_fill` hook. "strip": the triangles replaced by the ladder. Then the fill is smoothed as m02
smooths (size_iters "size", angle_iters "angle"; coast and outline nodes fixed, the separator nodes too unless
free_separators: connectivity is kept, so each region keeps its own fill), merged with the blocks by node ids, and the
seam band smoothed as in stitch.py.

tri_fill hook (m15's triangle fill): "module:function" with fill_tri(domain, h, fixed_loops) -> Mesh, the domain = the
region's boundary loops as exact loops, the mesh with those loop nodes first in loop order; every "tri" region is then
meshed by it instead of the n = 6 points (fill_tri_local is a reference implementation: Poisson points, a local
Delaunay with far helper points, size smoothing).

Measurement per region (`measure_regions`): elements, triangles, nodes (each fill node belongs to the region holding
most of its fill elements), irregular = interior nodes with valence != 4 (irr4) and with |360 - 90 q - 60 t| >= 60 deg
(irr_mix: q quads and t triangles at the node; regular for all-quad, all-triangle and 2 + 3 mixed nodes; a lone
triangle in a quad mesh gives three +30 nodes, not counted), densities per 10 cells of length (corridors: L = (nA +
nB) / 2; others: half the outline + coast edges on the boundary) and per 100 cells of area, |ln s/h| p99 and max, the
element ratio p99 across the edges of the region, quad and triangle angles.
"""
import heapq
import importlib
import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from meshom.delaunay import adjacency, background_mesh, sphere_delaunay
from meshom.domain import Domain, _place, loop_edges, resample_loops
from meshom.geo import R_EARTH_M, arc_length, tangent_basis
from meshom.mesh import Mesh, build_topology, signed_area
from meshom.size import SizeField

FILLS = ("strip", "cont", "m02", "tri")
CLS = ("corridor", "junction", "open", "coast")
END_MARGIN = 2           # B58 (the quantised join): dislocations keep this many columns from either cap of a ladder
END_PENALTY = 4.0        # ... a row termination there costs 1 + this in the row-count DP (finite: a plan still exists)
STRIP_QUAD_MAX = 150.0   # B46: a ladder patch as built with a quad angle >= this is no strip (the hard gate)
CLUSTER_QUAD_MAX = 150.0  # B62: a triangle cluster of >= 3 is broken by new quads (and triangles) below this angle
SEED_COINCIDE = 1e-9     # B66: two seeds (loop or separator nodes) closer than this unit-sphere chord (~6 mm) coincide


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _wrap90(x):
    return (np.asarray(x) + 45.0) % 90.0 - 45.0


def _coincident_chains(seeds: np.ndarray, chains: list, nbnd: int) -> list:
    """Ids of the chains (their node ids c["ids"] index seeds, the first nbnd are loop nodes) holding a node within
    SEED_COINCIDE of another seed, one of the two a separator node (B66): the CDT's hull keeps one of two coincident
    points, so a constraint at the other cannot be recovered."""
    from scipy.spatial import cKDTree

    pairs = cKDTree(seeds).query_pairs(SEED_COINCIDE, output_type="ndarray")
    pairs = pairs[pairs.max(axis=1) >= nbnd] if len(pairs) else pairs
    if len(pairs) == 0:
        return []
    bad = np.zeros(len(seeds), dtype=bool)
    bad[pairs.ravel()] = True
    return [ci for ci, c in enumerate(chains) if bad[c["ids"]].any()]


# ------------------------------------------------------------------------------------------------------- partition

@njit(cache=True)
def _k_nearest(nbr, w, src_t, src_o, src_d, K):
    """Dijkstra on the dual graph with up to K distinct source labels per triangle: (T,K) distances and labels."""
    T = nbr.shape[0]
    dist = np.full((T, K), np.inf)
    obj = np.full((T, K), -1, dtype=np.int64)
    cnt = np.zeros(T, dtype=np.int64)
    heap = [(np.float64(0.0), np.int64(0), np.int64(0))]
    heap.pop()
    for i in range(len(src_t)):
        heapq.heappush(heap, (src_d[i], src_t[i], src_o[i]))
    while len(heap) > 0:
        d, t, o = heapq.heappop(heap)
        c = cnt[t]
        if c >= K:
            continue
        dup = False
        for k in range(c):
            if obj[t, k] == o:
                dup = True
                break
        if dup:
            continue
        dist[t, c] = d
        obj[t, c] = o
        cnt[t] = c + 1
        for k in range(3):
            u = nbr[t, k]
            if u < 0 or cnt[u] >= K:
                continue
            dup = False
            for kk in range(cnt[u]):
                if obj[u, kk] == o:
                    dup = True
                    break
            if not dup:
                heapq.heappush(heap, (d + w[t, k], u, o))
    return dist, obj


@dataclass
class Partition:
    bg: Mesh                     # background mesh of the fill domain (boundary nodes first = fill ids)
    nbnd: int                    # boundary nodes (coast + outline) = the first fill ids
    reg: np.ndarray              # (Tb,) region of each background triangle
    regions: list                # region records (dict)
    chains: list                 # separators: dict(nodes = bg chain, left, right, ids = fill ids after resampling)
    sep_xyz: np.ndarray          # (S,3) separator nodes (fill ids nbnd .. nbnd + S - 1)
    he_keys: np.ndarray          # directed half-edges (fill ids a * N + b, N = nbnd + S), known region on the left
    he_reg: np.ndarray           # (and he_sep: the half-edge lies on a separator, set by resample_separators)
    cons: np.ndarray             # (E,2) constraint edges: loop edges (fill on the left) then separator edges
    n_loop_edges: int
    info: dict = field(default_factory=dict)


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def _region_stats(reg, nR, nbr, tri_obj_be, nobj):
    """Per region: triangle counts, boundary edges per object (R, nobj), shared dual edges between regions (dict)."""
    ntri = np.bincount(reg, minlength=nR)
    t_be, o_be = tri_obj_be
    cnt = np.zeros((nR, nobj), dtype=np.int64)
    np.add.at(cnt, (reg[t_be], o_be), 1)
    t, k = np.nonzero(nbr >= 0)
    u = nbr[t, k]
    m = reg[t] != reg[u]
    ra, rb = reg[t[m]], reg[u[m]]
    key, c = np.unique(ra * nR + rb, return_counts=True)
    shared = {}
    for kk, cc in zip(key, c):
        shared.setdefault(int(kk // nR), {})[int(kk % nR)] = int(cc)
    return ntri, cnt, shared


def partition(coast: list, outl: list, h: SizeField, o: dict) -> Partition:
    loops = list(coast) + list(outl)
    C, B = len(coast), len(outl)
    nobj = B + C
    dom = Domain(loops, "fill", exact=[True] * len(loops))
    bg = background_mesh(dom, h, factor=o["mix_bg"])
    P, Tb = bg.points, bg.tris
    nbnd = sum(len(lp) for lp in loops)
    assert np.array_equal(P[:nbnd], np.vstack(loops))
    loop_of = np.repeat(np.arange(len(loops)), [len(lp) for lp in loops])
    obj_of_loop = np.r_[B + np.arange(C), np.arange(B)].astype(np.int64)    # blocks 0..B-1, coast loops B..B+C-1
    obj_node = obj_of_loop[loop_of]
    nbr = adjacency(Tb)
    cen = _unit(P[Tb].sum(axis=1))
    hc = h(cen)
    # dual edge weights (units of h) and sources (triangles on a loop edge)
    t, k = np.nonzero(nbr >= 0)
    w = np.zeros(Tb.shape)
    w[t, k] = R_EARTH_M * arc_length(cen[t], cen[nbr[t, k]]) / (0.5 * (hc[t] + hc[nbr[t, k]]))
    tb, kb = np.nonzero(nbr < 0)
    a, b = Tb[tb, kb], Tb[tb, (kb + 1) % 3]
    ob = obj_node[a]
    d0 = R_EARTH_M * arc_length(cen[tb], _unit(P[a] + P[b])) / hc[tb]
    dist, obj = _k_nearest(nbr, w, tb.astype(np.int64), ob.astype(np.int64), d0, 3)
    d1, d2, d3 = dist[:, 0], dist[:, 1], dist[:, 2]
    wide = ~(d1 + d2 <= o["w_open"])
    junc = ~wide & (d3 - d2 < o["j_margin"])
    cls_t = np.where(wide, 2, np.where(junc, 1, 0))
    pa, pb = np.minimum(obj[:, 0], obj[:, 1]), np.maximum(obj[:, 0], obj[:, 1])
    key = np.where(cls_t == 0, (pa + 1) * (nobj + 2) + (pb + 1), -cls_t)
    same = key[t] == key[nbr[t, k]]
    reg = _components(len(Tb), t[same], nbr[t, k][same])
    _, reg = np.unique(reg, return_inverse=True)
    reg = reg.ravel()
    # merge small regions and corridors with < 2 edges on a side into a neighbour (several per pass, disjoint)
    merges = 0
    for _ in range(100):
        nR = int(reg.max()) + 1
        first = np.zeros(nR, dtype=np.int64)
        first[reg[::-1]] = np.arange(len(reg))[::-1]
        rkey = key[first]
        ntri, cnt, shared = _region_stats(reg, nR, nbr, (tb, ob), nobj)
        target = np.arange(nR)
        touched = np.zeros(nR, dtype=bool)
        for r in np.argsort(ntri, kind="stable"):
            if touched[r] or r not in shared:
                continue
            corr = rkey[r] > 0
            bad = ntri[r] < o["min_region"]
            if corr and not bad:
                oa, obb = int(rkey[r] // (nobj + 2)) - 1, int(rkey[r] % (nobj + 2)) - 1
                bad = cnt[r, oa] < 2 or cnt[r, obb] < 2
            if not bad:
                continue
            nb = {x: c for x, c in shared[r].items() if not touched[x]}
            if not nb:
                continue
            cand = ([x for x in nb if rkey[x] <= 0] if corr else []) or list(nb)
            x = max(cand, key=lambda y: nb[y])
            target[r] = x
            touched[r] = touched[x] = True
        moved = target != np.arange(nR)
        if not moved.any():
            break
        merges += int(moved.sum())
        key = np.where(moved[reg], rkey[target[reg]], key)
        _, reg = np.unique(target[reg], return_inverse=True)
        reg = reg.ravel()
    nR = int(reg.max()) + 1
    first = np.zeros(nR, dtype=np.int64)
    first[reg[::-1]] = np.arange(len(reg))[::-1]
    rkey = key[first]
    ntri, cnt, shared = _region_stats(reg, nR, nbr, (tb, ob), nobj)
    area = np.abs(0.5 * np.einsum("ij,ij->i", np.cross(P[Tb[:, 1]] - P[Tb[:, 0]], P[Tb[:, 2]] - P[Tb[:, 0]]),
                                  cen)) * R_EARTH_M**2 / hc**2
    area_r = np.bincount(reg, area, minlength=nR)
    regions = []
    for r in range(nR):
        mt = reg == r
        objs = np.unique(obj[mt][:, :3])
        objs = [int(x) for x in objs if x >= 0]
        coast_e = int(cnt[r, B:].sum())
        if rkey[r] > 0:
            oa, obb = int(rkey[r] // (nobj + 2)) - 1, int(rkey[r] % (nobj + 2)) - 1
            cls = "coast" if (oa >= B or obb >= B) else "corridor"
            objs = [oa, obb]
        else:
            cls = "junction" if rkey[r] == -1 else "open"
        regions.append({"id": r, "cls": cls, "objs": objs, "coast": bool(coast_e > 0), "n_bg_tris": int(ntri[r]),
                        "area_h2": float(area_r[r]), "outline_edges": int(cnt[r, :B].sum()), "coast_edges": coast_e,
                        "width_dd": float(np.median((d1 + d2)[mt])) if np.isfinite((d1 + d2)[mt]).any() else None})
    # ---- separators: bg edges between two regions, chained between branch nodes
    m = (reg[t] != reg[nbr[t, k]]) & (t < nbr[t, k])
    sa, sb = Tb[t[m], k[m]], Tb[t[m], (k[m] + 1) % 3]
    sl, sr = reg[t[m]], reg[nbr[t[m], k[m]]]                           # left of sa -> sb, right
    ns = len(sa)
    nn = len(P)
    deg = np.bincount(np.r_[sa, sb], minlength=nn)
    branch = deg != 2
    branch[:nbnd] = True
    inc_ptr = np.r_[0, np.cumsum(deg)]
    order = np.argsort(np.r_[sa, sb], kind="stable")
    inc = np.r_[np.arange(ns), np.arange(ns)][order]                    # separator edges per node (CSR)
    used = np.zeros(ns, dtype=bool)
    chains = []

    def walk(v0, e0):
        nodes, e, v = [v0], e0, v0
        while True:
            used[e] = True
            v = sb[e] if sa[e] == v else sa[e]
            nodes.append(int(v))
            if branch[v] or v == v0:
                break
            nxt = [x for x in inc[inc_ptr[v]:inc_ptr[v + 1]] if not used[x]]
            if not nxt:
                break
            e = nxt[0]
        e = e0
        left, right = (sl[e], sr[e]) if sa[e] == v0 else (sr[e], sl[e])
        return {"nodes": np.asarray(nodes, dtype=np.int64), "left": int(left), "right": int(right)}

    for v in np.flatnonzero(branch & (deg > 0)):
        for e in inc[inc_ptr[v]:inc_ptr[v + 1]]:
            if not used[e]:
                chains.append(walk(int(v), int(e)))
    for e in range(ns):                                                 # closed separators without a branch node
        if not used[e]:
            chains.append(walk(int(sa[e]), e))
    part = Partition(bg, nbnd, reg, regions, chains, np.zeros((0, 3)), np.zeros(0, np.int64),
                     np.zeros(0, np.int64), np.zeros((0, 2), np.int64), 0,
                     info={"n_parts": nR, "n_merges": merges, "n_separators": len(chains), "bg_tris": int(len(Tb))})
    part.loops, part.obj_node, part.B, part.C = loops, obj_node, B, C
    part.tb, part.kb, part.ob = tb, kb, ob
    return part


# --------------------------------------------------------------------------------------------- corridor geometry

def _runs(part: Partition, r: int) -> dict:
    """The outline / coast edges of region r per object: {obj: sorted local edge indices along its loop}."""
    Tb = part.bg.tris
    m = part.reg[part.tb] == r
    a = Tb[part.tb[m], part.kb[m]]
    offs = np.r_[0, np.cumsum([len(lp) for lp in part.loops])]
    li = np.searchsorted(offs, a, side="right") - 1
    out = {}
    for x, l_ in zip(a, li):
        out.setdefault(int(part.obj_node[x]), []).append((int(l_), int(x - offs[l_])))
    return {k: (v[0][0], np.sort(np.array([e for _, e in v]))) for k, v in out.items()}


def _contiguous(idx: np.ndarray, K: int):
    """Start and length of a cyclic run of edge indices idx (sorted) on a loop of K edges; None if not one run."""
    if len(idx) == K:
        return None
    gaps = np.flatnonzero(np.diff(np.r_[idx, idx[0] + K]) > 1)
    if len(gaps) != 1:
        return None
    start = idx[(gaps[0] + 1) % len(idx)]
    return int(start), len(idx)


def _rotation(xa: np.ndarray, xb: np.ndarray) -> tuple[float, float]:
    """Relative lattice rotation (deg, mod 90, in [0, 45]) between the edges of polyline xa and the nearest edges of
    polyline xb (tangent plane at the two edge midpoints' mean), and the coherence |mean e^(4 i phi)|."""
    from scipy.spatial import cKDTree

    ea, eb = xa[1:] - xa[:-1], xb[1:] - xb[:-1]
    ma, mb = _unit(xa[1:] + xa[:-1]), _unit(xb[1:] + xb[:-1])
    _, j = cKDTree(mb).query(ma)
    nrm = _unit(ma + mb[j])
    u = ea - np.einsum("ij,ij->i", ea, nrm)[:, None] * nrm
    v = eb[j] - np.einsum("ij,ij->i", eb[j], nrm)[:, None] * nrm
    phi = np.arctan2(np.einsum("ij,ij->i", np.cross(u, v), nrm), np.einsum("ij,ij->i", u, v))
    z = np.exp(4j * phi).mean()
    return float(abs(np.degrees(np.angle(z) / 4.0))), float(abs(z))


def _obliquity(x: np.ndarray, win: int = 3) -> float:
    """4-fold mean angle (deg, [0, 45]) of the edges of polyline x against its smoothed tangent (+-win nodes)."""
    n = len(x)
    e = x[1:] - x[:-1]
    mid = _unit(x[1:] + x[:-1])
    i = np.arange(n - 1)
    tan = x[np.clip(i + 1 + win, 0, n - 1)] - x[np.clip(i - win, 0, n - 1)]
    phi = np.arctan2(np.einsum("ij,ij->i", np.cross(tan, e), mid), np.einsum("ij,ij->i", tan, e))
    return float(abs(np.degrees(np.angle(np.exp(4j * phi).mean()) / 4.0)))


def _corners(x: np.ndarray, deg: float = 30.0) -> int:
    """Interior nodes of the open polyline x turning by more than deg (lattice corners of an outline run)."""
    if len(x) < 3:
        return 0
    a, b = x[1:-1] - x[:-2], x[2:] - x[1:-1]
    c = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return int(np.sum(c < np.cos(np.radians(deg))))


def _edge_classes(x: np.ndarray, win: int = 8, lwin: int = 5, tol_deg: float = 30.0):
    """Lattice-path classes of the edges of an outline run x (region on its left): 0 along, +1 a step toward the region
    (left), -1 a step away; None if an edge runs backward or lies > tol_deg off the local lattice axes. Outline edges
    are lattice edges: each edge's angle to the local chord (+-win nodes) minus the local lattice angle (4-fold mean
    over +-lwin edges, the axis nearest to the chord) is rounded to a multiple of 90 deg."""
    n = len(x)
    if n < 2:
        return None
    e = x[1:] - x[:-1]
    mid = _unit(x[1:] + x[:-1])
    i = np.arange(n - 1)
    tan = x[np.clip(i + 1 + win, 0, n - 1)] - x[np.clip(i - win, 0, n - 1)]
    nrm = np.cross(mid, tan)                                            # left of the run, seen from outside
    phi = np.arctan2(np.einsum("ij,ij->i", e, nrm) / np.linalg.norm(nrm, axis=1),
                     np.einsum("ij,ij->i", e, tan) / np.linalg.norm(tan, axis=1))
    z = np.exp(4j * phi)
    cz = np.r_[0, np.cumsum(z)]
    lo, hi = np.clip(i - lwin, 0, n - 1), np.clip(i + lwin + 1, 0, n - 1)
    psi = np.angle(cz[hi] - cz[lo]) / 4.0
    rel = np.degrees(phi - psi)
    q = np.round(rel / 90.0)
    if (np.abs(rel - 90.0 * q) > tol_deg).any():
        return None
    q = (q.astype(np.int64) + 4) % 4
    if (q == 2).any():
        return None
    return np.where(q == 0, 0, np.where(q == 1, 1, -1)).astype(np.int64)


def _path(cls: np.ndarray, j0: int, dj: int):
    """Lattice coordinates (I, J) of the nodes of a path: along -> I + 1, class +-1 -> J + dj * class."""
    I = np.r_[0, np.cumsum(cls == 0)]
    J = j0 + dj * np.r_[0, np.cumsum(cls)]
    return I.astype(np.int64), J.astype(np.int64)


def _pairing(n_long: int, k: int, e_long: np.ndarray, mode: str = "even", margin=0):
    """k disjoint pairs (i, i + 1) of consecutive along edges of the longer side (edge indices e_long, no step between),
    at evenly spaced positions (nearest free pair), None if there are not enough. Mode "spread" (B58): a pair with a
    step of the side at either of its column lines counts 0.75 cells farther (its termination would share a short
    column with the step's row end), and the pairs keep margin (an int, or (at C0, at C1)) along edges from the ends
    (spaced evenly between) while such pairs are free; mode "ends" (B58's shear): alternately nearest to the two ends,
    a small count mismatch pushed to the corridor ends."""
    ok = np.flatnonzero(np.diff(e_long) == 1)                           # along edge i and i + 1 adjacent in the run
    gap = np.diff(np.r_[e_long[0] - 1, e_long, e_long[-1] + 1])        # > 1: a step before along edge i (or after)
    near = 0.75 * ((gap[:-1] > 1) | (gap[1:] > 1)) * (mode == "spread")  # step at the line before / after edge i
    m0, m1 = (margin, margin) if np.isscalar(margin) else margin
    m0, m1 = (m0, m1) if mode == "spread" else (0, 0)
    inner = (ok >= m0) & (ok + 1 <= n_long - 1 - m1)
    n_eff = max(n_long - m0 - m1, min(n_long, 2 * k))
    off = m0 * (n_long - n_eff) / max(m0 + m1, 1)
    used = np.zeros(n_long, dtype=bool)
    out = []
    for t in range(k):
        want = off + (t + 0.5) * n_eff / k - 0.5
        if mode == "ends":
            want = (t // 2) * 2 if t % 2 == 0 else n_long - 2 - (t // 2) * 2
        free = [(i, m) for i, m in zip(ok, inner) if not used[i] and not used[i + 1]]
        cand = [i for i, m in free if m] or [i for i, _ in free]
        if not cand:
            return None
        i = min(cand, key=lambda x: abs(x - want) + near[x] + near[x + 1])
        used[i] = used[i + 1] = True
        out.append(int(i))
    return sorted(out)


def _strip_plan(ca: np.ndarray, cbv: np.ndarray, xa: np.ndarray, xbv: np.ndarray, h: SizeField, lam: float = 1.0,
                pairing: str = "even", end_rows=0):
    """Plan of a ladder strip between side A (region on its left) and side Bv (side B reversed, so both run from cap
    C0 to cap C1; its edge classes in that order: +1 = up, away from A). Columns: each pairs one along edge of one side
    with one or two along edges of the other (the |nA - nB| doubled columns, evenly spaced, are column terminations).
    Line s (s = 0..S, 0 and S the caps) runs from the top of side A's step nodes there to the bottom of side B's; its
    free part has e_s edges (e_0, e_S = the cap counts). Strip s has a left side of L_s = (A down-steps at s) + e_s +
    (B up-steps at s) edges and a right side of R_s = (A up-steps at s+1) + e_{s+1} + (B down-steps at s+1) edges; the
    |R_s - L_s| row terminations are its triangles. e is chosen by dynamic programming: minimum of the triangles plus
    lam * sum (e_s - w_s)^2, w_s = the length of line s's free part in h, |R_s - L_s| <= 2. Returns the plan or a
    reason. pairing: where the column terminations go (`_pairing`); "spread" and "ends" (the quantised join, B58) also
    keep every termination's triangle off the two sides (`strip_patch`) and, with end_rows (an int, or (at C0, at
    C1): the caps that border a junction, the coast or an open region, B58), every termination that many columns off
    those caps where the plan allows (the pairs; a row termination there costs END_PENALTY more)."""
    JA = _path(ca, 0, 1)[1]
    JB = _path(cbv, 0, 1)[1]
    ea, eb = np.flatnonzero(ca == 0), np.flatnonzero(cbv == 0)
    nA, nB = len(ea), len(eb)
    if nA < 2 or nB < 2:
        return "a side has < 2 along edges"
    k = abs(nA - nB)
    r0, r1 = (end_rows, end_rows) if np.isscalar(end_rows) else end_rows
    if k:
        pr = _pairing(max(nA, nB), k, ea if nA > nB else eb, pairing, (r0, r1))
        if pr is None or k > 0.3 * min(nA, nB):
            return "count mismatch too large for the column terminations"
        dbl = set(pr)
    strips, ia, ib = [], 0, 0
    while ia < nA and ib < nB:
        two_a = nA > nB and ia in dbl
        two_b = nB > nA and ib in dbl
        strips.append(([int(ea[ia])] + ([int(ea[ia + 1])] if two_a else []),
                       [int(eb[ib])] + ([int(eb[ib + 1])] if two_b else [])))
        ia += 2 if two_a else 1
        ib += 2 if two_b else 1
    S = len(strips)
    # line s: side corner nodes (indices into A / Bv): the end of strip s-1's edges and the start of strip s's
    a_end = [0] + [st[0][-1] + 1 for st in strips]                     # A node at the bottom-right of strip s-1
    a_beg = [st[0][0] for st in strips] + [len(xa) - 1]                # A node at the bottom-left of strip s
    b_end = [0] + [st[1][-1] + 1 for st in strips]
    b_beg = [st[1][0] for st in strips] + [len(xbv) - 1]
    lines = []
    for sl in range(S + 1):
        dA = int(JA[a_beg[sl]] - JA[a_end[sl]])
        dB = int(JB[b_beg[sl]] - JB[b_end[sl]])
        if sl == 0:
            dA, dB = int(JA[a_beg[0]] - JA[0]), int(JB[b_beg[0]] - JB[0])
            a_top, b_bot = 0, 0
        elif sl == S:
            dA, dB = int(JA[-1] - JA[a_end[S]]), int(JB[-1] - JB[b_end[S]])
            a_top, b_bot = len(xa) - 1, len(xbv) - 1
        else:
            a_top = a_beg[sl] if dA > 0 else a_end[sl]
            b_bot = b_beg[sl] if dB < 0 else b_end[sl]
        if (sl == 0 and (dA > 0 or dB < 0)) or (sl == S and (dA < 0 or dB > 0)):
            return "a side ends with a step along its cap"
        p0, p1 = xa[a_top], xbv[b_bot]
        w = float(R_EARTH_M * arc_length(p0[None], p1[None])[0] / h(_unit(p0 + p1)[None])[0])
        lines.append({"a_beg": a_beg[sl], "a_end": a_end[sl], "b_beg": b_beg[sl], "b_end": b_end[sl],
                      "a_top": a_top, "b_bot": b_bot, "dA": dA, "dB": dB, "w": w})
    # dynamic programming over e_s
    def L_(sl, e):
        ln = lines[sl]
        return max(-ln["dA"], 0) + e + max(ln["dB"], 0)

    def R_(sl, e):                                                      # right side of strip sl-1 at line sl
        ln = lines[sl]
        return max(ln["dA"], 0) + e + max(-ln["dB"], 0)

    cand = [np.arange(max(1, int(round(ln["w"])) - 2), int(round(ln["w"])) + 3) for ln in lines]
    cost = [lam * (cand[0] - lines[0]["w"]) ** 2]
    back = []
    for sl in range(1, S + 1):
        c_prev, c_cur = cand[sl - 1], cand[sl]
        d = np.abs(np.array([[R_(sl, e1) - L_(sl - 1, e0) for e1 in c_cur] for e0 in c_prev], dtype=float))
        # B58 (the quantised join): a row termination in the end_rows strips at either cap costs END_PENALTY more
        pen = 1.0 + END_PENALTY * (pairing != "even" and (sl - 1 < r0 or sl - 1 >= S - r1))
        tot = cost[-1][:, None] + np.where(d <= 2, pen * d, np.inf) + lam * (c_cur[None, :] - lines[sl]["w"]) ** 2
        back.append(np.argmin(tot, axis=0))
        cost.append(tot.min(axis=0))
    if not np.isfinite(cost[-1]).any():
        return "no row counts with <= 2 terminations per column"
    j = int(np.argmin(cost[-1]))
    es = [0] * (S + 1)
    for sl in range(S, 0, -1):
        es[sl] = int(cand[sl][j])
        j = int(back[sl - 1][j])
    es[0] = int(cand[0][j])
    for sl, ln in enumerate(lines):
        ln["e"] = es[sl]
    terms = [abs(R_(sl + 1, es[sl + 1]) - L_(sl, es[sl])) for sl in range(S)]
    ends = [t_ for sl, t_ in enumerate(terms) if sl < END_MARGIN or sl >= S - END_MARGIN]      # (reported)
    ends = ends if len(terms) > 2 * END_MARGIN else terms
    ws = np.array([ln["w"] for ln in lines])
    return {"strips": strips, "lines": lines, "m0": es[0], "m1": es[S], "row_terms": int(sum(terms)),
            "col_terms": k, "rows_min": int(min(es)), "rows_max": int(max(es)),
            "rows_err_p90": float(np.percentile(np.abs(np.array(es) - ws), 90)), "off_sides": pairing != "even",
            "end_row_terms": int(sum(ends))}


def burgers_vector(ca: np.ndarray, cbv: np.ndarray) -> tuple[int, int]:
    """Burgers vector b = (b_par, b_perp) of a corridor from its two lattice-path runs (A's edge classes ca and B's
    reversed, cbv, both from cap C0 to cap C1, +1 = a step up, away from A): b_par = nA_along - nB_along (column
    ends), b_perp = the net up-steps of A minus those of B (rows ending; 2 L sin(theta / 2) for a rotation theta).
    The caps' share of the circuit is left out (n_min = |b|_1)."""
    return int((ca == 0).sum() - (cbv == 0).sum()), int(ca.sum() - cbv.sum())


def quantise(b, width: float) -> str:
    """The corridor's quantisation (decisions B47): "none" for b = 0 (a ladder without terminations),
    "shear" for b_perp = 0 and |b_par| <= the width in cells (one integer transition; the mismatch as a shear, its
    column ends at the corridor ends), else "even" (|b|_1 dislocations evenly on the middle row, B58)."""
    bpar, bperp = b
    if bpar == 0 and bperp == 0:
        return "none"
    if bperp == 0 and abs(bpar) <= width:
        return "shear"
    return "even"


def _cap_chain(part: Partition, sep_ids: list, start: int, end: int):
    """The separators of a region joining node start to node end through interior branch nodes, in order; None if they
    do not form one such chain."""
    path, cur = [], start
    left = set(sep_ids)
    while True:
        nxt = [i for i in left if cur in (int(part.chains[i]["nodes"][0]), int(part.chains[i]["nodes"][-1]))]
        if len(nxt) != 1:
            return None
        i = nxt[0]
        left.discard(i)
        path.append(i)
        c = part.chains[i]["nodes"]
        cur = int(c[-1]) if int(c[0]) == cur else int(c[0])
        if cur == end:
            return path
        if cur < part.nbnd:                                             # reached another outline / coast node
            return None


def _beyond(part: Partition, r: int, ci: int) -> int:
    """The region on the other side of separator ci from region r."""
    c = part.chains[ci]
    return c["right"] if c["left"] == r else c["left"]


def _end_trim(part: Partition, r: int, A, Bs, ca, cb, cap0: list, cap1: list):
    """A step at a run end that runs along its cap (A starting toward B or ending away from it, the same for B) would
    overlap the cap: those end edges go to the region beyond the cap (beyond its first piece at A, its last piece at
    B), the cap moves to the next node. Returns (A, Bs, ca, cb trimmed, the moved edges (a, b, region)) or None (a
    side is all steps)."""
    def beyond(ci):
        c = part.chains[ci]
        return c["right"] if c["left"] == r else c["left"]

    other = {"A0": beyond(cap0[0]), "B0": beyond(cap0[-1]), "A1": beyond(cap1[0]), "B1": beyond(cap1[-1])}
    a0 = int(np.argmax(ca != 1))                                        # leading A steps toward B -> beyond C0
    a1 = len(ca) - int(np.argmax(ca[::-1] != -1))                       # trailing A steps away -> beyond C1
    b0 = int(np.argmax(cb != 1))                                        # B's first edges (C1 end) toward A
    b1 = len(cb) - int(np.argmax(cb[::-1] != -1))                       # B's last edges (C0 end) away from A
    if a0 >= a1 or b0 >= b1:
        return None
    moved = [(int(A[j]), int(A[j + 1]), other["A0"]) for j in range(a0)]
    moved += [(int(A[j]), int(A[j + 1]), other["A1"]) for j in range(a1, len(ca))]
    moved += [(int(Bs[j]), int(Bs[j + 1]), other["B1"]) for j in range(b0)]
    moved += [(int(Bs[j]), int(Bs[j + 1]), other["B0"]) for j in range(b1, len(cb))]
    return A[a0:a1 + 1], Bs[b0:b1 + 1], ca[a0:a1], cb[b0:b1], moved


def corridor_geometry(part: Partition, h: SizeField, o_lam: float = 1.0, obl_max: float = 20.0,
                      join: str = "rule", end_rows: int = END_MARGIN) -> None:
    """For each region: side runs, strip eligibility (bg level), nA, nB, rotation, obliquity, width, the strip plan;
    the Burgers vector b of a corridor whose sides are lattice paths (of the trimmed runs once a strip plan is made)
    and its quantisation. With join = "quantised" the plan's column terminations follow the quantisation (B58)."""
    P = part.bg.points
    for rec in part.regions:
        r = rec["id"]
        runs = _runs(part, r)
        sep_ids = [i for i, c in enumerate(part.chains) if (c["left"] == r) != (c["right"] == r)]
        seps = [part.chains[i] for i in sep_ids]
        rec["n_caps"] = len(seps)
        rec["strip_ok"] = False
        if rec["cls"] != "corridor":
            continue
        oa, ob = rec["objs"]
        ra, rb = runs.get(oa), runs.get(ob)
        if ra is None or rb is None:
            rec["why_not_strip"] = "a side has no outline edge"
            continue
        offs = np.r_[0, np.cumsum([len(lp) for lp in part.loops])]
        sides = []
        for li, idx in (ra, rb):
            K = len(part.loops[li])
            run = _contiguous(idx, K)
            if run is None:
                sides.append(None)
                continue
            s, n = run
            sides.append(offs[li] + (s + np.arange(n + 1)) % K)              # node ids (fill ids), loop order
        rec["nA"], rec["nB"] = len(ra[1]), len(rb[1])
        xa = P[sides[0]] if sides[0] is not None else None
        xb = P[sides[1]] if sides[1] is not None else None
        if xa is not None and xb is not None and len(xa) >= 3 and len(xb) >= 3:
            rec["rot_deg"], rec["rot_coh"] = _rotation(xa, xb)
            rec["obl_a"], rec["obl_b"] = _obliquity(xa), _obliquity(xb)
            from scipy.spatial import cKDTree

            dd, j = cKDTree(xb).query(xa)
            rec["width"] = float(np.median(R_EARTH_M * arc_length(xa, xb[j]) / h(_unit(xa + xb[j]))))
        else:
            rec["why_not_strip"] = "a side is not one contiguous run"
            continue
        rec["steps_a"], rec["steps_b"] = _corners(xa), _corners(xb)
        ca, cb = _edge_classes(xa), _edge_classes(xb)
        rec["nA_along"] = int((ca == 0).sum()) if ca is not None else None
        rec["nB_along"] = int((cb == 0).sum()) if cb is not None else None
        nal = [x for x in (rec["nA_along"], rec["nB_along"]) if x is not None]
        L = 0.5 * (rec["nA_along"] + rec["nB_along"]) if len(nal) == 2 else 0.5 * (rec["nA"] + rec["nB"])
        rec["mismatch"] = abs(rec["nA_along"] - rec["nB_along"]) if len(nal) == 2 else abs(rec["nA"] - rec["nB"])
        rec["mismatch_per10"] = 10.0 * rec["mismatch"] / L
        if ca is None or cb is None:
            rec["why_not_strip"] = "a side is not a monotone lattice path"
            continue
        inner_sep = any(c["left"] == r and c["right"] == r for c in part.chains)
        if join == "quantised":
            rec["b"] = burgers_vector(ca, cb[::-1])
            rec["quant"] = quantise(rec["b"], rec["width"])
            rec["cont_ok"] = not inner_sep                              # the continuation's loop is the region's
            c0 = _cap_chain(part, sep_ids, int(sides[0][0]), int(sides[1][-1]))
            c1 = _cap_chain(part, sep_ids, int(sides[0][-1]), int(sides[1][0]))
            if (not inner_sep and c0 and c1 and not set(c0) & set(c1) and set(c0) | set(c1) == set(sep_ids)):
                rec["_cont_caps"] = [c0, c1]                            # two caps joining the runs: lattice caps (B57)
                rec["_sides"] = sides
                rec["_cont_end_rows"] = tuple(end_rows * any(part.regions[_beyond(part, r, x)]["cls"] != "corridor"
                                                             for x in cap) for cap in (c0, c1))
        if max(rec["obl_a"], rec["obl_b"]) >= obl_max:
            rec["why_not_strip"] = "a lattice oblique to the gap"
            continue
        if inner_sep:
            rec["why_not_strip"] = "a separator inside the region"
            continue
        A, Bs = sides
        cap0 = _cap_chain(part, sep_ids, int(A[0]), int(Bs[-1]))        # C0: from A's start to B's end
        cap1 = _cap_chain(part, sep_ids, int(A[-1]), int(Bs[0]))        # C1: from A's end to B's start
        if cap0 is None or cap1 is None or set(cap0) & set(cap1) or set(cap0) | set(cap1) != set(sep_ids):
            rec["why_not_strip"] = "the separators do not form two caps joining the runs"
            continue
        rec["cap_pieces"] = [len(cap0), len(cap1)]
        tr = _end_trim(part, r, A, Bs, ca, cb, cap0, cap1)
        if tr is None:
            rec["why_not_strip"] = "a side is all steps"
            continue
        A, Bs, ca, cb, moved = tr
        other = {"A0": _beyond(part, r, cap0[0]), "A1": _beyond(part, r, cap1[0])}
        xa, xb = P[A], P[Bs]
        rec["trimmed_edges"] = len(moved)
        pairing = "even"
        if join == "quantised":
            rec["b"] = burgers_vector(ca, cb[::-1])
            rec["quant"] = quantise(rec["b"], rec["width"])
            pairing = {"shear": "ends"}.get(rec["quant"], "spread")
        er = (0, 0)
        if join == "quantised":                                         # B58: the caps facing a junction, coast, open
            er = rec["_end_rows"] = tuple(end_rows * (part.regions[other[k_]]["cls"] != "corridor")
                                          for k_ in ("A0", "A1"))
        plan = _strip_plan(ca, cb[::-1], xa, xb[::-1], h, o_lam, pairing, er)
        if isinstance(plan, str):
            rec["why_not_strip"] = plan
            continue
        rec["strip_ok"] = True
        rec.update(rows=plan["m0"], rows_end=plan["m1"], rows_min=plan["rows_min"], rows_max=plan["rows_max"],
                   row_terms=plan["row_terms"], col_terms=plan["col_terms"], rows_err_p90=plan["rows_err_p90"])
        rec["_plan"] = plan
        rec["_A"], rec["_B"] = A, Bs
        rec["_caps"] = [cap0, cap1]                                     # piece chains, in order from side A
        rec["_cap_rows"] = [plan["m0"], plan["m1"]]
        rec["_cap_ends"] = [(int(A[0]), int(Bs[-1])), (int(A[-1]), int(Bs[0]))]
        rec["_moved"] = moved


def resample_separators(part: Partition, h: SizeField, o: dict, grids: list | None = None) -> None:
    """Smooth and resample every separator chain; fill ids for its nodes; constraint edges and known half-edges.

    The two caps of a strip corridor become chords between its (trimmed) run ends; a cap made of several separators is
    one chord with its interior branch nodes moved onto it (at their arc fraction; the other separators ending there
    follow), its pieces resampled with counts summing to the strip's row count there. A corridor whose chords or moved
    separators would cross another constraint, or that shares a cap or a branch node with another strip, is no strip
    (and the resampling is redone without it). With lattice_caps (and the blocks given), each single-separator cap of a
    continuation corridor (B57) becomes its lattice cap (nodes at the blended map's integer points, their labels kept
    for the fill); one that would cross a constraint or share a cap with a strip keeps its separator."""
    strips = [rec for rec in part.regions if rec.get("strip_ok")]
    part.info["strip_caps_crossing"] = 0
    while True:
        res = _resample_try(part, h, o, strips)
        if res is True:
            break
        rec, why = res
        rec["strip_ok"] = False
        rec["why_not_strip"] = why
        part.info["strip_caps_crossing"] += why.startswith("a straight cap")
        strips = [r for r in strips if r is not rec]
    if grids is None or not o.get("lattice_caps"):
        return
    conts = []
    for rec in part.regions:                                            # the continuation corridors' lattice caps
        if not _wants_cont(rec, o):
            continue
        if "_cont_caps" not in rec:
            rec["lat_caps"] = "the separators do not form two caps"
            continue
        if rec.get("_no_lat_caps"):
            rec["lat_caps"] = rec["_no_lat_caps"]
            continue
        if "_lat_caps" not in rec:
            rec["_lat_caps"] = _lattice_caps(part, rec, grids, o)
        if rec["_lat_caps"] is None:
            rec["_no_lat_caps"] = rec["lat_caps"] = "no lattice path"
        else:
            rec["lat_caps"] = "ok"
            conts.append(rec)
    part.info["lattice_caps_dropped"] = part.info.get("lattice_caps_dropped", 0)
    while conts:
        res = _resample_try(part, h, o, strips, conts)
        if res is True:
            return
        rec, why = res
        if rec.get("_lat_variant", 0) < 1:                              # the other lattice path, then the separators
            rec["_lat_variant"] = 1
            rec["_lat_caps"] = _lattice_caps(part, rec, grids, o)
            if rec["_lat_caps"] is not None:
                continue
        rec["_no_lat_caps"] = rec["lat_caps"] = why
        part.info["lattice_caps_dropped"] += 1
        conts = [r for r in conts if r is not rec]
    _resample_try(part, h, o, strips)


def _cont_opts(o: dict, rec: dict | None = None) -> dict:
    """The continuation's options: o["cont"] over its end margins from end_rows (B58: at the caps that border a
    junction, the coast or an open region; both when the caps are not known)."""
    m = rec.get("_cont_end_rows", (o["end_rows"], o["end_rows"])) if rec is not None else (o["end_rows"],) * 2
    return {"end_margin": tuple(float(x) for x in m), **o["cont"]}


def _wants_cont(rec: dict, o: dict) -> bool:
    """classify's choice of the continuation (join = "quantised")."""
    if (rec["cls"] != "corridor" or rec["coast"] or o["force_fill"] or o["join"] != "quantised"
            or not rec.get("cont_ok") or rec.get("cont_failed")):
        return False
    obl = max(rec.get("obl_a", 90.0), rec.get("obl_b", 90.0))
    if obl < o["obl_strip_deg"] and rec.get("strip_ok"):
        return False
    return obl >= o["obl_strip_deg"] or bool(o["cont_fallback"])


def _block_run(part: Partition, obj: int, ids: np.ndarray, grids: list) -> np.ndarray:
    """Block node ids of the fill ids of block obj's outline nodes."""
    off = sum(len(lp) for lp in part.loops[:part.C + obj])
    return grids[obj].outline[np.asarray(ids) - off]


def _lattice_caps(part: Partition, rec: dict, grids: list, o: dict):
    """The continuation's corridor and schedule from the region's sides, and its two lattice caps
    (continuation.lattice_cap): [(xyz, u, v) of C0, of C1] (each from side A's end to side B's), None if a cap fails.
    A run end edge that would overlap its cap's first across step (A starting toward B, or B ending toward A, in the
    blended map's indices) goes to the region beyond the cap (rec["_cont_moved"]), the cap starts at the next node;
    the corridor is then made again from the trimmed sides (rec["_cont_cor"])."""
    from meshom.methods.m14 import continuation

    A, Bs = (np.asarray(x) for x in rec["_sides"])
    oa, ob = rec["objs"]
    caps_ci = rec["_cont_caps"]

    def beyond(ci):
        c = part.chains[ci]
        return c["right"] if c["left"] == rec["id"] else c["left"]

    moved = []
    for _ in range(6):
        if len(A) < 4 or len(Bs) < 4:
            return None
        try:
            cor = continuation.Corridor(grids[oa], _block_run(part, oa, A, grids), grids[ob],
                                        _block_run(part, ob, Bs, grids), _cont_opts(o, rec))
        except np.linalg.LinAlgError:
            return None
        sched = continuation.corridor_schedule(cor)
        Ts = continuation.segment_T(*sched)
        cut = False
        for end, T in ((0, Ts[0]), (1, Ts[-1])):
            ia, ib = (0, -1) if end == 0 else (-1, 0)
            D = np.round(cor.B.lk[ib] + T - cor.A.lk[ia])
            ac = int(np.argmax(np.abs(D)))
            e0 = np.zeros(2)
            e0[ac] = np.sign(D[ac])
            sa = cor.A.lk[1] - cor.A.lk[0] if end == 0 else cor.A.lk[-2] - cor.A.lk[-1]   # from the run end inward
            sb = cor.B.lk[-2] - cor.B.lk[-1] if end == 0 else cor.B.lk[1] - cor.B.lk[0]
            if np.array_equal(sa, e0):                                  # A's end edge runs up the cap
                k = (0, 1) if end == 0 else (len(A) - 2, len(A) - 1)
                moved.append((int(A[k[0]]), int(A[k[1]]), beyond(caps_ci[end][0])))
                A = A[1:] if end == 0 else A[:-1]
                cut = True
            if np.array_equal(sb, -e0):                                 # B's end edge runs down the cap
                k = (len(Bs) - 2, len(Bs) - 1) if end == 0 else (0, 1)
                moved.append((int(Bs[k[0]]), int(Bs[k[1]]), beyond(caps_ci[end][-1])))
                Bs = Bs[:-1] if end == 0 else Bs[1:]
                cut = True
        if not cut:
            break
    else:
        return None
    caps = [continuation.lattice_cap(cor, sched, e, rec.get("_lat_variant", 0)) for e in (0, 1)]
    if any(c is None for c in caps):
        return None
    rec["_cont_cor"] = (cor, sched, A.copy(), Bs.copy())
    rec["_cont_moved"] = moved
    rec["_cont_ends"] = [(int(A[0]), int(Bs[-1])), (int(A[-1]), int(Bs[0]))]
    return caps


def _cap_order(chains: list, cap: list, P: np.ndarray):
    """A composite cap's pieces in order from its side-A end: (piece ids, the interior branch nodes between them, their
    arc fractions along the cap)."""
    first = chains[cap[0]]["nodes"]
    cur = int(first[0]) if len(cap) == 1 or int(first[-1]) in (
        int(chains[cap[1]]["nodes"][0]), int(chains[cap[1]]["nodes"][-1])) else int(first[-1])
    lens, bn = [], []
    for k, ci in enumerate(cap):
        nd = chains[ci]["nodes"]
        nd = nd if int(nd[0]) == cur else nd[::-1]
        lens.append(np.linalg.norm(np.diff(P[nd], axis=0), axis=1).sum())
        cur = int(nd[-1])
        if k < len(cap) - 1:
            bn.append(cur)
    fr = np.cumsum(lens)[:-1] / np.sum(lens)
    return list(cap), bn, fr


def _resample_try(part: Partition, h: SizeField, o: dict, strips: list, conts: list = ()):
    P = part.bg.points
    nbnd = part.nbnd
    chains = part.chains
    for c in chains:                                                    # undo the shapes of an earlier attempt
        c.pop("ids", None)
        c.pop("lat", None)
        if "nodes0" in c:
            c["nodes"], c["left"], c["right"] = c["nodes0"], c["left0"], c["right0"]
        else:
            c["nodes0"], c["left0"], c["right0"] = c["nodes"], c["left"], c["right"]
        c["straight"] = False
    # 1. strip caps: chords, branch nodes moved onto them, piece counts
    piece = {}                                                          # chain id -> (u, v, n, strip rec, side)
    move = {}                                                           # bg branch node -> new position
    mover = {}                                                          # bg branch node -> strip rec
    for rec in strips:
        for side, (cap, m, (e0, e1)) in enumerate(zip(rec["_caps"], rec["_cap_rows"], rec["_cap_ends"])):
            seq, pts = [], []
            # the composite cap's bg chains in order from e0 (the run end of side A, perhaps before trimming)
            first = chains[cap[0]]["nodes"]
            cur = int(first[0]) if len(cap) == 1 or int(first[-1]) in (
                int(chains[cap[1]]["nodes"][0]), int(chains[cap[1]]["nodes"][-1])) else int(first[-1])
            for ci in cap:
                nd = chains[ci]["nodes"]
                nd = nd if int(nd[0]) == cur else nd[::-1]
                seq.append(ci)
                pts.append(P[nd])
                cur = int(nd[-1])
            if len(cap) > m:
                return rec, "fewer rows than cap pieces"
            lens = np.array([np.linalg.norm(np.diff(x, axis=0), axis=1).sum() for x in pts])
            fr = np.r_[0.0, np.cumsum(lens) / lens.sum()]
            p0, p1 = P[e0], P[e1]
            ends_ = [e0]
            for k_, ci in enumerate(seq[:-1]):
                nd = chains[ci]["nodes"]
                nxt = chains[seq[k_ + 1]]["nodes"]
                b = int(nd[0]) if int(nd[0]) in (int(nxt[0]), int(nxt[-1])) else int(nd[-1])
                if b < nbnd or b in mover:
                    return rec, "a cap branch node shared with another strip"
                move[b] = _unit((1 - fr[k_ + 1]) * p0 + fr[k_ + 1] * p1)
                mover[b] = rec
                ends_.append(b)
            ends_.append(e1)
            cnt = np.maximum(1, np.round(np.diff(fr) * m).astype(np.int64))
            while cnt.sum() > m:
                cnt[np.argmax(cnt)] -= 1
            while cnt.sum() < m:
                cnt[np.argmin(cnt)] += 1
            for ci, u, v, n in zip(seq, ends_[:-1], ends_[1:], cnt):
                if ci in piece:
                    return rec, "cap shared with another strip"
                piece[ci] = (u, v, int(n), rec, side)

    # 1b. lattice caps of the continuation corridors: the cap's lattice path, its interior branch nodes moved onto the
    # path node at their arc fraction (the separators ending there follow), each piece the path between them
    latc = {}                                           # chain id -> (u, v, xyz, (lu, lv), rec, side, end labels)
    for rec in conts:
        for side, (cap, (e0, e1)) in enumerate(zip(rec["_cont_caps"], rec["_cont_ends"])):
            if any(ci in piece or ci in latc for ci in cap):
                return rec, "a lattice cap shared with a strip or another corridor"
            xyz, lu, lv = rec["_lat_caps"][side]
            X = np.vstack([P[e0], xyz, P[e1]])
            nan = np.full((1, 2), np.nan)
            LU, LV = np.vstack([nan, lu, nan]), np.vstack([nan, lv, nan])
            n = len(X) - 1
            seq, bn, fr = _cap_order(chains, cap, P)
            j = np.clip(np.round(fr * n).astype(np.int64), 1, n - 1)
            if len(bn) and (np.diff(np.r_[0, j, n]) < 1).any():
                return rec, "fewer lattice cap nodes than cap pieces"
            for b, jk in zip(bn, j):
                if b < nbnd or b in mover:
                    return rec, "a lattice cap branch node shared with another cap"
                move[b] = X[jk]
                mover[b] = rec
            idx = np.r_[0, j, n].astype(np.int64)
            nodes_ = [int(e0)] + [int(b) for b in bn] + [int(e1)]
            for k, ci in enumerate(seq):
                a_, b_ = idx[k], idx[k + 1]
                el = ((LU[a_], LV[a_]) if 0 < a_ < n else None, (LU[b_], LV[b_]) if 0 < b_ < n else None)
                latc[ci] = (nodes_[k], nodes_[k + 1], X[a_ + 1:b_], (LU[a_ + 1:b_], LV[a_ + 1:b_]), rec, side, el)

    def pos_of(v):
        return move.get(v, P[v])

    def polyline(c):
        nodes = c["nodes"]
        closed = nodes[0] == nodes[-1]
        x = P[nodes].copy()
        x[0], x[-1] = pos_of(int(nodes[0])), pos_of(int(nodes[-1]))
        for _ in range(o["sep_smooth"]):
            if closed:
                x[:-1] = _unit(0.5 * x[:-1] + 0.25 * (np.roll(x[:-1], 1, axis=0) + np.roll(x[:-1], -1, axis=0)))
                x[-1] = x[0]
            elif len(x) > 2:
                x[1:-1] = _unit(0.5 * x[1:-1] + 0.25 * (x[:-2] + x[2:]))
        return x

    def place(x, n):
        hv = h(x)
        wgt = R_EARTH_M * arc_length(x[:-1], x[1:]) * 0.5 * (1.0 / hv[:-1] + 1.0 / hv[1:])
        n = n if n else max(3 if np.array_equal(x[0], x[-1]) else 1, int(round(wgt.sum())))
        return (_place(x, wgt, np.arange(1, n) * (wgt.sum() / n)) if n > 1 else np.zeros((0, 3))), float(wgt.sum())

    # 2. every chain's polyline: chords for the cap pieces, smoothed bg chains (moved ends) for the rest
    pos, xs = {}, {}
    for ci, c in enumerate(chains):
        if ci in piece:
            u, v, n, rec, side = piece[ci]
            a, b = pos_of(u), pos_of(v)
            x = _unit(a[None] + np.linspace(0.0, 1.0, 33)[:, None] * (b - a)[None])
            pos[ci] = place(x, n)
            # orientation u -> v (from side A): the strip lies right of C0's pieces and left of C1's
            other = c["right"] if c["left"] == rec["id"] else c["left"]
            c["left"], c["right"] = (other, rec["id"]) if side == 0 else (rec["id"], other)
            c["nodes"] = np.array([u, v], dtype=np.int64)
            c["straight"] = True
        elif ci in latc:
            u, v, xyz, lab, rec, side, el = latc[ci]
            x = np.vstack([pos_of(u), xyz, pos_of(v)])
            hv = h(x)
            pos[ci] = xyz, float(np.sum(R_EARTH_M * arc_length(x[:-1], x[1:]) * 0.5 * (1.0 / hv[:-1] + 1.0 / hv[1:])))
            other = c["right"] if c["left"] == rec["id"] else c["left"]
            c["left"], c["right"] = (other, rec["id"]) if side == 0 else (rec["id"], other)
            c["nodes"] = np.array([u, v], dtype=np.int64)
            c["straight"] = True
            c["lat"], c["lat_end"] = lab, el
        else:
            pos[ci] = place(polyline(c), 0)
        xs[ci] = np.vstack([pos_of(int(c["nodes"][0])), pos[ci][0], pos_of(int(c["nodes"][-1]))])
    # 3. crossings: cap chords and the chains with moved ends against every other constraint
    loop_segs = np.stack([np.vstack(part.loops), np.vstack([np.roll(lp, -1, axis=0) for lp in part.loops])], axis=1)
    segs = np.vstack([loop_segs] + [np.stack([xs[ci][:-1], xs[ci][1:]], axis=1) for ci in range(len(chains))])
    seg_of = np.concatenate([np.full(len(loop_segs), -1)] + [np.full(len(xs[ci]) - 1, ci)   # (no chains: one region)
                                                             for ci in range(len(chains))]).astype(np.int64)
    smid = segs.sum(axis=1)
    slen = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1).max(initial=0.0)

    def crossed(x, me):
        """The owners (chain id, -1 = a loop) of the constraint segments the polyline x crosses."""
        a, b = x[:-1], x[1:]
        near = (np.linalg.norm(0.5 * smid - 0.5 * (x[0] + x[-1]), axis=1)
                < 0.5 * np.linalg.norm(np.diff(x, axis=0), axis=1).sum() + 2 * slen) & (seg_of != me)
        c_, d_ = segs[near, 0], segs[near, 1]                           # (far segments: antipodal great-circle hits)

        def orient(p_, q_, r_):
            return np.einsum("...j,...j->...", np.cross(q_ - p_, r_ - p_), p_)

        A_, B_ = a[:, None], b[:, None]
        o1 = orient(A_, B_, c_[None]) * orient(A_, B_, d_[None])
        o2 = orient(c_[None], d_[None], A_) * orient(c_[None], d_[None], B_)
        return set(seg_of[near][((o1 < -1e-18) & (o2 < -1e-18)).any(axis=0)].tolist())

    def owner(ci):                                                      # the corridor that reshaped chain ci
        if ci in piece:
            return piece[ci][3]
        if ci in latc:
            return latc[ci][4]
        ends2 = (int(chains[ci]["nodes"][0]), int(chains[ci]["nodes"][-1]))
        return mover.get(ends2[0], mover.get(ends2[1]))

    for ci, c in enumerate(chains):
        if owner(ci) is None:
            continue
        hit = crossed(xs[ci], ci)
        if hit:                                                         # a lattice cap's corridor goes first
            lat = [owner(x) for x in hit if x >= 0 and x in latc] + ([owner(ci)] if ci in latc else [])
            lat += [owner(x) for x in hit if x >= 0 and any(owner(x) is r for r in conts)]
            why = "a lattice cap crosses another separator" if lat else "a straight cap crosses another separator"
            return (lat[0] if lat else owner(ci)), why
    # 4. node ids
    ends = {}
    xyz = []

    def node_id(v):
        if v < nbnd:
            return int(v)
        if v not in ends:
            ends[v] = nbnd + len(xyz)
            xyz.append(pos_of(v))
        return ends[v]

    for ci, c in enumerate(chains):
        nodes = c["nodes"]
        closed = nodes[0] == nodes[-1]
        inner, c["length_h"] = pos[ci]
        a = node_id(int(nodes[0]))
        ids = [a]
        for p_ in inner:
            ids.append(nbnd + len(xyz))
            xyz.append(p_)
        ids.append(a if closed else node_id(int(nodes[-1])))
        c["ids"] = np.asarray(ids, dtype=np.int64)
    hit = _coincident_chains(np.vstack([np.vstack(part.loops), np.asarray(xyz).reshape(-1, 3)]), chains, nbnd)
    own = [owner(ci) for ci in hit if ci in latc] + [owner(ci) for ci in hit if any(owner(ci) is r for r in conts)]
    own += [owner(ci) for ci in hit if owner(ci) is not None]
    if own:                                             # B66: a reshaped chain whose node coincides with another node
        part.info["caps_touching"] = part.info.get("caps_touching", 0) + 1
        return own[0], ("a lattice cap touches another separator" if any(own[0] is r for r in conts)
                        else "a straight cap touches another separator")
    override = {}
    for rec in strips:
        rec["_C"] = [np.concatenate([chains[ci]["ids"][:-1] for ci in cap] + [chains[cap[-1]]["ids"][-1:]])
                     for cap in rec["_caps"]]
        for a_, b_, rr in rec["_moved"]:
            override[(a_, b_)] = rr
    for rec in conts:                                                   # the lattice caps' trimmed run ends
        for a_, b_, rr in rec["_cont_moved"]:
            override[(a_, b_)] = rr
    part.sep_xyz = np.asarray(xyz).reshape(-1, 3)
    N = nbnd + len(part.sep_xyz)
    # constraint edges: loop edges, then separator edges; known half-edges with their left region
    le = loop_edges(part.loops)
    Tb = part.bg.tris
    M = len(P) + N
    hk = Tb.ravel() * M + np.roll(Tb, -1, axis=1).ravel()               # bg half-edges (boundary ids < nbnd agree)
    t_of = np.repeat(np.arange(len(Tb)), 3)
    srt = np.argsort(hk)
    lk = le[:, 0] * M + le[:, 1]
    ps = np.minimum(np.searchsorted(hk[srt], lk), len(hk) - 1)
    ok = hk[srt][ps] == lk
    he_k = [le[ok, 0] * N + le[ok, 1]]
    he_r = [part.reg[t_of[srt[ps[ok]]]]]
    he_s = [np.zeros(int(ok.sum()), dtype=bool)]
    if override:                                                        # trimmed run ends: the region beyond the cap
        key = {a_ * N + b_: rr for (a_, b_), rr in override.items()}
        he_r[0] = np.array([key.get(int(x), int(y)) for x, y in zip(he_k[0], he_r[0])], dtype=np.int64)
    part.he_loop = (le[ok], he_r[0])                                    # loop half-edges and their regions
    se = []
    for c in chains:
        i = c["ids"]
        e = np.column_stack([i[:-1], i[1:]])
        se.append(e)
        he_k += [e[:, 0] * N + e[:, 1], e[:, 1] * N + e[:, 0]]
        he_r += [np.full(len(e), c["left"]), np.full(len(e), c["right"])]
        he_s += [np.ones(2 * len(e), dtype=bool)]
    part.cons = np.vstack([le] + se).astype(np.int64)
    part.n_loop_edges = len(le)
    part.he_keys = np.concatenate(he_k).astype(np.int64)
    part.he_reg = np.concatenate(he_r).astype(np.int64)
    part.he_sep = np.concatenate(he_s)
    part.N = N
    return True


def merge_same_fill(part: Partition, on: bool = True) -> None:
    """Fill regions: adjacent non-strip regions with the same fill become one (merge_same_fill; a separator between two
    m02 regions only adds fronts that collide), named by their smallest region id; the separators inside a fill region
    are dropped (nodes, constraint edges, half-edges) and the half-edges relabelled by fill region (part.group)."""
    R = len(part.regions)
    parent = np.arange(R)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    fills = [r["fill"] for r in part.regions]
    for c in part.chains:
        a, b = c["left"], c["right"]
        if on and fills[a] == fills[b] and fills[a] not in ("strip", "cont"):
            ra, rb = find(a), find(b)
            parent[max(ra, rb)] = min(ra, rb)
    group = np.array([find(r) for r in range(R)], dtype=np.int64)
    part.group = group
    nbnd, N = part.nbnd, part.N
    kept = [c for c in part.chains if group[c["left"]] != group[c["right"]]]
    for c in part.chains:
        c["dropped"] = group[c["left"]] == group[c["right"]]
    used = np.zeros(N, dtype=bool)
    used[:nbnd] = True
    for c in kept:
        used[c["ids"]] = True
    m = np.cumsum(used) - 1
    part.sep_xyz = part.sep_xyz[used[nbnd:]]
    N2 = int(used.sum())
    for c in kept:
        c["ids"] = m[c["ids"]]
    for rec in part.regions:
        if "_C" in rec:
            rec["_C"] = [m[x] for x in rec["_C"]]
    le, lr = part.he_loop
    he_k = [le[:, 0] * N2 + le[:, 1]]
    he_r = [group[lr]]
    he_s = [np.zeros(len(le), dtype=bool)]
    se = []
    for c in kept:
        i = c["ids"]
        e = np.column_stack([i[:-1], i[1:]])
        se.append(e)
        he_k += [e[:, 0] * N2 + e[:, 1], e[:, 1] * N2 + e[:, 0]]
        he_r += [np.full(len(e), group[c["left"]]), np.full(len(e), group[c["right"]])]
        he_s += [np.ones(2 * len(e), dtype=bool)]
    part.cons = np.vstack([part.cons[:part.n_loop_edges]] + se).astype(np.int64)
    part.he_keys = np.concatenate(he_k).astype(np.int64)
    part.he_reg = np.concatenate(he_r).astype(np.int64)
    part.he_sep = np.concatenate(he_s)
    part.N = N2
    part.info["fill_regions"] = int(len(np.unique(group)))
    part.info["separators_kept"] = len(kept)


# ------------------------------------------------------------------------------------------------------ the strip

def strip_patch(xyz: np.ndarray, A, Bv, C0, C1, plan: dict, first_new: int):
    """Ladder strip of a plan (_strip_plan): line s's free part = C0 (s = 0), C1 (s = S) or e_s - 1 new nodes evenly on
    the chord from side A's top node there to side B's bottom node; each strip between lines s and s+1 is a column of
    quads from its A edge(s) to its B edge(s), with its |R - L| row terminations as triangles in the middle rows and a
    doubled A (B) edge as one column termination: an extra line up from the middle A node (down from the middle B
    node) to the middle row, closed by one quad and one triangle. Returns (new points, quads, tris), new ids first_new,
    ...; CCW when the region lies left of A."""
    A, Bv, C0, C1 = (np.asarray(x, dtype=np.int64) for x in (A, Bv, C0, C1))
    lines, strips = plan["lines"], plan["strips"]
    S = len(strips)
    new = []
    nid = [first_new]

    def add(p):
        new.append(p)
        nid[0] += 1
        return nid[0] - 1

    free = []
    for sl, ln in enumerate(lines):
        if sl == 0:
            free.append(C0)
        elif sl == S:
            free.append(C1)
        else:
            p0, p1 = xyz[A[ln["a_top"]]], xyz[Bv[ln["b_bot"]]]
            e = ln["e"]
            mid = [add(_unit((1 - f) * p0 + f * p1)) for f in np.arange(1, e) / e]
            free.append(np.r_[A[ln["a_top"]], mid, Bv[ln["b_bot"]]].astype(np.int64))

    def run(X, i, j):                                                   # nodes X[i..j] inclusive, either direction
        return X[i:j + 1] if i <= j else X[j:i + 1][::-1]

    quads, tris = [], []

    off_sides = plan.get("off_sides", False)
    side_nodes = set(A.tolist()) | set(Bv.tolist()) if off_sides else set()

    def ladder(Ls, Rs):
        """Quads between two node lines (bottom -> top) with |len diff| triangles in the middle rows; with off_sides
        each the nearest to the middle whose nodes are off both sides (a side's steps at a line end are outline nodes:
        B58)."""
        nl, nr = len(Ls) - 1, len(Rs) - 1
        d = nr - nl
        kabs = abs(d)
        nlow = min(nl, nr)
        pos, prev = [], 0
        for t in range(kabs):
            want = min(nlow, max(0, nlow // 2 - kabs // 2 + t))

            def free(p_):
                tri = (Ls[p_], Rs[p_ + t], Rs[p_ + t + 1]) if d > 0 else (Ls[p_ + t], Rs[p_], Ls[p_ + t + 1])
                return not any(int(v) in side_nodes for v in tri)

            ok = [p_ for p_ in range(prev, nlow + 1) if free(p_)] if off_sides else []
            pos.append(min(ok, key=lambda p_: abs(p_ - want)) if ok else max(want, prev))
            prev = pos[-1]
        i = j = t = 0
        while i < nl or j < nr:
            if d > 0 and t < kabs and (pos[t] <= i or i == nl):
                tris.append([Ls[i], Rs[j], Rs[j + 1]])
                j += 1
                t += 1
            elif d < 0 and t < kabs and (pos[t] <= j or j == nr):
                tris.append([Ls[i], Rs[j], Ls[i + 1]])
                i += 1
                t += 1
            else:
                quads.append([Ls[i], Rs[j], Rs[j + 1], Ls[i + 1]])
                i += 1
                j += 1

    def split(Ls, Rs, mid_node):
        """A strip with two bottom edges (Ls[0] - mid_node - Rs[0]): an extra line X from mid_node up to row tx, then
        one quad + one triangle, then the ladder above. Row tx is the middle row; with off_sides at least one row above
        a side's steps at the bottom of either line (the triangle never touches the outline: B58)."""
        tx = max(0, min(len(Ls), len(Rs)) // 2 - 1)
        if off_sides:
            steps = max(next((r for r, v in enumerate(X_) if int(v) not in side_nodes), len(X_)) for X_ in (Ls, Rs))
            tx = min(max(tx, steps), min(len(Ls), len(Rs)) - 2)
        X = [mid_node] + [add(_unit(0.5 * (xyz_all(Ls[r]) + xyz_all(Rs[r])))) for r in range(1, tx + 1)]
        for r in range(tx):
            quads.append([Ls[r], X[r], X[r + 1], Ls[r + 1]])
            quads.append([X[r], Rs[r], Rs[r + 1], X[r + 1]])
        quads.append([Ls[tx], X[tx], Rs[tx + 1], Ls[tx + 1]])
        tris.append([X[tx], Rs[tx], Rs[tx + 1]])
        ladder(Ls[tx + 1:], Rs[tx + 1:])

    def xyz_all(i):
        return xyz[i] if i < first_new else new[i - first_new]

    for sl, (eA, eB) in enumerate(strips):
        l0, l1 = lines[sl], lines[sl + 1]
        # left side of strip sl at line sl: A from the strip's bottom-left up to the line's A top, free part, B part
        Ls = np.r_[run(A, eA[0], l0["a_top"])[:-1], free[sl], run(Bv, l0["b_bot"], eB[0])[1:]].astype(np.int64)
        Rs = np.r_[run(A, eA[-1] + 1, l1["a_top"])[:-1], free[sl + 1], run(Bv, l1["b_bot"], eB[-1] + 1)[1:]]
        Rs = Rs.astype(np.int64)
        if len(eA) == 2:
            split(list(Ls), list(Rs), int(A[eA[1]]))
        elif len(eB) == 2:                                              # the same turned by 180 degrees
            split(list(Rs[::-1]), list(Ls[::-1]), int(Bv[eB[1]]))
        else:
            ladder(list(Ls), list(Rs))
    q = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    t = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    return (np.vstack(new) if new else np.zeros((0, 3))), q, t


def _strip_valid(pts: np.ndarray, quads: np.ndarray, tris: np.ndarray, quad_max: float = 170.0) -> bool:
    """No inverted element, no triangle angle >= 170 deg, no quad angle >= quad_max (the patch as built, before any
    smoothing: STRIP_QUAD_MAX for the ladder, B46)."""
    from meshom.quality.kernel import element_geometry

    m = Mesh(pts, tris, quads, np.zeros(len(pts), dtype=np.int8))
    if (signed_area(m) <= 0).any():
        return False
    g = element_geometry(m)
    qa, ta = np.degrees(g.quad_angles).max(initial=0.0), np.degrees(g.tri_angles).max(initial=0.0)
    return bool(qa < quad_max and ta < 170.0)


def _tri_max_deg(P: np.ndarray, t: np.ndarray) -> float:
    """Largest angle (deg) of the triangle t (3 node ids), inf if it is inverted (seen from outside)."""
    x = P[t]
    if np.dot(x[0], np.cross(x[1], x[2])) <= 0:
        return np.inf
    a = [np.arccos(np.clip(np.dot(_unit(x[(k + 1) % 3] - x[k]), _unit(x[(k + 2) % 3] - x[k])), -1.0, 1.0))
         for k in range(3)]
    return float(np.degrees(max(a)))


def _polygon_splits(p: np.ndarray):
    """The ways to split a polygon of 4-6 nodes (CCW) into quads and at most one triangle: [(quads, tris)]."""
    n = len(p)
    if n == 4:
        return [(p[None], np.zeros((0, 3), np.int64))]
    if n == 5:
        return [(np.array([[p[(i + 1) % 5], p[(i + 2) % 5], p[(i + 3) % 5], p[(i + 4) % 5]]]),
                 np.array([[p[(i + 4) % 5], p[i], p[(i + 1) % 5]]])) for i in range(5)]
    if n == 6:
        return [(np.array([[p[i], p[i + 1], p[i + 2], p[(i + 3) % 6]], [p[(i + 3) % 6], p[(i + 4) % 6],
                                                                         p[(i + 5) % 6], p[i]]]),
                 np.zeros((0, 3), np.int64)) for i in range(3)]
    return []


def break_clusters(P: np.ndarray, tris: np.ndarray, quads: np.ndarray, rt: np.ndarray, rq: np.ndarray,
                   quad_max: float = CLUSTER_QUAD_MAX, passes: int = 6, free_from: int | None = None,
                   tri_lim: tuple[float, float] | None = None):
    """Edge-connected clusters of >= 3 triangles (the tri_cluster_max gate). Per pass and cluster the best of:
    (a) two adjacent triangles merged into a quad; (b) a cluster of 3 or 4 whose union is a polygon of 5 or 6 nodes
    re-split into quads and at most one triangle; (c) an interior node (id >= free_from) with >= 2 of the cluster's
    triangles round it removed and its star (its elements' union, a polygon of 3-6 nodes) re-split so (a fan of 3
    triangles and a quad round a valence-4 node: a quad and a triangle; 4 triangles round one: a quad). Best = fewest
    triangles left in the cluster, then the squarest quads; every quad angle and triangle angle below quad_max; new
    elements in the region of the cluster's first triangle. Until every cluster has <= 2 triangles or nothing is left.
    tri_lim = (min, max) deg: every new triangle also keeps its angles in [min, max] (tangent plane, as the gates).
    Returns (tris, quads, rt, rq, the number of clusters changed, the removed nodes)."""
    from meshom.geo import tangent_coords
    from meshom.quality.kernel import _angles
    from meshom.recombine import _merge, _quad_cost

    lim = (quad_max - 90.0) / 90.0
    done, removed = 0, []
    for _ in range(passes):
        T = len(tris)
        if T < 3:
            break
        nbr = adjacency(tris)
        t_, k_ = np.nonzero(nbr > np.arange(T)[:, None])
        a, b = t_.astype(np.int64), nbr[t_, k_].astype(np.int64)
        lab = _components(T, a, b)
        size = np.bincount(lab)
        big = np.flatnonzero(size >= 3)
        if not len(big):
            break
        members = np.argsort(lab, kind="stable")
        split = np.searchsorted(lab[members], np.arange(len(size) + 1))
        n_nodes = int(max(tris.max(), quads.max() if len(quads) else 0)) + 1
        tri_of = [[] for _ in range(n_nodes)]                           # node -> triangles / quads
        quad_of = [[] for _ in range(n_nodes)]
        in_big = np.isin(lab, big)
        for t in np.flatnonzero(in_big):
            for v in tris[t]:
                tri_of[v].append(int(t))
        big_nodes = set(np.unique(tris[in_big]).tolist())
        for qi, q in enumerate(quads):
            for v in q:
                if int(v) in big_nodes:
                    quad_of[v].append(qi)
        for t in np.flatnonzero(~in_big):                               # triangles of other clusters at those nodes
            for v in tris[t]:
                if int(v) in big_nodes:
                    tri_of[v].append(int(t))
        cand = {}                                       # cluster -> (key, drop tris, drop quads, new q, new t, node)

        def offer(c, key, dt, dq, nq, nt, node):
            if c not in cand or key < cand[c][0]:
                cand[c] = (key, dt, dq, nq, nt, node)

        def valid(nq, nt):
            if not (float(_quad_cost(P, nq).max(initial=0.0)) < lim and all(_tri_max_deg(P, t) < quad_max for t in nt)):
                return False
            if tri_lim is None or not len(nt):
                return True
            ta = np.degrees(_angles(tangent_coords(P[np.asarray(nt, dtype=np.int64)])))
            return bool(ta.min() >= tri_lim[0] and ta.max() <= tri_lim[1])

        sel = size[lab[a]] >= 3
        if sel.any():                                                   # (a) pair merges
            q = _merge(tris[a[sel]], tris[b[sel]])
            cost = _quad_cost(P, q)
            for i in np.flatnonzero(cost < lim):
                c = int(lab[a[sel][i]])
                offer(c, (int(size[c]) - 2, float(cost[i])), [int(a[sel][i]), int(b[sel][i])], [], q[i:i + 1],
                      np.zeros((0, 3), np.int64), None)
        for c in big:
            m = members[split[c]:split[c + 1]]
            if len(m) <= 4:                                             # (b) the cluster's polygon
                lps = _region_loops(tris[m])
                if lps is not None and len(lps) == 1 and len(lps[0]) == len(m) + 2:
                    for nq, nt in _polygon_splits(lps[0]):
                        if valid(nq, nt):
                            offer(int(c), (len(nt), float(_quad_cost(P, nq).max())), m.tolist(), [], nq, nt, None)
            if free_from is None:
                continue
            for v in np.unique(tris[m]):                                # (c) remove an interior node
                tv, qv = tri_of[v], quad_of[v]
                if v < free_from or sum(lab[t] == c for t in tv) < 2 or len(tv) + len(qv) > 6:
                    continue
                if any(lab[t] != c for t in tv):
                    continue
                el = [tris[t] for t in tv] + [quads[x] for x in qv]
                lps = _boundary_loops(el)
                if lps is None or len(lps) != 1 or v in lps[0]:
                    continue
                pl = lps[0]
                opts = [(np.zeros((0, 4), np.int64), pl[None])] if len(pl) == 3 else _polygon_splits(pl)
                for nq, nt in opts:
                    if valid(nq, nt):
                        left = int(size[c]) - len(tv) + len(nt)
                        offer(int(c), (left, float(_quad_cost(P, nq).max(initial=0.0))), list(tv), list(qv), nq, nt,
                              int(v))
        if not cand:
            break
        drop_t = np.zeros(T, dtype=bool)
        drop_q = np.zeros(len(quads), dtype=bool)
        newq, newt, rnq, rnt = [], [], [], []
        used_q = set()
        for c, (key, dt, dq, nq, nt, node) in cand.items():
            if used_q & set(dq):                                        # two clusters sharing a quad: one per pass
                continue
            used_q |= set(dq)
            drop_t[dt] = True
            drop_q[dq] = True
            r0 = rt[members[split[c]]]
            newq.append(nq)
            newt.append(nt)
            rnq.append(np.full(len(nq), r0))
            rnt.append(np.full(len(nt), r0))
            if node is not None:
                removed.append(node)
            done += 1
        tris = np.vstack([tris[~drop_t]] + newt).astype(np.int64)
        rt = np.concatenate([rt[~drop_t]] + rnt).astype(np.int64)
        quads = np.vstack([quads[~drop_q]] + newq).astype(np.int64)
        rq = np.concatenate([rq[~drop_q]] + rnq).astype(np.int64)
    return tris, quads, rt, rq, done, removed


def _boundary_loops(elems: list):
    """Boundary loops of a set of elements (CCW node lists of 3 or 4), region on the left; None at a pinch node."""
    a, b = [], []
    for e in elems:
        e = [int(x) for x in e]
        for k in range(len(e)):
            a.append(e[k])
            b.append(e[(k + 1) % len(e)])
    fw = set(zip(a, b))
    nxt = {}
    for x, y in zip(a, b):
        if (y, x) not in fw:
            if x in nxt:
                return None
            nxt[x] = y
    loops, seen = [], set()
    for s0 in nxt:
        if s0 in seen:
            continue
        lp, x = [s0], nxt[s0]
        seen.add(s0)
        while x != s0:
            if x in seen or x not in nxt:
                return None
            lp.append(x)
            seen.add(x)
            x = nxt[x]
        loops.append(np.asarray(lp, dtype=np.int64))
    return loops


# --------------------------------------------------------------------------------------------------- the fills

def classify(part: Partition, o: dict) -> None:
    for rec in part.regions:
        cls = rec["cls"]
        reason = cls
        if cls == "coast" or rec["coast"]:
            f = o["coast_fill"]
            reason = "coast"
        elif cls == "junction":
            f = o["junction_fill"]
        elif cls == "open":
            f = o["open_fill"]
        elif o["force_fill"]:
            f, reason = o["force_fill"], "forced"
        elif o["join"] == "quantised":                                  # B47: b and the obliquity decide
            obl = max(rec.get("obl_a", 90.0), rec.get("obl_b", 90.0))
            if obl < o["obl_strip_deg"] and rec.get("strip_ok"):
                rec["join"] = "ladder"
                f, reason = "strip", f"ladder ({rec['quant']})"
            elif (rec.get("cont_ok") and not rec.get("cont_failed")
                  and (obl >= o["obl_strip_deg"] or o["cont_fallback"])):
                rec["join"] = "cont"                                    # m14/continuation.py (B57)
                why = "oblique" if obl >= o["obl_strip_deg"] else "no ladder"
                f, reason = "cont", f"{why}: the continuation ({rec['quant']})"
            elif rec.get("cont_failed"):
                rec["join"] = "m02"
                f, reason = "m02", "continuation failed: " + rec["cont_failed"]
            else:
                rec["join"] = "m02"
                f, reason = "m02", "no ladder: " + rec.get("why_not_strip", "not a corridor")
        else:
            rot, mis = rec.get("rot_deg"), rec.get("mismatch_per10")
            if rot is None:
                f, reason = "m02", "no rotation (sides not runs)"
            elif rot < o["rot_quad_deg"] and mis <= o["rows_tol"]:
                f, reason = "strip", "aligned"
            elif rot >= o["rot_tri_deg"] or mis > o["rows_tri"]:
                f, reason = "tri", "rotated" if rot >= o["rot_tri_deg"] else "mismatch"
            else:
                f, reason = "m02", "between"
        if f == "strip" and not rec.get("strip_ok"):
            f, reason = "m02", "strip impossible: " + rec.get("why_not_strip", "not a corridor")
        rec["fill"], rec["rule"] = f, reason


def _pack(bg: Mesh, h: SizeField, seeds: np.ndarray, mo: dict) -> np.ndarray:
    """m02's packing (m02_linf_frontal.points) on a given background mesh, seeded with the given fixed nodes."""
    from scipy.spatial import cKDTree

    from meshom.field import cross_field
    from meshom.methods.m02.packing import pack
    from meshom.methods.m02_linf_frontal import _locate, _vertex_triangle

    bnodes = np.flatnonzero(bg.node_flag == 1)
    nd = int(mo["n"])
    cf = cross_field(bg, h, n=nd, boundary_nodes=bnodes, smooth_steps=mo["smooth_steps"], method=mo["field"])
    Pb = np.ascontiguousarray(bg.points)
    Tb = np.ascontiguousarray(bg.tris, dtype=np.int64)
    nbr = adjacency(Tb)
    vt = _vertex_triangle(Tb, len(Pb))
    hb = h(Pb) / R_EARTH_M
    lhb = np.log(hb)
    ub = np.ascontiguousarray(cf.u)
    cen = _unit(Pb[Tb].sum(axis=1))
    rad = np.max(np.linalg.norm(Pb[Tb] - cen[:, None, :], axis=2), axis=1)
    e = np.vstack([Tb[:, [0, 1]], Tb[:, [1, 2]], Tb[:, [2, 0]]])
    g = float(np.max(np.abs(hb[e[:, 0]] - hb[e[:, 1]]) / np.linalg.norm(Pb[e[:, 0]] - Pb[e[:, 1]], axis=1)))
    alpha = float(mo["alpha"])
    c = np.sqrt(2.0) if nd == 4 else 2.0 / np.sqrt(3.0)
    rfac = c * alpha / max(1.0 - 0.5 * c * alpha * g, 0.25)
    stri = _locate(Pb, Tb, nbr, vt, cKDTree(Pb), seeds)
    sh = h(seeds) / R_EARTH_M
    fill = np.flatnonzero(bg.node_flag == 0) if mo["gap_fill"] else np.zeros(0, dtype=np.int64)
    area = 0.5 * np.linalg.norm(np.cross(Pb[Tb[:, 1]] - Pb[Tb[:, 0]], Pb[Tb[:, 2]] - Pb[Tb[:, 0]]), axis=1)
    n_est = float(np.sum(area / np.exp(lhb[Tb]).mean(axis=1) ** 2)) * (1.0 if nd == 4 else 2.0 / np.sqrt(3.0))
    cap = int(2.0 * n_est) + len(seeds) + 1000
    while True:
        Pp = np.zeros((cap, 3))
        ph = np.zeros(cap)
        ptri = np.zeros(cap, dtype=np.int64)
        Pp[: len(seeds)] = seeds
        ph[: len(seeds)] = sh
        ptri[: len(seeds)] = stri
        n, full, _ = pack(Pb, Tb, nbr, ub, lhb, cen, rad, Pp, ph, ptri, len(seeds), fill, vt[fill], alpha, rfac,
                          bool(mo["midpoint"]), float(mo["seam"]), float(mo["alpha_seam"]), nd)
        if not full:
            break
        cap *= 2
    return Pp[len(seeds):n].copy()


def _cdt(pts: np.ndarray, cons: np.ndarray, n_loop: int, info: dict | None = None) -> np.ndarray:
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tris = sphere_delaunay(pts, cons, cons[:n_loop])
    if info is not None and w:
        info["recover_warnings"] = info.get("recover_warnings", 0) + len(w)
    return tris


def _tri_regions(pts: np.ndarray, tris: np.ndarray, part: Partition) -> tuple[np.ndarray, list]:
    """Region of each triangle: components cut at every constraint edge, named by the known half-edges inside them:
    the separator half-edges decide (majority); a component touching no separator takes the majority of its loop
    half-edges. (A loop half-edge next to a separator end can sit on the other side of the smoothed, resampled separator
    than its background triangle did, so loop votes are not trusted where separator votes exist.) Returns (regions
    (T,), the region lists of components whose separator half-edges name more than one region: a real leak)."""
    n = len(pts)
    nbr = adjacency(tris)
    a = tris.ravel()
    b = np.roll(tris, -1, axis=1).ravel()
    ck = np.unique(np.minimum(part.cons[:, 0], part.cons[:, 1]) * n + np.maximum(part.cons[:, 0], part.cons[:, 1]))
    blocked = np.isin(np.minimum(a, b) * n + np.maximum(a, b), ck)
    nb = nbr.ravel()
    ok = (nb >= 0) & ~blocked
    t_of = np.repeat(np.arange(len(tris)), 3)
    comp = _components(len(tris), t_of[ok], nb[ok])
    N = part.N
    keys = (part.he_keys // N) * n + part.he_keys % N                    # the known half-edges (seed ids < N) in base
    hk = a * n + b                                                      # n: base N would alias edges with fill ids >= N
    srt = np.argsort(hk)
    pos = np.minimum(np.searchsorted(hk[srt], keys), len(hk) - 1)
    found = hk[srt][pos] == keys
    c = comp[t_of[srt[pos[found]]]]
    r = part.he_reg[found]
    sep = part.he_sep[found]
    nreg = len(part.regions)
    nc = int(comp.max()) + 1
    best = np.full(nc, -1, dtype=np.int64)
    conflicts = []
    for use in (sep, ~sep):                                             # separator votes first, then loop votes
        pair, cnt = np.unique(c[use] * nreg + r[use], return_counts=True)
        pc, pr = pair // nreg, pair % nreg
        bcnt = np.zeros(nc, dtype=np.int64)
        win = np.full(nc, -1, dtype=np.int64)
        for x, y, z in zip(pc, pr, cnt):
            if z > bcnt[x]:
                win[x], bcnt[x] = y, z
        if use is sep:
            multi = np.flatnonzero(np.bincount(pc, minlength=nc) > 1)
            conflicts = [sorted(set(pr[pc == x].tolist())) for x in multi]
        best = np.where(best < 0, win, best)
    return best[comp], conflicts


def _region_loops(tris: np.ndarray):
    """Boundary loops of a triangle set (region on the left) as node-id arrays; None at a pinch node."""
    a = tris.ravel()
    b = np.roll(tris, -1, axis=1).ravel()
    n = int(max(a.max(), b.max())) + 1
    fw = a * n + b
    bw = b * n + a
    free = ~np.isin(fw, bw)
    fa, fb = a[free], b[free]
    if len(np.unique(fa)) != len(fa):
        return None
    nxt = dict(zip(fa.tolist(), fb.tolist()))
    loops, seen = [], set()
    for s in fa.tolist():
        if s in seen:
            continue
        lp, x = [s], nxt[s]
        seen.add(s)
        while x != s:
            lp.append(x)
            seen.add(x)
            x = nxt[x]
        loops.append(np.asarray(lp, dtype=np.int64))
    return loops


def _hook(spec):
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def cont_fill(part: Partition, rec: dict, allp: np.ndarray, tris_r: np.ndarray, grids: list, h: SizeField, o: dict):
    """The lattice continuation (m14/continuation.py, B57) of corridor rec, whose region holds the triangles tris_r of
    the CDT (fill ids into allp): its boundary loop (region on the left) turned to start at side A's first outline
    node, the two outline runs as loop positions and as block node ids (loop order: A from cap C0 to C1, B from C1 to
    C0), the fill checked as the strips are (no inverted element, angles < 170 deg). Returns (Mesh with the loop's
    nodes first, the loop in fill ids, info) or the reason it failed."""
    from meshom.methods.m14 import continuation

    lps = _region_loops(tris_r)
    if lps is None or len(lps) != 1:
        return "the region is not one loop"
    lp = lps[0]
    offs = np.r_[0, np.cumsum([len(x) for x in part.loops])]
    runs = []
    for obj in rec["objs"]:
        li = part.C + obj                                               # loops: coast first, then the outlines
        on = (lp >= offs[li]) & (lp < offs[li + 1])
        st = np.flatnonzero(on & ~np.roll(on, 1))                       # starts of the cyclic runs of this outline
        if len(st) != 1 or on.all():
            return "a side is not one run of the region's loop"
        runs.append((int(st[0]), int(on.sum())))
    lp = np.roll(lp, -runs[0][0])
    pa = np.arange(runs[0][1])
    pb = (runs[1][0] - runs[0][0]) % len(lp) + np.arange(runs[1][1])
    if pb[-1] >= len(lp) or len(pa) < 3 or len(pb) < 3:
        return "the runs overlap or are too short"
    ga, gb = grids[rec["objs"][0]], grids[rec["objs"][1]]
    ra = ga.outline[lp[pa] - offs[part.C + rec["objs"][0]]]
    rb = gb.outline[lp[pb] - offs[part.C + rec["objs"][1]]]
    cor = sched = None
    known = {}
    cc = rec.get("_cont_cor")
    if cc is not None and np.array_equal(cc[2], lp[pa]) and np.array_equal(cc[3], lp[pb]):
        cor, sched = cc[0], cc[1]                                       # the lattice caps' corridor and schedule
        at = {int(x): k for k, x in enumerate(lp)}
        for ci in [x for cap in rec.get("_cont_caps", []) for x in cap]:
            c = part.chains[ci]
            if "lat" not in c or c.get("dropped"):
                continue
            ids = list(c["ids"][1:-1])
            lab = list(zip(*c["lat"]))
            for k, el in zip((0, -1), c["lat_end"]):                     # branch nodes between the pieces
                if el is not None:
                    ids.append(c["ids"][k])
                    lab.append(el)
            for x, (lu, lv) in zip(ids, lab):
                if int(x) in at:
                    known[at[int(x)]] = (lu, lv)
    try:
        fm, info = continuation.fill(allp[lp], pa, pb, ga, ra, gb, rb, h, _cont_opts(o, rec), cor, sched, known)
    except (np.linalg.LinAlgError, ValueError, IndexError) as e:      # a degenerate face map or an empty point set
        return f"failed ({type(e).__name__})"
    if not _strip_valid(fm.points, fm.quads, fm.tris):
        return "invalid (fold)"
    return fm, lp, info


def fill_tri_local(domain: Domain, h: SizeField, fixed_loops: list) -> Mesh:
    """Reference triangle fill of one region (the tri_fill interface): Poisson points (alpha 0.75), a local Delaunay
    with far helper points when the region fits in a hemisphere, every loop edge recovered, the outside removed,
    30 "size" + 3 "angle" sweeps with the loop nodes fixed. Loop nodes first, in loop order, flag 1."""
    from meshom.delaunay import poisson_points
    from meshom.smooth import smooth

    pts = poisson_points(domain, h, fixed_loops, alpha=0.75)
    nb = sum(len(lp) for lp in fixed_loops)
    e = loop_edges(fixed_loops)
    mu = _unit(pts.mean(axis=0))
    far = np.zeros((0, 3))
    if (pts @ mu).min() > -0.5:
        e1, e2 = tangent_basis(mu[None])
        far = _unit(np.vstack([-mu] + [-0.5 * mu + np.cos(x) * e1[0] + np.sin(x) * e2[0]
                                       for x in np.arange(6) * np.pi / 3]))
    allp = np.vstack([pts, far])
    tris = _cdt(allp, e, len(e))
    tris = tris[(tris < len(pts)).all(axis=1)]
    used = np.zeros(len(pts), dtype=bool)
    used[tris.ravel()] = True
    used[:nb] = True
    new = np.cumsum(used) - 1
    flag = np.zeros(int(used.sum()), dtype=np.int8)
    flag[:nb] = 1
    mesh = Mesh(pts[used], new[tris], np.zeros((0, 4), dtype=np.int64), flag)
    mesh = smooth(mesh, h, flag == 1, iters=30, kind="size")
    return smooth(mesh, h, flag == 1, iters=3, kind="angle")


def stitch_mix(domain: Domain, h: SizeField, grids: list, o: dict) -> tuple[Mesh, dict]:
    from meshom.methods import m02_linf_frontal as m02
    from meshom.methods.m14.stitch import element_rows
    from meshom.recombine import recombine
    from meshom.smooth import smooth

    if o["coast"] != "conform":
        raise NotImplementedError('coast="cut" is reserved (the run-level cut is m14_blocks_cut)')
    t_part = time.perf_counter()
    coast = resample_loops(domain, h)
    nc = sum(len(lp) for lp in coast)
    outl = [g.points[g.outline] for g in grids]
    part = partition(coast, outl, h, o)
    corridor_geometry(part, h, o["strip_lam"], o["obl_strip_deg"], o["join"], o["end_rows"])
    mo4 = m02._opts(dict(o["fill"], n=4))
    mo6 = m02._opts(dict(o["fill"], n=6))
    hook = _hook(o["tri_fill"]) if o["tri_fill"] else None
    part.info["strips_demoted_leak"] = 0
    tm = {"partition_s": time.perf_counter() - t_part}
    for attempt in range(6):
        t0 = time.perf_counter()
        resample_separators(part, h, o, grids)
        part.info["attempts"] = attempt + 1
        # strips (before the points: a strip that fails becomes "m02")
        classify(part, o)
        merge_same_fill(part, o["merge_same_fill"])
        nbnd, N = part.nbnd, part.N
        seeds = np.vstack([np.vstack(part.loops), part.sep_xyz])
        strips = {}
        nstrip_pts = 0
        redo = False                                                    # an invalid ladder that the continuation takes
        for rec in part.regions:
            if rec["fill"] != "strip":
                continue
            A, Bs = rec["_A"], rec["_B"]
            C0, C1 = rec["_C"]
            assert C0[0] == A[0] and C0[-1] == Bs[-1] and C1[0] == A[-1] and C1[-1] == Bs[0]
            new, q, tr = strip_patch(seeds, A, Bs[::-1], C0, C1, rec["_plan"], N + nstrip_pts)
            allp = np.vstack([seeds, np.zeros((nstrip_pts, 3)), new]) if len(new) else seeds
            if not _strip_valid(allp, q, tr, STRIP_QUAD_MAX):
                rec["fill"], rec["rule"] = "m02", "strip invalid (fold or a quad >= 150 deg)"
                rec["strip_ok"] = False
                rec["why_not_strip"] = "strip invalid (fold or a quad >= 150 deg)"
                redo |= _wants_cont(rec, o)
                continue
            strips[rec["id"]] = (new, q, tr)
            nstrip_pts += len(new)
        fills = np.array([rec["fill"] for rec in part.regions])
        t1 = time.perf_counter()
        tm["strip_s"] = tm.get("strip_s", 0.0) + t1 - t0                # (separators + strips, every attempt)
        if redo:                                                        # those corridors to the continuation first
            part.info["redo_invalid_strips"] = part.info.get("redo_invalid_strips", 0) + 1
            continue
        # packed points, kept by region
        kept, leaks = [], []
        for fill_kind, mo in (("m02", mo4), ("tri", mo6)):
            if not (fills == fill_kind).any() or (fill_kind == "tri" and hook is not None):
                continue
            pp = _pack(part.bg, h, seeds, mo)
            allp = np.vstack([seeds, pp])
            tris = _cdt(allp, part.cons, part.n_loop_edges, part.info)
            treg, cf = _tri_regions(allp, tris, part)
            leaks += cf
            preg = np.full(len(allp), -1, dtype=np.int64)
            preg[tris.ravel()] = np.repeat(treg, 3)
            sel = preg[N:]
            kept.append(pp[(sel >= 0) & (fills[np.maximum(sel, 0)] == fill_kind)])
        pin = np.vstack(kept) if kept else np.zeros((0, 3))
        t2 = time.perf_counter()
        tm["pack_s"] = tm.get("pack_s", 0.0) + t2 - t1
        # the triangulation of everything, labelled by region
        allp = np.vstack([seeds, pin])
        tris = _cdt(allp, part.cons, part.n_loop_edges, part.info)
        treg, conflicts = _tri_regions(allp, tris, part)
        leaks += conflicts
        if not leaks and not (treg < 0).any():
            # the continuation corridors (no packed points: their CDT triangles span the region); one that fails is
            # filled by m02 in the next attempt
            t4 = time.perf_counter()
            conts, failed = {}, 0
            for rec in part.regions:
                if rec["fill"] != "cont":
                    continue
                res = cont_fill(part, rec, allp, tris[treg == rec["id"]], grids, h, o)
                if isinstance(res, str):
                    rec["cont_failed"] = res
                    failed += 1
                else:
                    conts[rec["id"]] = res
                    rec["cont"] = res[2]
            tm["cont_s"] = tm.get("cont_s", 0.0) + time.perf_counter() - t4
            part.info["cont_failed"] = part.info.get("cont_failed", 0) + failed
            if not failed:
                break
            continue
        # a straight cap jumped over a thin neighbour (no segment crosses, the neighbour turns inside out): the strip
        # corridors involved, or whose straight caps border an involved region, lose their strip; rebuild
        inv = {r for r in range(len(part.regions)) if part.group[r] in {x for lk in leaks for x in lk}}
        demote = {r for r in inv if part.regions[r].get("strip_ok")}
        # a continuation corridor named in a leak loses its lattice caps; else those bordering a leaking region
        lat = {r for r in inv if part.regions[r].get("fill") == "cont" and part.regions[r].get("_lat_caps")
               and not part.regions[r].get("_no_lat_caps")}
        for c in part.chains:
            if c.get("straight") and (c["left"] in inv or c["right"] in inv):
                demote |= {x for x in (c["left"], c["right"]) if part.regions[x].get("strip_ok")}
                if not lat and not demote:
                    lat |= {x for x in (c["left"], c["right"]) if "lat" in c and part.regions[x].get("_lat_caps")}
        for r in lat:                                       # a lattice cap: the other path, then its separators
            x = part.regions[r]
            if x.get("_lat_variant", 0) < 1:
                x["_lat_variant"] = 1
                x.pop("_lat_caps", None)
            else:
                x["_no_lat_caps"] = "a lattice cap turns a neighbour inside out"
        if not demote and not lat:
            raise RuntimeError(f"m14_blocks_mix: regions leak into each other {leaks}, unlabelled triangles "
                               f"{int((treg < 0).sum())}, recover warnings {part.info.get('recover_warnings', 0)}")
        for r in demote:
            part.regions[r]["strip_ok"] = False
            part.regions[r]["why_not_strip"] = "a straight cap turns a thin neighbour inside out"
        part.info["strips_demoted_leak"] += len(demote)
    else:
        raise RuntimeError("m14_blocks_mix: regions still leak after 6 attempts")
    tf = fills[np.maximum(treg, 0)]
    keep = (treg >= 0) & (tf != "strip") & (tf != "cont") & ~((tf == "tri") & (hook is not None))
    # m02 regions: recombine, never across a separator (nodes duplicated per region)
    mq = keep & (tf == "m02")
    T2, Q2 = [tris[keep & (tf != "m02")]], []
    R2 = [treg[keep & (tf != "m02")]]
    RQ = []
    if mq.any():
        tm2, rm2 = tris[mq], treg[mq]
        pair = tm2.astype(np.int64) * len(part.regions) + rm2[:, None]
        up, inv = np.unique(pair, return_inverse=True)
        dup = Mesh(allp[up // len(part.regions)], inv.reshape(-1, 3).astype(np.int64), np.zeros((0, 4), np.int64),
                   np.zeros(len(up), np.int8))
        rc = recombine(dup, mo4["price"], mo4["recombine"])
        back = up // len(part.regions)
        rr = up % len(part.regions)
        T2.append(back[rc.tris])
        R2.append(rr[rc.tris[:, 0]])
        Q2.append(back[rc.quads])
        RQ.append(rr[rc.quads[:, 0]])
    # strips and hooked triangle regions: new nodes after the kept points
    extra, nextra = [], len(allp)
    off = len(allp) - N
    for rid, (new, q, tr) in strips.items():
        shift = lambda x: np.where(x >= N, x + off, x)                  # noqa: E731
        Q2.append(shift(q))
        RQ.append(np.full(len(q), rid))
        T2.append(shift(tr))
        R2.append(np.full(len(tr), rid))
        extra.append(new)
    nextra += sum(len(x) for x in extra)
    for rid, (fm, lp, _) in conts.items():                             # continuation corridors: loop nodes first
        gmap = np.r_[lp, nextra + np.arange(fm.n_nodes - len(lp))].astype(np.int64)
        T2.append(gmap[fm.tris])
        R2.append(np.full(fm.n_tris, rid))
        Q2.append(gmap[fm.quads])
        RQ.append(np.full(fm.n_quads, rid))
        extra.append(fm.points[len(lp):])
        nextra += fm.n_nodes - len(lp)
    n_hook = 0
    if hook is not None:
        for rec in part.regions:
            if rec["fill"] != "tri" or part.group[rec["id"]] != rec["id"]:   # once per fill region (B50: its id names
                continue                                                # the triangles of every region merged in it)
            lps = _region_loops(tris[(treg == rec["id"])])
            if lps is None:
                raise RuntimeError(f"tri_fill: region {rec['id']} has a pinch node")
            xl = [allp[lp] for lp in lps]
            fm = hook(Domain(xl, "region", exact=[True] * len(xl)), h, xl)
            nb_ = sum(len(lp) for lp in lps)
            gmap = np.r_[np.concatenate(lps), nextra + np.arange(fm.n_nodes - nb_)].astype(np.int64)
            T2.append(gmap[fm.tris])
            R2.append(np.full(fm.n_tris, rec["id"]))
            Q2.append(gmap[fm.quads])
            RQ.append(np.full(fm.n_quads, rec["id"]))
            extra.append(fm.points[nb_:])
            nextra += fm.n_nodes - nb_
            n_hook += 1
    fpts = np.vstack([allp] + extra)
    ftris = np.vstack(T2).astype(np.int64)
    fquads = np.vstack(Q2).astype(np.int64) if Q2 else np.zeros((0, 4), np.int64)
    rt = np.concatenate(R2).astype(np.int64)
    rq = np.concatenate(RQ).astype(np.int64) if RQ else np.zeros(0, np.int64)
    used = np.zeros(len(fpts), dtype=bool)
    used[ftris.ravel()] = True
    used[fquads.ravel()] = True
    used[:N] = True
    new_id = np.cumsum(used) - 1
    fpts, ftris, fquads = fpts[used], new_id[ftris], new_id[fquads]
    flag = np.zeros(len(fpts), dtype=np.int8)
    flag[:nc] = 1
    fmesh = Mesh(fpts, ftris, fquads, flag)
    t3 = time.perf_counter()
    tm["cdt_s"] = t3 - t2
    fixed = np.zeros(len(fpts), dtype=bool)
    fixed[:nbnd if o["free_separators"] else N] = True                 # coast and outline nodes (and separators)
    if mo4["size_iters"]:
        fmesh = smooth(fmesh, h, fixed, iters=mo4["size_iters"], kind="size")
    if mo4["angle_iters"]:
        fmesh = smooth(fmesh, h, fixed, iters=mo4["angle_iters"], kind="angle")
    if o["break_clusters"]:                                 # B62 (on the smoothed fill; not in the "tri" regions)
        tz = fills[np.maximum(rt, 0)] == "tri"
        t2, q2, rt2, rq, n_br, gone = break_clusters(fmesh.points, fmesh.tris[~tz], fmesh.quads, rt[~tz], rq,
                                                     free_from=nbnd)            # separator nodes may go too
        rt = np.r_[rt2, rt[tz]]
        t2 = np.vstack([t2, fmesh.tris[tz]]).astype(np.int64)
        keepn = np.ones(fmesh.n_nodes, dtype=bool)
        keepn[gone] = False                                             # removed valence-3 nodes (fill interior only)
        nid = np.cumsum(keepn) - 1
        fixed = fixed[keepn]
        fmesh = Mesh(fmesh.points[keepn], nid[t2], nid[q2], fmesh.node_flag[keepn])
        part.info["clusters_broken"] = n_br
        N -= int(np.sum(np.asarray(gone, dtype=np.int64) < N))           # removed separator nodes
        if n_br:                                                        # a few sweeps over the new elements
            fmesh = smooth(fmesh, h, fixed, iters=10, kind="size")
            fmesh = smooth(fmesh, h, fixed, iters=3, kind="angle")
    tm["fill_smooth_s"] = time.perf_counter() - t3
    # merge with the blocks: coast nodes, block nodes, then the other fill nodes (separators first)
    nb_f = nbnd
    offs = nc + np.r_[0, np.cumsum([len(g.points) for g in grids])].astype(np.int64)
    n_int = fmesh.n_nodes - nb_f
    NN = int(offs[-1]) + n_int
    pts = np.empty((NN, 3))
    pts[:nc] = fmesh.points[:nc]
    fmap = np.empty(fmesh.n_nodes, dtype=np.int64)
    fmap[:nc] = np.arange(nc)
    k = nc
    qb, ol_edges, block_nodes = [], [], []
    for b, g in enumerate(grids):
        pts[offs[b]:offs[b + 1]] = g.points
        lp = offs[b] + g.outline
        fmap[k:k + len(lp)] = lp
        k += len(lp)
        qb.append(g.quads + offs[b])
        ol_edges.append(np.column_stack([lp, np.roll(lp, -1)]))
        block_nodes.append(np.arange(offs[b], offs[b + 1]))
    fmap[nb_f:] = offs[-1] + np.arange(n_int)
    pts[offs[-1]:] = fmesh.points[nb_f:]
    nqb = sum(len(q) for q in qb)
    quads = np.vstack(qb + [fmap[fmesh.quads]]).astype(np.int64)
    tris_m = fmap[fmesh.tris]
    flag = np.zeros(NN, dtype=np.int8)
    flag[:nc] = 1
    mesh = Mesh(pts, tris_m, quads, flag)
    is_block = np.zeros(mesh.n_elems, dtype=bool)
    is_block[mesh.n_tris:mesh.n_tris + nqb] = True
    elem_region = np.r_[rt, np.full(nqb, -1), rq].astype(np.int64)
    gsize = np.bincount(part.group, minlength=len(part.regions))
    multi = (elem_region >= 0) & (gsize[np.maximum(elem_region, 0)] > 1)
    if multi.any():                                                     # merged fill regions: back to their regions
        from scipy.spatial import cKDTree

        el = [list(t) for t in mesh.tris] + [list(q) for q in mesh.quads]
        idx = np.flatnonzero(multi)
        cen = _unit(np.array([mesh.points[el[i]].mean(axis=0) for i in idx]))
        bgc = _unit(part.bg.points[part.bg.tris].sum(axis=1))
        _, j = cKDTree(bgc).query(cen, k=8)
        rj = part.reg[j]
        ok = part.group[rj] == elem_region[idx][:, None]
        first = np.argmax(ok, axis=1)
        elem_region[idx] = np.where(ok.any(axis=1), rj[np.arange(len(idx)), first], elem_region[idx])
    sep_nodes = offs[-1] + np.arange(N - nb_f)                         # separator nodes in the merged mesh
    ol_edges = np.vstack(ol_edges) if ol_edges else np.zeros((0, 2), np.int64)  # (no block: a narrow water domain)
    info = {"n_fill_tris": int(fmesh.n_tris), "n_fill_quads": int(fmesh.n_quads), "n_block_quads": int(nqb),
            "n_separator_nodes": int(N - nb_f), "n_strip_nodes": int(nstrip_pts), "cdt_conflicts": conflicts,
            "n_tri_hooked": n_hook, "n_cont": len(conts), **part.info, **{f"mix_{k_}": v for k_, v in tm.items()}}
    if o["band_size_iters"] or o["band_angle_iters"]:
        row, _, _ = element_rows(mesh, is_block, ol_edges)
        band = (row >= 0) & np.where(is_block, row < o["band_block"], row < o["band_fill"])
        in_band = np.zeros(NN, dtype=bool)
        out_band = np.zeros(NN, dtype=bool)
        for cells, of in ((mesh.tris, 0), (mesh.quads, mesh.n_tris)):
            bb = band[of:of + len(cells)]
            in_band[cells[bb].ravel()] = True
            out_band[cells[~bb].ravel()] = True
        movable = in_band & ~out_band & (flag == 0)
        info["band_nodes"] = int(movable.sum())
        if o["band_size_iters"]:
            mesh = smooth(mesh, h, ~movable, iters=o["band_size_iters"], kind="size")
        if o["band_angle_iters"]:
            mesh = smooth(mesh, h, ~movable, iters=o["band_angle_iters"], kind="angle")
    if o.get("size_tolerance") is not None:                             # G38: smoothness within the tolerance
        from meshom.methods.m14.stitch import tolerance_fit

        mesh = tolerance_fit(mesh, h, o, block_nodes, info)
    fills_r = np.array([rec["fill"] for rec in part.regions])
    rt_final = elem_region[:mesh.n_tris]
    st = {"info": info, "is_block": is_block, "outline_edges": ol_edges,
          "block_nodes": np.concatenate(block_nodes + [np.zeros(0, np.int64)]), "elem_region": elem_region,
          "sep_nodes": sep_nodes, "region_records": part.regions, "partition": part,
          "tri_zone": (rt_final >= 0) & (fills_r[np.maximum(rt_final, 0)] == "tri")}  # (m14_blocks.run measures)
    return mesh, st


# ---------------------------------------------------------------------------------------------------- measurement

def _p(x, q):
    return round(float(np.percentile(x, q)), 4) if len(x) else None


def measure_regions(mesh: Mesh, h: SizeField, er: np.ndarray, regions: list) -> list[dict]:
    """Per-region measures (module docstring) of a mesh whose elements carry region ids er (-1: block quads); regions
    = the partition's records (or those of m14.json: a smoothed mesh is re-measured with the same ids)."""
    from meshom.quality.kernel import element_geometry

    nreg = len(regions)
    n, T = mesh.n_nodes, mesh.n_tris
    topo = build_topology(mesh)
    E, ee = topo.edges, topo.edge_elems
    deg = np.bincount(E.ravel(), minlength=n)
    bnd = topo.boundary_node
    qn = np.bincount(mesh.quads.ravel(), minlength=n)
    tn = np.bincount(mesh.tris.ravel(), minlength=n)
    irr4 = (deg != 4) & ~bnd
    irrm = (np.abs(360 - 90 * qn - 60 * tn) >= 60) & ~bnd
    # node -> region holding most of its fill elements
    cells = [(mesh.tris, 0), (mesh.quads, T)]
    nn, rr = [], []
    for c, of in cells:
        r = er[of:of + len(c)]
        m = r >= 0
        nn.append(c[m].ravel())
        rr.append(np.repeat(r[m], c.shape[1]))
    nn, rr = np.concatenate(nn), np.concatenate(rr)
    key, cnt = np.unique(nn * nreg + rr, return_counts=True)
    kn, kr = key // nreg, key % nreg
    o_ = np.lexsort((kr, -cnt, kn))
    kn, kr = kn[o_], kr[o_]
    firstn = np.r_[True, kn[1:] != kn[:-1]]
    node_reg = np.full(n, -1, dtype=np.int64)
    node_reg[kn[firstn]] = kr[firstn]
    g = element_geometry(mesh)
    maxang = np.r_[np.degrees(g.tri_angles).max(axis=1, initial=0.0),
                   np.degrees(g.quad_angles).max(axis=1, initial=0.0)]
    area = np.abs(np.r_[g.tri_area, g.quad_area])
    hc = h(g.centroid)
    mean_edge = np.r_[g.tri_edges.mean(axis=1), g.quad_edges.mean(axis=1)]
    p, q = mesh.points[E[:, 0]], mesh.points[E[:, 1]]
    lnsh = np.abs(np.log(R_EARTH_M * arc_length(p, q) / h(_unit(p + q))))
    inner = ee[:, 1] >= 0
    e1 = np.where(inner, ee[:, 1], ee[:, 0])
    ratio = np.maximum(mean_edge[ee[:, 0]], mean_edge[e1]) / np.minimum(mean_edge[ee[:, 0]], mean_edge[e1])
    # (edge, region) pairs from both sides
    ep = np.r_[np.arange(len(E)), np.arange(len(E))[inner]]
    epr = np.r_[er[ee[:, 0]], er[ee[inner, 1]]]
    m = epr >= 0
    pk = np.unique(epr[m] * len(E) + ep[m])
    e_reg, e_id = pk // len(E), pk % len(E)
    e_split = np.searchsorted(e_reg, np.arange(nreg + 1))
    el_ord = np.argsort(er, kind="stable")
    el_split = np.searchsorted(er[el_ord], np.arange(nreg + 1))
    nd_ord = np.argsort(node_reg, kind="stable")
    nd_split = np.searchsorted(node_reg[nd_ord], np.arange(nreg + 1))
    is_tri = np.arange(mesh.n_elems) < T
    out = []
    for rec in regions:
        r = rec["id"]
        els = el_ord[el_split[r]:el_split[r + 1]]
        nds = nd_ord[nd_split[r]:nd_split[r + 1]]
        nds = nds[~bnd[nds]]
        eds = e_id[e_split[r]:e_split[r + 1]]
        tq = els[~is_tri[els]]
        tt = els[is_tri[els]]
        L = 0.5 * (rec["nA"] + rec["nB"]) if "nA" in rec else 0.5 * (rec["outline_edges"] + rec["coast_edges"])
        a_h2 = float(np.sum(area[els] / hc[els] ** 2))
        x = {k: v for k, v in rec.items() if not k.startswith("_")}
        for k_ in ("rot_deg", "rot_coh", "obl_a", "obl_b", "width", "width_dd", "area_h2", "mismatch_per10"):
            if x.get(k_) is not None:
                x[k_] = round(float(x[k_]), 3)
        x.update(elems=int(len(els)), tris=int(len(tt)), quads=int(len(tq)), nodes=int(len(nds)),
                 irr4=int(irr4[nds].sum()), irr_mix=int(irrm[nds].sum()),
                 n3=int((deg[nds] == 3).sum()), n5=int((deg[nds] == 5).sum()),
                 length_cells=round(float(L), 1), area_cells=round(a_h2, 1),
                 ln_sh_p99=_p(lnsh[eds], 99), ln_sh_max=round(float(lnsh[eds].max()), 4) if len(eds) else None,
                 ratio_p99=_p(ratio[eds[inner[eds]]], 99),
                 quad_max_deg=round(float(maxang[tq].max()), 2) if len(tq) else None,
                 quad_p99_deg=_p(maxang[tq], 99), tri_max_deg=round(float(maxang[tt].max()), 2) if len(tt) else None)
        if L > 0:
            x.update(irr4_per10=round(10.0 * x["irr4"] / L, 3), irr_mix_per10=round(10.0 * x["irr_mix"] / L, 3),
                     tris_per10=round(10.0 * x["tris"] / L, 3))
        if a_h2 > 0:
            x.update(irr4_per100a=round(100.0 * x["irr4"] / a_h2, 3),
                     irr_mix_per100a=round(100.0 * x["irr_mix"] / a_h2, 3))
        out.append(x)
    return out
