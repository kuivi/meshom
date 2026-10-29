"""Coast-row clean-up of conforming m14 after the repair (decisions B64).

On a real coast the coast loops are fixed, so a coast quad whose largest angle sits at a coast node keeps it whatever
the smoothing does: m14_blocks_rep on GSHHS i (global 100 km, West Africa 25 km) left 27 of 30 and 9 of 15 quads
above the 135 deg target at a bay corner of the coast row, a valence-2 coast node whose whole water angle (137-150 deg)
is one quad corner; the rest at a coast node of two elements or next to one. split_coast_quads cuts each such quad
along the diagonal through its largest corner into two triangles (the corner angle halves), only where both triangles
keep their angles inside [tri_min, tri_max] and no neighbour across the quad's edges is a triangle (no cluster of 3).
The triangle clusters left in the fill (interior, from the stitch and the repair's cavities) are then broken by
stitch_mix.break_clusters with its new quads kept below the target and its new triangles inside the hard
limits [20, 120] deg (a change that would leave one outside is not made). Conforming only (the domain has loops): the cut
mode never reaches it.
"""
import numpy as np

from meshom.geo import tangent_coords
from meshom.mesh import Mesh, build_topology
from meshom.quality.kernel import _angles
from meshom.quality.metrics import element_regions

TARGET_DEG = 135.0    # the quad_max_angle_target_deg gate (p99.9)
TRI_MAX_DEG = 120.0   # the tri_max_angle_hard_deg gate
TRI_MIN_DEG = 30.0    # a split triangle keeps every angle at least this
TRI_HARD_MIN_DEG = 20.0  # the tri_min_angle_hard_deg gate: a triangle of the cluster break keeps [20, 120]
ROWS = 3              # quads of the first ROWS coast rows (the gates' _coast3 subset) are split


def _deg(P: np.ndarray, cells: np.ndarray) -> np.ndarray:
    return np.degrees(_angles(tangent_coords(P[cells])))


def split_coast_quads(mesh: Mesh, target: float = TARGET_DEG, tri_max: float = TRI_MAX_DEG,
                      tri_min: float = TRI_MIN_DEG, frozen: np.ndarray | None = None) -> tuple[Mesh, int]:
    """Quads of the first ROWS coast rows with an angle above target, split through their largest corner (module
    docstring); frozen (T+Q,): cells never split or re-split (R24)."""
    if mesh.n_quads == 0:
        return mesh, 0
    topo = build_topology(mesh)
    T = mesh.n_tris
    qa = _deg(mesh.points, mesh.quads)
    mx, k = qa.max(axis=1), qa.argmax(axis=1)
    near = element_regions(mesh, topo, rows=ROWS)["_coast3"][T:]  # a node <= ROWS - 1 edge hops from the coast
    cand = np.flatnonzero((mx > target) & near & (True if frozen is None else ~np.asarray(frozen)[T:]))
    if not len(cand):
        return mesh, 0
    ee = topo.edge_elems
    inner = ee[:, 1] >= 0
    nb = [[] for _ in range(mesh.n_elems)]                       # element -> edge neighbours
    for a, b in ee[inner]:
        nb[a].append(int(b))
        nb[b].append(int(a))
    is_tri = np.zeros(mesh.n_elems, dtype=bool)
    is_tri[:T] = True
    ok = lambda ta: ta.max() < tri_max and ta.min() >= tri_min
    busy = np.zeros(mesh.n_elems, dtype=bool)
    drop, dropt, new, newq = [], [], [], []
    for qi in cand[np.argsort(-mx[cand])]:
        g = T + int(qi)
        q = np.roll(mesh.quads[qi], -int(k[qi]))
        tn = [n for n in nb[g] if is_tri[n] or busy[n]]
        if not tn:                                               # (a) two triangles through the largest corner
            t2 = np.array([[q[0], q[1], q[2]], [q[0], q[2], q[3]]], dtype=np.int64)
            if ok(_deg(mesh.points, t2)):
                is_tri[g] = True                                 # its neighbours see triangles from now on
                drop.append(int(qi))
                new.append(t2)
            continue
        if len(tn) != 1 or tn[0] >= T or busy[tn[0]]:            # (b) one triangle beside: the pentagon re-split
            continue
        t = int(tn[0])
        tri = mesh.tris[t]
        j = [i for i in range(4) if q[(i + 1) % 4] in tri and q[i] in tri]
        if not j:
            continue
        j = j[0]
        x = int(next(v for v in tri if v not in q))
        pent = np.r_[q[:j + 1], x, q[j + 1:]]                    # c = pent[0]
        best = None
        for d in (2, 3):                                         # diagonals from c
            qq = np.array([pent[[0, 1, 2, 3]] if d == 3 else pent[[0, 2, 3, 4]]], dtype=np.int64)
            tt = np.array([pent[[0, 3, 4]] if d == 3 else pent[[0, 1, 2]]], dtype=np.int64)
            qmx = _deg(mesh.points, qq).max()
            if qmx >= target or not ok(_deg(mesh.points, tt)):
                continue
            # the new triangle: no triangle across its outer edges (other than the removed one)
            outer = {frozenset(e) for e in ((tt[0, 0], tt[0, 1]), (tt[0, 1], tt[0, 2]), (tt[0, 2], tt[0, 0]))}
            hit = False
            for n in nb[g] + nb[t]:
                if n in (g, t) or not (is_tri[n] or busy[n]):
                    continue
                el = mesh.tris[n] if n < T else mesh.quads[n - T]
                ne = {frozenset((el[i], el[(i + 1) % len(el)])) for i in range(len(el))}
                hit |= bool(outer & ne)
            if not hit and (best is None or qmx < best[0]):
                best = (qmx, qq, tt)
        if best is None:
            continue
        busy[g] = busy[t] = True                                 # changed: a triangle for later checks
        drop.append(int(qi))
        dropt.append(t)
        new.append(best[2])
        newq.append(best[1])
    if not drop:
        return mesh, 0
    keep = np.ones(mesh.n_quads, dtype=bool)
    keep[drop] = False
    kt = np.ones(T, dtype=bool)
    kt[dropt] = False
    nt = np.vstack(new).astype(np.int64)
    out = Mesh(mesh.points, np.vstack([mesh.tris[kt], nt]), np.vstack([mesh.quads[keep], *newq]).astype(np.int64),
               mesh.node_flag, None if mesh.tri_zone is None else np.r_[mesh.tri_zone[kt], np.zeros(len(nt), bool)])
    return out, len(drop)


def coast_fix(mesh: Mesh, fixed: np.ndarray, target: float = TARGET_DEG,
              frozen: np.ndarray | None = None) -> tuple[Mesh, np.ndarray, dict]:
    """split_coast_quads, then the triangle clusters broken (new quads below target, new triangles inside the hard
    limits [TRI_HARD_MIN_DEG, TRI_MAX_DEG]; only free nodes removed).
    fixed: (N,) nodes the later fit keeps; returned for the new node numbering with the record. frozen (T+Q,): cells
    left as they are (R24): never split, their nodes kept fixed here, so the cluster break (which re-splits only the
    elements round a free node, and merges triangles) never takes them; frozen quads only."""
    from meshom.methods.m14.stitch_mix import break_clusters

    if frozen is not None and frozen.any():
        fixed = fixed.copy()
        fixed[mesh.quads[np.asarray(frozen)[mesh.n_tris:]].ravel()] = True
    m, n_split = split_coast_quads(mesh, target, frozen=frozen)
    tz = m.tri_zone if m.tri_zone is not None else np.zeros(m.n_tris, dtype=bool)
    order = np.r_[np.flatnonzero(fixed), np.flatnonzero(~fixed)]  # fixed nodes first: break_clusters keeps them
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    t2, q2, _, _, n_br, gone = break_clusters(m.points[order], inv[m.tris[~tz]], inv[m.quads],
                                              np.zeros(int((~tz).sum()), np.int64), np.zeros(m.n_quads, np.int64),
                                              quad_max=target, free_from=int(fixed.sum()),
                                              tri_lim=(TRI_HARD_MIN_DEG, TRI_MAX_DEG))
    keepn = np.ones(m.n_nodes, dtype=bool)                       # in the reordered numbering
    keepn[np.asarray(gone, dtype=np.int64)] = False
    nid = np.cumsum(keepn) - 1
    old = order[keepn]                                           # new node -> original node
    tris = np.vstack([nid[t2], nid[inv[m.tris[tz]]]]).astype(np.int64)
    out = Mesh(m.points[old], tris, nid[q2].astype(np.int64), m.node_flag[old],
               None if m.tri_zone is None else np.r_[np.zeros(len(t2), dtype=bool), np.ones(int(tz.sum()), bool)])
    return out, fixed[old], {"quads_split": n_split, "clusters_broken": int(n_br), "nodes_removed": len(gone)}
