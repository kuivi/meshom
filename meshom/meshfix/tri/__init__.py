"""Stage t of the mesh-fixing pipeline: improve a pure-triangle mesh in place (decisions B154-B157).

    python -m meshom.meshfix.pipeline --case NAME --stages t,a      (or --run RUN_DIR of a meshom triangle run)

Per round (kernels.py, B155 / B156):
1. valence flips over every interior edge: an edge whose flip lowers the weighted valence energy
   sum w (val - tgt)^2 over its four nodes (tgt 6 inside, round(theta / 60) + 1 at a coast node with water angle
   theta; w = 1 + kappa_d * (distance to the nearest other defect, defect_weights) [+ kappa * (distance to the
   coast), off by default], in local edge lengths, so that 5-7 pairs glide toward each other, where opposite pairs
   cancel) is flipped, its patch relaxed (`sweeps` Gauss-Seidel sweeps of relax_node over the movable nodes of the
   four) and kept only when the patch measure does not get worse (kernels.verdict: angles, size fidelity,
   neighbour ratio, energy; a glide may raise the energy by ea_slack); else undone;
2. `smooth` Gauss-Seidel sweeps of relax_node over the nodes the accepted flips touched and their neighbours
   (the nodes of bad triangles in the first round), each move kept only when its star's energy falls and nothing
   else gets worse.
Both run in parallel over coloured blocks (blocks.py) on meshes above 4 blocks.
Coast nodes (kind 1) and the artificial boundary (kind 2) never move here (A slides the coast afterwards, pipeline
order t,a); a flip never touches an artificial node (its incidence stays, B121). The stage stops when a round accepts
fewer than `min_flips` flips, when the irregular count has not fallen for `patience` rounds (glides
can cycle: on farc 41-43 flips per round went back and forth from round 9 on) or after `rounds` rounds.

Guard (B157): the whole-mesh measures of the result against the input (guard_summary); when any gets worse beyond
its slack, the input is returned unchanged (record "guard": "kept input"). The measures: interior irregular nodes,
obtuse triangles, max angle, min angle, max-angle p99, |ln s/h| p99 and max, element ratio p99 and max.
"""
import time

import numpy as np

from meshom.delaunay import adjacency
from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh, build_topology

METHOD = "t_tri"

DEFAULTS = {
    # B155: the acceptance rule's free values (a change may move a measure freely below / above these); None = the
    # input's own whole-mesh tail: max-angle p99, min-angle p1, |ln s/h| p99, element ratio p99
    "amax_free": None, "amin_free": None, "sz_free": None, "rt_free": None, "ea_slack": 0.2,
    # B156: operators
    "relax": 1.0, "w_size": 1.0, "w_ratio": 1.0, "sweeps": 3, "rings": 0, "vmin": 4, "vmax": 8,
    "kappa": 0.0, "kappa_d": 0.1,
    "rounds": 30, "min_flips": 1, "patience": 2, "smooth": 3, "seed_dev": 20.0, "parallel": True, "block_nodes": 2000,
    # B157: the stage guard
    "guard": True,
}

GUARD_SLACK = {"irregular": 0, "obtuse": 0, "amax": 0.5, "amin": 0.5, "amax_p99": 0.5, "lnsh_p99": 0.005,
               "lnsh_max": 0.02, "ratio_p99": 0.005, "ratio_max": 0.02}       # B157


def _prm(o: dict) -> np.ndarray:
    d = np.pi / 180.0
    return np.array([o["amax_free"] * d, o["amin_free"] * d, o["sz_free"], np.log(o["rt_free"]), o["ea_slack"],
                     o["relax"], o["sweeps"], o["rings"], o["vmin"], o["vmax"], o["w_size"], o["w_ratio"]],
                    dtype=np.float64)


def coast_target(P: np.ndarray, tris: np.ndarray, bnode: np.ndarray) -> np.ndarray:
    """Target valence: 6 inside; at a boundary node round(theta / 60) + 1 (theta its water angle), at least 2."""
    from meshom.quality.kernel import element_geometry

    n = len(P)
    m = Mesh(P, tris, np.zeros((0, 4), np.int64), np.zeros(n, np.int8))
    theta = np.bincount(tris.ravel(), weights=element_geometry(m).tri_angles.ravel(), minlength=n)
    tgt = np.full(n, 6, dtype=np.int64)
    tb = np.maximum(np.rint(np.degrees(theta) / 60.0).astype(np.int64), 1) + 1
    tgt[bnode] = tb[bnode]
    return tgt


def guard_summary(mesh: Mesh, h) -> dict:
    """Whole-mesh measures of the guard (module docstring), with the scorer's kernels (element tangent planes, h at
    the edge midpoints, the mean-edge ratio across interior edges)."""
    from meshom.geo import arc_length
    from meshom.quality.kernel import element_geometry

    P, T = mesh.points, mesh.tris
    topo = build_topology(mesh)
    a = np.degrees(element_geometry(mesh).tri_angles)
    amin, amax = a.min(1), a.max(1)
    E = topo.edges
    L = arc_length(P[E[:, 0]], P[E[:, 1]])
    mid = P[E[:, 0]] + P[E[:, 1]]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    r = np.abs(np.log(L * R_EARTH_M / h(mid)))
    deg = np.bincount(E.ravel(), minlength=mesh.n_nodes)
    inner = ~topo.boundary_node
    me = (arc_length(P[T[:, 0]], P[T[:, 1]]) + arc_length(P[T[:, 1]], P[T[:, 2]]) +
          arc_length(P[T[:, 2]], P[T[:, 0]])) / 3.0
    ee = topo.edge_elems
    ee = ee[ee[:, 1] >= 0]
    rho = np.exp(np.abs(np.log(me[ee[:, 0]] / me[ee[:, 1]]))) if len(ee) else np.ones(1)
    return {"irregular": int((inner & (deg != 6)).sum()), "obtuse": int((amax > 90).sum()),
            "amax": float(amax.max()), "amin": float(amin.min()), "amax_p99": float(np.percentile(amax, 99)),
            "amin_p1": float(np.percentile(amin, 1)),
            "lnsh_p99": float(np.percentile(r, 99)), "lnsh_max": float(r.max()),
            "ratio_p99": float(np.percentile(rho, 99)), "ratio_max": float(rho.max())}


def guard_worse(before: dict, after: dict) -> list[str]:
    """The guard's measures that got worse beyond their slack (amin: lower is worse; every other: higher)."""
    out = []
    for k, s in GUARD_SLACK.items():
        if k == "amin":
            if after[k] < before[k] - s:
                out.append(k)
        elif after[k] > before[k] + s:
            out.append(k)
    return out


def defect_weights(P: np.ndarray, val: np.ndarray, bnode: np.ndarray, lnode: np.ndarray, kappa: float,
                   cap: float = 20.0) -> np.ndarray:
    """kappa * min(cap, distance (in local edge lengths) to the nearest irregular interior node more than 1.5 local
    edge lengths away): the valence energy then falls when defects glide toward other defects, where pairs of
    opposite dislocations can cancel (B156, opt kappa_d)."""
    from scipy.spatial import cKDTree

    irr = np.flatnonzero((val != 6) & ~bnode)
    if len(irr) < 2:
        return np.zeros(len(P))
    k = min(8, len(irr))
    d, j = cKDTree(P[irr]).query(P, k=k)
    d = d.reshape(len(P), k) / np.maximum(lnode, 1e-12)[:, None]
    d[d < 1.5] = np.inf
    return kappa * np.minimum(d.min(1), cap)


def _dilate(tris: np.ndarray, mark: np.ndarray) -> np.ndarray:
    """mark grown by one ring (every node of a triangle with a marked node), int8."""
    out = np.zeros(len(mark), np.int8)
    out[tris[(mark[tris] > 0).any(1)].ravel()] = 1
    return out


def improve(mesh: Mesh, h, kind: np.ndarray, opts: dict | None = None, log=print) -> tuple[Mesh, dict]:
    """Stage t on a pure-triangle mesh (module docstring): (mesh, info). kind: 0 interior (moves), 1 coast (held),
    2 artificial boundary (held, incidence locked)."""
    from meshom.meshfix.tri import kernels as K

    o = {**DEFAULTS, **(opts or {})}
    t0 = time.perf_counter()
    P = np.ascontiguousarray(mesh.points, dtype=np.float64).copy()
    tris = np.ascontiguousarray(mesh.tris, dtype=np.int64).copy()
    n = len(P)
    nbr = adjacency(tris)
    vt = np.full(n, -1, np.int64)
    vt[tris.ravel()] = np.repeat(np.arange(len(tris), dtype=np.int64), 3)
    topo = build_topology(mesh)
    bnode = topo.boundary_node.copy()
    val = np.bincount(topo.edges.ravel(), minlength=n).astype(np.int64)
    mov = (np.asarray(kind) == 0) & ~bnode
    lock = np.asarray(kind) == 2
    tgt = coast_target(P, tris, bnode)
    w = np.ones(n)
    coast = np.flatnonzero((np.asarray(kind) == 1) & bnode)       # the real coast (never the artificial boundary)
    from meshom.geo import arc_length

    E0 = topo.edges
    L0 = arc_length(P[E0[:, 0]], P[E0[:, 1]])
    lnode = (np.bincount(E0[:, 0], weights=L0, minlength=n) + np.bincount(E0[:, 1], weights=L0, minlength=n)) \
        / np.maximum(val, 1)
    w_coast = np.zeros(n)
    if o["kappa"] > 0 and len(coast):
        from scipy.spatial import cKDTree

        d, _ = cKDTree(P[coast]).query(P)
        w_coast = o["kappa"] * np.minimum(d / np.maximum(lnode, 1e-12), 1e4)
    w = 1.0 + w_coast
    g0 = guard_summary(mesh, h)
    for k, gk in (("amax_free", "amax_p99"), ("amin_free", "amin_p1"), ("sz_free", "lnsh_p99"),
                  ("rt_free", "ratio_p99")):
        if o[k] is None:
            o[k] = g0[gk]
    prm = _prm(o)
    info = {"opts": o, "input": g0, "rounds": []}
    # first active set: nodes of bad triangles (worst |angle - 60| > seed_dev) and their neighbours
    from meshom.quality.kernel import element_geometry

    dev = np.degrees(np.abs(element_geometry(mesh).tri_angles - np.pi / 3).max(1))
    active = np.zeros(n, np.int8)
    active[tris[dev > o["seed_dev"]].ravel()] = 1
    cand = np.ones(n, np.int8)
    parts = []
    if o["parallel"] and n > 4 * o["block_nodes"]:
        from meshom.meshfix.tri import blocks

        tb = time.perf_counter()
        parts = [blocks.partition(P, tris, int(o["block_nodes"]), rotate=r) for r in (0, 1)]
        info["blocks"] = {"n": [int(b.max()) + 1 for b, _ in parts], "colours": [len(c) for _, c in parts],
                          "s": time.perf_counter() - tb}
        log(f"tri blocks: {info['blocks']}")
    best, since = np.inf, 0
    for rnd in range(int(o["rounds"])):
        tr = time.perf_counter()
        hr = np.asarray(h(P), dtype=np.float64) / R_EARTH_M
        if o["kappa_d"] > 0:
            w = 1.0 + w_coast + defect_weights(P, val, bnode, lnode, o["kappa_d"])
        touched = np.zeros(n, np.int8)
        fs = np.zeros(11, np.int64)
        args = (P, tris, nbr, vt, val, mov, lock, bnode, tgt, w, hr, prm)
        if parts:
            nf = 0
            c = cand
            for blk, colours in parts:                       # each partition: its colours in parallel
                deferred = np.zeros(n, np.int8)
                bptr, btri = blocks.csr(blk[tris[:, 0]], int(blk.max()) + 1)
                for col in colours:
                    bs = np.zeros((len(col), 11), np.int64)
                    nf += K.flip_blocks(*args, c, touched, bs, blk, col, bptr, btri, deferred)
                    fs += bs.sum(0)
                c = (deferred & cand) | _dilate(tris, _dilate(tris, touched))     # the next partition's candidates
            nf += K.flip_pass(*args, c & deferred, touched, fs)                 # the rest in sequence
        else:
            nf = K.flip_pass(*args, cand, touched, fs)
        t_flip = time.perf_counter() - tr
        active |= touched
        ring = _dilate(tris, active)
        ss = np.zeros(2, np.int64)
        for _ in range(int(o["smooth"])):
            if parts:
                blk, colours = parts[0]
                nptr, bnodes = blocks.csr(blk, int(blk.max()) + 1)
                deferred = np.zeros(n, np.int8)
                for col in colours:
                    bs = np.zeros((len(col), 2), np.int64)
                    K.smooth_blocks(P, tris, nbr, vt, mov, ring, hr, prm, touched, bs, blk, col, nptr, bnodes,
                                    deferred)
                    ss += bs.sum(0)
                K.smooth_pass(P, tris, nbr, vt, mov, ring & deferred, hr, prm, touched, ss)
            else:
                K.smooth_pass(P, tris, nbr, vt, mov, ring, hr, prm, touched, ss)
        irr = int(((val != 6) & ~bnode).sum())
        rec = {"round": rnd, "flips": int(nf), "tried": int(fs[0]), "rejected_measure": int(fs[2]),
               "rejected_topology": int(fs[3]), "verdicts": fs[4:].tolist(), "smooth_tried": int(ss[0]),
               "smooth_moved": int(ss[1]), "irregular": irr, "flip_s": t_flip, "s": time.perf_counter() - tr}
        info["rounds"].append(rec)
        log(f"tri round {rnd}: {rec}")
        active = np.zeros(n, np.int8)
        cand = _dilate(tris, _dilate(tris, touched))
        if irr < best:
            best, since = irr, 0
        else:
            since += 1
        if nf < o["min_flips"] or since >= o["patience"]:
            break
    out = Mesh(P, tris, np.zeros((0, 4), np.int64), mesh.node_flag)
    info["wall_s"] = time.perf_counter() - t0
    return out, info


def fix(case, opts: dict | None = None):
    """Bench contract: (Mesh, record). A mesh with quads is returned unchanged (stage t is for triangle meshes)."""
    o = {**DEFAULTS, **(opts or {})}
    t0 = time.perf_counter()
    mesh = case.mesh
    rec = {"method": METHOD, "case": case.name}
    if mesh.n_quads or not mesh.n_tris:
        rec.update(wall_s=0.0, skipped="the mesh has quads: stage t works on triangle meshes only")
        return mesh, rec
    out, info = improve(mesh, case.h, case.kind, o)
    if o["guard"]:
        b, a = guard_summary(mesh, case.h), guard_summary(out, case.h)
        worse = guard_worse(b, a)
        info.update(guard_before=b, guard_after=a, guard_worse=worse)
        if worse:
            info["guard"] = "kept input"
            print(f"tri guard: {worse} got worse, the input is kept", flush=True)
            out = mesh
    rec.update(wall_s=time.perf_counter() - t0, info=info)
    return out, rec
