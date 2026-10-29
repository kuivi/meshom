"""Local triangle calculus on S² (competitor B, decisions B126-B129; the fast B, B150-B153).

This is a local hybrid Q-zip prototype, not the arbitrary-path pure-quad
compass algorithm. Completed patch rewrites are judged after relaxation.
The boundary graph and its coordinates are preserved exactly.

Each sweep runs four phases in order (recasts of valence-3 stars, 3-5 quad
pairs, glides of triangles, EV/VE zips); a phase's seeds are listed when it
starts. A seed is skipped when it touches a node blocked by an operation
accepted earlier in the sweep; otherwise it is evaluated (`evaluate`: every
rewrite of the seed tried against the mesh as it is, which is left
unchanged) and its first accepted rewrite committed. The fast engine
(engine.py, B150) computes the same answer as this serial loop with read sets,
a cache across sweeps and worker processes.
"""
from collections import Counter
from functools import lru_cache
import time

import numpy as np

from meshom.geo import R_EARTH_M, arc_length
from meshom.mesh import Mesh, build_topology
from meshom.optimize import _convex_ok, boundary_loop, poly_splits
from meshom.quality.kernel import element_geometry
from .operators import Work, collapse_triangle_edge, split_triangle_tip

DEFAULTS = dict(passes=8, rings=2, fit_iter=100, fit_best=5, trigger_deg=118.0,
                tol=0.27, ratio=1.15, max_ratio=2.0, defect_weight=2.0,
                angle_weight=1.0, glide=True, recast=True, pair=True, zip=True,
                workers=None, cache=True, fit="numba", screen=True, window=None, inflight=1)

PHASES = ("recast", "pair", "glide", "zip")
READ_RINGS = 4      # B150: an evaluation reads the incidence and positions within max(4, rings + 2) element rings
                    # of its patches (the rewrite's `rings`, plus the fit's and the measure's two more)


@lru_cache(None)
def _patterns(n, nt):
    return tuple(poly_splits(tuple(range(n)), nt))


def splits(loop, nt):
    """All non-crossing quad/triangle partitions of an oriented patch loop."""
    return [tuple(tuple(loop[i] for i in c) for c in p) for p in _patterns(len(loop), nt)]


def trace_chord(work, elem, edge):
    """Trace opposite quad edges; stop at a triangle, coast, or a closed strip."""
    path, seen = [], set()
    a, b = edge
    while elem >= 0:
        if work.nv[elem] != 4:
            return path, "triangle", elem
        state = (elem, min(a, b), max(a, b))
        if state in seen:
            return path, "closed", elem
        seen.add(state)
        path.append(elem)
        c = work.cells(elem)
        k = next(i for i in range(4) if {c[i], c[(i + 1) % 4]} == {a, b})
        a, b = c[(k + 2) % 4], c[(k + 3) % 4]
        other = [e for e in work.edge_elems(a, b) if e != elem]
        if not other:
            return path, "boundary", -1
        elem = other[0]
    return path, "boundary", -1


def classify(mesh):
    """Composition-aware defect inventory, including (4,4,5) triangles."""
    topo = build_topology(mesh)
    tn = np.bincount(mesh.tris.ravel(), minlength=mesh.n_nodes)
    pure = (tn == 0) & ~topo.boundary_node
    return dict(n3=int(np.sum(pure & (topo.elems_per_node == 3))),
                n5=int(np.sum(pure & (topo.elems_per_node == 5))),
                other=int(np.sum(pure & ~np.isin(topo.elems_per_node, [3, 4, 5]))),
                triangle_patterns=dict(Counter(",".join(map(str, sorted(topo.elems_per_node[t])))
                                               for t in mesh.tris)))


def _sub(work, elems):
    elems = sorted(e for e in elems if work.nv[e])
    el, nv = work.el[elems], work.nv[elems]
    ids = np.unique(el[el >= 0])
    work._loc[ids] = np.arange(len(ids))
    tris = work._loc[el[nv == 3, :3]].reshape(-1, 3)
    quads = work._loc[el[nv == 4]].reshape(-1, 4)
    work._loc[ids] = -1
    return Mesh(work.P[ids], tris, quads, np.zeros(len(ids), np.int8)), ids, elems


def cluster_excess(work, seeds, reads=None, cap=64):
    """optimize._Work.cluster_excess (triangles outside the zones in clusters >= 3 reachable from the seed triangles,
    each cluster once, at most cap), recording the nodes of every visited triangle in reads (B150)."""
    seen, total = set(), 0
    for t in seeds:
        if t in seen or work.nv[t] != 3 or work.zone[t]:
            continue
        comp, i = [t], 0
        seen.add(t)
        while i < len(comp) and len(comp) < cap:
            for y in work.tri_nbrs(comp[i]):
                if y not in seen:
                    seen.add(y)
                    comp.append(y)
            i += 1
        if reads is not None:
            for y in comp[:i]:
                reads.update(work.cells(y))
        if len(comp) >= 3:
            total += len(comp)
    return total


def _measure(work, nodes, case, opts, reads=None):
    # A fixed collar includes every changed corner, edge and adjacent size pair.
    active = {e for v in nodes if work.alive[v] for e in work.at(v)}
    collar = {v for e in active for v in work.cells(e)}
    elems = active | {e for v in collar for e in work.at(v)}
    sub, ids, elems = _sub(work, elems)
    g = element_geometry(sub)
    qa, ta = np.degrees(g.quad_angles), np.degrees(g.tri_angles)
    topo = build_topology(sub)
    edges = topo.edges
    mid = sub.points[edges].sum(axis=1)
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    length = R_EARTH_M * arc_length(sub.points[edges[:, 0]], sub.points[edges[:, 1]])
    r = np.abs(np.log(length / case.h(mid)))
    means = np.r_[g.tri_edges.mean(axis=1), g.quad_edges.mean(axis=1)]
    pairs = topo.edge_elems[topo.edge_elems[:, 1] >= 0]
    ratio = np.maximum(means[pairs[:, 0]], means[pairs[:, 1]]) / np.minimum(means[pairs[:, 0]], means[pairs[:, 1]])
    qbad = np.maximum(qa - 100, 0) ** 2 + np.maximum(75 - qa, 0) ** 2
    tbad = np.maximum(ta - 100, 0) ** 2 + np.maximum(30 - ta, 0) ** 2
    energy = opts["angle_weight"] * (qbad.sum() + tbad.sum()) / 100.0
    energy += 50 * np.maximum(r - opts["tol"], 0).dot(np.maximum(r - opts["tol"], 0))
    energy += 50 * np.maximum(np.log(ratio / 1.10), 0).dot(np.maximum(np.log(ratio / 1.10), 0))
    pure = [v for v in nodes if work.alive[v] and not work.bnode[v] and work.ninc[v] > 0 and work.ninc[v] != 4
            and all(work.nv[e] == 4 for e in work.at(v))]
    tri = [e for e in elems if work.nv[e] == 3]
    p = [work.P[v] for v in pure]
    p += [work.P[list(work.cells(e))].mean(axis=0) for e in tri]
    if p:
        p = np.asarray(p)
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        d = case.dist_coast_km(p) * 1e3 / case.h(p)
        potential = np.minimum(d / 2.0, 1.0) + 0.02 * np.minimum(d, 40)
        energy += opts["defect_weight"] * potential.sum()
    hard = int((g.tri_area <= 0).sum() + (g.quad_area <= 0).sum())
    hard += int((ta < 20).sum() + (ta > 120).sum())
    hard += int((qa > 150).sum() + (np.sin(g.quad_angles) < 0.3).sum())
    min_edge = np.r_[g.tri_edges.min(axis=1), g.quad_edges.min(axis=1)]
    hard += int((min_edge < 0.01 * case.h(g.centroid)).sum())
    b2 = sum(sum(work.is_bedge(c[k], c[(k + 1) % 3]) for k in range(3)) >= 2
             for e in tri for c in [work.cells(e)])
    clusters = cluster_excess(work, tri, reads)
    return dict(energy=float(energy), hard=hard, ratio_bad=int((ratio > opts["ratio"]).sum()),
                ratio_max=float(ratio.max(initial=1)), edge_bad=int((r > 0.30).sum()),
                b2=b2, cluster=clusters, units=len(pure) + len(tri),
                gt110=int((qa.max(axis=1) > 110).sum()) if len(qa) else 0,
                gt120=int((qa.max(axis=1) > 120).sum()) if len(qa) else 0)


def _topology_ok(old, new):
    """The parts of _allowed that depend on the topology alone (B152: the same before and after a fit)."""
    return new["b2"] <= old["b2"] and new["cluster"] <= old["cluster"] and new["units"] <= old["units"]


def _allowed(old, new, opts):
    return (new["hard"] == 0 and new["b2"] <= old["b2"] and new["cluster"] <= old["cluster"]
            and new["ratio_max"] <= max(old["ratio_max"], opts["max_ratio"])
            and new["ratio_bad"] <= old["ratio_bad"]
            and new["edge_bad"] <= old["edge_bad"]
            and new["units"] <= old["units"]
            and new["energy"] < old["energy"] - 1e-7)


def _trial(work, rm, new, case, opts, nodes, old, fit=True, initial=None, reads=None):
    if not work.patch_ok(rm, new):
        return None
    if len(case.art_edges):
        # Fixed cut-out coordinates/edges alone do not protect the lattice
        # ports: changed incidence would hide defects at the artificial edge.
        old_inc = Counter((v, len(c)) for e in rm for c in [work.cells(e)] for v in c)
        new_inc = Counter((v, len(c)) for c in new for v in c)
        for v, k in old_inc.keys() | new_inc.keys():
            if v < len(case.kind) and case.kind[v] == 2 and old_inc[v, k] != new_inc[v, k]:
                return None
    before = work.P[list(nodes)].copy()
    rec = work.apply(rm, new, initial)
    if rec is None:
        return None
    if not _convex_ok(work.P, new):
        work.undo(rec)
        work.P[list(nodes)] = before
        return None
    if fit:
        work.fit(nodes, case.h, opts["tol"], dict(max_iter=opts["fit_iter"], rounds=1,
                 w_angle=2.0, w_wall=200.0, w_tol=50.0, ftol=1e-9), opts.get("fit", "numba"))
    measure = _measure(work, nodes, case, opts, reads)
    moves = {v: work.P[v].copy() for v in nodes if work.alive[v] and not work.fixed[v]}
    work.undo(rec)
    work.P[list(nodes)] = before
    return measure, moves


def _rewrite(work, rm, options, case, opts, initial=None, reads=None, base_out=None):
    """The best accepted rewrite of the elements rm among options (B127, B128), or None; the mesh is left as it was.
    Returns dict(nodes, before, after, tris_delta, op=(rm, new, moves)). base_out collects the patch nodes (the read
    set's centre, B150); reads collects the nodes the triangle-cluster searches visited."""
    base = {v for e in rm for v in work.cells(e)} | set(initial or {})
    if base_out is not None:
        base_out |= base
    nodes = work.rings(base, opts["rings"])
    old = _measure(work, nodes, case, opts, reads)
    candidates = []
    oldkey = {frozenset(work.cells(e)) for e in rm}
    for new in options:
        if {frozenset(c) for c in new} == oldkey:
            continue
        d = Counter(v for c in new for v in c)
        d.subtract(v for e in rm for v in work.cells(e))
        tn = Counter(v for c in new if len(c) == 3 for v in c)
        tn.subtract(v for e in rm if work.nv[e] == 3 for v in work.cells(e))
        du = sum(len(c) == 3 for c in new) - sum(work.nv[e] == 3 for e in rm)
        for v, delta in d.items():
            if work.bnode[v]:
                continue
            t0 = sum(work.nv[e] == 3 for e in work.at(v))
            n0 = work.ninc[v]
            du += int(n0 + delta > 0 and n0 + delta != 4 and t0 + tn[v] == 0)
            du -= int(n0 > 0 and n0 != 4 and t0 == 0)
        if du > 0:
            continue
        trial = _trial(work, rm, new, case, opts, nodes, old, fit=False, initial=initial, reads=reads)
        if trial:
            candidates.append((trial[0]["energy"], new, trial[0]))
    best = None
    for _, new, m0 in sorted(candidates, key=lambda c: c[0])[:opts["fit_best"]]:
        if opts.get("screen", True) and not _topology_ok(old, m0):
            continue                                        # B152: _allowed fails whatever the fit does
        result = _trial(work, rm, new, case, opts, nodes, old, initial=initial, reads=reads)
        if result and _allowed(old, result[0], opts) and (best is None or result[0]["energy"] < best[0]):
            best = (result[0]["energy"], new, result[1], result[0])
    if best is None:
        return None
    _, new, moves, after = best
    return dict(nodes=nodes, before=old, after=after,
                tris_delta=sum(len(c) == 3 for c in new) - sum(work.nv[e] == 3 for e in rm),
                op=(list(rm), tuple(new), moves))


# ------------------------------------------------------------------------------------------------ one seed

def _trigger(work, t, opts):
    star = {e for v in work.cells(t) for e in work.at(v)}
    sub, _, _ = _sub(work, star)
    qa = np.degrees(element_geometry(sub).quad_angles)
    return len(qa) and qa.max() >= opts["trigger_deg"]


def skip(work, kind, item, blocked) -> bool:
    """The serial loop's test before a seed is evaluated (monotone within a phase: once true, it stays true)."""
    if kind == "recast":
        return item in blocked or work.ninc[item] != 3
    if kind == "pair":
        e, f = item
        if not work.nv[e] or not work.nv[f]:
            return True
        return any(v in blocked for v in work.cells(e) + work.cells(f))
    return work.nv[item] != 3 or any(v in blocked for v in work.cells(item))


def evaluate(work, kind, item, case, opts, read_set=True) -> dict:
    """Evaluate one seed at the current mesh (the serial loop's body without the commit; the mesh is left exactly as
    it was): dict(res: the accepted rewrite (_rewrite's dict, its fresh node written -1) or None, tried: Counter,
    R: the read set (node ids whose incidence or position the evaluation may have read))."""
    base, reads, tried = set(), set(), Counter()
    res = None
    n0 = work.next_node
    if kind == "recast":
        v = item
        base.add(v)
        rm = work.at(v)
        loop = boundary_loop([work.cells(e) for e in rm])
        if loop is not None and len(loop) == 6:
            tried["recast_tried"] += 1
            res = _rewrite(work, rm, splits(loop, 2), case, opts, reads=reads, base_out=base)
            if res:
                res["node"] = v
    elif kind == "pair":
        e, f = item
        base.update(work.cells(e) + work.cells(f))
        loop = boundary_loop([work.cells(e), work.cells(f)])
        if loop is not None and len(loop) == 6:
            tried["pair_tried"] += 1
            res = _rewrite(work, [e, f], splits(loop, 2), case, opts, reads=reads, base_out=base)
    elif kind == "glide":
        t = item
        base.update(work.cells(t))
        if _trigger(work, t, opts):
            for q in work.nbrs(t):
                if work.nv[q] not in (3, 4) or work.zone[q]:
                    continue
                rm = [t, q]
                loop = boundary_loop([work.cells(e) for e in rm])
                if loop is None:
                    continue
                tried["glide_tried"] += 1
                res = _rewrite(work, rm, splits(loop, 1 if len(loop) % 2 else 0), case, opts, reads=reads,
                               base_out=base)
                if res:
                    break
    else:
        t = item
        base.update(work.cells(t))
        if _trigger(work, t, opts):
            proposals = []
            for k in range(3):
                candidate = collapse_triangle_edge(work, t, k)
                if candidate is not None:
                    proposals.append(candidate)
            for rm, new, initial in proposals:
                tried["EV_tried"] += 1
                res = _rewrite(work, rm, [new], case, opts, initial, reads=reads, base_out=base)
                if res:
                    res["kind"] = "EV"
                    break
            else:
                # VE cuts the four-quad fan into two equal halves.
                for tip in work.cells(t):
                    if work.ninc[tip] != 5 or work.bnode[tip]:
                        continue
                    neighbours = {v for e in work.at(tip) for v in work.cells(e)
                                  if v != tip and work.edge_elems(tip, v)} - set(work.cells(t))
                    for across in sorted(neighbours):
                        fresh = work.new_node(work.P[tip])
                        candidate = split_triangle_tip(work, t, tip, across, fresh)
                        res = None
                        if candidate is not None:
                            tried["VE_tried"] += 1
                            rm, new, initial = candidate
                            res = _rewrite(work, rm, [new], case, opts, initial, reads=reads, base_out=base)
                        work.discard_node(fresh)
                        if res:
                            res["kind"] = "VE"
                            _fresh_out(res, fresh)
                            break
                    if res:
                        break
    R = None
    if read_set:
        base.discard(-1)
        R = set(work.rings({v for v in base if v < n0}, max(READ_RINGS, opts["rings"] + 2))) | {v for v in reads
                                                                                               if v < n0}
        R = np.fromiter(R, dtype=np.int64, count=len(R))
    return dict(res=res, tried=tried, R=R)


def _fresh_out(res, fresh):
    """Write the fresh node of a VE rewrite as -1 (its id is given when the rewrite is committed)."""
    rm, new, moves = res["op"]
    new = tuple(tuple(-1 if v == fresh else v for v in c) for c in new)
    moves = {(-1 if v == fresh else v): x for v, x in moves.items()}
    res["op"] = (rm, new, moves)
    res["nodes"] = {(-1 if v == fresh else v) for v in res["nodes"]}


def commit(work, res):
    """Apply an accepted rewrite (a fresh node -1 gets the next node id); returns (undo record, the nodes it touched
    (removed and new cells, moved nodes), blocked nodes: its relaxed footprint)."""
    rm, new, moves = res["op"]
    nodes = res["nodes"]
    if -1 in moves:
        fresh = work.new_node(moves[-1])
        new = tuple(tuple(fresh if v == -1 else v for v in c) for c in new)
        moves = {(fresh if v == -1 else v): x for v, x in moves.items()}
        nodes = {(fresh if v == -1 else v) for v in nodes}
    old_cells = [v for e in rm for v in work.cells(e)]
    rec = work.apply(rm, new, moves)
    assert rec is not None, "a committed rewrite no longer fits (node table width)"
    work.mark(rec)
    touched = set(old_cells) | {v for c in new for v in c} | set(moves)
    return rec, touched, nodes


def phase_items(work, kind, seeds, blocked):
    """The seeds of a phase, listed when it starts (the serial loop's lists)."""
    if kind == "recast":
        return list(seeds)
    if kind == "pair":
        pairs = set()
        for v in seeds:
            if not work.alive[v] or v in blocked:
                continue
            for e in work.at(v):
                for f in work.nbrs(e):
                    if work.nv[e] == 4 and work.nv[f] == 4:
                        pairs.add(tuple(sorted((e, f))))
        return sorted(pairs)
    return np.flatnonzero((work.nv[:work.m] == 3) & ~work.zone[:work.m]).tolist()


def sweep_seeds(work):
    return [v for v in range(work.next_node) if work.alive[v] and not work.bnode[v]
            and work.ninc[v] == 3 and all(work.nv[e] == 4 for e in work.at(v))]


def fix(case, opts=None):
    """Connectivity rewrites and local angle/size relaxation, with fixed boundaries."""
    from .engine import Engine

    start = time.perf_counter()
    opts = {**DEFAULTS, **(opts or {})}
    if opts["workers"] is None:
        from meshom.meshfix.cavity import default_workers

        opts["workers"] = default_workers(case.mesh.n_nodes)
    record = dict(method="b_transport", case=case.name, opts=opts, before=classify(case.mesh),
                  passes=[], operations=[])
    eng = Engine(case, opts)
    try:
        for sweep in range(opts["passes"]):
            counts = eng.sweep(record["operations"])
            record["passes"].append(dict(sweep=sweep, **counts, wall_s=time.perf_counter() - start))
            print(record["passes"][-1], flush=True)
            if not sum(counts.get(k, 0) for k in ("recast", "glide", "pair", "EV", "VE")):
                break
    finally:
        eng.close()
    mesh, old_ids, _ = eng.work.to_mesh()
    record.update(after=classify(mesh), removed_nodes=case.mesh.n_nodes - mesh.n_nodes, stats=eng.stats)
    record["wall_s"] = time.perf_counter() - start
    return mesh, record
