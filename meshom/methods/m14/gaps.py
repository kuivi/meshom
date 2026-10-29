"""m14 adaptive gaps (decisions B63; option `gap_adapt`, default off): where two blocks stitch badly, the gap between
them is widened there and the fill stitched again.

Scoring (`score`). After a stitch every corridor of `stitch_mix.partition` between two blocks (class "corridor",
both objects block outlines; `_mix` hands its partition and element regions over, the m02 fill of stitch.py is
partitioned the same way from its own loops and each fill element is labelled by the background triangle holding its
centroid) is measured per block pair, its regions pooled, with `stitch_mix.measure_regions`: B53 defects per 10 cells
of corridor length, |ln s/h| p99 and the element ratio p99 over its edges, the quads' max-angle p99, and the largest
edge-connected triangle cluster with most of its triangles in it. Gates (GATES): defects <= 2.0 per 10 cells (G32's
level), |ln s/h| p99 <= 0.18 (G33), ratio p99 <= 1.15 (G35), quad p99 <= 135 deg (G34), clusters <= 2 triangles
(B62). The pair's exceedance = sum over the gates of max(0, value / limit - 1).

Trim (`trim_cells`, `rebuild`). For a failing pair each automatic block of the two loses the cells along its outline
run facing that corridor: every run edge's cell and the gap_rows - 1 cells behind it along the lattice normal (rows
counted inward), so the gap widens by about 2 gap_rows cells there and nowhere else. A block is rebuilt from its
first-stitch grid minus the union of its trims (cells in lattice indices, so trims of several corridors compose and
undo exactly), the cell set cleaned (blockmap.clean_cells, nothing removed comes back), node positions kept (no new
map, no new smoothing); a rebuild that leaves the block below min_cells or min_inradius, or with more than one outline
loop, drops that trim. Fixed blocks are never trimmed.

Rounds (`adapt`). Round r: the failing pairs not frozen get one more trim each, the worst exceedance first and at
most one per block (pairs sharing a block wait for a later round, so each change can be told apart), the whole
stitch runs again (the partition and the fill are global: there is no local re-stitch) and the trimmed pairs are
scored again. The keep rule (`keep`): the pair's exceedance falls and no gate it passed fails (`improved`), and the
summed exceedance of every corridor of its two blocks falls too (a trim must not move the trouble next door). The
pairs that did not improve themselves are undone first, then those whose blocks' corridors got worse together; after
an undo the stitch runs again and the rest are judged again, at most VERIFY times, after which a round still
unsettled is undone as a whole (undone pairs are frozen at their trims before the round). The returned mesh is the
stitch of the returned grids. Up to gap_rounds rounds; the loop ends when no pair is left to try. The record
(m14.json "gaps"): every pair's measures after the first stitch and after the last, per round the pairs tried, kept
and undone, the stitch times and the cells removed.
"""
import time

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from meshom.mesh import Mesh, build_topology

GATES = {"defects_per10": 2.0, "ln_sh_p99": 0.18, "ratio_p99": 1.15, "quad_p99_deg": 135.0, "tri_cluster": 2.0}
VERIFY = 2      # stitches after the first of a round that may undo trims; still unsettled then: the round is undone


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def gate_eval(m: dict, gates: dict = GATES) -> tuple[list, float]:
    """(failed gate names, exceedance = sum of max(0, value / limit - 1)) of one pair's measures m (ratio_p99: the
    excess over 1 against the limit's)."""
    failed, exc = [], 0.0
    for k, lim in gates.items():
        v = m.get(k)
        if v is None:
            continue
        x = (v - 1.0) / (lim - 1.0) if k == "ratio_p99" else v / lim
        if x > 1.0:
            failed.append(k)
            exc += x - 1.0
    return failed, exc


def improved(old: dict, new: dict | None) -> bool:
    """The pair's exceedance falls and no gate it passed fails."""
    if new is None:
        return False
    return new["excess"] < old["excess"] and set(new["failed"]) <= set(old["failed"])


def keep(p: tuple, before: dict, after: dict) -> bool:
    """The keep rule of a trimmed pair p (scores per pair before and after the stitch): p improved, and the summed
    exceedance of every corridor of its two blocks (p's neighbourhood; a corridor missing on one side counts with its
    other score, i.e. unchanged, or 0 when it is new) fell too."""
    if not improved(before[p], after.get(p)):
        return False
    nb = {q for q in set(before) | set(after) if set(q) & set(p)}
    b = sum(before[q]["excess"] if q in before else 0.0 for q in nb)
    a = sum(after[q]["excess"] if q in after else before[q]["excess"] for q in nb)
    return a < b


def locate(bg: Mesh, x: np.ndarray, k: int = 12) -> np.ndarray:
    """(len(x),) background triangle holding each unit vector x (its chord triangle, CCW seen from outside), -1 when
    none of the k nearest by centroid does."""
    P, T = bg.points, bg.tris
    cen = _unit(P[T].sum(axis=1))
    _, j = cKDTree(cen).query(x, k=min(k, len(T)))
    j = j.reshape(len(x), -1)
    a, b, c = P[T[j, 0]], P[T[j, 1]], P[T[j, 2]]
    xx = x[:, None, :]
    eps = -1e-12
    ins = ((np.einsum("ijk,ijk->ij", np.cross(a, b), xx) >= eps) & (np.einsum("ijk,ijk->ij", np.cross(b, c), xx) >= eps)
           & (np.einsum("ijk,ijk->ij", np.cross(c, a), xx) >= eps))
    first = np.argmax(ins, axis=1)
    return np.where(ins.any(axis=1), j[np.arange(len(x)), first], -1)


def _centroids(mesh: Mesh) -> np.ndarray:
    return _unit(np.r_[mesh.points[mesh.tris].sum(axis=1), mesh.points[mesh.quads].sum(axis=1)])


def _tri_clusters(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """(cluster label per triangle, cluster sizes): edge-connected triangle clusters."""
    T = mesh.n_tris
    if T == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    ee = build_topology(mesh).edge_elems
    m = (ee[:, 1] >= 0) & (ee[:, 0] < T) & (ee[:, 1] < T)
    lab = connected_components(coo_matrix((np.ones(int(m.sum())), (ee[m, 0], ee[m, 1])), shape=(T, T)),
                               directed=False)[1]
    return lab, np.bincount(lab)


def score(mesh: Mesh, h, st: dict, o: dict, part=None, er: np.ndarray | None = None, locate_all: bool = False) -> dict:
    """Per block pair (a, b), a < b (indices of the stitch's outline loops: the automatic blocks first): the pooled
    measures of its corridor regions, the gates failed, the exceedance, and its runs {block: local outline edge ids}.
    part / er: the partition and the element regions (stitch_mix's; else built from the stitch's fill loops and
    located). locate_all: mesh is not the stitch of st but made from it (gap_judge "post"): the partition is the
    stitch's, every element of mesh is located in it."""
    from meshom.methods.m14.stitch_mix import _runs, measure_regions, partition

    if part is None:
        part = st.get("partition")
        er = None if locate_all else st.get("elem_region")
    if part is None:
        part = partition(st["fill_coast"], st["fill_outl"], h, o)
    if er is None:
        er = np.full(mesh.n_elems, -1, dtype=np.int64)
        sel = np.ones(mesh.n_elems, dtype=bool) if locate_all else ~st["is_block"]
        t = locate(part.bg, _centroids(mesh)[sel])
        er[np.flatnonzero(sel)] = np.where(t >= 0, part.reg[np.maximum(t, 0)], -1)
    B, C = part.B, part.C
    pairs, reg_pair = {}, np.full(len(part.regions), -1, dtype=np.int64)
    for rec in part.regions:
        if rec["cls"] != "corridor":
            continue
        a, b = sorted(int(x) for x in rec["objs"])
        if not (0 <= a < B and 0 <= b < B):
            continue
        p = pairs.setdefault((a, b), {"id": len(pairs), "outline_edges": 0, "coast_edges": 0, "regions": [],
                                     "runs": {}})
        p["outline_edges"] += rec["outline_edges"]
        p["regions"].append(rec["id"])
        reg_pair[rec["id"]] = p["id"]
        for obj, (li, e) in _runs(part, rec["id"]).items():
            if obj in (a, b):
                assert li == C + obj
                p["runs"].setdefault(obj, []).append(e)
    if not pairs:
        return {}
    ep = np.where(er >= 0, reg_pair[np.maximum(er, 0)], -1)
    recs = sorted(pairs.values(), key=lambda x: x["id"])
    ms = measure_regions(mesh, h, ep, [{"id": x["id"], "outline_edges": x["outline_edges"], "coast_edges": 0}
                                       for x in recs])
    lab, size = _tri_clusters(mesh)
    clus = np.zeros(len(recs), dtype=np.int64)
    tp = ep[:mesh.n_tris]
    m = tp >= 0
    if m.any():                                         # (no triangle in any corridor: no cluster)
        key, cnt = np.unique(lab[m] * len(recs) + tp[m], return_counts=True)   # cluster -> its majority pair
        cl, pr = key // len(recs), key % len(recs)
        o_ = np.lexsort((-cnt, cl))
        cl, pr = cl[o_], pr[o_]
        first = np.r_[True, cl[1:] != cl[:-1]]
        np.maximum.at(clus, pr[first], size[cl[first]])
    out = {}
    for (a, b), x in pairs.items():
        m = ms[x["id"]]
        r = {"length_cells": m["length_cells"], "defects_per10": m.get("irr_mix_per10"), "ln_sh_p99": m["ln_sh_p99"],
             "ratio_p99": m["ratio_p99"], "quad_p99_deg": m["quad_p99_deg"], "tri_cluster": int(clus[x["id"]]),
             "tris": m["tris"], "elems": m["elems"], "n_regions": len(x["regions"]),
             "runs": {k: np.unique(np.concatenate(v)) for k, v in x["runs"].items()}}
        r["failed"], r["excess"] = gate_eval(r, o.get("gap_gates") or GATES)
        out[(a, b)] = r
    return out


def _cells(g) -> tuple[np.ndarray, np.ndarray]:
    """(cell lattice index of each quad (Q,2), node-id array over the lattice box with its origin): a block's cells."""
    from meshom.methods.m14.blockmap import _grid

    nid, i0, j0 = _grid(g.ij)
    return g.ij[g.quads].min(axis=1), (nid, np.array([i0, j0]))


def trim_cells(g, run_edges: np.ndarray, k: int) -> np.ndarray:
    """(M,2) lattice indices of the cells of grid g within k rows of its outline run (local outline edge ids): each run
    edge's cell and the k - 1 cells behind it along the lattice normal of the edge."""
    cij, _ = _cells(g)
    have = {tuple(c) for c in cij}
    K = len(g.outline)
    a, b = g.ij[g.outline[run_edges]], g.ij[g.outline[(run_edges + 1) % K]]
    lo = np.minimum(a, b)
    d = b - a
    along_i = d[:, 0] != 0
    out = set()
    for x, ai in zip(lo, along_i):
        n_ax = 1 if ai else 0                          # the normal axis: j for an edge along i
        c0 = x.copy()
        c1 = x.copy()
        c1[n_ax] -= 1                                  # the two cells beside the edge
        if tuple(c0) in have:
            c, s = c0, 1
        elif tuple(c1) in have:
            c, s = c1, -1
        else:
            continue
        for t in range(k):
            y = c.copy()
            y[n_ax] += s * t
            if tuple(y) in have:
                out.add(tuple(int(v) for v in y))
    return np.array(sorted(out), dtype=np.int64).reshape(-1, 2)


def rebuild(g, removed: np.ndarray, o: dict):
    """Grid g minus the cells `removed` (lattice indices), cleaned (nothing removed comes back), positions kept; None
    when the rest is below min_cells / min_inradius or not one outline loop."""
    from meshom.methods.m14.blockmap import BlockGrid, clean_cells, outline_loop

    if len(removed) == 0:
        return g
    cij, (nid, org) = _cells(g)
    shape = (nid.shape[0] - 1, nid.shape[1] - 1)
    cells = np.zeros(shape, dtype=bool)
    q_of = np.full(shape, -1, dtype=np.int64)
    ci, cj = (cij - org).T
    cells[ci, cj] = True
    q_of[ci, cj] = np.arange(len(cij))
    keep = cells.copy()
    r = removed - org
    inb = (r[:, 0] >= 0) & (r[:, 1] >= 0) & (r[:, 0] < shape[0]) & (r[:, 1] < shape[1])
    keep[r[inb, 0], r[inb, 1]] = False
    keep = clean_cells(keep, keep, keep)
    n_cells = int(keep.sum())
    if n_cells < o["min_cells"]:
        return None
    inrad = float(ndimage.distance_transform_cdt(np.pad(keep, 1), metric="taxicab").max())
    if inrad < o["min_inradius"]:
        return None
    loops = outline_loop(keep, nid)
    if len(loops) != 1:
        return None
    loop = loops[0][::-1] if g.info.get("flipped_orientation") else loops[0]
    quads = g.quads[q_of[keep]]
    used = np.unique(quads)
    new = np.full(len(g.points), -1, dtype=np.int64)
    new[used] = np.arange(len(used))
    info = dict(g.info, n_cells=n_cells, inradius_cells=inrad, gap_cells_removed=int(len(cij) - n_cells))
    return BlockGrid(g.points[used], new[quads], g.ij[used], new[loop], info, g.map)


def _brief(s: dict | None) -> dict | None:
    if s is None:
        return None
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "runs"}


def _key(p: tuple) -> str:
    return f"{p[0]}-{p[1]}"


def adapt(domain, h, grids: list, o: dict, fixed: list, mesh: Mesh, st: dict, post=None) -> tuple:
    """The rounds of the module docstring from the first stitch (mesh, st) of the automatic blocks `grids`; returns
    (mesh, st, grids, record). The returned mesh is always the stitch of the returned grids."""
    from meshom.methods.m14.stitch import stitch

    t0 = time.perf_counter()
    nauto, base, k_rows = len(grids), list(grids), int(o["gap_rows"])

    def judge(m: Mesh, s: dict) -> dict:
        """The pair scores of stitch (m, s): on the stitch, or (post) on the mesh the variant's later stages make of
        it, every element of it located in the stitch's partition."""
        if post is None:
            return score(m, h, s, o)
        return score(post(m, s), h, s, o, locate_all=True)

    cur = judge(mesh, st)
    rec = {"rounds": [], "pairs_first": len(cur), "failed_first": sum(bool(s["failed"]) for s in cur.values()),
           "first": {_key(p): _brief(s) for p, s in cur.items()}}
    trims: dict = {}                                    # pair -> one [(block, removed cells), ...] per kept round
    frozen: set = set()
    cur_grids = list(grids)

    def build(tr: dict, only: set | None = None) -> tuple[list, set]:
        """Grids: the base minus the trims (only the blocks in `only`, if given); the pairs whose block's rebuild
        fails."""
        out, bad, rem = list(base), set(), {}
        for p, lst in tr.items():
            for b, c in (x for rnd in lst for x in rnd):
                rem.setdefault(b, []).append(c)
        for b, cs in rem.items():
            if only is not None and b not in only:
                continue
            g = rebuild(base[b], np.unique(np.vstack(cs), axis=0), o)
            if g is None:
                bad |= {p for p, lst in tr.items() if any(x == b for rnd in lst for x, _ in rnd)}
            else:
                out[b] = g
        return out, bad

    for rnd in range(int(o["gap_rounds"])):
        cand = sorted((p for p, s in cur.items() if s["failed"] and p not in frozen and min(p) < nauto),
                      key=lambda p: -cur[p]["excess"])
        new, tried, busy = {p: list(v) for p, v in trims.items()}, [], set()
        for p in cand:                                  # one more trim per failing pair, worst first, one per block
            if set(p) & busy:
                continue
            add = [(b, trim_cells(cur_grids[b], e, k_rows)) for b, e in cur[p]["runs"].items() if b < nauto]
            add = [(b, c) for b, c in add if len(c)]
            test = dict(new)
            test[p] = test.get(p, []) + [add]
            if not add or p in build(test, {b for b, _ in add})[1]:
                frozen.add(p)                           # nothing to trim, or a block would get too small
                continue
            new, tried = test, tried + [p]
            busy |= set(p)
        if not tried:
            break
        rr = {"round": rnd + 1, "tried": [_key(p) for p in tried], "undone": [], "stitches": []}
        live = list(tried)
        for it in range(VERIFY + 1):                    # stitch; undo the trims that are not kept; stitch again
            g_try, _ = build(new)
            t1 = time.perf_counter()
            try:
                mesh2, st2 = stitch(domain, h, g_try, o, fixed)
            except RuntimeError as e:                   # the fill failed on the trimmed layout: the round is undone
                rr["error"] = str(e)[:300]
                bad = list(live)
            else:
                sc = judge(mesh2, st2)
                rr["stitches"].append(round(time.perf_counter() - t1, 2))
                bad = [p for p in live if not improved(cur[p], sc.get(p))]      # the pair itself first,
                bad = bad or [p for p in live if not keep(p, cur, sc)]        # then its blocks' corridors together
                if bad and it == VERIFY:                # still not settled: the whole round is undone
                    bad = list(live)
            for p in bad:                               # back to the trims it had before this round; frozen
                frozen.add(p)
                if trims.get(p):
                    new[p] = list(trims[p])
                else:
                    new.pop(p)
            rr["undone"] += [_key(p) for p in bad]
            live = [p for p in live if p not in bad]
            if not bad or not live:
                break
        rr["kept"] = [_key(p) for p in live]
        if live:                                        # the stitch of this round's kept trims, each one kept
            trims, mesh, st, cur_grids = new, mesh2, st2, g_try
            rr["before"] = {_key(p): _brief(cur[p]) for p in live}
            cur = sc
            rr["after"] = {_key(p): _brief(cur[p]) for p in live}
        rr["cells_removed"] = sum(int(g.info.get("gap_cells_removed", 0)) for g in cur_grids)
        rec["rounds"].append(rr)
    rec.update(trimmed={_key(p): len(v) for p, v in trims.items()}, failed_last=sum(bool(s["failed"])
               for s in cur.values()), pairs_last=len(cur), last={_key(p): _brief(s) for p, s in cur.items()},
               cells_removed=sum(int(g.info.get("gap_cells_removed", 0)) for g in cur_grids),
               s=round(time.perf_counter() - t0, 2))
    return mesh, st, cur_grids, rec


def trim_pairs(domain, h, grids: list, o: dict, fixed: list, mesh: Mesh, st: dict, seeds: np.ndarray) -> tuple:
    """The advisor's per-corridor gaps (option gap_trim): each request {"a": [lon, lat], "b": [lon, lat], "rows": k}
    names the two automatic blocks whose seeds (seeds: (B,3)) are nearest to a and b; when they share a corridor in
    the first stitch, both lose k cell rows along it (trim_cells, trims of one block composed), the blocks are
    rebuilt (a rebuild that fails drops that block's trims) and the stitch runs once more. Returns (mesh, st, grids,
    record)."""
    from meshom.geo import lonlat_to_xyz
    from meshom.methods.m14.stitch import stitch

    cur = score(mesh, h, st, o)
    rem, rec = {}, {"requests": [], "stitched": False}
    for q in o["gap_trim"]:
        xa, xb = (lonlat_to_xyz(np.array([q[s][0]]), np.array([q[s][1]]))[0] for s in ("a", "b"))
        a, b = (int(np.argmin(np.linalg.norm(seeds - x, axis=1))) for x in (xa, xb))
        p = (min(a, b), max(a, b))
        r = {"pair": _key(p), "rows": int(q["rows"]), "found": p in cur and a != b}
        if r["found"]:
            for blk, e in cur[p]["runs"].items():
                if blk < len(grids):
                    rem.setdefault(blk, []).append(trim_cells(grids[blk], e, int(q["rows"])))
        rec["requests"].append(r)
    out = list(grids)
    for blk, cs in rem.items():
        c = np.unique(np.vstack(cs), axis=0)
        g = rebuild(grids[blk], c, o)
        rec.setdefault("rebuilt", {})[str(blk)] = None if g is None else int(g.info.get("gap_cells_removed", 0))
        if g is not None:
            out[blk] = g
    if any(v for v in rec.get("rebuilt", {}).values()):
        mesh, st = stitch(domain, h, out, o, fixed)
        rec["stitched"] = True
    rec["before"] = {_key(p): _brief(s) for p, s in cur.items()}
    return mesh, st, out, rec
