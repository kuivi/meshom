"""m14 step 5: stitching.

The fill domain is the water minus the blocks: the coast loops (resampled at h as usual) plus one exact loop per block
outline (fill on the left, nodes taken as given). m02 fills it (packing -> Delaunay with the loops recovered ->
recombination -> its own smoothing with every loop node fixed; fill_method "tri": no recombination, then the population
control (meshom.popctl) inside the gap, loop nodes fixed). Merge: the fill's loop nodes come first in loop order,
so the outline ones are mapped to the block node ids; node order of the result: coast nodes (flag 1), the blocks, the
fill interior. Then the seam band: element rows by breadth-first search from the outline edges over element
adjacency (not across the outline), band_block rows on the block side and band_fill rows on the fill side; nodes of
band elements that touch no element outside the band (outline nodes included, coast nodes never) are smoothed jointly:
`smooth` kind "size", then a few "angle" sweeps, connectivity frozen. fill_method "tri" then runs the per-edge size fit
(meshom/size_fit.py) on the merged mesh with the coast and every block node (outlines included) fixed, so only
the triangle fill moves (its Lawson flips never touch the quads: boundary of nothing, flips need two triangles).
With `size_tolerance` set (decisions G38), `tolerance_fit` is the last stage instead: meshom.size_fit.smooth_fit of
every fill node (coast and block nodes fixed), smoothness first within |ln s/h| <= size_tolerance; the "tri" fill then
skips its size fit.

Fixed blocks (m14/strait.py; decision B59) are grids placed before the automatic blocks, with sides on the
coast. Conforming (domain loops given): the coast loops they touch are re-assembled (`merged_loops`): each fixed
block's non-coast outline runs (its mouths) join the coast pieces between them, walked forward along the loops (water on
the left) and resampled at h (corners kept, none within h/2 of a block corner); every node of such a loop is exact,
the run nodes are the block's own; loops no block touches stay as they are. The domain's `exact` flags are honoured.
Variant A (no domain loops, the land cut follows): a fixed block's whole outline is one more exact loop; so is, in
conforming mode, a hand-placed block without a coast side (m14/hand.py: its sides lie in water). Either way the
fixed blocks' quads join the mesh after the automatic blocks, their outline edges off the coast are seam edges, and the
band smoothing never moves their coast-side nodes.
"""
import numpy as np
from scipy.sparse import coo_matrix

from meshom.domain import Domain, _place, _place_jf, _turn_angles, _uncross, resample_jf, resample_loops
from meshom.geo import R_EARTH_M, arc_length
from meshom.mesh import Mesh, build_topology
from meshom.size import SizeField

MAX_ROWS = 8          # rows counted on each side of the outline (for the report)


def element_rows(mesh: Mesh, is_block: np.ndarray, outline_edges: np.ndarray, max_rows: int = MAX_ROWS):
    """(T+Q,) row of each element counted from the outline (0 = touches an outline edge), -1 beyond max_rows; rows
    run on each side separately (block elements are reached only from block elements). Also returns the topology and
    the (E,) outline-edge mask."""
    topo = build_topology(mesh)
    n = mesh.n_nodes
    ekey = topo.edges[:, 0] * n + topo.edges[:, 1]
    ol = np.sort(outline_edges, axis=1)
    is_ol = np.isin(ekey, ol[:, 0] * n + ol[:, 1])
    ee = topo.edge_elems
    M = mesh.n_elems
    row = np.full(M, -1, dtype=np.int64)
    start = ee[is_ol].ravel()
    row[start[start >= 0]] = 0
    inner = (ee[:, 1] >= 0) & ~is_ol
    a, b = ee[inner, 0], ee[inner, 1]
    same = is_block[a] == is_block[b]
    a, b = a[same], b[same]
    adj = coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(M, M)).tocsr()
    front = row == 0
    for r in range(1, max_rows + 1):
        reach = (adj @ front.astype(np.float64)) > 0
        new = reach & (row < 0)
        if not new.any():
            break
        row[new] = r
        front = new
    return row, topo, is_ol


def resample_open(p: np.ndarray, h: SizeField, corner_deg: float = 30.0, jf: bool = False):
    """An open coast piece resampled with spacing h (equal steps of the integral of ds/h, at least one), its ends and
    its corners (vertices turning by more than corner_deg, none within h/2 of an end) kept. jf: also the points'
    (vertex, fraction) along p."""
    if len(p) < 2:
        return (p, np.arange(len(p)), np.zeros(len(p))) if jf else p
    hv = h(p)
    w = R_EARTH_M * arc_length(p[:-1], p[1:]) * 0.5 * (1.0 / hv[:-1] + 1.0 / hv[1:])
    cum = np.r_[0.0, np.cumsum(w)]
    turn = np.zeros(len(p))
    if len(p) > 2:
        turn[1:-1] = _turn_angles(np.vstack([p, p[-1:] * 0 + 2 * p[-1:] - p[-2:-1]]))[1:-2]
    far = (cum > 0.5) & (cum < cum[-1] - 0.5)
    cuts = np.r_[0, np.flatnonzero((turn > np.radians(corner_deg)) & far), len(p) - 1]
    pieces, jl, fl = [], [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        ww = w[a:b]
        n = max(1, int(round(float(ww.sum()))))
        pieces.append(_place(p[a:b + 1], ww, np.arange(n) * (ww.sum() / n)))
        j, f = _place_jf(ww, np.arange(n) * (ww.sum() / n))
        jl.append(a + j)
        fl.append(f)
    if jf:                                                          # the points' (vertex, fraction) along p
        return np.vstack(pieces + [p[-1:]]), np.r_[np.concatenate(jl), len(p) - 1], np.r_[np.concatenate(fl), 0.0]
    return np.vstack(pieces + [p[-1:]])


def merged_loops(domain: Domain, h: SizeField, fixed: list, exact: list = ()) -> tuple[list, list, set]:
    """Conforming mode: the coast loops with the fixed blocks' coast sides replaced by their other sides (module
    docstring). Returns (loops, ids, touched): ids[k][i] = the node's index among the fixed blocks' nodes (in block
    order) or -1 for a coast node; touched = the domain loops that were re-assembled. The coast walks are refined
    until no chord crosses another of the fill's loops (the untouched loops at h, the `exact` outlines; C38)."""
    foff = np.r_[0, np.cumsum([len(f.grid.points) for f in fixed])]
    runs = [(int(foff[b]) + ids, f.grid.points[ids], s, e) for b, f in enumerate(fixed) for ids, s, e in f.runs]
    pos = lambda loc: loc[1] + loc[2]                               # noqa: E731  position along loop loc[0]
    used = [False] * len(runs)
    loops, idl, touched, srcs, sids, sjs, sfs = [], [], set(), [], [], [], []
    for r0 in range(len(runs)):
        if used[r0]:
            continue
        pts, ids, ri, S, sid, sj, sf = [], [], r0, [], [], [], []
        while not used[ri]:
            used[ri] = True
            gid, rp, _, end = runs[ri]
            pts.append(rp)
            ids.append(gid)
            L = domain.loops[end[0]]
            K = len(L)
            touched.add(end[0])
            rn = min((rj for rj in range(len(runs)) if runs[rj][2][0] == end[0]),
                     key=lambda rj: (pos(runs[rj][2]) - pos(end)) % K)
            st = runs[rn][2]
            cnt = (st[1] - end[1]) % K
            if cnt == 0 and st[2] < end[2]:
                cnt = K
            walk = np.vstack([rp[-1:], L[(end[1] + 1 + np.arange(cnt)) % K], runs[rn][1][:1]])
            walk = walk[np.r_[True, np.linalg.norm(np.diff(walk, axis=0), axis=1) > 1e-12]]
            piece, pj, pf = resample_open(walk, h, jf=True)
            piece, pj, pf = piece[1:-1], pj[1:-1], pf[1:-1]
            pts.append(piece)
            ids.append(np.full(len(piece), -1, dtype=np.int64))
            o0 = sum(len(x) for x in S)                                 # the source: run nodes, then the walk's
            S += [rp, walk[1:-1]]                                       # inner vertices (walk[j] = S[o0 + |rp| - 1 + j])
            sid += [gid, np.full(len(walk) - 2, -1, dtype=np.int64)]
            sj += [o0 + np.arange(len(rp)), o0 + len(rp) - 1 + pj]
            sf += [np.zeros(len(rp)), pf]
            ri = rn
        loops.append(np.vstack(pts))
        idl.append(np.concatenate(ids))
        srcs.append(np.vstack(S))
        sids.append(np.concatenate(sid))
        sjs.append(np.concatenate(sj))
        sfs.append(sf)
    if not loops:
        return loops, idl, touched
    rest = Domain([lp for i, lp in enumerate(domain.loops) if i not in touched], domain.name,
                  [i < len(domain.exact) and domain.exact[i] for i in range(len(domain.loops)) if i not in touched])
    r_out, r_src, r_js, r_fs = resample_jf(rest, h)
    nx = [None] * len(exact)
    out = loops + r_out + list(exact)
    sfs = [np.concatenate(x) for x in sfs]
    J, F = sjs + r_js + nx, sfs + r_fs + nx
    _uncross(srcs + r_src + nx, J, F, out)
    for k in range(len(loops)):
        if out[k] is not loops[k]:                                      # refined: run nodes exact, ids from S
            idk = np.where(F[k] == 0.0, sids[k][J[k]], -1)
            out[k][idk >= 0] = srcs[k][J[k][idk >= 0]]
            loops[k], idl[k] = out[k], idk
    return loops, idl, touched


def tolerance_fit(mesh: Mesh, h: SizeField, o: dict, block_nodes: list, info: dict) -> Mesh:
    """Step 5a (decisions G38, M38): meshom.size_fit.smooth_fit of the fill at o["size_tolerance"] with the coast
    (flag != 0) and every block node fixed; its record goes to info["tol_fit"]."""
    from meshom.size_fit import smooth_fit

    keep = mesh.node_flag != 0
    for bn in block_nodes:
        keep[bn] = True
    zone = mesh.tri_zone
    mesh, r = smooth_fit(mesh, h, float(o["size_tolerance"]), keep, **o.get("tol_fit", {}))
    mesh.tri_zone = zone
    info["tol_fit"] = {"tol": r["tol"], "before": r["before"], "after": r["after"], "free_nodes": r["free_nodes"],
                       "rounds": [(x["iters"], x["stop"], round(x["ratio_p99"], 4), round(x["ln_sh_p99"], 4))
                                  for x in r["rounds"]], "s": r["total_s"]}
    return mesh


def stitch(domain: Domain, h: SizeField, grids: list, o: dict, fixed: list = ()) -> tuple[Mesh, dict]:
    from meshom.methods import m02_linf_frontal as m02
    from meshom.recombine import recombine
    from meshom.smooth import smooth

    if o["fill_method"] == "mix":                                       # variant _mix: one fill per region class
        from meshom.methods.m14.stitch_mix import stitch_mix

        return stitch_mix(domain, h, grids, o)
    info = {}
    exact = [i < len(domain.exact) and domain.exact[i] for i in range(len(domain.loops))]
    cloops, cids = list(domain.loops), [None] * len(domain.loops)
    fixed = list(fixed)
    allg = list(grids) + [f.grid for f in fixed]
    outl = [g.points[g.outline] for g in grids]
    own = [not domain.loops or not f.coast.any() for f in fixed]       # the fixed blocks that are outline loops
    if any(f.coast.any() for f in fixed) and domain.loops:              # conforming: re-assembled coast loops
        ml, mids, touched = merged_loops(domain, h, fixed,
                                         outl + [f.grid.points[f.grid.outline] for f, w in zip(fixed, own) if w])
        keep = [i for i in range(len(domain.loops)) if i not in touched]
        cloops = ml + [domain.loops[i] for i in keep]
        cids = mids + [None] * len(keep)
        exact = [True] * len(ml) + [exact[i] for i in keep]
    outl += [f.grid.points[f.grid.outline] for f, w in zip(fixed, own) if w]   # variant A: cut later
    cdom = Domain(cloops, domain.name, exact)
    coast = resample_loops(cdom, h)
    cid = np.concatenate([np.full(len(lp), -1, dtype=np.int64) if c is None else c for lp, c in zip(coast, cids)]) \
        if coast else np.zeros(0, dtype=np.int64)
    nc = int((cid < 0).sum())
    fd = Domain(cloops + outl, domain.name, exact=exact + [True] * len(outl))
    if o["fill_method"] == "m07":                                       # position-field fill (Instant Meshes)
        from meshom.methods.m07_instant_meshes import instant_mesh

        fmesh, finfo = instant_mesh(fd, h, o["fill"])
        finfo = {k: v for k, v in finfo.items() if not isinstance(v, dict)}
    elif o["fill_method"] == "tri":                                     # triangles only (m02's asterisk packing)
        mo = m02._opts(dict({"n": 6}, **o["fill"]))
        fmesh, finfo = m02.triangle_mesh(fd, h, mo)
        fmesh = m02._smooth(fmesh, h, mo)
        if o["tri_popctl"]:                                             # popctl in the gap: loop nodes fixed
            from meshom.domain import loop_edges
            from meshom.popctl import popctl

            fmesh, pinfo = popctl(fmesh, h, None, loop_edges(resample_loops(fd, h)), rounds=o["tri_popctl"])
            info["fill_popctl"] = {"rounds": [(r["deleted"], r["inserted"], round(r["p99_before"], 4))
                                              for r in pinfo["rounds"]],
                                   "cleanup": pinfo["cleanup"], "p99_after": pinfo["p99_after"]}
        if o["tri_vflips"]:
            from meshom.popctl import valence_flips

            fmesh, info["fill_vflips"] = valence_flips(fmesh, h, fmesh.node_flag != 0)
    else:
        mo = m02._opts(o["fill"])
        fmesh, finfo = m02.triangle_mesh(fd, h, mo)
        fmesh = recombine(fmesh, mo["price"], mo["recombine"])
        fmesh = m02._smooth(fmesh, h, mo)
    info.update({f"fill_{k}": v for k, v in finfo.items()})
    ncl = len(cid)
    nb_f = ncl + sum(len(x) for x in outl)
    loops_f = resample_loops(fd, h)
    assert np.array_equal(fmesh.points[:nb_f], np.vstack(loops_f) if loops_f else fmesh.points[:0])
    offs = nc + np.r_[0, np.cumsum([len(g.points) for g in allg])].astype(np.int64)
    n_int = fmesh.n_nodes - nb_f
    N = int(offs[-1]) + n_int
    pts = np.empty((N, 3))
    fmap = np.empty(fmesh.n_nodes, dtype=np.int64)
    new = cid < 0
    fmap[:ncl][new] = np.arange(nc)
    fmap[:ncl][~new] = offs[len(grids)] + cid[~new]                     # the fixed blocks' mouth nodes
    pts[:nc] = fmesh.points[:ncl][new]
    k = ncl
    qb, ol_edges, block_nodes, bank_edges = [], [], [], []
    for b, g in enumerate(allg):
        pts[offs[b]:offs[b + 1]] = g.points
        lp = offs[b] + g.outline
        if b < len(grids) or own[b - len(grids)]:                      # outline loops of the fill
            fmap[k:k + len(lp)] = lp
            k += len(lp)
        qb.append(g.quads + offs[b])
        e = np.column_stack([lp, np.roll(lp, -1)])
        if b >= len(grids):                                            # a fixed block: its edges off the coast
            bk = fixed[b - len(grids)].bank()
            bank_edges.append(e[bk])
            e = e[~bk]
        ol_edges.append(e)
        block_nodes.append(np.arange(offs[b], offs[b + 1]))
    fmap[nb_f:] = offs[-1] + np.arange(n_int)
    pts[offs[-1]:] = fmesh.points[nb_f:]
    nqb = sum(len(q) for q in qb)
    quads = np.vstack(qb + [fmap[fmesh.quads]]).astype(np.int64) if allg else fmap[fmesh.quads]
    tris = fmap[fmesh.tris]
    flag = np.zeros(N, dtype=np.int8)
    flag[:nc] = 1
    pinned = np.concatenate([offs[len(grids) + i] + np.flatnonzero(f.coast) for i, f in enumerate(fixed)]) \
        if fixed else np.zeros(0, dtype=np.int64)
    if domain.loops:
        flag[pinned] = 1                                                # the fixed blocks' banks are the coast
    mesh = Mesh(pts, tris, quads, flag)
    is_block = np.zeros(mesh.n_elems, dtype=bool)
    is_block[mesh.n_tris:mesh.n_tris + nqb] = True
    fixed_id = np.full(mesh.n_elems, -1, dtype=np.int64)
    q0 = mesh.n_tris + sum(len(g.quads) for g in grids)
    for i, f in enumerate(fixed):
        fixed_id[q0:q0 + len(f.grid.quads)] = i
        q0 += len(f.grid.quads)
    ol_edges = np.vstack(ol_edges) if allg else np.zeros((0, 2), dtype=np.int64)
    info.update(n_fill_tris=int(fmesh.n_tris), n_fill_quads=int(fmesh.n_quads), n_block_quads=int(nqb),
                n_fixed_blocks=len(fixed))
    if allg and (o["band_size_iters"] or o["band_angle_iters"]):
        row, _, _ = element_rows(mesh, is_block, ol_edges)
        band = (row >= 0) & np.where(is_block, row < o["band_block"], row < o["band_fill"])
        in_band = np.zeros(N, dtype=bool)
        out_band = np.zeros(N, dtype=bool)
        for cells, off in ((mesh.tris, 0), (mesh.quads, mesh.n_tris)):
            bb = band[off:off + len(cells)]
            in_band[cells[bb].ravel()] = True
            out_band[cells[~bb].ravel()] = True
        movable = in_band & ~out_band & (flag == 0)
        movable[pinned] = False
        info["band_nodes"] = int(movable.sum())
        if o["band_size_iters"]:
            mesh = smooth(mesh, h, ~movable, iters=o["band_size_iters"], kind="size")
        if o["band_angle_iters"]:
            mesh = smooth(mesh, h, ~movable, iters=o["band_angle_iters"], kind="angle")
    if o["fill_method"] == "tri" and o["tri_fit"] and o.get("size_tolerance") is None:   # size fit: fill nodes only
        from meshom.size_fit import size_fit

        keep_fixed = flag != 0
        for bn in block_nodes:
            keep_fixed[bn] = True
        mesh, sinfo = size_fit(mesh, h, keep_fixed, p=o["tri_fit_p"], flips=o["tri_fit_flips"])
        info["fill_fit"] = {"p99_before": sinfo["p99_before"], "p99_after": sinfo["p99_after"],
                            "rounds": [(r["iters"], r["flips"], r["p99"]) for r in sinfo["rounds"]],
                            "s": sinfo["total_s"]}
    if o.get("size_tolerance") is not None:                             # G38: smoothness within the tolerance
        mesh = tolerance_fit(mesh, h, o, block_nodes, info)
    st = {"info": info, "is_block": is_block, "outline_edges": ol_edges,
          "block_nodes": np.concatenate(block_nodes) if allg else np.zeros(0, dtype=np.int64),
          "fill_coast": coast, "fill_outl": outl}                       # the fill's loops (m14/gaps.py partitions them)
    if fixed:                                       # protected in the land cut: the blocks with a coast side
        st.update(fixed_id=fixed_id, fixed_coast_nodes=pinned, fixed_bank_edges=np.vstack(bank_edges),
                  fixed_protect=np.array([bool(f.coast.any()) for f in fixed]),
                  fixed_kind=[f.record.get("kind", "strait") for f in fixed])   # hand / strait / band (R24)
    return mesh, st
