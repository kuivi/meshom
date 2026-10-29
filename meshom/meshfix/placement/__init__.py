"""c_placement: global placement of the defects from a continuous field, then re-extraction of the region
(competitor C of the mesh-fixing competition, decisions B130-B133).

    fix(case, opts) -> (Mesh, record)
    python -m meshom.meshfix.placement --case NAME [--opt key=value ...]     run, save, score

1. Region (fields.region, B130): the transition band and every defect, grown; the kept rest is fixed, its first rings
   (the collar) are the lattice the new rows are stitched to.
2. Fields (fields.py, B131): rows = integer level sets of v (least squares on n / h, the collar exact integers); the
   column stripes psi = a Ginzburg-Landau minimiser of the connection energy with the frequency 1 / h along the rows,
   psi = 1 on the collar. The zeros of psi are the column ends (dislocations): their number follows from the size
   field (the curl of e / h) and the coast (free boundary: zeros may leave), their layout from the energy (vortices
   repel: an even, staggered lattice, no lines).
3. Extraction (extract.py, B132): nodes where the phase crosses 2 pi m on each row; rows paired by a DP into quads and
   one triangle per column end; the rest of R (coast fringe, window sides) by a constrained Delaunay triangulation and
   recombination.
4. Relaxation (relax.py, B133): free-node geometric optimisation with a size wall against h.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh, build_topology
from meshom.meshfix.placement import extract as X
from meshom.meshfix.placement import fields as F

METHOD = "c_placement"
DEFAULTS = dict(kappa=1.0, gl_iters=5000, relax=True, recombine_price=0.3, coast_price=0.2, resplit=True, post_optimize=True, slide=True, match_collar=True,
                steiner_rounds=0, steiner_clear=0.4, guard=True,
                split_deg=145.0, relax_ring=2)


def _ccw_ok(points, cells, min_deg=5.0, max_deg=175.0):
    """(M,) cells CCW and convex with every corner in [min_deg, max_deg] (tangent plane)."""
    from meshom.methods.m07.extract import _corner_angles

    if len(cells) == 0:
        return np.zeros(0, bool)
    a = np.degrees(_corner_angles(points, cells))
    return (a.min(1) >= min_deg) & (a.max(1) <= max_deg) & (np.abs(a.sum(1) - 180.0 * (cells.shape[1] - 2)) < 1.0)


def _outline(mesh: Mesh, R: np.ndarray, el, nv):
    """Directed outline edges of the element set R (R on the left) and whether each is a mesh boundary edge."""
    a = np.concatenate([el[R, :3].ravel() if False else np.zeros(0, np.int64)])
    rows = np.flatnonzero(R)
    A, B = [], []
    for k in range(4):
        sel = rows[nv[rows] > k]
        nxt = np.where(nv[sel] == 3, (k + 1) % 3, (k + 1) % 4)
        A.append(el[sel, k])
        B.append(el[sel, nxt])
    A, B = np.concatenate(A), np.concatenate(B)
    n = mesh.n_nodes
    fwd = A.astype(np.int64) * n + B
    rev = B.astype(np.int64) * n + A
    out = ~np.isin(fwd, rev)                     # no element of R on the other side
    return np.column_stack([A[out], B[out]])


def fix(case, opts: dict | None = None):
    o = {**DEFAULTS, **F.DEFAULTS, **X.DEFAULTS, **(opts or {})}
    rec = {"method": METHOD, "case": case.name, "opts": {k: v for k, v in o.items()}}
    t00 = time.perf_counter()
    tm = {}
    mesh = case.mesh
    n0 = mesh.n_nodes
    topo = build_topology(mesh)
    bnd = topo.boundary_node

    t0 = time.perf_counter()
    reg = F.region(mesh, case.kind, case.h_node, o, topo)
    R, col, el, nv = reg["R"], reg["collar"], reg["el"], reg["nv"]
    rec["region"] = {k: v for k, v in reg.items() if isinstance(v, int)} | {"R": int(R.sum()), "collar": int(col.sum())}
    tm["region"] = time.perf_counter() - t0

    # local node set and triangles
    t0 = time.perf_counter()
    Rn = np.unique(el[R][el[R] >= 0])
    Cn = np.unique(el[col][el[col] >= 0])
    local = np.union1d(Rn, Cn)
    nl = len(local)
    loc = -np.ones(n0, np.int64)
    loc[local] = np.arange(nl)
    collar_node = np.zeros(nl, bool)
    collar_node[loc[Cn]] = True
    tr, owner = F.tri_split(mesh)
    use = R[owner] | col[owner]
    T = loc[tr[use]]
    T_owner = owner[use]
    T_inR = R[T_owner]
    E = np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    Ein = np.r_[T_inR, T_inR, T_inR]
    key = np.minimum(E[:, 0], E[:, 1]) * nl + np.maximum(E[:, 0], E[:, 1])
    order = np.lexsort((~Ein, key))
    ks = key[order]
    first = np.r_[True, ks[1:] != ks[:-1]]
    Eu = np.column_stack([ks[first] // nl, ks[first] % nl])
    EuR = Ein[order][first]
    # true mesh edges (no quad diagonals) for the row field
    TE = topo.edges
    okE = (loc[TE] >= 0).all(1)
    Em = loc[TE[okE]]
    ee = topo.edge_elems[okE]
    EmR = R[ee[:, 0]] | ((ee[:, 1] >= 0) & R[np.maximum(ee[:, 1], 0)])
    EmK = ~R[ee[:, 0]] | ((ee[:, 1] >= 0) & ~R[np.maximum(ee[:, 1], 0)])
    P = mesh.points[local]
    hn = case.h_node[local]
    find = X.edge_lookup(Eu, nl)
    tm["local"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ndir, _ = F.row_direction(P, Em, hn, None)
    v, rmax, r99 = F.row_field(P, Em, EmR, hn, ndir, collar_node, o, edge_kept=EmK)
    rec["v"] = {"min": float(v.min()), "max": float(v.max()), "collar_round_max": rmax, "collar_round_p99": r99}
    tm["rows"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    width = hn
    if o["match_collar"]:
        width = F.collar_width(P, Em, EmK, hn, ndir, collar_node)
    psi_lin, om = F.stripes(P, Eu, hn, ndir, collar_node, width=width)
    rng = np.random.default_rng(0)
    p0 = np.where(np.abs(psi_lin) > 1e-12, psi_lin / np.maximum(np.abs(psi_lin), 1e-300),
                  np.exp(2j * np.pi * rng.random(nl)))
    p0[collar_node] = 1.0
    L = F.connection_laplacian(Eu, om, nl)
    psi, glr = F.ginzburg_landau(L, collar_node, p0, kappa=o["kappa"], iters=o["gl_iters"])
    wind = F.winding(T, Eu, om, psi)
    rec["stripes"] = glr | {"zeros_plus": int((wind > 0).sum()), "zeros_minus": int((wind < 0).sum())}
    tm["stripes"] = time.perf_counter() - t0

    # ---------------------------------------------------------------- rows and their nodes
    t0 = time.perf_counter()
    edir = np.cross(ndir, P)
    outl = _outline(mesh, R, el, nv)                                 # global ids
    # outline edges toward the coast, the artificial boundary or the frozen coastal ring are soft (no lattice):
    # generated nodes keep clear of them and the fringe fills up to them
    frozen = reg["frozen"]
    ekey_all = topo.edges[:, 0] * n0 + topo.edges[:, 1]
    so_e = np.argsort(ekey_all)
    ok_ = np.minimum(outl[:, 0], outl[:, 1]) * n0 + np.maximum(outl[:, 0], outl[:, 1])
    eid = so_e[np.searchsorted(ekey_all[so_e], ok_)]
    e2 = topo.edge_elems[eid]
    other = np.where(R[e2[:, 0]], e2[:, 1], e2[:, 0])
    out_is_bnd = (other < 0) | frozen[np.maximum(other, 0)]
    rec["outline"] = {"edges": int(len(outl)), "soft": int(out_is_bnd.sum())}
    outline_nodes = np.unique(outl)
    snap_nodes = np.union1d(outline_nodes, Cn)                       # global ids a generated node may snap to
    snap_tree = cKDTree(mesh.points[snap_nodes])
    soft_nodes = np.zeros(n0, bool)
    soft_nodes[outl[out_is_bnd].ravel()] = True
    snap_tol = np.where(soft_nodes[snap_nodes], o["snap_soft"], o["snap"])
    seg = mesh.points[outl[out_is_bnd]]                              # coast / artificial outline segments
    from meshom.meshfix.cases import segment_distance_km

    new_pts = []
    rows = {}
    kmin, kmax = int(np.ceil(v.min())), int(np.floor(v.max()))
    n_drop = n_snap = 0
    for k in range(kmin, kmax + 1):
        pieces = []
        for es, t, closed in X.row_polylines(P, T, Eu, find, v, k - o["eps"]):
            Y, kk, _ = X.row_nodes(P, Eu, es, t, psi, om, hn, edir)
            if len(Y) == 0:
                continue
            hY = case.h(Y)
            d, j = snap_tree.query(Y)
            snapped = d * R_EARTH_M < snap_tol[j] * hY
            dc = segment_distance_km(Y, seg) * 1e3 if len(seg) else np.full(len(Y), np.inf)
            drop = ~snapped & (dc < o["clear"] * hY)
            n_drop += int(drop.sum())
            n_snap += int(snapped.sum())
            ids = np.empty(len(Y), np.int64)
            ids[snapped] = snap_nodes[j[snapped]]
            fresh = ~snapped & ~drop
            ids[fresh] = n0 + len(new_pts) + np.arange(fresh.sum())
            new_pts.extend(list(Y[fresh]))
            Yx = Y.copy()
            Yx[snapped] = mesh.points[ids[snapped]]
            ee = np.cross(np.cross(Yx, np.array([0, 0, 1.0])), Yx) if False else None
            # row direction at the points: interpolate edir by nearest local node
            _, jn = cKDTree(P).query(Yx)
            ev = edir[jn]
            # split at dropped nodes; remove consecutive duplicates (two points snapped to one node)
            cut = np.flatnonzero(drop)
            bounds = np.r_[-1, cut, len(Y)]
            for s0, s1 in zip(bounds[:-1] + 1, bounds[1:]):
                if s1 - s0 < 2:
                    continue
                sl = slice(s0, s1)
                pid = ids[sl]
                keep = np.r_[True, pid[1:] != pid[:-1]]
                pieces.append(dict(ids=pid[keep], X=Yx[sl][keep], h=hY[sl][keep], e=ev[sl][keep]))
        rows[k] = pieces
    rec["rows"] = {"levels": kmax - kmin + 1, "new_nodes": len(new_pts), "snapped": n_snap, "dropped": n_drop}
    tm["row_nodes"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    Q, Tt = [], []
    for k in range(kmin, kmax):
        qq, tt = X.strip_elements(rows.get(k, []), rows.get(k + 1, []), o)
        Q += qq
        Tt += tt
    allP = np.vstack([mesh.points, np.array(new_pts).reshape(-1, 3)])
    Q = np.array(Q, np.int64).reshape(-1, 4)
    Tt = np.array(Tt, np.int64).reshape(-1, 3)
    rec["strips"] = {"quads": len(Q), "tris": len(Tt)}

    # keep explicit elements inside R: centroid in an R element, valid shape, nodes new or on R's outline
    from meshom.meshfix.placement.relax import locate

    inR_node = np.zeros(len(allP), bool)
    inR_node[n0:] = True
    inR_node[outline_nodes] = True

    def inside(cells):
        """centroid and the points near each edge midpoint (pulled 20 % to the centroid) all in R: an element that
        straddles R's outline (rows snapped to outline nodes in a crossing order) is refused."""
        if len(cells) == 0:
            return np.zeros(0, bool)
        c = allP[cells].mean(1)
        probes = [c]
        k = cells.shape[1]
        for j in range(k):
            mpt = 0.5 * (allP[cells[:, j]] + allP[cells[:, (j + 1) % k]])
            probes.append(0.8 * mpt + 0.2 * c)
        ok = np.ones(len(cells), bool)
        for x in probes:
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            owner_el = locate(mesh, tr, owner, x)
            ok &= (owner_el >= 0) & R[np.maximum(owner_el, 0)]
        return ok

    okQ = inR_node[Q].all(1) & _ccw_ok(allP, Q) & inside(Q) if len(Q) else np.zeros(0, bool)
    okT = inR_node[Tt].all(1) & _ccw_ok(allP, Tt) & inside(Tt) if len(Tt) else np.zeros(0, bool)
    Q, Tt = Q[okQ], Tt[okT]
    # manifold: every directed edge at most once, else drop the later element (both if needed)
    Q, Tt = _manifold(Q, Tt, len(allP))
    rec["explicit"] = {"quads": len(Q), "tris": len(Tt)}
    tm["strips"] = time.perf_counter() - t0

    # ---------------------------------------------------------------- CDT of R with the explicit elements
    t0 = time.perf_counter()
    from meshom.delaunay import sphere_delaunay

    import warnings

    sub = set()
    for a_, b_, c_ in Tt:
        sub.add(tuple(sorted((a_, b_, c_))))
    for a_, b_, c_, d_ in Q:
        for t3 in ((a_, b_, c_), (a_, c_, d_), (a_, b_, d_), (b_, c_, d_)):
            sub.add(tuple(sorted(t3)))
    okey = np.minimum(outl[:, 0], outl[:, 1]) * 10**9 + np.maximum(outl[:, 0], outl[:, 1])
    set_ok = set(okey.tolist())
    n_steiner = 0
    for it in range(1 + o["steiner_rounds"]):
        pts_ids = np.unique(np.r_[outline_nodes, np.arange(n0, len(allP)), Q.ravel(), Tt.ravel()])
        gl = -np.ones(len(allP), np.int64)
        gl[pts_ids] = np.arange(len(pts_ids))
        cons = [gl[outl]]
        for cells in (Q, Tt):
            if len(cells):
                cons.append(np.column_stack([gl[cells].ravel(), gl[np.roll(cells, -1, 1)].ravel()]))
        cons = np.vstack(cons)
        ck = np.minimum(cons[:, 0], cons[:, 1]) * len(pts_ids) + np.maximum(cons[:, 0], cons[:, 1])
        _, ui = np.unique(ck, return_index=True)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            tri = sphere_delaunay(allP[pts_ids], edges=cons[np.sort(ui)], coast=gl[outl])
        tri = pts_ids[tri]
        if it == o["steiner_rounds"]:
            break
        # spikes: fringe triangles with two outline edges; a Steiner point mirrored across the third edge (0.8)
        nb_o = np.zeros(len(tri), np.int64)
        for j in range(3):
            a_, b_ = tri[:, j], tri[:, (j + 1) % 3]
            nb_o += np.isin(np.minimum(a_, b_) * 10**9 + np.maximum(a_, b_), okey)
        sp = tri[nb_o >= 2]
        if len(sp) == 0:
            break
        add = []
        tree_all = cKDTree(allP)
        for t3 in sp:
            for j in range(3):                       # apex = the node shared by the two outline edges
                u_, w_, v_ = t3[(j + 2) % 3], t3[j], t3[(j + 1) % 3]
                if (min(u_, w_) * 10**9 + max(u_, w_)) in set_ok and (min(w_, v_) * 10**9 + max(w_, v_)) in set_ok:
                    m_ = 0.5 * (allP[u_] + allP[v_])
                    p_ = m_ + 0.8 * (m_ - allP[w_])
                    p_ /= np.linalg.norm(p_)
                    hp = float(case.h(p_[None])[0])
                    d_, _ = tree_all.query(p_)
                    if d_ * R_EARTH_M > o["steiner_clear"] * hp:
                        add.append(p_)
                    break
        if not add:
            break
        add = np.array(add)
        oe = locate(mesh, tr, owner, add)
        add = add[(oe >= 0) & R[np.maximum(oe, 0)]]
        if len(add) == 0:
            break
        allP = np.vstack([allP, add])
        n_steiner += len(add)
    rec["cdt"] = {"tris": len(tri), "steiner": n_steiner, "warnings": [str(w.message) for w in wlist][:5]}
    # CDT triangles that lie in an explicit element
    st = np.sort(tri, 1)
    covered = np.array([tuple(r) in sub for r in st.tolist()], bool)
    fringe = tri[~covered]
    rec["fringe_tris"] = int(len(fringe))
    # holes: directed edges of the new region (explicit + fringe) without a twin that are not on the outline
    nn = len(allP)
    he = [np.column_stack([c.ravel(), np.roll(c, -1, 1).ravel()]) for c in (Q, Tt, fringe) if len(c)]
    he = np.vstack(he)
    hk = he[:, 0] * nn + he[:, 1]
    lone = he[~np.isin(he[:, 1] * nn + he[:, 0], hk)]
    lone = lone[~np.isin(lone[:, 0] * nn + lone[:, 1], outl[:, 0] * nn + outl[:, 1])]
    rec["holes_edges"] = int(len(lone))
    if len(lone):
        from meshom.geo import xyz_to_lonlat
        lo, la = xyz_to_lonlat(allP[lone[:, 0]])
        rec["holes_at"] = [[float(x), float(y), int(a), int(b), bool(a >= n0), bool(b >= n0)]
                           for x, y, (a, b) in zip(lo[:8], la[:8], lone[:8])]
        rec["cdt_tris_at_hole_nodes"] = int(np.isin(tri, lone).any(1).sum())
    if o.get("debug_dump"):
        np.savez(o["debug_dump"], allP=allP, Q=Q, Tt=Tt, tri=tri, outl=outl, fringe=fringe, lone=lone, n0=n0,
                 covered=covered)
    from meshom.recombine import recombine

    # a fringe triangle with two outline edges must merge (the boundary_edges gate): a high price
    okey = np.minimum(outl[:, 0], outl[:, 1]) * len(allP) + np.maximum(outl[:, 0], outl[:, 1])
    nb_out = np.zeros(len(fringe), np.int64)
    for j in range(3):
        a_, b_ = fringe[:, j], fringe[:, (j + 1) % 3]
        nb_out += np.isin(np.minimum(a_, b_) * len(allP) + np.maximum(a_, b_), okey)
    on_out = np.isin(fringe, outl).any(1)
    price = np.where(nb_out >= 2, 2.0, np.where(on_out, o["coast_price"], o["recombine_price"]))
    fm = recombine(Mesh(allP, fringe.astype(np.int64), np.zeros((0, 4), np.int64), np.zeros(len(allP), np.int8)),
                   price=price)
    tm["cdt"] = time.perf_counter() - t0

    # ---------------------------------------------------------------- assemble
    keepE = ~R
    kt = keepE[: mesh.n_tris]
    kq = keepE[mesh.n_tris:]
    tris_all = np.vstack([mesh.tris[kt], Tt, fm.tris]).astype(np.int64)
    quads_all = np.vstack([mesh.quads[kq], Q, fm.quads]).astype(np.int64)
    usedn = np.zeros(len(allP), bool)
    usedn[tris_all.ravel()] = True
    usedn[quads_all.ravel()] = True
    newid = -np.ones(len(allP), np.int64)
    newid[usedn] = np.arange(usedn.sum())
    out = Mesh(allP[usedn], newid[tris_all], newid[quads_all], np.zeros(int(usedn.sum()), np.int8))
    tp = build_topology(out)
    out.node_flag[:] = tp.boundary_node.astype(np.int8)
    # node kinds of the output: kept nodes keep theirs, new nodes are interior
    kind = np.zeros(len(allP), np.int8)
    kind[:n0] = case.kind
    okind = kind[usedn]
    if o["resplit"]:
        from meshom.methods.m07.extract import resplit_pentagons

        out, nres = resplit_pentagons(out)
        rec["resplit"] = int(nres)
    rec["assembled"] = {"nodes": out.n_nodes, "quads": out.n_quads, "tris": out.n_tris}
    tm["assemble"] = time.perf_counter() - t0

    def moving_mask(m_):
        """interior nodes of the output that lie in R (located in the input's R elements)"""
        loc_el = locate(mesh, tr, owner, m_.points)
        mv = (loc_el >= 0) & (R | frozen_near)[np.maximum(loc_el, 0)]
        return mv & ~build_topology(m_).boundary_node

    from meshom.meshfix.placement.relax import relax

    coast_tree = cKDTree(mesh.points[case.kind == 1]) if (case.kind == 1).any() else None

    def slide_mask(m_):
        if coast_tree is None or not o["slide"]:
            return None
        d, _ = coast_tree.query(m_.points)
        return (d * R_EARTH_M < 1.0) & build_topology(m_).boundary_node

    rnode = np.zeros(n0, bool)
    rnode[Rn] = True
    frozen_near = frozen & (np.where(el >= 0, rnode[np.maximum(el, 0)], False).any(1))
    if o["relax"]:
        t0 = time.perf_counter()
        out, rr = relax(out, case.h, moving_mask(out), o, slide=slide_mask(out))
        rec["relax"] = rr
        tm["relax"] = time.perf_counter() - t0
    if o["post_optimize"]:
        t0 = time.perf_counter()
        from meshom.optimize import optimize, record

        out2, info = optimize(out, case.h, case.tol, dict(steps=("flip", "merge", "cluster"), fit="off"),
                              fixed=~moving_mask(out))
        rec["optimize"] = record(info)
        out = out2
        from meshom.methods.m07.extract import resplit_pentagons

        out, nres2 = resplit_pentagons(out)
        rec["resplit_post"] = int(nres2)
        nsp = 0
        for _ in range(3):
            out, k_ = X.fix_spikes(out)
            nsp += k_
            if k_ == 0:
                break
        rec["spikes_fixed"] = int(nsp)
        out, nsq = X.split_bad_quads(out, o["split_deg"])
        rec["quads_split"] = int(nsq)
        out.node_flag[:] = build_topology(out).boundary_node.astype(np.int8)
        if o["relax"]:
            out, rr = relax(out, case.h, moving_mask(out), o, slide=slide_mask(out))
            rec["relax2"] = rr
        tm["post"] = time.perf_counter() - t0
    if o["guard"]:
        # do no harm (the control v5): keep the input when the result has more quads above 120 deg or a worse p99.9
        def qstat(m_):
            a_ = F.corner_max_angle(m_)
            return int((a_ > 120).sum()), float(np.percentile(a_, 99.9))
        q_in, q_out = qstat(mesh), qstat(out)
        rec["guard"] = {"input": q_in, "output": q_out}
        if q_out[0] > q_in[0] or q_out[1] > q_in[1] + 0.5:
            rec["guard"]["kept_input"] = True
            out = mesh
    rec["times"] = tm
    rec["wall_s"] = time.perf_counter() - t00
    return out, rec


def _manifold(Q, Tt, n):
    """Drop elements so that no directed edge is used twice (overlapping DP chains)."""
    cells = [(Q, 4), (Tt, 3)]
    keys, who = [], []
    for ci, (c, k) in enumerate(cells):
        if len(c) == 0:
            continue
        a, b = c.ravel(), np.roll(c, -1, 1).ravel()
        keys.append(a * n + b)
        who.append(np.c_[np.full(len(a), ci), np.repeat(np.arange(len(c)), k)])
    if not keys:
        return Q, Tt
    keys = np.concatenate(keys)
    who = np.vstack(who)
    o = np.argsort(keys, kind="stable")
    dup = np.r_[False, keys[o][1:] == keys[o][:-1]]
    bad = who[o][dup]
    keepQ = np.ones(len(Q), bool)
    keepT = np.ones(len(Tt), bool)
    keepQ[bad[bad[:, 0] == 0, 1]] = False
    keepT[bad[bad[:, 0] == 1, 1]] = False
    return Q[keepQ], Tt[keepT]


def run(case_name: str, opts: dict | None = None, score: bool = True, tag: str | None = None) -> dict:
    from meshom.compete import _save
    from meshom.meshfix.cases import load, meshfix_dir

    case = load(case_name)
    mesh, rec = fix(case, opts)
    d = meshfix_dir() / "results" / (METHOD if tag is None else f"{METHOD}_{tag}")
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{case_name}.npz"
    _save(mesh, f)
    f.with_suffix(".json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: rec[k] for k in rec if k not in ("opts",)}, indent=1, default=float), flush=True)
    if score:
        from meshom.meshfix.score import score_file

        r = score_file(case_name, f, case=case, label=METHOD)
        return r
    return rec


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="python -m meshom.meshfix.placement")
    ap.add_argument("--case", nargs="+", required=True)
    ap.add_argument("--opt", nargs="*", default=[])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-score", action="store_true")
    a = ap.parse_args(argv)
    opts = {}
    for kv in a.opt:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        opts[k] = v
    for c in a.case:
        run(c, opts, score=not a.no_score, tag=a.tag)


if __name__ == "__main__":
    main(sys.argv[1:])
