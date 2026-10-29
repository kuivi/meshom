"""m10: patch layout from separatrices + T-mesh quantisation + structured fill (QuadWild, Bi-MDF).

After Pietroni et al. 2021 (QuadWild: field tracing into a patch layout, quantisation, per-patch fill) with the
quantisation of Heistermann, Warnett & Bommes 2023 (Bi-MDF: integer arc counts, opposite patch sides equal, min
deviation from the target lengths) and the motorcycle graph of Eppstein et al. 2008 / Campen et al. 2015 (QGP) for
the T-mesh.

1. background_mesh(domain, h, 1.0) (its coast nodes are exactly resample_loops(domain, h)) and the "cones" cross
   field on it (parallel in g/h^2 except at cones, which follow rho_c); cones from singularities().
2. Separatrices from every cone (4 (1 - k) directions where the field is radial on a circle of radius h) and from
   every reflex coast corner (one per extra quarter turn), traced together as a motorcycle graph in the metric of h
   (meshom/methods/m10/trace.py): a trace stops at another trace (T-junction), head-on at another tip (joined), or
   at the coast (snapped to a coast node).
3. Faces of the trace + coast graph; four convex corners (cone sectors count one quarter turn each) = a patch.
4. Integer counts per arc by a small MILP (HiGHS): opposite patch sides equal, count ~ integral ds/h
   (meshom/methods/m10/layout.py).
5. Coons grid per patch with nodes at equal steps of ds/h on its sides; faces that are not patches (or keep a
   mismatch) get a constrained Delaunay fill from the bg nodes inside them, recombined (price 1/3).
6. meshom.smooth "size" sweeps, coast fixed.
"""
import sys
import time

import numpy as np

from meshom.delaunay import adjacency, background_mesh
from meshom.domain import Domain
from meshom.geo import R_EARTH_M, tangent_basis
from meshom.mesh import Mesh
from meshom.methods import register
from meshom.size import SizeField

DEFAULTS = dict(
    field="cones",          # cross_field method
    step_frac=0.25,         # trace step, units of h
    eps_join=0.75,          # head-on join distance, units of h
    cancel=1.0,             # +- cone pairs closer than this (units of h) cancel (a sub-cell dipole has no layout)
    straight=1.0,           # straight start from a cone, units of h (also the radius of the separatrix search)
    max_len=2.0 * np.pi,    # longest trace (radians)
    slack_cost=10.0,        # MILP penalty per unit mismatch of opposite patch sides (then: fallback fill)
    skew_rel=0.3,           # patches whose opposite sides differ by more (and by > 2 cells) go to the fallback
    milp_time=60.0,
    area_scale=True,        # scale arc targets by sqrt(patch area / (l_u l_v)) in units of h (curved patches)
    collapse=True,          # patch sides may quantise to 0 (thin strips collapse); each trace path >= 1
    coons_max_angle=145.0,  # a patch whose Coons grid has a larger quad angle (deg) goes to the fallback
    fallback="pack",        # fill of non-patch faces: "pack" (m02 L-inf packing from their boundary) | "bg" (bg nodes)
    price=1.0 / 3.0,        # recombination price of a fallback triangle
    size_iters=20,          # meshom.smooth "size" sweeps at the end (coast fixed)
    angle_iters=0,
    verbose=False,
)

LAST_INFO: dict = {}


def _opts(opts: dict | None) -> dict:
    o = dict(DEFAULTS)
    o.update(opts or {})
    return o


def _log(o: dict, msg: str) -> None:
    if o["verbose"]:
        print(f"[m10] {msg}", file=sys.stderr, flush=True)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _cancel_pairs(pos: np.ndarray, k: np.ndarray, hc: np.ndarray, dist: float) -> np.ndarray:
    """Keep mask: opposite cones closer than dist * h are removed in pairs, closest first."""
    from scipy.spatial import cKDTree

    keep = np.ones(len(pos), dtype=bool)
    if dist <= 0 or len(pos) < 2:
        return keep
    pairs = np.array(sorted(cKDTree(pos).query_pairs(dist * hc.max())), dtype=np.int64).reshape(-1, 2)
    d = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    for (i, j), dd in sorted(zip(pairs.tolist(), d), key=lambda x: x[1]):
        if keep[i] and keep[j] and k[i] * k[j] < 0 and dd < dist * 0.5 * (hc[i] + hc[j]):
            keep[i] = keep[j] = False
    return keep


def build_layout(domain: Domain, h: SizeField, o: dict, info: dict):
    from meshom.domain import resample_loops
    from meshom.field import cross_field, singularities
    from meshom.methods.m10 import layout as L
    from meshom.methods.m10.trace import field_at_points, motorcycle, near_dir, separatrix_angles

    t0 = time.perf_counter()
    bg = background_mesh(domain, h, factor=1.0)
    loops = resample_loops(domain, h)
    nb = sum(len(lp) for lp in loops)
    bnodes = np.flatnonzero(bg.node_flag == 1) if domain.loops else None
    t1 = time.perf_counter()
    cf = cross_field(bg, h, 4, boundary_nodes=bnodes, method=o["field"])
    ids, kk = singularities(cf)
    t2 = time.perf_counter()
    info.update(bg_nodes=bg.n_nodes, bg_s=t1 - t0, field_s=t2 - t1, n_cones=len(ids),
                n_cones_plus=int((kk > 0).sum()), n_cones_minus=int((kk < 0).sum()))

    Pb = np.ascontiguousarray(bg.points)
    Tb = np.ascontiguousarray(bg.tris, dtype=np.int64)
    nbr = adjacency(Tb)
    ub = np.ascontiguousarray(cf.u)
    hb = h(Pb) / R_EARTH_M
    lhb = np.log(hb)
    cen = _unit(Pb[Tb].sum(axis=1))
    rad = np.max(np.linalg.norm(Pb[Tb] - cen[:, None, :], axis=2), axis=1)
    vt = np.full(len(Pb), -1, dtype=np.int64)
    vt[Tb.ravel()] = np.repeat(np.arange(len(Tb), dtype=np.int64), 3)

    cpos = _unit(Pb[Tb[ids]].mean(axis=1))
    keep = _cancel_pairs(cpos, kk, h(cpos) / R_EARTH_M, float(o["cancel"]))
    info["n_cancelled"] = int((~keep).sum())
    lay = L.Layout()
    s_pos, s_dir, s_tri, s_org, s_str = [], [], [], [], []
    for t, k in zip(ids[keep], kk[keep]):
        c = _unit(Pb[Tb[t]].mean(axis=0))
        hc = float(np.exp(lhb[Tb[t]].mean()))
        e1, e2 = tangent_basis(c[None])
        th = separatrix_angles(Pb, Tb, nbr, ub, lhb, int(t), c, e1[0], e2[0], o["straight"] * hc, 360, float(k))
        node = lay.node(("cone", int(t)), c, L.KIND_CONE, nsep=len(th))
        for a in th:
            s_pos.append(c)
            s_dir.append(np.cos(a) * e1[0] + np.sin(a) * e2[0])
            s_tri.append(int(t))
            s_org.append(node)
            s_str.append(o["straight"] * hc)
    coast_loops, off = [], 0
    n_corner = 0
    w, d, e, out = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    for lp in loops:
        idx = np.arange(off, off + len(lp), dtype=np.int64)
        off += len(lp)
        coast_loops.append(idx)
        p = Pb[idx]
        nx, pv = np.roll(p, -1, axis=0), np.roll(p, 1, axis=0)
        t_out = _unit(nx - np.einsum("ij,ij->i", nx, p)[:, None] * p)
        t_back = _unit(pv - np.einsum("ij,ij->i", pv, p)[:, None] * p)
        A = np.arctan2(np.einsum("ij,ij->i", np.cross(t_out, t_back), p),
                       np.einsum("ij,ij->i", t_out, t_back)) % (2 * np.pi)
        for i in np.flatnonzero(np.abs(A - np.pi) > np.radians(45.0)):
            g = int(idx[i])
            node = lay.node(("coast", g), Pb[g], L.KIND_COAST, g)
            n_corner += 1
            m = int(round(A[i] / (0.5 * np.pi)))
            left = np.cross(p[i], t_out[i])
            for j in range(1, m):
                v = np.cos(0.5 * np.pi * j) * t_out[i] + np.sin(0.5 * np.pi * j) * left
                near_dir(Pb, Tb, ub, lhb, int(vt[g]), Pb[g].copy(), v, w, d, e, out)
                s_pos.append(Pb[g])
                s_dir.append(out.copy())
                s_tri.append(int(vt[g]))
                s_org.append(node)
                s_str.append(0.0)
    s_pos = np.ascontiguousarray(np.asarray(s_pos, dtype=np.float64).reshape(-1, 3))
    s_dir = np.ascontiguousarray(np.asarray(s_dir, dtype=np.float64).reshape(-1, 3))
    s_tri = np.asarray(s_tri, dtype=np.int64)
    s_org = np.asarray(s_org, dtype=np.int64)
    s_str = np.asarray(s_str, dtype=np.float64)
    org_pos = np.ascontiguousarray(np.asarray(lay.node_pos)[s_org].reshape(-1, 3))
    t3 = time.perf_counter()
    cap = max(100_000, 40 * len(s_pos))
    while True:
        tr = motorcycle(Pb, Tb, nbr, ub, lhb, cen, rad, s_pos, s_dir, s_tri, s_org, org_pos, s_str,
                        float(o["step_frac"]), float(o["eps_join"]), float(o["max_len"]), cap)
        if not tr[-1]:
            break
        cap *= 2
    t4 = time.perf_counter()
    et = tr[8]
    info.update(n_traces=len(s_pos), n_corners=n_corner, trace_points=int(tr[0]),
                n_hit=int((et == 1).sum()), n_join=int((et == 2).sum()), n_coast=int((et == 3).sum()),
                n_lost=int((et == 4).sum()), trace_s=t4 - t3)
    from scipy.spatial import cKDTree

    tree = cKDTree(Pb)

    def field_dir(X):
        X = np.ascontiguousarray(X)
        return field_at_points(Pb, Tb, nbr, ub, lhb, vt[tree.query(X)[1]], X)

    L.build(lay, tr, s_org, coast_loops, Pb, h, field_dir)
    info.update(lay.info)
    info["layout_s"] = time.perf_counter() - t4
    lay.bgd = dict(Pb=Pb, Tb=Tb, nbr=nbr, ub=ub, lhb=lhb, cen=cen, rad=rad, vt=vt)
    return bg, lay, nb, tr


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    from meshom.methods.m10 import layout as L
    from meshom.recombine import recombine
    from meshom.smooth import smooth

    o = _opts(opts)
    info: dict = {}
    bg, lay, nb, _ = build_layout(domain, h, o, info)
    t0 = time.perf_counter()
    scale = L.area_scale(lay, h) if o["area_scale"] else None
    L.demote_skewed(lay, scale, o["skew_rel"])
    counts = L.quantize(lay, o["slack_cost"], o["milp_time"], o["collapse"], scale)
    t1 = time.perf_counter()
    info.update(lay.info)
    P, quads, tris, finfo = L.fill(lay, counts, h, nb, bg.points, bg.node_flag,
                                   lay.bgd if o["fallback"] == "pack" else None, o["coons_max_angle"])
    finfo.pop("quad_face")
    info.update(finfo)
    info.update(quantize_s=t1 - t0, fill_s=time.perf_counter() - t1)
    # compact: drop unreferenced nodes (coast nodes stay first)
    used = np.zeros(len(P), dtype=bool)
    used[quads.ravel()] = True
    used[tris.ravel()] = True
    used[:nb] = True
    new = np.cumsum(used) - 1
    flag = np.zeros(int(used.sum()), dtype=np.int8)
    flag[:nb] = 1
    mesh = Mesh(P[used], new[tris], new[quads], flag)
    t2 = time.perf_counter()
    if mesh.n_tris:
        mesh = recombine(mesh, o["price"], "greedy")
    fixed = mesh.node_flag == 1
    if o["size_iters"]:
        mesh = smooth(mesh, h, fixed, iters=o["size_iters"], kind="size")
    if o["angle_iters"]:
        mesh = smooth(mesh, h, fixed, iters=o["angle_iters"], kind="angle")
    info.update(finish_s=time.perf_counter() - t2, quads=mesh.n_quads, tris=mesh.n_tris, nodes=mesh.n_nodes)
    LAST_INFO.clear()
    LAST_INFO.update(info)
    _log(o, " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in info.items()))
    return mesh
