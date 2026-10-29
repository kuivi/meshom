"""Patch layout from the traced separatrices, T-mesh quantisation, structured fill (m10).

Layout (a T-mesh): nodes = cones, coast nodes where traces start or end, T-junctions; arcs = trace pieces between
nodes and coast pieces between coast layout nodes. Faces come from the rotation system (outgoing arcs sorted by angle
at each node). A face corner is classified by the number of field quarter turns q of its interior angle (sector
angle 2 pi / n_sep at a cone with n_sep separatrices, 90 deg elsewhere): q = 1 convex corner, q = 2 flat (T-junction,
smooth coast), anything else bad. A water face with exactly four convex corners and nothing bad is a patch.

Quantisation (Bi-MDF problem of Heistermann et al. 2023, here as a small integer LP with scipy.optimize.milp/HiGHS):
integer x_a >= 0 per trace arc (coast arcs fixed by resample_loops), min sum w_a |x_a - l_a| with l_a = integral
ds/h (target count) and w_a = 1 / max(l_a, 1), subject to: opposite sides of every patch have equal sums (soft, slack
penalty `slack_cost`; a patch left with slack is demoted to the fallback), every patch side >= 1 (no collapsed patch,
as QuadWild; arcs may be 0, which merges two T-junctions from opposite sides of a trace into a crossing), arcs that
touch the coast or bound a fallback face >= 1.

Fill: a patch with side counts (nu, nv) gets a (nu+1) x (nv+1) grid; its boundary nodes are the arc nodes, placed at
equal steps of integral ds/h along each arc; interior nodes by transfinite (Coons) interpolation in 3D, projected to
the sphere. A fallback face is filled by its boundary nodes + the bg nodes inside it (>= 0.6 h from the boundary),
constrained Delaunay, recombination later.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from meshom.delaunay import recover_edges, remove_land, sphere_delaunay
from meshom.domain import Domain, in_water
from meshom.geo import R_EARTH_M, tangent_basis
from meshom.methods.m10.trace import END_COAST, END_HIT, END_JOIN, END_LOST

KIND_CONE, KIND_COAST, KIND_T = 0, 1, 2


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@dataclass
class Layout:
    node_pos: list = field(default_factory=list)       # (3,) per node
    node_kind: list = field(default_factory=list)
    node_bg: list = field(default_factory=list)        # bg (= mesh) coast node id, -1 otherwise
    node_nsep: list = field(default_factory=list)      # separatrices of a cone; 4 elsewhere
    keys: dict = field(default_factory=dict)
    arc_xyz: list = field(default_factory=list)        # polyline (K,3) per arc, start -> end
    arc_ends: list = field(default_factory=list)       # (n0, n1)
    arc_coast: list = field(default_factory=list)      # coast bg ids (start..end) or None for a trace arc
    arc_poly: list = field(default_factory=list)       # trace path (polyline) of a trace arc, -1 for a coast arc
    arc_len: np.ndarray | None = None                  # target count (integral ds/h); coast arcs: segment count
    faces: list = field(default_factory=list)          # half-edge cycles (he = 2 arc + (0 forward | 1 backward))
    face_q: list = field(default_factory=list)         # quarter turns at the end node of each half-edge
    face_land: list = field(default_factory=list)
    face_patch: list = field(default_factory=list)     # True: 4 convex corners, nothing bad
    face_alpha: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def node(self, key, pos, kind, bg=-1, nsep=4) -> int:
        i = self.keys.get(key)
        if i is None:
            i = len(self.node_pos)
            self.keys[key] = i
            self.node_pos.append(np.asarray(pos, dtype=np.float64))
            self.node_kind.append(kind)
            self.node_bg.append(bg)
            self.node_nsep.append(nsep)
        return i


def _follow(j: int, nextp: np.ndarray) -> list:
    out = []
    while j >= 0:
        out.append(j)
        j = int(nextp[j])
    return out


def _bend_to(TP: np.ndarray, L: np.ndarray, h) -> None:
    """Shift the points of the polyline L (its last point = the snapped coast node) within 2 h of its second-to-last
    point by the lateral offset of the snap, weighted linearly in arc length (in place)."""
    if len(L) < 3:
        return
    xyz = TP[L]
    end, p = xyz[-1], xyz[-2]
    hh = h(p[None])[0] / R_EARTH_M
    k0 = max(0, len(L) - 2 - 8)
    d = _unit(p - xyz[k0]) if k0 < len(L) - 2 else _unit(end - p)
    off = end - p
    off = off - np.dot(off, d) * d                                  # lateral part of the snap
    seg = np.linalg.norm(np.diff(xyz[:-1], axis=0), axis=1)
    back = np.r_[np.cumsum(seg[::-1])[::-1], 0.0]                   # arc length from each point to p
    wgt = np.clip(1.0 - back / (2.0 * hh), 0.0, 1.0)
    TP[L[:-1]] = _unit(xyz[:-1] + wgt[:, None] * off)


def build(lay: Layout, tr: tuple, s_org: np.ndarray, coast_loops: list, Pb: np.ndarray, h, field_dir) -> Layout:
    """Arcs and faces of the layout from the traces `tr` (trace.motorcycle output) and the coast loops (lists of bg
    node ids in loop order, water on the left). field_dir(X) -> (M,3): one cross-field direction at points X.

    A trace that ends at the coast was snapped to a coast node: its last 2 h are bent onto the node (the lateral
    offset spread linearly), so it meets the coast along the field."""
    n, TP, prev, nextp, ptrc, jlink, first, tip, etype, ea, eb, eu, _ = tr
    S = len(first)
    TP = TP[:n]
    polys = []
    poly_of = np.full(n, -1, dtype=np.int64)
    kpos = np.full(n, -1, dtype=np.int64)
    done = np.zeros(S, dtype=bool)
    for a in range(S):
        if done[a]:
            continue
        L = _follow(int(first[a]), nextp)
        done[a] = True
        n0 = int(s_org[a])
        if etype[a] == END_JOIN:
            b = int(ea[a])
            L = L + _follow(int(first[b]), nextp)[::-1]
            done[b] = True
            n1 = int(s_org[b])
        elif etype[a] == END_HIT:
            n1 = lay.node(("T", a), TP[tip[a]], KIND_T)
        elif etype[a] == END_COAST:
            bn = int(ea[a])
            n1 = lay.node(("coast", bn), Pb[bn], KIND_COAST, bn)
            _bend_to(TP, np.asarray(L, dtype=np.int64), h)
        else:
            n1 = lay.node(("end", a), TP[tip[a]], KIND_T)
        pid = len(polys)
        polys.append([np.asarray(L, dtype=np.int64), n0, n1, []])
        poly_of[L] = pid
        kpos[L] = np.arange(len(L))
    # T-junctions: split positions on the hit polylines
    for a in range(S):
        if etype[a] != END_HIT:
            continue
        c, d, u = int(ea[a]), int(eb[a]), float(eu[a])
        pid = poly_of[c]
        kc, kd = kpos[c], kpos[d]
        s = kc + u if kd == kc + 1 else kd + (1.0 - u)
        polys[pid][3].append((s, lay.keys[("T", a)]))
    # trace arcs
    for pid, (L, n0, n1, sp) in enumerate(polys):
        sp = [(0.0, n0)] + sorted(sp) + [(float(len(L) - 1), n1)]
        for (s0, a0), (s1, a1) in zip(sp[:-1], sp[1:]):
            k = np.arange(int(np.floor(s0)) + 1, int(np.ceil(s1)))
            k = k[(k > s0 + 1e-9) & (k < s1 - 1e-9)]
            xyz = np.vstack([lay.node_pos[a0][None], TP[L[k]], lay.node_pos[a1][None]])
            lay.arc_xyz.append(xyz)
            lay.arc_ends.append((a0, a1))
            lay.arc_coast.append(None)
            lay.arc_poly.append(pid)
    # coast arcs
    for ids in coast_loops:
        on = [i for i, b in enumerate(ids) if ("coast", int(b)) in lay.keys]
        if not on:
            lay.node(("coast", int(ids[0])), Pb[ids[0]], KIND_COAST, int(ids[0]))
            on = [0]
        m = len(ids)
        for i0, i1 in zip(on, on[1:] + [on[0] + m]):
            seg = np.asarray([ids[i % m] for i in range(i0, i1 + 1)], dtype=np.int64)
            lay.arc_xyz.append(Pb[seg])
            lay.arc_ends.append((lay.keys[("coast", int(seg[0]))], lay.keys[("coast", int(seg[-1]))]))
            lay.arc_coast.append(seg)
            lay.arc_poly.append(-1)
    # target counts
    ln = np.zeros(len(lay.arc_xyz))
    for i, xyz in enumerate(lay.arc_xyz):
        if lay.arc_coast[i] is not None:
            ln[i] = len(lay.arc_coast[i]) - 1
        elif len(xyz) > 1:
            mid = _unit(xyz[1:] + xyz[:-1])
            ln[i] = np.sum(np.linalg.norm(xyz[1:] - xyz[:-1], axis=1) * R_EARTH_M / h(mid))
    lay.arc_len = ln
    _faces(lay, h, field_dir)
    return lay


def _faces(lay: Layout, h, field_dir) -> None:
    """Rotation system (outgoing arcs sorted by the angle of their direction 0.5 h out), faces, and the quarter turns
    q at each face corner: at a cone the sector count (geometric angle / (2 pi / n_sep)); elsewhere from the field:
    each arc direction rounded to one of the 4 field directions k at the node, q = (k_in_twin - k_next) mod 4 (robust
    to oblique T-hits whose geometric angles of 45/135 deg would round wrongly)."""
    na = len(lay.arc_xyz)
    pos = np.asarray(lay.node_pos)
    hn = h(pos) / R_EARTH_M
    he_node = np.zeros(2 * na, dtype=np.int64)      # node the half-edge leaves
    he_ang = np.zeros(2 * na)
    e1, e2 = tangent_basis(pos)
    fd = field_dir(pos)
    f_ang = np.arctan2(np.einsum("ij,ij->i", fd, e2), np.einsum("ij,ij->i", fd, e1))
    for a, xyz in enumerate(lay.arc_xyz):
        n0, n1 = lay.arc_ends[a]
        for s, node, pts in ((0, n0, xyz), (1, n1, xyz[::-1])):
            p = pts[0]
            dist = np.linalg.norm(pts - p, axis=1)
            far = np.flatnonzero(dist >= 0.5 * hn[node])
            q = pts[far[0]] if len(far) else pts[-1]
            d = q - np.dot(q, p) * p
            he_node[2 * a + s] = node
            he_ang[2 * a + s] = np.arctan2(d @ e2[node], d @ e1[node])
    out = [[] for _ in range(len(pos))]
    for he in np.lexsort((he_ang, he_node)):
        out[he_node[he]].append(int(he))
    idx = np.zeros(2 * na, dtype=np.int64)
    for lst in out:
        for i, he in enumerate(lst):
            idx[he] = i
    nxt = np.zeros(2 * na, dtype=np.int64)
    alpha = np.zeros(2 * na)
    qfield = np.full(2 * na, -1, dtype=np.int64)
    for he in range(2 * na):
        tw = he ^ 1
        v = he_node[tw]
        lst = out[v]
        nx = lst[(idx[tw] - 1) % len(lst)]
        nxt[he] = nx
        alpha[he] = 2 * np.pi if nx == tw else (he_ang[tw] - he_ang[nx]) % (2 * np.pi)
        if lay.node_kind[v] != KIND_CONE:
            k_tw = np.rint((he_ang[tw] - f_ang[v]) / (0.5 * np.pi))
            k_nx = np.rint((he_ang[nx] - f_ang[v]) / (0.5 * np.pi))
            qf = (k_tw - k_nx) % 4 if nx != tw else 4
            qfield[he] = qf
    seen = np.zeros(2 * na, dtype=bool)
    nsep = np.asarray(lay.node_nsep)
    for he0 in range(2 * na):
        if seen[he0]:
            continue
        cyc = []
        he = he0
        while not seen[he]:
            seen[he] = True
            cyc.append(he)
            he = int(nxt[he])
        cyc = np.asarray(cyc, dtype=np.int64)
        land = any(lay.arc_coast[he // 2] is not None and he % 2 == 1 for he in cyc)
        v = he_node[cyc ^ 1]                                   # end node of each half-edge
        q = np.where(qfield[cyc] >= 0, qfield[cyc], np.rint(alpha[cyc] / (2 * np.pi / nsep[v]))).astype(np.int64)
        lay.faces.append(cyc)
        lay.face_q.append(q)
        lay.face_alpha.append(alpha[cyc] * nsep[v] / 4.0)        # corner angle in the cone metric
        lay.face_land.append(land)
        lay.face_patch.append((not land) and bool(np.all((q == 1) | (q == 2))) and int((q == 1).sum()) == 4)
    nf = len(lay.faces)
    lay.info.update(n_nodes_layout=len(pos), n_arcs=na, n_faces=nf,
                    n_water_faces=int(nf - sum(lay.face_land)), n_patches=int(sum(lay.face_patch)),
                    euler=len(pos) - na + nf)


def face_area_hat(lay: Layout, f: int, h) -> float:
    """Integral of dA / h^2 over face f: fan from the normalised mean of its boundary polyline, each fan triangle
    integrated with 4 points along the radius (weights ~ r)."""
    pts = []
    for he in lay.faces[f]:
        x = lay.arc_xyz[he // 2]
        pts.append((x if he % 2 == 0 else x[::-1])[:-1])
    poly = np.vstack(pts)
    m = _unit(poly.mean(axis=0))
    a, b = poly, np.roll(poly, -1, axis=0)
    det = np.einsum("ij,ij->i", np.cross(a, b), np.broadcast_to(m, a.shape))
    exc = 2.0 * np.arctan2(det, 1.0 + np.einsum("ij,ij->i", a, b) + a @ m + b @ m) * R_EARTH_M ** 2
    mid = _unit(a + b)
    r = (np.arange(4) + 0.5) / 4.0
    w = r / r.sum()
    acc = np.zeros(len(a))
    for rk, wk in zip(r, w):
        acc += wk / h(_unit(m + rk * (mid - m))) ** 2
    return float(np.sum(exc * acc))


def _sides(cyc: np.ndarray, q: np.ndarray) -> list:
    """Four lists of half-edges between the convex corners (corner = end node of a half-edge with q == 1)."""
    c = np.flatnonzero(q == 1)
    rot = np.r_[cyc[c[-1] + 1:], cyc[:c[-1] + 1]]
    qr = np.r_[q[c[-1] + 1:], q[:c[-1] + 1]]
    sides, cur = [], []
    for he, qq in zip(rot, qr):
        cur.append(int(he))
        if qq == 1:
            sides.append(cur)
            cur = []
    return sides


def area_scale(lay: Layout, h, lmin: float = 3.0, clip: tuple = (0.8, 1.4)) -> np.ndarray:
    """(A,) target scale per arc: sigma_f = sqrt(area_hat / (l_u l_v)) of each patch (l_u, l_v = mean lengths of its
    opposite sides; patches with l_u or l_v < lmin get 1), geometric mean over the patches on both sides. On a
    curved patch (sphere curvature not absorbed by the cones, or a residual of the trivial connection) the side lengths
    undercount the cells its area needs (a cube face: area 1.39 l^2)."""
    na = len(lay.arc_xyz)
    acc = np.zeros(na)
    cnt = np.zeros(na)
    sig = []
    for f, cyc in enumerate(lay.faces):
        if not lay.face_patch[f]:
            continue
        sd = _sides(cyc, lay.face_q[f])
        ls = [float(sum(lay.arc_len[he // 2] for he in side)) for side in sd]
        lu, lv = 0.5 * (ls[0] + ls[2]), 0.5 * (ls[1] + ls[3])
        if lu < lmin or lv < lmin:
            continue
        sg = float(np.clip(np.sqrt(face_area_hat(lay, f, h) / (lu * lv)), *clip))
        sig.append(sg)
        a = cyc // 2
        np.add.at(acc, a, np.log(sg))
        np.add.at(cnt, a, 1.0)
    lay.info.update(sigma_p50=float(np.median(sig)) if sig else 1.0, sigma_max=float(max(sig)) if sig else 1.0)
    return np.exp(np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0))


def demote_skewed(lay: Layout, scale: np.ndarray | None, rel: float = 0.3, absolute: float = 2.0,
                  corner: tuple = (55.0, 125.0)) -> int:
    """Patches whose opposite sides differ by more than rel (relative) and absolute (cells), or with a convex corner
    angle (cone metric, 0.5 h out) outside `corner` degrees, go to the fallback."""
    ln = lay.arc_len * (1.0 if scale is None else scale)
    n = 0
    lo, hi = np.radians(corner[0]), np.radians(corner[1])
    for f, cyc in enumerate(lay.faces):
        if not lay.face_patch[f]:
            continue
        a = lay.face_alpha[f][lay.face_q[f] == 1]
        if np.any((a < lo) | (a > hi)):
            lay.face_patch[f] = False
            n += 1
            continue
        sd = _sides(cyc, lay.face_q[f])
        ls = [float(sum(ln[he // 2] for he in side)) for side in sd]
        for k in range(2):
            a, b = ls[k], ls[k + 2]
            if abs(a - b) > absolute and abs(a - b) > rel * max(a, b):
                lay.face_patch[f] = False
                n += 1
                break
    lay.info["n_skewed"] = n
    return n


def quantize(lay: Layout, slack_cost: float = 50.0, time_limit: float = 60.0, collapse: bool = True,
             scale: np.ndarray | None = None) -> np.ndarray:
    """(A,) integer counts per arc (see the module docstring). Patches left with slack are demoted (face_patch).

    collapse: patch sides may be 0 (a thin strip collapses; QGP, Bi-MDF), with every trace path >= 1 (the non-zero
    path constraint per motorcycle trace of Heistermann 2023 §4.1.2); otherwise every patch side >= 1 (QuadWild)."""
    na = len(lay.arc_xyz)
    coast = np.array([c is not None for c in lay.arc_coast])
    ln = lay.arc_len * np.where(coast | (scale is None), 1.0, 1.0 if scale is None else scale)
    kind = np.asarray(lay.node_kind)
    ends = np.asarray(lay.arc_ends, dtype=np.int64)
    poly = np.asarray(lay.arc_poly, dtype=np.int64)
    var = np.full(na, -1, dtype=np.int64)
    free = np.flatnonzero(~coast)
    var[free] = np.arange(len(free))
    nx = len(free)
    lb_base = np.zeros(nx)
    touch = (kind[ends[free, 0]] == KIND_COAST) | (kind[ends[free, 1]] == KIND_COAST)
    lb_base[touch] = 1.0
    x = None
    for _round in range(4):
        patches = [f for f in range(len(lay.faces)) if lay.face_patch[f]]
        lb = lb_base.copy()
        for f in range(len(lay.faces)):
            if not lay.face_land[f] and not lay.face_patch[f]:
                a = lay.faces[f] // 2
                a = a[~coast[a]]
                lb[var[a]] = 1.0
        rows, cols, vals, lo, hi = [], [], [], [], []
        r = 0

        def add(row_terms, lo_v, hi_v):
            nonlocal r
            const = 0.0
            for a, s in row_terms:
                if coast[a]:
                    const += s * ln[a]
                else:
                    rows.append(r)
                    cols.append(var[a])
                    vals.append(s)
            lo.append(lo_v - const)
            hi.append(hi_v - const)
            r += 1
            return r - 1

        sides_of = {}
        eq_rows = []
        for f in patches:
            sd = _sides(lay.faces[f], lay.face_q[f])
            sides_of[f] = sd
            for k in range(2):
                terms = [(he // 2, 1.0) for he in sd[k]] + [(he // 2, -1.0) for he in sd[k + 2]]
                eq_rows.append((f, add(terms, 0.0, 0.0)))
            if not collapse:
                for k in range(4):
                    add([(he // 2, 1.0) for he in sd[k]], 1.0, np.inf)
        if collapse:
            for pid in np.unique(poly[poly >= 0]):
                add([(a, 1.0) for a in np.flatnonzero(poly == pid)], 1.0, np.inf)
        n_eq = len(eq_rows)
        nvar = 2 * nx + 2 * n_eq                              # x, d, slack+, slack-
        for i, (f, row) in enumerate(eq_rows):
            rows += [row, row]
            cols += [2 * nx + 2 * i, 2 * nx + 2 * i + 1]
            vals += [1.0, -1.0]
        # |x - l| <= d:  d - x >= -l,  d + x >= l;  and e >= |x - l| - band (convex extra cost beyond the band)
        lf = ln[free]
        band = 0.25 * lf + 1.0
        e0 = 2 * nx + 2 * n_eq
        nvar += nx
        for sgn in (-1.0, 1.0):
            rows += list(range(r, r + nx)) * 2
            cols += list(range(nx)) + list(range(nx, 2 * nx))
            vals += [sgn] * nx + [1.0] * nx
            lo += list(sgn * lf)
            hi += [np.inf] * nx
            r += nx
            rows += list(range(r, r + nx)) * 2
            cols += list(range(nx)) + list(range(e0, e0 + nx))
            vals += [sgn] * nx + [1.0] * nx
            lo += list(sgn * lf - band)
            hi += [np.inf] * nx
            r += nx
        A = coo_matrix((vals, (rows, cols)), shape=(r, nvar)).tocsr()
        w = 1.0 / np.maximum(lf, 1.0)
        c = np.r_[np.zeros(nx), w, np.full(2 * n_eq, slack_cost), 20.0 * w]
        integ = np.r_[np.ones(nx), np.zeros(nx + 2 * n_eq + nx)]
        res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ,
                   bounds=Bounds(np.r_[lb, np.zeros(nx + 2 * n_eq + nx)], np.full(nvar, np.inf)),
                   options={"time_limit": time_limit, "mip_rel_gap": 1e-3})
        if res.x is None:
            raise RuntimeError(f"m10 quantisation failed: {res.message}")
        sol = res.x
        x = np.rint(sol[:nx]).astype(np.int64)
        slack = sol[2 * nx:2 * nx + 2 * n_eq].reshape(-1, 2).sum(axis=1)
        bad = {f for (f, _), s in zip(eq_rows, slack) if s > 0.5}
        lay.info.update(milp_gap=float(getattr(res, "mip_gap", 0.0)), milp_status=int(res.status),
                        n_slack_patches=len(bad))
        if not bad:
            break
        for f in bad:
            lay.face_patch[f] = False
    counts = np.rint(ln).astype(np.int64)
    counts[free] = x
    dev = np.abs(counts[free] - ln[free])
    lay.info.update(n_patches_final=int(sum(lay.face_patch)), arc_dev_mean=float(dev.mean()) if nx else 0.0,
                    arc_dev_max=float(dev.max()) if nx else 0.0, n_zero_arcs=int((counts[free] == 0).sum()))
    return counts


class _UF:
    """Union-find over mesh node ids; the root is the member with the highest rank (coast > layout node > other)."""

    def __init__(self, rank: np.ndarray):
        self.p = np.arange(len(rank))
        self.rank = rank

    def find(self, i: int) -> int:
        p = self.p
        r = i
        while p[r] != r:
            r = p[r]
        while p[i] != r:
            p[i], i = r, p[i]
        return int(r)

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[rb] > self.rank[ra]:
            ra, rb = rb, ra
        self.p[rb] = ra
        return True


def _place(xyz: np.ndarray, h, k: int) -> np.ndarray:
    """k - 1 interior points at equal steps of integral ds/h along the polyline xyz."""
    if k <= 1:
        return np.zeros((0, 3))
    mid = _unit(xyz[1:] + xyz[:-1])
    w = np.linalg.norm(xyz[1:] - xyz[:-1], axis=1) / h(mid)
    cum = np.r_[0.0, np.cumsum(w)]
    u = cum[-1] * np.arange(1, k) / k
    j = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(w) - 1)
    f = np.clip((u - cum[j]) / np.where(w[j] > 0, w[j], 1.0), 0.0, 1.0)[:, None]
    return _unit((1.0 - f) * xyz[j] + f * xyz[j + 1])


def _dedupe(seq: list) -> list:
    return [s for i, s in enumerate(seq) if i == 0 or s != seq[i - 1]]


def _max_angle(X: np.ndarray) -> float:
    """Largest interior angle (radians, in [0, 2 pi); > pi = concave or inverted) of the quads X (Q,4,3), in the
    tangent plane at each quad's centre (CCW = positive)."""
    c = _unit(X.mean(axis=1))
    e1, e2 = tangent_basis(c)
    x = np.einsum("qkj,qj->qk", X, e1)
    y = np.einsum("qkj,qj->qk", X, e2)
    ax, ay = np.roll(x, 1, axis=1) - x, np.roll(y, 1, axis=1) - y
    bx, by = np.roll(x, -1, axis=1) - x, np.roll(y, -1, axis=1) - y
    a = np.arctan2(-(ax * by - ay * bx), ax * bx + ay * by) % (2 * np.pi)
    return float(a.max()) if a.size else 0.0


def _pack_points(P: np.ndarray, fixed: np.ndarray, seed: np.ndarray, h, bgd: dict, alpha: float = 0.75) -> np.ndarray:
    """New points of an L-inf frontal packing along the cross field (m02 kernel, Remacle 2013 / Baudouin 2014):
    the points P[fixed] only block, the queue starts at P[seed] (the fallback faces' boundary nodes), so the fronts
    grow only into the space the patches left free. Returns the new points."""
    from meshom.methods.m02.packing import _insert, _run_queue, walk

    Pb, Tb, nbr, ub, lhb, cen, rad, vt = (bgd[k] for k in ("Pb", "Tb", "nbr", "ub", "lhb", "cen", "rad", "vt"))
    X = np.vstack([P[fixed], P[seed]])
    near = cKDTree(Pb).query(X)[1]
    tri = np.array([walk(Pb, Tb, nbr, vt[j], x) for j, x in zip(near, X)], dtype=np.int64)
    tri = np.where(tri >= 0, tri, np.where(tri <= -2, -2 - tri, vt[near]))
    hx = h(X) / R_EARTH_M
    e = np.vstack([Tb[:, [0, 1]], Tb[:, [1, 2]], Tb[:, [2, 0]]])
    hb = np.exp(lhb)
    g = float(np.max(np.abs(hb[e[:, 0]] - hb[e[:, 1]]) / np.linalg.norm(Pb[e[:, 0]] - Pb[e[:, 1]], axis=1)))
    c = np.sqrt(2.0)
    rfac = c * alpha / max(1.0 - 0.5 * c * alpha * g, 0.25)
    n0 = len(X)
    cap = 2 * n0 + 1000
    while True:
        PP = np.zeros((cap, 3))
        ph = np.zeros(cap)
        pt = np.zeros(cap, dtype=np.int64)
        PP[:n0], ph[:n0], pt[:n0] = X, hx, tri
        head = np.full(len(Tb), -1, dtype=np.int64)
        nxt = np.full(cap, -1, dtype=np.int64)
        for i in range(n0):
            nxt[i] = head[pt[i]]
            head[pt[i]] = i
        stamp = np.zeros(len(Tb), dtype=np.int64)
        stack = np.empty(len(Tb), dtype=np.int64)
        n, _, full = _run_queue(Pb, Tb, nbr, ub, lhb, cen, rad, PP, ph, pt, head, nxt, stamp, 0, stack,
                                len(fixed), n0, alpha, rfac, True, 1.414, 0.6, 4)
        if not full:
            break
        cap *= 2
    return PP[n0:n].copy()


def fill(lay: Layout, counts: np.ndarray, h, nb: int, Pb: np.ndarray, bg_flag: np.ndarray, bgd: dict | None = None,
         max_angle: float = 160.0):
    """(points (N,3), quads (Q,4), tris (T,3), info): coast nodes 0..nb-1 = bg nodes 0..nb-1.

    Arc nodes get ids first; arcs of count 0 merge their end nodes and collapsed patches (a side sum 0) merge their
    two other sides node by node (union-find; merged positions = mean, coast nodes stay). Then the Coons grid of each
    patch (a patch whose grid has a quad angle > max_angle, e.g. a long thin strip whose sides were sized
    independently, goes to the fallback) and the fill of the other water faces."""
    nl = len(lay.node_pos)
    pts = [Pb[:nb]]
    rank = [np.full(nb, 3)]
    n_pts = nb
    node_id = np.full(nl, -1, dtype=np.int64)
    for i in range(nl):
        if lay.node_bg[i] >= 0:
            node_id[i] = lay.node_bg[i]
        else:
            node_id[i] = n_pts
            pts.append(lay.node_pos[i][None])
            rank.append(np.array([2]))
            n_pts += 1
    arc_nodes = []
    for a, xyz in enumerate(lay.arc_xyz):
        n0, n1 = lay.arc_ends[a]
        if lay.arc_coast[a] is not None:
            arc_nodes.append(np.asarray(lay.arc_coast[a], dtype=np.int64))
            continue
        inner = _place(xyz, h, int(counts[a]))
        pts.append(inner)
        rank.append(np.ones(len(inner), dtype=np.int64))
        arc_nodes.append(np.r_[node_id[n0], n_pts + np.arange(len(inner)), node_id[n1]].astype(np.int64))
        n_pts += len(inner)
    P0 = np.vstack(pts)
    uf = _UF(np.concatenate(rank))
    for a in range(len(arc_nodes)):
        if counts[a] == 0:
            uf.union(int(arc_nodes[a][0]), int(arc_nodes[a][-1]))

    def chain(hes):
        seq = []
        for he in hes:
            lst = arc_nodes[he // 2] if he % 2 == 0 else arc_nodes[he // 2][::-1]
            seq.extend(uf.find(int(i)) for i in lst)
        return _dedupe(seq)

    water = [f for f in range(len(lay.faces)) if not lay.face_land[f]]
    patch_sides = {f: _sides(lay.faces[f], lay.face_q[f]) for f in water if lay.face_patch[f]}
    n_collapsed = 0
    for _ in range(20):
        changed = False
        for f, sd in patch_sides.items():
            nu = int(sum(counts[he // 2] for he in sd[0]))
            nv = int(sum(counts[he // 2] for he in sd[1]))
            if nu > 0 and nv > 0:
                continue
            ch = [chain(s) for s in sd]
            for k in range(2):
                if (nu if k == 1 else nv) == 0 or True:
                    a_, b_ = ch[k], ch[k + 2][::-1]
                    if (k == 0 and nv == 0) or (k == 1 and nu == 0):
                        if len(a_) == len(b_):
                            for i, j in zip(a_, b_):
                                changed |= uf.union(i, j)
        if not changed:
            break
    roots = np.array([uf.find(i) for i in range(n_pts)], dtype=np.int64)
    acc = np.zeros((n_pts, 3))
    np.add.at(acc, roots, P0)
    P = np.where((np.arange(n_pts) < nb)[:, None], P0, _unit(acc + (acc == 0) * P0))   # coast nodes bit-exact
    quads, tris = [], []
    quad_face = []
    new_pts = []
    n_fallback, n_fail, n_mismatch, n_degen, n_sheared = 0, 0, 0, 0, 0
    fallback = [f for f in water if not lay.face_patch[f]]
    for f, sd in patch_sides.items():
        sd = [chain(s) for s in sd]
        nu, nv = len(sd[0]) - 1, len(sd[1]) - 1
        if nu == 0 or nv == 0:
            n_collapsed += 1
            continue
        if (len(sd[2]) - 1 != nu or len(sd[3]) - 1 != nv or sd[0][-1] != sd[1][0] or sd[1][-1] != sd[2][0]
                or sd[2][-1] != sd[3][0] or sd[3][-1] != sd[0][0]):
            n_mismatch += 1
            fallback.append(f)
            continue
        G = np.full((nu + 1, nv + 1), -1, dtype=np.int64)
        G[:, 0] = sd[0]
        G[nu, :] = sd[1]
        G[:, nv] = sd[2][::-1]
        G[0, :] = sd[3][::-1]
        X = P[G.clip(0)]
        if nu > 1 and nv > 1:
            B, T = P[G[:, 0]], P[G[:, nv]]
            L, R = P[G[0, :]], P[G[nu, :]]
            s = (np.arange(nu + 1) / nu)[:, None, None]
            t = (np.arange(nv + 1) / nv)[None, :, None]
            X = _unit(((1 - t) * B[:, None] + t * T[:, None] + (1 - s) * L[None] + s * R[None]
                       - ((1 - s) * (1 - t) * B[0] + s * (1 - t) * B[nu] + (1 - s) * t * T[0] + s * t * T[nu])))
        i, j = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
        if _max_angle(np.stack([X[i, j], X[i + 1, j], X[i + 1, j + 1], X[i, j + 1]], axis=-2).reshape(-1, 4, 3)) \
                > np.radians(max_angle):
            n_sheared += 1
            fallback.append(f)
            continue
        if nu > 1 and nv > 1:
            Xi = X[1:nu, 1:nv].reshape(-1, 3)
            G[1:nu, 1:nv] = (n_pts + np.arange(len(Xi))).reshape(nu - 1, nv - 1)
            new_pts.append(Xi)
            n_pts += len(Xi)
        qd = np.stack([G[i, j], G[i + 1, j], G[i + 1, j + 1], G[i, j + 1]], axis=-1).reshape(-1, 4)
        ok = (qd[:, [0, 0, 0, 1, 1, 2]] != qd[:, [1, 2, 3, 2, 3, 3]]).all(axis=1)
        n_degen += int((~ok).sum())
        quads.append(qd[ok])
        quad_face.append(np.full(int(ok.sum()), f))
    P = np.vstack([P] + new_pts)
    # fallback faces: boundary + interior points (field-aligned packing from the face boundaries, or the bg nodes),
    # constrained Delaunay per face
    seqs = {}
    for f in fallback:
        seq = chain(lay.faces[f])
        if len(seq) > 1 and seq[0] == seq[-1]:
            seq = seq[:-1]
        if len(seq) < 3 or len(set(seq)) != len(seq):
            n_fail += 1
            continue
        seqs[f] = seq
    if bgd is not None and seqs:
        used = np.zeros(len(P), dtype=bool)
        if quads:
            used[np.concatenate([q.ravel() for q in quads])] = True
        seed = np.unique(np.concatenate([np.asarray(sq, dtype=np.int64) for sq in seqs.values()]))
        used[seed] = False
        pool = _pack_points(P, np.flatnonzero(used), seed, h, bgd)
    else:
        pool = Pb[bg_flag == 0]
    tree = cKDTree(pool)
    extra = []
    for f, seq in seqs.items():
        n_fallback += 1
        poly = P[seq]
        m = _unit(poly.mean(axis=0))
        rmax = np.max(np.linalg.norm(poly - m, axis=1))
        cand = np.asarray(tree.query_ball_point(m, rmax), dtype=np.int64)
        if len(cand):
            cand = cand[in_water(Domain([poly]), pool[cand])]
        if len(cand):
            hb = h(pool[cand]) / R_EARTH_M
            dense = _unit(np.vstack([poly + f_ * (np.roll(poly, -1, axis=0) - poly) for f_ in (0.0, 0.25, 0.5, 0.75)]))
            dmin, _ = cKDTree(dense).query(pool[cand])
            cand = cand[dmin > 0.5 * hb]
        # far helper points so that the hull contains the sphere centre (a face is far smaller than a hemisphere)
        e1, e2 = tangent_basis(m[None])
        far = _unit(np.vstack([-m] + [-0.5 * m + np.cos(a) * e1[0] + np.sin(a) * e2[0]
                                      for a in np.arange(6) * np.pi / 3]))
        loc = np.vstack([poly, pool[cand], far])
        k = len(seq)
        e = np.column_stack([np.arange(k), np.roll(np.arange(k), -1)]).astype(np.int64)
        tt = sphere_delaunay(loc)
        tt, e, _ = recover_edges(loc, tt, e, return_edges=True)          # e split at touches (M40)
        tt = remove_land(loc, tt, e)
        tt = tt[(tt < k + len(cand)).all(axis=1)]
        gid = np.r_[np.asarray(seq, dtype=np.int64), n_pts + np.arange(len(cand))]
        extra.append(pool[cand])
        n_pts += len(cand)
        tris.append(gid[tt])
    P = np.vstack([P] + extra)
    quads = np.vstack(quads) if quads else np.zeros((0, 4), dtype=np.int64)
    tris = np.vstack(tris) if tris else np.zeros((0, 3), dtype=np.int64)
    info = dict(n_fallback=n_fallback, n_fallback_fail=n_fail, n_mismatch=n_mismatch, n_collapsed=n_collapsed,
                n_degenerate_quads=n_degen, n_sheared=n_sheared,
                quad_face=np.concatenate(quad_face) if quad_face else np.zeros(0, dtype=np.int64))
    return P, quads.astype(np.int64), tris.astype(np.int64), info
