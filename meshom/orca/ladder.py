# SPDX-License-Identifier: GPL-3.0-or-later
"""A structured transition between an ORCA grid and the same map refined 3 times (ORCA025 -> ORCA12), built in ORCA's
own index space: the "ladder" (decisions B114-B116; m16_orca `transition = "ladder"`).

Coordinates: fine F indices (x, y), x the fine F column (periodic, nf = 3 nc columns), y the fine F row, both
fractional; the map (meshom.orca.generator.lattice) evaluated there: I = i_offset_f + (x + 1/2) / q_f,
J = j_offset_f + (y + 1/2) / q_f. A coarse F point (r, c) is the fine F point (3 r + b_j, 3 c + b_i) (B108's offsets;
ORCA025 / ORCA12: b_i = -1, b_j = +1).

- The strip lies between two plain rows (B114): the coarse part's last F row y0 = 3 R_c + b_j and the fine part's
  first F row y1 = R_f, H = y1 - y0 fine rows. Its rows are j-lines y_k (k = 0 .. n); the row height grades as the
  size field's blend (B110): h(d) = 3 * 3^(-d / H) fine rows at the distance d = y - y0, n = round(2 H / (3 ln 3))
  rows (`rows`).
- Its columns are i-lines (B115). Each coarse line X_c = 3 c + b_i becomes the fine lines X_c - 1, X_c, X_c + 1 by
  two Y-splits (a node with two children on the next row, the triangle between them): event 1 at row k1(c), where the
  row height is near 3 / sqrt(2) (between the widths 3 and 1.5), splits X_c into the arms X_c +- 0.75; between the two
  events every arm drifts by -0.25 (smoothstep over the rows between the zones), so the arms reach X_c - 1 and
  X_c + 0.5; event 2 at row k2(c), where the row height is near sqrt(1.5), splits X_c + 0.5 into X_c and X_c + 1.
  2 nc Y-splits = the 2 nc column insertions, 2 nc triangles, nothing else irregular but the 2 nc valence-5 Y nodes.
  An arm opens by a1 per row until it is open (0.75, 0.5): a1 such that its tilt on the sphere is `theta_deg` (the
  triangle's apex 2 theta, the quad corner at the arm 90 + theta; the map's local metric): one row where ORCA's
  cells are square, more where they are flat (near NEMO's Siberian grid pole, on land).
- Relaxation (B115, `relax`, weights RELAX): the inner rows' x by least squares (y fixed: rows stay j-lines; the edge
  rows fixed): widths change slowly along a row (a split's width jump spread over its neighbours), lines keep small
  tilts, the Y-split triangles their base. Index space, ORCA025 -> ORCA12: element size ratio p99 1.23 -> 1.10 for
  quad max angle 108.0 -> 110.7 deg.
- Stagger (B116): k1(c) = z1 + s1 p1((m1 c) mod P1), k2(c) = z2 + s2 p2((m2 c) mod P2): each event's zone has P
  slots s rows apart (ORCA025: 12 consecutive rows, 12 slots 2 rows apart), the events of neighbouring coarse columns
  sit on rows at least 2 apart (ORCA025: 5 and 10) and the events of one row are P coarse columns apart, so no row
  or column line carries a seam.

The edge rows are the two grids' own F points (exact: the joint is node for node); the inner rows are the map at
fractional (x, y), interpolated linearly in x between samples every 1/SUB fine columns (error < 1 cm on ORCA12).
"""
from dataclasses import dataclass, field

import numpy as np

SUB = 8                 # map samples per fine column along a row (linear interpolation between them)
THETA_DEG = 18.0        # B115: an arm's tilt on its first row (the Y-split triangle's apex 2 theta)
H_EVENT = (3.0 / np.sqrt(2.0), np.sqrt(1.5))    # B115: ideal row heights of events 1 and 2 (geometric mean widths)
ZONE_ROWS = (12, 12)    # B116: slots in each event's zone (the stagger periods P1, P2)
ZONE_STEP = (1, 2)      # B116: the rows between consecutive stagger slots of each zone
RELAX = (1.0, 1.0, 10.0)    # B115: the relaxation's weights (widths along a row, tilts, the triangles' bases)


@dataclass
class LadderTopo:
    """The ladder in fine index space."""
    x: np.ndarray               # (N,) fine F column (unwrapped: may lie outside [0, nf))
    y: np.ndarray               # (N,) fine F row
    row: np.ndarray             # (N,) strip row k of each node
    quads: np.ndarray           # (Q, 4) CCW (x east, y north)
    tris: np.ndarray            # (T, 3)
    y_rows: np.ndarray          # (n + 1,) the rows' y
    k1: np.ndarray              # (nc,) event rows
    k2: np.ndarray
    split_nodes: np.ndarray     # (2 nc,) the Y nodes
    info: dict = field(default_factory=dict)
    parent: np.ndarray | None = None    # (N,) the node of the row below on the same line (-1 on row 0)
    row_lists: list | None = None       # per row the node ids in x order


def rows(y0: float, y1: float, k: int = 3) -> np.ndarray:
    """(n + 1,) the strip's rows y_k (B114): ln of the row height linear in the distance from y0, from k fine rows at
    y0 to 1 at y1, n = round((k - 1) H / (k ln k)) rows; the row centres at equal steps of N(d) = the number of rows
    of height h(d) = k k^(-d / H) up to d."""
    H = float(y1 - y0)
    lk = np.log(k)
    n_cont = (k - 1) * H / (k * lk)
    n = max(int(round(n_cont)), 2)
    u = np.linspace(0.0, n_cont, n + 1)                           # N(d) = H (k^(d/H) - 1) / (k ln k)
    d = H * np.log1p(u * k * lk / H) / lk
    return y0 + d


def _perm(P: int, nc: int, step_rows: float) -> np.ndarray:
    """(P,) the stagger slot of each residue (m c mod P): m coprime with P maximising the shortest vector of the
    lattice {(c, slot)} in (3 fine columns per coarse column, step_rows fine rows per slot) units, so the events of
    neighbouring columns lie far apart in both directions."""
    from math import gcd

    best, bm = -1.0, 1
    for m in range(1, P):
        if gcd(m, P) != 1:
            continue
        dmin = np.inf
        for dc in range(1, P):
            ds = (m * dc) % P
            ds = min(ds, P - ds)
            dmin = min(dmin, np.hypot(3.0 * dc, step_rows * ds))
        if dmin > best:
            best, bm = dmin, m
    return (bm * np.arange(P)) % P, bm


def _zone(hk: np.ndarray, h_ideal: float, nrows: int, step: int, lo: int, hi: int) -> np.ndarray:
    """(nrows,) rows of an event zone: the slots step rows apart, centred on the row whose height is nearest h_ideal,
    kept inside [lo, hi]."""
    c = int(np.argmin(np.abs(np.log(hk / h_ideal))))
    span = (nrows - 1) * step
    s = int(np.clip(c - span // 2, lo, hi - span))
    return s + step * np.arange(nrows)


def _open(total: float, a1: float, j: int, m: int) -> float:
    """An arm's offset j rows after its split (B115): a1 per row (the tilt theta on the sphere) up to `total`, and
    at least linear to `total` at j = m (the next event of the line, or the fine edge)."""
    return total * min(1.0, max(j * a1 / total, j / max(m, 1)))


def event_zones(hk: np.ndarray, nc: int, zone_rows=ZONE_ROWS, zone_step=ZONE_STEP):
    """The two event zones of a strip with row heights hk (B116): (z1, z2) the slots' rows, (p1, p2) the slot of each
    residue, (m1, m2) the stagger multipliers, (P1, P2) the periods, (s1, s2) the slot steps. A short strip takes
    fewer slots (event 2's step 1, then the larger period down, to 3)."""
    n = len(hk)
    P1, P2 = zone_rows
    s1, s2 = zone_step
    while (P1 - 1) * s1 + (P2 - 1) * s2 + 12 > n and (s2 > 1 or P1 > 3 or P2 > 3):   # a short strip: fewer slots
        if s2 > 1:
            s2 -= 1
        elif P1 >= P2 and P1 > 3 or P2 <= 3:
            P1 -= 1
        else:
            P2 -= 1
    p1, m1 = _perm(P1, nc, s1 * hk.mean())
    p2, m2 = _perm(P2, nc, s2 * hk.mean())
    z1 = _zone(hk, H_EVENT[0], P1, s1, 1, n - 6)
    z2 = _zone(hk, H_EVENT[1], P2, s2, z1[-1] + 4, n - 3)
    if z2[0] < z1[-1] + 4 or z1[0] < 1 or z2[-1] > n - 3:
        raise ValueError(f"ladder: {n} rows are too few for the two event zones ({P1} and {P2} slots)")
    return z1, z2, p1, p2, m1, m2, (P1, P2), (s1, s2)


def topology(nc: int, y_rows: np.ndarray, b_i: int = -1, theta_open=None, k: int = 3,
             zone_rows=ZONE_ROWS, zone_step=ZONE_STEP) -> LadderTopo:
    """The ladder's nodes and elements in fine index space (module docstring). theta_open(c, ev, h) -> the first-row
    opening a1 (fine columns) of event ev (1, 2) of coarse column c at row height h (None: h tan THETA_DEG)."""
    if k != 3:
        raise ValueError("ladder: only a refinement by 3 (ORCA025 -> ORCA12)")
    n = len(y_rows) - 1
    hk = np.diff(y_rows)                                          # (n,) height of row k -> k + 1
    z1, z2, p1, p2, m1, m2, (P1, P2), (s1, s2) = event_zones(hk, nc, zone_rows, zone_step)
    c = np.arange(nc)
    k1 = z1[p1[c % P1]]
    k2 = z2[p2[c % P2]]
    # the drift of the arms: 0 up to the end of zone 1 (+2 rows of opening), 1 from the start of zone 2
    ka, kb = z1[-1] + 2, z2[0]
    t = np.clip((np.arange(n + 1) - ka) / max(kb - ka, 1), 0.0, 1.0)
    drift = -0.25 * t * t * (3.0 - 2.0 * t)
    if theta_open is None:
        theta_open = lambda cc, ev, h: h * np.tan(np.radians(THETA_DEG))   # noqa: E731
    X = 3.0 * c + b_i
    a1e1 = np.array([theta_open(cc, 1, hk[k1[cc]]) for cc in c])
    a1e2 = np.array([theta_open(cc, 2, hk[k2[cc]]) for cc in c])
    xs, ys, rs, ids = [], [], [], {}

    def node(key, xv, kk):
        ids[key] = len(xs)
        xs.append(xv)
        ys.append(y_rows[kk])
        rs.append(kk)
        return ids[key]

    # per row, per coarse column, its nodes in x order as (key, children keys)
    row_nodes = []
    for kk in range(n + 1):
        lst = []
        for cc in range(nc):
            e1, e2 = k1[cc], k2[cc]
            if kk <= e1:
                keys = [("L", cc, kk)]
                pos = [X[cc]]
            else:
                o = _open(0.75, a1e1[cc], kk - e1, e2 - e1)
                if kk <= e2:
                    keys = [("A-", cc, kk), ("A+", cc, kk)]
                    pos = [X[cc] - o + drift[kk], X[cc] + o + drift[kk]]
                else:
                    o2 = _open(0.5, a1e2[cc], kk - e2, n - e2)
                    keys = [("A-", cc, kk), ("B-", cc, kk), ("B+", cc, kk)]
                    pos = [X[cc] - 0.75 + drift[kk], X[cc] + 0.5 - o2, X[cc] + 0.5 + o2]
            for kv, pv in zip(keys, pos):
                node(kv, pv, kk)
            lst.append((cc, keys))
        row_nodes.append(lst)

    def children(key):
        kind, cc, kk = key
        nk = kk + 1
        if kind == "L":
            return [("L", cc, nk)] if kk < k1[cc] else [("A-", cc, nk), ("A+", cc, nk)]
        if kind == "A+" and kk == k2[cc]:
            return [("B-", cc, nk), ("B+", cc, nk)]
        return [(kind, cc, nk)]

    quads, tris, split = [], [], []
    parent = np.full(len(xs), -1, np.int64)
    row_lists = [np.array([ids[kv] for _, keys in row_nodes[kk] for kv in keys], np.int64) for kk in range(n + 1)]
    for kk in range(n):
        flat = [kv for _, keys in row_nodes[kk] for kv in keys]
        ch = [[ids[q] for q in children(kv)] for kv in flat]
        me = [ids[kv] for kv in flat]
        for m_, cl in zip(me, ch):
            parent[cl] = m_
        L = len(flat)
        for a in range(L):
            b = (a + 1) % L
            quads.append((me[a], me[b], ch[b][0], ch[a][-1]))
            if len(ch[a]) == 2:
                tris.append((me[a], ch[a][1], ch[a][0]))
                split.append(me[a])
    x = np.array(xs)
    info = {"rows": int(n), "nc": int(nc), "nf": int(k * nc), "zone_steps": [s1, s2],
            "zone1_rows": [int(z1[0]), int(z1[-1])],
            "zone2_rows": [int(z2[0]), int(z2[-1])], "stagger_m": [int(m1), int(m2)], "periods": [P1, P2],
            "drift_rows": [int(ka), int(kb)], "row_height_first_last": [float(hk[0]), float(hk[-1])],
            "event_row_heights": [[float(hk[z1].min()), float(hk[z1].max())],
                                  [float(hk[z2].min()), float(hk[z2].max())]]}
    return LadderTopo(x, np.array(ys), np.array(rs), np.array(quads, np.int64), np.array(tris, np.int64),
                      np.asarray(y_rows, float), k1, k2, np.array(split, np.int64), info, parent, row_lists)


def relax(t: LadderTopo, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0, base=None) -> np.ndarray:
    """(N,) the nodes' x after a least-squares relaxation (B115), y fixed (the rows stay j-lines), rows 0 and n fixed
    (the grids' own F points): minimise alpha * sum (second difference of x along a row / h)^2 (widths change
    slowly along a row, spreading a split's width jump over its neighbours) + beta * sum ((x - x_parent) / h)^2
    (lines stay i-lines: tilts) + gamma * sum ((x_r - x_l - base) / h)^2 over the Y-split triangles' bases (base:
    (T,) target lengths, None = the nominal ones: the triangle keeps its shape). h = the row's height."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    n = len(t.y_rows) - 1
    nf = t.info["nf"]
    N = len(t.x)
    hk = np.diff(t.y_rows)
    fixed = (t.row == 0) | (t.row == n)
    rows_, cols_, vals_, rhs_ = [], [], [], []
    r = 0

    def term(idx, coef, off, w):
        """sum_j coef_j x[idx_j] + off, squared, weight w (arrays over terms)."""
        nonlocal r
        m = len(off)
        for j in range(idx.shape[1]):
            rows_.append(r + np.arange(m))
            cols_.append(idx[:, j])
            vals_.append(np.sqrt(w) * coef[:, j])
        rhs_.append(-np.sqrt(w) * off)
        r += m

    for kk in range(1, n):                                        # along a row (periodic: the wrap adds nf)
        ids = t.row_lists[kk]
        L = len(ids)
        a, b, c = ids, np.roll(ids, -1), np.roll(ids, -2)
        off = np.zeros(L)
        off[L - 2] += nf                                          # c wraps for the last two
        off[L - 1] += -2.0 * nf + nf                              # b and c wrap for the last one
        h = hk[kk] if kk < n else hk[-1]
        term(np.c_[a, b, c], np.tile([1.0, -2.0, 1.0], (L, 1)) / h, off / h, np.full(L, alpha))
    ch = np.flatnonzero(t.parent >= 0)                            # along a line
    hseg = hk[t.row[ch] - 1]
    term(np.c_[ch, t.parent[ch]], np.c_[1.0 / hseg, -1.0 / hseg], np.zeros(len(ch)), np.full(len(ch), beta))
    if len(t.tris) and gamma > 0:
        rr, ll = t.tris[:, 1], t.tris[:, 2]
        bb = (t.x[rr] - t.x[ll]) if base is None else np.asarray(base, float)
        ht = hk[t.row[t.tris[:, 0]]]
        term(np.c_[rr, ll], np.c_[1.0 / ht, -1.0 / ht], -bb / ht, np.full(len(rr), gamma))
    R = np.concatenate(rows_)
    C = np.concatenate(cols_)
    V = np.concatenate(vals_)
    rhs = np.concatenate(rhs_)
    A = coo_matrix((V, (R, C)), shape=(r, N)).tocsr()
    free = np.flatnonzero(~fixed)
    Af, Ax = A[:, free], A[:, np.flatnonzero(fixed)]
    rhs = rhs - Ax @ t.x[fixed]
    x = t.x.copy()
    x[free] = spsolve((Af.T @ Af).tocsc(), Af.T @ rhs)
    return x


def index_quality(t: LadderTopo, sx=1.0, sy=1.0) -> dict:
    """Angles of the ladder's elements in index space (x scaled by sx, y by sy): quad max angle percentiles, triangle
    min / max angles, quad aspect (the longer over the shorter mean of opposite sides), the element size ratio across
    inner edges (mean edge lengths)."""
    P = np.c_[t.x * sx, t.y * sy]
    nf = t.info["nf"] * sx

    def ang(c):
        q = P[c].copy()
        q[..., 0] -= nf * np.round((q[..., 0] - q[:, :1, 0]) / nf)          # the wrap column
        a = np.roll(q, -1, axis=1) - q
        b = np.roll(q, 1, axis=1) - q
        cr = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        dt = (a * b).sum(-1)
        return np.degrees(np.mod(np.arctan2(cr, dt), 2 * np.pi))

    qa, ta = ang(t.quads), ang(t.tris)

    def edges(c):
        d = P[np.roll(c, -1, axis=1)] - P[c]
        d[..., 0] -= nf * np.round(d[..., 0] / nf)
        return np.linalg.norm(d, axis=-1)

    qe, te = edges(t.quads), edges(t.tris)
    a02, a13 = qe[:, 0] + qe[:, 2], qe[:, 1] + qe[:, 3]
    asp = np.maximum(a02, a13) / np.minimum(a02, a13)
    qm = qa.max(axis=1)
    from meshom.mesh import Mesh, build_topology

    top = build_topology(Mesh(np.zeros((len(t.x), 3)), t.tris, t.quads, np.zeros(len(t.x), np.int8)))
    me = np.r_[te.mean(axis=1), qe.mean(axis=1)]
    ee = top.edge_elems[top.edge_elems[:, 1] >= 0]
    ratio = np.maximum(me[ee[:, 0]], me[ee[:, 1]]) / np.minimum(me[ee[:, 0]], me[ee[:, 1]])
    return {"ratio_p99": float(np.percentile(ratio, 99)), "ratio_p99_9": float(np.percentile(ratio, 99.9)),
            "ratio_max": float(ratio.max()), "quads": int(len(qa)), "tris": int(len(ta)),
            "quad_max_p50": float(np.percentile(qm, 50)), "quad_max_p99": float(np.percentile(qm, 99)),
            "quad_max_p99_9": float(np.percentile(qm, 99.9)), "quad_max": float(qm.max()), "quad_min": float(qa.min()),
            "quads_over_110_pct": float(100 * (qm > 110).mean()),
            "tri_min": float(ta.min()), "tri_max": float(ta.max()),
            "aspect_p99": float(np.percentile(asp, 99)), "aspect_max": float(asp.max()),
            "quad_edge_mean": qe.mean(axis=1), "tri_edge_mean": te.mean(axis=1)}


# ---- on the sphere ---------------------------------------------------------------------------------------------

def edge_rows(gc, gf, lat_coarse: float, lat_fine: float) -> tuple[int, int]:
    """(R_c, R_f) the strip's two plain rows (B114): R_c the last F row of the coarse grid whose points all lie south
    of lat_coarse (the coarse part's top), R_f the last F row of the fine grid whose points all lie south of
    lat_fine (the fine part's bottom: the fine part covers everything north of lat_fine)."""
    def last_south(g, lat):
        mx = g.lat["f"][:g.j_pole - 1, :-2].max(axis=1)
        ok = np.flatnonzero(mx <= lat)
        if not len(ok):
            raise ValueError(f"ladder: no F row lies south of {lat}")
        return int(ok[-1])
    return last_south(gc, lat_coarse), last_south(gf, lat_fine)


def _samples(gf, y_rows: np.ndarray, sub: int = SUB):
    """(n + 1, nf sub, 3) the fine map's points at the rows y_rows and x = 0, 1/sub, ... (fine F indices)."""
    from meshom.geo import lonlat_to_xyz
    from meshom.orca.generator import lattice, row_map

    p, c, q = gf.params, gf.coef, gf.q
    nf = gf.lon["f"].shape[1] - 2
    xs = np.arange(nf * sub) / sub
    zi = gf.info["i_offset"] + (xs + 0.5) / q
    jj = gf.info["j_offset"] + (np.asarray(y_rows, float) + 0.5) / q
    lon, lat = lattice(p, c, row_map(p, c), {"f": zi}, {"f": jj})
    return lonlat_to_xyz(lon["f"].ravel(), lat["f"].ravel()).reshape(len(jj), len(xs), 3)


def build(gc, gf, R_c: int, R_f: int, nest: dict, theta_deg: float = THETA_DEG, sub: int = SUB, relax_w=RELAX):
    """The ladder between the coarse grid's F row R_c and the fine grid's F row R_f (nest: meshom.orca.mesh.nesting
    of the two grids, k = 3) as a Mesh on the sphere (module docstring) and its record. The first-row opening of each
    Y-split is set so that its tilt on the sphere is theta_deg (the map's local metric sy / sx at the split)."""
    from meshom.mesh import Mesh

    k, b_i, b_j = nest["k"], nest["col_offset"], nest["row_offset"]
    nc = gc.lon["f"].shape[1] - 2
    nf = k * nc
    y0, y1 = k * R_c + b_j, R_f
    yr = rows(float(y0), float(y1), k)
    S = _samples(gf, yr, sub)                                       # (n + 1, nf sub, 3)
    n = len(yr) - 1
    tan_t = np.tan(np.radians(theta_deg))

    def metric(xv, kk):
        """sy / sx at fine column xv between rows kk and kk + 1: lengths per fine row and per fine column."""
        j = int(round((xv % nf) * sub)) % (nf * sub)
        j2 = (j + sub) % (nf * sub)
        sx = np.linalg.norm(S[kk, j2] - S[kk, j])
        sy = np.linalg.norm(S[kk + 1, j] - S[kk, j]) / (yr[kk + 1] - yr[kk])
        return sy / sx

    def opening(cc, ev, h):
        xv = k * cc + b_i + (0.0 if ev == 1 else 0.5)
        kk = int(np.argmin(np.abs(np.diff(yr) - h)))               # the event's row (its height h)
        return h * metric(xv, kk) * tan_t

    t = topology(nc, yr, b_i, opening, k)
    q_nominal = index_quality(t)
    if relax_w:                                                     # B115: spread the splits' width jumps
        from dataclasses import replace

        t = replace(t, x=relax(t, *relax_w))
    pts = np.empty((len(t.x), 3))
    for kk in range(n + 1):
        sel = np.flatnonzero(t.row == kk)
        if kk == 0:                                                 # the coarse grid's own F row R_c
            cc = np.round((t.x[sel] - b_i) / k).astype(np.int64) % nc
            from meshom.geo import lonlat_to_xyz
            pts[sel] = lonlat_to_xyz(gc.lon["f"][R_c, cc], gc.lat["f"][R_c, cc])
            continue
        if kk == n:                                                 # the fine grid's own F row R_f
            from meshom.geo import lonlat_to_xyz
            xi = np.round(t.x[sel]).astype(np.int64) % nf
            pts[sel] = lonlat_to_xyz(gf.lon["f"][R_f, xi], gf.lat["f"][R_f, xi])
            continue
        u = (t.x[sel] % nf) * sub
        j = np.floor(u).astype(np.int64)
        w = (u - j)[:, None]
        a, b = S[kk, j % (nf * sub)], S[kk, (j + 1) % (nf * sub)]
        p = (1.0 - w) * a + w * b
        pts[sel] = p / np.linalg.norm(p, axis=1, keepdims=True)
    mesh = Mesh(pts, t.tris, t.quads, np.zeros(len(pts), np.int8))
    q = index_quality(t)
    info = dict(t.info, y_rows=[float(y0), float(y1)], R_coarse=int(R_c), R_fine=int(R_f), theta_deg=float(theta_deg),
                sub=int(sub), relax=list(relax_w) if relax_w else None, nodes=int(mesh.n_nodes),
                quads=int(mesh.n_quads), tris=int(mesh.n_tris),
                insertions=int(len(t.split_nodes)), column_difference=int(nf - nc),
                index_quality={kq: round(v, 3) for kq, v in q.items() if isinstance(v, float)},
                index_quality_nominal={kq: round(v, 3) for kq, v in q_nominal.items() if isinstance(v, float)})
    return mesh, t, info


def preset_f_rows(name: str, rows) -> tuple[np.ndarray, np.ndarray]:
    """lon, lat (len(rows), columns) of a lattice preset's F rows, from the map (no grid built; as presets.build with
    no extra south rows): for config scripts that need the ladder's edge rows."""
    from meshom.orca.generator import lattice, row_map, set_coefficients
    from meshom.orca.presets import PRESETS

    pr = PRESETS[name]
    p = pr.params
    c = set_coefficients(p)
    rm = row_map(p, c)
    joff = (c.rjpj + rm.shift) - pr.pole_row / pr.q
    nu = int(round((p.jpi - 1) * pr.q))
    zi = pr.i_offset + (np.arange(nu) + 0.5) / pr.q
    jj = joff + (np.asarray(rows, float) + 0.5) / pr.q
    lon, lat = lattice(p, c, rm, {"f": zi}, {"f": jj})
    return lon["f"], lat["f"]


def preset_edge_rows(name_c: str, name_f: str, lat_coarse: float, lat_fine: float, guess=None) -> tuple[int, int]:
    """edge_rows from the map of two lattice presets: the rows searched in windows `guess` = (range_c, range_f) of
    F rows (the max latitude of an F row grows with the row in the north part)."""
    out = []
    for name, lat, rr in ((name_c, lat_coarse, guess[0]), (name_f, lat_fine, guess[1])):
        r = np.asarray(list(rr))
        _, la = preset_f_rows(name, r)
        ok = np.flatnonzero(la.max(axis=1) <= lat)
        if not len(ok) or ok[-1] == len(r) - 1 or ok[-1] == 0 and la[0].max() > lat:
            raise ValueError(f"preset_edge_rows: widen the window for {name}")
        out.append(int(r[ok[-1]]))
    return out[0], out[1]
