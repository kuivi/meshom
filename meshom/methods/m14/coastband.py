"""m14 thin coastal bands (decisions C36, C37, C25; option `coast_bands`, default off): fixed
blocks of a few rows extruded from the coast, only where h is flat to the coast; conforming mode only. They are built
after the strait blocks on the coast those leave free, and enter m14 through the fixed-block path of m14/strait.py
(B59): the automatic placement keeps block_gap from them, the stitch re-assembles the coast loops with their offshore
outline as exact nodes and never moves their coast row. The defect line of the coast moves to the band's offshore
seam, which the seam joins handle.

1. Free coast: every domain loop no fixed block touches (closed), and every piece of a touched loop from a fixed
   block's run end forward to the next run start on that loop (open; the walks of stitch.merged_loops).
2. Each piece is parametrised by u = integral ds/h and sampled every 1/SUB; the coast nodes are every SUB samples
   (about h apart).
3. Corners (C37): a raw vertex turning by more than CORNER_DEG (C18) whose node the coast filtered at 2 h confirms
   (turning by more than CORNER_DEG over the node and CORNER_WIN nodes on each side: a 90 deg corner turns ~ 18 deg
   per node after the filter, a sub-2h wiggle not at all). The piece is cut at its corners into segments; a closed
   loop is opened at a corner, else at its node of largest turning. Each segment is filtered on its own (a Gaussian
   in u, odd reflection with its ends kept, so a corner stays sharp): at sigma = FILTER (2 h; C16, C35) for the coast
   row, at NORMAL_FILTER (8 h) for the extrusion normals (to the water: the left of the loop). A band keeps GAP nodes
   from each segment end (the wedge between two bands' end cuts at a corner, and the coast next to a fixed block, is
   fill); a segment shorter than STRETCH_MIN + 2 GAP nodes gets no band.
4. Rows (C36, C25): per node n = min(n_max, k*, plateau / h, LFS / (2 h)) with k* = K_STAR rho_med / h_med (rho_med
   the median radius of curvature of the filtered coast over the segment, from the turning per node; h_med its median
   h); plateau = the rows j h along the normal with |ln h / h(coast node)| <= PLATEAU_TOL (h flat to the coast); LFS =
   the distance along the normal over which the water stays clear: the j-th sample (at j h) is clear when no filtered
   coast sample, raw coast too short for a band or fixed-block node lies within (CLEAR j - LFS_TOL) h of it (the
   opposite coast at W gives LFS ~ W / 1.8; the tolerance lets the 8 h normal lean against local wiggles of the 2 h
   coast). Per cell the smaller of its two nodes; where the cell's two normals converge (a bay) and their rays meet
   d_x from the coast (in the tangent plane), at most d_x / (BAY h) rows; the same across a corner wedge (`_wedge`):
   the rays of the last band node before a corner and the first after it (GAP from it on each side), where they meet
   d_x out (a concave corner seen from the water), bound the two end cells to d_x / (BAY h) rows (else the two bands'
   end cuts, 2 h apart, cross about cot(turn / 2) h out: 4 rows deep at turns of 40-50 deg, the 50 km crash of the
   re-assembled coast loop; sharper corners the LFS rule stops). The profile is regularised: at most one row
   of change per cell, peaks and valleys narrower than RUN_MIN cells removed (by lowering); cells with fewer than
   ROWS_MIN rows end the band (the dropped rows end in the fill).
5. Turning (C37): a band must pass |ln(1 + D Theta / L)| <= TURN_TOL as a whole, D = n h_med its depth, Theta its net
   turning (positive where convex), L its length: its offshore side carries the coast's cells, so this is the mean
   along-size error of its outer row (on a coast of uniform radius rho it allows D <= 0.105 rho, whatever the sweeps
   do; k* is the bound of a wavy coast, whose turning cancels). The deepest n not above the row rule that passes is
   taken; where none does down to ROWS_MIN, the band is cut: from its start the longest passing prefix (at least
   STRETCH_MIN nodes) is a band, the next starts CUT_GAP nodes on.
6. Build (C50 for the count and the ends): the segment's filtered coast between the band's two end nodes, blended
   onto the raw coast over END_RAW nodes at each end (the end nodes on the raw coast: the stitch's walk to the next
   block starts there without a spike), is cut into m cells of equal integral ds/h measured on the row at COUNT_ROW x
   the band's median depth (1: the offshore side meets the fill at h, the coast row takes the curvature's error,
   within TURN_TOL); each coast node extruded along the 8 h normal at the same u, row k at k h(coast node) on the
   great circle. SWEEPS "size" sweeps on the band extended by GHOST_ROWS times its depth of rows above
   it (measured on rows 0-3 of a 12-row extrusion; ghost cells from the first inverted one up are left out), the
   coast row and the two end cuts (the corner normals) fixed; the band's nodes are taken from the smoothed grid. A band cell inverted or above THETA_HARD
   after the sweeps lowers the rows there and the segment is planned again (at most REBUILD rounds; a band still bad
   is dropped). A band with a vertex of another coast loop inside its outline (an island under it, which the LFS
   samples h apart can miss) is dropped, as a strait block is (C39): the fill's land would leak into the water. Last,
   the coast loops are re-assembled with all bands as the stitch does (stitch.merged_loops, the other loops at h) and
   every band with an outline chord in a crossing is dropped, until none is left (`_crossing`; the fill needs simple
   loops: a loop crossing itself left the stitch fill's cross field NaN and m02's pack overflowing).
The automatic placement keeps band_gap (m14 option, 4.5 h; C50) from the bands' nodes, not block_gap (6 h): the fill
between a band and a block is narrower, its dislocation wall nearer the middle of the gap.
A band is a strait.FixedBlock: the outline with the fill on the left, the coast row as its coast mask, and one non-coast
run from its first coast node up the first end cut, along the offshore side, down the last end cut to its last coast
node, both ends located on the domain loop (loop, segment, t) by the raw samples at the same u (the band's end nodes
lie on the filtered coast). Its record (kind "band") carries the counts, the limits that bound it and per row the
size and angle numbers of the built band.
"""
import numpy as np
from scipy.spatial import cKDTree

from meshom.domain import Domain
from meshom.geo import R_EARTH_M, arc_length, xyz_to_lonlat
from meshom.mesh import Mesh, signed_area
from meshom.methods.m14.blockmap import BlockGrid
from meshom.methods.m14.strait import FixedBlock, _Coast
from meshom.quality.kernel import element_geometry
from meshom.size import SizeField

SUB = 8                  # samples per coast node (u step 1 / SUB)
FILTER = 2.0             # C35 / C16: the coast row filtered at sigma = 2 h
NORMAL_FILTER = 8.0      # extrusion normals of the coast smoothed at 8 h
CORNER_DEG = 30.0        # C18: a complex coast point turns by more than this
CORNER_WIN = 2           # C37: the filtered coast confirms a corner over the node and this many nodes on each side
GAP = 1                  # nodes left to the fill on each side of a corner and next to a fixed block
STRETCH_MIN = 10         # a band is at least this many nodes (h) long
ROWS_MIN = 2             # fewer rows: no band there
K_STAR = 0.32            # k* = 0.32 rho_med / h (after the size sweeps)
PLATEAU_TOL = 0.05       # h flat to the coast: |ln h / h(coast node)| <= this along the band's normal
CLEAR = 0.8              # the j-th normal sample (at j h) is clear when nothing lies within CLEAR j h - LFS_TOL h of
LFS_TOL = 0.25           # it: the opposite coast at W gives LFS ~ W / 1.8, local wiggles of the filtered coast pass
BAY = 2.0                # converging normals meeting at d_x allow d_x / (BAY h) rows
RUN_MIN = 3              # peaks / valleys of the row profile narrower than this many cells are removed
TURN_TOL = 0.10          # a band passes when |ln(1 + D Theta / L)| <= this (its depth drops, else it is cut)
CUT_GAP = 2              # nodes between two bands at a cut
SWEEPS = 30              # B25: "size" sweeps with the coast row fixed
THETA_HARD = 150.0       # G1 (quad_max_angle_hard_deg): a band cell above it after the sweeps lowers its rows
REBUILD = 3
GHOST_ROWS = 2           # the sweeps run on the band and this many times its depth more rows above it
COUNT_ROW = 1.0          # C50: the along count follows h on the row at this fraction of the band's depth (1: the
                         # offshore side meets the fill at h; the coast row takes the curvature's error, <= TURN_TOL)
END_RAW = 2.0            # C50: the coast row is blended onto the raw coast over this many nodes at each end (the
                         # band's end node on the raw coast: no spike where the walk to the next block starts)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _m(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return R_EARTH_M * arc_length(a, b)


def _gauss(X: np.ndarray, sig: float, closed: bool) -> np.ndarray:
    """Gaussian of the samples X (n,3) along their index at sigma = sig samples; closed: periodic, else odd reflection
    with the ends kept."""
    n = len(X)
    w = int(np.ceil(3.0 * sig))
    if not closed:
        w = min(w, n - 1)
    if w < 1:
        return X.copy()
    ker = np.exp(-0.5 * (np.arange(-w, w + 1) / sig) ** 2)
    ker /= ker.sum()
    i = np.arange(-w, n + w)
    if closed:
        Y = X[i % n]
    else:
        Y = X[np.clip(i, 0, n - 1)].copy()
        lo, hi = i < 0, i > n - 1
        Y[lo] = 2.0 * X[0] - X[np.clip(-i[lo], 0, n - 1)]
        Y[hi] = 2.0 * X[-1] - X[np.clip(2 * (n - 1) - i[hi], 0, n - 1)]
    out = _unit(np.column_stack([np.convolve(Y[:, d], ker, mode="valid") for d in range(3)]))
    if not closed:
        out[0], out[-1] = X[0], X[-1]
    return out


def _turns(F: np.ndarray) -> np.ndarray:
    """Signed turning (rad, > 0 to the left = toward the water) at the inner nodes of the open polyline F; 0 at ends."""
    th = np.zeros(len(F))
    if len(F) < 3:
        return th
    p, a, b = F[1:-1], F[1:-1] - F[:-2], F[2:] - F[1:-1]
    a = a - np.einsum("ij,ij->i", a, p)[:, None] * p
    b = b - np.einsum("ij,ij->i", b, p)[:, None] * p
    th[1:-1] = np.arctan2(np.einsum("ij,ij->i", p, np.cross(a, b)), np.einsum("ij,ij->i", a, b))
    return th


# ------------------------------------------------------------------------------------------------- free coast

def free_pieces(domain: Domain, fixed: list) -> list[dict]:
    """The coast left free by the fixed blocks (module docstring, 1.): per piece its loop, closed flag, vertices and,
    per piece segment, the loop segment and the loop parameters t at its two ends."""
    loops = domain.loops
    runs = [(s, e) for f in fixed for _, s, e in f.runs]
    touched = {s[0] for s, _ in runs} | {e[0] for _, e in runs}
    out = []
    for li, lp in enumerate(loops):
        if li not in touched and len(lp) >= 3:
            K = len(lp)
            out.append(dict(loop=li, closed=True, V=np.vstack([lp, lp[:1]]), seg=np.arange(K),
                            ta=np.zeros(K), tb=np.ones(K)))
    pos = lambda loc: loc[1] + loc[2]                                   # noqa: E731
    for _, e in runs:
        L = e[0]
        lp = loops[L]
        K = len(lp)
        s = min((s for s, _ in runs if s[0] == L), key=lambda s: (pos(s) - pos(e)) % K)
        pt = lambda loc: _unit(lp[loc[1]] + loc[2] * (lp[(loc[1] + 1) % K] - lp[loc[1]]))   # noqa: E731
        cnt = (s[1] - e[1]) % K
        if cnt == 0 and s[2] < e[2]:
            cnt = K
        if cnt == 0:
            V, seg, ta, tb = np.vstack([pt(e), pt(s)]), [e[1]], [e[2]], [s[2]]
        else:
            idx = (e[1] + 1 + np.arange(cnt)) % K
            V = np.vstack([pt(e)[None], lp[idx], pt(s)[None]])
            seg = np.r_[e[1], idx]
            ta = np.r_[e[2], np.zeros(cnt)]
            tb = np.r_[np.ones(cnt), s[2]]
        out.append(dict(loop=L, closed=False, V=V, seg=np.asarray(seg), ta=np.asarray(ta, float),
                        tb=np.asarray(tb, float)))
    return out


def _sample(pc: dict, h: SizeField) -> dict | None:
    """Uniform samples in u of a piece: points R, loop locations (seg, t), the node count N (SUB samples each) and
    the u of the raw vertices turning by more than CORNER_DEG (corner candidates, C18)."""
    V = pc["V"]
    keep = np.r_[True, np.linalg.norm(np.diff(V, axis=0), axis=1) > 1e-12]
    V, seg, ta, tb = V[keep], pc["seg"][keep[1:]], pc["ta"][keep[1:]], pc["tb"][keep[1:]]
    if len(V) < 2:
        return None
    hv = h(V)
    w = _m(V[:-1], V[1:]) * 0.5 * (1.0 / hv[:-1] + 1.0 / hv[1:])
    cum = np.r_[0.0, np.cumsum(w)]
    U = float(cum[-1])
    N = int(round(U))
    if N < STRETCH_MIN + 2 * GAP:
        return None
    ns = SUB * N
    u = np.arange(ns + 1) * (U / ns)
    if pc["closed"]:
        u = u[:-1]
    j = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(w) - 1)
    f = np.clip((u - cum[j]) / np.where(w[j] > 0, w[j], 1.0), 0.0, 1.0)
    R = _unit((1.0 - f)[:, None] * V[j] + f[:, None] * V[j + 1])
    tv = np.abs(_turns(np.vstack([V[-2:-1], V]) if pc["closed"] else V))    # closed: V[-1] == V[0]
    tv = tv[1:] if pc["closed"] else tv
    cu = cum[np.flatnonzero(tv > np.radians(CORNER_DEG))]
    return dict(R=R, seg=seg[j], t=ta[j] + f * (tb[j] - ta[j]), N=N, U=U, closed=pc["closed"], loop=pc["loop"],
                corner_u=cu, V=V)


def _corners(sm: dict) -> np.ndarray:
    """Corner nodes (C37): a raw vertex turning by more than CORNER_DEG (C18) whose node the filtered coast confirms,
    turning by more than CORNER_DEG over the node and CORNER_WIN nodes on each side (a sub-2h wiggle does not); of
    corners within CORNER_WIN nodes of each other the sharpest."""
    N, closed = sm["N"], sm["closed"]
    F = _gauss(sm["R"], FILTER * SUB, closed)
    k = np.arange(N) * SUB if closed else np.arange(N + 1) * SUB
    Fn = F[k]
    th = _turns(np.vstack([Fn[-1:], Fn, Fn[:1]]))[1:-1] if closed else _turns(Fn)
    cs = np.cumsum(np.r_[0.0, np.r_[th[-CORNER_WIN:], th, th[:CORNER_WIN]] if closed else np.r_[
        np.zeros(CORNER_WIN), th, np.zeros(CORNER_WIN)]])
    tau = cs[2 * CORNER_WIN + 1:] - cs[:-2 * CORNER_WIN - 1]
    cand = np.unique(np.rint(sm["corner_u"] / (sm["U"] / N)).astype(np.int64) % (N if closed else N + 1))
    cand = cand[np.abs(tau[cand]) > np.radians(CORNER_DEG)]
    if not closed:
        cand = cand[(cand > 0) & (cand < N)]
    out = []
    for c in cand[np.argsort(-np.abs(tau[cand]))]:
        if all(min(abs(c - x), N - abs(c - x)) > CORNER_WIN if closed else abs(c - x) > CORNER_WIN for x in out):
            out.append(int(c))
    sm["tau"] = tau
    return np.sort(np.asarray(out, dtype=np.int64))


# ---------------------------------------------------------------------------------------------- row planning

def _regular(n: np.ndarray) -> np.ndarray:
    """Row profile (cells): at most one row of change per cell; interior runs narrower than RUN_MIN removed (a peak
    lowered to its higher neighbour, a valley widened by lowering its lower neighbour); an end run narrower than
    RUN_MIN and higher than its neighbour lowered to it."""
    n = n.copy()
    for _ in range(4 * len(n) + 4):
        old = n.copy()
        for i in range(1, len(n)):
            n[i] = min(n[i], n[i - 1] + 1)
        for i in range(len(n) - 2, -1, -1):
            n[i] = min(n[i], n[i + 1] + 1)
        b = np.flatnonzero(np.diff(n)) + 1
        st, en = np.r_[0, b], np.r_[b, len(n)]
        for r in range(len(st)):
            a, z = st[r], en[r]
            if z - a >= RUN_MIN or len(st) == 1:
                continue
            v = n[a]
            left = n[a - 1] if r > 0 else None
            right = n[z] if r < len(st) - 1 else None
            nb = [x for x in (left, right) if x is not None]
            if all(v > x for x in nb):
                n[a:z] = max(nb)
            elif left is not None and right is not None and v < left and v < right:
                k = RUN_MIN - (z - a)
                if left <= right:
                    n[max(0, a - k):a] = v
                else:
                    n[z:z + k] = v
            else:
                continue
            break                                     # one change per pass: the runs moved
        if np.array_equal(old, n):
            break
    return n


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _cuts(a: int, b: int, nmax: int, seg: np.ndarray, th: np.ndarray, h_med: float) -> list[tuple[int, int, int]]:
    """The node range [a, b] as bands that each pass |ln(1 + D Theta / L)| <= TURN_TOL as a whole with D = n h_med
    (module docstring, 5.): [(a', b', n)]. From the start, the deepest n <= nmax (down to ROWS_MIN) for which the rest
    passes whole; else the longest passing prefix of at least STRETCH_MIN nodes (the deeper n on a tie) and the next
    band CUT_GAP nodes after it; where no prefix passes, the start moves on by one node."""
    out = []
    while b - a >= STRETCH_MIN:
        L = np.cumsum(seg[a:b])                          # L[k]: length from node a to node a + k + 1
        Th = -np.r_[0.0, np.cumsum(th[a + 1:b])][:b - a]  # net turning strictly between them, convex positive
        best = None                                      # (prefix end, n)
        for n in range(nmax, ROWS_MIN - 1, -1):
            x = 1.0 + n * h_med * Th / L
            ok = (x > 0) & (np.abs(np.log(np.maximum(x, 1e-12))) <= TURN_TOL)
            ok[:STRETCH_MIN - 1] = False
            if ok.any():
                e = a + int(np.flatnonzero(ok)[-1]) + 1
                if best is None or e > best[0]:
                    best = (e, n)
                if e == b:
                    break
        if best is None:
            a += 1
            continue
        out.append((a, best[0], best[1]))
        a = best[0] + CUT_GAP
    return out


# ---------------------------------------------------------------------------------------------- build

def _row_stats(P: np.ndarray, ids: np.ndarray, ncell: np.ndarray, h: SizeField, th_cell: np.ndarray,
               krow: np.ndarray) -> list[dict]:
    """Per node row k: |ln s/h| p99 / max of its along edges, p99 of the across edges from it to row k + 1, and the
    largest quad angle of cell row k."""
    out = []
    top = np.r_[ncell[:1], np.maximum(ncell[:-1], ncell[1:]), ncell[-1:]]
    for k in range(int(ncell.max()) + 1):
        i = np.flatnonzero(ncell >= k)
        a, b = P[ids[i, k]], P[ids[i + 1, k]]
        al = np.abs(np.log(_m(a, b) / h(_unit(a + b))))
        rec = {"row": k, "along_p99": round(float(np.percentile(al, 99)), 4), "along_max": round(float(al.max()), 4)}
        j = np.flatnonzero(top > k)
        if len(j) and k < int(ncell.max()):
            c, d = P[ids[j, k]], P[ids[j, k + 1]]
            ac = np.abs(np.log(_m(c, d) / h(_unit(c + d))))
            rec["across_p99"] = round(float(np.percentile(ac, 99)), 4)
            rec["across_max"] = round(float(ac.max()), 4)
            rec["theta_max"] = round(float(th_cell[krow == k].max()), 2)
        out.append(rec)
    return out


def _build(sg: dict, a: int, b: int, ncell_nodes: np.ndarray, h: SizeField):
    """The band between nodes a and b of segment sg, ncell_nodes = the rows of the cells [a, b): ((grid, coast mask,
    run), None, info), or (None, the bad cells as (cell, row), info) when a cell is inverted or above THETA_HARD
    after the sweeps."""
    from meshom.smooth import smooth

    q = sg["F"][SUB * a:SUB * b + 1]                           # the segment's filtered coast between the band's ends
    nrm = sg["NRM"][SUB * a:SUB * b + 1]
    if END_RAW > 0.0:                                          # C50: the coast row ends on the raw coast (blended
        ns = np.arange(len(q)) / SUB                           # over END_RAW nodes from each end)
        d = np.minimum(ns, ns[-1] - ns)
        wgt = np.where(d < END_RAW, 0.5 * (1.0 + np.cos(np.pi * np.minimum(d / END_RAW, 1.0))), 0.0)[:, None]
        q = _unit(q + wgt * (sg["R"][SUB * a:SUB * b + 1] - q))
    hv = h(q)
    w = _m(q[:-1], q[1:]) * 0.5 * (1.0 / hv[:-1] + 1.0 / hv[1:])
    if COUNT_ROW > 0.0:                                        # C50: u = integral ds/h along the row at COUNT_ROW x
        dep = float(np.median(ncell_nodes))                    # the median depth (one depth: a staircase step is no
        nq = _unit(nrm - np.einsum("ij,ij->i", nrm, q)[:, None] * q)       # along length)
        ang = (COUNT_ROW * dep * hv / R_EARTH_M)[:, None]
        X = _unit(np.cos(ang) * q + np.sin(ang) * nq)
        hx = h(X)
        w = _m(X[:-1], X[1:]) * 0.5 * (1.0 / hx[:-1] + 1.0 / hx[1:])
    cum = np.r_[0.0, np.cumsum(w)]
    m = max(1, int(round(float(cum[-1]))))
    u = np.linspace(0.0, cum[-1], m + 1)
    j = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(w) - 1)
    f = np.clip((u - cum[j]) / np.where(w[j] > 0, w[j], 1.0), 0.0, 1.0)[:, None]
    C = _unit((1.0 - f) * q[j] + f * q[j + 1])
    C[0], C[-1] = q[0], q[-1]
    nr = (1.0 - f) * nrm[j] + f * nrm[j + 1]
    nr = _unit(nr - np.einsum("ij,ij->i", nr, C)[:, None] * C)
    jm = np.clip(np.searchsorted(cum, 0.5 * (u[:-1] + u[1:]), side="right") - 1, 0, len(w) - 1)
    pc = np.minimum(a + jm // SUB, b - 1)                      # segment cell of each cell: the sample at its middle
    ncell = ncell_nodes[pc - a]
    top = np.r_[ncell[:1], np.maximum(ncell[:-1], ncell[1:]), ncell[-1:]]
    K = int(top.max())
    ids = np.full((m + 1, K + 1), -1, dtype=np.int64)
    hc = h(C)
    pts, cnt = [], 0
    for k in range(K + 1):                                     # row k at k h(coast node) along the normal
        i = np.flatnonzero(top >= k)
        ang = (k * hc[i] / R_EARTH_M)[:, None]
        ids[i, k] = cnt + np.arange(len(i))
        cnt += len(i)
        pts.append(_unit(np.cos(ang) * C[i] + np.sin(ang) * nr[i]))
    P = np.vstack(pts)
    ci, ck = np.nonzero(np.arange(K)[None, :] < ncell[:, None])
    quads = np.column_stack([ids[ci, ck], ids[ci + 1, ck], ids[ci + 1, ck + 1], ids[ci, ck + 1]])
    coast = np.zeros(len(P), dtype=bool)
    coast[ids[:, 0]] = True
    # SWEEPS "size" sweeps on the band and GHOST_ROWS x its depth more rows above it (12 rows for 4),
    # the coast row and the two end cuts fixed; the band's nodes are taken from the smoothed grid
    KG = K * (1 + GHOST_ROWS)
    ang = (np.arange(KG + 1)[None, :] * hc[:, None] / R_EARTH_M)[..., None]
    G = _unit(np.cos(ang) * C[:, None, :] + np.sin(ang) * nr[:, None, :])       # (m + 1, KG + 1, 3)
    gid = np.arange((m + 1) * (KG + 1)).reshape(m + 1, KG + 1)
    gi, gk = np.nonzero(np.ones((m, KG), dtype=bool))
    gq = np.column_stack([gid[gi, gk], gid[gi + 1, gk], gid[gi + 1, gk + 1], gid[gi, gk + 1]])
    gm = Mesh(G.reshape(-1, 3), np.zeros((0, 3), np.int64), gq, np.zeros(gid.size, np.int8))
    inv = (signed_area(gm) <= 0).reshape(m, KG)                 # ghost rows stop below an inverted ghost cell
    inv[:, :K] = False
    first = np.where(inv.any(axis=1), inv.argmax(axis=1), KG)
    gm = Mesh(gm.points, gm.tris, gq[gk < first[gi]], gm.node_flag)
    fix = np.zeros(gid.size, dtype=bool)
    fix[gid[:, 0]] = fix[gid[0]] = fix[gid[-1]] = True
    gm = smooth(gm, h, fix, iters=SWEEPS, kind="size")
    ii, kk = np.nonzero(ids >= 0)
    P = P.copy()
    P[ids[ii, kk]] = gm.points[gid[ii, kk]]
    mesh = Mesh(P, np.zeros((0, 3), np.int64), quads, np.zeros(len(P), np.int8))
    P = mesh.points
    g = element_geometry(mesh)
    th = np.degrees(g.quad_angles).max(axis=1)
    sa = signed_area(mesh)
    bad = (sa <= 0) | (th > THETA_HARD)
    info = dict(m=m, inverted=int((sa <= 0).sum()), theta_max=round(float(th.max()), 2))
    if bad.any():
        return None, np.unique(np.c_[pc[ci[bad]], ck[bad]], axis=0), info
    # outline (fill on the left): up the first end cut, along the offshore side, down the last end cut, the coast back
    run = list(ids[0, :ncell[0] + 1])
    for i in range(m):
        run.append(ids[i + 1, ncell[i]])
        if i + 1 < m and ncell[i + 1] != ncell[i]:
            step = 1 if ncell[i + 1] > ncell[i] else -1
            run += list(ids[i + 1, ncell[i] + step:ncell[i + 1] + step:step])
    run += list(ids[m, ncell[-1] - 1::-1])
    run = np.asarray(run, dtype=np.int64)
    outline = np.r_[run, ids[m - 1:0:-1, 0]]
    ij = np.zeros((len(P), 2), dtype=np.int64)
    ii, kk = np.nonzero(ids >= 0)
    ij[ids[ii, kk]] = np.column_stack([ii, kk])
    info.update(n_cells=int(len(quads)), rows=[int(ncell.min()), float(np.median(ncell)), int(ncell.max())],
                row_stats=_row_stats(P, ids, ncell, h, th, ck))
    grid = BlockGrid(P, quads, ij, outline, dict(n_cells=int(len(quads)), inradius_cells=float(ncell.min()) / 2))
    locs = [(sg["loop"], int(sg["seg"][x]), float(sg["t"][x])) for x in (SUB * a, SUB * b)]
    return (grid, coast, (run, locs[0], locs[1])), None, info


def _meet(P0: np.ndarray, P1: np.ndarray, n0: np.ndarray, n1: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Converging normals: the rays from P0 along n0 and from P1 along n1 (unit, tangent; seg = |P0 P1| in m) meet
    at a n0 = b n1 + (P1 - P0) in the tangent plane; min(a, b) (m) where both are positive, else inf."""
    mid = _unit(P0 + P1)
    t = P1 - P0
    t = _unit(t - np.einsum("ij,ij->i", t, mid)[:, None] * mid)
    w = np.cross(mid, t)
    ax, ay = np.einsum("ij,ij->i", n0, t), np.einsum("ij,ij->i", n0, w)
    bx, by = np.einsum("ij,ij->i", n1, t), np.einsum("ij,ij->i", n1, w)
    det = bx * ay - ax * by
    with np.errstate(divide="ignore", invalid="ignore"):
        ra, rb = -seg * by / det, -seg * ay / det
    return np.where((ra > 0) & (rb > 0), np.minimum(ra, rb), np.inf)


def _wedge(A: dict, B: dict, h: SizeField) -> float:
    """Corner wedge (C36): segment A ends where segment B starts (a corner, or where a closed loop was opened); the
    rays of A's last band node (GAP before its end) and of B's first (GAP after its start) meet d_x out where the
    corner is concave seen from the water: the end cells on both sides take at most d_x / (BAY h) rows (the bay rule
    across the wedge, so the two bands' end cuts never cross); inf where the rays diverge."""
    ka, kb = SUB * (A["N"] - GAP), SUB * GAP
    P = np.vstack([A["F"][ka], B["F"][kb]])
    n = np.vstack([A["NRM"][ka], B["NRM"][kb]])
    n = _unit(n - np.einsum("ij,ij->i", n, P)[:, None] * P)
    dx = float(_meet(P[:1], P[1:], n[:1], n[1:], _m(P[:1], P[1:]))[0])
    return float(np.floor(dx / (BAY * float(np.mean(h(P)))))) if np.isfinite(dx) else np.inf


def _crossing(domain: Domain, h: SizeField, fixed: list, bands: list) -> set:
    """The bands whose outline runs cross a chord of the coast loops the stitch re-assembles with them
    (stitch.merged_loops; the loops no block touches at h, as the fill sees them): indices into `bands`."""
    from meshom.domain import crossing_chords, resample_loops
    from meshom.methods.m14.stitch import merged_loops

    own = [f.grid.points[f.grid.outline] for f in fixed if not f.coast.any()]     # hand blocks in the water
    ml, ids, touched = merged_loops(domain, h, list(fixed) + list(bands), own)
    rest = [lp for i, lp in enumerate(domain.loops) if i not in touched]
    L = resample_loops(Domain(ml + rest + own, domain.name, [True] * len(ml) + [False] * len(rest) + [True] * len(own)),
                       h)
    off = np.r_[0, np.cumsum([len(x) for x in L])]
    ia = np.arange(off[-1])
    ib = ia + 1
    ib[off[1:] - 1] = off[:-1]
    hit = crossing_chords(np.vstack(L), ia, ib)
    g = np.concatenate(ids + [np.full(len(x), -1, dtype=np.int64) for x in rest + own])
    foff = np.r_[0, np.cumsum([len(f.grid.points) for f in list(fixed) + list(bands)])]
    nodes = g[np.r_[ia[hit], ib[hit]]]
    blk = np.searchsorted(foff, nodes[nodes >= 0], side="right") - 1 - len(fixed)
    return {int(b) for b in blk if b >= 0}


def _limits(sg: dict, h: SizeField, n_max: int, tree) -> dict:
    """Per node of a segment: the filtered coast nodes, turning, segment lengths, h, the plateau rows and the LFS
    rows; per cell the bay limit (module docstring, 4.)."""
    N = sg["N"]
    k = np.arange(N + 1) * SUB
    Fn = sg["F"][k]
    nn = sg["NRM"][k]
    nn = _unit(nn - np.einsum("ij,ij->i", nn, Fn)[:, None] * Fn)
    hn = h(Fn)
    seg = _m(Fn[:-1], Fn[1:])
    J = 2 * n_max + 1                                       # LFS / (2 h) <= n_max needs 2 n_max + 1 samples
    plateau, clear = np.full(N + 1, n_max), np.full(N + 1, J)
    flat, free = np.ones(N + 1, dtype=bool), np.ones(N + 1, dtype=bool)
    for j in range(1, J + 1):
        ang = (j * hn / R_EARTH_M)[:, None]
        x = _unit(np.cos(ang) * Fn + np.sin(ang) * nn)
        if j <= n_max:
            flat &= np.abs(np.log(h(x) / hn)) <= PLATEAU_TOL
            plateau = np.where(flat | (plateau < j), plateau, j - 1)
        d, _ = tree.query(x)
        free &= R_EARTH_M * 2.0 * np.arcsin(np.clip(0.5 * d, 0.0, 1.0)) >= (CLEAR * j - LFS_TOL) * hn
        clear = np.where(free | (clear < j), clear, j - 1)
    dx = _meet(Fn[:-1], Fn[1:], nn[:-1], nn[1:], seg)
    hc = 0.5 * (hn[:-1] + hn[1:])
    return dict(th=_turns(Fn), seg=seg, hn=hn, plateau=plateau, lfs=clear // 2, bay=np.floor(dx / (BAY * hc)))


def _plan(a: int, b: int, L: dict, n_max: int, cap: np.ndarray) -> tuple[list, np.ndarray, dict]:
    """The bands of the stretch [a, b] (segment nodes): [(a', b')], the per-cell rows, the stretch's limits."""
    th, seg, hn = L["th"], L["seg"], L["hn"]
    rho = seg[a:b].mean() / np.maximum(np.abs(th[a + 1:b]), 1e-12)
    h_med = float(np.median(hn[a:b + 1]))
    rho_med = float(np.median(rho)) if len(rho) else np.inf
    kstar = int(min(n_max, np.floor(K_STAR * rho_med / h_med)))
    r_node = np.minimum(np.minimum(kstar, L["plateau"][a:b + 1]), L["lfs"][a:b + 1])
    nc = np.zeros(len(seg), dtype=np.int64)
    nc[a:b] = _regular(np.minimum.reduce([r_node[:-1], r_node[1:], L["bay"][a:b], cap[a:b]]).astype(np.int64))
    plan = []
    for sa, sb in _segments(nc[a:b] >= ROWS_MIN):
        if sb - sa >= STRETCH_MIN:
            for pa, pb, n in _cuts(a + sa, a + sb, int(nc[a + sa:a + sb].max()), seg, th, h_med):
                nc[pa:pb] = np.minimum(nc[pa:pb], n)
                plan.append((pa, pb))
    lim = dict(n_max=n_max, k_star=kstar, rho_med_m=round(rho_med, 1) if np.isfinite(rho_med) else None,
               h_med_m=round(h_med, 1))
    return plan, nc, lim


def coast_bands(domain: Domain, h: SizeField, o: dict, fixed: list = ()) -> tuple[list[FixedBlock], dict]:
    """The coastal bands of a conforming domain on the coast the fixed blocks leave free (module docstring); returns
    the bands as FixedBlocks (after `fixed` in the stitch's block order) and a summary record."""
    n_max = int(o["coast_band_rows"])
    segs, short = [], []
    counts = dict(pieces=0, corners=0, stretches=0, cuts=0, rebuilt=0, dropped=0, islands=0, crossed=0)
    coast = _Coast(domain.loops) if domain.loops else None
    for pc in free_pieces(domain, list(fixed)):
        sm = _sample(pc, h)
        if sm is None:
            short.append(pc["V"])
            continue
        counts["pieces"] += 1
        N = sm["N"]
        cn = _corners(sm)
        counts["corners"] += len(cn)
        if sm["closed"]:                                    # open the loop at a corner, else at its sharpest node
            k0 = int(cn[0]) if len(cn) else int(np.argmax(np.abs(sm["tau"])))
            for key in ("R", "seg", "t"):
                x = np.roll(sm[key], -SUB * k0, axis=0)
                sm[key] = np.concatenate([x, x[:1]])
            cn = np.sort((cn - k0) % N)
            cn = cn[cn > 0]
        bnd = np.r_[0, cn, N]
        s0 = len(segs)
        for a, b in zip(bnd[:-1], bnd[1:]):                 # corner to corner: filtered and smoothed on its own
            sl = slice(SUB * int(a), SUB * int(b) + 1)
            R = sm["R"][sl]
            S = _gauss(R, NORMAL_FILTER * SUB, False)
            segs.append(dict(F=_gauss(R, FILTER * SUB, False), NRM=_unit(np.cross(S, np.gradient(S, axis=0))),
                             seg=sm["seg"][sl], t=sm["t"][sl], loop=sm["loop"], N=int(b - a), R=R,
                             wedge=[np.inf, np.inf]))
        js = list(range(s0, len(segs)))                     # the corner wedges: consecutive segments of the piece
        for ja, jb in list(zip(js[:-1], js[1:])) + ([(js[-1], js[0])] if sm["closed"] else []):
            if min(segs[ja]["N"], segs[jb]["N"]) > 2 * GAP:
                w = _wedge(segs[ja], segs[jb], h)
                segs[ja]["wedge"][1] = min(segs[ja]["wedge"][1], w)
                segs[jb]["wedge"][0] = min(segs[jb]["wedge"][0], w)
    # what the normals must keep clear of (LFS): the filtered coast, the raw coast too short for a band, fixed blocks
    tree = cKDTree(np.vstack([sg["F"] for sg in segs] + short + [f.grid.points for f in fixed]))
    bands = []
    for sg in segs:
        a, b = GAP, sg["N"] - GAP
        if b - a < STRETCH_MIN:
            continue
        counts["stretches"] += 1
        L = _limits(sg, h, n_max, tree)
        cap = np.full(sg["N"], n_max, dtype=np.int64)
        cap[a] = min(cap[a], sg["wedge"][0])                # the corner wedges bound the end cells (C36)
        cap[b - 1] = min(cap[b - 1], sg["wedge"][1])
        for _ in range(REBUILD + 1):
            plan, nc, lim = _plan(a, b, L, n_max, cap)
            built, again = [], False
            for pa, pb in plan:
                res, badc, info = _build(sg, pa, pb, nc[pa:pb], h)
                if res is None:
                    cap[badc[:, 0]] = np.minimum(cap[badc[:, 0]], badc[:, 1])
                    again = True
                else:
                    built.append((pa, pb, res, info))
            if not again:
                break
            counts["rebuilt"] += 1
        else:
            counts["dropped"] += len(plan) - len(built)
        counts["cuts"] += max(0, len(plan) - 1)
        for pa, pb, (grid, cmask, run), info in built:
            if coast.islands_inside(grid.points[grid.outline], {int(sg["loop"])}):     # as a strait block (C39)
                counts["islands"] += 1
                continue
            lo, la = xyz_to_lonlat(grid.points[[run[0][0], run[0][-1]]])
            rec = dict(kind="band", loop=int(sg["loop"]), nodes=[int(pa), int(pb)], **lim,
                       plateau_min=int(L["plateau"][pa:pb + 1].min()), lfs_rows_min=int(L["lfs"][pa:pb + 1].min()),
                       bay_min=float(L["bay"][pa:pb].min()), wedge=[float(x) for x in sg["wedge"]],
                       ends_lonlat=np.round(np.column_stack([lo, la]), 4).tolist(), status="accepted", **info,
                       fixed_index=len(fixed) + len(bands))
            bands.append(FixedBlock(grid, cmask, [run], rec))
    for _ in range(len(bands)):                             # a band whose outline still crosses a coast loop's
        x = _crossing(domain, h, fixed, bands) if bands else set()   # chord or another band is dropped (the fill
        if not x:                                           # needs simple loops; C36)
            break
        counts["crossed"] += len(x)
        bands = [b for i, b in enumerate(bands) if i not in x]
        for i, b in enumerate(bands):
            b.record["fixed_index"] = len(fixed) + i
    summary = dict(counts, bands=len(bands), cells=int(sum(len(b.grid.quads) for b in bands)),
                   coast_nodes=int(sum(int(b.coast.sum()) for b in bands)))
    return bands, summary
