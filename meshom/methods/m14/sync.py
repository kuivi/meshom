"""m14 frame and count synchronisation of the blocks before the stitch (decisions B54-B56).

Block graph (`corridors`): stitch_mix.partition of the fill (water minus the blocks); a corridor is a region of class
"corridor" (two blocks) or, in conforming mode, "coast" (a block and a coast loop; a channel between two coast loops is
not one). Side A = the block with the lower id, side B = the other block or the coast. Per edge of side A: the signed
4-fold rotation phi_e in (-45, 45] deg from A's outline edge to the nearest edge of side B (tangent plane at the mean
of the two midpoints; stitch_mix._rotation without the mean), smoothed by a 4-fold running mean over +-6 edges within
each contiguous piece of the run (SMOOTH = 6). Along counts nA, nB: the outline edges of a
side classified "along" (stitch_mix._edge_classes), only when the side is one contiguous lattice path of a block.
L = (edges of A + edges of B) / 2.

Frames (`solve_frames`, B54). Turning block b's lattice by psi_b (CCW seen from outside) changes phi_e to
phi_e + psi_b - psi_a. Minimised: sum over the corridor edges of |wrap90(phi_e + psi_b - psi_a)| (L1, weight 1 per
edge; the forced rotational dislocations 2 |sin(phi/2)| ~ |phi|). Spectral start: per connected component of the block
graph the top eigenvector v of the Hermitian W_ab = sum_e e^(4 i phi_e) (it maximises Re sum conj(z_a) W_ab z_b,
z = e^(4 i psi)), psi = arg(v) / 4. Then an LP over psi with each edge's 90-deg branch fixed at the current psi
(t_e >= +-(phi_e + m_e + psi_b - psi_a), minimise sum t_e; scipy linprog, HiGHS), the branches re-fixed, <= 3 rounds,
from the spectral start, from psi = 0 and from 8 random starts (seed 0: the fixed-branch LP stops in the basin of its
start; on band s1 the spectral start ends at a cost 15 % above the best of 40 random starts, 8 starts reach it within
1 %); the best by the wrapped cost is kept and wrapped to (-45, 45] (a quarter turn is the same lattice).
Conforming mode (B56): every coast loop is one node with a fixed frame psi = 0 whose "lattice" is the coast tangent, so
a block beside a coast turns to it. A component without a coast is gauge-free: one node is held during the LP and the
component is turned back so that the circular mean of 4 psi is 0 (no turn where nothing asks for one); a block without
corridors stays at 0. Cut mode: no coast term (the coast is a staircase through the cells).

Counts (`solve_counts`, B55). Scaling block b's cells by e^sigma_b multiplies its along counts by e^-sigma_b. Minimised:
sum over the block-block corridors with both counts of L (ln nA - sigma_a - ln nB + sigma_b)^2 + mu sum sigma^2 (a ridge
mu = 1e-3 x the total L fixes the free common scale), subject to |sigma_b| <= s_b = f_max - F_b (F_b = the half-range
of the block map's lambda: the size budget B12 leaves; scipy lsq_linear). The counts are taken on the turned blocks
(the second partition pass): a turn changes a run's along count by cos(turn), as much as the whole budget s_b.

Applied (`sync`): the block map is conformal, so the synchronised block is its map times e^(-i psi_b - sigma_b) and the
lattice is re-extracted from it (blockmap.extract: 4 x 4 offsets, trim, clean, the B17 overlap check, size sweeps; no
new solve). A turned block that fails (no grid, an overlap, fewer than min_cells or min_inradius) is tried once more with
its cells e^(-s_b) (the most cells its size budget allows; the count stage then starts from sigma_b = -s_b), and keeps
its first-pass grid when that fails too (a "fallback", no turn); a failed count shift keeps the turned grid. The record (m14.json "sync"): per block psi_deg, sigma, s_max; per corridor the objects
(region indices, "coast"), L, phi before / after (4-fold corridor means), the forced rotational dislocations before /
after, the along counts and b_par = the along mismatch after; before / after statistics per kind (block-block, coast).

Direct join (`merge`, B61; option merge, default off), run before the frames on the
first-pass blocks: a block-block corridor with b = 0 (Burgers vector of its two runs when both are lattice paths,
stitch_mix.burgers_vector) or a drift of at least merge_drift dislocations per 10 cells (drift_per10: the rotational
count left after the best rigid turn of the pair alone) is a candidate, the most drift dislocations first; the two
blocks' regions (or merged groups) with the corridor between them must pass place.union_region (a closed disc, F <=
f_max, h |grad lambda| <= grad_max) and give a grid from one map over the union (blockmap.region_grid). The merged
block is then one node of the block graph above. Record: m14.json "merge" (every candidate with its b, drift, union
F, gradient and result; the merged groups).
"""
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, lsq_linear
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from meshom.methods.m14.stitch_mix import _edge_classes, _runs, _unit, _wrap90, burgers_vector, partition

SMOOTH = 6              # phi smoothed over +-6 edges
LP_ROUNDS = 3           # LP rounds with the branches re-fixed
RANDOM_STARTS = 8       # LP starts besides the spectral one and psi = 0: uniform turns in (-45, 45], seed 0
RIDGE = 1e-3            # count shifts: ridge weight per unit of corridor length (gauge of the common scale)
COAST = -1              # side B of a coast corridor: the fixed coast frame


@dataclass
class Corridor:
    a: int                  # side A: block index (into the grids passed)
    b: int                  # side B: block index, or COAST
    phi: np.ndarray         # (E,) deg per edge of side A, signed 4-fold rotation A -> B, smoothed
    L: float                # (edges of A + edges of B) / 2
    na: int | None          # along edges of side A (None: not one contiguous lattice path)
    nb: int | None
    region: int = -1        # partition region id
    contig: bool = True     # both sides one contiguous run
    bvec: tuple | None = None   # Burgers vector (b_par, b_perp) of the two runs (block-block, both lattice paths)


def _pieces(idx: np.ndarray, K: int) -> list:
    """Sorted edge indices of one loop (K edges) split into cyclic runs of consecutive edges."""
    pcs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    if len(pcs) > 1 and pcs[0][0] == 0 and pcs[-1][-1] == K - 1:
        pcs[0] = np.r_[pcs[-1], pcs[0]]
        pcs.pop()
    return pcs


def _signed_rot(ea: np.ndarray, ma: np.ndarray, eb: np.ndarray, mb: np.ndarray) -> np.ndarray:
    """Per edge (vector ea, unit midpoint ma) of side A: the signed 4-fold angle (deg) to the nearest edge of side B."""
    _, j = cKDTree(mb).query(ma)
    nrm = _unit(ma + mb[j])
    u = ea - np.einsum("ij,ij->i", ea, nrm)[:, None] * nrm
    v = eb[j] - np.einsum("ij,ij->i", eb[j], nrm)[:, None] * nrm
    return _wrap90(np.degrees(np.arctan2(np.einsum("ij,ij->i", np.cross(u, v), nrm), np.einsum("ij,ij->i", u, v))))


def _smooth4(phi: np.ndarray, k: int) -> np.ndarray:
    """4-fold running mean (deg) over +-k entries."""
    if k <= 0 or len(phi) < 2:
        return phi
    cz = np.r_[0, np.cumsum(np.exp(4j * np.radians(phi)))]
    i = np.arange(len(phi))
    lo, hi = np.clip(i - k, 0, len(phi)), np.clip(i + k + 1, 0, len(phi))
    return np.degrees(np.angle(cz[hi] - cz[lo])) / 4.0


def corridors(coast: list, grids: list, h, o: dict, coast_term: bool) -> tuple[list, object]:
    """The block graph (module docstring): (corridors, the partition)."""
    part = partition(coast, [g.points[g.outline] for g in grids], h, o)
    P, B = part.bg.points, part.B
    offs = np.r_[0, np.cumsum([len(lp) for lp in part.loops])]
    out = []
    for rec in part.regions:
        if rec["cls"] not in ("corridor", "coast"):
            continue
        oa, ob = rec["objs"]                                            # blocks 0..B-1 first, so oa is a block
        if oa >= B or (ob >= B and not coast_term):
            continue
        runs = _runs(part, rec["id"])
        if oa not in runs or ob not in runs:
            continue
        sides = []
        for li, idx in (runs[oa], runs[ob]):
            K = len(part.loops[li])
            pcs = [offs[li] + np.r_[p, (p[-1] + 1) % K] for p in _pieces(idx, K)]   # node polylines, loop order
            sides.append((pcs, len(idx) < K))
        e = [np.vstack([P[p[1:]] - P[p[:-1]] for p in pcs]) for pcs, _ in sides]
        m = [np.vstack([_unit(P[p[1:]] + P[p[:-1]]) for p in pcs]) for pcs, _ in sides]
        raw = _signed_rot(e[0], m[0], e[1], m[1])
        n_a = np.cumsum([0] + [len(p) - 1 for p in sides[0][0]])
        phi = np.concatenate([_smooth4(raw[n_a[i]:n_a[i + 1]], SMOOTH) for i in range(len(n_a) - 1)])
        cls = [_edge_classes(P[pcs[0]]) if obj < B and len(pcs) == 1 and partial else None
               for (pcs, partial), obj in zip(sides, (oa, ob))]
        counts = [int((c == 0).sum()) if c is not None else None for c in cls]
        bvec = burgers_vector(cls[0], cls[1][::-1]) if cls[0] is not None and cls[1] is not None else None
        out.append(Corridor(int(oa), int(ob) if ob < B else COAST, phi, 0.5 * (len(e[0]) + len(e[1])),
                            counts[0], counts[1], int(rec["id"]),
                            all(len(pcs) == 1 and partial for pcs, partial in sides), bvec))
    return out, part


def _edges(cs: list, nb: int):
    """Flattened corridor edges: (a, b node ids with the coast = node nb, phi deg, corridor of each edge)."""
    a = np.concatenate([np.full(len(c.phi), c.a) for c in cs]).astype(np.int64)
    b = np.concatenate([np.full(len(c.phi), nb if c.b == COAST else c.b) for c in cs]).astype(np.int64)
    phi = np.concatenate([c.phi for c in cs])
    ci = np.concatenate([np.full(len(c.phi), i) for i, c in enumerate(cs)]).astype(np.int64)
    return a, b, phi, ci


def solve_frames(cs: list, nb: int, rounds: int = LP_ROUNDS, n_random: int = RANDOM_STARTS) -> tuple[np.ndarray, dict]:
    """Frame turns psi (nb,) in degrees (module docstring) and a small log."""
    n = nb + 1                                                         # node nb: the coast (psi = 0, fixed)
    log = {"lp_rounds": 0}
    if not cs:
        return np.zeros(nb), log
    a, b, phi, _ = _edges(cs, nb)
    pa = np.array([c.a for c in cs])
    pb = np.array([nb if c.b == COAST else c.b for c in cs])
    ncomp, lab = connected_components(coo_matrix((np.ones(len(cs)), (pa, pb)), shape=(n, n)), directed=False)
    Wm = np.zeros((n, n), complex)
    np.add.at(Wm, (a, b), np.exp(4j * np.radians(phi)))
    Wm = Wm + Wm.conj().T
    psi = np.zeros(n)
    deg = np.bincount(np.r_[pa, pb], minlength=n)
    anchor = deg == 0                                                  # held at 0: the coast, a block without corridors
    anchor[nb] = True
    held = anchor.copy()
    free = np.zeros(ncomp, dtype=bool)                                 # gauge-free components (no anchor)
    for k in range(ncomp):
        nodes = np.flatnonzero(lab == k)
        if len(nodes) < 2:
            continue
        v = np.linalg.eigh(Wm[np.ix_(nodes, nodes)])[1][:, -1]
        p = np.degrees(np.angle(v)) / 4.0
        if anchor[nodes].any():
            p -= p[np.flatnonzero(anchor[nodes])[0]]
            p[anchor[nodes]] = 0.0
        else:
            free[k] = True
            p -= np.degrees(np.angle(np.exp(4j * np.radians(p)).sum())) / 4.0
            held[nodes[np.argmax(deg[nodes])]] = True                  # the gauge during the LP
        psi[nodes] = _wrap90(p)

    def cost(ps):
        return float(np.abs(_wrap90(phi + ps[b] - ps[a])).sum())

    E = len(phi)
    rows = np.repeat(np.arange(2 * E), 3)
    cols = np.c_[b, a, n + np.arange(E), b, a, n + np.arange(E)].ravel()
    vals = np.tile([1.0, -1.0, -1.0, -1.0, 1.0, -1.0], E)
    A = coo_matrix((vals, (rows, cols)), shape=(2 * E, n + E)).tocsr()
    obj = np.r_[np.zeros(n), np.ones(E)]
    log["cost_zero"], log["cost_spectral"] = cost(np.zeros(n)), cost(psi)
    best = min((log["cost_zero"], np.zeros(n)), (log["cost_spectral"], psi.copy()), key=lambda x: x[0])
    rng = np.random.default_rng(0)
    starts = [psi.copy(), np.zeros(n)] + [np.where(anchor, 0.0, np.r_[rng.uniform(-45.0, 45.0, nb), 0.0])
                                          for _ in range(n_random)]
    for start in starts:                                               # the LP from every start, the best kept
        ps = start
        for _ in range(rounds):
            p0 = phi + ps[b] - ps[a]
            c = phi + (_wrap90(p0) - p0)                                # this round's branches
            bounds = [(x, x) if hd else (None, None) for x, hd in zip(ps, held)] + [(0, None)] * E
            res = linprog(obj, A_ub=A, b_ub=np.c_[-c, c].ravel(), bounds=bounds, method="highs")
            log["lp_rounds"] += 1
            if not res.success:
                log["lp_status"] = res.message
                break
            new = res.x[:n]
            q = phi + new[b] - new[a]
            same = np.array_equal(np.round((_wrap90(q) - q) / 90.0), np.round((c - phi) / 90.0))
            ps = new
            if cost(ps) < best[0]:
                best = (cost(ps), ps.copy())
            if same:
                break
    psi = best[1]
    log["cost_lp"] = best[0]
    for k in np.flatnonzero(free):                                     # gauge-free components: circular mean 0
        nodes = np.flatnonzero(lab == k)
        psi[nodes] -= np.degrees(np.angle(np.exp(4j * np.radians(psi[nodes])).sum())) / 4.0
    return _wrap90(psi[:nb]), log                                        # a quarter turn is the same lattice


def solve_counts(cs: list, s_max: np.ndarray, ridge: float = RIDGE, sig0: np.ndarray | None = None) -> np.ndarray:
    """Count shifts sigma (nb,), |sigma_b| <= s_max[b] (module docstring). sig0: the shift the counted blocks already
    carry (their counts are taken with it); returned is the total shift."""
    nb = len(s_max)
    sig0 = np.zeros(nb) if sig0 is None else np.asarray(sig0, dtype=float)
    rows = [c for c in cs if c.b != COAST and c.na and c.nb]
    if not rows:
        return sig0.copy()
    A = np.zeros((len(rows) + nb, nb))
    y = np.zeros(len(rows) + nb)
    for r, c in enumerate(rows):
        w = np.sqrt(c.L)
        A[r, c.a], A[r, c.b] = -w, w
        y[r] = -w * np.log(c.na / c.nb)
    mu = np.sqrt(ridge * sum(c.L for c in rows))
    A[len(rows):] = mu * np.eye(nb)                                    # the ridge on the total shift sig0 + d
    y[len(rows):] = -mu * sig0
    s = np.maximum(np.asarray(s_max, dtype=float), 0.0)
    d = lsq_linear(A, y, bounds=(-s - sig0 - 1e-15, s - sig0 + 1e-15)).x
    return (sig0 + d).clip(-s, s)


def stats(cs: list, psi: np.ndarray | None = None, sigma: np.ndarray | None = None) -> dict:
    """Per kind (block-block "blocks", block-coast "coast"): edges, forced rotational dislocations sum 2|sin(phi/2)|
    per 10 edges, share of the corridor length whose 4-fold mean |phi| < 25 deg (the strip's rule) and of the edges
    with |phi| < 10 deg; along mismatch sum |nA e^-sigma_a - nB e^-sigma_b| per 10 cells of L (block-block); forced =
    rotational + mismatch. psi (deg) and sigma as predicted on these corridors (None: as they are)."""
    out = {}
    for kind, sel in (("blocks", [c for c in cs if c.b != COAST]), ("coast", [c for c in cs if c.b == COAST])):
        if not sel:
            continue
        rot, n25, n10, ne = 0.0, 0.0, 0.0, 0
        for c in sel:
            p = c.phi if psi is None else _wrap90(c.phi + (psi[c.b] if c.b != COAST else 0.0) - psi[c.a])
            rot += float(2.0 * np.abs(np.sin(np.radians(p) / 2.0)).sum())
            mean = abs(np.degrees(np.angle(np.exp(4j * np.radians(p)).mean())) / 4.0)
            n25 += len(p) * (mean < 25.0)
            n10 += float((np.abs(p) < 10.0).sum())
            ne += len(p)
        d = {"corridors": len(sel), "edges": ne, "rot_per10": 10.0 * rot / ne, "share_lt25": n25 / ne,
             "share_edges_lt10": n10 / ne}
        cnt = [c for c in sel if c.na and c.nb]
        if kind == "blocks" and cnt:
            sg = np.zeros(max(max(c.a, c.b) for c in cnt) + 1) if sigma is None else sigma
            mis = sum(abs(c.na * np.exp(-sg[c.a]) - c.nb * np.exp(-sg[c.b])) for c in cnt)
            d.update(counted=len(cnt), mis_per10=10.0 * float(mis) / sum(c.L for c in cnt))
            d["forced_per10"] = d["rot_per10"] + d["mis_per10"]
        out[kind] = d
    return out


def _mean4(p: np.ndarray) -> float:
    return float(np.degrees(np.angle(np.exp(4j * np.radians(p)).mean())) / 4.0)


def extract_ok(g0, h, o: dict, psi_deg: float, sigma: float, why: list | None = None):
    """g0's map re-extracted turned by psi_deg and scaled by e^sigma; None when it fails (no grid, an overlap, fewer
    than min_cells cells or an inradius below min_inradius; the reason appended to `why`)."""
    from meshom.methods.m14.blockmap import extract

    g = extract(g0.map, h, o, float(np.radians(psi_deg)), float(sigma))
    bad = ("no grid" if g is None else "overlap" if g.info.get("overlap") else
           "small" if g.info["n_cells"] < o["min_cells"] or g.info["inradius_cells"] < o["min_inradius"] else None)
    if bad and why is not None:
        why.append(bad)
    return None if bad else g


def sync(coast: list, grids: list, h, o: dict, ids: list | None = None) -> dict:
    """Frames, then counts (module docstring), on the kept grids (replaced in place). coast = the resampled coast
    loops (none in cut mode); ids = the region index of each grid (for the record). Returns the m14.json record."""
    t0 = time.perf_counter()
    nb = len(grids)
    ids = list(range(nb)) if ids is None else list(ids)
    coast_term = bool(coast) and not o["cut"]
    base = list(grids)
    rec = {"coast_term": coast_term}
    cs1, part1 = corridors(coast, base, h, o, coast_term)
    psi, flog = solve_frames(cs1, nb)
    rec.update(flog)
    rec["pass1"] = {"before": stats(cs1), "predicted": stats(cs1, psi)}
    t1 = time.perf_counter()
    s_max = np.maximum(np.array([o["f_max"] - g.info.get("F_map", o["f_max"]) for g in base]), 0.0)
    sig0 = np.zeros(nb)                                                # the shift a turned block already carries
    fall, shrunk, why = [], [], []
    for k in np.flatnonzero(psi):
        g = extract_ok(base[k], h, o, psi[k], 0.0, why)
        if g is None and s_max[k] > 0:                                 # too small turned: the most cells allowed
            g = extract_ok(base[k], h, o, psi[k], -s_max[k], why)
            if g is not None:
                sig0[k] = -s_max[k]
                shrunk.append(int(k))
        if g is None:
            fall.append(int(k))
            psi[k] = 0.0
        else:
            grids[k] = g
    t2 = time.perf_counter()
    cs2, _ = corridors(coast, grids, h, o, coast_term)                   # the second pass: counts on turned blocks
    sigma = solve_counts(cs2, s_max, sig0=sig0)
    rec["pass2"] = {"turned": stats(cs2), "predicted": stats(cs2, np.zeros(nb), sigma - sig0)}
    t3 = time.perf_counter()
    fall2 = []
    for k in np.flatnonzero(sigma != sig0):
        g = extract_ok(base[k], h, o, psi[k], sigma[k], why)
        if g is None:                                                   # keeps the turned grid
            fall2.append(int(k))
            sigma[k] = sig0[k]
        else:
            grids[k] = g
    for k, g in enumerate(grids):
        g.info.update(sync_psi_deg=float(psi[k]), sync_sigma=float(sigma[k]), sync_s_max=float(max(s_max[k], 0.0)))
    rec["fallback_turn"] = [ids[k] for k in fall]
    rec["turn_shrunk"] = [ids[k] for k in shrunk]
    rec["fallback_count"] = [ids[k] for k in fall2]
    rec["fallback_why"] = why
    rec["corridors"] = []
    by_pair = {}                                                       # pass-2 corridors per pair, in order
    for c in cs2:
        by_pair.setdefault((c.a, c.b), []).append(c)
    for c in cs1:
        p1 = _wrap90(c.phi + (psi[c.b] if c.b != COAST else 0.0) - psi[c.a])
        r = {"a": ids[c.a], "b": "coast" if c.b == COAST else ids[c.b], "L": round(c.L, 1),
             "phi_deg": round(_mean4(c.phi), 2), "phi_pred_deg": round(_mean4(p1), 2),
             "rot_disl": round(float(2.0 * np.abs(np.sin(np.radians(c.phi) / 2.0)).sum()), 2),
             "rot_disl_pred": round(float(2.0 * np.abs(np.sin(np.radians(p1) / 2.0)).sum()), 2),
             "nA_along": c.na, "nB_along": c.nb}
        c2 = by_pair[(c.a, c.b)].pop(0) if by_pair.get((c.a, c.b)) else None
        if c2 is not None:
            r.update(phi_turned_deg=round(_mean4(c2.phi), 2),
                     rot_disl_turned=round(float(2.0 * np.abs(np.sin(np.radians(c2.phi) / 2.0)).sum()), 2),
                     nA_along_turned=c2.na, nB_along_turned=c2.nb)
            if c.b != COAST and c2.na and c2.nb:
                d = sigma - sig0                                        # pass-2 counts carry sig0
                r["b_par"] = round(float(c2.na * np.exp(-d[c.a]) - c2.nb * np.exp(-d[c.b])), 2)
        rec["corridors"].append(r)
    rec["timings"] = {"pass1_s": t1 - t0, "turn_s": t2 - t1, "pass2_s": t3 - t2, "count_s": time.perf_counter() - t3}
    return rec


def drift_per10(c: Corridor) -> float:
    """The corridor's forced rotational dislocations per 10 edges left after the best rigid turn of this pair alone
    (the "tree" bound per corridor): sum 2 |sin(phi_e - mean phi) / 2| over its edges, phi's 4-fold mean
    removed. On uniform's cube-face blocks it is the drift of the two outline lattices."""
    if len(c.phi) == 0:
        return 0.0
    d = _wrap90(c.phi - _mean4(c.phi))
    return float(20.0 * np.abs(np.sin(np.radians(d) / 2.0)).sum() / len(c.phi))


def is_candidate(c: Corridor, o: dict) -> bool:
    """A block-block corridor the direct join tries (B61): b = 0 (the two lattices already match, B35) or a drift of
    at least merge_drift dislocations per 10 cells (drift_per10; 0: every corridor)."""
    return c.b != COAST and (c.bvec == (0, 0) or drift_per10(c) >= o["merge_drift"])


def _bad_grid(g, o: dict):
    return ("no grid" if g is None else "overlap" if g.info.get("overlap") else
            "small" if g.info["n_cells"] < o["min_cells"] or g.info["inradius_cells"] < o["min_inradius"] else None)


def merge(coast: list, fit, regions: list, grids: list, kept: list, h, o: dict,
          fixed_pts: np.ndarray | None = None) -> tuple:
    """The direct join (B35, B61): before the frames, every block-block corridor of the first-pass
    blocks with b = 0 or a drift of at least merge_drift dislocations per 10 cells (drift_per10) is a candidate, the
    most drift dislocations (drift x L) first; a candidate whose two blocks' regions (a merged group's union so far)
    pass place.union_region (with the fixed blocks' nodes `fixed_pts`: the union keeps their block_gap, B59) gets one
    map over the union (blockmap.region_grid) and becomes one block, which then is one node of the frame and count
    synchronisation's block graph. Returns (regions, grids, kept) with every merged group in its first member's place
    and the others removed, and the m14.json record."""
    from meshom.methods.m14.blockmap import region_grid
    from meshom.methods.m14.place import union_region

    t0 = time.perf_counter()
    nb = len(kept)
    cs, _ = corridors(coast, [grids[k] for k in kept], h, o, False)
    cand = [(c, drift_per10(c)) for c in cs if c.b != COAST]
    cand.sort(key=lambda x: -x[1] * x[0].L)
    root = list(range(nb))

    def find(i):
        while root[i] != i:
            i = root[i]
        return i

    reg = {i: regions[k] for i, k in enumerate(kept)}
    grd = {i: grids[k] for i, k in enumerate(kept)}
    members = {i: [k] for i, k in enumerate(kept)}
    rec = {"candidates": []}
    for c, dr in cand:
        r = {"a": kept[c.a], "b": kept[c.b], "L": round(c.L, 1), "b_vec": c.bvec, "drift_per10": round(dr, 3),
             "region": c.region}
        rec["candidates"].append(r)
        if not is_candidate(c, o):
            r["result"] = "cheap"
            continue
        ga, gb = find(c.a), find(c.b)
        if ga == gb:
            r["result"] = "one block"
            continue
        others = [reg[g].tris for g in {find(i) for i in range(nb)} - {ga, gb}]
        u, info = union_region(fit, reg[ga].tris, reg[gb].tris, others, o, fixed_pts)
        r.update({k: round(v, 4) if isinstance(v, float) else v for k, v in info.items()})
        if u is not None:
            g = region_grid(fit, u.tris, h, o)
            info["why"] = _bad_grid(g, o)
        if u is None or info.get("why"):
            r["result"] = info["why"]
            continue
        u.seed, u.info = reg[ga].seed, {"merged": members[ga] + members[gb]}
        root[gb] = ga
        reg[ga], grd[ga] = u, g
        members[ga] += members[gb]
        r.update(result="merged", cells=round(u.cells, 1), B=round(u.B, 4), n_cells=int(g.info["n_cells"]))
    gone = {kept[i] for i in range(nb) if find(i) != i}
    new_id = {}
    regions2, grids2 = [], []
    for k in range(len(regions)):
        if k in gone:
            continue
        new_id[k] = len(regions2)
        i = kept.index(k) if k in kept else None
        regions2.append(reg[i] if i is not None else regions[k])
        grids2.append(grd[i] if i is not None else grids[k])
    kept2 = [new_id[k] for k in kept if k not in gone]
    rec.update(n_candidates=len(cand), n_merged=len(gone),
               groups=[members[i] for i in range(nb) if find(i) == i and len(members[i]) > 1],
               region_ids_after=new_id, merge_s=time.perf_counter() - t0)
    return regions2, grids2, kept2, rec
