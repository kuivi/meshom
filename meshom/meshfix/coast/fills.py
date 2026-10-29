"""Fills of a coast cavity with a re-spaced coast side (decisions B143).

D's patterns.fills (B135) changes a coast side's edge count by deleting the node nearest to its neighbours' chord or
by splitting the longest edge at its midpoint, and D's relaxation keeps coast nodes fixed: a merged edge doubles,
|ln s/h| rises by ~0.69, and none of ~6 000 coast adjustments was accepted. Here the whole coast side (a run of k
coast edges between two cavity corners) is re-spaced instead: its interior nodes are replaced by target - 1 new
coast nodes at equal steps of integral ds / h along the side's polyline, each projected onto the reference coast.
A side may change its count only while |ln(target / k)| <= adj_lnmax (with the collar's coast nodes sliding in the
relaxation the change spreads further). The interior of the cavity is D's: regular grid, ladders, splits
(patterns.patch_plans, build_patch, _split: imported, not changed).
"""
import numpy as np

from meshom.geo import R_EARTH_M
from meshom.meshfix.cavity.patterns import Fill, _split, _xs, build_patch, patch_plans, sides


def respace(F: Fill, side: list, target: int, h, ref, reach_h: float = 1.5) -> list | None:
    """The coast side with `target` edges: ends kept, interior nodes new (F.coast), at equal steps of ds / h along
    the side's polyline, projected onto the reference coast segments of the side's own shore."""
    if target < 1:
        return None
    X = F.arr(side)
    mid = X[1:] + X[:-1]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    hm = h(mid)
    ds = np.linalg.norm(np.diff(X, axis=0), axis=1) * R_EARTH_M / hm
    cum = np.r_[0.0, np.cumsum(ds)]
    if cum[-1] <= 0:
        return None
    u = cum / cum[-1]
    segs = set()
    for x, hh in zip(X, np.r_[hm[:1], hm]):
        segs.update(ref.allowed(x, reach_h * float(hh), K=12).tolist())
    segs = np.array(sorted(segs), dtype=np.int64)
    out = [side[0]]
    for j in range(1, target):
        t = j / target
        i = min(int(np.searchsorted(u, t, side="right")) - 1, len(X) - 2)
        f = (t - u[i]) / max(u[i + 1] - u[i], 1e-300)
        x = (1 - f) * X[i] + f * X[i + 1]
        x /= np.linalg.norm(x)
        x = ref.project(x[None], segs[None])[0]
        nid = F.add(x)
        F.coast.append(nid)
        out.append(nid)
    out.append(side[-1])
    return out


def coast_run(side: list, is_bedge) -> tuple | None:
    """(i0, i1): node index range of the longest run of consecutive coast edges of the side, or None."""
    best, i = None, 0
    n = len(side) - 1
    while i < n:
        if is_bedge(side[i], side[i + 1]):
            j = i
            while j < n and is_bedge(side[j], side[j + 1]):
                j += 1
            if best is None or j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return best


def n_ideal(F: Fill, side: list, h) -> float:
    """integral ds / h along the side's polyline (the h-ideal edge count of the side)."""
    X = F.arr(side)
    mid = X[1:] + X[:-1]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    return float((np.linalg.norm(np.diff(X, axis=0), axis=1) * R_EARTH_M / h(mid)).sum())


def coast_fills(P: np.ndarray, lp: tuple, cr: list, h, is_bedge, ref, layouts=None, max_k: int = 64,
                split: bool = True, max_adj: int = 6, adj_lnmax: float = 0.2, same: bool = True,
                bias: float = 0.0) -> list:
    """Every pattern of the cavity with its coast sides re-spaced. A coast side of k edges may take (B143): the
    count of the opposite side (the regular fill), the h-ideal count n* = round(exp(-bias) integral ds / h) (bias:
    the zone's median ln s/h), or keep k (same=True: the coast then only slides in the relaxation); a new count t
    only while |ln(n* / t)| <= adj_lnmax and |t - k| <= max_adj. Every combination of the coast sides' counts per
    rotation, then D's plans (regular, ladders, splits). List of (name, X new nodes, cells (-1 - j for X[j]), new
    coast refs)."""
    S = sides(lp, cr)
    e = [len(s) - 1 for s in S]
    runs = [coast_run(s, is_bedge) for s in S]           # the longest coast run of each side (node index range)
    cs = [r is not None for r in runs]
    out, seen = [], set()
    if not any(cs):
        return out
    F00 = Fill(P)
    # h-ideal count of a side: its non-coast edges as they are plus integral ds / h along its coast run
    nrun = [n_ideal(F00, S[i][r[0]: r[1] + 1], h) * np.exp(-bias) if r else 0.0 for i, r in enumerate(runs)]
    rlen = [r[1] - r[0] if r else 0 for r in runs]
    nstar = [e[i] - rlen[i] + nrun[i] for i in range(4)]

    def emit(name, F, cells):
        key = frozenset(tuple(sorted(c)) for c in cells if all(v >= 0 for v in c)) | {len(F.X)}
        if key in seen:
            return
        seen.add(key)
        out.append((name, _xs(F), cells, list(F.coast)))

    def targets(i: int, opp: int) -> list:
        k = e[i]
        out_t = [k] if same else []
        for t in (opp, int(round(nstar[i]))):
            tr = t - k + rlen[i]                          # the coast run's new count
            if tr >= 1 and t != k and t not in out_t and abs(t - k) <= max_adj and \
                    abs(np.log(max(nrun[i], 1e-9) / tr)) <= adj_lnmax:
                out_t.append(t)
        return out_t

    import itertools

    for r in range(4):
        q = [(r + j) % 4 for j in range(4)]                # bottom, right, top, left (loop sides)
        sd = {"b": S[q[0]], "r": S[q[1]], "t": S[q[2]][::-1], "l": S[q[3]][::-1]}
        idx = {"b": q[0], "r": q[1], "t": q[2], "l": q[3]}
        opp = {"b": "t", "t": "b", "r": "l", "l": "r"}
        keys = [k for k in "brtl" if cs[idx[k]]]
        opts = [targets(idx[k], e[idx[opp[k]]]) for k in keys]
        for combo in itertools.product(*opts):
            ad = {k: t for k, t in zip(keys, combo) if t != e[idx[k]]}
            F0 = Fill(P)
            sl = dict(sd)
            ok = True
            for key, tgt in ad.items():
                sid = sl[key]
                i0, i1 = coast_run(sid, is_bedge)
                new = respace(F0, sid[i0: i1 + 1], tgt - (len(sid) - 1) + (i1 - i0), h, ref)
                if new is None:
                    ok = False
                    break
                sl[key] = sid[:i0] + new + sid[i1 + 1:]
            if not ok:
                continue
            bottom, right, top, left = sl["b"], sl["r"], sl["t"], sl["l"]
            a, c, dR, bL = len(bottom) - 1, len(top) - 1, len(right) - 1, len(left) - 1
            if a < c or (a == c and r != 0 and dR == bL) or a - c > max_k:
                continue
            pre = ("adj" + "".join(f"{k}{t}" for k, t in sorted(ad.items()))) if ad else "slide"
            if dR == bL:
                top_coast = cs[idx["t"]] and runs[idx["t"]] == (0, e[idx["t"]])
                for name, nrow, bases in patch_plans(F0, bottom, right, top, left, h, layouts, tip_top=top_coast):
                    F = Fill(P)
                    F.X, F.coast = list(F0.X), list(F0.coast)
                    cells = build_patch(F, bottom, right, top, left, nrow, bases)
                    emit(f"{pre}_{'ladder' if a > c else 'regular'}-{name}_r{r}_k{a - c}", F, cells)
            elif split and a > c and dR > bL and dR - bL <= max_k:
                for name, F, cells in _split(F0, bottom, right, top, left, h, layouts, r):
                    emit(f"{pre}-{name}", F, cells)
    return out
