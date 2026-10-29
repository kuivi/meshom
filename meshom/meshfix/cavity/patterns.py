"""Fill patterns of a 4-sided grid cavity (decisions B135).

The four sides between the corners have edge counts e0..e3 (loop order, CCW, cavity on the left). Rotation r makes
side r the bottom (a = e_r edges, left to right), side r+1 the right side, side r+2 the top (c = e_{r+2}), side r+3
the left side.

- regular (a == c, left == right): a structured grid.
- ladder (a > c, left == right = R strips): k = a - c column ends, each one triangle with its base on row s and its
  tip on row s + 1 (the (4,4,5) dislocation of a quad-dominant mesh, v5's Y-split). Parity: k == a + b + c + d
  (mod 2). Rules that keep every node but the tips regular: a tip lies on an interior row (s <= R - 2; on a coast
  side the last strip is allowed too) and not at a row end (base index 1 .. n_s - 2); two triangles of one strip are
  >= 2 edges apart; a tip is no base corner of a triangle of the next strip. The triangles' places ("layouts") are
  free (at most one per strip on average: k <= the strips that may hold a tip); candidates: h-fitted rows (hfit, hfit2: the row counts follow integral ds / h), spread along the diagonal,
  along the anti-diagonal, in the middle strip, staggered over every strip.
- split (a > c and right > left: the cavity holds a chord turn, i.e. a 3-5 pair): a line of `left` edges from the
  bottom's node a - t to the top's node c - t cuts the cavity into a vertical ladder (a - t, left, c - t, left; the
  column ends) and a horizontal ladder on the right (right, t, left, t; the row ends). Every 3-5 turn becomes
  column-end plus row-end triangles. t in {2, 3, c // 2} within [2, c - 1].

Node positions: rows j = 1 .. R-1 run from the left side's j-th node to the right side's; row j has n_j edges and
its interior nodes at u = i / n_j, v = (1 - u) vL_j + u vR_j (arc fractions of the side nodes), placed by the Coons
patch of the four sides (3D, normalised to the sphere). The relaxation (relax.py) does the rest.
"""
import numpy as np

from meshom.geo import R_EARTH_M

GOLDEN = 0.6180339887


def sides(lp: tuple, cr: list) -> list[list[int]]:
    n = len(lp)
    out = []
    for q in range(4):
        i0, i1 = cr[q], cr[(q + 1) % 4]
        out.append([lp[(i0 + t) % n] for t in range((i1 - i0) % n + 1)])
    return out


class Fill:
    """New nodes of one candidate (id -1 - j for X[j]); positions of old nodes from P."""

    def __init__(self, P):
        self.P = P
        self.X = []
        self.coast = []                                    # ids of new coast nodes


    def pos(self, i: int) -> np.ndarray:
        return self.P[i] if i >= 0 else self.X[-1 - i]

    def arr(self, ids) -> np.ndarray:
        return np.array([self.pos(i) for i in ids])

    def add(self, x: np.ndarray) -> int:
        self.X.append(x / np.linalg.norm(x))
        return -len(self.X)


def _arc(X: np.ndarray) -> np.ndarray:
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
    return d / d[-1]


def _interp(X: np.ndarray, s: np.ndarray, u: float) -> np.ndarray:
    return np.array([np.interp(u, s, X[:, k]) for k in range(3)])


class _Coons:
    def __init__(self, F: Fill, bottom, right, top, left):
        self.XB, self.XT, self.XL, self.XR = F.arr(bottom), F.arr(top), F.arr(left), F.arr(right)
        self.sB, self.sT, self.sL, self.sR = _arc(self.XB), _arc(self.XT), _arc(self.XL), _arc(self.XR)

    def __call__(self, u: float, v: float) -> np.ndarray:
        XB, XT, XL, XR = self.XB, self.XT, self.XL, self.XR
        p = ((1 - v) * _interp(XB, self.sB, u) + v * _interp(XT, self.sT, u) + (1 - u) * _interp(XL, self.sL, v)
             + u * _interp(XR, self.sR, v) - ((1 - u) * (1 - v) * XB[0] + u * (1 - v) * XB[-1]
                                              + (1 - u) * v * XT[0] + u * v * XT[-1]))
        return p / np.linalg.norm(p)

    def many(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """__call__ at many (u, v) at once (the same formula, rows normalised)."""
        u = np.asarray(u, dtype=float)[:, None]
        v = np.asarray(v, dtype=float)[:, None]
        XB, XT, XL, XR = self.XB, self.XT, self.XL, self.XR

        def ip(X, s, t):
            return np.stack([np.interp(t[:, 0], s, X[:, k]) for k in range(3)], axis=1)

        p = ((1 - v) * ip(XB, self.sB, u) + v * ip(XT, self.sT, u) + (1 - u) * ip(XL, self.sL, v)
             + u * ip(XR, self.sR, v) - ((1 - u) * (1 - v) * XB[0] + u * (1 - v) * XB[-1]
                                         + (1 - u) * v * XT[0] + u * v * XT[-1]))
        return p / np.linalg.norm(p, axis=1, keepdims=True)


# ----------------------------------------------------------------------------------------------- layouts

def _layouts(k: int, R: int, ns: int) -> dict:
    """name -> list of (strip, fraction across) for k triangles over strips 0 .. ns-1."""
    q = (np.arange(k) + 0.5) / k
    strip_diag = np.minimum((q * ns).astype(int), ns - 1)
    out = {"diag": list(zip(strip_diag, q)), "anti": list(zip(strip_diag, 1.0 - q))}
    out["mid"] = [((ns - 1) // 2, f) for f in q]
    st = []
    per = int(np.ceil(k / ns))
    for t in range(k):
        s = t % ns
        j = t // ns
        f = (j + 0.5 + 0.5 * (s % 2) - 0.25) / per
        st.append((s, min(max(f, 0.0), 1.0)))
    out["stagger"] = st
    return out


def _hfit_layouts(co: _Coons, a: int, c: int, R: int, ns: int, h, nsamp: int = 16) -> dict:
    """Layouts whose row edge counts follow h (B135): the ideal count of row j is n*_j = integral ds / h along the
    row's Coons line, scaled so that rows 0 and R match a and c (the factor interpolated in ln between them); the
    counts are rounded, kept between c and a, non-increasing upward, and the strips past ns - 1 drop nothing. The
    triangles of a strip are spread evenly across it, shifted from strip to strip by the golden ratio, in two
    phases."""
    ideal = []
    for j in range(R + 1):
        u = np.linspace(0.0, 1.0, nsamp + 1)
        X = co.many(u, (1 - u) * co.sL[j] + u * co.sR[j])
        mid = X[1:] + X[:-1]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        ds = np.linalg.norm(np.diff(X, axis=0), axis=1) * R_EARTH_M
        ideal.append(float(np.sum(ds / h(mid))))
    ideal = np.array(ideal)
    t = np.arange(R + 1) / R
    tgt = ideal * np.exp(np.log(a / ideal[0]) * (1 - t) + np.log(c / ideal[R]) * t)
    n = [a]
    for j in range(1, R):
        n.append(int(min(n[-1], max(c, round(tgt[j])))))
    n.append(c)
    for j in range(ns + 1, R + 1):                       # rows above the last strip that may drop: all c
        n[j] = c
    for j in range(R - 1, 0, -1):
        n[j] = max(n[j], n[j + 1])
    drops = [n[s] - n[s + 1] for s in range(R)]
    out = {}
    for name, ph0 in (("hfit", 0.0), ("hfit2", 0.5)):
        lay = []
        for s, t_s in enumerate(drops):
            ph = (ph0 + s * GOLDEN) % 1.0
            for q in range(t_s):
                lay.append((s, (q + ph) / t_s))
        out[name] = lay
    return out


def _tips(base: set) -> set:
    """Tip indices on the upper row of the triangles with these base indices."""
    return {i - j for j, i in enumerate(sorted(base))}


def _place(a: int, lay: list, R: int, ns: int) -> tuple | None:
    """(row counts, base indices per strip) of a layout, or None (no legal placement)."""
    T = [[] for _ in range(R)]
    for s, f in lay:
        if s >= ns:
            return None
        T[int(s)].append(float(f))
    nrow = [a]
    bases = []
    for s in range(R):
        n = nrow[-1]
        fs = sorted(T[s])
        t = len(fs)
        nrow.append(n - t)
        if t == 0:
            bases.append(set())
            continue
        lo, hi = 1, n - 2
        if hi < lo or 2 * (t - 1) > hi - lo:
            return None
        idx = []
        for j, f in enumerate(fs):
            i = int(round(lo + f * (hi - lo)))
            idx.append(max(i, idx[-1] + 2 if idx else lo))
        if idx[-1] > hi:                                   # shift left, keep spacing
            for j in range(len(idx) - 1, -1, -1):
                idx[j] = min(idx[j], hi - 2 * (len(idx) - 1 - j))
            if idx[0] < lo:
                return None
        bases.append(set(idx))
    for s in range(R - 1):                                 # a tip must not be a base corner of the next strip
        tips = _tips(bases[s])
        if not tips or not bases[s + 1]:
            continue
        nn = nrow[s + 1]
        new = []
        for b in sorted(bases[s + 1]):
            for cand in (b, b + 1, b - 1, b + 2, b - 2, b + 3, b - 3):
                if 1 <= cand <= nn - 2 and cand not in tips and cand + 1 not in tips and \
                        all(abs(cand - x) >= 2 for x in new):
                    new.append(cand)
                    break
            else:
                return None
        bases[s + 1] = set(new)
    return nrow, bases


def patch_plans(F: Fill, bottom, right, top, left, h=None, layouts=None, tip_top: bool = False) -> list[tuple]:
    """(name, row counts, bases) of every legal ladder / regular plan of the patch."""
    a, c, R = len(bottom) - 1, len(top) - 1, len(left) - 1
    if len(right) - 1 != R or a < c or c < 1:
        return []
    k = a - c
    if k == 0:
        return [("regular", [a] * (R + 1), [set() for _ in range(R)])]
    ns = R if tip_top else R - 1
    if ns < 1 or k > ns:                                   # B135: at most one column end per strip on average
        return []
    lays = _layouts(k, R, ns)
    if h is not None:
        lays.update(_hfit_layouts(_Coons(F, bottom, right, top, left), a, c, R, ns, h))
    if layouts is not None:
        lays = {n: v for n, v in lays.items() if n in layouts}
    out, seen = [], set()
    for name, lay in lays.items():
        pl = _place(a, lay, R, ns)
        if pl is None:
            continue
        key = tuple(tuple(sorted(x)) for x in pl[1])
        if key in seen:
            continue
        seen.add(key)
        out.append((name, pl[0], pl[1]))
    return out


def build_patch(F: Fill, bottom, right, top, left, nrow, bases) -> list[tuple]:
    """Cells of the patch (new interior nodes appended to F)."""
    R = len(left) - 1
    co = _Coons(F, bottom, right, top, left)
    rows = [list(bottom)]
    for j in range(1, R):
        n = nrow[j]
        row = [left[j]]
        vL, vR = co.sL[j], co.sR[j]
        if n > 1:
            u = np.arange(1, n) / n
            for x in co.many(u, (1 - u) * vL + u * vR):
                row.append(F.add(x))
        row.append(right[j])
        rows.append(row)
    rows.append(list(top))
    cells = []
    for s in range(R):
        Xr, Yr = rows[s], rows[s + 1]
        ip = 0
        for i in range(len(Xr) - 1):
            if i in bases[s]:
                cells.append((Xr[i], Xr[i + 1], Yr[ip]))
            else:
                cells.append((Xr[i], Xr[i + 1], Yr[ip + 1], Yr[ip]))
                ip += 1
        assert ip == len(Yr) - 1, (s, ip, len(Yr))
    return cells


# ----------------------------------------------------------------------------------------------- the cavity

COAST_SHIFT_H = 0.25    # B135: a coast node may be removed when it lies within this many h of the merged edge


def _dev_m(F: Fill, a: int, v: int, b: int) -> float:
    """Distance (m) of node v to the great circle through a and b."""
    n = np.cross(F.pos(a), F.pos(b))
    n /= np.linalg.norm(n)
    return abs(float(F.pos(v) @ n)) * R_EARTH_M


def adjust(F: Fill, side: list, target: int, h) -> list | None:
    """The coast side with `target` edges: nodes removed (the one closest to the chord of its neighbours first, each
    within COAST_SHIFT_H h of it) or added (midpoints of the longest edges, new coast nodes of F); None when no legal
    removal is left."""
    side = list(side)
    while len(side) - 1 > target:
        best, bi = None, -1
        for i in range(1, len(side) - 1):
            d = _dev_m(F, side[i - 1], side[i], side[i + 1])
            if best is None or d < best:
                best, bi = d, i
        if bi < 0 or best > COAST_SHIFT_H * float(h(F.pos(side[bi])[None])[0]):
            return None
        del side[bi]
    while len(side) - 1 < target:
        L = [np.linalg.norm(F.pos(side[i + 1]) - F.pos(side[i])) for i in range(len(side) - 1)]
        i = int(np.argmax(L))
        nid = F.add(F.pos(side[i]) + F.pos(side[i + 1]))
        F.coast.append(nid)
        side.insert(i + 1, nid)
    return side


def fills(P: np.ndarray, lp: tuple, cr: list, h=None, layouts=None, max_k: int = 64, split: bool = True,
          is_bedge=None, max_adj: int = 4) -> list:
    """Every pattern of the cavity: list of (name, X new nodes (m,3), cells with -1 - j for X[j], new coast node
    ids). With is_bedge, a side made only of mesh-boundary (coast) edges may change its edge count by up to max_adj
    (adjust) to match the opposite side, and a coast top may hold triangle tips."""
    S = sides(lp, cr)
    e = [len(s) - 1 for s in S]
    cs = [is_bedge is not None and all(is_bedge(s[i], s[i + 1]) for i in range(len(s) - 1)) for s in S]
    out, seen = [], set()

    def emit(name, F, cells):
        key = frozenset(tuple(sorted(c)) for c in cells if all(v >= 0 for v in c)) | {len(F.X)}
        if key in seen:
            return
        seen.add(key)
        out.append((name, _xs(F), cells, list(F.coast)))

    for r in range(4):
        q = [(r + j) % 4 for j in range(4)]                # bottom, right, top, left (loop sides)
        sd = {"b": S[q[0]], "r": S[q[1]], "t": S[q[2]][::-1], "l": S[q[3]][::-1]}
        fl = {"b": cs[q[0]], "r": cs[q[1]], "t": cs[q[2]], "l": cs[q[3]]}
        variants = [({}, "")]
        if any(fl.values()):
            a0, c0, r0, l0 = e[q[0]], e[q[2]], e[q[1]], e[q[3]]
            adj = {}
            if r0 != l0:
                if fl["r"] and abs(r0 - l0) <= max_adj:
                    adj["r"] = l0
                elif fl["l"] and abs(r0 - l0) <= max_adj:
                    adj["l"] = r0
            if variants and (adj or True):
                v2 = dict(adj)
                if a0 != c0:
                    if fl["t"] and abs(a0 - c0) <= max_adj:
                        v2["t"] = a0
                    elif fl["b"] and abs(a0 - c0) <= max_adj:
                        v2["b"] = c0
                if adj:
                    variants.append((adj, "adj"))
                if v2 and v2 != adj:
                    variants.append((v2, "adj2"))
        for adj, tag in variants:
            F0 = Fill(P)
            sl = dict(sd)
            ok = True
            for key, tgt in adj.items():
                new = adjust(F0, sl[key], tgt, h)
                if new is None:
                    ok = False
                    break
                sl[key] = new
            if not ok:
                continue
            bottom, right, top, left = sl["b"], sl["r"], sl["t"], sl["l"]
            a, c, dR, bL = len(bottom) - 1, len(top) - 1, len(right) - 1, len(left) - 1
            if a < c or (a == c and r != 0 and not adj) or a - c > max_k:
                continue
            pre = f"{tag}" if tag else ""
            if dR == bL:
                for name, nrow, bases in patch_plans(F0, bottom, right, top, left, h, layouts, tip_top=fl["t"]):
                    F = Fill(P)
                    F.X, F.coast = list(F0.X), list(F0.coast)
                    cells = build_patch(F, bottom, right, top, left, nrow, bases)
                    kind = ("coast" + pre if pre else ("ladder" if a > c else "regular"))
                    emit(f"{kind}_{'ladder' if a > c else 'regular'}-{name}_r{r}_k{a - c}", F, cells)
            elif split and a > c and dR > bL and dR - bL <= max_k:
                for name, F, cells in _split(F0, bottom, right, top, left, h, layouts, r):
                    emit(("coast" + pre + "-" if pre else "") + name, F, cells)
    return out


def _xs(F: Fill) -> np.ndarray:
    return np.array(F.X).reshape(-1, 3)


def _split(Fb: Fill, bottom, right, top, left, h, layouts, r) -> list:
    """(name, Fill, cells) of the split patterns (module docstring); Fb holds the nodes made so far."""
    a, c, b = len(bottom) - 1, len(top) - 1, len(left) - 1
    out = []
    for t in sorted({2, 3, c // 2}):
        if t < 2 or t > c - 1 or b < 1:
            continue
        a1, c1 = a - t, c - t
        F0 = Fill(Fb.P)
        F0.X, F0.coast = list(Fb.X), list(Fb.coast)
        co = _Coons(F0, bottom, right, top, left)
        ub, ut = co.sB[a1], co.sT[c1]
        line = [bottom[a1]]
        if b > 1:
            v = np.asarray(co.sL[1:b])
            for x in co.many(ub * (1 - v) + ut * v, v):
                line.append(F0.add(x))
        line.append(top[c1])
        L = (bottom[: a1 + 1], line, top[: c1 + 1], left)                 # column ends (vertical ladder)
        Rp = (right, top[c1:][::-1], line, bottom[a1:][::-1])             # row ends (horizontal ladder)
        keep = ("hfit", "hfit2", "stagger", "regular")
        pl_l = patch_plans(F0, *L, h=h, layouts=layouts)
        pl_r = patch_plans(F0, *Rp, h=h, layouts=layouts)
        pl_l = [p for p in pl_l if p[0] in keep][:2] or pl_l[:2]
        pl_r = [p for p in pl_r if p[0] in keep][:2] or pl_r[:2]
        for nl, rl, bl in pl_l:
            for nr, rr, br in pl_r:
                F = Fill(Fb.P)
                F.X, F.coast = list(F0.X), list(F0.coast)
                cells = build_patch(F, *L, rl, bl) + build_patch(F, *Rp, rr, br)
                out.append((f"split_{nl}-{nr}_r{r}_t{t}_k{a - c}+{len(right) - len(left)}", F, cells))
    return out
