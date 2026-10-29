"""m14 hand-placed blocks: the `[[blocks]]` of the config (meshom.config.BlockCfg) as fixed blocks, built before the
strait blocks and the automatic placement (decisions U6, B59, B60).

resolve(): a block of the config -> its four sides on the sphere. meshom.run calls it in the parent, against the true
coast with treatment "cut" (the method itself meshes the closed sphere and sees no coast) and against the filtered
coast with treatment "loops" (meshom.coast_filter: the coast the method gets);
m14 calls it on its own domain loops when it gets the config's fields as they are (e.g. `opts["blocks"]` under
`meshom compete`).
1. Corners [lon, lat], counter-clockwise: the block lies on the left of each side (corner k -> corner k+1). A corner
   within snap_m (default SNAP_H h at the corner) of the coast moves onto its nearest coast point, unless both its
   sides are given and neither is "coast" (a cut-mode corner on land stays where it is put).
2. Each side has a kind:
   - "coast": the coast from the one corner forward along their coast loop (water on the left, so the block is on
     the left) to the other, Gaussian-filtered along its arc length at filter_m (default FILTER_H h, C35), ends kept;
   - "cut": the great-circle arc;
   - "lat": the parallel (the latitude linear in the longitude between the corners, the shorter way round);
   - "offset": an offshore curve parallel to the coast: the opposite side when that is a coast side (a
     coastal band), else the coast stretch between the corners' nearest coast points, extruded along the normals of
     its smoothing at OFFSET_SMOOTH_H h by the corners' distances from its ends (linear in between), cut where it
     passes nearest to the corners.
   Default kind (sides not given): "coast" when both corners lie within NEAR_H h of one coast loop and the forward
   coast path between them is shorter than PATH_CHORD x the chord, else "cut".
build(): counts m along sides 0 and 2 (`rows`, "cells along side 1-2") and n along sides 1 and 3 (`cols`), else the
rounded mean integral ds/h_b of the two opposite sides (the grid lines between them can be longer or shorter: the record
gives their range in h, `row_cells_h` / `col_cells_h`, as a guide for rows and cols, e.g. the Atlantic between 20 S
and 40 N is 59 h long at its parallels and 72 h at the equator); h_b = h_m (constant) or the run's h; then the harmonic
Dirichlet map of m14/strait.py (harmonic_grid; grid_block: 30 "size" sweeps, the sides fixed). A hand block is built
as given: its B22 numbers (|ln s/h_b|, quad angles, inverted cells) are recorded, not tested, and it is not split; one
whose map misses a lattice point is left out (status "map"). The coast mask is the nodes of its coast sides.

How a block enters the mesh (m14/stitch.py, m14/cut.py, meshom.run.cut_land):
- conforming (the domain has loops, no cut): a block with a coast side joins the coast loops through its non-coast
  outline runs (coast corner to coast corner, in outline order, located on the domain loops for
  stitch.merged_loops), as a strait block does; a block without one is one more exact outline of the fill (its sides
  must lie in water);
- cut (variant A, or `meshom run` with treatment "cut"): the whole outline is an exact loop of the closed-sphere fill
  and corners may lie on land. A block without coast sides is cut by the mask with everything else (ORCA style: the
  coast is the staircase of its cells; coast-to-coast blocks are built only this way). A built block with a coast
  side is protected in the cut (m14/cut.py fixed_banks: its cells wet, the
  element across each coast-side edge land; meshom.run.cut_land the same, from m14_blocks.CUT_FIXED).
"""
import numpy as np

from meshom.domain import Domain
from meshom.geo import R_EARTH_M, arc_length, lonlat_to_xyz, tangent_basis, xyz_to_lonlat
from meshom.methods.m14.strait import BG, FixedBlock, _seg_m, _unit, densify, grid_block, h_integral, \
    harmonic_grid, smooth_polyline
from meshom.size import SizeField, Uniform

KINDS = ("coast", "cut", "lat", "offset")
SNAP_H = 1.0             # B60: snap_m default, units of h at the corner
FILTER_H = 2.0           # B60 / C35: filter_m default, units of h
NEAR_H = 1.0             # B60: the default side rule: both corners within this many h of one coast loop ...
PATH_CHORD = 3.0         # ... and the forward coast path between them shorter than this many chords
OFFSET_SMOOTH_H = 8.0    # "offset" sides: the coast smoothed at this many h (normals of the coast at 8 h)
STEP_H = 0.25            # side polylines sampled at this many h


# --------------------------------------------------------------------------------------------- the coast

def _project(p: np.ndarray, lp: np.ndarray) -> tuple:
    """Nearest point of the closed loop lp to p: (point, seg, t, distance m); t = the arc fraction along seg."""
    A, B = lp, np.roll(lp, -1, axis=0)
    nrm = np.cross(A, B)
    ln = np.linalg.norm(nrm, axis=1)
    nrm = nrm / np.maximum(ln, 1e-300)[:, None]
    q = p[None, :] - (nrm @ p)[:, None] * nrm
    q = q / np.maximum(np.linalg.norm(q, axis=1), 1e-300)[:, None]
    inner = (np.einsum("ij,ij->i", np.cross(A, q), nrm) >= 0) & (np.einsum("ij,ij->i", np.cross(q, B), nrm) >= 0) \
        & (ln > 1e-15)
    dA, dB = arc_length(A, p[None]), arc_length(B, p[None])
    q = np.where(inner[:, None], q, np.where((dA <= dB)[:, None], A, B))
    d = arc_length(q, p[None])
    k = int(np.argmin(d))
    t = float(arc_length(A[k][None], q[k][None])[0] / max(float(arc_length(A[k][None], B[k][None])[0]), 1e-300))
    return q[k].copy(), k, min(max(t, 0.0), 1.0), float(R_EARTH_M * d[k])


class Shore:
    """The coast loops with a k-d tree over their vertices: nearest points per loop."""

    def __init__(self, loops: list):
        from scipy.spatial import cKDTree

        self.loops = list(loops)
        self.V = np.vstack(self.loops)
        self.loop = np.repeat(np.arange(len(self.loops)), [len(lp) for lp in self.loops])
        nxt = np.vstack([np.roll(lp, -1, axis=0) for lp in self.loops])
        ln = R_EARTH_M * arc_length(self.V, nxt)                    # segment i: vertex i -> its successor
        self.seg = float(ln.max())
        k = [len(lp) for lp in self.loops]
        prv = np.concatenate([o + np.roll(np.arange(n), 1) for o, n in zip(np.cumsum([0] + k[:-1]), k)])
        self.reach = np.maximum(ln, ln[prv])                        # the longest segment at each vertex, m
        self.tree = cKDTree(self.V)

    def near(self, p: np.ndarray, r_m: float) -> dict:
        """{loop: (point, seg, t, distance m)} of the loops that come within r_m of p."""
        rad = (r_m + self.seg) / R_EARTH_M
        cand = self.tree.query_ball_point(p, 2 * np.sin(0.5 * min(rad, np.pi)))
        out = {}
        for L in np.unique(self.loop[np.asarray(cand, dtype=np.int64)]) if len(cand) else []:
            x = _project(p, self.loops[L])
            if x[3] <= r_m:
                out[int(L)] = x
        return out

    def nearest(self, p: np.ndarray) -> tuple[int, tuple]:
        """(loop, (point, seg, t, distance m)) of the nearest coast point. The nearest vertex bounds the distance from
        above (d_v); a vertex of the nearest segment then lies within d_v + that segment's length, so the loops with a
        vertex i within d_v + reach[i] hold it (a fixed count of nearest vertices can miss a long segment)."""
        d, _ = self.tree.query(p)
        dv = 2.0 * np.arcsin(min(0.5 * float(d), 1.0)) * R_EARTH_M
        cand = np.asarray(self.tree.query_ball_point(p, 2 * np.sin(0.5 * min((dv + self.seg) / R_EARTH_M, np.pi))
                                                     * (1 + 1e-12)), dtype=np.int64)
        cand = cand[R_EARTH_M * arc_length(self.V[cand], p[None]) <= (dv + self.reach[cand]) * (1 + 1e-12) + 1e-6]
        best = None
        for L in np.unique(self.loop[cand]):
            x = _project(p, self.loops[L])
            if best is None or x[3] < best[1][3]:
                best = (int(L), x)
        return best

    def path(self, L: int, a: tuple, b: tuple) -> np.ndarray:
        """The coast from location a forward along loop L to location b (a, b = (point, seg, t, ...)), ends included."""
        lp = self.loops[L]
        K = len(lp)
        cnt = (b[1] - a[1]) % K
        if cnt == 0 and b[2] < a[2]:
            cnt = K
        q = np.vstack([a[0][None], lp[(a[1] + 1 + np.arange(cnt)) % K], b[0][None]])
        return q[np.r_[True, np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-12]]


# ---------------------------------------------------------------------------------------------- the sides

def _lat_side(a: np.ndarray, b: np.ndarray, step_m: float) -> np.ndarray:
    (la, lb), (pa, pb) = xyz_to_lonlat(np.vstack([a, b]))
    dl = (lb - la + 180.0) % 360.0 - 180.0
    span = R_EARTH_M * np.radians(abs(dl) * np.cos(np.radians(0.5 * (pa + pb))) + abs(pb - pa))
    t = np.linspace(0.0, 1.0, max(2, int(np.ceil(span / step_m))) + 1)
    p = lonlat_to_xyz(la + dl * t, pa + (pb - pa) * t)
    p[0], p[-1] = a, b
    return p


def _offset_side(a: np.ndarray, b: np.ndarray, base: np.ndarray, hb: float, step_m: float) -> np.ndarray:
    """The curve from a to b parallel to `base` (a coast piece running from near a to near b): base extruded along the
    normals of its smoothing at OFFSET_SMOOTH_H hb (rows along the coast, not a far side smoothed on its
    own) by the corners' distances from its ends (linear in between), cut at its points nearest to a and b (no hook
    where a corner lies short of the base's end), ends a and b."""
    q = densify(base, step_m)
    sm = smooth_polyline(q, OFFSET_SMOOTH_H * hb, step_m)
    s = np.r_[0.0, np.cumsum(_seg_m(q))]
    s = s / max(s[-1], 1e-300)
    t = np.r_[0.0, np.cumsum(_seg_m(sm))]
    N = _unit(np.cross(sm, np.gradient(sm, axis=0)))           # left of the smoothed base's direction
    N = np.column_stack([np.interp(s, t / max(t[-1], 1e-300), N[:, d]) for d in range(3)])
    N = _unit(N - np.einsum("ij,ij->i", N, q)[:, None] * q)
    sgn = np.sign((a - q[0]) @ N[0]) or 1.0
    da, db = (float(arc_length(x[None], y[None])[0]) for x, y in ((a, q[0]), (b, q[-1])))
    d = (1.0 - s) * da + s * db
    out = _unit(np.cos(d)[:, None] * q + sgn * np.sin(d)[:, None] * N)
    ia, ib = int(np.argmin(arc_length(out, a[None]))), int(np.argmin(arc_length(out, b[None])))
    out = out[ia:ib + 1] if ib - ia >= 2 else out[[0, -1]]
    return densify(np.vstack([a[None], out[1:-1], b[None]]), step_m)


def resolve(b: dict, loops: list | None, h: SizeField) -> dict:
    """A config block (BlockCfg's fields) -> {name, corners (4,3), kinds, sides_xyz (4 polylines, CCW, block on the
    left), rows, cols, h_m, snapped, loops} (module docstring, 1. and 2.). loops: the coast (None: no coast)."""
    C = lonlat_to_xyz(*np.asarray(b["corners"], dtype=np.float64).T)
    hb = np.full(4, float(b["h_m"])) if b.get("h_m") else h(C)
    shore = Shore(loops) if loops else None
    kinds = list(b["sides"]) if b.get("sides") else [None] * 4
    snapped = [False] * 4
    if shore is not None:
        for k in range(4):                         # a corner of a coast side (given or by the rule) snaps
            if kinds[k - 1] not in (None, "coast") and kinds[k] not in (None, "coast"):
                continue
            L, x = shore.nearest(C[k])
            if x[3] <= (b["snap_m"] if b.get("snap_m") is not None else SNAP_H * hb[k]):
                C[k], snapped[k] = x[0], x[3] > 0
    sides, side_loop = [], []
    for k in range(4):
        a, c = C[k], C[(k + 1) % 4]
        hk = 0.5 * (hb[k] + hb[(k + 1) % 4])
        step = STEP_H * hk
        L = -1
        if shore is not None and kinds[k] in (None, "coast"):
            na, nc = shore.near(a, NEAR_H * hb[k]), shore.near(c, NEAR_H * hb[(k + 1) % 4])
            chord = R_EARTH_M * float(arc_length(a[None], c[None])[0])
            for LL in sorted(set(na) & set(nc)):
                if _seg_m(shore.path(LL, na[LL], nc[LL])).sum() < PATH_CHORD * chord:
                    L = LL
                    break
            if kinds[k] == "coast" and L < 0:                     # given: the loop nearest to the first corner
                L = shore.nearest(a)[0]
            kinds[k] = "coast" if L >= 0 else "cut"
        kinds[k] = kinds[k] or "cut"
        if kinds[k] == "coast":
            xa, xc = _project(a, shore.loops[L]), _project(c, shore.loops[L])
            q = shore.path(L, xa, xc)
            q = np.vstack([a[None], q[1:-1], c[None]]) if len(q) > 2 else np.vstack([a, c])
            f = b["filter_m"] if b.get("filter_m") is not None else FILTER_H * hk
            sd = smooth_polyline(q, f, step)
        elif kinds[k] == "lat":
            sd = _lat_side(a, c, step)
        elif kinds[k] == "offset":
            sd = None                                  # below: it may follow the opposite (coast) side
        else:
            sd = densify(np.vstack([a, c]), step)
        sides.append(sd)
        side_loop.append(int(L))
    for k in range(4):
        if kinds[k] != "offset":
            continue
        a, c = C[k], C[(k + 1) % 4]
        hk = 0.5 * (hb[k] + hb[(k + 1) % 4])
        if kinds[(k + 2) % 4] == "coast":                   # a coastal band: parallel to its coast side
            base = sides[(k + 2) % 4][::-1]
        else:                                               # the coast between the corners' nearest coast points
            L, xa = shore.nearest(a)
            xc = _project(c, shore.loops[L])
            fw, bw = shore.path(L, xa, xc), shore.path(L, xc, xa)[::-1]
            base = fw if _seg_m(fw).sum() <= _seg_m(bw).sum() else bw
        sides[k] = _offset_side(a, c, base, hk, STEP_H * hk)
    return {"name": b.get("name", ""), "corners": C, "kinds": kinds, "sides_xyz": sides, "rows": b.get("rows"),
            "cols": b.get("cols"), "h_m": b.get("h_m"), "snapped": snapped, "loops": side_loop}


def outline(r: dict) -> np.ndarray:
    """The closed outline polygon of a resolved block (its four sides joined, CCW)."""
    return np.vstack([sd[:-1] for sd in r["sides_xyz"]])


# ---------------------------------------------------------------------------------------------- the blocks

def _corner_deg(P: np.ndarray) -> list:
    """Interior angles (deg) of the block at its four corners, in the tangent planes."""
    out = []
    for (i, j), (di, dj) in (((0, 0), (1, 1)), ((-1, 0), (-1, 1)), ((-1, -1), (-1, -1)), ((0, -1), (1, -1))):
        c = P[i, j]
        u, v = P[i + di, j] - c, P[i, j + dj] - c
        e1, e2 = tangent_basis(c[None])
        u2 = np.array([u @ e1[0], u @ e2[0]])
        v2 = np.array([v @ e1[0], v @ e2[0]])
        out.append(round(float(np.degrees(np.arccos(np.clip(u2 @ v2 / np.linalg.norm(u2) / np.linalg.norm(v2),
                                                              -1, 1)))), 2))
    return out


def hand_blocks(specs: list, domain: Domain, h: SizeField, o: dict) -> tuple[list[FixedBlock], list[dict]]:
    """The fixed blocks of opts["blocks"] (config fields or resolve()'s dicts) and one record each (module
    docstring)."""
    conform = bool(domain.loops) and not o["cut"]
    shore = Shore(domain.loops) if conform else None
    blocks, recs = [], []
    for i, b in enumerate(specs):
        r = b if "sides_xyz" in b else resolve(b, domain.loops or None, h)
        hb = Uniform(float(r["h_m"])) if r["h_m"] else h
        sides = r["sides_xyz"]
        L = [float(h_integral(densify(sd, BG * float(np.min(hb(sd)))), hb)[-1]) for sd in sides]
        m = int(r["rows"]) if r["rows"] else max(1, int(round(0.5 * (L[0] + L[2]))))
        n = int(r["cols"]) if r["cols"] else max(1, int(round(0.5 * (L[1] + L[3]))))
        lo, la = xyz_to_lonlat(r["corners"])
        rec = {"kind": "hand", "name": r["name"], "hand_index": i, "sides": r["kinds"], "m": m, "n": n,
               "corners_lonlat": np.round(np.column_stack([lo, la]), 4).tolist(), "snapped": r["snapped"],
               "side_cells_h": [round(x, 2) for x in L], "h_b": "h_m" if r["h_m"] else "h"}
        P, minfo = harmonic_grid(sides, hb, m, n)
        rec.update(minfo)
        if P is None:
            rec["status"] = "map"
            recs.append(rec)
            continue
        grid, _, acc = grid_block(P, hb, o["block_size_iters"])
        rec.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in acc.items()})
        rec["b22"] = bool(acc["lnsh_max"] <= 0.10 and acc["quad_max_deg"] <= 135.0 and acc["inverted"] == 0)
        Q = grid.points.reshape(m + 1, n + 1, 3)
        rec["corner_deg"] = _corner_deg(Q)
        for key, lines in (("row_cells_h", Q.transpose(1, 0, 2)), ("col_cells_h", Q)):   # the grid lines in h: a
            c = [float(h_integral(ln, hb)[-1]) for ln in lines]                          # guide for rows / cols
            rec[key] = [round(min(c), 2), round(max(c), 2)]
        nid = np.arange((m + 1) * (n + 1)).reshape(m + 1, n + 1)
        ccw = [nid[:, 0], nid[m, :], nid[::-1, n], nid[0, ::-1]]       # side k: corner k -> corner k+1
        coast = np.zeros((m + 1) * (n + 1), dtype=bool)
        for k in range(4):
            if r["kinds"][k] == "coast":
                coast[ccw[k]] = True
        runs = []
        if conform and coast.any():                 # the non-coast outline runs, outline order (sides 3, 2, 1, 0)
            order = [3, 2, 1, 0]
            s0 = next(j for j, k in enumerate(order) if r["kinds"][k] == "coast")
            order = order[s0 + 1:] + order[:s0 + 1]
            cur = []
            for k in order:
                if r["kinds"][k] != "coast":
                    seq = ccw[k][::-1]
                    cur.append(seq if not cur else seq[1:])
                    continue
                if cur:
                    ids = np.concatenate(cur)
                    ends = []
                    for v in (ids[0], ids[-1]):
                        Lv, x = shore.nearest(grid.points[v])
                        ends.append((Lv, x[1], x[2]))
                    runs.append((ids, ends[0], ends[1]))
                    cur = []
            rec["mouths"] = [{"lonlat": np.round(np.column_stack(xyz_to_lonlat(grid.points[[ids[0], ids[-1]]])),
                                                 4).tolist(), "cells": int(len(ids) - 1), "kind": "hand"}
                             for ids, _, _ in runs]
        rec["status"] = "built"
        rec["fixed_index"] = len(blocks)
        blocks.append(FixedBlock(grid, coast, runs, rec))
        recs.append(rec)
    return blocks, recs
