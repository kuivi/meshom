"""Seam test cases: lakes whose coasts fix the lattice of each region, so a
seam sits where two such lattices meet. Registered into meshom.cases.CASES (one import line there), so
`meshom compete --cases seam_rot15_k3` runs them; compete._score adds the seam measures (meshom/quality/seams.py) as
the record's "seam" block on every seam case (so `meshom compete`, a re-scoring and the _rep variants carry
it). `run_seams` (CLI: `python -m meshom.cases_seams run ...`) calls compete.run_case with the case's layout hint
(SEAM_HINTS) for the m14 methods, adds m14's block summary and appends the records to bench/competition_seams.jsonl
(or --jsonl). The seam gates are their own table (`summary`, `table`; `write_table` puts one between the seam-table
markers of a Markdown file), never a column of the 4-case ranking (decisions P29).

Layout hint (decisions P28): m14 left alone puts one block across each lake (its frame from the
tangent basis of its first node), so no block-block seam is exercised. SEAM_HINTS[case] = m14 options: `seeds` one per
region, at the region's centre on its axis (the bent bar's two arms from the corridor edge on the arm's axis to the
far end, the Y's three arms from their inner corners, the strait's two basins, the coast lake's region 8-48 H0 off the
coast), `grow_limit` = HINT_GROW_H h (the L-inf growth square of half-side 20 h, so the square's near edge crosses the
axis at the region's near boundary and a block stays in its arm), and `align` = True (each block frame turned to the
lake's B4 field, which follows the coasts except in the 45-deg bend's arm A, where a +1/4 cone at the convex corner
twists it), and `strait_blocks` = False except on seam_strait (a lake's inlet is a test artefact, and a fixed strait
block there would cut the lake from its ocean and mix strait blocks into the block-block seams measured here). The
hint is merged under the caller's opts and recorded in the record's opts.

Why lakes, not the closed sphere: in the open ocean the lattice orientation is free (decisions F11), so a rotation
between two regions is a property of a method's frames, not of (domain, h). A coast fixes it: a coast-aligned lattice
(FESOM-C coasts, coastal blocks, every coast-fitted fill) takes the coast's direction, and two regions with
coasts at an angle theta must meet somewhere with that misorientation. The lake also bounds the case to ~4e3 cells.

Geometry: polygons in the gnomonic plane at CENTRE (x along azimuth `x_az`, y to its left, units of H0 = 25 km at
scale 1), so straight edges are great circles; loops sampled every 5 km, water inside (left).
- seam_rot<theta>_k<k>: a bar 40 H0 wide bent by theta at x = 0 (each arm 43 H0 from the bend: a 40 H0 region beyond
  the corridor), the corridor the 6 H0 strip about the bend's bisector, h = H0 on arm A (west) and H0 (1 + k/100) on
  arm B, ln h a smoothstep across the corridor. Arms along the east-west axis at theta = 0.
- seam_junction: three arms 40 H0 wide at 120 deg meeting at a point (a Y); the three walls (bisectors, from the
  centre to the inner corners) are 30-deg corridors; the core (radius 6 H0) holds the disclination; h = H0.
- seam_coast: a 56 x 56 H0 lake, h = H0; the corridor is the 8 H0 strip between the main coast and the region
  8 H0 off it (coast at azimuth 60 deg, i.e. 30 deg off the east-west axis).
- seam_strait: a channel 12 H0 wide and 60 H0 long between two 60 x 40 H0 basins, h = H0 (same orientation).
Every lake sits in a land ring (to the circle RING_H0 beyond its farthest vertex) with a graded world ocean outside
(h = H0 + 0.1 per metre beyond the ring, at most H_FAR_M), joined to the lake by an inlet INLET_H0 wide through the
ring at one far end (40+ cells from every corridor): one water component, one coast loop. Two kernel limits force
this. (1) The sphere Delaunay of meshom.delaunay is the convex hull oriented by the sign of each facet's triple
product, valid only when the points surround the origin: a lake alone (every point within a hemisphere) gets far-side
facets of either sign along its straight coasts, one claims a coast edge for the water and remove_land keeps the far
side (a double cover, chi = -15: seam_rot0_k10, seam_rot15_k0 on the first runs); fixed since by far-cap helper
points (decisions M36), and the builders' ring=False gives the lake alone. (2) meshom.field integrates the
cross field along one BFS tree, so with the ocean as a second component the lake's field stayed at each node's
tangent basis (geographic north) with only the coast nodes fixed: m02 built a lattice 31.6 deg off seam_coast's
coast; fixed since by one BFS tree per component (decisions F15). The seam block's totals cover the lake (the cap inside the ring; the inlet's inner part included).
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from meshom.cases import CASES, STRAITS, StraitCorridor
from meshom.domain import Domain
from meshom.geo import R_EARTH_M, arc_length, lonlat_to_xyz, xyz_to_lonlat
from meshom.quality.seams import Core, Corridor, SeamSpec, seam_block

H0_M = 25e3
CENTRE_LONLAT = (-30.0, 0.0)
CORRIDOR_HALF = 3.0          # corridor half-width (H0): a corridor of 6 h = m14's block gap
TRIM = 3.0                   # corridor ends cut from the measures (H0): the coastal rows at the ends
ARM = 43.0                   # bend to far end (H0): 3 (corridor half) + 40 (region)
HALF_W = 20.0                # half the bar / arm width (H0)
LOOP_STEP_M = 5e3
JITTER_M = 2.0               # coast samples moved this far (at most) off their great circle: no exact in-circle ties
RING_H0 = 4.0                # land ring: the ocean's coast circle lies this far (H0) beyond the lake's farthest vertex
INLET_H0 = 8.0               # width of the inlet joining the lake to the world ocean (H0): wider than 6 h, or the
                             # field (F6) pushes the inlet corners' charge to its mouth as a cone in the lake
G_OCEAN = 0.1                # h slope in the world ocean beyond the ring
H_FAR_M = 400e3
SEAMS_JSONL = Path("bench/competition_seams.jsonl")
ROT_DEG = (0, 15, 30, 45)
K_PER100 = (0, 3, 10)
HINT_GROW_H = 20.0           # layout hint: m14's grow_limit (units of h), the half-side of a block's growth square
SEAM_TABLE_BEGIN, SEAM_TABLE_END = "<!-- seam-table:begin -->", "<!-- seam-table:end -->"


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@dataclass(frozen=True)
class Plane:
    """Gnomonic frame at CENTRE: x along azimuth x_az (deg, clockwise from north), y = x turned 90 deg to the left."""
    x_az: float = 90.0

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(c, ex, ey): the centre and the plane's axes (tangent at c)."""
        lon, lat = CENTRE_LONLAT
        c = lonlat_to_xyz(np.array([lon]), np.array([lat]))[0]
        lo, la = np.radians(lon), np.radians(lat)
        east = np.array([-np.sin(lo), np.cos(lo), 0.0])
        north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
        az = np.radians(self.x_az)
        ex = np.sin(az) * east + np.cos(az) * north
        return c, ex, np.cross(c, ex)

    def to_sphere(self, xy: np.ndarray) -> np.ndarray:
        c, ex, ey = self.frame()
        xy = np.atleast_2d(np.asarray(xy, dtype=float)) * (H0_M / R_EARTH_M)
        return _unit(c[None, :] + xy[:, :1] * ex[None, :] + xy[:, 1:2] * ey[None, :])


def _seg(a: np.ndarray, b: np.ndarray, rng) -> np.ndarray:
    """Great-circle samples from a (included, exact) towards b (excluded) every LOOP_STEP_M, the inner ones moved off
    the great circle by a fixed pseudo-random amount up to JITTER_M: exactly co-circular coast points are in-circle
    ties of the sphere Delaunay (decisions M13)."""
    ang = float(arc_length(a, b))
    k = max(1, int(np.ceil(ang * R_EARTH_M / LOOP_STEP_M)))
    t = np.arange(k)[:, None] / k
    x = _unit(np.sin((1 - t) * ang) * a + np.sin(t * ang) * b)
    off = rng.uniform(-JITTER_M, JITTER_M, k) / R_EARTH_M
    off[0] = 0.0
    return _unit(x + off[:, None] * _unit(np.cross(a, b))[None, :])


def _lake_loop(plane: Plane, verts: list, inlet_edge: int, ring: bool = True) -> tuple[np.ndarray, float]:
    """The single coast loop of a lake (polygon `verts`, CCW in the plane) joined to the world ocean: along the lake's
    coast to the inlet on edge inlet_edge (verts[i] -> verts[i+1], centred, INLET_H0 wide), out along the inlet to the
    ring circle, clockwise round the circle (the ocean on the left), back in along the inlet's other side, on along the
    lake. ring=False: the lake alone (its polygon, every point within one hemisphere). Returns (loop, ring radius m)."""
    c, ex, ey = plane.frame()
    V = np.asarray(verts, dtype=float)
    r_out = float(R_EARTH_M * arc_length(plane.to_sphere(V), c[None, :]).max()) + RING_H0 * H0_M
    if not ring:
        rng = np.random.default_rng(0)
        S = plane.to_sphere(np.vstack([V, V[:1]]))
        return np.vstack([_seg(u, w, rng) for u, w in zip(S[:-1], S[1:])]), r_out
    rho = r_out / R_EARTH_M
    rp = np.tan(rho) * R_EARTH_M / H0_M                               # the ring's radius in the plane
    a, b = V[inlet_edge], V[(inlet_edge + 1) % len(V)]
    d = (b - a) / np.linalg.norm(b - a)
    n_out = np.array([d[1], -d[0]])                                   # right of the edge = out of the lake
    mid = 0.5 * (a + b)
    p1, p2 = mid - 0.5 * INLET_H0 * d, mid + 0.5 * INLET_H0 * d

    def to_ring(p):
        pn = p @ n_out
        return p + (-pn + np.sqrt(pn * pn - p @ p + rp * rp)) * n_out

    c1, c2 = to_ring(p1), to_ring(p2)
    f1, f2 = np.arctan2(c1[1], c1[0]), np.arctan2(c2[1], c2[0])
    span = np.mod(f1 - f2, 2.0 * np.pi)                               # clockwise from c1 to c2, the long way
    k = int(np.ceil(span * R_EARTH_M * np.sin(rho) / LOOP_STEP_M))
    rng = np.random.default_rng(0)
    ph = f1 - span * np.arange(k) / k
    arc = _unit(np.cos(rho) * c + np.sin(rho) * (np.cos(ph)[:, None] * ex + np.sin(ph)[:, None] * ey))
    jit = rng.uniform(-JITTER_M, JITTER_M, k) / R_EARTH_M
    jit[0] = 0.0
    arc = _unit(arc + jit[:, None] * _unit(arc - c[None, :]))
    pts = list(V[: inlet_edge + 1]) + [p1]
    out = []
    S = plane.to_sphere(np.asarray(pts))
    for u, w in zip(S[:-1], S[1:]):
        out.append(_seg(u, w, rng))
    P1, C1, C2, P2 = plane.to_sphere(np.array([p1, c1, c2, p2]))
    out += [_seg(P1, C1, rng), arc, _seg(C2, P2, rng)]
    rest = plane.to_sphere(np.vstack([[p2], V[inlet_edge + 1:], V[:1]]))
    for u, w in zip(rest[:-1], rest[1:]):
        out.append(_seg(u, w, rng))
    return np.vstack(out), r_out


@dataclass(frozen=True)
class PlateauH:
    """h_a on side a (d < -w) of the great circle with this pole (d = R asin(x . pole)), h_b on side b (d > w), ln h
    a smoothstep in d between; beyond r_env_m from `centre` (the world ocean) h_a + g_env per metre, at most
    h_max_m."""
    pole: tuple
    h_a_m: float
    h_b_m: float
    w_m: float
    centre: tuple
    r_env_m: float
    g_env: float = G_OCEAN
    h_max_m: float = H_FAR_M

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        d = R_EARTH_M * np.arcsin(np.clip(xyz @ np.asarray(self.pole), -1.0, 1.0))
        t = np.clip((d + self.w_m) / (2.0 * self.w_m), 0.0, 1.0)
        lh = np.log(self.h_a_m) + (np.log(self.h_b_m) - np.log(self.h_a_m)) * t * t * (3.0 - 2.0 * t)
        r = R_EARTH_M * arc_length(xyz, np.asarray(self.centre)[None, :])
        out = r > self.r_env_m
        return np.where(out, np.minimum(self.h_a_m + self.g_env * (r - self.r_env_m), self.h_max_m), np.exp(lh))


def _lake(name: str, plane: Plane, verts: list, inlet_edge: int, pole: np.ndarray, h_b: float, corridors: tuple,
          cores: tuple = (), ring: bool = True):
    """Domain (one loop: the lake, its inlet and the ring; ring=False: the lake alone), h, and the seam spec (totals
    over the cap inside the ring)."""
    loop, r = _lake_loop(plane, verts, inlet_edge, ring)
    c = plane.frame()[0]
    h = PlateauH(tuple(pole), H0_M, h_b, CORRIDOR_HALF * H0_M, tuple(c), r)
    return Domain([loop], name), h, SeamSpec(corridors, cores, tuple(c), r - 0.5 * RING_H0 * H0_M)


def _corridor(plane: Plane, name: str, a: tuple, b: tuple, half: float, **kw) -> Corridor:
    p, q = plane.to_sphere([a, b])
    return Corridor(name, tuple(p), tuple(q), half * H0_M, **kw)


def seam_rot(theta_deg: float, k: float, ring: bool = True) -> tuple[Domain, PlateauH, SeamSpec]:
    """The bent bar (module docstring). Corridor from the convex bend corner (north) to the concave one (south):
    side a = arm A (west, lattice at +theta/2 to the corridor), side b = arm B (east, -theta/2)."""
    be = np.radians(0.5 * theta_deg)
    uA, uB = np.array([-np.cos(be), -np.sin(be)]), np.array([np.cos(be), -np.sin(be)])
    upA, upB = np.array([-np.sin(be), np.cos(be)]), np.array([np.sin(be), np.cos(be)])
    yt = HALF_W / np.cos(be)
    verts = [(0.0, yt), ARM * uA + HALF_W * upA, ARM * uA - HALF_W * upA, (0.0, -yt),
             ARM * uB - HALF_W * upB, ARM * uB + HALF_W * upB]
    pl = Plane()
    cor = _corridor(pl, "bend", (0.0, yt), (0.0, -yt), CORRIDOR_HALF, phi_a_deg=0.5 * theta_deg,
                    phi_b_deg=-0.5 * theta_deg, trim_m=(TRIM * H0_M, TRIM * H0_M))
    return _lake(f"seam_rot{theta_deg:g}_k{k:g}", pl, verts, 1, cor.pole, H0_M * (1.0 + k / 100.0), (cor,),
                 ring=ring)


def seam_junction(ring: bool = True) -> tuple[Domain, PlateauH, SeamSpec]:
    """Three arms at 90, 210, 330 deg; wall i between arm i and arm i+1 runs from the centre along the bisector to
    their inner corner (radius HALF_W / sin 60), side a = arm i; measured from 6 H0 (the core) to 3 H0 before the
    corner. Each wall: lattices at 30 and 60 deg to it (a 30-deg misorientation)."""
    pl = Plane()
    r_in = HALF_W / np.sin(np.radians(60.0))
    s_far = r_in * np.cos(np.radians(60.0)) + 40.0
    verts, cors = [], []
    for i in range(3):
        a = np.radians(90.0 + 120.0 * i)
        u, rgt = np.array([np.cos(a), np.sin(a)]), np.array([np.sin(a), -np.cos(a)])
        w0 = a - np.radians(60.0)
        verts += [r_in * np.array([np.cos(w0), np.sin(w0)]), s_far * u + HALF_W * rgt, s_far * u - HALF_W * rgt]
        w1 = a + np.radians(60.0)
        cors.append(_corridor(pl, f"wall{i}", (0.0, 0.0), tuple(r_in * np.array([np.cos(w1), np.sin(w1)])),
                              CORRIDOR_HALF, phi_a_deg=-60.0, phi_b_deg=60.0,
                              trim_m=(6.0 * H0_M, TRIM * H0_M)))
    core = Core("core", tuple(pl.to_sphere([[0.0, 0.0]])[0]), 6.0 * H0_M, 1)
    return _lake("seam_junction", pl, verts, 1, cors[0].pole, H0_M, tuple(cors), (core,), ring=ring)


def seam_coast(ring: bool = True) -> tuple[Domain, PlateauH, SeamSpec]:
    """56 x 56 H0 lake, the main coast y = 0; corridor y in [0, 8] under the region x in [-20, 20], y in [8, 48]
    (side a = the coast)."""
    pl = Plane(x_az=60.0)
    cor = _corridor(pl, "coast", (-20.0, 4.0), (20.0, 4.0), 4.0, coast_a=True)
    return _lake("seam_coast", pl, [(-28.0, 0.0), (28.0, 0.0), (28.0, 56.0), (-28.0, 56.0)], 2, cor.pole, H0_M, (cor,),
                 ring=ring)


def seam_strait(ring: bool = True) -> tuple[Domain, PlateauH, SeamSpec, StraitCorridor]:
    """Channel |x| <= 6, |y| <= 30 between basins |x| <= 30, 30 <= |y| <= 70 (H0); both sides coasts."""
    pl = Plane(x_az=60.0)
    verts = [(-30.0, -70.0), (30.0, -70.0), (30.0, -30.0), (6.0, -30.0), (6.0, 30.0), (30.0, 30.0), (30.0, 70.0),
             (-30.0, 70.0), (-30.0, 30.0), (-6.0, 30.0), (-6.0, -30.0), (-30.0, -30.0)]
    cor = _corridor(pl, "channel", (0.0, -30.0), (0.0, 30.0), 6.0, coast_a=True, coast_b=True, banks=True,
                    trim_m=(TRIM * H0_M, TRIM * H0_M))
    return (*_lake("seam_strait", pl, verts, 6, cor.pole, H0_M, (cor,), ring=ring),
            StraitCorridor(cor.p, cor.q, cor.half_width_m))


def _build() -> tuple[dict, dict, dict]:
    cases, seams, straits = {}, {}, {}
    for th in ROT_DEG:
        for k in K_PER100:
            d, h, s = seam_rot(th, k)
            cases[d.name], seams[d.name] = (d, h), s
    for f in (seam_junction, seam_coast):
        d, h, s = f()
        cases[d.name], seams[d.name] = (d, h), s
    d, h, s, st = seam_strait()
    cases[d.name], seams[d.name], straits[d.name] = (d, h), s, st
    return cases, seams, straits


def _arm_seed(axis_deg: float, near: float, g: float) -> np.ndarray:
    """Seed on the axis leaving the origin along axis_deg (plane, deg), g beyond `near` (H0): a growth square of
    half-side g about it has its near edge across the axis at `near`."""
    a = np.radians(axis_deg)
    return (near + g) * np.array([np.cos(a), np.sin(a)])


def _hint(plane: Plane, seeds: list, strait_blocks: bool = False) -> dict:
    lon, lat = xyz_to_lonlat(plane.to_sphere(np.asarray(seeds, dtype=float)))
    return {"seeds": [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)], "grow_limit": HINT_GROW_H,
            "align": True, "strait_blocks": strait_blocks}


def _hints() -> dict:
    """SEAM_HINTS (module docstring): the m14 layout hint per case, seeds in (lon, lat)."""
    hints = {}
    for th in ROT_DEG:
        for k in K_PER100:
            be = 0.5 * th                                  # arm A along 180 + be, arm B along -be
            near = CORRIDOR_HALF / np.cos(np.radians(be))  # the corridor edge on each arm's axis
            hints[f"seam_rot{th:g}_k{k:g}"] = _hint(Plane(), [_arm_seed(180.0 + be, near, HINT_GROW_H),
                                                              _arm_seed(-be, near, HINT_GROW_H * (1 + k / 100))])
    near = HALF_W / np.tan(np.radians(60.0))               # the Y's inner corners on each arm's axis
    hints["seam_junction"] = _hint(Plane(), [_arm_seed(90.0 + 120.0 * i, near, HINT_GROW_H) for i in range(3)])
    hints["seam_coast"] = _hint(Plane(x_az=60.0), [(0.0, 8.0 + HINT_GROW_H)])          # the region 8-48 H0 off
    hints["seam_strait"] = _hint(Plane(x_az=60.0), [(0.0, -30.0 - HINT_GROW_H), (0.0, 30.0 + HINT_GROW_H)], True)
    return hints


SEAM_CASES, SEAMS, SEAM_STRAITS = _build()
SEAM_HINTS = _hints()
CASES.update(SEAM_CASES)
STRAITS.update(SEAM_STRAITS)


# ---------------------------------------------------------------------------------------------------------- runner

def seam_opts(method: str, case: str, opts: dict | None, hint: bool = True) -> dict | None:
    """The options a seam run passes: SEAM_HINTS[case] under the caller's opts for m14's methods (not _igm, which
    places its own blocks and ignores seeds), the caller's opts for every other method."""
    if hint and method.startswith("m14") and not method.startswith("m14_blocks_igm") and case in SEAM_HINTS:
        return {**SEAM_HINTS[case], **(opts or {})}
    return opts


def run_seams(methods: list, cases: list, scale: float = 1.0, *, smooth: bool = False, opts: dict | None = None,
              timeout: float = 1800.0, out: Path = SEAMS_JSONL, hint: bool = True) -> list[dict]:
    """compete.run_case for each method x case (its plain records go to .tmp/C/compete_records.jsonl; _score adds
    the seam block), with SEAM_HINTS[case] merged under `opts` for the m14 methods when `hint` (recorded in the
    record's opts); m14's block summary added when the run wrote m14.json; records appended to `out`."""
    from meshom import bench, compete

    tmp = Path(".tmp/C/compete_records.jsonl")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    recs_all = []
    for m in methods:
        for c in cases:
            r = compete.run_case(m, c, scale, smooth=smooth, opts=seam_opts(m, c, opts, hint), timeout=timeout,
                                 out=tmp)
            recs = [r] + ([r.pop("after_smoothing")] if "after_smoothing" in r else [])
            work = (compete.WORK_DIR / f"{m}_{c}_s{scale:g}").resolve()
            for rec in recs:
                if rec["status"] == "ok" and (work / "m14.json").exists():
                    j = json.loads((work / "m14.json").read_text())
                    kept = [j["blocks"][i] for i in j.get("blocks_kept", [])]
                    rec["m14"] = {"n_blocks": j.get("n_blocks"), "coverage_area": j.get("coverage_area"),
                                  "irregular_in_blocks": j.get("irregular_in_blocks"),
                                  "block_cells": [b.get("n_cells") for b in kept],
                                  "frame_turn_deg": [b.get("frame_turn_deg") for b in kept]}
                print(compete.one_line(rec), "| seam", json.dumps(rec.get("seam", {}).get("total")), flush=True)
            recs = [compete._round(x) for x in recs]
            bench.append_jsonl(recs, out)
            recs_all += recs
    return recs_all


def rescore(path: Path, out: Path) -> None:
    """Recompute the seam block of every record in `path` from the saved meshes in .tmp/compete (for changed
    measures), written to `out`; the seam size gate at the record's size tolerance (G33)."""
    from meshom import bench, compete
    from meshom.methods import effective_tolerance
    from meshom.quality.seams import tolerance_seam_gates
    from meshom.size import scaled

    recs = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        f = compete.WORK_DIR / f"{r['method']}_{r['case']}_s{r['scale']:g}" / ("smooth.npz" if r["smoothed"]
                                                                                  else "raw.npz")
        if r.get("status") == "ok" and f.exists():
            r["seam"] = compete._round(seam_block(compete._load(f), scaled(CASES[r["case"]][1], r["scale"]),
                                                  SEAMS[r["case"]],
                                                  tolerance_seam_gates(effective_tolerance(r["method"], r["opts"]))))
        recs.append(r)
    bench.append_jsonl(recs, out)


_GATE_SHORT = {"seam_excess_per10_max": "excess", "seam_ln_sh_p99_max": "size", "seam_quad_p99_9_max": "quad",
               "seam_ratio_p99_max": "ratio", "seam_centre_share_min": "centre", "seam_lake_excess_per100_max": "lake"}


def summary(path: Path, scale: float = 1.0, smoothed: bool = False) -> str:
    """Markdown: one row per seam case with records at this scale (raw or smoothed, the last record per method),
    one column per method in the order of the records. Columns "2 D" = twice the minimum dislocation count of the
    case's corridors and "min" = the lake's minimum gross (2 D + the cores' |net|); per method "corridor gross / lake
    gross", m14's block count in brackets, the failed seam gates after a colon; the last row counts the cases that
    pass every seam gate (decisions G32-G37)."""
    last = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["scale"] == scale and r["smoothed"] == smoothed and r["case"] in SEAMS:
                last[(r["method"], r["case"])] = r
    methods = list(dict.fromkeys(m for m, _ in last))
    cases = [c for c in SEAM_CASES if any(c == k for _, k in last)]
    lines = ["| case | 2 D | min | " + " | ".join(methods) + " |", "|---|---|---|" + "---|" * len(methods)]
    passed = dict.fromkeys(methods, 0)
    for c in cases:
        ok = [last[(m, c)] for m in methods if (m, c) in last and last[(m, c)]["status"] == "ok"]
        ref = ["-", "-"] if not ok else [f"{v:.1f}" if v else "0" for v in
                                        (2.0 * ok[0]["seam"]["total"]["disl_min_case"],
                                         ok[0]["seam"]["mesh"]["gross_min_case"])]
        cells = []
        for m in methods:
            r = last.get((m, c))
            if r is None or r["status"] != "ok":
                cells.append("—" if r is None else r["status"])
                continue
            sb, nb = r["seam"], (r.get("m14") or {}).get("n_blocks")
            passed[m] += not sb["gates_failed"]
            cells.append(f"{sb['total']['gross']} / {sb['mesh']['gross']}" + ("" if nb is None else f" [{nb}]")
                         + (": " + ", ".join(_GATE_SHORT.get(g, g) for g in sb["gates_failed"])
                            if sb["gates_failed"] else ""))
        lines.append(f"| {c.removeprefix('seam_')} | " + " | ".join(ref + cells) + " |")
    lines.append(f"| pass (of {len(cases)}) | | | " + " | ".join(str(passed[m]) for m in methods) + " |")
    return "\n".join(lines) + "\n"


def write_table(md_path: Path, text: str) -> None:
    """`text` between the seam-table markers of md_path (compete.write_table)."""
    from meshom.compete import write_table as _write

    _write(md_path, text, SEAM_TABLE_BEGIN, SEAM_TABLE_END)


def table(path: Path) -> str:
    """Markdown: one row per corridor of the last record per method x case x scale x smoothed."""
    last = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            last[(r["method"], r["case"], r["scale"], r["smoothed"])] = r
    head = ("method", "case", "corridor", "L cells", "D case / real", "θ real", "irr / 10", "gross / 10",
            "excess / 10", "centre", "|ln s/h| p99", "quad p99.9", "ratio p99", "seam gates failed", "mesh gross",
            "mesh failed families", "blocks")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    order = list(CASES)
    for key in sorted(last, key=lambda k: (k[0], order.index(k[1]) if k[1] in order else 99, k[2], k[3])):
        r = last[key]
        if r.get("status") != "ok":
            lines.append(f"| {r['method']} | {r['case']} | {r['status']} |" + " |" * (len(head) - 3))
            continue
        s = r["seam"]
        for b in s["corridors"]:
            f = (lambda v, nd=2: "-" if v is None else f"{v:.{nd}f}")
            lines.append("| " + " | ".join([
                r["method"] + (" (sm)" if r["smoothed"] else ""), r["case"], b["name"], f(b["length_cells"], 1),
                f"{b['disl_min_case']:.1f} / {b['disl_min_real']:.1f}", f(b["theta_real_deg"], 1),
                f(b["irregular_per10"]), f(b["gross_per10"]), f(b["excess_per10"]), f(b["centre_share"]),
                f(b["ln_sh_p99"], 3), f(b["quad_p99_9_deg"], 1), f(b["ratio_p99"], 3),
                ", ".join(k.replace("seam_", "") for k in b["gates_failed"]) or "—", str(s["mesh"]["gross"]),
                ", ".join(r.get("gates_failed_families", [])) or "—",
                str((r.get("m14") or {}).get("n_blocks", "-"))]) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m meshom.cases_seams")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--methods", nargs="+", required=True)
    r.add_argument("--cases", nargs="+", default=list(SEAM_CASES))
    r.add_argument("--scale", type=float, default=1.0)
    r.add_argument("--smooth", action="store_true")
    r.add_argument("--opts", type=json.loads, default=None)
    r.add_argument("--timeout", type=float, default=1800.0)
    r.add_argument("--jsonl", default=str(SEAMS_JSONL))
    r.add_argument("--no-hint", dest="hint", action="store_false", help="m14 without the layout hint SEAM_HINTS")
    t = sub.add_parser("table")
    t.add_argument("--jsonl", default=str(SEAMS_JSONL))
    t.add_argument("--summary", action="store_true", help="one row per case (summary) instead of one per corridor")
    t.add_argument("--scale", type=float, default=1.0, help="--summary: the scale")
    t.add_argument("--smoothed", action="store_true", help="--summary: the smoothed records")
    t.add_argument("--out", default=None, help="write between the seam-table markers of this Markdown file")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run_seams(a.methods, a.cases, a.scale, smooth=a.smooth, opts=a.opts, timeout=a.timeout, out=Path(a.jsonl),
                  hint=a.hint)
        return
    text = summary(Path(a.jsonl), a.scale, a.smoothed) if a.summary else table(Path(a.jsonl))
    if a.out:
        write_table(Path(a.out), text)
        print(f"table: {a.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
