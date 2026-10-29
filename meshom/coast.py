"""Coastline from a polygon shapefile of land (GSHHS level 1, Natural Earth land, ...) as domain loops on the sphere.

- Rings follow the shapefile convention: an outer ring runs clockwise in lon-lat (land on its right), a hole
  counter-clockwise, so every ring already has water on its LEFT (meshom.domain); GSHHS L1 was checked (all rings
  clockwise). Rings are kept as given, not reoriented.
- Polygons cut at the antimeridian (GSHHS: "0-E" / "0-W" for Eurasia-Africa and a few islands) carry artificial edges
  along lon = +-180; they are removed and the pieces are chained end to start at equal latitude, so every land mass is
  one closed loop again. A ring left open by that chaining is an error (not met in GSHHS).
- `min_area_km2` drops polygons below the area (the shapefile field "area" in km^2 when present, as GSHHS; else the
  spherical area of the ring).
- Antarctica: GSHHS L1 leaves it out (GSHHG >= 2.3 keeps it in L5 = ice front and L6 = grounding line). For a source
  named GSHHS_<res>_L1.shp the polygons of GSHHS_<res>_L5.shp (antarctica = "ice_front", the default: FESOM2's
  practice, no ice-shelf cavities) or _L6.shp ("grounding") next to it are added; "none" adds nothing, and so does a
  missing L5 / L6 file (Coast.antarctica says what happened). The continent comes as two halves cut along 0° and
  +-180° and closed through the pole row (lat -90): the pole vertices and the meridian edges to them are dropped and
  the two coastal chains (each running east, water on the left) are joined end to start into one loop around the
  South Pole, so in_water and the land cut see the pole as land.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from meshom.domain import Domain, resample_loops
from meshom.geo import R_EARTH_M, arc_length, lonlat_to_xyz
from meshom.size import SizeField


@dataclass
class Coast:
    loops: list          # closed loops (K,3) unit vectors, water on the left, raw shapefile vertices
    area_km2: np.ndarray  # (n_loops,) area of the land polygon each loop came from
    antarctica: str = "none"   # what was added for Antarctica (the L5 / L6 file and its loop count, or why not)

    def points(self, max_step_m: float | None = None) -> np.ndarray:
        """(P,3) all loop vertices; with max_step_m, extra points on every segment longer than that (for distances)."""
        out = []
        for lp in self.loops:
            if max_step_m:
                nxt = np.roll(lp, -1, axis=0)
                k = np.maximum(1, np.ceil(R_EARTH_M * arc_length(lp, nxt) / max_step_m).astype(np.int64))
                if k.max() > 1:
                    s = np.repeat(np.arange(len(lp)), k)
                    t = (np.arange(len(s)) - np.repeat(np.cumsum(k) - k, k)) / np.repeat(k, k)
                    lp = (1.0 - t)[:, None] * lp[s] + t[:, None] * nxt[s]
                    lp /= np.linalg.norm(lp, axis=1, keepdims=True)
            out.append(lp)
        return np.vstack(out) if out else np.zeros((0, 3))

    def loops_at(self, h: SizeField) -> list[np.ndarray]:
        """The loops resampled at spacing h (equal steps of the integral of ds/h, no corners kept), loops with fewer
        than 3 steps dropped (islands smaller than the local h). Two coast stretches closer than h can cross after
        this: real coasts need the coast filter (meshom.coast_filter) first."""
        keep = []
        for lp in self.loops:
            hv = h(lp)
            nxt = np.roll(lp, -1, axis=0)
            w = R_EARTH_M * arc_length(lp, nxt) * 0.5 * (1.0 / hv + 1.0 / np.roll(hv, -1))
            if round(float(w.sum())) >= 3:
                keep.append(lp)
        return resample_loops(Domain(keep), h, corner_deg=180.0)

    def domain(self) -> Domain:
        """The raw loops as a Domain (in_water tests against the true coast)."""
        return Domain(list(self.loops), "coast")

    def islands(self, island_max_km2: float) -> np.ndarray:
        """(n_loops,) bool: the loop bounds an island, a land polygon of less than island_max_km2 on the sphere
        (loop_land_km2); the loop around the South Pole (Antarctica) is a continent whatever its area (S33)."""
        north = np.array([0.0, 0.0, 1.0])            # fan from the North Pole: > 0 when the land holds the South Pole
        return np.array([loop_land_km2(lp) < island_max_km2 and _fan_sr(lp, north) < 0.0 for lp in self.loops],
                        dtype=bool)

    def select(self, coasts: str, island_max_km2: float) -> "Coast":
        """The loops of one class: "all" (self), "continents" or "islands" (Coast.islands), as a Coast."""
        if coasts == "all":
            return self
        keep = self.islands(island_max_km2) == (coasts == "islands")
        return Coast([lp for lp, k in zip(self.loops, keep) if k], self.area_km2[keep], self.antarctica)


def _fan_sr(loop: np.ndarray, r: np.ndarray) -> float:
    """Sum of the signed areas (sr) of the spherical triangles (r, a, b) over the loop's edges a -> b (Van Oosterom
    and Strackee: tan(E/2) = r.(a x b) / (1 + r.a + a.b + b.r)): the area of the side of the loop that does not hold
    -r, positive when that side lies left of the loop (water), negative when right of it (land)."""
    a, b = loop, np.roll(loop, -1, axis=0)
    num = np.cross(a, b) @ r
    den = 1.0 + a @ r + b @ r + np.einsum("ij,ij->i", a, b)
    return float(2.0 * np.arctan2(num, den).sum())


def loop_land_km2(loop: np.ndarray) -> float:
    """Area (km^2) of the land right of a loop (water on the left), great-circle edges on the sphere of R_EARTH_M:
    minus the fan from the normalised vertex mean m, modulo the sphere (exact whichever side holds -m)."""
    m = loop.mean(axis=0)
    sr = (-_fan_sr(loop, m / np.linalg.norm(m))) % (4.0 * np.pi)
    return float(sr * (R_EARTH_M / 1e3) ** 2)


def _ring_area_km2(ll: np.ndarray) -> float:
    """Spherical area of a closed lon-lat ring, |sum (dlon) (sin lat1 + sin lat2) / 2| R^2 (edges as rhumb steps)."""
    lon, lat = np.radians(ll[:, 0]), np.radians(ll[:, 1])
    dl = np.diff(lon)
    dl = (dl + np.pi) % (2 * np.pi) - np.pi
    return float(abs(np.sum(dl * 0.5 * (np.sin(lat[:-1]) + np.sin(lat[1:])))) * (R_EARTH_M / 1e3) ** 2)


def _chains(ring: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """A closed lon-lat ring (first == last) -> ([the ring without its closing point], []) when it has no edge along
    lon = +-180, else ([], open chains between such edges, in ring order)."""
    p = ring[:-1] if np.array_equal(ring[0], ring[-1]) else ring
    on = np.abs(p[:, 0]) == 180.0
    art = on & np.roll(on, -1)                                    # edge q -> q+1 (cyclic) runs along the cut
    if not art.any():
        return [p], []
    s = (int(np.flatnonzero(art)[0]) + 1) % len(p)
    q, a = np.roll(p, -s, axis=0), np.roll(art, -s)               # a[-1] is True: chains end before the wrap
    out, start = [], 0
    for i in np.flatnonzero(a):
        if i > start:
            out.append(q[start:i + 1])
        start = i + 1
    return [], out


def _join(chains: list[np.ndarray]) -> list[tuple[np.ndarray, int]]:
    """Chain open pieces whose end (lon +-180, lat) is another's start; each cycle becomes one closed ring, returned
    with the index of its first piece."""
    key = lambda ll: round(float(ll[1]), 7)                       # noqa: E731
    start = {}
    for c, ch in enumerate(chains):
        start.setdefault(key(ch[0]), []).append(c)
    used = np.zeros(len(chains), dtype=bool)
    rings = []
    for c0 in range(len(chains)):
        if used[c0]:
            continue
        parts, c = [], c0
        while not used[c]:
            used[c] = True
            parts.append(chains[c][:-1])                          # the end equals the next chain's start
            nxt = [x for x in start.get(key(chains[c][-1]), []) if not used[x] or x == c0]
            if not nxt:
                raise ValueError(f"coast piece ending at {chains[c][-1].tolist()} does not continue across 180°")
            c = nxt[0]
        rings.append((np.vstack(parts), c0))
    return rings


def _to_loop(ll: np.ndarray) -> np.ndarray | None:
    x = lonlat_to_xyz(ll[:, 0], ll[:, 1])
    keep = np.linalg.norm(x - np.roll(x, 1, axis=0), axis=1) > 1e-12       # drop repeated points
    x = x[keep]
    return x if len(x) >= 3 else None


POLE_LAT = -89.9999            # a vertex at or below this latitude is on the pole row of a polygon closed via the pole
ANTARCTICA = {"ice_front": "L5", "grounding": "L6"}


def _pole_chain(ring: np.ndarray) -> np.ndarray | None:
    """A closed lon-lat ring (first == last) that runs through the South Pole -> its open coastal chain: the vertices
    after the run of pole vertices up to the one before it (cyclic), i.e. without the pole row and the meridian edges
    to it; None if the ring does not touch the pole."""
    p = ring[:-1] if np.array_equal(ring[0], ring[-1]) else ring
    pole = p[:, 1] <= POLE_LAT
    if not pole.any():
        return None
    last = int(np.flatnonzero(pole)[-1])
    q, m = np.roll(p, -(last + 1), axis=0), np.roll(pole, -(last + 1))     # q starts right after a pole vertex
    return q[: int(np.flatnonzero(m)[0])]                                  # up to the next pole vertex


def _join_pole_chains(chains: list[np.ndarray]) -> list[np.ndarray]:
    """Open chains (lon-lat) whose end is another's start on the sphere (0°, +-180° cuts) -> closed loops (K,3)."""
    xs = [lonlat_to_xyz(c[:, 0], c[:, 1]) for c in chains]
    used, loops = np.zeros(len(xs), dtype=bool), []
    for c0 in range(len(xs)):
        if used[c0]:
            continue
        parts, c = [], c0
        while not used[c]:
            used[c] = True
            parts.append(xs[c][:-1])                                       # the end is the next chain's start
            d = [np.linalg.norm(x[0] - xs[c][-1]) for x in xs]
            c = int(np.argmin(d))
            if d[c] > 1e-9:
                raise ValueError(f"Antarctic piece ending at {chains[c0][-1].tolist()} does not continue")
        loops.append(np.vstack(parts))
    return loops


def antarctica_file(path: str, antarctica: str) -> Path | None:
    """GSHHS_<res>_L5.shp / _L6.shp next to a GSHHS_<res>_L1.shp source (None: not GSHHS L1, "none", or missing)."""
    p = Path(path)
    if antarctica == "none" or not (p.name.upper().startswith("GSHHS_") and p.stem.upper().endswith("_L1")):
        return None
    f = p.with_name(p.stem[:-1] + ANTARCTICA[antarctica][1] + p.suffix)
    return f if f.exists() else None


@lru_cache(maxsize=4)
def read_coast(path: str, min_area_km2: float = 0.0, antarctica: str = "ice_front") -> Coast:
    """Land polygons of a shapefile -> Coast (closed loops, water on the left, merged across the antimeridian), plus
    Antarctica from the GSHHS L5 / L6 file next to an L1 source (module docstring)."""
    c = _read_rings(path, min_area_km2)
    f = antarctica_file(path, antarctica)
    if f is None:
        why = "not asked" if antarctica == "none" else (
            f"no GSHHS {ANTARCTICA[antarctica]} file next to {Path(path).name}"
            if Path(path).name.upper().startswith("GSHHS_") else "not a GSHHS L1 source")
        return Coast(c.loops, c.area_km2, f"none ({why})")
    a = _read_rings(str(f), min_area_km2)
    return Coast(c.loops + a.loops, np.r_[c.area_km2, a.area_km2],
                 f"{antarctica}: {f.name}, {len(a.loops)} loops, the largest {a.area_km2.max():.4g} km^2")


def _read_rings(path: str, min_area_km2: float) -> Coast:
    import shapefile

    r = shapefile.Reader(str(path))
    names = [f.name for f in r.fields[1:]]
    ia = names.index("area") if "area" in names else None
    closed, chains, areas, chain_areas, pole_chains, pole_area = [], [], [], [], [], 0.0
    for sr in r.iterShapeRecords():
        pts = np.asarray(sr.shape.points, dtype=np.float64)
        parts = list(sr.shape.parts) + [len(pts)]
        for a, b in zip(parts[:-1], parts[1:]):
            ring = pts[a:b]
            area = float(sr.record[ia]) if ia is not None else _ring_area_km2(ring)
            if area < min_area_km2:
                continue
            pc = _pole_chain(ring)
            if pc is not None:                     # Antarctica (GSHHS L5 / L6): a piece of the loop around the pole
                pole_chains.append(pc)
                pole_area = max(pole_area, area)    # GSHHS gives both halves the continent's area
                continue
            c, o = _chains(ring)
            closed += c
            areas += [area] * len(c)
            chains += o
            chain_areas += [area] * len(o)
    loops, out_area = [], []
    for ring, area in zip(closed, areas):
        x = _to_loop(ring)
        if x is not None:
            loops.append(x)
            out_area.append(area)
    for ring, c0 in _join(chains) if chains else []:
        x = _to_loop(ring)
        if x is not None:
            loops.append(x)
            out_area.append(chain_areas[c0])
    for x in _join_pole_chains(pole_chains) if pole_chains else []:
        keep = np.linalg.norm(x - np.roll(x, 1, axis=0), axis=1) > 1e-12
        loops.append(x[keep])
        out_area.append(pole_area)
    return Coast(loops, np.asarray(out_area))
