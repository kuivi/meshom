"""Coastline for the UI: land polygons from a shapefile (with Antarctica from GSHHS L5 / L6 next to a GSHHS L1 source,
as the core adds it), filtered by island area, as display polylines, a land mask on the preview grid and a KD-tree for
the distance-to-coast layer.

Sources (`sources()`): every polygon shapefile next to the configured one (GSHHS levels and resolutions, named
GSHHS_<res>_L<level>.shp), and the Natural Earth / other data of `fesom2_mesh_gen` when it is found next to meshom.
GSHHS rings carry an `area` field (km^2); other shapefiles get the ring area computed on the sphere. Multi-part shapes
are split into their rings.

Edges that are not coast are left out of the display polylines and of the distance tree:
- GSHHS cuts polygons at the antimeridian: edges running along lon = +-180 are artificial;
- Natural Earth country polygons share their land borders: a vertex used by two different records is a border vertex
  (NE is topologically consistent, shared vertices are bitwise equal), and an edge between two border vertices is a
  border, not coast. Edges along lat = -90 (Antarctica's closing edge) are dropped too.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapefile
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M, lonlat_to_xyz, xyz_to_lonlat


@dataclass
class Coast:
    rings: list            # list of (K,2) lon/lat float64, closed (first == last), area >= min_area_km2
    area_km2: np.ndarray   # (n_rings,)
    tree: cKDTree          # 3D unit vectors of the real coast vertices
    n_points: int
    coast_edge: list       # per ring (K-1,) bool: the edge i -> i+1 is coast (not a cut, a border or the pole edge)

    def land_mask(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """(ny, nx) bool on a regular lon-lat grid of cell centres (south to north): polygon fill with PIL. The grid
        may extend past +-180 (a window across the antimeridian): rings are drawn again shifted by -+360 there."""
        from PIL import Image, ImageDraw
        nx, ny = len(lon), len(lat)
        dx = (lon[1] - lon[0]) if nx > 1 else 1.0
        dy = (lat[1] - lat[0]) if ny > 1 else 1.0
        img = Image.new("1", (nx, ny), 0)
        d = ImageDraw.Draw(img)
        lo, hi = lon[0] - dx / 2, lon[-1] + dx / 2
        shifts = [s for s in (-360.0, 0.0, 360.0) if lo < 180.0 + s and hi > -180.0 + s]
        for r in self.rings:
            for s in shifts:
                if r[:, 0].max() + s < lo or r[:, 0].min() + s > hi:
                    continue
                px = (r[:, 0] + s - lo) / dx
                py = (r[:, 1] - (lat[0] - dy / 2)) / dy
                d.polygon(list(zip(px.tolist(), py.tolist())), fill=1)
        return np.array(img, dtype=bool)

    def water_fraction(self, lon: np.ndarray, lat: np.ndarray, sub: int | None = None) -> np.ndarray:
        """(ny, nx) the water share of each cell of a regular lon-lat grid of cell centres: the land mask on a grid sub
        times finer (default: sub-cells of about 4 km, 4 to 16 per cell; PIL's fill also takes the pixels on a
        polygon's edge, so a strait loses about a sub-cell on each side). A strait narrower than a cell keeps its
        cells water (> 0), where land_mask's cell-centre test made them land (Gibraltar, 14 km, in the 0.5° preview).
        Cached per grid (the global preview asks again after every change)."""
        dx = (lon[1] - lon[0]) if len(lon) > 1 else 1.0
        sub = sub or int(np.clip(np.ceil(abs(dx) / 0.035), 4, 16))
        key = (len(lon), len(lat), float(lon[0]), float(lon[-1]), float(lat[0]), float(lat[-1]), sub)
        cache = self.__dict__.setdefault("_water", {})
        if key not in cache:
            dy = (lat[1] - lat[0]) if len(lat) > 1 else 1.0
            fx = lon[0] - dx / 2 + (np.arange(len(lon) * sub) + 0.5) * dx / sub
            fy = lat[0] - dy / 2 + (np.arange(len(lat) * sub) + 0.5) * dy / sub
            land = self.land_mask(fx, fy).reshape(len(lat), sub, len(lon), sub)
            if len(cache) > 8:
                cache.clear()
            cache[key] = 1.0 - land.sum(axis=(1, 3), dtype=np.uint16) / float(sub * sub)
        return cache[key]

    def polylines(self, spacing_m: float | None = None) -> list[np.ndarray]:
        """Display polylines (lon/lat), split where a ring runs along a non-coast edge; optionally resampled to
        spacing_m (great-circle steps, the coast "at the chosen h"; pieces shorter than 3 steps are dropped)."""
        out = []
        for r, ce in zip(self.rings, self.coast_edge):
            for p in _pieces(r, ce):
                if spacing_m:
                    p = resample(p, spacing_m)
                    if p is None:
                        continue
                out.append(p)
        return out


def _pieces(r: np.ndarray, coast_edge: np.ndarray) -> list[np.ndarray]:
    if coast_edge.all():
        return [r]
    keep = np.flatnonzero(coast_edge)
    if len(keep) == 0:
        return []
    breaks = np.flatnonzero(np.diff(keep) > 1) + 1
    return [r[np.r_[seg, seg[-1] + 1]] for seg in np.split(keep, breaks)]


def resample(p: np.ndarray, spacing_m: float) -> np.ndarray | None:
    """Open or closed lon/lat polyline resampled at equal great-circle steps close to spacing_m."""
    x = lonlat_to_xyz(p[:, 0], p[:, 1])
    seg = R_EARTH_M * np.arccos(np.clip(np.einsum("ij,ij->i", x[:-1], x[1:]), -1, 1))
    s = np.r_[0.0, np.cumsum(seg)]
    n = int(round(s[-1] / spacing_m))
    if n < 3:
        return None
    t = np.linspace(0.0, s[-1], n + 1)
    j = np.clip(np.searchsorted(s, t, side="right") - 1, 0, len(seg) - 1)
    f = np.clip((t - s[j]) / np.where(seg[j] > 0, seg[j], 1.0), 0, 1)[:, None]
    y = (1 - f) * x[j] + f * x[j + 1]
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    lon, lat = xyz_to_lonlat(y)
    if np.abs(p[[0, -1], 0]).max() == 180.0:      # keep the exact antimeridian sign at piece ends
        lon[0], lon[-1] = p[0, 0], p[-1, 0]
    return np.column_stack([lon, lat])


def ring_area_km2(r: np.ndarray) -> float:
    """Area of a closed lon/lat ring on the sphere (trapezoids in lon x sin(lat), exact for lat-lon edges)."""
    lon, lat = np.radians(r[:, 0]), np.radians(r[:, 1])
    dlon = np.diff(lon)
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi
    return abs(float(np.sum(dlon * (np.sin(lat[:-1]) + np.sin(lat[1:])) / 2))) * (R_EARTH_M / 1e3) ** 2


@lru_cache(maxsize=4)
def load(path: str, min_area_km2: float = 0.0, antarctica: str = "ice_front") -> Coast:
    """The rings of a land-polygon shapefile, plus Antarctica from the GSHHS L5 / L6 file next to a GSHHS L1 source as
    the core adds it (meshom.coast.antarctica_file; [coast] antarctica)."""
    from meshom.coast import antarctica_file
    extra = antarctica_file(path, antarctica)
    rings, areas, recs, ia = [], [], [], None
    for fi, fp in enumerate([path] + ([str(extra)] if extra else [])):
        r = shapefile.Reader(fp)
        names = [f.name for f in r.fields[1:]]
        ia = names.index("area") if "area" in names else None
        for k, sr in enumerate(r.iterShapeRecords()):
            pts = np.asarray(sr.shape.points, dtype=np.float64)
            parts = list(sr.shape.parts) + [len(pts)]
            for a0, a1 in zip(parts[:-1], parts[1:]):
                ring = pts[a0:a1]
                if len(ring) < 4:
                    continue
                a = float(sr.record[ia]) if ia is not None else ring_area_km2(ring)
                if a < min_area_km2:
                    continue
                rings.append(ring)
                areas.append(a)
                recs.append(fi * 10_000_000 + k)
    # border vertices: used by two different records (Natural Earth countries); GSHHS has none
    if ia is None and len(set(recs)) > 1:
        allv = np.vstack([q[:-1] for q in rings])
        key = allv[:, 0] + 1j * allv[:, 1]
        rid = np.repeat(recs, [len(q) - 1 for q in rings])
        uk, cnt = np.unique(np.unique(np.column_stack([key, rid]), axis=0)[:, 0], return_counts=True)
        multi = np.isin(key, uk[cnt > 1])
        offs = np.cumsum([0] + [len(q) - 1 for q in rings])
        border = [np.r_[multi[o0:o1], multi[o0]] for o0, o1 in zip(offs[:-1], offs[1:])]
    else:
        border = [np.zeros(len(q), bool) for q in rings]
    coast_edge = []
    for q, b in zip(rings, border):
        cut = (np.abs(q[:-1, 0]) == 180.0) & (np.abs(q[1:, 0]) == 180.0)
        pole = (q[:-1, 1] <= -89.99) & (q[1:, 1] <= -89.99)
        # GSHHS L5 / L6 Antarctica: halves cut along 0° and +-180°, closed by meridian edges down to the pole row
        to_pole = (q[:-1, 0] == q[1:, 0]) & ((q[:-1, 1] <= -89.99) | (q[1:, 1] <= -89.99))
        coast_edge.append(~(cut | pole | to_pole | (b[:-1] & b[1:])))
    pts = [p for q, ce in zip(rings, coast_edge) for p in _pieces(q, ce)]
    allp = np.vstack(pts) if pts else np.zeros((1, 2))
    tree = cKDTree(lonlat_to_xyz(allp[:, 0], allp[:, 1]))
    return Coast(rings, np.asarray(areas), tree, len(allp), coast_edge)


# ------------------------------------------------------------------------------------------------------ sources

def data_roots(source: Path) -> list[Path]:
    """Directories searched for coastline data: the configured source's folder, the coastline data next to meshom
    (../coastlines: GSHHG 2.3.7, every resolution and level) and fesom2_mesh_gen's data folders
    (../fesom2_mesh_gen/data or ../fesom2_mesh_gen/fesom2_mesh_gen/data)."""
    from ui.common.scenario import main_checkout
    base = main_checkout().parent                                                 # .../mesh_generation
    roots = [source.parent]
    for p in (base / "coastlines", base / "fesom2_mesh_gen" / "data",
              base / "fesom2_mesh_gen" / "fesom2_mesh_gen" / "data"):
        if p.is_dir():
            roots.append(p)
    return roots


GSHHS_RES = {"f": "full", "h": "high", "i": "intermediate", "l": "low", "c": "crude"}
GSHHS_LEVELS = {"1": "land: continents and islands (Antarctica added from L5 / L6 next to it, [coast] antarctica)",
                "2": "lakes", "3": "islands in lakes", "4": "ponds in islands in lakes",
                "5": "Antarctica, ice front (added to L1 by [coast] antarctica = \"ice_front\")",
                "6": "Antarctica, grounding line (added to L1 by [coast] antarctica = \"grounding\")"}


def _describe(p: Path) -> dict:
    name = p.stem
    d = {"path": str(p), "name": p.name, "usable": False, "kind": "other"}
    if p.suffix == ".shp":
        try:
            st = shapefile.Reader(str(p)).shapeType
        except Exception as e:       # an unreadable file is listed, not used
            return {**d, "note": f"unreadable: {e}"}
        poly = st in (shapefile.POLYGON, shapefile.POLYGONZ, shapefile.POLYGONM)
        if name.upper().startswith("GSHHS_"):
            res, lev = name.split("_")[1], name.split("_")[2][1:]
            d.update(kind="GSHHS", level=int(lev) if lev.isdigit() else None,
                     label=f"GSHHS {res} ({GSHHS_RES.get(res, res)}) L{lev}: {GSHHS_LEVELS.get(lev, '?')}",
                     usable=poly and lev == "1",
                     note=None if lev == "1" else ("comes in with L1 through [coast] antarctica" if lev in "56"
                                                   else "inland water: a later layer, not a coastline"))
        elif "lakes" in name:
            d.update(kind="Natural Earth lakes" if name.startswith("ne_") else "lakes", usable=False,
                     note="lakes: a later layer, not a coastline")
        elif name.startswith("ne_") and ("admin" in name or "countries" in name):
            d.update(kind="Natural Earth countries", label=f"Natural Earth {name[3:]}", usable=False,
                     note="country polygons, not a land mask: every border between two countries would be a coast "
                          "(and meshom cannot join them at 180°); use GSHHS L1 or Natural Earth land (ne_*_land)")
        elif name.startswith("ne_"):
            d.update(kind="Natural Earth", label=f"Natural Earth {name[3:]}", usable=poly,
                     note=None if poly else "not polygons")
        else:
            d.update(kind="shapefile", usable=poly, note=None if poly else "not polygons")
    elif p.suffix in (".nc", ".tif"):
        d.update(kind="raster", note="bathymetry / mask raster: not a coastline source yet")
    return d


def _order(f: Path) -> tuple:
    """GSHHS by level, then resolution from full to crude; everything else after, by path."""
    n = f.stem.split("_")
    if f.stem.upper().startswith("GSHHS_") and len(n) == 3:
        return (0, n[2], "fhilc".find(n[1]), str(f))
    return (1, "", 0, str(f))


def sources(source: Path) -> list[dict]:
    seen, files = set(), []
    for root in data_roots(source):
        for f in sorted(set(root.rglob("*.shp")) | set(root.rglob("*.nc"))):
            if f.resolve() not in seen and f.stat().st_size > 0:
                seen.add(f.resolve())
                files.append(f)
    return [_describe(f) for f in sorted(files, key=_order)]
