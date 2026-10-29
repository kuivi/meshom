"""The UI scenario: the meshom config as the page edits it, the size field it describes and the coarse h preview.

One schema: the config is the core's `meshom.config.Config` (meshom/config.py, docs/config.md; `meshom run CONFIG`
reads the same file). This module only
- loads and saves it with the core (`load`, `dumps`, `save`: meshom.config) and converts it to and from the page's
  JSON (`to_json`, `from_json`): JSON has no inf (sent as "inf") and JavaScript has one number type (100000.0 comes
  back as 100000); meshom.config.from_dict converts the typed tables ([[blocks]] included), `coerce` the free ones
  ([mesh] opts against the method's DEFAULTS, [gates] against DEFAULT_GATES);
- builds the size field with the core: the layers with meshom.run.layer_fields (cached per layer: a coast_distance
  layer builds its KD-tree once), the gradient-limited h with meshom.run.size_field (the batch run's h: clip, limiter
  on its cubed-sphere grid, scale; cached per config hash, so only the first limited preview pays for the limiter);
- evaluates it on a lon-lat preview grid of cell centres (display and node estimate only: exact cell areas, the
  integral of h^-2 over the water) and colours it.
"""
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from meshom import config, run
from meshom.config import DEFAULT_GATES, Config, ZoneLayer
from meshom.geo import R_EARTH_M, lonlat_to_xyz
from meshom.size import Clipped, Layered, SizeField, layered, polygon_inside, scaled

UI_DIR = Path(__file__).resolve().parents[1]
REPO = UI_DIR.parent
EXAMPLE = UI_DIR / "common" / "scenario.toml"
SAVE_DEFAULT = REPO / ".tmp" / "ui" / "config.toml"     # where a page started on EXAMPLE saves: EXAMPLE stays as it is
DEFAULT_COAST = "../oceanmesh/tests/GSHHS_i_L1.shp"     # the coast panel's data folder when [coast] source = "none"


def main_checkout() -> Path:
    """Root of the main checkout (differs from REPO inside a git worktree)."""
    out = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=REPO, capture_output=True, text=True)
    return (REPO / out.stdout.strip()).resolve().parent if out.returncode == 0 else REPO


def resolve(path: str) -> Path:
    """A config path: absolute, or relative to the repository root (the main checkout's root inside a worktree)."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    for base in (REPO, main_checkout()):
        if (base / p).exists():
            return (base / p).resolve()
    return REPO / p


# ------------------------------------------------------------------------------------------------ config I/O

def load(path: Path = EXAMPLE) -> Config:
    return config.load(path)


def dumps(cfg: Config) -> str:
    return config.dumps(cfg)


def save(cfg: Config, path: Path) -> None:
    config.save(cfg, path)


def dumps_from(cfg: Config, variant: Path, folder: Path) -> str:
    """The config's TOML with [size] = only `from` = the h variant file (relative to the config's folder when it can
    be): the core reads the variant's [size] and layers (S35); the page writes this when its h is that variant."""
    import os
    import tomli_w
    d = config.to_dict(cfg)
    v = Path(variant).resolve()
    try:
        rel = os.path.relpath(v, Path(folder).resolve())
    except ValueError:
        rel = str(v)
    d["size"] = {"from": rel}
    return tomli_w.dumps(d)


def bathymetry_source(cfg: Config) -> Path | None:
    """The bottom grid of the page's profiles: the config's [bathymetry] source, else GEBCO 2024 next to the repository
    (../bathymetry, or fesom2_mesh_gen's copy) when there; None when none is found."""
    if cfg.bathymetry.source != "none":
        p = resolve(cfg.bathymetry.source)
        return p if p.exists() else None
    mg = main_checkout().parent
    for p in (mg / "bathymetry" / "GEBCO_2024.nc", mg / "fesom2_mesh_gen" / "fesom2_mesh_gen" / "data" / "gebco2024" / "GEBCO_2024.nc"):
        if p.exists():
            return p
    return None


def size_variants(folders: list[Path]) -> list[dict]:
    """The h variant files in the folders: *.toml holding only [size] (newest first): {path, name, layers}."""
    import tomllib
    out, seen = [], set()
    for f in folders:
        for p in sorted(Path(f).glob("*.toml"), key=lambda q: -q.stat().st_mtime) if Path(f).is_dir() else []:
            p = p.resolve()
            if p in seen:
                continue
            seen.add(p)
            try:
                d = tomllib.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if set(d) == {"size"}:
                out.append({"path": str(p), "name": p.name, "layers": len(d["size"].get("layers", []))})
    return out


def jsonable(v):
    """JSON-safe copy: numpy scalars to Python, non-finite floats to "inf" / "-inf" / "nan" (TOML has them)."""
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else str(float(v))
    if v is None or isinstance(v, (bool, int, str)):
        return v
    return repr(v)


def to_json(cfg: Config) -> dict:
    return jsonable(config.to_dict(cfg))


def coerce(new, old=None):
    """Undo what JSON in the browser does to TOML values: a number becomes a float where `old` (the same place in the
    loaded config, the method's DEFAULTS or DEFAULT_GATES) holds a float, and "inf" / "-inf" / "nan" become floats
    where `old` holds a float (the page shows non-finite floats as text)."""
    if isinstance(new, dict):
        o = old if isinstance(old, dict) else {}
        return {k: coerce(v, o.get(k)) for k, v in new.items()}
    if isinstance(new, list):
        o = old if isinstance(old, list) and len(old) == len(new) else [None] * len(new)
        return [coerce(v, ov) for v, ov in zip(new, o)]
    if isinstance(new, int) and not isinstance(new, bool) and isinstance(old, float):
        return float(new)
    if isinstance(new, str) and isinstance(old, float) and new in ("inf", "-inf", "nan"):
        return float(new)
    return new


def method_defaults(name: str) -> dict:
    import sys

    from meshom.methods import METHODS
    return dict(getattr(sys.modules[METHODS[name].__module__], "DEFAULTS", {}) or {})


def coerce_config(new: dict, old: dict) -> dict:
    """coerce() for a whole config dict: the loaded config's values, [mesh] opts against the chosen method's DEFAULTS,
    [gates] against DEFAULT_GATES. The typed tables are converted again by meshom.config.from_dict."""
    out = coerce(new, old)
    mesh = out.get("mesh", {})
    if mesh.get("opts"):
        try:
            mesh["opts"] = coerce(mesh["opts"], method_defaults(mesh["method"]))
        except KeyError:            # a method name the registry does not know: leave the options as they came
            pass
    if out.get("gates"):
        out["gates"] = coerce(out["gates"], DEFAULT_GATES)
    return out


def from_json(d: dict, loaded: Config | None = None) -> Config:
    """The page's config (JSON) as a core Config, typed as the loaded config (meshom.config.from_dict)."""
    old = to_json(loaded) if loaded is not None else {}
    return config.from_dict(coerce_config(d, old), loaded.name if loaded is not None else "config")


# --------------------------------------------------------------------------------------------- size field

def core_coast(cfg: Config):
    """The core's coast of the config (meshom.coast.read_coast, cached there), the path resolved as the UI does."""
    from meshom.coast import read_coast
    if cfg.coast.source == "none":
        return None
    return read_coast(str(resolve(cfg.coast.source)), cfg.coast.min_area_km2, cfg.coast.antarctica)


def active(cfg: Config) -> list[tuple[int, object]]:
    """(position in [[size.layers]], layer) of the layers a preview evaluates: enabled, and zones with >= 3 vertices,
    a "blend" zone also in its outer polygon (a zone being drawn is skipped)."""
    return [(i, ly) for i, ly in enumerate(cfg.size.layers)
            if ly.enabled and not (isinstance(ly, ZoneLayer) and (len(ly.polygon) < 3
                                                                 or ly.mode == "blend" and len(ly.outer) < 3))]


def effective(cfg: Config) -> Config:
    """cfg with only the active layers (what a batch run of the page's config would combine)."""
    return replace(cfg, size=replace(cfg.size, layers=[ly for _, ly in active(cfg)]))


_FIELDS: dict = {}


def layer_fields(cfg: Config, coast=None) -> tuple[list[str], list[int], list[SizeField]]:
    """(names, positions in [[size.layers]], core fields) of the active layers. Each field is built by
    meshom.run.layer_fields on a one-layer config with the same base h, and cached by the layer's parameters, the
    base h and (coast_distance) the coast."""
    base = cfg.base_h()
    names, index, fields = [], [], []
    for i, ly in active(cfg):
        coast_key = ((cfg.coast.source, cfg.coast.min_area_km2, cfg.coast.antarctica) if ly.kind == "coast_distance"
                     else None)
        key = (repr(ly), base, coast_key)
        if key not in _FIELDS:
            one = replace(cfg, size=replace(cfg.size, layers=[ly], h_max_m=base))
            if coast_key and coast is None:
                coast = core_coast(cfg)
            _FIELDS[key] = run.layer_fields(one, coast)[0][1]
            while len(_FIELDS) > 64:
                _FIELDS.pop(next(iter(_FIELDS)))
        names.append(ly.name or ly.kind)
        index.append(i)
        fields.append(_FIELDS[key])
    return names, index, fields


_LIMITED: dict = {}


def config_hash(cfg: Config) -> str:
    """Hash of what the run's h depends on: the active layers, the [size] keys but preview_deg, the coast, the scale."""
    d = config.to_dict(effective(cfg))
    d["size"].pop("preview_deg", None)
    key = {"size": d["size"], "coast": d["coast"], "scale": cfg.mesh.scale}
    return hashlib.sha1(json.dumps(jsonable(key), sort_keys=True).encode()).hexdigest()[:16]


def limited_field(cfg: Config, coast=None) -> tuple[SizeField, dict]:
    """The batch run's h (meshom.run.size_field: clip, gradient limiter, scale) and its record (limiter grid, node
    estimate, build seconds, cached or not); cached per config_hash (the last 2 configs: ~150 MB each at the
    limiter's 12 M node cap)."""
    key = config_hash(cfg)
    if key in _LIMITED:
        h, info = _LIMITED[key]
        return h, dict(info, cached=True)
    t0 = time.perf_counter()
    info: dict = {}
    eff = effective(cfg)
    h = run.size_field(eff, coast if coast is not None else core_coast(eff), info)
    info.update(seconds=round(time.perf_counter() - t0, 3), hash=key)
    _LIMITED[key] = (h, info)
    while len(_LIMITED) > 2:
        _LIMITED.pop(next(iter(_LIMITED)))
    return h, dict(info, cached=False)


_LAYERED: dict = {}


def running_h(cfg: Config) -> tuple[SizeField, list[str], list[int]]:
    """(the layers' h before clip and scale, names, positions in [[size.layers]]): meshom.size.layered over the active
    layers in file order (min; zone modes set / smooth on the running h, which starts at the base h when the first
    layer is such a zone), as meshom.run.background_h. Cached per config_hash: a smooth zone binds a cubed-sphere
    grid (seconds), and the page asks again after every change."""
    names, index, fields = layer_fields(cfg)
    if not fields:                                  # the page's "use this layer" off on every layer
        raise ValueError("every h layer is off: h is not defined; switch one on (\"use this layer\")")
    key = config_hash(cfg)
    if key not in _LAYERED:
        base = cfg.base_h()
        _LAYERED[key] = layered(fields, np.inf if base is None else base, 0.5 * run.h_floor(effective(cfg)))
        while len(_LAYERED) > 4:
            _LAYERED.pop(next(iter(_LAYERED)))
    return _LAYERED[key], names, index


def _steps(f: SizeField, xyz: np.ndarray, n_layers: int, base_pos: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(h, the position among the active layers of the layer that set it last) on xyz: for a Layered field the core's
    Layered.trace (a point no step changed keeps the base h: base_pos, the layer that gives it); else (min only) the
    smallest layer."""
    if isinstance(f, Layered):
        h, step = f.trace(xyz)
        return h, np.where(step < 0, base_pos, step)
    fs = list(f.fields) if n_layers > 1 else [f]
    stack = np.stack([g(xyz) for g in fs])
    win = np.argmin(stack, axis=0)
    return stack[win, np.arange(stack.shape[1])], win


def h_field(cfg: Config, limited: bool = False) -> SizeField:
    """The config's h as one SizeField: limited = the batch run's h (limited_field); else the same chain without the
    limiter (the layers in order: running_h, clip to [size] h_min_m / h_max_m, x [mesh] scale)."""
    if limited:
        return limited_field(cfg)[0]
    h = running_h(cfg)[0]
    if cfg.size.h_min_m is not None or cfg.size.h_max_m is not None:
        h = Clipped(h, cfg.size.h_min_m or 0.0, np.inf if cfg.size.h_max_m is None else cfg.size.h_max_m)
    return scaled(h, cfg.mesh.scale) if cfg.mesh.scale != 1.0 else h


# ------------------------------------------------------------------------------------------------------- preview

@dataclass
class Preview:
    lon: np.ndarray          # (nx,) cell-centre longitudes, deg (a window may run past 180)
    lat: np.ndarray          # (ny,) cell-centre latitudes, deg, south to north
    h: np.ndarray            # (ny, nx) metres
    water: np.ndarray        # (ny, nx) bool (False outside a window polygon too)
    winner: np.ndarray       # (ny, nx) index into names / layer_index of the layer that sets h (before the limiter)
    names: list
    n_nodes_est: float       # integral of h^-2 over water (quads: nodes ~ elements)
    grad_max: float          # max |grad h| over water, finite differences on the grid
    seconds: float
    layer_index: list = None  # position of each evaluated layer in [[size.layers]]
    grad_at: tuple = None     # (lon, lat) of grad_max
    inside: np.ndarray = None  # (ny, nx) bool: inside the window polygon (all True without one)
    limiter: dict = None      # limited_field's record when the limiter was applied


def grid(step_deg: float, bounds: tuple | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cell centres of a lon-lat grid: global, or inside bounds = (west, east, south, north) with east > west
    (east may exceed 180 for a window across the antimeridian)."""
    w, e, s, n = bounds or (-180.0, 180.0, -90.0, 90.0)
    nx, ny = max(int(round((e - w) / step_deg)), 1), max(int(round((n - s) / step_deg)), 1)
    return w + (np.arange(nx) + 0.5) * (e - w) / nx, s + (np.arange(ny) + 0.5) * (n - s) / ny


def preview(cfg: Config, coast, step_deg: float | None = None, bounds: tuple | None = None,
            polygon: list | None = None, limited: bool = False) -> Preview:
    """h on the lon-lat preview grid (global, or a window with its own step and an optional polygon [[lon, lat], ...]),
    the layer that wins, the node estimate and the max gradient. `coast` is the UI coast (land mask) or None (no land).
    h is the active layers' h in file order (running_h: min, zone modes set / smooth), clipped and scaled as the run
    does; limited: the run's h itself (limiter). The winner is the layer that set h last there."""
    t0 = time.perf_counter()
    step = step_deg or cfg.size.preview_deg
    lon, lat = grid(step, bounds)
    LON, LAT = np.meshgrid(lon, lat)
    xyz = lonlat_to_xyz(LON.ravel(), LAT.ravel())
    f, names, index = running_h(cfg)
    base, act = cfg.base_h(), [ly for _, ly in active(cfg)]
    base_pos = next((k for k, ly in enumerate(act) if ly.kind == "constant" and ly.h_m == base), 0)
    h0, win = _steps(f, xyz, len(names), base_pos)
    info = None
    if limited:
        hf, info = limited_field(cfg)
        h = hf(xyz).reshape(LON.shape)
    else:
        h = h0
        if cfg.size.h_min_m is not None or cfg.size.h_max_m is not None:
            h = np.clip(h, cfg.size.h_min_m or 0.0, np.inf if cfg.size.h_max_m is None else cfg.size.h_max_m)
        h = (h * cfg.mesh.scale).reshape(LON.shape)
    frac = coast.water_fraction(lon, lat) if coast is not None else np.ones(LON.shape)    # narrow straits stay water
    water = frac > 0
    inside = np.ones_like(water)
    if polygon is not None and len(polygon) >= 3:
        p = np.asarray(polygon, dtype=np.float64)
        inside = polygon_inside(xyz, lonlat_to_xyz(p[:, 0], p[:, 1])).reshape(LON.shape)
        water &= inside
    dl = np.radians(lon[1] - lon[0]) if len(lon) > 1 else np.radians(step)
    dp = np.radians(lat[1] - lat[0]) if len(lat) > 1 else np.radians(step)
    area = R_EARTH_M**2 * dl * (np.sin(np.radians(lat) + dp / 2) - np.sin(np.radians(lat) - dp / 2))
    n_est = float(np.sum(water * frac * area[:, None] / h**2))          # the water share of each cell
    gy = np.diff(h, axis=0) / (R_EARTH_M * dp)
    gx = np.diff(h, axis=1) / (R_EARTH_M * dl * np.cos(np.radians(lat))[:, None])
    g = np.zeros_like(h)
    g[:-1] = np.maximum(g[:-1], np.abs(gy))
    g[:, :-1] = np.maximum(g[:, :-1], np.abs(gx))
    gw = np.where(water, g, -1.0)
    k = np.unravel_index(int(np.argmax(gw)), g.shape)
    at = (float((lon[k[1]] + 180.0) % 360.0 - 180.0), float(lat[k[0]]))
    return Preview(lon, lat, h, water, win.reshape(LON.shape), names, n_est,
                   float(g[water].max()) if water.any() else 0.0, time.perf_counter() - t0, index, at, inside, info)


def h_rgba(p: Preview, h_lo: float | None = None, h_hi: float | None = None) -> np.ndarray:
    """(ny, nx, 4) uint8, north row first: log h through a perceptual colormap, land half-transparent grey,
    transparent outside a window polygon."""
    from matplotlib import colormaps
    lo = h_lo or float(p.h[p.water].min() if p.water.any() else p.h.min())
    hi = h_hi or float(p.h[p.water].max() if p.water.any() else p.h.max())
    t = (np.log(p.h) - np.log(lo)) / max(np.log(hi) - np.log(lo), 1e-9)
    rgba = (colormaps["viridis"](np.clip(t, 0, 1)) * 255).astype(np.uint8)
    rgba[~p.water] = (90, 90, 90, 255)
    if p.inside is not None:
        rgba[~p.inside] = 0
    return rgba[::-1]


# categorical colours of the "which layer wins" map, by position in [[size.layers]] (also used by the page legend)
LAYER_COLOURS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189), (140, 86, 75),
                 (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]


def winner_rgba(p: Preview) -> np.ndarray:
    """(ny, nx, 4) uint8, north row first: the colour of the layer that sets h (by its position in [[size.layers]]),
    land dark."""
    cols = np.array([LAYER_COLOURS[i % len(LAYER_COLOURS)] + (255,) for i in p.layer_index], np.uint8)
    rgba = cols[p.winner]
    rgba[~p.water] = (60, 60, 60, 255)
    if p.inside is not None:
        rgba[~p.inside] = 0
    return rgba[::-1]


def png_bytes(rgba: np.ndarray) -> bytes:
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba)).save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()
