# SPDX-License-Identifier: GPL-3.0-or-later
"""The cell size of an ORCA grid as a size field (`[[size.layers]] kind = "orca"`, decisions B101).

h(x) = max(sqrt(cell area), h_min_m) of the ORCA cells nearest to x, interpolated by inverse squared distance over the
K_NEAR nearest T points (continuous; at a T point its own cell's size). The cell area is that of the quad of its four
F corners (two triangles of chords, on the sphere of radius R_EARTH_M). The grid is the preset's own (no south rows
added): south of its first row the nearest cells of that row hold. Used as the running h of an ORCA run, so the cap
of m16_orca meets the ORCA cells at their size and the scores measure the ORCA part against its own cells.
"""
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M, lonlat_to_xyz

K_NEAR = 4          # B101: T points in the inverse-distance interpolation


@dataclass(frozen=True, eq=False)
class OrcaSize:
    tree: cKDTree
    h: np.ndarray            # (n,) sqrt(area) of each T point's cell, m
    h_min_m: float
    name: str

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        out = np.empty(len(xyz))
        for a in range(0, len(xyz), 2_000_000):
            d, k = self.tree.query(xyz[a:a + 2_000_000], k=K_NEAR, workers=-1)
            w = 1.0 / np.maximum(d, 1e-12) ** 2
            out[a:a + 2_000_000] = (w * self.h[k]).sum(axis=1) / w.sum(axis=1)
        return np.maximum(out, self.h_min_m)


def cell_sizes(g) -> tuple[np.ndarray, np.ndarray]:
    """(T points (n, 3), sqrt(cell area) (n,) m) of the grid's cells T rows 1 .. j_pole - 1 (unique columns)."""
    fx, fy = g.lon["f"][:, :-2], g.lat["f"][:, :-2]
    F = lonlat_to_xyz(fx.ravel(), fy.ravel()).reshape(fx.shape + (3,))
    sw, se = F[:-1, np.arange(F.shape[1]) - 1], F[:-1]
    nw, ne = F[1:, np.arange(F.shape[1]) - 1], F[1:]
    area = 0.5 * (np.linalg.norm(np.cross(se - sw, ne - sw), axis=-1)
                  + np.linalg.norm(np.cross(ne - sw, nw - sw), axis=-1)) * R_EARTH_M ** 2
    rows = slice(0, g.j_pole - 1)                              # T rows 1 .. j_pole - 1 (F rows r - 1, r)
    t = lonlat_to_xyz(g.lon["t"][1:g.j_pole, :-2].ravel(), g.lat["t"][1:g.j_pole, :-2].ravel())
    return t, np.sqrt(area[rows]).ravel()


CACHE_DIR = Path(__file__).resolve().parents[2] / ".tmp" / "cache" / "orca_sizes"   # B105: <repo>/.tmp
_SIZES: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_FIELDS: dict[tuple[str, float], OrcaSize] = {}


def _cache_key(name: str) -> str:
    """The preset and the code that builds its grid: a change of either gives a new cache file."""
    from meshom.orca import generator, presets

    src = b"".join(Path(f).read_bytes() for f in (generator.__file__, presets.__file__, __file__))
    return hashlib.sha1(name.encode() + repr(presets.PRESETS[name]).encode() + src).hexdigest()[:16]


def preset_cell_sizes(name: str) -> tuple[np.ndarray, np.ndarray]:
    """cell_sizes of preset `name`, built once (an ORCA025 grid takes ~76 s) and kept in this process and in
    CACHE_DIR/<name>_<key>.npz (decisions B105)."""
    if name in _SIZES:
        return _SIZES[name]
    f = CACHE_DIR / f"{name}_{_cache_key(name)}.npz"
    try:
        with np.load(f) as d:
            t, h = d["t"], d["h"]
    except (OSError, KeyError, ValueError):
        from meshom.orca import build

        t, h = cell_sizes(build(name, 0))
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp.npz")
            np.savez(tmp, t=t, h=h)
            os.replace(tmp, f)
        except OSError:
            pass
    _SIZES[name] = (t, h)
    return t, h


def orca_size(name: str, h_min_m: float = 0.0) -> OrcaSize:
    """The size field of preset `name` (meshom.orca.presets); the cell sizes and the field are cached (B105)."""
    key = (name, float(h_min_m))
    if key not in _FIELDS:
        t, h = preset_cell_sizes(name)
        _FIELDS[key] = OrcaSize(cKDTree(t), h, float(h_min_m), name)
    return _FIELDS[key]
