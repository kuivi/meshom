# SPDX-License-Identifier: GPL-3.0-or-later
"""The directions of an ORCA grid as a cross field (decisions B109): the frame of a fill between two lines of the same
ORCA map (m16_orca's coarse and refined parts), so the fill's rows only change size and do not turn.

At every T point of the preset's grid (T rows 1 .. j_pole - 1, the unique columns, as meshom.orca.sizes) the
i-direction is the chord T(j, i + 1) - T(j, i - 1) (columns periodic) projected on the tangent plane; ORCA's grid is
orthogonal, so the j-direction is its quarter turn and the cross {+-i, +-j} is the grid's. At a point x the frame is
the 4-fold average (power vectors e^(4 i phi), Knöppel 2013) of the K_NEAR nearest T points' i-directions, each
projected on x's tangent plane and weighted by the inverse squared distance: at a T point its own direction. Near
NEMO's grid poles the i-lines converge and the frame turns fast (a +1 singularity at each pole); the poles lie on
land.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from meshom.geo import tangent_basis

K_NEAR = 4          # B109: T points in the 4-fold average


@dataclass(frozen=True, eq=False)
class OrcaFrame:
    tree: cKDTree
    d: np.ndarray            # (n, 3) unit tangent i-direction at each T point
    name: str

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        """(M, 3) unit tangent vectors at the unit vectors xyz (M, 3): the frame's first direction."""
        x = np.asarray(xyz, dtype=np.float64)
        out = np.empty_like(x)
        e1, e2 = tangent_basis(x)
        for a in range(0, len(x), 1_000_000):
            s = slice(a, a + 1_000_000)
            dist, k = self.tree.query(x[s], k=K_NEAR, workers=-1)
            w = 1.0 / np.maximum(dist, 1e-12) ** 2
            dk = self.d[k]                                          # (m, K, 3)
            phi = np.arctan2(np.einsum("mkc,mc->mk", dk, e2[s]), np.einsum("mkc,mc->mk", dk, e1[s]))
            z = (w * np.exp(4j * phi)).sum(axis=1)
            ang = np.angle(z) / 4.0
            out[s] = np.cos(ang)[:, None] * e1[s] + np.sin(ang)[:, None] * e2[s]
        return out


def i_directions(t: np.ndarray, ni: int) -> np.ndarray:
    """(n, 3) the unit tangent i-direction at the T points t (n, 3), rows of ni unique columns (periodic)."""
    T = t.reshape(-1, ni, 3)
    d = np.roll(T, -1, axis=1) - np.roll(T, 1, axis=1)
    d = d - np.einsum("rcx,rcx->rc", d, T)[..., None] * T
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    return (d / np.maximum(n, 1e-300)).reshape(-1, 3)


_FRAMES: dict[str, OrcaFrame] = {}


def orca_frame(name: str) -> OrcaFrame:
    """The frame of preset `name` (meshom.orca.presets) from its cached T points (meshom.orca.sizes, B105)."""
    if name not in _FRAMES:
        from meshom.orca.presets import PRESETS
        from meshom.orca.sizes import preset_cell_sizes

        t, _ = preset_cell_sizes(name)
        p = PRESETS[name]
        ni = (p.params.jpi - 1) * p.q
        if len(t) % ni:
            raise ValueError(f"orca_frame {name}: {len(t)} T points are not rows of {ni} columns")
        _FRAMES[name] = OrcaFrame(cKDTree(t), i_directions(t, ni), name)
    return _FRAMES[name]
