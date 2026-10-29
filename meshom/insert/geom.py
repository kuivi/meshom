"""Plane geometry of an insert: the local stereographic chart, polylines, the strait's axis, transects and the
smoothed bank (decisions B158-B161).

All construction happens in a stereographic chart about the region's centre (conformal: angles are kept and a
harmonic map in the chart is harmonic on the sphere; the length scale drifts by 1 + r^2 / 4R^2, 0.15 % at 500 km).
Every quality number is measured on the sphere afterwards (meshom.quality.kernel), never in the chart.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M


class Chart:
    """Stereographic chart tangent at the unit vector c: fwd (M,3) -> (M,2) m, inv (M,2) -> (M,3)."""

    def __init__(self, c: np.ndarray):
        c = np.asarray(c, dtype=np.float64)
        self.c = c / np.linalg.norm(c)
        k = np.array([0.0, 0.0, 1.0]) if abs(self.c[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(k, self.c)
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(self.c, self.e1)

    def fwd(self, p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p)
        d = 1.0 + p @ self.c
        return 2.0 * R_EARTH_M * np.c_[p @ self.e1, p @ self.e2] / d[:, None]

    def inv(self, xy: np.ndarray) -> np.ndarray:
        u = np.atleast_2d(xy) / (2.0 * R_EARTH_M)
        r2 = (u ** 2).sum(1)
        f = 1.0 / (1.0 + r2)
        return (2 * f)[:, None] * (u[:, :1] * self.e1 + u[:, 1:] * self.e2) + ((1 - r2) * f)[:, None] * self.c

    def scale(self, xy: np.ndarray) -> np.ndarray:
        """Chart metres per sphere metre at xy: 1 + r^2 / 4R^2."""
        return 1.0 + (np.atleast_2d(xy) ** 2).sum(1) / (4.0 * R_EARTH_M ** 2)


class Polyline:
    """A plane polyline (K,2) with arc length s; at(s) points, project(q) the arc length of the nearest point."""

    def __init__(self, xy: np.ndarray):
        xy = np.asarray(xy, dtype=np.float64)
        keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-9]
        self.xy = xy[keep]
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))]
        self.L = float(self.s[-1])
        self._tree = None

    def at(self, s) -> np.ndarray:
        s = np.asarray(s, dtype=np.float64)
        return np.c_[np.interp(s, self.s, self.xy[:, 0]), np.interp(s, self.s, self.xy[:, 1])]

    def tangent(self, s, half: float) -> np.ndarray:
        """Unit chord direction between s - half and s + half (clipped to the ends)."""
        s = np.atleast_1d(np.asarray(s, dtype=np.float64))
        a, b = self.at(np.clip(s - half, 0, self.L)), self.at(np.clip(s + half, 0, self.L))
        d = b - a
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)

    def project(self, q: np.ndarray) -> np.ndarray:
        """Arc length of the closest point of the polyline to each q (nearest vertex, then its two segments)."""
        if self._tree is None:
            self._tree = cKDTree(self.xy)
        q = np.atleast_2d(q)
        j = self._tree.query(q)[1]
        best_s = self.s[j].astype(float)
        best_d = np.linalg.norm(self.xy[j] - q, axis=1)
        n = len(self.xy)
        for jj in (j - 1, j):
            ok = (jj >= 0) & (jj < n - 1)
            jj = np.clip(jj, 0, max(n - 2, 0))
            a, b = self.xy[jj], self.xy[np.minimum(jj + 1, n - 1)]
            d = b - a
            L2 = (d ** 2).sum(1) + 1e-30
            t = np.clip(((q - a) * d).sum(1) / L2, 0, 1)
            dist = np.linalg.norm(a + t[:, None] * d - q, axis=1)
            better = ok & (dist < best_d)
            best_d = np.where(better, dist, best_d)
            best_s = np.where(better, self.s[jj] + t * np.sqrt(L2), best_s)
        return best_s

    def distance(self, q: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.at(self.project(q)) - np.atleast_2d(q), axis=1)


def resample(xy: np.ndarray, step: float) -> np.ndarray:
    """The polyline at a spacing <= step (ends kept)."""
    pl = Polyline(xy)
    m = max(int(np.ceil(pl.L / step)), 1)
    return pl.at(np.linspace(0.0, pl.L, m + 1))


class Axis(Polyline):
    """The strait's axis: its centreline in the chart, extended at both ends by `ext` m along the end directions
    (tangent over `ext_dir` m), so transects reach the open basins. s0 = arc length of the path's first point."""

    def __init__(self, path_xy: np.ndarray, ext: float, step: float):
        p = resample(path_xy, step)
        pl = Polyline(p)
        half = min(0.25 * pl.L, 2.0 * step + 0.1 * pl.L)
        t0, t1 = pl.tangent(0.0, half)[0], pl.tangent(pl.L, half)[0]
        k = max(int(np.ceil(ext / step)), 1)
        head = p[0] - t0 * np.linspace(ext, 0, k + 1)[:-1, None]
        tail = p[-1] + t1 * np.linspace(0, ext, k + 1)[1:, None]
        super().__init__(np.vstack([head, p, tail]))
        self.s_path = (float(ext), float(ext + pl.L))

    def frame(self, s, half: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(points, unit tangents, unit left normals) at s."""
        t = self.tangent(s, half)
        return self.at(s), t, np.c_[-t[:, 1], t[:, 0]]


def ray_hits(o: np.ndarray, d: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First crossing of the rays o + t d (t > 0; o, d (M,2), d unit) with the segments seg (S,2,2): (t (M,) inf if
    none, segment index (M,) -1 if none)."""
    if len(seg) == 0:
        return np.full(len(o), np.inf), np.full(len(o), -1)
    a, b = seg[:, 0], seg[:, 1]
    e = b - a                                                     # (S,2)
    tt = np.full(len(o), np.inf)
    ii = np.full(len(o), -1)
    for lo in range(0, len(o), 256):
        oo, dd = o[lo:lo + 256, None, :], d[lo:lo + 256, None, :]
        den = dd[..., 0] * e[None, :, 1] - dd[..., 1] * e[None, :, 0]
        w = a[None] - oo
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[..., 0] * e[None, :, 1] - w[..., 1] * e[None, :, 0]) / den
            u = (w[..., 0] * dd[..., 1] - w[..., 1] * dd[..., 0]) / den
        ok = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        t = np.where(ok, t, np.inf)
        j = np.argmin(t, axis=1)
        tt[lo:lo + 256] = t[np.arange(len(j)), j]
        ii[lo:lo + 256] = np.where(np.isfinite(tt[lo:lo + 256]), j, -1)
    return tt, ii


def smooth_bank(xy: np.ndarray, sigma: float, cap, step: float, iters: int = 40) -> np.ndarray:
    """The coast polyline xy smoothed along its arc with a Gaussian of `sigma` m (ends pinned, a smoothstep taper
    over 2 sigma at each end), resampled at `step`, then held within `cap` m of the input coast both ways (B161):
    every point of the result within cap of the input polyline (nearest point; clamped) and every input vertex within
    cap of the result (the result pulled toward it with a Gaussian of sigma / 2 along its arc), iterated. cap: a
    scalar or a callable of (K,2) points."""
    pl = Polyline(xy)
    m = max(int(np.ceil(pl.L / step)), 8)
    t = np.linspace(0.0, pl.L, m + 1)
    raw = pl.at(t)
    if sigma <= 0 or pl.L <= 0:
        return raw
    capf = cap if callable(cap) else (lambda p: np.full(len(p), float(cap)))
    sm = gaussian_filter1d(raw, sigma / (pl.L / m), axis=0, mode="nearest")
    w = np.clip(t / (2 * sigma), 0, 1) * np.clip((t[-1] - t) / (2 * sigma), 0, 1)
    w = w * w * (3 - 2 * w)
    cur = raw + w[:, None] * (sm - raw)
    verts = pl.xy[1:-1]
    cv = capf(verts) if len(verts) else np.zeros(0)
    for _ in range(iters):
        q = pl.at(pl.project(cur))                            # forward: within cap of the input
        d = cur - q
        n = np.linalg.norm(d, axis=1)
        c = capf(cur)
        cur = q + np.minimum(1.0, c / np.maximum(n, 1e-12))[:, None] * d
        cur[0], cur[-1] = raw[0], raw[-1]
        if not len(verts):
            break
        cp = Polyline(cur)                                    # backward: every input vertex within cap of the result
        sv = cp.project(verts)
        pv = cp.at(sv)
        dv = np.linalg.norm(verts - pv, axis=1)
        bad = dv > cv
        if not bad.any():
            break
        tc = cp.s if len(cp.s) == len(cur) else np.linspace(0, cp.L, len(cur))
        for k in np.flatnonzero(bad):
            g = np.exp(-((tc - sv[k]) / max(0.5 * sigma, 1e-9)) ** 2)
            cur = cur + g[:, None] * ((verts[k] - pv[k]) / dv[k] * (dv[k] - 0.9 * cv[k]))[None]
        cur[0], cur[-1] = raw[0], raw[-1]
    return cur


def point_in_polygon(q: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """(M,) bool: q (M,2) inside the plane polygon poly (K,2) (even-odd rule)."""
    from matplotlib.path import Path as MPath

    return MPath(poly).contains_points(q)


def seg_distance(q: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """(M,) plane distance of q (M,2) to the nearest of the segments seg (S,2,2) (brute force in chunks)."""
    out = np.full(len(q), np.inf)
    if len(seg) == 0:
        return out
    a, e = seg[:, 0], seg[:, 1] - seg[:, 0]
    L2 = (e ** 2).sum(1) + 1e-30
    for lo in range(0, len(q), 512):
        qq = q[lo:lo + 512, None, :]
        t = np.clip(((qq - a[None]) * e[None]).sum(-1) / L2[None], 0, 1)
        p = a[None] + t[..., None] * e[None]
        out[lo:lo + 512] = np.linalg.norm(p - qq, axis=-1).min(1)
    return out
