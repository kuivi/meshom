"""Sphere geometry: lon/lat <-> unit vectors, arcs, tangent planes, FESOM2 rotation, stereographic inverse.

Points are (N,3) float64 unit vectors. Lengths in metres use R_EARTH_M; angles and ratios do not depend on it.
"""
import numpy as np

R_EARTH_M: float = 6_371_000.0


def lonlat_to_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon, lat = np.radians(lon_deg), np.radians(lat_deg)
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1)


def xyz_to_lonlat(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Degrees; lon in [-180, 180); lon = 0 exactly at a pole."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    h = np.hypot(x, y)
    lat = np.degrees(np.arctan2(z, h))
    lon = np.where(h > 0, np.degrees(np.arctan2(y, x)), 0.0)
    lon = np.where(lon >= 180.0, lon - 360.0, lon)
    return lon, lat


def arc_length(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Row-wise great-circle angle in radians, atan2(|p x q|, p.q) (accurate for tiny and near-antipodal arcs)."""
    return np.arctan2(np.linalg.norm(np.cross(p, q), axis=-1), np.einsum("...i,...i->...", p, q))


def tangent_basis(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal e1, e2 (M,3) of the tangent plane at unit c (M,3), with e1 x e2 = c.

    The reference axis is the coordinate axis along which |c| is smallest, so the basis is defined at the poles.
    """
    k = np.argmin(np.abs(c), axis=-1)
    a = np.zeros_like(c)
    np.put_along_axis(a, k[..., None], 1.0, axis=-1)
    e1 = a - np.einsum("...i,...i->...", a, c)[..., None] * c
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(c, e1)
    return e1, e2


def tangent_coords(p: np.ndarray, radius_m: float = R_EARTH_M) -> np.ndarray:
    """Corners p (M,k,3) of M elements -> (M,k,2) metres, gnomonic projection about the normalised corner mean.

    The plane is seen from outside the sphere, so a CCW element stays CCW (positive shoelace area).
    """
    c = p.mean(axis=1)
    c /= np.linalg.norm(c, axis=-1, keepdims=True)
    e1, e2 = tangent_basis(c)
    g = p / np.einsum("mki,mi->mk", p, c)[..., None]          # gnomonic: scale onto the plane c.x = 1
    return radius_m * np.stack([np.einsum("mki,mi->mk", g, e1), np.einsum("mki,mi->mk", g, e2)], axis=-1)


def euler_matrix(alpha_deg: float = 50.0, beta_deg: float = 15.0, gamma_deg: float = -90.0) -> np.ndarray:
    """FESOM2's rotation matrix (geographic to rotated): rotated = M @ geographic."""
    al, be, ga = np.radians([alpha_deg, beta_deg, gamma_deg])
    ca, sa, cb, sb, cg, sg = np.cos(al), np.sin(al), np.cos(be), np.sin(be), np.cos(ga), np.sin(ga)
    return np.array([
        [cg * ca - sg * cb * sa, cg * sa + sg * cb * ca, sg * sb],
        [-sg * ca - cg * cb * sa, -sg * sa + cg * cb * ca, cg * sb],
        [sb * sa, -sb * ca, cb],
    ])


def geo_to_rot(xyz: np.ndarray, m: np.ndarray) -> np.ndarray:
    """FESOM2 g2r: r = M g (row-wise)."""
    return xyz @ m.T


def rot_to_geo(xyz: np.ndarray, m: np.ndarray) -> np.ndarray:
    """FESOM2 r2g: g = M^T r (row-wise)."""
    return xyz @ m


def stereographic_inverse(x_m: np.ndarray, y_m: np.ndarray, lon0_deg: float, lat0_deg: float,
                          radius_m: float = R_EARTH_M) -> np.ndarray:
    """Spherical stereographic plane tangent at (lon0, lat0), x east, y north, metres -> (N,3) unit vectors.

    Conformal: a point at plane distance rho lies at arc c = 2 atan(rho / 2R) from the centre.
    """
    lo, la = np.radians(lon0_deg), np.radians(lat0_deg)
    c0 = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    x, y = np.asarray(x_m, dtype=np.float64), np.asarray(y_m, dtype=np.float64)
    rho = np.hypot(x, y)
    c = 2.0 * np.arctan(rho / (2.0 * radius_m))
    f = np.where(rho > 0, np.sin(c) / np.where(rho > 0, rho, 1.0), 1.0 / radius_m)   # sin(c)/rho -> 1/R at rho = 0
    return np.cos(c)[:, None] * c0 + (f * x)[:, None] * east + (f * y)[:, None] * north
