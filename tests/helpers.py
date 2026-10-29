"""Test helpers: small synthetic meshes built in a local east/north frame."""
import numpy as np

from meshom.geo import R_EARTH_M, lonlat_to_xyz
from meshom.mesh import Mesh


def _east_north(lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, la = np.radians(lon0), np.radians(lat0)
    c = lonlat_to_xyz(np.array([lon0]), np.array([lat0]))[0]
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    return c, east, north


def local_polygon(lon0: float, lat0: float, xy_m) -> np.ndarray:
    """East/north offsets in metres -> (k,3) unit vectors, inverse gnomonic map at (lon0, lat0)."""
    c, east, north = _east_north(lon0, lat0)
    xy = np.asarray(xy_m, dtype=np.float64) / R_EARTH_M
    p = c + xy[:, :1] * east + xy[:, 1:2] * north
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def land_loop(lon0: float, lat0: float, xy_km) -> np.ndarray:
    """A planar polygon (km, either orientation) about (lon0, lat0) -> a land loop on the sphere: clockwise (water on
    the left, the coast convention), vertices every 5 km (local_polygon)."""
    xy = np.asarray(xy_km, dtype=float)
    if shoelace(xy) < 0:
        xy = xy[::-1]
    out = []
    for a, b in zip(xy, np.roll(xy, -1, axis=0)):
        k = max(1, int(np.ceil(np.linalg.norm(b - a) / 5.0)))
        out.append(a + (b - a) * (np.arange(k)[:, None] / k))
    return local_polygon(lon0, lat0, np.vstack(out)[::-1] * 1e3)


def shoelace(q: np.ndarray) -> float:
    """Signed area of a planar polygon (k,2), > 0 for CCW."""
    x, y = q[:, 0], q[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _mesh(points, tris, quads) -> Mesh:
    tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    return Mesh(np.asarray(points, dtype=np.float64), tris, quads, np.zeros(len(points), dtype=np.int8))


def cube_sphere() -> Mesh:
    """8 nodes (+-1,+-1,+-1)/sqrt(3), 6 quads CCW seen from outside."""
    pts = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64) / np.sqrt(3)
    quads = []
    for a in range(3):
        b, c = [i for i in range(3) if i != a]
        for s in (-1, 1):
            face = []
            for u, v in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                want = np.zeros(3); want[a], want[b], want[c] = s, u, v
                face.append(int(np.argmin(np.linalg.norm(pts * np.sqrt(3) - want, axis=1))))
            p = pts[face]
            if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), p.mean(axis=0)) < 0:
                face = face[::-1]
            quads.append(face)
    return _mesh(pts, [], quads)


def structured_patch(nx: int, ny: int, dx_m: float, lon0: float = 0.0, lat0: float = 0.0, xs_m=None) -> Mesh:
    """nx x ny quads, rows dx_m apart, columns at xs_m (default i*dx_m), centred at (lon0, lat0).

    Node (j, i) has index j*(nx+1)+i; quad (j, i) has index j*nx+i (row-major) and is CCW.
    """
    xs = np.arange(nx + 1) * dx_m if xs_m is None else np.asarray(xs_m, dtype=np.float64)
    ys = np.arange(ny + 1) * dx_m
    X, Y = np.meshgrid(xs - 0.5 * (xs[0] + xs[-1]), ys - 0.5 * ys[-1])
    pts = local_polygon(lon0, lat0, np.column_stack([X.ravel(), Y.ravel()]))
    j, i = np.divmod(np.arange(nx * ny), nx)
    n = j * (nx + 1) + i
    quads = np.column_stack([n, n + 1, n + nx + 2, n + nx + 1])
    return _mesh(pts, [], quads)


def mesh_from_polygons(polys) -> Mesh:
    """Polygons (k,3) with k = 3 or 4 -> Mesh; corners closer than 1e-6 m become one node."""
    pts, elems = [], []
    tol = 1e-6 / R_EARTH_M
    for poly in polys:
        ids = []
        for p in np.asarray(poly, dtype=np.float64):
            d = np.linalg.norm(np.asarray(pts) - p, axis=1) if pts else np.array([])
            if d.size and d.min() < tol:
                ids.append(int(d.argmin()))
            else:
                pts.append(p); ids.append(len(pts) - 1)
        elems.append(ids)
    return _mesh(np.array(pts), [e for e in elems if len(e) == 3], [e for e in elems if len(e) == 4])


def split_quads(m: Mesh, ids) -> Mesh:
    """Each listed quad -> 2 triangles along its 0-2 diagonal; the other quads keep their order."""
    ids = np.asarray(list(ids), dtype=np.int64)
    keep = np.ones(m.n_quads, dtype=bool); keep[ids] = False
    q = m.quads[ids]
    new = np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    return Mesh(m.points, np.vstack([m.tris, new]), m.quads[keep], m.node_flag.copy())


def mixed_cube() -> Mesh:
    """cube_sphere with face 0 split into 2 triangles: 2 tris, 5 quads."""
    return split_quads(cube_sphere(), [0])


def triangulated_cube() -> Mesh:
    """Each cube face split into 2 triangles: 12 tris."""
    return split_quads(cube_sphere(), range(6))


def _cyclic(cells: np.ndarray) -> list[tuple]:
    out = []
    for c in cells.tolist():
        k = c.index(min(c))
        out.append(tuple(c[k:] + c[:k]))
    return sorted(out)


def same_elements(a: Mesh, b: Mesh) -> bool:
    """Equal as sets of cyclically normalised node tuples (orientation counts)."""
    return _cyclic(a.tris) == _cyclic(b.tris) and _cyclic(a.quads) == _cyclic(b.quads)


def random_rotation(rng) -> np.ndarray:
    """Proper 3x3 rotation (det +1), QR of a Gaussian matrix with the sign fix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q
