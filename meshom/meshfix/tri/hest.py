"""A size field estimated from a mesh that stores none (FESOM2 meshes; decision B154).

EstimatedH(mesh): per node l = the geometric mean of its incident edge lengths (great-circle), ln l smoothed by
`sweeps` Jacobi sweeps x <- 1/2 x + 1/2 mean over the edge neighbours (a radius of a few edges: the resolution design
stays, the cell-to-cell noise goes), then scaled by one factor so that the equilateral triangle count at h,
sum over triangles of area / (sqrt(3)/4 h^2) with h at the centroid, equals the mesh's triangle count (a method that
fits s = h keeps the node count). h(xyz) for any point: ln h interpolated from the `k` nearest nodes with Gaussian
weights exp(-(d / (0.5 h_i))^2) (smooth across nodes, so edge-midpoint sizes carry no nearest-node steps). The
estimate is frozen with the case: candidates never rebuild it. It measures roughness against the mesh's own
smoothed resolution, not against a designer's h.
"""
import numpy as np
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M, arc_length
from meshom.mesh import Mesh, build_topology

SWEEPS = 20         # B154: Jacobi sweeps of ln l
K_NEAR = 8          # B154: nodes in the interpolation of ln h
SIGMA_H = 0.5       # B154: Gaussian width of the interpolation weights, in units of the node's h


class EstimatedH:
    """Picklable size field h(xyz (M,3) unit vectors) -> (M,) metres, estimated from a triangle mesh."""

    def __init__(self, mesh: Mesh, sweeps: int = SWEEPS, k: int = K_NEAR, sigma_h: float = SIGMA_H):
        P = np.ascontiguousarray(mesh.points, dtype=np.float64)
        n = len(P)
        E = build_topology(mesh).edges
        L = R_EARTH_M * arc_length(P[E[:, 0]], P[E[:, 1]])
        deg = np.bincount(E.ravel(), minlength=n).astype(np.float64)
        ll = np.log(L)
        x = (np.bincount(E[:, 0], weights=ll, minlength=n) + np.bincount(E[:, 1], weights=ll, minlength=n)) / \
            np.maximum(deg, 1)
        for _ in range(sweeps):
            s = np.bincount(E[:, 0], weights=x[E[:, 1]], minlength=n) + \
                np.bincount(E[:, 1], weights=x[E[:, 0]], minlength=n)
            x = 0.5 * x + 0.5 * s / np.maximum(deg, 1)
        self.points = P
        self.lnh = x
        self.k = int(k)
        self.sigma_h = float(sigma_h)
        self.sweeps = int(sweeps)
        self._tree = cKDTree(P)
        # count-preserving scale
        cells = mesh.tris
        A, B, C = P[cells[:, 0]], P[cells[:, 1]], P[cells[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1) * R_EARTH_M ** 2
        cen = A + B + C
        cen /= np.linalg.norm(cen, axis=1, keepdims=True)
        h0 = self(cen)
        c2 = float(np.sum(area / (np.sqrt(3.0) / 4.0 * h0 ** 2)) / len(cells))
        self.scale = float(np.sqrt(c2))
        self.lnh = self.lnh + np.log(self.scale)

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        if len(xyz) == 0:
            return np.zeros(0)
        out = np.empty(len(xyz))
        k = min(self.k, len(self.points))
        for s in range(0, len(xyz), 500_000):
            p = xyz[s:s + 500_000]
            d, j = self._tree.query(p, k=k)
            d, j = d.reshape(len(p), k), j.reshape(len(p), k)
            lh = self.lnh[j]
            w = np.exp(-(d * R_EARTH_M / (self.sigma_h * np.exp(lh))) ** 2) + 1e-300
            out[s:s + 500_000] = np.exp((w * lh).sum(1) / w.sum(1))
        return out

    def describe(self) -> dict:
        return {"estimator": "EstimatedH (meshom/meshfix/tri/hest.py, B154)",
                "rule": "ln of the geometric-mean incident edge length per node, Jacobi-smoothed, scaled so the "
                        "equilateral triangle count at h equals the mesh's; Gaussian interpolation of ln h",
                "sweeps": self.sweeps, "k": self.k, "sigma_h": self.sigma_h, "scale": self.scale}
