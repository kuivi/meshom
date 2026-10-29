"""Mixed mesh on the sphere: data model, topology, orientation.

Global element index: e < T is tris[e], e >= T is quads[e - T]. Elements are CCW seen from outside the sphere.
"""
from dataclasses import dataclass

import numpy as np

from meshom.geo import tangent_coords


@dataclass
class Mesh:
    points: np.ndarray      # (N,3) float64 unit vectors
    tris: np.ndarray        # (T,3) int64
    quads: np.ndarray       # (Q,4) int64
    node_flag: np.ndarray   # (N,) int8
    tri_zone: np.ndarray | None = None   # (T,) bool: triangles of a declared triangle zone (a method's intentional
                                         # triangle fill; the cluster gate); None = no zone. Kept by orient_ccw,
                                         # smooth.smooth and repair (R23), saved in raw.npz / smooth.npz

    @property
    def n_nodes(self) -> int:
        return len(self.points)

    @property
    def n_tris(self) -> int:
        return len(self.tris)

    @property
    def n_quads(self) -> int:
        return len(self.quads)

    @property
    def n_elems(self) -> int:
        return len(self.tris) + len(self.quads)


@dataclass
class Topology:
    edges: np.ndarray            # (E,2) int64, n0 < n1
    edge_elems: np.ndarray       # (E,2) int64 global element ids; column 1 = -1 on boundary edges
    elems_per_node: np.ndarray   # (N,) int64
    boundary_node: np.ndarray    # (N,) bool
    elem_neighbours: np.ndarray  # (T+Q,) int64, number of edge-adjacent elements


def _half_edges(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directed edges (a -> b) of all elements with their global element id."""
    a, b, e = [], [], []
    for cells, off in ((mesh.tris, 0), (mesh.quads, mesh.n_tris)):
        k = cells.shape[1]
        a.append(cells.ravel())
        b.append(np.roll(cells, -1, axis=1).ravel())
        e.append(np.repeat(np.arange(len(cells), dtype=np.int64) + off, k))
    return np.concatenate(a), np.concatenate(b), np.concatenate(e)


def build_topology(mesh: Mesh) -> Topology:
    n = mesh.n_nodes
    a, b, e = _half_edges(mesh)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo * n + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    first = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    count = np.diff(np.r_[first, len(ks)])
    edges = np.column_stack([ks[first] // n, ks[first] % n])
    second = np.minimum(first + 1, len(ks) - 1)
    edge_elems = np.column_stack([e[order[first]], np.where(count >= 2, e[order[second]], -1)])

    nodes = np.concatenate([mesh.tris.ravel(), mesh.quads.ravel()])
    elems_per_node = np.bincount(nodes, minlength=n).astype(np.int64)
    boundary_node = np.zeros(n, dtype=bool)
    boundary_node[edges[edge_elems[:, 1] < 0].ravel()] = True
    inner = edge_elems[edge_elems[:, 1] >= 0]
    elem_neighbours = np.bincount(inner.ravel(), minlength=mesh.n_elems).astype(np.int64)
    return Topology(edges, edge_elems, elems_per_node, boundary_node, elem_neighbours)


def _shoelace(q: np.ndarray) -> np.ndarray:
    x, y = q[..., 0], q[..., 1]
    return 0.5 * np.sum(x * np.roll(y, -1, axis=-1) - np.roll(x, -1, axis=-1) * y, axis=-1)


def signed_area(mesh: Mesh) -> np.ndarray:
    """(T+Q,) m^2, shoelace in each element's tangent plane, > 0 for CCW."""
    parts = [_shoelace(tangent_coords(mesh.points[c])) for c in (mesh.tris, mesh.quads) if len(c)]
    return np.concatenate(parts) if parts else np.zeros(0)


def orient_ccw(mesh: Mesh) -> Mesh:
    """Copy of mesh with every element of negative signed area reversed (first node kept)."""
    neg = signed_area(mesh) < 0
    tris, quads = mesh.tris.copy(), mesh.quads.copy()
    t = neg[: mesh.n_tris]
    q = neg[mesh.n_tris:]
    tris[t] = tris[t][:, [0, 2, 1]]
    quads[q] = quads[q][:, [0, 3, 2, 1]]
    return Mesh(mesh.points, tris, quads, mesh.node_flag, mesh.tri_zone)
