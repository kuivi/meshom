"""The reference coast of a case as a segment graph (decisions B142): nearest segments, the segments a sliding coast
node may use, projection onto them, and the coast-shift check of a candidate.

The reference coast is the input mesh's boundary edges minus the artificial ones (= Case.coast_seg, with the node
ids kept so that segments know their neighbours along the coast). A sliding coast node may use only the segments
reachable along the coast from its nearest segment within `reach` (arc length): it never jumps across a strait to
the other shore.
"""
import numpy as np
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh


def seg_point(x: np.ndarray, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest points q (..., 3) on the chord segments A-B (..., 3) to x (..., 3) and the chord distances."""
    d = B - A
    t = np.einsum("...i,...i->...", x - A, d) / np.maximum(np.einsum("...i,...i->...", d, d), 1e-300)
    q = A + np.clip(t, 0.0, 1.0)[..., None] * d
    return q, np.linalg.norm(x - q, axis=-1)


class RefCoast:
    def __init__(self, mesh: Mesh, art_edges: np.ndarray | None = None, seg: np.ndarray | None = None):
        """From the input mesh's boundary edges minus art_edges, or (seg given) from the segments (S,2,3) of
        Case.coast_seg: the same edges in the same order, their end nodes recovered by exact position (from_segments;
        what a later pipeline stage has, whose case mesh is no longer the input's)."""
        if seg is not None:
            P, be = _graph_of(seg)
        else:
            from meshom.meshfix.cases import boundary_edges

            be = boundary_edges(mesh)
            if art_edges is not None and len(art_edges):
                n = mesh.n_nodes
                ak = set((np.minimum(art_edges[:, 0], art_edges[:, 1]) * n
                          + np.maximum(art_edges[:, 0], art_edges[:, 1])).tolist())
                k = np.minimum(be[:, 0], be[:, 1]) * n + np.maximum(be[:, 0], be[:, 1])
                be = be[~np.isin(k, list(ak))]
            P = mesh.points
        self.seg = be
        self.A, self.B = P[be[:, 0]].copy(), P[be[:, 1]].copy()
        m = self.A + self.B
        self.tree = cKDTree(m / np.linalg.norm(m, axis=1, keepdims=True))
        self.L = np.linalg.norm(self.B - self.A, axis=1) * R_EARTH_M
        # segment -> segments sharing an end node
        nodes = be.ravel()
        sid = np.repeat(np.arange(len(be)), 2)
        o = np.argsort(nodes, kind="stable")
        ns, ss = nodes[o], sid[o]
        self._by_node = {}
        for v, s in zip(ns.tolist(), ss.tolist()):
            self._by_node.setdefault(v, []).append(s)
        vids = np.unique(nodes)
        self.V = P[vids].copy()                            # reference vertices (the input coast nodes)
        self.vtree = cKDTree(self.V)

    @classmethod
    def from_segments(cls, seg: np.ndarray) -> "RefCoast":
        return cls(None, None, seg)

    def nearest(self, X: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
        """(segment id, chord distance) of the nearest reference segment to each of X (M,3)."""
        X = np.atleast_2d(X)
        k = min(k, len(self.A))
        _, idx = self.tree.query(X, k=k)
        idx = idx.reshape(len(X), k)
        _, d = seg_point(X[:, None, :], self.A[idx], self.B[idx])
        j = d.argmin(1)
        return idx[np.arange(len(X)), j], d[np.arange(len(X)), j]

    def allowed(self, x: np.ndarray, reach_m: float, K: int = 24) -> np.ndarray:
        """Up to K segment ids reachable along the coast from the nearest segment of x within reach_m (the
        nearest first); every id valid (the list is padded by repeating the nearest)."""
        s0 = int(self.nearest(x[None])[0][0])
        out, dist = [s0], {s0: 0.0}
        i = 0
        while i < len(out) and len(out) < K:
            s = out[i]
            i += 1
            for v in self.seg[s].tolist():
                for t in self._by_node.get(v, ()):
                    if t in dist:
                        continue
                    d = dist[s] + self.L[t]
                    if d - self.L[t] > reach_m:
                        continue
                    dist[t] = d
                    out.append(t)
                    if len(out) >= K:
                        break
        out = out[:K]
        return np.array(out + [s0] * (K - len(out)), dtype=np.int64)

    def allowed_many(self, X: np.ndarray, reach_m: np.ndarray, K: int = 24) -> np.ndarray:
        return np.array([self.allowed(x, float(r), K) for x, r in zip(X, np.broadcast_to(reach_m, len(X)))],
                        dtype=np.int64).reshape(len(X), K)

    def project(self, X: np.ndarray, segs: np.ndarray) -> np.ndarray:
        """X (M,3) projected onto the nearest of its allowed segments segs (M,K), on the sphere."""
        q, d = seg_point(X[:, None, :], self.A[segs], self.B[segs])
        j = d.argmin(1)
        q = q[np.arange(len(X)), j]
        return q / np.linalg.norm(q, axis=1, keepdims=True)

    def vertices_near(self, X: np.ndarray, r_chord: np.ndarray) -> np.ndarray:
        """Ids into V of the reference vertices within r_chord (per point) of any of X."""
        ids = set()
        for x, r in zip(X, np.broadcast_to(r_chord, len(X))):
            ids.update(self.vtree.query_ball_point(x, float(r)))
        return np.array(sorted(ids), dtype=np.int64)


def project(X: np.ndarray, SA: np.ndarray, SB: np.ndarray) -> np.ndarray:
    """X (M,3) onto the nearest of its segments SA-SB (M,K,3), on the sphere."""
    if len(X) == 0:
        return X
    d = SB - SA
    t = np.clip(np.einsum("mki,mki->mk", X[:, None] - SA, d) / np.maximum(np.einsum("mki,mki->mk", d, d), 1e-300),
                0.0, 1.0)
    q = SA + t[..., None] * d
    j = np.linalg.norm(X[:, None] - q, axis=2).argmin(1)
    q = q[np.arange(len(X)), j]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _graph_of(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(points, (S,2) node ids) of segments (S,2,3) whose shared ends are bitwise equal points (Case.coast_seg copies
    the input mesh's node coordinates)."""
    seg = np.ascontiguousarray(seg, dtype=np.float64).reshape(-1, 2, 3)
    if len(seg) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), np.int64)
    P, inv = np.unique(seg.reshape(-1, 3), axis=0, return_inverse=True)
    return P, inv.reshape(-1, 2).astype(np.int64)


def dist_to_edges(V: np.ndarray, X: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Chord distance of each of V (M,3) to the nearest edge E (K,2) of the positions X (inf without edges)."""
    if len(E) == 0 or len(V) == 0:
        return np.full(len(V), np.inf)
    _, d = seg_point(V[:, None, :], X[E[:, 0]][None], X[E[:, 1]][None])
    return d.min(1)
