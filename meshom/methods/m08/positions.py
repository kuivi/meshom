"""Re-solve the continuous positions with the integer jumps fixed (Huang 2018 §3.5, eq. 9):
min sum_(i,j) |p_j - p_i - rho_ij (D_x q_i + D_y t_i)|^2, p_i in the tangent plane at v_i, by Gauss-Seidel sweeps
started from the Instant Meshes positions (the flow changes the jumps only along short paths, so local sweeps
suffice)."""
import numpy as np
from numba import njit


@njit(cache=True)
def _sweeps(V, Q, ptr, idx, ox, oy, rs, O, iters, active):
    for _ in range(iters):
        for i in active:
            v0, v1, v2 = V[i, 0], V[i, 1], V[i, 2]
            q0, q1, q2 = Q[i, 0], Q[i, 1], Q[i, 2]
            qn = np.sqrt(q0 * q0 + q1 * q1 + q2 * q2)
            q0, q1, q2 = q0 / qn, q1 / qn, q2 / qn
            t0, t1, t2 = v1 * q2 - v2 * q1, v2 * q0 - v0 * q2, v0 * q1 - v1 * q0
            a0 = a1 = a2 = 0.0
            m = ptr[i + 1] - ptr[i]
            if m == 0:
                continue
            for k in range(ptr[i], ptr[i + 1]):
                j = idx[k]
                s = rs[k]
                a0 += O[j, 0] - s * (ox[k] * q0 + oy[k] * t0)
                a1 += O[j, 1] - s * (ox[k] * q1 + oy[k] * t1)
                a2 += O[j, 2] - s * (ox[k] * q2 + oy[k] * t2)
            a0, a1, a2 = a0 / m, a1 / m, a2 / m
            d = (a0 - v0) * v0 + (a1 - v1) * v1 + (a2 - v2) * v2
            O[i, 0], O[i, 1], O[i, 2] = a0 - d * v0, a1 - d * v1, a2 - d * v2


def directed_offsets(ptr: np.ndarray, idx: np.ndarray, edges: np.ndarray, n: int, D: np.ndarray,
                     r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per CSR entry (i -> j): the jump from i to j in i's own frame."""
    from meshom.methods.m08.flow import rot90

    i = np.repeat(np.arange(n), np.diff(ptr))
    j = idx
    key = edges[:, 0] * n + edges[:, 1]
    e = np.searchsorted(key, np.minimum(i, j) * n + np.maximum(i, j))
    fwd = i < j
    # stored (j, i): D is j -> i in j's frame; i -> j in i's frame = -R(-r) D  (r maps i-frame -> j-frame)
    back = -rot90(D[e], -r[e])
    o = np.where(fwd[:, None], D[e], back)
    return o[:, 0].astype(np.float64), o[:, 1].astype(np.float64)


def resolve_positions(lev, Q: np.ndarray, O: np.ndarray, rho: np.ndarray, D: np.ndarray, r: np.ndarray,
                      iters: int = 30, changed: np.ndarray | None = None, rings: int = 2) -> np.ndarray:
    """Gauss-Seidel on eq. 9. With `changed` (edge mask), only the vertices within `rings` rings of a changed edge
    move (the Instant Meshes positions elsewhere stay as they are: a global least-squares solve moves every lattice
    point off its nearest-to-vertex rounding and costs size fidelity, measured)."""
    n = lev.n
    ox, oy = directed_offsets(lev.ptr, lev.idx, lev.edges, n, D, r)
    i = np.repeat(np.arange(n), np.diff(lev.ptr))
    rs = 0.5 * (rho[i] + rho[lev.idx])
    if changed is None:
        active = np.arange(n)
    else:
        mask = np.zeros(n, dtype=bool)
        mask[lev.edges[changed].ravel()] = True
        for _ in range(rings):
            mask[lev.idx[mask[i]]] = True
        active = np.flatnonzero(mask)
    O = O.copy()
    _sweeps(lev.V, Q, lev.ptr, lev.idx, ox, oy, rs, O, iters, active)
    return O


def misfit(lev, Q: np.ndarray, O: np.ndarray, rho: np.ndarray, D: np.ndarray) -> np.ndarray:
    """|O_j - O_i - rho_ij (D_x q_i + D_y t_i)|^2 per edge (the collapse order of the extraction)."""
    i, j = lev.edges[:, 0], lev.edges[:, 1]
    q = Q[i] / np.linalg.norm(Q[i], axis=1, keepdims=True)
    t = np.cross(lev.V[i], q)
    s = 0.5 * (rho[i] + rho[j])
    d = O[j] - O[i] - s[:, None] * (D[:, :1] * q + D[:, 1:] * t)
    return np.einsum("ij,ij->i", d, d)
