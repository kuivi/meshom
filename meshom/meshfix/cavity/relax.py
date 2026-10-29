"""Local relaxation of a filled cavity and its collar on the sphere (decisions B136).

Energy (JAX, float64; with the size term against h):

    E = sum_quad corners [ cos^2 th + W_HINGE relu(cos TH_HINGE - cos th)^2 + W_BAR relu(SIN_MIN - sin th)^2 ]
      + W_TRI sum_tri corners (cos th - 1/2)^2 + W_BAR (same barrier)
      + W_SIZE sum_edges (ln(l / h_e) - bias)^2 + W_WALL sum_edges relu(|ln(l / h_e)| - wall)^2
      + W_RATIO sum_element pairs across an edge (ln(m_a / m_b))^2         (m = mean edge length of the element)

h_e = h at the edge midpoint of the start positions (frozen); bias = median ln(l/h) of the zone's input edges (the
mesh's own offset from h, so the relaxation does not pull the zone to a size the neighbours do not have); wall =
tol - WALL_MARGIN. Variables: tangent offsets of the free nodes in units of their mean edge length, retraction by
normalisation. scipy L-BFGS-B on the jitted value and gradient; arrays padded to powers of two so that jit compiles
a handful of shapes.
"""
import numpy as np

import jax
import jax.numpy as jnp
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

from meshom.meshfix.cavity.nrelax import (C_LIM, MAX_ITER, SIN_MIN, TH_HINGE, W_BAR, W_HINGE,  # noqa: F401
                                          W_RATIO, W_SIZE, W_TRI, W_WALL, WALL_MARGIN, _basis, edges_of,
                                          elem_pairs)


def _pad(n: int) -> int:
    return max(16, 1 << int(np.ceil(np.log2(n + 1))))


def _corner(X, C):
    p = X[C]
    a = jnp.roll(p, -1, 1) - p
    b = jnp.roll(p, 1, 1) - p
    a = a - (a * p).sum(-1, keepdims=True) * p
    b = b - (b * p).sum(-1, keepdims=True) * p
    na = jnp.linalg.norm(a, axis=-1) + 1e-30
    nb = jnp.linalg.norm(b, axis=-1) + 1e-30
    cs = (a * b).sum(-1) / (na * nb)
    sn = (jnp.cross(a, b) * p).sum(-1) / (na * nb)
    return cs, sn


@jax.jit
def _energy(v, P0, E1, E2, sc, fidx, Q, wq, T, wt, Ed, lh, we, bias, wall, EQ, ET, pa, pb, wp):
    v = v.reshape(-1, 2)
    X = P0.at[fidx].add(sc[:, None] * (v[:, 0:1] * E1 + v[:, 1:2] * E2))
    X = X / jnp.linalg.norm(X, axis=1, keepdims=True)
    cq, sq = _corner(X, Q)
    e = jnp.sum(wq[:, None] * (cq ** 2 + W_HINGE * jnp.maximum(C_LIM - cq, 0.0) ** 2
                               + W_BAR * jnp.maximum(SIN_MIN - sq, 0.0) ** 2))
    ct, st = _corner(X, T)
    e += jnp.sum(wt[:, None] * (W_TRI * (ct - 0.5) ** 2 + W_BAR * jnp.maximum(SIN_MIN - st, 0.0) ** 2))
    l = jnp.linalg.norm(X[Ed[:, 0]] - X[Ed[:, 1]], axis=1) + 1e-30
    r = jnp.log(l) - lh
    e += jnp.sum(we * (W_SIZE * (r - bias) ** 2 + W_WALL * jnp.maximum(jnp.abs(r) - wall, 0.0) ** 2))
    mq = jnp.linalg.norm(X[EQ] - X[jnp.roll(EQ, -1, 1)], axis=-1).mean(1)
    mt = jnp.linalg.norm(X[ET] - X[jnp.roll(ET, -1, 1)], axis=-1).mean(1)
    m = jnp.concatenate([mq, mt]) + 1e-30
    e += W_RATIO * jnp.sum(wp * (jnp.log(m[pa]) - jnp.log(m[pb])) ** 2)
    return e


_vg = jax.jit(jax.value_and_grad(_energy))


def relax(X0: np.ndarray, free: np.ndarray, quads: np.ndarray, tris: np.ndarray, h, tol: float,
          bias: float | None = None, max_iter: int = MAX_ITER) -> tuple[np.ndarray, dict]:
    """Relax the free nodes of the local mesh (X0, quads, tris); returns (positions, info)."""
    n = len(X0)
    Ed = edges_of(quads, tris)
    l0 = np.linalg.norm(X0[Ed[:, 0]] - X0[Ed[:, 1]], axis=1)
    mid = X0[Ed[:, 0]] + X0[Ed[:, 1]]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    from meshom.geo import R_EARTH_M

    lh = np.log(h(mid) / R_EARTH_M)
    if bias is None:
        bias = float(np.median(np.log(l0) - lh))
    fidx = np.flatnonzero(free)
    sc = np.zeros(len(fidx))
    s = np.zeros(n)
    c = np.zeros(n)
    np.add.at(s, Ed[:, 0], l0)
    np.add.at(s, Ed[:, 1], l0)
    np.add.at(c, Ed[:, 0], 1)
    np.add.at(c, Ed[:, 1], 1)
    sc = s[fidx] / np.maximum(c[fidx], 1)
    E1, E2 = _basis(X0[fidx])
    pairs = elem_pairs(quads, tris, n)

    Np, Fp = _pad(n), _pad(len(fidx))
    Qp, Tp, Ep, Pp = _pad(len(quads)), _pad(len(tris)), _pad(len(Ed)), _pad(len(pairs))
    dummy = Np - 1
    P0 = np.zeros((Np, 3))
    P0[:n] = X0
    P0[n:] = X0[0]
    fi = np.full(Fp, dummy)
    fi[: len(fidx)] = fidx
    E1p, E2p = np.zeros((Fp, 3)), np.zeros((Fp, 3))
    E1p[: len(fidx)], E2p[: len(fidx)] = E1, E2
    scp = np.zeros(Fp)
    scp[: len(fidx)] = sc
    Qa = np.zeros((Qp, 4), np.int64)
    Qa[: len(quads)] = quads
    Qa[len(quads):] = [0, 1, 2, 3] if n >= 4 else 0
    wq = np.zeros(Qp)
    wq[: len(quads)] = 1.0
    Ta = np.zeros((Tp, 3), np.int64)
    Ta[: len(tris)] = tris
    Ta[len(tris):] = [0, 1, 2]
    wt = np.zeros(Tp)
    wt[: len(tris)] = 1.0
    Ea = np.zeros((Ep, 2), np.int64)
    Ea[: len(Ed)] = Ed
    Ea[len(Ed):] = [0, 1]
    lha = np.zeros(Ep)
    lha[: len(Ed)] = lh
    wea = np.zeros(Ep)
    wea[: len(Ed)] = 1.0
    pa, pb = np.zeros(Pp, np.int64), np.zeros(Pp, np.int64)
    wp = np.zeros(Pp)
    if len(pairs):
        pa[: len(pairs)], pb[: len(pairs)] = pairs[:, 0], pairs[:, 1]
        wp[: len(pairs)] = 1.0
    # element arrays for the ratio term: quads then tris, indices into the same padded layout
    EQ, ET = Qa, Ta
    # remap pair ids of tris (len(quads) + t) to (Qp + t)
    if len(pairs):
        for arr in (pa, pb):
            m = arr[: len(pairs)] >= len(quads)
            arr[: len(pairs)][m] += Qp - len(quads)
    args = [jnp.asarray(x) for x in (P0, E1p, E2p, scp, fi, Qa, wq, Ta, wt, Ea, lha, wea)] + \
        [float(bias), float(tol - WALL_MARGIN)] + [jnp.asarray(x) for x in (EQ, ET, pa, pb, wp)]

    def f(v):
        e, g = _vg(jnp.asarray(v), *args)
        return float(e), np.asarray(g)

    v0 = np.zeros(2 * Fp)
    e0 = f(v0)[0]
    res = minimize(f, v0, jac=True, method="L-BFGS-B", options=dict(maxiter=max_iter, maxcor=20))
    v = res.x.reshape(-1, 2)[: len(fidx)]
    X = X0.copy()
    X[fidx] += sc[:, None] * (v[:, 0:1] * E1 + v[:, 1:2] * E2)
    X[fidx] /= np.linalg.norm(X[fidx], axis=1, keepdims=True)
    return X, {"e0": e0, "e1": float(res.fun), "nit": int(res.nit), "bias": bias}
