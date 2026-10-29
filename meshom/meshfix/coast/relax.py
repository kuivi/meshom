"""Relaxation of a coast zone with sliding coast nodes (decisions B144).

D's relaxation energy (cavity/relax.py, B136; a copy here, so D's own stays as it is) plus one term for
the sliding coast nodes:

    + W_COAST sum_slide (d / s)^2

d = chord distance of the node to the nearest of its allowed reference segments (refcoast.RefCoast.allowed: the
segments reachable along its own shore), s = the node's mean edge length. After L-BFGS the sliding nodes are
projected exactly onto their nearest allowed segment, so every coast node of an accepted candidate lies on the
reference coast.
"""
import numpy as np

import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from meshom.geo import R_EARTH_M
from meshom.meshfix.coast.refcoast import project  # noqa: F401 (moved there: no JAX import for it)

jax.config.update("jax_enable_x64", True)

W_HINGE = 50.0
TH_HINGE = 100.0
W_BAR = 1e3
SIN_MIN = 0.2
W_TRI = 0.5
W_SIZE = 0.2
W_WALL = 200.0
WALL_MARGIN = 0.02
W_CAP = 1e4              # B144: a second, stiff wall at CAP, below the acceptance cap 0.30 (D's 200 at 0.18
CAP = 0.27               # let the angle terms push zone edges to 0.33-0.41 |ln s/h|, rejected by the cap)
W_RATIO = 2.0
W_RWALL = 1e3            # B144: a wall on the element size ratio across an edge beyond RWALL (the gate's p99 is
RWALL = float(np.log(1.12))  # 1.15; collapsing a column doubles the neighbours' ratio before the relaxation)
W_COAST = 1e3            # B144: penalty on the distance of a sliding node to its allowed coast segments
MAX_ITER = 300
C_LIM = float(np.cos(np.radians(TH_HINGE)))


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
def _energy(v, P0, E1, E2, sc, fidx, Q, wq, T, wt, Ed, lh, we, bias, wall, w_wall, EQ, ET, pa, pb, wp, ci, cA, cB, cs,
            wc):
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
    e += jnp.sum(we * (W_SIZE * (r - bias) ** 2 + w_wall * jnp.maximum(jnp.abs(r) - wall, 0.0) ** 2
                       + W_CAP * jnp.maximum(jnp.abs(r) - CAP, 0.0) ** 2))
    mq = jnp.linalg.norm(X[EQ] - X[jnp.roll(EQ, -1, 1)], axis=-1).mean(1)
    mt = jnp.linalg.norm(X[ET] - X[jnp.roll(ET, -1, 1)], axis=-1).mean(1)
    m = jnp.concatenate([mq, mt]) + 1e-30
    lr = jnp.log(m[pa]) - jnp.log(m[pb])
    e += jnp.sum(wp * (W_RATIO * lr ** 2 + W_RWALL * jnp.maximum(jnp.abs(lr) - RWALL, 0.0) ** 2))
    # sliding coast nodes: squared distance to the nearest allowed segment, in units of the node's edge length
    x = X[ci][:, None, :]
    d = cB - cA
    t = jnp.clip(((x - cA) * d).sum(-1) / jnp.maximum((d * d).sum(-1), 1e-300), 0.0, 1.0)
    q = cA + t[..., None] * d
    dist2 = ((x - q) ** 2).sum(-1).min(1)
    e += W_COAST * jnp.sum(wc * dist2 / cs ** 2)
    return e


_vg = jax.jit(jax.value_and_grad(_energy))


def _basis(p):
    k = np.argmin(np.abs(p), axis=1)
    a = np.zeros_like(p)
    a[np.arange(len(p)), k] = 1.0
    e1 = a - (a * p).sum(1, keepdims=True) * p
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(p, e1)


def edges_of(quads: np.ndarray, tris: np.ndarray) -> np.ndarray:
    e = [np.stack([c, np.roll(c, -1, 1)], -1).reshape(-1, 2) for c in (quads, tris) if len(c)]
    e = np.sort(np.concatenate(e), axis=1)
    return np.unique(e, axis=0)


def elem_pairs(quads: np.ndarray, tris: np.ndarray, n: int) -> np.ndarray:
    """(K,2) element index pairs (quads first, then tris) across interior edges."""
    cells = [(c, np.roll(c, -1, 1)) for c in (quads, tris) if len(c)]
    a = np.concatenate([x.ravel() for x, _ in cells])
    b = np.concatenate([y.ravel() for _, y in cells])
    eid = np.concatenate([np.repeat(np.arange(len(quads)), 4), len(quads) + np.repeat(np.arange(len(tris)), 3)]) \
        if len(tris) else np.repeat(np.arange(len(quads)), 4)
    key = np.minimum(a, b) * n + np.maximum(a, b)
    o = np.argsort(key, kind="stable")
    ks = key[o]
    same = np.flatnonzero(ks[1:] == ks[:-1])
    return np.column_stack([eid[o[same]], eid[o[same + 1]]])


def relax(X0: np.ndarray, free: np.ndarray, quads: np.ndarray, tris: np.ndarray, h, tol: float,
          slide: np.ndarray | None = None, SA: np.ndarray | None = None, SB: np.ndarray | None = None,
          bias: float | None = None, max_iter: int = MAX_ITER, w_wall: float = W_WALL) -> tuple[np.ndarray, dict]:
    """Relax the free nodes of the local mesh (X0, quads, tris); slide (local node ids, a subset of the free ones)
    with their allowed segments SA, SB (len(slide), K, 3) slide along the coast and end projected onto it."""
    n = len(X0)
    Ed = edges_of(quads, tris)
    l0 = np.linalg.norm(X0[Ed[:, 0]] - X0[Ed[:, 1]], axis=1)
    mid = X0[Ed[:, 0]] + X0[Ed[:, 1]]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    lh = np.log(h(mid) / R_EARTH_M)
    if bias is None:
        bias = float(np.median(np.log(l0) - lh))
    fidx = np.flatnonzero(free)
    s = np.zeros(n)
    c = np.zeros(n)
    np.add.at(s, Ed[:, 0], l0)
    np.add.at(s, Ed[:, 1], l0)
    np.add.at(c, Ed[:, 0], 1)
    np.add.at(c, Ed[:, 1], 1)
    mean_e = s / np.maximum(c, 1)
    sc = mean_e[fidx]
    E1, E2 = _basis(X0[fidx])
    pairs = elem_pairs(quads, tris, n)
    slide = np.zeros(0, np.int64) if slide is None else np.asarray(slide, np.int64)
    K = SA.shape[1] if slide.size else 1

    Np, Fp = _pad(n), _pad(len(fidx))
    Qp, Tp, Ep, Pp, Cp = _pad(len(quads)), _pad(len(tris)), _pad(len(Ed)), _pad(len(pairs)), _pad(len(slide))
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
        for arr in (pa, pb):
            m = arr[: len(pairs)] >= len(quads)
            arr[: len(pairs)][m] += Qp - len(quads)
    ci = np.zeros(Cp, np.int64)
    cA, cB = np.zeros((Cp, K, 3)), np.zeros((Cp, K, 3))
    cA[:] = [1.0, 0.0, 0.0]
    cB[:] = [0.0, 1.0, 0.0]
    cs = np.ones(Cp)
    wc = np.zeros(Cp)
    if slide.size:
        ci[: len(slide)] = slide
        cA[: len(slide)], cB[: len(slide)] = SA, SB
        cs[: len(slide)] = np.maximum(mean_e[slide], 1e-12)
        wc[: len(slide)] = 1.0
    args = [jnp.asarray(x) for x in (P0, E1p, E2p, scp, fi, Qa, wq, Ta, wt, Ea, lha, wea)] + \
        [float(bias), float(tol - WALL_MARGIN), float(w_wall)] + \
        [jnp.asarray(x) for x in (Qa, Ta, pa, pb, wp, ci, cA, cB, cs, wc)]

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
    if slide.size:
        X[slide] = project(X[slide], SA, SB)
    return X, {"e0": e0, "e1": float(res.fun), "nit": int(res.nit), "bias": bias}
