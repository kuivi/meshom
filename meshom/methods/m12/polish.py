"""Optional size+shape polish of a closed quad mesh on the sphere (the variational alternative to pure OT).

Energy over the 4 corners of every quad (TMOP-type, Knupp 2012): corner edges e1 = p_next - p,
e2 = p_prev - p, a = (e1 x e2) . p (twice the corner triangle area);
    shape  mu = (|e1|^2 + |e2|^2) / (2 a) - 1   (0 for a square corner, -> inf as a -> 0+: no inversion)
    size   (ln (a / (c h^2)))^2                  (h at the quad centroid, c keeps the total area)
E = sum (mu + lam * size). Topology fixed; nodes are free on the sphere (positions normalised inside E).
Minimised by L-BFGS with a backtracking line search that rejects steps leaving the valid set; JAX (x64) gradients.
lam -> inf reproduces the equidistributed (OT) sizes; small lam favours square cells and gives up size fidelity.
"""
import numpy as np

from meshom.geo import R_EARTH_M
from meshom.size import SizeField


def _energy_fn(quads: np.ndarray, lam: float):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    qd = jnp.asarray(quads)

    def energy(xf, tgt):
        x = xf.reshape(-1, 3)
        p = x / jnp.linalg.norm(x, axis=1, keepdims=True)
        q = p[qd]                                                      # (Q,4,3)
        e1 = jnp.roll(q, -1, axis=1) - q
        e2 = jnp.roll(q, 1, axis=1) - q
        a = jnp.einsum("qki,qki->qk", jnp.cross(e1, e2), q)
        ok = jnp.all(a > 0)
        a_s = jnp.where(a > 0, a, 1.0)
        mu = (jnp.sum(e1 * e1, axis=2) + jnp.sum(e2 * e2, axis=2)) / (2.0 * a_s) - 1.0
        sz = jnp.log(a_s / tgt[:, None]) ** 2
        return jnp.where(ok, jnp.sum(mu + lam * sz), jnp.inf)

    return jax.jit(jax.value_and_grad(energy))


def polish(x: np.ndarray, quads: np.ndarray, h: SizeField, lam: float = 1.0, rounds: int = 3, iters: int = 150,
           verbose: bool = False) -> np.ndarray:
    """L-BFGS on the size+shape energy; h re-evaluated at the quad centroids at the start of each round."""
    vg = _energy_fn(quads, lam)
    xf = x.ravel().copy()
    for r in range(rounds):
        p = xf.reshape(-1, 3)
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        c = p[quads].sum(axis=1)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        h2 = (h(c) / R_EARTH_M) ** 2
        tgt = h2 * (4.0 * np.pi / h2.sum())                            # sum of quad targets = sphere area
        xf = _lbfgs(lambda v: vg(v, tgt), p.ravel(), iters, verbose, r)
    p = xf.reshape(-1, 3)
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def _lbfgs(vg, x0: np.ndarray, iters: int, verbose: bool, tag: int, mem: int = 10) -> np.ndarray:
    x = x0.copy()
    f, g = vg(x)
    f, g = float(f), np.asarray(g)
    S, Y = [], []
    for it in range(iters):
        q = g.copy()
        al = []
        for s, y in zip(reversed(S), reversed(Y)):
            a = s @ q / (y @ s)
            al.append(a)
            q -= a * y
        if S:
            q *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
        for (s, y), a in zip(zip(S, Y), reversed(al)):
            q += (a - y @ q / (y @ s)) * s
        d = -q
        gd = g @ d
        if gd >= 0:
            d, gd, S, Y = -g, -(g @ g), [], []
        step = 1.0 if S else 1e-3 / max(np.abs(g).max(), 1e-30)
        for _ in range(40):
            fn, gn = vg(x + step * d)
            fn = float(fn)
            if np.isfinite(fn) and fn <= f + 1e-4 * step * gd:
                break
            step *= 0.5
        else:
            break
        gn = np.asarray(gn)
        s, y = step * d, gn - g
        if y @ s > 1e-30:
            S.append(s)
            Y.append(y)
            if len(S) > mem:
                S.pop(0)
                Y.pop(0)
        x, df, f, g = x + s, f - fn, fn, gn
        if verbose and it % 50 == 0:
            print(f"  polish round {tag} it {it:4d}  E {f:.6g}", flush=True)
        if df < 1e-10 * abs(f):
            break
    return x
