"""Free-node geometric relaxation of c_placement's re-extracted region (B133), and point location.

E = sum over quad corners [cos^2 th + w_hinge relu(cos a_h - cos th)^2] + w_tri sum over triangle corners
    (cos th - 1/2)^2 + w_bar sum relu(s0 - sin th)^2 (all corners)
  + sum over edges with a moving end [w_size (ln l/h)^2 + w_wall relu(|ln l/h| - wall)^2],
h at the edge midpoints of the start (it moves little). Moving nodes are offsets in their tangent plane in units of
h, retracted to the sphere; scipy L-BFGS-B with a JAX value-and-grad, float64.
"""
import time

import numpy as np
from scipy.spatial import cKDTree

from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh

RELAX = dict(
    w_hinge=50.0,      # B133: hinge on quad corners above hinge_deg
    hinge_deg=100.0,
    w_tri=0.5,
    w_bar=1e3,         # B133: barrier on corners with sin th < 0.2
    bar_sin=0.2,
    w_size=0.5,        # B133: pull of every edge towards h
    w_wall=500.0,      # B133: wall on |ln l/h| above `wall`
    wall=0.16,
    w_bwall=2000.0,    # B133: wall on boundary (coast) edges above bwall (sliding must not break the coast sizes)
    bwall=0.15,
    w_tri_hinge=50.0,  # B133: hinges on triangle corners above tri_hi_deg / below tri_lo_deg
    tri_hi_deg=105.0,
    tri_lo_deg=30.0,
    w_ratio=2.0,       # B133: (ln of the mean-edge ratio of edge-adjacent elements)^2
    w_rwall=200.0,     # B133: wall on that ratio above ratio_wall
    ratio_wall=1.10,
    relax_iters=1500,
    slide_h=0.25,      # B133: a coast node slides along its coast edges by at most this * h (the coast rule) ...
    slide_frac=0.4,    # ... and this share of each edge
    relax_ring=0,      # extra kept rings that may move (0: only the region)
)


def locate(mesh: Mesh, tr: np.ndarray, owner: np.ndarray, x: np.ndarray, k: int = 12) -> np.ndarray:
    """(M,) the element (owner id) whose split triangle (tr, CCW) contains each unit point x, -1 if none."""
    P = mesh.points
    c = P[tr].mean(1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    tree = cKDTree(c)
    _, nn = tree.query(x, k=min(k, len(tr)))
    out = -np.ones(len(x), np.int64)
    for j in range(nn.shape[1]):
        t = tr[nn[:, j]]
        a, b, cc = P[t[:, 0]], P[t[:, 1]], P[t[:, 2]]
        s1 = np.einsum("ij,ij->i", np.cross(a, b), x)
        s2 = np.einsum("ij,ij->i", np.cross(b, cc), x)
        s3 = np.einsum("ij,ij->i", np.cross(cc, a), x)
        hit = (out < 0) & (s1 >= -1e-15) & (s2 >= -1e-15) & (s3 >= -1e-15)
        out[hit] = owner[nn[hit, j]]
    return out


def relax(mesh: Mesh, h, moving: np.ndarray, opts: dict | None = None, slide: np.ndarray | None = None):
    """slide: (N,) coast nodes that may slide along their two coast edges (the reference polyline itself) by up to
    slide_h * h and slide_frac of each edge."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    jax.config.update("jax_enable_x64", True)
    o = {**RELAX, **(opts or {})}
    t0 = time.perf_counter()
    moving = moving.copy()
    if o["relax_ring"]:
        from meshom.meshfix.placement.fields import elements

        el, nv = elements(mesh)
        bnd = mesh.node_flag.astype(bool)
        for _ in range(int(o["relax_ring"])):
            touch = moving[np.where(el >= 0, el, 0)] & (el >= 0)
            em = touch.any(1)
            nodes = np.unique(el[em][el[em] >= 0])
            moving[nodes[~bnd[nodes]]] = True
    slide = np.zeros(mesh.n_nodes, bool) if slide is None else slide.copy()
    if slide.any():
        from meshom.meshfix.cases import boundary_edges

        be = boundary_edges(mesh)
        nxt = -np.ones(mesh.n_nodes, np.int64)
        prv = -np.ones(mesh.n_nodes, np.int64)
        cnt = np.bincount(be[:, 0], minlength=mesh.n_nodes)
        nxt[be[:, 0]] = be[:, 1]
        prv[be[:, 1]] = be[:, 0]
        slide &= (cnt == 1) & (nxt >= 0) & (prv >= 0) & ~moving
        # only coast nodes next to the moving region
        near = np.zeros(mesh.n_nodes, bool)
        for C in (mesh.quads, mesh.tris):
            if len(C):
                hitc = moving[C].any(1)
                near[C[hitc].ravel()] = True
        slide &= near
    act = moving | slide
    # the sub-problem: elements with a moving or sliding node
    Qm = mesh.quads[act[mesh.quads].any(1)]
    Tm = mesh.tris[act[mesh.tris].any(1)] if mesh.n_tris else np.zeros((0, 3), np.int64)
    nodes = np.unique(np.r_[Qm.ravel(), Tm.ravel()])
    loc = -np.ones(mesh.n_nodes, np.int64)
    loc[nodes] = np.arange(len(nodes))
    Ql, Tl = loc[Qm], loc[Tm]
    P0 = mesh.points[nodes].copy()
    free = np.flatnonzero(moving[nodes])
    e = np.vstack([np.stack([Ql, np.roll(Ql, -1, 1)], -1).reshape(-1, 2),
                   np.stack([Tl, np.roll(Tl, -1, 1)], -1).reshape(-1, 2)])
    e = np.unique(np.sort(e, 1), axis=0)
    e = e[act[nodes][e].any(1)]
    sl = np.flatnonzero(slide[nodes])
    sl_g = nodes[sl]
    Pn, Pp = mesh.points[nxt[sl_g]] if len(sl) else np.zeros((0, 3)), mesh.points[prv[sl_g]] if len(sl) else np.zeros((0, 3))
    hs = h(P0[sl]) / R_EARTH_M if len(sl) else np.zeros(0)
    ln_ = np.linalg.norm(Pn - P0[sl], axis=1) if len(sl) else np.zeros(0)
    lp_ = np.linalg.norm(Pp - P0[sl], axis=1) if len(sl) else np.zeros(0)
    sig = np.minimum(np.minimum(o["slide_h"] * hs / np.maximum(ln_, 1e-12), o["slide_h"] * hs / np.maximum(lp_, 1e-12)),
                     o["slide_frac"]) if len(sl) else np.zeros(0)
    mid = P0[e[:, 0]] + P0[e[:, 1]]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    he = h(mid) / R_EARTH_M
    # edge-adjacent element pairs (elements: quads then triangles, local ids)
    cells = [(Ql, 4), (Tl, 3)]
    ha, hb, hel = [], [], []
    off = 0
    for C, k in cells:
        if len(C):
            ha.append(C.ravel())
            hb.append(np.roll(C, -1, 1).ravel())
            hel.append(np.repeat(np.arange(len(C)) + off, k))
        off += len(C)
    ha, hb, hel = np.concatenate(ha), np.concatenate(hb), np.concatenate(hel)
    nn_ = len(nodes)
    kf = ha * nn_ + hb
    kr = hb * nn_ + ha
    so = np.argsort(kf)
    pos = np.searchsorted(kf[so], kr)
    pos = np.minimum(pos, len(kf) - 1)
    hit = kf[so][pos] == kr
    pairs = np.column_stack([hel[hit], hel[so][pos[hit]]])
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    pf = P0[free]
    z = np.array([0, 0, 1.0])
    a = np.where(np.abs(pf[:, 2:3]) < 0.9, z[None], np.array([[1.0, 0, 0]]))
    E1 = np.cross(a, pf)
    E1 /= np.linalg.norm(E1, axis=1, keepdims=True)
    E2 = np.cross(pf, E1)
    hf = h(pf) / R_EARTH_M
    P0j, E1j, E2j, hj, fj = map(jnp.asarray, (P0, E1, E2, hf, free))
    slj, Pnj, Ppj, sigj = map(jnp.asarray, (sl, Pn, Pp, sig))
    nfree, nsl = len(free), len(sl)
    bnd_e = mesh.node_flag.astype(bool)[nodes][e].all(1).astype(float)
    Qj, Tj, ej, hej, prj, bej = map(jnp.asarray, (Ql, Tl, e, he, pairs, bnd_e))
    lrw = np.log(o["ratio_wall"])
    c_h = np.cos(np.radians(o["hinge_deg"]))
    c_thi, c_tlo = np.cos(np.radians(o["tri_hi_deg"])), np.cos(np.radians(o["tri_lo_deg"]))

    def pos(z):
        x = z[: 2 * nfree].reshape(-1, 2)
        X = P0j.at[fj].add(hj[:, None] * (x[:, :1] * E1j + x[:, 1:] * E2j))
        if nsl:
            sv = sigj * jnp.tanh(z[2 * nfree:])
            p = P0j[slj]
            q = p + jnp.maximum(sv, 0.0)[:, None] * (Pnj - p) + jnp.maximum(-sv, 0.0)[:, None] * (Ppj - p)
            X = X.at[slj].set(q)
        return X / jnp.linalg.norm(X, axis=1, keepdims=True)

    def corners(X, C):
        p = X[C]
        u = jnp.roll(p, -1, 1) - p
        w = jnp.roll(p, 1, 1) - p
        u = u - (u * p).sum(-1, keepdims=True) * p
        w = w - (w * p).sum(-1, keepdims=True) * p
        nu = jnp.linalg.norm(u, axis=-1)
        nw = jnp.linalg.norm(w, axis=-1)
        cs = (u * w).sum(-1) / (nu * nw)
        sn = (jnp.cross(u, w) * p).sum(-1) / (nu * nw)
        return cs, sn

    def energy(x):
        X = pos(x)
        en = 0.0
        if len(Ql):
            cq, sq = corners(X, Qj)
            en += jnp.sum(cq ** 2) + o["w_hinge"] * jnp.sum(jnp.maximum(c_h - cq, 0.0) ** 2)
            en += o["w_bar"] * jnp.sum(jnp.maximum(o["bar_sin"] - sq, 0.0) ** 2)
        if len(Tl):
            ct, st = corners(X, Tj)
            en += o["w_tri"] * jnp.sum((ct - 0.5) ** 2) + o["w_bar"] * jnp.sum(jnp.maximum(o["bar_sin"] - st, 0.0) ** 2)
            en += o["w_tri_hinge"] * (jnp.sum(jnp.maximum(c_thi - ct, 0.0) ** 2) + jnp.sum(jnp.maximum(ct - c_tlo, 0.0) ** 2))
        l = jnp.linalg.norm(X[ej[:, 0]] - X[ej[:, 1]], axis=1)
        r = jnp.log(l / hej)
        en += o["w_size"] * jnp.sum(r ** 2) + o["w_wall"] * jnp.sum(jnp.maximum(jnp.abs(r) - o["wall"], 0.0) ** 2)
        en += o["w_bwall"] * jnp.sum(bej * jnp.maximum(jnp.abs(r) - o["bwall"], 0.0) ** 2)
        if len(pairs):
            means = []
            if len(Ql):
                pq = X[Qj]
                means.append(jnp.linalg.norm(pq - jnp.roll(pq, -1, 1), axis=-1).mean(1))
            if len(Tl):
                pt = X[Tj]
                means.append(jnp.linalg.norm(pt - jnp.roll(pt, -1, 1), axis=-1).mean(1))
            lm = jnp.log(jnp.concatenate(means))
            dr = lm[prj[:, 0]] - lm[prj[:, 1]]
            en += o["w_ratio"] * jnp.sum(dr ** 2) + o["w_rwall"] * jnp.sum(jnp.maximum(jnp.abs(dr) - lrw, 0.0) ** 2)
        return en

    fg = jax.jit(jax.value_and_grad(energy))

    def f(x):
        val, g = fg(jnp.asarray(x))
        return float(val), np.asarray(g)

    x0 = np.zeros(2 * len(free) + len(sl))
    e0 = f(x0)[0]
    res = minimize(f, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=int(o["relax_iters"]), maxcor=20))
    X = np.asarray(pos(jnp.asarray(res.x)))
    pts = mesh.points.copy()
    pts[nodes] = X
    out = Mesh(pts, mesh.tris, mesh.quads, mesh.node_flag, mesh.tri_zone)
    return out, dict(free=int(len(free)), sliding=int(len(sl)), iters=int(res.nit), e0=float(e0), e1=float(res.fun), msg=str(res.message),
                     s=time.perf_counter() - t0)
