"""m04: Lp-CVT (L_inf Lloyd) points aligned with the cross field -> sphere Delaunay -> recombination.

Levy & Liu 2010: minimise the Lp-CVT energy (p = 8 ~ L_inf) whose minimisers are Voronoi cells
that are squares of side h aligned with an anisotropy frame; the Delaunay dual of a square lattice is right triangles,
which pair into squares. Ekelschot et al. 2018: the same with a
metric, BFGS, and recombination (Blossom-Quad) of the right triangles.

1. Coast nodes from resample_loops (fixed). Points: poisson_points with the node count of square cells,
   N ~ integral of h^-2 dA over the water (minus half a cell per coast node).
2. Frame per point from meshom.field.cross_field (default "cones", on background_mesh(field_factor = 1)),
   interpolated as the 4th-power mean of the k nearest field nodes.
3. `rounds` outer rounds; each freezes per point the frame and h (at the current position) and runs L-BFGS-B
   (scipy, `iters` iterations, each displacement coordinate bounded by max_step * h per round) on the tangent
   displacements of the free points. Energy and exact gradient (meshom/methods/m04/energy.py, Numba) over the
   full-sphere Voronoi diagram, recomputed at every evaluation (Lawson flips from the previous triangulation, Qhull
   when a triangle inverted). Only free points own cells; the coast nodes' cells reach over the empty land and clip
   the cells of the free points next to the coast (coast nodes stay fixed and clip only by their Voronoi
   cells). A point that crossed onto land would own a cell over the whole land mass, so the energy keeps points in
   the water (the rare point left in land is dropped).
4. Multilevel start (`levels` = L, as Levy-Liu §5.3 and Ekelschot §III.B): optimise at 2^L h first, then insert the
   midpoint of every water Delaunay edge (a square lattice becomes the lattice of half the spacing, still aligned),
   swap in the coast nodes of the finer level, optimise again. L is lowered until 2^L max|grad h| <= g_lim: on steeper
   coarse levels the per-owner size lets coarse cells invade fine regions and points collapse (disc, scale 4, L = 2:
   |ln s/h| p99 2.1).
5. sphere_delaunay + recover_edges + remove_land, recombine (greedy, price), meshom.smooth kind "size" (10 sweeps)
   and one "angle" sweep (coast fixed).
"""
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from meshom.delaunay import background_mesh, poisson_points, sphere_delaunay
from meshom.domain import Domain, in_water, loop_edges, resample_loops
from meshom.geo import R_EARTH_M, tangent_basis
from meshom.mesh import Mesh
from meshom.methods import register
from meshom.size import SizeField, scaled

DEFAULTS = dict(
    p=8,                  # L_p norm of the energy (8 ~ L_inf, Levy-Liu)
    rounds=8,             # outer rounds (frames and h re-evaluated at the current points)
    iters=40,             # L-BFGS iterations per round
    levels=2,             # multilevel start: optimise at 2^levels h first, refine by edge midpoints
    coarse_rounds=10,     # rounds on the coarser levels
    g_lim=0.5,            # a level is used only if 2^level max|grad h| <= g_lim (steeper coarse levels collapse)
    field="cones",        # meshom.field.cross_field method ("cones" | "smoothest" | "none" = Levi-Civita only)
    field_k=4,            # nearest field nodes in the frame interpolation
    field_factor=1.0,     # background_mesh(factor) of the cross field (0.5: same quality, band scale 1 peak 3.2 GB)
    price=1.0 / 3.0,      # recombination price of a leftover triangle
    recombine="greedy",
    post_smooth="size",   # meshom.smooth kind after recombination (None = no smoothing)
    post_iters=10,
    post_angle=1,         # then this many meshom.smooth "angle" sweeps (fixes the few obtuse leftover triangles)
    seed=0,
    max_step=0.3,         # bound on each displacement coordinate per round, units of h (None = unbounded)
    lawson=True,          # keep the Delaunay triangulation by Lawson flips between evaluations (Qhull fallback)
    verbose=False,
)
ALPHA_SQ = 0.838          # poisson_points alpha for density 1/h^2 (density ~ 0.70 / (alpha h)^2, calibrated by m01)


def _opts(opts: dict | None) -> dict:
    o = dict(DEFAULTS)
    o.update(opts or {})
    return o


def _log(o: dict, *a) -> None:
    if o["verbose"]:
        print("[m04]", *a, file=sys.stderr, flush=True)


def _fibonacci(m: int) -> np.ndarray:
    i = np.arange(m) + 0.5
    z = 1.0 - 2.0 * i / m
    ph = np.pi * (1.0 + 5**0.5) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(ph), r * np.sin(ph), z])


def square_count(domain: Domain, h: SizeField, m: int = 200_000) -> float:
    """Number of square cells of side h over the water: integral of h^-2 dA."""
    lat = _fibonacci(m)
    wet = in_water(domain, lat) if domain.loops else np.ones(m, dtype=bool)
    return float(4.0 * np.pi * R_EARTH_M**2 / m * np.sum(h(lat[wet]) ** -2.0))


class Frames:
    """Cross-field frame u (unit tangent) at arbitrary points: 4th-power mean of the k nearest field nodes."""

    def __init__(self, field, k: int = 4):
        self.tree = cKDTree(field.mesh.points)
        self.u = field.u
        self.k = k

    def __call__(self, x: np.ndarray) -> np.ndarray:
        d, j = self.tree.query(x, k=self.k)
        e1, e2 = tangent_basis(x)
        uj = self.u[j]                                                    # (N,k,3)
        ang = np.arctan2(np.einsum("nkd,nd->nk", uj, e2), np.einsum("nkd,nd->nk", uj, e1))
        w = 1.0 / (d * d + (1e-3 * d[:, :1]) ** 2 + 1e-300)
        phi = np.angle(np.sum(w * np.exp(4j * ang), axis=1)) / 4.0
        return np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2


def _frames(domain: Domain, h: SizeField, o: dict) -> tuple[Frames, float]:
    """Frames of the cross field and max |grad h| (p99.9 over the edges of the field's background mesh)."""
    from meshom.field import cross_field
    from meshom.size import uniform

    bg = background_mesh(domain, h, factor=o["field_factor"])
    bn = np.flatnonzero(bg.node_flag == 1) if domain.loops else None
    h_conn = uniform(1.0) if o["field"] == "none" else h               # "none": h-blind Levi-Civita field
    cf = cross_field(bg, h_conn, n=4, boundary_nodes=bn, method="cones" if o["field"] == "none" else o["field"])
    a, b = bg.tris.ravel(), np.roll(bg.tris, -1, axis=1).ravel()
    hv = h(bg.points)
    g = np.abs(hv[a] - hv[b]) / (np.linalg.norm(bg.points[a] - bg.points[b], axis=1) * R_EARTH_M)
    return Frames(cf, o["field_k"]), float(np.percentile(g, 99.9))


def _initial_points(domain: Domain, h: SizeField, loops: list, seed: int) -> np.ndarray:
    nb = sum(len(lp) for lp in loops)
    n_star = square_count(domain, h) - 0.5 * nb
    pts = poisson_points(domain, h, loops, alpha=ALPHA_SQ, seed=seed)
    corr = np.sqrt((len(pts) - nb) / max(n_star, 1.0))
    if abs(corr - 1.0) > 0.01:
        pts = poisson_points(domain, h, loops, alpha=ALPHA_SQ * corr, seed=seed)
    return pts


def lp_cvt(x: np.ndarray, nfix: int, h: SizeField, frames, o: dict, rounds: int) -> tuple[np.ndarray, dict]:
    """Minimise the Lp-CVT energy over the free points x[nfix:] (unit vectors). Returns the points and stats."""
    from meshom.methods.m04.energy import delaunay_update, lp_energy, unit_square_integral

    p = int(o["p"])
    n = len(x)
    free = np.arange(nfix, n)
    own = np.zeros(n, dtype=bool)
    own[nfix:] = True
    h_ref = float(np.median(h(x[free])))
    s = h_ref / R_EARTH_M                                   # variable unit: tangent displacement of h_ref
    cp = unit_square_integral(p)
    stats = dict(evals=0, iters=0, qhull=0, t_hull=0.0, t_energy=0.0, energy=[])
    state = dict(tris=None)
    for r in range(rounds):
        hv = h(x)
        u = frames(x)
        v = np.cross(x, u)
        inv_h = R_EARTH_M / hv
        wgt = (hv / h_ref) ** 2 / cp
        x0 = x.copy()
        uf, vf = u[free], v[free]

        def place(z):
            z = z.reshape(-1, 2)
            q = x0[free] + s * (z[:, :1] * uf + z[:, 1:] * vf)
            nq = np.linalg.norm(q, axis=1)
            xc = x0.copy()
            xc[free] = q / nq[:, None]
            return xc, nq

        def fg(z):
            xc, nq = place(z)
            t0 = time.perf_counter()
            if o["lawson"]:
                state["tris"], full = delaunay_update(xc, state["tris"])
                tris = state["tris"][0]
            else:
                tris, full = sphere_delaunay(xc), True
            stats["qhull"] += full
            t1 = time.perf_counter()
            e, g, _ = lp_energy(xc, tris, own, u, v, inv_h, wgt, p)
            stats["t_hull"] += t1 - t0
            stats["t_energy"] += time.perf_counter() - t1
            stats["evals"] += 1
            g = g[free]
            xf = xc[free]
            g = (g - np.einsum("ij,ij->i", g, xf)[:, None] * xf) / nq[:, None]
            gz = s * np.column_stack([np.einsum("ij,ij->i", g, uf), np.einsum("ij,ij->i", g, vf)])
            return e, gz.ravel()

        e0 = fg(np.zeros(2 * len(free)))[0]
        bounds = None
        if o["max_step"]:
            b = np.repeat(o["max_step"] * hv[free] / h_ref, 2)
            bounds = list(zip(-b, b))
        res = minimize(fg, np.zeros(2 * len(free)), jac=True, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=int(o["iters"]), maxcor=10, ftol=1e-12, gtol=1e-8))
        x = place(res.x)[0]
        stats["iters"] += int(res.nit)
        stats["energy"].append((float(e0) / len(free), float(res.fun) / len(free)))
        _log(o, f"round {r}: E/N {e0 / len(free):.4f} -> {res.fun / len(free):.4f}, {res.nit} it, "
                f"{res.nfev} ev, {res.message}")
    return x, stats


def _triangulate(x: np.ndarray, bedges: np.ndarray) -> np.ndarray:
    return sphere_delaunay(x, bedges)


def _refine(x: np.ndarray, nb: int, bedges: np.ndarray, domain: Domain, h_fine: SizeField,
            loops_fine: list) -> np.ndarray:
    """Interior points plus the midpoints of all water Delaunay edges (coast edges excepted), with the coast nodes
    of the finer level in front; midpoints in land or closer than h/2 to a fine coast node are dropped."""
    tris = _triangulate(x, bedges)
    a = tris.ravel()
    b = np.roll(tris, -1, axis=1).ravel()
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = np.unique(lo * len(x) + hi)
    e = np.column_stack([key // len(x), key % len(x)])
    if len(bedges):
        bkey = np.minimum(bedges[:, 0], bedges[:, 1]) * len(x) + np.maximum(bedges[:, 0], bedges[:, 1])
        e = e[~np.isin(key, bkey)]
    mid = x[e[:, 0]] + x[e[:, 1]]
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    new = np.vstack([x[nb:], mid])
    if loops_fine:
        coast = np.vstack(loops_fine)
        new = new[in_water(Domain(loops_fine, domain.name), new)]
        d, j = cKDTree(coast).query(new)
        new = new[d * R_EARTH_M > 0.5 * h_fine(coast[j])]
        return np.vstack([coast, new])
    return new


def point_set(domain: Domain, h: SizeField, o: dict) -> tuple[np.ndarray, list, dict]:
    """Optimised points (coast nodes first), the coast loops, and stats."""
    t0 = time.perf_counter()
    frames, g_max = _frames(domain, h, o)
    t_field = time.perf_counter() - t0
    levels = int(o["levels"])
    while levels > 0 and g_max * 2.0**levels > o["g_lim"]:
        levels -= 1
    stats = dict(t_field=t_field, g_max=g_max, n_levels=levels, levels=[])
    _log(o, f"field {t_field:.1f} s, max |grad h| {g_max:.3f}, levels {levels}")
    x = None
    for lev in range(levels, -1, -1):
        hl = scaled(h, 2.0**lev) if lev else h
        loops = resample_loops(domain, hl)
        nb = sum(len(lp) for lp in loops)
        if x is None:
            x = _initial_points(domain, hl, loops, o["seed"])
        else:
            x = _refine(x, nb_prev, bedges_prev, domain, hl, loops)
        rounds = o["rounds"] if lev == 0 else int(o["coarse_rounds"])
        t1 = time.perf_counter()
        x, st = lp_cvt(x, nb, hl, frames, o, rounds)
        st["t"] = time.perf_counter() - t1
        st["n"] = len(x)
        stats["levels"].append(st)
        _log(o, f"level {lev}: {len(x)} points, {st['iters']} it, {st['evals']} ev ({st['qhull']} qhull), {st['t']:.1f} s "
                f"(hull {st['t_hull']:.1f}, energy {st['t_energy']:.1f})")
        nb_prev, bedges_prev = nb, loop_edges(loops)
    return x, loops, stats


def triangle_mesh(domain: Domain, h: SizeField, opts: dict | None = None) -> Mesh:
    """Delaunay triangulation of the Lp-CVT points (right triangles where the cells are aligned squares)."""
    o = _opts(opts)
    x, loops, stats = point_set(domain, h, o)
    nb = sum(len(lp) for lp in loops)
    tris = _triangulate(x, loop_edges(loops))
    used = np.zeros(len(x), dtype=bool)
    used[tris.ravel()] = True
    used[:nb] = True
    if not used.all():                                      # a free point left in land: drop it
        remap = np.cumsum(used) - 1
        x, tris = x[used], remap[tris]
    flag = np.zeros(len(x), dtype=np.int8)
    flag[:nb] = 1
    return Mesh(x, tris, np.zeros((0, 4), dtype=np.int64), flag)


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """Mixed output: Lp-CVT triangles recombined into quads, then size smoothing with the coast fixed."""
    from meshom.recombine import recombine

    o = _opts(opts)
    mesh = recombine(triangle_mesh(domain, h, o), o["price"], o["recombine"])
    if o["post_smooth"]:
        from meshom.smooth import smooth

        mesh = smooth(mesh, h, mesh.node_flag == 1, iters=o["post_iters"], kind=o["post_smooth"])
    if o["post_angle"]:
        from meshom.smooth import smooth

        mesh = smooth(mesh, h, mesh.node_flag == 1, iters=o["post_angle"], kind="angle")
    return mesh
