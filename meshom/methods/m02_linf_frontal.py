"""m02: L-infinity frontal point insertion along the cross field -> sphere Delaunay -> recombination.

Remacle et al. 2013 (L-inf frontal Delaunay, Gmsh algorithm 8), in its packing form (Baudouin et al. 2014, Gmsh
algorithm 9) as Georgiadis et al. 2017 used it for coastal ocean meshes on the sphere:

1. background_mesh(domain, h, bg_factor) and meshom.field.cross_field on it ("cones" by default: integrable field whose
   singularities follow rho_c; coast nodes fixed to the coast tangent).
2. Points (meshom/methods/m02/packing.py, Numba): a FIFO seeded with the resampled coast nodes in loop order (closed
   sphere: one bg node, the farthest from the field's cones); each popped point proposes 4 points at metric distance 1
   (geodesic, along its 4 cross directions, midpoint-corrected for the turning of the field); a candidate is kept if
   it is in the water (the walk on the bg mesh does not leave it) and no point lies in its L-inf box of half-size
   alpha h in the local cross frame. A step blocked by the other front at a gap D >= seam h gets a point at D / 2
   instead (an L-inf off-centre, Remacle 2013 §5). When the queue is empty, gaps are filled from the bg nodes with the
   same exclusion, each new point growing its own front.
3. sphere_delaunay with the coast recovered and the land removed, unreferenced nodes dropped (coast nodes kept first).
4. Optional valence flips (m11 ops, kind "valence"), then recombine(price) (greedy), then meshom.smooth "size"
   sweeps followed by a short "angle" run, coast fixed.

Variant m02_linf_frontal_tri (the triangle path, Georgiadis 2017 §2 with an asterisk field): the same packing with the
6-direction field (n = 6) and a hexagonal exclusion zone, Delaunay, the same smoothing, no recombination.

Option frame (B109): the field is not solved but set from an ORCA grid's i/j directions (meshom.orca.frame): the
fill between two lines of one ORCA map (m16_orca's refined part) keeps ORCA's rows and columns.

Option seeds (used by m15_tri): K > 1 or "auto" grows K fronts at once from a Fibonacci set of seeds in
the water (after the coast nodes in the FIFO), which removes most of the one-seed seam strain (scale 1, after the
smoothing: band 0.285 -> 0.155 with 484 seeds). The default 1 is the old code path.
"""
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from meshom.delaunay import adjacency, background_mesh, sphere_delaunay
from meshom.domain import Domain, loop_edges, resample_loops
from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh
from meshom.methods import OVERRIDES, register
from meshom.size import SizeField

DEFAULTS = dict(
    bg_factor=1.0,          # background mesh spacing (units of h) for field and point location (0.5: 3-4x slower,
                            # no better at scale 1)
    n=4,                    # field directions: 4 (cross, quads) or 6 (asterisk, triangles; the _tri variant)
    field="cones",          # cross_field method: "cones" | "smoothest"
    seed_loops=None,        # B112: None = every loop's nodes start fronts; [k, ...] = only these loops' (the others
                            # are fixed points that the fronts stop at): one front, no seam where two fronts meet
    frame=None,             # B109: an ORCA preset's name: the field is that grid's i/j directions (meshom.orca.frame),
                            # not solved (n = 4 only; `field` and `smooth_steps` not used)
    smooth_steps=0,         # MBO steps of the cross field
    alpha=0.75,             # L-inf exclusion half-size, units of the mean h of the two points (Gmsh 0.7)
    midpoint=True,          # turn each step by the field rotation seen at its half (follows the integral line)
    seam=1.414,             # > 0: a step blocked at a gap D >= seam * h ahead gets a point at D / 2 (off-centre)
    alpha_seam=0.6,         # exclusion of those seam points
    gap_fill=True,          # fill the gaps left by colliding fronts from the bg nodes
    seeds=1,                # fronts: 1 (closed sphere: one seed far from the cones; with a coast: the coast nodes
                            # only), K > 1 (K Fibonacci seeds in the water, grown at once with the coast), "auto"
    flips=None,             # None | "valence" | "delaunay": m11 flips on the triangle mesh before recombining
    price=1.0 / 3.0,        # recombination price of a leftover triangle (1 ~ all quads; 1/3 keeps pairs > ~60 deg)
    recombine="greedy",
    size_iters=30,          # meshom.smooth kind "size" sweeps after recombination (coast fixed)
    angle_iters=3,          # then a short kind "angle" run
    verbose=False,
)


def _opts(opts: dict | None) -> dict:
    o = dict(DEFAULTS)
    o.update(opts or {})
    return o


def _log(o: dict, msg: str) -> None:
    if o["verbose"]:
        print(f"[m02] {msg}", file=sys.stderr, flush=True)


def _vertex_triangle(tris: np.ndarray, n: int) -> np.ndarray:
    vt = np.full(n, -1, dtype=np.int64)
    vt[tris.ravel()] = np.repeat(np.arange(len(tris), dtype=np.int64), 3)
    return vt


def _locate(Pb, Tb, nbr, vt, tree, x: np.ndarray) -> np.ndarray:
    """bg triangle of each point x (walk from a triangle of the nearest bg node; clamped to the last triangle when the
    point lies just outside the bg mesh, as coast nodes between bg coast chords do)."""
    from meshom.methods.m02.packing import walk

    _, near = tree.query(x)
    out = np.empty(len(x), dtype=np.int64)
    for i in range(len(x)):
        t = walk(Pb, Tb, nbr, vt[near[i]], x[i])
        out[i] = t if t >= 0 else (-2 - t if t <= -2 else vt[near[i]])
    return out


PACK_DOUBLINGS = 3          # pack's point buffer starts at 2 n_est and doubles at most this often (16 n_est), then
                            # the packing fails (decisions M37): a NaN field never stops filling it

SEED_CAP_RAD = 0.4          # seeds: a front's cap has at most this radius (rad) ...
SEED_MIN_NODES = 500        # ... and at least this many nodes (K = 30-1000 work at scale 1)
SEED_NODES = 1000           # ... aiming at this many (m15 scan, scale 1, after m02's smoothing: band 26 / 100 / 485
                            # seeds 0.185 / 0.163 / 0.155, disc 26 / 100 / 248 0.158 / 0.138 / 0.140)


def n_seeds(seeds, water_sr: float, n_est: float) -> int:
    """Number of fronts: an int as given; "auto" = one per SEED_NODES nodes, at least the caps of radius SEED_CAP_RAD
    that cover the water, at most one per SEED_MIN_NODES nodes, at least 1."""
    if seeds != "auto":
        return int(seeds)
    k_cap = int(np.ceil(water_sr / (2.0 * np.pi * (1.0 - np.cos(SEED_CAP_RAD)))))
    return max(1, min(max(k_cap, int(n_est // SEED_NODES)), int(n_est // SEED_MIN_NODES)))


def seed_points(k: int, water_sr: float, water_nodes: np.ndarray, coast: np.ndarray | None) -> np.ndarray:
    """K seeds: a Fibonacci set of K points, each snapped to the nearest water bg node (duplicates dropped, Fibonacci
    order kept); with a coast, points on land and points within half a cap radius of a coast node are dropped (the
    coast nodes are fronts already)."""
    from meshom.delaunay import _fibonacci

    fib = _fibonacci(k)
    _, idx = cKDTree(water_nodes).query(fib)
    _, first = np.unique(idx, return_index=True)
    idx = idx[np.sort(first)]
    s = water_nodes[idx]
    if coast is not None:
        r_cap = np.arccos(max(-1.0, 1.0 - water_sr / (2.0 * np.pi * k)))
        d, _ = cKDTree(coast).query(s)
        s = s[d > 2.0 * np.sin(0.25 * r_cap)]              # chord of half the cap radius
    return s


def points(domain: Domain, h: SizeField, opts: dict | None = None) -> tuple[np.ndarray, int, dict]:
    """Packed points (unit vectors): the resampled coast nodes first (nb of them), then the interior points."""
    from meshom.field import cross_field, singularities
    from meshom.methods.m02.packing import pack

    o = _opts(opts)
    info = {}
    t0 = time.perf_counter()
    bg = background_mesh(domain, h, factor=o["bg_factor"])
    t1 = time.perf_counter()
    bnodes = np.flatnonzero(bg.node_flag == 1) if domain.loops else None
    nd = int(o["n"])
    if o.get("frame"):                                          # B109: ORCA's own directions
        from meshom.field import CrossField
        from meshom.orca.frame import orca_frame

        if nd != 4:
            raise ValueError("m02: `frame` needs the cross field (n = 4)")
        cf = CrossField(bg, orca_frame(str(o["frame"]))(bg.points), 4, np.log(h(bg.points)))
    else:
        cf = cross_field(bg, h, n=nd, boundary_nodes=bnodes, smooth_steps=o["smooth_steps"], method=o["field"])
    t2 = time.perf_counter()
    info.update(bg_nodes=bg.n_nodes, bg_s=t1 - t0, field_s=t2 - t1)

    Pb = np.ascontiguousarray(bg.points)
    Tb = np.ascontiguousarray(bg.tris, dtype=np.int64)
    nbr = adjacency(Tb)
    vt = _vertex_triangle(Tb, len(Pb))
    hb = h(Pb) / R_EARTH_M                                     # angles on the unit sphere
    lhb = np.log(hb)
    ub = np.ascontiguousarray(cf.u)
    cen = Pb[Tb].sum(axis=1)
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    rad = np.max(np.linalg.norm(Pb[Tb] - cen[:, None, :], axis=2), axis=1)
    e = np.vstack([Tb[:, [0, 1]], Tb[:, [1, 2]], Tb[:, [2, 0]]])
    g = float(np.max(np.abs(hb[e[:, 0]] - hb[e[:, 1]]) / np.linalg.norm(Pb[e[:, 0]] - Pb[e[:, 1]], axis=1)))
    alpha = float(o["alpha"])
    c = np.sqrt(2.0) if nd == 4 else 2.0 / np.sqrt(3.0)          # corner of the box / hexagon, units of its half-width
    rfac = c * alpha / max(1.0 - 0.5 * c * alpha * g, 0.25)
    tree = cKDTree(Pb)

    area = 0.5 * np.linalg.norm(np.cross(Pb[Tb[:, 1]] - Pb[Tb[:, 0]], Pb[Tb[:, 2]] - Pb[Tb[:, 0]]), axis=1)
    n_est = float(np.sum(area / np.exp(lhb[Tb]).mean(axis=1) ** 2)) * (1.0 if nd == 4 else 2.0 / np.sqrt(3.0))
    loops = resample_loops(domain, h)
    nb = sum(len(lp) for lp in loops)
    k_seeds = n_seeds(o["seeds"], float(area.sum()), n_est)
    q0 = 0                                                      # B112: points before q0 exclude but do not grow
    if k_seeds > 1:                                             # K fronts at once
        seeds = seed_points(k_seeds, float(area.sum()), Pb[bg.node_flag == 0], np.vstack(loops) if nb else None)
        info["seeds"] = int(len(seeds))
        if nb:
            seeds = np.vstack(loops + [seeds])
    elif nb and o.get("seed_loops") is not None:               # B112: only these loops grow fronts
        grow = set(int(k) for k in o["seed_loops"])
        order = [k for k in range(len(loops)) if k not in grow] + [k for k in range(len(loops)) if k in grow]
        offs = np.r_[0, np.cumsum([len(lp) for lp in loops])]
        perm = np.concatenate([np.arange(offs[k], offs[k + 1]) for k in order])
        seeds = np.vstack(loops)[perm]
        q0 = int(sum(len(loops[k]) for k in order if k not in grow))
    elif nb:
        seeds = np.vstack(loops)
    else:                                                       # closed sphere: the bg node farthest from the cones
        ids, _ = singularities(cf)
        if len(ids):
            sc = Pb[Tb[ids]].sum(axis=1)
            dist, _ = cKDTree(sc / np.linalg.norm(sc, axis=1, keepdims=True)).query(Pb)
            seeds = Pb[[int(np.argmax(dist))]]
        else:
            seeds = Pb[:1]
    stri = _locate(Pb, Tb, nbr, vt, tree, seeds)
    sh = h(seeds) / R_EARTH_M

    fill = np.flatnonzero(bg.node_flag == 0) if o["gap_fill"] else np.zeros(0, dtype=np.int64)
    fill_tri = vt[fill]
    cap = int(2.0 * n_est) + len(seeds) + 1000
    for _ in range(PACK_DOUBLINGS + 1):
        P = np.zeros((cap, 3))
        ph = np.zeros(cap)
        ptri = np.zeros(cap, dtype=np.int64)
        P[: len(seeds)] = seeds
        ph[: len(seeds)] = sh
        ptri[: len(seeds)] = stri
        n, full, n_gap = pack(Pb, Tb, nbr, ub, lhb, cen, rad, P, ph, ptri, len(seeds), fill, fill_tri, alpha,
                              rfac, bool(o["midpoint"]), float(o["seam"]), float(o["alpha_seam"]), nd, q0)
        if not full:
            break
        cap *= 2
    else:                                                       # a field with NaN packs NaN points without end
        raise RuntimeError(f"m02 pack: more than {cap // 2} points for an estimate of {n_est:.0f} "
                           f"(field finite: {bool(np.isfinite(ub).all())})")
    if q0:                                                      # B112: the loop nodes back in loop order
        P[:nb] = P[:nb][np.argsort(perm)]
        info["seed_loops"] = sorted(grow)
    t3 = time.perf_counter()
    info.update(pack_s=t3 - t2, n_points=int(n), n_gap=int(n_gap), n_est=n_est, g=g)
    return P[:n].copy(), nb, info


def triangle_mesh(domain: Domain, h: SizeField, opts: dict | None = None) -> tuple[Mesh, dict]:
    o = _opts(opts)
    pts, nb, info = points(domain, h, o)
    t0 = time.perf_counter()
    tris = sphere_delaunay(pts, loop_edges(resample_loops(domain, h)) if nb else None)
    used = np.zeros(len(pts), dtype=bool)
    used[tris.ravel()] = True
    used[:nb] = True
    new = np.cumsum(used) - 1
    flag = np.zeros(int(used.sum()), dtype=np.int8)
    flag[:nb] = 1
    mesh = Mesh(pts[used], new[tris], np.zeros((0, 4), dtype=np.int64), flag)
    info.update(delaunay_s=time.perf_counter() - t0, n_dropped=int((~used).sum()))
    if o["flips"]:
        from meshom.methods.m11.ops import flip_edges

        mesh, nf = flip_edges(mesh, kind=o["flips"])
        info["n_flips"] = nf
    _log(o, " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in info.items()))
    return mesh, info


def _smooth(mesh: Mesh, h: SizeField, o: dict) -> Mesh:
    from meshom.smooth import smooth

    fixed = mesh.node_flag == 1
    if o["size_iters"]:
        mesh = smooth(mesh, h, fixed, iters=o["size_iters"], kind="size")
    if o["angle_iters"]:
        mesh = smooth(mesh, h, fixed, iters=o["angle_iters"], kind="angle")
    return mesh


@register("m02_linf_frontal_tri", overrides={"n": 6})          # and another entry path: no recombination
def generate_triangles(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """Triangle-only: packing along the asterisk field (n = 6), Delaunay, smoothing; no recombination."""
    o = _opts(dict(OVERRIDES["m02_linf_frontal_tri"], **(opts or {})))
    return _smooth(triangle_mesh(domain, h, o)[0], h, o)


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    from meshom.recombine import recombine

    o = _opts(opts)
    mesh, _ = triangle_mesh(domain, h, o)
    t0 = time.perf_counter()
    mesh = recombine(mesh, o["price"], o["recombine"])
    t1 = time.perf_counter()
    mesh = _smooth(mesh, h, o)
    _log(o, f"recombine_s={t1 - t0:.3g} smooth_s={time.perf_counter() - t1:.3g} tris={mesh.n_tris} "
            f"quads={mesh.n_quads}")
    return mesh
