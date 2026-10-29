"""m09: global seamless parametrisation + integer rounding + quad extraction (MIQ / QuadCover / integer-grid maps).

Bommes, Zimmer, Kobbelt 2009 (MIQ); Kälberer, Nieser, Polthier 2007 (QuadCover); Bommes et al. 2013 (integer-grid
maps); extraction in the spirit of Ebke et al. 2013 (QEx).

1. background(): coast nodes at h, interior at bg_factor * h; meshom.field.cross_field ("cones": its singularities follow rho_c; coast
   nodes fixed to the coast tangent).
2. meshom/methods/m09/param.py: per-face cross field, combing on a dual BFS tree, cut graph through the cones and the
   coast loops, seamless map w = u + i v with dw ~ (R/h) e^{-i Theta} dx in least squares, transitions rho in
   {1, i, -1, -i} plus integer translations across the seams; the coast is an integer iso-line with one unit per
   resampled coast edge (the coast nodes land on integer points); MIQ greedy rounding of cones, coast pieces and free
   chain translations.
3. meshom/methods/m09/extract.py: unit cells of the map -> quads; failed cells -> constrained Delaunay triangles,
   optionally recombined (price).
4. Optional meshom.smooth sweeps (coast fixed).

The background mesh keeps the coast nodes resampled at h (each coast edge is one map unit, so they land on integer
points and stay exact) and has interior points at bg_factor * h.
"""
import sys
import time

import numpy as np

from meshom.domain import Domain, loop_edges, resample_loops
from meshom.mesh import Mesh
from meshom.methods import register
from meshom.size import SizeField

DEFAULTS = dict(
    bg_factor=1.0,          # interior spacing of the background mesh (units of h); coast nodes stay at h
    field="cones",          # cross_field method: "cones" | "smoothest"
    smooth_steps=0,         # MBO steps of the cross field
    rounding="greedy",      # "greedy" (MIQ, one Gaussian integer at a time) | "direct" (all at once)
    batch_frac=0.0,         # > 0: greedy rounds this fraction of the free integers per re-solve
    stiffen=8,              # local stiffening iterations on the continuous map (MIQ §5), while faces are flipped
    stiffen_post=8,         # stiffening iterations after rounding (integers kept), while the flip count falls
    stiffen_c=1.0,          # weight factor per iteration: 1 + c * distortion
    price=1.0 / 3.0,        # recombination price of the fill triangles (0: keep them as triangles)
    size_iters=0,           # meshom.smooth "size" sweeps (coast fixed)
    angle_iters=0,          # then "angle" sweeps
    verbose=False,
)


def _log(o: dict, msg: str) -> None:
    if o["verbose"]:
        print(f"[m09] {msg}", file=sys.stderr, flush=True)


def background(domain: Domain, h: SizeField, factor: float) -> Mesh:
    """Triangle mesh of the water: the coast nodes resampled at h (they become integer points of the map, one unit
    per coast edge), interior Poisson points at factor * h (a finer mesh spreads the rounding distortion)."""
    from meshom.delaunay import poisson_points, sphere_delaunay
    from meshom.size import scaled

    loops = resample_loops(domain, h)
    pts = poisson_points(domain, scaled(h, factor), loops)
    tris = sphere_delaunay(pts, loop_edges(loops) if loops else None)
    used = np.zeros(len(pts), dtype=bool)
    used[tris.ravel()] = True
    nb = sum(len(lp) for lp in loops)
    used[:nb] = True
    new = np.cumsum(used) - 1
    flag = np.zeros(int(used.sum()), dtype=np.int8)
    flag[:nb] = 1
    return Mesh(pts[used], new[tris], np.zeros((0, 4), dtype=np.int64), flag)


def run(domain: Domain, h: SizeField, opts: dict | None = None) -> tuple[Mesh, dict]:
    from meshom.field import cross_field, singularities
    from meshom.methods.m09.extract import extract
    from meshom.methods.m09.param import parametrize
    from meshom.recombine import recombine

    o = dict(DEFAULTS)
    o.update(opts or {})
    info = {}
    t0 = time.perf_counter()
    bg = background(domain, h, o["bg_factor"])
    t1 = time.perf_counter()
    bnodes = np.flatnonzero(bg.node_flag == 1) if domain.loops else None
    cf = cross_field(bg, h, n=4, boundary_nodes=bnodes, smooth_steps=o["smooth_steps"], method=o["field"])
    t2 = time.perf_counter()
    info.update(bg_nodes=bg.n_nodes, bg_s=t1 - t0, field_s=t2 - t1, field_cones=len(singularities(cf)[0]))
    prm = parametrize(bg.points, bg.tris, cf.u, h(bg.points), o["rounding"], o["batch_frac"], o["stiffen"],
                      o["stiffen_post"], o["stiffen_c"])
    t3 = time.perf_counter()
    info.update(param_s=t3 - t2, **{k: v for k, v in prm.info.items() if not k.startswith("_")})   # "_": debug arrays
    loops = resample_loops(domain, h)
    coast = np.vstack(loops) if loops else np.zeros((0, 3))
    pts, tris, quads, xinfo = extract(bg.points, bg.tris, prm, coast, loop_edges(loops))
    t4 = time.perf_counter()
    info.update(extract_s=t4 - t3, **xinfo)
    nb = len(coast)
    flag = np.zeros(len(pts), dtype=np.int8)
    flag[:nb] = 1
    mesh = Mesh(pts, tris, quads, flag)
    if o["price"] > 0 and len(tris):
        mesh = recombine(mesh, o["price"])
    t5 = time.perf_counter()
    if o["size_iters"] or o["angle_iters"]:
        from meshom.smooth import smooth

        fixed = mesh.node_flag == 1
        if o["size_iters"]:
            mesh = smooth(mesh, h, fixed, iters=o["size_iters"], kind="size")
        if o["angle_iters"]:
            mesh = smooth(mesh, h, fixed, iters=o["angle_iters"], kind="angle")
    t6 = time.perf_counter()
    info.update(recombine_s=t5 - t4, smooth_s=t6 - t5, nodes=mesh.n_nodes, tris=mesh.n_tris, quads=mesh.n_quads)
    _log(o, " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in info.items()))
    return mesh, info


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    return run(domain, h, opts)[0]
