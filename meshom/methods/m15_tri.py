"""m15_tri: the size-faithful pure-triangle pipeline (seeds + population control + fit).

1. Seeded packing: m02's frontal packing along the 6-fold field (`m02_linf_frontal.points`, n = 6) grown from
   K fronts at once: the resampled coast nodes plus K Fibonacci seeds in the water (`seeds="auto"`: one per 1000
   nodes, at least the caps of radius 0.4 rad that cover the water, at most one per 500 nodes; seeds within half a cap
   radius of the coast dropped). One FIFO, so the fronts advance in rounds (layer by layer, in seed order):
   deterministic.
2. Sphere Delaunay of the points, the coast loops recovered and the land removed (m02.triangle_mesh); the coast nodes
   come first (node_flag 1) and stay fixed from here on. The coast conforms to the resampled loops as in m02 (the
   cut variant exists at the run level: `[coast] treatment = "cut"` in `meshom run`).
3. Population control (meshom/popctl.py): rounds of deletions of valence <= 4 nodes and of one end of edges below
   ln(s/h) = -0.18, midpoint insertions above +0.22, over independent sets; re-triangulation with the coast
   recovered; 10 `size` sweeps.
4. Per-edge size fit (meshom/size_fit.py): L-BFGS on mean (ln s/h)^6 with the shape barrier, Lawson flips between
   rounds, coast fixed. (An L-BFGS CVT before the fit was not needed: the fit met its numbers.) With
   `size_tolerance` set (decisions G38) the last stage is meshom.size_fit.smooth_fit instead: the element size ratio
   between neighbours as small as possible within |ln s/h| <= size_tolerance.

Also registered here (the fit as a last step of the existing triangle methods): `m01_triangles_fit`
(m01_triangles, then the fit) and `m02_linf_frontal_tri_fit` (m02_linf_frontal_tri, then the fit).
`run(domain, h, opts)` returns (mesh, info) with the stage times and the per-stage records.
"""
import sys
import time

from meshom.domain import Domain, loop_edges, resample_loops
from meshom.mesh import Mesh
from meshom.methods import register
from meshom.size import SizeField

DEFAULTS = dict(
    seeds="auto",            # fronts of the packing: "auto" (m02_linf_frontal.n_seeds) or K
    packing={},              # further m02 options (m02_linf_frontal.DEFAULTS; n = 6 is set)
    popctl_rounds=8,         # population control: rounds (0 = off), thresholds on ln(s/h), size sweeps per round
    t_del=-0.18,
    t_ins=0.22,
    sweeps=10,
    vflips=False,            # valence flips after the population control (popctl.valence_flips)
    fit=True,                # the per-edge size fit as the last stage
    fit_flips=True,          # Lawson flips between the fit's rounds
    fit_p=6,
    fit_iters=150,           # L-BFGS iterations per round
    fit_rounds=3,
    fit_target=None,         # stop the fit when |ln s/h| p99 <= this (None: run to convergence / the caps)
    size_tolerance=None,     # G38: None = the size fit (s = h) is the last stage; t = meshom.size_fit.smooth_fit
                             # instead: sizes as smooth as possible within |ln s/h| <= t (coast fixed, flips as fit_flips)
    tol_fit={},              # options of that smooth_fit (M38)
    verbose=False,
)


def _log(o: dict, msg: str) -> None:
    if o["verbose"]:
        print(f"[m15] {msg}", file=sys.stderr, flush=True)


def coast_edges(domain: Domain, h: SizeField):
    """Constraint edges of the resampled coast loops (into the first mesh nodes), None on the closed sphere."""
    return loop_edges(resample_loops(domain, h)) if domain.loops else None


def run(domain: Domain, h: SizeField, opts: dict | None = None) -> tuple[Mesh, dict]:
    from meshom.methods import m02_linf_frontal as m02
    from meshom.popctl import popctl
    from meshom.size_fit import size_fit

    o = dict(DEFAULTS)
    o.update(opts or {})
    t = {}
    t0 = time.perf_counter()
    mo = m02._opts(dict({"n": 6, "seeds": o["seeds"]}, **o["packing"]))
    mesh, pinfo = m02.triangle_mesh(domain, h, mo)
    t1 = time.perf_counter()
    t["packing_delaunay_s"] = t1 - t0
    info = {"packing": {k: v for k, v in pinfo.items() if not isinstance(v, dict)}}
    if o["popctl_rounds"]:
        mesh, cinfo = popctl(mesh, h, None, coast_edges(domain, h), rounds=o["popctl_rounds"], t_del=o["t_del"],
                             t_ins=o["t_ins"], sweeps=o["sweeps"])
        info["popctl"] = cinfo
    if o["vflips"]:
        from meshom.popctl import valence_flips

        mesh, info["vflips"] = valence_flips(mesh, h, mesh.node_flag != 0)
    t2 = time.perf_counter()
    t["popctl_s"] = t2 - t1
    if o["size_tolerance"]:                             # G38: smoothness within the tolerance last (0 = off)
        from meshom.size_fit import smooth_fit

        mesh, info["tol_fit"] = smooth_fit(mesh, h, float(o["size_tolerance"]), None,
                                           **dict({"flips": o["fit_flips"]}, **o["tol_fit"]))
    elif o["fit"]:
        mesh, finfo = size_fit(mesh, h, None, p=o["fit_p"], max_iter=o["fit_iters"], rounds=o["fit_rounds"],
                               target_p99=o["fit_target"], flips=o["fit_flips"])
        info["fit"] = finfo
    t["fit_s"] = time.perf_counter() - t2
    info["timings"] = t
    _log(o, f"nodes {mesh.n_nodes} {t} popctl {info.get('popctl', {}).get('rounds')} "
            f"fit p99 {info.get('fit', {}).get('p99_after')}")
    return mesh, info


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    return run(domain, h, opts)[0]


def _fit_opts(opts: dict | None) -> tuple[dict, dict]:
    """Split method options into the base method's and the fit's (keys fit_p, fit_iters, fit_rounds, fit_target)."""
    base = dict(opts or {})
    fit = {k: base.pop(k) for k in ("fit_p", "fit_iters", "fit_rounds", "fit_target") if k in base}
    return base, {"p": fit.get("fit_p", DEFAULTS["fit_p"]), "max_iter": fit.get("fit_iters", DEFAULTS["fit_iters"]),
                  "rounds": fit.get("fit_rounds", DEFAULTS["fit_rounds"]),
                  "target_p99": fit.get("fit_target", DEFAULTS["fit_target"])}


@register("m01_triangles_fit", overrides={})          # composed: m01_triangles, then the fit
def generate_m01_fit(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """m01_triangles (Poisson discs + 60 springs), then the per-edge size fit with the coast fixed."""
    from meshom.methods import m01_isotropic as m01
    from meshom.size_fit import size_fit

    base, fk = _fit_opts(opts)
    return size_fit(m01.triangle_mesh(domain, h, base), h, None, **fk)[0]


@register("m02_linf_frontal_tri_fit", overrides={})   # composed: m02_linf_frontal_tri, then the fit
def generate_m02_fit(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """m02_linf_frontal_tri (one-seed asterisk packing + its smoothing), then the per-edge size fit."""
    from meshom.methods import m02_linf_frontal as m02
    from meshom.size_fit import size_fit

    base, fk = _fit_opts(opts)
    return size_fit(m02.generate_triangles(domain, h, base), h, None, **fk)[0]
