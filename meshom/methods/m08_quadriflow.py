"""m08: QuadriFlow (Huang, Zhou, Nießner, Shewchuk, Guibas 2018, "QuadriFlow: a scalable and robust method for
quadrangulation") on the sphere with a per-vertex scale h: Instant Meshes (m07) + a global minimum-cost flow on the
integer jumps that removes the position singularities.

background_mesh -> hierarchy -> orientation field ("cones" = meshom.field.cross_field; "im" = the paper's
extrinsic 4-RoSy, as m07) -> orientation singularities from the frame turns of the background triangles (kept) ->
4-PoSy position field with rho = h / R (m07) -> integer jumps per background edge -> min-cost flow (m08/flow.py) that
makes every triangle's jump sum zero at the least change of the jumps (or keeps a singularity whose removal costs more
than `keep`) -> positions re-solved with the jumps fixed (m08/positions.py, paper §3.5) -> extraction with the new
jumps (m07/extract.py) -> size smoothing with the coast fixed. Notes and numbers: docs/methods/m08_quadriflow.md.
"""
import time

import numpy as np

from meshom.delaunay import background_mesh
from meshom.domain import Domain
from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh
from meshom.methods import OVERRIDES, register
from meshom.size import SizeField

DEFAULTS = dict(
    factor=0.5,              # background mesh spacing / h
    orientation="cones",     # "cones" | "smoothest" (meshom.field.cross_field) | "im" (Instant Meshes GS, h-blind)
    iters=6,
    fine_iters=6,
    flow=True,               # False: plain m07 (for comparisons on the same input)
    cap=1,                   # |jump component| <= cap after the flow (QuadriFlow starts at 2)
    keep=np.inf,             # price of keeping a position singularity (inf: remove all the network can)
    coast=True,              # the coast absorbs Burgers vectors along it
    cost="l2",               # flow cost: "l2" (d - u)^2 + eps |d - d*| (u = continuous offset), "l1" |d - d*|
    eps=0.25,
    resolve_iters=100,       # Gauss-Seidel sweeps of the position re-solve (0: keep the IM positions)
    rings="auto",            # re-solve within rings of a changed jump; None: everywhere (paper); auto: None for
                             # keep = inf, else 1 (a global solve moves every lattice point; measured)
    drift=1.5,               # re-solved positions are kept within drift x rho of their vertex
    max_settle=100_000,      # nodes settled per shortest-path search before giving up (the unit is kept)
    price=1.0 / 3.0,
    snap=0.5,
    merge=0.35,
    spurious=0.1,
    post_smooth="size",
    post_iters=10,
    seed=0,
)


def quadriflow(domain: Domain, h: SizeField, opts: dict | None = None) -> tuple[Mesh, dict]:
    """The mesh and a dict of counts and stage times."""
    from meshom.methods.m07.extract import extract
    from meshom.methods.m07.fields import (edge_frames, edge_jumps, orientation_field, position_field,
                                           restrict_constraints, restrict_orientation)
    from meshom.methods.m07.hierarchy import build_hierarchy
    from meshom.methods.m07_instant_meshes import _boundary_tangents
    from meshom.methods.m08.flow import continuous_offsets, own_frame, remove_position_singularities
    from meshom.methods.m08.positions import misfit, resolve_positions

    o = {**DEFAULTS, **(opts or {})}
    tm = {}
    t0 = time.perf_counter()
    bg = background_mesh(domain, h, factor=o["factor"])
    tm["background"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    levels = build_hierarchy(bg.points, bg.tris)
    tm["hierarchy"] = time.perf_counter() - t0

    n = bg.n_nodes
    CQ, CO = np.zeros((n, 3)), np.zeros((n, 3))
    CQw, COw = np.zeros(n), np.zeros(n)
    bnodes = np.zeros(0, dtype=np.int64)
    if domain.loops:
        bnodes, tang = _boundary_tangents(bg)
        CQ[bnodes], CQw[bnodes] = tang, 1.0
        CO[bnodes], COw[bnodes] = bg.points[bnodes], 1.0
    cons = restrict_constraints(levels, CQ, CQw, CO, COw)

    t0 = time.perf_counter()
    if o["orientation"] == "im":
        Q = orientation_field(levels, cons, o["iters"], o["fine_iters"], o["seed"])
    else:
        from meshom.field import cross_field

        cf = cross_field(bg, h, boundary_nodes=bnodes if len(bnodes) else None,
                         method="cones" if o["orientation"] == "cones" else "smoothest")
        Q = restrict_orientation(levels, cf.u)
        del cf
    tm["orientation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rho = [h(lev.V) / R_EARTH_M for lev in levels]
    O = position_field(levels, Q, rho, cons, o["iters"], o["fine_iters"], False, o["seed"])
    lev0 = levels[0]
    Q0, rho0 = Q[0], rho[0]
    del Q, rho, cons, levels
    diff, err = edge_jumps(lev0, Q0, O, rho0, False)
    tm["position"] = time.perf_counter() - t0

    st = {}
    if o["flow"]:
        t0 = time.perf_counter()
        D, r = own_frame(diff, edge_frames(lev0, Q0))
        U = continuous_offsets(lev0.V, lev0.edges, Q0, O, rho0, D)
        Dn, st = remove_position_singularities(bg.tris, lev0.edges, n, D, r, lev0.V, Q0, cap=o["cap"],
                                               keep=o["keep"], coast=o["coast"], max_settle=o["max_settle"],
                                               cost=o["cost"], eps=o["eps"], U=U, seed=o["seed"])
        tm["flow"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        if o["resolve_iters"]:
            chg = (Dn != D).any(axis=1)
            rings = (None if o["keep"] == np.inf else 1) if o["rings"] == "auto" else o["rings"]
            O = resolve_positions(lev0, Q0, O, rho0, Dn, r, o["resolve_iters"], None if rings is None else chg,
                                  rings or 0)
            err = misfit(lev0, Q0, O, rho0, Dn)
            # where the jumps cannot follow the geometry (strain of removed singularities) the least-squares
            # positions drift; keep each representative within `drift` x rho of its vertex for the extraction
            dv = O - lev0.V
            dn = np.linalg.norm(dv, axis=1)
            st["drift_p99"] = float(np.percentile(dn / rho0, 99))
            O = lev0.V + dv * np.minimum(1.0, o["drift"] * rho0 / np.maximum(dn, 1e-300))[:, None]
        diff = Dn
        tm["resolve"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mesh, st2 = extract(domain, h, lev0.V, lev0.edges, O, rho0, diff, err, o["price"], o["snap"], o["merge"],
                        o["spurious"])
    st.update(st2)
    tm["extract"] = time.perf_counter() - t0
    if o["post_smooth"]:
        from meshom.smooth import smooth

        t0 = time.perf_counter()
        mesh = smooth(mesh, h, mesh.node_flag == 1, iters=o["post_iters"], kind=o["post_smooth"])
        tm["smooth"] = time.perf_counter() - t0
    st["time"] = {k: round(v, 2) for k, v in tm.items()}
    return mesh, st


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """Orientation from the shared cross field (cones at rho_c), all position singularities removed."""
    return quadriflow(domain, h, opts)[0]


@register("m08_quadriflow_im", overrides={"orientation": "im"})
def generate_im(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """The Instant Meshes orientation field (h-blind, 8 cones on the sphere), all position singularities removed."""
    return quadriflow(domain, h, {**OVERRIDES["m08_quadriflow_im"], **opts})[0]


@register("m08_quadriflow_keep", overrides={"orientation": "im", "keep": 12.0})
def generate_keep(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """The Instant Meshes orientation field; a position singularity is removed only when an opposite one (or the coast)
    is within a flow cost of 12 (about 6 h): the meshom variant that keeps the singularities the grading needs."""
    return quadriflow(domain, h, {**OVERRIDES["m08_quadriflow_keep"], **opts})[0]
