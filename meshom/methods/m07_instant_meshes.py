"""m07: Instant Meshes (Jakob, Tarini, Panozzo, Sorkine-Hornung 2015, "Instant field-aligned meshes") with a
per-vertex scale h, on the sphere.

background_mesh(domain, h, factor) -> multiresolution graph hierarchy (m07/hierarchy.py) -> orientation field
(m07/fields.py: "cones" = meshom.field.cross_field, the shared field parallel in g/h^2 with cones at rho_c; "im" =
the paper's extrinsic 4-RoSy Gauss-Seidel on the hierarchy, h-blind; "smoothest" = cross_field Knöppel) -> 4-PoSy
position field with per-vertex scale rho = h / R (edge scale = mean of the two ends in the matching step) ->
integer jumps per background edge -> extraction (m07/extract.py: collapse to clusters, coast snap, Delaunay faces with
lattice quads, recombination of the leftover triangles, pentagon re-split) -> optional size smoothing with
the coast fixed. Notes and numbers: docs/methods/m07_instant_meshes.md.

m08 (QuadriFlow) can reuse the steps: build_hierarchy, orientation (restrict_orientation / orientation_field),
position_field, edge_jumps + edge_frames (replace the integer jumps), extract.
"""
import time

import numpy as np

from meshom.delaunay import adjacency, background_mesh
from meshom.domain import Domain
from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh
from meshom.methods import OVERRIDES, register
from meshom.size import SizeField

DEFAULTS = dict(
    factor=0.5,              # background mesh spacing / h (the paper subdivides the input to <= rho / 2)
    orientation="cones",     # "cones" | "smoothest" (meshom.field.cross_field) | "im" (paper's GS, h-blind)
    iters=6,                 # Gauss-Seidel sweeps per hierarchy level (paper: 6)
    fine_iters=6,            # sweeps on the finest level
    scale="mean",            # per-edge scale in the position matching: "mean" (both ends) | "own" (each its own)
    conformal=0.0,           # lattice spacing h e^(conformal f), f = conformal factor of the field (0 = h; cones only)
    price=1.0 / 3.0,         # triangle price for the lattice-quad test and the recombination of leftovers
    snap=0.5,                # clusters within snap * h of the coast go onto the nearest coast node
    merge=0.35,              # clusters closer than merge * h are merged
    spurious=0.1,            # clusters with fewer members than spurious * mean are dropped (paper)
    post_smooth="size",      # meshom.smooth kind after extraction (None to skip), coast fixed
    post_iters=10,
    seed=0,
)


def _boundary_tangents(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Boundary nodes of a triangle mesh and the unit tangent of their outgoing boundary edge (water on the left)."""
    nbr = adjacency(mesh.tris)
    t, k = np.nonzero(nbr < 0)
    a, b = mesh.tris[t, k], mesh.tris[t, (k + 1) % 3]
    p = mesh.points
    d = p[b] - p[a]
    d -= np.einsum("ij,ij->i", d, p[a])[:, None] * p[a]
    return a, d / np.linalg.norm(d, axis=1, keepdims=True)


def instant_mesh(domain: Domain, h: SizeField, opts: dict | None = None) -> tuple[Mesh, dict]:
    """The mesh and a dict of counts and stage times."""
    from meshom.methods.m07.extract import extract
    from meshom.methods.m07.fields import (conformal_factor, edge_jumps, orientation_field, position_field,
                                           restrict_constraints, restrict_orientation, restrict_scale)
    from meshom.methods.m07.hierarchy import build_hierarchy

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
    tm["orientation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rho0 = h(bg.points) / R_EARTH_M
    if o["conformal"] and o["orientation"] != "im":
        f = conformal_factor(bg, h, cf.u)
        s = np.exp(o["conformal"] * f)
        s *= np.sqrt(np.sum(levels[0].A / (rho0 * s) ** 2) / np.sum(levels[0].A / rho0**2))   # keep the count
        rho0 = rho0 * s
    rho = restrict_scale(levels, rho0) if o["conformal"] else [h(lev.V) / R_EARTH_M for lev in levels]
    own = o["scale"] == "own"
    O = position_field(levels, Q, rho, cons, o["iters"], o["fine_iters"], own, o["seed"])
    diff, err = edge_jumps(levels[0], Q[0], O, rho[0], own)
    tm["position"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mesh, st = extract(domain, h, levels[0].V, levels[0].edges, O, rho[0], diff, err, o["price"], o["snap"],
                       o["merge"], o["spurious"])
    tm["extract"] = time.perf_counter() - t0
    if o["post_smooth"]:
        from meshom.smooth import smooth

        t0 = time.perf_counter()
        mesh = smooth(mesh, h, mesh.node_flag == 1, iters=o["post_iters"], kind=o["post_smooth"])
        tm["smooth"] = time.perf_counter() - t0
    st["levels"] = len(levels)
    st["time"] = {k: round(v, 2) for k, v in tm.items()}
    return mesh, st


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    return instant_mesh(domain, h, opts)[0]


@register("m07_instant_meshes_im", overrides={"orientation": "im"})
def generate_im(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """The paper's own orientation field (extrinsic 4-RoSy Gauss-Seidel on the hierarchy), h-blind."""
    return instant_mesh(domain, h, {**OVERRIDES["m07_instant_meshes_im"], **opts})[0]
