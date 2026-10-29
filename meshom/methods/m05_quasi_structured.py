"""m05: quasi-structured quad meshing (Reberol, Georgiadis, Remacle 2021; Gmsh algorithm 11), mixed output.

1. m02 (L-inf frontal packing along the cross field -> Delaunay -> recombination -> smoothing) gives a quad-dominant
   mesh whose irregular spots are mostly dislocation pairs (a leftover triangle next to a valence-5 node).
2. Cavity remeshing (meshom/methods/m05/remesh.py, paper §6.2): from every irregular vertex grow a convex disk of
   elements and replace it by a quad pattern (regular grid; triangle / pentagon around one singularity; 3-5
   transition patterns) when the irregularity falls and the local geometry (angles, edge size against h) does not
   get worse. No global midpoint subdivision: the cavity's own boundary loop is kept, so its triangles disappear in
   pairs; cavities that cannot be improved stay as they are (mixed output).
3. meshom.smooth "size" sweeps then a short "angle" run (coast fixed).

Variant m05_quasi_structured_loose: the same with looser geometric acceptance (more irregular spots removed, worse
size fidelity).
"""
import sys
import time

import numpy as np

from meshom.domain import Domain
from meshom.mesh import Mesh
from meshom.methods import OVERRIDES, register
from meshom.size import SizeField

DEFAULTS = dict(
    max_cavity=300,       # elements per cavity
    max_steps=256,        # growth steps (snapshots) per seed
    tries=3,              # remeshable snapshots tried per seed
    max_fails=1,          # a seed that failed this many times is retried only after a remesh next to it
    local_iters=20,       # "size" sweeps on cavity + rings after a replacement
    rings=2,              # element layers around the cavity smoothed with it (their outer nodes fixed)
    max_dln=0.1,          # skip cavities whose element count change shifts the mean size by more (|ln|, half the
                          # log of old / new element count, a triangle counted as half a quad)
    tol_angle=40.0,       # a replacement may not raise the worst corner deviation above max(before + eps, tol) (deg)
    tol_size=0.25,        # ... nor the worst edge |ln(s/h)| above max(before + eps, tol)
    eps_angle=2.0,        # slack on "before" (the ring around the cavity moves too)
    eps_size=0.02,
    rounds=3,             # repeats of the pass sequence while something changes
    steps=4,              # passes of the sequence used (1: regular only, 2: + singular, 4: + transitions)
    size_iters=30,        # final meshom.smooth "size" sweeps (coast fixed)
    angle_iters=3,        # then "angle" sweeps
    m02=None,             # options passed to m02
    verbose=False,
)


def _opts(opts: dict | None) -> dict:
    o = dict(DEFAULTS)
    o.update(opts or {})
    return o


LOOSE = dict(tol_angle=45.0, tol_size=0.4)   # paper-like: shape validity first, size may drift (cube-sphere-like)


@register("m05_quasi_structured_loose", overrides=LOOSE)
def generate_loose(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """Looser acceptance: worst corner deviation up to 45 deg, worst edge |ln(s/h)| up to 0.4."""
    return generate(domain, h, dict(OVERRIDES["m05_quasi_structured_loose"], **(opts or {})))


def coast_row(mesh: Mesh) -> np.ndarray:
    """Coast nodes and the nodes of every element touching the coast: m02 has already smoothed this row and the
    cavities never touch it, so the final smoothing keeps it as it is."""
    coast = mesh.node_flag == 1
    fixed = coast.copy()
    for el in (mesh.tris, mesh.quads):
        fixed[el[coast[el].any(axis=1)].ravel()] = True
    return fixed


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    from meshom.methods.m02_linf_frontal import generate as m02
    from meshom.methods.m05.remesh import improve
    from meshom.smooth import smooth

    o = _opts(opts)
    t0 = time.perf_counter()
    mesh = m02(domain, h, o["m02"] or {})
    t1 = time.perf_counter()
    mesh, st = improve(mesh, h, o)
    t2 = time.perf_counter()
    fixed = coast_row(mesh)
    if o["size_iters"]:
        mesh = smooth(mesh, h, fixed, iters=o["size_iters"], kind="size")
    if o["angle_iters"]:
        mesh = smooth(mesh, h, fixed, iters=o["angle_iters"], kind="angle")
    if o["verbose"]:
        st.pop("log", None)
        print(f"[m05] m02_s={t1 - t0:.3g} cavity_s={t2 - t1:.3g} smooth_s={time.perf_counter() - t2:.3g} {st}",
              file=sys.stderr, flush=True)
    return mesh
