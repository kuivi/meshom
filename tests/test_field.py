"""cross field with the connection of g^ = g/h^2 (meshom/field.py)."""
import numpy as np

from meshom.cases import CASES
from meshom.delaunay import background_mesh
from meshom.domain import resample_loops
from meshom.field import cross_field, singularities
from meshom.geo import R_EARTH_M, arc_length
from meshom.size import scaled, uniform


def _cube_check(mesh, field):
    ids, idx = singularities(field)
    c = mesh.points[mesh.tris[ids]].mean(axis=1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    d = np.degrees(np.arccos(np.clip(c @ c.T, -1.0, 1.0)))
    np.fill_diagonal(d, 180.0)
    return ids, idx, d.min(axis=1)


def test_uniform_sphere_eight_quarter_singularities():
    dom, h = CASES["uniform"]
    hs = scaled(h, 4)
    mesh = background_mesh(dom, hs, 0.5)
    ids, idx, nn = _cube_check(mesh, cross_field(mesh, hs))
    assert idx.sum() == 2.0                               # Poincare-Hopf: chi = 2
    assert len(ids) == 8 and np.all(idx == 0.25)
    assert np.all(np.abs(nn - 70.5) < 3.0)                # the cones form a cube (corner spacing 70.5 deg)
    ids, idx, _ = _cube_check(mesh, cross_field(mesh, hs, method="smoothest", smooth_steps=3))
    assert len(ids) == 8 and np.all(idx == 0.25)
    ids, idx, _ = _cube_check(mesh, cross_field(mesh, hs, n=6))
    assert np.isclose(idx.sum(), 2.0) and len(ids) == 12 and np.allclose(idx, 1.0 / 6.0)


def test_disc_singularities_follow_rho_c():
    """rho_c on `disc`: ~+20 on the fine rim (r = 500 km), ~-20 over the ramp (500-900 km), +8 from K elsewhere.
    Scale 3 at factor 0.5: ~12 cells across the ramp. Without the *d ln h term (uniform h) the field sees only K."""
    dom, h = CASES["disc"]
    hs = scaled(h, 3)
    mesh = background_mesh(dom, hs, 0.5)
    centre = np.asarray(hs.h.center)

    def counts(field):
        ids, idx = singularities(field)
        r = R_EARTH_M * arc_length(mesh.points[mesh.tris[ids]].mean(axis=1), centre[None, :])
        return (idx.sum(), np.sum((r > 400e3) & (r < 650e3) & (idx > 0)),
                np.sum((r > 500e3) & (r < 1100e3) & (idx < 0)), np.sum((r > 1100e3) & (idx > 0)))

    total, rim, ramp, far = counts(cross_field(mesh, hs))
    assert total == 2.0
    assert 14 <= rim <= 26 and 14 <= ramp <= 26 and 4 <= far <= 12, (rim, ramp, far)
    total, rim, ramp, far = counts(cross_field(mesh, uniform(1.0)))
    assert total == 2.0 and rim + ramp == 0 and far == 8, (rim, ramp, far)


def test_continent_boundary_aligned():
    dom, h = CASES["continent"]
    hs = scaled(h, 4)
    mesh = background_mesh(dom, hs, 0.5)
    lengths = [len(lp) for lp in resample_loops(dom, scaled(hs, 0.5))]   # boundary nodes first, loop by loop
    nb = sum(lengths)
    assert np.all(mesh.node_flag[:nb] == 1)
    bnd = np.arange(nb)
    field = cross_field(mesh, hs, boundary_nodes=bnd)

    p = mesh.points[:nb]
    nxt, prv = np.empty(nb, dtype=np.int64), np.empty(nb, dtype=np.int64)
    off = np.r_[0, np.cumsum(lengths)]
    for a, b in zip(off[:-1], off[1:]):
        i = np.arange(a, b)
        nxt[i], prv[i] = np.roll(i, -1), np.roll(i, 1)

    def tangent(src, dst):
        d = p[dst] - p[src]
        t = d - np.einsum("ij,ij->i", d, p[src])[:, None] * p[src]
        return t / np.linalg.norm(t, axis=1, keepdims=True)

    t_out, t_in = tangent(bnd, nxt), -tangent(bnd, prv)
    turn = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", t_in, t_out), -1.0, 1.0)))
    smooth = turn < 30.0                                  # corners (the channel mouths) are checked separately
    t = t_in + t_out
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    u = field.u[:nb]
    ang = np.degrees(np.arctan2(np.einsum("ij,ij->i", np.cross(t, u), p), np.einsum("ij,ij->i", t, u))) % 90.0
    err = np.minimum(ang, 90.0 - ang)
    assert smooth.sum() > 0.95 * nb
    assert err[smooth].max() < 1.0
    # corners: the cross takes one of the two coast directions up to half the deviation of the corner from 90 deg
    a_in = np.degrees(np.arctan2(np.einsum("ij,ij->i", np.cross(t_in, u), p), np.einsum("ij,ij->i", t_in, u))) % 90.0
    e_in = np.minimum(a_in, 90.0 - a_in)
    assert np.all(e_in[~smooth] <= 0.5 * np.abs(turn[~smooth] - 90.0) + 1.0)
    # Poincare-Hopf with boundary: chi = -1 (sphere minus 3 discs) and four 270-degree water corners: sum 0
    ids, idx = singularities(field)
    assert idx.sum() == 0.0
    # no cone in the coastal rows (COAST_CONE_DISTANCE = 3 h; node spacing on the coast 0.5 h)
    c = mesh.points[mesh.tris[ids]].mean(axis=1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    d = np.min(np.arccos(np.clip(c @ p.T, -1.0, 1.0)), axis=1) * R_EARTH_M / hs(c)
    assert d.min() > 2.5


def _lake_and_ocean(case: str, lake_r_h0: float | None = None, lake_first: bool = True):
    """A seam lake of meshom.cases_seams (or a round lake of radius lake_r_h0 H0) alone inside its land ring, plus the
    world ocean beyond the ring as a separate water component (no inlet); h as the seam cases. Returns (domain, h,
    plane centre)."""
    from meshom import cases_seams as cs
    from meshom.domain import Domain

    verts = {"seam_coast": [(-28, 0), (28, 0), (28, 56), (-28, 56)],
             "seam_strait": [(-30, -70), (30, -70), (30, -30), (6, -30), (6, 30), (30, 30), (30, 70), (-30, 70),
                             (-30, 30), (-6, 30), (-6, -30), (-30, -30)]}
    pl = cs.Plane(x_az=60.0)
    c, ex, ey = pl.frame()

    def circle(r_m, sign):                                                # sign -1: clockwise
        rho, k = r_m / R_EARTH_M, int(np.ceil(2.0 * np.pi * r_m / cs.LOOP_STEP_M))
        ph = sign * 2.0 * np.pi * np.arange(k)[:, None] / k
        return cs._unit(np.cos(rho) * c + np.sin(rho) * (np.cos(ph) * ex + np.sin(ph) * ey))

    if lake_r_h0 is None:
        h = getattr(cs, case)()[1]
        rng = np.random.default_rng(0)
        s = pl.to_sphere(np.asarray(verts[case] + verts[case][:1], dtype=float))
        lake = np.vstack([cs._seg(a, b, rng) for a, b in zip(s[:-1], s[1:])])
    else:
        r_env = (lake_r_h0 + cs.RING_H0) * cs.H0_M
        h = cs.PlateauH(tuple(ey), cs.H0_M, cs.H0_M, 3.0 * cs.H0_M, tuple(c), r_env)
        lake = circle(lake_r_h0 * cs.H0_M, 1.0)
    ocean = circle(h.r_env_m, -1.0)                                         # clockwise: the ocean outside on the left
    return Domain([lake, ocean] if lake_first else [ocean, lake], case), h, c


def test_two_water_components_each_anchored():
    """A lake and a separate world ocean each get their own tree root and anchor, and their own cones.
    Before: the component without the first fixed node kept psi = 0 (geographic north): seam_coast's lake (ocean loop
    first) 31.4 deg off its coast at every node; and the global quantisation put a -1/4 cone into seam_strait's lake
    (charge 0), so the trivial connection had no solution (CG residual 1.5e3) and the lake's field broke (45 deg)."""
    for case, lake_first in (("seam_coast", False), ("seam_strait", True)):
        dom, h, c = _lake_and_ocean(case, lake_first=lake_first)
        mesh = background_mesh(dom, h, 1.0)
        cf = cross_field(mesh, h, boundary_nodes=np.flatnonzero(mesh.node_flag == 1))
        p, u = mesh.points, cf.u
        r = R_EARTH_M * arc_length(p, c[None, :])
        hv = h(p)

        def err(t, sel):                                                  # 4-fold angle of u to the tangent t, deg
            a = np.degrees(np.arctan2(np.einsum("ij,ij->i", np.cross(t[sel], u[sel]), p[sel]),
                                      np.einsum("ij,ij->i", t[sel], u[sel]))) % 90.0
            return np.minimum(a, 90.0 - a)

        from meshom.cases_seams import Plane
        ex = Plane(x_az=60.0).frame()[1]
        lake = (mesh.node_flag == 0) & (r < h.r_env_m - 50e3)
        assert err(ex[None, :] - (p @ ex)[:, None] * p, lake).max() < 2.0  # the lake's lattice follows its coast
        ring = (mesh.node_flag == 0) & (r > h.r_env_m) & (r < h.r_env_m + 1.5 * hv)
        assert np.percentile(err(np.cross(c[None, :], p), ring), 99) < 15.0  # the ocean's coastal rows follow the ring
        ids, idx = singularities(cf)
        q = mesh.points[mesh.tris[ids]].mean(axis=1)
        in_lake = R_EARTH_M * arc_length(q / np.linalg.norm(q, axis=1, keepdims=True), c[None, :]) < h.r_env_m
        assert not in_lake.any()                                          # right-angle coast: no cone in the lake
        assert idx.sum() == 1.0                                           # the ocean: sphere minus a disc


def test_small_lake_quantised_on_its_own():
    """A round lake 2.5 H0 in radius has no 3 h shell: its coast charge (+1, a disc with a fixed coast) stays in it as 4
    cones of +1/4 instead of going to the ocean's shell (before: 0 cones placed in the lake, +8 in the ocean, the
    trivial connection unsolvable (CG residual 5.5e6) and 12 cones measured in the lake)."""
    dom, h, c = _lake_and_ocean("round", lake_r_h0=2.5)
    mesh = background_mesh(dom, h, 1.0)
    cf = cross_field(mesh, h, boundary_nodes=np.flatnonzero(mesh.node_flag == 1))
    ids, idx = singularities(cf)
    q = mesh.points[mesh.tris[ids]].mean(axis=1)
    in_lake = R_EARTH_M * arc_length(q / np.linalg.norm(q, axis=1, keepdims=True), c[None, :]) < h.r_env_m
    assert np.array_equal(idx[in_lake], np.full(4, 0.25))
    assert idx.sum() == 2.0
