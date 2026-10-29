"""Quantised seam joins on a synthetic corridor: two block lattices below and above a gap of w = 4 cells,
L = 100 cells long, lattice rotation theta and a
count mismatch of k cells per 100 as a size ramp across the gap, joined by the ladder strip and by the lattice
continuation under the corridor rule of the Burgers vector (stitch_mix.burgers_vector / quantise).

Geometry in a stereographic plane (units of h0 = 10 km at 30 W 20 N) mapped to the sphere conformally;
h(p) = h0 hp(y) cos^2(c/2) makes one plane unit exactly one h on the sphere. Measured over the fill
and block rows 0-2, 3 cells off the caps; dislocations = gross / 2 (gross = sum |4 - valence| + triangles) per 10
cells, the minimum = |b|_1 of the two lattices over the same stretch, |ln s/h| p99 over the edges, quad p99.9.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from meshom.geo import R_EARTH_M, arc_length, stereographic_inverse
from meshom.mesh import Mesh, build_topology, signed_area
from meshom.methods.m14 import continuation as C
from meshom.methods.m14 import stitch_mix as S
from meshom.methods.m14.blockmap import clean_cells, outline_loop
from meshom.methods.m14.stitch import element_rows
from meshom.quality.kernel import element_geometry

LON0, LAT0 = -30.0, 20.0
H0 = 10_000.0


def _frame():
    lo, la = np.radians(LON0), np.radians(LAT0)
    c0 = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])
    north = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    return c0, east, north


C0, EAST, NORTH = _frame()


def to_sphere(xy: np.ndarray) -> np.ndarray:
    return stereographic_inverse(xy[:, 0] * H0, xy[:, 1] * H0, LON0, LAT0)


def to_plane(p: np.ndarray) -> np.ndarray:
    d = 1.0 + p @ C0
    return np.column_stack([2.0 * R_EARTH_M * (p @ EAST) / d, 2.0 * R_EARTH_M * (p @ NORTH) / d]) / H0


class StereoH:
    """One plane unit = h; ln h ramps by kappa across the gap |y| < w/2, or a given plane function hp(x, y)."""

    def __init__(self, kappa: float, w: float, hp=None):
        self.kappa, self.w, self._hp = kappa, w, hp

    def hp(self, xy):
        if self._hp is not None:
            return self._hp(xy)
        return np.exp(self.kappa * np.clip(xy[:, 1] / self.w, -0.5, 0.5))

    def __call__(self, xyz):
        xyz = np.atleast_2d(xyz)
        cc = xyz @ C0
        near = cc > np.cos(0.3)
        out = np.full(len(xyz), 50.0 * H0)
        out[near] = H0 * self.hp(to_plane(xyz[near])) * 0.5 * (1.0 + cc[near])
        return out


class Lat:
    """Affine lattice X(i, j) = O + i a e_u + j a e_v in the plane."""

    def __init__(self, O, ang_deg, a):
        self.O = np.asarray(O, dtype=float)
        t = np.radians(ang_deg)
        self.M = a * np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        self.Mi = np.linalg.inv(self.M)

    def X(self, ij):
        return self.O + np.asarray(ij, dtype=float) @ self.M.T

    def U(self, x):
        return (np.asarray(x) - self.O) @ self.Mi.T


def block(X, U, rect):
    """Cells of the lattice map X (index -> plane; U its inverse) with centre in rect, cleaned: (plane points, quads,
    outline loop (fill on the left), lattice indices)."""
    x0, x1, y0, y1 = rect
    cor = U(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [0.5 * (x0 + x1), y0], [0.5 * (x0 + x1), y1]]))
    i0, j0 = np.floor(cor.min(axis=0)).astype(int) - 2
    i1, j1 = np.ceil(cor.max(axis=0)).astype(int) + 2
    ni, nj = i1 - i0, j1 - j0
    ii, jj = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    cen = X(np.column_stack([ii.ravel() + i0 + 0.5, jj.ravel() + j0 + 0.5]))
    inside = ((cen[:, 0] >= x0) & (cen[:, 0] <= x1) & (cen[:, 1] >= y0) & (cen[:, 1] <= y1)).reshape(ni, nj)
    cells = clean_cells(inside, inside)
    nid = np.arange((ni + 1) * (nj + 1)).reshape(ni + 1, nj + 1)
    ci, cj = np.nonzero(cells)
    q = np.column_stack([nid[ci, cj], nid[ci + 1, cj], nid[ci + 1, cj + 1], nid[ci, cj + 1]])
    used = np.unique(q)
    new = np.full(nid.size, -1, dtype=np.int64)
    new[used] = np.arange(len(used))
    ni_, nj_ = np.unravel_index(used, nid.shape)
    ij = np.column_stack([ni_ + i0, nj_ + j0])
    loops = outline_loop(cells, nid)
    assert len(loops) == 1
    return X(ij), new[q], new[loops[0]], ij


def _run(loop, xy, start_xy, end_xy):
    a = int(np.argmin(np.linalg.norm(xy[loop] - start_xy, axis=1)))
    b = int(np.argmin(np.linalg.norm(xy[loop] - end_xy, axis=1)))
    return loop[(a + np.arange((b - a) % len(loop) + 1)) % len(loop)]


def make_case(theta, k, w=4.0, psi=0.0, L=100.0, D=10.0, offset=(0.37, 0.21)):
    """The corridor: A at psi - theta/2 (spacing e^(-kappa/2)) below the gap, B at psi + theta/2
    (e^(kappa/2)) above it, kappa = ln(1 + k/100)."""
    kappa = np.log(1.0 + k / 100.0)
    latA = Lat((0.0, -w / 2), psi - theta / 2, np.exp(-kappa / 2))
    latB = Lat((offset[0], w / 2 + offset[1]), psi + theta / 2, np.exp(kappa / 2))
    c = dict(theta=theta, k=k, w=w, psi=psi, L=L, kappa=kappa, h=StereoH(kappa, w),
             U_A=latA.U, U_B=latB.U)
    c["blocks"] = [block(latA.X, latA.U, (0.0, L, -w / 2 - D, -w / 2)), block(latB.X, latB.U, (0.0, L, w / 2, w / 2 + D))]
    c["ends"] = [((0.0, -w / 2), (L, -w / 2)), ((L, w / 2), (0.0, w / 2))]
    return c


class Ramp:
    """The conformal lattice of a size ramp: u(z) = R(rot) (e^(i a z) - 1) / (i a) + off in the plane z = x + iy;
    its cell size is e^(a y) (the band case's ramp: |grad ln h| = a per cell, a = g = 0.1) and its rows turn by a per
    cell along x. along=True turns the ramp along the corridor (z -> i z: cells e^(a x))."""

    def __init__(self, a, rot_deg=0.0, off=(0.0, 0.0), along=False):
        self.a, self.r = a, np.exp(1j * np.radians(rot_deg))
        self.off = complex(*off)
        self.q = 1j if along else 1.0

    def U(self, x):
        z = (x[..., 0] + 1j * x[..., 1]) * self.q
        u = self.r * (np.exp(1j * self.a * z) - 1.0) / (1j * self.a) + self.off
        return np.stack([u.real, u.imag], axis=-1)

    def X(self, ij):
        u = (np.asarray(ij, dtype=float)[..., 0] + 1j * np.asarray(ij, dtype=float)[..., 1] - self.off) / self.r
        z = np.log(1.0 + 1j * self.a * u) / (1j * self.a) / self.q
        return np.stack([z.real, z.imag], axis=-1)


def make_ramp_case(a, theta=0.0, w=4.0, L=60.0, D=10.0, along=False):
    """Two blocks of the ramp lattice (B turned by theta and shifted by a fraction of a cell) across a gap of w cells;
    h follows the ramp exactly, so the blocks are exact and the gap holds only the extrapolation's error."""
    A, B = Ramp(a, -theta / 2, along=along), Ramp(a, theta / 2, (0.37, 0.21), along=along)
    hp = (lambda xy: np.exp(a * xy[:, 0])) if along else (lambda xy: np.exp(a * xy[:, 1]))
    c = dict(theta=theta, k=0.0, w=w, psi=0.0, L=L, kappa=0.0, h=StereoH(0.0, w, hp), U_A=A.U, U_B=B.U, ramp=(A, B))
    c["blocks"] = [block(A.X, A.U, (0.0, L, -w / 2 - D, -w / 2)), block(B.X, B.U, (0.0, L, w / 2, w / 2 + D))]
    c["ends"] = [((0.0, -w / 2), (L, -w / 2)), ((L, w / 2), (0.0, w / 2))]
    return c


def corridor(c, join="quantised"):
    """Blocks on the sphere, the trimmed runs, b, the quantisation, the strip plan, straight caps and the fill loop."""
    (xa, qa, la, ija), (xb, qb, lb, ijb) = c["blocks"]
    na = len(xa)
    xy = np.vstack([xa, xb])
    P = to_sphere(xy)
    c["nblock"], c["quads_b"] = len(P), np.vstack([qa, qb + na])
    c["gA"] = SimpleNamespace(points=P[:na], ij=ija)
    c["gB"] = SimpleNamespace(points=P[na:], ij=ijb)
    A = _run(la, xa, *c["ends"][0]).astype(np.int64)
    Bs = (_run(lb, xb, *c["ends"][1]) + na).astype(np.int64)
    ca, cb = S._edge_classes(P[A]), S._edge_classes(P[Bs])
    a0, a1 = int(np.argmax(ca != 1)), len(ca) - int(np.argmax(ca[::-1] != -1))
    b0, b1 = int(np.argmax(cb != 1)), len(cb) - int(np.argmax(cb[::-1] != -1))
    A, Bs, ca, cb = A[a0:a1 + 1], Bs[b0:b1 + 1], ca[a0:a1], cb[b0:b1]
    c["b"] = S.burgers_vector(ca, cb[::-1])
    c["quant"] = S.quantise(c["b"], round(c["w"])) if join == "quantised" else "even"
    plan = S._strip_plan(ca, cb[::-1], P[A], P[Bs][::-1], c["h"], 1.0,
                         {"shear": "ends", "even": "spread", "none": "spread"}[c["quant"]] if join == "quantised"
                         else "even")
    c["plan"] = None if isinstance(plan, str) else plan
    c["A"], c["Bs"] = A, Bs
    pts, nid, caps = [P], len(P), []
    for (s, e), m in (((A[0], Bs[-1]), "m0"), ((A[-1], Bs[0]), "m1")):
        p0, p1 = xy[s], xy[e]
        n = c["plan"][m] if c["plan"] else max(1, int(round(np.linalg.norm(p1 - p0))))
        mid = (1 - np.arange(1, n)[:, None] / n) * p0 + np.arange(1, n)[:, None] / n * p1
        pts.append(to_sphere(mid) if len(mid) else np.zeros((0, 3)))
        caps.append(np.r_[s, nid + np.arange(len(mid)), e].astype(np.int64))
        nid += len(mid)
    c["P"] = np.vstack(pts)
    c["C0"], c["C1"] = caps
    c["loop"] = np.r_[A, c["C1"][1:-1], Bs, c["C0"][1:-1][::-1]].astype(np.int64)
    c["a_loop"] = np.arange(len(A))
    c["b_loop"] = len(A) + len(c["C1"]) - 2 + np.arange(len(Bs))
    return c


def _fill_smooth(fm, h, size_iters=30, angle_iters=3):
    from meshom.smooth import smooth

    fixed = fm.node_flag == 1
    fm = smooth(fm, h, fixed, iters=size_iters, kind="size")
    return smooth(fm, h, fixed, iters=angle_iters, kind="angle")


def fill_ladder(c):
    if c["plan"] is None:
        return None
    P, loop = c["P"], c["loop"]
    new, q, tr = S.strip_patch(P, c["A"], c["Bs"][::-1], c["C0"], c["C1"], c["plan"], len(P))
    allp = np.vstack([P, new]) if len(new) else P
    if not S._strip_valid(allp, q, tr):
        return None
    nl = len(loop)
    idx = np.full(len(allp), -1, dtype=np.int64)
    idx[loop] = np.arange(nl)
    idx[len(P):] = nl + np.arange(len(new))
    fpts = np.vstack([P[loop], new]) if len(new) else P[loop]
    flag = np.zeros(len(fpts), np.int8)
    flag[:nl] = 1
    fm = _fill_smooth(Mesh(fpts, idx[tr].reshape(-1, 3), idx[q].reshape(-1, 4), flag), c["h"])
    return fm.points, fm.tris, fm.quads, nl


def fill_cont(c, **o):
    from meshom.methods import m14_blocks as M

    o = {"end_margin": float(M.DEFAULTS["end_rows"]), **o}             # B58's configured margin (as _cont_opts)
    na = len(c["gA"].points)
    fm, info = C.fill(c["P"][c["loop"]], c["a_loop"], c["b_loop"], c["gA"], c["A"], c["gB"], c["Bs"] - na, c["h"], o)
    c["cont_info"] = info
    return fm.points, fm.tris, fm.quads, len(c["loop"])


def merge(c, fpts, ftris, fquads, nl):
    """Blocks + fill (the fill's first nl nodes are c['loop'] in order) after stitch.py's seam band smoothing (3 block
    + 4 fill rows, 20 "size" + 3 "angle" sweeps)."""
    from meshom.smooth import smooth

    P = c["P"]
    gmap = np.r_[c["loop"], len(P) + np.arange(len(fpts) - nl)].astype(np.int64)
    pts = np.vstack([P, fpts[nl:]])
    quads = np.vstack([c["quads_b"], gmap[fquads].reshape(-1, 4)])
    tris = gmap[ftris].reshape(-1, 3) if len(ftris) else np.zeros((0, 3), np.int64)
    mesh = Mesh(pts, tris.astype(np.int64), quads.astype(np.int64), np.zeros(len(pts), np.int8))
    mesh = Mesh(pts, mesh.tris, mesh.quads, build_topology(mesh).boundary_node.astype(np.int8))
    is_block = np.zeros(mesh.n_elems, bool)
    is_block[mesh.n_tris:mesh.n_tris + len(c["quads_b"])] = True
    A, Bs = c["A"], c["Bs"]
    ol = np.vstack([np.column_stack([A[:-1], A[1:]]), np.column_stack([Bs[:-1], Bs[1:]])])
    row, _, _ = element_rows(mesh, is_block, ol)
    band = (row >= 0) & np.where(is_block, row < 3, row < 4)
    in_band, out_band = np.zeros(mesh.n_nodes, bool), np.zeros(mesh.n_nodes, bool)
    for cells, off in ((mesh.tris, 0), (mesh.quads, mesh.n_tris)):
        bb = band[off:off + len(cells)]
        in_band[cells[bb].ravel()] = True
        out_band[cells[~bb].ravel()] = True
    fixed = ~(in_band & ~out_band & (mesh.node_flag == 0))
    mesh = smooth(mesh, c["h"], fixed, iters=20, kind="size")
    mesh = smooth(mesh, c["h"], fixed, iters=3, kind="angle")
    return mesh, is_block, ol


def _hops(mesh, src):
    """Graph distance (edges) of every node to the node set src."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    E = build_topology(mesh).edges
    g = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(mesh.n_nodes, mesh.n_nodes))
    return dijkstra(g, directed=False, indices=np.unique(src), min_only=True)


def measure(c, mesh, is_block, ol, xmargin=3.0):
    topo = build_topology(mesh)
    n, T, L = mesh.n_nodes, mesh.n_tris, c["L"]
    deg = np.bincount(topo.edges.ravel(), minlength=n)
    bnd = topo.boundary_node
    row, _, _ = element_rows(mesh, is_block, ol)
    g = element_geometry(mesh)
    cxy = to_plane(g.centroid)
    meas = (cxy[:, 0] >= xmargin) & (cxy[:, 0] <= L - xmargin) & ((~is_block) | ((row >= 0) & (row <= 2)))
    is_tri = np.arange(mesh.n_elems) < T
    qn, tn = np.bincount(mesh.quads.ravel(), minlength=n), np.bincount(mesh.tris.ravel(), minlength=n)
    nmeas = np.zeros(n, bool)
    for cells, off in ((mesh.tris, 0), (mesh.quads, T)):
        nmeas[cells[meas[off:off + len(cells)]].ravel()] = True
    nxy = to_plane(mesh.points)
    nodes = nmeas & ~bnd & (nxy[:, 0] >= xmargin) & (nxy[:, 0] <= L - xmargin)
    defect = nodes & (np.abs(360.0 - 90.0 * qn - 60.0 * tn) >= 60.0)
    cv = 4 - deg
    tri_m = meas & is_tri
    gross = int(np.abs(cv[nodes]).sum() + tri_m.sum())
    per10 = 10.0 / (L - 2 * xmargin)
    E, ee = topo.edges, topo.edge_elems
    inner = ee[:, 1] >= 0
    e1 = np.where(inner, ee[:, 1], ee[:, 0])
    em = (meas[ee[:, 0]] | meas[e1]) & inner
    p, q = mesh.points[E[:, 0]], mesh.points[E[:, 1]]
    mid = (p + q) / np.linalg.norm(p + q, axis=1, keepdims=True)
    lnsh = np.abs(np.log(R_EARTH_M * arc_length(p, q) / c["h"](mid)))
    qmax = np.degrees(g.quad_angles).max(axis=1)
    # where the cores sit: hop distance (edges) of the defect nodes to the nearer outline run; a triangle with a node
    # on an outline or a defect node on it is a dislocation on the outline row
    hop = np.minimum(_hops(mesh, c["A"]), _hops(mesh, c["Bs"]))
    tri_hop = hop[mesh.tris[tri_m[:T]]].min(axis=1) if tri_m.any() else np.zeros(0)
    x0, x1 = np.array([xmargin, 0.0]), np.array([L - xmargin, 0.0])
    b = (c["U_A"](x1) - c["U_A"](x0)) - (c["U_B"](x1) - c["U_B"](x0))
    return dict(disl_per10=0.5 * gross * per10, min_per10=float(np.abs(b).sum()) * per10, gross=gross,
                defects=int(defect.sum()), tris=int(tri_m.sum()),
                lnsh_p99=float(np.percentile(lnsh[em], 99)), lnsh_max=float(lnsh[em].max()),
                quad_p999=float(np.percentile(qmax[meas[T:]], 99.9)), quad_max=float(qmax[meas[T:]].max()),
                outline_row=int((tri_hop < 1).sum() + (hop[defect] < 1).sum()),
                near_outline=int((hop[defect] < 2).sum()),
                inverted=int((signed_area(mesh) <= 0).sum()))


def run(theta, k, join, w=4.0, psi=0.0, rule="quantised", **o):
    c = corridor(make_case(theta, k, w, psi), rule)
    f = fill_ladder(c) if join == "ladder" else fill_cont(c, **o)
    if f is None:
        return dict(theta=theta, k=k, join=join, ok=False, quant=c["quant"], b=c["b"])
    mesh, is_block, ol = merge(c, *f)
    r = measure(c, mesh, is_block, ol)
    r.update(theta=theta, k=k, w=w, psi=psi, join=join, ok=True, quant=c["quant"], b=c["b"],
             cont=c.get("cont_info"))
    return r


# ------------------------------------------------------------------------------------------------------------ tests

@pytest.mark.parametrize("theta,k", [(0, 0), (0, 3), (0, 10), (15, 0), (30, 10)])
def test_burgers_vector_and_rule(theta, k):
    """b of the trimmed runs equals the closure of their lattice indices (A's index change minus B's, both from cap C0
    to cap C1) and Frank-Bilby, b_par = L ln(1 + k/100) cos(theta/2), b_perp = 2 L sin(theta/2) (L = 100 cells), to
    +-2 (the trimmed run ends: +-1 per run); the maps' b agrees to +-1; the rule: b = 0 none, b_perp = 0 and
    |b_par| <= w shear, else even."""
    c = corridor(make_case(theta, k))
    ija, ijb, na = c["gA"].ij, c["gB"].ij, len(c["gA"].points)
    closure = (ija[c["A"][-1]] - ija[c["A"][0]]) - (ijb[c["Bs"][0] - na] - ijb[c["Bs"][-1] - na])
    assert tuple(int(x) for x in closure) == c["b"]
    L, kap, th = 100.0, np.log(1 + k / 100.0), np.radians(theta)
    assert abs(c["b"][0] - L * kap * np.cos(th / 2)) <= 2.0 and abs(c["b"][1] - 2 * L * np.sin(th / 2)) <= 2.0
    assert c["quant"] == ("none" if theta == 0 and k == 0 else "shear" if theta == 0 and k == 3 else "even")
    cor = C.Corridor(c["gA"], c["A"], c["gB"], c["Bs"] - len(c["gA"].points), {})
    assert np.allclose(cor.burgers_split(), c["b"], atol=1.0)
    assert cor.mode() == ("none" if c["quant"] in ("none", "shear") else "even")


def test_schedule_even_spacing_and_hysteresis():
    """|b|_1 = 36 events for delta = (0.1 s, 0.26 s) over 100 cells, evenly spaced (Bresenham: kinds interleaved,
    never two column ends in a row), |delta - T| < 1 throughout; noise of +-0.1 around 0.45 makes no event (eta) while
    plain rounding makes pairs; "none" keeps one T."""
    s = np.linspace(0.0, 100.0, 401)
    delta = np.column_stack([0.1 * s, 0.26 * s])
    pos, T0, steps = C.schedule(s, delta, "even")
    assert len(pos) == 36
    assert np.abs(np.diff(pos) - 100.0 / 36).max() < 0.2
    kinds = steps[:, 0] != 0
    assert kinds.sum() == 10 and not (kinds[1:] & kinds[:-1]).any()
    Ts = C.segment_T(pos, T0, steps)
    T = Ts[np.searchsorted(pos, s, side="right")]
    assert np.abs(delta - T).max() < 1.0
    noisy = np.column_stack([0.45 + 0.1 * np.sin(3.0 * s), np.zeros_like(s)])
    assert len(C.schedule(s, noisy, "even")[0]) == 0 and len(C.schedule(s, noisy, "round")[0]) > 10
    pos, T0, steps = C.schedule(s, delta, "none")
    assert len(pos) == 0 and np.array_equal(T0, np.round(delta[200]))


def test_pairing_modes():
    """The ladder's column terminations (B58): "even" at the nearest free pair to the even position (9.5: pairs 9-10
    and 10-11 tie, the first taken), "spread" off the pair that borders a step of the side (the run edge between along
    edges 8 and 9), "ends" (the shear) alternately at the two corridor ends."""
    e = np.r_[0:9, 10:21]
    assert S._pairing(20, 1, e, "even") == [9] and S._pairing(20, 1, e, "spread") == [10]
    assert S._pairing(20, 3, np.arange(20), "ends") == [0, 2, 18]


def test_classify_quantised():
    """join = "quantised": the ladder below obl_strip_deg (strip-eligible), the continuation above it (a corridor whose
    continuation failed: m02; below obl_strip_deg without a ladder: m02, or the continuation with cont_fallback),
    coasts and junctions as before; rot_quad_deg does not decide."""
    from meshom.methods import m14_blocks as M

    assert M.DEFAULTS["join"] == "quantised" and M.DEFAULTS["cont_fallback"] and M.DEFAULTS["lattice_caps"]
    o = dict(M.DEFAULTS, rot_quad_deg=0.0, cont_fallback=False)
    obl = dict(cls="corridor", coast=False, obl_a=30.0, obl_b=10.0, strip_ok=False, quant="even", b=(0, 5),
               cont_ok=True)
    regs = [dict(id=0, cls="corridor", coast=False, obl_a=5.0, obl_b=8.0, strip_ok=True, quant="even", b=(1, 9),
                 rot_deg=40.0, mismatch_per10=9.0, cont_ok=True),
            dict(id=1, **obl), dict(id=2, **obl, cont_failed="invalid (fold)"),
            dict(id=3, **dict(obl, obl_a=5.0, obl_b=8.0)),
            dict(id=4, cls="coast", coast=True), dict(id=5, cls="junction", coast=False)]
    S.classify(SimpleNamespace(regions=regs), o)
    assert [r["fill"] for r in regs] == ["strip", "cont", "m02", "m02", "m02", "m02"]
    assert regs[0]["join"] == "ladder" and regs[1]["join"] == "cont" and regs[2]["join"] == "m02"
    S.classify(SimpleNamespace(regions=regs), dict(o, cont_fallback=True))
    assert [r["fill"] for r in regs[:4]] == ["strip", "cont", "m02", "cont"]


@pytest.mark.parametrize("join", ["ladder", "cont"])
@pytest.mark.parametrize("theta,k", [(t, k) for t in (0, 15, 30, 45) for k in (0, 3, 10)])
def test_joins_on_synthetic_corridor(theta, k, join):
    """Synthetic corridor: dislocations <= 1.2 x the minimum for theta <= 30 deg with the ladder and the
    continuation, |ln s/h| p99 <= 0.18, quad p99.9 <= 135 deg, no dislocation on an outline row; at 45 deg the count
    only (measured: the continuation 1.28 x on average, 1.42 x at 30 deg)."""
    r = run(theta, k, join)
    assert r["ok"] and r["inverted"] == 0
    assert r["disl_per10"] <= 1.2 * r["min_per10"] + 1e-9
    if theta <= 30:
        assert r["lnsh_p99"] <= 0.18 and r["quad_p999"] <= 135.0 and r["outline_row"] == 0


@pytest.mark.parametrize("theta,psi,k", [(0, 30, 0), (15, 15, 3), (5, 30, 3), (15, 30, 3)])
def test_continuation_on_oblique_lattices(theta, psi, k):
    """Lattices oblique to the gap (mean angle psi): a side's obliquity is >= obl_strip_deg
    (20 deg), so the rule's join is the continuation; parallel oblique lattices stay defect-free, otherwise the
    synthetic gates hold and no core sits on an outline row (B58; the staircase corners reach into the gap)."""
    c = corridor(make_case(theta, k, 4.0, psi))
    assert max(S._obliquity(c["P"][c["A"]]), S._obliquity(c["P"][c["Bs"]])) >= 20.0
    mesh, is_block, ol = merge(c, *fill_cont(c))
    r = measure(c, mesh, is_block, ol)
    assert r["inverted"] == 0 and r["outline_row"] == 0
    assert r["disl_per10"] <= 1.2 * r["min_per10"] + 1e-9
    assert r["lnsh_p99"] <= 0.18 and r["quad_p999"] <= 135.0


def test_continuation_on_the_band_ramp():
    """Two blocks of the band ramp's conformal lattice (|grad ln h| = 0.1 per cell: the lattice fans by
    5.7 deg per cell) across a 4-cell gap; the outline faces' affine extrapolation is off by <= 0.3 cells (about
    a d^2 / 2 at d = 2) and the continuation makes no dislocation where none is needed."""
    c = corridor(make_ramp_case(0.1, 0.0, 4.0, L=15.0))
    cor = C.Corridor(c["gA"], c["A"], c["gB"], c["Bs"] - len(c["gA"].points), {})
    A, B = c["ramp"]
    mxy = to_plane(cor.chart.xyz(cor.mid))
    assert np.abs(cor.delta - (A.U(mxy) - B.U(mxy))).max() < 0.3
    mesh, is_block, ol = merge(c, *fill_cont(c))
    r = measure(c, mesh, is_block, ol)
    assert r["gross"] == 0 and r["inverted"] == 0 and r["lnsh_p99"] < 0.12


# ------------------------------------------------------------------------------------------------- schedules

def test_two_row_rule_schedule_and_pairing():
    """B58's 2-row rule (at caps that border a junction, the coast or an open region): the continuation's events
    within end_margin of a cap move to it (1 cell apart where they would stack), the others stay; the ladder's spread
    column terminations keep 2 along edges from either end while such pairs are free (the shear's "ends" stay at the
    ends); a row termination in the two end strips of a ladder costs END_PENALTY more in the row-count DP."""
    s = np.linspace(0.0, 30.0, 121)
    delta = np.column_stack([np.zeros_like(s), 0.5 * s])               # 15 events, the first at 1 cell
    pos0 = C.schedule(s, delta, "even", 0.15)[0]
    pos, T0, steps = C.schedule(s, delta, "even", 0.15, 2.0)
    assert pos0.min() < 2.0 and len(pos) == len(pos0) == 15
    assert pos.min() >= 2.0 - 1e-9 and pos.max() <= 28.0 + 1e-9 and (np.diff(pos) >= 1.0 - 1e-9).all()
    inside = (pos0 >= 3.0) & (pos0 <= 27.0)
    assert np.allclose(pos[inside], pos0[inside])                       # the events off the caps do not move
    assert np.array_equal(C.schedule(s, delta, "even", 0.15, (2.0, 0.0))[0][-1:], pos0[-1:])
    e = np.arange(20)
    pr = S._pairing(20, 5, e, "spread", 2)
    assert min(pr) >= 2 and max(pr) + 1 <= 17 and np.ptp(np.diff(pr)) <= 1
    assert min(S._pairing(20, 5, e, "spread", 0)) < 2
    assert S._pairing(20, 2, e, "ends", 2) == [0, 18]
    # a straight 30-cell corridor narrowing from 7 to 5 cells: the 2 row terminations stay off the end strips
    xa = stereographic_inverse(np.arange(31) * H0, np.zeros(31), LON0, LAT0)
    top = 7.0 - 2.0 * np.clip((np.arange(31)[::-1] - 26) / 3.0, 0.0, 1.0)   # narrows over the last cells at x = 0
    xb = stereographic_inverse(np.arange(31)[::-1] * H0, top * H0, LON0, LAT0)
    hh = StereoH(0.0, 4.0, lambda xy: np.ones(len(xy)))
    ca, cb = S._edge_classes(xa), S._edge_classes(xb)
    plans = {m: S._strip_plan(ca, cb[::-1], xa, xb[::-1], hh, 1.0, m, 2) for m in ("even", "spread")}
    assert all(not isinstance(p, str) for p in plans.values())
    assert plans["spread"]["end_row_terms"] <= plans["even"]["end_row_terms"]
    assert plans["spread"]["end_row_terms"] == 0


def test_strip_gate_rejects_a_ladder_quad_of_150_deg_as_built():
    """B46: a ladder patch as built with a quad corner >= STRIP_QUAD_MAX = 150 deg is no strip: a row of 3 sheared
    quads (corners 90 +- psi deg) passes the gate at 145 deg and fails it at 160 deg (the 170 deg fold guard would
    pass both)."""
    tris = np.zeros((0, 3), np.int64)
    q = np.array([[i, i + 1, i + 5, i + 4] for i in range(3)], dtype=np.int64)
    for deg, ok in ((145.0, True), (160.0, False)):
        t = np.tan(np.radians(deg - 90.0))
        P = to_sphere(np.array([[i, 0.0] for i in range(4)] + [[i + t, 1.0] for i in range(4)]))
        built = np.degrees(element_geometry(Mesh(P, tris, q, np.zeros(len(P), np.int8))).quad_angles).max()
        assert (signed_area(Mesh(P, tris, q, np.zeros(len(P), np.int8))) > 0).all() and abs(built - deg) < 0.5
        assert S._strip_valid(P, q, tris, S.STRIP_QUAD_MAX) is ok


def test_two_row_rule_in_the_configured_run(monkeypatch):
    """B58 in the configured run: m14_blocks' DEFAULTS end_rows = 2 reaches the quantised join (uniform, scale 4,
    fill_method "mix": every corridor cap there faces a junction): each ladder's end strips (rec _end_rows) and each
    continuation's end margins (_cont_opts) are 2 cells at both caps."""
    from meshom.cases import CASES
    from meshom.methods import m14_blocks as M
    from meshom.size import scaled

    parts, margins = [], []
    geo, cont = S.corridor_geometry, S._cont_opts

    def geo_spy(part, *a, **k):
        parts.append(part)
        return geo(part, *a, **k)

    def cont_spy(o, rec=None):
        r = cont(o, rec)
        margins.append(r["end_margin"])
        return r

    monkeypatch.setattr(S, "corridor_geometry", geo_spy)
    monkeypatch.setattr(S, "_cont_opts", cont_spy)
    dom, h0 = CASES["uniform"]
    M.run(dom, scaled(h0, 4.0), {"fill_method": "mix"})
    ladders = [r["_end_rows"] for r in parts[0].regions if "_end_rows" in r]
    assert len(ladders) >= 5 and set(ladders) == {(2, 2)}
    assert len(margins) >= 1 and set(margins) == {(2.0, 2.0)}


def test_break_clusters_resplits_pentagon_and_hexagon():
    """B62: a fan of 3 triangles (a pentagon) becomes one quad + one triangle, a strip of 4 (a hexagon) two quads, 3
    triangles round a valence-3 interior node the one triangle without it, a pair stays; regions follow the cluster's
    first triangle; nothing is inverted."""
    xy = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [1, 1], [0, 1],           # hexagon strip 0-1-2-3-4-5
                   [5, 0], [6, 0], [6.6, 0.8], [5.5, 1.4], [4.6, 0.7],      # pentagon 6..10
                   [8, 0], [9, 0], [8.5, 0.9], [9.5, 0.9],                  # a lone pair 11-12-13, 12-14-13
                   [11, 0], [13, 0], [12, 1.7], [12, 0.6],                  # triangle 15-16-17, interior node 18
                   [21, 1], [20, 0], [22, 0], [22, 2], [20, 2], [19, 1]],   # a fan of 3 + a quad round node 19
                  dtype=float)
    P = to_sphere(xy)
    tris = np.array([[0, 1, 4], [0, 4, 5], [1, 2, 3], [1, 3, 4],
                     [6, 7, 8], [6, 8, 9], [6, 9, 10],
                     [11, 12, 13], [12, 14, 13],
                     [15, 16, 18], [16, 17, 18], [17, 15, 18],
                     [19, 20, 21], [19, 21, 22], [19, 22, 23]], dtype=np.int64)
    rt = np.array([5, 5, 5, 5, 7, 7, 7, 9, 9, 3, 3, 3, 2, 2, 2])
    t2, q2, rt2, rq2, n, gone = S.break_clusters(P, tris, np.array([[19, 23, 24, 20]]), rt, np.array([2]),
                                                 free_from=18)
    assert n == 4 and len(q2) == 4 and len(t2) == 5 and sorted(gone) == [18, 19]
    assert sorted(rq2.tolist()) == [2, 5, 5, 7] and sorted(rt2.tolist()) == [2, 3, 7, 9, 9]
    assert [15, 16, 17] in t2.tolist() and not (t2 == 19).any() and not (q2 == 19).any()
    m = Mesh(P, t2, q2, np.zeros(len(P), np.int8))
    assert (signed_area(m) > 0).all()


def test_continuation_wired_on_a_real_corridor():
    """The continuation in stitch_mix (join = "quantised"): disc at scale 4 has a corridor between two real blocks with
    a lattice oblique to the gap (obliquity >= 20 deg); it is filled by the continuation with its lattice caps (the cap
    nodes carry their labels), it is quad-dominant, the mesh is valid, and its defects stay within 1.5 x the
    dislocations its schedule places plus the two caps."""
    from meshom.cases import CASES
    from meshom.compete import irregular_counts
    from meshom.methods import m14_blocks as M
    from meshom.size import scaled

    dom, h0 = CASES["disc"]
    mesh, info = M.run(dom, scaled(h0, 4.0), {"fill_method": "mix", "join": "quantised", "lattice_caps": True})
    cont = [r for r in info["regions"] if r["fill"] == "cont"]
    assert info["stitch"]["n_cont"] == len(cont) >= 1
    r = cont[0]
    assert max(r["obl_a"], r["obl_b"]) >= 20.0 and r["lat_caps"] == "ok" and r["cont"]["cap_labels"] > 0
    assert r["quads"] > 5 * r["tris"]
    assert r["cont"]["events"] >= np.abs(r["cont"]["b"]).sum() - 1 and r["irr_mix"] <= 1.5 * r["cont"]["events"] + 10
    assert (signed_area(mesh) > 0).all() and irregular_counts(mesh)["euler_ok"]
