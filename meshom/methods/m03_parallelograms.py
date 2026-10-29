"""m03: frontal packing of parallelograms (squares of side h in the cross frame) -> sphere Delaunay -> recombination.

Carrier Baudouin, Remacle, Marchandise, Henrotte, Geuzaine 2014 (AMSES 1:8), section "Surface meshing: the packing of
parallelograms algorithm"; Gmsh algorithm 9 (gmsh.info).

1. Field: meshom.field.cross_field on background_mesh(domain, h, field_factor) (default "cones"; coast nodes aligned
   with the coast). field_factor 1.0 by default: at 0.5 the background mesh and field take ~85 % of the run time (3x
   slower at scale 1) and the packed mesh is no better (disc |ln s/h| p99 0.26 vs 0.24). At any point the field is
   sampled from the 4 nearest background nodes: their u projected into the point's tangent plane, averaged as
   power-4 vectors with inverse-square-distance weights.
2. Packing. Seeds: the resampled coast nodes (frame = coast tangent, in loop order) or, on the closed sphere, one
   background node far from the cones. A FIFO queue: the point at its head proposes four candidates, one step of
   length L along each of -t1, -t2, +t1, +t2 of its own frame (geodesic on the sphere; L = h at the point as in Gmsh,
   or by default the mean of h at the point and at the first guess). A candidate is kept (and queued) iff it is in the
   water and lies outside the exclusion square of every point kept so far: the square of side 2 k h_q (Gmsh: k = 0.7;
   default 0.6: the widest gap left between meeting fronts is (1 + k) h, so a smaller k shortens the long edges at
   the seams, k < 0.6 adds triangles) centred on point q and aligned with q's frame, tested in q's tangent plane
   (points whose parent is the candidate's parent are exempt, as in Gmsh). The packing works layer by layer: a
   layer's candidates are generated and their h and frames evaluated in batch (NumPy), then accepted one by one in
   FIFO order (Numba, multilevel 3D spatial hash; a point of exclusion radius rho sits in the level whose cell size
   >= rho, so a query checks 27 cells per level) -- which is exactly the sequential FIFO, since a candidate depends
   only on its parent.
3. Triangulation: sphere_delaunay (+ recover_edges, remove_land with a coast) of the packed points: right triangles
   where the packing is a lattice. Optional gap fill (opts["fill"]): circumcentres of triangles with a circumradius
   above fill * h, inserted greedily with the same exclusion test.
4. recombine(price) and post_iters sweeps of meshom.smooth.smooth(kind="size") with the coast fixed.

Difference from m02 (L-infinity frontal-Delaunay, Remacle 2013): m02 keeps a triangulation during the front and
inserts one point per active edge, at the L-infinity optimal position for that edge; here no mesh exists during the
packing: each point proposes its own four lattice neighbours along its cross frame, and an exclusion square (not a
circumradius test on triangles) decides; the triangulation is built once at the end.
"""
import numpy as np
from numba import njit

from meshom.domain import Domain, in_water, loop_edges, resample_loops
from meshom.geo import R_EARTH_M, tangent_basis
from meshom.mesh import Mesh
from meshom.methods import register
from meshom.size import SizeField

DEFAULTS = {
    "k": 0.6,                  # exclusion square half-side / h (Gmsh FACTOR = 0.7)
    "step": "mean",            # "point": L = h(p) (Gmsh); "mean": L = (h(p) + h(first guess)) / 2
    "field": "cones",          # cross_field method
    "field_factor": 1.0,       # background mesh spacing / h (0.5: 3x slower, no better)
    "fill": 0.0,               # > 0: insert circumcentres of triangles with circumradius > fill * h (0 = off)
    "price": 1.0 / 3.0,        # recombination: price of a leftover triangle
    "recombine": "greedy",
    "post_smooth": "size",     # meshom.smooth kind after recombination, coast fixed (None = off)
    "post_iters": 40,
}
_LEVELS = 24
_K_NEAR = 4


def _opts(opts: dict | None) -> dict:
    o = dict(DEFAULTS)
    o.update(opts or {})
    return o


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ------------------------------------------------------------------------------------------------ field sampling

class _FieldSampler:
    """Cross frame (t1, t2) at arbitrary points from a CrossField on its triangle mesh (power-4 IDW of 4 nodes)."""

    def __init__(self, field):
        from scipy.spatial import cKDTree

        self.pts = field.mesh.points
        self.u = field.u
        self.tree = cKDTree(self.pts)

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d, j = self.tree.query(x, k=_K_NEAR)
        e1, e2 = tangent_basis(x)
        uj = self.u[j]                                                     # (M,k,3)
        a1 = np.einsum("mki,mi->mk", uj, e1)
        a2 = np.einsum("mki,mi->mk", uj, e2)
        w = 1.0 / (d + 1e-9) ** 2
        z = np.sum(w * np.exp(4j * np.arctan2(a2, a1)), axis=1)
        phi = np.angle(z) / 4.0
        t1 = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
        return t1, np.cross(x, t1)


# ------------------------------------------------------------------------------------------------ packing kernel

@njit(cache=True, inline="always")
def _bucket(lv, ix, iy, iz, mask):
    return ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (lv * 2654435761)) & mask


@njit(cache=True)
def _level(rho, s0):
    lv = 0
    s = s0
    while s < rho and lv < _LEVELS - 1:
        s *= 2.0
        lv += 1
    return lv


@njit(cache=True)
def _insert(i, P, H, LV, NXT, head, used, s0, k, mask):
    rho = 1.42 * k * H[i] * 1.01
    lv = _level(rho, s0)
    s = s0 * 2.0**lv
    b = _bucket(lv, np.int64(np.floor(P[i, 0] / s)), np.int64(np.floor(P[i, 1] / s)),
                np.int64(np.floor(P[i, 2] / s)), mask)
    LV[i] = lv
    NXT[i] = head[b]
    head[b] = i
    used[lv] = True


@njit(cache=True)
def _excluded(c, par, P, T1, T2, H, PAR, NXT, head, used, s0, k, mask):
    """True if point c lies in the exclusion square of a stored point (other than its parent's children)."""
    for lv in range(_LEVELS):
        if not used[lv]:
            continue
        s = s0 * 2.0**lv
        cx = np.int64(np.floor(c[0] / s))
        cy = np.int64(np.floor(c[1] / s))
        cz = np.int64(np.floor(c[2] / s))
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    q = head[_bucket(lv, cx + dx, cy + dy, cz + dz, mask)]
                    while q >= 0:
                        if PAR[q] != par or par < 0:
                            dot = c[0] * P[q, 0] + c[1] * P[q, 1] + c[2] * P[q, 2]
                            if dot > 0.0:
                                lim = k * H[q]
                                x = c[0] * T1[q, 0] + c[1] * T1[q, 1] + c[2] * T1[q, 2]
                                if abs(x) < lim:
                                    y = c[0] * T2[q, 0] + c[1] * T2[q, 1] + c[2] * T2[q, 2]
                                    if abs(y) < lim:
                                        return True
                        q = NXT[q]
    return False


@njit(cache=True)
def _accept(n, cand, ct1, ct2, ch, cpar, P, T1, T2, H, PAR, LV, NXT, head, used, s0, k, mask):
    """Sequential FIFO acceptance of one layer's candidates; kept ones are appended at n, n+1, ... Returns new n."""
    for i in range(len(cand)):
        if _excluded(cand[i], cpar[i], P, T1, T2, H, PAR, NXT, head, used, s0, k, mask):
            continue
        P[n] = cand[i]
        T1[n] = ct1[i]
        T2[n] = ct2[i]
        H[n] = ch[i]
        PAR[n] = cpar[i]
        _insert(n, P, H, LV, NXT, head, used, s0, k, mask)
        n += 1
    return n


@njit(cache=True)
def _insert_all(n0, n1, P, H, LV, NXT, head, used, s0, k, mask):
    for i in range(n0, n1):
        _insert(i, P, H, LV, NXT, head, used, s0, k, mask)


class _Packing:
    """Growable point store with the multilevel spatial hash (all sizes in radians of the unit sphere)."""

    def __init__(self, cap: int, h_min: float, k: float):
        self.k = k
        self.s0 = 1.42 * k * h_min * 1.01
        m = 1 << int(np.ceil(np.log2(max(4 * cap, 1024))))
        self.mask = m - 1
        self.head = np.full(m, -1, dtype=np.int64)
        self.used = np.zeros(_LEVELS, dtype=np.bool_)
        self.n = 0
        self._alloc(cap)

    def _alloc(self, cap: int) -> None:
        old = getattr(self, "P", None)
        arrs = {"P": (cap, 3), "T1": (cap, 3), "T2": (cap, 3), "H": (cap,)}
        for name, shp in arrs.items():
            a = np.zeros(shp)
            if old is not None:
                a[: self.n] = getattr(self, name)[: self.n]
            setattr(self, name, a)
        for name in ("PAR", "LV", "NXT"):
            a = np.full(cap, -1, dtype=np.int64)
            if old is not None:
                a[: self.n] = getattr(self, name)[: self.n]
            setattr(self, name, a)

    def reserve(self, extra: int) -> None:
        if self.n + extra > len(self.P):
            self._alloc(2 * (self.n + extra))

    def add_fixed(self, p, t1, t2, hr) -> None:
        """Seeds: stored without the exclusion test."""
        m = len(p)
        self.reserve(m)
        s = slice(self.n, self.n + m)
        self.P[s], self.T1[s], self.T2[s], self.H[s] = p, t1, t2, hr
        self.PAR[s] = -1
        _insert_all(self.n, self.n + m, self.P, self.H, self.LV, self.NXT, self.head, self.used, self.s0, self.k,
                    self.mask)
        self.n += m

    def accept(self, cand, ct1, ct2, ch, cpar) -> np.ndarray:
        """Indices of the kept candidates' new points."""
        self.reserve(len(cand))
        n0 = self.n
        self.n = _accept(n0, np.ascontiguousarray(cand), np.ascontiguousarray(ct1), np.ascontiguousarray(ct2),
                         np.ascontiguousarray(ch), np.ascontiguousarray(cpar, dtype=np.int64), self.P, self.T1,
                         self.T2, self.H, self.PAR, self.LV, self.NXT, self.head, self.used, self.s0, self.k,
                         self.mask)
        return np.arange(n0, self.n)


# ------------------------------------------------------------------------------------------------ driver

def _geodesic(p: np.ndarray, d: np.ndarray, ang: np.ndarray) -> np.ndarray:
    return _unit(np.cos(ang)[:, None] * p + np.sin(ang)[:, None] * d)


def _coast_frames(loops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    t = []
    for lp in loops:
        d = np.roll(lp, -1, axis=0) - np.roll(lp, 1, axis=0)
        t.append(_unit(d - np.einsum("ij,ij->i", d, lp)[:, None] * lp))
    p = np.vstack(loops)
    t1 = np.vstack(t)
    return t1, np.cross(p, t1)                                            # t2 points into the water (left)


def pack(domain: Domain, h: SizeField, o: dict, stats: dict | None = None) -> tuple[np.ndarray, int]:
    """Packed points (boundary nodes first, np.vstack(resample_loops(domain, h))) and the boundary count."""
    from meshom.delaunay import background_mesh
    from meshom.field import cross_field, singularities

    loops = resample_loops(domain, h)
    nb = sum(len(lp) for lp in loops)
    water_dom = Domain(loops, domain.name)
    bg = background_mesh(domain, h, o["field_factor"])
    bnodes = np.flatnonzero(bg.node_flag == 1) if loops else None
    field = cross_field(bg, h, boundary_nodes=bnodes, method=o["field"])
    sample = _FieldSampler(field)
    hb = h(bg.points)
    k = o["k"]
    store = _Packing(int(len(bg.points) / 3) + nb + 1024, float(hb.min()) / R_EARTH_M, k)

    if loops:
        bp = np.vstack(loops)
        t1, t2 = _coast_frames(loops)
        store.add_fixed(bp, t1, t2, h(bp) / R_EARTH_M)
    else:                                                                 # seed: background node farthest from cones
        from scipy.spatial import cKDTree

        tid, _ = singularities(field)
        cones = _unit(bg.points[bg.tris[tid]].sum(axis=1))
        far = int(np.argmax(cKDTree(cones).query(bg.points)[0])) if len(tid) else 0
        sp = bg.points[far:far + 1]
        t1, t2 = sample(sp)
        store.add_fixed(sp, t1, t2, h(sp) / R_EARTH_M)

    layer = np.arange(store.n)
    n_layers = 0
    while len(layer):
        n_layers += 1
        p = store.P[layer]
        hr = store.H[layer]
        dirs = np.stack([-store.T1[layer], -store.T2[layer], store.T1[layer], store.T2[layer]], axis=1)
        par = np.repeat(layer, 4)
        p4 = np.repeat(p, 4, axis=0)
        h4 = np.repeat(hr, 4)
        d4 = dirs.reshape(-1, 3)
        cand = _geodesic(p4, d4, h4)
        if o["step"] == "mean":
            h4 = 0.5 * (h4 + h(cand) / R_EARTH_M)
            cand = _geodesic(p4, d4, h4)
        if loops:
            wet = in_water(water_dom, cand)
            cand, par = cand[wet], par[wet]
        ct1, ct2 = sample(cand)
        layer = store.accept(cand, ct1, ct2, h(cand) / R_EARTH_M, par)
    if stats is not None:
        stats.update(n_layers=n_layers, n_packed=store.n, n_bg=len(bg.points))
    pts = store.P[: store.n].copy()
    if o["fill"] > 0:
        pts = _fill(pts, nb, loops, water_dom, h, store, sample, o["fill"])
    return pts, nb


def _triangulate(pts: np.ndarray, bedges: np.ndarray) -> np.ndarray:
    from meshom.delaunay import sphere_delaunay

    return sphere_delaunay(pts, bedges)


def _fill(pts, nb, loops, water_dom, h, store, sample, fill, rounds=5):
    """Insert circumcentres of Delaunay triangles with circumradius > fill * h (largest first, exclusion-tested)."""
    bedges = loop_edges(loops)
    for _ in range(rounds):
        tris = _triangulate(pts, bedges)
        a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
        cc = _unit(np.cross(b - a, c - a))
        rc = np.linalg.norm(cc - a, axis=1) * R_EARTH_M
        big = rc > fill * h(cc)
        if loops:
            big &= in_water(water_dom, cc)
        if not big.any():
            break
        cand = cc[big][np.argsort(-(rc[big] / h(cc[big])))]
        ct1, ct2 = sample(cand)
        new = store.accept(cand, ct1, ct2, h(cand) / R_EARTH_M, np.full(len(cand), -1))
        if len(new) == 0:
            break
        pts = store.P[: store.n].copy()
    return pts


def triangle_mesh(domain: Domain, h: SizeField, opts: dict | None = None, stats: dict | None = None) -> Mesh:
    o = _opts(opts)
    pts, nb = pack(domain, h, o, stats)
    loops_e = loop_edges(resample_loops(domain, h)) if domain.loops else np.zeros((0, 2), dtype=np.int64)
    tris = _triangulate(pts, loops_e)
    used = np.zeros(len(pts), dtype=bool)
    used[tris.ravel()] = True
    used[:nb] = True
    if not used.all():                                                    # points cut off by land removal
        remap = np.cumsum(used) - 1
        pts, tris = pts[used], remap[tris]
    flag = np.zeros(len(pts), dtype=np.int8)
    flag[:nb] = 1
    return Mesh(pts, tris.astype(np.int64), np.zeros((0, 4), dtype=np.int64), flag)


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    from meshom.recombine import recombine

    o = _opts(opts)
    mesh = recombine(triangle_mesh(domain, h, o), o["price"], o["recombine"])
    if o["post_smooth"]:
        from meshom.smooth import smooth

        mesh = smooth(mesh, h, mesh.node_flag == 1, iters=o["post_iters"], kind=o["post_smooth"])
    return mesh
