"""m13: cubed sphere with refinement tiles (SQuadGen, P. A. Ullrich; CAM-SE / E3SM RRM grids; Guba et al. 2014).

1. Base: equiangular cubed sphere of nb x nb element blocks per panel, 2 nb elements per panel edge,
   nb = round(sqrt(4 pi / 6) R / (2 h_max)) (element area = h_max^2 on average).
2. Refinement (m13/refine.py): level map "element size > tol * h" (tol = sqrt 2, so every element ends within a
   factor sqrt 2 of h), normalised as in SQuadGen; each level refines the marked blocks by 2 and paves one ring of
   blocks around them with the LOWCONN tiles (standard / inner corner / outer corner + connecting quads + inner-corner
   merges): conforming, all quads, no hanging nodes.
3. SQuadGen spring smoothing (zero-rest-length springs k = 0.1 on all edges, nodes of the tiles grown by
   spring_dist - 1 rings, spring_iters steps; E3SM recipe 3 / 20), then meshom.smooth kind="size" (size_iters).
   Coast cases add a narrow-water criterion to the map (>= `narrow` elements across water, width from the facing
   bank), so a channel a few h wide is not closed by the cut below.
4. Coast: m12's cut (cells fully in water at >= coast_gap h from the coast, one coastal row of quads and triangles
   to the resampled coast, guarded Laplacian on the rows next to it), then the size smoothing with coast nodes fixed.
"""
import numpy as np

from meshom.domain import Domain
from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh
from meshom.methods import OVERRIDES, register
from meshom.size import SizeField

DEFAULTS = {
    "nb": None,              # blocks per panel edge (elements 2 nb); None: from max h
    "tol": float(np.sqrt(2.0)),  # refine a block while its elements are > tol * h somewhere in it
    "narrow": 4,             # coast cases: >= this many elements across narrow water (0: off)
    "spring_dist": 3,        # SQuadGen --smooth_dist (graph rings around the tiles; < 0: whole mesh)
    "spring_iters": 20,      # SQuadGen --smooth_iter (E3SM recipe: 3 / 20)
    "size_iters": 20,        # meshom.smooth kind="size" sweeps afterwards (0: off)
    "coast": "strip",        # m12 coast cut (m12/coast.py)
    "coast_gap": 0.4,
    "smooth_rows": 3,
    "smooth_iters": 30,
    "verbose": False,
}


def _fibonacci(m: int) -> np.ndarray:
    i = np.arange(m) + 0.5
    z = 1.0 - 2.0 * i / m
    ph = np.pi * (1.0 + 5**0.5) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(ph), r * np.sin(ph), z])


class NarrowCap:
    """Largest element size from narrow water: for each coast vertex p, w_p = distance to the nearest coast vertex q
    across the water (q - p within 60 deg of p's water normal, and q's water normal facing back); a point whose
    nearest coast vertex p is within w_p gets elements <= w_p / n_across (m)."""

    def __init__(self, loops: list[np.ndarray], n_across: float, r_max_m: float):
        from scipy.spatial import cKDTree
        v = np.vstack(loops)
        t = np.vstack([np.roll(lp, -1, axis=0) - np.roll(lp, 1, axis=0) for lp in loops])
        nrm = np.cross(v, t)                                            # water (left) normal
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        self.tree = cKDTree(v)
        k = min(len(v), 1024)
        self.w = np.empty(len(v))
        for a in range(0, len(v), 512):                                 # chunks: (512, k) neighbour arrays
            vv, nn = v[a:a + 512], nrm[a:a + 512]
            d, j = self.tree.query(vv, k=k, distance_upper_bound=r_max_m / R_EARTH_M)
            ok = np.isfinite(d)
            jj = np.where(ok, j, 0)
            dv = v[jj] - vv[:, None, :]
            dist = np.linalg.norm(dv, axis=2)
            across = ok & (np.einsum("nki,ni->nk", dv, nn) > 0.5 * dist) & (np.einsum("nki,nki->nk", -dv, nrm[jj]) > 0)
            self.w[a:a + 512] = R_EARTH_M * np.where(across, dist, np.inf).min(axis=1)
        self.n = n_across
        fin = self.w[np.isfinite(self.w)]
        self.r = (fin.max() if len(fin) else 0.0) / R_EARTH_M

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        out = np.full(len(xyz), np.inf)
        if self.r > 0:
            d, j = self.tree.query(xyz, distance_upper_bound=self.r)
            ok = np.isfinite(d)
            w = self.w[j[ok]]
            out[ok] = np.where(R_EARTH_M * d[ok] <= w, w / self.n, np.inf)
        return out


def build(domain: Domain, h: SizeField, opts: dict | None = None) -> tuple[Mesh, dict]:
    """The mesh and diagnostics (closed sphere: info["tag"] per quad, 0 = quartered block, k = tile of level k)."""
    from meshom.methods.m13.refine import refine, spring_smooth
    from meshom.smooth import smooth

    o = {**DEFAULTS, **(opts or {})}
    hs = h(_fibonacci(400_000))
    h_min, h_max = float(hs.min()), float(hs.max())
    nb = o["nb"] or max(1, int(round(np.sqrt(4 * np.pi / 6) * R_EARTH_M / (2 * h_max))))
    cap = None
    if domain.loops and o["narrow"]:
        s0 = np.sqrt(4 * np.pi / 6) * R_EARTH_M / (2 * nb)              # level-0 element size (nominal)
        cap = NarrowCap(domain.loops, o["narrow"], 1.3 * o["narrow"] * s0)
    x, quads, tag = refine(h, nb, h_min, h_max, o["tol"], o["verbose"], cap)
    if o["spring_iters"] > 0 and (tag > 0).any():
        mob = np.zeros(len(x), dtype=bool)
        mob[quads[tag > 0].ravel()] = True
        x = spring_smooth(x, quads, mob, o["spring_dist"], o["spring_iters"])
    info = {"nb": nb, "levels": int(tag.max()), "h_min": h_min, "h_max": h_max}
    if o["verbose"]:
        print(f"m13: nb {nb}, levels {info['levels']}, nodes {len(x)}, quads {len(quads)}, "
              f"tile quads {int((tag > 0).sum())}", flush=True)
    if not domain.loops or not o.get("cut", True):                     # cut=False: refined sphere (diagnostics)
        mesh = Mesh(x, np.zeros((0, 3), dtype=np.int64), quads, np.zeros(len(x), dtype=np.int8))
        info["tag"] = tag
    else:
        from meshom.methods.m12.coast import cut_to_coast
        mesh = cut_to_coast(x, quads, domain, h, o)
    if o["size_iters"] > 0:
        mesh = smooth(mesh, h, np.flatnonzero(mesh.node_flag == 1), iters=o["size_iters"], kind="size")
    return mesh, info


@register()
def generate(domain: Domain, h: SizeField, opts: dict | None = None) -> Mesh:
    return build(domain, h, opts)[0]


@register("m13_squadgen_spring", overrides={"size_iters": 0})
def generate_spring(domain: Domain, h: SizeField, opts: dict | None = None) -> Mesh:
    """SQuadGen as E3SM runs it: tiles + springs (3 rings, 20 steps), no size smoothing."""
    return build(domain, h, {**OVERRIDES["m13_squadgen_spring"], **(opts or {})})[0]


@register("m13_squadgen_raw", overrides={"size_iters": 0, "spring_iters": 0})
def generate_raw(domain: Domain, h: SizeField, opts: dict | None = None) -> Mesh:
    """Tiles only: no spring, no size smoothing (the coast cut keeps its own row smoothing)."""
    return build(domain, h, {**OVERRIDES["m13_squadgen_raw"], **(opts or {})})[0]
