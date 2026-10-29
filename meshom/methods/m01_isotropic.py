"""m01: isotropic points -> sphere Delaunay -> recombination (the "triangles first, then glue" baseline).

Triangle stage (also the triangle-only path of meshom, variant m01_triangles):
1. resample_loops(domain, h) -> fixed coast nodes; poisson_points (variable-radius Poisson disc, alpha chosen so the
   node count matches an equilateral mesh of edge h: N* = integral of 2 / (sqrt(3) h^2) dA over the water).
2. `iters` smoothing steps of the interior nodes, re-triangulating after each step (sphere_delaunay, then
   recover_edges + remove_land when there is a coast):
   - "spring" (default, 60 steps, dt 0.3; Persson & Strang 2004 DistMesh, as OceanMesh2D / JIGSAW-GEO 2017 §4.1):
     repulsive edge springs, F = max(L0 - L, 0), L0 = Fscale h(mid) sqrt(sum L^2 / sum h^2), x += dt F in the tangent
     plane, re-normalised.
   - "odt" (Chen & Xu 2004, Chen & Holst 2011 eq. 4.11): a node moves to the rho|tau|-weighted mean of the
     circumcentres of its incident triangles, rho = h^-4 at the triangle centroid (equidistributes |tau| ~ h^2);
     on the sphere the circumcentre is the unit normal of the triangle's plane, the mean is projected back.
     Converges in ~20 steps but stalls at a worse mesh than springs (scale 4 uniform: theta_max p99 / max 88.5 / 102
     deg, theta_min 27 deg vs springs 86 / 96, 38 deg).
   A step is halved (per node, up to 6 times) where it would invert a triangle of the current triangulation, so no
   node crosses the fixed coast; the coast nodes never move.
3. The final triangulation is the (constrained) Delaunay triangulation of the smoothed nodes.

Mixed variant m01_isotropic: meshom.recombine.recombine (greedy, or exact on small meshes) on that triangle mesh with a
triangle price; the default price 1/3 merges a pair only when the quad's worst corner deviation is below 60 deg (the
150 deg hard gate), so leftover triangles are where no acceptable pair exists (~10 %). Then 10 sweeps of
meshom.smooth.smooth(kind="size") with the coast fixed: quad p99.9 max angle ~146 -> ~135 deg at scale 4 (kind
"angle" squares the quads further but doubles |ln(s/h)| p99, 0.18 -> 0.37).
"""
import numpy as np

from meshom.delaunay import poisson_points, sphere_delaunay
from meshom.domain import Domain, in_water, loop_edges, resample_loops
from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh
from meshom.methods import register
from meshom.size import SizeField

DEFAULTS = {
    "smoother": "spring",    # "spring" | "odt" | "odt+spring" | "none"
    "iters": 60,             # smoothing steps (each followed by re-triangulation)
    "odt_iters": 10,         # "odt+spring": the first odt_iters steps are ODT, the rest springs
    "alpha": None,           # Poisson-disc exclusion factor; None = calibrated to the equilateral node count
    "price": 1.0 / 3.0,      # recombination: price of a leftover triangle (quad cost = max |theta - 90| / 90)
    "recombine": "greedy",   # "greedy" | "exact" (networkx, small meshes only)
    "post_smooth": "size",   # meshom.smooth.smooth kind after recombination ("size" | "angle" | None), coast fixed
    "post_iters": 10,
    "seed": 0,
    "dt": 0.3,               # spring time step (DistMesh uses 0.2; 0.4 oscillates)
    "fscale": 1.2,           # spring rest-length inflation (DistMesh)
}
ALPHA_REF = 0.78             # maximal Poisson disc: density ~ 0.7 / (alpha h)^2 vs 2 / (sqrt(3) h^2) equilateral


def _opts(opts: dict | None) -> dict:
    o = dict(DEFAULTS)
    o.update(opts or {})
    return o


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _fibonacci(m: int) -> np.ndarray:
    i = np.arange(m) + 0.5
    z = 1.0 - 2.0 * i / m
    ph = np.pi * (1.0 + 5**0.5) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(ph), r * np.sin(ph), z])


def target_nodes(domain: Domain, h: SizeField, m: int = 200_000) -> float:
    """Node count of an equilateral triangle mesh with edge h over the water: integral of 2 / (sqrt(3) h^2) dA."""
    lat = _fibonacci(m)
    wet = in_water(domain, lat) if domain.loops else np.ones(m, dtype=bool)
    return float(4.0 * np.pi * R_EARTH_M**2 / m * np.sum(2.0 / np.sqrt(3.0) / h(lat[wet]) ** 2))


def _triangulate(points: np.ndarray, bedges: np.ndarray) -> np.ndarray:
    return sphere_delaunay(points, bedges)


def _orient(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), a)


def _accumulate(n: int, idx: np.ndarray, val: np.ndarray) -> np.ndarray:
    """(n,3) sums of the rows val (M,3) into idx (M,)."""
    return np.column_stack([np.bincount(idx, val[:, k], minlength=n) for k in range(3)])


def _odt_target(points: np.ndarray, tris: np.ndarray, h: SizeField) -> np.ndarray:
    """Per node: rho|tau|-weighted mean of the circumcentres of its triangles, rho = h^-4 (projected to the sphere)."""
    n = len(points)
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    nrm = np.cross(b - a, c - a)
    two_area = np.linalg.norm(nrm, axis=1)
    cc = nrm / two_area[:, None]                       # spherical circumcentre (CCW -> same side as the triangle)
    hc = h(_unit(a + b + c))
    w = two_area * (hc.min() / hc) ** 4
    wc = w[:, None] * cc
    num = sum(_accumulate(n, tris[:, k], wc) for k in range(3))
    den = np.bincount(tris.ravel(), np.repeat(w, 3), minlength=n)
    out = points.copy()
    used = den > 0
    out[used] = _unit(num[used] / den[used, None])
    return out


def _spring_target(points: np.ndarray, tris: np.ndarray, h: SizeField, dt: float, fscale: float) -> np.ndarray:
    """DistMesh step: repulsive springs with rest length fscale * h scaled to the current total edge length.

    Per half-edge a -> b the push on a from b (each interior edge has both half-edges; a coast edge has one, but its
    nodes are fixed)."""
    n = len(points)
    a = tris.ravel()
    b = np.roll(tris, -1, axis=1).ravel()
    p, q = points[a], points[b]
    v = p - q
    L = np.linalg.norm(v, axis=1)
    hm = h(_unit(p + q)) / R_EARTH_M
    L0 = hm * fscale * np.sqrt(np.sum(L**2) / np.sum(hm**2))
    force = _accumulate(n, a, (np.maximum(L0 - L, 0.0) / L)[:, None] * v)
    force -= np.einsum("ij,ij->i", force, points)[:, None] * points
    return _unit(points + dt * force)


def _safe_move(points: np.ndarray, target: np.ndarray, tris: np.ndarray, free: np.ndarray,
               halvings: int = 6) -> np.ndarray:
    """Move free nodes toward target; the step of the nodes of a triangle that would invert is halved (up to
    `halvings` times), then such nodes stay put. Fixed nodes are not touched (bit for bit)."""
    step = free.astype(np.float64)
    new = points
    for k in range(halvings + 1 + len(points)):
        mv = step > 0
        new = points.copy()
        new[mv] = _unit(points[mv] + step[mv, None] * (target[mv] - points[mv]))
        bad = _orient(new, tris) <= 0.0
        if not bad.any():
            break
        nodes = np.unique(tris[bad])
        step[nodes] = step[nodes] * 0.5 if k < halvings else 0.0
    return new


def triangle_mesh(domain: Domain, h: SizeField, opts: dict | None = None) -> Mesh:
    """Smoothed isotropic triangle mesh of the water with edge length ~h (boundary nodes first, node_flag 1)."""
    o = _opts(opts)
    loops = resample_loops(domain, h)
    nb = sum(len(lp) for lp in loops)
    bedges = loop_edges(loops)
    alpha = o["alpha"]
    pts = poisson_points(domain, h, loops, alpha=alpha or ALPHA_REF, seed=o["seed"])
    if alpha is None:                                   # one correction of the density toward the equilateral count
        n_star = target_nodes(domain, h)
        corr = np.sqrt((len(pts) - nb) / max(n_star - nb, 1.0))
        if abs(corr - 1.0) > 0.01:
            pts = poisson_points(domain, h, loops, alpha=ALPHA_REF * corr, seed=o["seed"])
    free = np.ones(len(pts), dtype=bool)
    free[:nb] = False
    tris = _triangulate(pts, bedges)
    for it in range(o["iters"] if o["smoother"] != "none" else 0):
        if o["smoother"] == "odt" or (o["smoother"] == "odt+spring" and it < o["odt_iters"]):
            target = _odt_target(pts, tris, h)
        else:
            target = _spring_target(pts, tris, h, o["dt"], o["fscale"])
        pts = _safe_move(pts, target, tris, free)
        tris = _triangulate(pts, bedges)
    flag = np.zeros(len(pts), dtype=np.int8)
    flag[:nb] = 1
    return Mesh(pts, tris, np.zeros((0, 4), dtype=np.int64), flag)


@register("m01_triangles", overrides={})          # another entry path: no recombination
def generate_triangles(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """Triangle-only output (the triangle v0 path)."""
    return triangle_mesh(domain, h, opts)


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """Mixed output: the triangle mesh recombined into quads (leftover triangles where no pair is acceptable)."""
    from meshom.recombine import recombine

    o = _opts(opts)
    mesh = recombine(triangle_mesh(domain, h, o), o["price"], o["recombine"])
    if o["post_smooth"]:
        from meshom.smooth import smooth

        mesh = smooth(mesh, h, mesh.node_flag == 1, iters=o["post_iters"], kind=o["post_smooth"])
    return mesh
