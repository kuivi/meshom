"""Standard picture set of one mesh (`meshom compete --figures`, `meshom figures`): fixed file names plus index.json.

render_run(mesh, domain, h, out_dir, title) writes into out_dir:
  overview.png        3 rows x 4 orthographic views (centred where ln h changes fastest, its antipode, the north and
                      the south pole): elements coloured by ln(s/h) (s = mean edge length, h at the centroid); by the
                      quad max angle (triangles grey, quads above 135 deg dotted); triangles with the irregular
                      interior nodes. Each view is a raster of the nearest element per pixel, so it
                      stays readable at 1e5-1e6 elements; the coast is a thin line.
  zoom_angle.png      ~30 x 30 cells around the quad with the largest corner angle (no quads: the worst triangle)
  zoom_size.png       around the edge with the largest |ln s/h| (s = great-circle length, h at the midpoint)
  zoom_irregular.png  around the densest cluster of irregular interior nodes (most such nodes within 3 local cells
                      of one; cell = mean edge length at the node)
  zoom_seam.png       where fronts met: the |ln s/h| > 0.15 edge maximising (such edges) x (irregular nodes) within
                      3 local cells (no such edge: the centre of zoom_irregular, said in the title)
  zoom_coast.png      domains with coast loops only: the boundary node of the worst element touching the coast
                      (worst = largest (max angle - 90) / 60 for quads, (max angle - 60) / 60 for triangles)
  zoom_strait.png     domains with coast loops only: the narrowest water gap between two coast stretches (the loops
                      resampled at h; pairs of points on different loops, or on one loop but far apart along it, whose
                      midpoint is water; the middle of the near-narrowest stretch)
  index.json          the files, each zoom's centre (lon, lat), radius and local numbers
Zooms are drawn in the gnomonic plane at their centre (north up, metres), never in raw lon-lat: quads white-ish and
coloured by their max angle above ~110 deg, triangles shaded, edges coloured by |ln s/h|, irregular interior nodes
marked, a scale bar in km, the local numbers in the title.

Irregular = interior node whose valence (edge count) differs from the regular one: 4, or 6 when the mesh has more
triangles than quads (a triangle-only variant would otherwise mark every node). Markers: regular - 1 (3 or 5),
regular + 1 (5 or 7), other; index.json gives the regular valence and the counts keyed by valence.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from meshom.domain import Domain, in_water, resample_loops  # noqa: E402
from meshom.geo import R_EARTH_M, arc_length, lonlat_to_xyz, xyz_to_lonlat  # noqa: E402
from meshom.mesh import Mesh, build_topology  # noqa: E402
from meshom.quality.kernel import element_geometry  # noqa: E402
from meshom.size import SizeField  # noqa: E402

NAMES = ("overview", "zoom_angle", "zoom_size", "zoom_irregular", "zoom_seam", "zoom_coast", "zoom_strait")
ZOOM_CELLS = 30           # a zoom shows ~ZOOM_CELLS x ZOOM_CELLS cells of the local element size
KERNEL_CELLS = 3.0        # radius of the irregular / seam density kernel, in local cells (mean edge length)
SEAM_LN = 0.15            # |ln s/h| of a seam edge (the size-fidelity target)
QUAD_BAD_DEG = 135.0      # quads dotted in the overview (the p99.9 gate)
NPX = 440                 # overview raster, pixels across one view

QUAD_NORM = Normalize(90.0, 160.0)
QUAD_CMAP = LinearSegmentedColormap.from_list("quad_angle", [(0.0, "#ffffff"), (20 / 70, "#fbf5e6"),
                                                             (45 / 70, "#fdb863"), (60 / 70, "#e34a33"),
                                                             (1.0, "#7f0000")])
EDGE_NORM = Normalize(0.0, 0.4)
EDGE_CMAP = LinearSegmentedColormap.from_list("ln_sh", [(0.0, "#3c3c3c"), (0.2, "#3c3c3c"), (0.375, "#f39c12"),
                                                        (1.0, "#b2182b")])
LN_NORM = Normalize(-0.4, 0.4)
LN_CMAP = plt.get_cmap("RdBu_r")
TRI_FACE, TRI_DARK, QUAD_PLAIN, LAND, COAST = "#c3d0e3", "#8a8a8a", "#f6f6f6", "#e6e0cc", "#0b3d91"


@dataclass
class _Data:
    """Everything the pictures need, computed once per mesh (all vectorised)."""
    mesh: Mesh
    h: SizeField
    edges: np.ndarray        # (E,2)
    edge_elems: np.ndarray   # (E,2), column 1 = -1 on boundary edges
    bnd: np.ndarray          # (N,) boundary node
    valence: np.ndarray      # (N,) edges at the node
    node_size: np.ndarray    # (N,) mean length of the node's edges, m
    regular: int             # 4, or 6 on a mesh with more triangles than quads
    irregular: np.ndarray    # (N,) interior node with valence != regular
    edge_len: np.ndarray     # (E,) great-circle length s, m
    edge_ln: np.ndarray      # (E,) ln(s/h), h at the midpoint
    edge_mid: np.ndarray     # (E,3)
    cent: np.ndarray         # (M,3) element centroids
    cell: np.ndarray         # (M,) mean edge length, m
    elem_ln: np.ndarray      # (M,) ln(cell / h(centroid))
    elem_r: np.ndarray       # (M,) chord from the centroid to the farthest corner (unit sphere)
    max_angle: np.ndarray    # (M,) largest corner angle, deg (tangent plane of the element)
    touches_bnd: np.ndarray  # (M,) element has a boundary node
    tree: cKDTree            # over cent

    def marks(self) -> tuple:
        """(valence or 0 = other, marker, colour, label) of the irregular node classes."""
        r = self.regular
        return ((r - 1, "^", "#1f4fd1", f"valence {r - 1}"), (r + 1, "p", "#1a9850", f"valence {r + 1}"),
                (0, "X", "#9c27b0", "other valence"))

    def classes(self, val: np.ndarray) -> dict:
        """Counts of irregular valences val by class, keyed "<regular-1>", "<regular+1>", "other"."""
        lo, hi = self.regular - 1, self.regular + 1
        return {str(lo): int((val == lo).sum()), str(hi): int((val == hi).sum()),
                "other": int(((val != lo) & (val != hi)).sum())}

    def select(self, val: np.ndarray, k: int) -> np.ndarray:
        lo, hi = self.regular - 1, self.regular + 1
        return (val == k) if k else ((val != lo) & (val != hi))

    def nodes_of(self, m: int) -> np.ndarray:
        t = self.mesh.n_tris
        return self.mesh.tris[m] if m < t else self.mesh.quads[m - t]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _prepare(mesh: Mesh, h: SizeField) -> _Data:
    topo = build_topology(mesh)
    g = element_geometry(mesh)
    p = mesh.points
    deg = np.bincount(topo.edges.ravel(), minlength=mesh.n_nodes)
    a, b = p[topo.edges[:, 0]], p[topo.edges[:, 1]]
    mid = _unit(a + b)
    length = R_EARTH_M * arc_length(a, b)
    edge_ln = np.log(length / h(mid))
    node_size = np.bincount(topo.edges.ravel(), np.repeat(length, 2), minlength=mesh.n_nodes) / np.maximum(deg, 1)
    cell = np.concatenate([g.tri_edges.mean(axis=1), g.quad_edges.mean(axis=1)])
    cent = g.centroid
    rad = [np.linalg.norm(p[c] - cent[off:off + len(c), None, :], axis=2).max(axis=1)
           for c, off in ((mesh.tris, 0), (mesh.quads, mesh.n_tris)) if len(c)]
    ang = np.degrees(np.concatenate([g.tri_angles.max(axis=1, initial=0.0), g.quad_angles.max(axis=1, initial=0.0)]))
    tb = np.concatenate([topo.boundary_node[mesh.tris].any(axis=1), topo.boundary_node[mesh.quads].any(axis=1)])
    reg = 4 if mesh.n_quads >= mesh.n_tris else 6
    return _Data(mesh, h, topo.edges, topo.edge_elems, topo.boundary_node, deg, node_size, reg,
                 ~topo.boundary_node & (deg != reg),
                 length, edge_ln, mid, cent, cell, np.log(cell / h(cent)), np.concatenate(rad), ang, tb, cKDTree(cent))


def _east_north(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """East and north unit vectors at unit c (at a pole: east = +y, so lon 0 is at the bottom of a north-pole view)."""
    e = np.cross([0.0, 0.0, 1.0], c)
    e = e / np.linalg.norm(e) if np.linalg.norm(e) > 1e-9 else np.array([0.0, 1.0, 0.0])
    return e, np.cross(c, e)


def _lonlat(c: np.ndarray) -> list[float]:
    lon, lat = xyz_to_lonlat(c)
    return [round(float(lon), 3), round(float(lat), 3)]


def _fibonacci(m: int) -> np.ndarray:
    i = np.arange(m) + 0.5
    z = 1.0 - 2.0 * i / m
    ph = np.pi * (1.0 + 5**0.5) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(ph), r * np.sin(ph), z])


def _densest(pts: np.ndarray, rad: np.ndarray) -> tuple[int, int]:
    """Row of pts with the most pts within its chord radius (a box kernel density); ties by the count at 2 x."""
    tree = cKDTree(pts)
    n1 = np.asarray(tree.query_ball_point(pts, rad, return_length=True))
    n2 = np.asarray(tree.query_ball_point(pts, 2.0 * rad, return_length=True))
    i = int(np.lexsort((n2, n1))[-1])
    return i, int(n1[i])


def _strait(domain: Domain, h: SizeField):
    """(centre, width_m, a, b): the narrowest water gap between two coast stretches, or None."""
    loops = resample_loops(domain, h)
    pts = np.vstack(loops)
    lid = np.repeat(np.arange(len(loops)), [len(lp) for lp in loops])
    seg = [R_EARTH_M * arc_length(lp, np.roll(lp, -1, axis=0)) for lp in loops]
    s = np.concatenate([np.r_[0.0, np.cumsum(x)[:-1]] for x in seg])
    length = np.array([x.sum() for x in seg])[lid]
    k = min(49, len(pts))
    dist, nb = cKDTree(pts).query(pts, k=k)
    dist, nb = R_EARTH_M * dist[:, 1:], nb[:, 1:]
    ds = np.abs(s[nb] - s[:, None])
    ds = np.minimum(ds, length[:, None] - ds)
    ok = (lid[nb] != lid[:, None]) | (ds > 4.0 * dist)     # another loop, or the same loop folded back
    dist = np.where(ok, dist, np.inf)
    j = np.argmin(dist, axis=1)
    rows = np.arange(len(pts))
    dmin, partner = dist[rows, j], nb[rows, j]
    order = np.argsort(dmin)
    order = order[np.isfinite(dmin[order])]
    mid = _unit(pts + pts[partner])
    for k in range(0, len(order), 256):                        # the narrowest gap whose midpoint is water
        chunk = order[k:k + 256]
        wet = in_water(domain, mid[chunk])
        if wet.any():
            break
    else:
        return None
    near = order[dmin[order] <= 1.05 * dmin[chunk[wet][0]]]      # all near-narrowest pairs (a channel: its length)
    near = near[in_water(domain, mid[near])]
    i = near[np.argmax(mid[near] @ _unit(mid[near].mean(axis=0)))]
    return mid[i], float(dmin[i]), pts[i], pts[partner[i]]


# ----------------------------------------------------------------------------------------------------------- overview

def _views(h: SizeField) -> list[tuple[float, float]]:
    """Centre of the region where ln h changes fastest (lat clipped to +-60), its antipode, both poles."""
    f = _fibonacci(20_000)
    lh = np.log(h(f))
    dist, nb = cKDTree(f).query(f, k=7)
    g = (np.abs(lh[nb[:, 1:]] - lh[:, None]) / dist[:, 1:]).max(axis=1)
    if g.max() <= 1e-9:
        lon0, lat0 = 0.0, 0.0
    else:
        m = f[g >= 0.5 * g.max()].mean(axis=0)
        c = m / np.linalg.norm(m) if np.linalg.norm(m) > 0.3 else f[np.argmax(g)]
        lon0, lat0 = map(float, xyz_to_lonlat(c))
        lat0 = float(np.clip(lat0, -60.0, 60.0))
    return [(lon0, lat0), ((lon0 + 360.0) % 360.0 - 180.0, -lat0), (0.0, 90.0), (0.0, -90.0)]


def _rgba(c) -> np.ndarray:
    return np.array(to_rgba(c))


def _overview(d: _Data, domain: Domain, out: Path, title: str, counts: dict) -> dict:
    T = d.mesh.n_tris
    x = (np.arange(NPX) + 0.5) / NPX * 2.0 - 1.0
    X, Y = np.meshgrid(x, x)
    r2 = X * X + Y * Y
    disc = r2 < 1.0
    views = _views(d.h)
    fig = plt.figure(figsize=(17.0, 13.2))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1, 0.07], left=0.06, right=0.95, top=0.9, bottom=0.02,
                          wspace=0.04, hspace=0.06)
    rows = ("ln(s/h) per element\nblue finer, red coarser than h", "quad max angle\ntriangles grey, dots > 135°",
            "triangles (grey) and irregular\ninterior nodes")
    irr = np.flatnonzero(d.irregular)
    bad = T + np.flatnonzero(d.max_angle[T:] > QUAD_BAD_DEG)
    for col, (lon, lat) in enumerate(views):
        c = lonlat_to_xyz(np.array(lon), np.array(lat))
        e, n = _east_north(c)
        P = X[disc][:, None] * e + Y[disc][:, None] * n + np.sqrt(1.0 - r2[disc])[:, None] * c
        dist, idx = d.tree.query(P, workers=-1)
        wet = dist <= 1.8 * d.elem_r[idx]
        is_tri = idx < T
        layers = [LN_CMAP(LN_NORM(d.elem_ln[idx])),
                  np.where(is_tri[:, None], _rgba(TRI_DARK), QUAD_CMAP(QUAD_NORM(d.max_angle[idx]))),
                  np.where(is_tri[:, None], _rgba(TRI_DARK), _rgba(QUAD_PLAIN))]
        for row, colours in enumerate(layers):
            img = np.zeros((NPX, NPX, 4))
            colours = np.where(wet[:, None], colours, _rgba(LAND))
            img[disc] = colours
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img, origin="lower", extent=(-1, 1, -1, 1), interpolation="nearest")
            ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=0.6, color="#777777"))
            for lp in domain.loops:
                w = lp @ c
                xy = np.column_stack([lp @ e, lp @ n])
                xy[w < 0] = np.nan
                xy = np.vstack([xy, xy[:1]])
                ax.plot(xy[:, 0], xy[:, 1], color=COAST, lw=0.5)
            if row == 1 and len(bad):
                v = bad[d.cent[bad] @ c > 0.2]                  # not at the limb: views overlap
                ax.scatter(d.cent[v] @ e, d.cent[v] @ n, s=10, c=d.max_angle[v], cmap=QUAD_CMAP, norm=QUAD_NORM,
                           edgecolors="k", linewidths=0.3, zorder=3)
            if row == 2:
                v = irr[d.mesh.points[irr] @ c > 0.2]
                size = float(np.clip(4000.0 / max(len(v), 1), 0.4, 8.0))
                val = d.valence[v]
                t = np.flatnonzero(d.cent[:T] @ c > 0.2)          # triangles: squares around the node dots, so
                ax.scatter(d.cent[t] @ e, d.cent[t] @ n,          # they stay visible at 1e5+ elements
                           s=2.5 * size if d.regular == 4 else 0.3, marker="s", c=TRI_DARK, linewidths=0, zorder=2)
                for k, mk, colour, _ in sorted(d.marks(), key=lambda m: -int(d.select(val, m[0]).sum())):
                    q = d.mesh.points[v[d.select(val, k)]]          # the most frequent class first, rare on top
                    ax.scatter(q @ e, q @ n, s=size, marker="o", c=colour, linewidths=0, zorder=3)
            ax.set_xlim(-1.02, 1.02)
            ax.set_ylim(-1.02, 1.02)
            ax.set_aspect("equal")
            ax.axis("off")
            if row == 0:
                ax.set_title(f"centre {lon:.0f}°E, {lat:.0f}°N", fontsize=10)
            if col == 0:
                ax.text(-1.12, 0.0, rows[row], rotation=90, ha="center", va="center", fontsize=10,
                        transform=ax.transData)
    fig.colorbar(ScalarMappable(LN_NORM, LN_CMAP), cax=fig.add_subplot(gs[0, 4]), label="ln(s/h)",
                 extend="both")
    fig.colorbar(ScalarMappable(QUAD_NORM, QUAD_CMAP), cax=fig.add_subplot(gs[1, 4]),
                 label="quad max angle (°)", extend="max")
    lax = fig.add_subplot(gs[2, 4])
    lax.axis("off")
    handles = [Patch(fc=TRI_DARK, label="triangle"), Patch(fc=LAND, label="land")]
    handles += [Line2D([], [], ls="", marker="o", color=colour, label=lab) for _, _, colour, lab in d.marks()]
    lax.legend(handles=handles, loc="center left", fontsize=9, frameon=False, bbox_to_anchor=(-1.5, 0.5))
    c = counts
    fig.suptitle(f"{title}\n{c['n_nodes']} nodes, {c['n_quads']} quads, {c['n_tris']} triangles; interior irregular "
                 f"(valence ≠ {d.regular}) {_classes(c['irregular'])}; quad θmax max {c['quad_max_deg']}°; "
                 f"|ln s/h| p99 {c['ln_sh_p99']}, max {c['ln_sh_max']}", fontsize=11)
    fig.savefig(out / "overview.png", dpi=100)
    plt.close(fig)
    return {"name": "overview", "file": "overview.png", "views_lonlat": [[round(a, 3), round(b, 3)] for a, b in views]}


# -------------------------------------------------------------------------------------------------------------- zooms

def _scale_bar(ax, R: float) -> None:
    w = 2.0 * R
    target = 0.2 * w / 1e3
    p = 10.0 ** np.floor(np.log10(target))
    km = max(m for m in (1, 2, 5, 10) if m * p <= target) * p
    x0, y0 = -R + 0.05 * w, -R + 0.05 * w
    halo = [withStroke(linewidth=7, foreground="white")]
    ax.plot([x0, x0 + km * 1e3], [y0, y0], color="k", lw=4, solid_capstyle="butt", zorder=8, path_effects=halo)
    ax.text(x0 + 0.5 * km * 1e3, y0 + 0.015 * w, f"{km:g} km", ha="center", va="bottom", fontsize=11, zorder=8,
            path_effects=[withStroke(linewidth=3, foreground="white")])


def _classes(c: dict) -> str:
    """{"3": 5, "5": 9, "other": 0} -> "n3 / n5 / other 5 / 9 / 0"."""
    k = list(c)
    return f"n{k[0]} / n{k[1]} / other {c[k[0]]} / {c[k[1]]} / {c['other']}"


def _fmt(v, nd: int = 1) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _zoom(d: _Data, domain: Domain, out: Path, title: str, name: str, centre: np.ndarray, head: str,
          mark: dict | None = None, min_radius_m: float = 0.0, note: str | None = None, extra: dict | None = None) -> dict:
    mesh, T = d.mesh, d.mesh.n_tris
    c = _unit(np.asarray(centre, dtype=np.float64))
    e, n = _east_north(c)
    _, near = d.tree.query(c, k=min(40, len(d.cell)))
    cell = float(np.median(d.cell[np.atleast_1d(near)]))
    R = max(0.5 * ZOOM_CELLS * cell, min_radius_m)

    def proj(p: np.ndarray) -> np.ndarray:
        return R_EARTH_M * np.stack([p @ e, p @ n], axis=-1) / (p @ c)[..., None]

    ids = np.asarray(d.tree.query_ball_point(c, r=(1.5 * R + 3.0 * cell) / R_EARTH_M), dtype=np.int64)
    ids = ids[d.cent[ids] @ c > 0.2]
    ti, qi = ids[ids < T], ids[ids >= T] - T
    fig = plt.figure(figsize=(10.6, 9.0))
    ax = fig.add_axes([0.02, 0.02, 0.74, 0.84])
    ax.add_collection(PolyCollection(proj(mesh.points[mesh.tris[ti]]), facecolors=TRI_FACE, edgecolors="none"))
    ax.add_collection(PolyCollection(proj(mesh.points[mesh.quads[qi]]), edgecolors="none",
                                     facecolors=QUAD_CMAP(QUAD_NORM(d.max_angle[T + qi]))))
    insel = np.zeros(mesh.n_elems, dtype=bool)
    insel[ids] = True
    ee = d.edge_elems
    eids = np.flatnonzero(insel[ee[:, 0]] | ((ee[:, 1] >= 0) & insel[np.maximum(ee[:, 1], 0)]))
    absln = np.abs(d.edge_ln[eids])
    ax.add_collection(LineCollection(proj(mesh.points[d.edges[eids]]), colors=EDGE_CMAP(EDGE_NORM(absln)),
                                     linewidths=np.where(absln > SEAM_LN, 1.7, 0.6), zorder=2))
    for lp in domain.loops:
        xy = proj(lp)
        xy[lp @ c < 0.2] = np.nan
        xy = np.vstack([xy, xy[:1]])
        ax.plot(xy[:, 0], xy[:, 1], color=COAST, lw=1.4, zorder=3)

    nodes = np.unique(d.edges[eids])
    irr = nodes[d.irregular[nodes]]
    ixy, ival = proj(mesh.points[irr]), d.valence[irr]
    size = float(np.clip(34.0 * (ZOOM_CELLS * cell / (2.0 * R)) ** 2, 5.0, 40.0))
    for k, mk, colour, _ in d.marks():
        sel = d.select(ival, k)
        ax.scatter(ixy[sel, 0], ixy[sel, 1], s=size, marker=mk, c=colour, edgecolors="white", linewidths=0.5,
                   zorder=4)

    mark = mark or {}
    if "elem" in mark:
        q = proj(mesh.points[d.nodes_of(mark["elem"])])
        q = np.vstack([q, q[:1]])
        ax.plot(q[:, 0], q[:, 1], color="#ff00c8", lw=2.6, zorder=6)
    if "edge" in mark:
        q = proj(mesh.points[d.edges[mark["edge"]]])
        ax.plot(q[:, 0], q[:, 1], color="#ff00c8", lw=3.2, zorder=6)
        ax.add_patch(Circle((0.0, 0.0), 1.5 * cell, fill=False, ec="#ff00c8", lw=1.6, zorder=6))
    if "node" in mark:
        q = proj(mesh.points[mark["node"]])
        ax.scatter([q[0]], [q[1]], s=140, marker="*", c="#ff00c8", edgecolors="k", linewidths=0.6, zorder=7)
    if "circle_m" in mark:
        ax.add_patch(Circle((0.0, 0.0), mark["circle_m"], fill=False, ec="#ff00c8", lw=1.8, ls="--", zorder=6))
    if "segment" in mark:
        q = proj(np.asarray(mark["segment"]))
        ax.annotate("", q[1], q[0], arrowprops=dict(arrowstyle="<->", color="#ff00c8", lw=2.0), zorder=7)

    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    _scale_bar(ax, R)

    # local numbers: elements, edges and irregular nodes inside the square
    box = lambda xy: np.abs(xy).max(axis=-1) <= R  # noqa: E731
    ib = ids[box(proj(d.cent[ids]))]
    qa, ta = d.max_angle[ib[ib >= T]], d.max_angle[ib[ib < T]]
    eb = eids[box(proj(d.edge_mid[eids]))]
    lb = np.abs(d.edge_ln[eb])
    vb = ival[box(ixy)]
    num = {"n_quads": int(len(qa)), "n_tris": int(len(ta)),
           "quad_max_angle_deg": round(float(qa.max()), 2) if len(qa) else None,
           "tri_max_angle_deg": round(float(ta.max()), 2) if len(ta) else None,
           "ln_sh_p99": round(float(np.percentile(lb, 99)), 4) if len(lb) else None,
           "ln_sh_max": round(float(lb.max()), 4) if len(lb) else None,
           "irregular": d.classes(vb)}
    num.update(extra or {})
    lon, lat = _lonlat(c)
    line2 = (f"in view: {num['n_quads']} quads, {num['n_tris']} triangles; quad θmax max "
             f"{_fmt(num['quad_max_angle_deg'])}°, tri θmax max {_fmt(num['tri_max_angle_deg'])}°; |ln s/h| p99 "
             f"{_fmt(num['ln_sh_p99'], 3)}, max {_fmt(num['ln_sh_max'], 3)}; irregular {_classes(num['irregular'])}")
    fig.suptitle(f"{title} — {name} at {lon:.2f}°E {lat:.2f}°N, cell ≈ {cell / 1e3:.1f} km\n{head}"
                 + (f" [{note}]" if note else "") + f"\n{line2}", fontsize=10, y=0.985)

    handles = [Patch(fc="white", ec="#3c3c3c", label="quad"), Patch(fc=TRI_FACE, ec="#3c3c3c", label="triangle")]
    handles += [Line2D([], [], ls="", marker=mk, color=colour, markeredgecolor="white", markersize=8, label=lab)
                for _, mk, colour, lab in d.marks()]
    handles += [Line2D([], [], color=COAST, lw=1.4, label="coast"), Line2D([], [], color="#ff00c8", lw=2.4,
                                                                          label="zoom target")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.775, 0.86), fontsize=9, frameon=False)
    fig.colorbar(ScalarMappable(QUAD_NORM, QUAD_CMAP), cax=fig.add_axes([0.80, 0.36, 0.025, 0.26]),
                 label="quad max angle (°), faces", extend="max")
    fig.colorbar(ScalarMappable(EDGE_NORM, EDGE_CMAP), cax=fig.add_axes([0.80, 0.04, 0.025, 0.26]),
                 label="|ln s/h|, edges (thick > 0.15)", extend="max")
    fig.savefig(out / f"{name}.png", dpi=100)
    plt.close(fig)
    entry = {"name": name, "file": f"{name}.png", "center_lonlat": [lon, lat], "radius_km": round(R / 1e3, 2),
             "cell_km": round(cell / 1e3, 2), "headline": head, "numbers": num}
    if note:
        entry["note"] = note
    return entry


# ------------------------------------------------------------------------------------------------------------- driver

def render_run(mesh: Mesh, domain: Domain, h: SizeField, out_dir, title: str = "") -> list[Path]:
    """Write the standard picture set of `mesh` (target size h, coast from `domain`) into out_dir; returns the PNG
    paths and index.json (last). Stale files of the set in out_dir are removed first."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for nm in NAMES:
        (out / f"{nm}.png").unlink(missing_ok=True)
    d = _prepare(mesh, h)
    T, p = mesh.n_tris, mesh.points
    absln = np.abs(d.edge_ln)
    val = d.valence[d.irregular]
    counts = {"n_nodes": mesh.n_nodes, "n_quads": mesh.n_quads, "n_tris": T, "regular_valence": d.regular,
              "irregular": d.classes(val),
              "quad_max_deg": round(float(d.max_angle[T:].max()), 2) if mesh.n_quads else None,
              "tri_max_deg": round(float(d.max_angle[:T].max()), 2) if T else None,
              "ln_sh_p99": round(float(np.percentile(absln, 99)), 4), "ln_sh_max": round(float(absln.max()), 4)}
    entries = [_overview(d, domain, out, title, counts)]

    if mesh.n_quads:
        m = T + int(np.argmax(d.max_angle[T:]))
        head = f"worst quad: max corner angle {d.max_angle[m]:.1f}°"
    else:
        m = int(np.argmax(d.max_angle))
        head = f"no quads; worst triangle: max corner angle {d.max_angle[m]:.1f}°"
    entries.append(_zoom(d, domain, out, title, "zoom_angle", d.cent[m], head, {"elem": m},
                         extra={"target_angle_deg": round(float(d.max_angle[m]), 2)}))

    i = int(np.argmax(absln))
    s_km = d.edge_len[i] / 1e3
    head = f"largest |ln s/h| = {absln[i]:.3f}: edge s = {s_km:.1f} km, h = {s_km / np.exp(d.edge_ln[i]):.1f} km"
    entries.append(_zoom(d, domain, out, title, "zoom_size", d.edge_mid[i], head, {"edge": i},
                         extra={"target_ln_sh": round(float(d.edge_ln[i]), 4)}))

    irr = np.flatnonzero(d.irregular)
    if len(irr):
        rad = KERNEL_CELLS * d.node_size[irr] / R_EARTH_M          # 3 local cells (mean edge length at the node)
        j, cnt = _densest(p[irr], rad)
        irr_centre, irr_rad = p[irr[j]], rad[j] * R_EARTH_M
        head = (f"densest irregular cluster: {cnt} irregular interior nodes within {KERNEL_CELLS:g} cells = "
                f"{irr_rad / 1e3:.0f} km (dashed) of the centre; {len(irr)} in the mesh")
        entries.append(_zoom(d, domain, out, title, "zoom_irregular", irr_centre, head, {"circle_m": irr_rad},
                             extra={"target_count": cnt}))
    else:
        irr_centre, irr_rad = d.cent[m], KERNEL_CELLS * float(d.cell[m])
        entries.append(_zoom(d, domain, out, title, "zoom_irregular", irr_centre, "no irregular interior nodes",
                             note="centre of zoom_angle"))

    bad = np.flatnonzero(absln > SEAM_LN)
    if len(bad):
        # a seam: |ln s/h| > 0.15 edges AND irregular nodes together; score = their product within 3 local cells of
        # a bad edge (ties and the no-irregular case: by the bad-edge count)
        at, r = d.edge_mid[bad], KERNEL_CELLS * d.edge_len[bad] / R_EARTH_M
        n_bad = np.asarray(cKDTree(at).query_ball_point(at, r, return_length=True))
        n_irr = (np.asarray(cKDTree(p[irr]).query_ball_point(at, r, return_length=True)) if len(irr)
                 else np.zeros(len(bad), dtype=np.int64))
        j = int(np.lexsort((n_bad, n_bad * n_irr))[-1])
        head = (f"seam: {n_bad[j]} edges with |ln s/h| > {SEAM_LN:g} and {n_irr[j]} irregular nodes within "
                f"{KERNEL_CELLS:g} cells = {r[j] * R_EARTH_M / 1e3:.0f} km (dashed); {len(bad)} such edges in the mesh")
        entries.append(_zoom(d, domain, out, title, "zoom_seam", at[j], head, {"circle_m": r[j] * R_EARTH_M},
                             note=None if n_irr[j] else "no irregular node near a bad edge: densest bad edges",
                             extra={"target_bad_edges": int(n_bad[j]), "target_irregular": int(n_irr[j])}))
    else:
        entries.append(_zoom(d, domain, out, title, "zoom_seam", irr_centre,
                             f"no edge with |ln s/h| > {SEAM_LN:g}: no seam found", {"circle_m": irr_rad},
                             note="same centre as zoom_irregular"))

    if domain.loops:
        cand = np.flatnonzero(d.touches_bnd)
        badness = (d.max_angle - np.where(np.arange(mesh.n_elems) < T, 60.0, 90.0)) / 60.0
        m = int(cand[np.argmax(badness[cand])])
        nodes = d.nodes_of(m)
        b = int(nodes[d.bnd[nodes]][0])
        kind = "quad" if m >= T else "triangle"
        head = f"worst element touching the coast: {kind}, max corner angle {d.max_angle[m]:.1f}° (star: its coast node)"
        entries.append(_zoom(d, domain, out, title, "zoom_coast", p[b], head, {"elem": m, "node": b},
                             extra={"target_angle_deg": round(float(d.max_angle[m]), 2), "target_kind": kind}))
        st = _strait(domain, d.h)
        if st is None:
            entries.append(_zoom(d, domain, out, title, "zoom_strait", p[b], "no water gap between coast stretches",
                                 note="same centre as zoom_coast"))
        else:
            centre, width, a, bb = st
            _, near = d.tree.query(centre, k=min(40, len(d.cell)))
            cells = width / float(np.median(d.cell[np.atleast_1d(near)]))
            head = f"narrowest water gap: {width / 1e3:.0f} km ≈ {cells:.1f} cells across (arrow)"
            entries.append(_zoom(d, domain, out, title, "zoom_strait", centre, head, {"segment": [a, bb]},
                                 min_radius_m=1.5 * width,
                                 extra={"target_width_km": round(width / 1e3, 2), "target_cells": round(cells, 2)}))

    (out / "index.json").write_text(json.dumps({"title": title, "counts": counts, "figures": entries}, indent=1))
    return [out / x["file"] for x in entries] + [out / "index.json"]
