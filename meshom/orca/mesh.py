# SPDX-License-Identifier: GPL-3.0-or-later
"""An ORCA grid (meshom.orca.generator) as a meshom mesh, and NEMO's land-sea mask on it (decisions B71, B78-B86).

- Cells and nodes: the mesh nodes are the F points (cell corners), a cell is the quad of the four F points around a
  T point, F(i-1, j-1), F(i, j-1), F(i, j), F(i-1, j) (NEMO: F(i, j) is the north-east corner of T(i, j)), CCW seen
  from outside. Columns are the grid's unique ones (its two wrap columns dropped), periodic in i.
- The north fold (`cells`, when the rows reach the pole line): F pivot (the pole line is F row j_p): F(i, j_p) and
  F(m(i), j_p) are one node; T pivot (the pole line is T row j_p): F(i, j_p) is F(m(i), j_p - 1), and the cells of
  row j_p come twice (once from each side): one copy is kept. m(i) = K - i (mod the column count), K from the grid's
  lattice (the mirror I -> 2 - I). F points that coincide (the poles: the code clamps the meridians' last points onto
  the pole line's ends) are merged; a cell with three distinct corners becomes a triangle, fewer: dropped; a cell
  smaller than SLIVER_FRAC of its row's median area (the slivers along the clamped pole line) is dropped (counted).
- NEMO's mask (`transfer_mask`): a cell is water when NEMO's mask says its T point is water (tmask /
  bottom_level > 0), transferred cell for cell where NEMO's T point is the generator's (within MATCH_FRAC of the
  cell's shorter side); elsewhere (eORCA's ice-shelf rows, ORCA2's hand-made Mediterranean patch, a grid without a
  file) the depth rule `depth_mask`: water when the mean depth over the cell from GEBCO, land counted as 0 m, is at
  least MIN_DEPTH_M (B83).
- The clean-up FESOM3 needs (`fesom_cleanup`, B84): cells with fewer than two edge neighbours removed, repeatedly
  (fesom_ini stops on them); where water cells meet only at a corner (a pinch) one fan is
  removed, the one that loses the fewest cells and cuts no channel (NEMO does not connect diagonal cells either; a
  node split, tried first, leaves two nodes at one position, which meshom's boundary Delaunay rejects); separate water
  bodies (NEMO's closed seas) are kept.
"""
from dataclasses import dataclass

import numpy as np

from meshom.geo import R_EARTH_M, lonlat_to_xyz
from meshom.mesh import Mesh

MATCH_FRAC = 0.25       # B82: NEMO's T point within this fraction of the cell (its shorter side) of the generator's
MIN_DEPTH_M = 10.0      # B83: the depth rule's minimum mean cell depth (calibrated against NEMO's masks)
MERGE_RAD = 1e-9        # F points closer than this (6 mm) are one node (the fold's mirror, the poles' clamp)
SLIVER_FRAC = 0.01      # B84: a cell smaller than this fraction of its row's median area is a pole sliver: dropped


@dataclass
class OrcaCells:
    """The cells of T rows j0 .. j1 (0-based rows of the grid) as a mesh, all of them (no mask)."""
    mesh: Mesh                  # quads (and the few pole triangles), CCW; nodes: the F points used
    cell_ij: np.ndarray         # (n_elems, 2) the T point (row, column) of each element, tris first then quads
    node_ij: np.ndarray         # (n_nodes, 2) the F point (row, column) of each node (one representative)
    rows: tuple                 # (j0, j1)
    ni: int                     # unique columns
    folded: bool
    info: dict


def mirror_k(g) -> int:
    """K of the F columns' mirror m(i) = K - i (mod ni): I_F(i) = i_offset + i / q + 1 / (2q), I -> 2 - I."""
    q = g.q
    ioff = g.info.get("i_offset", 0.5)
    k = (2.0 - 2.0 * ioff) * q - 1.0
    if abs(k - round(k)) > 1e-9:
        raise ValueError(f"the F columns have no integer mirror (K = {k})")
    return int(round(k))


def cells(g, j0: int, j1: int) -> OrcaCells:
    """The cells of T rows j0 .. j1 (1 <= j0 <= j1 <= g.j_pole) with their F corners (rows j0 - 1 .. j1); the fold
    closed when j1 == g.j_pole (module docstring)."""
    lon, lat = g.lon["f"][:, :-2], g.lat["f"][:, :-2]
    ni = lon.shape[1]
    nr = j1 - j0 + 2                                               # F rows j0 - 1 .. j1
    fid = np.arange(nr * ni, dtype=np.int64).reshape(nr, ni)
    pts = lonlat_to_xyz(lon[j0 - 1:j1 + 1].ravel(), lat[j0 - 1:j1 + 1].ravel())
    folded = j1 == g.j_pole
    info = {"rows": [int(j0), int(j1)], "folded": bool(folded), "pivot": g.pivot}
    alias = np.arange(nr * ni, dtype=np.int64)
    if folded:
        K = mirror_k(g)
        m = (K - np.arange(ni)) % ni
        top = fid[-1]
        if g.pivot == "F":                                         # F(i, j_p) = F(m(i), j_p)
            alias[top] = np.minimum(top, top[m])
        else:                                                      # F(i, j_p) = F(m(i), j_p - 1)
            alias[top] = fid[-2][m]
        src = np.flatnonzero(alias != np.arange(len(alias)))
        gap = np.linalg.norm(pts[src] - pts[alias[src]], axis=1)
        info["fold_nodes"] = int(len(src))
        info["fold_max_gap_m"] = float(gap.max() * R_EARTH_M) if len(gap) else 0.0
    # coincident F points (the poles' clamp): one node each
    key = np.round(pts / MERGE_RAD).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    rep = first[inv.ravel()]
    alias = rep[alias]                                             # fold first, then position
    alias = alias[alias]
    info["merged_coincident"] = int((rep != np.arange(len(rep))).sum())
    r, i = np.meshgrid(np.arange(1, nr), np.arange(ni), indexing="ij")
    im = (i - 1) % ni
    q = np.stack([fid[r - 1, im], fid[r - 1, i], fid[r, i], fid[r, im]], axis=-1).reshape(-1, 4)
    q = alias[q]
    cij = np.stack([r.ravel() + j0 - 1, i.ravel()], axis=1)
    if folded and g.pivot == "T":                                  # the pivot row's cells come twice: keep one copy
        srt = np.sort(q, axis=1)
        _, keep = np.unique(srt, axis=0, return_index=True)
        dup = np.ones(len(q), bool)
        dup[keep] = False
        info["fold_duplicate_cells"] = int(dup.sum())
        q, cij = q[~dup], cij[~dup]
    nd = _n_distinct(q)
    P = pts
    area = 0.5 * (np.linalg.norm(np.cross(P[q[:, 1]] - P[q[:, 0]], P[q[:, 2]] - P[q[:, 0]]), axis=1)
                  + np.linalg.norm(np.cross(P[q[:, 2]] - P[q[:, 0]], P[q[:, 3]] - P[q[:, 0]]), axis=1))
    med = np.zeros(int(cij[:, 0].max()) + 1 if len(cij) else 0)
    for r in np.unique(cij[:, 0]):
        sel = cij[:, 0] == r
        med[r] = np.median(area[sel])
    sliver = (area < SLIVER_FRAC * med[cij[:, 0]]) & (nd >= 3)
    info["pole_slivers_dropped"] = int(sliver.sum())
    nd = np.where(sliver, 0, nd)
    tri_sel = nd == 3
    drop = nd < 3
    info["pole_cells_to_triangles"] = int(tri_sel.sum())
    info["pole_cells_dropped"] = int((drop & ~sliver).sum())
    tris = np.array([_tri_of(row) for row in q[tri_sel]], np.int64).reshape(-1, 3)
    quads = q[nd == 4]
    used = np.unique(np.r_[tris.ravel(), quads.ravel()])
    new = np.full(len(pts), -1, np.int64)
    new[used] = np.arange(len(used))
    mesh = Mesh(pts[used], new[tris], new[quads], np.zeros(len(used), np.int8))
    node_ij = np.stack(np.unravel_index(used, (nr, ni)), axis=1)
    node_ij[:, 0] += j0 - 1
    cell_ij = np.r_[cij[tri_sel], cij[nd == 4]]
    return OrcaCells(mesh, cell_ij, node_ij, (j0, j1), ni, folded, info)


def _n_distinct(q: np.ndarray) -> np.ndarray:
    s = np.sort(q, axis=1)
    return 1 + (np.diff(s, axis=1) != 0).sum(axis=1)


def _tri_of(row: np.ndarray) -> list:
    out = []
    for v in row:                                                  # keep the cyclic order, drop repeats
        if not out or (v != out[-1]):
            out.append(v)
    if out[0] == out[-1]:
        out.pop()
    return out


def sub_cells(oc: OrcaCells, keep: np.ndarray) -> tuple[Mesh, np.ndarray]:
    """The kept elements of oc.mesh (tris first, then quads) with their used nodes: (mesh, old node ids)."""
    m = oc.mesh
    T = m.n_tris
    tris, quads = m.tris[keep[:T]], m.quads[keep[T:]]
    used = np.unique(np.r_[tris.ravel(), quads.ravel()])
    new = np.full(m.n_nodes, -1, np.int64)
    new[used] = np.arange(len(used))
    return Mesh(m.points[used], new[tris], new[quads], np.zeros(len(used), np.int8)), used


# ---- an ORCA part bounded by a line (B100) ------------------------------------------------------------------------

def region_last_rows(g, j0: int, polygon) -> np.ndarray:
    """(ni,) per column i the last T row of the ORCA part bounded by a line (B100): from row j0 up, the rows whose T
    point lies inside `polygon` ([[lon, lat], ...] degrees, a polygon of the lon-lat plane, straight edges), up to the
    first that does not (so a column's cells are contiguous from the south and the edge is one staircase loop). Every
    column must start inside and leave it below the fold row (j_pole - 1)."""
    from matplotlib.path import Path as MplPath

    lon, lat = g.lon["t"][:, :-2], g.lat["t"][:, :-2]
    ni = lon.shape[1]
    rows = slice(j0, g.j_pole - 1)
    inside = MplPath(np.asarray(polygon, float)).contains_points(
        np.column_stack([lon[rows].ravel(), lat[rows].ravel()])).reshape(-1, ni)
    if not inside[0].all():
        raise ValueError(f"region: {int((~inside[0]).sum())} columns start outside the polygon (row {j0})")
    if inside.all(axis=0).any():
        raise ValueError(f"region: {int(inside.all(axis=0).sum())} columns reach the fold inside the polygon")
    return j0 + np.argmin(inside, axis=0) - 1


def region_cells(oc: OrcaCells, last: np.ndarray) -> OrcaCells:
    """The cells of oc in rows <= last[column] (B100), with their used nodes."""
    keep = oc.cell_ij[:, 0] <= last[oc.cell_ij[:, 1]]
    mesh, used = sub_cells(oc, keep)
    info = dict(oc.info, region_rows=[int(last.min()), int(last.max())], region_cells=int(keep.sum()),
                region_cells_dropped=int((~keep).sum()))
    return OrcaCells(mesh, oc.cell_ij[keep], oc.node_ij[used], oc.rows, oc.ni, oc.folded, info)


def staircase(g, last: np.ndarray) -> np.ndarray:
    """(L, 3) the north edge of the ORCA part with last T rows `last` (B100) as one loop of F points, eastward (the cap
    north of it on the loop's left, as the F row of a row cut): along each column's top edge F(last_i, i - 1) ->
    F(last_i, i), then down or up F column i from row last_i to last_{i+1}."""
    lon, lat = g.lon["f"][:, :-2], g.lat["f"][:, :-2]
    ni = lon.shape[1]
    rr, cc = [], []
    for i in range(ni):
        a, b = int(last[i]), int(last[(i + 1) % ni])
        r = np.arange(a, b + 1) if b >= a else np.arange(a, b - 1, -1)
        rr.append(r)
        cc.append(np.full(len(r), i))
    rr, cc = np.concatenate(rr), np.concatenate(cc)
    return lonlat_to_xyz(lon[rr, cc], lat[rr, cc])


# ---- a refined ORCA part around the north fold (B108) ---------------------------------------------------------------

def top_rows(g, lat_min: float) -> np.ndarray:
    """(ni,) per column i the first T row of the refined part (B108): the T rows from the pole line's row (j_pole; at
    a T pivot the pivot row) down while the T point lies north of lat_min, contiguous from the top (the part around
    the fold is one disc, its edge a staircase); j_pole + 1 where the column's top T point does not (columns that end
    at NEMO's grid poles south of lat_min). Every column must leave above T row 1 (else an error)."""
    lat = g.lat["t"][1:g.j_pole + 1, :-2]
    inside = lat > lat_min
    rev = inside[::-1]
    if rev.all(axis=0).any():
        raise ValueError(f"top_rows: {int(rev.all(axis=0).sum())} columns lie north of {lat_min} down to T row 1")
    return g.j_pole + 1 - np.argmin(rev, axis=0)


def top_cells(oc: OrcaCells, first: np.ndarray) -> OrcaCells:
    """The cells of oc in rows >= first[column] (B108), with their used nodes."""
    keep = oc.cell_ij[:, 0] >= first[oc.cell_ij[:, 1]]
    mesh, used = sub_cells(oc, keep)
    info = dict(oc.info, top_rows=[int(first.min()), int(first.max())], top_cells=int(keep.sum()),
                top_cells_dropped=int((~keep).sum()), top_empty_columns=int((first > oc.rows[1]).sum()))
    return OrcaCells(mesh, oc.cell_ij[keep], oc.node_ij[used], oc.rows, oc.ni, oc.folded, info)


def edge_loop(mesh: Mesh) -> tuple[np.ndarray, list]:
    """The edge of a disc-like part (B108): (L, 3) its longest boundary loop as F points, oriented with the part on
    its right (the fill outside it on its left), and the other loops' node counts: holes inside the part where the
    generator's cells around a grid pole collapse (`cells`: merged, dropped), left open. An error at a pinch node."""
    from meshom.methods.m14.cut import boundary_loops, half_edges

    a, b, e, tw = half_edges(mesh)
    bnd = tw < 0
    if np.bincount(a[bnd], minlength=mesh.n_nodes).max() > 1:
        raise ValueError("edge_loop: the part's boundary has a pinch node")
    loops = sorted(boundary_loops(mesh, (a, b, e, tw)), key=len, reverse=True)
    return mesh.points[loops[0][::-1]], [int(len(lp)) for lp in loops[1:]]


def nesting(gc, gf) -> dict:
    """B108: the fine grid gf is the coarse grid gc refined k = q_f / q_c times on the same map when every k-th F line of
    gf is an F line of gc: F_f(k r + b_j, k c + b_i) = F_c(r, c). The offsets from the lattices (I_F = i_offset +
    (c + 1/2) / q, J_F = j_offset + (r + 1/2) / q), then checked on the points of the rows both grids have: (k, b_i,
    b_j, max distance m, compared points). An error when the lattices do not nest."""
    from fractions import Fraction

    if gc.params != gf.params:
        raise ValueError("nesting: the two grids are not one map (different generator parameters)")
    k = Fraction(gf.q, gc.q)
    if k.denominator != 1 or k <= 1:
        raise ValueError(f"nesting: q {gf.q} is not a multiple > 1 of q {gc.q}")
    fr = lambda x: Fraction(x).limit_denominator(1000)          # noqa: E731
    ic, jc = fr(gc.info["i_offset"]) + Fraction(1, 2 * gc.q), fr(gc.info["j_offset"]) + Fraction(1, 2 * gc.q)
    i_f, j_f = fr(gf.info["i_offset"]) + Fraction(1, 2 * gf.q), fr(gf.info["j_offset"]) + Fraction(1, 2 * gf.q)
    bi, bj = (ic - i_f) * gf.q, (jc - j_f) * gf.q
    if bi.denominator != 1 or bj.denominator != 1:
        raise ValueError(f"nesting: the F lines do not nest (offsets {bi}, {bj} fine steps)")
    k, bi, bj = int(k), int(bi), int(bj)
    lc, pc = gc.lon["f"][:, :-2], gc.lat["f"][:, :-2]
    lf, pf = gf.lon["f"][:, :-2], gf.lat["f"][:, :-2]
    nc = lc.shape[1]
    r = np.arange(lc.shape[0] - 2)                               # the fold rows aside
    r = r[(k * r + bj >= 0) & (k * r + bj < lf.shape[0] - 2)]
    c = np.arange(nc)
    R, C = np.meshgrid(r, c, indexing="ij")
    Rf, Cf = k * R + bj, (k * C + bi) % lf.shape[1]
    d = _dist_km(lc[R, C], pc[R, C], lf[Rf, Cf], pf[Rf, Cf]) * 1e3
    return {"k": k, "col_offset": bi, "row_offset": bj, "max_m": float(d.max()) if d.size else 0.0,
            "p99_m": float(np.percentile(d, 99)) if d.size else 0.0, "points": int(d.size)}


# ---- NEMO's mask ------------------------------------------------------------------------------------------------

def _dist_km(lon1, lat1, lon2, lat2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_M / 1e3 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def transfer_values(g, rl: np.ndarray, rp: np.ndarray, values: np.ndarray, row_offset: int,
                    col_offset: int) -> tuple[np.ndarray, np.ndarray]:
    """A reference field on the T points of g (unique columns), cell for cell: (values (nj, ni), mapped (nj, ni) bool).
    Our T (j, i) is the reference's (j + row_offset, i + col_offset) (T points rl, rp: lon, lat); mapped where that
    point exists and lies within MATCH_FRAC of our cell's shorter side (the distance to the next T point east or north,
    the smaller) of ours; elsewhere values are 0."""
    lon, lat = g.lon["t"][:, :-2], g.lat["t"][:, :-2]
    nj, ni = lat.shape
    rm = np.asarray(values)
    out = np.zeros((nj, ni), rm.dtype)
    mapped = np.zeros((nj, ni), bool)
    jr = np.arange(nj) + row_offset
    ir = np.arange(ni) + col_offset
    okj = (jr >= 0) & (jr < rl.shape[0])
    oki = (ir >= 0) & (ir < rl.shape[1])
    J, I = np.meshgrid(np.flatnonzero(okj), np.flatnonzero(oki), indexing="ij")
    d = _dist_km(lon[J, I], lat[J, I], rl[J + row_offset, I + col_offset], rp[J + row_offset, I + col_offset])
    Jn = np.minimum(J + 1, nj - 1)
    Jn = np.where(Jn == J, J - 1, Jn)
    dz = _dist_km(lon[J, I], lat[J, I], lon[J, (I + 1) % ni], lat[J, (I + 1) % ni])
    dm = _dist_km(lon[J, I], lat[J, I], lon[Jn, I], lat[Jn, I])
    hit = d <= MATCH_FRAC * np.minimum(dz, dm)
    mapped[J[hit], I[hit]] = True
    out[J[hit], I[hit]] = rm[J[hit] + row_offset, I[hit] + col_offset]
    return out, mapped


def transfer_mask(g, ref: dict, row_offset: int, col_offset: int, mask_key: str) -> tuple[np.ndarray, np.ndarray]:
    """NEMO's water flag for every T point of g (unique columns): (wet (nj, ni) bool, mapped (nj, ni) bool), by
    `transfer_values` of the reference's mask array `mask_key` (water > 0)."""
    v, mapped = transfer_values(g, ref["glamt"], ref["gphit"], np.asarray(ref[mask_key]), row_offset, col_offset)
    return v > 0, mapped


def depth_mask(g, gebco, variable: str = "elevation", min_depth_m: float = MIN_DEPTH_M,
               rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The depth rule (B83) on the T cells of g: (wet (nj, ni), mean depth (nj, ni) m). A cell's box on the GEBCO grid
    is the lon-lat box around its T point with half-widths half the cell's zonal and meridional extents (from its F
    corners); its mean depth counts land as 0 m. rows: only these T rows (others: False / nan)."""
    import netCDF4

    from meshom.bathymetry import _band_sums, _open_grid

    lon, lat = g.lon["t"][:, :-2], g.lat["t"][:, :-2]
    nj, ni = lat.shape
    fx, fy = g.lon["f"][:, :-2], g.lat["f"][:, :-2]
    rows = np.arange(1, nj) if rows is None else np.asarray(rows)
    rows = rows[(rows >= 1) & (rows < nj)]
    J, I = np.meshgrid(rows, np.arange(ni), indexing="ij")
    J, I = J.ravel(), I.ravel()
    Im = (I - 1) % ni
    corners = [(J - 1, Im), (J - 1, I), (J, I), (J, Im)]
    cx = lonlat_to_xyz(np.stack([fx[a, b] for a, b in corners], 1).ravel(),
                       np.stack([fy[a, b] for a, b in corners], 1).ravel()).reshape(-1, 4, 3)
    dx = 0.5 * (np.linalg.norm(cx[:, 1] - cx[:, 0], axis=1) + np.linalg.norm(cx[:, 2] - cx[:, 3], axis=1))
    dy = 0.5 * (np.linalg.norm(cx[:, 3] - cx[:, 0], axis=1) + np.linalg.norm(cx[:, 2] - cx[:, 1], axis=1))
    with netCDF4.Dataset(gebco) as ds:
        var, glat, glon = _open_grid(ds, variable)
        ny, nx = len(glat), len(glon)
        dlat, dlon = (glat[-1] - glat[0]) / (ny - 1), (glon[-1] - glon[0]) / (nx - 1)
        la, lo = lat[J, I], lon[J, I]
        ci = np.clip(np.rint((la - glat[0]) / dlat).astype(np.int64), 0, ny - 1)
        cj = np.rint((lo - glon[0]) / dlon).astype(np.int64) % nx
        hi = np.floor(0.5 * dy / np.radians(abs(dlat))).astype(np.int64)
        coslat = np.maximum(np.cos(np.radians(la)), 1e-6)
        hj = np.minimum(np.floor(0.5 * dx / (np.radians(abs(dlon)) * coslat)), nx // 2 - 1).astype(np.int64)
        box = np.column_stack([ci, cj, hi, hj])
        ssum, _ = _band_sums(var, box, 600)
    n_rows = np.minimum(ci + hi + 1, ny) - np.maximum(ci - hi, 0)
    total = n_rows * (2 * hj + 1)
    depth = np.full((nj, ni), np.nan)
    depth[J, I] = -ssum / np.maximum(total, 1)
    wet = np.zeros((nj, ni), bool)
    wet[J, I] = depth[J, I] >= min_depth_m
    return wet, depth


# ---- the clean-up FESOM3 needs ------------------------------------------------------------------------------------

PINCH_REACH = 30        # B84: a pinch's fan removal must leave its neighbours connected within this many cell steps


class _Adjacency:
    """Edge neighbours of every element (from the half-edges) and the elements around every node."""

    def __init__(self, mesh: Mesh, he: tuple):
        a, b, e, tw = he
        self.T = mesh.n_tris
        self._nbr = {}
        self.a, self.b, self.e, self.tw = a, b, e, tw
        order = np.argsort(a, kind="stable")
        self.he_sorted = order
        self.he_start = np.searchsorted(a[order], np.arange(mesh.n_nodes + 1))

    def fans(self, node: int, keep: np.ndarray) -> list:
        """The fans of kept elements around `node`: kept elements with a corner there, grouped through their kept
        edge neighbours across the edges at `node`."""
        hs = self.he_sorted[self.he_start[node]:self.he_start[node + 1]]      # half-edges leaving node
        els = sorted({int(self.e[h]) for h in hs if keep[self.e[h]]})
        parent = {x: x for x in els}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for h in hs:                                       # edge node -> b: its two sides
            t = self.tw[h]
            if t >= 0 and keep[self.e[h]] and keep[self.e[t]]:
                parent[find(int(self.e[h]))] = find(int(self.e[t]))
        groups = {}
        for x in els:
            groups.setdefault(find(x), []).append(x)
        return list(groups.values())

    def nbr(self, el: int) -> list:
        """The edge neighbours of element el (half-edges are per element in order: 3 per triangle, then 4 per quad)."""
        r = self._nbr.get(el)
        if r is None:
            h0 = 3 * el if el < self.T else 3 * self.T + 4 * (el - self.T)
            n = 3 if el < self.T else 4
            t = self.tw[h0:h0 + n]
            r = [int(self.e[x]) for x in t if x >= 0]
            self._nbr[el] = r
        return r

    def loss(self, drop: list, keep: np.ndarray) -> tuple[int, list]:
        """Cells lost when `drop` is removed: the spur cascade it starts (cells left with fewer than two kept edge
        neighbours, repeatedly), and a split (PINCH_REACH large) when the dropped cells' kept neighbours are no longer
        connected within PINCH_REACH steps. Returns (loss, the cells removed)."""
        k = keep.copy()
        gone = list(drop)
        k[drop] = False
        stack = [n for d in drop for n in self.nbr(d) if k[n]]
        while stack:
            c = stack.pop()
            if k[c] and sum(1 for n in self.nbr(c) if k[n]) < 2:
                k[c] = False
                gone.append(c)
                stack.extend(n for n in self.nbr(c) if k[n])
        rest = sorted({n for d in gone for n in self.nbr(d) if k[n]})
        split = 0
        if len(rest) > 1:
            seen = {rest[0]}
            front = [rest[0]]
            for _ in range(PINCH_REACH):
                nxt = []
                for c in front:
                    for n in self.nbr(c):
                        if k[n] and n not in seen:
                            seen.add(n)
                            nxt.append(n)
                front = nxt
                if not front or all(r in seen for r in rest):
                    break
            if not all(r in seen for r in rest):
                split = PINCH_REACH
        return len(gone) + split, gone


def fesom_cleanup(mesh: Mesh, keep: np.ndarray, keep_seas: bool = True) -> tuple[Mesh, np.ndarray, dict]:
    """B84 on the elements `keep` of `mesh`, until nothing changes: spurs (fewer than two kept edge neighbours:
    fesom_ini stops on them) removed; at a pinch node (kept elements meeting there only at a
    corner: meshom's boundary loops and FESOM's edge lists need one fan per node) one fan removed at a time, the one
    whose removal loses the fewest cells (`_Adjacency.loss`: itself, the spur cascade it starts, and a channel cut
    when its neighbours no longer connect within PINCH_REACH steps), ties to the smaller fan; NEMO does not connect
    diagonal cells either. Separate water bodies are kept (keep_seas) or all but the largest removed. Returns (mesh of
    the kept elements, the kept element mask on `mesh`, info: counts and where (lon, lat) of each removed element)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from meshom.geo import xyz_to_lonlat
    from meshom.methods.m14.cut import _spurs, half_edges

    keep0 = keep.copy()
    keep = keep.copy()
    he = half_edges(mesh)
    a, b, e, tw = he
    spurs = np.zeros(len(keep), bool)
    pinch = np.zeros(len(keep), bool)
    adj = None
    rounds = 0
    while True:
        rounds += 1
        k0 = int(keep.sum())
        while True:
            sp = _spurs(he, keep)
            if not sp.any():
                break
            keep &= ~sp
            spurs |= sp
        hk = keep[e]
        inner = hk & (tw >= 0) & keep[e[np.maximum(tw, 0)]]
        cand = np.flatnonzero(np.bincount(a[hk & ~inner], minlength=mesh.n_nodes) > 1)
        if len(cand):
            adj = adj or _Adjacency(mesh, he)
            for nd in cand:
                while True:
                    fans = adj.fans(int(nd), keep)
                    if len(fans) < 2:
                        break
                    best = min(((adj.loss(f, keep), len(f), i) for i, f in enumerate(fans)),
                               key=lambda t: (t[0][0], t[1], t[2]))
                    gone = best[0][1]
                    keep[gone] = False
                    for g_ in gone:
                        (pinch if g_ in fans[best[2]] else spurs)[g_] = True
        if int(keep.sum()) == k0:
            break
    M = len(keep)
    inner = keep[e] & (tw >= 0) & keep[e[np.maximum(tw, 0)]]
    g = coo_matrix((np.ones(int(inner.sum())), (e[inner], e[tw[inner]])), shape=(M, M))
    _, lab = connected_components(g, directed=False)
    lab = np.where(keep, lab, -1)
    sizes = np.bincount(lab[keep])
    main = int(np.argmax(sizes))
    seas = keep & (lab != main)
    if not keep_seas:
        keep &= ~seas
    T = mesh.n_tris
    tris, quads = mesh.tris[keep[:T]], mesh.quads[keep[T:]]
    used = np.unique(np.r_[tris.ravel(), quads.ravel()])
    ren = np.full(mesh.n_nodes, -1, np.int64)
    ren[used] = np.arange(len(used))
    out = Mesh(mesh.points[used], ren[tris], ren[quads], np.zeros(len(used), np.int8))
    allc = np.r_[mesh.points[mesh.tris].mean(1), mesh.points[mesh.quads].mean(1)] if M else np.zeros((0, 3))

    def where(mask):
        if not mask.any():
            return []
        lo, la = xyz_to_lonlat(allc[mask])
        return [[round(float(x), 2), round(float(y), 2)] for x, y in zip(lo, la)]

    info = {"water_cells_in": int(keep0.sum()), "rounds": rounds,
            "spurs_removed": int(spurs.sum()), "spurs_at": where(spurs),
            "pinch_cells_removed": int(pinch.sum()), "pinch_at": where(pinch),
            "water_bodies": int(len(np.flatnonzero(sizes))), "main_body_cells": int(sizes[main]),
            "other_bodies_cells": int(seas.sum()), "other_bodies_kept": bool(keep_seas),
            "other_bodies": _bodies(lab, seas, allc), "water_cells_out": int(keep.sum())}
    return out, keep, info


def _bodies(lab: np.ndarray, seas: np.ndarray, cent: np.ndarray) -> list:
    from meshom.geo import xyz_to_lonlat

    out = []
    for c in np.unique(lab[seas]):
        sel = lab == c
        lo, la = xyz_to_lonlat(cent[sel].mean(0)[None] / np.linalg.norm(cent[sel].mean(0)))
        out.append({"cells": int(sel.sum()), "at": [round(float(lo[0]), 2), round(float(la[0]), 2)]})
    return sorted(out, key=lambda d: -d["cells"])
