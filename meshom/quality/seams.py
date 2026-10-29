"""Seam measures: defects, size and shape in a corridor between two lattices
(two blocks, a block and a coast, the two banks of a channel), against the minimum an ideal mesh needs there.

Corridor geometry. A corridor is the rectangle around a great-circle arc p -> q (its centreline): s = distance along
that great circle from p (m), d = signed distance from it (m; d < 0 is side "a", on the right of p -> q seen from
outside the sphere; d > 0 side "b"). Measured part: trim_a <= s <= L - trim_b, |d| <= half_width. Elements count by
their centroid, nodes and edge midpoints by position; boundary nodes and edges are left out (coast rows are the coast
gates' business, G17-G19).

Reference: the minimum dislocation count. Lattices of steps (a1, a2) on side a and (b1, b2) on side b, their first axis
at phi_a, phi_b (deg, mod 90) to the centreline, meet along a measured length L. A closed circuit across the corridor
picks up the Burgers vector (Frank-Bilby) B = L (cos phi_a / a1 - cos phi_b / b1, sin phi_b / b2 - sin phi_a / a2) in
lattice steps (side b's axes renamed by the quarter turn that brings phi_b nearest to phi_a). A quad-mesh dislocation
(a valence-3/5 pair, or a triangle and a valence-5 node: two unit charges) carries one step along one axis, so at
least D = |B_x| + |B_y| dislocations sit in the corridor and its gross charge is >= 2 D. Same orientation: D = |L/a -
L/b| = |N - M| (the row-count mismatch); symmetric tilt (phi = -/+ theta/2, a = b = h): D = 2 (L/h) sin(theta/2)
(Read-Shockley's wall). D is the minimum at exact size: a mesh may trade dislocations for size error (a side stretched
by e^eps absorbs a mismatch of eps; a conformal bend absorbs rotation at a size gradient of the same order).

Measures per corridor (`seam_block`): interior nodes, n3 / n5 / other, triangles, irregular nodes (valence != 4; != 6
where every element at the node is a triangle, as meshom.repair), B53 defects (|360 - 90 q - 60 t| >= 60 deg), gross
charge (sum |deg - regular| + triangles of a quad-dominant mesh) and net (quad-dominant only), all per 10 cells of
corridor length; the gross excess over 2 D (D of the case's design and D of the lattices the mesh realised on the two
sides); |ln s/h| (interior edges), quad max angle and element size ratio (G7) in the corridor; the share of the
irregular nodes within 1 h of the centreline; for channels (both sides coasts) the cells across, bank to bank.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from meshom.geo import R_EARTH_M, arc_length
from meshom.mesh import Mesh, build_topology
from meshom.quality.kernel import element_geometry

SIDE_BAND_CELLS = 6.0      # the realised lattice of a side is read from quads 0-6 local cells beyond the corridor edge
CENTRE_CELLS = 1.0         # "on the centreline": |d| <= this many local h

CENTRE_MIN_IRREGULAR = 5   # the centre-share gate applies from this many irregular nodes in the corridor

# Proposed seam gates; not in DEFAULT_GATES, evaluated into each seam block only.
SEAM_GATES = {
    "seam_excess_per10_max": 2.0,     # gross charge above 2 D (realised lattices) per 10 corridor cells
    "seam_ln_sh_p99_max": 0.18,       # the corridor's interior edges (P22's seam-band level; G9's 0.15 stays global)
    "seam_quad_p99_9_max": 135.0,     # G2 on the corridor's quads
    "seam_ratio_p99_max": 1.15,       # G7 on the corridor's interior edges
    "seam_centre_share_min": 0.5,     # a wall on the mid-line, not a scatter (>= CENTRE_MIN_IRREGULAR nodes)
    "seam_lake_excess_per100_max": 1.0,   # the case total: gross above the design minimum per 100 interior nodes
}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@dataclass(frozen=True)
class Corridor:
    """A seam corridor around the great-circle arc p -> q (unit vectors as tuples). phi_*_deg: the lattice of each
    side relative to the centreline in the case's design (deg, mod 90); coast_*: that side is a coast (its lattice
    is the coast rows: phi 0, step h). banks: both sides are coasts and the cells across are counted."""
    name: str
    p: tuple
    q: tuple
    half_width_m: float
    phi_a_deg: float = 0.0
    phi_b_deg: float = 0.0
    trim_m: tuple = (0.0, 0.0)
    coast_a: bool = False
    coast_b: bool = False
    banks: bool = False

    @property
    def pole(self) -> np.ndarray:
        return _unit(np.cross(np.asarray(self.p), np.asarray(self.q)))

    @property
    def length_m(self) -> float:
        return float(R_EARTH_M * arc_length(np.asarray(self.p), np.asarray(self.q)))

    def coords(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(s, d) in metres: distance along the centreline great circle from p, signed distance from it (> 0 left)."""
        p, n = np.asarray(self.p), self.pole
        t = np.cross(n, p)
        s = R_EARTH_M * np.arctan2(xyz @ t, xyz @ p)
        d = R_EARTH_M * np.arcsin(np.clip(xyz @ n, -1.0, 1.0))
        return s, d

    def point(self, s: float, d: float) -> np.ndarray:
        p, n = np.asarray(self.p), self.pole
        x = np.cos(s / R_EARTH_M) * p + np.sin(s / R_EARTH_M) * np.cross(n, p)
        return np.cos(d / R_EARTH_M) * x + np.sin(d / R_EARTH_M) * n

    def span(self) -> tuple[float, float]:
        return self.trim_m[0], self.length_m - self.trim_m[1]

    def inside(self, xyz: np.ndarray) -> np.ndarray:
        s, d = self.coords(xyz)
        s0, s1 = self.span()
        return (s >= s0) & (s <= s1) & (np.abs(d) <= self.half_width_m)


@dataclass(frozen=True)
class Core:
    """A junction core: the disc of radius radius_m about centre; net_ref = the net charge it must hold (sign
    free), so its gross charge is >= |net_ref|."""
    name: str
    centre: tuple
    radius_m: float
    net_ref: int = 0


@dataclass(frozen=True)
class SeamSpec:
    """The corridors and cores of a case; centre / radius_m: the cap over which the totals are taken (the case's
    lake; None: the whole mesh)."""
    corridors: tuple
    cores: tuple = field(default_factory=tuple)
    centre: tuple | None = None
    radius_m: float | None = None


def dislocations(L_m: float, phi_a: float, steps_a: tuple, phi_b: float, steps_b: tuple) -> float:
    """Minimum dislocation count D = |B_x| + |B_y| (module docstring) for lattices at phi_a, phi_b (deg) to the
    centreline with steps (along axis 1, along axis 2) in metres, over a corridor length L_m; the smallest D over the
    quarter-turn renamings of side b's axes."""
    best = np.inf
    ra = np.radians(phi_a)
    for k in range(-2, 3):
        rb = np.radians(phi_b + 90.0 * k)
        sb = steps_b if k % 2 == 0 else steps_b[::-1]
        bx = L_m * (np.cos(ra) / steps_a[0] - np.cos(rb) / sb[0])
        by = L_m * (np.sin(rb) / sb[1] - np.sin(ra) / steps_a[1])
        best = min(best, float(abs(bx) + abs(by)))
    return best


def _side_lattice(mesh: Mesh, cen: np.ndarray, hc: np.ndarray, c: Corridor, sign: int) -> dict | None:
    """Realised lattice of one side: quads whose centroid lies 0-SIDE_BAND_CELLS local cells beyond the corridor edge
    (within the measured span); 4-fold mean edge angle to the centreline (deg, (-45, 45]), its coherence, and the mean
    step along each axis family (m)."""
    T = mesh.n_tris
    s, d = c.coords(cen[T:])
    s0, s1 = c.span()
    dd = sign * d - c.half_width_m
    m = (s >= s0) & (s <= s1) & (dd >= 0.0) & (dd <= SIDE_BAND_CELLS * hc[T:])
    if m.sum() < 8:
        return None
    qd = mesh.quads[m]
    x = cen[T:][m]
    n = c.pole
    u = _unit(np.cross(n, x))                         # along the centreline (parallel circle through x)
    v = np.cross(x, u)
    P = mesh.points
    e = P[np.roll(qd, -1, axis=1)] - P[qd]            # (M,4,3)
    et = e - np.einsum("mkd,md->mk", e, x)[..., None] * x[:, None, :]
    ang = np.arctan2(np.einsum("mkd,md->mk", et, v), np.einsum("mkd,md->mk", et, u))
    ln = np.linalg.norm(e, axis=2) * R_EARTH_M
    z = np.sum(ln * np.exp(4j * ang))
    phi = float(np.degrees(np.angle(z) / 4.0))
    coh = float(np.abs(z) / ln.sum())
    rel = np.mod(np.degrees(ang) - phi + 45.0, 180.0) - 45.0     # axis 1 within (-45, 45], axis 2 within (45, 135]
    ax1 = rel <= 45.0
    st1 = float(ln[ax1].mean()) if ax1.any() else float(ln.mean())
    st2 = float(ln[~ax1].mean()) if (~ax1).any() else float(ln.mean())
    return {"phi_deg": round(phi, 2), "coherence": round(coh, 3), "step1_m": st1, "step2_m": st2,
            "step1_over_h": round(st1 / float(np.mean(hc[T:][m])), 4),
            "step2_over_h": round(st2 / float(np.mean(hc[T:][m])), 4), "quads": int(m.sum())}


def _p(x: np.ndarray, q: float):
    return round(float(np.percentile(x, q)), 4) if len(x) else None


def _mx(x: np.ndarray):
    return round(float(x.max()), 4) if len(x) else None


class _MeshData:
    """Topology and per-element / per-edge / per-node quantities shared by all corridors of one mesh."""

    def __init__(self, mesh: Mesh, h):
        self.mesh = mesh
        n, T = mesh.n_nodes, mesh.n_tris
        self.topo = topo = build_topology(mesh)
        E, ee = topo.edges, topo.edge_elems
        self.deg = np.bincount(E.ravel(), minlength=n)
        self.bnd = topo.boundary_node
        qn = np.bincount(mesh.quads.ravel(), minlength=n)
        tn = np.bincount(mesh.tris.ravel(), minlength=n)
        self.reg = np.where((qn == 0) & (tn > 0), 6, 4)
        self.quad_dom = mesh.n_quads >= mesh.n_tris
        zone = np.zeros(T, dtype=bool) if mesh.tri_zone is None else np.asarray(mesh.tri_zone, dtype=bool)
        self.tri_free = ~zone                             # triangles that carry a charge (outside a declared zone)
        in_zone = np.zeros(n, dtype=bool)
        in_zone[mesh.tris[zone].ravel()] = True
        dev = np.abs(360 - 90 * qn - 60 * tn)
        self.mixdef = (dev >= 60) & ~self.bnd
        # charge per node: |deg - regular| (meshom.repair); at nodes of a declared triangle zone the angle sum
        # 90 q + 60 t against 360 deg in units of 60 deg (B53), so a block-to-triangle join of 2 quads + 3 triangles
        # is regular
        self.node_charge = np.where(in_zone, np.rint(dev / 60.0), np.abs(self.deg - self.reg)).astype(np.int64)
        self.node_charge[self.bnd] = 0
        self.irr = self.node_charge > 0
        self.zone_node = in_zone
        g = element_geometry(mesh)
        self.cen = g.centroid
        self.hc = h(g.centroid)
        self.maxang = np.r_[np.degrees(g.tri_angles).max(axis=1, initial=0.0),
                            np.degrees(g.quad_angles).max(axis=1, initial=0.0)]
        self.is_tri = np.arange(mesh.n_elems) < T
        mean_edge = np.r_[g.tri_edges.mean(axis=1), g.quad_edges.mean(axis=1)]
        p, q = mesh.points[E[:, 0]], mesh.points[E[:, 1]]
        self.mid = _unit(p + q)
        self.lnsh = np.abs(np.log(R_EARTH_M * arc_length(p, q) / h(self.mid)))
        self.inner = ee[:, 1] >= 0
        e1 = np.where(self.inner, ee[:, 1], ee[:, 0])
        self.ratio = (np.maximum(mean_edge[ee[:, 0]], mean_edge[e1])
                      / np.minimum(mean_edge[ee[:, 0]], mean_edge[e1]))
        self.hn = h(mesh.points)

    def charge(self, nodes: np.ndarray, tris_in: np.ndarray) -> dict:
        """Counts over the interior nodes of mask `nodes` and the triangles of element mask `tris_in` (T,)."""
        nd = nodes & ~self.bnd
        d = self.deg[nd]
        q4 = (self.reg[nd] == 4) & ~self.zone_node[nd]
        t_free = int((tris_in & self.tri_free).sum())
        gross = int(self.node_charge[nd].sum()) + (t_free if self.quad_dom else 0)
        net = int((4 - d[q4]).sum()) + t_free if self.quad_dom and not self.zone_node.any() else None
        n3, n5 = int(((d == 3) & q4).sum()), int(((d == 5) & q4).sum())
        return {"nodes": int(nd.sum()), "n3": n3, "n5": n5, "n_other": int(self.irr[nd].sum()) - n3 - n5,
                "T": int(tris_in.sum()), "irregular": int(self.irr[nd].sum()), "defects_mix": int(self.mixdef[nd].sum()),
                "gross": gross, "net": net}


def _cells_across(md: _MeshData, c: Corridor) -> dict | None:
    """Bank to bank: edge hops from each boundary node of bank b in the span to the nearest boundary node of bank a."""
    P = md.mesh.points
    s, d = c.coords(P)
    s0, s1 = c.span()
    bank = md.bnd & (s >= s0) & (s <= s1)
    tol = 0.5 * md.hn
    a = np.flatnonzero(bank & (np.abs(d + c.half_width_m) <= tol))
    b = np.flatnonzero(bank & (np.abs(d - c.half_width_m) <= tol))
    if len(a) == 0 or len(b) == 0:
        return None
    E = md.topo.edges
    n = md.mesh.n_nodes
    g = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    dist = dijkstra(g, directed=False, indices=a, unweighted=True, min_only=True)
    x = dist[b]
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    return {"min": int(x.min()), "median": float(np.median(x)), "max": int(x.max()), "bank_nodes": int(len(b))}


def corridor_block(md: _MeshData, h, c: Corridor) -> dict:
    """The seam measures of one corridor (module docstring)."""
    mesh = md.mesh
    s0, s1 = c.span()
    L = s1 - s0
    sm = 0.5 * (s0 + s1)
    h_a = float(h(c.point(sm, -c.half_width_m)[None, :])[0])
    h_b = float(h(c.point(sm, c.half_width_m)[None, :])[0])
    hcl = float(np.sqrt(h_a * h_b))
    cells = L / hcl
    el = c.inside(md.cen)
    nd = c.inside(mesh.points)
    ed = c.inside(md.mid) & md.inner
    ch = md.charge(nd, el[: mesh.n_tris])
    qd = el & ~md.is_tri
    # references: the case's design and the realised side lattices
    d_case = dislocations(L, c.phi_a_deg, (h_a, h_a), c.phi_b_deg, (h_b, h_b))
    sides = {}
    for name, sign, coast, phi0, hh in (("a", -1, c.coast_a, c.phi_a_deg, h_a), ("b", 1, c.coast_b, c.phi_b_deg, h_b)):
        sides[name] = None if coast else _side_lattice(mesh, md.cen, md.hc, c, sign)
    real = []
    for name, coast, phi0, hh in (("a", c.coast_a, c.phi_a_deg, h_a), ("b", c.coast_b, c.phi_b_deg, h_b)):
        sl = sides[name]
        real.append((0.0, (hh, hh)) if coast or sl is None else (sl["phi_deg"], (sl["step1_m"], sl["step2_m"])))
    d_real = dislocations(L, real[0][0], real[0][1], real[1][0], real[1][1])
    theta_real = None
    if not (c.coast_a and c.coast_b):
        th = real[1][0] - real[0][0]
        theta_real = round(float(np.mod(th + 45.0, 90.0) - 45.0), 2)
    # defects on the centreline
    _, dn = c.coords(mesh.points)
    irr_in = nd & md.irr
    centre = irr_in & (np.abs(dn) <= CENTRE_CELLS * md.hn)
    per10 = 10.0 / cells if cells > 0 else np.nan
    out = {"name": c.name, "length_cells": round(cells, 2), "width_cells": round(2.0 * c.half_width_m / hcl, 2),
           "h_a_m": round(h_a, 1), "h_b_m": round(h_b, 1), "elems": int(el.sum()), **ch,
           "irregular_per10": round(ch["irregular"] * per10, 3), "defects_mix_per10": round(ch["defects_mix"] * per10, 3),
           "gross_per10": round(ch["gross"] * per10, 3),
           "disl_min_case": round(d_case, 2), "disl_min_real": round(d_real, 2),
           "theta_case_deg": None if (c.coast_a and c.coast_b) else round(c.phi_b_deg - c.phi_a_deg, 2),
           "theta_real_deg": theta_real,
           "excess_per10": round((ch["gross"] - 2.0 * d_real) * per10, 3),
           "excess_case_per10": round((ch["gross"] - 2.0 * d_case) * per10, 3),
           "gross_over_min": round(ch["gross"] / (2.0 * d_case), 3) if d_case >= 1.0 else None,
           "centre_share": round(float(centre.sum() / irr_in.sum()), 3) if irr_in.any() else None,
           "ln_sh_p99": _p(md.lnsh[ed], 99), "ln_sh_max": _mx(md.lnsh[ed]),
           "quad_p99_9_deg": _p(md.maxang[qd], 99.9), "quad_max_deg": _mx(md.maxang[qd]),
           "tri_max_deg": _mx(md.maxang[el & md.is_tri]),
           "ratio_p99": _p(md.ratio[ed], 99), "ratio_max": _mx(md.ratio[ed]),
           "sides": {k: (None if v is None else {kk: vv for kk, vv in v.items() if not kk.endswith("_m")})
                     for k, v in sides.items()}}
    if c.banks:
        out["cells_across"] = _cells_across(md, c)
    return out


def tolerance_seam_gates(tol: float | None) -> dict:
    """SEAM_GATES with the size gate at max(its 0.18, tol) for a mesh built at the size tolerance tol (G33, as G38
    on the whole mesh; methods.effective_tolerance); SEAM_GATES when tol is None."""
    if not tol:
        return SEAM_GATES
    return {**SEAM_GATES, "seam_ln_sh_p99_max": max(SEAM_GATES["seam_ln_sh_p99_max"], float(tol))}


def seam_gates(block: dict, gates: dict | None = None) -> dict:
    """Failed seam gates of a corridor block: {gate: value}; a gate without a value passes (as G20)."""
    g = SEAM_GATES if gates is None else gates
    checks = [("seam_excess_per10_max", block.get("excess_per10"), lambda v, t: v <= t),
              ("seam_ln_sh_p99_max", block.get("ln_sh_p99"), lambda v, t: v <= t),
              ("seam_quad_p99_9_max", block.get("quad_p99_9_deg"), lambda v, t: v <= t),
              ("seam_ratio_p99_max", block.get("ratio_p99"), lambda v, t: v <= t),
              ("seam_centre_share_min", block.get("centre_share") if block.get("irregular", 0) >= CENTRE_MIN_IRREGULAR
               else None, lambda v, t: v >= t)]
    return {k: v for k, v, ok in checks if k in g and v is not None and not ok(v, g[k])}


def seam_block(mesh: Mesh, h, spec: SeamSpec, gates: dict | None = None) -> dict:
    """The record's "seam" block: one entry per corridor and per core, the totals over the corridors and the whole
    mesh, the failed seam gates."""
    md = _MeshData(mesh, h)
    cors = [corridor_block(md, h, c) for c in spec.corridors]
    for b in cors:
        b["gates_failed"] = seam_gates(b, gates)
    cores = []
    for k in spec.cores:
        dist = R_EARTH_M * arc_length(mesh.points, np.asarray(k.centre)[None, :])
        cdist = R_EARTH_M * arc_length(md.cen, np.asarray(k.centre)[None, :])
        nd = dist <= k.radius_m
        cores.append({"name": k.name, "net_ref": k.net_ref,
                      **md.charge(nd, (cdist <= k.radius_m)[: mesh.n_tris])})
    if spec.centre is None:
        all_nodes, all_tris = np.ones(mesh.n_nodes, dtype=bool), np.ones(mesh.n_tris, dtype=bool)
    else:
        cc = np.asarray(spec.centre)[None, :]
        all_nodes = R_EARTH_M * arc_length(mesh.points, cc) <= spec.radius_m
        all_tris = (R_EARTH_M * arc_length(md.cen, cc) <= spec.radius_m)[: mesh.n_tris]
    tot = md.charge(all_nodes, all_tris)
    cells = sum(b["length_cells"] for b in cors)
    gross_c = sum(b["gross"] for b in cors)
    d_case = sum(b["disl_min_case"] for b in cors)
    out = {"corridors": cors, "cores": cores,
           "total": {"length_cells": round(cells, 2), "gross": gross_c, "disl_min_case": round(d_case, 2),
                     "disl_min_real": round(sum(b["disl_min_real"] for b in cors), 2),
                     "irregular_per10": round(10.0 * sum(b["irregular"] for b in cors) / cells, 3) if cells else None,
                     "excess_per10": round(10.0 * (gross_c - 2.0 * sum(b["disl_min_real"] for b in cors)) / cells, 3)
                     if cells else None},
           "mesh": {"nodes": tot["nodes"], "T": tot["T"], "gross": tot["gross"], "irregular": tot["irregular"],
                    "defects_mix": tot["defects_mix"],
                    "gross_outside_corridors": tot["gross"] - gross_c - sum(k["gross"] for k in cores),
                    "gross_min_case": round(2.0 * d_case + sum(abs(k.net_ref) for k in spec.cores), 2)}}
    m = out["mesh"]
    m["excess_per100"] = round(100.0 * (m["gross"] - m["gross_min_case"]) / max(m["nodes"], 1), 3)
    g = SEAM_GATES if gates is None else gates
    lake = ({"seam_lake_excess_per100_max": m["excess_per100"]} if "seam_lake_excess_per100_max" in g
            and m["excess_per100"] > g["seam_lake_excess_per100_max"] else {})
    out["gates_failed"] = sorted({k for b in cors for k in b["gates_failed"]} | set(lake))
    return out
