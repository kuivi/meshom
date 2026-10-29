"""m11: local operators to h (MMG / Omega_h pattern; independent-set passes), then recombination.

coarse mesh at factor * h (closed sphere: background_mesh; with a coast: a subset of the resampled coast nodes, corners
kept, interior Poisson points at factor * h) -> passes of split (> hi) / collapse (< lo) / Delaunay flips / size
smoothing (meshom.smooth, kind "size") / flips, until a pass splits and collapses at most stop_frac * N edges ->
polish (smoothing + valence flips, Botsch & Kobbelt 2004) -> recombine (greedy; price 1/3 merges a pair only when the
quad's largest angle stays below 150 deg, the hard gate) -> size smoothing of the mixed mesh. Lengths are metric
lengths (units of h, ops.metric_length). The coast nodes of the result are exactly resample_loops(domain, h): coast
chords of the coarse mesh are split by the skipped coast nodes. Operators: meshom/methods/m11/ops.py.
"""
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from meshom.delaunay import background_mesh, poisson_points, sphere_delaunay
from meshom.domain import Domain, in_water, loop_edges, resample_loops
from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh
from meshom.methods import register
from meshom.methods.m11.ops import Coast, collapse_edges, compact, edge_ratios, flip_edges, split_edges
from meshom.size import SizeField, scaled

DEFAULTS = dict(factor=4.0, lo=0.7, hi=1.4, max_angle_deg=150.0, max_passes=60, stop_frac=1e-3, smooth_iters=3,
                flips="delaunay", polish=10, polish_flips="valence", price=1.0 / 3.0, post_smooth=10, verbose=False)
BINS = (0.0, 0.5, 0.7, 1.0, 1.4, 2.0, np.inf)


def _turn(lp: np.ndarray) -> np.ndarray:
    """Turning angle (rad) at each vertex of a closed loop."""
    a = lp - np.roll(lp, 1, axis=0)
    b = np.roll(lp, -1, axis=0) - lp
    a -= np.einsum("ij,ij->i", a, lp)[:, None] * lp
    b -= np.einsum("ij,ij->i", b, lp)[:, None] * lp
    c = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.arccos(np.clip(c, -1.0, 1.0))


def _coarse_ids(lp: np.ndarray, factor: float, min_loop: int = 8, corner_deg: float = 30.0) -> np.ndarray:
    """Every ~factor-th node of a resampled loop, corners kept, at least min_loop nodes on a loop without corners."""
    k = len(lp)
    corners = np.flatnonzero(_turn(lp) > np.radians(corner_deg))
    if len(corners) == 0:
        m = min(k, max(min_loop, int(round(k / factor))))
        return np.unique(np.round(np.linspace(0, k, m + 1)[:-1]).astype(np.int64))
    ids = []
    for a, b in zip(corners, np.r_[corners[1:], corners[0] + k]):
        m = max(1, int(round((b - a) / factor)))
        ids.append(a + np.round(np.linspace(0, b - a, m + 1)[:-1]).astype(np.int64))
    return np.concatenate(ids) % k


def coarse_mesh(domain: Domain, h: SizeField, factor: float) -> tuple[Mesh, Coast | None]:
    """Start mesh at factor * h. With a coast, nodes 0..B-1 are all resampled coast nodes at h (node_flag 1), only a
    subset of them in use; interior points closer than factor * h / 2 to the coast are dropped."""
    if not domain.loops:
        return background_mesh(domain, h, factor=factor), None
    fine = resample_loops(domain, h)
    coast = Coast.from_loops(fine)
    fine_all = np.vstack(fine)
    sel = [coast.start[i] + _coarse_ids(lp, factor) for i, lp in enumerate(fine)]
    pts = poisson_points(domain, scaled(h, factor), [fine_all[s] for s in sel])
    inner = pts[sum(len(s) for s in sel):]
    d, _ = cKDTree(fine_all).query(inner)
    inner = inner[in_water(Domain(fine, domain.name), inner) & (R_EARTH_M * d >= 0.5 * factor * h(inner))]
    ids = np.concatenate(sel)
    local = np.vstack([fine_all[ids], inner])
    e = loop_edges([fine_all[s] for s in sel])
    tris = sphere_delaunay(local, e)
    gid = np.r_[ids, coast.n + np.arange(len(inner), dtype=np.int64)]
    flag = np.r_[np.ones(coast.n, dtype=np.int8), np.zeros(len(inner), dtype=np.int8)]
    return Mesh(np.vstack([fine_all, inner]), gid[tris], np.zeros((0, 4), dtype=np.int64), flag), coast


def _record(stage, mesh: Mesh, h, lo, hi, t, **counts) -> dict:
    r = edge_ratios(mesh, h)
    hist = np.histogram(r, bins=BINS)[0] / max(len(r), 1)
    return dict(stage=stage, nodes=int(len(np.unique(mesh.tris))), tris=mesh.n_tris, **counts,
                in_window=float(np.mean((r >= lo) & (r <= hi))), p1=float(np.percentile(r, 1)),
                p50=float(np.median(r)), p99=float(np.percentile(r, 99)), hist=[round(float(x), 4) for x in hist],
                time_s=round(t, 3))


def remesh(domain: Domain, h: SizeField, opts: dict | None = None) -> tuple[Mesh, list[dict]]:
    """Triangle mesh of the water with edge lengths in [lo, hi] * h (as far as the operators get), and the
    per-pass log (counts, edge-length histogram in units of h over BINS, time)."""
    from meshom.smooth import smooth

    o = {**DEFAULTS, **(opts or {})}
    lo, hi = o["lo"], o["hi"]
    t0 = time.perf_counter()
    mesh, coast = coarse_mesh(domain, h, o["factor"])
    log = [_record("coarse", mesh, h, lo, hi, time.perf_counter() - t0)]

    def note(rec):
        log.append(rec)
        if o["verbose"]:
            print(rec, file=sys.stderr, flush=True)

    note(log.pop())
    for p in range(o["max_passes"]):
        ts = time.perf_counter()
        mesh, ns = split_edges(mesh, h, hi, coast)
        mesh, nc = collapse_edges(mesh, h, lo, hi, o["max_angle_deg"])
        mesh, nf = flip_edges(mesh, o["flips"])
        mesh = smooth(mesh, h, mesh.node_flag != 0, iters=o["smooth_iters"], kind="size")
        mesh, nf2 = flip_edges(mesh, o["flips"])
        note(_record(f"pass {p + 1}", mesh, h, lo, hi, time.perf_counter() - ts, splits=ns, collapses=nc,
                     flips=nf + nf2))
        if ns + nc <= o["stop_frac"] * log[-1]["nodes"]:
            break
    ts = time.perf_counter()
    nf = 0
    for _ in range(o["polish"]):
        mesh = smooth(mesh, h, mesh.node_flag != 0, iters=o["smooth_iters"], kind="size")
        mesh, k = flip_edges(mesh, o["polish_flips"])
        nf += k
    note(_record("polish", mesh, h, lo, hi, time.perf_counter() - ts, flips=nf))
    out = compact(mesh, keep_fixed=False)
    if coast is not None and out.n_nodes - int((out.node_flag == 0).sum()) < coast.n:
        print(f"m11: {coast.n - int((out.node_flag != 0).sum())} coast nodes not inserted", file=sys.stderr)
    return out, log


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    from meshom.recombine import recombine

    from meshom.smooth import smooth

    o = {**DEFAULTS, **opts}
    mesh, _ = remesh(domain, h, o)
    mesh = recombine(mesh, o["price"])
    return smooth(mesh, h, mesh.node_flag != 0, iters=o["post_smooth"], kind="size") if o["post_smooth"] else mesh


@register("m11_local_ops_tri", overrides={})          # another entry path: no recombination
def generate_tri(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    return remesh(domain, h, opts)[0]
