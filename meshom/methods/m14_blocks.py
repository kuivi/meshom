"""m14: curvilinear blocks placed automatically where g / h^2 is nearly flat, then stitched by m02.

1. Fitness map from h alone (m14/fitness.py): background mesh at 1.0 h, the B4 cross field and its cones, rho_c per
   node, distances to cones / coast / complex coast in units of h.
2. Placement (m14/place.py): greedy, largest first, regions grown from the clearest admissible node (>= 3 h from
   cones and coast, 6 h from a complex coast, 6 h from other blocks), kept a closed disc, accepted by the rule:
   the conformal size correction lambda (Laplace lambda = -(K + Laplace ln h), lambda = 0 on the outline) has
   F = (max - min)/2 <= 0.08 and h |grad lambda| <= 0.05 (P1 gradient, max over the background triangles);
   pre-screen |B| <= min(1.3, 0.08 L) during growth. A block whose map overlaps itself is retried on 80 % of its
   growth prefix; lambda, F, the gradient, B and the cell estimate are recomputed for the retained region, which
   must pass the acceptance rule again (else the block is dropped). Regions below `drop_small` = 0.25 of the median
   region's cells are dropped and the greedy runs again without seeding there (B68): their area goes to the fill.
   Given seeds (`seeds`) share a growth front (`front`, B67) between peers (`front_peer` = 6 h, B69): grown one after
   the other, two seeds whose gap lies within 6 h of their bisector are peers, and each region of a peer pair is kept
   on its own side of the pair's bisector, block_gap / 2 from it, so the gap between them is centred on that line; a
   seed beside a much larger region (the greedy's later seeds) keeps the order's layout.
3. Block map (m14/blockmap.py): zero-cone cross field of the region (trivial connection, free boundary) and m09's
   least-squares map with the target size h e^(lambda - c), free boundary, no cuts, no integers.
4. Block grid: nodes at the integer points, whole cells inside the region, cells with |ln s/h| > 0.10 trimmed, the
   cell set cleaned (opening/closing, spurs, notches, pinches), `smooth` kind "size" with the outline fixed.
4a. Direct join (`merge`, off by default; m14/sync.py merge): a block-block corridor
   of the first-pass blocks with b = 0 or enough drift whose two regions' union with the corridor between them passes
   the acceptance rule (place.union_region) gets one map over the union (blockmap.region_grid): the two blocks become
   one, without a seam. m14.json "merge" lists the candidates and the merged groups.
4b. Synchronisation (`sync`, m14/sync.py): the block graph from stitch_mix.partition of the
   first-pass blocks; per block a frame turn psi_b (an L1 LP over the corridor edges' 4-fold rotations; in conforming
   mode a block beside the coast turns toward the coast tangent, a fixed frame) and a count shift sigma_b within
   f_max - F_b (bounded least squares on the corridors' along counts, taken on the turned blocks); each block is
   re-extracted from its own map times e^(-i psi_b - sigma_b). psi_b, sigma_b go to the block records, the corridors
   to m14.json "sync".
5. Stitching (m14/stitch.py): m02 fills the water minus the blocks with the outlines as exact loops (nodes shared),
   merge by node ids, then a joint "size" + "angle" smoothing of a 3-row block-side + 4-row fill-side band.
5a. Smoothness within the size tolerance (option `size_tolerance`, 0.20 by default, 0 = off; decisions G38, M38):
   the last stage of the stitch (every fill, _mix too) is meshom.size_fit.smooth_fit on the whole fill, block and
   coast nodes fixed: the element size ratio between neighbours as small as possible, every edge within |ln s/h| <= t.
   m14_blocks_rep runs m14 without it and the fit after its repair instead (rep_variants, R22).
5b. Adaptive gaps (`gap_adapt`, off by default; m14/gaps.py, B63): every block-block corridor of the stitch is scored
   with the per-corridor seam measures (defects per 10 cells, |ln s/h| p99, ratio p99, quad p99, triangle clusters);
   a failing one gets both blocks trimmed by gap_rows cell rows along it (the gap widens there only), the stitch runs
   again, and a trim stays only where its corridor and its blocks' corridors together improve; up to gap_rounds
   rounds. m14.json "gaps" holds the per-corridor measures before and after and the rounds.
6. Measurement (m14/report.py): coverage, irregular nodes inside blocks and in the seam band, statistics per row from
   the outline; per block its seed, sizes and outline (lon-lat, with the corner indices: report.outline_record);
   returned by run(). Under meshom compete / meshom run (MESHOM_RUN_DIR set) it is deferred to the runner's
   "diagnostics" stage (compete.DEFERRED, outside the timed generation) and written to <run dir>/m14.json; so is
   the _cut variant's coast fit.
7. Optimisation (option `optimize`, on by default; decisions Z1): meshom.optimize on the final
   mesh (after the _cut variant's cut) at the size tolerance, the last stage (optimize_stage); its record is
   info["optimize"] and <run dir>/optimize.json; the measurement of step 6 is of the mesh before it (its stitch
   record indexes that mesh). m14_blocks_rep runs m14 without it and the stage after its repair, coast stage and
   fit instead. In the generation time (timings "optimize_s"; "total_s" stays the generation before it).

Fixed strait blocks (m14/strait.py; decisions S31, S32, B59, C35; option `strait_blocks`, on): before
step 2, one boundary-fitted block per passage of the coast's medial axis (meshom.lfs: W <= 20 h, |dW/ds| <= n g, both
ends mouths), n = max(strait_n_min, round(median W/h)) cells across at h = W/n, ends = cross chords orthogonal to the
centreline, banks = the coast filtered at 2 h; kept when it passes B22 and follows h (|ln s/h| p99 <= 0.10: h must
resolve the strait with n cells, which the S22 zone of `[coast] auto_straits` provides), with no island inside. The
automatic placement keeps block_gap from them; the stitch re-assembles the coast loops with their mouths (conforming)
or takes their outlines as exact loops before the land cut (variant A: the coast is fitted in straits, a staircase
elsewhere). Not with _igm or _mix; only where the domain has loops (`meshom run` with treatment "cut" hands the method
the closed sphere and cuts afterwards, so there are no strait blocks there). m14.json
gets "fixed_blocks" (one record per block: counts, widths, B22 numbers, the mouths as corridors, and the seam band at
its mouths) and "straits" (every passage piece tried).

Hand-placed blocks (m14/hand.py; decisions U6, B60; option `blocks`, the config's `[[blocks]]`):
built first, before the strait blocks (which keep block_gap from them), through the same fixed-block path: four
corners and four sides (coast / cut / lat / offset), the harmonic Dirichlet map, built as given with its B22 numbers
recorded. A block with a coast side is fitted as a strait block is (conforming: the coast loops re-assembled with its
other sides; cut: protected in the land cut); a block without one is an exact outline of the fill: in water when the
domain is conforming, over land and sea in cut mode (variant A, or `meshom run` with treatment "cut", where meshom.run
resolves the sides against the true coast), cut by the mask with everything else (ORCA style; coast-to-coast blocks
only this way). m14.json: the blocks in "fixed_blocks" (kind "hand"), every entry in "hand_blocks". Not with _igm or
_mix.

Thin coastal bands (m14/coastband.py; decisions C36, C37, C25; option `coast_bands`, default
off; conforming only, not with _cut, _igm or _mix): after the strait blocks, on the coast they leave free, fixed blocks
of n = min(coast_band_rows, k*, plateau / h, LFS / (2 h)) rows extruded from the coast filtered at 2 h along the normals
of the coast smoothed at 8 h, only where h is flat to the coast; split at the coast's corners (the wedges between two
bands' end cuts are fill), no deeper than the coast's turning allows; the along count follows h on the offshore row
and the coast row's ends lie on the raw coast (C50). They take the strait blocks' path: the placement keeps band_gap
(4.5 h, C50) from them, the stitch re-assembles the coast loops with their offshore outline, their coast row is never
moved. m14.json gets "coast_bands" (counts) and one "fixed_blocks" record per band (kind "band": rows, limits,
per-row size and angle, the seam band at its offshore outline).

Variant m14_blocks_igm: B4 cones are allowed inside blocks (<= 8 per block, >= 5 h apart, >= 4 h from the block
edge); the block map then is m09's parametrisation with cuts and integers inside the block (free outline) and the
cells are extracted by walks across the seams (m09.extract).

Variant m14_blocks_mix: the fill domain is partitioned into corridors between two blocks,
junctions, open holes and coast corridors (m14/stitch_mix.py); each region gets its class's fill (a ladder strip
between the two block lattices, m02's quad-dominant fill or triangles) with the region boundaries as shared nodes.
Default = the quantised join (B47): a ladder strip in every corridor whose lattices run along the
gap (obliquity < 20 deg) and whose ladder plan is valid, the lattice continuation (m14/continuation.py) in the other
corridors whose sides are lattice paths (with lattice caps), m02 everywhere else, meshed as one fill region; triangle
clusters of >= 3 broken (B62). join = "rule" is the measured rule of the m14 note (strip below 25 deg rotation, else
m02).

Variant m14_blocks_cut (coast treatment A, FESOM2 style; the default is B, conforming, FESOM-C style): steps 1-5
run on the closed sphere with the same h, as if there were no land; then the land is cut away by the mask
(m14/cut.py: elements whose centroid is on land removed, FESOM2's clean-up of spurs, pinched nodes and small
basins, renumbering with the new boundary first). The coast becomes a staircase through the block cells; nothing
moves. m14.json gets "cut" (counts, and the coast fit: distances to the true coast, water area lost / gained).

No automatic blocks (`auto_blocks` False, B109): the placement is skipped and the fill (with its own options, e.g.
m02's ORCA `frame`) takes the whole domain; m16_orca's fill between two ORCA parts.

Placement region (`place_region`, off; B106): blocks are seeded and grown only on the bg nodes inside a spherical
polygon, everything outside goes to the fill (one block over a uniform zone with `max_blocks` 1; m14/place.py).

Water-aware placement under a land cut (`water_place`, on; B70): when the land is cut away after the method (variant
_cut: its own domain; `meshom run` with treatment "cut": meshom.run hands the true coast over as the option `land`),
the placement seeds only in the water of the true coast and counts a region's cells in the water (min_cells,
drop_small); land still does not stop the growth (an island inside a block is cut out of it). m14.json gets
"region_water_coverage", "seeds_on_land" and per block "seed_wet", "region_water_cells".
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from meshom.domain import Domain
from meshom.mesh import Mesh
from meshom.methods import OVERRIDES, register
from meshom.size import SizeField

DEFAULTS = dict(
    bg_factor=1.0,          # background mesh of the fitness map and the block maps (units of h)
    cone_margin=3.0,        # admissible: >= this many h from every B4 cone
    coast_margin=3.0,       # ... from the coast
    complex_margin=6.0,     # ... from a complex coast point
    complex_turn_deg=30.0,  # a resampled coast vertex turning more than this is a complex coast point
    block_gap=6.0,          # blocks keep this many h apart (an accepted region dilated by it leaves the admissible set)
    min_cells=100.0,        # smallest block (area / h^2), i.e. 10 x 10 cells
    min_inradius=5.0,       # ... and inradius in cells after the grid is built
    min_clear=4.0,          # stop when no admissible node is this many h from the non-admissible set
    max_blocks=32,          # B8: at most this many automatic blocks; None / 0: no cap, the greedy runs until no seed is
                            # min_clear (m14_blocks_rep: None, rep_variants.REP_MAX_BLOCKS)
    seeds=None,             # [[lon, lat], ...]: grow exactly these seeds, in order (the advisor's handle; with a
                            # shared front, `front`)
    grow="linf",            # growth order: "linf" (L-inf norm in the cross-field frame: squares) | "metric" (discs)
    grow_limit=None,        # Dijkstra limit (units of h), None = none
    front="seeds",          # shared growth front (B67): given seeds grow each in its Voronoi cell (units of h) and
                            # > block_gap / 2 from the cell boundary, grow_limit not applied (place._front_pass);
                            # True: the greedy placement's seeds too (off there: coverage falls); False: off
                            # (m14_blocks_mix: off)
    front_peer=6.0,         # with the shared front: only seed pairs whose gap, grown one after the other, lies within
                            # this many h of their bisector share a front (place._peer_pass, B69: the greedy's own seeds
                            # given back keep its layout); None = every pair (the Voronoi cells of B67)
    drop_small=0.25,        # greedy placement: regions below this fraction of the median region's cells are dropped and
                            # the greedy reruns without seeding there; their area goes to the fill (B68; None = off)
    water_place=True,       # B70: under a land cut (variant _cut, or `land` given by meshom run's treatment "cut"):
                            # seeds only in the water, a region's cells (min_cells, drop_small) counted in water cells;
                            # land still does not stop the growth (its cells are cut afterwards)
    land=None,              # the true coast of the land cut that meshom run applies after the method (a Domain; set by
                            # meshom.run for treatment "cut"); the _cut variant uses its own domain
    compact=1.6,            # growth stops beyond compact x the equivalent-disc radius from the seed
    a_max=1e9,              # largest region (cells)
    f_max=0.08,             # acceptance: (max lambda - min lambda) / 2
    grad_max=0.05,          # acceptance: max h |grad lambda| over the region's bg triangles (P1 gradient)
    b_cap=1.3,              # pre-screen |B| <= min(b_cap, b_slope L) during growth
    b_slope=0.08,
    bisect=8,               # bisection steps for the largest accepted growth prefix
    margin_rings=0,         # the block map runs on the region plus this many bg triangle rings (cells: centre in it)
    overlap_retries=4,      # a block whose map overlaps itself is retried with 80 % of its growth prefix
    align=False,            # turn each block frame by a constant to match the global B4 field (consistent lattices)
    sync=True,              # frame and count synchronisation of the blocks before the stitch (m14/sync.py)
    merge=False,            # direct join (B61): two blocks whose union with their corridor passes the acceptance rule
                            # become one map (m14/sync.py merge, before the sync)
    merge_drift=0.0,        # ... candidates: corridors with b = 0 or a drift >= this many dislocations per 10 cells
    target="conformal",     # block map target size: "conformal" = h e^(lambda - c) | "h"
    trim=0.10,              # drop cells with an edge |ln s/h| above this
    shrink=0,               # erode each block's cleaned cell set by this many cells (3 x 3 square): wider gaps
                            # to the other blocks and the coast at the same placement
    block_size_iters=30,    # `smooth` kind "size" sweeps on each block, outline fixed
    band_block=3,           # seam band rows on the block side
    band_fill=4,            # ... on the fill side
    band_size_iters=20,     # joint band smoothing: "size" sweeps, then "angle" sweeps
    band_angle_iters=3,
    fill_method="m02",      # the fill between blocks and coast: "m02" (L-inf frontal) | "tri" (m02 triangles, no
                            # recombination: quads only inside the blocks) | "m07" (Instant Meshes) | "mix" (variant
                            # _mix: one fill per region class, m14/stitch_mix.py)
    fill={},                # options of the fill method (m02_linf_frontal.DEFAULTS / m07_instant_meshes.DEFAULTS)
    tri_popctl=8,           # fill_method "tri": population-control rounds in the gap (0 = off)
    tri_fit=True,           # fill_method "tri": per-edge size fit of the fill after the merge
    tri_vflips=True,        # fill_method "tri": valence flips after the population control (popctl.valence_flips)
    tri_fit_flips=True,     # fill_method "tri": Lawson flips between the fit's rounds
    tri_fit_p=6,            # fill_method "tri": exponent of the fit
    size_tolerance=0.20,    # G38; 0 / None = the fill follows h; t = the allowed |ln s/h|: after
                            # the seam band smoothing (and instead of the "tri" fill's size fit) size_fit.smooth_fit
                            # moves the fill nodes (blocks and coast fixed) to make sizes change as gently as
                            # possible within t
    tol_fit={},             # options of that smooth_fit (p, w_tol, w_h, w_angle, max_iter, rounds; M38)
    optimize=True,          # Z1: meshom.optimize as the method's last stage (after the cut of
                            # _cut; m14_blocks_rep: after its repair, coast stage and fit) at the run's size
                            # tolerance; 0 / False = off (meshom run's [mesh] optimize forces it either way)
    optimize_opts={},       # its options (meshom.optimize.DEFAULTS)
    # variant _mix (fill_method="mix"): partition of the fill into corridors / junctions / open / coast regions
    mix_bg=1.0,             # background mesh of the partition and the packing (units of h)
    j_margin=2.0,           # junction: the third-nearest object within this many h of the second-nearest
    w_open=14.0,            # open: the two nearest objects more than this many h apart (in total)
    min_region=6,           # regions with fewer background triangles are merged into a neighbour
    sep_smooth=3,           # Laplacian passes on each separator before it is resampled at h
    rot_quad_deg=25.0,      # corridor rule (measured, docs/methods/m14_blocks.md "m14_blocks_mix"): a strip-eligible
                            # corridor with rotation < this and mismatch <= rows_tol -> "strip"
    rot_tri_deg=90.0,       # ... rotation >= this or mismatch > rows_tri -> "tri" (90 / inf: never; 25 was tried
                            # first: the triangle fill lost in every class, 9-53 defects per 10 cells)
    rows_tol=3.0,           # row-count mismatch |nA - nB| per 10 cells of corridor length, strip limit
    rows_tri=float("inf"),  # ... triangle limit
    strip_lam=1.0,          # strip row counts: fewest triangles + strip_lam * sum (rows - width in h)^2 over lines
    obl_strip_deg=20.0,     # a strip needs both lattices within this angle of the gap direction (else its cells shear)
    join="quantised",       # corridor rule: "quantised" (B47: the Burgers vector b and the obliquity
                            # decide; the ladder below obl_strip_deg with B58's placement, the lattice continuation
                            # above it, B57) | "rule" (rot_quad_deg / rows_tol, the measured table of the m14 note)
    cont={},                # options of the continuation fill (m14/continuation.py DEFAULTS)
    end_rows=2,             # join = "quantised" (B58): no dislocation within this many cells of a corridor cap that
                            # borders a junction, the coast or an open region: ladder column / row ends, continuation
                            # events
    cont_fallback=True,     # join = "quantised": the continuation also where the ladder is impossible or invalid
                            # (obliquity below obl_strip_deg), else m02
    lattice_caps=True,      # join = "quantised": a continuation corridor's two caps as lattice paths of its blended
                            # map (B57), else its separators
    break_clusters=True,    # _mix: triangle clusters >= 3 broken (a pair merged, a polygon re-split, a node removed;
                            # B62)
    free_separators=True,   # the final fill smoothing moves the separator nodes too (the regions' topology is kept)
    merge_same_fill=True,   # adjacent non-strip regions with the same fill are meshed as one (no separator between)
    junction_fill="m02",    # fill of junctions (three or more objects near): "m02" | "tri"
    open_fill="m02",        # fill of open regions (wider than w_open): "m02" | "tri"
    coast_fill="m02",       # fill of coast corridors and of regions touching the coast: "m02" | "tri"
    coast="conform",        # "conform" (the coast loops are mesh edges); "cut" reserved (m14_blocks_cut cuts the run)
    force_fill=None,        # "strip" | "m02" | "tri": this fill on every block-block corridor (decision table)
    tri_fill=None,          # "module:function" fill_tri(domain, h, fixed_loops) -> Mesh for the "tri" regions (m15)
    igm=False,              # variant _igm: cones inside blocks
    igm_max_cones=8,
    igm_cone_gap=5.0,       # cones inside a block at least this many h apart
    igm_cone_edge=4.0,      # ... and this many h from the block edge
    igm_min_fill=576.0,     # the growth leaves at least this many cells of water to the fill
    igm_max_share=0.45,     # ... and one block covers at most this share of the water
    igm_stiffen=8,          # m09 stiffening iterations (before and after rounding)
    igm_trim=0.3,           # boundary cells with an edge |ln s/h| above this are peeled
    cut=False,              # variant _cut: build on the closed sphere, then cut the land away (coast treatment A)
    blocks=None,            # hand-placed fixed blocks, built first: the config's [[blocks]] (meshom.config.BlockCfg
                            # fields, or resolved by meshom.run against the true coast; m14/hand.py, B60)
    strait_blocks=True,     # fixed blocks in the straits of the medial axis first (m14/strait.py; S31)
    strait_n_min=5,         # S22 / S23: at least this many cells across a strait block
    strait_w_max=20.0,      # S31: a strait is at most this many h wide
    strait_grad=0.1,        # S31: |dW/ds| <= n strait_grad along a strait
    coast_bands=False,      # thin coastal bands as fixed blocks after the strait blocks (m14/coastband.py; C36, C37),
                            # conforming mode only (not _cut, _igm, _mix)
    coast_band_rows=4,      # C36: n_max, the most rows a band takes
    band_gap=4.5,           # C50: the automatic placement keeps this many h from the bands' nodes (None: block_gap)
    gap_adapt=False,        # adaptive gaps (m14/gaps.py, B63): a block-block corridor failing its seam gates after the
                            # stitch gets both blocks trimmed along it and the stitch runs again; a trim is kept only
                            # where the corridor's seam measures improve
    gap_rows=2,             # ... cell rows trimmed per block and round (the gap widens by about twice this, in h)
    gap_rounds=6,           # ... rounds (at most one trim per block each; 1-3 stitches per round)
    gap_gates=None,         # ... the per-corridor gates (None: gaps.GATES)
    gap_judge="raw",        # ... the mesh a trim is judged on: "raw" = the stitch; "post" = what the calling variant's
                            # later stages make of it (m14_blocks_rep: repair, fit, optimisation stage; B63)
    auto_blocks=True,       # B109: False = no automatic block (no placement: the fill takes the whole domain, the
                            # hand / strait / band blocks as before)
    place_region=None,      # B106: [[lon, lat], ...] a spherical polygon (great-circle edges, as a size zone's
                            # `polygon`): automatic blocks are seeded and grown only on bg nodes inside it, everything
                            # outside goes to the fill; None = the whole domain (not with _igm)
    seed_grow_limit=None,   # with `seeds` and the shared front: a Dijkstra limit (units of h) per seed, None = none
                            # (the LLM advisor's lever)
    gap_trim=None,          # [{"a": [lon, lat], "b": [lon, lat], "rows": k}, ...]: after the first stitch the two
                            # blocks nearest to a and b (by seed) lose k cell rows along their corridor (gaps.trim_cells)
                            # and the stitch runs once more (the advisor's per-corridor gap lever; gaps.trim_pairs)
    verbose=False,
)


def optimize_stage(mesh: Mesh, h: SizeField, tol: float | None, opts: dict | None,
                   frozen: np.ndarray | None = None) -> tuple[Mesh, dict]:
    """Z1: meshom.optimize as the method's last stage (no extra fixed nodes, as meshom run's stage; frozen: the cells
    m14_blocks_rep freezes, R24); (mesh, record): the record (optimize.record) also goes to
    <MESHOM_RUN_DIR>/optimize.json when that is set."""
    from meshom.optimize import optimize, record

    out, oinfo = optimize(mesh, h, tol, opts, frozen=frozen)
    rec = _jsonable(record(oinfo))
    d = os.environ.get("MESHOM_RUN_DIR")
    if d:
        (Path(d) / "optimize.json").write_text(json.dumps(rec, indent=1))
    return out, rec


def _log(o: dict, msg: str) -> None:
    if o["verbose"]:
        print(f"[m14] {msg}", file=sys.stderr, flush=True)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def build_blocks(domain: Domain, h: SizeField, o: dict, fixed: list = (), land: Domain | None = None) -> tuple:
    """Steps 1-4: (fitness, regions, grids, placement log, timings); fixed: the fixed blocks placed before (B59); land:
    the true coast of a land cut that follows (B70: the placement's water)."""
    from meshom.methods.m14.blockmap import block_grid
    from meshom.methods.m14.fitness import fitness
    from meshom.methods.m14.place import lambda_solve, place, region_lambda, tri_capacity, with_margin

    t = {}
    t0 = time.perf_counter()
    fit = fitness(domain, h, o, land)
    t1 = time.perf_counter()
    t["fitness_s"] = t1 - t0
    if o["igm"]:
        from meshom.methods.m14.igm import place_igm

        regions, plog = place_igm(fit, o)
    elif not o.get("auto_blocks", True):                           # B109: the fill takes the domain
        regions, plog = [], {"auto_blocks": False}
    else:
        band = [f.record.get("kind") == "band" for f in fixed]
        kw = {"fixed_pts": np.vstack([f.grid.points for f, b in zip(fixed, band) if not b])} if not all(band) else {}
        if any(band):
            kw["band_pts"] = np.vstack([f.grid.points for f, b in zip(fixed, band) if b])
        regions, plog = place(fit, o, **kw)
    t2 = time.perf_counter()
    t["place_s"] = t2 - t1
    grids, kept = [], []
    q, a = tri_capacity(fit)
    for r in regions:
        for attempt in range(o["overlap_retries"] + 1):
            mt = with_margin(fit.bg.tris, r.tris, o["margin_rings"])    # map region: the block region + a margin
            nodes, Tl = np.unique(fit.bg.tris[mt], return_inverse=True)
            Tl = Tl.reshape(-1, 3)
            in_r = np.isin(mt, r.tris)
            lam, _, _ = lambda_solve(fit.bg.points[nodes], Tl, (fit.area + fit.lap)[nodes], fit.hv[nodes])
            if o["igm"]:
                from meshom.methods.m14.igm import block_grid_igm

                g = block_grid_igm(fit.bg.points[nodes], Tl, lam, in_r, r, h, o, fit.u[nodes])
            else:
                g = block_grid(fit.bg.points[nodes], Tl, lam, h, o, in_r, fit.u[nodes])
            if g is None or not g.info.get("overlap"):
                break
            # the map's image overlaps itself (a curved tail): retry with a shorter growth prefix (still a disc)
            plog["overlap_retries"] = plog.get("overlap_retries", 0) + 1
            r.tris = r.tris[: int(0.8 * len(r.tris))]
            r.B, r.cells = float(q[r.tris].sum()), float(a[r.tris].sum())
            if o["igm"]:                                  # _igm regions carry no lambda
                r.nodes = np.unique(fit.bg.tris[r.tris])
                continue
            r.nodes, r.lam, r.F, r.grad = region_lambda(fit, r.tris)
            if r.F > o["f_max"] or r.grad > o["grad_max"]:      # the retained region must pass the rule again
                plog["overlap_retry_rejected"] = plog.get("overlap_retry_rejected", 0) + 1
                g = None
                break
        if g is None or g.info["n_cells"] < o["min_cells"] or g.info["inradius_cells"] < o["min_inradius"]:
            plog["dropped_small"] = plog.get("dropped_small", 0) + 1
            grids.append(g)
            continue
        grids.append(g)
        kept.append(len(grids) - 1)
    t["blocks_s"] = time.perf_counter() - t2
    return fit, regions, grids, kept, plog, t


# the built fixed blocks of the last run() for meshom run's land cut (run.cut_land, called on the returned mesh in the
# same process): "protect" = the cells of the blocks with a coast side (element mask), "coast_nodes" = their bank
# nodes, "bank_edges" = their bank edges (mesh ids, FixedBlock.bank); "fixed_id" = the fixed block of each element
# (-1: none), "fixed_kind" = each block's kind (hand / strait / band: m14_blocks_rep freezes some of them, R24); None
# when no fixed block was built
CUT_FIXED: dict | None = None


def run(domain: Domain, h: SizeField, opts: dict | None = None, post=None) -> tuple[Mesh, dict]:
    """m14 (module docstring). post: gap_adapt's judge (B63, gap_judge "post"): mesh, stitch record -> the mesh the
    later stages of the calling variant make of it (m14_blocks_rep: repair, fit, optimisation stage); None = raw."""
    from meshom.methods.m14.report import block_records, measure
    from meshom.methods.m14.stitch import stitch

    o = dict(DEFAULTS)
    o.update(opts or {})
    o["size_tolerance"] = float(o["size_tolerance"]) if o["size_tolerance"] else None   # G38: 0 = off
    t_start = time.perf_counter()
    bdom = Domain([], domain.name) if o["cut"] else domain             # variant A: built as if there were no land
    fixed, sinfo, hrec = [], None, None
    if o["blocks"] and not o["igm"] and o["fill_method"] != "mix":    # hand-placed blocks first (B60)
        from meshom.methods.m14.hand import hand_blocks

        t7 = time.perf_counter()
        fixed, hrec = hand_blocks(o["blocks"], domain, h, o)
        _log(o, f"hand blocks {len(fixed)} of {len(o['blocks'])}")
        hand_s = time.perf_counter() - t7
    if o["strait_blocks"] and domain.loops and not o["igm"] and o["fill_method"] != "mix":
        from meshom.methods.m14.strait import strait_blocks

        t6 = time.perf_counter()
        sb, sinfo = strait_blocks(domain, h, o, [f.grid.points for f in fixed])
        for f in sb:
            f.record["fixed_index"] += len(fixed)
        fixed = fixed + sb
        _log(o, f"strait blocks {len(sb)} of {sinfo['n_passages']} passages")
        sinfo["s"] = time.perf_counter() - t6
    binfo = None
    if o["coast_bands"] and domain.loops and not o["cut"] and not o["igm"] and o["fill_method"] != "mix":
        from meshom.methods.m14.coastband import coast_bands

        t7 = time.perf_counter()
        bands, binfo = coast_bands(domain, h, o, fixed)
        fixed = fixed + bands
        binfo["s"] = time.perf_counter() - t7
        _log(o, f"coast bands {len(bands)}: {binfo}")
    land = domain if o["cut"] and domain.loops else o["land"]      # B70: the water of the land cut that follows
    fit, regions, grids, kept, plog, t = build_blocks(bdom, h, o, fixed, land)
    if sinfo is not None:
        t["strait_s"] = sinfo["s"]
    if hrec is not None:
        t["hand_s"] = hand_s
    if binfo is not None:
        t["coast_bands_s"] = binfo["s"]
    merge_rec = None
    if o["merge"] and not o["igm"] and len(kept) > 1:
        from meshom.domain import resample_loops
        from meshom.methods.m14.sync import merge

        t5 = time.perf_counter()
        regions, grids, kept, merge_rec = merge(resample_loops(bdom, h), fit, regions, grids, kept, h, o,
                                                np.vstack([f.grid.points for f in fixed]) if fixed else None)
        t["merge_s"] = time.perf_counter() - t5
    sync_rec = None
    if o["sync"] and not o["igm"] and kept:
        from meshom.domain import resample_loops
        from meshom.methods.m14.sync import sync

        t5 = time.perf_counter()
        kg = [grids[k] for k in kept]
        sync_rec = sync(resample_loops(bdom, h), kg, h, o, kept)
        for k, g in zip(kept, kg):
            grids[k] = g
        t["sync_s"] = time.perf_counter() - t5
    t3 = time.perf_counter()
    mesh, st = stitch(bdom, h, [grids[k] for k in kept], o, fixed)
    t["stitch_s"] = time.perf_counter() - t3
    trim_rec = None
    if o["gap_trim"] and not o["igm"] and len(kept) > 1:              # the advisor's per-corridor gaps
        from meshom.methods.m14.gaps import trim_pairs

        t5 = time.perf_counter()
        seeds = np.array([fit.bg.points[regions[k].seed] for k in kept])
        mesh, st, kg, trim_rec = trim_pairs(bdom, h, [grids[k] for k in kept], o, fixed, mesh, st, seeds)
        for k, g in zip(kept, kg):
            grids[k] = g
        t["gap_trim_s"] = time.perf_counter() - t5
    gap_rec = None
    if o["gap_adapt"] and not o["igm"] and len(kept) > 1:              # adaptive gaps (B63)
        from meshom.methods.m14.gaps import adapt

        t5 = time.perf_counter()
        mesh, st, kg, gap_rec = adapt(bdom, h, [grids[k] for k in kept], o, fixed, mesh, st, post=post)
        for k, g in zip(kept, kg):
            grids[k] = g
        t["gaps_s"] = time.perf_counter() - t5
        _log(o, f"gaps: {gap_rec['failed_first']} of {gap_rec['pairs_first']} pairs failed, trimmed "
                f"{gap_rec['trimmed']}, {gap_rec['failed_last']} fail after")
    cut_info = None
    if o["cut"]:
        from meshom.methods.m14.cut import cut

        t5 = time.perf_counter()
        mesh, st, cut_info = cut(mesh, st, domain, h)
        t["cut_s"] = time.perf_counter() - t5
    global CUT_FIXED
    CUT_FIXED = ({"protect": (st["fixed_id"] >= 0) & st["fixed_protect"][np.maximum(st["fixed_id"], 0)],
                  "coast_nodes": st["fixed_coast_nodes"], "bank_edges": st["fixed_bank_edges"],
                  "fixed_id": st["fixed_id"], "fixed_kind": st["fixed_kind"]}
                 if "fixed_id" in st else None)
    if o["fill_method"] == "tri":                         # the triangle fill is a declared triangle zone
        mesh.tri_zone = np.ones(mesh.n_tris, dtype=bool)
    t["total_s"] = time.perf_counter() - t_start         # generation only (the diagnostics below are timed apart)
    if o["fill_method"] == "mix" and st.get("tri_zone") is not None and st["tri_zone"].any():
        mesh.tri_zone = st["tri_zone"]                    # _mix: the triangles of its "tri" regions
    name = ("m14_blocks_igm" if o["igm"] else "m14_blocks_tri" if o["fill_method"] == "tri" else
            "m14_blocks_mix" if o["fill_method"] == "mix" else "m14_blocks")
    info = {"method": name + ("_cut" if o["cut"] else ""), "domain": domain.name, "h": repr(h),
            "params": {k: v for k, v in o.items() if k not in ("verbose", "blocks", "land")},  # blocks: "hand_blocks"
            "n_nodes": mesh.n_nodes, "n_tris": mesh.n_tris, "n_quads": mesh.n_quads,
            "n_regions": len(regions), "n_blocks": len(kept), "blocks_kept": kept,
            "region_coverage": sum(float(fit.area[r.nodes].sum()) for r in regions) / float(fit.area.sum()),
            "n_cones_field": int(len(fit.cone_pos)),                                       # _cut: of the sphere
            **({} if fit.wet is None else {"region_water_coverage": sum(float(fit.area[r.nodes][fit.wet[r.nodes]].sum())
                                                                    for r in regions) / float(fit.area[fit.wet].sum()),
                                           "seeds_on_land": int(sum(not fit.wet[r.seed] for r in regions))}),
            "placement": plog, "blocks": block_records(regions, grids, fit), "stitch": st["info"], "timings": t,
            "total_s": t["total_s"]}
    if merge_rec is not None:
        info["merge"] = merge_rec
    if sync_rec is not None:
        info["sync"] = sync_rec
    if gap_rec is not None:
        info["gaps"] = gap_rec
    if trim_rec is not None:
        info["gap_trim"] = trim_rec
    if sinfo is not None or hrec is not None or binfo is not None:
        info["fixed_blocks"] = [f.record for f in fixed]
    if sinfo is not None:
        info["straits"] = {k: v for k, v in sinfo.items() if k != "s"}
    if hrec is not None:
        info["hand_blocks"] = hrec                        # every [[blocks]] entry, built or not
    if binfo is not None:
        info["coast_bands"] = {k: v for k, v in binfo.items() if k != "s"}

    def diagnostics() -> None:
        """measure (and the _cut coast fit): fills info, writes <MESHOM_RUN_DIR>/m14.json when that is set."""
        if cut_info is not None:
            if domain.loops:
                from meshom.methods.m14.cut import coast_fit

                t5 = time.perf_counter()
                cut_info["coast_fit"] = coast_fit(mesh, domain, h)
                t["coast_fit_s"] = time.perf_counter() - t5
            info["cut"] = cut_info
        t4 = time.perf_counter()
        rep = measure(mesh, h, st)
        if "fixed_id" in st:                              # the seam band at each fixed block's mouths
            from meshom.methods.m14.report import fixed_seams

            for rec, sm in zip(info["fixed_blocks"], fixed_seams(mesh, h, st)):
                rec["seam"] = sm
        t["measure_s"] = time.perf_counter() - t4
        info.update(_jsonable({**info, **rep}))
        if "region_records" in st:                        # _mix: one record per fill region, measured here
            from meshom.methods.m14.stitch_mix import measure_regions

            t6 = time.perf_counter()
            info["regions"] = _jsonable(measure_regions(mesh, h, st["elem_region"], st["region_records"]))
            t["regions_measure_s"] = time.perf_counter() - t6
        _log(o, f"blocks {len(kept)}/{len(regions)} coverage {rep['coverage_area']:.3f} irregular in blocks "
                f"{rep['irregular_in_blocks']} seam {rep['seam_band']} {t}")
        if d:
            (Path(d) / "m14.json").write_text(json.dumps(info, indent=1))
            if "elem_region" in st:                       # _mix: region of each element (to re-measure a smoothed mesh)
                np.save(Path(d) / "m14_elem_region.npy", st["elem_region"])
            elif "fill_outl" in st:                       # the m02 fill's loops (gaps.score partitions them again)
                np.savez(Path(d) / "m14_fill_loops.npz", n_coast=len(st["fill_coast"]),
                         *(list(st["fill_coast"]) + list(st["fill_outl"])))

    d = os.environ.get("MESHOM_RUN_DIR")
    if d:                                                 # under the runner: after the timed generate stage
        from meshom.compete import DEFERRED

        DEFERRED.append(diagnostics)
    else:
        diagnostics()
    if o["optimize"]:                                     # Z1: the last stage (diagnostics: of the mesh before it)
        t8 = time.perf_counter()
        out, info["optimize"] = optimize_stage(mesh, h, o["size_tolerance"], o["optimize_opts"])
        t["optimize_s"] = time.perf_counter() - t8
        _log(o, f"optimize {t['optimize_s']:.1f} s: {info['optimize']['before']} -> {info['optimize']['after']}")
        return out, info
    return mesh, info


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    return run(domain, h, opts)[0]


@register("m14_blocks_igm", overrides={"igm": True})
def generate_igm(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    return run(domain, h, dict(OVERRIDES["m14_blocks_igm"], **(opts or {})))[0]


@register("m14_blocks_tri", overrides={"fill_method": "tri"})
def generate_tri(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """Quads only inside the blocks; the water between them and the coast is triangles (no recombination)."""
    return run(domain, h, dict(OVERRIDES["m14_blocks_tri"], **(opts or {})))[0]


@register("m14_blocks_mix", overrides={"fill_method": "mix", "front": False})
def generate_mix(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """The fill partitioned into corridors, junctions, open and coast regions, each filled by its class's rule. Given
    seeds grow one after the other (front off, B67: with the shared front the seam lakes went 5 -> 4 of 15 and
    seam_rot45_k10 raised B52's region-leak error)."""
    return run(domain, h, dict(OVERRIDES["m14_blocks_mix"], **(opts or {})))[0]


@register("m14_blocks_cut", overrides={"cut": True})
def generate_cut(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    """Coast treatment A: blocks and fill on the closed sphere (no land), then the land cut away by the mask."""
    return run(domain, h, dict(OVERRIDES["m14_blocks_cut"], **(opts or {})))[0]
