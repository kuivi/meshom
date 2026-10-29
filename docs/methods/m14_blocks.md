# m14 — curvilinear blocks placed from h, stitched by m02 (`m14_blocks`, `m14_blocks_igm`)

Code: `meshom/methods/m14_blocks.py`, `meshom/methods/m14/{fitness,place,blockmap,stitch,report,igm,cut}.py`; test
`tests/test_m14_blocks.py`, `tests/test_m14_rules.py`. Results: the final m14 competition runs (re-scored with the
revised gates of 2026-10-22) and the exploration runs; per run `<run dir>/m14.json` (coverage, blocks, rows, seam band,
parameters).

**After the gate revision (2026-10-22).** (1) Under the revised gates no m14 variant passes every gate any more: the
"only method with every applicable gate on all four cases" of the final run held for the old gate set only.
`m14_blocks` at scale 1 raw fails 1 / 1 / 1 / 2 gate families: the triangle-cluster gate on every case (2 / 5 / 15 / 9
edge-connected clusters of 3–4 fill triangles, in fill rows 1–6 and on the band ramp, none at the coast) and on
continent the coastal size target once the coast's own resampled edges leave the coast set (coast |ln s/h| p99
0.143 → 0.158; first three coastal rows 0.208; the channel 0.018). m07_im and m08_keep lead with 0 / 0 / 0 / 1. (2) The acceptance rule now bounds the P1 gradient of λ on each background
triangle (h |∇λ| ≤ 0.05, h = the triangle's mean), not the slope along its edges, which is a lower bound of it. Re-run
of the placement on the final runs' fitness maps (a probe; the old rule reproduces the final runs' regions to the rounding): the region count is unchanged everywhere; uniform
(s4, s1), disc s4, continent s4 and `_cut` s4 keep identical regions; under the new rule 3 of 28 old regions exceed
0.05 on disc s1 (0.050–0.059; 5 regions change), 3 of 10 on band s4 (up to 0.096; 4 change), 17 of 32 on band s1 (up
to 0.071; 26 change through the greedy order), 1 of 18 on continent s1 (0.051; 1 changes), 13 of 32 on `_cut`
continent s1 (up to 0.070; 21 change); the region area shrinks by ≤ 0.5 % (band s4). The meshes of the records were
built with the old rule and are not regenerated. (3) A region retried after a self-overlapping map now gets its λ, F,
gradient, B and cell estimate recomputed and must pass the acceptance rule again (in the final runs one retry,
continent s1, which passes: F 0.047, gradient 0.008). (4) The seam-band edge statistics now use fill rows 0–3 like
its elements and nodes; the numbers below are corrected. (5) `measure` and `_cut`'s `coast_fit`
run after the timed generation, in the runner's "diagnostics" stage (`diagnostics_s`); the final records' times still
include them (0.1–1.1 s; `_cut` continent s1 4.8 s of 53.7 s).

- **What.** (1) Fitness map: bg mesh 1.0 h, B4 field + cones, ρ_c per dual cell (`compete._node_rho`). (2) Greedy
  placement from the clearest admissible node (≥ 3 h from cones and coast, 6 h from complex coast and other blocks),
  growth over bg triangles in L∞ order of the cross frame (squares along field lines), kept a disc, stopped by the
  pre-screen |B| ≤ min(1.3, 0.08 L) or a compactness rule (next triangle > 1.6 × equivalent-disc radius from the seed);
  accepted = largest prefix whose λ (Δλ = −(K + Δ ln h), λ|∂ = 0, cotan Laplacian, AMG-CG) has F ≤ 0.08 and
  h|∇λ| ≤ 0.05. (3) Block map: `cross_field(..., cones=False)` on the region, m09 `parametrize(..., free_boundary=True)`
  with target h e^(λ−c): one solve, no cuts, no integers. (4) Grid: integer points (best of 4×4 offsets); a cell is
  kept when its centre is in the region (corners just outside are extrapolated from the centre's face); trim |ln s/h|
  > 0.10, opening/closing, holes refilled, spurs/notches/pinches; a curved region whose map image overlaps itself
  (outline self-crossing) is retried with 80 % of its growth prefix; 30 `smooth` "size" sweeps, outline fixed.
  (5) m02 fills water minus blocks with the outlines as exact loops (`Domain.exact`), merge by node ids, then 20 "size"
  + 3 "angle" sweeps on a 3 block + 4 fill row band. `_igm`: good cones (≥ 5 h apart) inside blocks (4 h discs taken
  whole, ≤ 8), residual pre-screen only, m09 with cuts and integers, `m09.extract`, quad-graph cleaning, trim 0.3.
- **Scale 1, raw, `m14_blocks`** (uniform / disc / band / continent): 52k / 215k / 426k / 198k nodes; triangles 1.13 /
  0.72 / 0.85 / 0.92 %; quad θmax 140 / 135 / 140 / 135°, p99.9 123 / 120 / 125 / 123°; elem ratio p99 1.09 / 1.09 /
  1.11 / 1.09; **|ln s/h| p99 0.148 / 0.128 / 0.140 / 0.135** (m02 0.17–0.27, m07 0.11–0.13, m09 0.27–0.63, m10 0.20–0.40,
  m13 0.22–0.35); irregular interior nodes 1.28 / 0.83 / 1.02 / 1.08 % (m02 0.63–1.30 %, m07 0.64–1.16 %);
  gates 18/18, 18/18, 18/18, 24/24 under the earlier gate set (m07 18/18 ×3 + 25/26; m02 16/18 + 22/24);
  under the revised gates failed families 1 / 1 / 1 / 2 (above); 8.2 / 40 / 80 / 42 s in the final run (band:
  fitness 27, placement 24, blocks 22, stitch 9 s), peak 0.26–1.2 GB (exploration runs). Runner `--smooth` (10 angle
  sweeps on everything) costs size fidelity: 16/18 on uniform and band (old set). Scale 4: valid everywhere, 14–21
  gates (old set; as all methods: g = 0.4), 2–4 failed families.
- **Coverage** (block cells' area / water; cells share): scale 1 0.873 / 0.880 / 0.766 / 0.858 (0.855 / 0.857 / 0.839 /
  0.826), 6 / 24 / 32 / 18 blocks; scale 4 0.530 / 0.743 / 0.521 / 0.701 (the 6 h gaps are 22° wide at scale 4).
  **Inside blocks: 0 irregular nodes on every run**, θmax max 98–108°, p99.9 94–95°, aspect p99 1.03–1.05, |ln s/h|
  p99 0.078 (= F + noise), elem ratio p99 1.006–1.009 (hand-made ocean model grids: p99.9 90–92°, ratio 1.00–1.07).
- **Seams** (outline + 4 fill rows, scale 1): irregular 7.8 / 4.0 / 3.9 / 6.0 % of their nodes, 508 / 619 / 1058 / 1021
  triangles, |ln s/h| p99 0.243 / 0.192 / 0.206 / 0.211, elem ratio p99 1.155 / 1.135 / 1.154 / 1.141 (max 1.28–1.44;
  the runs' m14.json took the edges of fill rows 0–5: 0.24 / 0.21 / 0.22 / 0.22 and 1.156 / 1.150 / 1.175 / 1.145), quad
  θmax 135–140°, p99.9 127–133°; across the outline itself ratio p99 1.02–1.09. All defects of the mesh sit there and in
  the grading zones: the total count is about m02's, only concentrated (pictures: clean blocks with the smooth ±0.08 conformal
  pattern, one dislocation line per gap where the two fronts collide, a dense ring in the 3 h strip round the island).
- **Aspect allowance** (placement only, region coverage, capacity scaled by LP factors; scale 1, a = 1.0 / 1.2 /
  1.35 / 1.5): per-edge size gate (C, ×1 / 1.18 / 1.18 / 1.18): uniform 0.893 / 0.892 / 0.892 / 0.892, disc 0.897 /
  0.918 / 0.918 / 0.918, band 0.789 / 0.796 / 0.796 / 0.796, continent 0.902 / 0.907 / 0.907 / 0.907 — the margins, not
  curvature, limit blocks; a √area gate (B, ×2.6 / 3.4 / 4.2 at 1.2 / 1.35 / 1.5) would give 0.89–0.96 / 0.95–0.98 /
  0.91–0.94 / 0.94–0.97 with 2–10 huge blocks.
- **`_igm`, scale 1:** uniform 0.889 coverage in 2 blocks holding the 8 cube cones, triangles 0.58 %, irregular 0.80 %,
  but |ln s/h| p99 0.268 (cone distortion, as m09/m10/m13) → 16/18; disc 0.908 (15 cones), 16/18, 0.198; band 0.838
  (40 cones), 16/18, 0.217; continent fails: the two big blocks (95 % of the water) yield no single simple outline after cleaning,
  coverage 0.054, the rest is m02 (22/24). Block quads near cones 120–135°. Under the revised gates: failed families
  1 / 2 / 2 / 2 (size everywhere, triangle clusters on disc, band, continent).
- **Tried and dropped:** frames turned to match the global B4 field (`align`): more seam triangles (uniform 124 → 170).
  m07 as the fill (`fill_method="m07"`): 369 vs 85 seam triangles, and it fails at its default bg factor 0.5 with exact
  loops. A 2-ring map margin: reaches into ramps (Δ ln h ≈ −g² per cell), replaced by corner extrapolation.
- **Not done:** `m14_blocks_energy` (Garanzha energy), the direct join of matching blocks (the 6 h gap of the greedy
  placement never lets two blocks touch), the 1 → 4 background split, the LLM advisor (the `seeds` option and m14.json
  replay exactly: rerun from the saved params and from the saved seeds both give the identical mesh).
- **Production would need:** a seam fill that interpolates between two fixed lattices instead of colliding fronts
  (a strip/transfinite or map-based fill of the gaps, or merging neighbouring regions into one map), placement that
  sizes gaps from the count mismatch rather than a fixed 6 h, a wider or coast-aligned strip at smooth coasts, a
  fill whose points do not lie in one hemisphere (or a Delaunay that tolerates it: `_igm` caps a block at 45 % of the
  water for that reason), and `_igm`'s cleaning made robust for blocks with many cones.

## `m14_blocks_tri`: triangles between the blocks

Same placement and blocks, `fill_method="tri"`: m02's packing along the asterisk field (n = 6) and its smoothing,
no recombination, so quads exist only inside the blocks. Scale 1, raw, uniform / disc / band / continent:

- Triangles 26.4 / 26.8 / 30.2 / 31.9 % of the elements (`m14_blocks`: 0.7–1.1 %); coverage unchanged (0.87 / 0.88 /
  0.77 / 0.86). Time and memory as `m14_blocks`.
- Every quad is a block quad: θmax p99.9 102.6–104.2°, max 109.7–118.1°, no n3, no dislocation lines, the block edges
  join the triangles one to one (zoom_seam pictures).
- |ln s/h| p99 0.211 / 0.209 / 0.214 / 0.221 → 16/18, 16/18, 16/18, 20/23: the only failed gate is size fidelity
  (0.15), as for `m01_triangles` and `m02_linf_frontal_tri` (0.20). Under the revised gates its triangle fill is a
  declared triangle zone (`Mesh.tri_zone`; the cluster gate counts only triangles outside zones), so it fails 1 / 1 /
  1 / 2 families: size everywhere, and on continent the element ratio p99 of the first three coastal rows (1.160).
  The packing without the recombination + smoothing step is less size-faithful than the recombined fill, and the seam
  band shows |ln s/h| p99 0.25–0.28 (rows 0–3; quad fill 0.19–0.24). The valence counts in the seam statistics are meaningless for triangles (every valence-6 node counts as
  "other").
- Scale 4: 48–68 % triangles (coverage 0.52–0.74), 14/18 and 17/23.

Reading: the join itself is clean (quads in blocks, triangles for the transitions); what
is missing is a size-faithful triangle generator for the gaps. A triangle fill with a proper size-driven smoothing
(ODT / CVT with h, or m01's spring relaxation with more sweeps) is the next thing to try before judging the variant.

**m15 plug (`docs/methods/m15_tri.md`):** `fill_method="tri"` now runs m15's population control in the
gap (`tri_popctl=8` rounds, loop nodes fixed), valence flips (`tri_vflips`) and, after the merge and the band
smoothing, the per-edge size fit on the fill nodes only (`tri_fit`, coast and every block node fixed). Numbers in
`m15_tri.md` §"m14 plug"; the old behaviour is `tri_popctl=0, tri_vflips=False, tri_fit=False`.

## `m14_blocks_cut`: the mesh cut by the land mask (variant A) vs conforming (variant B)

Two coast traditions: **B** (`m14_blocks`, FESOM-C style) builds along the coast (loops
resampled at h, blocks `coast_margin` = 3 h off it, the m02 fill conforms); **A** (`m14_blocks_cut`, FESOM2 style)
runs steps 1–5 on `Domain([], name)` with the same h, as if there were no land, then cuts the land away
(`meshom/methods/m14/cut.py`; option `cut=True`, `m14_blocks` defaults unchanged). Nothing moves: the block cells
stay intact where they are water and the coast becomes a staircase through them (as in every curvilinear ocean grid).
`m14.json` gets `"cut"` (counts) with `"coast_fit"`; `region_coverage` there is a share of the whole sphere, while
`coverage_area` is measured on the cut mesh (share of the water).

**FESOM2 rules** (the land cut and clean-up after D. Sein (AWI), fesom2_mesh_gen):
- wet test per element: fesom2_mesh_gen counts an element wet when at least half of 9 sub-triangle samples of the
  mask raster are water. Here: the element centroid against the original coast loops (`meshom.domain.in_water`), one
  sample.
- clean-up loop until the wet count stops changing: spurs (elements with ≥ 2 boundary edges, repeated; FESOM2 rejects
  triangles with a single neighbour) — for a mixed mesh an element with < 2 edge neighbours (triangle ≥ 2, quad ≥ 3
  boundary edges; the gate `elem_neighbours_min`); pinched nodes (at a node with > 2 boundary edges the largest fan of
  wet elements stays); components (only the largest basin).
- extract: kept elements, used nodes renumbered; here the new boundary loops come first (loop order, water on the
  left, `node_flag` 1), then the other nodes in their old order.
- Not reproduced: strait edits, the closing of thin channels and necks (by a minimum node count across), the final
  unprotected pass (nothing is protected here).
- In this repo m12 already cuts a closed-sphere quad mesh (`meshom/methods/m12/coast.py`, `clean_cells`: quads only,
  at a pinch it drops the cells with the most boundary edges instead of keeping the largest fan).

**What the cut did** (continent; s1 / s4): 217 442 / 14 024 elements on the sphere, 19 507 / 1 283 removed by the mask
(9 074 / 158 of them block cells), 7 / 0 by the clean-up (all spurs; no pinch, no stray basin), 216 255 → 198 052 /
13 820 → 12 796 nodes; 3 boundary loops as the domain, Euler −4 / −4 on every run; cut + coast fit 0.5 + 2 s at s1.
With no land the cut is a no-op: `m14_blocks_cut` on `uniform` s4 equals `m14_blocks` array for array (points, tris,
quads, node_flag; test and the run's mesh).

**A vs B, continent** (exploration runs; boundary elements = elements with a boundary edge / touching
a boundary node):

| | A s4 raw | B s4 raw | A s1 raw | B s1 raw | A s1 smooth | B s1 smooth |
|---|---|---|---|---|---|---|
| gates | 22/26 | 21/26 | 24/26 | **24/24** | 24/26 | 23/24 |
| failed families, revised gates (final runs; "tri" / "all") | 2 / 3 | 4 / 4 | 3 / 4 | 2 / 2 | 3 / 4 | 2 / 2 |
| failed | ratio p99 coast 1.38, ln s/h 0.20 / coast 0.37 / int. 0.18 | θ p99.9 coast 137.8, ratio p99 coast 1.38, ln s/h 0.20 ×3 | ratio p99 coast 1.151, ln s/h coast 0.215 | – | ratio p99 coast 1.170, ln s/h coast 0.225 | ln s/h coast 0.154 |
| nodes | 12 796 | 12 791 | 198 052 | 197 567 | | |
| triangles % | 2.4 | 2.9 | 0.77 | 0.92 | | |
| n3 / n5 / T | 22 / 268 / 304 | 29 / 376 / 375 | 164 / 1579 / 1519 | 164 / 1918 / 1814 | | |
| \|ln s/h\| p99 all / coast | 0.204 / 0.369 | 0.199 / 0.204 | 0.136 / 0.215 | 0.135 / 0.143 | 0.145 / 0.225 | 0.147 / 0.154 |
| quad θmax p99.9 coast (max) | 130.1 (132.1) | 137.8 (140.1) | 126.4 (130.3) | 109.6 (111.7) | 124.7 (126.2) | 107.4 (110.6) |
| boundary nodes / elements | 416 / 321 / 466 | 335 / 335 / 347 | 1769 / 1153 / 1821 | 1348 / 1348 / 1352 | | |
| block coverage (blocks) | 0.760 (6) | 0.701 (5) | 0.885 (32) | 0.858 (18) | | |
| time s / peak MB | 4.8 / 221 | 3.3 / 374 | 54.6 / 694 | 41.5 / 533 | | |

**Staircase** (`cut.coast_fit`; B for reference): boundary node → true coast max / p99 / median **0.74 / 0.68 / 0.32
h** at s1 (s4: 0.70 / 0.67 / 0.29 h; B ≤ 3·10⁻⁵ h); true coast → boundary max / p99 0.68 / 0.46 h (s4 0.56 / 0.47;
B 0.02 / 0.09 h, the chord between resampled nodes) — no stretch of coast is missing, the channel stays open at both
scales. Water area gained (mesh on land) / lost: 0.029 % / 0.031 % of the water at s1 (0.13 % / 0.13 % at s4; net
−0.002 % / −0.005 %): a band of mean width 8.3 km (s1) / 35 km (s4) along the 34 900 km of coast, ≈ 0.17 h of the
open coast. Boundary length / coast length 1.27 (s4 1.23; ≈ 4/π, the classic staircase; B 1.000); boundary nodes
turning by > 45°: 71 % (s4 55 %; an earlier staircase study 72 % / 56 %; the real global grids 45–63 %; B 0.3 %, the
4 mouth corners).

**What the scorer's coast columns mean for a cut mesh.** `_score` and the gates do not assume that boundary nodes lie
on the resampled coast: they use only `chi_domain = 2 − len(domain.loops)`, and the `_coast` subsets are the elements
touching any boundary node (`quality/metrics.py`, `_edge_coast`). The Euler sum counts staircase corners correctly
(one quad at a convex corner +1, three at a concave one −1). So the columns are valid as the element quality of the
coastal row, but they say nothing about how the boundary represents the coast: a staircase of perfect squares scores
90° and |ln s/h| ≈ 0 in the "coast" columns. Here they happen to favour B, for a reason unrelated to the staircase:
at s1, 195 of the 206 coastal edges with |ln s/h| > 0.15 in A lie where h < 50 km (the channel and its ramp), where
A's block seams run through the channel; on the open coast (h = 50 km) A is better (p99 0.136 vs B 0.198; s4 0.170 vs
0.259). Only the m14 test helper `_check_valid` assumes coast nodes = resampled loops (B only); the cut test has its
own checks. Proposal (not done, the scorer is unchanged): a coast-fit block in every coastal record (this
`coast_fit`, moved to `meshom/quality/`): boundary → coast and coast → boundary distance in h, water area gained /
lost, boundary length ratio, share of turning boundary nodes; reported as columns, not gates, unless a fitted coast
becomes a requirement.

**Pictures** (the runs' `figures/raw/`; `zoom_island`, `zoom_opencoast` and `zoom_mouth` rendered extra):
- `zoom_opencoast.png` (cap coast, h = 50 km), `zoom_island.png`: A is the clean block lattice up to the coast, cut
  as a staircase of 1-cell steps, 0 irregular nodes and 0 triangles in view; B has the conforming coast row and,
  2–3 cells off, a dislocation line of irregular nodes and triangles where the coast front meets the block (open
  coast 3 / 26 n3 / n5, 23 triangles in view; round the island a dense ring: 6 / 72 / 2 other, 66 triangles,
  |ln s/h| p99 0.28 in view).
- `zoom_strait.png` (the 150 km channel), `zoom_mouth.png`, `zoom_coast.png`: B's channel is a perfect 10-cell
  structured strip along the channel (m02 between two parallel coasts); in A the block lattice is oblique to the
  channel (the zero-cone frame of the closed sphere knows nothing of the coast), the channel has a staircase on both
  banks and a seam of irregular nodes and triangles along its middle; at s4 (2.7 cells across) A's channel is a
  jagged 2–3-cell zigzag of fill elements, B's a clean 3-cell strip.
- `overview.png`: A's layout is the cubed-sphere-like block pattern of the closed sphere with the land cut out; the
  seams run regardless of the coast (one through the channel).

**Reading.** For an ocean model the A coast is the familiar one: a staircase offset by at most 0.74 h, with 27 %
more boundary length and a corner at 55–71 % of the boundary nodes — what NEMO, MOM and MITgcm run on, and
FESOM2's land-sea cut — and it gives the cleanest open-ocean coast here: block cells reach the coast with no front
collision, 17 % fewer irregular spots (n5 + T 3098 vs 3732 at s1; seam band 1087 vs 1475 irregular nodes) and 3
points more block coverage (6 at s4); 0 irregular nodes inside blocks in both. It costs what a
staircase costs (coast-normal transport and Kelvin waves on a zigzag, a 0.17 h band of land/water misplaced), the
lost alignment in channels (an oblique lattice and a seam inside the strait; at 2–3 cells across the strait is barely
connected), and 30 % more time and memory at s1 (54.6 vs 41.5 s: placement 20 vs 9 s for 32 instead of 18 blocks,
the land meshed and thrown away = 18 203 nodes, and 4.3 s of coast-fit measurement inside the method's time). B keeps FESOM-C's
fitted coast and perfect channels but pays a ring of irregular nodes round every coast. For FESOM3 (fitted coast,
design) neither is the answer alone; A plus a one-row fitted strip looks like the best of both.

**Open questions.**
1. Hybrid A + strip: cut, then fit one row between the staircase and the resampled coast (m12's `strip` / `snap`,
   `meshom/methods/m12/coast.py`): block cells up to one row from the coast, a fitted coast, no front collision. Try
   it as `m14_blocks_cut` + `coast="strip"`?
2. Channels in A: let the frame of the closed-sphere field follow the coast (the B4 field with coast directions but
   the blocks placed over land), or keep the conforming variant inside straits narrower than ~20 h?
3. Wet test: centroid (here) vs FESOM2's 9-sample fraction ≥ 0.5; strait rules (closing thin channels, strait edits)
   for real coasts with 1-cell straits.
4. Scorer: add the coast-fit columns above?

## Gap and shrink sweep

**Hypothesis.** The seams are bad (4–8 % irregular nodes in the seam band, |ln s/h| p99 0.21–0.24, one
dislocation line per gap where two m02 fronts collide) because the blocks sit too close: smaller blocks and wider gaps
give the free elements room to absorb the mismatch between two lattices. Tested by a parameter sweep, no new
mechanism: scale 1, the four cases, raw and after the runner's common smoothing, both fills (`m14_blocks` = m02
quad-dominant fill, `m14_blocks_tri` = triangles only).

**Knobs.** `block_gap` (h): an accepted region's outline dilated by it leaves the admissible set (realised narrowest
gap between two block outlines: 5.3 h at 6, 23.1 h at 24 on uniform). `shrink` = k (new option, default 0): each
block's cleaned cell set is eroded by k cells (3 × 3 square) and cleaned again, after the block map and before the
outline and the block smoothing; the placement is unchanged, block–block gaps grow by 2k cells and block–coast gaps by
k. No existing knob shrinks a block uniformly: `trim` drops cells by their size error (it fires where |λ − c| is
largest, centre or edge depending on the sign, and rarely, as F ≤ 0.08 < 0.10), `compact` changes the growth shape,
`min_clear` the seed count, `a_max` caps the area. The erosion acts on the cell set rather than on the region before
the map: same gaps, the λ admission and the map stay as they were, and `clean_cells` keeps one simple outline. Blocks
that fall below 100 cells or inradius 5 are dropped (disc: 24 → 15 → 11 blocks at k = 0 / 2 / 4).

**Runs.** A sweep script calls `compete.run_case` with exact method names (`meshom compete --methods m14_blocks`
also matches `_igm` and `_tri`; `meshom compete --opts JSON` passes method options); 192 compete records (gates,
quality, irregular counts) and as many m14 records (the m14 report of the raw mesh, and the same report re-measured on
the smoothed mesh: connectivity is unchanged, so the block quads and outlines are rebuilt from the mesh). Slurm: 7 jobs
of 7–30 min, 16 CPUs.

**Caveat (gate revision, 2026-10-22).** The sweep records were scored before the gate revision: their gates are the old set (no triangle-cluster, boundary, pinch or Euler gate, the coast size set with the coast edges, no coast3 / strait subsets) and their seam |ln s/h| and ratio took fill rows 0–5 for the edges (0.01–0.02 higher than rows 0–3). The comparisons between gaps hold; the gate counts are not comparable with the final competition's.

**Columns.** gates raw · smoothed; coverage = block area / water; blocks kept; tris % of all elements; irreg n (%) =
interior nodes with valence ≠ 4, count and share (the same after smoothing; "-" for the triangle fill, where every
valence-6 node counts); seam irreg % = in the seam band (outline nodes + fill rows 0–3); fill irreg % = in the whole
fill; |ln s/h| p99 of all edges / of the seam band / of the whole fill (raw · smoothed); ratio p99 = element size ratio
across edges, all / seam band; fill tris θ>100° % = fill triangles with a corner above 100°; time = generate, s. The
seam band is 4 rows wide: in a wide gap the collision line lies outside it, so it is read together with the whole
fill and the total count.

### Step 1: `block_gap` ∈ {6, 9, 12, 18, 24} h

| method | case | options | gates raw · sm | coverage | blocks | tris % | irreg n (%) | seam irreg % | fill irreg % | \|ln s/h\| p99 | seam \|ln s/h\| p99 | fill \|ln s/h\| p99 | ratio p99 | seam ratio p99 | fill tris θ>100° % | time s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m14_blocks | uniform | block_gap=6 | 18/18 · 16/18 | 0.873 | 6 | 1.1 | 664 (1.28) | 7.8 | 7.88 | 0.148 · 0.159 | 0.244 · 0.264 | 0.244 · 0.264 | 1.089 · 1.099 | 1.156 · 1.178 | 0.00 · 0.00 | 9 |
| m14_blocks | uniform | block_gap=9 | 18/18 · 16/18 | 0.821 | 6 | 1.0 | 577 (1.11) | 2.9 | 5.13 | 0.143 · 0.154 | 0.216 · 0.234 | 0.220 · 0.239 | 1.084 · 1.094 | 1.137 · 1.156 | 0.00 · 0.00 | 7 |
| m14_blocks | uniform | block_gap=12 | 16/18 · 16/18 | 0.768 | 6 | 1.2 | 764 (1.46) | 0.9 | 5.39 | 0.163 · 0.177 | 0.208 · 0.223 | 0.236 · 0.253 | 1.096 · 1.108 | 1.126 · 1.145 | 0.16 · 0.16 | 7 |
| m14_blocks | uniform | block_gap=18 | 16/18 · 16/18 | 0.655 | 6 | 1.7 | 1101 (2.09) | 0.9 | 5.45 | 0.203 · 0.217 | 0.164 · 0.174 | 0.258 · 0.274 | 1.116 · 1.133 | 1.092 · 1.105 | 0.00 · 0.00 | 8 |
| m14_blocks | uniform | block_gap=24 | 16/18 · 16/18 | 0.573 | 4 | 1.8 | 1197 (2.27) | 0.6 | 4.91 | 0.212 · 0.228 | 0.150 · 0.154 | 0.254 · 0.275 | 1.121 · 1.138 | 1.074 · 1.086 | 0.00 · 0.00 | 7 |
| m14_blocks | disc | block_gap=6 | 18/18 · 18/18 | 0.880 | 24 | 0.7 | 1795 (0.83) | 4.0 | 5.37 | 0.128 · 0.137 | 0.210 · 0.227 | 0.220 · 0.239 | 1.090 · 1.098 | 1.150 · 1.167 | 0.06 · 0.19 | 39 |
| m14_blocks | disc | block_gap=9 | 18/18 · 18/18 | 0.852 | 15 | 0.7 | 1805 (0.84) | 1.6 | 4.60 | 0.133 · 0.143 | 0.189 · 0.204 | 0.225 · 0.242 | 1.092 · 1.101 | 1.130 · 1.146 | 0.07 · 0.07 | 37 |
| m14_blocks | disc | block_gap=12 | 18/18 · 18/18 | 0.842 | 11 | 0.6 | 1514 (0.70) | 0.5 | 3.66 | 0.132 · 0.140 | 0.162 · 0.176 | 0.211 · 0.230 | 1.089 · 1.097 | 1.106 · 1.118 | 0.32 · 0.32 | 36 |
| m14_blocks | disc | block_gap=18 | 18/18 · 16/18 | 0.795 | 7 | 0.7 | 1814 (0.83) | 0.6 | 3.51 | 0.148 · 0.158 | 0.152 · 0.163 | 0.219 · 0.237 | 1.093 · 1.102 | 1.099 · 1.111 | 0.00 · 0.13 | 35 |
| m14_blocks | disc | block_gap=24 | 16/18 · 16/18 | 0.763 | 7 | 0.7 | 1799 (0.83) | 0.5 | 3.07 | 0.152 · 0.160 | 0.144 · 0.151 | 0.213 · 0.232 | 1.093 · 1.103 | 1.089 · 1.099 | 0.13 · 0.13 | 33 |
| m14_blocks | band | block_gap=6 | 18/18 · 16/18 | 0.766 | 32 | 0.8 | 4336 (1.02) | 3.9 | 5.93 | 0.140 · 0.152 | 0.220 · 0.238 | 0.241 · 0.262 | 1.111 · 1.119 | 1.175 · 1.193 | 0.22 · 0.22 | 80 |
| m14_blocks | band | block_gap=9 | 18/18 · 16/18 | 0.734 | 32 | 0.9 | 4617 (1.09) | 2.5 | 5.56 | 0.145 · 0.158 | 0.214 · 0.229 | 0.240 · 0.259 | 1.112 · 1.120 | 1.169 · 1.185 | 0.08 · 0.18 | 79 |
| m14_blocks | band | block_gap=12 | 18/18 · 16/18 | 0.704 | 32 | 0.9 | 4770 (1.12) | 1.6 | 5.20 | 0.147 · 0.160 | 0.191 · 0.209 | 0.237 · 0.257 | 1.112 · 1.121 | 1.155 · 1.170 | 0.28 · 0.33 | 79 |
| m14_blocks | band | block_gap=18 | 16/18 · 16/18 | 0.606 | 30 | 0.9 | 4553 (1.07) | 1.0 | 4.09 | 0.152 · 0.164 | 0.175 · 0.191 | 0.225 · 0.244 | 1.112 · 1.121 | 1.143 · 1.155 | 0.05 · 0.11 | 79 |
| m14_blocks | band | block_gap=24 | 16/18 · 16/18 | 0.512 | 20 | 0.9 | 4588 (1.07) | 0.9 | 3.58 | 0.176 · 0.186 | 0.175 · 0.189 | 0.236 · 0.248 | 1.113 · 1.123 | 1.143 · 1.154 | 0.16 · 0.29 | 78 |
| m14_blocks | continent | block_gap=6 | 24/24 · 23/24 | 0.858 | 18 | 0.9 | 2112 (1.08) | 6.0 | 5.83 | 0.135 · 0.147 | 0.216 · 0.236 | 0.222 · 0.242 | 1.088 · 1.099 | 1.145 · 1.164 | 0.06 · 0.17 | 41 |
| m14_blocks | continent | block_gap=9 | 22/24 · 22/24 | 0.801 | 17 | 1.0 | 2469 (1.26) | 3.2 | 5.23 | 0.151 · 0.163 | 0.202 · 0.219 | 0.223 · 0.241 | 1.093 · 1.105 | 1.137 · 1.155 | 0.14 · 0.19 | 43 |
| m14_blocks | continent | block_gap=12 | 21/24 · 21/24 | 0.788 | 14 | 1.1 | 2746 (1.39) | 2.9 | 5.48 | 0.155 · 0.168 | 0.203 · 0.218 | 0.229 · 0.247 | 1.100 · 1.111 | 1.138 · 1.155 | 0.23 · 0.28 | 40 |
| m14_blocks | continent | block_gap=18 | 21/24 · 21/24 | 0.718 | 12 | 1.1 | 2897 (1.47) | 1.5 | 4.54 | 0.172 · 0.186 | 0.176 · 0.192 | 0.232 · 0.252 | 1.104 · 1.117 | 1.110 · 1.123 | 0.27 · 0.35 | 41 |
| m14_blocks | continent | block_gap=24 | 21/24 · 21/24 | 0.717 | 10 | 0.9 | 2033 (1.03) | 1.4 | 3.19 | 0.165 · 0.173 | 0.165 · 0.178 | 0.210 · 0.226 | 1.090 · 1.101 | 1.102 · 1.115 | 0.00 · 0.06 | 38 |
| m14_blocks_tri | uniform | block_gap=6 | 16/18 · 16/18 | 0.873 | 6 | 26.4 | - | - | - | 0.211 · 0.213 | 0.272 · 0.273 | 0.284 · 0.290 | 1.135 · 1.137 | 1.183 · 1.184 | 0.04 · 0.01 | 9 |
| m14_blocks_tri | uniform | block_gap=9 | 16/18 · 16/18 | 0.820 | 6 | 34.8 | - | - | - | 0.215 · 0.217 | 0.249 · 0.249 | 0.275 · 0.277 | 1.141 · 1.144 | 1.170 · 1.172 | 0.04 · 0.01 | 7 |
| m14_blocks_tri | uniform | block_gap=12 | 16/18 · 16/18 | 0.768 | 6 | 42.3 | - | - | - | 0.226 · 0.228 | 0.257 · 0.260 | 0.273 · 0.275 | 1.147 · 1.149 | 1.171 · 1.173 | 0.02 · 0.01 | 7 |
| m14_blocks_tri | uniform | block_gap=18 | 16/18 · 16/18 | 0.654 | 6 | 55.8 | - | - | - | 0.216 · 0.218 | 0.250 · 0.253 | 0.249 · 0.251 | 1.134 · 1.137 | 1.162 · 1.165 | 0.02 · 0.02 | 8 |
| m14_blocks_tri | uniform | block_gap=24 | 16/18 · 16/18 | 0.572 | 4 | 64.0 | - | - | - | 0.219 · 0.223 | 0.241 · 0.245 | 0.243 · 0.246 | 1.133 · 1.136 | 1.159 · 1.161 | 0.02 · 0.01 | 7 |
| m14_blocks_tri | disc | block_gap=6 | 16/18 · 16/18 | 0.879 | 24 | 26.8 | - | - | - | 0.209 · 0.211 | 0.272 · 0.272 | 0.280 · 0.283 | 1.137 · 1.139 | 1.185 · 1.187 | 0.04 · 0.02 | 40 |
| m14_blocks_tri | disc | block_gap=9 | 16/18 · 16/18 | 0.852 | 15 | 31.4 | - | - | - | 0.206 · 0.209 | 0.260 · 0.260 | 0.266 · 0.268 | 1.130 · 1.132 | 1.174 · 1.177 | 0.01 · 0.01 | 40 |
| m14_blocks_tri | disc | block_gap=12 | 16/18 · 16/18 | 0.842 | 11 | 33.0 | - | - | - | 0.208 · 0.211 | 0.261 · 0.263 | 0.266 · 0.269 | 1.132 · 1.134 | 1.174 · 1.178 | 0.03 · 0.02 | 37 |
| m14_blocks_tri | disc | block_gap=18 | 16/18 · 16/18 | 0.794 | 7 | 39.5 | - | - | - | 0.209 · 0.212 | 0.261 · 0.261 | 0.258 · 0.261 | 1.129 · 1.131 | 1.175 · 1.178 | 0.02 · 0.01 | 35 |
| m14_blocks_tri | disc | block_gap=24 | 16/18 · 16/18 | 0.763 | 7 | 43.6 | - | - | - | 0.209 · 0.212 | 0.256 · 0.258 | 0.252 · 0.255 | 1.127 · 1.129 | 1.170 · 1.172 | 0.02 · 0.01 | 34 |
| m14_blocks_tri | band | block_gap=6 | 16/18 · 16/18 | 0.766 | 32 | 30.2 | - | - | - | 0.214 · 0.216 | 0.277 · 0.278 | 0.279 · 0.281 | 1.144 · 1.146 | 1.188 · 1.190 | 0.03 · 0.02 | 81 |
| m14_blocks_tri | band | block_gap=9 | 16/18 · 16/18 | 0.734 | 32 | 33.9 | - | - | - | 0.210 · 0.213 | 0.266 · 0.265 | 0.269 · 0.272 | 1.140 · 1.142 | 1.181 · 1.182 | 0.02 · 0.02 | 84 |
| m14_blocks_tri | band | block_gap=12 | 16/18 · 16/18 | 0.704 | 32 | 36.9 | - | - | - | 0.214 · 0.217 | 0.262 · 0.262 | 0.269 · 0.272 | 1.143 · 1.145 | 1.178 · 1.180 | 0.02 · 0.02 | 81 |
| m14_blocks_tri | band | block_gap=18 | 16/18 · 16/18 | 0.606 | 30 | 43.3 | - | - | - | 0.213 · 0.216 | 0.259 · 0.260 | 0.257 · 0.260 | 1.137 · 1.139 | 1.177 · 1.179 | 0.02 · 0.01 | 86 |
| m14_blocks_tri | band | block_gap=24 | 16/18 · 16/18 | 0.512 | 20 | 48.2 | - | - | - | 0.214 · 0.216 | 0.260 · 0.260 | 0.253 · 0.256 | 1.133 · 1.135 | 1.177 · 1.179 | 0.02 · 0.01 | 79 |
| m14_blocks_tri | continent | block_gap=6 | 20/23 · 18/23 | 0.858 | 18 | 31.9 | - | - | - | 0.221 · 0.221 | 0.291 · 0.291 | 0.287 · 0.289 | 1.149 · 1.151 | 1.194 · 1.197 | 0.04 · 0.03 | 42 |
| m14_blocks_tri | continent | block_gap=9 | 19/23 · 19/23 | 0.801 | 17 | 40.1 | - | - | - | 0.225 · 0.227 | 0.270 · 0.271 | 0.276 · 0.279 | 1.151 · 1.153 | 1.187 · 1.189 | 0.04 · 0.02 | 43 |
| m14_blocks_tri | continent | block_gap=12 | 18/23 · 18/23 | 0.787 | 14 | 41.9 | - | - | - | 0.229 · 0.231 | 0.277 · 0.278 | 0.280 · 0.283 | 1.154 · 1.156 | 1.189 · 1.191 | 0.05 · 0.03 | 42 |
| m14_blocks_tri | continent | block_gap=18 | 21/23 · 21/23 | 0.717 | 12 | 50.8 | - | - | - | 0.225 · 0.227 | 0.268 · 0.268 | 0.262 · 0.265 | 1.142 · 1.145 | 1.185 · 1.188 | 0.04 · 0.03 | 41 |
| m14_blocks_tri | continent | block_gap=24 | 18/23 · 18/23 | 0.716 | 10 | 50.9 | - | - | - | 0.231 · 0.233 | 0.276 · 0.276 | 0.270 · 0.272 | 1.152 · 1.155 | 1.186 · 1.188 | 0.04 · 0.03 | 39 |

Where the irregular nodes of `m14_blocks` go (raw; seam band / beyond the band / whole fill per 100 outline edges):

| case | 6 h | 9 h | 12 h | 18 h | 24 h |
|---|---|---|---|---|---|
| uniform | 650 / 14 / 28 | 289 / 288 / 26 | 87 / 677 / 35 | 76 / 1025 / 55 | 42 / 1155 / 71 |
| disc | 1057 / 738 / 26 | 388 / 1417 / 32 | 104 / 1410 / 29 | 128 / 1686 / 38 | 105 / 1694 / 37 |
| band | 1769 / 2567 / 36 | 1183 / 3434 / 40 | 756 / 4014 / 43 | 393 / 4160 / 47 | 317 / 4271 / 56 |
| continent | 1475 / 637 / 30 | 807 / 1662 / 39 | 671 / 2075 / 46 | 290 / 2607 / 58 | 256 / 1777 / 42 |

### Step 2: `shrink` k ∈ {0, 2, 4} cells at the default gap (6 h) and the best gap of step 1 (9 h, both fills)

| method | case | options | gates raw · sm | coverage | blocks | tris % | irreg n (%) | seam irreg % | fill irreg % | \|ln s/h\| p99 | seam \|ln s/h\| p99 | fill \|ln s/h\| p99 | ratio p99 | seam ratio p99 | fill tris θ>100° % | time s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m14_blocks | uniform | block_gap=6 | 18/18 · 16/18 | 0.873 | 6 | 1.1 | 664 (1.28) | 7.8 | 7.88 | 0.148 · 0.159 | 0.244 · 0.264 | 0.244 · 0.264 | 1.089 · 1.099 | 1.156 · 1.178 | 0.00 · 0.00 | 9 |
| m14_blocks | uniform | block_gap=9 | 18/18 · 16/18 | 0.821 | 6 | 1.0 | 577 (1.11) | 2.9 | 5.13 | 0.143 · 0.154 | 0.216 · 0.234 | 0.220 · 0.239 | 1.084 · 1.094 | 1.137 · 1.156 | 0.00 · 0.00 | 7 |
| m14_blocks | uniform | block_gap=6_shrink=2 | 18/18 · 16/18 | 0.795 | 6 | 1.1 | 624 (1.21) | 1.4 | 4.98 | 0.149 · 0.162 | 0.208 · 0.227 | 0.221 · 0.239 | 1.090 · 1.102 | 1.133 · 1.153 | 0.18 · 0.18 | 10 |
| m14_blocks | uniform | block_gap=6_shrink=4 | 16/18 · 16/18 | 0.723 | 6 | 1.1 | 632 (1.23) | 0.5 | 3.89 | 0.159 · 0.172 | 0.161 · 0.176 | 0.221 · 0.240 | 1.093 · 1.106 | 1.093 · 1.107 | 0.00 · 0.18 | 9 |
| m14_blocks | uniform | block_gap=9_shrink=2 | 16/18 · 16/18 | 0.746 | 6 | 1.2 | 731 (1.41) | 0.8 | 4.80 | 0.160 · 0.173 | 0.189 · 0.204 | 0.229 · 0.245 | 1.093 · 1.105 | 1.108 · 1.122 | 0.16 · 0.16 | 8 |
| m14_blocks | uniform | block_gap=9_shrink=4 | 16/18 · 16/18 | 0.675 | 6 | 1.5 | 1013 (1.96) | 0.9 | 5.34 | 0.189 · 0.205 | 0.157 · 0.168 | 0.249 · 0.272 | 1.114 · 1.129 | 1.097 · 1.110 | 0.25 · 0.50 | 8 |
| m14_blocks | disc | block_gap=6 | 18/18 · 18/18 | 0.880 | 24 | 0.7 | 1795 (0.83) | 4.0 | 5.37 | 0.128 · 0.137 | 0.210 · 0.227 | 0.220 · 0.239 | 1.090 · 1.098 | 1.150 · 1.167 | 0.06 · 0.19 | 39 |
| m14_blocks | disc | block_gap=9 | 18/18 · 18/18 | 0.852 | 15 | 0.7 | 1805 (0.84) | 1.6 | 4.60 | 0.133 · 0.143 | 0.189 · 0.204 | 0.225 · 0.242 | 1.092 · 1.101 | 1.130 · 1.146 | 0.07 · 0.07 | 37 |
| m14_blocks | disc | block_gap=6_shrink=2 | 18/18 · 18/18 | 0.823 | 15 | 0.7 | 1827 (0.85) | 0.7 | 3.96 | 0.136 · 0.147 | 0.161 · 0.173 | 0.217 · 0.234 | 1.093 · 1.102 | 1.101 · 1.115 | 0.13 · 0.13 | 40 |
| m14_blocks | disc | block_gap=6_shrink=4 | 16/18 · 16/18 | 0.775 | 11 | 0.7 | 1934 (0.90) | 0.3 | 3.42 | 0.152 · 0.162 | 0.134 · 0.140 | 0.219 · 0.237 | 1.095 · 1.104 | 1.064 · 1.073 | 0.13 · 0.19 | 41 |
| m14_blocks | disc | block_gap=9_shrink=2 | 18/18 · 16/18 | 0.805 | 11 | 0.7 | 1897 (0.88) | 0.4 | 3.82 | 0.148 · 0.158 | 0.149 · 0.159 | 0.224 · 0.242 | 1.094 · 1.104 | 1.084 · 1.095 | 0.13 · 0.19 | 38 |
| m14_blocks | disc | block_gap=9_shrink=4 | 16/18 · 16/18 | 0.763 | 8 | 0.7 | 1907 (0.89) | 0.2 | 3.24 | 0.154 · 0.163 | 0.134 · 0.135 | 0.223 · 0.240 | 1.095 · 1.105 | 1.050 · 1.058 | 0.27 · 0.27 | 38 |
| m14_blocks | band | block_gap=6 | 18/18 · 16/18 | 0.766 | 32 | 0.8 | 4336 (1.02) | 3.9 | 5.93 | 0.140 · 0.152 | 0.220 · 0.238 | 0.241 · 0.262 | 1.111 · 1.119 | 1.175 · 1.193 | 0.22 · 0.22 | 80 |
| m14_blocks | band | block_gap=9 | 18/18 · 16/18 | 0.734 | 32 | 0.9 | 4617 (1.09) | 2.5 | 5.56 | 0.145 · 0.158 | 0.214 · 0.229 | 0.240 · 0.259 | 1.112 · 1.120 | 1.169 · 1.185 | 0.08 · 0.18 | 79 |
| m14_blocks | band | block_gap=6_shrink=2 | 18/18 · 16/18 | 0.680 | 32 | 0.8 | 4377 (1.03) | 0.8 | 4.57 | 0.148 · 0.160 | 0.173 · 0.190 | 0.230 · 0.250 | 1.110 · 1.119 | 1.121 · 1.135 | 0.28 · 0.25 | 81 |
| m14_blocks | band | block_gap=6_shrink=4 | 16/18 · 16/18 | 0.605 | 32 | 0.9 | 4695 (1.11) | 0.4 | 4.04 | 0.152 · 0.165 | 0.127 · 0.138 | 0.224 · 0.243 | 1.110 · 1.119 | 1.075 · 1.086 | 0.05 · 0.13 | 83 |
| m14_blocks | band | block_gap=9_shrink=2 | 16/18 · 16/18 | 0.653 | 32 | 0.9 | 4696 (1.10) | 0.7 | 4.48 | 0.151 · 0.164 | 0.155 · 0.169 | 0.228 · 0.248 | 1.111 · 1.121 | 1.106 · 1.117 | 0.24 · 0.32 | 81 |
| m14_blocks | band | block_gap=9_shrink=4 | 16/18 · 16/18 | 0.584 | 32 | 0.9 | 4428 (1.04) | 0.3 | 3.58 | 0.153 · 0.165 | 0.115 · 0.123 | 0.219 · 0.237 | 1.108 · 1.117 | 1.065 · 1.072 | 0.24 · 0.33 | 82 |
| m14_blocks | continent | block_gap=6 | 24/24 · 23/24 | 0.858 | 18 | 0.9 | 2112 (1.08) | 6.0 | 5.83 | 0.135 · 0.147 | 0.216 · 0.236 | 0.222 · 0.242 | 1.088 · 1.099 | 1.145 · 1.164 | 0.06 · 0.17 | 41 |
| m14_blocks | continent | block_gap=9 | 22/24 · 22/24 | 0.801 | 17 | 1.0 | 2469 (1.26) | 3.2 | 5.23 | 0.151 · 0.163 | 0.202 · 0.219 | 0.223 · 0.241 | 1.093 · 1.105 | 1.137 · 1.155 | 0.14 · 0.19 | 43 |
| m14_blocks | continent | block_gap=6_shrink=2 | 24/24 · 22/24 | 0.812 | 15 | 1.1 | 2699 (1.38) | 3.2 | 5.96 | 0.149 · 0.163 | 0.212 · 0.230 | 0.231 · 0.248 | 1.097 · 1.109 | 1.138 · 1.156 | 0.00 · 0.05 | 41 |
| m14_blocks | continent | block_gap=6_shrink=4 | 22/24 · 22/24 | 0.749 | 13 | 1.0 | 2338 (1.20) | 1.2 | 4.07 | 0.154 · 0.167 | 0.166 · 0.180 | 0.218 · 0.236 | 1.095 · 1.107 | 1.103 · 1.116 | 0.00 · 0.10 | 42 |
| m14_blocks | continent | block_gap=9_shrink=2 | 22/24 · 22/24 | 0.780 | 15 | 1.2 | 3004 (1.53) | 2.3 | 5.82 | 0.165 · 0.178 | 0.195 · 0.213 | 0.240 · 0.260 | 1.104 · 1.116 | 1.124 · 1.141 | 0.12 · 0.25 | 41 |
| m14_blocks | continent | block_gap=9_shrink=4 | 22/24 · 22/24 | 0.720 | 13 | 1.2 | 2888 (1.48) | 1.3 | 4.59 | 0.168 · 0.182 | 0.173 · 0.188 | 0.229 · 0.248 | 1.104 · 1.116 | 1.110 · 1.125 | 0.17 · 0.21 | 42 |
| m14_blocks_tri | uniform | block_gap=6 | 16/18 · 16/18 | 0.873 | 6 | 26.4 | - | - | - | 0.211 · 0.213 | 0.272 · 0.273 | 0.284 · 0.290 | 1.135 · 1.137 | 1.183 · 1.184 | 0.04 · 0.01 | 9 |
| m14_blocks_tri | uniform | block_gap=9 | 16/18 · 16/18 | 0.820 | 6 | 34.8 | - | - | - | 0.215 · 0.217 | 0.249 · 0.249 | 0.275 · 0.277 | 1.141 · 1.144 | 1.170 · 1.172 | 0.04 · 0.01 | 7 |
| m14_blocks_tri | uniform | block_gap=6_shrink=2 | 16/18 · 14/18 | 0.795 | 6 | 38.8 | - | - | - | 0.228 · 0.229 | 0.262 · 0.265 | 0.278 · 0.282 | 1.149 · 1.152 | 1.174 · 1.175 | 0.04 · 0.04 | 9 |
| m14_blocks_tri | uniform | block_gap=6_shrink=4 | 16/18 · 16/18 | 0.722 | 6 | 48.4 | - | - | - | 0.228 · 0.231 | 0.256 · 0.256 | 0.270 · 0.271 | 1.146 · 1.149 | 1.169 · 1.171 | 0.04 · 0.03 | 9 |
| m14_blocks_tri | uniform | block_gap=9_shrink=2 | 16/18 · 14/18 | 0.745 | 6 | 45.3 | - | - | - | 0.230 · 0.231 | 0.263 · 0.265 | 0.274 · 0.278 | 1.148 · 1.151 | 1.173 · 1.174 | 0.04 · 0.03 | 8 |
| m14_blocks_tri | uniform | block_gap=9_shrink=4 | 16/18 · 16/18 | 0.674 | 6 | 53.9 | - | - | - | 0.224 · 0.228 | 0.261 · 0.261 | 0.261 · 0.263 | 1.142 · 1.144 | 1.169 · 1.171 | 0.03 · 0.03 | 10 |
| m14_blocks_tri | disc | block_gap=6 | 16/18 · 16/18 | 0.879 | 24 | 26.8 | - | - | - | 0.209 · 0.211 | 0.272 · 0.272 | 0.280 · 0.283 | 1.137 · 1.139 | 1.185 · 1.187 | 0.04 · 0.02 | 40 |
| m14_blocks_tri | disc | block_gap=9 | 16/18 · 16/18 | 0.852 | 15 | 31.4 | - | - | - | 0.206 · 0.209 | 0.260 · 0.260 | 0.266 · 0.268 | 1.130 · 1.132 | 1.174 · 1.177 | 0.01 · 0.01 | 40 |
| m14_blocks_tri | disc | block_gap=6_shrink=2 | 16/18 · 16/18 | 0.823 | 15 | 36.1 | - | - | - | 0.220 · 0.222 | 0.272 · 0.273 | 0.277 · 0.280 | 1.145 · 1.148 | 1.182 · 1.184 | 0.03 · 0.02 | 42 |
| m14_blocks_tri | disc | block_gap=6_shrink=4 | 16/18 · 16/18 | 0.775 | 11 | 43.2 | - | - | - | 0.218 · 0.220 | 0.273 · 0.276 | 0.265 · 0.267 | 1.137 · 1.139 | 1.178 · 1.181 | 0.04 · 0.03 | 42 |
| m14_blocks_tri | disc | block_gap=9_shrink=2 | 16/18 · 16/18 | 0.805 | 11 | 38.7 | - | - | - | 0.216 · 0.218 | 0.272 · 0.272 | 0.271 · 0.272 | 1.138 · 1.140 | 1.178 · 1.179 | 0.04 · 0.03 | 39 |
| m14_blocks_tri | disc | block_gap=9_shrink=4 | 16/18 · 16/18 | 0.763 | 8 | 44.7 | - | - | - | 0.219 · 0.222 | 0.271 · 0.272 | 0.263 · 0.266 | 1.135 · 1.137 | 1.178 · 1.179 | 0.03 · 0.02 | 41 |
| m14_blocks_tri | band | block_gap=6 | 16/18 · 16/18 | 0.766 | 32 | 30.2 | - | - | - | 0.214 · 0.216 | 0.277 · 0.278 | 0.279 · 0.281 | 1.144 · 1.146 | 1.188 · 1.190 | 0.03 · 0.02 | 81 |
| m14_blocks_tri | band | block_gap=9 | 16/18 · 16/18 | 0.734 | 32 | 33.9 | - | - | - | 0.210 · 0.213 | 0.266 · 0.265 | 0.269 · 0.272 | 1.140 · 1.142 | 1.181 · 1.182 | 0.02 · 0.02 | 84 |
| m14_blocks_tri | band | block_gap=6_shrink=2 | 16/18 · 14/18 | 0.680 | 32 | 38.2 | - | - | - | 0.220 · 0.223 | 0.271 · 0.272 | 0.274 · 0.276 | 1.148 · 1.151 | 1.181 · 1.183 | 0.03 · 0.02 | 85 |
| m14_blocks_tri | band | block_gap=6_shrink=4 | 16/18 · 16/18 | 0.605 | 32 | 44.9 | - | - | - | 0.222 · 0.225 | 0.269 · 0.270 | 0.267 · 0.269 | 1.146 · 1.148 | 1.179 · 1.181 | 0.03 · 0.02 | 88 |
| m14_blocks_tri | band | block_gap=9_shrink=2 | 16/18 · 14/18 | 0.653 | 32 | 41.3 | - | - | - | 0.221 · 0.224 | 0.268 · 0.268 | 0.271 · 0.273 | 1.148 · 1.150 | 1.181 · 1.182 | 0.04 · 0.03 | 85 |
| m14_blocks_tri | band | block_gap=9_shrink=4 | 16/18 · 16/18 | 0.584 | 32 | 47.4 | - | - | - | 0.221 · 0.224 | 0.263 · 0.264 | 0.263 · 0.266 | 1.144 · 1.146 | 1.176 · 1.178 | 0.02 · 0.02 | 86 |
| m14_blocks_tri | continent | block_gap=6 | 20/23 · 18/23 | 0.858 | 18 | 31.9 | - | - | - | 0.221 · 0.221 | 0.291 · 0.291 | 0.287 · 0.289 | 1.149 · 1.151 | 1.194 · 1.197 | 0.04 · 0.03 | 42 |
| m14_blocks_tri | continent | block_gap=9 | 19/23 · 19/23 | 0.801 | 17 | 40.1 | - | - | - | 0.225 · 0.227 | 0.270 · 0.271 | 0.276 · 0.279 | 1.151 · 1.153 | 1.187 · 1.189 | 0.04 · 0.02 | 43 |
| m14_blocks_tri | continent | block_gap=6_shrink=2 | 19/23 · 19/23 | 0.812 | 15 | 38.7 | - | - | - | 0.231 · 0.233 | 0.286 · 0.286 | 0.287 · 0.289 | 1.159 · 1.161 | 1.192 · 1.192 | 0.02 · 0.02 | 42 |
| m14_blocks_tri | continent | block_gap=6_shrink=4 | 19/23 · 19/23 | 0.748 | 13 | 47.4 | - | - | - | 0.231 · 0.233 | 0.284 · 0.283 | 0.275 · 0.276 | 1.154 · 1.156 | 1.188 · 1.189 | 0.04 · 0.03 | 44 |
| m14_blocks_tri | continent | block_gap=9_shrink=2 | 19/23 · 19/23 | 0.780 | 15 | 43.0 | - | - | - | 0.234 · 0.236 | 0.285 · 0.286 | 0.283 · 0.285 | 1.158 · 1.161 | 1.188 · 1.190 | 0.03 · 0.02 | 43 |
| m14_blocks_tri | continent | block_gap=9_shrink=4 | 19/23 · 19/23 | 0.720 | 13 | 50.8 | - | - | - | 0.235 · 0.237 | 0.284 · 0.284 | 0.274 · 0.277 | 1.156 · 1.158 | 1.190 · 1.191 | 0.03 · 0.02 | 43 |

### Step 3: the triangle fill's size fidelity (default gap)

`fill={"size_iters": 60 | 120}` (2×, 4× m02's 30), `band_size_iters=40` (2×), both together; and, after the
diagnostic below, m02's existing valence flips `fill={"flips": "valence"}` (m11's Botsch–Kobbelt flips towards valence
6, new angles kept within 30–110°, before the smoothing), alone and with 120 sweeps.

| method | case | options | gates raw · sm | coverage | blocks | tris % | irreg n (%) | seam irreg % | fill irreg % | \|ln s/h\| p99 | seam \|ln s/h\| p99 | fill \|ln s/h\| p99 | ratio p99 | seam ratio p99 | fill tris θ>100° % | time s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m14_blocks_tri | uniform | block_gap=6 | 16/18 · 16/18 | 0.873 | 6 | 26.4 | - | - | - | 0.211 · 0.213 | 0.272 · 0.273 | 0.284 · 0.290 | 1.135 · 1.137 | 1.183 · 1.184 | 0.04 · 0.01 | 9 |
| m14_blocks_tri | uniform | fill.size_iters=60 | 16/18 · 16/18 | 0.873 | 6 | 26.4 | - | - | - | 0.210 · 0.213 | 0.272 · 0.274 | 0.285 · 0.290 | 1.135 · 1.137 | 1.183 · 1.184 | 0.04 · 0.01 | 8 |
| m14_blocks_tri | uniform | fill.size_iters=120 | 16/18 · 16/18 | 0.873 | 6 | 26.4 | - | - | - | 0.210 · 0.213 | 0.272 · 0.274 | 0.286 · 0.289 | 1.135 · 1.137 | 1.183 · 1.184 | 0.04 · 0.01 | 9 |
| m14_blocks_tri | uniform | band_size_iters=40 | 16/18 · 16/18 | 0.873 | 6 | 26.4 | - | - | - | 0.211 · 0.213 | 0.272 · 0.273 | 0.284 · 0.290 | 1.135 · 1.136 | 1.184 · 1.184 | 0.04 · 0.01 | 9 |
| m14_blocks_tri | uniform | band_size_iters=40_fill.size_iters=120 | 16/18 · 16/18 | 0.873 | 6 | 26.4 | - | - | - | 0.210 · 0.213 | 0.273 · 0.274 | 0.286 · 0.289 | 1.135 · 1.137 | 1.183 · 1.184 | 0.04 · 0.01 | 9 |
| m14_blocks_tri | uniform | fill.flips=valence | 16/18 · 16/18 | 0.873 | 6 | 26.4 | - | - | - | 0.206 · 0.209 | 0.266 · 0.268 | 0.271 · 0.274 | 1.125 · 1.127 | 1.173 · 1.177 | 0.02 · 0.01 | 8 |
| m14_blocks_tri | uniform | fill.flips=valence_fill.size_iters=120 | 16/18 · 16/18 | 0.873 | 6 | 26.4 | - | - | - | 0.206 · 0.210 | 0.267 · 0.269 | 0.272 · 0.273 | 1.125 · 1.127 | 1.174 · 1.178 | 0.02 · 0.01 | 11 |
| m14_blocks_tri | disc | block_gap=6 | 16/18 · 16/18 | 0.879 | 24 | 26.8 | - | - | - | 0.209 · 0.211 | 0.272 · 0.272 | 0.280 · 0.283 | 1.137 · 1.139 | 1.185 · 1.187 | 0.04 · 0.02 | 40 |
| m14_blocks_tri | disc | fill.size_iters=60 | 16/18 · 16/18 | 0.879 | 24 | 26.8 | - | - | - | 0.208 · 0.211 | 0.273 · 0.272 | 0.279 · 0.283 | 1.137 · 1.139 | 1.184 · 1.187 | 0.04 · 0.03 | 41 |
| m14_blocks_tri | disc | fill.size_iters=120 | 16/18 · 16/18 | 0.879 | 24 | 26.8 | - | - | - | 0.208 · 0.211 | 0.273 · 0.272 | 0.279 · 0.282 | 1.137 · 1.139 | 1.184 · 1.187 | 0.04 · 0.03 | 41 |
| m14_blocks_tri | disc | band_size_iters=40 | 16/18 · 16/18 | 0.879 | 24 | 26.8 | - | - | - | 0.209 · 0.211 | 0.272 · 0.272 | 0.280 · 0.283 | 1.137 · 1.139 | 1.185 · 1.187 | 0.04 · 0.03 | 41 |
| m14_blocks_tri | disc | band_size_iters=40_fill.size_iters=120 | 16/18 · 16/18 | 0.879 | 24 | 26.8 | - | - | - | 0.208 · 0.211 | 0.273 · 0.272 | 0.279 · 0.281 | 1.137 · 1.139 | 1.184 · 1.187 | 0.04 · 0.03 | 44 |
| m14_blocks_tri | disc | fill.flips=valence | 16/18 · 16/18 | 0.880 | 24 | 26.8 | - | - | - | 0.203 · 0.206 | 0.263 · 0.267 | 0.267 · 0.270 | 1.126 · 1.128 | 1.175 · 1.177 | 0.01 · 0.01 | 41 |
| m14_blocks_tri | disc | fill.flips=valence_fill.size_iters=120 | 16/18 · 16/18 | 0.880 | 24 | 26.8 | - | - | - | 0.202 · 0.205 | 0.264 · 0.267 | 0.266 · 0.270 | 1.126 · 1.128 | 1.175 · 1.176 | 0.02 · 0.01 | 43 |
| m14_blocks_tri | band | block_gap=6 | 16/18 · 16/18 | 0.766 | 32 | 30.2 | - | - | - | 0.214 · 0.216 | 0.277 · 0.278 | 0.279 · 0.281 | 1.144 · 1.146 | 1.188 · 1.190 | 0.03 · 0.02 | 81 |
| m14_blocks_tri | band | fill.size_iters=60 | 16/18 · 16/18 | 0.766 | 32 | 30.2 | - | - | - | 0.213 · 0.216 | 0.277 · 0.278 | 0.279 · 0.281 | 1.144 · 1.146 | 1.188 · 1.190 | 0.03 · 0.02 | 82 |
| m14_blocks_tri | band | fill.size_iters=120 | 16/18 · 16/18 | 0.766 | 32 | 30.2 | - | - | - | 0.213 · 0.215 | 0.277 · 0.278 | 0.279 · 0.281 | 1.144 · 1.146 | 1.188 · 1.190 | 0.03 · 0.02 | 86 |
| m14_blocks_tri | band | band_size_iters=40 | 16/18 · 16/18 | 0.766 | 32 | 30.2 | - | - | - | 0.214 · 0.216 | 0.277 · 0.278 | 0.279 · 0.281 | 1.144 · 1.146 | 1.188 · 1.190 | 0.03 · 0.02 | 88 |
| m14_blocks_tri | band | band_size_iters=40_fill.size_iters=120 | 16/18 · 16/18 | 0.766 | 32 | 30.2 | - | - | - | 0.213 · 0.215 | 0.277 · 0.278 | 0.279 · 0.281 | 1.144 · 1.146 | 1.188 · 1.190 | 0.03 · 0.02 | 90 |
| m14_blocks_tri | band | fill.flips=valence | 16/18 · 16/18 | 0.766 | 32 | 30.2 | - | - | - | 0.207 · 0.211 | 0.270 · 0.273 | 0.268 · 0.271 | 1.135 · 1.137 | 1.179 · 1.182 | 0.02 · 0.01 | 82 |
| m14_blocks_tri | band | fill.flips=valence_fill.size_iters=120 | 16/18 · 16/18 | 0.766 | 32 | 30.2 | - | - | - | 0.207 · 0.210 | 0.271 · 0.273 | 0.267 · 0.271 | 1.135 · 1.137 | 1.180 · 1.182 | 0.02 · 0.01 | 85 |
| m14_blocks_tri | continent | block_gap=6 | 20/23 · 18/23 | 0.858 | 18 | 31.9 | - | - | - | 0.221 · 0.221 | 0.291 · 0.291 | 0.287 · 0.289 | 1.149 · 1.151 | 1.194 · 1.197 | 0.04 · 0.03 | 42 |
| m14_blocks_tri | continent | fill.size_iters=60 | 20/23 · 18/23 | 0.858 | 18 | 31.9 | - | - | - | 0.220 · 0.221 | 0.291 · 0.291 | 0.287 · 0.289 | 1.149 · 1.151 | 1.195 · 1.197 | 0.04 · 0.02 | 42 |
| m14_blocks_tri | continent | fill.size_iters=120 | 20/23 · 18/23 | 0.858 | 18 | 31.9 | - | - | - | 0.220 · 0.221 | 0.291 · 0.292 | 0.287 · 0.289 | 1.149 · 1.151 | 1.195 · 1.197 | 0.04 · 0.02 | 46 |
| m14_blocks_tri | continent | band_size_iters=40 | 20/23 · 18/23 | 0.858 | 18 | 31.9 | - | - | - | 0.221 · 0.221 | 0.291 · 0.291 | 0.287 · 0.289 | 1.149 · 1.151 | 1.195 · 1.197 | 0.04 · 0.03 | 46 |
| m14_blocks_tri | continent | band_size_iters=40_fill.size_iters=120 | 20/23 · 18/23 | 0.858 | 18 | 31.9 | - | - | - | 0.220 · 0.221 | 0.291 · 0.291 | 0.287 · 0.289 | 1.149 · 1.151 | 1.195 · 1.197 | 0.04 · 0.02 | 47 |
| m14_blocks_tri | continent | fill.flips=valence | 20/23 · 20/23 | 0.858 | 18 | 31.9 | - | - | - | 0.215 · 0.217 | 0.286 · 0.289 | 0.277 · 0.280 | 1.137 · 1.140 | 1.186 · 1.189 | 0.03 · 0.02 | 42 |
| m14_blocks_tri | continent | fill.flips=valence_fill.size_iters=120 | 20/23 · 20/23 | 0.858 | 18 | 31.9 | - | - | - | 0.214 · 0.217 | 0.286 · 0.289 | 0.277 · 0.280 | 1.137 · 1.139 | 1.186 · 1.188 | 0.03 · 0.02 | 47 |

Diagnostic of the default triangle fill (raw, uniform / disc / band / continent): 30 / 28 / 26 / 30 % of the fill nodes have valence 5 or 7 (2046 of 6830, 8461 of 30441, 18932 of 71463, 10188 of 34005).
Edges between two valence-6 nodes: |ln s/h| p99 0.154 / 0.160 / 0.161 / 0.155 (36–47 % of the fill edges); edges
touching a valence-5/7 node: 0.328 / 0.316 / 0.320 / 0.326 (37–39 %); outline and coast edges 0.245–0.254. Per
triangle, the size of the equal-area equilateral triangle: |ln(a_eq/h)| p99 0.10 for triangles with three valence-6
corners, 0.20 over all. The size smoother is at its fixed point after 30 sweeps; the error sits at the 5/7 nodes of
the packing (picture: orange stars of long and short spokes around them, all over the gap, not at the outlines).
Valence flips cut the 5/7 nodes to 26 / 24 / 23 / 25 % and the p99 by 0.005–0.007.

### Best setting per case

| fill | case | default (6 h) | best | best: gates · coverage · irregular (seam band) · seam \|ln s/h\| · seam ratio |
|---|---|---|---|---|
| m14_blocks | uniform | 18/18 · 0.873 · 664 (7.8 %) · 0.244 · 1.156 | `block_gap=9` | 18/18 · 0.821 · 577 (2.9 %) · 0.216 · 1.137 |
| m14_blocks | disc | 18/18 · 0.880 · 1795 (4.0 %) · 0.210 · 1.150 | `block_gap=12` | 18/18 · 0.842 · 1514 (0.5 %) · 0.162 · 1.106 |
| m14_blocks | band | 18/18 · 0.766 · 4336 (3.9 %) · 0.220 · 1.175 | default | (cleanest seam at 18/18: `block_gap=6, shrink=2`: 0.680 · 4377 (0.8 %) · 0.173 · 1.121) |
| m14_blocks | continent | 24/24 · 0.858 · 2112 (6.0 %) · 0.216 · 1.145 | default | (cleanest seam at 24/24: `block_gap=6, shrink=2`: 0.812 · 2699 (3.2 %) · 0.212 · 1.138) |
| m14_blocks_tri | uniform | 16/18 · \|ln s/h\| p99 0.211 · seam 0.272 · 26.4 % tris | `fill.flips=valence` | 16/18 · 0.206 · 0.266 · 26.4 % |
| m14_blocks_tri | disc | 16/18 · 0.209 · 0.272 · 26.8 % | `fill.flips=valence`, `size_iters=120` | 16/18 · 0.202 · 0.264 · 26.8 % |
| m14_blocks_tri | band | 16/18 · 0.214 · 0.277 · 30.2 % | `fill.flips=valence`, `size_iters=120` | 16/18 · 0.207 · 0.271 · 30.2 % |
| m14_blocks_tri | continent | 20/23 (sm 18) · 0.221 · 0.291 · 31.9 % | `fill.flips=valence`, `size_iters=120` | 20/23 (sm 20) · 0.214 · 0.286 · 31.9 % |

Criterion: `m14_blocks`, the fewest irregular nodes among the settings of steps 1–2 that keep the default's raw gate
count (ties: coverage); `m14_blocks_tri`, the lowest global |ln s/h| p99 (its one failing gate).

### Reading

- **The seam band clears, the defects stay.** A wider gap (or a shrink) empties the 4 rows next to the outline:
  irregular 4–8 % → 0.5–1.5 % at 18–24 h or k = 4, seam |ln s/h| p99 0.21–0.24 → 0.14–0.18 at 18–24 h (0.12–0.17
  at k = 4), seam ratio p99 1.15–1.18 → 1.05–1.14. But the irregular nodes move out of the band about as fast as they leave it
  (uniform 650 + 14 at 6 h, 42 + 1155 at 24 h): the total falls by at most 13–16 % and otherwise stays or grows (uniform 664 → 577 at 9 h, then 764–1197;
  disc 1514–1814; band 4336 → 4553–4770; continent 2112 → 2469–2897, 2033 at 24 h), and per 100 outline edges it rises from
  26–36 to 37–71. The pictures show why: at 24 h the two fronts still meet in one dislocation line (valence-5 +
  triangle pairs) in the middle of the gap, as at 6 h; the number of dislocations on it follows the row-count mismatch
  of the two lattices (two nearly aligned lattices 14 h apart: a handful), not the room between them, and m02's own
  noise (fill |ln s/h| p99 0.21–0.26) now covers a wider strip. The hypothesis holds for the band next to the outline,
  not for the seam as a whole.
- **Price.** Coverage falls 1–6 points per 3 h of gap (uniform 0.873 → 0.573 at 24 h, band 0.766 → 0.512); the water given
  up goes to the less size-faithful m02 fill, so the global |ln s/h| p99 rises with the fill share and crosses 0.15 at
  9–12 h on uniform and continent, 18–24 h on disc and band: gates 18/18 → 16/18, 24/24 → 21/24 (size fidelity only).
  Shrinking also drops small blocks (disc 24 → 11) and on continent adds 11–28 % irregular nodes.
- **Optimum, quad fill:** 9 h on uniform (−13 % irregular nodes, 5 points of coverage), 12 h on disc (−16 %, 4
  points), the default 6 h on band and continent; `block_gap=6, shrink=2` keeps the raw gates on all four cases and clears the
  seam band to 1.4 / 0.7 / 0.8 / 3.2 % for 5–9 points of coverage (disc: 9 h + shrink 2 is cleaner, 0.4 %). No single wider gap is a win; the seam needs a fill that
  interpolates between the two lattices (see "Production would need"), not more room.
- **Triangle fill:** the gap and the shrink do not move its failing gate (|ln s/h| p99 0.21–0.24 at every gap and shrink); a
  wider gap only trades quads for triangles (26–32 % → 44–64 % at 24 h) for a 0.01–0.03 lower seam |ln s/h|, and the
  shrink makes the global value worse (0.216–0.235). Optimum: the default gap. More size smoothing changes nothing
  (≤ 0.001: 30 sweeps are the fixed point); the residual sits at the 26–30 % valence-5/7 nodes of the packing, valence
  flips gain 0.005–0.007. Reaching 0.15 needs a triangle fill with far fewer 5/7 nodes, not a smoothing setting.

Figures looked at (the runs' `figures/raw/`; `zoom_gap.png`, the narrowest gap between two blocks, drawn by the
sweep script): `m14_blocks` uniform at `block_gap=6` and `block_gap=24` (one collision line at both widths), uniform at
`block_gap=6, shrink=4` (aligned lattices, few dislocations), band at 6 and 18 h, disc at 12 h (the narrowest gap is
the ramp), `m14_blocks_tri` band at 24 h (5/7 stars). The full picture sets (overview, zoom_angle/size/irregular/seam,
raw and smoothed) were made for uniform 6 / 9 / 24 / 6 + shrink 4, disc 6 / 12, band 6 / 18 (`m14_blocks`) and all
four cases at 6 h, uniform and band at 24 h (`_tri`).

## `m14_blocks_mix`: one fill per corridor class (2026-10-22)

The idea: stitch the blocks with different fills for different conditions instead of one m02 front collision per gap.
Code `meshom/methods/m14/stitch_mix.py` (dispatched from `stitch.py` by `fill_method="mix"`, options in
`m14_blocks.DEFAULTS`, registered as `m14_blocks_mix`); test `tests/test_m14_mix.py`;
every run's `m14.json` gets `regions` (one record per region) and `m14_elem_region.npy` (the region of each element).
Blocks and placement are those of `m14_blocks`, unchanged.

**Partition** (`partition`). The fill domain (water minus blocks) gets a background mesh at 1 h (the coast resampled at
h, every loop taken as given). Objects = the block outlines and the coast loops. A Dijkstra on the dual graph of the
background triangles (centroid hops, length / h, so distances run inside the water and are counted in cells) keeps the
three nearest distinct objects per triangle, d1 ≤ d2 ≤ d3. A triangle is **open** if d1 + d2 > `w_open` = 14 h (a hole
in the layout), else a **junction** if d3 − d2 < `j_margin` = 2 h (a third object about as near as the second: the Y
where three blocks meet, with thin arms along the lines where the second-nearest block changes, ending at the block
corners), else a **corridor** of the pair {o1, o2} (a **coast** corridor when one of them is a coast loop; a channel is
the pair of its two coast loops). Regions = edge-connected components of equal labels; components of < 6 triangles and
corridors with < 2 edges on one side are merged into the neighbour sharing the most edges. Every background triangle is
in exactly one region (test: the regions' areas plus the block quads' areas = 4π to 10⁻⁶).

**Shared boundaries** (`resample_separators`). Separators = background edges between two regions, chained between
branch nodes (outline / coast nodes and nodes where three regions meet), smoothed (3 Laplacian passes, ends fixed) and
resampled at h; they become constraint edges of one sphere triangulation, with the coast and outline nodes, so each
region is meshed with its boundary nodes given (the exact-loop mechanism of `Domain(exact=...)`). The two caps of a
strip corridor are instead straight chords between the ends of its two outline runs (a cap made of several separators:
one chord, the interior branch nodes moved onto it and the separators ending there following them); a chord that would
cross another constraint, or that turns a thin neighbour inside out (caught by the triangulation: a component whose
separator half-edges name two regions), takes the corridor back to "no strip". Packed points are assigned to regions by
that constrained triangulation; m02's recombination never pairs triangles across a separator.

**Classes** (`corridor_geometry`), per block-block corridor: its two sides (one contiguous outline run per block);
**rotation** = the relative rotation of the two lattices mod 90° (4-fold circular mean of the angle between each side-A
edge and the nearest side-B edge, in the tangent plane at their midpoint; outline edges are lattice edges); **mismatch**
= |nA − nB| of the along edges, per 10 cells of length; **width** = median distance from side A to side B in h;
**obliquity** of each side = its lattice's angle to the gap (4-fold mean of its edges against its smoothed tangent);
**coast** = the region touches a coast loop (coast class). Each side's edges are classified as along / step toward /
step away (against the local lattice axis nearest to the run's chord): a lattice path.

**The three fills.**
- **strip** (`_strip_plan`, `strip_patch`): the simplest correct strip between two lattice-path sides, a ladder of
  columns. Each column pairs one along edge of each side (where the counts differ, the |nA − nB| doubled columns,
  evenly spaced, are column terminations: an extra line from the middle node to the middle row, closed by one quad and
  one triangle). Column line s runs from side A's topmost node there to side B's lowest; its free part has e_s edges
  (e at the two ends = the caps), chosen by dynamic programming: fewest row terminations Σ|R_s − L_s| + Σ(e_s − w_s)²
  (w_s = the line's length in h, ≤ 2 terminations per column). A side's lattice step ends a row at the outline (absorbed
  by the corner quads, no defect); unmatched steps of the two sides and width changes are row terminations (one triangle
  and one valence-5 node each, in the middle rows). Interior nodes on the straight line between the two side nodes of
  their column, at the level fraction. Straight parallel sides with equal counts: all quads, every node regular; a
  mismatch of 2: exactly 2 triangles (tests). Eligible when both sides are lattice paths, both obliquities < 20°, the
  separators form two caps, end steps that would run along a cap are handed to the region beyond it, and a plan exists
  (column terminations ≤ 0.3 of the shorter side).
- **m02**: m02's packing along its cross field (seeded with every coast, outline and separator node), Delaunay,
  recombination, its smoothing (30 "size" + 3 "angle" sweeps).
- **tri**: the same packing with the 6-direction field, no recombination (m02's triangle path = today's `m14_blocks_tri`
  fill). The hook `tri_fill="module:function"` swaps in `fill_tri(domain, h, fixed_loops) -> Mesh` per triangle region
  (the region's loops as exact loops, loop nodes first; `stitch_mix.fill_tri_local` is a reference: Poisson points,
  local Delaunay with far helper points, size smoothing) — the place for m15's S3 + S4.

After the fills: one smoothing of the whole fill as m02 smooths (coast and outline nodes fixed; the separator nodes
free, since the connectivity keeps each region's fill: −0.002 in the whole-mesh |ln s/h| p99 of every region meshed on
its own, junctions 0.29 → 0.24), the merge with the blocks and stitch.py's seam-band smoothing.

**Approaches that failed**: a transfinite strip with one fill edge per outline node gave
reflex 185° quads at outline steps; requiring step-free runs left no corridor (every run has 2–19 lattice steps, from
the blocks' rounded corners and lattices that bend along an 80-cell gap); a stepped strip with one row count per
corridor gave 3–26 rows across a 6.8 h gap on `uniform` s1 (|ln s/h| p99 0.5–1.4). The ladder carries the defects the
two lattices need. Lattices oblique to the gap still shear it (one run, all 8 case / scale pairs): at obliquity ≥ 20°,
12 of 27 strips had a quad of 143–169° and 22 a |ln s/h| p99 > 0.3; below 20°, 2 of 63 above 143° (max 146.6°) and 1
above 0.3: hence `obl_strip_deg` = 20.

**Decision table**. The same
partition meshed four times: every block-block corridor forced to strip (where eligible, else m02) / m02 / tri, with
junctions, open and coast regions m02 / m02 / tri, every region on its own (`merge_same_fill=False`); the fourth run
is the default rule. Classes: "strip-able" = the corridors that are strips in the strip run (the same corridors in the
m02 and tri rows), split by rotation (10° / 25°). Defects = interior nodes with |360 − 90 q − 60 t| ≥ 60°
(q quads, t triangles at the node: counts valence-3/5 quad nodes and valence-5/7 triangle nodes alike, not the +30°
corners of a lone triangle) per 10 cells of corridor length; irregular (valence ≠ 4) only for the quad fills; |ln s/h|
p99 per region, averaged with the length as weight (max over the regions).

Scale 1, raw (42 strip-able corridors, 2858 cells of length; 120 others, 7409):

| class | fill | regions | length cells | defects / 10 cells | irregular / 10 | triangles / 10 | \|ln s/h\| p99 (mean / max) | ratio p99 | quad max |
|---|---|---|---|---|---|---|---|---|---|
| strip-able, rot < 10° | **strip** | 30 | 2218 | **2.11** | 2.26 | 2.26 | **0.195** / 0.262 | 1.102 | 146.6 |
| | m02 | 30 | 2218 | 3.50 | 4.27 | 3.69 | 0.210 / 0.334 | 1.101 | 140.2 |
| | tri | 30 | 2218 | 19.76 | - | 148.4 | 0.281 / 0.312 | 1.186 | (tri 104.7) |
| strip-able, rot 10-25° | **strip** | 9 | 486 | **2.62** | 2.64 | 2.60 | **0.194** / 0.255 | 1.109 | 145.5 |
| | m02 | 9 | 486 | 3.63 | 4.35 | 3.75 | 0.197 / 0.324 | 1.118 | 135.9 |
| | tri | 9 | 486 | 21.24 | - | 155.2 | 0.269 / 0.280 | 1.180 | (tri 103.9) |
| strip-able, rot ≥ 25° | strip | 3 | 154 | 3.64 | 4.03 | 4.09 | 0.214 / 0.705 | 1.128 | 138.8 |
| | **m02** | 3 | 154 | 4.48 | 5.06 | 4.61 | **0.180** / 0.275 | 1.134 | 131.2 |
| | tri | 3 | 154 | 25.45 | - | 160.6 | 0.293 / 0.313 | 1.195 | (tri 100.3) |
| no strip, rot < 10° | **m02** | 45 | 2544 | **2.88** | 3.60 | 3.52 | **0.209** / 0.317 | 1.140 | 138.4 |
| | tri | 45 | 2544 | 24.67 | - | 139.5 | 0.289 / 0.326 | 1.192 | (tri 104.7) |
| no strip, rot 10-25° | **m02** | 37 | 2437 | **4.83** | 6.58 | 5.84 | **0.242** / 0.345 | 1.156 | 137.6 |
| | tri | 37 | 2437 | 25.81 | - | 137.5 | 0.294 / 0.340 | 1.200 | (tri 105.9) |
| no strip, rot ≥ 25° | **m02** | 26 | 1531 | **5.70** | 8.26 | 7.25 | **0.254** / 0.334 | 1.171 | 135.0 |
| | tri | 26 | 1531 | 26.53 | - | 141.4 | 0.295 / 0.314 | 1.196 | (tri 104.9) |
| no strip, rotation not measured | **m02** | 12 | 897 | **5.65** | 7.78 | 6.71 | **0.246** / 0.335 | 1.159 | 134.9 |
| | tri | 12 | 897 | 25.73 | - | 153.5 | 0.286 / 0.303 | 1.195 | (tri 105.0) |
| junction | **m02** | 106 | 308 | **14.34** (15.6 / 100 cells of area) | 21.82 | 20.55 | 0.281 / 0.497 | 1.170 | 135.0 |
| | tri | 106 | 308 | 43.61 (48.2) | - | 219.6 | **0.276** / 0.405 | 1.184 | (tri 130.3) |
| open | **m02** | 71 | 3373 | **12.16** (6.7) | 15.63 | 12.51 | **0.250** / 0.467 | 1.152 | 140.7 |
| | tri | 71 | 3373 | 52.53 (29.1) | - | 436.6 | 0.274 / 0.455 | 1.182 | (tri 125.3) |
| coast | **m02** | 9 | 805 | **2.67** (3.5) | 3.52 | 3.17 | **0.141** / 0.332 | 1.195 | 136.6 |
| | tri | 9 | 805 | 9.42 (12.5) | - | 174.8 | 0.205 / 0.342 | 1.192 | (tri 100.5) |

Scale 4 says the same with fewer regions: strip-able rot < 10° strip 2.01 / m02 5.30 / tri 19.7 defects per 10 cells,
|ln s/h| 0.187 / 0.260 / 0.267; rot 10-25° 2.93 / 4.21 / 24.0, 0.191 / 0.213 / 0.284; rot ≥ 25° (2 corridors) 3.53 /
3.83 / 19.1, 0.293 / 0.205 / 0.278; not strip-able m02 5.5-6.7 vs tri 23-28; junction m02 18.1 vs tri 39.6; open 14.9
vs 46.7; coast 4.0 vs 12.7. Corridor by corridor (scale 1): the strip has fewer defects than m02 in 27 of 42 corridors,
as many in 9, more in 6 (by 0.1-0.4 per 10 cells); the mismatch does not change that up to the eligibility limit (≤ 3
per 10 cells: strip 2.3 / 1.8 / 2.9, m02 3.8 / 1.9 / 5.3 at 0-1 / 1-2 / 2-3 per 10 cells).


The per-region baseline flatters the strip: m02 confined to the same regions also pays for the caps. **Against the m02
fill it replaces** (`m02merged`: the same partition with no strip and the regions merged, which reproduces
`m14_blocks` exactly: 51/591/592, 131/1645/1560, 394/3880/3618, 164/1918/1814 n3/n5/T at scale 1) and on the same
corridors (`allstrip`: merged, a strip in every strip-able corridor):

| strip-able corridors, scale 1 | corridors | length cells | irregular / 10: strip · m02 | triangles / 10 | \|ln s/h\| p99: strip · m02 | strip fewer / equal / more |
|---|---|---|---|---|---|---|
| rot < 10° | 30 | 2218 | **2.25** · 3.95 | 2.26 · 3.51 | 0.195 · 0.199 | 17 / 3 / 10 |
| rot 10-25° | 9 | 486 | **2.60** · 3.71 | 2.60 · 3.23 | 0.193 · 0.187 | 7 / 1 / 1 |
| rot ≥ 25° | 3 | 154 | 4.09 · 4.61 | 4.09 · 4.09 | 0.214 · **0.161** | 3 / 0 / 0 |
| uniform | 11 | 1016 | **2.71** · 5.87 | 2.73 · 5.21 | 0.212 · 0.230 | 11 / 0 / 0 |
| disc | 13 | 610 | 2.13 · **1.95** | 2.13 · 1.76 | 0.193 · **0.160** | 4 / 3 / 6 |
| band | 14 | 1116 | **2.36** · 3.48 | 2.35 · 3.03 | 0.185 · 0.187 | 10 / 1 / 3 |
| continent | 4 | 115 | 1.83 · 1.91 | 1.83 · 2.00 | 0.175 · 0.139 | 2 / 0 / 2 |

Scale 4: rot < 10° 2.01 · 6.24 (12 / 0 / 2), rot 10-25° 2.80 · 4.82, rot ≥ 25° 4.51 · 5.04 with |ln s/h| 0.285 · 0.220;
uniform 2.40 · 8.96, disc 3.13 · 6.80, band 2.37 · 3.53. The strip halves the defects of long aligned gaps (uniform's
80-cell cube edges, band's), not of disc's short gaps at scale 1 (mean 47 cells), where m02 alone is already at 2 per
10 cells. And it costs outside the corridor: the junction between strips becomes a closed m02 piece. Decomposed against
`m02merged` (scale 1, irregular nodes): uniform −322 in the strips, +73 in the junctions and open holes next to them;
band −122 / +83 (+50 elsewhere); disc +16 / +53 (+9).

**The rule, from the numbers** (the defaults): a strip where the corridor is strip-able and its rotation is below
25° (`rot_quad_deg` = 25: above, fewer defects but a worse size, 0.214 · 0.161; `rows_tol` = 3: the mismatch costs
nothing up to the eligibility limit); m02 everywhere else, junctions, open holes and coast corridors included; no
triangles (`rot_tri_deg` = 90, `rows_tri` = inf: today's triangle fill has 9-53 defects per 10 cells and the worst size
in every class; a triangle fill at ≥ 25° is `rot_tri_deg=25`). Rotation does not decide strip or not: it raises m02's
density (2.9 / 4.8 / 5.7 per 10 cells in the non-strip corridors) more than the
strip's. What keeps corridors from the strip is the obliquity of their lattices to the gap (28 / 32 / 23 of the 56 / 57
/ 37 corridors of disc / band / continent at scale 1), then a column mismatch above 0.3 of the shorter side (4-7) and
sides that are not one run (2-6). Strip share of the corridor length under the rule: 0.90 / 0.23 / 0.25 / 0.04 at
scale 1 (uniform / disc / band / continent), 0.52 / 0.29 / 0.44 / 0 at scale 4. `merge_same_fill` = True: the non-strip
regions are one m02 fill with the strips cut out as exact boundaries (with every region on its own, m02 everywhere
gives 267/2409, 555/4903, 214/2441 n3/n5 on disc / band / continent, |ln s/h| p99 0.149-0.155: each separator is a
pair of colliding fronts).

**Per case** (`m14_blocks` / `_tri` from the final run; seam
band = outline nodes + fill rows 0-3, from `m14.json`; the triangle fill's valence counts mean nothing; time = generate,
s):

| case | s | method | gates raw · sm | tris % | n3 / n5 / T | \|ln s/h\| p99 raw · sm | ratio p99 | quad max | seam irregular % | seam tris | seam \|ln s/h\| p99 | time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 4 | m14_blocks | 16/18 · 14/18 | 4.7 | 12 / 156 / 160 | 0.228 · 0.246 | 1.148 | 127.4 | 9.2 | 108 | 0.257 | 1.5 |
| uniform | 4 | m14_blocks_tri | 14/18 · 14/18 | 68.2 | 0 / 589 / 3606 | 0.255 · 0.261 | 1.177 | 107.1 | 95.9 | 1692 | 0.257 | 1.4 |
| uniform | 4 | m14_blocks_mix | 16/18 · 14/18 | 4.6 | 6 / 142 / 154 | 0.201 · 0.224 | 1.139 | 129.7 | 7.9 | 108 | 0.224 | 4.0 |
| disc | 4 | m14_blocks | 14/18 · 12/18 | 2.6 | 34 / 358 / 358 | 0.199 · 0.215 | 1.204 | 136.1 | 7.9 | 239 | 0.264 | 2.9 |
| disc | 4 | m14_blocks_tri | 14/18 · 14/18 | 48.0 | 0 / 1421 / 8974 | 0.234 · 0.248 | 1.189 | 112.6 | 95.9 | 4277 | 0.258 | 2.7 |
| disc | 4 | m14_blocks_mix | 14/18 · 12/18 | 2.3 | 16 / 304 / 322 | 0.185 · 0.200 | 1.193 | 129.2 | 6.2 | 210 | 0.239 | 3.0 |
| band | 4 | m14_blocks | 14/18 · 14/18 | 2.5 | 69 / 698 / 690 | 0.204 · 0.220 | 1.296 | 139.4 | 6.5 | 290 | 0.255 | 5.1 |
| band | 4 | m14_blocks_tri | 14/18 · 14/18 | 48.2 | 0 / 2770 / 17708 | 0.261 · 0.289 | 1.230 | 114.1 | 95.2 | 7244 | 0.288 | 5.2 |
| band | 4 | m14_blocks_mix | 14/18 · 14/18 | 2.6 | 59 / 713 / 714 | 0.207 · 0.222 | 1.299 | 139.5 | 6.6 | 316 | 0.257 | 6.1 |
| continent | 4 | m14_blocks | 21/26 · 19/26 | 2.9 | 29 / 376 / 375 | 0.199 · 0.215 | 1.143 | 140.1 | 8.4 | 226 | 0.243 | 3.1 |
| continent | 4 | m14_blocks_tri | 17/23 · 17/23 | 52.9 | 0 / 1327 / 9429 | 0.257 · 0.260 | 1.180 | 112.2 | 92.5 | 3558 | 0.293 | 5.3 |
| continent | 4 | m14_blocks_mix | 21/26 · 19/26 | 2.9 | 29 / 376 / 375 | 0.199 · 0.215 | 1.143 | 140.1 | 8.4 | 226 | 0.243 | 7.0 |
| uniform | 1 | m14_blocks | 18/18 · 16/18 | 1.1 | 51 / 591 / 592 | 0.148 · 0.159 | 1.089 | 140.2 | 7.8 | 508 | 0.244 | 8.2 |
| uniform | 1 | m14_blocks_tri | 16/18 · 16/18 | 26.4 | 0 / 2750 / 16002 | 0.211 · 0.213 | 1.135 | 109.7 | 95.1 | 8350 | 0.272 | 8.2 |
| uniform | 1 | m14_blocks_mix | 18/18 · 16/18 | 0.8 | 12 / 394 / 408 | 0.144 · 0.157 | 1.079 | 146.6 | 4.8 | 353 | 0.229 | 8.8 |
| disc | 1 | m14_blocks | 18/18 · 18/18 | 0.7 | 131 / 1645 / 1560 | 0.128 · 0.137 | 1.090 | 135.0 | 4.0 | 619 | 0.210 | 39.8 |
| disc | 1 | m14_blocks_tri | 16/18 · 16/18 | 26.8 | 0 / 9093 / 67738 | 0.209 · 0.211 | 1.137 | 113.1 | 95.1 | 24608 | 0.272 | 39.7 |
| disc | 1 | m14_blocks_mix | 18/18 · 18/18 | 0.8 | 135 / 1715 / 1634 | 0.131 · 0.142 | 1.092 | 145.6 | 4.3 | 682 | 0.215 | 41.6 |
| band | 1 | m14_blocks | 18/18 · 16/18 | 0.8 | 394 / 3880 / 3618 | 0.140 · 0.152 | 1.111 | 139.7 | 3.9 | 1058 | 0.220 | 80.4 |
| band | 1 | m14_blocks_tri | 16/18 · 16/18 | 30.2 | 0 / 17718 / 154958 | 0.214 · 0.216 | 1.144 | 118.1 | 94.9 | 43339 | 0.277 | 82.1 |
| band | 1 | m14_blocks_mix | 18/18 · 16/18 | 0.9 | 374 / 3906 / 3674 | 0.141 · 0.154 | 1.111 | 139.5 | 3.9 | 1085 | 0.218 | 87.4 |
| continent | 1 | m14_blocks | 24/24 · 23/24 | 0.9 | 164 / 1918 / 1814 | 0.135 · 0.147 | 1.088 | 135.0 | 6.0 | 1021 | 0.216 | 41.6 |
| continent | 1 | m14_blocks_tri | 20/23 · 18/23 | 31.9 | 0 / 9723 / 76434 | 0.221 · 0.221 | 1.149 | 116.2 | 93.8 | 24714 | 0.291 | 41.7 |
| continent | 1 | m14_blocks_mix | 24/24 · 23/24 | 0.9 | 170 / 1944 / 1856 | 0.137 · 0.149 | 1.089 | 135.3 | 6.2 | 1057 | 0.217 | 48.6 |

- **Scale 1**: the gates are those of `m14_blocks` on every case and stage (18/18, 18/18, 18/18, 24/24 raw; 16, 18, 16,
  23 smoothed). Uniform, where 90 % of the corridor length is strip: n3 + n5 642 → 406 (−37 %), triangles 592 → 408,
  seam band irregular 7.8 → 4.8 %, seam |ln s/h| p99 0.244 → 0.229, whole-mesh p99 0.148 → 0.144, ratio p99 1.089 →
  1.079; but quad max 140.2 → 146.6°. Disc / band / continent (23 / 25 / 4 % strip): within +0.1 … +4 % of `m14_blocks`
  in irregular nodes (1850 vs 1776, 4280 vs 4274, 2114 vs 2082) and triangles (+1.5 … +4.7 %), |ln s/h| p99 +0.001 …
  +0.003, seam band 4.3 / 3.9 / 6.2 % (4.0 / 3.9 / 6.0 %). `m14_blocks_tri`: 26-32 % triangles, 16/18 and 20/23.
- **Scale 4**: uniform and disc gain (n3 + n5 168 → 148, 392 → 320; |ln s/h| p99 0.228 → 0.201, 0.199 → 0.185; seam
  band 9.2 → 7.9 %, 7.9 → 6.2 %), band is even (767 → 772), continent has no strip-able corridor and is `m14_blocks`
  node for node. Gates as `m14_blocks` (16, 14, 14, 21).
- **Under the revised gates** (the 16 final records re-scored, 0 mismatches in the unchanged fields): scale 1 raw 20/20, 19/20, 19/20, 41/44, failed families 0 / 1 / 1 / 2 = 4
  (`m14_blocks` 19/20 ×3, 41/44 = 5): uniform has no triangle cluster of ≥ 3 (`m14_blocks` 2); disc / band / continent
  keep 6 / 16 / 10 (5 / 15 / 9), continent also the coastal size; smoothed 6 (7); scale 4 18, 16, 16, 34 (17, 15, 15,
  34). The `_tri` rows above are the old triangle fill; the current `m14_blocks_tri` with m15's population
  control and fit ranks first of the m14 variants (1 failed family raw, 4 smoothed).
- **Cost**: +0.6 … +7 s at scale 1 (partition 0.2-4.2 s, packing 0.1-2.6 s, region measures ≤ 0.9 s), memory as
  `m14_blocks` (≤ 1.2 GB). Every run valid (CCW, Euler, manifold); one strip of the triangle variant on disc s1 and none
  of the default runs needed the leak retry.

**Figures** (final run): the standard set of each run and `zoom_fill.png` (elements coloured by fill,
the longest strip in the centre). Looked at: `zoom_fill` uniform s1 (rows parallel to the gap, the terminations as a sparse line of
triangle + valence-5 pairs in the middle row, about one per 4 cells), band s1 (16 row + 13 column terminations along a
160-cell strip, evenly spaced), `zoom_seam` uniform s1 (the worst seam is now the Y where three strips meet: a closed
m02 piece), `zoom_angle` uniform s1 (the 146.6° quads: the last columns of a strip whose cap chord meets the corridor
obliquely), `zoom_seam` disc s1 (the worst seam is an m02 corridor between two lattices oblique to it, as before).

**Reading.** The rule, measured: the strip is the right fill where two lattices run along the gap; it halves
the defects of long aligned gaps and is a little more size-faithful there; triangles (today's) are wrong everywhere;
m02 stays for junctions, holes, coasts and the oblique corridors. With greedy independent block frames most corridors
of the graded cases are oblique, so the total gain is small (uniform −37 % irregular nodes, the others ±4 %). The
defects that remain sit in the oblique corridors and in the junctions. **Not done / next**: a lattice-continuation fill
for oblique but parallel lattices (extend one block's lattice across the gap; the ladder's columns shear there);
caps perpendicular to the corridor at its ends (the 146° quads); strips that pass through a junction (two collinear
corridors as one strip); block frames turned gap by gap so more corridors are strip-able;
the triangle classes with m15's fill (`tri_fill` hook; `fill_tri_local` is only the reference); `coast="cut"` (reserved,
raises). The region measures run in `m14_blocks.run`'s deferred diagnostics stage, like `measure`
(`timings.regions_measure_s`, not in the generate time).

## Direct join and the `b_cap` probe (2026-10-23)

Code: option `merge` (default off) and `merge_drift` in `m14_blocks.DEFAULTS`; `m14/sync.py` `merge`, `is_candidate`,
`drift_per10`; `m14/place.py` `union_region`; `m14/blockmap.py` `region_grid`; test `tests/test_m14_merge.py`;
decisions B35, B61 (B14 kept).

**What.** Before the frame synchronisation, every block-block corridor of the first-pass blocks with b = 0 or a drift
of at least `merge_drift` (0: every corridor) dislocations per 10 cells is a candidate, the most drift dislocations
(drift × L) first. Drift = Σ 2|sin((φ_e − φ̄)/2)| per 10 edges, the rotational count that the best rigid turn of the
pair leaves (a lower bound per corridor). The union of the two regions (or merged groups) takes the bg
triangles between them: the directions to the nearest outline node of each region are ≥ 150° apart, so the corridor
ends with the facing sides. It also keeps the partition's corridor rules (d_a + d_b ≤ 14 h, a third object ≥ 2 h
farther), the cone / coast margins and the 6 h gap to every other block; holes are filled and pinch fans closed. It is
merged when it is a closed disc that passes the placement's acceptance rule (F ≤ 0.08, h|∇λ| ≤ 0.05) and its one map
gives a grid. The merged block is one node of the sync's block graph.

**The `b_cap` probe** (uniform s1, sync on; drift = the `_mix` corridors' B53
defects; "sync forced" = the sync record's predicted rotational + along count per 10 cells after the turns and shifts):

| b_cap | merge | blocks | block-block corridors | coverage (area) | `_mix` corridor defects: total / length / per 10 | sync forced / 10 | `m14_blocks` seam band irregular | gates `m14_blocks` / `_mix` |
|---|---|---|---|---|---|---|---|---|
| **1.3** | off | 6 | 12 | **0.874** | **299** / 1128 / **2.65** | 3.49 | 645 | 20/21 / 20/21 |
| 0.6 | off | 14 | 32 | 0.749 | 630 / 1816 / 3.47 | 2.66 | 828 | 18/21 / 18/21 |
| 0.3 | off | 23 | 57 | 0.720 | 482 / 1943 / 2.48 | 2.55 | 769 | 18/21 / 11/21 |
| 1.3 | on | 6 (0 of 12 merged: union F 0.088–0.105) | 12 | 0.874 | 299 / 1128 / 2.65 | 3.49 | 645 | 20/21 / 20/21 |
| 0.6 | on | 4 (10 of 32 merged, 47 % of the length) | 10 | 0.831 | 386 / 1094 / 3.53 | 4.04 | 596 | 18/21 / 18/21 |
| 0.3 | on | 4 (19 of 57 merged, 43 %) | 8 | 0.831 | 425 / 1310 / 3.24 | 4.18 | 589 | 20/21 / 20/21 |

A lower cap buys no seam density: the blocks are smaller and drift less per cell, but there are 1.6–1.7× as many
corridor cells, so the corridor total rises (299 → 630 / 482) and coverage drops 12–15 points; 0.6 and 0.3 also fail the
size gate (more m02 fill) and 0.3 `_mix` the quad angle gates. With the merge the small blocks grow back into 4 long
blocks (B 1.3–3.2, F 0.03–0.08: an elongated union has a smaller F per unit of B than a square). Coverage comes back to
0.831, not 0.874. The defects of the remaining corridors fall less than the merged length (−39 % for 47 %, −12 % for
43 %, against the same cap without the merge) and stay above the default layout (386 / 425 against 299). At the
default cap no union passes: two cube faces with their corridor have F 0.088–0.105. **`b_cap` stays 1.3** (B14).

**Why the merge moves drift instead of removing it.** On a block's outline λ = 0, and the two lattices of a seam turn
against each other at the rate |∂_n λ_A| + |∂_n λ_B| along it (the conformal factor's normal derivative on each side);
one map has ∂_n λ continuous across, so a merged seam has no drift. But ∮ ∂_n λ ds = −(π/2) B over each block's outline
(Δλ = −(K + Δ ln h)). A merged block of larger B puts that turning on its remaining outline, and a seam's count grows
with its drift times its length (∫|θ| ds for θ drifting linearly along it). Measured as sync's predicted rotational
count per 10 cells of the remaining corridors: uniform 2.93 → 2.97 / 3.59 (b_cap 0.6 / 0.3 merged), disc 2.26 → 3.00,
band 1.59 → 3.05, continent 1.81 → 3.22.

**The merge at the default cap** (s1, raw; base = the same build with `merge = False`, against `merge = True`):

| method | case | blocks | merged (length share) | coverage | corridor defects (B53) total / per 10 | seam band irregular | n3 / n5 / T | gates | generation s (merge) |
|---|---|---|---|---|---|---|---|---|---|
| m14_blocks | uniform | 6 → 6 | 0 of 12 | 0.873 → 0.873 | | 645 → 645 | 53/595/588 → same | 20/21 → 20/21 | 14 → 15 (3) |
| m14_blocks | disc | 24 → 12 | 12 of 56 (0.23) | 0.880 → 0.893 | | 1259 → 1101 | 167/1785/1694 → 199/1880/1780 | 20/21 → 20/21 | 60 → 110 (53) |
| m14_blocks | band | 32 → 7 | 25 of 57 (0.42) | 0.766 → 0.800 | | 1437 → 1222 | 319/3449/3302 → 289/2939/2776 | 20/21 → 20/21 | 119 → 379 (254) |
| m14_blocks | continent | 18 → 6 | 12 of 37 (0.26) | 0.859 → 0.875 | | 1396 → 1532 | 167/1891/1798 → 192/2024/1916 | 42/45 → 44/47 (same 3 failed) | 66 → 116 (51) |
| m14_blocks_mix | uniform | 6 → 6 | 0 of 12 | 0.874 → 0.874 | 299 / 2.65 → same | 405 → 405 | same | 20/21 → 20/21 | |
| m14_blocks_mix | disc | 24 → 12 | 12 of 56 (0.23) | 0.880 → 0.893 | 873 / 3.32 → 793 / 3.75 (−9 %) | 1220 → 1116 | 143/1783/1716 → 199/1882/1778 | 18/21 → 20/21 | 61 → 111 (53) |
| m14_blocks_mix | band | 32 → 7 | 25 of 57 (0.42) | 0.766 → 0.800 | 826 / 2.11 → 586 / 2.72 (−29 %) | 1501 → 1203 | 327/3544/3404 → 283/2907/2764 | 18/21 → 20/21 | 122 → 380 (254) |
| m14_blocks_mix | continent | 18 → 6 | 12 of 37 (0.26) | 0.859 → 0.875 | 817 / 2.89 → 850 / 3.87 (+4 %) | 1323 → 1440 | 156/1824/1746 → 181/1949/1842 | 42/45 → 44/47 (same 3 failed) | 68 → 115 (50) |

- **Merged seams:** 0 irregular nodes among the block nodes inside the former corridors (uniform b_cap 0.3 / 0.6: 4162 /
  3819 nodes; disc 2475, band 7782, continent 2921; one irregular outline node of 426 on uniform 0.3). 17–30 % of a
  former corridor's area is not covered by the merged block (its ends and the notches where the facing sides are
  rotated); that part stays fill, with 19–57 irregular nodes per case.
- **The `_mix` gates** that sync costs on disc and band (strip quads of 150.3° / 155°) come back with the merged layout,
  because those corridors are gone, not because a join got better.
- **Not on by default:** uniform gains nothing at the default cap, and below it the merged layout is worse than the
  default (above). Continent's seams get worse (+4 % corridor defects, +10 % seam band irregular). The merge costs 50–254 s
  at s1: one map per accepted union, and a group's map is solved again at every merge it takes (band: 25 maps, a final
  block of B 9.4 over the plateau).
- **Next, if wanted:** maps only for the final groups (one solve per group, a failed group falls back to its members);
  a union that fails F narrowly shrunk to pass (uniform's cube pairs at 0.088); curvature pushed into the junction cones
  (one map per pair with λ continuous across the seam, the cubed-sphere trade), the only lever
  left for uniform's drift.

## Smoothness within a size tolerance (2026-10-23; decisions G38, M38, R22)

**Default (2026-10-23): t = 0.20 for every m14 method (`m14_blocks_rep` included); `size_tolerance = 0` fits h
as before.** `size_tolerance` = t (`[mesh] size_tolerance`, or the method option): the cell size may deviate from h by
|ln s/h| ≤ t, and within it the mesh is made as smooth as possible. `meshom.size_fit.smooth_fit` minimises the
element size ratio across every interior edge (ln(M_a/M_b)², M = mean edge, the G7 quantity) plus a mild
corner-angle term, with the tolerance as a soft wall (from t − 0.01, Huber pull), an angle wall at 60–120° (quads) /
35–100° (triangles) and a stiff wall 5° inside the hard angle gates; L-BFGS, 500 iterations × 5 rounds. The size gate
is judged at t (G38); the ratio gates stay.

**Where it acts.** Plain m14 and `_mix`: the last stage of the stitch (after the seam-band smoothing), on every fill
node, block and coast nodes fixed. `_rep`: m14 and the repair run as without a tolerance (the repair's bad edge at
max(t, 0.15)), then the smooth fit on the repaired mesh with the remaining block nodes fixed. (First tried: the fit in
the stitch, then the repair with its bad edge at t. The repair then left triangle clusters of 3 — disc s1 at 0.10 /
0.20 / 0.30, uniform at 0.10, CORE2-like at all three — and at 0.30 it raised the CORE2-like ratio to 1.150.)

**Why the fill fails G7 without it.** CORE2-like m14 before the repair (closed sphere): every element-ratio
failure sits between two fill elements. Block–block edges: p99 1.022. Block–fill: 1.060. Fill–fill: 1.212, with 4.5 %
of them above 1.15. The fill quads' mean edge scatters around h by e^−0.17 … e^+0.14 (p1 … p99), and neighbours jump
between the two ends. This is node-position noise left by m02's packing and the h-driven smoothing, not h itself:
the smooth fit removes most of it at t = 0.10 already. Pre-repair on the cached mesh, after the land cut: ratio p99
1.197 → 1.126 / 1.120 / 1.115 / 1.110 at t = 0.10 / 0.15 / 0.20 / 0.30; 38/39 gates at each, the remaining one being
the pre-existing `tri_cluster_max`.

**CORE2-like config** (the CORE2-like `m14_blocks_rep` mesh with `size_tolerance`, GEBCO stage off; 16 CPUs):

| method | t | nodes | ratio p99 / max | \|ln s/h\| p99 | quad p99.9 / max | tri min / max | gates | wall |
|---|---|---|---|---|---|---|---|---|
| m14_blocks_rep | none | 143 271 | 1.206 / 1.532 | 0.254 | 127.2 / 135.0 | 31.5 / 97.3 | 31/39 (ratio, size) | 1077 s |
| m14_blocks_rep | 0.10 | 143 332 | **1.121** / 1.247 | 0.096 | 124.8 / 142.5 | 45.8 / 80.7 | **39/39** | 1109 s |
| m14_blocks_rep | 0.20 | 143 380 | **1.110** / 1.248 | 0.194 | 120.1 / 130.9 | 38.4 / 99.9 | **39/39** | 968 s |
| m14_blocks_rep | 0.30 | 143 601 | 1.105 / 1.233 | 0.290 | 120.0 / 126.7 | 34.8 / 106.7 | 38/39 (tri clusters) | 1134 s |
| m14_blocks_mix | none | 144 356 | 1.197 / 1.565 | 0.242 | 130.2 / 154.3 | 29.1 / 108.8 | 28/39 | 84 s |
| m14_blocks_mix | 0.10 | 144 331 | 1.126 / 1.307 | 0.097 | 132.4 / 145.1 | 30.4 / 115.1 | 38/39 (tri clusters) | 161 s |
| m14_blocks_mix | 0.20 | 144 542 | 1.113 / 1.407 | 0.193 | 121.5 / 145.1 | 35.4 / 107.6 | 38/39 (tri clusters) | 192 s |
| m14_blocks_mix | 0.30 | 144 497 | 1.105 / 1.244 | 0.290 | 120.2 / 134.3 | 31.3 / 112.8 | 38/39 (tri clusters) | 168 s |
| m15_tri | none | 162 557 | 1.109 / 1.218 | 0.117 | — | 42.8 / 84.5 | 29/29 | 85 s |
| m15_tri | 0.20 | 162 707 | 1.086 / 1.192 | 0.180 | — | 38.5 / 91.0 | 29/29 | 230 s |

The `none` rows reproduce an earlier mixed-versus-triangle comparison (rep: 1.206 / 0.254, 31/39). `_mix`'s
`tri_cluster_max` (3) is there without the tolerance too.

**Scale 1, raw** (`meshom compete`), gates passed:

| method | t | uniform | disc | band | continent | ratio p99 (u / d / b / c) | \|ln s/h\| p99 (u / d / b / c) |
|---|---|---|---|---|---|---|---|
| m14_blocks_rep | none | 21/21 | 21/21 | 21/21 | 41/43 (size) | 1.076 / 1.087 / 1.104 / 1.078 | 0.133 / 0.123 / 0.133 / 0.124 |
| m14_blocks_rep | 0.10 | 21/21 | 21/21 | 21/21 | 40/43 (size, quad p99.9) | 1.034 / 1.051 / 1.080 / 1.035 | 0.090 / 0.090 / 0.091 / 0.090 |
| m14_blocks_rep | 0.20 | 21/21 | 21/21 | 21/21 | 43/43 | 1.027 / 1.046 / 1.075 / 1.029 | 0.174 / 0.168 / 0.185 / 0.171 |
| m14_blocks_rep | 0.30 | 21/21 | 21/21 | 21/21 | 43/43 | 1.027 / 1.044 / 1.074 / 1.028 | 0.180 / 0.176 / 0.194 / 0.175 |
| m14_blocks_mix | none | 21/21 | 21/21 | 20/21 (clusters) | 43/45 (size) | 1.075 / 1.089 / 1.105 / 1.086 | 0.142 / 0.136 / 0.143 / 0.145 |
| m14_blocks_mix | 0.10 | 21/21 | 21/21 | 20/21 | 43/45 (size) | 1.046 / 1.060 / 1.082 / 1.049 | 0.096 / 0.093 / 0.095 / 0.094 |
| m14_blocks_mix | 0.20 | 21/21 | 21/21 | 20/21 | 45/45 | 1.032 / 1.052 / 1.076 / 1.038 | 0.184 / 0.177 / 0.189 / 0.189 |
| m14_blocks_mix | 0.30 | 21/21 | 21/21 | 20/21 | 45/45 | 1.031 / 1.050 / 1.074 / 1.037 | 0.183 / 0.176 / 0.202 / 0.200 |
| m15_tri | none | 15/15 | 15/15 | 15/15 | 35/35 | 1.058 / 1.056 / 1.064 / 1.055 | 0.093 / 0.091 / 0.100 / 0.091 |
| m15_tri | 0.20 | 15/15 | 15/15 | 15/15 | 35/35 | 1.051 / 1.051 / 1.057 / 1.049 | 0.126 / 0.112 / 0.126 / 0.114 |

At t = 0.10 the coast subsets of continent miss (the coast nodes are fixed, so their edges cannot move into the
wall). The quad angle tails improve with the tolerance: `_rep` max 120–122° at 0.20 against 130–135° without, because
the angle wall acts on every fill quad. Times at t = 0.20: `_rep` −4 to +57 % at scale 1 (uniform 49 → 77 s, band 419 → 404 s),
CORE2-like −10 to +5 % over the three tolerances. `_mix` +43 to +49 % at scale 1, +129 % on CORE2-like (84 → 192 s).
m15_tri ×2.7–3.6 (CORE2-like 85 → 230 s).

**Recommendation:** t = 0.20 for the m14 methods. It is the only setting that passes every gate on CORE2-like
(`_rep`) and on all four scale-1 cases (`_rep` and, except band's pre-existing clusters, `_mix`). t = 0.10 keeps the
size error at G9's level and below, and still brings the ratio to 1.12. For m15_tri the gain is small (ratio
1.109 → 1.086) at 2–4× the time, so leave it off there. This became
the default (above).

## Shared growth front: seams on the bisector (2026-10-25; decisions B67)

Code: option `front` ("seeds" by default, True, False; `m14_blocks_mix` off) in `m14_blocks.DEFAULTS`; `m14/place.py` `_front_pass`,
`front_cells`, `front_points`; test `tests/test_m14_rules.py::test_shared_front_centres_the_gap`.

**Why.** After B65 three seam lakes failed only the centre share (G36): rot45_k0, rot45_k10 and the junction. The
wall sat in the middle of the gap, and the gap was off the bisector. With given seeds the first seed's growth square
took the 45° bend's concave corner, and the second block kept `block_gap` from it. In arm A of the 45° bend the B4
field is also twisted by about 40° (the convex corner's +1/4 cone), so A's L∞ square was turned toward B's lattice.
The gap edges on rot45_k0 were 2.3 h and 5.4 h from the bisector, and the gap was 17 h wide near the convex corner.

**What.** Given seeds now share the space. Each bg node belongs to the seed whose front reaches it first when all
fronts advance at once in the isotropic metric |dx| / h (the seeds' Voronoi cells in units of h). A region grows
inside its own cell and stays more than `block_gap` / 2 from the cell boundary, so the gap is centred on the line where
the fronts meet. Growth and acceptance are as before; `grow_limit` is not applied, because the cell is the limit.
The cells use the isotropic metric, not the growth's L∞ field metric: the twisted field of arm A put that front
1-1.5 h off the bisector.

**Seam lakes** (`m14_blocks_rep` s1, every seam gate; centre share per corridor, corridor / lake gross, lake excess
per 100 nodes):

| case | base centre | base gross | base lake | base gates | front centre | front gross | front lake | front gates |
|---|---|---|---|---|---|---|---|---|
| rot0_k0 | 0.50 | 4 / 18 | 0.55 | pass | 0.75 | 8 / 22 | 0.67 | pass |
| rot0_k3 | 1.00 | 6 / 32 | 0.92 | pass | - | 0 / 8 | 0.19 | pass |
| rot0_k10 | 0.00 | 2 / 24 | 0.59 | pass | 1.00 | 2 / 36 | 1.00 | pass |
| rot15_k0 | 0.89 | 18 / 30 | 0.37 | pass | 0.90 | 20 / 30 | 0.37 | pass |
| rot15_k3 | 0.56 | 18 / 34 | 0.45 | pass | 0.64 | 24 / 32 | 0.39 | pass |
| rot15_k10 | 0.62 | 16 / 50 | 0.88 | pass | 0.80 | 19 / 54 | 1.01 | lake |
| rot30_k0 | 0.65 | 33 / 66 | 0.89 | pass | 0.88 | 34 / 52 | 0.47 | pass |
| rot30_k3 | 0.53 | 32 / 64 | 0.81 | pass | 0.79 | 36 / 52 | 0.43 | pass |
| rot30_k10 | 0.50 | 32 / 64 | 0.76 | pass | 0.71 | 32 / 48 | 0.23 | pass |
| rot45_k0 | 0.26 | 25 / 74 | 0.51 | centre | 0.72 | 33 / 56 | -0.03 | pass |
| rot45_k3 | 0.55 | 25 / 82 | 0.73 | pass | 0.80 | 33 / 67 | 0.27 | pass |
| rot45_k10 | 0.39 | 37 / 74 | 0.44 | centre | 0.56 | 36 / 74 | 0.44 | pass |
| junction | 0.67 / 0.67 / 0.22 | 34 / 85 | 0.42 | centre | 0.88 / 0.78 / 0.70 | 50 / 91 | 0.54 | pass |
| coast | - | 0 / 4 | 0.13 | pass | - | 0 / 4 | 0.13 | pass |
| strait | - | 0 / 8 | 0.16 | pass | - | 0 / 8 | 0.16 | pass |
| pass (of 15) | | | | 12 | | | | 14 |

The default run (front "seeds", no opts) gives the same 15 records. With the
front, every bend's and wall's gap lies within 0.5 h of the bisector. The gap is 6-8 h wide, and its edges
are 3.2-4.2 h from the bisector on both sides. The one failure, rot15_k10, misses the lake total
(G37: 1.01 against 1.0). Its extra charges are coast-gap dislocations (B65), not in the corridor.

The other variants on the same lakes, front off → on: `m14_blocks` 6 → 7 of 15 (centre failures gone; lake failures
on rot0_k0, rot0_k10 and the 45° bends), `m14_blocks_mix` 5 → 4. With the front, `_mix` raised B52's region-leak error
on rot45_k10 and its rot45_k0 lake gross rose 130 → 232. So `m14_blocks_mix` keeps the front off (its
override). `_tri` and `_cut` were not measured.

**The greedy placement** (`front = True`; competition s1; corridor B53 defects per 10 cells from `gaps.score` on the
final meshes):

| case | defects / 10 base → front | coverage base → front | mesh triangles base → front |
|---|---|---|---|
| uniform | 2.44 → 2.61 | 0.873 → 0.864 | 288 → 300 |
| disc | 2.12 → 1.81 | 0.880 → 0.759 | 868 → 1230 |
| band | 1.61 → 1.26 | 0.766 → 0.624 | 1916 → 2384 |
| continent | 1.73 → 1.79 | 0.861 → 0.814 | 846 → 1004 |

All gates still pass. But the cells cut the large first regions, and their neighbours cannot fill the freed space
within their own limits (B12-B14). Refilling the leftovers greedily gained only half a coverage point.
So the default leaves the greedy placement as it was.

**Drift on cube-face layouts: the price of curvature in the junction cones.** Uniform's seams drift
because every block has λ = 0 on its outline. Its outline lattice lines then curve with the enclosed curvature (B ≈
1.2, about 27° per side). The alternative is straight-sided block images, with the curvature in the corner cones at
the junctions. A gnomonic cube face has great-circle sides and 120° corners. Its conformal factor with straight
images (Neumann λ, a sink of π/6 at each cube corner) has λ ~ ¼ ln r at the corners. A probe gives its
half-range F over the block minus a junction disc of radius r_j:

| h | block | Dirichlet F (today) | F, r_j = 3 / 6 / 9 / 12 h |
|---|---|---|---|
| 100 km (uniform s1) | face (B = 1.33) | 0.086 | 0.32 / 0.24 / 0.17 / 0.14 |
| 100 km | quarter face (B = 0.33, one cube corner) | 0.019 | 0.31 / 0.23 / 0.17 / 0.14 |
| 50 km | face | 0.087 | 0.42 / 0.32 / 0.27 / 0.22 |
| 50 km | quarter face | 0.019 | 0.42 / 0.31 / 0.26 / 0.22 |

The corner singularity sets the price, not the block's size (a quarter face costs as much as a whole face), and the
price grows with resolution (ln of the cells to the corner). At 6 h junction discs it is |ln s/h| ≈ ±0.23-0.31 inside
the blocks, against `f_max` 0.08, the cell trim 0.10 (B22) and the size tolerance 0.20. A finer split helps only the
blocks without a cube corner: their corners carry fractional excess, which the junctions then hold as cycle residuals,
so the drift moves rather than goes. Not built. Within today's budgets, the drift stays the price of λ = 0
outlines.

## Coastal bands: the offshore seam (2026-10-26; decisions C50)

Code: `m14/coastband.py` `COUNT_ROW`, `END_RAW`, `_build`; option `band_gap` (4.5 h) in `m14_blocks.DEFAULTS`;
`m14/place.py` `clear_of_fixed`; tests `tests/test_coastband.py::test_count_row_follows_h_offshore`,
`::test_coast_row_ends_on_the_raw_coast`, `::test_band_gap_of_the_placement`.

**Re-measure.** The first bands (C36, C37) predate the 0.20 tolerance, B64, B65 and the optimisation stage. Today,
`m14_blocks_rep` + the stage, continent B at scale 1: 45/45 gates with and without bands; coast rows 1-3 (B53) 84
defects without, 23 with; the 30 offshore corridors (350 km off the cap coast, seam gates at the tolerance, G33) 27
pass without, 25 with (at the old 0.18 size gate 24 / 21; the first bands: 22 / 8). The corridor counts move by ±3 between
scale replicas (the global m02 fill), so every comparison below sums the replicas 0.98 / 1 / 1.02.

**What changed** (only with `coast_bands` on):
1. The band's along count follows h on its offshore row, not on the coast row: on a convex coast the offshore row of
   a band counted on the coast is (1 + D / ρ) too long (continent 0.055), a count mismatch the fill must take.
2. The automatic blocks keep 4.5 h from a band, not 6 h (`band_gap`): a narrower fill, its wall nearer the middle.
3. The coast row's ends are blended onto the raw coast (2 nodes): on GSHHS the band's end node sat up to 0.86 h off
   the raw coast (the 2 h filter), and the walk to the next block started with a spike (triangles with two coast
   edges, a quad of 157° next to a band on the global 100 km run).

| run (replicas) | corridors pass (tol / 0.18) | coast rows 1-3 | gates all pass |
|---|---|---|---|
| continent B, bands off (3) | 79 / 72 of 90 | 84-107 | 3 of 3 |
| continent B, bands before (3) | 78 / 64 | 23-28 | 3 of 3 |
| continent B, bands now (3) | 82 / 76 | 23-35 | 3 of 3 |
| global 100 km, off / before / now (5 / 3 / 5) | - | 805-873 / 549-566 / 516-583 | 4 / 2 / 4 |
| West Africa 200 km, off / before / now (7 / 3 / 7) | - | 319-375 / 247-275 / 234-285 | 6 / 0 / 3 |

Tried and dropped (C50): band nodes free in the final fit, 2-3 nodes from a corner, a band cell size limit, a row
taper at the ends, `band_gap` 3 or 5 h.

**Reading.** On the continent the bands now keep the coast rows clean (a quarter of the defects) and their offshore
seam passes as often as the block interior that sits there without bands. On the real coast they remove a third (global)
and a fifth (West Africa) of the coast-row defects, but West Africa at 200 km loses a coast gate in 4 of 7 replicas
(1 of 7 without): both gates sit at their edge in that run, and most of the bad quads are far from any band. `coast_bands`
stays off by default.

## Coastal bands at 50 km: the corner wedge (2026-10-26; decisions C36)

Code: `m14/coastband.py` `_wedge`, `_meet`, `_crossing`; tests `tests/test_coastband.py::
test_bands_at_a_concave_corner_do_not_cross`, `::test_crossing_bands_are_dropped`.

**The crash.** With bands on, global conforming and West Africa at h_max 50 km crashed in 5 of 6 builds (m02's pack
overflowing on a NaN cross field). Every crossing of the re-assembled coast loops was between two
consecutive bands of the same loop, on the two sides of a concave corner (seen from the water): the Tanga corner near
the Pemba channel (39 E, 6 S), the base of the Antarctic Peninsula at the Ronne front (60 W, 75 S) and the Arica bend
(70 W, 18 S). Not a narrow channel: the 3000 km2 filter drops Zanzibar and Pemba, and the crossing sat 100-150 km
off the coast. Each band ends 1 node from the corner, so the two end cuts start 2 h apart and converge: they meet
about cot(turn / 2) h out, inside a 4-row band for turns of 40-50 deg. Sharper corners the LFS rule already stops
(the other segment's coast is in its tree); gentler ones the 2 h filter does not confirm as corners, and one band
bends round them under the bay rule.

**The fix.** The bay rule (converging normals meeting d_x out allow d_x / (2 h) rows) now also runs across each corner
wedge: the rays of the last band node before the corner and the first after it bound both end cells; the row profile
ramps up one row per cell from there. Then the coast loops are re-assembled with all bands as the stitch does, and a
band with an outline chord in a crossing is dropped (it never fired in the runs below: 0 bands dropped). With bands
off nothing changes (the bands are not built).

**Measured** (`meshom run`, m14_blocks_rep + the optimisation stage; replicas `[mesh] scale` 0.98 / 1 / 1.02; ranges or
sums over the replicas; "rows" = B53 defects of the interior nodes 1-3 edge hops from the coast; the seam band is
m14.json's, on the stitch's mesh before the repair, fit and optimisation stage, automatic and fixed outlines together):

| config | bands | built | all gates pass | coast quad p99.9 (deg) | coast size p99 | rows 1-3 (sum) | irregular | triangles | nodes (k) | wall (min) | seam irr / 10 | seam \|ln s/h\| p99 | seam quad p99.9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glob100 | off | 3/3 | 2/3 | 133.1-134.5 | 0.184-0.190 | 2529 | 1529-1671 | 1471-1604 | 36.4-38.1 | 7-11 | 5.04-5.30 | 0.264-0.284 | 130.7-133.5 |
| glob100 | on | 3/3 | 3/3 | 133.4-134.4 | 0.190-0.195 | 1691 | 1279-1361 | 1260-1277 | 35.8-38.8 | 6-8 | 4.15-5.13 | 0.274-0.292 | 133.3-134.3 |
| glob50 | off | 3/3 | 2/3 | 134.5-135.9 | 0.190 | 6262 | 4182-4283 | 3840-4105 | 148.8-161.1 | 18-20 | 4.54-5.37 | 0.270-0.279 | 131.6-133.4 |
| glob50 | on | 3/3 | 2/3 | 133.9-136.9 | 0.190-0.194 | 3980 | 3419-3716 | 3306-3463 | 146.9-157.3 | 17-22 | 4.24-4.46 | 0.277-0.281 | 133.8-134.6 |
| glob25 | off | 3/3 | 3/3 | 133.9-134.5 | 0.190 | 13610 | 9544-11330 | 9108-10410 | 566.3-643.7 | 36-37 | 4.41-4.51 | 0.267-0.271 | 131.3-132.2 |
| glob25 | on | 3/3 | 3/3 | 134.3-134.7 | 0.190 | 8367 | 8757-9282 | 8124-8535 | 578.2-618.1 | 37-45 | 4.43-4.75 | 0.278-0.287 | 134.7-135.5 |
| waf100 | off | 3/3 | 3/3 | 132.8-133.4 | 0.189-0.190 | 2309 | 1559-1682 | 1488-1563 | 39.6-43.0 | 7-13 | 3.55-4.51 | 0.243-0.257 | 130.2-133.8 |
| waf100 | on | 3/3 | 3/3 | 133.1-134.7 | 0.190-0.198 | 1673 | 1448-1494 | 1323-1440 | 40.7-42.6 | 8-13 | 4.34-5.64 | 0.268-0.284 | 132.1-134.6 |
| waf50 | off | 3/3 | 2/3 | 134.3-139.4 | 0.190 | 5487 | 3870-4146 | 3650-3907 | 148.8-164.6 | 15-24 | 4.73-5.01 | 0.272-0.276 | 133.1-133.2 |
| waf50 | on | 3/3 | 2/3 | 133.8-140.0 | 0.190-0.193 | 3760 | 3344-3804 | 3049-3539 | 150.7-164.7 | 14-15 | 3.93-4.46 | 0.265-0.278 | 131.7-135.1 |
| a2loops | off | 3/3 | 3/3 | 134.4-134.8 | 0.190 | 11783 | 9785-10379 | 9002-9664 | 141.9-155.3 | 36-48 | 3.48-4.04 | 0.245-0.254 | 130.0-133.2 |
| a2loops | on | 3/3 | 3/3 | 133.6-134.8 | 0.187-0.190 | 11095 | 9261-10220 | 8580-9333 | 142.1-154.7 | 38-42 | 3.70-3.93 | 0.256-0.264 | 131.8-134.3 |

**Reading.** Every band build now succeeds (50 km: 6 of 6, was 1 of 6), no gate is lost (global 100 km passes in 3
of 3 replicas against 2 of 3 without) and the coast-row defects fall by 6 % (A2) to 33-39 % (global 25-100 km) and
28-31 % (West Africa). But the coast size p99 rises in four of six configurations (West Africa 100 km up to 0.198
against the 0.20 gate, global 100 km 0.195; 0.190 without), and the seam band's |ln s/h| p99 and quad p99.9 rise
(0.005-0.03, 1-3 deg) where the bands' offshore seams join it. So `coast_bands` stays off by default (decisions
C50).

## Layout follow-ups: small blocks, peer fronts, the gap judge (2026-10-26; decisions B68, B69, B63)

Code: options `drop_small` (0.25), `front_peer` (6 h), `gap_judge` ("raw") in `m14_blocks.DEFAULTS`; `m14/place.py`
`place` (`greedy`, `DROP_RERUNS`), `_peer_pass` (`PEER_ROUNDS`); `m14/gaps.py` `adapt(post=)`, `score(locate_all=)`;
`rep_variants.m14_blocks_rep` (its later stages as `post`). Tests `tests/test_m14_rules.py::
test_drop_small_leaves_the_small_region_to_the_fill`, `::test_peer_front_keeps_the_greedy_layout_of_its_own_seeds`,
`::test_shared_front_centres_the_gap[seeds-6.0]`, `tests/test_m14_gaps.py::test_post_judge_keeps_the_trim_the_later_stages_fix`.
Every number below is a sum over the h-scale replicas 0.98 / 1 / 1.02 (the m02
fill moves corridor counts by ±3 between runs): failing block-block corridors (seam gates, size gate at the 0.20
tolerance), their B53 defects, whole-mesh irregular nodes (valence ≠ 4), block coverage (mean).

**Small blocks (`drop_small`).** The greedy fills the space left beside its large blocks with small ones (disc: six
regions of 22 000-31 000 cells, then twenty of 100-1 000). Each small block brings four corridors and little area.
After the greedy pass the regions below 0.25 × the median are dropped, their seed neighbourhoods and nodes may not
seed again, and the greedy runs again (deterministic, so the regions before the first dropped one come back unchanged).
The freed area goes to the fill. Not with given seeds.

| case | base: failing / defects / irregular / coverage | 0.25 | 0.33 | 0.40 | 0.50 |
|---|---|---|---|---|---|
| uniform | 32 / 829 / 918 / 0.873 | unchanged | — | unchanged | — |
| disc | 75 / 1560 / 2987 / 0.882 | 72 / 1549 / 2966 / 0.882 | 55 / 1398 / 2965 / 0.880 | 53 / 1354 / 2953 / 0.880 | 49 / 1239 / 2890 / 0.878 |
| band | 52 / 1838 / 7052 / 0.754 | 47 / 1813 / 6944 / 0.755 | 47 / 1779 / 6711 / 0.756 | 55 / 1942 / 6716 / 0.751 | 44 / 1808 / 6831 / 0.736 |
| continent | 34 / 1470 / 2952 / 0.871 | 16 / 1283 / 2863 / 0.863 | 18 / 1267 / 3001 / 0.861 | 15 / 1201 / 3117 / 0.856 | 15 / 1185 / 3286 / 0.852 |
| mesh A2 | 33 / 297 / 12 237 / 0.273 | unchanged (none below) | — | 28 / 290 / 11 686 / 0.276 | — |

Every gate passes in every run. 0.25 is the only fraction that makes nothing worse on any case (continent: failing
corridors 34 → 16, irregular −3 %, coverage −0.8 points); larger fractions cut more corridors on disc but add
irregular nodes on continent (+2 to +11 %) and defects on band. On A2 0.25 drops nothing (median 380-530 cells, the
smallest block 146); 0.40 helps there (its 32-block cap refills the freed slots). Default 0.25 (B68).

**Seeds given back (`front_peer`).** Handing m14 the greedy's own seeds lost coverage with B67's front because its
cells are equal-weight Voronoi cells of unequal blocks: the bisector between a large region's seed and a small
neighbour's lies about half the large radius from its seed, so the cells cut the large regions to a third (disc
31 000 → 11 000 cells) while the small seeds' cells grow as large and their regions cannot fill them within the
acceptance rule (placement probe, region coverage greedy → front: disc 0.897 → 0.847, continent 0.902 → 0.868, band
0.787 → 0.647; regrowing every region afterwards without its cell gave 0.885 / 0.892 / 0.802 but a different layout and
changed the seam lakes: dropped). The peer front first grows the seeds one after the other, then shares a front only
where two regions' gap lies within 6 h of their bisector (|c_ij − c_ji| / 2, c_ij = how far region i reaches past the
bisector): there each region keeps to its side of the pair's bisector, block_gap / 2 from it (B67's rule for the pair).
A small seed beside a large region stops a gap away from its edge, tens of h from the bisector: no peers, the order's
layout. On every seam lake (15 cases × 3 scales) the peer pairs are the lake's seeds and the placement is identical to
B67's (byte-identical meshes). Built meshes, greedy → its seeds with B67 → with B69 (coverage, irregular):

| case | greedy | seeds, B67 | seeds, B69 |
|---|---|---|---|
| uniform | 0.873 / 918 | 0.864 / 898 | 0.864 / 898 (all six cube-face seeds are peers) |
| disc | 0.882 / 2987 | 0.730 / 4507 | 0.880 / 3004 |
| band | 0.754 / 7052 | 0.612 / 9019 | 0.750 / 7091 |
| continent | 0.871 / 2952 | 0.828 / 3451 | 0.883 / 2955 |
| mesh A2 (scale 1) | 0.288 / 4149 | 0.223 / 4002 | 0.284 / 3958 (3 replicas: 0.268 / 12 001 against 0.273 / 12 237) |

The placement with given seeds takes longer (disc 28 seeds: 8 → 39 s). Default 6 h (B69).

**The gap judge (`gap_judge`, B63).** `gap_adapt` kept a trim only when the corridor's measures improved on the raw
stitch, before `_rep`'s repair; on seam_rot15_k10 it undid the 2-row trim of corridor 0-1 that passes the lake's gate
after the repair. With `gap_judge = "post"` m14_blocks_rep hands m14 its later stages (repair,
coast stage, fit, the optimisation stage) and every stitch is judged on the mesh they make of it, its elements located
in the stitch's partition, the corridor size gate at the tolerance. On rot15_k10 (scale 1) it keeps that trim: lake
excess 1.01 → 0.95 per 100, the lake passes. Over the cases (the new defaults on; failing corridors / defects / whole-mesh
irregular / generation s):

| case | `gap_adapt` off | raw judge | post judge |
|---|---|---|---|
| uniform | 32 / 829 / 918 / 145 | 33 / 839 / 925 / 220 | 33 / 832 / 918 / 1284 |
| disc | 72 / 1549 / 2966 / 612 | 66 / 1495 / 2982 / 906 | 70 / 1476 / 2932 / 5890 |
| band | 47 / 1813 / 6944 / 1395 | 52 / 1770 / 6911 / 2262 | 45 / 1737 / 7007 / 12 073 |
| continent | 16 / 1283 / 2863 / 1010 | 18 / 1281 / 2915 / 1135 | 15 / 1279 / 2988 / 4502 |
| seam lakes (45; seam gates passed, lake excess, irregular) | 37, 22.66, 16 305 | 37, 23.20, 16 222 | 38, 22.31, 16 391 |

The post judge lowers corridor defects (disc −5 %, band −4 %) and gains one lake replica, but adds whole-mesh irregular
nodes on band (+1 %), continent (+4 %) and the lakes (+0.5 %), and repeats the repair per stitch: 3-10× the build time
(mesh A2, ~880 s of repair per stitch, was not run). 12 of the 15 lakes keep no trim. `gap_adapt` stays off and "raw"
stays its judge.

## Water-aware placement under the cut and no block cap (2026-10-26; decisions B70, B8)

Code: option `water_place` (true) in `m14_blocks.DEFAULTS`; `max_blocks` (32; `m14_blocks_rep`: none,
`rep_variants.REP_MAX_BLOCKS`; None or 0 = no cap, 0 for `[mesh] opts`);
`m14/fitness.py` `fitness(land=)` → `Fitness.wet`; `m14/place.py` `place` (seeds only at wet nodes, `Region.water`,
`Region.value` for `min_cells` and `drop_small`; the greedy without a cap); `m14_blocks.run` takes the land of the cut
(variant `_cut`: its domain; option `land`: `meshom/run.py` `_worker` hands the true coast to the m14 methods and to
`m14_blocks_rep` under treatment "cut"). m14.json: `region_water_coverage`, `seeds_on_land`, per block `seed_wet`,
`region_water_cells`. Tests `tests/test_m14_rules.py::test_water_place_seeds_in_the_water_and_counts_water_cells`,
`::test_max_blocks_none_places_until_no_seed_is_clear`, `::test_meshom_run_cut_hands_the_land_to_the_placement`.

**Why.** Under the cut the method meshes the closed sphere and never saw the land: on mesh A2, 6 of 32 seeds lay on
land (the second-largest block, 8 913 cells, on Antarctica at the South Pole; the North Pole block of 66 000 cells about
half land) and the regions held 53 % water. Every A2 variant also hit the 32-block cap.

**What.** Seeds only in the water; a region counts its water cells (each background triangle's cells times its share
of wet nodes); land still does not stop the growth (its cells are cut afterwards, an island inside a block is cut out
of it). `m14_blocks_rep` has no cap: the greedy runs until no admissible seed is `min_clear` from the rest (B68 still
drops the small ones); the other m14 variants keep 32 (below).

**Measured** (`m14_blocks_rep`, tolerance 0.20, the optimisation stage on; water coverage = final-mesh elements inside
the block outlines, by area / by count; irregular = interior nodes of valence ≠ 4):

| A2, scale 1 | blocks | water coverage | 5 largest blocks (water cells) | seeds on land | irregular | nodes | gates |
|---|---|---|---|---|---|---|---|
| cap 32, land-blind (the old default) | 32 | 0.306 / 0.365 | 32 092, 3094, 1903, 1491, 1487 | 6 | 4149 | 144 439 | 39/39 |
| cap 32, water | 32 | 0.313 / 0.371 | 32 095, 4721, 1488, 905, 902 | 0 | 3910 | 144 185 | 39/39 |
| cap 64, water | 64 | 0.387 / 0.418 | 32 040, 4717, 1484, 918, 889 | 0 | 3644 | 144 322 | 39/39 |
| no cap, land-blind | 104 | 0.412 / 0.432 | 32 131, 3089, 1488, 1335, 1316 | 33 | 3825 | 143 783 | 39/39 |
| **no cap, water** | 80 | 0.410 / 0.433 | 32 055, 4717, 1493, 901, 896 | 0 | 3801 | 144 073 | 39/39 |

| A2 islands not refined, scale 1 | blocks | water coverage | 5 largest | seeds on land | irregular | nodes | gates |
|---|---|---|---|---|---|---|---|
| cap 32, land-blind (the old default) | 32 | 0.362 / 0.404 | 32 896, 3105, 3060, 2944, 1392 | 8 | 3333 | 137 639 | 39/39 |
| cap 32, water | 32 | 0.376 / 0.428 | 32 937, 4293, 3726, 3409, 1424 | 0 | 3338 | 140 263 | 39/39 |
| cap 64, water | 63 | 0.452 / 0.465 | 32 886, 3617, 2906, 2742, 1219 | 0 | 3257 | 136 734 | 39/39 |
| no cap, land-blind | 101 | 0.448 / 0.461 | 32 905, 3107, 2899, 2747, 1220 | 36 | 3471 | 136 736 | 39/39 |
| **no cap, water** | 65 | 0.453 / 0.466 | 32 921, 3631, 2897, 2742, 1217 | 0 | 3360 | 136 810 | 39/39 |

The block cells of a region are not fixed by the placement: with more neighbours the synchronisation's count shift
(sigma, within f_max − F) coarsens a block by up to 8 %, so the same region gives fewer cells (A2 islands not
refined: the second region, 4275 water cells, is 4292 cells with 32 blocks and 3630 without the cap); compare the
coverage by area.

Replicas (h × 0.98 / 1 / 1.02; irregular summed, coverage the mean; every run 39/39 or 21/21):

| case | old default (cap 32, land-blind) | cap 64, water | new default (no cap, water) |
|---|---|---|---|
| A2: irregular / water coverage area / cells / blocks | 12 237 / 0.285 / 0.363 / 31-32 | 11 078 / 0.396 / 0.425 / 59-64 | 11 204 (−8.4 %) / 0.409 / 0.435 / 61-80 |
| A2 islands not refined | 10 245 / 0.373 / 0.416 / 32 | 10 108 / 0.443 / 0.465 / 62-63 | 10 078 (−1.6 %) / 0.446 / 0.469 / 62-72 |
| band (conforming: the cap only) | 6944 / 0.755 / — / 32 | — | 6626 (−4.6 %) / 0.764 / — / 40-45 |

A cap of 64 is a wash against no cap (irregular A2 −1.1 %, islands not refined +0.3 %; water coverage 1-1.3 points
lower); it would bind again on finer meshes, so `_rep` has none.

**Only `_rep`.** The competition re-run with no cap in every variant and band replicas of the raw
variants show that without the repair the extra seams cost: band, cap 32 → none, irregular
summed over h × 0.98 / 1 / 1.02: `m14_blocks` 12 390 → 13 607 (+9.8 %), `m14_blocks_mix` 12 195 → 12 977 (+6.4 %),
and `_mix` at h × 1.02 built a broken mesh (25/39 gates, the Euler check failed; 21/21 with the cap). So the cap stays
32 for `m14_blocks`, `_tri`, `_mix`, `_cut`, `_igm` (`max_blocks` = 0 lifts it) and only `m14_blocks_rep` runs
without it (−4.6 % on band, −8.4 % / −1.6 % on the A2 meshes).

Uniform (6 blocks), disc (25) and continent (12) never reach 32: unchanged. Seeds on land 5-8 → 0 on every A2
replica. The placement takes longer without the cap (A2 10 → 48 s; the build 840-1020 s either way).

**Tried and dropped** (placement probe, no cap): the seed's clearance measured to the land too (the most open water
first): the largest block 35 646 → 26 193 water cells, total water −1 %; land farther than 6 h from the water not
admissible (to spend a block's room on water): the Arctic block 35 646 → 26 697 water cells (Greenland's interior
becomes an obstacle), total water −2 %.
