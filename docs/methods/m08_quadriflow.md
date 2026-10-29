# m08: QuadriFlow (Instant Meshes + a min-cost flow that removes position singularities) with a per-vertex scale h

Code: `meshom/methods/m08_quadriflow.py` (`m08_quadriflow` = shared cross field "cones", every position singularity
removed; `m08_quadriflow_im` = the Instant Meshes field, removed; `m08_quadriflow_keep` = the IM field, a singularity
is removed only when its partner is within flow cost 12 ≈ 6 h), steps in `meshom/methods/m08/` (`flow.py`,
`positions.py`), m07's hierarchy / fields / extraction imported unchanged; test `tests/test_m08_quadriflow.py`.
Results: the competition runs (`meshom compete`), scales 4 and 1, raw and `--smooth`. Paper: Huang, Zhou, Nießner,
Shewchuk and Guibas 2018, QuadriFlow: a scalable and robust method for quadrangulation, Comput. Graph. Forum 37(5),
doi:10.1111/cgf.13498, §3.2–3.6; code github.com/hjwdzh/QuadriFlow.

- **What.** m07 up to the integer jumps (background 0.5 h) → per background edge the jump D in the frame of its lower
  vertex and the frame turn r (m07 `edge_jumps` + `edge_frames`; D = R(ki)·diff, checked: the other sign gives 13 723
  instead of 113 singular faces) → per background triangle the frame holonomy (cone, kept) and the jump sum turned
  into its first vertex's frame (≠ 0: position singularity; paper eq. 3) → gauge: each face equation turned by g_f ∈
  Z₄ from a union-find with Z₄ offsets over the dual edges in decreasing hop distance from the cones (maximum spanning
  tree, so the unbalanced seams run between neighbouring cones; QuadriFlow: edge order, cone faces last) → network: 2
  rows per face, every jump component with opposite signs in its two faces is an arc, the others are fixed (paper),
  the coast component along the coast is an arc to a ground node (a row may end on the coast) → cost (d − u)² + 0.25
  |d − d*| per component, u = continuous offset of the IM positions in lattice units (= local h), clipped to d* ± ½;
  |d| ≤ 1 → exact min-cost flow by successive shortest paths with node potentials (one unit per Dijkstra, early stop
  at the first deficit, lazy potential update; equal to networkx network simplex on test networks: 307 = 307, 760 =
  760), strict mode with QuadriFlow's random pre-fix of component imbalance by fixed components, a search that settles
  > 10⁵ nodes gives up (the unit stays); `keep`: every row also has an arc to ground at that price → positions
  re-solved with the jumps fixed (§3.5, Gauss–Seidel, 100 sweeps; global for strict, within one ring of a changed jump
  for `keep`), kept within 1.5 ρ of their vertex → m07 extraction → 10 sweeps `smooth("size")`, coast fixed. Not done:
  §3.4 (greedy collapse + local SAT against inverted parameter triangles), long-jump subdivision (the extraction
  ignores non-unit jumps), the multi-resolution flow, LEMON / Boykov–Kolmogorov (not available; the SSP is exact and
  local).
- **Cost model matters.** QuadriFlow's L1 cost |d − d*| (unit per component) removes the same singularities but folds
  the parametrisation: inverted parameter triangles 6 → 424 (uniform s4), 18 → 1829 (disc), 15 → 3139 (continent), and
  the extraction gives 6–13 % triangles. With (d − u)² the flow prefers to re-round jumps whose IM rounding was
  marginal: 6 → 5, 18 → 16, 15 → 110; triangles 0.8–2.3 %. A global position re-solve (paper §3.5) also costs size by
  itself: with no jump changed it moves uniform s4 |ln s/h| p99 0.156 → 0.202 and band 0.197 → 0.369 (every lattice
  point leaves its nearest-to-vertex rounding), hence the local re-solve for `keep`.
- **Strict removal works on the integers, not on the mesh size.** Scale 4: singular faces 113 → 0 (uniform), 409 → 2
  (disc), 1311 → 8 (band), 380 → 0 (continent). Scale 1 the 10⁵-node search cap leaves the far units: 471 → 74, 1325 →
  415, 4815 → 1772, 1532 → 718 (cones). Scale 1, raw (final records; m07, same input, in brackets):
  `m08_quadriflow` n3/n5/T 10/134/144 (7/470/480) uniform, 40/607/620 (27/1352/1356) disc, 229/2699/2818
  (120/4677/4664) band, 38/909/1160 (41/1495/1498) continent; irregular interior nodes 0.32 / 0.31 / 0.77 / 0.52 %
  (m07 0.65–1.15 %, m02/m04 ~1 %); quad θmax p99.9 117 / 119 / 130 / 124°; but |ln s/h| p99 0.40 / 0.32 / 0.37 / 0.41
  (m07 0.11–0.13) and 16/18, 16/18, 10/18, 12/26 gates (the earlier gate set; failed gate families under
  the revised gates 2 / 2 / 5 / 7: size and triangle clusters everywhere, band and continent also hard angle and
  ratio gates).
  Removing a dislocation does not remove the strain it relieved: the IM dislocations are mostly the ones the size
  field needs (the cube-like field's conformal factor on the sphere, the grading); without them the lattice is
  integrable only with cells of size h·e^f. Even the cancelling of close pairs costs size: keep = 3 / 6 / 12 / 25
  (cones, global re-solve) gives |ln s/h| p99 0.21 / 0.27 / 0.27 / 0.27 on uniform s4, keep 6 / 12 with the local
  re-solve 0.22 / 0.24 (m07 0.156).
- **Orientation: the flow reverses m07's verdict on closed cases.** Strict, scale 4: irregular 0.5 % cones vs 1.2 % im
  (uniform), 0.9 vs 1.4 % (disc), 3.6 vs 13 % (band, im: 12.5 % triangles, |ln s/h| 1.9: an 8-cone field cannot carry
  the grading once all dislocations go) — cones fewer; continent 1.7 % cones vs 0.44 % im (im fewer, and |ln s/h| 0.35
  vs 0.47). Scale 1: cones 0.26 / 0.32 / 0.78 / 0.56 %, im 0.40 / 0.48 / 0.55 / 0.22 % (im disc breaks: quad 168°,
  ratio 4.0; im band, |ln s/h| 0.18, only looks good because the search cap leaves 2391 of 2740 singular faces). With
  the flow the cones at ρ_c pay: they let the integer layout follow h; without it (m07) they only cost dislocations.
- **`m08_quadriflow_keep` (IM field, keep 12) is the usable variant:** scale 1 raw (final records) T 470 / 1330 /
  2718 / 1112 (m07_im 508 / 1466 / 2880 / 1210, −6…−9 %), irregular 0.93 / 0.64 / 0.66 / 0.56 %, quad θmax p99.9 114
  / 114 / 117 / 113°, continent coast 130.4 / 127.9°, element ratio p99 1.05–1.08, |ln s/h| p99 0.132 / 0.122 / 0.123
  / 0.115 (m07_im 0.116 / 0.112 / 0.119 / 0.105). Gates (the earlier gate set) 18/18 on uniform, disc, band
  raw, smoothed 16/18 / 18/18 / 18/18 (uniform smoothed: |ln s/h| 0.160), continent 25/26 raw (coast |ln s/h| 0.205)
  and 24/26 smoothed. Under the revised gates it is, with m07_im, the best method: failed gate families 0 / 0 / 0 / 1
  raw (continent: the coastal size target, coast 0.221, first three rows 0.197; the strait passes at 0.048; no
  triangle cluster of 3), 1 / 0 / 0 / 2 smoothed. The flow removes only 7–12 % of the singular faces at scale 1 (508 → 446, 1397 → 1302, 2740 → 2553, 1117 →
  977): most IM dislocations have no opposite partner within ~6 h. Scale 4: T 82 / 300 / 510 / 211 vs 114 / 354 / 620
  / 325 for m07_im at |ln s/h| 0.20–0.25 vs 0.15–0.18.
- **Time** (final Levante run, scale 1: `m08_quadriflow` 18 / 93 / 225 / 130 s, `_im` 18 / 83 / 195 / 92 s, `_keep`
  13 / 64 / 142 / 71 s, m07 13 / 68 / 158 / 88 s). Earlier Mac runs (runner, with Numba compilation; flow stage in
  brackets):
  `m08_quadriflow` 14 / 58 / 143 / 90 s (flow 2.7 / 13 / 38 / 25 s), `_im` 11 / 55 / 126 / 53 s (flow 3.9 / 16 / 39 /
  15 s), `_keep` 8 / 41 / 99 / 48 s (flow 2.8 / 5.5 / 11 / 5.0 s, mostly building the network: 0.3–1.2·10⁶ settled
  nodes in all) for uniform / disc / band / continent; m07 7 / 38 / 95 / 54 s. Peak memory band: cones 3.03 GB (m07
  cones 3.2 GB), im 2.8 GB, keep 2.2 GB. The strict flow is slow where units travel far (band s4: 2.6·10⁷ settled
  nodes for 1311 units, 8 s; s1: 1.3·10⁸, mostly the 914 searches that hit the cap). Background factor 1.0 (less memory) is
  not an option: with ~1 background vertex per cell the extraction gives 30 % triangles (uniform
  s4, flow off); 0.75 gives 4.5 %.
- **Fails / weak.** Strict removal and h-grading do not mix; band at scale 4 fails for
  every variant on size like the other methods; continent strict: quad 147° at the coast, triangles < 20°, coast |ln
  s/h| 0.62; inverted parameter triangles are left to the Delaunay extraction (no SAT stage).
- **Production would need.** A target dislocation density from h (the dislocation analogue of ρ_c: the Burgers vector
  density that a lattice of size h with the chosen field needs, from the curl of the field's frame relative to ∇ln h),
  used as the flow's supplies, so that the flow cancels only unforced dislocations and moves the forced ones to where
  h wants them; a primal–dual (multi-source) or cost-scaling solver for long transport; QuadriFlow's §3.4 inversion
  repair; the local position re-solve on the device (one Jacobi stencil).
