# m07: Instant Meshes (orientation + position fields, extraction) with a per-vertex scale h

Code: `meshom/methods/m07_instant_meshes.py` (`m07_instant_meshes` = orientation from `meshom.field.cross_field`
"cones"; `m07_instant_meshes_im` = the paper's own h-blind orientation field), steps in `meshom/methods/m07/`
(`hierarchy.py`, `fields.py`, `extract.py`); test `tests/test_m07_instant_meshes.py`. Results: the competition runs
(`meshom compete`), scales 4 and 1, raw and `--smooth`. Paper: Jakob, Tarini, Panozzo and Sorkine-Hornung 2015,
Instant field-aligned meshes, ACM TOG 34(6) 189, doi:10.1145/2816795.2818078, §3; code github.com/wjakob/instant-meshes.

- **What.** `background_mesh(factor 0.5)` (the paper subdivides to ≤ ρ/2) → multiresolution graph hierarchy (pairwise
  matching, ~log₂N levels; the score is the released code's max area ratio, the paper's min ratio stalls the
  coarsening: 316 levels instead of 17) → orientation: "cones" (shared field, parallel in g/h² with cones at ρ_c)
  restricted to the coarse levels, or "im" (extrinsic 4-RoSy Gauss–Seidel, 6 sweeps per level, coarse to fine) →
  4-PoSy position field, same GS, lattice points matched by the 4 × 4 floor-cell search at the edge midpoint, final
  rounding to the lattice point nearest the vertex → integer jumps per background edge → extraction. Coast: boundary
  vertices constrain the orientation (coast tangent) and one lattice line through the coast vertex (paper §3.7 edge
  brush). All sweeps are Numba loops (sequential GS, no colouring).
- **Per-vertex scale.** ρ_i = h(v_i)/R. In the matching step of edge (i, j) both lattices use the edge scale
  (ρ_i + ρ_j)/2, so neighbours agree when their lattice points are one edge-midpoint h apart (first order in |∇h|);
  the final rounding at i uses ρ_i. Variant `scale="own"` (each lattice its own ρ, as QuadriFlow's unused scale option):
  same within noise (disc scale 4: |ln s/h| p99 0.147 vs 0.162, T 294 vs 308). Option `conformal=λ`: spacing
  h·e^{λf}, f = best-fit conformal factor of the field (its rotation relative to the g/h² transport, least squares on
  the dual graph), see below.
- **Extraction (deviation).** IM's collapse of zero-jump edges (union-find in error order, skip when the two clusters
  are joined by a unit edge) gives the output vertices (weighted mean of the representative positions); clusters
  within 0.35 h merge; clusters within 0.5 h of the coast snap onto the nearest `resample_loops` node (the coast nodes
  stay exact). The faces are not walked (IM's walk can be non-manifold): the sphere Delaunay of the vertices with the
  coast recovered is classified by the lattice (unit-jump) edges; a triangle pair across a non-lattice edge whose
  four outer edges are lattice/coast edges becomes a quad (98–99.6 % of the quads); `recombine(price 1/3)` pairs the
  leftover triangles (pentagon → quad + triangle); each leftover triangle and the quad across its longest edge are
  re-split into the best of the 5 quad + triangle splits (gate-normalised worst angle); 10 sweeps `smooth("size")`,
  coast fixed. Valid by construction (CCW manifold, Euler sum exact on every run).
- **Numbers, scale 1, as produced (with `--smooth` in brackets), `m07_instant_meshes`:** 51 k / 215 k / 421 k / 197 k
  nodes (uniform / disc / band / continent; node count ∫dA/h², |ln s/h| bias ≈ 0); triangles 0.9 / 0.6 / 1.1 / 0.8 %;
  quad θmax max 133 / 137 / 144 / 147°, p99.9 114 / 114 / 120 / 115° (110–115°); continent coast 141.7 / 128.2°
  (130.8 / 122.0°); element ratio p99 1.06–1.09 (1.10–1.11), max 1.16–1.65; |ln s/h| p99 0.114 / 0.108 / 0.132 /
  0.112 (0.13–0.16); triangle θmax ≤ 107°, θmin ≥ 27°; quad corner SJ ≥ 0.55. Gates 18/18 on uniform, disc, band; continent 25/26 (coast
  |ln s/h| p99 0.166 > 0.15); with the common angle smoothing band drops to 16/18 (|ln s/h| 0.162). Euler sum 8 / −4
  on every run. `_im`: the same quality (all closed cases 18/18 also smoothed; continent 25/26, coast 0.170).
- **Irregular nodes, scale 1, n3 / n5 / T vs ρ+ / ρ−:** uniform 13 / 469 / 470 vs 8 / 0; disc 25 / 1335 / 1338 vs
  26 / −18; band 118 / 4728 / 4712 vs 94 / −86; continent 43 / 1570 / 1564 vs 13 / −6. Irregular interior nodes
  0.6–1.2 % (m01_isotropic ~27 %, m11 ~35 %; at scale 4: 2.4–4 %). The net count is right, the gross count is
  50–75 × ρ_c on the closed cases (continent 120–280 ×): nearly every triangle sits next to a valence-5 node, a position singularity (T-junction dipole,
  paper §3.6). They keep every cell square and of size h where the field cannot: on the uniform sphere the cube-like
  field's conformal scale varies like ¼ ln r around each cone, and the triangles sit near the cones (median distance
  8 h vs 16 h for random points); per cone ~14 dipoles at scale 4, ~59 at scale 1 (grows ~1/h).
- **Orientation: "cones" vs "im" (paper's GS, h-blind, 8 cones on the sphere).** Same quality, but "im" has fewer
  dipoles on graded cases: band T 2706 vs 4712, continent 1156 vs 1564 (uniform 514 vs 470, disc 1398 vs 1338). The
  cones field puts ~150 cones on band (ρ_c), and each cone costs its own dipoles; the h-blind field pays only for the
  transition (T-junction rows). For a position-field method the cones are not free: fewer cones beat cones at ρ_c.
  `smoothest` (Knöppel): between the two (disc scale 4: T 392). Peak memory band scale 1: 3.2 GB cones (over the ~3 GB budget; the 1.6 M-node background), 2.4 GB im.
- **Trade-off knob `conformal=λ`** (spacing h·e^{λf}; f from the field, −0.54 within 1 h of a cone, theory ¼ ln r):
  λ = 0.25 at scale 1: T 470 → 288 (uniform), 1338 → 944 (disc), 4712 → 3946 (band), 1564 → 1394 (continent), at
  |ln s/h| p99 0.114 → 0.154, 0.108 → 0.151, 0.132 → 0.323, 0.112 → 0.195; λ ≥ 0.5 does not lower the count further
  (scale 4) and costs more size fidelity. Default 0. A good λ needs a cleaner f near the cones.
- **Time, scale 1 (runner, includes the shared stages):** 7 / 38 / 95 / 54 s; band: background mesh at 0.5 h
  (1.58 M nodes) ~45 %, cross field ~25 %, position field (25 levels × 6 sweeps) ~15 %, hierarchy ~10 %
  (np.unique/lexsort per level), extraction ~5 %. `_im` 7 / 34 / 77 / 37 s.
- **Works.** Field-aligned near-square quads with 0.6–1.2 % irregular nodes and ≤ 1.1 % isolated triangles, all size
  gates at scale 1 (raw), exact coasts, valid by construction, linear time, every sweep a local stencil (GPU-ready).
- **Fails / weak.** Gross irregular count ≫ ρ_c (dipoles, see above); continent coast |ln s/h| p99 0.166–0.17 (the
  first row between snapped resampled coast nodes and the lattice: T-junctions where the two banks of the channel
  disagree with the lattice); isolated quads up to 144–147° (raw) at singularities; scale 4 (coarse) misses the
  element ratio and |ln s/h| gates like the other methods.
- **Production would need.** Dipole removal (m08: QuadriFlow's flow on the integer jumps, `fields.edge_jumps` /
  `edge_frames` / `extract.extract` take replaced jumps), a clean conformal factor or fewer cones (merge ± cone pairs
  closer than a few h), coloured or Jacobi sweeps on the device instead of sequential Numba GS, a background mesh at
  0.5 h without the full Poisson + Delaunay cost (the hierarchy only needs a graph), coast-aware lattice phase (coast
  nodes placed on the lattice instead of resampled independently), and IM's oriented-face walk only where the
  Delaunay classification disagrees.
