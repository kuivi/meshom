# m04 — Lp-CVT (L∞ Lloyd) aligned with the cross field → Delaunay → recombination

**What.** `meshom/methods/m04_lpcvt.py`, `meshom/methods/m04/energy.py`, `tests/test_m04_lpcvt.py`. Lévy & Liu 2010
(Lp centroidal Voronoi tessellation, ACM TOG 29(4), doi:10.1145/1833349.1778856) energy Σ_a (h_a/h_ref)² ∫_Vor(a) ‖(R/h_a)[e1;e2](y − x_a)‖₈⁸ dA, frame = "cones" cross field
(aligned square cell = 1, rotated 45° = 3.2). Spherical Voronoi = dual of the hull, split per triangle corner into two
signed triangles (owner, edge-midpoint, circumcentre) in the owner's gnomonic plane (exact tiling); closed-form ∫xᵖ and
exact gradient (App. A–B; finite differences agree to 1e-6), Numba. scipy L-BFGS-B, 8 rounds × 40 it, frame and h
frozen per round, step ≤ 0.3 h per round (cuts Qhull fallbacks 94 → 18 of ~350 evaluations); Delaunay kept by Lawson
flips. Multilevel start (Lévy–Liu §5.3): 10 rounds at 4h and 2h, refined by water-edge midpoints, only while
2^L·max|∇h| ≤ 0.5 (steeper: points collapse, disc ×4 |ln s/h| p99 2.1). Coast nodes fixed, not owners (their cells
over land clip the free cells). Then Delaunay + recover + remove_land, recombine(1/3), 10 "size" + 1 "angle" sweep.
Deviation: field on `background_mesh(factor 1.0)` (0.5: same quality, band ×1 peak 3.2 GB).

**Scale 1 raw (runner `--smooth`).** uniform / disc / band / continent: 50.7k / 213k / 420k / 197k nodes; triangles
0.9–1.1 %; quad θmax max 132 / 141 / 141 / 134°, p99.9 127.1–127.7° (124.2–124.5°), continent coast 125.8 / 123.1°;
elem ratio p99 1.075–1.096; |ln s/h| p99 0.128–0.137 (0.150–0.160), continent coast 0.223; gates 18/18 ×3 and 25/26
(16/18, 23/26). n3/n5/T vs ρ+/ρ−: 42/536/510 vs 8/0; 170/2090/1974 vs 26/−18; 504/5064/4638 vs 94/−86; 174/2177/2030
vs 13/−5 — 1.1–1.3 % irregular interior nodes vs ~27 % (m01_isotropic) and ~35 % (m11). Time 20 / 95 / 195 / 92 s, peak
0.5–2.0 GB. Scale 4: 16/18, 14/18, 14/18, 21/26 (elem ratio p99 1.15–1.30 and |ln s/h| p99 0.22–0.24 at g = 0.4).

**Works.** Square aligned quads, ~1 % triangles, grading passes at scale 1, coast rows follow the coast. The field
variant hardly matters (disc ×4 T: cones 484, smoothest 554, h-blind 490); p = 6 ≈ 8, p = 12 doubles T. Multilevel
helps (continent ×2 n3/n5/T 140/988/887 vs 653/1851/1303).
**Fails.** Irregular counts stay 20–70× ρ_c: dislocation pairs (leftover triangle + 5-valent node), ~1 per 100 nodes,
growing like 1/h. The energy wants them: its size term is stiff (∝ (s/h)¹⁰) while a lattice following the cones field
must change size around each cone. From an equiangular cubed sphere with the same count (8 defects, |ln s/h| p99 0.212)
8 rounds create 42 pairs and reach 0.178; the Poisson start ends at ~100 pairs and 0.143. Defects are enriched near
cones (15 % within 5h vs 1.5 % of nodes) but spread over whole faces. Fixed coast nodes fix the lattice phase on each
bank: in the channel the phases conflict (66 of 68 coast edges with |ln s/h| > 0.2).
**Time** (disc ×1): field 6 s, levels 5 / 17 / 63 s (at h: 344 evaluations, Delaunay upkeep 29 s, energy 10 s, rest
scipy), recombine + smoothing 1 s.
**Production would need.** A size term that admits the conformal factor of the field (or Ekelschot-style insertion /
removal that annihilates dislocation pairs), coast nodes sliding along the coast, and the optimiser on the device (the
energy is a per-triangle gather/scatter, JAX-ready; the Delaunay upkeep is the serial part).
