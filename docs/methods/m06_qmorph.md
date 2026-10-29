# m06: Q-Morph (indirect advancing quad front over a triangle mesh)

Code: `meshom/methods/m06_qmorph.py` (driver; `m06_qmorph` = bisector sides, `m06_qmorph_field` = cross-field sides),
`meshom/methods/m06/qmorph.py` (Numba kernel); test `tests/test_m06_qmorph.py`. Results: the competition runs
(`meshom compete`), scales 4 and 1, raw and `--smooth`. Source: Owen, Staten, Canann and Saigal 1999, Q-Morph: an
indirect approach to advancing front quad meshing, Int. J. Numer. Meth. Eng. 44(9), 1317–1340.

- **What.** Input: m02_linf_frontal_tri (default) or m01_triangles, coast nodes first. Front = region boundary
  half-edges (the coast, level 0). **Closed sphere:** 3 quads around the valence-6 node nearest a +1/4 cone of the
  cross field (a cube corner); the 6 edges of its star are the first front. Owen's steps on the unit sphere in tangent
  planes: front angles at both ends; < 135° reuses the neighbour front edge as side (state 1), else k = round(θ/90°)
  elements meet there and the side direction is θ/k (bisector) or, in `_field`, the nearest cross-field direction
  (used only if within 45° at both ends of the edge); side = existing edge within 30°, else a swap, else a split on
  the ray; new top nodes moved to distance h; top edge recovered by swaps; triangles inside cleared by flood fill.
  Top edge > 1.7 h → quad + triangle (front gains an edge), < 0.6 h → triangle (front loses one). Heap order: front
  level (min of the end levels), then fail count, then state. Seams: a ray walk measures the gap to the next front;
  < 1.4 h snaps the quad onto it, < 2.5 h makes the row take half the gap. After a refusal: local Delaunay repair.
  End: leftover clusters with inner nodes and a 3–6 node loop are re-split (`close_pockets`), the rest get Delaunay
  flips and recombination (price 1/3), then `smooth` "size" ×30 + "angle" ×3, coast fixed.
- **Scale 1, `_field`, raw** (final competition runs, Levante; runner smoothing in brackets):
  nodes 50.4k / 211k / 414k / 193k (uniform / disc / band / continent); triangles 1.0 / 0.8 / 1.2 / 0.8 %; quad θmax
  max 141.7 / 149.8 / 150.0 / 147.5°, p99.9 127.7 / 126.8 / 129.1 / 127.2° (125.1–127.4); continent coast 134.8 /
  134.5°; elem ratio p99 1.09–1.13; **|ln s/h| p99 0.169 / 0.172 / 0.193 / 0.167** (0.177–0.206), the best of the
  frontal methods (m02 0.17–0.27, m03 0.22–0.31). n3 / n5 / T vs ρ+ / ρ−: 83 / 575 / 518 vs 8 / 0; 309 / 1940 / 1716
  vs 26 / −18; 1078 / 5881 / 5042 vs 94 / −86; 250 / 1678 / 1508 vs 13 / −6 → irregular interior nodes 1.32 / 1.08 /
  1.71 / 1.02 % (m02 0.84 / 0.63 / 1.30 / 1.19 %, m03 similar, m01_isotropic ~27 %).
- **Hard-gate failures on disc and band (final run; they rule out production use):** disc keeps three triangles that
  break θmin (6.1°), θmax (132.8°) and the element ratio max (2.56), 5 700–8 200 km from the disc; band one quad at
  150.05° and a triangle with θmin 19.0°. The finishing pass does not repair them and the runner's smoothing keeps
  them; two fresh runs on Levante give the identical meshes. Gates (the earlier gate set) 16/18, **10/18,
  12/18**, 22/26; failed gate families under the revised gates 2 / 5 / 4 / 3 (size p99 and triangle clusters on
  every case; + ratio max, tri θmax, tri θmin on disc; + quad θmax, tri θmin on band; + coastal ratio p99 on
  continent, where the coast |ln s/h| p99 is 0.32). An earlier run on a Mac with older code gave "16/18 closed,
  20/26 continent, tri θmax ≤ 118°". Not for a production shortlist until local finishing repairs the surviving
  slivers. Scale 4: 2.5–3.3 % triangles, 3.3–4.8 % irregular, p99.9 129.8–134.8°.
- **Field vs bisector** (scale 1): the bisector variant grows concentric rings with ~4k expansions on uniform and ends
  at 11 % triangles, **15–16 % irregular nodes** (uniform 952 / 6630 vs 73 / 563), quad max 155–164°, tri min 3.4°:
  the cross field cuts irregular nodes ~12× and is the variant worth keeping.
- **Input mesh** (`_field`): m01 vs m02 are equal within noise at scale 4; at scale 1 continent m01 gives 0.90 %
  irregular / 0.67 % triangles / p99 0.164 vs m02 1.12 % / 0.83 % / 0.174, uniform equal, but m01 costs 136 vs 23 s.
  m02 is the default.
- **Time** (final run, scale 1: 5.4 / 26.8 / 59.0 / 31.6 s). Split from the earlier Mac run (continent s1, 23 s): m02
  triangles 14.5 s (63 %), cross field 4.1 s, Q-Morph advance 1.4 s, finish 3.0 s; band s1 55 s, 1.0 GB peak there.
  The advance itself is cheap; the input mesh dominates.
- **Fails / where defects come from:** triangles and n3/n5 pairs sit where fronts meet (seams, closure pockets) and
  where a front must expand (around islands, at channel mouths, grading); the raw advance still leaves rare slivers
  (raw tri max 176°) that flips + smoothing usually remove, not always (disc and band above). Defects are ~1.5× m02's; seam rows are near-uniform (hence the
  good size fidelity) but kinked (p99.9 ~127° vs m02 ~120°).
- **Production would need:** fronts from all coasts and seeds with level-synchronous seam closing (Owen's seam /
  transition-split templates instead of the free triangle), a size-aware choice of where a front gains or loses
  edges (along cross-field singular lines), Owen's isoparametric front smoothing during the advance, and an input
  mesh aligned with the field (the field variant on m02 already is). Sequential front: no GPU path.
