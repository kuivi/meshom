# m02: L∞ frontal points along the cross field → sphere Delaunay → recombination

Code: `meshom/methods/m02_linf_frontal.py` (variants `m02_linf_frontal` = mixed, `m02_linf_frontal_tri` = triangles on
the 6-direction field), Numba kernels `meshom/methods/m02/packing.py`; test `tests/test_m02_linf_frontal.py`. Results:
the competition runs (`meshom compete`), scales 4 and 1, raw and `--smooth`.

- **What.** The packing form of Remacle 2013 (Baudouin 2014, Gmsh algo 9) as Georgiadis 2017 used it on the sphere:
  `background_mesh(h, 1.0)` + `cross_field` ("cones", coast nodes fixed to the tangent); a FIFO seeded with the
  resampled coast nodes in loop order (closed sphere: one bg node, the farthest from the cones); each popped point
  proposes 4 points at metric distance 1 (`s = logmean(h_p, h_q)`) along great circles in its cross directions, each
  direction turned by the field rotation seen at the half step (follows the integral line). A candidate is kept if
  the walk on the bg mesh stays in the water and no point lies in its L∞ box of half-size α·(h_q + h_j)/2, α = 0.75,
  in its cross frame (Gmsh 0.7). Neighbours: points bucketed by bg triangle, breadth-first search over bg triangles
  (Georgiadis §2.4: no R-tree). Seams: a step blocked by the other front at a gap D ≥ √2·h gets a point at D/2
  (exclusion 0.6; an L∞ off-centre in the spirit of Remacle §5); a final gap fill from bg nodes exists but is no longer
  triggered. Then `sphere_delaunay` + `recover_edges` + `remove_land`, `recombine(price 1/3)`, 30 `smooth` "size"
  sweeps + 3 "angle" sweeps (coast fixed). `_tri`: same code with n = 6 and a hexagonal exclusion zone
  (max_k |v·d_k| < lim over the 3 field directions), no recombination.
- **Mixed, scale 1, raw** (uniform / disc / band / continent): 52.7k / 221k / 437k / 198k nodes; triangles 0.8 / 0.6 /
  1.1 / 1.0 %; quad θmax max 138.3 / 137.4 / 142.0 / 138.0°, p99.9 119.8 / 118.5 / 125.5 / 124.4°; continent coast
  104.4 / 103.9°; quad SJ ≥ 0.62; elem size ratio p99 1.08–1.12, max ≤ 1.51; |ln s/h| p99 0.173 / 0.206 / 0.266 /
  0.253 (median 0.010–0.018; coast p99 0.097). Gates 16/18 (closed), 22/24 (continent): only size fidelity p99
  (whole, interior) fails. `--smooth` changes < 1.5° and nothing in the gates. Euler sum right on every run.
- **Irregular nodes, scale 1, n3 / n5 / T vs ρ+ / ρ−:** uniform 24 / 418 / 406 vs 8 / 0; disc 80 / 1304 / 1264 vs
  26 / −18; band 452 / 5176 / 4842 vs 94 / −86; continent 222 / 2098 / 1914 vs 13 / −5.5. Irregular interior nodes
  0.6–1.3 % (m01_isotropic ~27 %, m11 ~35 %): ~25× fewer, but still 50–400× the net ρ_c. Almost all are dislocation
  pairs (a leftover triangle next to a valence-5 node: a lattice row ends) along front collisions and transition
  rings; their count grows as 1/h (uniform T: 102 at scale 4 → 406 at scale 1, 16× the nodes), so their share falls
  as N^-1/2. The coastal rows are structured (coast quads ≤ 104°, coast |ln s/h| p99 0.10): the failure mode of
  the Gmsh-built FESOM-C Mediterranean meshes (bad quads at the coast) does not appear.
- **Triangles (`_tri`), scale 1:** 98–98.7 % valence-6 nodes (m01_triangles 74.6 %); θmax max 102–105°, p99
  77.7–85.9°; θmin ≥ 28.9°; elem ratio p99 1.09–1.10; |ln s/h| p99 0.197 / 0.217 / 0.285 / 0.231 (m01 0.20);
  2.1 / 10.5 / 23.6 / 13.5 s (m01 17–132 s); gates 12/14 and 18/20, only size fidelity fails.
- **Size fidelity** is the open failure: most edges are exactly h (p50 0.00 before smoothing), the p99 comes from
  ~5 % of edges at the seams (gaps in [α, 1+α]·h), short and long alike. The seam off-centre took p99 0.228 → 0.197
  (uniform), 0.255 → 0.226 (disc), 0.319 → 0.272 (continent); size sweeps 10 → 30 → 100 bring uniform 0.197 → 0.173
  → 0.159 and continent 0.272 → 0.253 → 0.234 (slow Jacobi diffusion; quad angles unchanged within 1°).
- **Variants measured (scale 4 unless said):** price 1 → 1.1–1.9 % triangles but T-shaped pairs merge into 180°
  quads (raw max 179.5°); price 1/3 bounds raw quads at 150°. α 0.7 / 0.8: +2–6 % / −1–4 % nodes, size p99 0.02–0.03
  better / 0.03–0.05 worse, irregular counts ±10 %. Midpoint turn off: no clear change. Field "smoothest": band n5 716 vs 962, but disc and
  continent 15 % more, and 2–3× slower (LU). Valence flips (m11 ops): no-op (angle limits). bg factor 0.5 vs 1.0 at scale 1:
  same quality, 3–4× slower; 2.0: same quality (continent fewer triangles, size p99 +0.01–0.03), half the time.
- **Time, scale 1, band (437k nodes, 22.8 s):** background mesh 9.2 s (Poisson points + hull), cross field 4.9 s,
  packing 3.6 s (~120k points/s, one thread; Georgiadis: 100k/s), Delaunay + recovery 1.6 s, recombination 1.2 s,
  smoothing 1.3 s. Peak RSS 0.43–1.26 GB.
- **A production version would need:** removal of the dislocation pairs (triangle + valence-5 → move/cancel, i.e.
  m05's cavity remeshing or quad diagonal swaps / doublet collapses), a seam treatment that spreads the collision gap
  over several rows (or an ODT/size relaxation with a multigrid flavour instead of Jacobi sweeps), a cheaper
  background (2h, structured or reused from the size engine), a parallel packing by fronts (Georgiadis §2.5) or a
  position-field point generator on the GPU, and coast seeds ordered per loop with layer/plateau seeds.
