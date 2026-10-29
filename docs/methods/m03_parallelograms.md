# m03: packing of parallelograms → sphere Delaunay → recombination

Code: `meshom/methods/m03_parallelograms.py` (`m03_parallelograms`); test `tests/test_m03_parallelograms.py`. Results:
the competition runs (`meshom compete`), scales 4 and 1, raw and `--smooth`.

- **What** (Carrier Baudouin, Remacle, Marchandise, Henrotte and Geuzaine 2014, A frontal approach to hex-dominant
  mesh generation, Adv. Model. Simul. Eng. Sci. 1:8, doi:10.1186/2213-7467-1-8, the 2D part; as in Gmsh's algorithm 9,
  gmsh.info). Cross field `cones` on
  `background_mesh(domain, h, 1.0)`, sampled at any point by power-4 IDW of the 4 nearest background nodes. FIFO from
  the resampled coast nodes (frame = coast tangent) or, on the closed sphere, from one node far from the cones. Each
  point proposes 4 candidates one geodesic step L along ±t1, ±t2 of its own frame (L = mean of h at the point and at
  the first guess); a candidate is kept iff in the water and outside the exclusion square (half-side k·h_q, aligned
  with q's frame, tested in q's tangent plane) of every point kept so far. Layer-synchronous: a layer's candidates are
  evaluated in batch, then accepted one by one in FIFO order in Numba with a multilevel 3D spatial hash (exactly the
  sequential FIFO). Then `sphere_delaunay` + `recover_edges` + `remove_land`, `recombine(price=1/3)`, 40 sweeps of
  `smooth(kind="size")`, coast fixed.
- **How it differs from m02** (L∞ frontal-Delaunay, Remacle 2013): no mesh during the front; each point proposes its
  own four lattice neighbours along its frame and an exclusion square decides (no per-edge optimal point, no
  circumradius test on active triangles); one triangulation at the end.
- **Scale 1, raw (runner smoothing in brackets):** nodes 52.8k / 220k / 438k / 191k (uniform / disc / band /
  continent); triangles 0.81 / 0.59 / 1.09 / 1.04 %; quad θmax max 139.5 / 142.1 / 145.6 / 143.6° (138.6–140.3),
  p99.9 119.2 / 120.1 / 126.8 / 124.5° (114.5–121.9); continent coast 135.7 / 132.2°; elem size ratio p99 1.06–1.09
  (1.09–1.12); |ln s/h| p99 0.22 / 0.24 / 0.31 / 0.27, median 0.01–0.02. Gates 16/18 closed, 24/26 continent; only
  size fidelity p99 fails. Euler sum right on every run.
- **Irregular nodes, n3 / n5 / T vs ρ+ / ρ−:** uniform 8 / 416 / 428 vs 8 / 0; disc 53 / 1337 / 1308 vs 26 / −18;
  band 322 / 4998 / 4796 vs 94 / −86; continent 126 / 2083 / 1990 vs 13 / −6. n3 + n5 = 0.6–1.2 % of the nodes
  (m01_isotropic ~27 %, m11 ~35 %); on uniform n3 = 8 = the cube's cones. T ≈ n5: almost every triangle sits next to a
  valence-5 node, an index-neutral dislocation (a lattice row that ends) where fronts meet with a misfit.
- **Scale 4:** 2.0–3.0 % triangles, quad max 136.7–142.1°, p99.9 129.4–134.8°, |ln s/h| p99 0.21–0.29, 0.4–1.6 s.
  Elem ratio p99 fails on band (1.29), disc (1.17) and the continent coast, as for every method (g = 0.4).
- **Fails / where the size error comes from:** seams. Where two fronts meet, a candidate is rejected until the gap
  exceeds (1 + k)·h, so a strip of quads up to 1.6·h long runs along the seam; in graded zones the columns diverge
  and a new column starts only at that gap. On uniform 0.4 % of the edges are > 1.35 h, and only 8 % of them touch a
  defect: the error is in the seam strips, not at the dislocations. Smoothing spreads it slowly (uniform, size sweeps
  10 / 30 / 100: p99 0.27 / 0.24 / 0.21).
- **Option sweeps (scale 1 unless noted):** exclusion k 0.5 / 0.55 / 0.6 / 0.7 (Gmsh): disc |ln s/h| p99 0.236 /
  0.236 / 0.241 / 0.260, continent 0.283 / 0.273 / 0.271 / 0.300, continent triangles 1.64 / 1.15 / 1.04 / 0.91 %
  → 0.6. Background mesh at 0.5·h (the value first planned) instead of 1.0·h: 3× slower (disc 32 vs 12 s, band peak 2.9 vs
  1.0 GB) and no better (disc p99 0.26 vs 0.24) → 1.0. Field `smoothest` (scale 4): more triangles and twisted
  sizes (band p99 0.38 vs 0.26, continent coast quad max 144 vs 135°). Gap fill by circumcentres > 0.9·h (scale 4):
  +15 % triangles, p99 −0.01 (off). Step L = h(p) as in Gmsh vs the mean: equal within noise.
- **Time, scale 1:** 2.5 / 12.3 / 26.2 / 15.2 s, peak 0.47–1.03 GB. disc: background mesh 5.1 s, cross field 2.7 s,
  packing 1.9 s (405 layers, field sampling 0.9 s), Delaunay + recovery 1.0 s, recombine 0.6 s, 40 size sweeps 1.4 s.
  The packing is not the bottleneck; the background mesh and the field are.
- **A production version would need:** a seam treatment (detect strips of stretched quads and insert a row where the
  strip is wider than ~1.4·h, or a field-aligned anisotropic relaxation across the seam); fewer seams (fronts ordered
  by distance in g^ instead of BFS layers, or fronts from the cones outward); corner frames from the quantised coast
  field instead of a central-difference tangent (continent coast quad max 135.7°); a device version of the layer
  acceptance (independent sets of non-conflicting candidates per layer, like GPU Poisson-disc sampling). The method
  is a strong candidate: fast, ~1 % triangles, ~1 % irregular nodes, and its only failed gate comes from a known,
  local defect.
