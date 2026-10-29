# m01: isotropic points → sphere Delaunay → recombination

Code: `meshom/methods/m01_isotropic.py` (variants `m01_isotropic` = mixed, `m01_triangles` = triangle-only v0 path);
test `tests/test_m01_isotropic.py`. Results: the competition runs (`meshom compete`), scales 4 and 1, raw and `--smooth`.

- **What.** Coast nodes from `resample_loops` (fixed), `poisson_points` with α calibrated so that N ≈ ∫2/(√3 h²) dA
  (equilateral edge h); 60 DistMesh spring steps (Persson & Strang 2004, as OceanMesh2D and JIGSAW-GEO 2017 §4.1;
  repulsive springs, rest length 1.2·h scaled to the current total length, dt 0.3) with a full re-triangulation
  (`sphere_delaunay` + `recover_edges` + `remove_land`) after every step; a node's step is halved where a triangle
  would invert, so no node crosses the coast. ODT (Chen & Xu 2004, circumcentre mean with ρ = h⁻⁴) is an option.
  `m01_triangles` stops here; `m01_isotropic` adds `recombine(price=1/3, "greedy")` and 10 sweeps of
  `smooth(kind="size")` (coast fixed).
- **Triangle v0 (`m01_triangles`), scale 1:** all triangle gates pass on all four cases: θmax max 101–106°, p99
  86.7° (≤ 90), θmin 33–36°; element ratio p99 1.10. Only |ln s/h| p99 = 0.20 fails (target 0.15; bias +0.004):
  the spread comes from the 25 % of nodes with valence 5 or 7 (12.7 % each), not from the size field. The common
  angle smoothing lowers θmax p99 to 78° without harm.
- **Mixed (`m01_isotropic`), scale 1, as produced:** 9.7–9.9 % triangles; quad θmax max 142–146°, p99.9 134.7–134.8°
  (target 135° met by a hair; continent coast 135.2° misses); quad corner SJ ≥ 0.5; element ratio p99 1.13–1.14;
  |ln s/h| p99 0.237, bias −0.057 (the equilateral node density 1.155/h² is 15 % too dense for squares). Gates
  16/18 on the closed cases, 22/26 on continent. With the runner's angle smoothing on top: quad p99.9 → 128°, but
  |ln s/h| p99 → 0.38 and element ratio p99 → 1.26, gates 14/18 (h-blind squaring of an irregular mesh).
- **Irregular nodes (the expected weakness), scale 1, n3 / n5 / T vs ρ+ / ρ−:** uniform 4915 / 10923 / 6166 vs 8 / 0;
  disc 20533 / 45087 / 25318 vs 26 / −18; band 40468 / 88838 / 49958 vs 94 / −86; continent 18611 / 40989 / 23232
  vs 13 / −6. About 27 % of the interior nodes are 3- or 5-valent: thousands of times the net count. The quads are
  not aligned: raw pairs are 60/120° rhombi (median worst-corner deviation 27.6°, 15° after size smoothing).
- **Price trade-off** (uniform, scale 4, before smoothing): price 1 → 0.3 % triangles, quad max / p99.9 167 / 163°;
  0.5 → 2.4 %, 161 / 158°; 1/3 → 9.5 %, 148 / 146°; 0.25 → 18 %, 135 / 134°, but triangle clusters ≥ 3 appear.
- **Greedy vs exact** (networkx), objective Σ cost + p·T, all cases at scales 8 and 16: gap 0.2–0.5 % at price 1/3,
  2.8–4.3 % at price 1; exact needs 58–84 s for 7·10³ triangles (greedy 0.02 s). The matching is not the
  bottleneck; the isotropic points are.
- **Smoother choice** (uniform, scale 4): ODT converges in ~20 steps but stalls at θmax p99 / max 88.5 / 102°,
  θmin 27°; springs reach 86 / 97°, 38°. ODT then springs is no better than springs alone; dt 0.4 oscillates.
- **Density factor:** triangles built at 1.075·h (node density 1/h², right for squares) bring the size-smoothed
  |ln s/h| p99 from 0.238 to 0.190 and the bias from −0.058 to +0.013 (uniform, scale 4). Not the default because
  the coast spacing also grows (the continent coast gets worse).
- **Time, scale 1:** 14 s (uniform, 59k nodes), 65 s (disc, 247k), 130 s (band, 485k), 102 s (continent, 226k);
  peak RSS 0.6–1.8 GB. About 85 % goes into the 60 full re-triangulations (a Qhull hull every step, then edge
  recovery and land removal, ≈ 65 ms per step at 3·10⁴ nodes); recombination and size smoothing take < 1 s at
  3·10⁴ nodes. The timings are noisy: other runs shared the machine.
- **Scale-4 caveat:** `scaled(h, 4)` also multiplies |∇h| by 4 (g = 0.4), so at scale 4 the element-ratio gate
  fails on band for any method (m01: 1.30), while at scale 1 it passes (1.14).
- **A production version would need:** Lawson flips instead of a hull per step, and re-triangulation only when a node
  moved more than ~0.1·h; a JAX/GPU spring kernel; edge split and collapse to fix local density (the |ln s/h| floor
  of ~0.2 comes from valence defects); the 1/h² node density for quads; a per-triangle price from its shape. No
  smoothing fixes the topology: ~27 % irregular nodes is built into recombining isotropic points, and quad
  orientation needs a cross field (m02 onward).
