# m15_tri — seeded packing + population control + per-edge size fit (pure triangles)

Code: `meshom/methods/m15_tri.py` (methods `m15_tri`, `m01_triangles_fit`, `m02_linf_frontal_tri_fit`),
`meshom/size_fit.py` (S1/S4), `meshom/popctl.py` (S3), the `seeds` option of `meshom/methods/m02_linf_frontal.py` (S2),
the m14 plug in `meshom/methods/m14/stitch.py` (options `tri_popctl`, `tri_fit`, `tri_vflips`, `tri_fit_flips` of
`m14_blocks`); tests `tests/test_size_fit.py`, `tests/test_m15_tri.py`. Results: the competition runs (`meshom compete`;
exploration, 8 CPUs, and the final setup, 16 CPUs), scales 4 and 1, raw and `--smooth`, and probes of the single
stages with the method code.

## Pipeline

1. **S2, seeded packing.** m02's frontal packing along the 6-fold field (n = 6, hexagonal exclusion α = 0.75 h, seam
   off-centres) grown from the coast nodes plus K seeds at once. `seeds="auto"` (`m02_linf_frontal.n_seeds`): one seed
   per 1000 nodes, at least the caps of radius 0.4 rad that cover the water (26 on the sphere), at most one per 500
   nodes; the seeds are a Fibonacci set of K points snapped to the nearest water bg node, duplicates dropped, those
   within half a cap radius of a coast node dropped. All fronts share one FIFO, so they advance in rounds (layer by
   layer, in seed order): deterministic. `seeds=1` (the default of `m02_*`) is the old code path, node for node.
2. **Delaunay + coast**: `sphere_delaunay` → `recover_edges` → `remove_land` (m02.triangle_mesh); the coast conforms to
   the resampled loops, coast nodes first and fixed from here on. (The cut variant exists at the run level,
   `[coast] treatment = "cut"`; not built here.)
3. **S3, population control** (`popctl`, Numba selection): 10 `size` sweeps, then ≤ 8 rounds of — free nodes of
   valence ≤ 4 deleted, then one end of every edge with ln(s/h) < −0.18 (the end with the smaller mean ln(s/h)), then
   the midpoint of every interior edge with ln(s/h) > +0.22, all as independent sets (a deleted node blocks its ring,
   an insertion its two ends; fixed priorities, ties by index); sphere Delaunay of the new point set with the coast
   recovered; 10 `size` sweeps — and ≤ 2 cleanup rounds that only delete valence ≤ 4 nodes.
4. **S4, per-edge size fit** (`size_fit`): minimise (1/n_e) Σ_e (ln ℓ_e/h_e)^6 + μ (1/n_e) Σ_t max(0, 0.5 − q_t)² over
   the free nodes; ℓ = chord, **h_e = h at the edge midpoint, frozen per round** (re-evaluated at the start of each
   round, so on a ramp the fit converges over the rounds to the score's midpoint h); q_t = det(a, b, c) / ((√3/2) h_t²),
   inactive on any triangle whose edges are within e^{±0.15} h. Variables = tangent offsets in units of h(x0); own
   L-BFGS (m = 10, two-loop in Numba) with an Armijo line search that rejects any trial with a non-positive triangle,
   so a valid start never inverts; ≤ 3 rounds × 150 iterations, Lawson flips between rounds; energy and gradient by a
   per-node gather over a CSR of the edges and blocked sums, so the result is bit-identical for any thread count.
   Fixed: node_flag ≠ 0, the boundary, the caller's mask.
5. S5 (L-BFGS CVT before the fit) was not built: S4 met its numbers without it.

## Stages against the acceptance numbers set beforehand (scale 1, raw, probes with the method code)

S1 = `size_fit` on the saved final meshes of `m01_triangles` / `m02_linf_frontal_tri`, scored with
`compete._score`. S2–S4: one probe of the stages (`seeds="auto"`, the final code), one Levante shared node, 4 CPUs.
Angles = min θmin – max θmax over all triangles; irr. = interior nodes with valence ≠ 6.

| stage | acceptance | uniform | disc | band | continent |
|---|---|---|---|---|---|
| S1 m01_triangles + fit | p99 ≤ 0.145, ratio p99 ≤ 1.15, no inverted | **0.132**, 1.131, 0 | **0.133**, 1.132, 0 | **0.133**, 1.133, 0 | **0.134** (coast 0.165), 1.132, 0 |
| S1 m02_tri + fit | ≤ 0.13 on u/d; b, c reported | **0.120** | **0.128** | 0.198 | 0.135 (coast 0.125) |
| S2 seeds, after m02's own smoothing | ≤ 0.165 u/d/b, c ≤ 0.20 | **0.153** (K 58) | **0.139** (246) | **0.155** (484) | **0.143** (207) |
| S3 population control | p99 ≤ 0.13 u/d/b, c ≤ 0.15 (whole and coast); ratio p99 ≤ 1.08; angles in [35°, 105°]; irr. ≤ 2 % | **0.127**; 1.079; 43.2–83.5°; 1.66 % | **0.115**; 1.073; 40.7–86.2°; 1.06 % | **0.123**; 1.078; 40.0–87.2°; 1.40 % | **0.117** (coast 0.117); 1.074; 39.7–86.2°; 1.14 % |
| S4 size fit (= m15_tri) | p99 ≤ 0.12 u/d/b, c ≤ 0.13 (coast ≤ 0.15); angles in [40°, 100°]; ratio p99 ≤ 1.08, max ≤ 1.35; irr. ≤ 2 %; all gates | **0.093**; 47.2–77.6°; 1.058 / 1.137; 1.66 %; 14/14 | **0.091**; 44.3–82.1°; 1.056 / 1.190; 1.06 %; 14/14 | **0.100**; 43.1–86.8°; 1.064 / 1.232; 1.40 %; 14/14 | **0.091** (coast 0.110); 43.7–81.4°; 1.055 / 1.197; 1.14 %; 20/20 |

Every acceptance number is met. S1 extras: m01 + fit keeps its 25 % irregular nodes and angles 37.9–92.9°; the
continent coast subset of m01 + fit is 0.165 (its only failed gate, 19/20); m02_tri + fit fails on band (one seed,
12/14), as expected.

**Seed count** (S2 → S3 → S4, p99; K = seeds used):

| case | K = 1 | K ≈ caps (21–26) | K = 53–100 | K = N/1000 (auto) |
|---|---|---|---|---|
| uniform | 0.197 → 0.152 → 0.107 | 0.156 → 0.125 → 0.093 | 0.151 → 0.126 → 0.092 (60) | 0.153 → 0.127 → 0.093 (58) |
| disc | 0.219 → 0.160 → 0.114 | 0.158 → 0.121 → 0.096 | 0.138 → 0.112 → 0.090 (100) | 0.139 → 0.115 → 0.090 (246) |
| band | 0.285 → 0.186 → 0.154 | 0.185 → 0.143 → 0.106 | 0.163 → 0.128 → 0.101 (100) | 0.155 → 0.123 → 0.100 (484) |
| continent | 0.231 → 0.160 → 0.115 | 0.168 → 0.130 → 0.100 | 0.153 → 0.123 → 0.096 (53) | 0.143 → 0.117 → 0.091 (207) |

(K = 1, the caps and 53–100 ran before the valence-4 deletion was added to S3; their S3/S4 angles were 30–108°.)
The cap rule alone (0.4 rad, 26 seeds) misses S2 on band (0.185): a 485 k-node band needs ~500 fronts, so the count
rule (one per 1000 nodes) decides at scale 1 and the cap rule only on small meshes (scale 4: 7 seeds, bounded by
≥ 500 nodes per cap).

**Valence-4 nodes.** The first S3 left 8–43 interior nodes of valence 4 (band 43 of 485 k); every triangle outside
[40°, 100°] after S4 touched one (band 38.0–103.5°, continent 39.5–100.1°, a 4-7-7 fan at 90/38/38°): the fit cannot
open a valence-4 fan. Deleting them first in each round (and in two cleanup rounds) gave the S3/S4 angles above
(S4 band 43.1–86.8°).

## Time and determinism

Band at scale 1 (0.48 M nodes) end to end (`m15_tri.run`, scoring excluded, Levante shared node):

| threads (job CPUs) | packing + Delaunay | population control | size fit | total |
|---|---|---|---|---|
| 1 (2) | 36 s | 120 s | 41 s | **197 s** (budget: 5 min on one core) |
| 8 (8) | 37 s | 59 s | 18 s | **114 s**; rerun 112 s, identical |

The mesh is bit-identical across reruns and across thread counts (same SHA-1 of points and triangles on 1 and 8
threads): the packing is serial, Qhull deterministic, the `size` sweeps are Jacobi maps, and the fit sums in fixed
blocks. Population control is the largest stage: 8 Qhull re-triangulations of the whole sphere (~3 s each at 0.5 M
nodes) plus 90 `size` sweeps; the packing is serial Numba (13 k nodes/s).

## Competition records (`meshom compete`, exploration, 8 CPUs, raw)

| method | scale | uniform | disc | band | continent |
|---|---|---|---|---|---|
| **m15_tri** | 1 | 0.093; 47.2–77.6°; 1.058 / 1.137; 1.66 %; **14/14**; 12 s | 0.091; 44.3–82.1°; 1.056 / 1.190; 1.06 %; **14/14**; 57 s | 0.100; 43.1–86.8°; 1.064 / 1.232; 1.40 %; **14/14**; 111 s | 0.091 (coast 0.110); 43.7–81.4°; 1.055 / 1.197; 1.13 %; **20/20**; 68 s |
| m15_tri | 4 | 0.107; 46.6–77.4°; 14/14 | 0.109; 38.8–91.7°; 14/14 | 0.131; 37.6–99.8°; 12/14 (ratio p99 1.187) | 0.119 (coast 0.298); 30.9–98.8°; 18/20 (coast size, coast ratio 1.22) |
| m01_triangles_fit | 1 | 0.132; 43.5–90.3°; 1.131; 25.4 %; 14/14; 23 s | 0.133; 14/14; 142 s | 0.133; 14/14; 299 s | 0.134 (coast 0.165); 19/20; 195 s |
| m02_linf_frontal_tri_fit | 1 | 0.120; 41.5–92.0°; 1.069; 1.35 %; 14/14; 7 s | 0.128; 14/14; 31 s | 0.196; 12/14; 67 s | 0.135 (coast 0.126); 20/20; 34 s |

Cells: |ln s/h| p99; angle range; elem ratio p99 / max; interior valence ≠ 6; gates; time (the method call). At scale 4
|∇h| is 0.4 on band and continent and every method fails the ratio gate there; continent's coast rows at
scale 4 are 1–2 cells from the channel walls. `--smooth` (the runner's 10 "angle" sweeps) moves m15_tri to 0.124–0.137
at scale 1, gates unchanged (14/14, 20/20). The fit variants: the fit alone repairs m01's tail (0.20 → 0.13) but keeps
its 25 % defects (median 0.083, ratio p99 1.13), and it cannot remove m02's one-seed seams on band (0.196).

## m14 plug (`m14_blocks_tri`)

`fill_method="tri"`: m02's asterisk packing of the gap (one front per outline and coast, no extra seeds) and its
smoothing, then S3 in the gap (8 rounds, every loop node fixed), valence flips (`popctl.valence_flips`, Σ(deg − 6)²
falls, new edge |ln s/h| ≤ 0.35), merge and the band smoothing as before, then S4 on the merged mesh with the coast and
every block node (outlines included) fixed. Scale 1, raw (exploration records; fill / seam numbers from `m14.json`):

| | acceptance | uniform | disc | band | continent |
|---|---|---|---|---|---|
| gates | 18/18 u/d/b, 23/23 c | **18/18** | **18/18** | **18/18** | 22/23 (coast p99 0.159) |
| whole \|ln s/h\| p99 (before) | – | 0.143 (0.211) | 0.141 (0.209) | 0.143 (0.214) | 0.149 (0.221) |
| fill p99 | ≤ 0.15 | 0.172 | 0.173 | 0.171 | 0.178 |
| seam band p99 | ≤ 0.18 | **0.175** | 0.184 | 0.189 | 0.194 |
| fill valence ≠ 6 (before) | reported | 20.8 % (26–30 %) | 18.6 % | 18.1 % | 17.9 % |
| time | – | 11 s | 53 s | 113 s | 60 s |

The **fit alone** on the gap (S1, `tri_popctl=0`, no flips): 16/18, 16/18, 16/18, 20/23; p99 0.165 / 0.163 / 0.165 /
0.170, fill 0.184 / 0.184 / 0.184 / 0.188, fill valence ≠ 6 28–31 %. Population control helps (0.165 → 0.143), the
valence flips a little (continent 20/23 → 22/23, valence ≠ 6 −0.3 points), fit exponent 8 or 10 ≤ 0.004. The rest is
topology: in a 6 h gap between two fixed outlines with unrelated lattice phases population control does not converge
(~5 % of the fill nodes deleted and inserted every round; keeping the round with the lowest p99 made it
worse), and 18–21 % of the fill nodes stay at valence 5 or 7. The fill's own p99 needs a gap fill that grows from
the outline edges (frontal-Delaunay), not more relaxation.

## What missed, and why

- m14 plug: fill p99 ≤ 0.15 (0.171–0.178), seam band ≤ 0.18 on disc / band / continent (0.184–0.194) and continent's
  coast subset (0.159, 22/23): the gap's topology (above).
- m15_tri at scale 4: band ratio (g = 0.4) and continent coast rows (every method fails there); nothing at scale 1.
- m01_triangles_fit, continent coast subset 0.165 (19/20): m01's coast rows are Poisson discs, the fit keeps their
  topology.

## Final setup (the final competition runs, 16 CPUs)

The final-setup runs reproduce every exploration number above node for node (same meshes; the methods do not depend
on the thread count). Times at scale 1, raw, uniform / disc / band / continent: `m15_tri` 12 / 56 / 105 / 66 s,
`m01_triangles_fit` 23 / 147 / 306 / 198 s, `m02_linf_frontal_tri_fit` 6 / 30 / 63 / 33 s, `m14_blocks_tri`
10 / 46 / 93 / 49 s. Figures (the standard set of every run):
m15_tri's `zoom_coast` on continent shows the channel mouth corner with valence-6 rows along both walls (θmax 78°).

## Not done, open

- S5 (L-BFGS CVT) not built (not needed). No run beyond scale 1.
- The cut coast variant for m15 exists only at the run level (`meshom run`, `[coast] treatment = "cut"`); measured on
  the first FESOM2-style global meshes: 162 k and 1.02 M nodes, 29/29 gates,
  |ln s/h| p99 0.117 / 0.106, angles 42.8–84.5° / 43.2–83.7°, 75 s / 501 s for 0.28 / 1.76 M sphere nodes.
- Ramp-heavy h (real coasts with g = 0.1 everywhere): measured there too; the gates hold, but 4.8 % of the interior
  nodes are irregular at g = 0.1 (2.8 % at g = 0.04) against 1–2 % on the competition cases (97–99 % plateau edges).

## Size tolerance (2026-10-23; decisions G38, M38)

With `size_tolerance` t the last stage is `meshom.size_fit.smooth_fit` instead of the size fit: the element size ratio
as small as possible within |ln s/h| ≤ t, Lawson flips between rounds as `fit_flips`. At t = 0.20: scale 1, ratio p99 1.058 / 1.056 / 1.064 / 1.055 → 1.051 / 1.051 / 1.057 / 1.049
(uniform / disc / band / continent), |ln s/h| p99 0.09–0.10 → 0.11–0.13, every gate still passes, time 11 / 53 / 110 /
69 s → 40 / 193 / 369 / 185 s. CORE2-like: ratio 1.109 → 1.086, 29/29, 85 → 230 s. The gain is small at 2–4× the time,
so the size fit (no tolerance) stays the recommendation for m15. Comparison with m14: docs/methods/m14_blocks.md
"Smoothness within a size tolerance".
