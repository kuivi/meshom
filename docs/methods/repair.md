# repair — local cavity remeshing of the defect clusters (`meshom/repair.py`)

The idea: "select a zone and remesh" as in SMS, automated from the defect maps; a larger zone is fine, so
the cavity grows when it does not improve. Code: `meshom/repair.py` (`repair(mesh, h, domain, opts, is_block=None)
-> (Mesh, info)`), variants `meshom/methods/rep_variants.py` (`m14_blocks_rep`, `m02_linf_frontal_rep`,
`m03_parallelograms_rep`: the base method, then `repair`, options under `opts["repair"]`), CLI `meshom repair RUN_DIR
[--polygon '[[lon, lat], ...]'] [--mesh raw.npz] [--out repaired] [--opts JSON] [--case CASE [--scale S]] [--figures]`
on a `meshom run` folder or a `meshom compete` run directory (section "CLI" at the end). One zone by hand (the UI's
"select a zone and remesh"): `repair_region(mesh, h, domain, polygon_lonlat, opts, is_block=None)` (below). Tests:
`tests/test_repair.py`. Thresholds: `docs/decisions.md` §11.

## Mechanism

1. **Detect** (once per pass, whole mesh). Defect elements: elements touching an irregular interior node (valence ≠ 4;
   ≠ 6 where all elements at the node are triangles), quads with a corner above 135°, triangles of a quad-dominant
   mesh, triangles failing a hard angle gate, the two elements of an edge with |ln s/h| > 0.15. Elements touching a
   coast node (the coast row, the metrics' `_coast` subset) are never seeds and never part of a cavity: coast nodes
   never move and are never removed, the coastal size fidelity stays exactly as it was and no size jump is added
   next to the coast row.
   **Clusters** are the edge-connected components of the defect set dilated by one element. On the competition
   meshes these components are whole dislocation walls (m02 uniform s1: 100 clusters up to 1176 elements, each wall
   around one of the 8 cube cones, net +1; m14 band s1: one component of 22 924 elements over the band ramp, seams of
   1000+ elements), so a cluster larger than `window` = 100 elements is cut into windows (breadth-first balls grown
   from the worst defect element left). Rank: w_irr (gross − |net|) + w_elem (bad elements) + w_edge (bad edges),
   then centroid z, y, x; a window with score ≤ 0 (its charge is all needed) is left alone.
2. **Cavity**: the window plus `rings` node rings of elements (block elements only from the first `block_rows` = 2 rows
   of the original block outline), closed: a pinch node gets its whole fan, a hole is filled, concave loop corners
   (turning right by more than 45°) get their fans for two sweeps (squarer cavities, so Coons applies more often).
   One outer loop (islands inside are allowed); at most 3000 elements. Its loops are exact: the loop nodes are fixed
   and shared with the rest of the mesh.
3. **Candidates**, in this order, each a patch with the loops fixed:
   - `ops` — local topology operations on the existing patch, greedy on the gross charge: quad edge rotation (the
     other two splits of the hexagon of two quads), triangle–quad pentagon re-split (moves a triangle: the glide of
     a T–5 dislocation), triangle pair merge or flip, quad diagonal collapse (removes a 3–3 diamond), doublet
     removal. When no single operation lowers the gross charge, a depth-2 lookahead tries up to 40 zero-cost moves
     (glides) each followed by a lowering operation next to it. Then 20 "size" + 3 "angle" sweeps of `meshom.smooth`
     with the loop fixed. With no operation this is plain local smoothing.
   - `coons` — one loop with exactly four left turns above 45° and no right turn below −45°: a transfinite quad grid
     when opposite sides have equal edge counts (0 irregular nodes inside); when one pair differs by k, a strip whose
     rows change their count by one per transition triangle, the k transitions spread over the middle rows (the
     "strip between the two loop sides" for a seam whose rows do not match). Then the same smoothing.
   - `m02` — m02's own fill of the cavity: its cross field on the cavity's triangulated elements (loop tangent
     fixed), L∞ frontal packing seeded with the loop nodes, sphere Delaunay closed by an antipodal dummy point, loop
     edges recovered, recombination (price 1/3), m02's smoothing (30 size + 3 angle sweeps). Only for cavities of at
     most 800 elements.
   - `m02_tri` (off by default) — the same with the asterisk field, triangles only.
4. **Measure** each patch on the cavity plus its first ring (valences complete at every closure node): gross charge
   Σ|deg − regular| + triangles (quad-dominant mesh), net charge (below), bad elements (quad θmax > 135°, triangle hard
   angle gates), bad edges (|ln s/h| > 0.15 on cavity edges, element size ratio > 1.15 across edges with a cavity
   element), per-gate counts of the hard gates (quad θmax > 150°, SJ < 0.3, triangle θmax > 120°, θmin < 20°, size ratio
   > 2, < 2 neighbours, > 32 elements per node, non-manifold edges), inverted elements. badness = 2 gross + 2 bad
   elements + 1 bad edges.
5. **Accept** the lowest badness (ties to the earlier kind) only if it is strictly below the old one, nothing is
   inverted, no hard gate has more violations than before, neither the gross charge nor the count of bad elements
   grows, no size jump (element size ratio > 1.15) is added next to the coast row, the net charge is the same and the
   patch topology is right (its boundary half-edges are exactly the loops, consistent orientation, same Euler
   characteristic), so the global Euler sum stays. Size jumps elsewhere may be traded, but the whole mesh keeps them
   below 0.9 % of its interior edges when the input passes the ratio gate (p99 ≤ 1.15, i.e. 1 %) and never adds any
   when it fails (`ratio_guard` "budget"; patches over the budget are rejected in rank order). These counts are the
   local form of "no gate violation that was not there" for the target gates; without them the first runs lost the
   coastal size gate on m14 continent, made 149° quads on m14 and m03 disc and lost the ratio gate on m14 uniform s4.
6. **Schedule.** Clusters in rank order; a cavity whose first ring meets a cavity accepted in the same pass waits for
   the next pass (the mesh there changed); rejected cavities block nothing. Each pass ends by applying the accepted
   patches (new nodes appended, node ids stable, coast first). A failed window is remembered with the nodes it saw and
   retried only when one of them changes. Ring stages 2, 4, 8 (`rings`, doubled up to `rings_max`): every window with a
   positive score is tried again with the larger cavity (a window that a 2-ring patch improved can still gain from
   more room); `grow_failed_only` keeps the 4- and 8-ring stages to windows that failed with fewer rings (m02 disc s1:
   |ln s/h| p99 0.170 and 16/18 instead of 0.148 and 18/18, in half the time). At most `max_rounds` = 3 rounds of the
   stages; stop when a round accepts nothing.
   The cavities of a wave (an independent set of the pending windows in rank order) are evaluated by a pool of spawned
   processes (`workers`, default the CPUs available, ≤ 16); the result does not depend on the number of workers.
   Deterministic: two runs give identical meshes, with 1 or 2 workers (test).

## One zone by hand: `repair_region`

The cavity is the set of elements whose centroid lies inside a polygon `[[lon, lat], ...]` (great-circle edges; either
orientation, the smaller side is the inside), plus `rings` node rings, closed as above. Coast-row elements may be taken
here (`region_coast_row`, default True): coast nodes are loop nodes of every cavity, so they stay fixed; deep block rows
are never taken. Same candidates, same measures, same acceptance rule; `force` = True applies the lowest-badness
patch whose topology is right, with nothing inverted and the net charge kept, even when it does not improve. `info`
has the cavity size, every candidate's measures with its acceptance, the chosen kind, `forced`, and the whole-mesh
summary before and after (`accepted` = a patch was applied). Example (m02 uniform s1, the worst quad, a 8° × 8° square
around 132°W 43°N, 61 elements inside, cavity of 198 with 2 rings): no candidate improves it (old badness 43; the best
patches lose on bad edges); with `force` the ops patch removes the 138.3° quad (whole-mesh quad θmax 138.3 → 132.7°)
and adds 2 bad edges and 5 size jumps, all reported. Test: a lon/lat square of ±2 cells around a rotated edge removes
both 3–5 pairs and leaves the nodes 6 cells away untouched.

## The index budget, and why a dislocation stays

A refill keeps the cavity's boundary and Euler characteristic. With c(v) = 4 − deg(v) at interior nodes,
c(v) = 2 − elements(v) at coast nodes and +1 per triangle, the **net charge of the closure** (cavity interior plus
its loop nodes)

    net = Σ_closure c(v) + T_cavity

is invariant under any refill: Euler's formula fixes Σ_interior(4 − deg) + Σ_loop(2 − e_loop) + T = 4χ, the loop nodes'
outside edges are fixed, so the charge a fill moves onto or off the loop nodes is exactly what the interior gains or
loses. The fill can lower only the gross charge Σ|c|, never below |net|; `info["budget_hist"]` records the net of
every cavity tried.

The net charge is not the only invariant. A 3–5 pair (or a triangle next to a valence-5 node, the T–5 pair that
dominates m02 and m14) has net 0 but is a dislocation: a disclination dipole whose Burgers vector is one lattice
row. Inside a fixed loop whose lattice is regular, the loop fixes the total Burgers vector, so a single pair can be
moved but not removed. Only pairs whose Burgers vectors cancel can go: a rotated edge (two opposite 3–5 pairs, the
test case), zigzags, quadrupoles, and opposite T–5 pairs that a glide brings together (the lookahead). The walls of
m02 (front collisions) and the seams of m14 (two block lattices meeting at a tilt angle θ, one dislocation per
1/θ cells) are walls of like-signed dislocations: their count is set by the lattices on both sides, which the
cavity loop crosses. A local fill cannot change that, at any ring count; only moving the cone or the lattice on one
side of the wall can (a global change). This is the main limit of the repair.

## Results on the saved final meshes (scale 1)

`meshom repair` on the `raw.npz` of the final serial competition runs (m14 with `m14.json` for the block quads),
default options, 10 CPUs (`workers` = 10) on `shared`; outputs per run: `repaired.npz`, `repair.json`
with `compete._score` before and after, `figures/repaired/`. Gross / net: charges as above (net = the Euler sum, 8 on the sphere, −4 on continent). Bad quads:
θmax > 135°; bad edges: |ln s/h| > 0.15. Accepted: ops / coons (of which strips) / m02.

| method | case | n3/n5/other/T before → after | gross / net | bad quads | bad edges | \|ln s/h\| p99 | elem ratio p99 | gates | accepted | time s |
|---|---|---|---|---|---|---|---|---|---|---|
| m14_blocks | uniform | 51/591/22/592 → 17/315/2/310 | 1278 → 646 / 8 | 3 → 0 | 1001 → 597 | 0.148 → 0.128 | 1.089 → 1.075 | 18/18 → 18/18 | 46/3(2)/50 | 46 |
| m14_blocks | disc | 131/1645/19/1560 → 75/937/5/880 | 3374 → 1902 / 8 | 0 → 0 | 2705 → 2081 | 0.128 → 0.117 | 1.090 → 1.086 | 18/18 → 18/18 | 172/4(3)/97 | 110 |
| m14_blocks | band | 394/3880/62/3618 → 256/2176/14/1956 | 8016 → 4416 / 8 | 25 → 1 | 7117 → 6191 | 0.140 → 0.133 | 1.111 → 1.110 | 18/18 → 18/18 | 511/12(10)/186 | 286 |
| m14_blocks | continent | 164/1918/30/1814 → 96/1182/5/1096 | 3956 → 2384 / −4 | 0 → 0 | 2950 → 2718 | 0.135 → 0.132 | 1.088 → 1.089 | 24/24 → 24/24 | 238/10(8)/83 | 138 |
| m02_linf_frontal | uniform | 24/418/2/406 → 23/361/1/348 | 852 → 734 / 8 | 1 → 0 | 2537 → 921 | 0.173 → 0.147 | 1.084 → 1.084 | 16/18 → **18/18** | 185/14(12)/33 | 74 |
| m02_linf_frontal | disc | 80/1304/16/1264 → 54/1048/7/1016 | 2680 → 2132 / 8 | 7 → 1 | 15302 → 3413 | 0.206 → 0.148 | 1.088 → 1.088 | 16/18 → **18/18** | 1440/74(59)/133 | 483 |
| m02_linf_frontal | band | 452/5176/55/4842 → 307/3633/11/3356 | 10580 → 7318 / 8 | 25 → 3 | 62496 → 30257 | 0.266 → 0.235 | 1.115 → 1.114 | 16/18 → 16/18 | 4300/166(118)/405 | 2122 |
| m02_linf_frontal | continent | 221/2097/19/1914 → 103/1505/4/1410 | 4270 → 3026 / −4 | 10 → 4 | 29680 → 9876 | 0.253 → 0.199 | 1.097 → 1.092 | 22/24 → 22/24 | 2084/115(89)/299 | 755 |
| m03_parallelograms | uniform | 8/416/6/428 → 12/350/2/350 | 864 → 716 / 8 | 5 → 3 | 4793 → 2336 | 0.222 → 0.192 | 1.061 → 1.075 | 16/18 → 16/18 | 391/27(23)/20 | 191 |
| m03_parallelograms | disc | 53/1337/8/1308 → 58/1100/3/1056 | 2714 → 2220 / 8 | 10 → 1 | 20023 → 13139 | 0.241 → 0.214 | 1.072 → 1.081 | 16/18 → 16/18 | 1012/62(46)/79 | 507 |
| m03_parallelograms | band | 322/4998/56/4796 → 232/3494/20/3310 | 10228 → 7076 / 8 | 46 → 6 | 69433 → 43233 | 0.305 → 0.272 | 1.095 → 1.108 | 16/18 → 16/18 | 3240/156(121)/280 | 1916 |
| m03_parallelograms | continent | 126/2081/17/1988 → 108/1411/11/1324 | 4229 → 2865 / −4 | 16 → 5 | 37802 → 17456 | 0.271 → 0.240 | 1.070 → 1.087 | 24/26 → 24/26 | 2680/162(124)/181 | 936 |
| m05_quasi_structured | uniform | 44/286/0/250 → 44/270/0/234 | 580 → 548 / 8 | 0 → 0 | 1666 → 1257 | 0.168 → 0.159 | 1.094 → 1.093 | 16/18 → 16/18 | 118/9(6)/7 | 40 |
| m05_quasi_structured | disc | 107/781/1/684 → 112/752/1/650 | 1574 → 1516 / 8 | 0 → 0 | 13186 → 4655 | 0.193 → 0.153 | 1.092 → 1.093 | 16/18 → 16/18 | 1130/54(44)/41 | 317 |
| m05_quasi_structured | band | 496/2200/6/1724 → 513/2069/5/1574 | 4432 → 4166 / 8 | 2 → 0 | 58359 → 41581 | 0.270 → 0.265 | 1.129 → 1.129 | 16/18 → 16/18 | 3182/105(49)/78 | 1251 |
| m05_quasi_structured | continent | 244/1002/4/766 → 240/938/2/702 | 2020 → 1884 / −4 | 1 → 1 | 24209 → 14356 | 0.249 → 0.236 | 1.111 → 1.111 | 22/24 → 22/24 | 1715/59(34)/83 | 507 |

Every repaired mesh keeps its Euler sum and its coast nodes (ids, positions), has no inverted element and no new
hard-gate violation; the coastal size fidelity is unchanged (coast |ln s/h| p99 0.143 / 0.097 / 0.119 / 0.097 on
continent) and no coastal size jump above 1.15 is added (coastal ratio p99 1.1134 → 1.1138, 1.107 → 1.104, 1.101 →
1.101, 1.102 → 1.102).
The size-jump budget never bound at scale 1 (0 patches rejected; 0.8–0.9 % of the interior edges allowed, 0.01–0.70 %
used before). Largest quad angle: never above the old maximum where the old one exceeded 135° (m14 uniform 140.2 →
130.0°, m03 band 145.6 → 139.9°); m05 disc 131.6 → 133.0° (below 135°, allowed). Triangle clusters of ≥ 3 (the G13
gate): m14 2 / 5 / 15 / 9 → 0 / 0 / 0 / 3 (those left are in the coast row), m03 0 / 5 / 14 / 16 →
0, m02 0 / 1 / 4 / 2 → 0 / 0 / 0 / 1. Element size ratio p99 rises a little on m03 (1.061 → 1.075, 1.095 → 1.108),
within the budget.

**Reading.**
- **m14_blocks: the gross charge falls by 40–49 %** (triangles 1560 → 880 on disc, 3618 → 1956 on band), all gates kept,
  size fidelity better everywhere (band 0.140 → 0.133). The m02 fill between the blocks and on the band ramp holds many
  T–5 pairs whose Burgers vectors cancel within a few cells (zigzags where two fronts meet a block outline); the local
  operations and the refills remove them. What stays is one dislocation per row of mismatch along each seam (the
  pictures: `figures/repaired/zoom_seam.png` of band shows the same walls, thinner).
- **m02: gross −14 / −20 / −31 / −29 %, bad edges −52 to −78 %**; **uniform and disc now pass all 18 gates** (|ln s/h|
  p99 0.173 → 0.147, 0.206 → 0.148). Most of the size gain comes from plain local "size" smoothing of a cavity (20 624
  of the 22 444 accepted ops patches had no operation): the global smoothing of m02 fixes one scale λ for the whole
  mesh, a cavity's own λ lets a lens of stretched quads relax. Band and continent keep failing size fidelity (0.235,
  0.199): their lenses are larger than an 8-ring cavity and lie far from any defect, so no window reaches
  them.
- **m03: gross −17 to −32 %, bad edges −34 to −54 %**, gates unchanged (size fidelity 0.19–0.27 stays above 0.15).
- **m05: gross −4 to −7 %**: m05 already turned m02's pairs into its quasi-structured patterns; what is left is needed
  (net) or a wall. Bad edges −25 to −65 % (smoothing), size fidelity still 0.153–0.265.
- **Candidates** (all 16 meshes): accepted 22 444 ops patches (20 624 smoothing only, 1820 with 9235 operations in
  total), 1032 Coons patches (748 of them strips with transition triangles), 2055 m02 refills; tried 196 139 cavities
  per kind. The Coons patch applies to 15 % of the cavities (28 849 valid); m02 refills are valid in 81 %, accepted in
  1.3 %. `m02_tri` was off (0 of 1291 accepted on m02 uniform in the first probe: triangles count as defects in a
  quad-dominant mesh).

**Figures looked at** (before: the run directories' `figures/raw/`; after: `figures/repaired/`):
cavity renders of the 6 largest windows (a probe) of m02 uniform before and after (dislocation
walls of T–5 pairs around each cube cone, net +1 per wall; afterwards the same walls with fewer pairs) and of m14 band
(seam windows of 24–48 dislocations, net 0, the ramp component of 22 924 elements); m14 band `zoom_irregular` and
`zoom_seam` (the ramp's mixed clusters thinned; the seam walls and a row of triangles along a block outline remain);
m02 disc `zoom_seam` (the dense rim clusters are gone, the worst seam is now a far-field wall); m03 band `overview`
(the collision walls on rows 3 and the lenses on row 1 unchanged, see below).

## Growing the cavity: rings 2, 4, 8

Round 1 of each mesh: gross / net (irregular nodes + triangles) after the stage with 2, 4 and 8 rings, then after all
rounds; acceptance = accepted / tried cavities per ring count over all rounds.

| method | case | before | after 2 rings | after 4 | after 8 | end | accepted / tried at 2 · 4 · 8 |
|---|---|---|---|---|---|---|---|
| m14_blocks | uniform | 1278 / 8 (664 + 592) | 688 (352 + 334) | 674 (348 + 324) | 668 (344 + 322) | 646 (334 + 310) | 77/264 (29 %) · 12/151 (8 %) · 10/122 (8 %) |
| m14_blocks | disc | 3374 / 8 (1795 + 1560) | 1992 (1058 + 928) | 1934 (1033 + 896) | 1912 (1023 + 884) | 1902 (1017 + 880) | 201/897 (22 %) · 48/553 (9 %) · 24/406 (6 %) |
| **m14_blocks** | **band** | 8016 / 8 (4336 + 3618) | 4684 (2581 + 2088) | 4562 (2517 + 2032) | 4484 (2478 + 1992) | 4416 (2446 + 1956) | 545/2366 (23 %) · 101/1385 (7 %) · 63/1282 (5 %) |
| m14_blocks | continent | 3956 / −4 (2112 + 1814) | 2530 (1365 + 1158) | 2476 (1331 + 1138) | 2418 (1300 + 1112) | 2384 (1283 + 1096) | 235/943 (25 %) · 49/569 (9 %) · 47/569 (8 %) |
| m02_linf_frontal | uniform | 852 / 8 (444 + 406) | 762 (397 + 364) | 742 (389 + 352) | 734 (385 + 348) | 734 (385 + 348) | 174/680 (26 %) · 40/428 (9 %) · 18/328 (5 %) |
| **m02_linf_frontal** | **disc** | 2680 / 8 (1400 + 1264) | 2254 (1170 + 1078) | 2210 (1152 + 1052) | 2178 (1132 + 1040) | 2132 (1109 + 1016) | 894/4266 (21 %) · 552/2926 (19 %) · 201/2135 (9 %) |
| m02_linf_frontal | band | 10580 / 8 (5683 + 4842) | 7746 (4155 + 3572) | 7526 (4045 + 3464) | 7404 (3984 + 3406) | 7318 (3951 + 3356) | 2237/14054 (16 %) · 1119/10406 (11 %) · 1515/15432 (10 %) |
| m02_linf_frontal | continent | 4270 / −4 (2337 + 1914) | 3298 (1756 + 1538) | 3140 (1678 + 1458) | 3044 (1625 + 1416) | 3026 (1612 + 1410) | 1121/5330 (21 %) · 707/4660 (15 %) · 670/4665 (14 %) |
| m03_parallelograms | uniform | 864 / 8 (430 + 428) | 792 (401 + 388) | 762 (385 + 374) | 720 (366 + 352) | 716 (364 + 350) | 155/928 (17 %) · 145/1102 (13 %) · 138/1124 (12 %) |
| m03_parallelograms | disc | 2714 / 8 (1398 + 1308) | 2376 (1230 + 1142) | 2268 (1180 + 1084) | 2232 (1164 + 1064) | 2220 (1161 + 1056) | 339/2521 (13 %) · 340/2895 (12 %) · 474/4787 (10 %) |
| m03_parallelograms | band | 10228 / 8 (5376 + 4796) | 7916 (4144 + 3736) | 7544 (3968 + 3546) | 7256 (3828 + 3406) | 7076 (3746 + 3310) | 1317/10498 (13 %) · 1004/10023 (10 %) · 1355/16121 (8 %) |
| m03_parallelograms | continent | 4229 / −4 (2224 + 1988) | 3255 (1731 + 1510) | 3103 (1663 + 1428) | 2923 (1559 + 1352) | 2865 (1530 + 1324) | 716/4720 (15 %) · 898/6332 (14 %) · 1409/10898 (13 %) |
| m05_quasi_structured | uniform | 580 / 8 (330 + 250) | 564 (324 + 240) | 556 (318 + 238) | 548 (314 + 234) | 548 (314 + 234) | 72/420 (17 %) · 42/366 (11 %) · 20/254 (8 %) |
| m05_quasi_structured | disc | 1574 / 8 (889 + 684) | 1524 (869 + 654) | 1522 (869 + 652) | 1518 (867 + 650) | 1516 (865 + 650) | 563/3010 (19 %) · 473/2359 (20 %) · 189/1865 (10 %) |
| m05_quasi_structured | band | 4432 / 8 (2702 + 1724) | 4216 (2611 + 1600) | 4182 (2597 + 1580) | 4168 (2589 + 1574) | 4166 (2587 + 1574) | 1116/8486 (13 %) · 896/8975 (10 %) · 1353/10978 (12 %) |
| m05_quasi_structured | continent | 2020 / −4 (1250 + 766) | 1972 (1229 + 740) | 1922 (1200 + 718) | 1888 (1181 + 704) | 1884 (1180 + 702) | 529/3496 (15 %) · 602/3779 (16 %) · 726/5385 (13 %) |

(The net stays 8 / −4 in every column.)
**What growing buys:** the 2-ring stage takes 91–94 % of the whole gross gain on m14 and 76–87 % on m02 (m14 band:
−3332 of −3600; m02 disc: −426 of −548); the 4-ring stage removes another 1.1–3.7 % of the starting gross, the 8-ring
stage 0.5–2.2 %, at an acceptance rate that falls from 16–29 % (2 rings) to 5–14 % (8 rings) on larger cavities. On m03
the larger cavities matter more (uniform: −72 / −30 / −42 at 2 / 4 / 8 rings, the 2-ring stage only 49 % of the gain;
continent 71 %): its defects come with fans of anisotropic quads that a small cavity cannot relax. The bad edges keep
falling with the ring count (m02 disc: 15 302 → 9268 → 4528 → 3708 after the 2-, 4- and 8-ring stages of round 1,
|ln s/h| p99 0.206 → 0.178 → 0.151 → 0.148: only the 8-ring stage brings it under 0.15): larger
cavities give the local smoothing room. In the m14 seams the cavities reach the first two block rows (26 133 block
elements replaced over the four m14 meshes); the strip candidate fires there but does not remove the walls: a strip
with k fewer edges on one side needs k transition triangles, each with its valence-5 node, i.e. the same gross charge
as the k dislocations it replaces; it is accepted only when it improves the shapes or the size (10 strips on m14
band).

## The `_rep` variants in the competition

Final setup: the final competition job for `m14_blocks_rep` and `m02_linf_frontal_rep` (16 CPUs on
`shared`, 15 000 MB, timeout 1800 s, figures rendered afterwards by `meshom figures`). Exploration run: `meshom compete
--methods m14_blocks_rep m02_linf_frontal_rep --cases uniform disc band continent --scale 4 1 --smooth --figures
--timeout 3600` (8 CPUs on `interactive`). Both give the same meshes as the saved-mesh repairs above;
only the times differ. Final setup, raw
(\|ln s/h\| and gates also after the runner's common smoothing):

| method | case | scale | n3/n5/other/T | quad θmax max | \|ln s/h\| p99 raw / smoothed | elem ratio p99 | gates raw / smoothed (base raw) | time s (base) |
|---|---|---|---|---|---|---|---|---|
| m14_blocks_rep | uniform | 1 | 17/315/2/310 | 130.0 | 0.128 / 0.140 | 1.075 | 18/18 / 18/18 (18/18) | 50 (8) |
| m14_blocks_rep | disc | 1 | 75/937/5/880 | 135.0 | 0.117 / 0.128 | 1.086 | 18/18 / 18/18 (18/18) | 139 (40) |
| m14_blocks_rep | band | 1 | 256/2176/14/1956 | 136.8 | 0.133 / 0.147 | 1.110 | 18/18 / 18/18 (18/18) | 334 (80) |
| m14_blocks_rep | continent | 1 | 96/1182/5/1096 | 135.0 | 0.132 / 0.144 | 1.089 | 24/24 / 23/24 (24/24) | 168 (42) |
| m02_linf_frontal_rep | uniform | 1 | 23/361/1/348 | 131.0 | 0.147 / 0.161 | 1.084 | **18/18** / 16/18 (16/18) | 56 (3) |
| m02_linf_frontal_rep | disc | 1 | 54/1048/7/1016 | 137.0 | 0.148 / 0.153 | 1.088 | **18/18** / 16/18 (16/18) | 411 (17) |
| m02_linf_frontal_rep | band | 1 | 307/3633/11/3356 | 140.4 | 0.235 / 0.244 | 1.114 | 16/18 / 16/18 (16/18) | 1673 (38) |
| m02_linf_frontal_rep | continent | 1 | 103/1505/4/1410 | 138.0 | 0.199 / 0.210 | 1.092 | 22/24 / 22/24 (22/24) | 643 (23) |
| m14_blocks_rep | uniform | 4 | 6/82/0/84 | 126.7 | 0.241 / 0.279 | 1.147 | 16/18 / 14/18 (16/18) | 35 (2) |
| m14_blocks_rep | disc | 4 | 9/181/3/186 | 132.8 | 0.207 / 0.228 | 1.205 | 14/18 / 14/18 (14/18) | 44 (3) |
| m14_blocks_rep | band | 4 | 44/450/5/424 | 134.9 | 0.207 / 0.227 | 1.304 | 14/18 / 14/18 (14/18) | 198 (5) |
| m14_blocks_rep | continent | 4 | 23/230/0/215 | 140.1 | 0.197 / 0.213 | 1.144 | 21/26 / 19/26 (21/26) | 45 (3) |
| m02_linf_frontal_rep | uniform | 4 | 8/80/1/82 | 128.1 | 0.207 / 0.237 | 1.126 | 16/18 / 16/18 (16/18) | 32 (1) |
| m02_linf_frontal_rep | disc | 4 | 16/220/2/216 | 134.5 | 0.187 / 0.209 | 1.188 | 14/18 / 14/18 (14/18) | 59 (1) |
| m02_linf_frontal_rep | band | 4 | 39/517/8/502 | 133.5 | 0.250 / 0.270 | 1.311 | 14/18 / 14/18 (14/18) | 496 (3) |
| m02_linf_frontal_rep | continent | 4 | 19/250/2/243 | 132.6 | 0.195 / 0.210 | 1.136 | 22/26 / 20/26 (22/26) | 127 (2) |

No gate is lost against the base method in any raw run (the earlier rule set lost the ratio gate on m14 uniform
scale 4: R18). Re-scored with the revised gates: failed gate families at scale 1
raw m14_blocks_rep 0 / 0 / 0 / 2 = 2 (m14_blocks 5: the triangle-cluster gate passes on uniform, disc, band; on
continent 3 clusters remain in the coast row, and the coastal size p99 0.158 is m14_blocks' own), m02_linf_frontal_rep
0 / 0 / 1 / 2 = 3 (m02 7); smoothed 2 and 5; scale 4 m14_rep 1 / 2 / 2 / 3 (4 more in m14_blocks), m02_rep 1 / 2 / 2 / 2
(2 more in m02). The runner's common angle smoothing (h-blind) gives back part of the size gain: m02_rep uniform and disc
pass 18/18 raw but 16/18 smoothed (0.161, 0.153), and m14_rep continent loses its coastal size gate when smoothed (as
m14_blocks does). At scale 4 the 8-ring cavities are large compared with the mesh and the repair costs 30–500 s on
meshes the base methods make in 1–5 s.

## What the repair cannot do, and why

- **Dislocation walls stay.** The seams of m14 and the front-collision lines of m02, m03 and m05 are walls of
  like-signed dislocations (T–5 or 3–5 pairs): their number is fixed by the lattices on both sides of the wall, which
  any cavity loop crosses. A fill can move them, merge cancelling pairs, or trade a T–5 pair for a 3–5 pair (the same
  gross charge); it cannot remove them. After the repair the m14 band seam still shows its walls, thinner
  (`figures/repaired/zoom_seam.png`); on m02 disc the worst seam moved from the dense rim (11 km cells, 24 bad edges in
  view) to a far-field wall of T–5 pairs (52 km cells). Removing a wall needs a global change: moving a cone, or
  rotating / rescaling the lattice on one side (for m14: block frames aligned gap by gap, or gaps sized from the row
  mismatch).
- **Lenses far from defects stay.** 52–81 % of m02/m03/m05's bad edges lie more than 3 h from any defect, in lenses
  of ~3000 km; the repair only opens cavities around defect clusters (an edge with |ln s/h| >
  0.15 is a defect, but a lens is far larger than an 8-ring cavity). m02 band and continent, m03 and m05 keep failing
  size fidelity.
- **The coast row is never touched** (by design, to keep coast nodes and the coastal statistics); the defects of the
  first coastal row (m14 continent: 2 triangle clusters there) stay.
- **Net charge, by construction:** gross ≥ |net| per cavity. Budget histograms: net 0 for 32–70 % of the cavities
  tried (m14 band 32 %, the ramp), the rest −1 to −27 (mostly negative: valence-5 nodes of the ramps and rims) and at
  most +2.
- **Time.** The defect maps are rebuilt over the whole mesh in every pass, in one process, and the passes chain when
  accepted cavities overlap their neighbours' rings (m02 band s1, first rule set: 380 passes, detection 540 s, cavity
  building 321 s): m02 band s1 takes 2122 s with 10 workers on `shared` and 1673 s with 16 in the final setup, close to
  its 1800 s limit (an earlier rule set timed out there); the parallel evaluation helps less than the sequential part
  costs.
- **Not built:** a single 3–5 pair "annihilated by collapsing its edge" (a proposed template): collapsing the edge
  between a valence-3 and a valence-5 node of a quad mesh turns the two quads on it into triangles and leaves a
  valence-6 node (3 + 5 − 2), so the gross charge rises; the sound operations that remove 3–5 pairs are the edge
  rotation of a rotated-edge quadrupole (test `test_rotated_edge_pairs_annihilated`) and the glide + merge of the
  lookahead. A Coons patch inside a block (only rows 0–1 are taken), annular block joins, coast nodes sliding along the
  coast.

## CLI

```
meshom repair RUN_DIR [--polygon '[[lon, lat], ...]'] [--mesh raw.npz] [--out repaired] [--opts JSON] [--figures]
meshom repair RUN_DIR --case CASE [--scale S] ...        # a `meshom compete` run directory
```

- **Domain and h.** Without `--case`, RUN_DIR is a `meshom run` folder: its `config.resolved.toml` (else
  `config.toml`) is read and the coast and h are built as `meshom run` builds them (`run.coast_of`, `run.size_field`:
  layers, clip, limiter, scale); the domain is the closed sphere for `[coast] source = "none"`, the coast loops
  resampled at h for `treatment = "loops"`, and the mesh's own boundary loops (`run._boundary_domain`) for
  `treatment = "cut"`, as the run scores it; gates are the config's (`cfg.gates_all()`). With `--case` the
  competition case at h × `--scale` (default: from the directory name `<method>_<case>_s<S>`), as before.
- **Input** `--mesh` (relative to RUN_DIR; default `raw.npz`): `smooth.npz`, or an earlier repair's `<NAME>.npz`, so
  repairs chain. **Output** `--out NAME` (a stem relative to RUN_DIR; default `repaired`): `<NAME>.npz` and
  `<NAME>.json`; `--figures` renders the picture set into `RUN_DIR/figures/<NAME>/`. (Until 2026-10-22 `--out` was a
  directory and the record was `repair.json`, the name the `_rep` variants write into a compete run directory.)
- **Record** `<NAME>.json`: `input` (the `--mesh` name), `polygon` (as given, null for the automatic repair), `opts`
  (the options used, every default filled in, `candidates` among them), `before` and `after` (`compete._score` of the
  whole mesh, the record `quality.json` holds: quality summary, size fidelity, irregular counts, ρ_c, gates passed /
  failed / families), `repair` (the info record of `repair` or `repair_region`: accepted / tried per kind, rounds,
  whole-mesh summaries; for a zone the cavity size, each candidate's measures, `kind`, `forced`, `old` / `new` local
  measures), `run`, `case` and `scale` or `config`; with `--polygon` also `region` = {`before`, `after`}:
  `region_summary` of the zone (elements with the centroid inside, without the rings; counts, n3 / n5 / n_other of
  the interior nodes inside, the compute_quality and size_fidelity statistics of those elements, the gates on them).
- **Fills tried**: `--opts '{"candidates": ["ops", "coons", "m02", "m02_tri"]}'` (any subset, in the order tried;
  default `["ops", "coons", "m02"]`, `m02_tri` off), for the automatic repair and for a zone.
- **Blocks (m14).** The block elements keep their protection (cavities take only the first `block_rows` rows): the
  mask is the input npz's `is_block`, else `m14.json`'s block-quad prefix for the method's own `raw.npz` / `smooth.npz`
  (not after a `meshom run` land cut, which renumbers the elements). `<NAME>.npz` carries `is_block` (the repair's
  `is_block_after`) when the input had blocks, so a chained repair keeps the protection; its rows are counted from the
  current block outline, so where a repair replaced the outer block rows the next one may take `block_rows` more.
  Test: `test_cli_chain_m14_blocks` (m14_blocks uniform s4 repaired twice: every deep block element of each input
  stays as it was, still marked).
