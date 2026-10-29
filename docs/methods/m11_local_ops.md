# m11 — local operators to h (MMG / Omega_h pattern), then recombination

**What.** `meshom/methods/m11_local_ops.py` (`m11_local_ops` mixed, `m11_local_ops_tri` triangles only) and the
reusable passes in `meshom/methods/m11/ops.py` (`split_edges`, `collapse_edges`, `flip_edges` "delaunay"|"valence",
`compact`, `Coast`, `metric_length`, `edge_ratios`; triangle meshes, fixed nodes = `node_flag != 0`). Start: background
mesh at 4h (with a coast: every ~4th resampled coast node, corners kept; skipped coast nodes are inserted later by
coast-chord splits, so the result keeps exactly `resample_loops(domain, h)`). Each pass: split > 1.4 (metric midpoint
on the arc), collapse < 0.7 (link condition, no inversion, no new angle > 150°, no new edge > 1.4), Delaunay flips,
3 sweeps of `meshom.smooth` "size", flips. Each operator is a greedy maximal independent set in priority order (no two
operations share a triangle; Omega_h independent sets, Ibanez and Shephard 2017, doi:10.1137/16M1063496; PRAgMaTIc
schedule, Rokos, Gorman, Southern and Kelly 2013, arXiv:1308.2480). Metric length = arc / logmean(h_a, h_b) (Ibanez et
al. 2017, First benchmark of the Unstructured Grid Adaptation Working Group, §4). Stop when a pass splits and collapses ≤ 1e-3·N edges. Then 10 rounds of
size smoothing + valence flips (Botsch & Kobbelt 2004; new angles kept in [30°, 110°]),
`recombine(price=1/3)` (a pair merges only if its quad stays ≤ 150°), 10 size sweeps on the mixed mesh.

**Convergence (scale 1).** Five refinement passes take the count from 4h to h (each pass ~×1.9 nodes); after pass 5,
98.4 % of edges are in [0.7, 1.4]·h; pass 9: 99.87 %, 150–300 operations left; after polish 99.9 %, edge/h
p1/p50/p99 = 0.78/1.02/1.27, histogram [<0.5, 0.5–0.7, 0.7–1, 1–1.4, 1.4–2] = 0 / 0.06 / 42.0 / 57.9 / 0.02 %,
the same on all four cases. Time per pass (refined) 0.18 s at 57 k nodes, 1.0 s at 241 k, 2.4 s at 473 k; at 241 k:
split 0.21, collapse 0.22, two flip rounds 0.31, 3 size sweeps 0.26 s, of which 4 × 0.14 s is the adjacency rebuild
(argsort) each op does; recombine 1.2 s. band (473 k nodes): coarse 0.4 s, passes 1–5 4.4 s, passes 6–9 9.8 s,
polish 11 s, remesh 30 s, method 33 s.

**Numbers, scale 1, raw (with `--smooth` in brackets).** Mixed: 57 k / 241 k / 473 k / 221 k nodes (uniform /
disc / band / continent), 8.4 % triangles; quad θmax max 148–150°, p99.9 140.5–140.9° (131.6–132.1°; continent
coast 140.5 (134.8)); elem size ratio p99 1.16 (1.28); |ln s/h| p99 0.26, median 0.066 (0.40); n3/n5/T =
7918/12119/5026 on uniform (ρ+ = 8), 33337/50503/21124 on disc (ρ ±26/−18); time 2.8 / 14.1 / 32.9 / 14.4 s; peak 0.4–1.3 GB. Gates
12/18 (14/18), continent 17/26 (20/26). Triangles only: tri θmax max 110°, min angle ≥ 27.5°, elem ratio p99 1.18,
|ln s/h| p99 0.27, n5 16 % of nodes, 2.3 / 14.3 / 35.7 / 14.2 s. Euler sum correct on every run (8, continent −4).

**Works.** Robust and fast: the passes converge in 9 passes on every case without tuning, the sphere needs no
projection machinery (renormalise), coasts are exact, graded fields (disc, band) behave like uniform.
**Fails.** Size fidelity p99 0.26 > 0.15 and elem ratio p99 1.16–1.18 > 1.15: the [0.7, 1.4] window plus irregular
Delaunay connectivity; more size sweeps at fixed connectivity do not tighten it (disc, scale 4: 9 → 60 sweeps, p99
0.304 → 0.306), valence flips do a little (0.304 → 0.274). Recombining an isotropic mesh cannot follow a cross
field: n3 + n5 ≈ 35 % of nodes against ρ+ = 8 on uniform, quad p99.9 ≈ 141° > 135°.
Price 0.25 instead of 1/3 gives quads ≤ 135° by construction but 23 % triangles (disc, scale 4).
**Production would need.** Metric-orthogonal / right-triangle point placement or cross-field-aligned splits before
recombination (Tenkès–Alauzet 2022), quad operators after it (diagonal swap, doublet/diamond collapse), a maintained half-edge structure instead of per-op adjacency rebuilds, coast nodes sliding along the
coast, and the independent sets as Luby rounds on padded arrays for the device.
