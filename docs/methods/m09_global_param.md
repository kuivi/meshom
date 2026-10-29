# m09 — global seamless parametrisation (integer-grid map) → quad extraction

**What.** `meshom/methods/m09_global_param.py`, `meshom/methods/m09/{param,extract}.py`, `tests/test_m09_global_param.py`.
MIQ (Bommes, Zimmer and Kobbelt 2009, doi:10.1145/1531326.1531383) / QuadCover / integer-grid maps (Bommes et al.
2013, doi:10.1145/2461912.2462014) on a background mesh (coast at h,
interior at 1.0 h; 0.5 h is 10–30× slower and worse) with the "cones" field. Field combed on a dual BFS tree; cut graph =
tree complement pruned to cones + coast loops; map w = u + iv with target dw = (R/h) e^{-iΘ} dx, transitions i^r w + t.
Simplification: **rigid coast** (integer iso-line, one unit per resampled coast edge: coast nodes land on integer points).
Constraints eliminated by Gauss–Jordan over Gaussian integers; Schur complement onto the integers (cones, coast anchors,
free chain translations: 7–171); **rounding = MIQ greedy, one Gaussian integer at a time**, then a parity repair. Local
stiffening (MIQ §5) of flipped faces: ≤ 8 iterations before rounding, then while flips fall. Extraction (simpler than
QEx): unit cells per face, corners traced across seams by straight walks in the map, merged by position; cells touching
a fold (+2 rings) dropped, holes filled by constrained Delaunay, recombine(1/3). superlu with a nested-dissection ordering.

**Scale 1 raw (runner-smoothed).** uniform / disc / band / continent: 50k / 211k / 398k / 166k nodes; triangles 0 /
0.002 / 0.03 / 0.02 %. n3/n5/other/T 8/0/0/0, 23/19/0/4, 43/141/14/134, 3/27/4/32 vs ρ+/ρ− 8/0, 26/−18, 94/−86, 13/−5:
**0.02–0.05 % irregular interior nodes, 25–60× fewer than m02/m03/m04 (0.6–1.3 %)**; uniform = the cube. Quad θmax
max/p99.9 140/108, 178/116, 179/146, 180/160° (123/102, 128/112, 175/134, 176/157°; m02–m04: p99.9 118–128°), continent
coast 158/146°. Elem ratio p99 1.02–1.10. |ln s/h| p99 0.27 / 0.27 / 0.44 / 0.63 (coast 0.88; m02–m04 0.13–0.31).
Gates 12/14, 8→16/18, 4→6/18, 5→8/24. Time 2.8 / 30 / 255 / 67 s, peak 0.65 / 2.3 / 3.2 / 2.2 GB. Scale 4: 0.4–3.4 s,
n3/n5/T 8/0/0, 20/14/2, 52/56/16, 1/17/29, p99.9 131–159°, |ln s/h| p99 0.26–0.55. **h-gradient misfit:** energy/target
0.7–0.9 % (uniform, disc), 2 % (band ×4), 6.7 % (continent ×1); map |ln scale| p99 0.21 / 0.21 / 0.26 (×4) / 0.44,
anisotropy p99 1.2–1.75 (continent 3.2).
**Works.** Irregular nodes ≈ the field's cones, long regular rows, almost no triangles, exact coast rows, smooth interior
grading. **Fails.** (1) Worst quads at the rounded cones (rounding moves them ≤ ~0.7 h): need smoothing. (2) Folds
where cones are < 3 h apart (band: 38 pairs, min 1.2 h) or 2.5–3.5 h from the rigid coast (band ×1 160 flipped faces,
continent 377): stiffening clears the continuous map, 7–9 stay after rounding → triangle-filled holes. (3) Rigid coast
pins the scale: continent edges exact within 10 h of the coast, +14 % beyond 50 h (curvature of g/h² not balanced by
the quantised cones drifts into the interior); |ln s/h| p99 ≈ 1 within 20 h of the coast. (4) Parity: an odd coast edge
count (continent ×4: 335) leaves one cone half-integer.
**Time** (band ×1): background 8, field 4, parametrisation 235 s (10 complex LUs of 400k unknowns, ~23 s each, from
stiffening; without it 41 s but 279 triangles and a broken Euler sum), extraction 4 s.
**Production would need.** A foldover-free map (IGM / Myles 2014 constraints or QGP quantisation) with close ± cones
merged in the field; a non-rigid coast (units per coast piece as integers, coast resampled from the map); CHOLMOD/PARDISO
or device PCG instead of superlu.
