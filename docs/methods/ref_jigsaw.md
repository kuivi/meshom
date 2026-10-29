# ref_jigsaw — JIGSAW as a reference baseline (`ref_jigsaw`, `ref_jigsaw_fit`)

Code: `meshom/methods/ref_jigsaw.py`; test `tests/test_ref_jigsaw.py`. Results: the competition runs (final
setup: 16 CPUs, scales 4 and 1, raw and `--smooth`; an exploration run gave the same meshes and scores, times within
2 %); per run `.tmp/compete/<run>/ref_jigsaw.json` (grid, time split, JIGSAW's own stage
times and peak RSS, cut and coast fit) and `.tmp/compete/<run>/jigsaw/` (JIGSAW's input files `geom.msh`, `hfun.msh`,
`opts.jig` and its log).

**Not a meshom method.** The `ref_` prefix marks an external generator run as is, as a yardstick for our measures.
JIGSAW (D. Engwirda, github.com/dengwirda/jigsaw-python) is the generator behind the FESOM2 meshes (fesom2_mesh_gen,
D. Sein, AWI), triangles only.

## What is called

- The `jigsaw` executable, version 0.9.14 (conda-forge `jigsaw` 0.9.14 + `jigsawpy` 0.3.3, the separate env
  `/home/a/a270126/conda/envs/meshom-jigsaw`; the main env `meshom` stays without it,
  `environment.yml` names it in a comment). jigsawpy is not imported: meshom writes JIGSAW's text files itself, runs
  the executable in a subprocess (under GNU `time` for its peak RSS), parses its log and reads the output mesh back
  (the ctypes path of jigsawpy segfaults with a lon-lat hfun in this build; the file path works). JIGSAW
  0.9.14 here is single-threaded (no OpenMP in the build) and deterministic: the same input files give the same mesh
  bit for bit (the closed-sphere test).
- Run directory: `MESHOM_RUN_DIR` (set by `meshom compete` to `.tmp/compete/<method>_<case>_s<scale>/`), else
  `.tmp/ref_jigsaw/<pid>/<variant>`; inputs in `jigsaw/`, the output `mesh.msh` deleted after reading (`raw.npz` holds
  the mesh). Inputs are 1–5 MB per run.

## Settings (those of fesom2_mesh_gen)

| option | value | why |
|---|---|---|
| geom | `ELLIPSOID-MESH`, radii 6371 km × 3 | fesom2_mesh_gen; = `R_EARTH_M` |
| hfun | `ELLIPSOID-GRID`, h in km on a lon-lat grid (below) | fesom2_mesh_gen's resolution grid |
| `HFUN_SCAL` | absolute | fesom2_mesh_gen |
| `HFUN_HMIN` / `HFUN_HMAX` | min / max of the grid values | fesom2_mesh_gen |
| `MESH_DIMS` | 2 | fesom2_mesh_gen |
| `MESH_KERN` | default (`delfront`, frontal-Delaunay) | fesom2_mesh_gen leaves it |
| `OPTM_ITER` / `OPTM_QTOL` | 32 / 1e-5 | fesom2_mesh_gen defaults |
| `VERBOSITY` | 0 | fesom2_mesh_gen |
| gradient limiting | **off** | fesom2_mesh_gen runs JIGSAW's `marche` only when a gradient limit is configured; our h is already limited (cases g ≤ 0.1, `meshom run`: `size.limit_gradient`); `jigsaw()` itself does not limit |
| `GEOM_FEAT`, `MESH_TOP1` | TRUE, only `_fit` with a coast | JIGSAW's own examples pair them with edge geometry |
| everything else | JIGSAW default (`MESH_RAD2` 1.05, `MESH_OFF2` 0.90, `MESH_EPS1` 0.33, `GEOM_ETA1` 45°, `OPTM_KERN` odt+dqdx, zip/div on) | |

**The h grid.** JIGSAW interpolates an `ELLIPSOID-GRID` bilinearly in lon-lat, so meshom's analytic h is sampled on a
tensor grid: lon −180…180 and lat −90…90 inclusive, start step 0.5°, then every lon and lat interval is halved while
bilinear interpolation misses h by more than 1 % (relative) at an edge midpoint or a cell centre of any row or column,
down to 0.005°. The −180 and 180 columns are the same points (equal values; JIGSAW detects the period from
cos(lon_first) = cos(lon_last)). **Poles:** the ±90° rows hold h at the pole (one value per row), so JIGSAW's pole
extrapolation is not used. Grids: uniform 721 × 361 (0.5°), disc 991 × 560 (down to 0.016° / 0.031°), band 2881 × 363
(0.125°), continent 1425 × 1019 (0.031°), 0.26–1.45 M values; bilinear error on 5·10⁵ Fibonacci points max 0.9–1.1 %
(at the kinks of h), p99 < 10⁻⁷. The grid does not depend on the scale (g/h is scale-free). The cell-centre test
refines both axes, so band's lon axis is refined too (2881 instead of 721 columns; 0.8 s, harmless).

## Coast

- **`ref_jigsaw`** (treatment A, what fesom2_mesh_gen does): JIGSAW meshes the closed sphere; the elements whose
  centroid is on land are removed and FESOM2's clean-up (spurs, pinched nodes, the largest basin) applied —
  `meshom.run.cut_land` = `meshom/methods/m14/cut.py` (the rules after D. Sein (AWI), fesom2_mesh_gen; one centroid
  sample instead of fesom2_mesh_gen's 3 × 3, no strait edits).
- **`ref_jigsaw_fit`** (treatment B, conforming): the coast loops resampled at h (`resample_loops`) go to JIGSAW as
  geometry (`POINT` lon/lat in radians, `EDGE2` closing each loop) with `GEOM_FEAT` and `MESH_TOP1`. **Conforming
  edges on the sphere work in this build**: every boundary edge of the water mesh is a JIGSAW feature edge (1347/1347
  at scale 1, 352/352 at scale 4). JIGSAW meshes the whole sphere with the coast embedded; the elements with the
  centroid on land are dropped, no clean-up. Corners turning ≥ 45° (`GEOM_ETA1`; the four channel mouths) become fixed
  nodes; the other given vertices are not kept (8 of 1348 are mesh nodes): JIGSAW resamples the arcs by its own h.
  Option `fix_coast_nodes` tags the given vertices −1 (JIGSAW "user features") so every one is a mesh node.
- Without a coast both variants write byte-identical files and give the same mesh (test; identical records).

## Results (final setup, raw; both variants identical on the closed cases)

| case | scale | nodes | gates | \|ln s/h\| p99 (bias) | tri θmax max / p99 | θmin | elem ratio p99 / max | time s | failed |
|---|---|---|---|---|---|---|---|---|---|
| uniform | 4 | 3 773 | 12/14 | 0.168 (−0.008) | 78.9 / 77.3 | 44.5 | 1.090 / 1.17 | 1.5 | size |
| uniform | 1 | 59 546 | **14/14** | **0.145** (−0.003) | 79.3 / 75.7 | 43.9 | 1.079 / 1.18 | 32.9 | — |
| disc | 4 | 15 733 | 12/14 | 0.168 (−0.006) | 89.5 / 77.0 | 39.4 | 1.118 / 1.44 | 8.7 | size |
| disc | 1 | 248 966 | **14/14** | **0.134** (−0.002) | 82.5 / 75.1 | 42.1 | 1.079 / 1.28 | 229 | — |
| band | 4 | 30 746 | 10/14 | 0.166 (−0.004) | 86.9 / 77.1 | 39.0 | 1.177 / 1.52 | 17.0 | size, elem ratio (g = 0.4) |
| band | 1 | 488 459 | **14/14** | **0.124** (−0.002) | 82.2 / 74.4 | 41.1 | 1.078 / 1.27 | 548 | — |
| continent A (cut) | 4 | 14 497 | 16/20 | 0.162 (−0.005) | 88.6 / 76.8 | 38.8 | 1.089 / 1.53 | 13.7 | size (coast 0.187), coast elem ratio 1.181 |
| continent A (cut) | 1 | 228 283 | 19/20 | 0.129 (−0.002) | 80.4 / 74.7 | 43.0 | 1.077 / 1.23 | 244 | coast size 0.162 |
| continent B (fit) | 4 | 14 492 | 16/20 | 0.168 (−0.004) | 86.9 / 77.0 | 41.8 | 1.089 / 1.46 | 13.4 | size (coast 0.288), coast elem ratio 1.201 |
| continent B (fit) | 1 | 228 063 | **20/20** | **0.132** (−0.002) | 81.3 / 75.0 | 40.9 | 1.078 / 1.21 | 241 | — |

- Scale 1 reproduces an earlier probe (0.149 / 0.133 / 0.127 with case-tailored grids; here 0.145 / 0.134 /
  0.124). Scale 4 fails the size gate everywhere (0.162–0.168), uniform included; the cause was not analysed (on disc,
  band and continent the slope of 4 h is 0.4). Band at scale 4 fails the element ratio as every method.
- The runner's common angle smoothing makes JIGSAW worse: |ln s/h| p99 +0.004 to +0.027 (scale 1 +0.009 to +0.018;
  uniform 0.145 → 0.154, 12/14), θmin down 1–9°.
- Valence: 96–97 % valence 6 (probe); valence-5 interior nodes 1.2–1.9 % at scale 1 (continent 3203 of 226 785).

**Against our triangle generators (scale 1, raw, final records):** m01_triangles |ln s/h| p99 0.201 / 0.202 / 0.202 /
0.203, gates 12/14 ×3 + 17/20, 19 / 124 / 270 / 180 s; m02_linf_frontal_tri 0.197 / 0.219 / 0.285 / 0.231, 12/14 ×3 +
18/20, 3.5 / 18 / 39 / 21 s. JIGSAW: 0.145 / 0.134 / 0.124 / 0.129–0.132, all gates (A misses only the coastal size by
0.012), 33 / 229 / 548 / 241–244 s. JIGSAW's triangles are rounder (θmax p99 74–76° vs m01 86.7°, θmin 41–44° vs
33–36°) and its element ratio p99 1.08 vs 1.10. So JIGSAW is the bar for size fidelity (0.06–0.08 better than ours),
and it is 1.3–2× slower than m01 and 9–14× slower than m02_tri; m01's springs on JIGSAW's unoptimised points reach the
same 0.142 / 0.133 / 0.125, so the gap is in the point placement + topology, not in the relaxation.

## Time split (scale 1, s; one core)

| case | h grid (+ check) | files | JIGSAW wall (own: refine + **optimise**) | read | cut | coast fit | total |
|---|---|---|---|---|---|---|---|
| uniform | 0.07 + 0.09 | 0.04 | 32.6 (0.8 + **31.6**) | 0.15 | — | — | 32.9 |
| disc | 1.3 + 0.1 | 0.1 | 226.4 (3.9 + **222.0**) | 0.6 | — | — | 228.6 |
| band | 0.8 + 0.1 | 0.2 | 545.4 (8.1 + **535.0**) | 1.3 | — | — | 547.8 |
| continent A | 5.3 + 0.2 | 0.2 | 225.1 (4.0 + **220.0**) | 0.6 | 1.3 | 10.8 | 243.5 |
| continent B | 5.4 + 0.2 | 0.2 | 222.5 (4.6 + **217.0**) | 0.6 | 0.5 | 11.0 | 240.5 |

JIGSAW's own reported stage times (3 digits in its log) sum to its wall time within that rounding (file I/O
included). 97–98 % of it is the optimiser (32 passes of ODT + zip/div + flips): the frontal-Delaunay refinement alone
makes ~6·10⁴ nodes/s. The
coast-fit measurement (`m14.cut.coast_fit`, 4.5 % of the continent wall time) is part of the method call and so of the
table's time; `coast_fit: False` drops it. JIGSAW's peak RSS (GNU time, in `ref_jigsaw.json` of the exploration run):
46 / 167 / 322 / 174–175 MB at scale 1 (10–34 MB at scale 4), ~0.7 kB per node. The table's "peak MB" is the
Python child only (and carries the parent's size).

## Coast behaviour (continent)

| | boundary → coast (max / p99, × h) | coast → boundary (max) | boundary length / coast | nodes turning > 45° | coast \|ln s/h\| p99 | coast elem ratio p99 |
|---|---|---|---|---|---|---|
| A cut, scale 1 | 0.66 / 0.55 | 0.59 | 1.106 | 52 % | 0.162 | 1.088 |
| B fit, scale 1 | 0.020 / 0.019 | 0.050 | 1.000 | 0.3 % | 0.126 | 1.071 |
| B fit + `fix_coast_nodes`, scale 1 | 0.000 | 0.021 | — | — | 0.079 | — |
| A cut, scale 4 | 0.60 / 0.53 | 0.55 | 1.114 | 56 % | 0.187 | 1.181 |
| B fit, scale 4 | 0.026 / 0.015 | 0.125 | 0.998 | 1.1 % | 0.288 | 1.201 |
| B fit + `fix_coast_nodes`, scale 4 | 0.000 | 0.090 | — | — | 0.200 | 1.207 |

- **A** is the FESOM2 staircase: 219 spurs removed at scale 1 (41 at scale 4), the channel stays open (3 loops),
  net water −0.019 % (scale 4 −0.030 %); 73 % of boundary nodes have 1 or ≥ 3 elements (the "coast turn"; m14_blocks_cut
  71 %). Every boundary node within 0.66 h of the true coast (test bound 1 h; m14_blocks_cut 0.74 h).
- **B** conforms exactly (boundary nodes on the resampled polyline; the 0.02 h are its chords against the 5 km input
  loops). Its one weak spot is JIGSAW's own 1-D sampling of the coast: at scale 4, 20 % of the coast edges are more
  than 0.15 off h (p1 −0.30: short edges between the fixed corners and where JIGSAW's 1-D front closes); at scale 1
  coastal p99 0.126 passes. With our resampled nodes kept (`fix_coast_nodes`) the coastal p99 drops
  to 0.079 at scale 1 (all gates, overall 0.132) and 0.200 at scale 4 (overall 0.173 vs 0.168). Default off: the
  reference shows JIGSAW's own coast handling.
- Pictures (`zoom_strait.png` of the two continent runs at scale 1): in the channel B is a defect-free
  strip of triangles aligned with the coast (0 valence-5/7 nodes, |ln s/h| p99 0.038 in the view), grown by the front
  from the straight coast edges; A's staircase leaves 21 / 20 valence-5 / 7 nodes and p99 0.177 in the same view.
  B's worst edge (`zoom_size`) is a 33 km coast edge on the 50 km open coast, where JIGSAW's 1-D front closes.
- JIGSAW returns triangles with a run-dependent orientation (all clockwise on band and on continent's closed sphere,
  all counter-clockwise elsewhere); meshom flips them to CCW.

## What JIGSAW cannot do for us

- **Quads.** Triangles only (`TRIA3`); no mixed output, no cross-field alignment. As a FESOM3 generator it would need
  meshom's recombination on top (m01's path: ~10 % triangles, quad θmax p99.9 ≈ 135°, |ln s/h| 0.24).
- **Our size measure.** It optimises its own cost (ODT + radius-edge / valence moves), not |ln s/h| per edge; it
  reaches 0.12–0.15 on our measure because its placement is good, and the runner's smoothing only moves it away.
  At scale 4 (slope 0.4) it fails the size gate like every method.
- **h input.** Only a lon-lat grid or an unstructured mesh with values; an analytic or KD-tree h must be sampled
  (here ≤ 5.6 s and 1.45 M values; a real coast-distance field at 1 km would need a much larger grid).
- **Coast nodes and flags.** It resamples the coast itself (short edges between features) unless every vertex is
  forced; no FESOM node order or boundary flags (meshom adds them); land removal and FESOM2's clean-up are ours.
- **Speed and parallelism.** One core in this build; the optimiser is 97–98 % of the time (~900 nodes/s at
  4.9·10⁵ nodes). JIGSAW 1.x has an OpenMP optimiser (not tested here).
- **Licence.** Custom, non-OSI (free for research and private use; distribution as part of a commercial system only by
  arrangement with the author). It can be run as a reference; its code cannot go into meshom.

Not done: JIGSAW 1.1.0 (bundled with jigsaw-python, OpenMP optimiser) — not built; the unstructured hfun
input; a separable refinement test for the h grid (would keep band at 721 columns).
