---
name: meshom-outputs
description: What a meshom run delivers and how to read it - the model formats (FESOM-C, FESOM2, FESOM3) and their checks, fesom_ini, quality.json and its gates, stages.json, the figures, meshom write for a saved mesh, and meshom quality for any existing mesh. Read it when you judge a run or hand model files over.
---

# Outputs: model files, records, figures

## Model formats (`[mesh] format`)

All writers are in `meshom/io/fesom.py`. Coordinates are geographic lon-lat in degrees, not rotated; node ids are
line numbers 1..N; node flag 1 on boundary nodes, 0 elsewhere (never 2: no open boundaries yet); elements clockwise,
triangles first, then quads (O3-O9).

| format | files | elements | depth | checks in the `write` stage |
|---|---|---|---|---|
| `"fesom-c"` (default) | `nod2d.out`, `elem2d.out`, `depth.out`, placeholder `aux3d.out` | 4 columns; a triangle is `n1 n2 n3 n1` | `depth.out`, positive down | none |
| `"fesom2"` | `nod2d.out`, `elem2d.out`, `aux3d.out` | 3 columns, **triangles only** (quads are dropped) | `aux3d.out`: nl, the levels, the node depths negative down | `fesom2_checks` (O21) |
| `"fesom3"` | `nod2d.out`, `elem2d.out`, `aux3d.out` | 4 columns as FESOM-C | `aux3d.out` as FESOM2 | `fesom3_checks` (O25); `fesom_ini` with `[mesh] fesom3_ini` (O26) |

- `stages.json` → `write` → `fesom2_checks` / `fesom3_checks` → `stop` lists what would stop the model or its
  preprocessor: fewer than 2 neighbours, more than 32 elements or neighbour nodes at a node (quad diagonals counted
  for FESOM3), a node near the rotated North Pole (FESOM2), an element holding a rotated pole, a bow-tie or
  non-convex quad (FESOM3, in rotated lon-lat and on the sphere), an element not clockwise, too few levels, an
  `aux3d.out` node count that differs, more than one water component. It must be empty. Other counts are reported
  only (unused nodes, depths above zbar(5), more than 11 elements at a node for coupled runs).
- **FESOM3 needs `fesom_ini`** (FESOM3's preprocessor: edges, node levels, partitions) once before any model run.
  `[mesh] fesom3_ini = "/path/to/fesom_ini"` runs it with 1 partition in the `write` stage (folder `fesom_ini/`);
  the record `fesom3_ini` has `ok`, `exit_status` and the stop lines. `fesom_ini` exits 0 even on errors: trust `ok`
  and the log. For an N-rank model run, run `fesom_ini` yourself with N partitions on a copy of the three files.
  The writer removes files `fesom_ini` derived and old partition folders, so rerun it after every rewrite (O34).
- FESOM2: partition with FESOM2's `fesom_meshpart`; namelist hints in [`triangles.md`](triangles.md).
- FESOM-C: `depth.out` is what the model uses; open boundaries must be marked by hand ([`regional.md`](regional.md)).

## `quality.json`

`{"raw": record}`, plus `"smooth"` / `"optimized"` when those stages ran (the last one is the final mesh's).

| field | meaning |
|---|---|
| `status` | `ok`, `error` or `timeout` (then `error` holds the text) |
| `n_nodes`, `wall_s`, `generate_s`, `peak_rss_mb`, `commit`, `host`, `opts` | run facts (`opts` includes the size tolerance the method got) |
| `quality` | `n_tris`, `n_quads`, `tri_share`, angle statistics (`quad_max_angle_deg`, `tri_min_angle_deg`, ...), `quad_sj`, `elem_size_ratio`, `size_fidelity` (\|ln s/h\|: p99, median, bias), `tri_clusters`, `boundary` |
| `irregular` | interior nodes of valence 3 (`n3`), 5 (`n5`) and other, triangles `T`, `euler_ok` |
| `rho_c` | the irregular nodes an ideal quad mesh following h needs |
| `gates_passed`, `gates_total` | applicable gates only (a gate without a value is not counted; G20) |
| `gates_failed` | `{gate name with its subset suffix: value}` |
| `gates_failed_families` | the failed gates without the suffix: **drive this to empty** |
| `straits` | per strait: `name`, `profile`, `cells_target`, `cells_min`, `width_min_m`, `closed_sections`, `ends_component`, `connected`, `depth_min_m`, `ok` |
| `water` | `connectivity_ok`, `n_components`, `components` (elements, nodes, km², box, straits ending there), `not_connected` |
| `passages` | `n_narrow` and each narrow passage left (C51) |

**Gate subsets** (G17, G18): each gate is evaluated on the whole mesh and on `_coast` (elements touching the coast),
`_interior`, `_coast3` (the first three coast rows) and `_strait` (test cases only). `size_fidelity_p99_coast3`
failing means the size target fails in the first three coast rows. A mixed real-coast mesh with a land cut has 39
applicable gates, a triangle-only one 29. The gates and their defaults: `meshom.config.DEFAULT_GATES`;
`[gates]` overrides them.

**A mesh is acceptable when** `status` is `ok`, `gates_failed_families` is empty, `water.connectivity_ok` is true,
every strait has `ok` true (or you can say why one does not) and the format checks' `stop` is empty.

## `stages.json`

One record per stage with `wall_s`, `peak_rss_mb`, host and commit; use them to size the next job. Worth reading:

- `size`: `n_nodes_est_sphere`, `limiter` → `capped` (true: the limiter grid was too coarse), `strait_profiles`.
- `post`: the land cut's removals by rule, `n_components`, `passages`.
- `coast_filter` (`loops`): loops and basins before and after (a dropped basin you wanted means a missing strait).
- `bathymetry`: nodes raised to the floor, nodes without an ocean cell, `policy`, `regions`.
- `write`: the format checks, `fesom3_ini`, `water`.
- `straits`: `n_ok`, `n_connected`, `water`.

## Figures (`[run] figures = true`)

`figures/raw/` (`meshom/figures.py`); open them, they are PNGs. `index.json` gives each zoom's centre, radius and
local numbers. Zooms are drawn in a local tangent plane, never in raw lon-lat.

| file | shows |
|---|---|
| `overview.png` | 3 x 4 globe views: ln(s/h) per element, quad max angle (triangles grey), triangles and irregular nodes |
| `zoom_angle.png` | the quad with the largest corner angle |
| `zoom_size.png` | the edge with the largest \|ln s/h\| |
| `zoom_irregular.png` | the densest cluster of irregular nodes |
| `zoom_seam.png` | where fronts met: size errors and irregular nodes together |
| `zoom_coast.png` | the worst element touching the coast (conforming coasts) |
| `zoom_strait.png` | the narrowest water gap (conforming coasts) |

## `meshom write`: model files of a saved mesh

```
meshom write RUN_DIR [--mesh raw|smooth|optimized|edited-NN|repaired|NAME|PATH.npz] [--format F] [--out DIR] [--fesom-ini]
```

Default mesh: the newest `edited-NN.npz`, else the run's final mesh. Output `RUN_DIR/fesom-<mesh>/` (never the
run's own `fesom/`): the files, the depth recomputed with the run's `[bathymetry]` (regions included, then the
node edits of `<mesh>.depth_edits.json`), the format's checks, the run's scoring, the strait report and the passage
check (reported, never widened), all in `fesom-<mesh>/quality.json`. `--fesom-ini` runs the config's `fesom3_ini`.
The first write builds `RUN_DIR/cache/`; later writes are fast (O32, O33).

## Checking an existing mesh

`meshom quality FESOM_DIR [--gates FILE] [--out DIR]` reads any FESOM mesh folder (`nod2d.out`, `elem2d.out`),
computes the quality summary and the gates (no h, so no size gate) and writes `report.md`. `meshom figures` renders
the picture set of a saved mesh of a test case.

## Config keys used

- `[mesh]`: `format`, `fesom3_ini`
- `[run]`: `figures`
- `[bathymetry]`: `levels`, `min_layers`
- `[gates]`: `size_fidelity_p99`, `tri_cluster_max`, `boundary_rule`, `euler_ok`

## Sources

- Code: `meshom/io/fesom.py` (writers, `fesom2_checks`, `fesom3_checks`, `run_fesom_ini`, `read_fesom`),
  `meshom/compete.py` (the scored record), `meshom/quality/metrics.py` (`check_gates`), `meshom/figures.py`,
  `meshom/run.py` (`write_mesh`, `water_record`).
- Docs: [`docs/config.md` "Running"](../config.md#running), [`docs/config.md` "`[gates]`"](../config.md#gates).
- Decisions: G17, G18, G20, O3-O9, O21, O25, O26, O32-O34, C51, C52.
