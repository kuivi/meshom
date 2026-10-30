---
name: meshom-troubleshooting
description: The common failures, warnings and missed gates of meshom runs, what each message means and which config change fixes it, plus the known limits of the tool today. Read it when a run fails, times out, prints a warning or misses a gate.
---

# Troubleshooting and known limits

Change one thing at a time and build into a new run folder, so the old result stays for comparison. Judge
smoothness at scale 1 only.

## Messages

| message | meaning | what to change |
|---|---|---|
| `<path>: unknown key (known: ...)` | a key that is not in the schema (a typo, or a key of another table) | use one of the names listed; [`docs/config.md`](../config.md) |
| `size.layers[i].h_out_m: needed without a constant layer or [size] h_max_m` | a zone has no h to grow to | add a `constant` layer or `[size] h_max_m`, or give `h_out_m` |
| `status: timeout` in `quality.json` | the method ran longer than `[run] timeout` (default 1800 s) | raise `[run] timeout` and the job's `--time` to 2-3 times the expected time ([`workflow.md`](workflow.md)) |
| `WATER NOT CONNECTED: N water components; #i ... km2, box [...]` | a basin is cut off at the mesh's h (a strait missing from the list, or closed by a region `off`) | find the basin by its box; add a `[[straits]]` entry from basin to basin, or open the list with `widen`; `water_components` is then also in the format checks' `stop` |
| `NARROW PASSAGES: N through-passages below K cells across: ...` | a passage the widening rounds did not open | add a `[[straits]]` entry there (`profile = "widen"`, `cells = 3`) |
| `fesom2_checks` / `fesom3_checks` → `stop` not empty | a rule the model or `fesom_ini` stops on | read which count is non-zero ([`outputs.md`](outputs.md)); FESOM3 non-convex quads in rotated lon-lat sit near the rotated poles (40° W 75° N, 140° E 75° S) |
| `fesom3_ini` → `ok` false (exit status 0) | `fesom_ini` failed; it exits 0 even on errors | read `fesom_ini/fesom_ini.out` |
| `m16_orca: NEMO's mask needs ref_dir ...`, `... levels_dir ...`, `... set gebco` | an ORCA input is missing | [`orca.md`](orca.md) "Data, once" |
| `m16_orca: region and lat_north exclude each other`, `lat_north ... reaches the fold` | inconsistent ORCA options | pick one way to bound the ORCA part; keep it below the fold |
| `[insert NAME] REFUSED ...; fix: ...` | an insert rule failed | apply the fix it names ([`inserts.md`](inserts.md)) |
| `...: the run's own fesom/ is never overwritten` | `meshom write --out` pointed at `RUN_DIR/fesom` | write to another folder (default `fesom-<mesh>/`) |
| `n_nodes ... is not the mesh's ...: the edits of another mesh` | a `depth_edits.json` of another mesh | fix its `mesh` and `n_nodes`, or remove it |

## Symptoms

| symptom (where you see it) | likely cause | knob |
|---|---|---|
| far too many or too few nodes (`stages.json` → `size` → `n_nodes_est_sphere`) | layers too fine or too coarse; a zone's `h_out_m` below the open-ocean h refines the whole globe | `[mesh] scale` for a uniform change, else the layer `h_m`; leave `h_out_m` unset ([`resolution.md`](resolution.md)) |
| a refined zone has no effect | `[size] h_min_m` clips it | lower `h_min_m` |
| `elem_size_ratio_p99` fails with an m14 method | no or too small size tolerance | keep `size_tolerance` at 0.20 (check `opts` in `quality.json`); never 0 for m14 |
| `size_fidelity_p99` fails, m14 | edges beyond t next to fixed coast or block nodes | t = 0.20 is the measured best; on the `_coast*` subsets of a conforming coast use `treatment = "cut"` |
| `size_fidelity_p99` fails on an ORCA part | ORCA's cells are anisotropic by design | report it; do not fix it |
| `tri_cluster_max` fails | clusters of ≥ 3 triangles in the fill | use `m14_blocks_rep` (not `_mix` or plain `m14_blocks`); then `meshom repair RUN_DIR` or `meshom optimize RUN_DIR` ([`cleanup.md`](cleanup.md)); t = 0.30 adds clusters |
| `quad_max_angle_target_deg_coast*` fails, conforming coast | bay corners at the coast | `coast_fix` (on by default in `m14_blocks_rep`), or `treatment = "cut"` |
| a strait with `connected: false` or `closed_sections > 0` | the path runs over land, or the corridor is too narrow | check the path against the coast (well inside both basins); raise `cells` or `width_m`; `profile = "widen"` |
| a strait with `cells_min` below `cells_target` | a narrow corridor | `[coast] strait_widen_cells = 3` or the strait's `cells = 3` |
| the Mediterranean missing | Gibraltar closed by an explicit `enabled = false` | remove that entry (C49) |
| the Black Sea missing | its four straits not opened (a region `off`, disabled, or no list) | `straits = "fesom2_mesh_gen"`, or accept it (CORE2 has none) |
| `stages.json` → `size` → `limiter` → `capped` true | the smallest h is below about 14 km at the 12 M-node cap | raise `[size] limiter_max_nodes` (about 1.1 GB per 12 M nodes) |
| quads in a FESOM2 run are missing, `fesom2_checks` shows unused nodes | a quad method written as `fesom2` | use `m15_tri` for FESOM2 |
| a bad spot visible in a figure | local defects | `meshom repair RUN_DIR --polygon '[[lon, lat], ...]'` |
| FESOM2 blows up in a strait (Marmara, Gibraltar) | small deep basins, vertical CFL | `strait_h_min_m = 20000`, `[bathymetry] smooth_passes = 4`, `min_layers = 4` (`fesom2_core2like_straits45.toml`); or a shorter step with w-splitting |
| the run is slow above 1 M nodes | `m14_blocks_rep`'s repair dominates | expected; do not switch on `gap_adapt` (3-7 times slower, no gain) |
| a different node count on another machine | results reproduce per platform, not across (L9) | compare on one platform |

## Known limits

- **Regional meshes with open boundaries are not built**: no regional domain, no node flag 2 ([`regional.md`](regional.md)).
- **No ice-shelf cavities** (FESOM3 has none either); no lakes in a global mesh (by rule, C48).
- **A cut-off basin is reported, not repaired automatically**: add the strait yourself.
- **Conforming coasts** (`treatment = "loops"`) are weaker than the land cut on real coasts (36-39 of 39 gates for
  m14, 24 of 29 for `m15_tri`), and the coast filter costs time and memory (about 5 GB for a CORE2-like coast
  with 10 km straits).
- **The land cut's point-in-polygon test** grows with elements times coast vertices: about 2 minutes at 3.5 M
  elements, more beyond.
- **The limiter works across land**: a strait refinement can leak through thin land into the next sea.
- **Strait corridors are centred on the list's paths**: an off-centre path widens unevenly, a path across a small
  island carves through it.
- **Leftover triangle clusters** can remain on the largest `m14_blocks_rep` meshes (the 3.87 M-node example: 38/39,
  `tri_cluster_max` 3) and on conforming global meshes.
- **Seams between blocks hold the defects**; blocks cover about 30 % of a real ocean.
- **Gate thresholds are first settings**, not calibrated against model behaviour (G0).
- **Largest meshes built**: 12.2 M nodes (`m15_tri`, a test case), 3.87 M (`m14_blocks_rep`, real coast),
  1.65 M (ORCA025 with a 4.5 km Arctic).

## Config keys used

- `[mesh]`: `scale`, `size_tolerance`, `format`
- `[mesh.opts]` `m14_blocks_rep`: `coast_fix`, `gap_adapt`
- `[size]`: `h_min_m`, `limiter_max_nodes`
- `[[size.layers]]` `zone`: `h_out_m`
- `[coast]`: `treatment`, `straits`, `strait_widen_cells`, `strait_h_min_m`, `passage_min_cells`
- `[[straits]]`: `name`, `path`, `cells`, `width_m`, `profile`, `enabled`
- `[bathymetry]`: `smooth_passes`, `min_layers`
- `[run]`: `timeout`
- `[gates]`: `elem_size_ratio_p99`, `size_fidelity_p99`, `tri_cluster_max`, `quad_max_angle_target_deg`

## Sources

- Code: `meshom/cli.py` (the printed lines), `meshom/run.py` (`water_text`, `write_mesh`), `meshom/config.py`
  (`ConfigError`), `meshom/insert/core.py`, `meshom/insert/region.py` (`refusal`, `advise`),
  `meshom/methods/m16_orca.py`.
- Docs: [`docs/config.md`](../config.md), `docs/decisions.md`.
- Decisions: C24, C44, C48, C49, C51, C52, G0, G38, L9, O25, O26, S10.
