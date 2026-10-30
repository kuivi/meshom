---
name: meshom-workflow
description: The loop from a mesh request to model files with meshom - config, quick test, build on a compute node, read the run folder, iterate, write. Read it before your first run and whenever you size a batch job.
---

# Workflow: from a request to model files

## The loop

1. **Understand the request**: the model (FESOM3, FESOM2, FESOM-C), global or regional, the resolution in km where
   it matters, the straits that must stay open, the region of focus. [`choosing.md`](choosing.md) turns that into
   a method and an example config.
2. **Write the config**: copy the closest `docs/examples/*.toml` to a new file. Change:
   - `[mesh] out`: a new run folder (default `.tmp/runs/<config file stem>`, relative to the working directory);
   - `[coast] source` and `[bathymetry] source`: your copies of the coast shapefile and the elevation grid (the
     examples carry the paths of the machine they were measured on);
   - only then what the request asks (the size layers, the straits, the method, the format).
3. **Quick test**: the same config with `[mesh] scale = 4` (16 times fewer nodes, minutes) and another `out`. It
   checks the coast, the straits, the water connectivity and the depth stage. Do not judge smoothness there: scale
   multiplies |∇h| too (S15).
4. **Estimate the size**: `stages.json` of any run, stage `size`, field `n_nodes_est_sphere` = ∫h⁻² dA over the
   whole sphere, land included (S25). With a land cut the final node count is roughly the ocean share of it
   (about 0.6 for a global mesh).
5. **Build**: `meshom run CONFIG` on a compute node (below).
6. **Read the result**: the summary line, then `quality.json`, `stages.json`, the figures
   ([`outputs.md`](outputs.md)).
7. **Iterate**: change one thing, new `out`, build again, compare the two `quality.json`. The symptom table of
   [`troubleshooting.md`](troubleshooting.md) maps failures to the knob.
8. **Deliver**: the model files of the run are in `RUN_DIR/fesom/`. If you improved the mesh afterwards
   ([`cleanup.md`](cleanup.md), [`inserts.md`](inserts.md)), write that mesh with `meshom write`.

## Commands

```
meshom run CONFIG [--out DIR]           # build the mesh the config describes
meshom runs [DIR]                       # one line per run folder under DIR (default .tmp/runs): status, size, gates
meshom write RUN_DIR [--mesh NAME] [--format fesom-c|fesom2|fesom3] [--out DIR] [--fesom-ini]
                                        # a saved mesh of the run as model files (default: the run's final mesh)
meshom repair RUN_DIR [--polygon '[[lon, lat], ...]'] [--mesh raw.npz] [--out repaired]
meshom optimize RUN_DIR [--mesh raw.npz] [--out optimized]
meshom quality FESOM_DIR [--gates FILE]  # quality and gates of any FESOM mesh folder (no h: no size gate)
python -m meshom.insert --run RUN_DIR --config FILE.toml        # inserts, see inserts.md
```

`meshom ui` starts a web interface for people (config editing, mesh viewing, node edits); you do not need it.

## What a run does

`meshom run` (`meshom/run.py`) runs these stages in order; each appends one record to `stages.json` with its wall
time `wall_s` and peak memory `peak_rss_mb`:

| stage | what it does |
|---|---|
| `coast` | reads `[coast] source` (or meshes the closed sphere), adds Antarctica, joins polygons cut at ±180° |
| `size` | builds h from the layers, the strait zones and the gradient limiter; the node estimate |
| `coast_filter` | only with `treatment = "loops"`: smooths the coast to h, drops cut-off basins |
| `blocks` | only with `[[blocks]]`: lays the hand-placed blocks on the sphere |
| `generate` | the method fills the domain in a child process, killed after `[run] timeout` |
| `post` | only with `treatment = "cut"`: the land cut, FESOM's clean-up, narrow passages widened |
| `optimize`, `smooth` | the optimisation stage when the run applies it; the common smoothing when `[run] smooth` |
| `score` | quality summary, size fidelity, irregular nodes, gates |
| `figures` | only with `[run] figures = true`: the standard picture set |
| `bathymetry` | only with a `[bathymetry] source`: node depths and the depth rules |
| `write` | the model files in `fesom/`, the format checks, the water check |
| `straits`, `passages` | the strait report and the narrow-passage check of the final mesh |

## The run folder

| file | content |
|---|---|
| `config.resolved.toml` | the config with every default filled in and absolute paths: the reproducible input |
| `stages.json` | one record per stage (time, memory, host, commit and the stage's own numbers) |
| `quality.json` | the scored record(s): `raw`, and `smooth` / `optimized` when those stages ran |
| `raw.npz`, `optimized.npz`, `smooth.npz` | the meshes (unit vectors `points`, `tris`, `quads`, `node_flag`) |
| `depth.npy` | node depths, metres positive down (with a bathymetry source) |
| `fesom/` | the model files of the final mesh in `[mesh] format` |
| `figures/raw/` | the pictures and `index.json` (with `[run] figures = true`) |
| `m14.json`, `repair.json`, `tol_fit_final.json`, `optimize.json` | the m14 methods' block layout and stage records |
| `orca.json`, `orca_belt.npz` | `m16_orca`: the grid, its rows, the mask and the joints |
| `cache/` | `meshom write`'s caches; delete any time (and when the coast file changed under the same config) |

The final mesh is `optimized.npz` when the run applied the optimisation stage, else `smooth.npz` when smoothing ran,
else `raw.npz`. A failed run still writes `quality.json` with `status` `error` or `timeout` and the error text.

## Running on an HPC system with Slurm

Run every build on a compute node. `scripts/run_example.sbatch` runs `meshom run` on each config given, from the
submit folder, then lists the runs; its log is `meshom-run-<job id>.log` in the submit folder:

```
sbatch -A <project> scripts/run_example.sbatch CONFIG [CONFIG ...]
sbatch -A <project> --partition=<partition> --cpus-per-task=16 --mem=30G --time=04:00:00 \
       scripts/run_example.sbatch CONFIG
```

It uses the `python` of the environment active at submission (`PY=/path/to/python` overrides it) and sets
`NUMBA_NUM_THREADS` to the job's CPUs and the other thread counts to 1. Its defaults (8 CPUs, 15 GB, 1 h) suit
small meshes only; override them on the command line. `scripts/on_node.sh -c CPUS -m MEM -t HH:MM:SS -- COMMAND`
runs one command on a compute node and streams its output (blocking `srun`); it names a fixed Slurm account in its
last line: change it to your project before you use it.

Set `[run] timeout` (seconds; default 1800, the method's child process is killed after it) and the job's `--time`
to two or three times the expected time.

## Time and memory

Measured on 2-socket AMD EPYC 7763 nodes (whole `meshom run` with GEBCO depth unless noted; the GEBCO read alone is
about 1.5-2 min whatever the mesh size):

| mesh | method | nodes | CPUs | wall | peak memory |
|---|---|---|---|---|---|
| global 150 km, cut (`global_150km_m14.toml`) | m14_blocks | 60 k | 8 | under a minute | small |
| CORE2-like, FESOM2 (`fesom2_core2like.toml`) | m15_tri | 163 k | 16 | 4 min | 2.1 GB |
| CORE2-like mixed, 90 widened straits, FESOM3 | m14_blocks_rep | 144 k | 16 | 16 min | about 2 GB |
| the same at `scale = 0.5` | m14_blocks_rep | 572 k | 16 | 32 min | 3.0 GB |
| FESOM2 at `scale = 0.4` (`fesom2_finer.toml`) | m15_tri | 1.02 M | 32 | 14 min | 3.1 GB |
| fine global mixed (`fesom3_meshB.toml`) | m14_blocks_rep | 3.87 M | 32 | 6 h 17 min | 13.8 GB |
| ORCA025 whole (`orca025_whole.toml`) | m16_orca | 0.9 M cells | 32 | 3 min | 29 GB |
| ORCA025 + 4.5 km Arctic (`orca025_arctic45.toml`) | m16_orca | 1.52 M | 128 | 69 min | 74 GB |

Rough sizing: 16 CPUs and 30 GB up to about 1 M nodes; `m14_blocks_rep` above 1 M nodes takes hours (its repair
dominates). Each example config states the resources it was run with in its header comment.

## Config keys used

- `[mesh]`: `out`, `scale`, `format`
- `[coast]`: `source`
- `[bathymetry]`: `source`
- `[run]`: `timeout`, `smooth`, `figures`

## Sources

- Code: `meshom/run.py` (the pipeline, `write_mesh`, `mesh_file`), `meshom/cli.py` (the commands),
  `scripts/run_example.sbatch`, `scripts/on_node.sh`.
- Docs: [`docs/config.md` "Running"](../config.md#running), [`docs/config.md` "`[run]`"](../config.md#run).
- Decisions: S15, S25 (scale, node estimate), O15 (provenance), O32, O33 (`meshom write` and its caches).
