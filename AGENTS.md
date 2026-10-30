# meshom for AI agents

Read this file first when you are asked to build, change or check an ocean mesh with meshom. It is short on purpose:
the details are in the topic files of `docs/agents/`, listed in the index at the end. Open only the ones the task
needs.

## What meshom is

- A mesh generator on the sphere for the ocean models FESOM3 (global, mixed quads and triangles), FESOM2 (global,
  triangles) and FESOM-C (regional; open boundaries are not built yet).
- One TOML config describes one mesh: the coast, the size function h (target edge length in metres), the fill
  method, the depth and the output format. `meshom run CONFIG` builds it into a run folder.
- Mixed meshes are mostly near-square quads, with triangles only in size transitions and along complex coasts.
- Every run records its quality (`quality.json`, `stages.json`) and checks the files against what the model needs.
- The config reference is [`docs/config.md`](docs/config.md); every threshold and default has a row with its reason
  in [`docs/decisions.md`](docs/decisions.md) (cited here by its id, e.g. G38, C51).

## Install and check

```
mamba env create -f environment.yml        # or conda; the environment is named meshom
mamba activate meshom                      # it installs this repository in editable mode
python -c "import meshom; print(meshom.__file__)"      # must point into this checkout
meshom --help                              # the commands: run, runs, write, repair, optimize, quality, ...
python -m pytest -q tests/test_config.py   # a quick test of the install
```

No data is shipped. A real mesh needs a land-polygon shapefile (GSHHG's `GSHHS_i_L1.shp` with its L5 file for
Antarctica, or Natural Earth) for `[coast] source`, and a GEBCO-like elevation grid (netCDF) for `[bathymetry]
source`. ORCA meshes also need NEMO's grid files ([`docs/agents/orca.md`](docs/agents/orca.md)).

## The workflow

1. Pick the closest example in `docs/examples/` ([`docs/agents/choosing.md`](docs/agents/choosing.md)) and copy it
   to a new config file. Set `[mesh] out` to a new run folder and point the data paths at your files.
2. Check the config cheaply: run it once with `[mesh] scale = 4` (16 times fewer nodes) and a different `out`.
3. Build it: `meshom run CONFIG` on a compute node (`sbatch -A <project> scripts/run_example.sbatch CONFIG`).
4. Read the run folder: the summary line, `quality.json` (gates, water, straits), `stages.json` (time, memory per
   stage) and the figures in `figures/raw/` ([`docs/agents/outputs.md`](docs/agents/outputs.md)).
5. Change one thing, build again into a new folder, compare. Stop when the acceptance rules below hold.
6. The model files are in `RUN_DIR/fesom/`. A repaired, optimised or hand-edited mesh of the run becomes model
   files with `meshom write RUN_DIR --mesh NAME`.

A mesh is acceptable when `status` is `ok`, `gates_failed_families` is empty (or each failure is explained), the
water is one connected component, every strait record has `ok` true and the format checks' `stop` list is empty.

## Rules for agents

- Run builds on compute nodes through the batch system (Slurm), never on a shared login node. Size the job from
  the timings in [`docs/agents/workflow.md`](docs/agents/workflow.md) and set `[run] timeout` to match.
- Sizes are metres of great-circle length on the sphere; areas are km²; angles and coordinates are degrees.
  Never judge sizes or angles in raw longitude-latitude.
- Start from an example config; change only what the task asks; keep every run in its own folder.
- meshom does not validate inputs beyond unknown keys: check units and orders yourself (a `box` is
  `[west, east, south, north]` in `[[strait_regions]]` but `[lon_min, lat_min, lon_max, lat_max]` in
  `[[bathymetry.regions]]`).
- Never hand-edit a mesh, a quality record or a gate value to make a check pass. Fix the cause in the config, or
  report the failure with its numbers.
- Read the failure message: refusals and warnings (`WATER NOT CONNECTED`, `NARROW PASSAGES`, an insert's
  `REFUSED ... fix: ...`) name the cause and the change that fixes it
  ([`docs/agents/troubleshooting.md`](docs/agents/troubleshooting.md)).
- A global mesh keeps only the water connected to the world ocean; Gibraltar is always open. Do not switch these off
  for a global ocean mesh (C48, C49).
- Report what you built: the run folder, nodes, quads and triangles, gates passed / total and the failed families,
  the water and strait checks, the wall time and the peak memory.

## Index

| file | read it when |
|---|---|
| [`docs/agents/workflow.md`](docs/agents/workflow.md) | you run meshom: the loop from a request to model files, the run folder, time, memory, Slurm |
| [`docs/agents/choosing.md`](docs/agents/choosing.md) | you must decide which mesh type, method, resolution class and example config fit a request |
| [`docs/agents/triangles.md`](docs/agents/triangles.md) | you build a triangle mesh (FESOM2, or triangles for FESOM3) with `m15_tri` |
| [`docs/agents/mixed.md`](docs/agents/mixed.md) | you build a mixed quad-dominant FESOM3 mesh (`m14_blocks_rep`) or place blocks by hand |
| [`docs/agents/orca.md`](docs/agents/orca.md) | the request names ORCA or NEMO: whole ORCA grids, ORCA joined to a refined region, nested ORCA |
| [`docs/agents/regional.md`](docs/agents/regional.md) | the request is a regional mesh with open boundaries (FESOM-C): what works and what does not |
| [`docs/agents/resolution.md`](docs/agents/resolution.md) | you set or change the size function h: layers, zones, gradient limit, scale, size tolerance |
| [`docs/agents/coast-straits.md`](docs/agents/coast-straits.md) | you choose the coast data and treatment, small islands, straits and narrow passages, water connectivity |
| [`docs/agents/bathymetry.md`](docs/agents/bathymetry.md) | you set node depths: the grid, the depth rules, strait sills, depth regions, node edits |
| [`docs/agents/inserts.md`](docs/agents/inserts.md) | you add a finer structured patch to a finished mesh: a strait, a bay, a coastal strip, a current |
| [`docs/agents/cleanup.md`](docs/agents/cleanup.md) | you want to understand or improve mesh quality after the fill: FESOM rules, repair, optimisation, post-processing |
| [`docs/agents/outputs.md`](docs/agents/outputs.md) | you read a run's results or deliver model files: formats, `quality.json`, `stages.json`, figures, checks |
| [`docs/agents/troubleshooting.md`](docs/agents/troubleshooting.md) | a run fails, times out, warns or misses a gate; you need the known limits |
