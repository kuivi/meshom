---
name: meshom-cleanup
description: The clean-up and quality stages of a meshom mesh - the land cut with FESOM's rules, narrow passages, repair, the size fit, the optimisation stage, smoothing - and the post-processing commands for a finished mesh (meshom repair, meshom optimize, the mesh-fixing pipeline), with how to read their records. Read it when a mesh misses quality gates or you want to improve a finished mesh.
---

# Clean-up, repair and post-processing

## What already runs inside `meshom run`

| step | when | what it does | record |
|---|---|---|---|
| land cut | `[coast] treatment = "cut"` | removes every element whose centroid is on land, keeps strait corridors and coast-sided hand blocks as water | `stages.json` → `post` |
| FESOM clean-up | with the land cut | after D. Sein (AWI), fesom2_mesh_gen: removes spurs (elements with one neighbour), pinched nodes, every water piece but the largest; one unprotected pass at the end | `post` → `n_removed_land`, `n_removed_clean`, `removed_clean_by`, `n_components` |
| narrow passages | `[coast] passage_min_cells` ≥ 2 | widens every through-passage below K cells and cuts again, up to 3 rounds (C51) | `post` → `passages` |
| repair | `m14_blocks_rep` | defect clusters cut out and remeshed locally; a candidate is kept only when it scores better | `repair.json` |
| size fit | m14 methods (t = 0.20) | sizes as smooth as possible within \|ln s/h\| ≤ t (G38, M38) | `tol_fit_final.json` |
| coast fix | `m14_blocks_rep`, conforming coasts | cleans the first coast rows (B64) | `tol_fit_final.json` → `coast_fix` |
| optimisation stage | default for the m14 methods; `[mesh] optimize = true` for others | edge flips, triangle pairs merged into quads ≤ 135°, triangle clusters ≥ 3 broken, cancelling irregular pairs removed, a local size fit; boundary and coast nodes fixed (Z1-Z13) | `optimize.json` (inside the method), or `optimized.npz` and the `optimized` record of `quality.json` (applied by the run, e.g. after a land cut) |
| common smoothing | `[run] smooth = true` | angle smoothing with the boundary fixed, scored as `smooth` | `smooth.npz` |

Keep `[run] smooth = false` for `m14_blocks_rep` and `m15_tri`: the common smoothing makes their sizes worse. Keep
the optimisation stage at its default.

## The FESOM rules the mesh must meet

- Every element has at least 2 edge neighbours; at most 32 elements and 32 neighbour nodes at a node (G10, G11).
- No triangle with two coast edges (FESOM2: it would have one neighbour; G14, `boundary_rule = "tri"`).
- Every boundary node has exactly 2 boundary edges: no pinched coast (G15).
- Quads strictly convex, no bow-ties (G6, O12); FESOM3 also needs every quad convex in rotated lon-lat and every
  element clockwise (O25).
- One water component (C52).

The land cut's clean-up enforces the first three on a cut mesh; the gates and the format checks report the rest.

## Post-processing a finished mesh

Each command reads the run folder's `config.resolved.toml` to rebuild the domain and h, writes a new mesh next to
the others and a record with the scores before and after. None changes `fesom/`; turn the new mesh into model files
with `meshom write RUN_DIR --mesh NAME` ([`outputs.md`](outputs.md)).

### `meshom repair`: local remeshing of defect clusters

```
meshom repair RUN_DIR [--mesh raw.npz] [--out repaired] [--opts JSON] [--figures]
meshom repair RUN_DIR --polygon '[[lon, lat], ...]'     # only one zone, e.g. a bad spot seen in a figure
```

Writes `repaired.npz` and `repaired.json` (`before`, `after`, the repair's counts). Repairs chain (`--mesh
repaired.npz`). Options: `meshom.repair.DEFAULTS` (e.g. `candidates`, the fills tried; `--opts '{"force": true}'`
with `--polygon` takes the best fill even without a gain). A dislocation (a lone irregular pair) often cannot be
removed locally; see `docs/methods/repair.md`.

### `meshom optimize`: the optimisation stage on a saved mesh

```
meshom optimize RUN_DIR [--mesh raw.npz] [--out optimized] [--tol T] [--opts JSON]
```

The same stage the m14 methods run last (Z1), for any method's mesh: writes `<NAME>.npz`, `<NAME>.json` and the
record `<NAME>` in `quality.json`. Options: `meshom.optimize.DEFAULTS` (`steps`, `passes`, `quad_max`, ...).

### The mesh-fixing pipeline: whole-mesh improvement

```
python -m meshom.meshfix.pipeline --run RUN_DIR [--stages d,b,e,a] [--opt d.workers=64 ...]
```

Research-grade, not part of any default run (B141-B157). It improves a finished meshom mesh everywhere, moving coast
nodes along the real coast, and writes `RUN_DIR/fixed_<stages>.npz`, `.json` and `.score.json` (scores against the
input). Stages: `d` cavity remeshing, `b` local defect transport, `e` defects carried to the coast and absorbed there,
`a` free-node geometry with a coast slide, `c` global placement, `t` valence flips for pure-triangle meshes. The
default order `d,b,e,a` was best on every test case (B149); `d,e,a` is the fast order (stage `b` runs on one thread
and dominates the time). A triangle mesh (`m15_tri`) takes `--stages t,a`. Then
`meshom write RUN_DIR --mesh fixed_d_b_e_a`. Check the coast gates afterwards: the coast size fidelity may rise
toward its gate.

## Reading the records

- `quality.json` → `raw` (or `optimized`): `gates_failed_families` is the list to drive to empty; `irregular` gives
  the interior nodes of valence 3, 5 and other and the triangle count `T`; `rho_c` the irregular nodes an ideal quad
  mesh following h needs (curvature and size changes force some).
- A repair or optimisation record: compare `before` and `after` gate by gate; a gain in one family that costs
  another is not a gain.
- Figures: `figures/raw/zoom_irregular.png`, `zoom_seam.png`, `zoom_angle.png` show where the defects are
  ([`outputs.md`](outputs.md)); `meshom repair --figures` and `meshom optimize --figures` render the new mesh's set.

## Config keys used

- `[mesh]`: `optimize`, `optimize_opts`, `size_tolerance`
- `[mesh.opts]` `m14_blocks_rep`: `repair`, `coast_fix`, `freeze_blocks`, `optimize`
- `[mesh.optimize_opts]`: `steps`, `passes`, `quad_max`, `relax_rings`, `fit`
- `[coast]`: `treatment`, `passage_min_cells`
- `[run]`: `smooth`
- `[gates]`: `elem_neighbours_min`, `elems_per_node_max`, `boundary_edges_max`, `boundary_rule`, `pinch_nodes_max`, `quad_sj_hard`, `tri_cluster_max`

## Sources

- Code: `meshom/run.py` (`cut_land`), `meshom/methods/m14/cut.py` (the clean-up rules), `meshom/passages.py`,
  `meshom/repair.py`, `meshom/size_fit.py`, `meshom/optimize.py`, `meshom/smooth.py`,
  `meshom/meshfix/pipeline.py`.
- Docs: `docs/methods/repair.md` ([CLI](../methods/repair.md#cli)), [`docs/config.md` "`[mesh]`"](../config.md#mesh),
  [`docs/config.md` "Running"](../config.md#running).
- Decisions: B64, B141, B149, B157, C51, C52, G6, G10, G11, G14, G15, G38, M38, O12, O25, R22, R24, Z1-Z13.
