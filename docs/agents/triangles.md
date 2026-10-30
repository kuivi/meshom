---
name: meshom-triangles
description: How to build a triangle-only mesh with m15_tri (FESOM2, or triangles for FESOM3) - the example configs and their measured results, the method's options, the gates that apply, and the FESOM2 namelist and partitioning. Read it when the request is a FESOM2 mesh or a triangle mesh.
---

# Triangle meshes (`m15_tri`)

## What the method does

`m15_tri` (`meshom/methods/m15_tri.py`) builds a pure triangle mesh that follows h closely:

1. **Seeded packing**: frontal point packing along a 6-fold field from the coast and from many seeds at once
   (`seeds = "auto"`: about one seed per 1000 nodes), then a Delaunay triangulation on the sphere.
2. **Population control** (`meshom/popctl.py`): up to `popctl_rounds` rounds that delete nodes of valence ≤ 4 and
   one end of every edge much shorter than h (ln(s/h) < `t_del`), and insert a midpoint on every edge much longer
   (ln(s/h) > `t_ins`), each round re-triangulated.
3. **Per-edge size fit** (`meshom/size_fit.py`): the free nodes move so that every edge is as close to h as
   possible, with flips between rounds; a valid triangle never inverts. Deterministic for any thread count.

With `[coast] treatment = "cut"` (the global default) the method meshes the closed sphere and the run cuts the
land away and applies FESOM's clean-up ([`cleanup.md`](cleanup.md)).

## Build one

1. Copy the example that matches the request:

| need | config | measured |
|---|---|---|
| CORE2-like, Gibraltar only (no strait list) | `docs/examples/fesom2_core2like.toml` | 162 557 nodes, 29/29 gates, the Black Sea dropped (cut off at 30 km) |
| CORE2-like, every listed strait widened, FESOM2 at 45 min | `docs/examples/fesom2_core2like_widen.toml` | 162 705 nodes, 29/29, one water component, 86/86 straits connected; Black Sea, Marmara and Azov left out as in CORE2 |
| CORE2-like, straits resolved at 20 km, FESOM2 at 45 min | `docs/examples/fesom2_core2like_straits45.toml` | one water component (the Black Sea connected), FESOM2 runs at 45 min without w-splitting |
| CORE2-like, straits resolved at 10 km | `docs/examples/fesom2_core2like_straits.toml` | 181 674 nodes, 29/29, 90/90 straits; FESOM2 needs a 15 min step with w-splitting |
| about 1 M nodes (`scale = 0.4`) | `docs/examples/fesom2_finer.toml` | 1.02 M nodes, 29/29, 14 min on 32 CPUs, 3.1 GB |
| Arctic focus: Baltic off, north of 63° N resolved at 10 km, the rest widened | `docs/examples/fesom2_arctic_straits.toml` | 173 058 nodes, 29/29, one water component (the Baltic dropped) |

2. Set `[mesh] out`, `[coast] source`, `[bathymetry] source` to your files ([`workflow.md`](workflow.md)).
3. Change the resolution with the size layers ([`resolution.md`](resolution.md)) or uniformly with `[mesh] scale`
   (nodes scale with 1/scale²).
4. Keep `method = "m15_tri"` and `format = "fesom2"` (or `"fesom3"`: FESOM3 reads triangle meshes too).
5. Run, then check that `quality.json` → `raw` → `gates_failed_families` is empty, `water.connectivity_ok` is true,
   every strait has `ok` true and `stages.json` → `write` → `fesom2_checks` → `stop` is empty.

## Options (`[mesh.opts]`)

Leave them at their defaults unless a measurement says otherwise. The full list: `DEFAULTS` in
`meshom/methods/m15_tri.py`.

- `seeds` (`"auto"`): fronts of the packing, or a number.
- `popctl_rounds` (8; 0 = off), `t_del` (−0.18), `t_ins` (0.22): the population control.
- `fit` (true), `fit_rounds` (3), `fit_iters` (150): the size fit.
- `size_tolerance` (none): with a value t the last stage makes sizes as smooth as possible within |ln s/h| ≤ t
  instead of fitting s = h. Measured: a small gain (element ratio p99 1.109 to 1.086 on the CORE2-like mesh) at
  2-4 times the time, so the default stays none for `m15_tri` (G38, M38).

## Gates on a triangle mesh

A triangle-only mesh on a real coast with a land cut has 29 applicable gates (the quad gates and `tri_cluster_max`,
`tri_share_max` do not apply; G20). The ones that matter (defaults in `meshom.config.DEFAULT_GATES`):

| gate | default | meaning |
|---|---|---|
| `tri_min_angle_hard_deg` | 20 | every triangle (G5) |
| `tri_max_angle_hard_deg` | 120 | every triangle (G3) |
| `tri_max_angle_target_deg` | 90 | p99 of the largest angle, on meshes without quads (G4) |
| `elem_size_ratio_p99`, `elem_size_ratio_max` | 1.15, 2.0 | mean edge of the two elements across each interior edge (G7, G8) |
| `size_fidelity_p99` | 0.15 | p99 of \|ln(s/h)\| per edge (G9) |
| `boundary_edges_max` with `boundary_rule = "tri"` | 1 | no triangle with two coast edges: FESOM2 rejects it (G14) |
| `elem_neighbours_min`, `elems_per_node_max` | 2, 32 | FESOM's neighbour limits (G10, G11) |

Each gate is evaluated on the whole mesh and on the subsets `_coast`, `_interior`, `_coast3` (G17, G18).

## FESOM2 after the build

- Files: `nod2d.out`, `elem2d.out` (3 columns, clockwise), `aux3d.out` (the `[bathymetry] levels` and the node
  depths, negative down) in `RUN_DIR/fesom/`. Coordinates are geographic, not rotated (O9, O11).
- `fesom2_checks` → `stop` lists what would stop FESOM2 with its default namelist (O21): an element with fewer than
  2 neighbours, more than 32 elements or neighbours at a node, a node near the rotated North Pole, an element
  holding a rotated pole, an `aux3d.out` node count that differs.
- Namelist: `force_rotation = .true.` (Euler angles 50 / 15 / −90), `use_depthfile = 'aux3d'`, `use_cavity =
  .false.`, `which_depth_n2e = 'mean'`, `MeshPath` = the `fesom/` folder with a trailing `/`.
- Partitioning: FESOM2's own partitioner `fesom_meshpart` (O22).
- Time step: CORE2-like meshes with widened straits, or straits resolved at 20 km with 4 depth smoothing passes, run
  at 45 min; straits resolved at 10 km need 15 min and w-splitting (O23, O29).

## Config keys used

- `[mesh]`: `method`, `format`, `scale`, `size_tolerance`
- `[mesh.opts]` `m15_tri`: `seeds`, `popctl_rounds`, `t_del`, `t_ins`, `fit`, `fit_rounds`, `fit_iters`, `size_tolerance`
- `[coast]`: `treatment`, `straits`, `strait_profile`
- `[gates]`: `tri_min_angle_hard_deg`, `tri_max_angle_hard_deg`, `tri_max_angle_target_deg`, `elem_size_ratio_p99`, `elem_size_ratio_max`, `size_fidelity_p99`, `boundary_edges_max`, `boundary_rule`, `elem_neighbours_min`, `elems_per_node_max`

## Sources

- Code: `meshom/methods/m15_tri.py`, `meshom/popctl.py`, `meshom/size_fit.py`, `meshom/io/fesom.py`
  (`write_fesom2`, `fesom2_checks`), `meshom/quality/metrics.py` (gates).
- Docs: `docs/methods/m15_tri.md`, [`docs/config.md` "`[gates]`"](../config.md#gates).
- Decisions: G3, G4, G5, G7-G11, G14, G17, G18, G20, G38, M38, O9, O11, O21-O23, O29.
