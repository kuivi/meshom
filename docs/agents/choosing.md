---
name: meshom-choosing
description: Which mesh, method, resolution class and example config fit a request (global FESOM3 mixed, FESOM2 triangles, ORCA, regional FESOM-C). Read it at the start of a task, before you write a config.
---

# Choosing a mesh, a method and a starting config

## First questions

1. **Which model?** FESOM3 reads mixed meshes (quads and triangles) and triangle meshes; FESOM2 reads triangles
   only; FESOM-C reads mixed meshes with open boundaries. The model sets `[mesh] format`.
2. **Global or regional?** meshom builds global meshes (the closed sphere minus land). Regional meshes with open
   boundaries are not built yet ([`regional.md`](regional.md)); a global mesh with a fine region of focus is.
3. **Which resolution class?** Below.
4. **Does it have to be an ORCA grid?** Then `m16_orca` ([`orca.md`](orca.md)).
5. **Which straits must be open, and at what size?** [`coast-straits.md`](coast-straits.md).

## Decision table

| request | method | `[mesh] format` | start from |
|---|---|---|---|
| global mixed mesh for FESOM3, coarse (CORE2 class) | `m14_blocks_rep` | `"fesom3"` | `fesom2_core2like_widen.toml` with the method and format changed ([`mixed.md`](mixed.md)) |
| global mixed mesh for FESOM3, fine (NG5 class, 1-5 M nodes) | `m14_blocks_rep` | `"fesom3"` | `fesom3_meshB.toml` |
| global mixed mesh for FESOM3 with a fine region (e.g. the Arctic) | `m14_blocks_rep` | `"fesom3"` | `fesom3_meshC_arctic.toml` |
| global triangle mesh for FESOM2 | `m15_tri` | `"fesom2"` | `fesom2_core2like*.toml`, `fesom2_finer.toml` ([`triangles.md`](triangles.md)) |
| triangle mesh for FESOM3 | `m15_tri` | `"fesom3"` | a FESOM2 example with the format changed |
| an Arctic-focus triangle mesh | `m15_tri` | `"fesom2"` | `fesom2_arctic_straits.toml` |
| NEMO's ORCA grid, whole | `m16_orca` | `"fesom3"` | `orca2_whole.toml`, `orca1_whole.toml`, `orca025_whole.toml` |
| ORCA up to a latitude or a line, a meshom mesh beyond | `m16_orca` | `"fesom3"` | `orca1_40n_arctic10.toml`, `orca025_arctic45.toml`, `orca025_arctic45_v3.toml` |
| ORCA with a finer ORCA inside (around the fold, or a box) | `m16_orca` | `"fesom3"` | `orca025_arctic_v4.toml`, `orca025_arctic_v5.toml`, `orca025_gs_nest.toml` |
| a finer patch in one strait, bay, coastal strip or current of a finished mesh | `python -m meshom.insert` | the run's | `inserts.toml`, `inserts_gulf_stream.toml` ([`inserts.md`](inserts.md)) |
| a quick look at a method on a coarse globe | `m14_blocks`, `m02_linf_frontal` | `"fesom-c"` | `global_150km_m14.toml`, `global_150km_m02.toml` |
| a regional FESOM-C mesh with open boundaries | not built | | [`regional.md`](regional.md) |

All example configs are in `docs/examples/`. Never combine a quad method with `format = "fesom2"`: the FESOM2 writer
writes triangles only, and the quads are dropped (the `fesom2_checks` then show unused nodes).

## Resolution classes

| class | open ocean | coasts, high latitudes | nodes (global) | straits | model time step |
|---|---|---|---|---|---|
| coarse (CORE2 class) | 100-110 km | 25-30 km | 140-180 k | `widen`: at the background size, at least 2 cells across | long (FESOM2 45 min) |
| intermediate | about 50 km | 10-25 km | 0.5-1 M | `widen`, or `resolve` at 10-20 km | shorter |
| fine (NG5 class) | 10-15 km | 5 km | 1-5 M | `resolve`: a size zone along each strait (5 km) | about 10 min or less |
| regional focus | coarse far away | 1-10 km in the focus | depends on the focus area | per region: resolve in the focus, widen elsewhere | set by the smallest cells |

Classes coarser than about 100 km are for quick tests only: they decide nothing about the quality of a real mesh.
Metre-scale meshes are regional FESOM-C work, not built yet.

## Methods

The registry is `meshom.methods.METHODS`; `meshom compete --help` shows how to score methods on the built-in test
cases. Use these:

- **`m14_blocks_rep`** (mixed, the production method): structured quad blocks placed from h, the rest filled by
  frontal quads, defect clusters repaired, sizes smoothed within the size tolerance 0.20, then the optimisation
  stage. Passed every gate on the competition cases and 39/39 on a real CORE2-like coast; its meshes run in FESOM3.
- **`m15_tri`** (triangles): seeded packing, population control and a per-edge size fit. Passed every gate on
  every case and scale tried, up to 12.2 M nodes; 29/29 on real coasts; its meshes run in FESOM2.
- **`m16_orca`**: NEMO's ORCA grids as quad meshes, alone or joined to a meshom part (its caps are built by
  `m14_blocks_rep`).
- `ref_jigsaw`: JIGSAW as a reference baseline for comparison (needs a separate JIGSAW install, not part of the
  meshom environment); never for production.

The other registered methods (`m01_isotropic` to `m13_squadgen` and their variants, `m14_blocks`,
`m14_blocks_mix`, `m14_blocks_tri`, `m14_blocks_cut`, `m14_blocks_igm`) are research variants: each fails gates on
some case or costs far more memory or time. Each has a note in `docs/methods/`.

## Coast treatment

- `[coast] treatment = "cut"` (default): the method meshes the closed sphere and the land is cut away; the coast is
  a staircase of whole cells. Use it for every global mesh: all measured global meshes are cut.
- `treatment = "loops"`: the coast smoothed to h is the mesh boundary (a conforming coast). Needed for coast-aligned
  cells; weaker today on real coasts (some coast gates fail). See [`coast-straits.md`](coast-straits.md).

## Config keys used

- `[mesh]`: `method`, `format`, `scale`
- `[coast]`: `treatment`, `straits`, `strait_profile`

## Sources

- Code: `meshom/methods/__init__.py` (the registry, `defaults`, `effective_tolerance`),
  `meshom/methods/rep_variants.py`, `meshom/methods/m15_tri.py`, `meshom/methods/m16_orca.py`.
- Docs: `docs/methods/m14_blocks.md`, `docs/methods/m15_tri.md`, [`docs/config.md` "`[mesh]`"](../config.md#mesh),
  [`docs/config.md` "NEMO's ORCA grids"](../config.md#nemos-orca-grids-method--m16_orca).
- Decisions: C1 (coast treatment), C45 (strait profiles), G38 (size tolerance), O1 (formats).
