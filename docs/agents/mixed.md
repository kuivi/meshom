---
name: meshom-mixed
description: How to build a mixed quad-dominant mesh for FESOM3 with m14_blocks_rep - the steps inside the method, the example configs with measured results, the options worth changing, hand-placed blocks, and what a good mixed mesh looks like. Read it when the request is a FESOM3 mesh with quads.
---

# Mixed meshes (`m14_blocks_rep`)

## What the method does

`m14_blocks_rep` (`meshom/methods/rep_variants.py` over `meshom/methods/m14_blocks.py`) is the production method
for mixed meshes:

1. **Blocks**: structured curvilinear quad grids placed where h allows a defect-free lattice: first the hand-placed
   `[[blocks]]`, then strait blocks (conforming coasts only), then a greedy automatic placement from h.
2. **Fill**: the space between blocks, coast and land is filled by frontal quads (`m02_linf_frontal`), stitched to
   the blocks.
3. **Repair** (`meshom/repair.py`): clusters of defects (triangles, irregular nodes, bad angles) are cut out and
   remeshed locally; a candidate is kept only when it scores better.
4. **Size fit within the tolerance** (`meshom/size_fit.py`): the free nodes move so that sizes change between
   neighbours as gently as possible while every edge keeps |ln(s/h)| ≤ t, t = 0.20 by default (G38, M38, R22).
5. **Optimisation stage** (`meshom/optimize.py`): flips, triangle pairs merged into quads, triangle clusters broken,
   cancelling irregular pairs removed, a local fit (Z1). Under a land cut the run applies it after the cut.

The method does all of this inside the `generate` stage; its records are `m14.json` (block layout and coverage),
`repair.json`, `tol_fit_final.json` and `optimize.json` in the run folder.

## Build one

| need | start from | measured |
|---|---|---|
| coarse CORE2 class, every listed strait widened | `docs/examples/fesom2_core2like_widen.toml` with `method = "m14_blocks_rep"`, `format = "fesom3"`, `[run] timeout = 3600.0` | 144 439 nodes (135 800 quads, 3 720 triangles), 39/39 gates, element ratio p99 1.110, \|ln s/h\| p99 0.194, one water component, every strait at ≥ 2 cells, 16 min on 16 CPUs; ran 10 days in FESOM3 at 30 min |
| the same at half the size | the above with `[mesh] scale = 0.5`, `timeout` ≥ 3600 | 572 280 nodes, 39/39, 32 min, 3.0 GB |
| fine NG5 class, every strait resolved at 5 km | `docs/examples/fesom3_meshB.toml` | 3.87 M nodes (36 912 triangles), 38/39 gates (`tri_cluster_max` 3), one water component, 92/92 straits ≥ 3.3 cells, 6 h 17 min on 32 CPUs, 13.8 GB; FESOM3 5 days at 6 min |
| global with a 10 km Arctic | `docs/examples/fesom3_meshC_arctic.toml` | a CORE2-like base, strait regions (Baltic off, Arctic resolved), 10 km north of 60° N |
| a quick look at the block layout | `docs/examples/global_150km_m14.toml` (plain `m14_blocks`) | seconds to minutes; not a production setting |

The widen example disables the four Black Sea straits (CORE2 has no Black Sea); keep or remove those `[[straits]]`
tables as the request wants. With them open the CORE2-class mixed mesh keeps the Black Sea connected.

Steps: copy the config, set `out` and the data paths, keep `[mesh] size_tolerance` unset (0.20 for every m14
method), keep `[run] smooth = false` (the common smoothing makes these meshes worse on size), set `[run] figures =
true`, run, read the gates and the figures ([`outputs.md`](outputs.md)).

## What a good mixed mesh looks like

- Quads dominate: on the coarse global mesh 97 % of the elements are quads; the gate `tri_share_max` allows up to 35 %.
- No cluster of three or more edge-connected triangles outside declared triangle zones (`tri_cluster_max` = 2, G13).
- Quad corners: every corner below 150° (`quad_max_angle_hard_deg`), p99.9 below 135°, every corner sine ≥ 0.3
  (`quad_sj_hard`) (G1, G2, G6).
- Smooth sizes: element size ratio across each interior edge p99 ≤ 1.15 and max ≤ 2.0 (G7, G8); |ln(s/h)| p99
  within the tolerance t = 0.20 (G38).
- Defects sit in the seams between blocks and fill; the blocks themselves are regular. On real global coasts the
  blocks cover about 30 % of the water.
- A real-coast mixed mesh with a land cut has 39 applicable gates; `gates_failed_families` should be empty.

## Options worth knowing (`[mesh.opts]`)

The full list: `DEFAULTS` in `meshom/methods/m14_blocks.py`, plus the `_rep` options below. Change one at a time.

- `size_tolerance` is better set as `[mesh] size_tolerance`. Never 0 for m14: the fill then fails the ratio gate.
  0.10 and 0.20 pass on the CORE2-like mesh, 0.30 adds a triangle cluster.
- `block_gap` (6, in h): blocks keep this distance apart; the fill takes the gap.
- `max_blocks`: no cap for `m14_blocks_rep` (the greedy runs until no seed is left; 32 for the other variants; B8).
- `drop_small` (0.25): blocks below a quarter of the median block's cells are left to the fill (B68).
- `place_region` (none): a `[[lon, lat], ...]` polygon; blocks only inside it. With `max_blocks = 1`: one block
  over a uniform fine zone (B106).
- `auto_blocks` (true): false = no automatic block; the fill takes the domain.
- `water_place` (true): under a land cut, seeds only in the water and a block counted by its water cells (B70).
- `strait_blocks` (true): blocks in the passages of the medial axis (conforming coasts only).
- `coast_bands` (false): thin quad bands along a conforming coast; cleaner coast rows, slightly worse coast size.
- `seeds` (none): a list of `[lon, lat]` grows exactly these blocks in order; `front_peer` (6 h) keeps such a
  layout stable (B69).
- `repair` (a table of `meshom.repair` options), `coast_fix` (true, conforming coasts only, B64), `freeze_blocks`
  (`["hand"]`: the block kinds the later stages leave alone; `"strait"`, `"band"` possible, `[]` = none; R24).
- `gap_adapt` (false): adaptive gaps; 3-7 times slower and no gain with the tolerance (B63). Leave it off.
- `optimize` (true) and `optimize_opts`: the optimisation stage; `[mesh] optimize` sets the same.

## Hand-placed blocks (`[[blocks]]`)

Use them only where a region needs a hand layout (a strait, a basin, an ocean you want as one lattice). The
automatic placement covers the rest. The m14 methods build them first (`m14_blocks`, `m14_blocks_cut`,
`m14_blocks_tri`, `m14_blocks_rep`; not `_igm`, `_mix`; B60).

```toml
[[blocks]]
name = "Atlantic"
corners = [[-42.0, -20.0], [15.0, -20.0], [-7.0, 40.0], [-76.0, 40.0]]   # [lon, lat], counter-clockwise
sides = ["lat", "cut", "lat", "cut"]   # per side: "coast" | "cut" (great circle) | "lat" (parallel) | "offset"
rows = 64                              # cells along sides 1-2 and 3-4 (default: from h)
cols = 70                              # cells along sides 2-3 and 4-1 (default: from h)
```

- Corners counter-clockwise: the block lies on the left of each side. `h_m` sets a constant cell size inside
  (default: the run's h); `snap_m` (1 h) moves corners onto a nearby coast; `filter_m` (2 h) smooths coast sides.
- Default `rows` / `cols` follow the sides only. A block whose middle is longer than its sides (a wide ocean
  between two parallels) needs them set by hand: compare `row_cells_h` / `col_cells_h` in `m14.json`
  `fixed_blocks`, which say what fits.
- With `[mesh] scale` ≠ 1, `h_m` is scaled and `rows` / `cols` divided by the scale.
- Under a land cut a block without a `coast` side is cut by the land like everything else; a block with a `coast`
  side is kept as water. With a conforming coast a block without a `coast` side must lie in water.
- `m14_blocks_rep` keeps a hand block exactly as drawn: repair, fit and optimisation work only around it.
- The block's quality is in `m14.json` → `fixed_blocks` (`b22`: every edge within 0.10 of h, every angle ≤ 135°).

## Config keys used

- `[mesh]`: `method`, `format`, `scale`, `size_tolerance`, `optimize`, `optimize_opts`
- `[mesh.opts]` `m14_blocks_rep`: `size_tolerance`, `block_gap`, `max_blocks`, `drop_small`, `place_region`, `auto_blocks`, `water_place`, `strait_blocks`, `coast_bands`, `seeds`, `front_peer`, `repair`, `coast_fix`, `freeze_blocks`, `gap_adapt`, `optimize`, `optimize_opts`
- `[[blocks]]`: `name`, `corners`, `sides`, `rows`, `cols`, `h_m`, `snap_m`, `filter_m`
- `[run]`: `timeout`, `smooth`, `figures`
- `[gates]`: `tri_share_max`, `tri_cluster_max`, `quad_max_angle_hard_deg`, `quad_max_angle_target_deg`, `quad_sj_hard`, `elem_size_ratio_p99`, `elem_size_ratio_max`, `size_fidelity_p99`

## Sources

- Code: `meshom/methods/m14_blocks.py`, `meshom/methods/m14/` (`hand.py` for hand blocks), `meshom/methods/rep_variants.py`,
  `meshom/repair.py`, `meshom/size_fit.py`, `meshom/optimize.py`.
- Docs: `docs/methods/m14_blocks.md`
  ([size tolerance](../methods/m14_blocks.md#smoothness-within-a-size-tolerance-2026-10-23-decisions-g38-m38-r22)),
  `docs/methods/repair.md`, [`docs/config.md` "`[[blocks]]`"](../config.md#blocks).
- Decisions: B8, B60, B63, B64, B68, B69, B70, B106, G1, G2, G6, G7, G8, G13, G27, G38, M38, R22, R24, Z1.
