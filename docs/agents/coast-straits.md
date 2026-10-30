---
name: meshom-coast-straits
description: Coast data and treatment (land cut or conforming coast), small islands, the rule that a global mesh keeps only the ocean-connected water, the built-in strait list after D. Sein, strait profiles and regions, added straits, automatic straits and the narrow-passage rule. Read it when you choose the coast or when straits, basins or connectivity matter.
---

# Coast and straits

## Coast source

- `[coast] source`: a land-polygon shapefile. The examples use GSHHG's `GSHHS_i_L1.shp` (intermediate resolution,
  level 1 = land against ocean); Natural Earth land or a shapefile of your own work too. `"none"` meshes the closed
  sphere (no land: method tests only).
- `antarctica`: GSHHS level 1 has no Antarctica. `"ice_front"` (default) adds the L5 file found next to the source
  (the ice-shelf front: no cavities, FESOM2's practice), `"grounding"` the L6 file (grounding line), `"none"`
  nothing (C20). The `coast` stage record says what was added.
- `min_area_km2`: land polygons below this area are dropped (the global examples use 3000, the finer ones 1000).
  The size function decides which smaller islands survive the land cut anyway.
- Polygons cut at the antimeridian are joined back; the poles and ±180° are ordinary points.

## Coast treatment

- **`treatment = "cut"`** (default; C1): the method meshes the closed sphere, then every element whose centroid is on
  land is removed and the element mask is cleaned with FESOM2's rules after D. Sein (AWI), fesom2_mesh_gen (no
  spurs, no pinched nodes, the largest basin only). The coast is a staircase of whole cells. Robust on any real
  coast. Use it for every global mesh.
- **`treatment = "loops"`** (conforming coast): the coast is smoothed to h by the coast filter (C16: water narrower
  than about 2 h closes unless a strait keeps it open, land narrower than h goes unless that would split land or
  join waters, cut-off basins dropped), resampled at h, and becomes the mesh boundary. `smoothing_m` sets the filter
  radius in metres instead of h/2 (0 = no filter). Needed for coast-aligned cells; weaker on real coasts today (an
  m14 mesh passes 36-39 of 39 gates, `m15_tri` 24 of 29) and costly (the CORE2-like coast with 10 km straits: 209 s
  and 4.9 GB for the filter alone).

## Water connectivity (C44, C48, C52)

A global mesh keeps only the water connected to the world ocean: no lakes, no cut-off basins. A basin the coast cuts
off at the mesh's h is dropped (at 25-35 km the Black Sea, unless its straits are opened). Every written mesh gets a
`water` record (`connectivity_ok`, `n_components`, each component's elements, nodes, km² and box). When the water
is not one piece, `meshom run` and `meshom write` print `WATER NOT CONNECTED` with each cut-off piece and go on, and
`water_components` enters the format checks' `stop` list: open the missing strait. `keep_basins_km2` (keep every
basin of at least that area) is for special cases only, never for a global ocean mesh.

## The strait list

`[coast] straits = "fesom2_mesh_gen"` selects meshom's built-in list of 90 strait centrelines after D. Sein (AWI),
fesom2_mesh_gen: its 88 curated straits plus the Great Belt and Marmara_West (C27). Each entry has a path from
basin to basin, the channel width kept open, a target size and a sill depth (`meshom/data/straits_fesom2_mesh_gen.json`).
**Gibraltar is opened on every mesh with a real coast**, even with `straits = "none"` (C49); only an explicit
`[[straits]]` entry `name = "Gibraltar"`, `enabled = false` closes it. Do not do that for an ocean mesh.

A strait acts in four places: the size (a zone along the path, `resolve` only), the land cut (elements within half
the corridor of the path stay water and are protected from the clean-up), the depth (a floor near the path, see
[`bathymetry.md`](bathymetry.md)) and the report.

**Corridor width** W (C29, C45): max(`width_m`, (`cells` · 1.15 + 1/√3) · h) on triangle meshes;
max(`width_m`, (`cells` · 1.15 + √2) · e^t · h) on meshes with quads, t the size tolerance. At 2 cells and t = 0.20
that is 4.54 h on quads against 2.88 h on triangles.

## Profiles and regions (C45-C47)

| profile | size | use for | measured |
|---|---|---|---|
| `widen` | no zone: the strait is meshed at the background h; the corridor is at least `strait_widen_cells` (2) cells of that h, widened into the land | coarse meshes with a long time step (CORE2 class) | `m15_tri` CORE2-like: smallest edge 21.8 km, 29/29, all straits connected |
| `resolve` (default) | a size zone `h_m` (raised to `strait_h_min_m`) along the path | fine meshes (NG5 class), a region of focus | `m15_tri` CORE2-like at 10 km: smallest edge 8.7 km, 29/29, 90/90 |

- `[coast] strait_profile` sets the profile of every listed strait when there are no regions.
- `[[strait_regions]]` (file order, first match wins): `profile` (`"resolve"`, `"widen"` or `"off"` = not opened)
  for the straits whose middle point lies in the region's `box` = `[west, east, south, north]` (west > east crosses
  180°) or `polygon`. With any region, a strait in no region is not opened. Gibraltar is still widened.
- A `[[straits]]` entry's own `profile` wins over both and opens the strait wherever it lies.

Example `docs/examples/fesom2_arctic_straits.toml`:

```toml
[coast]
straits = "fesom2_mesh_gen"
strait_h_min_m = 10000.0

[[strait_regions]]
name = "Baltic"
profile = "off"
box = [9.0, 31.0, 53.0, 66.0]

[[strait_regions]]
name = "north of 63N"
profile = "resolve"
box = [-180.0, 180.0, 63.0, 90.0]

[[strait_regions]]
name = "everything else"
profile = "widen"
box = [-180.0, 180.0, -90.0, 90.0]
```

**The Black Sea** depends on four straits: Bosphorus, Dardanelles, Marmara_West, Kerch. With the list and any
profile they connect it. CORE2 has no Black Sea: `fesom2_core2like_widen.toml` disables these four, because a
widened Bosphorus (an 86 km corridor) made FESOM2 blow up there.

**Time step**: straits resolved at 10 km need a shorter model step than CORE2's 45 min (FESOM2: 15 min with
w-splitting); at 20 km with 4 depth smoothing passes FESOM2 runs at 45 min (`fesom2_core2like_straits45.toml`;
C31, O23, O29).

## Adding or changing a strait

```toml
[[straits]]
name = "Bosphorus"                 # a built-in name overrides only the fields given; a new name adds a strait
path = [[29.147, 41.247], [29.065, 41.146], [28.99, 40.979]]   # [lon, lat], from one basin well into the other
width_m = 2050.0                   # channel width kept open
h_m = 10000.0                      # target size along the path (resolve)
cells = 2                          # cells across wanted
depth_m = 35.0                     # depth floor near the path
profile = "resolve"                # optional
```

`enabled = false` drops a strait. Bering and Fram are not in the list (wide enough to stay open by themselves);
`fesom2_arctic_straits.toml` adds them as resolved entries to refine them.

## Narrow passages (C51)

`[coast] passage_min_cells` = K (default 2; 3 where straits matter; 0 = off): every through-passage of the final
mesh, listed or not, gets at least K cells across. The test is topological (no node off the coast across it).
With `cut` the run adds a widened corridor and cuts again (up to 3 rounds); with `loops` it rebuilds with a
`[[straits]]` entry per passage. One-cell fjords and small dead-end inlets are left alone. `stages.json` → `post` →
`passages` lists every widened passage; a `NARROW PASSAGES` line means one is left: add a `[[straits]]` entry there
(`profile = "widen"`, `cells = 3`). With `loops` only K = 2 converges reliably.

## Automatic straits (S22, S24, S31)

`auto_straits = true` finds passages on the medial axis of the water and gives each a size zone h = W/n (n =
`strait_n_min`, 5). With a coarse h and a large `h_min_m` only passages wider than about 100 km open; Gibraltar and
the Bosphorus stay with the list. Leave it off unless the request asks for it.

## The strait report

`quality.json` → `straits`: per strait `profile`, `cells_target`, `cells_min` (cells across at the narrowest of 40
sections), `closed_sections`, `connected` (both ends in the main water), `depth_min_m`, `ok`.

## Config keys used

- `[coast]`: `source`, `antarctica`, `min_area_km2`, `treatment`, `smoothing_m`, `keep_basins_km2`, `straits`, `strait_h_min_m`, `strait_profile`, `strait_widen_cells`, `passage_min_cells`, `auto_straits`, `strait_n_min`
- `[[straits]]`: `name`, `path`, `width_m`, `h_m`, `cells`, `depth_m`, `profile`, `enabled`
- `[[strait_regions]]`: `name`, `profile`, `box`, `polygon`

## Sources

- Code: `meshom/coast.py`, `meshom/coast_filter.py`, `meshom/straits.py`, `meshom/passages.py`, `meshom/lfs.py`,
  `meshom/methods/m14/cut.py` (the land cut's clean-up), `meshom/data/straits_fesom2_mesh_gen.json`.
- Docs: [`docs/config.md` "`[coast]`"](../config.md#coast),
  ["The coast filter"](../config.md#the-coast-filter-coast-treatment--loops),
  ["Straits"](../config.md#straits-coast-straits-straits),
  ["Strait profiles"](../config.md#strait-profiles-coast-strait_profile-strait_regions),
  ["Narrow passages"](../config.md#narrow-passages-coast-passage_min_cells),
  ["Automatic straits"](../config.md#automatic-straits-coast-auto_straits).
- Decisions: C1, C16, C20, C27, C29, C31, C44-C49, C51, C52, S22, S24, S31, O23, O29.
