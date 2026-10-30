---
name: meshom-resolution
description: How to set the size function h (the target edge length on the sphere) - layer kinds, file order and zone modes, the global clip, strait zones, the gradient limiter, scale and the size tolerance - with worked config snippets and the common pitfalls. Read it whenever you change where the mesh is fine or coarse.
---

# Resolution: the size function h

All resolution comes from one size function h on the sphere: the target edge length in metres at every point. The
methods follow h; nothing else sets the cell size (S20).

## How h is assembled

```
h = scale × limit( min( clip(the layers in file order, h_min_m, h_max_m), strait zones ), grad_max )
```

1. The `[[size.layers]]` are applied **in file order** to a running h (S34). Every kind but `zone`, and a zone of
   `mode = "min"` (the default), takes the minimum: the finest criterion wins, in any order.
2. `[size] h_min_m` / `h_max_m` clip the result.
3. Strait zones (`resolve` straits, `auto_straits`) come after the clip, so a 10 km strait zone survives
   `h_min_m = 20000` ([`coast-straits.md`](coast-straits.md)).
4. The gradient limiter makes h grow by at most `grad_max` per unit distance (|∇h| ≤ g: neighbouring cells differ by
   at most a factor 1 + g; default 0.1). It runs across land (S3, S7-S13).
5. `[mesh] scale` multiplies the final h: node count ∝ 1/scale², and |∇h| is multiplied by scale too (S15).

The **base h** is the largest enabled `constant` layer, else `h_max_m`, else the largest `h_m` / `h_out_m` of the
layers. Coast layers stop growing there and zones grow to it.

## Layer kinds

| kind | keys | h |
|---|---|---|
| `constant` | `h_m` | h_m everywhere |
| `latitude` | `h_m` (at the equator), `h_min_m` | max(h_m cos(lat), h_min_m) |
| `coast_distance` | `h_m` (at the coast), `grad`, `plateau_m` (0), `coasts` (`"all"`, `"continents"`, `"islands"`), `island_max_km2` (100 000) | h_m + grad · max(d − plateau_m, 0), at most the base h; d = geodesic distance to the coasts of that class |
| `zone` | `h_m`, `transition_m`, `polygon`, `h_out_m`, `mode`, `smooth_m`, `outer`, `h_factor` | `min`: h_m inside the polygon, linear in distance to `h_out_m` (default: the base h) over `transition_m`; other modes below |
| `orca` | `orca` (a preset), `h_min_m` | an ORCA grid's own cell size ([`orca.md`](orca.md), B101) |

Every layer also takes `name` and `enabled` (false: kept in the file, left out of the run).

**Zone modes** (S34, S36, S37). Put a `set` or `smooth` zone **after** the layers it should override:

- `mode = "set"`: h = h_m inside the polygon whatever the layers before it give (finer or coarser), linear to the
  running h over `transition_m`. Later layers still apply on top.
- `mode = "smooth"` (`smooth_m`): the running h inside is replaced by its ln h average over a disc of radius
  `smooth_m`, blended over `transition_m`. Use it to remove steps that a quad lattice cannot follow.
- `mode = "blend"` (`outer`, `transition_m = 0`): h_m inside `polygon`, the running h outside `outer`, ln h linear
  between the two edges. With `h_factor` instead of `h_m`: h_factor × the running h inside (B110). Used to join an
  ORCA part to a fine region.

**Coast classes** (S33): an island is a land polygon smaller than `island_max_km2` on the sphere; Antarctica is a
continent. Two `coast_distance` layers can refine continents and islands differently; `coasts = "continents"` alone
leaves islands unrefined. Coastal refinement is optional: many FESOM meshes refine toward the coast only in some
regions.

**h variants** (S35): keep a size function in its own file of `[size]` and `[[size.layers]]` only, and point a
config at it with `[size] from = "h.toml"` (relative to the config's folder). The config's own `[size]` keys
override the file's; its layers come after the file's. `config.resolved.toml` holds the merged layers.

**Polygons** have great-circle edges and must fit in a hemisphere. A band around the globe is split into several
zones, or given as a ring of points along one latitude (the edges bulge slightly poleward between the points).

## Worked snippets

A CORE2-like global size function (all `docs/examples/fesom2_core2like*.toml`): 110 km open ocean, 110 km cos(lat)
down to 25 km, 30 km within 30 km of the coast, 25 km north of 52° N.

```toml
[size]
grad_max = 0.1
h_min_m = 20000.0
h_max_m = 110000.0

[[size.layers]]
kind = "constant"
h_m = 110000.0

[[size.layers]]
kind = "latitude"
h_m = 110000.0
h_min_m = 25000.0

[[size.layers]]
kind = "coast_distance"
h_m = 30000.0
plateau_m = 30000.0
grad = 0.1

[[size.layers]]
kind = "zone"
name = "north of 52N"
h_m = 25000.0
transition_m = 1000000.0
polygon = [[-180.0, 52.0], [-150.0, 52.0], [-120.0, 52.0], [-90.0, 52.0], [-60.0, 52.0], [-30.0, 52.0],
           [0.0, 52.0], [30.0, 52.0], [60.0, 52.0], [90.0, 52.0], [120.0, 52.0], [150.0, 52.0]]
```

A fine region of focus (10 km in the Nordic Seas, graded over 1500 km). Lower `[size] h_min_m` too, or the clip
undoes the zone:

```toml
[[size.layers]]
kind = "zone"
name = "Nordic Seas 10 km"
h_m = 10000.0
transition_m = 1500000.0
polygon = [[-20.0, 62.0], [15.0, 62.0], [20.0, 72.0], [10.0, 82.0], [-20.0, 82.0], [-30.0, 72.0]]
```

A coarser area that overrides the coast layer before it (e.g. a coarse Southern Ocean coast):

```toml
[[size.layers]]
kind = "zone"
mode = "set"
name = "coarse Weddell Sea"
h_m = 60000.0
transition_m = 500000.0
polygon = [[-60.0, -78.0], [-10.0, -78.0], [-10.0, -65.0], [-60.0, -65.0]]
```

## Size tolerance

`[mesh] size_tolerance` = t lets each edge deviate from h by up to |ln(s/h)| ≤ t (0.20 ≈ ±22 %); within it the
method makes sizes change between neighbours as gently as possible (G38, M38). Default 0.20 for every m14 method,
none for the others. The `size_fidelity_p99` gate becomes t unless `[gates]` sets it; the element-ratio gates
(1.15, 2.0) never change. Measured on the CORE2-like `m14_blocks_rep` mesh: no tolerance 31/39 gates, t = 0.10 and
0.20 39/39 (element ratio p99 1.121 and 1.110), t = 0.30 38/39. The tolerance is a soft wall: single edges next to
fixed coast or block nodes may lie beyond it.

## Before you run

- **Node estimate**: `stages.json` → `size` → `n_nodes_est_sphere` = ∫h⁻² dA over the whole sphere (land included).
  A global mesh with a land cut ends at roughly 0.6 of it. Check it before a long build.
- **Limiter grid**: a cubed sphere at half the smallest h, capped at `limiter_max_nodes` (12 M ≈ 7.1 km spacing;
  about 1.1 GB per 12 M nodes; S10). If `stages.json` → `size` → `limiter` → `capped` is true, fine features of h
  (sizes below about 14 km, narrow strait zones) are smeared: raise the cap.
- **Pitfall, `h_out_m`**: a min zone reaches `h_out_m` at `transition_m` and keeps it everywhere beyond, on the
  whole sphere. Setting it below the open-ocean h refines the whole globe. Leave it unset unless that is wanted.
- **Pitfall, the clip**: `[size] h_min_m` clips every layer, zones included: a 10 km zone under
  `h_min_m = 20000` stays 20 km. Strait zones are not affected.
- **Pitfall, test scale**: judge smoothness at scale 1 only; a scale-4 test has 4 times the gradient.

## Config keys used

- `[mesh]`: `scale`, `size_tolerance`
- `[size]`: `layers`, `grad_max`, `h_min_m`, `h_max_m`, `limiter_max_nodes`, `from`
- `[[size.layers]]` `constant`: `h_m`, `name`, `enabled`
- `[[size.layers]]` `latitude`: `h_m`, `h_min_m`
- `[[size.layers]]` `coast_distance`: `h_m`, `grad`, `plateau_m`, `coasts`, `island_max_km2`
- `[[size.layers]]` `zone`: `h_m`, `transition_m`, `polygon`, `h_out_m`, `mode`, `smooth_m`, `outer`, `h_factor`, `name`
- `[[size.layers]]` `orca`: `orca`, `h_min_m`
- `[gates]`: `size_fidelity_p99`, `elem_size_ratio_p99`, `elem_size_ratio_max`

## Sources

- Code: `meshom/size.py` (layers, `limit_gradient`), `meshom/run.py` (`size_field`, `background_h`,
  `layer_fields`), `meshom/config.py` (`Config.base_h`, `dumps_size`, `loads_size`), `meshom/size_fit.py`.
- Docs: [`docs/config.md` "`[size]` and `[[size.layers]]`"](../config.md#size-and-sizelayers).
- Decisions: S3, S7-S13, S15, S20, S25, S33-S37, B101, B110, G38, M38.
