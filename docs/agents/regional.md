---
name: meshom-regional
description: What meshom can and cannot do today for regional meshes with open boundaries (FESOM-C) - the FESOM-C writer works, open boundaries and a regional domain do not exist yet - and the closest things that do work. Read it when a request asks for a regional, coastal or FESOM-C mesh.
---

# Regional meshes (FESOM-C)

## Say it plainly

meshom builds global meshes. **Regional meshes with open boundaries are not built yet.** There is no config key for
a regional domain (a polygon that cuts the sphere down to a region), and the writers never set the open-boundary
node flag 2 that FESOM-C needs: every boundary node is written with flag 1 (coast), every other node with 0 (O6).
FESOM3 has no open boundaries at all.

If the request is a regional FESOM-C mesh, tell the requester this before you build anything, and offer one of the
options below.

## What works

- **The FESOM-C file format**: `[mesh] format = "fesom-c"` (the default) writes `nod2d.out` (flag 1 on boundary
  nodes), `elem2d.out` with 4 columns (a triangle written `n1 n2 n3 n1`, clockwise), `depth.out` (metres, positive
  down) and a placeholder `aux3d.out` that FESOM-C's partitioner reads only for level counts (O1-O8). FESOM-C is
  sigma-coordinate, so the placeholder is fine; `depth.out` is what the model uses. There are no FESOM-C checks in
  the write stage.
- **A global mesh with a fine region of focus**: a `zone` size layer (or several) gives the region its resolution,
  graded to the coarse ocean ([`resolution.md`](resolution.md)); `[[strait_regions]]` resolve the straits there and
  widen or close them elsewhere ([`coast-straits.md`](coast-straits.md)). Example:
  `docs/examples/fesom3_meshC_arctic.toml`.
- **Coast-aligned cells**: `[coast] treatment = "loops"` makes the smoothed coastline the mesh boundary (weaker than
  the land cut on real coasts today; see [`coast-straits.md`](coast-straits.md)).
- **A finer patch in a finished mesh**: inserts in a strait, a bay, along a coast ([`inserts.md`](inserts.md)).
- **A lake or a cut-off basin on purpose**: `[coast] keep_basins_km2` keeps every water basin of at least that area
  besides the largest (never in a global ocean mesh: C48). The water check then reports more than one component.
- **Scoring an existing regional mesh**: `meshom quality FESOM_DIR` reads a FESOM mesh folder;
  `meshom quality MESH.msh --format gmsh41 --center LON LAT` reads a Gmsh 4.1 file in a stereographic plane around
  the given centre. Both compute the quality summary and the gates (no h, so no size gate) and write `report.md`.

## A workaround, outside meshom

1. Build a global mesh with the region refined as wanted, `format = "fesom-c"`, `treatment = "cut"` or `"loops"`.
2. Cut the region out of `nod2d.out` / `elem2d.out` with your own script, renumber the nodes, and set flag 2 on the
   nodes of the new open boundary (FESOM-C expects both end nodes of an open stretch flagged).
3. Score the result with `meshom quality FESOM_DIR`.

meshom does not check step 2. Say so in your report.

## Config keys used

- `[mesh]`: `format`
- `[coast]`: `treatment`, `keep_basins_km2`
- `[[size.layers]]` `zone`: `h_m`, `transition_m`, `polygon`, `h_out_m`
- `[[strait_regions]]`: `name`, `profile`, `box`, `polygon`

## Sources

- Code: `meshom/io/fesom.py` (`write_fesomc`, `read_fesom`), `meshom/cli.py` (`meshom quality`), `meshom/run.py`.
- Docs: [`docs/config.md` "`[mesh]`"](../config.md#mesh), [`docs/config.md` "`[coast]`"](../config.md#coast).
- Decisions: O1-O8 (FESOM-C files, node flag), C48 (only the ocean-connected water), C1 (coast treatment).
