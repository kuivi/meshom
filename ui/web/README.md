# meshom web UI (v2): Python server + deck.gl GlobeView

    meshom ui [--config FILE] [--runs DIR]... [--refs DIR]... (each option names one folder; repeat it for more) [--port 8765] [--host H] [--token T | --no-token]
              [--allow-submit]

It prints `http://<host>:<port>/?token=<32 hex>`: open that URL (any recent browser; deck.gl 9.4.0 is loaded from
unpkg, pinned in `static/index.html`). `meshom ui --help` lists the options; `python -m ui.web.server` takes the same
ones plus `--mesh` (mesh shown first) and `--run-cmd`. Without the console script (or in a second checkout, when the
editable install points to another one): `PYTHONPATH=$PWD python -m meshom.cli ui ...`. Python: the standard-library HTTP
server, numpy, scipy, numba, matplotlib, pillow, pyshp, tomli_w (the `meshom-ui` env, or `meshom`, which has pyshp
and tomli_w now). The page is plain HTML + ES modules (no framework, no build step).

**Access.** Every start makes a random token (`--token T` sets one, `--no-token` switches it off on a loopback
host). The first page load carries it in the URL; the server answers with a cookie (`meshom_token_<port>`), which the
browser sends with every `/api/...` call; a request without the right token (query, cookie or `X-Meshom-Token`
header) gets 403 and a one-line page. Only `/static/*` is public. `--host` other than the loopback (a compute node's
hostname) listens on the cluster-internal interface: it needs the token (`--no-token` is refused, as is `--host
0.0.0.0` without a token) and is still reachable only from inside the cluster; never give a public address, and
never a non-loopback host on a login node. With `--allow-submit` whoever holds the URL submits jobs as you.

### Three ways to run

1. **Local:** in the repository, `meshom ui`; open the printed `http://127.0.0.1:8765/?token=...`. The browser
   draws on the GPU.
2. **Levante login node + one ssh hop** (the server is one process: h preview 0.5 s, a mesh payload < 0.1 s; fine
   for light use; stop it with Ctrl-C when done):

       # on the login node (hostname tells which one, e.g. levante1)
       cd <your clone of meshom>                    # with the meshom environment active
       python -m meshom.cli ui --port 8765 [--allow-submit]
       # on your own computer, to the login node the server runs on
       ssh -N -L 8765:127.0.0.1:8765 <user>@levante1.dkrz.de
       # open the printed http://127.0.0.1:8765/?token=...

3. **Interactive job + `--host $(hostname)` + one ssh hop** (for the gradient-limited preview of fine configs: the
   example config builds a 12 M node limiter grid in 24–31 s and ~1.2 GB; for the 10⁷-element reference meshes):

       # on the login node
       salloc -p interactive -A <project> -c 4 --mem=7G -t 04:00:00
       # in the job's shell: `hostname` must print the compute node (e.g. l40051); if it prints a login node,
       # `srun --pty bash` first
       cd <your clone of meshom>
       python -m meshom.cli ui --host $(hostname) --port 8765
       # it prints the URL and the tunnel; on your own computer, through the login node only:
       ssh -N -L 8765:l40051:8765 <user>@levante.dkrz.de
       # open http://127.0.0.1:8765/?token=...   (the token printed by the server)

   The tunnel's far end is the node's cluster-internal name, resolved by the login node, so no ssh login on the
   compute node is needed (the two-hop `ssh -J` tunnel fails with "Too many authentication failures"). Meshes are
   still built by batch jobs, never by the server.

- **Action log.** The server writes `.tmp/ui/logs/web-<date>-<time>-<host>.jsonl` (`--log FILE` names another; the
  path is printed at start), JSON lines: every API request when it starts (`req`) and ends (`done`: status, ms), and
  the page's events (`static/log.js`): clicks, form choices, messages and errors with stacks, map clicks, view changes
  (at most one a second), server calls slower than 1 s or failed, main-thread stalls over 1 s, and every 5 s a `beat`
  (animation frames, map renders, view changes, wheel and pointer input on the map, the open calls, the busy box, the
  controller, the layers, a lost graphics context). A frozen map shows as input without renders or view changes. The
  token is never written.
- **One schema with the core.** The config is `meshom.config.Config` (`docs/config.md`); `ui/common/scenario.py`
  only converts it to and from the page's JSON and evaluates it with the core: the layers with
  `meshom.run.layer_fields`, the gradient-limited h with `meshom.run.size_field` (the batch run's h). "Save TOML"
  writes `meshom.config.dumps`; `meshom run` reads the file as it is. The example `ui/common/scenario.toml` is
  never overwritten: a page started on it saves to `.tmp/ui/config.toml` (the status line names the file). `docs/config.md`, "What the UI writes", lists
  every key the page writes.
- **Tools** (the map bar's button, every tab; a window at the right edge): a ruler - "Draw a line", clicks on the map,
  Finish or Enter - with each leg and the total along great circles, and the profile along that line (600 points):
  h of the page's config (the run's h when Size h shows it) and the bottom of `[bathymetry] source`, else GEBCO 2024
  (`../bathymetry`), as two charts on one distance axis; hovering shows the values and the point on the map; "Download
  CSV" (U31). With a mesh open, the bottom chart also shows the nodes of the elements the line crosses as dots at their
  mesh depth, the hover names the nearest node (mesh depth, the bottom grid at the node, its offset from the line),
  the nodes are dotted on the map; "Nodes CSV". "Mesh depth: nodes in an area": select the mesh's nodes with a box or a
  polygon on the map and see their count, mesh depth (min, median, mean, max), the bottom grid at them and how many sit
  at the run's floor (U34); then "set to" / "at least" / "at most" D m and "Apply to these N nodes": an edit of that
  mesh in `<run>/<mesh>.depth_edits.json` (the core's C2), shown at once, written into the model files at the next
  Write model files (a depth below the run's floor gives the floor); the mesh's edits listed with remove and "Undo the
  last edit"; "Save as a config rule" adds the area and rule to the page's config as `[[bathymetry.regions]]` (U35).
  Reading GEBCO needs `netcdf4` in the server's Python (`mamba install -y netcdf4`).
- **The depth of a mesh** (colour "depth", the profile, the node tool): its last write's `depth_base.npy` (the depth
  before the node edits; an older write: depth.out for fesom-c, aux3d.out for fesom2 / fesom3), else the run's
  depth.npy (its final mesh), with the mesh's current node edits applied as the next write will (U34, U35); after a
  write from the page the mesh shown takes the new depth. A hand edit saved as edited-NN carries the depth edits.
- **Depth rules by area** (Coast, under [bathymetry]): the `[[bathymetry.regions]]` list - name, a box or a polygon
  drawn on the map (outlined in the Coast tab), set to / at least / at most m, fly to, remove - saved with the config;
  they reach the runs made from it. The Write window shows the rules and node edits a write applied, and warns when
  the mesh's edits are newer than its last write (U35).
- **Which runs are listed:** every folder given with `--runs` and every folder added in the page (Runs, "Add
  folder…": a browser of the server's folders, starting from the run roots, `.tmp`, `../meshom/.tmp` and
  `../meshom/.worktrees`; added folders are remembered in the browser) lists the run folders right in it; its
  "subfolders" tick (off by default) lists those at any depth below it too (decision U29). The table's "folder" column says where each run sits. A large tree's first search runs in the
  background ("searching…" in the window; `../meshom` whole: 280 s the first time, 0.6 s after); the page then fetches
  the list again only when it changed.
- **Runs (panel 7):** "Run" writes `<runs>/<stamp>-<method>/config.toml` (the page's config, `[mesh] out` = that
  folder, `[coast] source` absolute) and `run.sbatch` (account ab0995;
  partition `shared` by default, CPUs, memory and time from the panel; `--output` into the folder), whose command is
  the core's: `PYTHONPATH=<repo> <python> -m meshom.cli run <config> --out <dir>`. Only a server started with
  `--allow-submit` calls `sbatch`; otherwise the folder, the script and the command are shown (dry run). The run list
  reads what `meshom run` writes (`config.resolved.toml`, `stages.json`, `quality.json` with its raw / smooth records,
  `raw.npz`, `smooth.npz`; the line of `meshom runs`), the state from squeue, the log tail; "raw" / "smooth" load a
  mesh. Default run roots: `.tmp/ui/runs` (new runs), `.tmp/runs` (`meshom run` by hand), `.tmp/compete`.
  End to end: the closed-sphere config of `tests/test_run.py` posted to `/api/run` with `--allow-submit` (partition
  `shared`, 4 CPUs) → `meshom run` writes its folder → the mesh view loads it with its quality.
- **Reference meshes (panel 7, below the runs):** the meshes other models built, one folder each with `mesh.npz`
  in meshom's npz layout (the layout of a run's `raw.npz`: points, tris, quads, node_flag), with `quality.json`
  (`{"raw": record}`: compute_quality, irregular counts, gates, no h) and `overlay.npz` when it has them, under
  `../curvilinear_grids/` (23 grids: NEMO, MITgcm, MOM6, POP, MPIOM, HYCOM, ROMS, Delft3D) and
  `../triangle_meshes/` (FESOM2, fesom2_mesh_gen, MPAS-Ocean, ICON-O, ADCIRC); `--refs DIR` (repeatable) replaces
  the two. "load" opens one like a run (overview raster, zoom windows, irregular nodes, histograms with the
  quality.json numbers, the worst elements). eORCA12 (8.2 M quads) opens in 2.0–2.6 s (mesh.npz + overlay.npz
  read), its overview raster takes 0.6 s and a 450 km zoom window (4 486 cells) 0.5–0.7 s; use the overview and
  zoom windows, not "Full", on the 10⁷-element meshes (eORCA12, HYCOM, FESOM2 ng5).
- **Demo mesh** (10⁶ elements, for the mesh view): `scripts/on_node.sh -p interactive -c 8 -m 15G -t 00:40:00 --
  env PYTHONPATH=$PWD python -m ui.common.prepare_mesh --scale 0.45` (177 s) → `.tmp/ui/mesh/{mesh,overlay}.npz`. A
  run folder without `overlay.npz` gets its overlay computed by the server on first load and cached in
  `.tmp/ui/cache/` (4 s for 10⁶ elements); its |ln s/h| per edge uses the run's h (`config.resolved.toml` through
  `meshom.run.size_field`, or the compete case in the folder name).
- **Headless measurements and screenshots:** `scripts/on_node.sh -p interactive -c 16 -m 28G -t 00:50:00 -- env -u
  DISPLAY PYTHONPATH=$PWD python -m ui.web.measure [--quick]` (Chromium with SwiftShader; DISPLAY must be unset) →
  `.tmp/ui/measure_web_v2.json`, `docs/figures/ui/web-v2/*.png`.
- **Tests:** `PYTHONPATH=$PWD python -m pytest tests/test_ui_web.py tests/test_ui_common.py` (on a node).

## Panels

The tabs follow the work: **Set up** (Size h, Coast, Method, Blocks) → **Run** (Runs) → **Inspect** (Mesh, Reference
meshes, Compare); the map views (Globe, Flat, North, South) sit below them. Over the map: messages and errors (top), "Computing
… n s (last time m s)" for any server call over 0.5 s, lon / lat / h / the layer that sets h under the cursor (top
left), the legend in words (bottom left), a scale bar (bottom right). Developer numbers (timings, payload sizes, paths,
the batch template, fps) appear with "details for developers" in the footer.

**Floating windows** (`static/floatwin.js`, `static/sorttable.js`): "Open in a window" in Runs, Reference meshes,
Compare and above Mesh's worst elements opens the same data in a window over the map; the panel keeps its compact
list. A window moves by its title bar, resizes at its corner, comes to the front on a click; several can be open;
Escape closes the front one (a shape being drawn on the map is cancelled first); position and size are remembered per
browser (localStorage `meshom.windows`). Their tables sort on a header click (again: descending; numbers as numbers,
'–' last) and filter by the text in the box above them. Runs: run, state, nodes, gates passed, h (the run config's
finest–base h after `[mesh] scale`, sorted by the base; `h_km`, `scale` in `/api/runs`), date (the last quality
record's, else the folder's), with the mesh and "+ compare" buttons; the window lists every run (the panel the newest
60) and refreshes with the panel every 10 s. Reference meshes: family, resolution (the source's words; sorted by the
typical element size `res_km` of `/api/refs`: median √area of the quality record, the equilateral edge for triangle
meshes), nodes, triangle share, MB, load / + compare. Worst elements: a click on a row flies there, as in the panel.

| panel | what it does | stubbed / pending |
|---|---|---|
| Size h | h is drawn here (and in Coast, Method) even with a mesh open; "the open mesh over h (faint)" lays that mesh over it. The layers apply in order (`meshom.size.layered`): a zone's mode refine / set h / smooth h (`mode`, `smooth_m`), a coast layer's coasts all / continents / islands (`coasts`, `island_max_km2`; "Split into continents + islands"); "h variants": the page's [size] saved as `h-NAME.toml` next to the config and loaded back, Save refers to an unchanged variant with `[size] from` (U30). `[[size.layers]]` as a list (▸ opens a layer's form): add (constant, latitude law, distance to coast, zone), remove, reorder (colours only: h is the minimum), enable/disable (`enabled = false`); zones drawn by clicks (great-circle edges), corners dragged; a new zone's transition from grad_max, (base h − h inside) / grad_max, until a width is typed; map: h, "which layer sets h", ρ_c h² (1°, 0.5° or 2°); node estimate ∫h⁻² dA; the steepest change of h, in the layers and in the run, with Fly to; **the run's h by default**: the layers' h comes first, the gradient-limited h (`meshom.run.size_field`: clip, the core's limiter, scale) follows in the background; "Look closer at a region": zoom windows (box or polygon, own step, may cross 180°), "Make a refinement zone" appends a zone layer | — |
| Coast | source: none (the closed sphere), GSHHG 2.3.7 in `../coastlines` (every resolution and level, labelled; L1 usable) and every shapefile next to the configured one and in `../fesom2_mesh_gen/**/data`; Antarctica: ice front (L5, default) / grounding line (L6) / none (`[coast] antarctica`); drop islands below an area; keep seas above an area (`[coast] keep_basins_km2`; empty: the largest basin only); the coast drawn at the coastal h; **Bathymetry**: the `[bathymetry]` keys as plain inputs (source, variable, min_depth_m, depth_at, levels) | the core's coast distance (`meshom.coast`) keeps Natural Earth's land borders as coast |
| Method | `meshom.methods.METHODS` with each module's `DEFAULTS` as a form (numbers, bools, strings, JSON for None/lists/dicts); changed options go to `[mesh] opts`, kept per method while the page is open | a variant's own overrides are not shown yet (the form says so; the core will declare them in `meshom.methods.OVERRIDES`) |
| Blocks | the hand `[[blocks]]`: "Draw a block" (four corners clicked on the map, made counter-clockwise), numbered corners dragged on the map, a form per block (the sides' kinds or the core's rule, rows / cols / h) and the TOML text; the m14 layout of a loaded run (`m14.json`): seeds, sizes, ∫ρ_c, fly-to, block outlines and corners (also over the mesh: Mesh, "m14 blocks over the mesh") | coast and offset sides drawn straight (the run lays them along the coast) |
| Runs | the run button writes the page's config (saved or not) into a new run folder with `run.sbatch` (submitted only with `--allow-submit`; the button says which); the run list (state, stages with raw / smooth, the quality.json lines, log); "raw" / "smooth" open the mesh in Mesh; "open config" makes a run's config the page's (the page reloads; saves then go to `.tmp/ui/config.toml`) | — |
| Mesh | the open mesh named on top; overview raster (worst value per pixel; pixels inside large elements filled), the 2000 worst (by the colour field), a zoom window, all elements (a newly opened mesh above 2 M elements opens as the overview); opening a mesh frames it whole; colour by max angle / gate, size ratio, **cell size** (the mean edge of each element in km, log scale over the 1st-99th percentile rounded to two significant digits, viridis; the overview shows the largest element per pixel; U33), tri/quad, **depth** (only when the mesh's folder has the core's `depth.npy`: per element the mean of its nodes' depths, YlGnBu from 0 m to the 99th percentile rounded up to 500 m; the overview shows the deepest element per pixel; otherwise the option is greyed out with the reason); **edges** and **irregular-node dots** only when a typical cell covers ≥ 6 px (dots off by default on triangle meshes, n5 / n7 there); the coast drawn is the one the mesh was built with (its run's config); **the gates as a table** (what, where, value, limit, passed; failed first) from `check_gates` on the mesh's quality.json; a mesh repaired with `meshom repair` (`repaired-NN.npz` / `.json`, listed in Runs) opens with a before / after box (the zone and the whole mesh, worse numbers marked; Show before / Show after; the zone outlined); "Remesh a zone" is shown greyed out (planned, U27); **Edit by hand** (an option, off by default, zoom window only; U16): drag a node (its elements follow, their max angle and size ratio recomputed in the page on the sphere), add a node by a click inside an element (split into 3 / 4 triangles), undo, "Save as edited-NN" (the server replays the edits on the full mesh and only writes `edited-NN.npz` / `.json` next to it, never over raw / smooth / earlier files; listed in Runs like raw / smooth; reference meshes are not edited); coast / boundary nodes and the window's rim stay; "remesh with these edits" greyed out (planned); **Write model files…** (the Mesh panel for the mesh shown, the hand editor for the last saved edited-NN; only for a mesh of a `meshom run` folder: its config.resolved.toml gives the format, the bathymetry and the gates; U32): a window picks the run's mesh (raw, smooth, optimized, repaired-NN, edited-NN) and the format (the run's by default), its Write button runs the core's `meshom write` on the server (meshom.run.write_mesh: <run>/fesom-<mesh>/, the depth recomputed from the run's [bathymetry], the mesh scored with the run's gates; one write at a time, the window shows the seconds and the stage write_mesh is in; the first write of a run computes its size field and depth, later ones read them from <run>/cache/, ~30 s after a small edit: after a change of the coast file without a config change delete that cache/ by hand, O33), with fesom_ini inside the write when ticked ([mesh] fesom3_ini, format fesom3; its log in <run>/fesom_ini-<mesh>/, its record in the write's); the window shows the folder, the format's checks (the STOP items in words first, every item folded), the gates (value, limit), the straits (connected, cells across of the target, the narrowest width and depth; a row flies there) and the narrow passages below [coast] passage_min_cells as a warning list (where, cells across, h, kind, the strait near; a row flies there; folded above 10); a mesh written before shows its record from the folder's quality.json; worst-element list with fly-to and a ring; histograms (quad and triangle θmax with their gate lines, size ratio with a note on what is counted, \|ln s/h\| per edge) | histogram bins come from the overlay arrays (quality.json holds percentiles, not bins) |
| Reference meshes | the meshes other models built (curvilinear grids, triangle meshes; `mesh.npz` + `quality.json`); "load" opens one in Mesh in place of the current mesh; "+ compare" puts it into Compare | — |
| Compare | 2–4 meshes picked with "+ compare" (Runs: raw / smooth / repaired-NN; Reference meshes), a column each: nodes, quads, triangles, triangle share, n3 and n5 per 1000 nodes, largest quad angle (max, p99.9), largest triangle angle (p99, max), size jump p99, \|ln(edge/h)\| p99 (– when unknown), the gates every mesh has (passed of them, the failed ones by name); best value per row in bold green (counts not ranked); "show" opens a column in Mesh. `POST /api/compare` reads only the quality records (quality.json raw / smooth, repaired-NN.json "after"), no mesh | no maps side by side; the picks are kept for the page session only |

Config round trip: the page edits only the keys the user touches; the server converts the page's JSON back with the
core's `from_dict` (JavaScript's integers become floats where the schema has floats; `[mesh] opts` and `[gates]` are
typed by the method's `DEFAULTS` and `DEFAULT_GATES`; "inf" becomes inf), so load → edit → save → load gives the same
values and the same TOML text (tested).

## Payloads

- **mesh.bin** (`ui/web/payload.py`): indexed, shared vertices — header JSON, float32 node lon/lat, uint32 triangles
  and quads, uint32 element ids (subsets only), float32 max angle and size ratio per element. The page expands it to
  per-element vertices (flat colours) and hands deck.gl the triangle indices, so nothing is triangulated in the
  browser; on the flat map elements are unwrapped and the ones across ±180° copied (820 for the demo mesh). When the
  flat view reaches past ±180° every layer is drawn again shifted by ±360° (`modelMatrix`), so the map is continuous.
- A colour change swaps only the colour array of the layer (`updateTriggers`); a new layer or data object would make
  deck.gl rebuild and upload 10⁶ polygons.

## Timings (10⁶-element mesh: 981 286 nodes, 978 932 quads, 1 718 triangles)

Ranges are over three measurement runs (`ui/web/measure.py`, which writes the raw numbers to `.tmp/ui/measure_*.json`).

| step | size | server | in the browser (SwiftShader) |
|---|---|---|---|
| h preview, global 0.5° (259 200 points) | 0.38 MB | 0.41 s (first call 1.1 s: coastline load) | 0.51–0.66 s round trip |
| h preview, global 1° | 0.12 MB | 0.14 s | — |
| gradient-limited h preview, global 0.5° (example config: zone 8 km → the limiter's grid at its 12 M node cap, 7.1 km) | 0.38 MB | first 23.8–31.4 s (the limiter, on two nodes), then 0.49–0.53 s (cached per config hash; unlimited 0.46–0.50 s) | — |
| gradient-limited zoom window 30° × 17° at 0.05° (cached limiter) | — | 0.48 s | — |
| ρ_c h², 2° / 1° / 0.5° (10 k / 41 k / 165 k points) | 29 / 97 / 327 KB | 0.13 / 0.31–0.52 / 1.5–1.9 s | 1° 0.53–0.60 s, 0.5° 2.0–2.2 s |
| zoom window 30° × 17° at 0.05° / 0.02° (0.2 M / 1.3 M points) | 0.13 / 0.46 MB | 0.34 / 2.05 s | 46° × 11° box at 0.05°: 0.36–0.41 s |
| first mesh request (10⁶ mesh + overlay read from disk) | — | 0.35–0.52 s | — |
| **all 980 650 elements, mesh.bin** | **31.4 MB** (prototype per-element polys.bin: 58.8 MB) | 0.043–0.046 s | fetch 44–107 ms, decode < 1 ms, expand 70–153 ms, first frame 0.12–0.23 s, total 0.43–1.2 s (prototype: fetch 0.5–0.7 s, first frame 0.6–2.5 s on the globe, 3.8–4.4 s on the maps) |
| same, flat map: unwrap + 820 copies across ±180° / north map: projection | — | — | geometry 110–165 ms / 175–200 ms; view switch 0.3–1.5 s typical, SwiftShader outliers 4–14 s and once 50 s (buffer churn) |
| colour field change on all elements | — | — | 70–122 ms (only the colour array is swapped) |
| frames while the globe turns, all elements | — | — | median 16.6 ms in one run, 1.4 s in another (SwiftShader is erratic); flat map 16 ms |
| zoom window 700 km (2 600 elements, edges) | 90 KB | 0.02–0.03 s | fetch 29 ms, first frame 18–30 ms; 57 copies at 180° on the flat map |
| the 2000 worst elements / overview PNG / irregular nodes (4 489) | 96 / 83 / 40 KB | 0.03–0.04 / 0.03–0.05 / 0.001 s | worst: first frame 40 ms |
| histograms / worst list (40) / methods (first call imports meshom.methods) | 1.9 / 4.6 / 15 KB | 0.08 / 0.03 / 0.6–2.1 s | — |
| page load (config, coastline 1.4 MB, first preview, methods) | — | — | 1.6–3.3 s |

Server: one Python process on a Levante compute node (AMD EPYC 7763), timed with urllib. Browser: headless Chromium
156 with SwiftShader (WebGL2 on the CPU, 16 cores, no GPU), window 1400 × 880 — frame times are upper bounds for a
laptop GPU; the page shows deck.gl's fps counter to read them off on your own computer.
