# meshom

A mesh generator on the sphere for the ocean models **FESOM3** (global meshes: the closed sphere) and **FESOM-C**
(regional meshes with open boundaries); it also writes FESOM2 triangle meshes. meshom builds **mixed meshes**: mostly
near-rectangular quads, with triangles only where they help, in smooth size transitions and along complex coasts.

All resolution comes from **one size function h on the sphere**, and the smoothness of size transitions is the main
requirement. Sizes and element quality are measured on the sphere or in a local tangent plane, never in raw
longitude-latitude.

**AI agents: start at [`AGENTS.md`](AGENTS.md).** It holds the workflow, the rules and an index of short topic
guides, each read only when the task needs it.

An illustrated tour of what meshom builds, with figures and measured numbers: [`docs/overview.md`](docs/overview.md).

## How it works

One run builds one mesh from one TOML config (`meshom run CONFIG`). The stages:

1. **Coast**: land polygons (GSHHS, Natural Earth or a shapefile of your own), small islands dropped, the water
   connected to the world ocean kept; straits from a built-in list or found on the coast's medial axis are opened and
   resolved; narrow passages are widened to a minimum number of cells.
2. **Size function h**: layers applied in file order, most of them by their minimum (constant, latitude, distance to
   the coast, zones you draw, an ORCA grid's own cell size), then the strait zones, then a gradient limiter on the
   sphere (|∇h| ≤ `grad_max`).
3. **Mesh**: a method fills the water. The main ones are `m14_blocks` (structured quad blocks placed along the coast
   and in the open ocean, stitched by a frontal fill), `m02_linf_frontal` (frontal quads), `m15_tri` (triangles) and
   `m16_orca` (NEMO's tripolar ORCA grids as quad meshes, alone or joined to a refined region). `meshom compete`
   lists and scores them all.
4. **Clean-up and optimisation**: the land cut, the rules FESOM needs (no spurs, no pinched nodes), local repair of
   defect clusters, topology flips and merges, node smoothing within a size tolerance.
5. **Depth and output**: node depths from a bathymetry grid (e.g. GEBCO) with the coastal and strait depth rules, and
   the model files in FESOM-C, FESOM2 or FESOM3 format, with the quality record of every stage (`quality.json`,
   `stages.json`).

The config reference is [`docs/config.md`](docs/config.md); ready configs are in [`docs/examples/`](docs/examples)
(CORE2-like FESOM2 meshes with straits, mixed FESOM3 meshes, ORCA meshes with a refined Arctic). Every hard rule,
threshold and default is listed with its reason in [`docs/decisions.md`](docs/decisions.md); the methods are
described in [`docs/methods/`](docs/methods).

```
meshom run docs/examples/global_150km_m14.toml      # build a mesh: the run folder is .tmp/runs/<config name>
meshom runs                                         # one line per run: status, size, quality
meshom write RUN_DIR --format fesom3                # write a saved or hand-edited mesh as model files again
```

**Inserts** (optional, never made by `meshom run`): a finer structured patch can be fitted into any finished mesh
afterwards, in a strait, along a coast or along a current, with the host mesh round it kept or cleaned:
`python -m meshom.insert --run RUN_DIR --config FILE.toml`. The recipe and examples: `docs/examples/inserts.toml`.

On Levante, runs go to a compute node: `sbatch -A <project> scripts/run_example.sbatch CONFIG` (the log:
`meshom-run-<jobid>.log`).

## The web UI

`meshom ui` starts a small web server: you edit the config and see h on the globe, look at a finished mesh (worst
elements, irregular nodes, zoom windows), edit node depths and nodes, and write model files. The browser draws the
map; the server never builds a mesh (its "Run" writes an sbatch script). Details: [`ui/web/README.md`](ui/web/README.md).

**On your own computer:** `meshom ui` in the repository, then open the printed `http://127.0.0.1:8765/?token=...`.

**On Levante, viewed in the browser of your own computer:** the server runs as a Slurm job on a compute node, and an
ssh tunnel through the login node brings its port to your computer.

1. On Levante, in the repository, with the meshom environment active:

       sbatch -A <project> scripts/ui.sbatch [--runs DIR] [--refs DIR]

2. When the job runs, its log `meshom-ui-<jobid>.log` prints the address and the tunnel command, e.g.

       meshom UI (web): http://l40051.lvt.dkrz.de:8765/?token=3f9c...  config ...
         tunnel: ssh -N -L 8765:l40051.lvt.dkrz.de:8765 <user>@levante.dkrz.de   then open http://127.0.0.1:8765/?token=3f9c...

3. On your own computer, run that `ssh -N -L ...` line (with your Levante user name) and keep it open; open
   `http://127.0.0.1:8765/?token=...` in the browser.
4. When done: Ctrl-C in the tunnel's terminal, and `scancel <jobid>` on Levante.

The far end of the tunnel is the compute node's cluster-internal name, resolved by the login node, so no ssh login on
the compute node is needed. If port 8765 is taken on the node, pass another one: `sbatch ... scripts/ui.sbatch
--port 8790` (and use it in the tunnel). The token in the URL is the access key: whoever has the URL can use the
server.

## Data

No data is shipped. The configs read coastlines from GSHHG (www.soest.hawaii.edu/pwessel/gshhg/) or Natural Earth,
bathymetry from GEBCO (www.gebco.net), and, for the ORCA meshes, NEMO's grid files (www.nemo-ocean.eu; the Met Office
eORCA grids, the DRAKKAR eORCA05 mask). Each config names the files it reads; `docs/config.md` says which keys take
them.

## Credits

- The strait list built into meshom and the strait and depth rules follow D. Sein (AWI), fesom2_mesh_gen.
- The ORCA rules of `meshom.orca.fesom3` follow FESOM3's conversion of NEMO grids (github.com/FESOM/fesom3).
- `meshom/orca/generator.py` is a port of G. Madec's ORCA grid generator (NEMO; Madec and Imbard, 1996), see
  [`NOTICE`](NOTICE).

## Licence

GNU General Public License v3 or later ([`LICENSE`](LICENSE)); the origin of the ported ORCA generator is in
[`NOTICE`](NOTICE).
