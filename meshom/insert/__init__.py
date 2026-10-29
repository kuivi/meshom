"""Curvilinear inserts fitted into a finished mesh: step 1, a strait as one structured block (decisions B158-B163);
step 2, banks on the real coast, a size floor, bays as blocks of their own, straighter joins (B164-B169).

    python -m meshom.insert --run RUN_DIR --insert 'strait=Gibraltar; cells=10' [--name N] [--out DIR]
    python -m meshom.insert --run RUN_DIR --config FILE.toml        (every [[inserts]] entry of FILE)

In a region of interest a logically rectangular quad grid that zooms in (a "banana" grid) replaces the host's
cells; on the two joins its nodes are the host's nodes, fixed, so the result conforms without hanging nodes. Any host
works (ORCA, meshom's quad-dominant meshes, triangles). Steps (meshom.insert.core):

1. region (meshom.insert.region): the strait's width profile along its axis, the neck, the size target; candidate
   joins grow outward from the neck until the three rules hold (corner, funnel, room), or the region is refused
   with the failing rule and what would fix it;
2. joins: ordered chains of host edges (tier 1 along host grid lines, tier 2 otherwise);
3. block (meshom.insert.block): Winslow harmonic map in a local stereographic chart, joins fixed, banks sliding along
   the real coast (meshom.insert.realcoast: the run's coast smoothed at the insert's cell size, bays bridged; step
   1's capped host coast with coast = "host"); rows along the strait at equal steps of the integral of ds / h, never
   below the size floor h_min_m; Y-splits (staggered) when the counts change; bays as blocks of their own
   (meshom.insert.network);
4. merge by shared node ids (meshom.insert.assemble); the join ring cleaned by the mesh-fixing pipeline
   (meshom.meshfix.pipeline.run_stages) with the block's core fixed (cut out of the case: its outline is artificial
   boundary), against the run's h composed with the insert's h layer;
5. scores by zone (core, join ring, outside; meshom.insert.measure), the splice into the whole mesh, depth by the
   run's bathymetry rules and model files when the run wrote them.

Later steps: cape junctions (L-shaped joins), coastal strips and the Gulf Stream corridor, inserts reserved inside the
builder.
"""
