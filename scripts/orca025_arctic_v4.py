"""The ORCA025 + Arctic config, version 4 (B113): the Arctic as ORCA itself, refined 3 x 3. ORCA025 per column south
of LAT_ORCA (B100's staircase, with B100's dip to 46 N between 62 E and 91 E south of NEMO's Siberian grid pole, over
land), ORCA12 (= ORCA025 x 3 on the same map) per column from the fold down while the T point lies north of LAT_FINE
(B108), FESOM3's rules on both ORCA parts (B94); between them a fill by m14_blocks_rep without blocks whose m02 field
is ORCA025's own directions (B109), grown from the fine edge only (B112); h = ORCA025's cell size scaled by 1/3 north of LAT_FINE and by 3^-(1 - t)
between the two lines (B110); Penny Strait at least 150 m (O35).

    python scripts/orca025_arctic_v4.py RUN_OUT CONFIG [LAT_ORCA [LAT_FINE [LEVELS_DIR [REF_DIR]]]]
"""
import sys

import numpy as np

out, path = sys.argv[1], sys.argv[2]
lat_orca = float(sys.argv[3]) if len(sys.argv) > 3 else 56.0
lat_fine = float(sys.argv[4]) if len(sys.argv) > 4 else 62.0
levels = sys.argv[5] if len(sys.argv) > 5 else ".tmp/orca_levels"
ref_dir = sys.argv[6] if len(sys.argv) > 6 else ".tmp/orca_refs"


def lat_line(a, b, la, step=1.0):
    lo = np.linspace(a, b, int(round((b - a) / step)) + 1)
    return np.c_[lo, np.full(len(lo), la)]


def dedup(p):
    return p[np.r_[True, np.abs(np.diff(p, axis=0)).max(axis=1) > 1e-9]]


def dense(pts, step=1.0):
    p = np.array(pts, float)
    o = []
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.abs(b - a).max() / step)))
        o.append(np.c_[np.linspace(a[0], b[0], n, endpoint=False), np.linspace(a[1], b[1], n, endpoint=False)])
    o.append(p[-1:])
    return np.vstack(o)


EDGE = dedup(np.vstack([lat_line(-180, 62, lat_orca), dense([(62, lat_orca), (66, 46), (86, 46), (91, lat_orca)]),
                        lat_line(91, 180, lat_orca)]))
REGION = np.vstack([[[-181.0, -91.0], [-181.0, lat_orca]], EDGE, [[181.0, lat_orca], [181.0, -91.0]]])
OUTER = EDGE[:-1]                                    # around the pole: 180 E is 180 W
FINE = lat_line(-180, 180, lat_fine)[:-1]            # 1 deg steps: the great-circle chords bulge < 0.1 km


def fmt(p):
    return "[" + ", ".join(f"[{x:.4f}, {y:.4f}]" for x, y in p) + "]"


R = "/work/ab0995/a270126/projects/mesh_generation"
txt = f'''# ORCA025 + the Arctic as ORCA itself, refined 3 x 3 (v4, B113): NEMO's ORCA025 per
# column south of {lat_orca:g} N (B100's staircase, with B100's dip to 46 N over Kazakhstan south of NEMO's Siberian grid
# pole), ORCA12 = ORCA025 x 3 on the same map per column from the fold down while the T point lies north of {lat_fine:g} N
# (B108), FESOM3's rules on both (B94); between them m14_blocks_rep without blocks, its m02 fill along ORCA025's own
# directions (B109), cut by the coast. h: ORCA025's own cell size (B101), a third of it north of {lat_fine:g} N, scaled by
# 3^-(1 - t) between the two lines (B110). Penny Strait at least 150 m (O35).
# Written by scripts/orca025_arctic_v4.py.
[mesh]
method = "m16_orca"
out = "{out}"
format = "fesom3"
fesom3_ini = "{R}/fesom_F_build/build_intel_dp/bin/fesom_ini"

[mesh.opts]
orca = "ORCA025"
mask = "fesom3"
levels_dir = "{levels}"
cap_method = "m14_blocks_rep"
ref_dir = "{ref_dir}"
gebco = "{R}/bathymetry/GEBCO_2024.nc"
fine_orca = "ORCA12"
fine_lat = {lat_fine}
region = {fmt(REGION)}

[mesh.opts.cap_opts]
# B109: no block, no strait block: the fill between two lines of the same ORCA map takes ORCA025's directions
auto_blocks = false
strait_blocks = false

[mesh.opts.cap_opts.fill]
# B109: m02's field = ORCA025's own i/j directions; B112: one front, from the fine edge (the cap's loop 1): the rows
# merge columns as h grows south and stop at the coarse edge; two fronts sheared the band where they met
frame = "ORCA025"
seed_loops = [1]

[coast]
source = "{R}/coastlines/GSHHS_shp/i/GSHHS_i_L1.shp"
antarctica = "ice_front"
min_area_km2 = 0.0
treatment = "cut"
passage_min_cells = 2

[bathymetry]
source = "{R}/bathymetry/GEBCO_2024.nc"
variable = "elevation"
min_depth_m = 10.0
depth_at = "cells_deepest"
levels = "core2"
min_layers = 4
smooth_passes = 4

[[bathymetry.regions]]
# the strait between Bathurst Island and Devon Island's Grinnell Peninsula: 80 m on v2 against a sill of ~157 m
name = "Penny Strait"
box = [-97.5, 75.8, -94.5, 77.0]
min_depth_m = 150.0

[size]
grad_max = 0.1
h_max_m = 40000.0

[[size.layers]]
kind = "orca"
name = "ORCA025 cell size"
orca = "ORCA025"
h_min_m = 1000.0

[[size.layers]]
kind = "zone"
mode = "blend"
name = "ORCA12 = ORCA025 / 3 north of {lat_fine:g} N, scaled down from ORCA025's cells at {lat_orca:g} N"
h_factor = {1.0 / 3.0!r}
transition_m = 0.0
polygon = {fmt(FINE)}
outer = {fmt(OUTER)}

[run]
timeout = 14400.0
figures = false
'''
open(path, "w").write(txt)
print(path, len(REGION), len(OUTER), len(FINE))
