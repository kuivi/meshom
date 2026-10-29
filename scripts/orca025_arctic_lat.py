"""The ORCA025 + Arctic config with latitude edges (v3, B107): ORCA025 per column south of LAT_ORCA (B100's staircase;
B100's dip to 46 N between 62 E and 91 E south of NEMO's Siberian grid pole, over land, unless DIP is 0), h = the
fine h north of LAT_FINE (a latitude circle, the zone's polygon and the m14 block's place_region, B106), the blend of
S37 between the two circles, and the depth rules by area of `[[bathymetry.regions]]` (O35): Penny Strait at least
150 m.

    python scripts/orca025_arctic_lat.py H_IN_M RUN_OUT CONFIG [LAT_ORCA [LAT_FINE [PASSAGE_MIN_CELLS [DEPTH_AT
        [LEVELS_DIR [REF_DIR [DIP]]]]]]]
"""
import sys

import numpy as np

h_in, out, path = float(sys.argv[1]), sys.argv[2], sys.argv[3]
lat_orca = float(sys.argv[4]) if len(sys.argv) > 4 else 56.0
lat_fine = float(sys.argv[5]) if len(sys.argv) > 5 else 62.0
passages = int(sys.argv[6]) if len(sys.argv) > 6 else 2
depth_at = sys.argv[7] if len(sys.argv) > 7 else "cells_deepest"
levels = sys.argv[8] if len(sys.argv) > 8 else ".tmp/orca_levels"
ref_dir = sys.argv[9] if len(sys.argv) > 9 else ".tmp/orca_refs"
dip = int(sys.argv[10]) if len(sys.argv) > 10 else 1


def lat_line(a, b, la, step=1.0):
    lo = np.linspace(a, b, int(round((b - a) / step)) + 1)
    return np.c_[lo, np.full(len(lo), la)]


def dedup(p):
    keep = np.r_[True, np.abs(np.diff(p, axis=0)).max(axis=1) > 1e-9]
    return p[keep]


def dense(pts, step=1.0):
    p = np.array(pts, float)
    out = []
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.abs(b - a).max() / step)))
        out.append(np.c_[np.linspace(a[0], b[0], n, endpoint=False), np.linspace(a[1], b[1], n, endpoint=False)])
    out.append(p[-1:])
    return np.vstack(out)


# ORCA's edge: the latitude circle LAT_ORCA from 180 W to 180 E, with B100's dip south of NEMO's Siberian grid pole
# (73 E 50 N) over Kazakhstan's land
if dip:
    EDGE = dedup(np.vstack([lat_line(-180, 62, lat_orca), dense([(62, lat_orca), (66, 46), (86, 46), (91, lat_orca)]),
                            lat_line(91, 180, lat_orca)]))
else:
    EDGE = lat_line(-180, 180, lat_orca)
REGION = np.vstack([[[-181.0, -91.0], [-181.0, lat_orca]], EDGE, [[181.0, lat_orca], [181.0, -91.0]]])
OUTER = EDGE[:-1]                                    # around the pole: 180 E is 180 W
FINE = lat_line(-180, 180, lat_fine)[:-1]            # 1 deg steps: the great-circle chords bulge < 0.1 km


def fmt(p):
    return "[" + ", ".join(f"[{x:.4f}, {y:.4f}]" for x, y in p) + "]"


R = "/work/ab0995/a270126/projects/mesh_generation"
txt = f'''# ORCA025 joined to a {h_in / 1e3:g} km quad Arctic with latitude edges (v3, B107): NEMO's ORCA025 per
# column south of {lat_orca:g} N (B100's staircase{", with B100's dip to 46 N over Kazakhstan south of NEMO's Siberian grid pole" if dip else ""}),
# FESOM3's rules on the ORCA cells (B94); north of it a cap by m14_blocks_rep against the staircase, cut by the coast,
# one block north of {lat_fine:g} N (B106). h: ORCA025's own cell size (B101), {h_in / 1e3:g} km north of {lat_fine:g} N, a blend
# between the two latitude circles (S37). Penny Strait at least 150 m (O35).
# Written by scripts/orca025_arctic_lat.py.
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
region = {fmt(REGION)}

[mesh.opts.cap_opts]
# B106: one block over the uniform Arctic (north of {lat_fine:g} N), none in the transition; the B pre-screen off
place_region = {fmt(FINE)}
max_blocks = 1
b_cap = 100.0

[coast]
source = "{R}/coastlines/GSHHS_shp/i/GSHHS_i_L1.shp"
antarctica = "ice_front"
min_area_km2 = 0.0
treatment = "cut"
passage_min_cells = {passages}

[bathymetry]
source = "{R}/bathymetry/GEBCO_2024.nc"
variable = "elevation"
min_depth_m = 10.0
depth_at = "{depth_at}"
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
name = "{h_in / 1e3:g} km north of {lat_fine:g} N, blended to ORCA's cells at {lat_orca:g} N"
h_m = {h_in}
transition_m = 0.0
polygon = {fmt(FINE)}
outer = {fmt(OUTER)}

[run]
timeout = 14400.0
figures = false
'''
open(path, "w").write(txt)
print(path, len(REGION), len(OUTER), len(FINE))
