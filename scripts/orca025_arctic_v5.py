"""The ORCA025 + Arctic config, version 5 (B117): ORCA025 up to its last F row lying wholly south of LAT_ORCA,
ORCA12 (= ORCA025 x 3 on the same map) from its last F row lying wholly south of LAT_FINE to the fold (every
column), FESOM3's rules on both ORCA parts (B94); between the two plain rows the ladder (B114-B116: m16_orca
`transition = "ladder"`, a structured strip in ORCA's index space), cut by the coast; h = ORCA025's cell size scaled
by 3^-(1 - t) between the two rows (B110, the rows as the blend's lines); the Danish straits widened (they lie in the
ladder: the coast cut closes them, B117); Penny Strait at least 150 m in a box that
also holds Belcher Channel and the north of Queens Channel (O35, B117). Run it on a compute node (the two rows are
evaluated on the ORCA12 map: ~4320 meridians).

    python scripts/orca025_arctic_v5.py RUN_OUT CONFIG [LAT_ORCA [LAT_FINE [LEVELS_DIR [REF_DIR]]]]
"""
import sys

import numpy as np

from meshom.orca.ladder import preset_edge_rows, preset_f_rows

out, path = sys.argv[1], sys.argv[2]
lat_orca = float(sys.argv[3]) if len(sys.argv) > 3 else 56.0
lat_fine = float(sys.argv[4]) if len(sys.argv) > 4 else 62.0
levels = sys.argv[5] if len(sys.argv) > 5 else ".tmp/orca_levels"
ref_dir = sys.argv[6] if len(sys.argv) > 6 else ".tmp/orca_refs"

r_c, r_f = preset_edge_rows("ORCA025", "ORCA12", lat_orca, lat_fine, (range(700, 860), range(2250, 2560)))
lo_c, la_c = preset_f_rows("ORCA025", [r_c])
lo_f, la_f = preset_f_rows("ORCA12", [r_f])
OUTER = np.c_[lo_c[0], la_c[0]]                      # the coarse edge row, eastward (1440 points)
FINE = np.c_[lo_f[0], la_f[0]][::3]                  # the fine edge row, every third point (1440)
PENNY = [-97.5, 75.5, -93.5, 77.35]                  # B117: Penny Strait, Belcher Channel, the north of Queens Channel


def danish():
    from meshom.straits import builtin

    out = []
    for d in builtin("fesom2_mesh_gen"):
        if d["name"] in ("Oresund", "Great_Belt", "Little_Belt"):
            w = f"width_m = {float(d['width_m'])}\n" if d.get("width_m") else ""
            out.append(f'[[straits]]\nname = "{d["name"]}"\npath = {fmt(d["path"])}\n{w}profile = "widen"\n')
    assert len(out) == 3, out
    return "\n".join(out)


def fmt(p):
    return "[" + ", ".join(f"[{x:.5f}, {y:.5f}]" for x, y in p) + "]"


R = "/work/ab0995/a270126/projects/mesh_generation"
DANISH = danish()
txt = f'''# ORCA025 + the Arctic as ORCA12, joined by the ladder (v5, B117): NEMO's ORCA025 up to
# its F row {r_c} (the last lying wholly south of {lat_orca:g} N: {la_c.min():.2f}-{la_c.max():.2f} N), ORCA12 = ORCA025 x 3 on the same
# map from its F row {r_f} (the last lying wholly south of {lat_fine:g} N: {la_f.min():.2f}-{la_f.max():.2f} N) to the fold, FESOM3's rules on
# both (B94); between the two plain rows meshom.orca.ladder's strip in ORCA's index space (B114-B116), cut by the
# coast. h: ORCA025's own cell size (B101) x 3^-(1 - t) between the two rows (B110). Penny Strait at least 150 m (O35).
# Written by scripts/orca025_arctic_v5.py.
[mesh]
method = "m16_orca"
out = "{out}"
format = "fesom3"
fesom3_ini = "{R}/fesom_F_build/build_intel_dp/bin/fesom_ini"

[mesh.opts]
orca = "ORCA025"
mask = "fesom3"
levels_dir = "{levels}"
ref_dir = "{ref_dir}"
gebco = "{R}/bathymetry/GEBCO_2024.nc"
fine_orca = "ORCA12"
transition = "ladder"
ladder_lat = [{lat_orca}, {lat_fine}]

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
# Penny Strait (Bathurst Island - Grinnell Peninsula, ~157 m) with Belcher Channel (-94.1 E, 77.17 N) and the north
# of Queens Channel (75.64 N), where v3 / v4 had their sills outside the old box [-97.5, 75.8, -94.5, 77.0]
name = "Penny Strait"
box = {PENNY}
min_depth_m = 150.0

# B117: the Danish straits lie in the ladder (v4: in ORCA025, NEMO's mask): the coast cut closed them and the Baltic
# (ORCA12 north of ~58 N) was dropped as a sea cut off at the joint; widened corridors keep them open (paths and
# widths of meshom's built-in list fesom2_mesh_gen)
{DANISH}
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
name = "ORCA12 = ORCA025 / 3 north of ORCA12's F row {r_f}, scaled down from ORCA025's cells at its F row {r_c}"
h_factor = {1.0 / 3.0!r}
transition_m = 0.0
polygon = {fmt(FINE)}
outer = {fmt(OUTER)}

[run]
timeout = 14400.0
figures = false
'''
open(path, "w").write(txt)
print(path, r_c, r_f, len(OUTER), len(FINE))
