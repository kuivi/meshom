"""The ORCA025 config with ORCA12 nested in a Gulf Stream box (B176): NEMO's ORCA025 outside the coarse box, ORCA12
(= ORCA025 x 3 on the same map) inside the fine box, FESOM3's rules on both ORCA parts (B94), between the two boxes
meshom.orca.nest's frame (B172-B175) cut by the coast; h = ORCA025's cell size scaled by 3^-(1 - t) across the frame
(B110, the two box edges as the blend's lines). With ARCTIC = 1 the nest is added to the ORCA025 + Arctic ORCA12
mesh joined by the ladder (B117). Run it on a compute node (the ORCA025 grid is built to place the box).

    python scripts/orca025_gs_nest.py RUN_OUT CONFIG [ARCTIC [BOX [LEVELS_DIR [REF_DIR]]]]

BOX: "c0,c1,r0,r1" (ORCA025 F lines of the fine box) or "lon0,lat0,lon1,lat1" (degrees, with a dot: converted to the
enclosing F lines); default the lon-lat box -80, 25, -40, 50.
"""
import sys

import numpy as np

from meshom.orca import build
from meshom.orca.ladder import preset_edge_rows, preset_f_rows
from meshom.orca.nest import H_DEFAULT, box_from_lonlat, boxes

out, path = sys.argv[1], sys.argv[2]
arctic = len(sys.argv) > 3 and sys.argv[3] == "1"
box_arg = sys.argv[4] if len(sys.argv) > 4 else "-80.0,25.0,-40.0,50.0"
levels = sys.argv[5] if len(sys.argv) > 5 else ".tmp/orca_levels"
ref_dir = sys.argv[6] if len(sys.argv) > 6 else ".tmp/orca_refs"

g = build("ORCA025")
if "." in box_arg:
    LONLAT = [float(v) for v in box_arg.split(",")]
    BOX = list(box_from_lonlat(g, LONLAT))
else:
    LONLAT = None
    BOX = [int(v) for v in box_arg.split(",")]
nk = {"k": 3, "col_offset": -1, "row_offset": 1}                    # ORCA025 -> ORCA12 (meshom.orca.mesh.nesting)
CBOX = boxes(BOX, H_DEFAULT, 3, nk)["coarse_box"]
nc = g.lon["f"].shape[1] - 2


def loop(b):
    """The box's edge as F points of ORCA025, every coarse F point, counter-clockwise in index space."""
    c0, c1, r0, r1 = b
    rr = np.r_[np.full(c1 - c0, r0), np.arange(r0, r1), np.full(c1 - c0, r1), np.arange(r1, r0, -1)]
    cc = np.r_[np.arange(c0, c1), np.full(r1 - r0, c1), np.arange(c1, c0, -1), np.full(r1 - r0, c0)]
    return np.c_[g.lon["f"][rr, cc % nc], g.lat["f"][rr, cc % nc]]


FINE, OUTER = loop(BOX), loop(CBOX)


def fmt(p):
    return "[" + ", ".join(f"[{x:.5f}, {y:.5f}]" for x, y in p) + "]"


def corners(b):
    c0, c1, r0, r1 = b
    return ", ".join(f"{g.lon['f'][r, c % nc]:.1f} E {g.lat['f'][r, c % nc]:.1f} N"
                     for r, c in ((r0, c0), (r0, c1), (r1, c1), (r1, c0)))


def builtin(names):
    from meshom.straits import builtin as bl

    out = []
    for d in bl("fesom2_mesh_gen"):
        if d["name"] in names:
            w = f"width_m = {float(d['width_m'])}\n" if d.get("width_m") else ""
            out.append(f'[[straits]]\nname = "{d["name"]}"\npath = {fmt(d["path"])}\n{w}profile = "widen"\n')
    assert len(out) == len(names), out
    return "\n".join(out)


R = "/work/ab0995/a270126/projects/mesh_generation"
ladder = ""
arctic_txt = ""
zones = f'''[[size.layers]]
kind = "zone"
mode = "blend"
name = "ORCA12 = ORCA025 / 3 in the nest's fine box, scaled down from ORCA025's cells at its coarse box"
h_factor = {1.0 / 3.0!r}
transition_m = 0.0
polygon = {fmt(FINE)}
outer = {fmt(OUTER)}
'''
if arctic:
    r_c, r_f = preset_edge_rows("ORCA025", "ORCA12", 56.0, 62.0, (range(700, 860), range(2250, 2560)))
    if CBOX[3] >= r_c:
        raise SystemExit(f"the nest's coarse box reaches the ladder's coarse row {r_c}: {CBOX}")
    lo_c, la_c = preset_f_rows("ORCA025", [r_c])
    lo_f, la_f = preset_f_rows("ORCA12", [r_f])
    ladder = 'fine_orca = "ORCA12"\ntransition = "ladder"\nladder_lat = [56.0, 62.0]\n'
    arctic_txt = f'''
[[bathymetry.regions]]
# Penny Strait (Bathurst Island - Grinnell Peninsula, ~157 m) with Belcher Channel and the north of Queens Channel (B117)
name = "Penny Strait"
box = [-97.5, 75.5, -93.5, 77.35]
min_depth_m = 150.0

# B117: the Danish straits lie in the ladder: widened corridors keep them open (meshom's built-in list fesom2_mesh_gen)
{builtin(("Oresund", "Great_Belt", "Little_Belt"))}'''
    zones += f'''
[[size.layers]]
kind = "zone"
mode = "blend"
name = "ORCA12 = ORCA025 / 3 north of ORCA12's F row {r_f}, scaled down from ORCA025's cells at its F row {r_c}"
h_factor = {1.0 / 3.0!r}
transition_m = 0.0
polygon = {fmt(np.c_[lo_f[0], la_f[0]][::3])}
outer = {fmt(np.c_[lo_c[0], la_c[0]])}
'''
txt = f'''# ORCA025 with ORCA12 nested in a Gulf Stream box (B176{", with the Arctic as ORCA12 joined by the ladder, B117" if arctic else ""}):
# fine box ORCA025 F columns {BOX[0]}-{BOX[1]}, rows {BOX[2]}-{BOX[3]} (corners {corners(BOX)}{"; from the lon-lat box " + str(LONLAT) if LONLAT else ""}),
# coarse box {CBOX} (corners {corners(CBOX)}); FESOM3's rules on both ORCA parts (B94); between the boxes
# meshom.orca.nest's frame ({H_DEFAULT} fine steps, B172-B175), cut by the coast. h: ORCA025's own cell size (B101) x 3^-(1 - t)
# across the frame (B110). Written by scripts/orca025_gs_nest.py.
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
{ladder}
[mesh.opts.nest]
fine_orca = "ORCA12"
box = {BOX}
width = {H_DEFAULT}

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
{arctic_txt}
[size]
grad_max = 0.1
h_max_m = 40000.0

[[size.layers]]
kind = "orca"
name = "ORCA025 cell size"
orca = "ORCA025"
h_min_m = 1000.0

{zones}
[run]
timeout = 14400.0
figures = false
'''
open(path, "w").write(txt)
print(path, "box", BOX, corners(BOX), "coarse box", CBOX, corners(CBOX))
