"""The ORCA025 + Arctic config from drawn lines: the curves are interpolating cubic splines (scipy splprep, k = 3,
s = 0) through the control points in the north polar stereographic plane, the segments between them straight in
lon-lat; ORCA's edge as m16_orca's `region` (B100), the 4.5 km edge and ORCA's edge as the `blend` zone's polygons
(S37).

    python scripts/orca025_arctic_lines.py H_IN_M RUN_OUT CONFIG [PASSAGE_MIN_CELLS [DEPTH_AT [LEVELS_DIR [REF_DIR]]]]
"""
import sys

import numpy as np
from scipy.interpolate import splev, splprep


def ps(lo, la):
    r = np.tan(np.radians(90 - np.asarray(la, float)) / 2)
    return r * np.sin(np.radians(lo)), -r * np.cos(np.radians(lo))


def ips(x, y):
    r = np.hypot(x, y)
    return np.degrees(np.arctan2(x, -y)), 90 - 2 * np.degrees(np.arctan(r))


def smooth(pts, n=200):
    p = np.array(pts, float)
    x, y = ps(p[:, 0], p[:, 1])
    tck, _ = splprep([x, y], k=min(3, len(p) - 1), s=0)
    xs, ys = splev(np.linspace(0, 1, n), tck)
    lo, la = ips(np.asarray(xs), np.asarray(ys))
    return np.c_[lo, la]


def seg(pts, n=40):
    p = np.array(pts, float)
    return np.vstack([np.c_[np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)] for a, b in zip(p[:-1], p[1:])])


ORCA_LAT = 50.0
Y1 = smooth([(-95.5, 58.6), (-87.3, 61.3), (-80.5, 63.6), (-75.0, 65.1)])
Y2 = smooth([(-65.9, 63.2), (-59.3, 64.9), (-52.6, 66.5)])
Y3 = smooth([(-43.7, 61.6), (-30.0, 61.1), (-15.4, 61.0), (-3.3, 60.8), (7.2, 60.6)])
B1 = smooth([(-90.3, 66.8), (-85.0, 66.4), (-80.1, 68.2), (-73.7, 72.6), (-67.3, 77.0)])
B2 = smooth([(-26.9, 74.2), (-6.9, 73.0), (7.3, 71.3), (17.9, 69.5)])
ORCA_POLY = np.vstack([seg([(-100, 45), (-100, 58.6), (-95.5, 58.6)]), Y1,
                       seg([(-75.0, 65.1), (-68.0, 64.9), (-65.9, 63.2)]), Y2,
                       seg([(-52.6, 66.5), (-48.0, 64.0), (-43.7, 61.6)]), Y3,
                       seg([(7.2, 60.6), (12, 62.5), (20, 66.6), (26, 66.6), (31, 64.5), (31, 45), (-100, 45)])])
RED = np.vstack([seg([(-180, 62), (-155, 62), (-155, 66), (-100, 67.5), (-90.3, 66.8)]), B1,
                 seg([(-67.3, 77.0), (-50, 80), (-26.9, 74.2)]), B2,
                 seg([(17.9, 69.5), (20, 69.5), (30, 66), (30, 63.5), (60, 63.5), (60, 65), (160, 66), (165, 62),
                      (180, 62)])])
h_in, out, path = float(sys.argv[1]), sys.argv[2], sys.argv[3]
passages = int(sys.argv[4]) if len(sys.argv) > 4 else 0
depth_at = sys.argv[5] if len(sys.argv) > 5 else "node"
levels = sys.argv[6] if len(sys.argv) > 6 else ".tmp/orca_levels"
ref_dir = sys.argv[7] if len(sys.argv) > 7 else ".tmp/orca_refs"


def lat_line(a, b, la, step=1.0):
    lo = np.linspace(a, b, int(round((b - a) / step)) + 1)
    return np.c_[lo, np.full(len(lo), la)]


def dense(pts, step=1.0):
    """Straight lon-lat segments sampled every `step` degrees at most."""
    p = np.array(pts, float)
    out = []
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.abs(b - a).max() / step)))
        out.append(np.c_[np.linspace(a[0], b[0], n, endpoint=False), np.linspace(a[1], b[1], n, endpoint=False)])
    out.append(p[-1:])
    return np.vstack(out)


def dedup(p):
    keep = np.r_[True, np.abs(np.diff(p, axis=0)).max(axis=1) > 1e-9]
    return p[keep]


# ORCA's edge as one line from 180 W to 180 E (the drawn polygon north of 50 N, 50 N elsewhere; B100: the dip
# south of NEMO's Siberian grid pole at 73 E 50 N, over land)
EDGE = dedup(np.vstack([
    lat_line(-180, -100, 50.0), dense([(-100, 50), (-100, 58.6), (-95.5, 58.6)]), Y1,
    dense([(-75.0, 65.1), (-68.0, 64.9), (-65.9, 63.2)]), Y2,
    dense([(-52.6, 66.5), (-48.0, 64.0), (-43.7, 61.6)]), Y3,
    dense([(7.2, 60.6), (12, 62.5), (20, 66.6), (26, 66.6), (31, 64.5), (31, 50)]),
    lat_line(31, 62, 50.0), dense([(62, 50), (66, 46), (86, 46), (91, 50)]), lat_line(91, 180, 50.0)]))
REGION = np.vstack([[[-181.0, -91.0], [-181.0, 50.0]], EDGE, [[181.0, 50.0], [181.0, -91.0]]])
OUTER = EDGE[:-1]                                    # around the pole: 180 E is 180 W
RED = dedup(np.vstack([
    dense([(-180, 62), (-155, 62), (-155, 66), (-100, 67.5), (-90.3, 66.8)]), B1,
    dense([(-67.3, 77.0), (-50, 80), (-26.9, 74.2)]), B2,
    dense([(17.9, 69.5), (20, 69.5), (30, 66), (30, 63.5), (60, 63.5), (60, 65), (160, 66), (165, 62), (180, 62)])]))[:-1]


def fmt(p):
    return "[" + ", ".join(f"[{x:.4f}, {y:.4f}]" for x, y in p) + "]"


R = "/work/ab0995/a270126/projects/mesh_generation"
txt = f'''# ORCA025 joined to a {h_in / 1e3:g} km quad Arctic:
# NEMO's ORCA025 per column up to ORCA's edge (B100: a drawn line, smoothed; 50 N elsewhere, with a dip to
# 46 N over Kazakhstan south of NEMO's Siberian grid pole), FESOM3's rules on the ORCA cells (B94: NEMO's levels from
# FESOM3's ORCA025 inputs, levels_dir = scripts/grids/nemo_inputs_npz.py's nemo025q.npz); north of the edge a cap by
# m14_blocks_rep against the staircase, cut by the coast, one block inside the fine edge (B106). h: ORCA025's own
# cell size (B101), {h_in / 1e3:g} km inside the 4.5 km edge, a blend between the two edges
# (S37).
# Written by scripts/orca025_arctic_lines.py (the control points as splines).
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
# B106: one block over the uniform Arctic (inside the {h_in / 1e3:g} km edge), none in the transition (the fill adapts
# it to ORCA's cells): blocks only inside the zone's polygon, at most one, the B pre-screen off (b_cap 100: the line
# charge of h's kink at the zone's edge counts in B and grows as h falls, 1.7 at 9 km, 4.5 at 4.5 km for the whole
# zone; the acceptance F <= 0.08 still decides: 0.018)
place_region = {fmt(RED)}
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
name = "{h_in / 1e3:g} km Arctic, blended to ORCA's cells at ORCA's edge"
h_m = {h_in}
transition_m = 0.0
polygon = {fmt(RED)}
outer = {fmt(OUTER)}

[run]
timeout = 14400.0
figures = false
'''
open(path, "w").write(txt)
print(path, len(REGION), len(OUTER), len(RED))
