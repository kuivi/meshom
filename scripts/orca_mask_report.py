"""The land-sea mask of an m16_orca run's ORCA part against NEMO's (decisions B82-B85): cell by cell where NEMO's mask
maps (its T point is ours), the cells that differ (count, km2, where), the cells of the depth rule, and named places:
the Red Sea (and whether it is connected to the ocean through Bab-el-Mandeb), the Caspian, Gibraltar, the Indonesian
passages. Needs the run's orca_belt.npz (T points, NEMO's flags, the mask's clean-up) and its final mesh.

    python scripts/orca_mask_report.py RUN_DIR [MESH]      # MESH: raw.npz (default); writes RUN_DIR/orca_mask.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meshom.compete import _load  # noqa: E402
from meshom.geo import R_EARTH_M, lonlat_to_xyz  # noqa: E402

# (name, lon0, lon1, lat0, lat1): boxes of T points
PLACES = [("Red Sea", 32.0, 44.0, 12.5, 30.0), ("Bab-el-Mandeb", 42.5, 44.5, 11.5, 13.5),
          ("Caspian", 46.5, 55.0, 36.5, 47.5), ("Gibraltar", -6.5, -4.5, 35.5, 36.5),
          ("Lombok Strait", 115.0, 116.5, -9.5, -8.0), ("Makassar Strait", 116.5, 119.5, -4.0, 1.0),
          ("Ombai Strait", 124.0, 126.0, -9.0, -8.0), ("Timor Passage", 125.0, 128.0, -11.0, -9.5),
          ("Torres Strait", 141.0, 143.5, -11.0, -9.5), ("Bering Strait", -170.5, -167.5, 65.0, 66.5),
          ("Danish straits", 9.5, 13.0, 54.5, 58.0), ("Bosphorus", 28.5, 29.5, 40.8, 41.4)]


def main(run: Path, mesh_name: str = "raw.npz") -> dict:
    b = np.load(run / "orca_belt.npz")
    tlon, tlat = b["tlon"], b["tlat"]
    nj, ni = tlat.shape
    nemo, mapped, wetmap = b["nemo_wet"], b["nemo_mapped"], b["wetmap"]
    mesh = _load(run / mesh_name)
    P = mesh.points
    # the final mesh's ORCA cells: elements whose nodes are all ORCA nodes, by their centroid's nearest T point
    tree_b = cKDTree(b["points"])
    d, _ = tree_b.query(P, distance_upper_bound=1e-9)
    onb = np.isfinite(d)
    present = b["present"] if "present" in b.files else np.ones((nj, ni), bool)
    pidx = np.flatnonzero(present.ravel())                     # one copy of a T pivot row's cells
    T = lonlat_to_xyz(tlon.ravel()[pidx], tlat.ravel()[pidx])
    tt = cKDTree(T)
    final = np.zeros((nj, ni), bool)
    for arr in (mesh.tris, mesh.quads):
        if not len(arr):
            continue
        sel = onb[arr].all(axis=1)
        c = P[arr[sel]].mean(axis=1)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        _, k = tt.query(c)
        final.ravel()[pidx[k]] = True
    # cell areas from the F corners
    F = lonlat_to_xyz(b["lon"].ravel(), b["lat"].ravel()).reshape(nj + 1, ni, 3)
    im = (np.arange(ni) - 1) % ni
    a, bb, cc, dd = F[:-1, im], F[:-1], F[1:], F[1:, im]
    area = 0.5 * (np.linalg.norm(np.cross(bb - a, cc - a), axis=-1) + np.linalg.norm(np.cross(cc - a, dd - a), axis=-1))
    area *= (R_EARTH_M / 1e3) ** 2
    rec = {"run": str(run), "cells": int(nj * ni), "cells_mapped": int(mapped.sum()),
           "cells_depth_rule": int((~mapped).sum()), "final_water_cells": int(final.sum())}
    m = mapped & present
    for key, sel in (("both_water", m & final & nemo), ("ours_water_nemo_land", m & final & ~nemo),
                     ("nemo_water_ours_land", m & ~final & nemo)):
        rec[key] = {"cells": int(sel.sum()), "km2": round(float(area[sel].sum()), 1)}
    clean = (wetmap == 1)                                           # water by the mask, removed by the clean-up
    rec["removed_by_cleanup"] = {"cells": int(clean.sum()), "km2": round(float(area[clean].sum()), 1)}
    rec["removed_in_run_cut"] = int((wetmap == 2).sum() - (final & (wetmap == 2)).sum())
    ndiff = m & ~final & nemo
    rec["nemo_water_ours_land_at"] = [[round(float(x), 2), round(float(y), 2)]
                                      for x, y in zip(tlon[ndiff], tlat[ndiff])][:200]
    dr = ~mapped
    rec["depth_rule"] = {"cells": int(dr.sum()), "water": int((dr & final).sum()),
                         "lat_range": [round(float(tlat[dr].min()), 2), round(float(tlat[dr].max()), 2)] if dr.any()
                         else None}
    # water bodies of the final ORCA cells (edge-connected through the grid, the fold not followed: the places below
    # lie south of it)
    idx = np.arange(nj * ni).reshape(nj, ni)
    e1 = (final & np.roll(final, -1, axis=1))
    e2 = (final[:-1] & final[1:])
    u = np.r_[idx[e1], idx[:-1][e2]]
    v = np.r_[np.roll(idx, -1, axis=1)[e1], idx[1:][e2]]
    g = coo_matrix((np.ones(len(u)), (u, v)), shape=(nj * ni, nj * ni))
    _, lab = connected_components(g, directed=False)
    lab = np.where(final.ravel(), lab, -1).reshape(nj, ni)
    main = np.argmax(np.bincount(lab[final]))
    places = {}
    for name, x0, x1, y0, y1 in PLACES:
        box = (((tlon - x0) % 360) <= ((x1 - x0) % 360)) & (tlat >= y0) & (tlat <= y1)
        box &= present
        bm = box & mapped
        places[name] = {"cells": int(box.sum()), "nemo_water": int((bm & nemo).sum()),
                        "ours_water": int((box & final).sum()), "ours_water_mapped": int((bm & final).sum()),
                        "differ": int((bm & (final != nemo)).sum()),
                        "ours_in_main_ocean": int((box & final & (lab == main)).sum()),
                        "depth_rule_cells": int((box & ~mapped).sum())}
    rec["places"] = places
    (run / "orca_mask.json").write_text(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    r = main(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "raw.npz")
    print(json.dumps({k: v for k, v in r.items() if k != "nemo_water_ours_land_at"}, indent=1))
