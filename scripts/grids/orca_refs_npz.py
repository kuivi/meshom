"""Convert the NEMO reference grids (ORCA2 domcfg, eORCA1 / eORCA025 / eORCA12 GO6 grid_mask) once to .npz, so that
meshom's ORCA tests and comparisons need no netCDF reader (env meshom-grids has netCDF4).

    python scripts/grids/orca_refs_npz.py OUT_DIR [NAME=PATH ...]  ->  OUT_DIR/{orca2,eorca1,eorca025,eorca12}.npz
NAME=PATH converts only those files (absolute paths): eorca05=<...>/eORCA05_mesh_mask.nc4 (MEOM / DRAKKAR eORCA05.L121,
ige-meom-opendap.univ-grenoble-alpes.fr thredds meomopendap/extract/MEOM/eORCA05.L121/eORCA05.L121-I, downloaded
2026-10-27).

Each npz: glam<p>, gphi<p> (float64, rows x columns as in the file, halo columns kept) for every point p in t, u, v, f
the file has, tmask (int8, the surface level; ORCA2: top_level > 0), closea_mask / bottom_level / top_level /
Bathymetry when the file has them, and attrs (str: the file's global attributes that say the fold type).
"""
import sys
from pathlib import Path

import netCDF4
import numpy as np

MG = next(p for p in Path(__file__).resolve().parents if (p / "curvilinear_grids").is_dir())
GRIDS = {"orca2": "nemo_orca2/ORCA_R2_zps_domcfg.nc", "eorca1": "nemo_eorca1/grid_mask_eORCA1-GO6.nc",
         "eorca025": "nemo_eorca025/grid_mask_eORCA025-GO6.nc", "eorca12": "nemo_eorca12/grid_mask_eORCA12-GO6.nc"}


def main(out: Path, only: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    base = MG / "curvilinear_grids"
    for name, rel in (only or GRIDS).items():
        with netCDF4.Dataset(base / rel) as d:
            d.set_auto_mask(False)
            arr = {}
            for p in "tuvf":
                for c in ("glam", "gphi"):
                    if f"{c}{p}" in d.variables:
                        arr[f"{c}{p}"] = np.asarray(d[f"{c}{p}"][...], np.float64).squeeze()
            for v in ("closea_mask", "bottom_level", "top_level", "Bathymetry"):
                if v in d.variables:
                    arr[v] = np.asarray(d[v][...]).squeeze()
            if "tmask" in d.variables:
                tm = np.asarray(d["tmask"][...]).squeeze()
                arr["tmask"] = (tm[0] if tm.ndim == 3 else tm).astype(np.int8)       # the surface level
            elif "top_level" in d.variables:
                arr["tmask"] = (np.asarray(d["top_level"][...]).squeeze() > 0).astype(np.int8)
            attrs = {k: str(d.getncattr(k)) for k in d.ncattrs() if k in ("NFtype", "NFold", "Iperio", "Jperio")}
        np.savez(out / f"{name}.npz", attrs=np.array(repr(attrs)), **arr)
        print(name, {k: v.shape for k, v in arr.items()}, attrs, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), dict(a.split("=", 1) for a in sys.argv[2:]) or None)
