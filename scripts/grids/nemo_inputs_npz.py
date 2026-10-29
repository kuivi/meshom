"""Convert NEMO's mesh-mask inputs of a FESOM3 ORCA conversion (mesh_hgr.nc, mesh_zgr.nc, mask.nc as FESOM3's
converter reads them) once to one .npz, so that meshom.orca.fesom3 and its tests need no netCDF reader.

    python scripts/grids/nemo_inputs_npz.py NEMO_DIR OUT.npz

The arrays keep the files' types and layout (rows x columns, the two wrap columns kept): glamt, gphit, glamf, gphif,
nav_lon, nav_lat (float32), nav_lev (float32, the T levels' depths), mbathy (int16, NEMO's bottom level: water > 0),
tmaskutil (int8) when mask.nc is there.
"""
import sys
from pathlib import Path

import netCDF4
import numpy as np


def _read(path: Path, name: str) -> np.ndarray:
    with netCDF4.Dataset(path) as d:
        v = d.variables[name]
        v.set_auto_maskandscale(False)
        a = np.array(v[:])
    return a[0] if a.ndim == 3 and a.shape[0] == 1 else a


def main(src: Path, out: Path) -> None:
    arr = {k: _read(src / "mesh_hgr.nc", k) for k in ("glamt", "gphit", "glamf", "gphif", "nav_lon", "nav_lat",
                                                      "nav_lev")}
    arr["mbathy"] = _read(src / "mesh_zgr.nc", "mbathy")
    if (src / "mask.nc").exists():
        arr["tmaskutil"] = _read(src / "mask.nc", "tmaskutil")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arr)
    print({k: (v.dtype.name, v.shape) for k, v in arr.items()})


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
