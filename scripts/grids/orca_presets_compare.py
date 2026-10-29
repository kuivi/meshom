"""Each ORCA preset (meshom.orca.presets) against its NEMO reference grid, point by point (T and F, great-circle km):
median / p99 / max per band (south of 20 S, 20 S - 20 N, north of 20 N, the fold rows: the last three), outside
eORCA's southern ice-shelf rows (the reference rows before its first exact parallel) and, for ORCA2, outside its
hand-made Mediterranean / Red Sea / Asia patch (reported apart: the points farther than a quarter of the local grid
spacing from ours, meshom.orca.mesh.MATCH_FRAC, the rule that decides where NEMO's mask transfers). Per-row medians
and maxima go to <out>/orca_presets_compare.json and the figure <out>/figures/presets_vs_reference.png.

    python scripts/grids/orca_presets_compare.py REF_DIR OUT_DIR [PRESET ...]
REF_DIR: the npz of scripts/grids/orca_refs_npz.py.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from meshom.orca.presets import PRESETS, build  # noqa: E402

R_KM = 6371.229


def dist_km(l1, p1, l2, p2):
    p1, p2, dl = np.radians(p1), np.radians(p2), np.radians(l2 - l1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def stats(a):
    return {"n": int(a.size), "median_km": float(np.median(a)), "p99_km": float(np.percentile(a, 99)),
            "max_km": float(a.max())} if a.size else None


def compare(name, ref_dir):
    pr = PRESETS[name]
    t0 = time.time()
    g = build(name)
    t_build = time.time() - t0
    ref = np.load(Path(ref_dir) / f"{pr.ref}.npz")
    out = {"preset": name, "build_s": round(t_build, 1), "rows": int(g.lat["t"].shape[0]),
           "cols": int(g.lat["t"].shape[1]), "pivot": g.pivot, "info": {k: v for k, v in g.info.items()
                                                                          if not isinstance(v, dict)}}
    rp = ref["gphit"]
    circ = np.flatnonzero(np.ptp(rp[:, 2:-2], axis=1) < 1e-6)
    first_orca = int(circ.min()) if pr.ref != "orca2" else 0
    out["ref_first_orca_row"] = first_orca
    for p in "tf":
        lo, la = g.lon[p][:, :-2], g.lat[p][:, :-2]
        rl, rla = ref["glam" + p], ref["gphi" + p]
        nj, ni = la.shape
        rows = np.arange(nj)
        rr = rows + pr.ref_row_offset
        ok = (rr >= first_orca) & (rr < rl.shape[0])
        rows, rr = rows[ok], rr[ok]
        cols = np.arange(ni)
        rc = cols + pr.ref_col_offset
        okc = (rc >= 0) & (rc < rl.shape[1])
        cols, rc = cols[okc], rc[okc]
        d = dist_km(lo[np.ix_(rows, cols)], la[np.ix_(rows, cols)], rl[np.ix_(rr, rc)], rla[np.ix_(rr, rc)])
        patch = np.zeros(d.shape, bool)
        if pr.ref == "orca2":
            from meshom.orca.mesh import MATCH_FRAC

            L, A = lo[np.ix_(rows, cols)], la[np.ix_(rows, cols)]
            dz = dist_km(L, A, np.roll(L, -1, axis=1), np.roll(A, -1, axis=1))
            dm = np.abs(np.gradient(A, axis=0)) * np.pi / 180 * R_KM
            patch = d > MATCH_FRAC * np.minimum(dz, dm)
        m = rla[rr][:, rc].mean(axis=1)
        fold = rr >= rl.shape[0] - 3
        bands = {"south": (m < -20) & ~fold, "tropics": (m >= -20) & (m < 20) & ~fold,
                 "north": (m >= 20) & ~fold, "fold": fold}
        rec = {}
        for b, sel in bands.items():
            a = d[sel][~patch[sel]]
            rec[b] = stats(a)
        rec["all"] = stats(d[~patch])
        if patch.any():
            rec["orca2_patch"] = stats(d[patch])
        rec["per_row"] = {"row": rr.tolist(), "lat": m.round(4).tolist(),
                          "median_km": np.nanmedian(np.where(patch, np.nan, d), axis=1).round(6).tolist(),
                          "max_km": np.nanmax(np.where(patch, np.nan, d), axis=1).round(6).tolist()}
        out[p] = rec
    return out


def figure(recs, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(recs), figsize=(5 * len(recs), 4.5), squeeze=False)
    for ax, r in zip(axs[0], recs):
        pr = r["f"]["per_row"]
        ax.semilogy(pr["lat"], np.maximum(pr["max_km"], 1e-5), ".", ms=2, label="max")
        ax.semilogy(pr["lat"], np.maximum(pr["median_km"], 1e-5), ".", ms=2, label="median")
        ax.set_title(f"{r['preset']} vs {PRESETS[r['preset']].ref}: F points")
        ax.set_xlabel("row mean latitude (deg)")
        ax.set_ylabel("distance (km)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=90)


if __name__ == "__main__":
    ref_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    names = sys.argv[3:] or [n for n, p in PRESETS.items() if p.ref]
    (out / "figures").mkdir(parents=True, exist_ok=True)
    recs = []
    for n in names:
        r = compare(n, ref_dir)
        recs.append(r)
        for p in "tf":
            print(n, p.upper(), {k: v for k, v in r[p].items() if k != "per_row"}, flush=True)
        print(n, "build", r["build_s"], "s", r["rows"], "x", r["cols"], r["pivot"], flush=True)
    (out / "orca_presets_compare.json").write_text(json.dumps(recs))
    figure(recs, out / "figures" / "presets_vs_reference.png")
