"""The valence-3 floor (decision B125): how much of a mesh's quad angle tail sits at interior pure-quad valence-3
nodes, where one corner is >= 120 deg whatever the geometry (three corners around a node sum to 360 deg).

    python -m meshom.meshfix.geom.floor --case NAME --mesh FILE.npz ...
"""
import argparse
import json
import sys

import numpy as np

from meshom.mesh import Mesh, build_topology
from meshom.quality.kernel import element_geometry


def floor_report(mesh: Mesh) -> dict:
    """Quads > 110 / > 120 deg split into those touching an interior pure-quad valence-3 node ("floor") and the
    rest; the largest corner at each n3 node (its excess over 120 deg: what geometry could still remove)."""
    topo = build_topology(mesh)
    N, T = mesh.n_nodes, mesh.n_tris
    nq = np.bincount(mesh.quads.ravel(), minlength=N)
    nt = np.bincount(mesh.tris.ravel(), minlength=N) if T else np.zeros(N, int)
    n3 = (nq == 3) & (nt == 0) & ~topo.boundary_node
    qa = np.degrees(element_geometry(mesh).quad_angles)
    qmax = qa.max(axis=1)
    at3 = n3[mesh.quads]                               # (Q,4) corner at an n3 node
    touch = at3.any(axis=1)
    # largest corner at each n3 node
    cm = np.full(N, 0.0)
    np.maximum.at(cm, mesh.quads[at3], qa[at3])
    out = {"n3_nodes": int(n3.sum()),
           "quads_gt120": int(((qmax > 120)).sum()), "quads_gt120_at_n3": int(((qmax > 120) & touch).sum()),
           "quads_gt110": int(((qmax > 110)).sum()), "quads_gt110_at_n3": int(((qmax > 110) & touch).sum()),
           "n3_max_corner_median": float(np.median(cm[n3])) if n3.any() else None,
           "n3_max_corner_p90": float(np.percentile(cm[n3], 90)) if n3.any() else None,
           "n3_max_corner_max": float(cm[n3].max()) if n3.any() else None,
           "n3_excess_over_120_sum": float(np.maximum(cm[n3] - 120.0, 0).sum()) if n3.any() else None,
           "quads_gt120_not_n3_max": float(qmax[(qmax > 120) & ~touch].max())
           if ((qmax > 120) & ~touch).any() else None}
    return out


def main(argv=None) -> None:
    from meshom.compete import _load
    from meshom.meshfix.cases import load

    ap = argparse.ArgumentParser(prog="python -m meshom.meshfix.geom.floor")
    ap.add_argument("--case", required=True)
    ap.add_argument("--mesh", nargs="*", default=[])
    a = ap.parse_args(argv)
    case = load(a.case, with_h=False)
    for f in [None] + a.mesh:
        m = case.mesh if f is None else _load(f)
        print(f or "input", json.dumps(floor_report(m)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
