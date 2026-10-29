"""Hand edits of a mesh (the Mesh panel's "Edit by hand", static/edit.js): the page sends its edits as a list of
operations in the order they were made; the server replays them on the full mesh file and writes the result as
edited-NN.npz next to it (the npz layout of raw.npz: points, tris, quads, node_flag[, tri_zone]) with edited-NN.json
(the input file and the operations). No meshing: a node moves, or an element is split around a new node; topology
beyond that (flips, collapses) belongs to the core.

Operations:
  {"op": "move", "node": i, "lon": .., "lat": ..}    node i to lon / lat (degrees)
  {"op": "split", "elem": e, "lon": .., "lat": ..}   a new node at lon / lat inside element e: a triangle (a, b, c)
                                                      becomes (a, b, p), (b, c, p), (c, a, p), a quad (a, b, c, d) the
                                                      four triangles (a, b, p) ... (d, a, p); corner order is kept, so
                                                      CCW elements stay CCW
Numbering (the page uses the same): nodes 0 .. N-1 as in the file, added nodes N, N+1, ... in the order made;
elements 0 .. T+Q-1 as meshom numbers them (triangles first, then quads), the triangles a split makes T+Q, T+Q+1, ...
in the order made (a split element is gone and its number is not reused). In the file the kept triangles come first,
then the new ones in the order made; the kept quads keep their order; added nodes get node_flag 0.

The files are never overwritten: NN is one more than the largest edited-NN.npz in the folder, and the file is opened
for exclusive creation.
"""
import datetime
import json
import re
from pathlib import Path

import numpy as np

from meshom.geo import lonlat_to_xyz

NAME = re.compile(r"edited-(\d+)\.npz")


def apply(arrays: dict, ops: list[dict]) -> dict:
    """The arrays of a mesh npz with the operations applied (see the module docstring)."""
    pts, tris, quads, flag = arrays["points"].copy(), arrays["tris"], arrays["quads"], arrays["node_flag"]
    N, T, Q = len(pts), len(tris), len(quads)
    alive = np.ones(T + Q, bool)
    new_pts: list[np.ndarray] = []
    new_tris: list[tuple] = []
    new_alive: list[bool] = []
    for op in ops:
        xyz = lonlat_to_xyz(float(op["lon"]), float(op["lat"]))
        if op["op"] == "move":
            i = int(op["node"])
            if i < N:
                pts[i] = xyz
            else:
                new_pts[i - N] = xyz
        elif op["op"] == "split":
            e = int(op["elem"])
            if e < T + Q:
                c = [int(x) for x in (tris[e] if e < T else quads[e - T])]
                alive[e] = False
            else:
                c = list(new_tris[e - T - Q])
                new_alive[e - T - Q] = False
            p = N + len(new_pts)
            new_pts.append(xyz)
            for k in range(len(c)):
                new_tris.append((c[k], c[(k + 1) % len(c)], p))
                new_alive.append(True)
        else:
            raise ValueError(f"unknown edit operation {op['op']!r}")
    nt = np.array(new_tris, dtype=tris.dtype).reshape(-1, 3)[np.array(new_alive, bool)]
    out = {"points": np.vstack([pts, np.array(new_pts, pts.dtype).reshape(-1, 3)]),
           "tris": np.vstack([tris[alive[:T]], nt]), "quads": quads[alive[T:]],
           "node_flag": np.r_[flag, np.zeros(len(new_pts), flag.dtype)]}
    if "tri_zone" in arrays:                       # new triangles are not part of a declared triangle zone
        out["tri_zone"] = np.r_[arrays["tri_zone"][alive[:T]], np.zeros(len(nt), bool)]
    return out


def save(mf: Path, ops: list[dict]) -> dict:
    """Replay ops on the mesh file mf and write edited-NN.npz / .json next to it, and mf's depth edits as
    edited-NN.depth_edits.json (node ids are kept: new nodes come last); returns what was written."""
    mf = Path(mf)
    with np.load(mf) as z:
        out = apply({k: z[k] for k in z.files}, ops)
    nn = max((int(m.group(1)) for p in mf.parent.glob("edited-*.npz") if (m := NAME.fullmatch(p.name))), default=0) + 1
    path = mf.parent / f"edited-{nn:02d}.npz"
    with open(path, "xb") as f:                     # never over an existing file
        np.savez(f, **out)
    moves = sum(op["op"] == "move" for op in ops)
    path.with_suffix(".json").write_text(json.dumps({
        "input": mf.name, "date": datetime.datetime.now().isoformat(timespec="seconds"), "moves": moves,
        "splits": len(ops) - moves, "ops": ops}, indent=1))
    de = mf.with_name(f"{mf.stem}.depth_edits.json")     # the input's depth edits: its node ids hold here too
    if de.exists():
        e = json.loads(de.read_text())
        n = len(out["points"])
        if e.get("n_nodes") == n - sum(op["op"] == "split" for op in ops):     # the input's edits, not another mesh's
            e.update(mesh=path.name, n_nodes=n)
            path.with_name(f"{path.stem}.depth_edits.json").write_text(json.dumps(e))
    return {"saved": str(path), "name": path.name, "n_nodes": len(out["points"]), "n_tris": len(out["tris"]),
            "n_quads": len(out["quads"])}
