"""Binary payloads of the web UI (little-endian; the page decodes them in static/app.js, the tests with decode_*).

mesh.bin (indexed, shared vertices):
    b"MSHB", uint32 header_len, header JSON (space-padded to a multiple of 4 bytes):
        {"n_nodes": N, "n_tris": T, "n_quads": Q, "subset": bool, "fields": ["angle", "size_ratio", "size_km"]}
    float32 lonlat[2 N]          node lon/lat (a subset carries only the nodes its elements use, renumbered)
    uint32  tris[3 T]            local node ids, CCW seen from outside
    uint32  quads[4 Q]
    uint32  elem_id[T + Q]       global element ids (tris first, then quads; only when subset)
    float32 <field>[T + Q]       per element, in the order of "fields" ("size_km": the mean edge length, km);
                                 "depth" (the mean depth of the element's
                                 nodes, m) follows only for a mesh with a depth (meshview.depth_of: its last
                                 write, else the run's depth.npy)
  with node_fields (the hand editor's zoom window, static/edit.js), the header has "node_fields": ["node_id",
  "node_flag"] and the payload ends with
    uint32  node_id[N]           global node ids of the local nodes
    uint8   node_flag[N]         the file's node_flag (coast / open-boundary nodes: != 0), padded to 4 bytes
  The page expands it to per-element vertices (flat colours) and passes pre-triangulated indices to deck.gl, so no
  polygon is triangulated in the browser; longitudes are unwrapped per element only for the flat map, where elements
  across ±180° also get a copy shifted by ∓360°.
points.bin:  uint32 n, float32 lonlat[2 n], uint8 cls[n] (padded to 4 bytes)
coast.bin:   uint32 n_paths, uint32 starts[n_paths + 1], float32 lonlat[2 V]
"""
import json
import struct

import numpy as np

MESH_FIELDS = ("angle", "size_ratio", "size_km")


def mesh_bin(mv, elems: np.ndarray | None = None, fields=None, node_fields: bool = False) -> bytes:
    if fields is None:
        fields = MESH_FIELDS + (("depth",) if "depth" in mv.fields else ())
    T = len(mv.tris)
    if elems is None:
        tris, quads, ll, eid = mv.tris, mv.quads, mv.lonlat, None
        e, used = slice(None), np.arange(len(mv.lonlat))
    else:
        e = np.sort(np.asarray(elems, dtype=np.int64))
        et, eq = e[e < T], e[e >= T] - T
        used = np.unique(np.r_[mv.tris[et].ravel(), mv.quads[eq].ravel()])
        tris, quads = np.searchsorted(used, mv.tris[et]), np.searchsorted(used, mv.quads[eq])
        ll, eid = mv.lonlat[used], e
    h = {"n_nodes": len(ll), "n_tris": len(tris), "n_quads": len(quads), "subset": eid is not None,
         "fields": list(fields)}
    if node_fields:
        h["node_fields"] = ["node_id", "node_flag"]
    head = json.dumps(h).encode()
    head += b" " * (-len(head) % 4)
    parts = [b"MSHB", struct.pack("<I", len(head)), head, np.ascontiguousarray(ll, "<f4").tobytes(),
             np.ascontiguousarray(tris, "<u4").tobytes(), np.ascontiguousarray(quads, "<u4").tobytes()]
    if eid is not None:
        parts.append(eid.astype("<u4").tobytes())
    parts += [np.ascontiguousarray(mv.fields[f][e], "<f4").tobytes() for f in fields]
    if node_fields:
        flag = mv.node_flag[used] if mv.node_flag is not None else np.zeros(len(used))
        parts += [used.astype("<u4").tobytes(), flag.astype(np.uint8).tobytes(), b"\0" * (-len(used) % 4)]
    return b"".join(parts)


def decode_mesh_bin(b: bytes) -> dict:
    assert b[:4] == b"MSHB"
    hl = struct.unpack("<I", b[4:8])[0]
    h = json.loads(b[8:8 + hl])
    o = 8 + hl
    n, t, q = h["n_nodes"], h["n_tris"], h["n_quads"]

    def take(dtype, count, shape=None):
        nonlocal o
        a = np.frombuffer(b, dtype, count, o)
        o += a.nbytes
        return a.reshape(shape) if shape else a
    out = {"header": h, "lonlat": take("<f4", 2 * n, (n, 2)), "tris": take("<u4", 3 * t, (t, 3)),
           "quads": take("<u4", 4 * q, (q, 4))}
    out["elem_id"] = take("<u4", t + q) if h["subset"] else np.arange(t + q, dtype=np.uint32)
    for f in h["fields"]:
        out[f] = take("<f4", t + q)
    if h.get("node_fields"):
        out["node_id"], out["node_flag"] = take("<u4", n), take(np.uint8, n)
        o += -n % 4
    assert o == len(b)
    return out


def points_bin(lonlat: np.ndarray, cls: np.ndarray) -> bytes:
    n = len(lonlat)
    c = np.ascontiguousarray(cls, np.uint8).tobytes()
    return struct.pack("<I", n) + np.ascontiguousarray(lonlat, "<f4").tobytes() + c + b"\0" * (-n % 4)


def decode_points_bin(b: bytes) -> tuple[np.ndarray, np.ndarray]:
    n = struct.unpack("<I", b[:4])[0]
    return np.frombuffer(b, "<f4", 2 * n, 4).reshape(n, 2), np.frombuffer(b, np.uint8, n, 4 + 8 * n)


def coast_bin(paths: list) -> bytes:
    starts = np.r_[0, np.cumsum([len(p) for p in paths])].astype("<u4")
    ll = np.vstack(paths).astype("<f4") if paths else np.zeros((0, 2), "<f4")
    return struct.pack("<I", len(paths)) + starts.tobytes() + ll.tobytes()
