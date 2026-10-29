"""meshom.repair: local cavity remeshing of defect clusters on hand-built quad patches (boundary = coast, first).

- One rotated edge in a regular lattice: two 3-5 pairs whose Burgers vectors cancel (a single 3-5 pair carries one
  lattice row and cannot be removed inside a fixed regular loop). Repair removes all four irregular nodes, keeps the
  Euler sum, every quad convex, the coast nodes untouched.
- Three quad sectors around a valence-3 node (net charge +1) plus one rotated edge: the cavity's budget is +1, so at
  least one irregular spot stays; the removable pairs go.
- A clean lattice is left exactly as it is; two runs give identical meshes, with or without the process pool.
- repair_region: a polygon drawn around the rotated edge remeshes that zone only, with the same result.
"""
import numpy as np

from meshom.compete import irregular_counts
from meshom.domain import Domain
from meshom.mesh import Mesh, build_topology, signed_area
from meshom.quality.metrics import compute_quality
from meshom.geo import R_EARTH_M, xyz_to_lonlat
from meshom.repair import repair, repair_region, summary
from meshom.size import Uniform
from meshom.smooth import smooth
from tests.helpers import local_polygon, mesh_from_polygons, split_quads, structured_patch

DX = 10e3
H = Uniform(DX)
DOM = Domain([], "patch")


def _boundary_first(m: Mesh) -> tuple[Mesh, int]:
    """Renumber so that the boundary nodes come first (flag 1)."""
    b = build_topology(m).boundary_node
    perm = np.r_[np.flatnonzero(b), np.flatnonzero(~b)]
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    flag = np.zeros(m.n_nodes, dtype=np.int8)
    flag[: int(b.sum())] = 1
    return Mesh(m.points[perm], inv[m.tris], inv[m.quads], flag), int(b.sum())


def _rotate(m: Mesh, qa: int, qb: int) -> Mesh:
    """Replace quads qa, qb (sharing an edge) by the other split of their hexagon (an edge rotation)."""
    a, b = m.quads[qa].tolist(), m.quads[qb].tolist()
    for k in range(4):
        u, v = a[k], a[(k + 1) % 4]
        if u in b and b[(b.index(u) - 1) % 4] == v:
            break
    k1, k2 = a.index(v), b.index(u)
    poly = (a[k1:] + a[:k1])[:-1] + (b[k2:] + b[:k2])[:-1]
    new = [poly[1:5], poly[4:] + poly[:2]]
    keep = [i for i in range(m.n_quads) if i not in (qa, qb)]
    return Mesh(m.points, m.tris, np.vstack([m.quads[keep], np.array(new)]), m.node_flag)


def _settle(m: Mesh) -> Mesh:
    return smooth(m, H, build_topology(m).boundary_node, iters=20, kind="angle")


def _lattice_with_rotated_edge() -> Mesh:
    m = structured_patch(8, 8, DX, 10.0, 40.0)
    m = _rotate(m, 3 * 8 + 3, 3 * 8 + 4)                 # the edge between quads (3, 3) and (3, 4)
    return _boundary_first(_settle(m))[0]


def _three_sectors(n: int = 12) -> Mesh:
    """An equilateral triangle cut into 3 quads around its centre (valence 3), each split n x n, one edge rotated."""
    r = 6.0 * DX * n / 6.0 * 1.5
    c = np.array([[r * np.cos(np.radians(a)), r * np.sin(np.radians(a))] for a in (90.0, 210.0, 330.0)])
    mid = 0.5 * (c + np.roll(c, -1, axis=0))
    polys = []
    for i in range(3):
        A, B, C, D = c[i], mid[i], np.zeros(2), mid[i - 1]
        q = np.array([A, B, C, D])
        x, y = q[:, 0], q[:, 1]
        if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
            A, B, C, D = A, D, C, B
        for i0 in range(n):
            for j0 in range(n):
                uv = [(i0, j0), (i0 + 1, j0), (i0 + 1, j0 + 1), (i0, j0 + 1)]
                pts = [(1 - u / n) * (1 - v / n) * A + u / n * (1 - v / n) * B + u / n * v / n * C
                       + (1 - u / n) * v / n * D for u, v in uv]
                polys.append(local_polygon(-30.0, 20.0, np.array(pts)))
    m = mesh_from_polygons(polys)
    topo = build_topology(m)
    deg = np.bincount(topo.edges.ravel(), minlength=m.n_nodes)
    centre = int(np.flatnonzero((deg == 3) & ~topo.boundary_node)[0])
    # rotate the edge of two quads off the boundary and the centre, nearest to the point 0.3 r above the centre
    target = local_polygon(-30.0, 20.0, np.array([[0.0, 0.3 * r]]))[0]
    ee = topo.edge_elems
    best, pick = np.inf, None
    for e in np.flatnonzero(ee[:, 1] >= 0):
        qa, qb = ee[e] - m.n_tris
        nodes = set(m.quads[qa]) | set(m.quads[qb])
        if centre in nodes or topo.boundary_node[list(nodes)].any():
            continue
        mp = m.points[topo.edges[e]].mean(axis=0)
        d = np.linalg.norm(mp / np.linalg.norm(mp) - target)
        if d < best:
            best, pick = d, (qa, qb)
    return _boundary_first(_settle(_rotate(m, *pick)))[0]


def _check(out: Mesh, inp: Mesh) -> None:
    nb = int((inp.node_flag == 1).sum())
    assert np.array_equal(out.points[:nb], inp.points[:nb]) and (out.node_flag[:nb] == 1).all()
    b = build_topology(out).boundary_node
    assert b[:nb].all() and b.sum() == nb                                  # the coast is the same node set
    assert (signed_area(out) > 0).all()
    assert compute_quality(out).summary["quad_sj"]["min"] > 0                 # every quad convex
    ic, ic0 = irregular_counts(out), irregular_counts(inp)
    assert ic["euler_ok"] and ic["euler_sum"] == ic0["euler_sum"]


def test_rotated_edge_pairs_annihilated():
    m = _lattice_with_rotated_edge()
    ic0 = irregular_counts(m)
    assert (ic0["n3"], ic0["n5"], ic0["T"]) == (2, 2, 0)
    out, info = repair(m, H, DOM)
    _check(out, m)
    ic = irregular_counts(out)
    assert (ic["n3"], ic["n5"], ic["n_other"], ic["T"]) == (0, 0, 0, 0)
    assert sum(info["accepted"].values()) >= 1


def test_budget_kept():
    m = _three_sectors()
    s0 = summary(m, H)
    ic0 = irregular_counts(m)
    assert (ic0["n3"], ic0["n5"], ic0["T"]) == (3, 2, 0) and s0["gross"] == 5
    out, info = repair(m, H, DOM)
    _check(out, m)
    s = summary(out, H)
    ic = irregular_counts(out)
    interior_net = ic["n3"] - ic["n5"] + ic["T"]
    assert s["gross"] >= 1 and ic["n3"] + ic["T"] >= 1                     # the budget +1 cannot be removed
    assert s["gross"] < s0["gross"] and interior_net == 1                  # the removable quadrupole is gone
    assert "1" in info["budget_hist"]                                       # a cavity with budget +1 was tried


def test_clean_lattice_unchanged():
    m = _boundary_first(structured_patch(10, 10, DX, 0.0, 0.0))[0]
    out, info = repair(m, H, DOM)
    assert np.array_equal(out.points, m.points) and np.array_equal(out.quads, m.quads) and out.n_tris == 0
    assert sum(info["tried"].values()) == 0


def test_deterministic():
    """Two runs give the same mesh, also with the cavities of a wave evaluated by a pool of 2 processes."""
    n = 40
    base = structured_patch(n, n, DX, 5.0, -20.0)
    m = base
    for j in (6, 19, 32):
        for i in (6, 19, 32):
            q = [tuple(x) for x in m.quads.tolist()]
            m = _rotate(m, q.index(tuple(base.quads[j * n + i])), q.index(tuple(base.quads[j * n + i + 1])))
    m = _boundary_first(_settle(m))[0]
    assert irregular_counts(m)["n3"] == 18
    a, ia = repair(m, H, DOM, {"workers": 1})
    b, ib = repair(m, H, DOM, {"workers": 2})
    c, _ = repair(m, H, DOM, {"workers": 1})
    for x in (b, c):
        assert np.array_equal(a.points, x.points) and np.array_equal(a.quads, x.quads)
        assert np.array_equal(a.tris, x.tris)
    assert ia["accepted"] == ib["accepted"] and ib["pool_cavities"] >= 4 and ia["pool_cavities"] == 0
    ic = irregular_counts(a)
    assert (ic["n3"], ic["n5"], ic["T"]) == (0, 0, 0)


def test_region_polygon():
    """A lon/lat square of +-2 cells around the rotated edge: the zone is remeshed, its 3-5 pairs go, the rest stays."""
    m = _lattice_with_rotated_edge()
    topo = build_topology(m)
    deg = np.bincount(topo.edges.ravel(), minlength=m.n_nodes)
    bad = np.flatnonzero((deg != 4) & ~topo.boundary_node)
    c = m.points[bad].mean(axis=0)
    lon, lat = xyz_to_lonlat(c / np.linalg.norm(c))
    d = np.degrees(2.0 * DX / R_EARTH_M)
    e = d / np.cos(np.radians(lat))
    poly = [[lon - e, lat - d], [lon + e, lat - d], [lon + e, lat + d], [lon - e, lat + d]]
    out, info = repair_region(m, H, DOM, poly, {"rings": 1})
    assert info["accepted"] and info["n_core"] >= 4 and info["old"]["gross"] == 4 and info["new"]["gross"] == 0
    _check(out, m)
    ic = irregular_counts(out)
    assert (ic["n3"], ic["n5"], ic["T"]) == (0, 0, 0)
    far = np.linalg.norm(m.points - c, axis=1) > 6 * DX / R_EARTH_M          # nodes far from the zone: untouched
    keep = [tuple(p) for p in m.points[far]]
    assert set(keep) <= {tuple(p) for p in out.points}


def test_candidates_subset():
    """opts["candidates"] names the fills tried: with ["ops"] alone no m02 (or Coons) refill is tried or accepted."""
    m = _lattice_with_rotated_edge()
    out, info = repair(m, H, DOM, {"candidates": ["ops"]})
    assert set(info["tried"]) == set(info["accepted"]) == {"ops"} and info["options"]["candidates"] == ["ops"]
    assert sum(info["accepted"].values()) >= 1 and "fill_m02" not in info["time_s"]
    _check(out, m)


def _zone_and_rotated_edge() -> Mesh:
    """12 x 12 lattice: a declared triangle zone (quads (8, 8), (8, 9), (9, 8), (9, 9) split, tri_zone) and a rotated
    edge between quads (3, 3) and (3, 4) far from it."""
    m = split_quads(structured_patch(12, 12, DX, 10.0, 40.0), [104, 105, 116, 117])
    m = _boundary_first(_settle(_rotate(m, 39, 40)))[0]
    m.tri_zone = np.ones(m.n_tris, dtype=bool)
    return m


def _zone_tris(m: Mesh) -> set:
    return {tuple(sorted(map(tuple, m.points[t]))) for t in m.tris[m.tri_zone]}


def test_tri_zone_kept():
    """A declared triangle zone (Mesh.tri_zone) is intentional: repair never takes it into a
    cavity and returns its mask; the defects elsewhere are still repaired."""
    m = _zone_and_rotated_edge()
    ic0 = irregular_counts(m)
    for opts in ({"candidates": (), "max_rounds": 1, "rings_max": 2}, {}):
        out, info = repair(m, H, DOM, opts)
        assert out.tri_zone is not None and out.n_tris == 8 and out.tri_zone.all()
        assert _zone_tris(out) == _zone_tris(m)
        _check(out, m)
    assert sum(info["accepted"].values()) >= 1
    ic = irregular_counts(out)
    assert ic["n3"] == ic0["n3"] - 2 and ic["n5"] == ic0["n5"] - 2           # the rotated edge's two 3-5 pairs go
    poly = [[9.0, 39.0], [11.0, 39.0], [11.0, 41.0], [9.0, 41.0]]           # the whole patch
    out, info = repair_region(m, H, DOM, poly, {"rings": 1})
    assert out.tri_zone is not None and out.n_tris == 8 and out.tri_zone.all() and _zone_tris(out) == _zone_tris(m)


SPHERE_RUN = """
[mesh]
method = "m01_isotropic"
[coast]
source = "none"
[size]
grad_max = 0.1
[[size.layers]]
kind = "constant"
h_m = 1500000.0
[[size.layers]]
kind = "zone"
h_m = 500000.0
transition_m = 0.0
polygon = [[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]
[run]
timeout = 300.0
smooth = false
figures = false
"""
RECORD_KEYS = {"input", "polygon", "opts", "before", "after", "repair"}
SCORE_KEYS = {"n_nodes", "quality", "irregular", "rho_c", "gates_passed", "gates_total", "gates_failed",
              "gates_failed_families"}


def test_cli_on_run_folder(tmp_path):
    """`meshom repair RUN_DIR` on a `meshom run` folder (domain and h from its config.resolved.toml), then a zone of
    the repaired mesh chained with --mesh / --out and the fills limited to ops."""
    import json

    from meshom.cli import main
    from meshom.compete import _load

    cfg = tmp_path / "sphere.toml"
    cfg.write_text(SPHERE_RUN)
    d = tmp_path / "sphere"
    main(["run", str(cfg), "--out", str(d)])
    main(["repair", str(d)])
    r1 = json.loads((d / "repaired.json").read_text())
    assert RECORD_KEYS <= set(r1) and r1["input"] == "raw.npz" and r1["polygon"] is None
    assert set(r1["before"]) == set(r1["after"]) == SCORE_KEYS and r1["after"]["irregular"]["euler_sum"] == 8
    q = json.loads((d / "quality.json").read_text())["raw"]           # domain, h and gates rebuilt as the run's
    assert {k: r1["before"][k] for k in SCORE_KEYS} == {k: q[k] for k in SCORE_KEYS}
    assert r1["opts"]["candidates"] == ["ops", "coons", "m02"] and "accepted" in r1["repair"]
    m1 = _load(d / "repaired.npz")
    assert m1.n_elems > 0 and r1["after"]["n_nodes"] == m1.n_nodes
    poly = [[-25.0, -25.0], [25.0, -25.0], [25.0, 25.0], [-25.0, 25.0]]
    main(["repair", str(d), "--mesh", "repaired.npz", "--out", "r2", "--polygon", json.dumps(poly),
          "--opts", '{"candidates": ["ops"], "force": true}'])
    r2 = json.loads((d / "r2.json").read_text())
    assert RECORD_KEYS | {"region"} <= set(r2) and r2["input"] == "repaired.npz" and r2["polygon"] == poly
    assert r2["opts"]["candidates"] == ["ops"] and set(r2["repair"]["candidates"]) == {"ops"}
    assert r2["before"]["n_nodes"] == m1.n_nodes and r2["after"]["irregular"]["euler_sum"] == 8
    for k in ("before", "after"):
        z = r2["region"][k]
        assert z["n_elems"] > 0 and z["quality"]["size_fidelity"]["p99"] is not None and "gates_total" in z
    assert (d / "r2.npz").exists() and not (d / "repair.json").exists()


def test_cli_chain_m14_blocks(tmp_path, monkeypatch):
    """m14_blocks on uniform at scale 4 (a compete run directory with m14.json), repaired, then the repaired mesh
    repaired again: repaired.npz carries is_block, and each repair leaves the deep block elements of its input
    (beyond the first block_rows rows from the block outline) exactly as they were, still marked as block."""
    import json

    from meshom import compete
    from meshom.cli import main
    from meshom.repair import DEFAULTS, _deep_blocks, _elements

    monkeypatch.setattr(compete, "WORK_DIR", tmp_path / "work")
    r = compete.run_case("m14_blocks", "uniform", 4.0, out=tmp_path / "c.jsonl")
    assert r["status"] == "ok", r.get("error")
    d = tmp_path / "work" / "m14_blocks_uniform_s4"
    m0 = compete._load(d / "raw.npz")
    b0 = np.zeros(m0.n_elems, dtype=bool)
    nb = json.loads((d / "m14.json").read_text())["stitch"]["n_block_quads"]
    b0[m0.n_tris:m0.n_tris + nb] = True
    main(["repair", str(d), "--case", "uniform"])
    main(["repair", str(d), "--case", "uniform", "--mesh", "repaired.npz", "--out", "r2"])
    assert sum(json.loads((d / "repaired.json").read_text())["repair"]["accepted"].values()) > 0

    def keys(m, sel):
        el, nv = _elements(m)
        return {tuple(m.points[e[:k]].ravel()) for e, k in zip(el[sel], nv[sel])}

    for src, mask, dst in ((m0, b0, "repaired.npz"), (None, None, "r2.npz")):
        if src is None:                                   # the chained step: the mask written by the first repair
            src = compete._load(d / "repaired.npz")
            with np.load(d / "repaired.npz") as z:
                mask = z["is_block"]
        out = compete._load(d / dst)
        with np.load(d / dst) as z:
            ob = z["is_block"]
        deep = _deep_blocks(_elements(src)[0], mask, src.n_nodes, DEFAULTS["block_rows"])
        assert deep.sum() > 0 and len(ob) == out.n_elems
        assert keys(src, deep) <= keys(out, ob)            # every deep block element kept as it was, still a block


def test_coast_row_dislocation_ends_on_coast():
    """B65: an edge rotated in the row next to the coast leaves one 3-5 pair inside (its partner sits on the coast).
    The default repair never takes the coast row, so the pair stays; with coast_row ("after": a coast stage after
    the usual repair; "all") the pair ends on the coast."""
    m = _boundary_first(_settle(_rotate(structured_patch(8, 8, DX, 10.0, 40.0), 3, 4)))[0]
    ic0 = irregular_counts(m)
    assert (ic0["n3"], ic0["n5"]) == (1, 1)
    out, _ = repair(m, H, DOM, {"workers": 1})
    assert (irregular_counts(out)["n3"], irregular_counts(out)["n5"]) == (1, 1)
    for mode in ("after", "all"):
        out, info = repair(m, H, DOM, {"coast_row": mode, "workers": 1})
        _check(out, m)
        ic = irregular_counts(out)
        assert (ic["n3"], ic["n5"], ic["n_other"], ic["T"]) == (0, 0, 0, 0)
        acc = info["coast_stage"]["accepted"] if mode == "after" else info["accepted"]
        assert sum(acc.values()) >= 1
