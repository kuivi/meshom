"""meshom config, size layers, gradient limiter and `meshom run` / `meshom runs`."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from meshom import config
from meshom.cli import main
from meshom.geo import R_EARTH_M, arc_length, lonlat_to_xyz
from meshom.size import Latitude, coast_distance, cube_edges, cube_grid, disc, limit_gradient, zone

pytest.importorskip("tomli_w")

ROOT = Path(__file__).resolve().parents[1]
GSHHS = Path("/work/ab0995/a270126/projects/mesh_generation/oceanmesh/tests/GSHHS_i_L1.shp")
if not GSHHS.exists():
    GSHHS = ROOT.parent / "oceanmesh/tests/GSHHS_i_L1.shp"


def _example() -> str:
    text = (ROOT / "docs/config.md").read_text()
    return re.search(r"```toml\n(.*?)```", text, re.S).group(1)


def test_config_round_trip_and_unknown_key():
    cfg = config.loads(_example(), "example")
    assert config.loads(config.dumps(cfg), "example") == cfg
    assert [ly.kind for ly in cfg.size.layers] == ["constant", "latitude", "coast_distance", "zone"]
    assert cfg.base_h() == 400e3 and cfg.gates_all()["size_fidelity_p99"] == 0.15
    with pytest.raises(config.ConfigError, match=r"size\.layers\[3\]\.transiton_m: unknown key"):
        config.loads(_example().replace("transition_m =", "transiton_m ="))
    with pytest.raises(config.ConfigError, match=r"run\.smoth: unknown key"):
        config.loads(_example().replace("smooth = false", "smoth = false"))


def test_disabled_layer_and_reserved_keys():
    """enabled = false keeps a layer in the file and out of the run (fields, base h, limiter grid); [coast]
    smoothing_m is a reserved key that round-trips; enabled = true is not written."""
    from meshom.run import h_floor, layer_fields

    text = """
[mesh]
method = "m01_isotropic"
[coast]
smoothing_m = 5000.0
[[size.layers]]
kind = "constant"
h_m = 400000.0
[[size.layers]]
kind = "constant"
h_m = 900000.0
enabled = false
[[size.layers]]
kind = "latitude"
h_m = 400000.0
h_min_m = 100000.0
enabled = false
[[size.layers]]
kind = "zone"
h_m = 200000.0
transition_m = 0.0
polygon = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
enabled = true
"""
    cfg = config.loads(text)
    assert [ly.kind for ly in cfg.layers()] == ["constant", "zone"] and cfg.coast.smoothing_m == 5000.0
    assert cfg.base_h() == 400e3 and h_floor(cfg) == 200e3
    assert [n for n, _ in layer_fields(cfg, None)] == ["constant", "zone"]
    out = config.dumps(cfg)
    assert config.loads(out) == cfg and out.count("enabled") == 2 and "smoothing_m = 5000.0" in out


def _ll(lon, lat):
    return lonlat_to_xyz(np.atleast_1d(np.asarray(lon, float)), np.atleast_1d(np.asarray(lat, float)))


def test_layers_hand_values():
    lat = Latitude(100e3, 30e3)
    assert lat(_ll([0, 0, 0], [0, 60, 80])) == pytest.approx([100e3, 50e3, 30e3])
    cd = coast_distance(_ll(0.0, 0.0), h_m=20e3, plateau_m=30e3, grad=0.05, h_cap_m=100e3)
    d10 = R_EARTH_M * np.radians(10.0)                                    # 1111.95 km along the equator
    assert cd(_ll([0.0, 10.0, 30.0], [0.0, 0.0, 0.0])) == pytest.approx([20e3, 20e3 + 0.05 * (d10 - 30e3), 100e3])
    z = zone([[170.0, -5.0], [-170.0, -5.0], [-170.0, 5.0], [170.0, 5.0]], 10e3, 500e3, 100e3)  # across 180°
    h = z(_ll([180.0, 180.0, 0.0], [0.0, 7.0, 0.0]))
    # the top edge is a great circle: at 180° it bulges to atan(tan 5° / cos 10°) = 5.077°N
    d = np.radians(7.0 - np.degrees(np.arctan(np.tan(np.radians(5.0)) / np.cos(np.radians(10.0))))) * R_EARTH_M
    assert h == pytest.approx([10e3, 10e3 + 90e3 * d / 500e3, 100e3], rel=1e-9)


def test_zone_in_a_hemisphere_not_of_its_vertex_mean():
    """S19: a zone that fits in a hemisphere is a zone, even when its vertex mean is not a centre of one (the
    vertex (70, 0) is 112° from the mean of this thin 140°-wide triangle; every vertex has x > 0)."""
    z = zone([[-70.0, -10.0], [70.0, 0.0], [-70.0, 10.0]], 20e3, 500e3, 100e3)
    q = _ll([-60.0, 0.0, 60.0, 0.0, 180.0], [0.0, 0.0, 0.0, 17.0, 0.0])
    n = np.cross(_ll(70.0, 0.0)[0], _ll(-70.0, 10.0)[0])                  # (0, 17): 2.5° north of the top edge (14.5°N here)
    d = R_EARTH_M * np.arcsin(abs(q[3] @ n) / np.linalg.norm(n))
    h = z(q)
    assert h[:3] == pytest.approx([20e3, 20e3, 20e3]) and h[4] == 100e3
    assert h[3] == pytest.approx(20e3 + 80e3 * d / 500e3, rel=1e-9)


def test_limiter_disc_step():
    """A 10 -> 50 km step at r = 500 km is graded into 10 + g (r - 500 km); already graded fields stay as they are."""
    g, spacing = 0.1, 25e3
    step = disc(-30.0, 40.0, 500e3, 10e3, 500e3 + 1.0, 50e3)          # the disc case with its ramp squeezed to 1 m
    lim = limit_gradient(step, None, g, spacing_m=spacing)
    nid, hn = lim.node_id, lim.h_node
    _, _, xyz = cube_grid(lim.n)
    e = cube_edges(nid)
    d = R_EARTH_M * arc_length(xyz[e[:, 0]], xyz[e[:, 1]])
    slope = np.abs(hn[e[:, 0]] - hn[e[:, 1]]) / d
    dln = np.abs(np.log(hn[e[:, 0]] / hn[e[:, 1]])) * np.minimum(hn[e[:, 0]], hn[e[:, 1]]) / d
    assert slope.max() <= g * (1 + 1e-9) and dln.max() <= g * (1 + 1e-9)
    assert step(xyz).max() / step(xyz).min() == 5.0 and lim.info["nodes_lowered"] > 0
    c = np.asarray(step.center)
    r = R_EARTH_M * arc_length(xyz, c[None, :])
    assert np.all(hn[r < 500e3] == 10e3) and np.all(hn[r > 950e3] == 50e3)       # where h was graded: unchanged
    # on the ramp: 10 km + g (graph distance to the nearest node with r < 500 km) >= the exact 10 km + g (r - 500 km);
    # the excess is the step's position on the grid (<= one cell) plus the 16-stencil metrication (<= 2.7 %)
    ramp = (r > 520e3) & (r < 880e3)
    exact = 10e3 + g * (r[ramp] - 500e3)
    excess = hn[ramp] - exact
    assert excess.min() > -1e-6 and np.all(excess <= g * (lim.info["spacing_m"] + 0.027 * (r[ramp] - 500e3)))
    err = hn[ramp] / exact - 1.0
    far = r[ramp] > 700e3
    # the interpolant between nodes: exact where the field is constant over the cell
    q = np.random.default_rng(0).normal(size=(20000, 3))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rq = R_EARTH_M * arc_length(q, c[None, :])
    hq = lim(q)
    assert np.all(hq[rq > 1000e3] == 50e3) and np.all(hq[rq < 450e3] == 10e3)
    # the disc case itself is graded at g = 0.1: the limiter leaves it as it is
    dc = disc(-30.0, 40.0, 500e3, 10e3, 900e3, 50e3)
    lim2 = limit_gradient(dc, None, g, spacing_m=spacing)
    assert lim2.info["relaxations"] == 0 and np.array_equal(lim2.h_node, dc(xyz))
    smooth = (np.abs(rq - 500e3) > 2 * spacing) & (np.abs(rq - 900e3) > 2 * spacing)    # bilinear error: kinks only
    dq = np.abs(lim2(q) - dc(q))
    assert dq.max() < g * spacing and np.all(dq[smooth] <= 2e-3 * dc(q)[smooth])
    print(f"limiter n={lim.n} nodes={len(hn)} edges={len(e)} max slope/g={slope.max() / g:.6f} "
          f"ramp excess max {excess.max() / (g * lim.info['spacing_m']):.3f} g dx, rel err max {err.max():.4f} "
          f"(r > 700 km: {err[far].max():.4f}) lowered {lim.info['nodes_lowered']}")


def _worst_slope(h, x, d_m):
    """max over the points x and 12 directions of |h(q) - h(p)| / d_m, p and q d_m apart about each point."""
    from meshom.geo import tangent_basis

    e1, e2 = tangent_basis(x)
    d, worst = d_m / R_EARTH_M, 0.0
    for a in np.linspace(0.0, np.pi, 12, endpoint=False):
        b = np.cos(a) * e1 + np.sin(a) * e2
        worst = max(worst, float(np.max(np.abs(h(np.cos(d / 2) * x + np.sin(d / 2) * b)
                                               - h(np.cos(d / 2) * x - np.sin(d / 2) * b))) / d_m))
    return worst


def test_limited_field_slope_by_scale():
    """S11: the limited field meets grad_max exactly on the stencil edges only. The bilinear cells overshoot within
    one cell of a kink of h (where a ramp starts at a step, a cone apex, a ridge): any exact-at-node interpolant
    that is linear along the cell edges does, since a cone's nodes rise g L along two cell edges from its apex
    (bilinear: sqrt 2 g in square cells, up to 2 g in the 60° cells at a cube corner). Two cells apart (the
    shortest mesh edge, h_floor) only the stencil's metrication is left. Case: a step zone at a cube corner,
    g = 0.1, cells of h_floor / 2 = 100 km."""
    from meshom.run import size_field

    cfg = config.loads('[mesh]\nmethod = "m14_blocks_rep"\n[size]\ngrad_max = 0.1\nh_min_m = 200000.0\n'
                       'h_max_m = 500000.0\n[[size.layers]]\nkind = "constant"\nh_m = 500000.0\n[[size.layers]]\n'
                       'kind = "zone"\nh_m = 200000.0\ntransition_m = 0.0\n'
                       'polygon = [[35.0,25.0],[55.0,25.0],[55.0,45.0],[35.0,45.0]]\n')
    h = size_field(cfg, None)
    _, _, grid = cube_grid(h.n)
    e = cube_edges(h.node_id)
    edge = np.abs(h.h_node[e[:, 0]] - h.h_node[e[:, 1]]) / (R_EARTH_M * arc_length(grid[e[:, 0]], grid[e[:, 1]]))
    rng = np.random.default_rng(29)
    x = lonlat_to_xyz(rng.uniform(25, 65, 20000), rng.uniform(15, 55, 20000))
    L = h.info["spacing_m"]
    point, cell, two = (_worst_slope(h, x, d) / 0.1 for d in (2.0, L, 2 * L))
    print(f"slope / g: stencil edges {edge.max() / 0.1:.6f}, 2 m {point:.4f}, one cell {cell:.4f}, two {two:.4f}")
    assert edge.max() <= 0.1 * (1 + 1e-9)
    assert 1.5 < point < 2.0                    # 1.65 measured (2026-10-27): the bilinear overshoot at the kink
    assert cell < 1.15 and two < 1.06           # 1.11, 1.050: metrication near a cube corner (S11)


SPHERE = """
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
smooth = true
figures = true
"""


def test_run_closed_sphere(tmp_path, capsys):
    cfg_path = tmp_path / "sphere.toml"
    cfg_path.write_text(SPHERE)
    main(["run", str(cfg_path), "--out", str(tmp_path / "runs" / "sphere")])
    d = tmp_path / "runs" / "sphere"
    q = json.loads((d / "quality.json").read_text())
    assert q["raw"]["status"] == "ok" and q["raw"]["irregular"]["euler_sum"] == 8, q["raw"].get("error")
    stages = [s["stage"] for s in json.loads((d / "stages.json").read_text())]
    assert stages[:3] == ["coast", "size", "generate"] and "write" in stages
    lim = json.loads((d / "stages.json").read_text())[1]["limiter"]
    assert lim["nodes_lowered"] > 0 and lim["h_min_m"] == 500e3          # the zone's step was graded
    res = config.load(d / "config.resolved.toml")
    assert res.mesh.out == str(d) and res.size.layers == config.load(cfg_path).size.layers
    n = int((d / "fesom/nod2d.out").read_text().split()[0])
    final = q["smooth"] if "smooth" in q else q["raw"]
    assert n == final["n_nodes"] and (d / "fesom/elem2d.out").exists()
    assert (d / "figures/raw/index.json").exists()
    main(["runs", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert re.search(r"^sphere\s+ok\s+m01_isotropic", out, re.M), out


def test_run_again_after_the_partitioner(tmp_path):
    """`meshom run` into a run folder whose fesom/ holds a partition dist_<N>/ (fesom_meshpart / fesom_ini ran there)
    runs again and removes the old partition with the old files."""
    from meshom.run import run

    d = tmp_path / "again"
    cfg = config.loads(f'[mesh]\nmethod = "m00_background"\nout = "{d}"\n[coast]\nsource = "none"\n[size]\n'
                       'grad_max = 0.1\n[[size.layers]]\nkind = "constant"\nh_m = 800000.0\n[run]\nfigures = false\n',
                       "again")
    run(cfg)
    (d / "fesom" / "dist_2").mkdir()
    (d / "fesom" / "dist_2" / "rpart.out").write_text("2\n")
    (d / "fesom" / "edges.out").write_text("stale\n")
    q = run(cfg)
    assert q["raw"]["status"] == "ok"
    assert sorted(f.name for f in (d / "fesom").iterdir()) == ["aux3d.out", "depth.out", "elem2d.out", "nod2d.out"]
    final = q["optimized" if "optimized" in q else "smooth" if "smooth" in q else "raw"]
    w = final["water"]                                                  # C52: without straits too
    assert w["connectivity_ok"] and w["n_components"] == 1
    assert w["components"][0]["n_elems"] == final["quality"]["n_elems"]
    write = [s for s in json.loads((d / "stages.json").read_text()) if s["stage"] == "write"][-1]
    assert write["water"] == w


@pytest.mark.skipif(not GSHHS.exists(), reason="GSHHS coastline not found")
def test_run_gshhs_m02(tmp_path):
    text = f"""
[mesh]
method = "m02_linf_frontal"
[coast]
source = "{GSHHS}"
min_area_km2 = 20000.0
[size]
grad_max = 0.1
[[size.layers]]
kind = "constant"
h_m = 400000.0
[[size.layers]]
kind = "coast_distance"
h_m = 150000.0
plateau_m = 0.0
grad = 0.1
[run]
timeout = 600.0
"""
    cfg = config.loads(text, "gshhs")
    from meshom.run import run

    q = run(cfg, tmp_path / "gshhs")
    r = q["raw"]
    assert r["status"] == "ok", r.get("error")
    ic = r["irregular"]
    assert ic["euler_ok"] and r["quality"]["n_quads"] > 0
    st = {s["stage"]: s for s in json.loads((tmp_path / "gshhs/stages.json").read_text())}
    assert st["post"]["n_removed_land"] > 0 and st["coast"]["n_loops"] > 10
    print(f"gshhs: {r['n_nodes']} nodes, {r['quality']['n_quads']} quads, {r['quality']['n_tris']} tris, "
          f"loops {ic['chi_domain']}, gates {r['gates_passed']}/{r['gates_total']}, wall {r['wall_s']} s, "
          f"stages " + ", ".join(f"{k} {v['wall_s']:.1f}" for k, v in st.items()))


def test_base_h_without_constant_layer():
    """A config of zones alone (the UI's h builder allows it) has a base: the largest h_m / h_out_m of the layers."""
    from meshom.config import loads
    cfg = loads('[mesh]\nmethod = "m01_isotropic"\n[coast]\nsource = "none"\n[[size.layers]]\nkind = "zone"\nname = "z"\n'
                'h_m = 50000.0\ntransition_m = 500000.0\npolygon = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]\n'
                '[[size.layers]]\nkind = "latitude"\nh_m = 200000.0\nh_min_m = 80000.0\n')
    assert cfg.base_h() == 200000.0
    from meshom.run import layer_fields
    assert len(layer_fields(cfg, None)) == 2


def _synthetic_bathymetry(path: Path) -> None:
    """1-degree grid (cell centres, lat ascending, GEBCO's layout): ocean at -2000 m, a land block 60-120E x 30S-30N
    at +50 m, a shallow cap north of 60N at -3 m."""
    netCDF4 = pytest.importorskip("netCDF4")
    lat, lon = np.arange(-89.5, 90.0, 1.0), np.arange(-179.5, 180.0, 1.0)
    z = np.full((len(lat), len(lon)), -2000, dtype=np.int16)
    z[np.ix_((lat > -30) & (lat < 30), (lon > 60) & (lon < 120))] = 50
    z[lat > 60, :] = -3
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        ds.createVariable("elevation", "i2", ("lat", "lon"))[:] = z


def test_run_bathymetry_fesom2(tmp_path):
    """[bathymetry] + [mesh] format = "fesom2" on the closed sphere: the box mean takes ocean cells only (partly-land
    boxes stay at 2000 m), nodes whose box is all land are filled from their neighbours, shallow nodes are raised to
    the floor (min_depth_m 10 m, but for fesom2 the bottom of the 4th CORE2 layer, 30 m: min_layers), and depth.npy,
    stages.json, aux3d.out (CORE2 levels by default, depths negative) agree."""
    from meshom.geo import xyz_to_lonlat
    from meshom.io.fesom import read_aux3d, read_fesom

    nc = tmp_path / "bathy.nc"
    _synthetic_bathymetry(nc)
    text = SPHERE.replace('method = "m01_isotropic"', 'method = "m01_triangles"\nformat = "fesom2"')
    text = text.replace("smooth = true\nfigures = true", "smooth = false\nfigures = false")
    text += f'\n[bathymetry]\nsource = "{nc}"\nmin_depth_m = 10.0\nsmooth_passes = 0\n'
    cfg = config.loads(text, "bathy")
    assert config.loads(config.dumps(cfg), "bathy") == cfg and cfg.bathymetry.levels == "core2"
    from meshom.run import run

    q = run(cfg, tmp_path / "bathy")
    assert q["raw"]["status"] == "ok", q["raw"].get("error")
    d = tmp_path / "bathy"
    depth = np.load(d / "depth.npy")
    st = {s["stage"]: s for s in json.loads((d / "stages.json").read_text())}
    b = st["bathymetry"]
    m = read_fesom(d / "fesom")
    lon, lat = xyz_to_lonlat(m.points)
    zbar, aux = read_aux3d(d / "fesom/aux3d.out")
    assert len(zbar) == 48 and zbar[-1] == 6250.0 and np.allclose(aux, depth)
    assert b["n_nodes"] == m.n_nodes == len(depth) and b["n_without_ocean_cell"] >= 1 and b["n_fallback"] == 0
    assert np.allclose(depth[lat < 50.0], 2000.0, rtol=1e-12, atol=0)   # incl. boxes partly on the land block
    assert (lat > 70.0).any() and np.all(depth[lat > 70.0] == 30.0) and b["min_depth_m"] == 30.0
    assert b["n_raised_to_min_depth"] == int((depth == 30.0).sum()) and abs(b["policy"]["volume_change_rel"]) < 1e-12
    block = (lat > -25) & (lat < 25) & (lon > 65) & (lon < 115)
    assert block.any() and np.allclose(depth[block], 2000.0)    # filled from neighbours
    c = st["write"]["fesom2_checks"]
    assert st["write"]["format"] == "fesom2" and c["aux3d_depths"] == m.n_nodes and c["unused_nodes"] == 0
    assert (d / "fesom/elem2d.out").read_text().split("\n")[1].count(" ") == 2


def test_write_edited_mesh(tmp_path, capsys):
    """`meshom write` on a hand-edited mesh (ui/web/edit.py: a moved node, a split triangle): the files in the
    run's format (fesom2) and in fesom3 on request, with the depth recomputed on the edited nodes, the format's
    checks, the quality record in <out>/quality.json; the run's own fesom/ untouched."""
    import sys

    from meshom.geo import xyz_to_lonlat
    from meshom.io.fesom import read_aux3d, read_fesom
    from meshom.run import run

    sys.path.insert(0, str(ROOT))
    from ui.web.edit import save as save_edit

    nc = tmp_path / "bathy.nc"
    _synthetic_bathymetry(nc)
    text = SPHERE.replace('method = "m01_isotropic"', 'method = "m01_triangles"\nformat = "fesom2"')
    text = text.replace("smooth = true\nfigures = true", "smooth = false\nfigures = false")
    text += f'\n[bathymetry]\nsource = "{nc}"\nmin_depth_m = 10.0\n'
    d = tmp_path / "run"
    q = run(config.loads(text, "edit"), d)
    assert q["raw"]["status"] == "ok", q["raw"].get("error")
    own = {f.name: f.read_bytes() for f in (d / "fesom").iterdir()}
    with np.load(d / "raw.npz") as z:
        n, t = len(z["points"]), len(z["tris"])
        lon0, lat0 = xyz_to_lonlat(z["points"][z["tris"][3]].mean(axis=0, keepdims=True))   # a triangle's centre
    save_edit(d / "raw.npz", [{"op": "move", "node": 5, "lon": 0.0, "lat": 0.0}])        # edited-01
    save_edit(d / "raw.npz", [{"op": "split", "elem": 3, "lon": float(lon0[0]), "lat": float(lat0[0])},
                              {"op": "move", "node": 5, "lon": 90.0, "lat": 0.0}])       # edited-02: onto the land
    main(["write", str(d)])                                                            # newest edit, run's format
    line = capsys.readouterr().out
    o = d / "fesom-edited-02"
    assert "edited-02.npz" in line and "(fesom2)" in line and "checks" in line, line
    m = read_fesom(o)
    assert m.n_nodes == n + 1 and m.n_tris == t + 2
    lon, lat = xyz_to_lonlat(m.points)
    assert abs(lon[5] - 90.0) < 1e-6 and abs(lat[5]) < 1e-6
    _, aux = read_aux3d(o / "aux3d.out")
    assert len(aux) == n + 1 and np.all(aux >= 30.0) and aux[5] == 2000.0        # land node filled from neighbours
    rec = json.loads((o / "quality.json").read_text())["edited-02"]
    assert rec["n_nodes"] == n + 1 and rec["format"] == "fesom2" and rec["fesom2_checks"]["unused_nodes"] == 0
    assert rec["gates_total"] == q["raw"]["gates_total"] and rec["irregular"]["euler_sum"] == 8
    assert [s["stage"] for s in rec["stages"]] == ["size", "score", "bathymetry", "write"]
    from meshom.run import write_mesh

    r3 = write_mesh(d, "raw", "fesom3", tmp_path / "f3")
    assert r3["fesom3_checks"]["n_nodes"] == n and r3["fesom3_checks"]["elems_not_clockwise"] == 0
    assert (tmp_path / "f3/elem2d.out").read_text().split("\n")[1].count(" ") == 3   # 4 columns
    assert json.loads((tmp_path / "f3/quality.json").read_text())["raw"]["format"] == "fesom3"
    assert {f.name: f.read_bytes() for f in (d / "fesom").iterdir()} == own
    with pytest.raises(ValueError, match="never overwritten"):
        write_mesh(d, "raw", None, d / "fesom")


REGIONS = """
[[bathymetry.regions]]
name = "deep box"
box = [-30.0, -40.0, 30.0, 0.0]
min_depth_m = 2500.0
[[bathymetry.regions]]
name = "shallow cap"
polygon = [[150.0, -20.0], [-150.0, -20.0], [-150.0, 20.0], [150.0, 20.0]]
max_depth_m = 20.0
"""


def test_depth_regions_in_the_run_and_node_edits_in_the_write(tmp_path):
    """C1-C3 (decisions O35-O38): the [[bathymetry.regions]] round-trip the config and act in meshom run's bathymetry
    stage (a box raised to 2500 m; a polygon across 180 lowered to 20 m, which the fesom2 floor makes 30 m: depth.npy,
    aux3d.out and the stage's "regions" record agree); meshom write applies them again, saves depth_base.npy (no node
    edit in it) and then the mesh's <stem>.depth_edits.json in order, with the floor, recorded as "depth_edits"."""
    from meshom.geo import xyz_to_lonlat
    from meshom.io.fesom import read_aux3d
    from meshom.run import run, write_mesh

    nc = tmp_path / "bathy.nc"
    _synthetic_bathymetry(nc)
    text = SPHERE.replace('method = "m01_isotropic"', 'method = "m01_triangles"\nformat = "fesom2"')
    text = text.replace("smooth = true\nfigures = true", "smooth = false\nfigures = false")
    text += f'\n[bathymetry]\nsource = "{nc}"\nmin_depth_m = 10.0\n' + REGIONS
    cfg = config.loads(text, "regions")
    assert config.loads(config.dumps(cfg), "regions") == cfg and len(cfg.bathymetry.regions) == 2
    assert cfg.bathymetry.regions[1].polygon[1] == [-150.0, -20.0] and cfg.bathymetry.regions[0].depth_m is None
    with pytest.raises(config.ConfigError, match=r"bathymetry.regions\[0\].min_depth: unknown key"):
        config.loads(text.replace("min_depth_m = 2500.0", "min_depth = 2500.0"), "bad")
    d = tmp_path / "run"
    q = run(cfg, d)
    assert q["raw"]["status"] == "ok", q["raw"].get("error")
    depth = np.load(d / "depth.npy")
    with np.load(d / "raw.npz") as z:
        lon, lat = xyz_to_lonlat(z["points"])
    box = (lon >= -30) & (lon <= 30) & (lat >= -40) & (lat <= 0)
    cap = (np.abs(lon) >= 150) & (np.abs(lat) < 19.5)
    assert box.sum() >= 3 and cap.sum() >= 3
    assert np.all(depth[box] == 2500.0) and np.all(depth[cap] == 30.0)
    near = (np.abs(lon) >= 149) & (np.abs(lat) < 25)              # the cap's great-circle edges bow out of lat 20
    rest = ~box & ~near & (lat < 50.0) & ~((lat > -30) & (lat < 30) & (lon > 60) & (lon < 120))
    assert np.allclose(depth[rest], 2000.0)
    st = {s["stage"]: s for s in json.loads((d / "stages.json").read_text())}
    reg = st["bathymetry"]["regions"]
    assert [r["name"] for r in reg] == ["deep box", "shallow cap"] and reg[0]["n_changed"] == box.sum()
    assert reg[1]["n_clamped_to_floor"] >= cap.sum() and reg[1]["n_changed"] >= cap.sum()
    assert np.allclose(read_aux3d(d / "fesom/aux3d.out")[1], depth)

    r = write_mesh(d, "raw")                                       # no edits file: the base is the files' depth
    o = d / "fesom-raw"
    base = np.load(o / "depth_base.npy")
    assert np.array_equal(base, depth) and np.allclose(read_aux3d(o / "aux3d.out")[1], base)
    assert "depth_edits" not in r and r["stages"][2]["regions"] == reg
    i, j = np.flatnonzero(box)[:2], np.flatnonzero(rest)[:2]
    edits = {"mesh": "raw.npz", "n_nodes": len(depth),
             "edits": [{"op": "set", "depth_m": 100.0, "nodes": [int(k) for k in i], "outline": [], "date": "x"},
                       {"op": "max", "depth_m": 5.0, "nodes": [int(i[0]), *map(int, j)], "outline": [], "date": "x"}]}
    (d / "raw.depth_edits.json").write_text(json.dumps(edits))
    r = write_mesh(d, "raw")
    assert np.array_equal(np.load(o / "depth_base.npy"), base)     # C3: before the node edits
    aux = read_aux3d(o / "aux3d.out")[1]
    want = base.copy()
    want[i] = 100.0
    want[[i[0], *j]] = 30.0                                        # max 5 m: below the floor, raised to it
    assert np.allclose(aux, want)
    assert r["depth_edits"] == {"file": str(d / "raw.depth_edits.json"), "n_edits": 2, "n_nodes_changed": 4,
                                "n_clamped_to_floor": 3, "floor_m": 30.0}
    assert json.loads((o / "quality.json").read_text())["raw"]["depth_edits"] == r["depth_edits"]
    edits["n_nodes"] += 1
    (d / "raw.depth_edits.json").write_text(json.dumps(edits))
    with pytest.raises(ValueError, match="another mesh"):
        write_mesh(d, "raw")


def test_write_caches_passages_progress_and_fesom_ini(tmp_path):
    """write_mesh on a run folder with a coast (treatment "loops": the domain is the filtered coast, cached with h):
    progress called per stage, the passages block of the run's record (check only: a one-cell channel is reported),
    the size cache built once, reused, rebuilt when the config file changes, the rho_c cache equal to a new
    integration (decisions O33), fesom_ini on the written fesom3 folder on request (a stand-in binary)."""
    if not GSHHS.exists():
        pytest.skip("GSHHS test coast not on disk")
    from dataclasses import replace

    from meshom import compete
    from meshom.run import write_line, write_mesh
    from tests.test_passages import basins, grid_mesh

    d = tmp_path / "run"
    d.mkdir()
    ini = tmp_path / "fake_fesom_ini"                 # writes the derived files fesom_ini makes, as fesom_ini would
    ini.write_text("#!/bin/sh\nd=$(sed -n 's/^MeshPath=\"\\(.*\\)\\/\"$/\\1/p' namelist.config)\n"
                   "for f in edgenum.out edges.out edge_tri.out nlvls.out; do : > \"$d/$f\"; done\necho ' Total time 0.1'\n")
    ini.chmod(0o755)
    cfg = config.loads(f"""
[mesh]
method = "m01_triangles"
format = "fesom3"
fesom3_ini = "{ini}"
[coast]
source = "{GSHHS}"
min_area_km2 = 1000000.0
treatment = "loops"
[size]
grad_max = 0.1
[[size.layers]]
kind = "constant"
h_m = 400000.0
""", "wr")
    config.save(cfg, d / "config.resolved.toml")
    compete._save(grid_mesh(basins(1)), d / "edited-01.npz")
    calls = []
    r = write_mesh(d, progress=calls.append, fesom_ini=True)
    stages = [s["stage"] for s in r["stages"]]
    assert calls == stages == ["size", "score", "bathymetry", "write", "straits", "passages", "fesom_ini"]
    assert r["stages"][0]["cache"] == "built" and (d / "cache/size.pkl").exists()
    p = r["passages"]
    assert p["cells"] == 2 and p["n_narrow"] == 1 and p["narrow"][0]["kind"] == "pools"
    assert p["narrow"][0]["cells_across"] == 1 and "path" not in p["narrow"][0]
    f = r["fesom_ini"]
    assert f["ok"] is True and f["log"] == str(d / "fesom_ini-edited-01" / "fesom_ini.out")
    assert (d / "fesom-edited-01/edges.out").exists()
    assert json.loads((d / "fesom-edited-01/quality.json").read_text())["edited-01"]["passages"] == p
    assert "1 passages below 2 cells" in write_line(r) and "fesom_ini ok" in write_line(r)
    r2 = write_mesh(d)
    assert r2["stages"][0]["cache"] == "hit" and "fesom_ini" not in r2 and r2["quality"] == r["quality"]
    assert r["stages"][1]["rho_c_cache"] == "built" and r2["stages"][1]["rho_c_cache"] == "hit"
    (d / "cache/rho_c.json").unlink()                                   # integrated again: the same value
    r2b = write_mesh(d)
    assert r2b["stages"][1]["rho_c_cache"] == "built" and r2b["rho_c"] == r2["rho_c"] == r["rho_c"]
    config.save(replace(cfg, size=replace(cfg.size, layers=[replace(cfg.size.layers[0], h_m=200000.0)])),
                d / "config.resolved.toml")
    r3 = write_mesh(d)
    assert r3["stages"][0]["cache"] == "built"
    assert r3["quality"] != r["quality"]                                  # h halved: the size fidelity moves


def test_mesh_file_default_is_the_final_mesh(tmp_path):
    """`meshom write DIR` without --mesh takes the run's final mesh: the newest edited-NN, else optimized.npz (the
    run's optimisation stage, Z1: the mesh of its fesom/), else smooth, else raw."""
    from meshom.run import mesh_file

    for f in ("raw.npz", "smooth.npz", "optimized.npz"):
        (tmp_path / f).touch()
    assert mesh_file(tmp_path).name == "optimized.npz"
    (tmp_path / "optimized.npz").unlink()
    assert mesh_file(tmp_path).name == "smooth.npz"
    (tmp_path / "optimized.npz").touch()
    (tmp_path / "edited-01.npz").touch()
    assert mesh_file(tmp_path).name == "edited-01.npz"
    assert mesh_file(tmp_path, "raw").name == "raw.npz"


@pytest.mark.parametrize("method", ["m14_blocks", "m14_blocks_rep"])
def test_worker_hands_blocks_and_the_cut_record_to_m14_rep(tmp_path, monkeypatch, method):
    """run._worker gives the config's [[blocks]] and, under a land cut, the method's fixed-block record
    (m14_blocks.CUT_FIXED) to every m14 method, m14_blocks_rep (registered in rep_variants) included (before: it
    got no blocks and no record)."""
    import pickle

    import meshom.run as mrun
    from meshom import compete
    from meshom.domain import closed_sphere
    from meshom.methods import m14_blocks
    from meshom.size import Uniform

    inputs = tmp_path / "inputs.pkl"
    with open(inputs, "wb") as f:
        pickle.dump((closed_sphere(), Uniform(500e3), closed_sphere(), [], [{"name": "hand"}], None), f)
    seen = {}

    def fake_generate(name, domain, h, opts, smooth, d, post=None):
        m14_blocks.CUT_FIXED = "RECORD"                  # what the method leaves for the cut
        seen["blocks"] = opts.get("blocks", [])
        post(None)

    monkeypatch.setattr(compete, "_generate", fake_generate)
    monkeypatch.setattr(mrun, "cut_land", lambda m, land, keep, straits, fixed, *rest: seen.setdefault("fixed", fixed))
    monkeypatch.setattr(m14_blocks, "CUT_FIXED", None)
    mrun._worker(json.dumps({"method": method, "opts": {}, "smooth": False, "dir": str(tmp_path),
                             "inputs": str(inputs)}))
    assert seen == {"blocks": [{"name": "hand"}], "fixed": "RECORD"}
