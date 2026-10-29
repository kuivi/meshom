"""`meshom run` with treatment "loops" meshes the coast filtered to h (meshom.coast_filter, decisions C16): no element
in a bay narrower than 2 h, no coast edge below 0.8 h. Without the filter the run meshes the bay: 5 elements in
it, the shortest coast edge 0.71 h; with it: 0 and 0.89 h."""
import numpy as np

from meshom.geo import R_EARTH_M, arc_length, xyz_to_lonlat
from tests.helpers import land_loop, local_polygon

LON0, LAT0 = 20.0, 10.0


def test_a_loops_run_meshes_the_filtered_coast(tmp_path, monkeypatch):
    """`meshom run` with treatment "loops" on a coast with a bay 40 km wide and 600 km deep at h = 200 km (method
    m00_background): no element in the bay, no coast edge below 0.8 h; a hand-placed block is laid against the same
    filtered coast (no coast vertex in the bay)."""
    import shapefile

    from meshom import compete
    from meshom.config import loads
    from meshom.mesh import build_topology
    import meshom.run
    from meshom.run import run
    from meshom.size import polygon_inside

    lp = land_loop(LON0, LAT0, [(-1500, -1000), (1500, -1000), (1500, 0), (20, 0), (20, -600), (-20, -600), (-20, 0),
                                (-1500, 0)])
    lon, lat = xyz_to_lonlat(lp)
    w = shapefile.Writer(str(tmp_path / "land"), shapeType=shapefile.POLYGON)
    w.field("area", "N", decimal=3)
    w.poly([np.c_[np.r_[lon, lon[:1]], np.r_[lat, lat[:1]]].tolist()])
    w.record(3.0e6)
    w.close()
    cfg = loads(f'[mesh]\nmethod = "m00_background"\nout = "{tmp_path / "run"}"\n'
                f'[coast]\nsource = "{tmp_path / "land.shp"}"\ntreatment = "loops"\n'
                f'[size]\ngrad_max = 0.1\nh_max_m = 200000.0\n[[size.layers]]\nkind = "constant"\nh_m = 200000.0\n'
                f'[[blocks]]\nname = "b"\ncorners = [[19.0, 10.5], [21.0, 10.5], [21.0, 12.0], [19.0, 12.0]]\n')
    seen = []
    monkeypatch.setattr(meshom.run, "hand_blocks", lambda cfg, coast, h: seen.append(coast) or [])
    q = run(cfg)["raw"]
    assert q["status"] == "ok"
    m = compete._load(tmp_path / "run" / "raw.npz")
    cent = np.r_[m.points[m.tris].mean(axis=1), m.points[m.quads].mean(axis=1)]
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    bay = local_polygon(LON0, LAT0, np.array([[-20e3, -590e3], [20e3, -590e3], [20e3, -10e3], [-20e3, -10e3]]))
    assert not polygon_inside(cent, bay).any()
    assert len(seen) == 1 and not polygon_inside(np.vstack(seen[0].loops), bay).any()
    topo = build_topology(m)
    e = topo.edges[topo.edge_elems[:, 1] < 0]
    s = R_EARTH_M * arc_length(m.points[e[:, 0]], m.points[e[:, 1]])
    assert s.min() > 0.8 * 200e3, s.min()
