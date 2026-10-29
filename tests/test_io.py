import numpy as np
import pytest

from helpers import mixed_cube, same_elements, triangulated_cube
from meshom.geo import euler_matrix, lonlat_to_xyz, rot_to_geo
from meshom.io.fesom import (fesom2_checks, fesom3_checks, read_aux3d, read_fesom, write_fesom2, write_fesom3,
                             write_fesomc)
from meshom.io.gmsh import read_gmsh41
from meshom.mesh import Mesh, orient_ccw, signed_area

# MSH 4.1 ASCII (the Gmsh reference manual, "MSH file format"): node blocks of dims 0, 1, 2; tags 10..60;
# a line element on 50-60, a quad 10-20-30-40 and a triangle 20-50-30; node 60 is used only by the line.
TINY_MSH41 = """$MeshFormat
4.1 0 8
$EndMeshFormat
$PhysicalNames
1
2 1 "surface"
$EndPhysicalNames
$Entities
1 1 1 0
1 0 0 0 0
1 2e4 5e3 0 3e4 5e3 0 0 0
1 0 0 0 2e4 1e4 0 1 1 0
$EndEntities
$Nodes
3 6 10 60
0 1 0 1
10
0 0 0
1 1 0 2
50
60
20000 5000 0
30000 5000 0
2 1 0 3
20
30
40
10000 0 0
10000 10000 0
0 10000 0
$EndNodes
$Elements
3 3 1 3
1 1 1 1
1 50 60
2 1 3 1
2 10 20 30 40
2 1 2 1
3 20 50 30
$EndElements
"""



LEV = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]


def test_fesom3_files(tmp_path):
    """nod2d.out (flag 1 on boundary nodes), elem2d.out with 4 columns (triangle n1 n2 n3 n1, clockwise), aux3d.out
    with the levels and node depths negative down; no depth.out, stale derived files removed."""
    m = mixed_cube()                                                     # 2 triangles, 5 quads
    m = Mesh(m.points, m.tris, m.quads[1:], m.node_flag)                 # one quad removed: a 4-node hole
    for f in ("depth.out", "nlvls.out", "edges.out"):
        (tmp_path / f).write_text("stale\n")
    depth = 100.0 + 10.0 * np.arange(m.n_nodes)
    write_fesom3(m, tmp_path, depth, LEV)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["aux3d.out", "elem2d.out", "nod2d.out"]
    nod = np.loadtxt(tmp_path / "nod2d.out", skiprows=1)
    assert (tmp_path / "nod2d.out").read_text().split()[0] == str(m.n_nodes)
    assert np.array_equal(nod[:, 0], np.arange(1, m.n_nodes + 1))
    assert sorted(np.flatnonzero(nod[:, 3] == 1)) == sorted(mixed_cube().quads[0])
    raw = np.loadtxt(tmp_path / "elem2d.out", skiprows=1, dtype=np.int64)
    assert (tmp_path / "elem2d.out").read_text().split()[0] == "6" and raw.shape == (6, 4)
    assert np.all(raw[:2, 3] == raw[:2, 0]) and np.all(raw[2:, 3] != raw[2:, 0])
    raw -= 1
    assert np.all(signed_area(Mesh(m.points, raw[:2, :3], raw[2:], m.node_flag)) < 0)   # clockwise
    v = (tmp_path / "aux3d.out").read_text().split()
    assert v[0] == "6" and [float(x) for x in v[1:7]] == [-x for x in LEV] and float(v[7]) == -100.0
    assert len(v) == 7 + m.n_nodes
    back = read_fesom(tmp_path)
    assert same_elements(back, m) and np.allclose(back.points, m.points, atol=1e-11)
    c = fesom3_checks(tmp_path)
    assert c["elem2d_columns"] == 4 and c["elems_not_clockwise"] == 0 and c["elems_anticlockwise_rotated"] == 0


def _rot_mesh(rlon, rlat, tris, quads) -> Mesh:
    """A mesh given in FESOM's rotated lon-lat (Euler 50/15/-90), elements CCW there."""
    pts = rot_to_geo(lonlat_to_xyz(np.asarray(rlon, float), np.asarray(rlat, float)), euler_matrix())
    return Mesh(pts, np.asarray(tris, np.int64).reshape(-1, 3), np.asarray(quads, np.int64).reshape(-1, 4),
                np.zeros(len(pts), np.int8))


def test_fesom3_checks_nonconvex_quad_and_order(tmp_path):
    """3 x 3 quads at the rotated equator pass; a reflex corner in rotated lon-lat is flagged non-convex (fesom_ini
    STOP); a reversed element line is flagged anticlockwise."""
    i, j = np.meshgrid(np.arange(4.0), np.arange(4.0), indexing="ij")
    rl, rt = i.ravel(), j.ravel()
    quads = [[4 * a + b, 4 * (a + 1) + b, 4 * (a + 1) + b + 1, 4 * a + b + 1] for a in range(3) for b in range(3)]
    write_fesom3(_rot_mesh(rl, rt, [], quads), tmp_path, np.full(16, 100.0), LEV)
    c = fesom3_checks(tmp_path)
    assert c["stop"] == [] and c["n_quads"] == 9 and c["quads_nonconvex_rotated"] == 0 and c["quads_bowtie_rotated"] == 0
    lines = (tmp_path / "elem2d.out").read_text().splitlines()
    n1, n2, n3, n4 = lines[1].split()
    (tmp_path / "elem2d.out").write_text("\n".join([lines[0], f"{n1} {n4} {n3} {n2}"] + lines[2:]) + "\n")
    c = fesom3_checks(tmp_path)
    assert c["elems_not_clockwise"] == 1 and c["elems_anticlockwise_rotated"] == 1 and c["stop"] == ["elems_not_clockwise"]
    rl, rt = rl.copy(), rt.copy()
    rl[5] = rt[5] = 1.7                       # node (1, 1) into the quad (1..2, 1..2): a reflex corner there
    write_fesom3(_rot_mesh(rl, rt, [], quads), tmp_path, np.full(16, 100.0), LEV)
    c = fesom3_checks(tmp_path)
    assert c["quads_nonconvex_rotated"] == 1 and c["quads_bowtie_rotated"] == 0 and c["elems_not_clockwise"] == 0
    assert c["quads_nonconvex_sphere"] == 1 and c["stop"] == ["quads_nonconvex_rotated", "quads_nonconvex_sphere"]


def test_fesom3_checks_quad_nonconvex_on_the_sphere_only(tmp_path):
    """A quad convex in rotated lon-lat, non-convex on the sphere (corner turn signs +, +, -, + with both diagonal
    sub-triangles of one sense), ringed by 8 triangles: FESOM3's mesh check rejects it (all four corner turns), so
    the checks must stop on it."""
    rl = [-5.908449025201931, 9.2900235260443, 4.6577856773095885, -12.409720275855115, -25.0, 25.0, 25.0, -25.0]
    rt = [80.22047669148068, 80.158203642545, 80.37317669990583, 80.71026299223672, 79.0, 79.0, 82.0, 82.0]
    tris = [[4 + i, 4 + (i + 1) % 4, (i + 1) % 4] for i in range(4)] + [[4 + i, (i + 1) % 4, i] for i in range(4)]
    write_fesom3(orient_ccw(_rot_mesh(rl, rt, tris, [[0, 1, 2, 3]])), tmp_path, np.full(8, 100.0), LEV)
    c = fesom3_checks(tmp_path)
    assert c["quads_nonconvex_rotated"] == 0 and c["elems_not_clockwise"] == 0 and c["elems_lt2_neighbours"] == 0
    assert c["quads_nonconvex_sphere"] == 1 and c["stop"] == ["quads_nonconvex_sphere"]


def test_fesom3_rewrite_removes_stale_partitions(tmp_path):
    """dist_<N> partition folders of an earlier mesh are removed (FESOM3 reads dist_<npes>/rpart.out for a
    multi-rank run); other folders stay."""
    for d in ("dist_2", "dist_16", "figures"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "rpart.out").write_text("2\nold mesh partition\n")
    write_fesom3(mixed_cube(), tmp_path, np.full(8, 100.0), LEV)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["aux3d.out", "elem2d.out", "figures", "nod2d.out"]


@pytest.mark.parametrize("fmt", ["fesom2", "fesom-c"])
def test_rewrite_removes_derived_files_and_partitions(tmp_path, fmt):
    """A FESOM2 or FESOM-C folder written again holds none of the earlier mesh's derived files or dist_<N>
    folders: FESOM2's partitioner never overwrites existing edge and level files, so the model would run on the old
    mesh's edges."""
    stale = ("edgenum.out", "edges.out", "edge_tri.out", "nlvls.out", "elvls.out", "elvls_raw.out")
    for f in stale:
        (tmp_path / f).write_text("stale\n")
    for d in ("dist_2", "figures"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "rpart.out").write_text("2\nold mesh partition\n")
    if fmt == "fesom2":
        m = triangulated_cube()
        write_fesom2(m, tmp_path, np.full(m.n_nodes, 100.0), LEV)
        want = ["aux3d.out", "elem2d.out", "figures", "nod2d.out"]
    else:
        m = mixed_cube()
        write_fesomc(m, tmp_path, np.full(m.n_nodes, 100.0))
        want = ["aux3d.out", "depth.out", "elem2d.out", "figures", "nod2d.out"]
    assert sorted(f.name for f in tmp_path.iterdir()) == want


def test_fesom3_checks_33_elements_at_a_node(tmp_path):
    """A fan of 33 triangles: 33 elements and 33 neighbour nodes at the centre, both above FESOM3's 32."""
    t = np.linspace(0.0, 2.0 * np.pi, 33, endpoint=False)
    rl, rt = np.r_[0.0, np.cos(t)], np.r_[0.0, np.sin(t)]
    tris = [[0, 1 + k, 1 + (k + 1) % 33] for k in range(33)]
    write_fesom3(_rot_mesh(rl, rt, tris, []), tmp_path, np.full(34, 100.0), LEV)
    c = fesom3_checks(tmp_path)
    assert c["elems_per_node_max"] == 33 and c["node_neighbours_max"] == 33 and c["elems_lt2_neighbours"] == 0
    assert c["stop"] == ["elems_per_node_max", "node_neighbours_max"]


@pytest.mark.data
def test_read_fesom2_pi(fesom2_dir):
    m = read_fesom(fesom2_dir / "test/meshes/pi")
    assert (m.n_nodes, m.n_tris, m.n_quads) == (3140, 5839, 0)
    assert np.all(signed_area(m) > 0)


def test_fesomc_roundtrip(tmp_path):
    m = mixed_cube()                      # cube_sphere with face 0 split into 2 triangles
    m.node_flag[:] = 0; m.node_flag[3] = 2
    write_fesomc(m, tmp_path, depth_m=np.full(m.n_nodes, 100.0))
    raw = np.loadtxt(tmp_path / "elem2d.out", skiprows=1, dtype=np.int64)
    assert raw.shape == (7, 4) and np.all(raw[:2, 3] == raw[:2, 0]) and np.all(raw[2:, 3] != raw[2:, 0])
    m2 = read_fesom(tmp_path)
    assert np.allclose(m2.points, m.points, atol=1e-11)
    assert same_elements(m2, m)           # equal as sets of cyclically normalised node tuples
    assert np.array_equal(m2.node_flag, m.node_flag)
    lines = (tmp_path / "aux3d.out").read_text().split()
    assert lines[:6] == ["5", "0", "1", "2", "3", "4"] and len(lines) == 6 + m.n_nodes


def test_fesomc_file_is_clockwise(tmp_path):
    m = mixed_cube(); write_fesomc(m, tmp_path, depth_m=np.ones(m.n_nodes))
    raw = np.loadtxt(tmp_path / "elem2d.out", skiprows=1, dtype=np.int64) - 1
    tri = Mesh(m.points, raw[:2, :3], raw[2:], m.node_flag)
    assert np.all(signed_area(tri) < 0)


def test_fesom2_roundtrip(tmp_path):
    m = triangulated_cube()               # each cube face split into 2 triangles
    write_fesom2(m, tmp_path, depth_m=np.full(m.n_nodes, 4000.0), levels_m=np.arange(0.0, 60.0, 10.0))   # FESOM2 needs zbar(5)
    v = (tmp_path / "aux3d.out").read_text().split()
    assert v[0] == "6" and float(v[7]) == -4000.0 and len(v) == 7 + m.n_nodes
    assert np.all(np.loadtxt(tmp_path / "nod2d.out", skiprows=1)[:, 3] == 0)
    assert same_elements(read_fesom(tmp_path), m)


def test_fesom2_roundtrip_cut_mesh(tmp_path):
    """A mesh with a boundary: flag 1 exactly on the boundary nodes, 3 columns clockwise, aux3d levels and depths
    (negative down in the file) read back, and the FESOM2 checks recomputed from the files."""
    m = triangulated_cube()
    m = Mesh(m.points, m.tris[1:], m.quads, m.node_flag)              # one triangle removed: a 3-node hole
    depth = 100.0 + 10.0 * np.arange(m.n_nodes)
    write_fesom2(m, tmp_path, depth, [0.0, 5.0, 10.0, 20.0, 30.0, 40.0])
    raw = np.loadtxt(tmp_path / "elem2d.out", skiprows=1, dtype=np.int64) - 1
    assert raw.shape == (m.n_tris, 3) and np.all(signed_area(Mesh(m.points, raw, m.quads, m.node_flag)) < 0)
    back = read_fesom(tmp_path)
    assert same_elements(back, m) and np.allclose(back.points, m.points, atol=1e-11)
    assert sorted(np.flatnonzero(back.node_flag == 1)) == sorted(np.unique(triangulated_cube().tris[0]))
    zbar, d = read_aux3d(tmp_path / "aux3d.out")
    assert np.array_equal(zbar, [0.0, 5.0, 10.0, 20.0, 30.0, 40.0]) and np.allclose(d, depth)
    assert float((tmp_path / "aux3d.out").read_text().split()[7]) == -100.0
    c = fesom2_checks(tmp_path)
    assert c["flag_equals_boundary"] and c["boundary_nodes"] == 3 and c["elem2d_columns"] == 3
    assert c["nl"] == 6 and c["unused_nodes"] == 0 and c["elems_lt2_neighbours"] == 0
    # the hole is at (161.6E, 17.5N): both rotated poles (40W 75N, 140E 75S) lie in water elements, which FESOM2
    # cannot take (singular metric); the box check finds no node within 2.5 deg of 40W 75N (nearest 39.8 deg)
    assert c["elems_containing_rotated_north_pole"] == 1 and c["elems_containing_rotated_south_pole"] == 1
    assert c["nodes_in_rotated_pole_box"] == 0 and c["stop"] == ["elems_containing_rotated_north_pole",
                                                                 "elems_containing_rotated_south_pole"]


@pytest.mark.data
def test_read_medi_v9_1(medi_dir):
    m = read_fesom(medi_dir / "mesh_v9_1")
    assert (m.n_nodes, m.n_tris, m.n_quads) == (135764, 18, 132454)
    assert (m.node_flag == 2).sum() == 89 and np.all(signed_area(m) > 0)


def test_read_gmsh41_tiny(tmp_path):
    (tmp_path / "t.msh").write_text(TINY_MSH41)
    m = read_gmsh41(tmp_path / "t.msh", 16.0, 37.0)
    assert (m.n_nodes, m.n_tris, m.n_quads) == (5, 1, 1) and np.all(signed_area(m) > 0)


@pytest.mark.data
def test_read_gmsh41_medi_v5_1_pair(medi_dir):
    g = medi_dir / "mesh_v5_1/gmsh"
    a = read_gmsh41(g / "medi_v5_1.msh", 16.0, 37.0)
    b = read_gmsh41(g / "medi_v5_1_gmsh495.msh", 16.0, 37.0)
    assert (a.n_tris, a.n_quads) == (50041, 608641) and (b.n_tris, b.n_quads) == (4, 631245)
