"""Method competition: rho_c integrals, irregular counts, the runner and the table.

rho_c = (2/pi)(K + Laplace ln h) is the net density of irregular spots (valence-3 / -5 nodes, triangles) that an ideal
quad mesh following h must have. Its integral over the closed sphere is 4 chi = 8.

The runner (`run_case`, CLI `meshom compete`) runs one method on one test case at one scale in a child process (so a
hanging method is killed after the timeout and the peak RSS is the method's own), scores the mesh in this process
(quality summary, size fidelity, irregular counts against rho_c, gates; on a seam case of meshom.cases_seams also
the "seam" block, meshom/quality/seams.py) and appends one JSON line per scored mesh to bench/competition.jsonl;
`table` renders those lines as Markdown, `family_table` the failed gate families per
method and case (the ranking statistic). The seam gates never enter the ranking: cases_seams renders them as their
own table (decisions P29).
"""
import datetime
import importlib.util
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

import meshom
from meshom import bench
from meshom.cases import CASES, STRAITS
from meshom.cases_seams import SEAMS
from meshom.config import load_gates, tolerance_gates
from meshom.delaunay import adjacency, background_mesh
from meshom.domain import Domain, in_water
from meshom.geo import R_EARTH_M
from meshom.mesh import Mesh, build_topology
from meshom.quality.metrics import check_gates, compute_quality, gate_family, size_fidelity
from meshom.size import SizeField, scaled

COMPETITION_JSONL = Path("bench/competition.jsonl")
WORK_DIR = Path(".tmp/compete")     # meshes of the last run per method x case x scale (raw.npz, smooth.npz)
TABLE_BEGIN, TABLE_END = "<!-- competition-table:begin -->", "<!-- competition-table:end -->"
NO_SMOOTH = "--smooth skipped: meshom.smooth is not available"
_ROOT = Path(meshom.__file__).resolve().parents[1]

_TWO_OVER_PI = 2.0 / np.pi

# Deferred diagnostics: callables a method appends during its (timed) generate stage when MESHOM_RUN_DIR is set, e.g.
# m14's own measurement and coast fit; _generate runs them after that stage in the stage "diagnostics", so the
# method's time excludes them (record field diagnostics_s).
DEFERRED: list = []


def _fibonacci(m: int) -> np.ndarray:
    i = np.arange(m) + 0.5
    z = 1.0 - 2.0 * i / m
    ph = np.pi * (1.0 + 5**0.5) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(ph), r * np.sin(ph), z])


def _node_rho(points: np.ndarray, tris: np.ndarray, bedges: np.ndarray, f: np.ndarray) -> tuple:
    """Per node: spherical dual area (sr, = integral of K dA on the unit sphere) and the integral of Laplace f over the
    node's barycentric dual cell (cotangent Laplacian + boundary flux of the P1 gradient)."""
    n = len(points)
    p = points[tris]                                                    # (T,3,3)
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    exc = 2.0 * np.arctan2(det, 1.0 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c)
                           + np.einsum("ij,ij->i", c, a))
    area = np.bincount(tris.ravel(), np.repeat(exc / 3.0, 3), minlength=n)
    lap = np.zeros(n)
    for k in range(3):                                                  # cot at corner k weights edge (i, j)
        i, j = tris[:, (k + 1) % 3], tris[:, (k + 2) % 3]
        u, v = p[:, (k + 1) % 3] - p[:, k], p[:, (k + 2) % 3] - p[:, k]
        cot = np.einsum("ij,ij->i", u, v) / np.linalg.norm(np.cross(u, v), axis=1)
        w = 0.5 * cot * (f[j] - f[i])
        lap += np.bincount(i, w, minlength=n) - np.bincount(j, w, minlength=n)
    if len(bedges):
        # boundary flux: grad f of the water triangle on each boundary edge, outward normal = right of n0 -> n1
        key = tris[:, [0, 1, 2]] * n + tris[:, [1, 2, 0]]
        tid, kk = np.nonzero(np.isin(key, bedges[:, 0] * n + bedges[:, 1]))
        q = p[tid]
        nrm = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
        two_a = np.linalg.norm(nrm, axis=1)
        nh = nrm / two_a[:, None]
        grad = np.zeros((len(tid), 3))
        for m in range(3):
            e = q[:, (m + 2) % 3] - q[:, (m + 1) % 3]
            grad += f[tris[tid, m]][:, None] * np.cross(nh, e) / two_a[:, None]
        i0, i1 = tris[tid, kk], tris[tid, (kk + 1) % 3]
        d = points[i1] - points[i0]
        mid = points[i0] + points[i1]
        out = np.cross(d, mid / np.linalg.norm(mid, axis=1, keepdims=True))
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        flux = np.einsum("ij,ij->i", grad, out) * np.linalg.norm(d, axis=1)
        lap += np.bincount(i0, 0.5 * flux, minlength=n) + np.bincount(i1, 0.5 * flux, minlength=n)
    return area, lap


def rho_c_integrals(domain: Domain, h: SizeField, n_target: int = 150_000) -> dict:
    """(2/pi) times the integrals of K and of Laplace ln h over the water, and of their sum split by sign.

    Keys: curv (= (2/pi) water area / R^2; 8 on the closed sphere), lap_plus, lap_minus (positive and negative parts
    of (2/pi) Laplace ln h), plus, minus (positive and negative parts of rho_c, node by node), total (= plus + minus),
    n_nodes of the integration mesh. Positive rho_c asks for valence-3 nodes or triangles, negative for valence-5
    nodes.

    Discretisation: background_mesh of the domain at the scale that gives ~n_target nodes (rho_c does not depend on
    the scale of h); cotangent Laplacian of ln h with the boundary flux of the P1 gradient added on coast edges, so
    the Laplace integral equals the flux of grad ln h through the coast (Green), and kinks of h (delta lines of
    Laplace ln h) are captured exactly in total. The coast's own geodesic-curvature term is not included. The
    node-by-node sign split nets a kink against the adjacent ramp over one cell, so lap_plus is a little below the
    delta-line value: disc 18.2 vs 20.0, band 86.9 vs 101.8 at the default n_target (~2-3 s per call).
    """
    lat = _fibonacci(200_000)
    wet = in_water(domain, lat) if domain.loops else np.ones(len(lat), dtype=bool)
    inv_h2 = (4.0 * np.pi * R_EARTH_M**2 / len(lat)) * np.sum(h(lat[wet]) ** -2.0)   # ~ node count at factor 1
    mesh = background_mesh(domain, h, factor=float(np.sqrt(inv_h2 / n_target)))
    t, k = np.nonzero(adjacency(mesh.tris) < 0)                 # half-edges without a twin: the coast, water left
    bedges = np.column_stack([mesh.tris[t, k], mesh.tris[t, (k + 1) % 3]])
    area, lap = _node_rho(mesh.points, mesh.tris, bedges, np.log(h(mesh.points)))
    rho = _TWO_OVER_PI * (area + lap)
    return {"curv": _TWO_OVER_PI * float(area.sum()),
            "lap_plus": _TWO_OVER_PI * float(lap[lap > 0].sum()),
            "lap_minus": _TWO_OVER_PI * float(lap[lap < 0].sum()),
            "plus": float(rho[rho > 0].sum()), "minus": float(rho[rho < 0].sum()), "total": float(rho.sum()),
            "n_nodes": mesh.n_nodes}



# ------------------------------------------------------------------------------------------------ irregular counts

def irregular_counts(mesh: Mesh) -> dict:
    """Irregular spots: interior nodes with 3, 5, or other than 3-5 edges (n3, n5, n_other), triangles T, Euler sum.

    euler_sum = sum over interior nodes of (4 - edges at the node) + sum over boundary nodes of (2 - elements at the
    node) + T. It equals 4 chi (chi = V - E + F of the mesh) on every edge-manifold mesh with simple boundary loops;
    on a closed mesh euler_ok also requires chi = 2 per component, i.e. euler_sum = 8 on the sphere.
    `components` counts the edge-connected pieces (closed seas after a land cut). A valence-3
    node or a triangle carries +1 (rho_c > 0), a valence-5 node -1 (rho_c < 0).
    """
    topo = build_topology(mesh)
    deg = np.bincount(topo.edges.ravel(), minlength=mesh.n_nodes)
    bnd = topo.boundary_node
    d = deg[~bnd]
    euler = int((4 - d).sum() + (2 - topo.elems_per_node[bnd]).sum() + mesh.n_tris)
    chi = int(mesh.n_nodes - len(topo.edges) + mesh.n_elems)
    closed = not bool(bnd.any())
    inner = topo.edge_elems[topo.edge_elems[:, 1] >= 0]                      # connected water components
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = coo_matrix((np.ones(len(inner)), (inner[:, 0], inner[:, 1])), shape=(mesh.n_elems, mesh.n_elems))
    k = int(connected_components(g, directed=False)[0]) if mesh.n_elems else 0
    return {"n3": int((d == 3).sum()), "n5": int((d == 5).sum()), "n_other": int(((d < 3) | (d > 5)).sum()),
            "T": mesh.n_tris, "n_interior": int(d.size), "euler_sum": euler, "chi": chi, "closed": closed,
            "components": k, "euler_ok": euler == 4 * chi and (chi == 2 * k or not closed)}


# ---------------------------------------------------------------------------------------------------------- runner

@lru_cache(maxsize=None)
def _rho_c(case: str) -> dict:
    """rho_c integrals of a case at reference scale; cached, they do not depend on the scale of h."""
    domain, h = CASES[case]
    return rho_c_integrals(domain, h)


def smooth_available() -> bool:
    return importlib.util.find_spec("meshom.smooth") is not None


def _save(mesh: Mesh, path: Path, **arrays) -> None:
    """The mesh as .npz; `arrays`: optional per-element or per-node arrays stored next to it (`meshom repair`:
    is_block, the block mask of the elements, tris first), which _load ignores."""
    extra = {} if mesh.tri_zone is None else {"tri_zone": np.asarray(mesh.tri_zone, dtype=bool)}
    np.savez(path, points=mesh.points, tris=mesh.tris, quads=mesh.quads, node_flag=mesh.node_flag, **extra, **arrays)


def _load(path: Path) -> Mesh:
    with np.load(path) as z:
        return Mesh(z["points"], z["tris"], z["quads"], z["node_flag"], z["tri_zone"] if "tri_zone" in z else None)


def _generate(method: str, domain: Domain, h: SizeField, opts: dict, smooth: bool, d: Path, post=None) -> None:
    """Child-process body shared by `meshom compete` and `meshom run`: run METHODS[method] on (domain, h), then the
    diagnostics it deferred (stage "diagnostics", see DEFERRED), then post(mesh) -> (mesh, info) when given (stage
    "post", e.g. the land cut of `meshom run`), then the common smoothing (boundary nodes fixed) when asked; save
    raw.npz, smooth.npz and the stage records stages.json into d."""
    from meshom.methods import METHODS

    os.environ["MESHOM_RUN_DIR"] = str(d)              # a method may write its own records there (m14: m14.json)
    rec: list[dict] = []
    DEFERRED.clear()
    with bench.stage("generate", rec) as meta:
        mesh = METHODS[method](domain, h, opts)
        meta.update(n_nodes=mesh.n_nodes)
    if DEFERRED:
        with bench.stage("diagnostics", rec):
            while DEFERRED:
                DEFERRED.pop(0)()
    if post is not None:
        with bench.stage("post", rec) as meta:
            mesh, info = post(mesh)
            meta.update(info)
    _save(mesh, d / "raw.npz")
    (d / "stages.json").write_text(json.dumps(rec))
    if smooth:
        from meshom.smooth import smooth as smooth_mesh

        fixed = np.flatnonzero(build_topology(mesh).boundary_node)
        with bench.stage("smooth", rec):
            zone = mesh.tri_zone
            mesh = smooth_mesh(mesh, h, fixed)
            mesh.tri_zone = zone                          # the smoothing moves nodes only
        _save(mesh, d / "smooth.npz")
        (d / "stages.json").write_text(json.dumps(rec))


def _worker(spec_json: str) -> None:
    """Child process of run_case: the test case at its scale through _generate."""
    a = json.loads(spec_json)
    domain, h0 = CASES[a["case"]]
    _generate(a["method"], domain, scaled(h0, a["scale"]), a["opts"], a["smooth"], Path(a["dir"]))


def _spawn(code: str, arg: str, timeout: float) -> tuple[str, str | None]:
    """Run `python -c code arg` with this meshom first on the path; ("ok" | "error" | "timeout", error text)."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(_ROOT), os.environ.get("PYTHONPATH")])))
    try:
        r = subprocess.run([sys.executable, "-c", code, arg], env=env, stderr=subprocess.PIPE, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", f"killed after {timeout:g} s"
    if r.returncode:
        return "error", "\n".join(r.stderr.strip().splitlines()[-15:])
    return "ok", None


def gate_results(quality: dict, irregular: dict, gates: dict) -> list[dict]:
    """check_gates on a record's quality summary plus its Euler check, applicable gates only (value not None)."""
    return [x for x in check_gates({**quality, "euler_ok": irregular["euler_ok"]}, gates) if x["value"] is not None]


def failed_families(gates_failed) -> list[str]:
    """The failed gates de-duplicated by family (gate name without the region suffix), sorted."""
    return sorted({gate_family(g) for g in gates_failed})


def _score(mesh: Mesh, domain: Domain, h: SizeField, rho_c: dict, gates: dict, strait=None, seam=None,
           tol: float | None = None) -> dict:
    """Score a mesh: quality summary with the region subsets (strait: the case's corridor, cases.STRAITS), size
    fidelity, irregular counts and the Euler check, gates. gates_failed_families = the failed gates by family; with
    N/A counted as not failed, len() of it is the ranking statistic (the same family list for every method).
    seam: the case's seam spec (cases_seams.SEAMS) -> the record's "seam" block with its own gates (SEAM_GATES),
    which never enter gates_failed (decisions P29), its size gate at the run's size tolerance `tol` (G33, G38)."""
    s = compute_quality(mesh, strait).summary
    s.update(size_fidelity(mesh, h, strait))
    ic = irregular_counts(mesh)
    ic["chi_domain"] = 2 * ic["components"] - len(domain.loops)   # per water component a sphere minus one disc per loop
    # (closed seas kept by the land cut are their own components: the Mediterranean of a 150 km mesh, 2026-10-22)
    ic["euler_ok"] = ic["euler_ok"] and ic["euler_sum"] == 4 * ic["chi_domain"]
    g = gate_results(s, ic, gates)
    failed = {x["gate"]: x["value"] for x in g if not x["passed"]}
    out = {"n_nodes": mesh.n_nodes, "quality": s, "irregular": ic, "rho_c": rho_c,
           "gates_passed": sum(x["passed"] for x in g), "gates_total": len(g),
           "gates_failed": failed, "gates_failed_families": failed_families(failed)}
    if seam is not None:
        from meshom.quality.seams import seam_block

        from meshom.quality.seams import tolerance_seam_gates

        out["seam"] = seam_block(mesh, h, seam, tolerance_seam_gates(tol))
    return out


def _round(x):
    """Floats to 6 significant digits, recursively (keeps the tracked JSON lines small)."""
    if isinstance(x, float):
        return float(f"{x:.6g}")
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return [_round(v) for v in x] if isinstance(x, list) else x


def run_case(method: str, case: str, scale: float, *, smooth: bool = False, opts: dict | None = None,
             timeout: float = 1800.0, out: Path = COMPETITION_JSONL, gates: dict | None = None,
             figures: bool = False) -> dict:
    """Run METHODS[method] on CASES[case] with h scaled by `scale`, score the mesh, append the record to `out`.

    The method runs in a child process killed after `timeout` s; wall_s and peak_rss_mb are the child's (the peak
    includes the interpreter and imports; the time includes Numba compilation, not the method's deferred diagnostics,
    which are timed as diagnostics_s). smooth=True also applies the common
    smoothing (meshom.smooth, boundary nodes fixed) in the child and scores that mesh as a second record
    (smoothed=True, wall_s = generate + smooth), returned under the key "after_smoothing". A failed or timed-out run
    is recorded with status "error" / "timeout", the error text and no scores. The run's size tolerance
    (methods.effective_tolerance: opts["size_tolerance"], else the method's default, 0.20 for m14; 0 = off) sets the
    size_fidelity_p99 gate (config.tolerance_gates: a mesh is judged against the tolerance it was built for).
    figures=True renders the standard picture set (meshom.figures.render_run) of each scored mesh into
    <run dir>/figures/raw/ and figures/smooth/.
    Returns the record of the raw mesh.
    """
    if smooth and not smooth_available():
        print(NO_SMOOTH)
        smooth = False
    opts = opts or {}
    work = (WORK_DIR / f"{method}_{case}_s{scale:g}").resolve()
    work.mkdir(parents=True, exist_ok=True)
    for f in ("raw.npz", "smooth.npz", "stages.json"):
        (work / f).unlink(missing_ok=True)
    for f in (work / "figures").glob("*/*"):                    # pictures of an earlier run of this method x case
        f.unlink()
    spec = json.dumps({"method": method, "case": case, "scale": scale, "smooth": smooth, "opts": opts,
                       "dir": str(work)})
    status, err = _spawn("import sys; from meshom.compete import _worker; _worker(sys.argv[1])", spec, timeout)

    stages = json.loads((work / "stages.json").read_text()) if (work / "stages.json").exists() else []
    stages = {x["stage"]: x for x in stages}
    domain, h0 = CASES[case]
    h = scaled(h0, scale)
    from meshom.methods import effective_tolerance
    tol = effective_tolerance(method, opts)
    gates = tolerance_gates(load_gates() if gates is None else gates, tol, gates)   # G38; a caller's gate stays
    recs = []
    for smoothed, stage, f in [(False, "generate", "raw.npz")] + ([(True, "smooth", "smooth.npz")] if smooth else []):
        rec = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "method": method, "case": case,
               "scale": float(scale), "smoothed": smoothed, "opts": opts, **bench.machine_info()}
        if stage in stages and (work / f).exists():
            gen = stages["generate"]["wall_s"]
            rec.update(status="ok", wall_s=gen + (stages[stage]["wall_s"] if smoothed else 0.0), generate_s=gen,
                       peak_rss_mb=stages[stage]["peak_rss_mb"],
                       **({"diagnostics_s": stages["diagnostics"]["wall_s"]} if "diagnostics" in stages else {}),
                       **_score(_load(work / f), domain, h, _rho_c(case), gates, STRAITS.get(case),
                                SEAMS.get(case), tol))
        else:
            rec.update(status=status, error=err)
        recs.append(_round(rec))
    bench.append_jsonl(recs, out)
    if figures:
        from meshom.figures import render_run

        for r, f in zip(recs, ("raw", "smooth")):
            if r["status"] == "ok":
                render_run(_load(work / f"{f}.npz"), domain, h, work / "figures" / f,
                           title=f"{method} | {case} | scale {scale:g} | {'smoothed' if r['smoothed'] else 'raw'}")
    if smooth:
        recs[0]["after_smoothing"] = recs[1]
    return recs[0]


# ----------------------------------------------------------------------------------------------------------- table

_HEAD = ("method", "case", "scale", "smoothed", "nodes", "tris %", "quad θmax max / p99.9", "coast", "interior",
         "tri θmax max", "elem ratio p99 / max", "\\|ln s/h\\| p99", "\\|ln s/h\\| p99 coast / coast3 / strait",
         "elem ratio p99 coast3 / strait", "n3 / n5 / T", "ρ+ / ρ−", "Euler", "gates", "failed gate families",
         "time s", "peak MB")


def _f(v, nd: int) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _row(r: dict) -> list[str]:
    key = [r["method"], r["case"], f"{r['scale']:g}", "yes" if r["smoothed"] else "no"]
    if r["status"] != "ok":
        return key + [r["status"]] + [""] * (len(_HEAD) - len(key) - 1)
    q, ic, rho = r["quality"], r["irregular"], r["rho_c"]

    def qa(reg: str) -> str:
        x = q["quad_max_angle_deg" + reg]
        return f"{_f(x['max'], 1)} / {_f(x['p99_9'], 1)}"

    def st(name: str, regs: tuple, nd: int) -> str:
        return " / ".join(_f(q.get(name + reg, {}).get("p99"), nd) for reg in regs)

    er = q["elem_size_ratio"]
    fam = r.get("gates_failed_families", failed_families(r["gates_failed"]))
    return key + [
        str(q["n_nodes"]), _f(100.0 * q["n_tris"] / max(q["n_elems"], 1), 1), qa(""), qa("_coast"), qa("_interior"),
        _f(q["tri_max_angle_deg"]["max"], 1), f"{_f(er['p99'], 2)} / {_f(er['max'], 2)}",
        _f(q["size_fidelity"]["p99"], 3), st("size_fidelity", ("_coast", "_coast3", "_strait"), 3),
        st("elem_size_ratio", ("_coast3", "_strait"), 2), f"{ic['n3']} / {ic['n5']} / {ic['T']}",
        f"{rho['plus']:.1f} / {rho['minus']:.1f}",
        f"{ic['euler_sum']} / {4 * ic['chi_domain']}" + ("" if ic["euler_ok"] else " FAIL"),
        f"{r['gates_passed']}/{r['gates_total']}", ", ".join(_FAM_SHORT.get(x, x) for x in fam) or "—",
        _f(r["wall_s"], 1), _f(r["peak_rss_mb"], 0)]


# short names of the gate families in the table
_FAM_SHORT = {"quad_max_angle_hard_deg": "quad θmax", "quad_max_angle_target_deg": "quad θ p99.9",
              "tri_max_angle_hard_deg": "tri θmax", "tri_max_angle_target_deg": "tri θ p99",
              "tri_min_angle_hard_deg": "tri θmin", "quad_sj_hard": "quad SJ", "elem_size_ratio_p99": "ratio p99",
              "elem_size_ratio_max": "ratio max", "size_fidelity_p99": "size p99",
              "elem_neighbours_min": "neighbours", "elems_per_node_max": "elems/node",
              "tri_cluster_max": "tri clusters", "boundary_edges_max": "boundary edges",
              "pinch_nodes_max": "pinch", "euler_ok": "Euler"}


def _last(path: Path) -> list[dict]:
    """The last record per method x case x scale x smoothed, sorted by method, case, scale (largest first)."""
    path = Path(path)
    last: dict = {}
    for line in (path.read_text() if path.exists() else "").splitlines():
        if line.strip():
            r = json.loads(line)
            last[(r["method"], r["case"], r["scale"], r["smoothed"])] = r
    order = list(CASES)
    return sorted(last.values(), key=lambda r: (r["method"], order.index(r["case"]) if r["case"] in order
                                                else len(order), -r["scale"], r["smoothed"]))


def table(path: Path) -> str:
    """Markdown table of the records in `path`: one row per method x case x scale x smoothed (the last record)."""
    lines = ["| " + " | ".join(_HEAD) + " |", "|" + "---|" * len(_HEAD)]
    lines += ["| " + " | ".join(_row(r)) + " |" for r in _last(path)]
    return "\n".join(lines) + "\n"


def record_families(r: dict, gates: dict | None = None) -> list[str] | None:
    """Failed gate families of a scored record: as stored (gates=None) or re-evaluated on its summary under `gates`
    (e.g. another boundary_rule); None for a failed run."""
    if r.get("status") != "ok":
        return None
    if gates is None:
        return r.get("gates_failed_families", failed_families(r["gates_failed"]))
    return failed_families(x["gate"] for x in gate_results(r["quality"], r["irregular"], gates) if not x["passed"])


def family_table(path: Path, scale: float = 1.0, smoothed: bool = False, gates: dict | None = None) -> str:
    """Markdown table: failed gate families per method and case at one scale, raw or smoothed, sorted by their sum
    over the cases (the ranking statistic: the same family list for every method, N/A = not failed)."""
    fam: dict = {}
    for r in _last(path):
        if r["scale"] == scale and r["smoothed"] == smoothed:
            fam.setdefault(r["method"], {})[r["case"]] = record_families(r, gates)
    cases = [c for c in CASES if any(c in per for per in fam.values())]   # only cases with records (seam cases apart)
    rows = sorted(fam.items(), key=lambda kv: (sum(len(v or ()) + 99 * (v is None) for v in kv[1].values()), kv[0]))
    lines = ["| method | " + " | ".join(cases) + " | sum |", "|---|" + "---|" * (len(cases) + 1)]
    for m, per in rows:
        cells = []
        for c in cases:
            v = per.get(c, "missing")
            cells.append("—" if v == "missing" else "run failed" if v is None else
                         f"{len(v)}" + (": " + ", ".join(_FAM_SHORT.get(x, x) for x in v) if v else ""))
        total = sum(len(v) for v in per.values() if v is not None)
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {total} |")
    return "\n".join(lines) + "\n"


def write_table(md_path: Path, text: str, begin: str = TABLE_BEGIN, end: str = TABLE_END) -> None:
    """Put `text` between the table markers (begin, end) of md_path; appended (or a new file) when they are
    missing."""
    md_path = Path(md_path)
    block = f"{begin}\n{text}{end}"
    old = md_path.read_text() if md_path.exists() else ""
    if begin in old:
        new = old[: old.index(begin)] + block + old[old.index(end) + len(end):]
    else:
        new = old + ("\n" if old else "") + block + "\n"
    md_path.write_text(new)


def one_line(r: dict) -> str:
    """Short console summary of a record."""
    head = f"{r['method']} {r['case']} x{r['scale']:g}{' smoothed' if r['smoothed'] else ''}: "
    if r["status"] != "ok":
        return head + f"{r['status']}\n{r['error']}"
    q, ic = r["quality"], r["irregular"]
    peak = "-" if r.get("peak_rss_mb") is None else f"{r['peak_rss_mb']:.0f}"
    return head + (f"{q['n_nodes']} nodes, {q['n_tris']} tris, {q['n_quads']} quads; n3/n5/T {ic['n3']}/{ic['n5']}/"
                   f"{ic['T']}, Euler {ic['euler_sum']} ({'ok' if ic['euler_ok'] else 'FAIL'}); gates "
                   f"{r['gates_passed']}/{r['gates_total']}; {r['wall_s']:.1f} s, {peak} MB")
