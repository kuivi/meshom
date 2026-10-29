"""meshom web UI: a Python standard-library HTTP server + deck.gl (GlobeView) in the browser.

    meshom ui [--config FILE] [--runs DIR ...] [--refs DIR ...] [--port 8765] [--allow-submit]
    python -m ui.web.server [same options] [--mesh PATH] [--host 127.0.0.1] [--run-cmd TEMPLATE]

The server answers JSON and small binary arrays; the page draws. It reads and writes the meshom config (the core's
schema, meshom/config.py: the file `meshom run CONFIG` reads) and never builds a mesh: "Run" writes a run folder with
the config and an sbatch script that runs `meshom run` and submits it only with --allow-submit (ui/web/runs.py).
Binary formats: ui/web/payload.py.

Access: `meshom ui` makes a random token (32 hex; `--token T` sets one, `--no-token` switches it off) and prints
http://<host>:<port>/?token=<token>. Every request but /static/* needs it: the token in the query string (the first
page load; the reply sets the cookie meshom_token_<port>, which the browser then sends with every /api/... call), the
cookie, or the header X-Meshom-Token; anything else gets 403 and a one-line page. `--host` (default 127.0.0.1) may be
a compute node's hostname (`--host $(hostname)`): the server then listens on the cluster-internal interface and
your own computer reaches it through the login node (`ssh -N -L 8765:<node>:8765 <user>@levante.dkrz.de`). A host other than the
loopback needs the token (`--no-token` is refused there); never give a public address.

Endpoints (GET unless marked POST):
  /api/config                 the loaded config (JSON of meshom.config.to_dict) and its TOML; POST saves {config, path?}
  POST /api/toml              {config} -> TOML text (no save); {text} -> parsed value (the [blocks] editor)
  POST /api/preview           {config, step, polar: [north|south], limited} -> h and "which layer wins" PNGs + stats;
                              limited: the batch run's h (meshom.run.size_field: the core's gradient limiter, cached
                              per config hash), with the limiter's record
  POST /api/preview/window    {config, bounds: [w, e, s, n], polygon?, step, h_lo, h_hi, polar, limited} -> PNGs
  POST /api/rho               {config, step, polar, limited} -> rho_c h^2 PNG + integrals (ui/common/rho.py)
  /api/coast/sources          coastline files found (GSHHS levels, Natural Earth, other)
  /api/coast.bin?spacing_m=   coastline of the current config as polylines
  /api/methods                meshom.methods.METHODS with each module's DEFAULTS
  /api/mesh/info              the current mesh: counts, irregular nodes, its quality.json record, what it is
  /api/mesh/overview.png      worst value per pixel (field=q|size_ratio|kind|depth: the deepest|size_km: the largest, proj=north|south)
  /api/mesh/mesh.bin          indexed mesh: sel=all|worst|window|tris (n=, lon=, lat=, r_km=; edit=1: with global
                              node ids and node_flag, for the hand editor)
  /api/mesh/irregular.bin     irregular interior nodes (class 1 n3, 2 n5, 3 other)
  /api/mesh/hist              histograms of quad / triangle θmax, element size ratio, |ln s/h| per edge (bins from the
                              overlay, percentiles and gates from quality.json when the mesh has one)
  /api/mesh/worst.json?n=     the worst elements (by q) with centroids
  POST /api/mesh/load         {path} -> make a run folder's mesh, a reference mesh or a mesh file the current mesh
  POST /api/compare           {paths} -> the quality records of 2-4 meshes side by side (rows of numbers, the best
                              marked; the gates every mesh has), from quality.json / repaired-NN.json only
  POST /api/mesh/edit         {ops} -> the hand edits of the page replayed on the current mesh, written as
                              edited-NN.npz / .json next to it (ui/web/edit.py; reference meshes are not edited)
  /api/mesh/write?path=       Write model files: a mesh's run folder, its meshes and their last write (ui/web/write.py)
  POST /api/mesh/write        {path, format, fesom_ini} -> meshom.run.write_mesh in a thread, one at a time (409)
  /api/mesh/write/job         the write: idle | running (elapsed_s) | done (the folder, checks, gates, straits) | error
  /api/runs                   run folders under --runs (any depth); /api/runs/log?path=; /api/runs/blocks?path=
  /api/dirs?path=             a folder's subfolders (the page's folder browser); no path: the places to start
  POST /api/profile           {config, line, n, limited, mesh}: distance, h and the bottom (GEBCO) along a line (Tools);
                              with the shown mesh's path, the nodes of its elements the line crosses and their depth
  POST /api/mesh/depth/select {path, box | polygon}: the shown mesh's nodes inside, their depth statistics (depth.py)
  POST /api/mesh/depth/edit|undo|remove, GET /api/mesh/depth/edits: the mesh's node edits (depth.py)
  /api/size/variants          h variant files (size-only TOML) next to the config and in .tmp/ui
  POST /api/size/save         {config, path}: the page's [size] as an h variant file (meshom.config.dumps_size)
  POST /api/size/load         {path}: an h variant file as the page's [size]
  POST /api/runs/roots        {"add": folder, "deep": with its subfolders} / {"remove": folder}: the folders
                              searched for runs (a root again: its deep tick set)
  /api/refs                   reference meshes under --refs (mesh.npz + quality.json per folder)
  POST /api/run               {config, resources?} -> run folder + sbatch script (+ job id with --allow-submit)
"""
import argparse
import base64
import hashlib
import hmac
import zlib
import ipaddress
import secrets
import socket
import json
import re
import sys
import threading
import time
import tomllib
import traceback
from dataclasses import replace
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from meshom import config as mconfig
from ui.common import coast as coastmod
from ui.common import meshview, scenario
from ui.web import depth, payload, runs, write

STATIC = Path(__file__).resolve().parent / "static"
CTYPES = {"js": "text/javascript", "css": "text/css", "html": "text/html; charset=utf-8", "png": "image/png"}


def b64png(rgba: np.ndarray) -> str:
    return base64.b64encode(scenario.png_bytes(rgba)).decode()


def polar_xy(lon, lat, north: bool):
    colat = np.radians(90.0 - lat if north else 90.0 + lat)
    r = np.minimum(np.tan(colat / 2), 4.0)
    lo = np.radians(lon)
    return r * np.sin(lo), (-r if north else r) * np.cos(lo)


def polar_inv(x, y, north: bool):
    colat = np.degrees(2 * np.arctan(np.hypot(x, y)))
    lon = np.degrees(np.arctan2(x, -y)) if north else np.degrees(np.arctan2(x, y))
    return lon, (90 - colat) if north else (colat - 90)


def polar_rgba(rgba: np.ndarray, north: bool, n: int = 720) -> np.ndarray:
    """Polar stereographic raster of a global lon-lat RGBA image (rows north first): plane coordinates in [-1, 1]
    reach the equator (x = r sin lon, y = -r cos lon in the north, +r cos lon in the south, r = tan(colat / 2));
    nearest lookup. Row 0 is the top of the image (+y)."""
    y, x = np.meshgrid(np.linspace(1, -1, n), np.linspace(-1, 1, n), indexing="ij")
    lon, lat = polar_inv(x, y, north)
    ny, nx = rgba.shape[:2]
    i = np.clip(((90 - lat) / (180 / ny)).astype(int), 0, ny - 1)
    j = np.clip(((lon + 180) / (360 / nx)).astype(int), 0, nx - 1)
    out = rgba[i, j].copy()
    out[np.hypot(x, y) > 1] = 0
    return out


def polar_window(rgba: np.ndarray, bounds, north: bool, n: int = 600) -> tuple[np.ndarray, list]:
    """A window's lon-lat image (rows north first, bounds w, e, s, n) on the polar plane: the image over the plane
    bounding box of the window [x0, y0, x1, y1] (row 0 at y1), transparent outside the window."""
    w, e, s, nn = bounds
    t = np.linspace(0, 1, 90)
    bl = np.r_[w + (e - w) * t, np.full(90, e), e - (e - w) * t, np.full(90, w)]
    bp = np.r_[np.full(90, s), s + (nn - s) * t, np.full(90, nn), nn - (nn - s) * t]
    x, y = polar_xy(bl, bp, north)
    pole = (nn >= 89.999) if north else (s <= -89.999)
    x0, x1 = min(x.min(), 0 if pole else x.min()), max(x.max(), 0 if pole else x.max())
    y0, y1 = min(y.min(), 0 if pole else y.min()), max(y.max(), 0 if pole else y.max())
    span = max(x1 - x0, y1 - y0, 1e-9)
    nx_, ny_ = max(int(n * (x1 - x0) / span), 2), max(int(n * (y1 - y0) / span), 2)
    Y, X = np.meshgrid(np.linspace(y1, y0, ny_), np.linspace(x0, x1, nx_), indexing="ij")
    lon, lat = polar_inv(X, Y, north)
    lon_u = w + (lon - w) % 360.0
    inside = (lon_u <= e) & (lat >= s) & (lat <= nn)
    ih, iw = rgba.shape[:2]
    i = np.clip(((nn - lat) / (nn - s) * ih).astype(int), 0, ih - 1)
    j = np.clip(((lon_u - w) / (e - w) * iw).astype(int), 0, iw - 1)
    out = rgba[i, j].copy()
    out[~inside] = 0
    return out, [float(x0), float(y0), float(x1), float(y1)]


jsonable = scenario.jsonable


_METHODS: list = []


def readout_grid(p) -> dict:
    """The preview's h (km, float32, NaN on land) and its winning layer (uint8, 255 on land) as deflated base64 arrays
    on the preview grid (cell centres lon0 + (i + 0.5) step, lat0 + (j + 0.5) step), for the page's cursor readout."""
    h = np.where(p.water, p.h / 1000.0, np.nan).astype(np.float32)
    win = np.where(p.water, p.winner, 255).astype(np.uint8)
    step = float(p.lon[1] - p.lon[0]) if len(p.lon) > 1 else 1.0
    pack = lambda a: base64.b64encode(zlib.compress(np.ascontiguousarray(a).tobytes(), 6)).decode()  # noqa: E731
    return {"nx": len(p.lon), "ny": len(p.lat), "lon0": float(p.lon[0] - step / 2), "lat0": float(p.lat[0] - step / 2),
            "step": step, "h": pack(h), "winner": pack(win)}


# variants composed of another method and a stage: meshom.methods.defaults(name) is not
# the options they take; their form says so
COMPOSED = {"m01_triangles", "m11_local_ops_tri", "m01_triangles_fit", "m02_linf_frontal_tri_fit", "ref_jigsaw_fit"}


def methods_catalog() -> list[dict]:
    """METHODS with module, a one-line description, the options it runs with when given none
    (meshom.methods.defaults: the module's DEFAULTS with the variant's OVERRIDES on top), the overrides, and whether it
    is a composed variant (imported once: ~2 s)."""
    if not _METHODS:
        from meshom.methods import METHODS, OVERRIDES, defaults
        for name, fn in METHODS.items():
            mod = sys.modules[fn.__module__]
            doc = (fn.__doc__ or mod.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")
            _METHODS.append({"name": name, "module": fn.__module__.rsplit(".", 1)[-1], "doc": doc[:400],
                             "defaults": jsonable(defaults(name)), "overrides": jsonable(OVERRIDES.get(name, {})),
                             "composed": name in COMPOSED or name.endswith("_rep")})
    return _METHODS


class ActionLog:
    """The action log (JSON lines, appended): every API request when it starts ("req") and when it ends ("done": status,
    ms), and the page's events (ui/web/static/log.js: clicks, choices, view changes, errors, a heartbeat of the map),
    to trace what happened before the page froze. Thread-safe; the token never goes in."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._next = 0

    def write(self, rec: dict) -> None:
        at = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
        line = json.dumps({"at": at, **rec, **({"at": at} if "at" in rec else {})}, default=str)   # the server's time wins
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")

    def next_id(self) -> int:
        with self._lock:
            self._next += 1
            return self._next


class State:
    def __init__(self, config: Path, mesh: Path, run_roots: list[Path], allow_submit: bool = False,
                 run_cmd: str = runs.RUN_CMD, ref_roots: list[Path] | None = None):
        self.config_path = config
        self.loaded = scenario.load(config)          # as read from disk: the type template of from_json()
        self.cfg = self.loaded
        self.mesh_path = mesh
        self.run_roots = run_roots
        self.ref_roots = ref_roots if ref_roots is not None else runs.REF_ROOTS
        self.allow_submit = allow_submit
        self.run_cmd = run_cmd
        self.lock = threading.Lock()
        self.heavy = threading.Lock()                # the gradient limiter (numba parallel kernels): one at a time,
        #                                              while the quick previews go on under self.lock
        self.opened_from: str | None = None          # the run folder whose config the page opened
        self.token: str | None = None                # None: no access control (tests, scripts in this process)
        self.cookie = "meshom_token"
        self.deep: set[Path] = set()                 # run roots searched with their subfolders (the page's tick)
        self.ready: set[tuple[Path, bool]] = set()   # (run root, deep) searched once (first_scans)
        self.scanning: dict[tuple[Path, bool], threading.Thread] = {}
        self.log: ActionLog | None = None            # None: no action log (tests, scripts in this process)
        self.writer = write.Writer()                 # Write model files: one write_mesh at a time, in a thread

    def first_scans(self, wait_s: float = 10.0) -> list[Path]:
        """The run roots searched at least once; a root's first search (minutes for a large tree on Lustre: 11 500
        folders, 2 400 runs in 280 s; again 0.6 s) runs in a thread, awaited up to wait_s in all (a small folder is
        listed at once even on a busy node: its first search imports the core, > 2 s on one CPU)."""
        keys = [(r, r in self.deep) for r in self.run_roots]
        for k in keys:
            if k not in self.ready and k not in self.scanning:
                t = threading.Thread(target=self._first_scan, args=(k,), daemon=True)
                self.scanning[k] = t
                t.start()
        end = time.monotonic() + wait_s
        for t in list(self.scanning.values()):
            t.join(max(0.0, end - time.monotonic()))
        return [r for r, d in keys if (r, d) in self.ready]

    def _first_scan(self, k: tuple[Path, bool]) -> None:
        try:
            runs.scan_runs([k[0]], {k[0]} if k[1] else set())
        finally:
            self.ready.add(k)
            self.scanning.pop(k, None)

    def roots_json(self) -> dict:
        return {"roots": [str(r) for r in self.run_roots], "deep": [str(r) for r in self.run_roots if r in self.deep]}

    def variant_dir(self) -> Path:
        """Where h variants are saved by default: next to the config (the example's saves: .tmp/ui)."""
        p = Path(self.config_path)
        return (scenario.SAVE_DEFAULT if p.resolve() == scenario.EXAMPLE.resolve() else p).resolve().parent

    def places(self) -> list[dict]:
        """Where the folder browser starts: the run roots, this checkout's and the main checkout's .tmp, the
        checkout ../meshom next to it (its .tmp and .worktrees hold test meshes) when this is another clone, the
        folder next to the repository; those that exist."""
        main = scenario.main_checkout()
        core = main.parent / "meshom"
        cands = [*self.run_roots, scenario.REPO / ".tmp", main / ".tmp", main / ".worktrees", core / ".tmp",
                 core / ".worktrees", main.parent]
        out, seen = [], set()
        for p in map(Path, cands):
            p = p.resolve()
            if p.is_dir() and p not in seen:
                seen.add(p)
                out.append({"path": str(p), "root": p in self.run_roots})
        return out

    def coast(self, cfg: mconfig.Config | None = None) -> coastmod.Coast | None:
        """The UI coast (display polylines, land mask) of the config; None for the closed sphere."""
        c = (cfg or self.cfg).coast
        return None if c.source == "none" else coastmod.load(str(scenario.resolve(c.source)), float(c.min_area_km2),
                                                               c.antarctica)

    def take(self, cfg: dict | None) -> mconfig.Config:
        if cfg is not None:
            self.cfg = scenario.from_json(cfg, self.loaded)
        return self.cfg

    def mesh_config(self, mf: Path | None = None) -> tuple[mconfig.Config, Path] | None:
        """The config a mesh was built with: config.resolved.toml written by `meshom run`, else the config.toml the UI
        wrote, next to the mesh file; None for meshes without one (compete folders, reference meshes)."""
        d = (mf or meshview.mesh_file(self.mesh_path)).parent
        for name in ("config.resolved.toml", "config.toml"):
            if (d / name).exists():
                try:
                    return mconfig.load(d / name), d / name
                except (mconfig.ConfigError, tomllib.TOMLDecodeError):      # a draft of an older UI
                    continue
        return None

    def mesh_h(self, mf: Path):
        """The size field a mesh was built for: the run's config through the core's size_field (limiter and scale
        included), else the compete case of the folder name (<method>_<case>_s<scale>); None if unknown (reference
        meshes)."""
        mc = self.mesh_config(mf)
        if mc is not None:
            with self.heavy:
                return scenario.h_field(mc[0], limited=True)
        d = mf.parent
        from meshom.cases import CASES
        from meshom.size import scaled
        m = re.fullmatch(r".+_(" + "|".join(map(re.escape, CASES)) + r")_s([0-9.eE+-]+)", d.name)
        return scaled(CASES[m.group(1)][1], float(m.group(2))) if m else None

    def mv(self) -> meshview.MeshView:
        mf = meshview.mesh_file(self.mesh_path)
        return meshview.load(self.mesh_path, lambda: self.mesh_h(mf))

    def quality(self, mf: Path | None = None) -> dict | None:
        """The quality.json record of the current mesh (or of mesh file mf): "smooth" for smooth.npz, else "raw"
        (`meshom run` writes {"raw": record, "smooth": record}; a reference mesh's folder holds {"raw": record} next
        to mesh.npz); a repaired-NN.npz: the "after" of repaired-NN.json."""
        mf = mf or meshview.mesh_file(self.mesh_path)
        if mf.name.startswith("repaired-"):                 # meshom repair: NAME.json's whole-mesh score after it
            return (runs.read_json(mf.with_suffix(".json")) or {}).get("after")
        if mf.name.startswith("edited-"):                   # edited by hand: not scored (quality.json is raw's, smooth's)
            return None
        q = runs.read_json(mf.parent / "quality.json")
        if not isinstance(q, dict):
            return None
        return q.get(mf.stem if mf.stem in ("smooth", "optimized") else "raw")


class Handler(BaseHTTPRequestHandler):
    state: State
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200, headers: dict | None = None) -> None:
        self._code = code
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _given_token(self) -> tuple[str | None, bool]:
        """(the token the request carries, whether it came in the query string)."""
        q = parse_qs(urlparse(self.path).query).get("token")
        if q:
            return q[0], True
        if self.headers.get("X-Meshom-Token"):
            return self.headers["X-Meshom-Token"], False
        try:
            c = SimpleCookie(self.headers.get("Cookie", ""))
        except CookieError:
            return None, False
        return (c[self.state.cookie].value if self.state.cookie in c else None), False

    def _allowed(self) -> bool:
        """True when the server has no token, the path is a static file, or the request carries the token."""
        tok = self.state.token
        if tok is None or urlparse(self.path).path.startswith("/static/"):
            return True
        given, _ = self._given_token()
        return given is not None and hmac.compare_digest(given.encode(), tok.encode())

    def _forbidden(self) -> None:
        self._send(b"meshom ui: 403, no or wrong token: open the URL meshom ui printed (http://.../?token=...)\n",
                   "text/plain; charset=utf-8", 403)

    def _json(self, obj, code: int = 200) -> None:
        self._send(json.dumps(obj).encode(), "application/json", code)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        self._logged(self._get)

    def do_POST(self):
        self._logged(self._post)

    def _logged(self, f) -> None:
        """f under the token check, its errors reported to the page (and stderr) instead of a dropped connection, and,
        with an action log, the request written when it starts and when it ends."""
        if not self._allowed():
            return self._forbidden()
        u, log = urlparse(self.path), self.state.log
        if log is None or not u.path.startswith("/api/") or u.path == "/api/log":
            log = None
        else:
            rid, t0 = log.next_id(), time.perf_counter()
            q = "&".join(f"{k}={v[0]}" for k, v in parse_qs(u.query).items() if k != "token")
            log.write({"ev": "req", "id": rid, "method": self.command, "path": u.path, **({"q": q} if q else {}),
                       "thread": threading.get_ident() % 10000})
        self._code = None
        try:
            f()
        except Exception as e:
            traceback.print_exc()
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            if log:
                log.write({"ev": "error", "id": rid, "error": f"{type(e).__name__}: {e}"})
        finally:
            if log:
                log.write({"ev": "done", "id": rid, "path": u.path, "status": self._code,
                           "ms": round((time.perf_counter() - t0) * 1000)})

    # ------------------------------------------------------------------------------------------------------ GET
    def _get(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        st = self.state
        if u.path in ("/", "/index.html"):
            given, in_query = self._given_token()
            cookie = ({"Set-Cookie": f"{st.cookie}={given}; Path=/; HttpOnly; SameSite=Strict"}
                      if st.token is not None and in_query else None)
            return self._send((STATIC / "index.html").read_bytes(), CTYPES["html"], headers=cookie)
        if u.path.startswith("/static/"):
            f = STATIC / u.path.removeprefix("/static/")
            return self._send(f.read_bytes(), CTYPES.get(f.suffix[1:], "application/octet-stream"))
        if u.path == "/api/config":
            return self._json({"path": str(st.config_path), "opened_from": st.opened_from, "config": scenario.to_json(st.cfg),
                               "toml": scenario.dumps(st.cfg), "layer_colours": scenario.LAYER_COLOURS,
                               "allow_submit": st.allow_submit, "run_cmd": st.run_cmd, "resources": runs.RESOURCES,
                               "mesh": str(st.mesh_path), "size_from": st.cfg.size_from or None,
                               "variant_dir": str(st.variant_dir())})
        if u.path == "/api/size/variants":          # h variant files: size-only TOML next to the config and in .tmp/ui
            return self._json({"variants": scenario.size_variants([st.variant_dir(), scenario.SAVE_DEFAULT.parent])})
        if u.path == "/api/coast/sources":
            none = st.cfg.coast.source == "none"
            src = scenario.resolve(scenario.DEFAULT_COAST if none else st.cfg.coast.source)
            return self._json({"current": "none" if none else str(src), "sources": coastmod.sources(src)})
        if u.path == "/api/coast.bin":
            sp = float(q["spacing_m"]) if q.get("spacing_m") else None
            mc = st.mesh_config() if q.get("of") == "mesh" else None                # the current mesh's own coast
            c = st.coast(mc[0]) if mc else st.coast()
            return self._send(payload.coast_bin(c.polylines(sp) if c else []), "application/octet-stream")
        if u.path == "/api/methods":
            return self._json({"methods": methods_catalog()})
        if u.path == "/api/mesh/write":             # Write model files: what a mesh's run folder can write
            try:
                return self._json(jsonable(write.targets(Path(q["path"]), st.writer.job())))
            except ValueError as e:                  # no config.resolved.toml
                return self._json({"error": str(e)}, 400)
        if u.path == "/api/mesh/write/job":
            return self._json(jsonable(st.writer.job()))
        if u.path == "/api/mesh/depth/edits":        # the node tool: the mesh's depth edits
            return self._json(jsonable({"file": str(meshview.edits_file(Path(q["path"]))),
                                        "edits": depth.edits_list(Path(q["path"]))}))
        if u.path.startswith("/api/mesh/"):
            return self._mesh(u.path.removeprefix("/api/mesh/"), q)
        if u.path == "/api/runs":
            ready = st.first_scans()
            found, scans = runs.scan_runs(ready, st.deep)
            scans += [{"path": str(r), "scanning": True} for r in st.run_roots if r not in ready]
            out = json.dumps({**st.roots_json(), "runs": found, "scans": scans, "allow_submit": st.allow_submit}).encode()
            tag = hashlib.blake2b(out, digest_size=8).hexdigest()       # the page asks again with known=tag
            if q.get("known") == tag:
                return self._json({"same": True, "tag": tag})
            return self._send(out, "application/json", headers={"X-Meshom-Tag": tag})
        if u.path == "/api/dirs":                    # the page's folder browser ("Add folder" in Runs)
            if not q.get("path"):
                return self._json({"places": st.places()})
            p = Path(q["path"]).expanduser().resolve()
            if not p.is_dir():
                return self._json({"error": f"no such folder: {p}"}, 400)
            return self._json(runs.list_dirs(p))
        if u.path == "/api/runs/log":
            p = Path(q["path"])
            lf = runs.log_file(p) if p.is_dir() else p
            return self._json({"log": str(lf) if lf else None, "tail": runs.tail(lf, int(q.get("n", 40)))})
        if u.path == "/api/runs/blocks":
            return self._json(runs.blocks(Path(q["path"])))
        if u.path == "/api/refs":
            return self._json({"roots": [str(r) for r in st.ref_roots], "refs": runs.list_refs(st.ref_roots)})
        return self._json({"error": "not found"}, 404)

    def _mesh(self, what: str, q: dict):
        st = self.state
        t0 = time.perf_counter()
        mv = st.mv()
        if what == "info":
            T = len(mv.tris)
            rec = st.quality()
            mc = st.mesh_config()
            sl, lat = np.sort(mv.lonlat[:, 0].astype(np.float64)), mv.lonlat[:, 1]
            gaps = np.diff(np.r_[sl, sl[0] + 360.0])                   # the widest empty longitude band ends at the
            k = int(gaps.argmax())                                       # mesh's west edge
            gap, west = float(gaps[k]), float(sl[(k + 1) % len(sl)])
            coast = (None if mc is None or mc[0].coast.source == "none" else
                     {"source": mc[0].coast.source, "min_area_km2": mc[0].coast.min_area_km2,
                      "antarctica": mc[0].coast.antarctica, "config": str(mc[1])})
            return self._json({"path": mv.path, "n_nodes": len(mv.lonlat), "n_tris": T, "n_quads": len(mv.quads),
                               "extent": {"global": gap < 30.0, "west": west,
                                          "width": float(360.0 - gap), "south": float(lat.min()), "north": float(lat.max())},
                               "cell_km": mv.cell_km(), "coast": coast, "config": str(mc[1]) if mc else None,
                               "repair": (runs.repair_summary(runs.read_json(Path(mv.path).with_suffix(".json")) or {})
                                          if Path(mv.path).name.startswith("repaired-") else None),
                               "gates": jsonable(runs.gate_table(rec, mc[0].gates_all() if mc else mconfig.DEFAULT_GATES)),
                               "q_p99": float(np.percentile(mv.fields["q"], 99)), "q_max": float(mv.fields["q"].max()),
                               "irregular": mv.counts(), "has_ln_sh": mv.edge_ln is not None,
                               "has_depth": mv.depth_hi is not None,     # meshview.depth_of: colour "depth"
                               "depth_source": mv.depth_source,
                               "depth": None if mv.depth_hi is None else {"lo": 0, "hi": mv.depth_hi},
                               "size_km": dict(zip(("lo", "hi"), mv.size_range)),       # colour "size_km", log
                               "m14": (Path(mv.path).parent / "m14.json").exists(),
                               "run_dir": str(Path(mv.path).parent), "load_s": round(time.perf_counter() - t0, 3),
                               "quality": jsonable(runs.record_summary(rec)) if rec else None})
        if what == "overview.png":
            f = q.get("field", "q")
            img = np.ascontiguousarray(meshview.quality_rgba(mv.overview(f, float(q.get("step", 0.5))), f,
                                                             mv.depth_hi, mv.size_range)[::-1])
            if q.get("proj") in ("north", "south"):
                img = polar_rgba(img, q["proj"] == "north")
            return self._send(scenario.png_bytes(img), "image/png")
        if what == "mesh.bin":
            sel = q.get("sel", "all")
            elems = {"all": lambda: None, "worst": lambda: mv.worst(int(q.get("n", 2000)), q.get("field", "q")),
                     "tris": lambda: np.arange(len(mv.tris)),
                     "window": lambda: mv.window(float(q["lon"]), float(q["lat"]), float(q.get("r_km", 500)))}[sel]()
            return self._send(payload.mesh_bin(mv, elems, node_fields=q.get("edit") == "1"),
                              "application/octet-stream")
        if what == "irregular.bin":
            return self._send(payload.points_bin(*mv.irregular()), "application/octet-stream")
        if what == "hist":
            rec = st.quality()
            return self._json(jsonable({"histograms": runs.with_record(mv.histograms(), rec), "counts": mv.counts(),
                                        "quality": runs.record_summary(rec) if rec else None}))
        if what == "worst.json":
            return self._json({"worst": mv.worst_list(int(q.get("n", 50)), q.get("field", "q"))})
        return self._json({"error": "not found"}, 404)

    # ----------------------------------------------------------------------------------------------------- POST
    def _post(self):
        u = urlparse(self.path)
        st = self.state
        body = self._body()
        if u.path == "/api/log":                     # the page's events (ui/web/static/log.js)
            if st.log:
                for e in body.get("events", []):
                    st.log.write({"page": body.get("page"), **e})
            return self._json({"ok": True})
        if u.path == "/api/runs/roots":              # a folder to search for runs, added or removed from the page;
            p = Path(body.get("add") or body.get("remove")).expanduser().resolve()      # deep: with its subfolders
            if body.get("add"):
                if not p.is_dir():
                    return self._json({"error": f"no such folder: {p}"}, 400)
                if p not in st.run_roots:
                    st.run_roots.append(p)
                (st.deep.add if body.get("deep") else st.deep.discard)(p)
                st.first_scans()
            elif p == st.run_roots[0]:
                return self._json({"error": f"{p} is where new runs are written: it stays in the list"}, 400)
            elif p in st.run_roots:
                st.run_roots.remove(p)
                st.deep.discard(p)
            return self._json({**st.roots_json(), "path": str(p)})
        if u.path == "/api/preview":
            limited = bool(body.get("limited", False))
            with st.lock:
                cfg = st.take(body.get("config"))
            with st.heavy if limited else st.lock:
                p = scenario.preview(cfg, st.coast(cfg), body.get("step"), limited=limited)
            rgba, win = scenario.h_rgba(p), scenario.winner_rgba(p)
            out = {"seconds": round(p.seconds, 3), "n_nodes_est": p.n_nodes_est, "grad_max": p.grad_max,
                   "grad_at": p.grad_at, "h_min": float(p.h[p.water].min()), "h_max": float(p.h[p.water].max()),
                   "layers": p.names, "layer_index": p.layer_index, "png": b64png(rgba), "png_winner": b64png(win),
                   "limited": limited, "limited_applied": p.limiter is not None, "grad_limit": cfg.size.grad_max,
                   "scale": cfg.mesh.scale, "limiter": jsonable(p.limiter) if p.limiter else None,
                   "grid": readout_grid(p)}
            for pole in body.get("polar", []):
                out[f"png_{pole}"] = b64png(polar_rgba(rgba, pole == "north"))
                out[f"png_winner_{pole}"] = b64png(polar_rgba(win, pole == "north"))
            return self._json(out)
        if u.path == "/api/preview/window":
            w, e, s, n = (float(x) for x in body["bounds"])
            limited = bool(body.get("limited", False))
            with st.lock:
                cfg = st.take(body.get("config"))
            with st.heavy if limited else st.lock:
                p = scenario.preview(cfg, st.coast(cfg), float(body["step"]), (w, e, s, n), body.get("polygon"),
                                     limited)
            rgba = scenario.h_rgba(p, body.get("h_lo"), body.get("h_hi"))
            hw = p.h[p.water]
            out = {"seconds": round(p.seconds, 3), "shape": list(p.h.shape), "n_nodes_est": p.n_nodes_est,
                   "grad_max": p.grad_max, "grad_at": p.grad_at, "h_min": float(hw.min()) if hw.size else None,
                   "h_max": float(hw.max()) if hw.size else None, "png": b64png(rgba), "bounds": [w, e, s, n]}
            for pole in body.get("polar", []):
                img, pb = polar_window(rgba, (w, e, s, n), pole == "north")
                out[f"png_{pole}"], out[f"bounds_{pole}"] = b64png(img), pb
            return self._json(out)
        if u.path == "/api/rho":
            from ui.common import rho
            limited = bool(body.get("limited", False))
            with st.lock:
                cfg = st.take(body.get("config"))
            with st.heavy if limited else st.lock:
                r = rho.rho_preview(scenario.h_field(cfg, limited), st.coast(cfg), float(body.get("step", 1.0)))
            rgba = rho.rho_rgba(r)
            out = {k: r[k] for k in ("v0", "n_points", "plus", "minus", "total", "curv")}
            out.update(seconds=round(r["seconds"], 3), png=b64png(rgba), step=float(body.get("step", 1.0)))
            for pole in body.get("polar", []):
                out[f"png_{pole}"] = b64png(polar_rgba(rgba, pole == "north"))
            return self._json(out)
        if u.path == "/api/config":
            cfg = st.take(body["config"])
            path = Path(body.get("path") or st.config_path)
            if path.resolve() == scenario.EXAMPLE.resolve():          # the repository's example stays as it is
                path = Path(scenario.SAVE_DEFAULT)
            path.parent.mkdir(parents=True, exist_ok=True)
            st.config_path = path                                     # "Save as": later saves go there too
            text = scenario.dumps(cfg)
            if body.get("size_from"):                    # the page's h is that variant, unchanged: refer to it (S35)
                text = scenario.dumps_from(cfg, Path(body["size_from"]), path.parent)
                st.cfg = replace(cfg, size_from=str(Path(body["size_from"]).resolve()))
            path.write_text(text)
            return self._json({"saved": str(path), "toml": text})
        if u.path == "/api/mesh/depth/select":       # the Tools node tool: the nodes in a box or polygon, their depth
            mv = st.mv()
            if body.get("path") != mv.path:
                return self._json({"error": f"the mesh shown is {Path(mv.path).name}, not {Path(str(body.get('path'))).name}"}, 400)
            ids = depth.select(mv, body.get("box"), body.get("polygon"))
            return self._json(jsonable(depth.summary(mv, ids, scenario.bathymetry_source(st.cfg), st.cfg.bathymetry.variable)))
        if u.path == "/api/mesh/depth/edit":         # the node tool: an edit of the nodes in an area
            mv = st.mv()
            if body.get("path") != mv.path:
                return self._json({"error": f"the mesh shown is {Path(mv.path).name}, not {Path(str(body.get('path'))).name}"}, 400)
            try:
                ids = depth.add_edit(mv, body["op"], body["depth_m"], body.get("box"), body.get("polygon"))
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            area = depth.summary(st.mv(), ids, scenario.bathymetry_source(st.cfg), st.cfg.bathymetry.variable)
            return self._json(jsonable({"edits": depth.edits_list(Path(mv.path)), "area": area}))
        if u.path in ("/api/mesh/depth/undo", "/api/mesh/depth/remove"):
            try:
                depth.remove_edit(body["path"], body.get("index") if u.path.endswith("remove") else None)
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            return self._json(jsonable({"edits": depth.edits_list(Path(body["path"]))}))
        if u.path == "/api/profile":                 # the Tools window: h and the bottom along a line
            from meshom.geo import lonlat_to_xyz
            from ui.common import profile
            cfg = scenario.from_json(body["config"], st.loaded) if body.get("config") else st.cfg
            lon, lat, d = profile.sample_line(body["line"], int(body.get("n", 600)))
            limited = bool(body.get("limited"))
            with st.heavy if limited else st.lock:
                hf = scenario.h_field(cfg, limited)
            h = hf(lonlat_to_xyz(lon, lat))
            land = profile.land_at(st.coast(cfg), lon, lat)
            src, elev, note = scenario.bathymetry_source(cfg), None, None
            if src is None:
                note = "no bathymetry grid found: set [bathymetry] source (GEBCO 2024) in the config"
            else:
                try:
                    elev = profile.elevation_at(src, lon, lat, cfg.bathymetry.variable)
                except ImportError:
                    note = "the server's Python has no netCDF4: the bottom cannot be read (mamba install netcdf4)"
            nodes = None
            mv = st.mv() if body.get("mesh") else None
            if mv is not None and mv.path == body["mesh"]:          # the mesh shown: the nodes the line crosses
                nd = profile.line_nodes(mv, body["line"])
                nelev = None
                if elev is not None and len(nd["id"]):
                    nelev = profile.elevation_at(src, nd["lon"], nd["lat"], cfg.bathymetry.variable).tolist()
                nodes = {"id": nd["id"].tolist(), "d_km": (nd["d_m"] / 1000.0).tolist(),
                         "off_km": (nd["off_m"] / 1000.0).tolist(), "lon": nd["lon"].tolist(), "lat": nd["lat"].tolist(),
                         "depth_m": None if mv.node_depth is None else mv.node_depth[nd["id"]].tolist(),
                         "elev_m": nelev, "mesh": Path(mv.path).name, "depth_source": mv.depth_source}
            return self._json(jsonable({"d_km": (d / 1000.0).tolist(), "lon": lon.tolist(), "lat": lat.tolist(),
                                        "h_km": (h / 1000.0).tolist(), "elev_m": None if elev is None else elev.tolist(),
                                        "land": land.tolist(), "bathy": str(src) if src else None,
                                        "note": note, "limited": limited, "nodes": nodes}))
        if u.path == "/api/size/save":               # the page's [size] as an h variant file (size-only TOML)
            cfg = scenario.from_json(body["config"], st.loaded)
            path = Path(body["path"]).expanduser()
            path = path if path.is_absolute() else st.variant_dir() / path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(mconfig.dumps_size(cfg.size))
            return self._json({"saved": str(path)})
        if u.path == "/api/size/load":               # an h variant file as the page's [size]
            path = Path(body["path"]).expanduser().resolve()
            size = mconfig.loads_size(path.read_text(), path.parent)
            return self._json({"size": scenario.to_json(replace(st.cfg, size=size))["size"], "path": str(path)})
        if u.path == "/api/config/open":                 # a run's config becomes the page's; saves go elsewhere
            p = Path(body["path"])
            f = next((p / n for n in ("config.toml", "config.resolved.toml") if (p / n).exists()), p) if p.is_dir() else p
            cfg = scenario.load(f)
            cfg = replace(cfg, mesh=replace(cfg.mesh, out=None))       # `meshom run FILE` must not write into that run
            with st.lock:
                st.loaded = st.cfg = cfg
                st.config_path = Path(scenario.SAVE_DEFAULT)
                st.opened_from = str(f.parent)
            return self._json({"opened": str(f), "save_to": str(st.config_path)})
        if u.path == "/api/toml":
            if "text" in body:
                try:
                    return self._json({"value": tomllib.loads(body["text"])})
                except tomllib.TOMLDecodeError as e:
                    return self._json({"error": str(e)})
            c = body["config"]
            if "mesh" in c:                              # a whole config: typed by the core (as Save writes it)
                cfg = scenario.from_json(c, st.loaded)
                return self._json({"toml": scenario.dumps_from(cfg, Path(body["size_from"]), st.variant_dir())
                                   if body.get("size_from") else scenario.dumps(cfg)})
            import tomli_w                               # one table (the [blocks] editor)
            return self._json({"toml": tomli_w.dumps(scenario.coerce(c, scenario.to_json(st.loaded)))})
        if u.path == "/api/mesh/load":
            p = Path(body["path"])
            meshview.mesh_file(p)                        # raises if there is no mesh
            st.mesh_path = p
            return self._json({"mesh": str(p)})
        if u.path == "/api/compare":                     # quality records side by side; no mesh is read
            cols = []
            for p in body["paths"]:
                mf = meshview.mesh_file(Path(p))
                mc = st.mesh_config(mf)
                cols.append({"path": str(p), "file": str(mf), "rec": st.quality(mf),
                             "gates": mc[0].gates_all() if mc else mconfig.DEFAULT_GATES})
            return self._json(jsonable(runs.compare(cols)))
        if u.path == "/api/mesh/edit":
            mf = meshview.mesh_file(st.mesh_path).resolve()
            if any(Path(r).resolve() in mf.parents for r in st.ref_roots):
                return self._json({"error": f"{mf.parent.name} is a reference mesh: it is not edited (shared data)"}, 400)
            from ui.web import edit
            return self._json(edit.save(mf, body.get("ops", [])))
        if u.path == "/api/mesh/write":
            try:
                return self._json(jsonable(st.writer.start(Path(body["path"]), body.get("format"),
                                                           bool(body.get("fesom_ini")))))
            except write.Busy as e:
                return self._json({"error": str(e)}, 409)
            except ValueError as e:                  # no config.resolved.toml
                return self._json({"error": str(e)}, 400)
        if u.path == "/api/run":
            cfg = st.take(body.get("config"))
            root = st.run_roots[0]
            r = runs.submit(cfg, root, scenario.REPO, sys.executable, st.run_cmd, st.allow_submit,
                            body.get("resources"))
            return self._json(r)
        return self._json({"error": "not found"}, 404)


def make_handler(config: Path, mesh: Path, run_roots: list[Path] | None = None, allow_submit: bool = False,
                 run_cmd: str = runs.RUN_CMD, ref_roots: list[Path] | None = None,
                 token: str | None = None, log_path: Path | None = None) -> type[Handler]:
    """The request handler bound to a fresh state, without a socket: make_server serves it, ui/web/inproc.py calls it
    in this process. token: the access token every request but /static/* must carry; None: none. log_path: the action
    log (ActionLog); None: none."""
    roots = [Path(r).resolve() for r in run_roots or [".tmp/ui/runs", ".tmp/runs", ".tmp/compete"]]
    roots[0].mkdir(parents=True, exist_ok=True)
    state = State(config, mesh, roots, allow_submit, run_cmd, ref_roots)
    state.token = token
    state.log = ActionLog(Path(log_path)) if log_path else None
    Handler.state = state                         # the last state made, for scripts that reach into it
    return type("BoundHandler", (Handler,), {"state": state})    # one state per handler (several in one process)


def make_server(config: Path, mesh: Path, port: int, host: str = "127.0.0.1", run_roots: list[Path] | None = None,
                allow_submit: bool = False, run_cmd: str = runs.RUN_CMD,
                ref_roots: list[Path] | None = None, token: str | None = None,
                log_path: Path | None = None) -> ThreadingHTTPServer:
    """The server (not started) of make_handler's handler; the cookie is named after the port."""
    handler = make_handler(config, mesh, run_roots, allow_submit, run_cmd, ref_roots, token, log_path)
    srv = ThreadingHTTPServer((host, port), handler)
    handler.state.cookie = f"meshom_token_{srv.server_address[1]}"
    return srv


def loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:                             # a hostname: a node's cluster-internal name
        return False


def parser(prog: str = "meshom-ui-web") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="meshom web UI: open http://HOST:PORT/ in a browser")
    ap.add_argument("--config", type=Path, default=scenario.EXAMPLE,
                    help="meshom config (TOML) to load and save (the example is never overwritten: its saves go to .tmp/ui/config.toml)")
    ap.add_argument("--mesh", type=Path, default=Path(".tmp/ui/mesh"), help="mesh file or run folder shown first")
    ap.add_argument("--runs", type=Path, action="append",
                    help="a folder with run folders in it (repeatable; the first gets new runs; the page adds more and can search a "
                         "folder's subfolders too). "
                         "Default .tmp/ui/runs .tmp/runs .tmp/compete")
    ap.add_argument("--refs", type=Path, action="append",
                    help="reference meshes' parent (repeatable; folders with mesh.npz + quality.json). Default: "
                         + " ".join(str(r) for r in runs.REF_ROOTS))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on: 127.0.0.1 (default), or a compute node's hostname ($(hostname)) to be "
                         "reached through the login node (needs the token; never a public address)")
    tk = ap.add_mutually_exclusive_group()
    tk.add_argument("--token", help="access token (default: a random 32-hex token, printed in the URL)")
    tk.add_argument("--no-token", action="store_true", help="no access control (only with a loopback --host)")
    ap.add_argument("--allow-submit", action="store_true", help="Run submits the sbatch script (else a dry run)")
    ap.add_argument("--run-cmd", default=runs.RUN_CMD,
                    help="batch command; {repo}, {python}, {config}, {dir} are filled in")
    ap.add_argument("--log", type=Path,
                    help="action log: the page's clicks, view changes, errors and every request, as JSON lines "
                         "(default .tmp/ui/logs/web-<date>-<time>-<host>.jsonl)")
    return ap


def main(argv: list[str] | None = None, prog: str = "meshom-ui-web") -> None:
    ap = parser(prog)
    a = ap.parse_args(argv)
    if a.no_token and not loopback(a.host):
        ap.error(f"--host {a.host} listens beyond this machine: it needs the token (drop --no-token)")
    token = None if a.no_token else (a.token or secrets.token_hex(16))
    log = a.log or scenario.REPO / ".tmp" / "ui" / "logs" / time.strftime(f"web-%Y%m%d-%H%M%S-{socket.gethostname()}.jsonl")
    srv = make_server(a.config, a.mesh, a.port, a.host, a.runs, a.allow_submit, a.run_cmd, a.refs, token, log)
    srv.RequestHandlerClass.state.log.write({"ev": "start", "config": a.config, "mesh": a.mesh,
                                             "host": a.host, "port": srv.server_address[1]})
    port = srv.server_address[1]
    query = f"?token={token}" if token else ""
    print(f"meshom UI (web): http://{a.host}:{port}/{query}  config {a.config}"
          f"{'  (submit allowed)' if a.allow_submit else ''}", flush=True)
    print(f"  action log: {log}", flush=True)
    if not loopback(a.host):
        print(f"  tunnel: ssh -N -L {port}:{a.host}:{port} <user>@levante.dkrz.de   then open "
              f"http://127.0.0.1:{port}/{query}", flush=True)
    t0 = time.time()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print(f"stopped after {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
