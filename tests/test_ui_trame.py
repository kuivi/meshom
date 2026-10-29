"""UI prototype 2 (trame + PyVista): the app starts and serves its page (VTK renders offscreen)."""
import os
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

pytest.importorskip("trame")
pytest.importorskip("trame_vtk")
pytest.importorskip("pyvista")
pytest.importorskip("shapefile")

from ui_helpers import write_tiny_mesh  # noqa: E402

from ui.common import scenario  # noqa: E402


def test_trame_app_serves_page(tmp_path):
    if not scenario.resolve(scenario.load().coast.source).exists():
        pytest.skip("coastline not found")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    env.update(LP_NUM_THREADS="2", PYTHONPATH=os.pathsep.join([str(scenario.REPO), env.get("PYTHONPATH", "")]))
    p = subprocess.Popen([sys.executable, "-m", "ui.trame.app", "--server", "--port", str(port),
                          "--mesh", str(write_tiny_mesh(tmp_path / "mesh"))], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        body = None
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/index.html", timeout=2) as r:
                    body = r.read()
                break
            except OSError:
                time.sleep(0.5)
        assert body is not None and b"<html" in body.lower()
    finally:
        p.terminate()
        p.wait(10)
