"""UI prototype 3 (PySide6 + pyvistaqt): the window builds, previews h and shows a mesh (needs a display: a real one,
or the conda-forge Xvfb of the meshom-ui env on Levante)."""
import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyvistaqt")
pytest.importorskip("shapefile")

from ui_helpers import write_tiny_mesh  # noqa: E402

from ui.common import scenario  # noqa: E402
from ui.qt import xvfb  # noqa: E402


@pytest.fixture(scope="module")
def display():
    if not scenario.resolve(scenario.load().coast.source).exists():
        pytest.skip("coastline not found")
    if os.environ.get("DISPLAY", "").startswith(":") or os.uname().sysname == "Darwin":
        yield None
        return
    if not xvfb.available():
        pytest.skip("no display and no Xvfb")
    os.environ["LP_NUM_THREADS"] = os.environ.get("LP_NUM_THREADS", "2")
    yield xvfb.start(80 + os.getpid() % 9)              # -terminate: exits after Qt disconnects at process exit


def test_qt_window(display, tmp_path):
    from PySide6 import QtWidgets

    from ui.qt.app import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test"])
    cfg = tmp_path / "c.toml"
    scenario.save(scenario.load(), cfg)
    w = MainWindow(cfg, write_tiny_mesh(tmp_path / "mesh"))
    w.show()
    app.processEvents()
    assert "nodes ≈" in w.h_stats.text()
    w.set_view("north")
    assert w.set_mesh("full")["n_elems"] == 18 * 8 + 2 * 18
    w.save()
    assert "polygon" in cfg.read_text()
    w.close()
