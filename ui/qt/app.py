"""Prototype 3: desktop app — PySide6 + pyvistaqt (VTK in a Qt window). Same PyVista scene as the trame prototype.

    python -m ui.qt.app [--config FILE] [--mesh DIR]

Runs where the display is (a desktop or laptop); over ssh -X from Levante every frame crosses the network.
"""
import argparse
import json
import sys
from pathlib import Path

from PySide6 import QtCore, QtWidgets
from pyvistaqt import QtInteractor

from meshom.config import ZoneLayer
from ui.common import scenario
from ui.common.vtkscene import MESH_MODES, VIEWS, Scene


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: Path, mesh: Path):
        super().__init__()
        self.setWindowTitle("meshom · UI prototype · Qt")
        self.plotter = QtInteractor(self)
        self.setCentralWidget(self.plotter.interactor)
        self.sc = Scene(self.plotter, config, mesh)
        self.sc.on_zone_edit = self.zone_edited
        L = self.sc.cfg.size.layers
        self.base = next(ly for ly in L if ly.kind == "constant")
        self.coastl = next(ly for ly in L if ly.kind == "coast_distance")

        dock = QtWidgets.QDockWidget("controls", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)
        dock.setWidget(panel)
        dock.setMinimumWidth(310)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        views = QtWidgets.QHBoxLayout()
        self.view_buttons = QtWidgets.QButtonGroup(self)
        for v in VIEWS:
            b = QtWidgets.QPushButton(v)
            b.setCheckable(True)
            b.setChecked(v == "globe")
            self.view_buttons.addButton(b)
            views.addWidget(b)
            b.clicked.connect(lambda _=False, v=v: self.set_view(v))
        lay.addLayout(views)

        box = QtWidgets.QGroupBox("1 · h builder (min of layers)")
        form = QtWidgets.QFormLayout(box)
        z = self.sc.zone() or ZoneLayer(h_m=10000.0, transition_m=300000.0)
        self.inputs = {}
        for key, label, val, step in (("base_h", "base h, km", self.base.h_m / 1e3, 5),
                                      ("coast_h", "h at coast, km", self.coastl.h_m / 1e3, 1),
                                      ("plateau", "plateau, km", self.coastl.plateau_m / 1e3, 5),
                                      ("grad", "coast |∇h|", self.coastl.grad, 0.01),
                                      ("zone_h", "zone h, km", z.h_m / 1e3, 1),
                                      ("zone_w", "zone transition, km", z.transition_m / 1e3, 25)):
            w = QtWidgets.QDoubleSpinBox()
            w.setRange(0.0, 1e5)
            w.setDecimals(3 if key == "grad" else 1)
            w.setSingleStep(step)
            w.setValue(val)
            w.valueChanged.connect(self.schedule_preview)
            form.addRow(label, w)
            self.inputs[key] = w
        row = QtWidgets.QHBoxLayout()
        self.draw_btn = QtWidgets.QPushButton("Draw polygon")
        self.finish_btn = QtWidgets.QPushButton("Finish")
        self.finish_btn.setEnabled(False)
        self.draw_btn.clicked.connect(self.draw)
        self.finish_btn.clicked.connect(self.finish)
        row.addWidget(self.draw_btn)
        row.addWidget(self.finish_btn)
        form.addRow(row)
        self.show_h = QtWidgets.QCheckBox("show h")
        self.show_h.setChecked(True)
        self.show_h.toggled.connect(self.schedule_preview)
        form.addRow(self.show_h)
        self.h_stats = QtWidgets.QLabel()
        self.h_stats.setStyleSheet("font-family: monospace; font-size: 11px")
        form.addRow(self.h_stats)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("2 · Coastline")
        form = QtWidgets.QFormLayout(box)
        self.min_area = QtWidgets.QDoubleSpinBox()
        self.min_area.setRange(0, 1e7)
        self.min_area.setSingleStep(50)
        self.min_area.setValue(self.sc.cfg.coast.min_area_km2)
        self.min_area.editingFinished.connect(self.coast_changed)
        form.addRow("min island area, km²", self.min_area)
        self.coast_at_h = QtWidgets.QCheckBox("resample at h at coast")
        self.coast_at_h.toggled.connect(self.coast_changed)
        form.addRow(self.coast_at_h)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("6 · Mesh view")
        form = QtWidgets.QFormLayout(box)
        self.field = QtWidgets.QComboBox()
        self.field.addItems(["q", "size_ratio", "kind"])
        self.field.currentTextChanged.connect(lambda _: self.set_mesh(self.sc.mesh_mode))
        form.addRow("overlay", self.field)
        modes = QtWidgets.QHBoxLayout()
        self.mode_buttons = QtWidgets.QButtonGroup(self)
        for m in MESH_MODES:
            b = QtWidgets.QPushButton(m)
            b.setCheckable(True)
            b.setChecked(m == "off")
            self.mode_buttons.addButton(b)
            modes.addWidget(b)
            b.clicked.connect(lambda _=False, m=m: self.set_mesh(m))
        form.addRow(modes)
        self.mesh_stats = QtWidgets.QLabel()
        self.mesh_stats.setStyleSheet("font-family: monospace; font-size: 11px")
        self.mesh_stats.setWordWrap(True)
        form.addRow(self.mesh_stats)
        lay.addWidget(box)

        row = QtWidgets.QHBoxLayout()
        save, run = QtWidgets.QPushButton("Save TOML"), QtWidgets.QPushButton("Run (batch)")
        save.clicked.connect(self.save)
        run.clicked.connect(self.run)
        row.addWidget(save)
        row.addWidget(run)
        lay.addLayout(row)
        self.toml = QtWidgets.QPlainTextEdit()
        self.toml.setReadOnly(True)
        self.toml.setStyleSheet("font-family: monospace; font-size: 10px")
        lay.addWidget(self.toml)
        lay.addStretch(1)
        self.statusBar().showMessage(f"config {self.sc.config_path}")

        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.preview)
        self.sc.rebuild()
        self.sc.look()
        self.stats()

    # -------------------------------------------------------------------------------------------------- actions
    def read_inputs(self) -> None:
        g = {k: w.value() for k, w in self.inputs.items()}
        self.base.h_m, self.coastl.h_m = g["base_h"] * 1e3, g["coast_h"] * 1e3
        self.coastl.plateau_m, self.coastl.grad = g["plateau"] * 1e3, g["grad"]
        if self.sc.zone():
            self.sc.zone().h_m, self.sc.zone().transition_m = g["zone_h"] * 1e3, g["zone_w"] * 1e3
        self.sc.show_h = self.show_h.isChecked()

    def schedule_preview(self, *_) -> None:
        self.timer.start()

    def preview(self) -> float:
        self.read_inputs()
        s = self.sc.update_h()
        self.stats()
        self.plotter.render()
        return s

    def stats(self) -> None:
        p = self.sc.preview
        self.h_stats.setText(f"h {p.h[p.water].min() / 1e3:.1f}–{p.h[p.water].max() / 1e3:.1f} km over water\n"
                             f"nodes ≈ ∫h⁻² dA = {p.n_nodes_est / 1e6:.3f} M\nmax |∇h| = {p.grad_max:.3f}\n"
                             f"preview {self.sc.timings['preview_s']:.2f} s")
        self.toml.setPlainText(scenario.dumps(self.sc.cfg))

    def zone_edited(self) -> None:
        """Every handle move: redraw the outline now, recompute h when the moves stop (debounced)."""
        self.sc.update_zone(widgets=False)
        self.plotter.render()
        self.timer.start()

    def coast_changed(self, *_) -> None:
        self.sc.cfg.coast.min_area_km2 = self.min_area.value()
        self.sc.coast_at_h = self.coast_at_h.isChecked()
        self.sc.update_coast()
        self.preview()

    def set_view(self, v: str) -> None:
        self.sc.set_view(v)
        self.plotter.render()

    def set_mesh(self, m: str) -> dict:
        self.sc.mesh_mode, self.sc.field = m, self.field.currentText()
        r = self.sc.update_mesh()
        self.plotter.render()
        r["render_ms"] = 1e3 * self.plotter.renderer.GetLastRenderTimeInSeconds()
        self.mesh_stats.setText(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
        return r

    def draw(self) -> None:
        self.sc.start_drawing()
        self.draw_btn.setEnabled(False)
        self.finish_btn.setEnabled(True)
        self.statusBar().showMessage("drawing: click on the globe to add vertices")
        self.plotter.render()

    def finish(self) -> None:
        self.sc.finish_drawing()
        self.draw_btn.setEnabled(True)
        self.finish_btn.setEnabled(False)
        self.statusBar().clearMessage()
        self.stats()
        self.plotter.render()

    def save(self) -> None:
        self.statusBar().showMessage(f"saved {self.sc.save_config()}")

    def run(self) -> None:
        self.statusBar().showMessage("dry run (the core has no `meshom run CONFIG` yet): scripts/on_node.sh -c 8 "
                                     f"-t 01:00:00 -- python -m ui.common.prepare_mesh --method "
                                     f"{self.sc.cfg.mesh.method}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=scenario.EXAMPLE)
    ap.add_argument("--mesh", type=Path, default=Path(".tmp/ui/mesh"))
    a, rest = ap.parse_known_args(argv)
    app = QtWidgets.QApplication([sys.argv[0], *rest])
    w = MainWindow(a.config, a.mesh)
    w.resize(1400, 850)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
