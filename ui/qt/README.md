# Prototype 3 — desktop app (PySide6 + pyvistaqt)

    PYTHONPATH=$PWD mamba run -n meshom-ui python -m ui.qt.app [--config ui/common/scenario.toml] [--mesh .tmp/ui/mesh]

- **A desktop or laptop (e.g. a Mac):** run the command above; VTK draws with the local GPU in a native window.
- **Levante:** only through `ssh -X` (every frame crosses the network; not usable for the 10⁶-element mesh) or
  headless for tests: `ui/qt/xvfb.py` starts the conda-forge Xvfb of the `meshom-ui` env (it needs a three-function
  `libcrypto.so.10` shim, built with gcc into `.tmp/ui/xlib`, because that Xvfb is a CentOS-7 build).
- **Headless measurements:** `scripts/on_node.sh -c 16 -t 00:30:00 -- env -u DISPLAY LP_NUM_THREADS=16
  PYTHONPATH=$PWD mamba run -n meshom-ui python -m ui.qt.measure`.

The scene (`ui/common/vtkscene.py`) is shared with the trame prototype; this directory holds only the Qt shell.
