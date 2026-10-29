# Prototype 2 — Python only, in the browser (trame + PyVista)

    PYTHONPATH=$PWD mamba run -n meshom-ui python -m ui.trame.app [--config ui/common/scenario.toml] [--mesh .tmp/ui/mesh] [--port 8766] [--server]

opens http://localhost:8766/ (`--server`: do not open a browser). VTK renders on the server; the browser receives
JPEG images over a websocket.

- **A desktop or laptop (e.g. a Mac):** run the command above in the `meshom-ui` env; VTK uses the local GPU.
- **Levante:** rendering is CPU (Mesa llvmpipe through EGL), 0.15–0.4 s per frame with 16 threads for the full mesh,
  so run it on a compute node, not the login node:
  `scripts/on_node.sh -c 16 -t 02:00:00 -- env -u DISPLAY LP_NUM_THREADS=16 PYTHONPATH=$PWD mamba run -n meshom-ui
  python -m ui.trame.app --server --host 0.0.0.0 --port 8766`, then on your own computer
  `ssh -N -L 8766:<node>:8766 levante.dkrz.de` and open http://127.0.0.1:8766/. The stream is ~25 KB per image.
  On the DKRZ JupyterHub the same trame app can run inside a notebook (not tried here).
- **Headless measurements:** `scripts/on_node.sh -c 16 -t 00:30:00 -- env -u DISPLAY LP_NUM_THREADS=16
  PYTHONPATH=$PWD mamba run -n meshom-ui python -m ui.trame.measure`.

The scene (`ui/common/vtkscene.py`) is shared with the Qt prototype; this directory holds only the trame shell.
