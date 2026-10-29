# meshom UI prototypes

Three prototype stacks for the same scenario:

| dir | stack | start (from the repository root) |
|---|---|---|
| `web/` | Python stdlib HTTP server + deck.gl 9.4.0 GlobeView in the browser — **the chosen stack, v2 with the panels** (`web/README.md`) | `PYTHONPATH=$PWD mamba run -n meshom-ui python -m ui.web.server` |
| `trame/` | trame 4 + PyVista/VTK, rendered on the server, shown in the browser | `PYTHONPATH=$PWD mamba run -n meshom-ui python -m ui.trame.app` |
| `qt/` | PySide6 + pyvistaqt desktop window | `PYTHONPATH=$PWD mamba run -n meshom-ui python -m ui.qt.app` |

Shared: `common/scenario.py` (the TOML config, size layers combined by min, the coarse h preview, windows), `common/coast.py`
(coastline shapefiles: GSHHS, Natural Earth), `common/rho.py` (ρ_c on the preview grid), `common/meshview.py` (a finished
mesh: overview raster, worst elements, zoom windows, irregular nodes, histograms),
`common/vtkscene.py` (the PyVista scene of the trame and Qt prototypes), `common/scenario.toml` (the example config).

The mesh view needs a finished mesh. Build it once on a compute node (a batch run; the UI never builds it):

    scripts/on_node.sh -c 8 -t 01:00:00 -- env PYTHONPATH=$PWD \
        mamba run -n meshom-ui python -m ui.common.prepare_mesh --scale 0.45     # -> .tmp/ui/mesh (~10^6 elements)

The env: `ui/environment-ui.yml`. The coastline is read from `../oceanmesh/tests/GSHHS_i_L1.shp` (relative to the
main checkout). Tests: `PYTHONPATH=$PWD mamba run -n meshom-ui python -m pytest tests/test_ui_*.py`.
