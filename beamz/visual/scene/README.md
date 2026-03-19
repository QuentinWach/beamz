Interactive 3D scene viewer for BeamZ designs and simulations.

This package is BeamZ-specific. It contains:

- a compact scene schema in `_scene.py`
- BeamZ adapters in `_beamz.py`
- notebook/browser entry points in `_widget.py` and `_browser.py`
- bundled frontend assets in `static/`

Public entry points:

- `beamz.visual.scene.view3d(...)`
- `beamz.visual.scene.beamz_to_scene(...)`
- `Simulation.show()` / `Simulation.to_scene()`
