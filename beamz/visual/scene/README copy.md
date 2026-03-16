# beamz.visual.scene

`beamz.visual.scene` is a notebook-first 3D visualization widget for BeamZ-style FDTD geometry and simulation setup. It uses a compact Python scene graph, exposes that scene to the browser through `anywidget`, and renders it with Three.js.

Current scope:

- `view3d(scene_or_beamz_object)` entry point
- compact JSON scene graph for structures, sources, monitors, and the simulation domain
- Three.js frontend with orbit controls, clipping planes, transparency, and picking
- BeamZ-compatible duck-typed adapter so the widget can evolve independently from BeamZ internals

## Install

```bash
pip install -e ".[widget,dev]"
```

## Example

```python
from beamz.visual.scene import view3d

result = view3d(simulation_or_design)
result
```

Behavior:

- in Jupyter notebooks, `view3d(...)` returns an `anywidget`
- outside notebooks, `view3d(...)` writes a temporary HTML page and opens it in your browser

Demo scene:

```python
from beamz.visual.scene import demo_widget

demo_widget()
```

Example files:

- `examples/demo_scene.py`
- `examples/demo_beamz_like.py`

## Notes

- The Python package is usable without `anywidget` for scene generation and testing.
- Rendering the widget requires the optional `widget` extra.
- The frontend currently imports Three.js directly as an ES module, which keeps the package lightweight while the widget contract stabilizes.
