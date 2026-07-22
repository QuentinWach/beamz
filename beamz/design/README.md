# Design

Module to define the design of complex structures parametrically as well as mesh it into grids of material values.

- `core.py`: Main design API and rasterization orchestration.
- `discretization.py`: Design-to-`MaterialGrid` material placement API.
- `structures.py`: Geometry primitives (rectangle, circle, ring, polygon, taper, etc.).
- `materials.py`: Material classes (`Material`, `CustomMaterial`) for spatially constant or custom material models.
- `meshing.py`: 2D/3D rasterization into material-property grids.
- `gds.py`: Optional gdsfactory-backed component and GDS import/export.

`CustomMaterial` infers `max_permittivity` from sampled grids. Callable
permittivity models must declare `max_permittivity=...` when used with
`GridSpec.auto(...)`; explicit uniform grids do not require that bound.
Material evaluation and geometry errors abort rasterization rather than silently
substituting fallback properties.

Install the optional layout dependency and import cells through the design API:

```python
from beamz.design import import_component

imported = import_component("mmi1x2", layer=(1, 0), xy_padding=2e-6)
design = imported.design
input_port = imported.port("o1")
```

Pass an explicit gdsfactory `LayerStack` when the full PDK stack should define
the vertical geometry. BeamZ does not discover or patch vendor PDK packages.
