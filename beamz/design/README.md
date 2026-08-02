# Design

Module to define the design of complex structures parametrically as well as mesh it into grids of material values.

- `core.py`: Main design API and rasterization orchestration.
- `discretization.py`: Design-to-`MaterialGrid` material placement API.
- `structures.py`: Geometry primitives (rectangle, circle, ring, polygon, taper, etc.).
- `materials.py`: the homogeneous `Material` value accepted by geometric designs.
- `raster/`: Rust-backed geometry rasterization and external-format importers.
- `gds.py`: Optional gdsfactory-backed component and GDS import/export.

Spatial coefficient arrays enter the solver through `MaterialGrid`; geometric
designs use homogeneous materials and the Rust rasterizer. Geometry errors abort
rasterization rather than silently substituting fallback properties.

Install the optional layout dependency and import cells through the design API:

```python
from beamz.design import import_component

imported = import_component("mmi1x2", layer=(1, 0), xy_padding=2e-6)
design = imported.design
input_port = imported.port("o1")
```

Pass an explicit gdsfactory `LayerStack` when the full PDK stack should define
the vertical geometry. BeamZ does not discover or patch vendor PDK packages.
