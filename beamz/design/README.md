# Design

Module to define the design of complex structures parametrically as well as mesh it into grids of material values.

- `core.py`: Main design API and rasterization orchestration.
- `structures.py`: Geometry primitives (rectangle, circle, ring, polygon, taper, etc.).
- `materials.py`: Material classes (`Material`, `CustomMaterial`) for spatially constant or custom material models.
- `library.py`: Starter material presets (`VACUUM`, `AIR`, `SIO2`, `SIN`, `SI3N4`, `GOLD`, `ALUMINUM`, `COPPER`).
- `meshing.py`: 2D/3D rasterization into material-property grids.
- `io.py`: GDS import/export helpers and gdsfactory integration.
