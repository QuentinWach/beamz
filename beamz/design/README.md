# Design

Module to define the design of complex structures parametrically as well as mesh it into grids of material values.

+ core.py       / Main module to define and organize complex design geometries, materials, ...
+ structures.py / Polygon objects to define geometry within the design
+ materials.py  / Material models (`Material`, `CustomMaterial`, and dispersive classes:
                 `SellmeierMaterial`, `DrudeMaterial`, `LorentzMaterial`,
                 `DebyeMaterial`, `PoleResidueMaterial`, `DrudeLorentzMaterial`)
+ library.py    / Predefined static material catalog + lookup helpers:
                 `list_materials()`, `get_material()`, `material_info()`
+ meshing.py    / Turns parametric design into rasterized grid.
+ io.py         / Import and export of the design as .gds, .gltf, etc.

## Notes
- Dispersive models are operating-point models and must be converted using
  `to_material(frequency=... or wavelength=...)` before mesh/simulation use.
- Static catalog constants in `library.py` are referenced to 1.55 um.
