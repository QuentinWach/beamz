# Design

Module to define complex structures parametrically and rasterize them into material grids.

+ core.py       / Main module to define and organize geometry and materials.
+ structures.py / Polygon objects to define geometry within the design.
+ materials.py  / Canonical material models (`Material`, `CustomMaterial`, `PECMaterial`,
                 `PMCMaterial`, `SellmeierMaterial`, `DrudeMaterial`,
                 `LorentzMaterial`, `DebyeMaterial`, `PoleResidueMaterial`,
                 `Material2D`, `AnisotropicMaterial`).
+ library.py    / Curated function-based material registry + `material_library` dataset
                 (`MaterialItem`, variants/default/medium contract, including
                 dispersive Sellmeier/Drude/Lorentz/Debye models).
+ meshing.py    / Turns parametric design into rasterized grids.
+ io.py         / Import and export of designs as .gds, .gltf, etc.

## Notes
- Dispersive models are operating-point models and must be converted using
  `to_material(frequency=... or wavelength=...)` before mesh/simulation use.
- The runtime material catalog is intentionally curated and minimal.
