# Design

Module to define complex structures parametrically and rasterize them into material grids.

+ core.py       / Main module to define and organize geometry and materials.
+ structures.py / Polygon objects to define geometry within the design.
+ materials.py  / Canonical material models (`Material`, `CustomMaterial`,
                 `SellmeierMaterial`, `DrudeMaterial`,
                 `LorentzMaterial`, `DebyeMaterial`, `PoleResidueMaterial`,
                 `Material2D`, `AnisotropicMaterial`).
+ library.py    / Curated function-based material registry + `material_library` dataset
                 (`MaterialItem`, variants/default/medium contract, including
                 dispersive Sellmeier/Drude/Lorentz/Debye models).
+ meshing.py    / Turns parametric design into rasterized grids.
+ io.py         / Import and export of designs as .gds, .gltf, etc.

## Notes
- Dispersive models run natively in `Simulation.step()` and the core
  `Simulation.run()` loop through ADE updates.
- `Simulation.run_jit_scan()` remains a compatibility wrapper and routes to
  `run()`.
- Thermal coupling with dispersive ADE materials is currently not supported.
- The runtime material catalog is intentionally curated and minimal.

## Quick Visualization
```python
from beamz.design.library import material_library
from beamz.design.materials import Material

# Constant material summary
Material(name="SimpleOxide", permittivity=2.1, conductivity=1e-4).show()

# Dispersive material from library
material_library["Gold"].medium.show(wavelength_range_um=(0.45, 1.8))

# Uniaxial LiNbO3 (extraordinary axis along z)
lno = material_library["LiNbO3"].medium("z")
lno.show(wavelength_range_um=(0.5, 2.0))
```
