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

## Dispersion Validation
Use the pulse-through-slab showcase to inspect time-domain behavior and compare
extracted spectral response against analytic material models:

```bash
uv run python examples/7_dispersion_showcase.py --case all
```

Useful options:
- `--case sellmeier|drude|debye|all`
- `--fast` for shorter runs
- `--save` to write figures/animations to `artifacts/dispersion_showcase/`

Reading the plots:
- Left panel: time-domain field evolution through the slab.
- Right panels: extracted vs reference `n`, `k`, `Re(eps_r)`, `Im(eps_r)`.
- The summary table reports RMSE and max absolute errors in passband.

Caveats:
- Coarse grids and short windows bias extracted values; trends are more reliable
  than absolute metrology.
- If source-driven ADE fields become non-finite for a case, the showcase falls
  back to model-based spectral extraction and labels that mode in the summary.
