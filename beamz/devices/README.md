# Devices

Simulation devices define how fields enter, leave, and are sampled from a
simulation.

- `modes/`: native finite-difference mode solving, modal results and sweeps,
  BeamZ-shaped discrete modes, and guarded Yee-grid refinement.
- `modes/specs.py`: immutable mode-selection and mode-data values shared by
  sources, monitors, and ports.
- `sources/specs.py`: immutable public source values and compatibility re-exports
  for `ModeSpec` and `ModeData`.
- `sources/time.py`: sampled and analytic temporal waveforms.
- `sources/solve.py`: source-plane extraction and compatibility wrappers around
  the native mode solver.
- `sources/mode_profiles.py`: mode-profile geometry, interpolation, and power scaling.
- `sources/mode_launch.py`: 2D/3D launch planning from solved profiles.
- `sources/planar_tfsf.py`: discrete 3D total-field/scattered-field residuals.
- `sources/compiler.py`: the single lowering boundary into executable source plans.
- `monitors/monitors.py`: immutable public monitor specifications.
- `monitors/compiler.py`: grid placement and packed acquisition plans. Runtime
  accumulation belongs to `simulation.observe`, not to device specifications.
- `ports.py`: named modal port metadata used by S-parameter analysis.
- `boundaries.py`: immutable PEC, sponge `Absorber`, and PML specifications.
- `_placement.py`: shared grid-snapping rules for sources and monitors.
- `_boundary_compile.py`: grid-aware PEC/PML/absorber lowering kept separate from
  the public boundary values for the same reason as source and monitor compilation.
- `_immutable.py`: array freezing and canonicalization shared by every device spec.
- `visualization.py`: data-only visual descriptions consumed by analysis plotting.

These files separate public values, numerical planning, and runtime execution. Avoid
adding per-device facade modules; a new file should own a distinct numerical stage or
be folded into the nearest existing owner.
