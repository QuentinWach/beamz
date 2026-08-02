# Changelog

## Unreleased (target: v0.5.0)

This is a breaking, pre-1.0 architecture release. BeamZ now separates immutable
simulation configuration, compiled execution, evolving runtime state, and detached
analysis results. Public devices are immutable specifications, compiled plans and
caches are private implementation details, and completed results no longer retain or
mutate a live simulation.

Existing v0.4 code may require import and lifecycle changes. The most common
migrations are listed below; removed compatibility modules are not preserved as
aliases.

### Added

- Integrated the complete MicroMode finite-difference eigensolver into
  `beamz.devices.modes`, including the latest guarded Yee-grid refinement and
  validation work from `beamzorg/micromode@80c57d8`.
- Added a public native mode-solver package for rasterized solves, modal sweeps,
  overlap analysis, and optional HDF5 result persistence through
  `beamz[mode-io]`.
- Added immutable `SimulationState`, `SimulationRun`, and detached
  `SimulationResults` ownership contracts.
- Added canonical immutable source, monitor, boundary, port, material, geometry, and
  topology specifications with functional `updated_copy(...)` operations.
- Added result-owned source normalization, modal projection, S-parameter extraction,
  labeled-data adapters, plotting, and field-video support through `beamz.analysis`.
- Added explicit compilation, continuation, sharding, cache, packaging, public-API,
  architecture, and physics regression coverage.

### Changed

- Mode sources, monitors, and ports now share `ModeSpec` and `ModeData` from
  `beamz.devices.modes`; the former source import paths remain compatibility
  re-exports.
- Removed the external `micromode` runtime dependency. Mode solving and source
  compilation now use the BeamZ-owned implementation directly.
- `Simulation.run()` is now the normal complete execution path and returns detached
  `SimulationResults`. Use `Simulation.advance()` for continuation or checkpointing,
  and `Simulation.step()` only for single-timestep debugging.
- Simulation configuration changes now return a new value through
  `Simulation.updated_copy(...)`; simulations and device specifications no longer own
  mutable runtime fields or monitor buffers.
- Grid materialization is deferred until compilation or execution. `GridSpec`
  describes the requested grid, while `Design.rasterize()` returns material data.
- The Python `RegularGrid` rasterizer and callback-based `CustomMaterial` API
  were removed. Rust rasterization is the only geometry path for `Design`,
  simulations, imported GDS, STL, Gmsh, meshio, and raw-mesh scenes. Pass
  pre-sampled spatial coefficients directly as `MaterialGrid`.
- GDS support is now optional and loaded lazily. Install `beamz[gds]` before calling
  the GDS import or export functions.
- Plotting and labeled-data conversion are loaded lazily from `beamz.analysis` so the
  numerical engine does not depend on presentation or xarray state.

### Breaking API migration

| v0.4 API | v0.5 API | Migration notes |
| --- | --- | --- |
| `beamz.simulation.core.Simulation` | `beamz.Simulation` or `beamz.simulation.Simulation` | The legacy `core` module was removed. Configuration is immutable. |
| `Simulation.run_compiled()` and `compile_simulation()` | `Simulation.run()` | All execution now uses the compiled engine. `Simulation.compile()` remains available for advanced inspection or prewarming. |
| `Simulation.run_compiled_until_decay()` | Repeated `Simulation.advance()` calls | There is no built-in decay-loop replacement. Evaluate the stopping criterion from each detached result and pass its state into the next chunk. |
| `RunState`, `EngineState`, and live simulation fields | `SimulationState` | `step()` returns state. `advance()` returns `SimulationRun(results, state)` for continuation and branching. |
| Mutable simulation snapshots and monitor side channels | `SimulationResults` and `MonitorResults` | Read acquisitions from `results.monitor(name)` or `results.monitors[name]`; a completed result is detached from the simulation. |
| `CompiledSimulation`, `CompiledRunConfig`, and `MonitorState` | No public replacement | Compiled plans, executable caches, and monitor accumulators are private implementation details. |
| `RegularGrid` | `GridSpec`, `Design.rasterize()`, and `MaterialGrid` | Use `GridSpec` for simulation grid configuration and rasterize only when direct material arrays are required. |
| `Medium` | `Material` | Materials are immutable value specifications. |
| Public `Structure` construction | `Box`, `Rectangle`, `Circle`, `Ring`, `CircularBend`, `Polygon`, `Taper`, or `Sphere` | Use concrete immutable geometry and `updated_copy(...)` or `with_material(...)` for changes. |
| Generic `Monitor` | `FieldMonitor`, `FieldRecorder`, `FluxMonitor`, or `ModeMonitor` | Configure the concrete acquisition required by the analysis. |
| `Boundary`, `BoundarySpec`, and `AbsorbingLayer` | `PEC`, `PML`, and `Absorber` | Boundaries are canonical immutable device specifications; use `Absorber` for the former sponge-style absorbing layer. |
| `PortSpec` | `Port` | A single canonical port creates matching source and monitor specifications. |
| `ModeSolver` and public `solve_modes()` workflows | `ModeSpec` on `ModeSource` or `ModeMonitor` | Mode solving is planned internally. Read modal results with `results.mode(name)` or `beamz.analysis.mode_data(...)`. |
| `beamz.data` xarray wrappers | `SimulationResults.to_xarray()` or `beamz.analysis.to_xarray(...)` | Labeled data is created lazily from detached results. Generic `colocate_dataset`, `field_intensity`, and `poynting_vector` helpers were removed without one-to-one replacements. |
| `beamz.visual` plotting functions and browser scene | `design.plot()`, `simulation.plot()`, `simulation.view3d()`, and `results.plot_field()` | Plotting now uses static Matplotlib-backed analysis adapters; the interactive browser viewer was removed. |
| `TopologyManager` | `TopologySpec` and `TopologyState` | Immutable optimization configuration is separate from evolving density and optimizer state. |
| Top-level `transform_density`, `compute_overlap_gradient`, and `create_optimization_mask` | `beamz.optimization.autodiff.transform_density` and `beamz.optimization.topology` helpers | Advanced optimization functions are no longer part of the top-level `beamz` namespace. |
| Top-level `ShardingConfig` | `beamz.simulation.model.ShardingConfig` | Sharding remains an advanced execution option passed to `compile()`, `advance()`, or `run()`. |
| `beamz.devices.sources.signals.ramped_cosine` | `beamz.ramped_cosine` or `beamz.devices.sources.time.ramped_cosine` | Source-time definitions now live together in the source-time module. |

The normal lifecycle is now:

```python
simulation = simulation.updated_copy(sources=new_sources)
results = simulation.run()

# Use this form only when continuation state is required.
first = simulation.advance(num_steps=100)
second = simulation.advance(state=first.state, num_steps=100)
```

### Removed

- Removed the legacy `beamz.simulation.core`, `beamz.simulation.compiled`,
  `beamz.simulation.fields`, `beamz.simulation.ops`, `beamz.simulation.specs`, and
  `beamz.simulation.yee` public modules.
- Removed the legacy `beamz.data` and `beamz.visual` packages in favor of detached
  result adapters and `beamz.analysis`.
- Removed compatibility aliases and mutable compiler/device hooks that duplicated the
  canonical immutable APIs described above.

## v0.4.3 - 2026-06-26

### Changed
- Updated cosine crossing and modal source example notebooks.

### Fixed
- Fixed mode-source flux calculation and expanded related mode-source and tidy API coverage.

## v0.4.2 - 2026-06-22

### Changed
- Improved matplotlib field plotting with permittivity overlays for field-frame and DFT field views.
- Refined material overlay styling so real fields use darker structure overlays and power fields use lighter overlays.

### Fixed
- Expanded visualization coverage for real-field and power-field overlay behavior.

## v0.4.1 - 2026-06-22

### Changed
- Improved matplotlib field plotting aliases, marker orientation handling, and DFT field component extraction.
- Clarified README terminology around sub-pixel averaging.

### Fixed
- Preserved `SimulationResults.plot_field` behavior while forwarding monitor and field aliases through keyword arguments.
- Added clearer errors for missing DFT field components and expanded visualization coverage.

## v0.4.0 - 2026-06-20

### Added
- Added TFSF support for mode sources.
- Added CPML waveguide benchmarking documentation and scripts.
- Added expanded tests for mode sources, PML/CPML behavior, animation, visualization, and curl kernels.

### Changed
- Improved CPML behavior to reduce waveguide reflections.
- Updated examples and notebooks for modal source and monitor workflows.
- Refined license labeling and documentation around recommended and experimental examples.

## v0.3.2 - 2026-06-13

### Changed
- Reduced compiled-engine memory use and simulation compile overhead for larger 3D runs.
- Simplified boundary and compiled simulation internals while preserving the public API.
- Streamlined examples, benchmark scripts, and development tooling to reduce repository size.
- Updated project licensing metadata to Apache-2.0.

### Fixed
- Improved 3D permittivity handling and memory estimation in simulation setup.
- Fixed mode profile data handling for visualization workflows.
- Adjusted tests and CI coverage around compiled-engine, boundary, and 3D constitutive behavior.

## v0.3.1 - 2026-05-31

### Added
- Added a compact demo example for the demux workflow.

### Changed
- Reduced memory load in boundary and compiled simulation paths.

### Fixed
- Fixed the demux example.

## v0.3.0 - 2026-05-26

### Added
- Added modal port workflows with `Port` and `ModeMonitor` support.
- Added xarray-backed result accessors and plotting conveniences for simulation data.
- Added matplotlib visualization helpers for snapshots, fields, layouts, mode fields, and Tidy3D-style DFT views.
- Added UBC PDK support and improved gdsfactory component handling.
- Added broader 2D/3D physics, monitor, source, and engine-equivalence test coverage.

### Changed
- Replaced the Tidy3D mode-solver dependency path with `micromode`.
- Improved 3D mode-source normalization, source quadrature handling, Yee phase-plane calibration, and monitor DFT/modal projection behavior.
- Refactored CPML/PML handling, including sponge-style absorbing layers and compatibility through `AbsorbingLayer`.
- Improved material sampling on Yee components, full-PEC 3D sampling, and source scattering/S-parameter calculations.
- Promoted the Design material API and added simulation-domain/depth convenience aliases.
- Updated development tooling around `uv`, `ruff`, `vulture`, and Makefile audit commands.

### Fixed
- Fixed 2D and 3D mode-source handedness, power normalization, and source-plane phase-referenced test expectations.
- Fixed cropped Yee profile interpolation and modal monitor projection alignment.
- Fixed 3D CPML compiled-engine behavior and related monitor/source reconstruction cases.
- Restored the dipole example and reverted an incompatible material-handling refactor.
