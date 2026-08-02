# BeamZ rasterization

This package turns analytic geometry, GDSFactory layer stacks, and closed
triangle meshes into static real tensor material arrays. Permittivity and
permeability may be isotropic, diagonal, or symmetric positive-definite 3x3
tensors; conductivity may be symmetric positive-semidefinite.

## API

```python
import beamz
import beamz.design.raster as raster

scene = raster.Scene(
    materials=(
        beamz.Material(),
        beamz.Material(permittivity=((12.1, 0, 0), (0, 11.8, 0), (0, 0, 10.9))),
    ),
    objects=(
        raster.Object(
            raster.Box((0, 0, 0), (2e-6, 0.5e-6, 0.22e-6)),
            material_id=1,
        ),
    ),
)
grid = raster.Grid.uniform(
    (0, -0.5e-6, -0.1e-6),
    (2e-6, 1e-6, 0.4e-6),
    (80, 60, 8),
)
result = raster.rasterize(scene, grid)
```

The public package exports:

- `Scene`, `Object`, and scalar/diagonal/symmetric-tensor `Material`
- `Grid`, including explicit nonuniform edges and `Grid.uniform()`
- boxes, spheres, cylinders, extruded polygons, tapered polygons, and meshes
- `RasterOptions`, `RasterResult`, and `CompiledScene`
- `compile_scene()`, `rasterize()`, and `inspect_mesh()`

`RasterOptions` has three choices: `quality` (`fast`, `balanced`, or
`reference`), `smoothing` (`volume`, `farjadpour_diagonal`, or
`farjadpour_full`), and `components` (`all`, `two_dimensional_tm`, or
`two_dimensional_te`). Adaptive
tolerances and threading remain internal. Farjadpour modes use the generalized
local-interface tensor transform for permittivity and permeability;
conductivity remains volume averaged.

`RasterResult` contains immutable grid edges, cell-centered `tensors`, summary
diagnostics, and constitutive `yee_tensors` integrated independently over the
Ex/Ey/Ez and Hx/Hy/Hz dual volumes. Tensor arrays use one leading component for
isotropic materials, three `(xx, yy, zz)` components for diagonal materials,
and six `(xx, yy, zz, xy, xz, yz)` components for full symmetric materials. It
does not produce dense material-ID, boundary-mask, or error arrays.

Farjadpour smoothing is applied only when the surface patches crossing one Yee
support form a reliable, sign-invariant lamination axis. Corners, non-coplanar
mesh patches, unresolved geometry, and overlapping objects fall back to tensor
volume averaging. Summary diagnostics count each fallback reason without
allocating dense diagnostic fields.

Compile once when rasterizing one scene on several grids:

```python
compiled = raster.compile_scene(scene)
coarse = compiled.rasterize(coarse_grid)
fine = compiled.rasterize(fine_grid)
```

Passing `cache_directory=` enables atomic, schema-versioned NPZ caching with
corrupt-cache recovery.

## Standalone versus simulation use

Standalone rasterization accepts uniform and nonuniform rectilinear grids.
BeamZ's current FDTD engine requires a single uniform resolution and supports
componentwise diagonal material coefficients. `MaterialGrid.from_raster_result()`
therefore rejects nonuniform and `farjadpour_full` results. It extracts the
target diagonal from each `farjadpour_diagonal` support tensor and never
silently discards off-diagonal terms.

Imported scenes use the same simulation workflow as BeamZ designs; the
solver-specific conversion stays internal:

```python
import beamz
from beamz.design.raster import Grid
from beamz.design.raster.importers import from_mesh

scene = from_mesh("device.msh", material=beamz.Material(permittivity=12.0))
raster_grid = Grid.uniform(minimum, maximum, shape)
simulation = beamz.Simulation(
    scene=scene,
    raster_grid=raster_grid,
    run_time=1e-12,
)
```

The simulation domain is inferred from the grid shape and resolution, while the
raster origin is retained for source and monitor coordinates.

For standalone inspection, `scene.rasterize(raster_grid)` returns the general
`RasterResult`. `MaterialGrid.from_raster_result()` remains an advanced explicit
boundary for callers that need to retain or transform that result themselves.

`Design.rasterize()` returns a `MaterialGrid` directly, always uses the Rust
engine, and selects the reduced TMz or TEz work set requested by the simulation.
Pre-sampled
spatial coefficients enter `Simulation` directly as `MaterialGrid`; there is no
second Python geometry rasterizer or engine-selection switch.

Normal simulations retain volume averaging unless an explicit native policy is
requested:

```python
import beamz

simulation = beamz.Simulation(
    design=design,
    raster_options=raster.RasterOptions(smoothing="farjadpour_diagonal"),
    run_time=1e-12,
)
```

The compiler consumes those native Yee arrays directly. Full Farjadpour tensors,
nonuniform grids, off-diagonal volume tensors, and non-unit permeability fail at
the `MaterialGrid` boundary with a capability-specific message. Conductive
materials propagate in FDTD but are rejected for `ModeSource` until the mode
bridge represents their frequency-dependent complex permittivity. The current
2D and public convenience mode solvers likewise reject direct-Yee grids rather
than solving a cell-centered approximation; the 3D launch planner retains its
Yee-aware refinement path.

## Importers

```python
from beamz.design.raster.importers import (
    from_gdsfactory,
    from_mesh,
    from_mesh_arrays,
    repair_mesh,
)
```

- `from_gdsfactory()` reads PDK layer stacks, derived layer regions, sidewalls,
  material maps, and callable or tabulated `z_to_bias` profiles.
- `from_mesh()` reads meshio-supported surfaces such as STL and extracts closed
  boundaries from Gmsh tetrahedral physical regions.
- `from_mesh_arrays()` accepts raw vertices and triangles.
- `repair_mesh()` uses optional trimesh operations and returns an auditable
  immutable report.

All importers return reusable `Scene` objects. GDSFactory, meshio, and trimesh
remain optional dependencies.

PDK material names have no implicit production constants. Pass `material_map`
with the intended values. `use_builtin_materials=True` is an explicit opt-in to
approximate nondispersive values near 1.55 µm.

Meshes must be closed, consistently oriented, manifold, nondegenerate, and free
of self-intersections. `inspect_mesh()` and scene compilation share the same
native scale-aware validity predicate.

## Development

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
env -u CONDA_PREFIX uv run maturin develop
uv run pytest tests/unit/raster \
  tests/integration/test_native_design_rasterization.py
```
