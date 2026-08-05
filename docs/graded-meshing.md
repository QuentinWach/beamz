# Automatic Graded Meshing

A good photonics mesh must resolve wavelength and small gaps without turning the
entire simulation domain into the finest cell size. It must also avoid abrupt
fine-to-coarse transitions, which add truncation error and can create numerical
reflections. BeamZ's graded mesher builds one nonuniform rectilinear axis at a
time, then uses the resulting physical edge arrays throughout rasterization,
the Yee updates, CPML, and the CFL calculation.

## Quick start

```python
import beamz as bz

spec = bz.GridSpec.auto(
    wavelength=1.55 * bz.um,
    min_steps_per_wvl=16,
    min_feature_cells=6,
    max_scale=1.2,
)

grid = spec.realize(design)
print(grid.shape)
print(grid.quality_report())

simulation = bz.Simulation(
    design=design,
    grid_spec=spec,
    sources=sources,
    monitors=monitors,
    run_time=2e-12,
)
assert simulation.grid == grid
```

`GridSpec.auto()` is the canonical automatic meshing constructor and produces a
geometry-aware nonuniform grid. Use `GridSpec.uniform()` when equal cell widths
are required explicitly.

An explicit uniform spacing is retained exactly on every active axis. When a
domain extent is not divisible by that spacing, the computational grid is
padded by less than one terminal cell. Consequently, the requested design size
and realized computational extent can differ slightly, while
`spec.realize(design)`, `simulation.grid`, and the rasterized material grid
remain identical.

## How the policy is resolved

For each active axis, BeamZ:

1. collects the domain bounds, projected structure bounds, user snapping
   points, and mesh-override boundaries;
2. assigns local upper bounds on cell width from wavelength in every material,
   simulation size, structure thickness, and transversely overlapping gaps;
3. propagates those limits through a continuous piecewise-linear spacing
   envelope;
4. integrates reciprocal spacing to place cell edges, while retaining every
   mandatory coordinate exactly; and
5. verifies the realized cells and locally refines until both the spacing caps
   and adjacent-cell ratio hold.

The realization is deterministic. Even a tiny snapped interval is treated as a
local spacing constraint and graded into its neighbors rather than left as an
isolated sliver.

The policy combines material-wavelength targets, a maximum consecutive-cell
scale, lower spacing bounds, and regional overrides. BeamZ's spacing-envelope
implementation is independent, so behavioral parity does not require edge
coordinates to match bit for bit. A pinned differential suite covers
homogeneous, dielectric, coupled-gap, ring, override, and snapping cases. It
gates cell count within 4%, normalized spacing-profile error within 8%, domain
bounds, explicit snapping, and the requested adjacent-cell ratio. BeamZ defaults
to `max_scale=1.3`; 1.1–1.2 is a useful conservative range for especially gentle
transitions.

With the default `min_feature_cells=1`, local resolution is primarily set by
wavelength in material, while a narrow feature or gap is guaranteed at least
one cell. Increase it explicitly when a coupling gap or thin film needs more
cells for a convergence study; this is deliberate extra refinement rather than
part of cross-solver parity.

Semantic primitives report physical dimensions directly: circle and sphere
diameters, ring and bend wall thicknesses, taper output widths, and box sizes.
For explicit polygons, BeamZ measures distances between opposing material or
gap boundaries. It retains the nearest reliable face pair, builds a local
normal-direction refinement corridor, and merges overlapping corridors into a
one-dimensional spacing envelope. A narrow neck therefore refines the axis
normal to that neck near the feature instead of forcing the finest spacing over
the polygon's whole bounding box. Adjacent vertex spacing is never interpreted
as a feature size, so adding more points to a curved boundary does not force a
finer mesh.

## Geometry treatment

A rectilinear mesh cannot make its grid lines circular. Curved interfaces are
therefore handled by subpixel constitutive averaging rather than by visually
staircasing the material at cell centers:

```python
material_grid = design.rasterize(
    grid,
    smoothing="farjadpour_full",
    quality="balanced",
)
```

The Farjadpour interface transform is based on electromagnetic perturbation
theory and was shown to recover quadratic convergence for arbitrarily sloped
dielectric interfaces. Sharp corners and unreliable local interface normals
fall back to volume tensor averaging and are counted in raster diagnostics.

## Local control

Refine a region without changing the physical design:

```python
spec = bz.GridSpec.auto(
    wavelength=1.55 * bz.um,
    overrides=(
        bz.MeshOverride(
            center=(8 * bz.um, 3 * bz.um),
            size=(3 * bz.um, 1 * bz.um),
            dl=(30 * bz.nm, 20 * bz.nm),
        ),
    ),
    snapping_points=((8 * bz.um, None),),
)
```

A normal override only refines the automatic material target. Set
`enforced=True` to replace that target inside the override. `dl_min` is a global
lower bound on requested refinement and `dl_max` is a global upper bound on
automatic cell width.

## Allocation safety

Automatic refinement is checked before material or field arrays are allocated.
The defaults permit at most 200,000 cells on one active axis and 20,000,000
active Cartesian cells. Tighten these limits for interactive work or raise them
explicitly for a deliberately large production run:

```python
spec = bz.GridSpec.auto(
    wavelength=1.55 * bz.um,
    max_cells_per_axis=50_000,
    max_total_cells=20_000_000,
)
```

Uniform cell counts and conservative graded-axis lower bounds are checked before
edge generation. The graded mesher then rechecks its exact count before every
temporary edge/owner allocation and after refinement passes; the active
Cartesian product is rechecked as each axis is finalized. An exceeded budget
reports the predicted shape or offending axis, the smallest requested spacing,
the limiting interval when available, a conservative memory estimate, and the
violated limit. Prefer raising `dl_min` or relaxing a local override before
increasing a budget.

## Result coordinates

Compiled field arrays use a solver-local grid, while analysis results expose
public physical coordinates. Imported grids with nonzero origins and centered
domains are translated through the same `coordinate_offset` contract in xarray,
plots, and videos; the offset is never applied twice. For 2D `xy`, `xz`, and
`yz` simulations, stored raster axes are mapped explicitly into physical XYZ
before offsets are applied, including two-coordinate sources and three-coordinate
monitors. Three-dimensional slice results retain their exact compiled sample
region, so videos use the actual nonuniform tangential edges and snapped plane
position rather than reconstructing them from one scalar resolution.

## Quality checks

The realized grid reports measurable invariants:

```python
report = grid.quality_report()
assert report.satisfies_max_scale(1.2, active_axes=("x", "y"))
print(report.x.minimum_spacing, report.x.maximum_spacing)
print(report.max_adjacent_ratio)
```

These checks establish mesh construction quality, not solution convergence.
For production resonators, repeat the simulation with higher
`min_steps_per_wvl`, more cells across the coupling gap, and stricter
`max_scale`; compare resonance wavelength, linewidth/Q, and port power. A
single attractive mesh plot is not a convergence study.

## Current boundary

`GaussianSource`, `CustomSource`, `ModeSource`, and field, flux, and mode
monitors work on graded grids. Mode solving, source launch, monitor colocation,
and modal projection consume the realized transverse coordinates and local
integration weights. A Gaussian beam still selects an isotropic uniform grid
at the same material-wavelength target because that operator requires constant
spacing. The core FDTD propagation, rasterizer, CPML, and CFL calculation all
consume rectilinear metrics directly.

## References

- A. Farjadpour et al., [“Improving accuracy by subpixel smoothing in the finite-difference time domain”](https://doi.org/10.1364/OL.31.002972), *Optics Letters* 31, 2972–2974 (2006).
- C. Kottke, A. Farjadpour, and S. G. Johnson, [“Perturbation theory for anisotropic dielectric interfaces, and application to subpixel smoothing of discretized numerical methods”](https://arxiv.org/abs/0708.1031) (2007).
- A. Spanakis-Misirlis, [“Efficient Non-Uniform Structured Mesh Generation Algorithm for Computational Electromagnetics”](https://arxiv.org/abs/2209.10260) (2022).
