# Differential cases

Solver-neutral JSON specifications live in `cases/`. They describe physical
inputs and observable outputs without exposing any solver's internal field
layout. `case_schema.py` validates them before an adapter is allowed to run.

Oracle priority is explicit in every case:

1. analytical solution;
2. mathematical invariant;
3. independent-solver consensus;
4. historical BeamZ regression data.

External adapters should return only the named observables (for example
effective index, reflected power, or an S-parameter). Optional Meep and FDTDX
jobs can therefore be added without turning either solver into the sole truth.

## Tidy3D rectilinear-grid references

`tidy3d_grid_references.json` contains only the x/y grid-boundary coordinates
exported from Tidy3D 2.12.0 for the solver-neutral cases declared in
`tidy3d_grid_adapter.py`. The cases cover a homogeneous domain, high-index and
coupled rectangles, a ring/bus coupler, an explicit mesh override, snapping
points, and an enforced coarse override. BeamZ's normal test suite consumes the
frozen coordinates and does not depend on Tidy3D.

Regenerate the reference file explicitly with the pinned external version:

```shell
uv run --with tidy3d==2.12.0 \
  python scripts/export_tidy3d_grid_references.py
```

The exporter fails for any other Tidy3D version so a solver upgrade cannot
silently move the differential baseline.
