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
