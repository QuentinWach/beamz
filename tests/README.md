# BeamZ testing constitution

BeamZ tests are organized by the evidence they provide. Passing code coverage
is useful engineering evidence, but it is not proof that a simulated physical
quantity is correct.

## Primary evidence classes

Every collected test receives exactly one primary marker from its directory:

- `contract`: the API or an execution path behaves as promised;
- `invariant`: a mathematical or physical identity is preserved;
- `validation`: a BeamZ observable agrees with an independent oracle;
- `characterization`: the suite records current behavior without claiming that
  the behavior is physically correct.

A smoke test belongs in `characterization/` and uses the additional `smoke`
marker. “The field is finite/nonzero” is valid smoke evidence, never analytical
validation.

`differential`, `hardware`, `performance`, `compiled`, `pdk`, and `simulation`
are orthogonal scope markers. They do not replace a primary evidence class.
Collection fails when a file is outside the evidence-oriented directory tree or
when an explicit primary marker conflicts with its placement.

## Directory ownership

```text
tests/
├── unit/                    # Pure-function and local behavior contracts
├── contracts/               # Public API, immutability, caching, serialization
├── kernels/                 # Discrete operators and local-update invariants
├── integration/             # Multiple BeamZ subsystems and execution paths
├── characterization/        # Honest smoke checks and exploratory behavior
├── validation/
│   ├── analytical/          # Comparison with independently derived formulas
│   ├── invariants/          # Maxwell, energy, reciprocity, symmetry
│   ├── convergence/         # Measured refinement order
│   └── regression/          # Named historical physics failures
├── differential/            # Solver-neutral cross-solver observables
├── hardware/                # CPU/GPU, precision, sharding, multi-device
├── performance/             # Runtime, compilation, memory, scaling
├── docs/                    # Executed documentation and examples
└── pdk/                     # External PDK-dependent cases
```

## Rules for scientific claims

1. State the observable, independent oracle, error definition, and tolerance.
2. Normalize power measurements against an incident/reference run.
3. Keep analytical helper tests in `unit/`; they test the oracle, not BeamZ.
4. A convergence claim must calculate and gate a convergence order.
5. Report measured values even when a test passes.
6. Calibrate tolerances from a documented refinement study, not a single run.
7. Use deterministic random generators and include the seed in failures.
8. Do not hide a validation case with `__test__ = False`, an unmarked skip, or
   an unbounded tolerance. Move unfinished work to a case specification or
   characterize it honestly.

Named gates live in `validation/tolerances.py`. Validation tests use the
`validation_metrics` fixture so each assertion records its measured value,
reference, absolute/relative error, resolution, backend, and tolerance
rationale:

```python
def test_observable(validation_metrics):
    validation_metrics.check(
        "reflectance",
        measured=measured_R,
        reference=analytical_R,
        tolerance="analytical_coarse",
        resolution="20 ppw",
    )
```

Emit the collected measurements as portable JSON with:

```bash
python -m pytest -m validation --validation-report=validation-results.json
```

## Local gates

The compact pull-request-style suite excludes explicitly slow simulations,
external PDKs, and hardware-specific work:

```bash
python -m pytest -m "not slow and not pdk and not hardware"
```

Run a focused evidence class with, for example:

```bash
python -m pytest -m invariant
python -m pytest -m validation
python -m pytest -m characterization
```

The full CPU suite remains:

```bash
python -m pytest tests/
```
