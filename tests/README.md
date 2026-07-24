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

The CI evidence job uploads this JSON alongside the coverage data. Threshold
metrics preserve whether a value is an equality target, upper bound, or lower
bound; a result below a −40 dB ceiling is not misreported as being “far from
the reference.”
Unexpected warnings fail the suite. Plain skips in `validation/` and
`differential/` are converted to failures so missing numerical evidence cannot
silently pass; named strict `xfail` regressions remain visible and allowed.
Hypothesis exercises randomized public-API and interpolation contracts, while
fault-sensitivity tests demonstrate that representative missing-derivative and
wrong-sign operator mutations are rejected by the numerical invariants.

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

## Current verification boundary

The compact gate now makes the following quantitative claims:

- reference-normalized normal-incidence Fresnel power;
- three-level second-order Yee-curl convergence;
- plane-wave phase velocity, impedance, amplitude, direction, and polarization
  against the discrete Yee relation in every 2D plane;
- bounded CPML packet reflection in vacuum and a dielectric;
- slab-waveguide effective index against the analytical dispersion equation;
- straight-waveguide launch, reflection, transmission, reference-plane, and
  resolution metrics;
- reciprocal forward/reverse straight-waveguide transmission, apart from the
  strict named regression for reverse 2D mode-source leakage;
- directional derivatives for the available optimization primitives;
- one-step lossless Yee time reversibility and mutation-sensitive
  divergence-of-curl detection;
- execution of all notebooks in reduced mode against an isolated built wheel.

These claims are emitted as structured JSON. The pull-request gate enforces
combined statement-and-branch coverage at 80%, changed-line coverage at 100%,
and the risk-weighted floors in `tests/coverage_policy.json`: 87% for public
configuration, 90.5% for the solver core, 77% for PIC analysis, and 93.5% for
numerical kernels. The top-level public API also has an executable inventory:
adding an export requires classifying it, and every public configuration object
shares frozen-state and nested-immutability contracts. Available functional
copy APIs are checked as well. The longer-term 90% global branch-inclusive
target remains a target, not a current claim.

The suite deliberately does **not** yet claim complete Mie scattering, cavity
resonance/Q validation, end-to-end FDTD adjoints, external-solver consensus,
CUDA parity, multi-GPU equivalence, or stable performance trends. Those require,
respectively, closed-contour flux/TFSF support, calibrated long ringdowns,
integrated adjoint execution, installed external adapters, real accelerators,
and a controlled benchmark host. Their schemas and evidence directories are in
place, but promoting them to validation requires the missing independent
observable or execution environment. Large 3D sweeps, long-time CPML stability,
and broad convergence/performance matrices belong in weekly or controlled
hardware runs rather than the compact pull-request gate.
