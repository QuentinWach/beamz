# Test Structure

The suite currently mixes several kinds of tests. New tests should be placed and
marked by contract, not just by subsystem.

## Test categories

- `unit`: Fast tests for public API behavior and pure helper logic.
- `component`: Small real-component tests with limited simulation scope.
- `integration`: End-to-end tests that span multiple subsystems.
- `characterization`: Numerical sweeps and behavior-profiling tests. These are
  useful for research and regression analysis.
- `compiled`: Tests that exercise the compiled/JAX execution path.
- `pdk`: Tests that depend on an external PDK or design-kit install.
- `slow`: Expensive tests. These remain marked for discoverability, but the
  default CI gate runs them.

## Running tests

The pull-request and main-branch CI gates run the complete test suite with
coverage:

```bash
python -m pytest tests/ --cov=beamz --cov-report=term-missing --cov-report=xml --cov-fail-under=75
```

For local development, use the same batch:

```bash
make test
```

## Placement guidance

- Keep API normalization and validation tests in focused files such as
  `test_simulation_api.py`.
- Keep small compiled-engine and monitor tests in compiled/component files.
- Keep expensive physics validation in integration-oriented files.
- Keep broad numerical sweeps and exploratory residual checks in
  characterization files, not mixed into core API tests. They still run in CI;
  the marker is for ownership and selective local debugging.

## Rules of thumb

- Prefer public-behavior assertions over private attribute checks.
- Avoid asserting internal helper names or exact implementation wording in
  exception messages.
- Do not freeze known defects as expected behavior.
- Mark heavy tests explicitly so they can be found and run selectively during
  local debugging.
