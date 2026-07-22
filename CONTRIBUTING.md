# Contributing to BeamZ

Thank you for contributing. Bug fixes and focused improvements are welcome as
pull requests. For new features or public API changes, open an issue first so we
can agree on the direction before substantial work begins.

## AI-Assisted Contributions

AI-assisted coding workflows are allowed, but AI tools are only tools. The human
author owns the change and is responsible for reviewing, understanding, testing,
and defending everything they submit.

Contributors must be transparent when a meaningful part of a change was created
or rewritten with AI assistance. Mention the tool or model used in the pull
request description or commit notes, and describe which parts of the change it
affected. Maintainers may ask for extra explanation, review, or testing for
AI-assisted changes.

## Set up your environment

BeamZ supports Python 3.10 through 3.14 and uses
[uv](https://docs.astral.sh/uv/) for environments and dependencies. The default
development version is defined in `.python-version`.

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on macOS
brew install uv
```

Clone BeamZ and install its contributor tooling:

```bash
git clone https://github.com/beamzorg/beamz.git
cd beamz
uv sync --extra dev --extra test
```

CPU JAX is sufficient for the contributor checks. For CUDA notebooks on RunPod,
use the environment described in [`docker/runpod/README.md`](docker/runpod/README.md).

## Make a change

Keep changes focused and include tests for new behavior or regressions. Update
the relevant package README and examples when behavior or interfaces change.

Manage dependencies with uv so `pyproject.toml` and `uv.lock` stay synchronized:

```bash
uv add "package>=1.0"                 # Runtime dependency
uv add --optional dev "tool>=1.0"    # Contributor-only dependency
uv remove package
```

Do not edit `uv.lock` by hand.

## Test while you work

Run the smallest relevant test while iterating:

```bash
uv run python -m pytest tests/test_physics_energy.py -v
uv run python -m pytest \
  tests/test_physics_energy.py::test_energy_conservation -v
make test-single FILE=test_physics_energy.py
```

Tests belong in `tests/`, use `test_*.py` names, and should reuse fixtures from
`tests/conftest.py` where appropriate. Existing markers include `slow`,
`design`, and `simulation`.

## Run the contributor checks

Before opening a pull request, run:

```bash
make format       # Apply Ruff formatting and safe lint fixes
make audit        # Lint, type-check, find dead code, and run tests
make build        # Build the wheel and source distribution
```

Use `make help` to see individual checks. CI repeats the quality checks, builds
the package, and runs the test suite on every supported Python version.

## Respect the architecture and public API

Do not add, remove, rename, or relocate a public export or top-level package without an
approved architecture change.

An approved public API change must update the existing API freeze, architecture
contract tests, compatibility paths, and the relevant package README in the same
pull request. Internal refactors must preserve existing import paths through a
compatibility facade.

## Open a pull request

Before submitting, confirm that:

- The change is focused and its behavior is explained.
- New behavior and bug fixes have tests.
- Formatting, audit, and build checks pass.
- Dependency changes include the updated `uv.lock`.
- Public API changes have prior approval and the required compatibility work.
- Material AI assistance is disclosed as described above.

The pull request template asks for the same evidence and must be completed by
the author.

## Repository map

```text
beamz/          Main package
  design/       Geometry and meshing
  simulation/   FDTD engine
  devices/      Sources, monitors, and boundaries
  analysis/     Results, modal analysis, and plotting
  optimization/ Topology and inverse-design tools
tests/          Test suite and architecture contracts
examples/       Scripts and notebooks
docs/           Documentation work area
```

For project usage and design context, start with the [project README](README.md)
and the README in the package area you plan to change.
