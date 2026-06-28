# Development Guide

This guide covers the development workflow for BEAMZ using modern Python tooling with **uv**.

BEAMZ is very young and in a very crowded space - but it is **quickly growing**.

[![Star History Chart](https://api.star-history.com/svg?repos=QuentinWach/beamz,ymahlau/fdtdx,NanoComp/meep,facebookresearch/Khronos.jl,flaport/fdtd,zer011b/fdtd3d,thliebig/openEMS-Project,flexcompute/tidy3d&type=timeline&logscale&legend=bottom-right)](https://www.star-history.com/#QuentinWach/beamz&ymahlau/fdtdx&NanoComp/meep&facebookresearch/Khronos.jl&flaport/fdtd&zer011b/fdtd3d&thliebig/openEMS-Project&flexcompute/tidy3d&type=timeline&logscale&legend=bottom-right)

## AI-Assisted Contributions

AI-assisted coding workflows are allowed, but AI tools are only tools. The human
author owns the change and is responsible for reviewing, understanding, testing,
and defending everything they submit.

Contributors must be transparent when a meaningful part of a change was created
or rewritten with AI assistance. Mention the tool or model used in the pull
request description or commit notes, and describe which parts of the change it
affected. Maintainers may ask for extra explanation, review, or testing for
AI-assisted changes.

## Prerequisites

Install uv:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS with Homebrew
brew install uv
```

## Quick Setup

```bash
git clone https://github.com/QuentinWach/beamz.git
cd beamz
uv sync  # Installs all dependencies
```

## Development Workflow

### Environment Management

```bash
# Install package in dev mode
uv sync

# Install with all extras (dev, test, gpu)
uv sync --all-extras

# Install specific extra
uv sync --extra gpu

# Update dependencies
uv sync --upgrade
```

### Running Commands

All development commands use the Makefile:

```bash
make help          # Show all available commands
make test          # Run tests with coverage
make format        # Format code and fix package lint issues
make lint          # Check code quality
make dead-code     # Check for high-confidence dead code
make audit         # Run lint, dead-code checks, and tests
make build         # Build distribution
```

Or use uv directly:
```bash
uv run python -m pytest tests/
uv run --extra lint ruff check beamz/
uv run --extra lint ruff format beamz/
```

### Adding Dependencies

```bash
# Add runtime dependency
uv add <package-name>

# Add dev dependency
uv add --dev <package-name>

# Add with version constraint
uv add "numpy>=1.24,<2.0"

# Remove dependency
uv remove <package-name>
```

## Testing

### Run Tests

```bash
# All tests with coverage
make test

# Single test file
make test-single FILE=test_physics_energy.py

# Specific test function
uv run python -m pytest tests/test_physics_energy.py::test_energy_conservation -v

# Tests by marker
uv run python -m pytest -m design
uv run python -m pytest -m simulation
```

### Writing Tests

- Place tests in `tests/` directory
- Use `test_*.py` naming convention
- Use pytest fixtures from `tests/conftest.py`
- Add markers for categorization:
  ```python
  @pytest.mark.slow
  @pytest.mark.design
  @pytest.mark.simulation
  ```

## Code Quality

### Formatting

```bash
# Auto-format code
make format

# Check formatting (CI)
make format-check
```

### Linting

```bash
make lint
make dead-code
make audit
```

### Configuration

All tool configurations are in `pyproject.toml`:
- pytest
- ruff (formatting, import sorting, linting)
- vulture (high-confidence dead-code audit)

## Version Release

### Using the Release Script

```bash
# Update version and create git tag
make version VERSION=0.1.X

# Or manually:
python release_version.py 0.1.X
```

This will:
1. Update version in `pyproject.toml` and `beamz/__init__.py`
2. Create git tag `v0.1.X`
3. Push tag to remote repository

### Create GitHub Release

```bash
export GITHUB_TOKEN=your_token_here
python release_version.py 0.1.X --message "Release notes"
```

Options:
- `--no-push`: Don't push tag to remote
- `--force`: Force overwrite existing tag
- `--skip-version-update`: Skip updating version files

## Package Publishing

### Build and Publish

```bash
# Build distribution
make build
# or: uv build

# Publish to PyPI
make publish
# or: uv run twine upload dist/*
```

**Note**: The old `patch_wheel.py` step is no longer needed with uv/hatchling.

### Publishing Workflow

1. Update version: `make version VERSION=0.1.X`
2. Build: `make build`
3. Test in test environment
4. Publish: `make publish`

## Project Structure

```
beamz/
├── beamz/              # Main package
│   ├── design/         # Geometry and meshing
│   ├── simulation/     # FDTD engine
│   ├── optimization/   # Topology optimization
│   ├── devices/        # Sources and monitors
│   └── visual/         # Visualization
├── tests/              # Test suite
├── examples/           # Example scripts
├── pyproject.toml      # Project config (source of truth)
├── uv.lock            # Dependency lockfile
├── Makefile           # Development shortcuts
└── release_version.py # Version bump + tag helper
```

## Configuration Files

### pyproject.toml
- Single source of truth for project metadata
- Uses `hatchling` build backend
- Contains all tool configurations
- Minimum Python: 3.10

### uv.lock
- Reproducible dependency lockfile
- Committed to version control
- Auto-updated by `uv add/remove`

### .python-version
- Specifies Python 3.11 for development
- Auto-detected by uv

## Troubleshooting

### Dependency Issues

```bash
# Clear cache and reinstall
uv cache clean
uv sync

# Regenerate lockfile
uv lock --upgrade
```

### Python Version Issues

```bash
# List available Python versions
uv python list

# Install specific version
uv python install 3.11

# Use specific version
uv sync --python 3.11
```

### Import Errors

Ensure package is installed in editable mode:
```bash
uv sync
```

## CI/CD

GitHub Actions workflows use uv:
- `.github/workflows/tests.yml` - Run tests on push/PR
- Configured for Python 3.11
- Uses `astral-sh/setup-uv@v5` action
- Runs the full suite with coverage and enforces the current coverage baseline

## Best Practices

1. **Always use `uv add/remove`** for dependencies (keeps lockfile in sync)
2. **Run `make format`** before committing
3. **Run focused `uv run python -m pytest ...` commands** while iterating
4. **Run `make test`** before pushing
5. **Keep lockfile committed** (ensures reproducibility)
6. **Use `make` commands** for consistency
7. **Prefix one-off commands with `uv run`** (e.g., `uv run python script.py`)

## Additional Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [Project README](README.md)
- [Benchmark Artifact Policy](BENCHMARK_ARTIFACT_POLICY.md)
