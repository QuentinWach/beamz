# Development Guide

This guide covers the development workflow for BEAMZ using modern Python tooling with **uv**.

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
make test-fast     # Run quick tests
make format        # Format code (black + isort)
make lint          # Check code quality
make docs-serve    # Serve docs locally
make build         # Build distribution
```

Or use uv directly:
```bash
uv run pytest tests/
uv run black beamz/
uv run mkdocs serve
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

# Fast tests (skip @pytest.mark.slow)
make test-fast

# Single test file
make test-single FILE=test_physics_energy.py

# Specific test function
uv run pytest tests/test_physics_energy.py::test_energy_conservation -v

# Tests by marker
uv run pytest -m design
uv run pytest -m simulation
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
```

### Configuration

All tool configurations are in `pyproject.toml`:
- pytest
- black (line-length: 88, target: py310+)
- isort (black-compatible)

## Documentation

### Local Development

```bash
make docs-serve
# Opens at http://127.0.0.1:8000
```

### Deployment

```bash
make docs-deploy
```

This deploys to the `gh-pages` branch.

## Version Release

### Using the Release Script

```bash
# Update version and create git tag
make version VERSION=0.1.X

# Or manually:
python release_version.py 0.1.X
```

This will:
1. Update version in `setup.py`, `pyproject.toml`, and `beamz/__init__.py`
2. Create git tag `v0.1.X`
3. Push tag to remote repository

### Create GitHub Release

```bash
export GITHUB_TOKEN=your_token_here
python release_version.py 0.1.X --message "Release notes"
```

Options:
- `--tag-only`: Only create git tag
- `--no-push`: Don't push tag to remote
- `--draft`: Create draft GitHub release
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
├── docs/               # Documentation source
├── pyproject.toml      # Project config (source of truth)
├── uv.lock            # Dependency lockfile
├── Makefile           # Development shortcuts
└── CLAUDE.md          # AI assistant guide
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
- Configured for Python 3.10 and 3.11
- Uses `astral-sh/setup-uv@v5` action

## Best Practices

1. **Always use `uv add/remove`** for dependencies (keeps lockfile in sync)
2. **Run `make format`** before committing
3. **Run `make test-fast`** during development
4. **Run `make test`** before pushing
5. **Keep lockfile committed** (ensures reproducibility)
6. **Use `make` commands** for consistency
7. **Prefix one-off commands with `uv run`** (e.g., `uv run python script.py`)

## Additional Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [Contributing Guide](CONTRIBUTING.md)
- [Quick Start](QUICKSTART.md)
- [AI Assistant Guide](CLAUDE.md)