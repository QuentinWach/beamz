# Contributing to BEAMZ

Thank you for your interest in contributing to BEAMZ! This guide will help you get started.

## Getting Started

### Prerequisites

1. **Install uv** (Python package manager):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # macOS with Homebrew
   brew install uv

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/beamz.git
   cd beamz
   ```

3. **Set up the development environment**:
   ```bash
   # Install all dependencies (including dev and test)
   uv sync --all-extras
   ```

That's it! You're ready to develop.

## Development Workflow

### Make Your Changes

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** to the codebase

3. **Format your code**:
   ```bash
   make format
   ```

4. **Run linting**:
   ```bash
   make lint
   ```

5. **Run tests**:
   ```bash
   # Quick tests (skip slow ones)
   make test-fast

   # Full test suite with coverage
   make test
   ```

### Commit Your Changes

```bash
git add .
git commit -m "Brief description of your changes"
git push origin feature/your-feature-name
```

### Submit a Pull Request

1. Go to the [BEAMZ repository](https://github.com/QuentinWach/beamz)
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill in the PR template with:
   - Description of changes
   - Related issue numbers (if applicable)
   - Testing done

## Code Style

- **Python version**: 3.10+ (3.11 recommended)
- **Formatting**: Black (line length 88)
- **Import sorting**: isort (Black-compatible)
- **Linting**: flake8

All of these are automatically applied/checked with:
```bash
make format  # Apply formatting
make lint    # Check for issues
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Follow the naming convention: `test_*.py`
- Use pytest fixtures from `tests/conftest.py`
- Use markers for test categorization:
  ```python
  @pytest.mark.slow
  @pytest.mark.design
  @pytest.mark.simulation
  @pytest.mark.optimization
  ```

### Running Tests

```bash
# Run all tests
make test

# Run fast tests only (skip @pytest.mark.slow)
make test-fast

# Run specific test file
make test-single FILE=test_physics_energy.py

# Run tests for a specific module
uv run pytest -m design
uv run pytest -m simulation

# Run a specific test function
uv run pytest tests/test_physics_energy.py::test_energy_conservation -v
```

## Adding Dependencies

### Runtime Dependencies

```bash
# Add a new package
uv add <package-name>

# Add with version constraint
uv add "numpy>=1.24,<2.0"
```

This will update both `pyproject.toml` and `uv.lock`.

### Development Dependencies

```bash
# Add a dev tool
uv add --dev <package-name>
```

### Optional Dependencies

For optional features (like GPU support), manually edit `pyproject.toml`:
```toml
[project.optional-dependencies]
gpu = [
    "torch>=2.6.0",
]
```

Then run `uv lock` to update the lockfile.

## Documentation

### Building Documentation Locally

```bash
# Serve docs with live reload
make docs-serve

# Or directly:
uv run mkdocs serve
```

Visit http://127.0.0.1:8000 to view the documentation.

### Documentation Style

- Use Google-style docstrings
- Include examples in docstrings where helpful
- Keep explanations clear and concise

## Project Structure

```
beamz/
├── beamz/              # Main package
│   ├── design/         # Design and geometry
│   ├── simulation/     # FDTD simulation engine
│   ├── optimization/   # Topology optimization
│   ├── devices/        # Sources and monitors
│   └── visual/         # Visualization tools
├── tests/              # Test suite
├── examples/           # Example scripts
├── docs/               # Documentation
├── pyproject.toml      # Project configuration
├── uv.lock            # Dependency lockfile
└── Makefile           # Development shortcuts
```

## Common Issues

### Dependency Resolution Errors

```bash
# Clear cache and reinstall
uv cache clean
uv sync
```

### Python Version Mismatch

```bash
# Install the required Python version
uv python install 3.11

# Sync with specific Python version
uv sync --python 3.11
```

### Import Errors

Make sure you've installed the package in development mode:
```bash
uv sync
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/QuentinWach/beamz/issues)
- **Discussions**: [GitHub Discussions](https://github.com/QuentinWach/beamz/discussions)
- **Documentation**: [Project Docs](https://quentinwach.github.io/beamz/)

## Code of Conduct

There is no code of conduct. I rule with an iron fist.

## License

By contributing to BEAMZ, you agree that your contributions will be licensed under the same license as the project.
