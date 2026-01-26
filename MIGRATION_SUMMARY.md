# Project Management Migration Summary

## Overview

This project has been migrated from traditional pip/setuptools to **uv**, a modern Python package manager that provides:
- ⚡ **10-100x faster** dependency resolution and installation
- 🔒 **Reproducible builds** via lockfile (uv.lock)
- 🎯 **Single tool** for environment, dependency, and project management
- 🔄 **Drop-in replacement** for pip/pip-tools/virtualenv/pyenv

## What Changed

### Files Added

1. **`.python-version`**
   - Pins Python version to 3.11 for development
   - Auto-detected by uv

2. **`uv.lock`** (630KB)
   - Lockfile for reproducible installations
   - **Must be committed to git**
   - Auto-updated by `uv add`/`uv remove`

3. **`Makefile`**
   - Convenient shortcuts for all common tasks
   - Type `make help` to see all commands

4. **`CONTRIBUTING.md`**
   - Contributor guide with uv-based workflow
   - Setup instructions, testing guide, code style

5. **`QUICKSTART.md`**
   - Quick reference for developers
   - Common commands, example usage

6. **`.uvignore`**
   - Excludes docs/tests from package builds

### Files Modified

1. **`pyproject.toml`**
   - Changed build backend: `setuptools` → `hatchling`
   - Updated `requires-python` to `>=3.10` (was `>=3.8`, incompatible with jaxlib)
   - Removed Python 3.8/3.9 from classifiers, added 3.12
   - Moved dev dependencies to `[dependency-groups]` (uv standard)
   - Updated black target versions to py310/py311/py312

2. **`CLAUDE.md`**
   - Added comprehensive uv workflow documentation
   - Dependency management commands
   - Troubleshooting section
   - Best practices for development

3. **`DEVELOPMENT.md`**
   - Complete rewrite with uv-centric workflow
   - Added Makefile usage
   - Removed outdated setuptools references
   - Added troubleshooting guide

4. **`.github/workflows/tests.yml`**
   - Updated to use uv for CI/CD
   - Uses `astral-sh/setup-uv@v5` action
   - Faster CI runs (uv caching)

5. **`.gitignore`**
   - Kept `.venv/` in gitignore
   - **Removed** `uv.lock` and `.python-version` from gitignore (should be committed)

### Files Unchanged (Legacy Support)

- **`setup.py`**: Kept for backward compatibility
- **`pytest.ini`**: Still works, but config also in pyproject.toml
- **`patch_wheel.py`**: No longer needed with hatchling

## Migration Guide for Developers

### Old Way → New Way

| Old Command | New Command |
|------------|-------------|
| `pip install -e ".[dev,test]"` | `uv sync --all-extras` |
| `pip install numpy` | `uv add numpy` |
| `pip uninstall numpy` | `uv remove numpy` |
| `python -m pytest tests/` | `uv run pytest tests/` or `make test` |
| `python -m black beamz/` | `uv run black beamz/` or `make format` |
| `python -m build` | `uv build` or `make build` |
| `python -m venv .venv` | `uv sync` (auto-creates .venv) |
| `pip freeze > requirements.txt` | Not needed (uv.lock handles this) |

### Getting Started (New Setup)

```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and setup
git clone <repo-url>
cd beamz
uv sync  # That's it! No pip, no venv, no requirements.txt

# 3. Run tests
make test

# 4. Start developing
make format  # Format code
make lint    # Check quality
```

### Common Tasks

```bash
# Development
make install         # Install package
make test           # Run all tests
make test-fast      # Run quick tests
make format         # Format code
make lint           # Lint code

# Adding dependencies
uv add requests              # Add to dependencies
uv add --dev pytest-mock    # Add to dev dependencies
uv add "numpy>=1.24,<2.0"   # With version constraint

# Documentation
make docs-serve     # Local preview
make docs-deploy    # Deploy to GitHub Pages

# Building
make build          # Build distribution
make publish        # Upload to PyPI
```

## Benefits

### For Developers

1. **Faster setup**: `uv sync` is ~10x faster than `pip install`
2. **Reproducible**: `uv.lock` ensures everyone has identical dependencies
3. **Less confusion**: One tool, clear commands
4. **Better errors**: Clear dependency conflict messages
5. **Convenience**: Makefile shortcuts for everything

### For CI/CD

1. **Faster builds**: uv's caching dramatically reduces CI time
2. **Deterministic**: Lockfile ensures same deps in CI and local
3. **Simpler**: Fewer steps in GitHub Actions

### For Project Maintainers

1. **Single source of truth**: `pyproject.toml` for all config
2. **Modern standards**: PEP 621, PEP 631 compliant
3. **Easier updates**: `uv sync --upgrade` updates everything safely
4. **Better dependency resolution**: Handles complex constraints better

## Breaking Changes

### Python Version Requirement

- **Old**: Python >=3.8
- **New**: Python >=3.10

**Reason**: jaxlib (required dependency) only supports Python 3.9+ for older versions and 3.10+ for newer versions. To avoid confusion and support recent jaxlib features, we standardized on 3.10+.

**Impact**: Users on Python 3.8/3.9 must upgrade to 3.10 or later.

### Build Backend

- **Old**: setuptools
- **New**: hatchling

**Reason**: Hatchling is more modern, faster, and requires no setup.py.

**Impact**:
- `python setup.py install` no longer works (use `uv sync` or `pip install -e .`)
- `patch_wheel.py` is no longer needed

## Rollback Plan

If issues arise, you can revert to the old setup:

```bash
# Use old workflow
python -m pip install -e ".[dev,test]"
python -m pytest tests/
```

The old `setup.py` is still present and functional.

## Testing Status

✅ All tests passing with uv
✅ Package imports correctly: `uv run python -c "import beamz; print(beamz.__version__)"`
✅ Fast tests complete successfully: `make test-fast`
✅ Makefile working: `make help` shows all commands
✅ GitHub Actions updated to use uv

## Next Steps

### Immediate

1. ✅ uv setup complete
2. ✅ Tests passing
3. ✅ Documentation updated
4. ⏳ **Review and test the changes**
5. ⏳ **Commit the changes**
6. ⏳ **Update README with uv installation**

### Future

1. Remove `setup.py` once fully confident (keep for now)
2. Remove `pytest.ini` (config moved to pyproject.toml)
3. Remove `patch_wheel.py` (no longer needed)
4. Consider removing Python 3.10 support when jaxlib requires 3.11+

## Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [PEP 621 - Storing project metadata in pyproject.toml](https://peps.python.org/pep-0621/)

## Support

Questions? Check:
- `make help` - See all available commands
- `DEVELOPMENT.md` - Development workflow guide
- `CONTRIBUTING.md` - Contributor guide
- `QUICKSTART.md` - Quick reference
- `CLAUDE.md` - AI assistant guide

---

**Migration completed**: 2026-01-26
**uv version**: 0.9.10
**Python version**: 3.11 (development), 3.10+ (minimum)
