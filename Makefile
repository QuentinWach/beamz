.PHONY: help install install-dev test test-all-cpu docs lint typecheck dead-code audit format clean build publish

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in development mode
	uv sync

install-dev:  ## Install package with dev dependencies
	uv sync --all-extras

test:  ## Run the compact pull-request evidence gate with branch coverage
	uv run python -m pytest tests/ -m "not slow and not pdk and not hardware" -v --tb=short --cov=beamz --cov-branch --cov-report=xml --cov-report=json:coverage.json --cov-report=term-missing --cov-fail-under=80 --validation-report=validation-results.json
	uv run python scripts/check_coverage_policy.py coverage.json tests/coverage_policy.json

test-all-cpu:  ## Run every CPU test, including slow characterization
	uv run python -m pytest tests/ -v --tb=short --validation-report=validation-results.json

docs:  ## Build strict docs and execute documentation contracts
	uv run mkdocs build --strict
	BEAMZ_DOCS_TEST=1 uv run python -m pytest tests/docs/ -v --tb=short

test-single:  ## Run a single test file (usage: make test-single FILE=test_physics_energy.py)
	uv run python -m pytest tests/$(FILE) -v --tb=short

lint:  ## Run linting checks
	uv run --extra lint ruff check beamz/ tests/ examples/ scripts/ release_version.py vulture_allowlist.py
	uv run --extra lint ruff format --check beamz/ tests/ examples/ scripts/ release_version.py vulture_allowlist.py

typecheck:  ## Run package-wide static type checking
	uv run --extra lint pyright

dead-code:  ## Run dead code checks with intentional dynamic API allowlist
	uv run --extra lint vulture beamz/ vulture_allowlist.py --min-confidence 60

audit: lint typecheck dead-code test  ## Run core code quality audit

format:  ## Format code and fix package lint issues
	uv run --extra lint ruff format beamz/ examples/ tests/
	uv run --extra lint ruff check --fix beamz/ examples/

format-check:  ## Check if code is formatted correctly
	uv run --extra lint ruff format --check beamz/ examples/ tests/
	uv run --extra lint ruff check beamz/ examples/

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build distribution packages
	uv build --clear
	uv run --no-project python scripts/check_distributions.py

publish:  ## Publish to PyPI (requires credentials)
	uv run twine upload dist/*

version:  ## Create new version release (usage: make version VERSION=0.1.X)
	python release_version.py $(VERSION)
