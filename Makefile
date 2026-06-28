.PHONY: help install install-dev test lint dead-code audit format clean build publish

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in development mode
	uv sync

install-dev:  ## Install package with dev dependencies
	uv sync --all-extras

test:  ## Run all tests with coverage
	uv run python -m pytest tests/ -v --tb=short --cov=beamz --cov-report=xml --cov-report=term-missing --cov-fail-under=75

test-single:  ## Run a single test file (usage: make test-single FILE=test_physics_energy.py)
	uv run python -m pytest tests/$(FILE) -v --tb=short

lint:  ## Run linting checks
	uv run --extra lint ruff check beamz/ examples/

dead-code:  ## Run high-confidence dead code checks
	uv run --extra lint vulture beamz/ --min-confidence 80

audit: lint dead-code test  ## Run core code quality audit

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
	uv build

publish:  ## Publish to PyPI (requires credentials)
	uv run twine upload dist/*

version:  ## Create new version release (usage: make version VERSION=0.1.X)
	python release_version.py $(VERSION)
