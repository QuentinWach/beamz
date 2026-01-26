.PHONY: help install install-dev test test-fast lint format clean docs build publish

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
	uv run pytest tests/ -v --tb=short --cov=beamz --cov-report=xml --cov-report=term-missing

test-fast:  ## Run tests excluding slow ones
	uv run pytest tests/ -v -m "not slow" --tb=short

test-single:  ## Run a single test file (usage: make test-single FILE=test_physics_energy.py)
	uv run pytest tests/$(FILE) -v --tb=short

lint:  ## Run linting checks
	uv run flake8 beamz/ tests/

format:  ## Format code with black and isort
	uv run black beamz/ tests/
	uv run isort beamz/ tests/

format-check:  ## Check if code is formatted correctly
	uv run black --check beamz/ tests/
	uv run isort --check beamz/ tests/

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs-serve:  ## Serve documentation locally
	uv run mkdocs serve

docs-deploy:  ## Deploy documentation to GitHub Pages
	uv run mkdocs gh-deploy

build:  ## Build distribution packages
	uv build

publish:  ## Publish to PyPI (requires credentials)
	uv run twine upload dist/*

version:  ## Create new version release (usage: make version VERSION=0.1.X)
	python release_version.py $(VERSION)
