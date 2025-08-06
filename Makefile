.PHONY: help install install-dev test lint format type-check clean build docs

# Default target
help:
	@echo "Available commands:"
	@echo "  make install      Install package with UV"
	@echo "  make install-dev  Install package with dev dependencies"
	@echo "  make test         Run tests"
	@echo "  make lint         Run linting (ruff)"
	@echo "  make format       Format code (black)"
	@echo "  make type-check   Run type checking (mypy)"
	@echo "  make clean        Clean build artifacts"
	@echo "  make build        Build package"
	@echo "  make docs         Build documentation"

# Check if UV is installed
check-uv:
	@command -v uv >/dev/null 2>&1 || { echo "UV is not installed. Install from: https://github.com/astral-sh/uv"; exit 1; }

# Install package
install: check-uv
	uv pip install -e .

# Install package with all dev dependencies
install-dev: check-uv
	uv pip install -e ".[dev]"

# Install with specific providers
install-openai: check-uv
	uv pip install -e ".[openai]"

install-all-providers: check-uv
	uv pip install -e ".[all]"

# Run tests
test:
	pytest tests/ -v --cov=mesh --cov-report=term-missing

# Run quick local tests (no pytest required)
test-local:
	python test_setup.py

# Run linting
lint:
	ruff check mesh/ tests/

# Format code
format:
	black mesh/ tests/ examples/
	ruff check --fix mesh/ tests/

# Type checking
type-check:
	mypy mesh/

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build package
build: clean
	uv pip install build
	python -m build

# Build documentation
docs:
	mkdocs build

# Serve documentation locally
docs-serve:
	mkdocs serve

# Create virtual environment with UV
venv: check-uv
	uv venv
	@echo "Virtual environment created. Activate with: source .venv/bin/activate"

# Update dependencies
update-deps: check-uv
	uv pip compile pyproject.toml -o requirements.txt
	uv pip compile pyproject.toml --extra dev -o requirements-dev.txt

# Run pre-commit hooks
pre-commit:
	pre-commit run --all-files