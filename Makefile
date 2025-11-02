.PHONY: help install install-dev test test-security test-functional test-unit lint format security clean build docker run-dev run-prod docs

# Default target
help:
	@echo "Opti3D Development Commands"
	@echo "==========================="
	@echo "install        Install production dependencies"
	@echo "install-dev    Install development dependencies"
	@echo "test           Run all tests"
	@echo "test-unit      Run unit tests only"
	@echo "test-functional Run functional tests only"
	@echo "test-security  Run security tests only"
	@echo "lint           Run code linting (flake8, pylint, mypy)"
	@echo "format         Format code with black"
	@echo "security       Run security scans (bandit, safety)"
	@echo "clean          Clean build artifacts"
	@echo "build          Build Python package"
	@echo "docker         Build Docker image"
	@echo "run-dev        Run development server"
	@echo "run-prod       Run production server"
	@echo "docs           Build documentation"

# Installation
install:
	uv sync --frozen --no-dev

install-dev:
	uv sync --frozen
	pre-commit install

# Testing
test:
	uv run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	uv run pytest tests/ -v -m "unit" --cov=src

test-functional:
	uv run pytest tests/ -v -m "integration or not unit" --cov=src

test-security:
	uv run pytest tests/test_security_compliance.py tests/test_flask_app.py::TestSecurityFeatures -v

# Code quality
lint:
	@echo "Running flake8..."
	uv run flake8 src/ tests/
	@echo "Running pylint..."
	uv run pylint src/ tests/
	@echo "Running mypy..."
	uv run mypy src/

format:
	uv run black src/ tests/
	uv run isort src/ tests/

# Security
security:
	@echo "Running bandit security scan..."
	uv run bandit -r src/
	@echo "Running safety dependency check..."
	uv run safety check
	@echo "Running security tests..."
	$(MAKE) test-security

# Build and clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	uv build

# Docker
docker:
	docker build -t opti3d:latest .

docker-run:
	docker run -p 5000:5000 opti3d:latest

# Development servers
run-dev:
	uv run flask --app src/app run --debug

run-prod:
	uv run gunicorn --bind 0.0.0.0:5000 src.app:app

# Documentation
docs:
	cd docs && uv run make html

docs-serve:
	cd docs/_build/html && uv run python -m http.server 8000

# Development workflow
dev-setup: install-dev
	@echo "Development environment ready!"

ci-test: lint security test
	@echo "All CI checks passed!"

# Release
version-patch:
	uv run bump2version patch

version-minor:
	uv run bump2version minor

version-major:
	uv run bump2version major

release: clean build test
	@echo "Ready for release!"
