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
	pip install -e .

install-dev:
	pip install -e ".[dev,security,test,docs]"
	pre-commit install

# Testing
test:
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	python -m pytest tests/ -v -m "unit" --cov=src

test-functional:
	python -m pytest tests/ -v -m "integration or not unit" --cov=src

test-security:
	python -m pytest tests/test_security_compliance.py tests/test_flask_app.py::TestSecurityFeatures -v

# Code quality
lint:
	@echo "Running flake8..."
	flake8 src/ tests/
	@echo "Running pylint..."
	pylint src/ tests/
	@echo "Running mypy..."
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# Security
security:
	@echo "Running bandit security scan..."
	bandit -r src/
	@echo "Running safety dependency check..."
	safety check
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
	python -m build

# Docker
docker:
	docker build -t opti3d:latest .

docker-run:
	docker run -p 5000:5000 opti3d:latest

# Development servers
run-dev:
	python -m flask --app src/app run --debug

run-prod:
	gunicorn --bind 0.0.0.0:5000 src.app:app

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Development workflow
dev-setup: install-dev
	@echo "Development environment ready!"

ci-test: lint security test
	@echo "All CI checks passed!"

# Release
version-patch:
	bump2version patch

version-minor:
	bump2version minor

version-major:
	bump2version major

release: clean build test
	@echo "Ready for release!"
