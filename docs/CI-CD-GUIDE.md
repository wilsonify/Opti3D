# Opti3D CI/CD Pipeline Guide

This document describes the modern CI/CD pipeline that replaces sys.path manipulation with proper build and deployment automation.

## 🚀 Overview

The Opti3D project uses GitHub Actions for continuous integration and deployment, providing:
- Automated security testing
- Code quality checks
- Multi-environment deployments
- Container-based builds
- Comprehensive reporting

## 📁 Pipeline Structure

### GitHub Actions Workflow
Location: `.github/workflows/ci-cd.yml`

The pipeline includes these jobs:

1. **Security Tests** - Runs vulnerability scans, static analysis, and security compliance tests
2. **Functional Tests** - Runs unit, integration, and application tests
3. **Build** - Creates Python packages and validates them
4. **Docker Build** - Creates container images for deployment
5. **Deploy Staging** - Deploys to staging environment (develop branch)
6. **Deploy Production** - Deploys to production (releases only)
7. **Documentation** - Builds and deploys documentation
8. **Notifications** - Sends status notifications

### Package Configuration
Location: `pyproject.toml`

Modern Python packaging with:
- Proper dependency management
- Development dependencies
- Tool configurations (pytest, black, flake8, etc.)
- Entry points for CLI tools

## 🛠️ Local Development

### Quick Setup
```bash
# Clone and setup
git clone https://github.com/wilsonify/Opti3D.git
cd Opti3D
./scripts/setup_dev_env.sh

# Or manually
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,security,test]"
pre-commit install
```

### Development Commands
```bash
# Install dependencies
make install-dev

# Run all checks (CI equivalent)
make ci-test

# Run specific test types
make test              # All tests
make test-security     # Security tests only
make test-functional   # Functional tests only
make test-unit         # Unit tests only

# Code quality
make lint              # Run all linters
make format            # Format code
make security          # Run security scans

# Build and deployment
make build             # Build Python package
make docker            # Build Docker image
make clean             # Clean artifacts
```

## 🔒 Security Testing

### Automated Security Checks
The pipeline runs these security tools:

1. **Safety** - Scans dependencies for known vulnerabilities
2. **Bandit** - Static analysis for security issues in Python code
3. **Custom Security Tests** - Application-specific security tests
4. **Flake8/Pylint** - Code quality with security-focused rules

### Security Test Markers
Tests are marked with pytest markers:
```python
@pytest.mark.security
def test_sql_injection_protection():
    # Security test implementation
    pass
```

### Running Security Tests Locally
```bash
# Run all security checks
make security

# Or individual tools
bandit -r src/
safety check
python -m pytest tests/ -m security
```

## 📦 Build and Deployment

### Package Building
```bash
# Build Python package
make build

# Output in dist/
# - opti3d-1.0.0-py3-none-any.whl
# - opti3d-1.0.0.tar.gz
```

### Docker Deployment
```bash
# Build Docker image
make docker

# Run locally
make docker-run

# Or manually
docker build -t opti3d:latest .
docker run -p 5000:5000 opti3d:latest
```

### Environment-specific Deployment

#### Staging (develop branch)
- Triggers on push to `develop` branch
- Runs after successful tests
- Deploys to staging environment
- Runs smoke tests

#### Production (releases)
- Triggers on GitHub releases
- Runs full test suite
- Creates Docker images
- Deploys to production
- Runs health checks

## 🧪 Testing Strategy

### Test Categories
1. **Unit Tests** - Fast, isolated component tests
2. **Integration Tests** - Component interaction tests
3. **Security Tests** - Security-focused tests
4. **End-to-End Tests** - Full application tests

### Test Configuration
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests", 
    "security: marks tests as security-related tests",
    "unit: marks tests as unit tests",
]
```

### Coverage Reporting
- HTML reports in `htmlcov/`
- XML reports for CI integration
- Coverage uploaded to Codecov

## 🔧 Configuration

### Required Secrets
Configure these in GitHub repository settings:

- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password/token
- `GITHUB_TOKEN` - Automatically provided by GitHub Actions

### Environment-specific Settings
- **Staging**: Configured via GitHub environments
- **Production**: Requires approval and additional checks

## 📊 Monitoring and Reporting

### Artifacts
Each pipeline run generates:
- Security scan reports (JSON)
- Coverage reports (HTML/XML)
- Build packages (wheel/tar.gz)
- Docker images (multi-platform)

### Notifications
Configure notifications in the `notify` job:
- Slack integration
- Email notifications
- Teams webhooks

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure package is installed in development mode
   pip install -e .
   ```

2. **Permission Issues**
   ```bash
   # Make setup script executable
   chmod +x scripts/setup_dev_env.sh
   ```

3. **Docker Build Failures**
   ```bash
   # Check Dockerfile syntax
   docker build --no-cache -t opti3d:test .
   ```

### Debug Mode
Run tests with verbose output:
```bash
python -m pytest tests/ -v -s --tb=long
```

## 🔄 Migration from Old System

### Before (sys.path manipulation)
```python
import sys
sys.path.append('src')
from stldeli import optimizer
```

### After (proper package installation)
```python
from stldeli import optimizer
# Package installed via: pip install -e .
```

### Benefits of New System
- ✅ No sys.path manipulation needed
- ✅ Proper dependency management
- ✅ Automated testing and deployment
- ✅ Security scanning
- ✅ Multi-environment support
- ✅ Container-based deployment
- ✅ Comprehensive reporting

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python Packaging Guide](https://packaging.python.org/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Pytest Documentation](https://docs.pytest.org/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and run `make ci-test`
4. Submit a pull request
5. CI pipeline will run automatically

For questions or issues, please open an issue on GitHub.
