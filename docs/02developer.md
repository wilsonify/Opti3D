# Opti3D Developer Guide

Welcome to the Opti3D development team. 

This guide is designed to help you get oriented quickly, 
understand the structure and purpose of each component, 
and begin contributing with confidence.

## Introduction

## Quick Start

Opti3D is a Flask-based web application for analyzing and optimizing 3D models in STL format. 

It integrates computational geometry algorithms, robust security features, and a modern Python toolchain to streamline 3D file optimization.

This guide walks through setup, architecture, workflows, testing, and performance considerations to support effective development and collaboration.

### Prerequisites

- Python 3.13
- uv - Modern Python package manager (faster than pip!)
- Git - Version control
- Docker - For containerized development and deployment

### Setup in 5 Minutes

1. Clone the repository
2. Install uv (if not already installed)
3. Install dependencies
4. Run the development server 

```bash

git clone https://github.com/wilsonify/Opti3D.git
cd Opti3D
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --dev
make run-dev
```

That's it! The application will be running at `http://localhost:5000` with hot reload enabled.

## Project Structure

```
Opti3D/
├── src/                       # Core application code
│   ├── app.py                 # Flask web application
│   ├── stldeli/               # STL processing library
│   │   ├── stl_optimizer.py   # Core optimization algorithms
│   │   └── deli.py            # External tool integration
│   └── security_middleware.py # Security layer
├── static/                    # Frontend assets
├── tests/                     # Test suite
├── docs/                      # Documentation
├── pyproject.toml             # Project configuration
├── uv.lock                    # Dependency lock file
├── Dockerfile                 # Container configuration
└── Makefile                   # Development commands
```

## Development Workflow

### Commands

```bash
make install-dev     # Install dependencies
make run-dev         # Run development server
make test            # Run all tests
make lint            # Code style checks
make format          # Apply formatting
make build           # Build for production
```

### Contributing

1. Make an Issue on GitHub to discuss your proposed changes

1. Fork the repository on GitHub

1. Create a feature branch

1. **Make your changes**
   - Write code following our style
   - Add tests for new functionality
   - Update documentation as needed

1. **Test your changes**
   ```bash
   make test
   make lint
   ```
1. Create a Pull Request
   - Provide a clear description of changes
   - Link any relevant issues
   - Request review from team members

## Architecture Overview

Opti3D consists of three main layers:

1. Flask Application (src/app.py)
   - file upload
   - API endpoints
   - sessions
   - logging

1. STL Optimizer (src/stldeli/stl_optimizer.py)
   - mesh analysis
   - vertex merging
   - surface smoothing
   - size reduction

1. Security Middleware (src/security_middleware.py)
   - CSRF protection
   - rate limiting
   - input validation
   - file verification

### API Endpoints

| Endpoint | Method | Purpose | Rate Limit |
|----------|--------|---------|------------|
| `/` | GET | Main application interface | None |
| `/api/upload` | POST | File upload and analysis | 5/min |
| `/api/optimize` | POST | Process optimization | 10/min |
| `/api/download/<filename>` | GET | Download optimized file | None |
| `/health` | GET | Health check endpoint | None |

## Testing

Testing ensures stability and reproducibility across optimization scenarios.

1. Run all tests with coverage
1. Run specific test files
1. Run with coverage report

```bash
make test
uv run pytest tests/test_stl_optimizer.py -v
uv run pytest tests/ --cov=src --cov-report=html
```

### Test Categories

- Unit: Individual function testing
- Integration: Component interaction testing
- End-to-end: full workflow testing

#### Example:

```python
import pytest
from stldeli.stl_optimizer import analyze_stl_mesh

def test_mesh_analysis():
    """Test mesh analysis with sample data"""
    # Arrange
    test_file = "tests/fixtures/sample.stl"
    
    # Act
    result = analyze_stl_mesh(test_file)
    
    # Assert
    assert result['triangles'] > 0
    assert 'dimensions' in result
    assert result['mesh_health'] in ['good', 'fair', 'poor']
```

## 🔧 Development Tools

### Code Quality

We use several tools to maintain code quality:

| Tool           | Purpose             |
| -------------- | ------------------- |
| **Black**      | Formatting          |
| **Flake8**     | Linting             |
| **Pylint**     | Static analysis     |
| **MyPy**       | Type checking       |
| **Pre-commit** | Automated Git hooks |

### Security Tools

| Tool        | Purpose                              |
| ----------- | ------------------------------------ |
| **Bandit**  | Source vulnerability detection       |
| **Safety**  | Dependency scanning                  |
| **Semgrep** | Static analysis and rule enforcement |

## Performance and Optimization

### Memory Management

1. Process STL files in manageable chunks
1. Prefer NumPy arrays for vector operations
1. Release temporary memory explicitly
1. Track memory usage during optimization

### Algorithmic Efficiency

1. Use spatial indexing for nearest-vertex queries
1. Apply parallelization where beneficial
1. Cache expensive calculations
1. Abort early when convergence criteria are met

### Performance Benchmarks

| File Size | Analysis | Light Opt | Medium Opt | Aggressive Opt |
| --------- | -------- | --------- | ---------- | -------------- |
| <10 MB    | 0.1 s    | 0.3 s     | 0.5 s      | 0.8 s          |
| 10–50 MB  | 0.8 s    | 1.5 s     | 2.8 s      | 4.2 s          |
| 50–100 MB | 2.1 s    | 4.2 s     | 7.1 s      | 11.3 s         |


## Security Guidelines

### Input Validation

Always validate user input:

```python
def validate_file_upload(file):
    """Validate uploaded files"""
    if not file or not file.filename:
        raise ValueError("No file provided")
    
    # Check file extension
    if not file.filename.lower().endswith('.stl'):
        raise ValueError("Invalid file type")
    
    # Validate content
    if not is_valid_stl(file.stream):
        raise ValueError("Invalid STL format")
    
    # Check size limits
    if file.content_length > MAX_FILE_SIZE:
        raise ValueError("File too large")
```

### Security Best Practices

- Use CSRF tokens for all state-changing operations
- Sanitize all user inputs
- Implement rate limiting
- Use secure headers
- Never trust client-side data
- Log security events

## Containerized Development

### Building Images

```bash
docker build -t opti3d:latest .
docker run -p 5000:5000 opti3d:latest
```


## Code Style Guide

- Follow PEP 8 and include type hints
- Write clear, concise docstrings
- Keep functions small and testable

### Commit Messages

Use conventional commit format:

```
feat: add new optimization algorithm
fix: resolve memory leak in file processing
docs: update API documentation
test: add tests for mesh validation
refactor: improve code readability
```

### Pull Request Process

1. **Description**: Clearly explain what you changed and why
2. **Testing**: Include tests for new functionality
3. **Documentation**: Update relevant documentation
4. **Review**: Respond to review feedback promptly

## Common Tasks

### Adding New Optimization Levels

1. Update `stl_optimizer.py` with new parameters
2. Add validation logic
3. Update frontend options in `static/js/`
4. Add comprehensive tests
5. Update API documentation

### Adding New File Formats

1. Create parser in `stldeli/` module
2. Add validation logic
3. Update API endpoints
4. Add file type detection
5. Write tests and documentation

### Debugging Common Issues

**Memory Issues:**
- Check for large file handling
- Verify cleanup of temporary files
- Profile memory usage with `memory_profiler`

**Performance Issues:**
- Use profiling decorators
- Check algorithm complexity
- Optimize database queries if applicable

**Security Issues:**
- Run security tests: `make test-security`
- Check input validation
- Verify rate limiting is working

## 📚 Learning Resources

### Recommended Reading

- [Python Best Practices](https://docs.python-guide.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [NumPy for Computational Geometry](https://numpy.org/doc/stable/)
- [STL File Format Specification](http://www.fabbers.com/tech/STL_Format)

### Internal Resources

- [User Guide](01user.md) - End-user documentation
- [Administrator Guide](03admin.md) - Deployment and operations


## Getting Help

### Troubleshooting

**Common Issues:**

1. **"uv command not found"**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **"Tests failing with import errors"**
   ```bash
   uv sync --dev
   ```

3. **"Docker build fails"**
   ```bash
   docker system prune -f
   docker build --no-cache -t opti3d:latest .
   ```

### Asking for Help

- Check existing GitHub issues first
- Create detailed bug reports with:
  - Steps to reproduce
  - Expected vs actual behavior
  - Environment details
  - Error messages and logs

### Team Communication

- Use GitHub issues for bug reports and feature requests
- Use pull requests for code review

---

## Welcome to the Team!

We're excited to have you contribute to Opti3D! 
Whether you're interested in computational geometry, 
web development, or user experience design, 
there's fascinating work to be done.

Start with the Quick Start section above, 
explore the codebase, and don't hesitate to ask questions. 
Every contribution helps make 3D file optimization more accessible to everyone.

Happy coding!

---

*For user-facing documentation, see the [User Guide](01user.md).  
For deployment and administration, see the [Administrator Guide](03admin.md).*
