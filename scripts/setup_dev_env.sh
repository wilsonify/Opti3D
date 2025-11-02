#!/bin/bash
# Development environment setup script for Opti3D

set -e

echo "🔧 Setting up Opti3D development environment..."

# Check if Python 3.9+ is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "📋 Found Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode
echo "📥 Installing Opti3D in development mode..."
pip install -e .

# Install development dependencies
echo "🛠️ Installing development dependencies..."
pip install -e ".[dev,security,test]"

# Install pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️ Please edit .env file with your configuration"
fi

echo "✅ Development environment setup complete!"
echo ""
echo "🚀 To get started:"
echo "   source venv/bin/activate"
echo "   python -m pytest tests/"
echo "   python -m stldeli --help"
echo ""
echo "🔍 Useful commands:"
echo "   make test          # Run all tests"
echo "   make lint          # Run linting"
echo "   make security      # Run security checks"
echo "   make format        # Format code"
echo "   make clean         # Clean build artifacts"
