#!/bin/bash
# Run the mesh test suite

set -e  # Exit on error

echo "🧪 Running Mesh Framework Tests"
echo "=============================="

# Run pytest with coverage
python -m pytest tests/ -v --cov=mesh --cov-report=term-missing

# Run the basic functionality test separately if needed
# python tests/test_basic_functionality.py

echo ""
echo "✅ All tests completed!"