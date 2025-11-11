#!/bin/bash
# Run tests with subprocess coverage tracking enabled
# This captures coverage in multiprocessing code (e.g., driver.py worker processes)
#
# Usage:
#   ./tests_with_subprocess_coverage.sh                    # Run all tests
#   ./tests_with_subprocess_coverage.sh embodied/tests/test_driver.py  # Specific test

set -e

echo "🔍 Running tests with subprocess coverage tracking enabled..."
echo "   This may be slower than regular test runs."
echo ""

# Enable subprocess coverage
export COVERAGE_PROCESS_START=.coveragerc
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Clean up old coverage data
echo "🧹 Cleaning old coverage data..."
rm -f .coverage .coverage.*

# Run tests with coverage using 'coverage run' instead of 'pytest --cov'
# This is more reliable for subprocess coverage tracking
echo "🧪 Running tests..."
if [ $# -eq 0 ]; then
    # No arguments - run all tests
    coverage run -m pytest dreamerv3/tests/ embodied/tests/ -v
else
    # Arguments provided - run specific tests
    coverage run -m pytest "$@" -v
fi

# Combine coverage data from all subprocesses
echo ""
echo "📊 Combining coverage data from subprocesses..."
coverage combine

# Generate reports
echo ""
echo "📈 Generating coverage reports..."
coverage html -d .local/htmlcov_subprocess
coverage json -o .local/coverage_subprocess.json

# Show final report
echo ""
echo "📈 Final coverage report (including subprocesses):"
coverage report --precision=2

echo ""
echo "✅ Done! HTML report available at: .local/htmlcov_subprocess/index.html"
echo "   JSON report available at: .local/coverage_subprocess.json"
