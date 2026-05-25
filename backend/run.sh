#!/bin/bash
# Usage: cd backend && ./run.sh
# Set INNERSIGHT_DATA_DIR to point to your CERT r4.2 dataset folder

set -e

# Create venv if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt -q

echo "Running tests..."
pytest tests/ -v --tb=short
if [ $? -ne 0 ]; then
    echo "Tests failed. Fix them before starting the server."
    exit 1
fi

echo "Starting Flask server on port 5001..."
flask --app api run --host 0.0.0.0 --port 5001
