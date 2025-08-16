#!/usr/bin/env bash
# setup.sh
# Script to reproduce environment for this project.

set -e  # exit on error

PYTHON_VERSION=3.12
VENV_NAME=".venv"

echo ">>> Creating virtual environment with Python $PYTHON_VERSION"

# Ensure pyenv or system python can provide the right version
if command -v pyenv >/dev/null 2>&1; then
    pyenv install -s $PYTHON_VERSION
    pyenv local $PYTHON_VERSION
    python -m venv $VENV_NAME
else
    python$PYTHON_VERSION -m venv $VENV_NAME
fi

echo ">>> Activating virtual environment"
source $VENV_NAME/bin/activate

echo ">>> Upgrading pip"
pip install --upgrade pip wheel setuptools

if [ -f "requirements.txt" ]; then
    echo ">>> Installing requirements from requirements.txt"
    pip install -r requirements.txt
else
    echo "!!! No requirements.txt found, skipping"
fi

echo ">>> Setup complete!"
echo ""
echo "Activate your environment with:"
echo "  source $VENV_NAME/bin/activate"
