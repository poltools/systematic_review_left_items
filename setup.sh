#!/usr/bin/env bash
# setup.sh — reproduce environment for this project.

set -euo pipefail

# --- config ---
PYTHON_VERSION="3.12"
VENV_NAME=".venv"

# --- sanity: ensure bash ---
if [ -z "${BASH_VERSION:-}" ]; then
  echo "ERROR: This script must be run with bash (not sh). Try: bash setup.sh"
  exit 1
fi

echo ">>> Creating virtual environment with Python ${PYTHON_VERSION}"

# Decide which python to use
choose_python() {
  if command -v pyenv >/dev/null 2>&1; then
    pyenv install -s "${PYTHON_VERSION}"
    if [ ! -f .python-version ]; then
      pyenv local "${PYTHON_VERSION}"
    else
      echo ">>> Existing .python-version detected; leaving as-is."
    fi
    echo "python"
    return
  fi

  if command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
    echo "python${PYTHON_VERSION}"
  elif command -v python3 >/dev/null 2>&1; then
    echo "python3"
    echo "!!! python${PYTHON_VERSION} not found; using $(python3 --version) instead."
  elif command -v python >/dev/null 2>&1; then
    echo "python"
    echo "!!! python${PYTHON_VERSION} not found; using $(python --version) instead."
  else
    echo ""
  fi
}

PYTHON="$(choose_python)"
if [ -z "$PYTHON" ]; then
  echo "ERROR: No suitable Python found. Install python${PYTHON_VERSION} or pyenv."
  exit 1
fi

# Check venv availability
if ! "$PYTHON" -c "import venv" 2>/dev/null; then
  echo "ERROR: The 'venv' module is not available for $($PYTHON -V)."
  echo "On Debian/Ubuntu: sudo apt-get install python3-venv or python${PYTHON_VERSION}-venv"
  exit 1
fi

echo ">>> Creating virtual environment at ${VENV_NAME}"
"$PYTHON" -m venv "${VENV_NAME}"

echo ">>> Activating virtual environment"
if [ -f "${VENV_NAME}/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "${VENV_NAME}/bin/activate"
elif [ -f "${VENV_NAME}/Scripts/activate" ]; then
  # shellcheck source=/dev/null
  source "${VENV_NAME}/Scripts/activate"
else
  echo "ERROR: Could not find activate script in ${VENV_NAME}/(bin|Scripts)."
  exit 1
fi

echo ">>> Upgrading pip tooling"
python -m pip install --upgrade pip wheel setuptools

if [ -f "requirements.txt" ]; then
  echo ">>> Installing requirements from requirements.txt"
  python -m pip install -r requirements.txt
else
  echo "!!! No requirements.txt found, skipping"
fi

# --- NEW: unzip llama_classified if present ---
if [ -f "data/llama_classified.zip" ]; then
  echo ">>> Unzipping data/llama_classified.zip into data/"
  unzip -o "data/llama_classified.zip" -d "data/"
else
  echo "!!! No data/llama_classified.zip found, skipping unzip"
fi

echo ">>> Setup complete!"
echo
if [ -f "${VENV_NAME}/bin/activate" ]; then
  echo "Activate your environment with:"
  echo "  source ${VENV_NAME}/bin/activate"
else
  echo "Activate your environment (Windows/Git Bash) with:"
  echo "  source ${VENV_NAME}/Scripts/activate"
fi
