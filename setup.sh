#!/usr/bin/env bash
set -euo pipefail

# Install system dependencies for building Sibernetic and the PyTorch solver
if [[ "$(uname)" == "Darwin" ]]; then
    if ! command -v brew >/dev/null 2>&1; then
        echo "Homebrew not found. Please install it from https://brew.sh/" >&2
        exit 1
    fi
    brew update
    brew install python glew freeglut clinfo opencl-headers || true
else
    apt-get update
    apt-get install -y python3-dev ocl-icd-opencl-dev libglu1-mesa-dev freeglut3-dev libglew-dev clinfo pocl-opencl-icd
fi

# Install Python packages
python3 -m pip install torch ruff pytest pyneuroml || echo "Warning: failed to install pyneuroml"

# Verify pyneuroml installed correctly if available
python3 - <<'EOF'
try:
    import pyneuroml
    print(pyneuroml.__version__)
except Exception:
    print('pyneuroml not installed, continuing')
EOF
