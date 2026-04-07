#!/bin/bash
# Sibernetic macOS Setup Script (for ow-0.9.8b branch)
# Tested on Apple Silicon (M1/M2/M3/M4/M5) with macOS

set -e

echo "=== Sibernetic macOS Setup (v0.9.8b) ==="
echo ""

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    HOMEBREW_PREFIX="/opt/homebrew"
    echo "Detected Apple Silicon ($ARCH)"
else
    HOMEBREW_PREFIX="/usr/local"
    echo "Detected Intel Mac ($ARCH)"
fi

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "ERROR: Homebrew not found. Install it from https://brew.sh"
    exit 1
fi

echo ""
echo "Installing dependencies via Homebrew..."

# Install dependencies
brew install cmake python@3.11 glew freeglut || true

# Python setup - Homebrew doesn't symlink versioned binaries by default
# Check multiple possible locations
PYTHON_BIN=""
PIP_BIN=""

# Try Homebrew's versioned path first
if [ -f "$HOMEBREW_PREFIX/opt/python@3.11/bin/python3.11" ]; then
    PYTHON_BIN="$HOMEBREW_PREFIX/opt/python@3.11/bin/python3.11"
    PIP_BIN="$HOMEBREW_PREFIX/opt/python@3.11/bin/pip3.11"
elif [ -f "$HOMEBREW_PREFIX/bin/python3.11" ]; then
    PYTHON_BIN="$HOMEBREW_PREFIX/bin/python3.11"
    PIP_BIN="$HOMEBREW_PREFIX/bin/pip3.11"
elif [ -f "$HOMEBREW_PREFIX/opt/python@3.11/bin/python3" ]; then
    PYTHON_BIN="$HOMEBREW_PREFIX/opt/python@3.11/bin/python3"
    PIP_BIN="$HOMEBREW_PREFIX/opt/python@3.11/bin/pip3"
elif command -v python3.11 &> /dev/null; then
    PYTHON_BIN=$(which python3.11)
    PIP_BIN=$(which pip3.11 2>/dev/null || echo "")
elif command -v python3 &> /dev/null; then
    # Fall back to system python3 (should be 3.9+ on modern macOS)
    PYTHON_BIN=$(which python3)
    PIP_BIN=$(which pip3 2>/dev/null || echo "")
    echo "Note: Using system python3 instead of python@3.11"
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: Python 3 not found. Try: brew install python@3.11"
    exit 1
fi

# If pip not found, derive it from python
if [ -z "$PIP_BIN" ] || [ ! -f "$PIP_BIN" ]; then
    PIP_BIN="$PYTHON_BIN -m pip"
fi

echo "Using Python: $PYTHON_BIN"
echo "Using pip: $PIP_BIN"

echo ""
echo "Installing numpy for the system Python (required by C++ binary)..."

# Install numpy to the Python that CMake will link against
# This is needed because the embedded Python interpreter doesn't see venv packages
$PIP_BIN install numpy

echo ""
echo "Building Sibernetic..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE=$PYTHON_BIN \
    -DSIBERNETIC_USE_GRAPHICS=ON \
    -DSIBERNETIC_USE_PYTHON=ON \
    -DSIBERNETIC_USE_METAL=ON

# Build
cmake --build . --parallel $(sysctl -n hw.ncpu)

cd ..

echo ""
echo "=== Build Complete ==="
echo ""
echo "To run Sibernetic (from the sibernetic root directory):"
echo "  cd $(pwd)"
echo "  PYTHONPATH=. ./build/bin/Sibernetic -f worm"
echo ""
echo "Or without graphics:"
echo "  PYTHONPATH=. ./build/bin/Sibernetic -no_g -f worm"
echo ""
echo "To run with c302 neural simulation:"
echo "  pip3.11 install pyneuroml matplotlib  # if not installed"
echo "  python3.11 sibernetic_c302.py -duration 1.0"
echo ""
