#!/bin/bash
# Sibernetic macOS Setup Script
# Tested on Apple Silicon (M1/M2/M3/M4/M5) with macOS

set -e

echo "=== Sibernetic macOS Setup ==="
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
brew install cmake python@3.11 numpy glew freeglut || true

# Python setup
PYTHON_VERSION="3.11"
PYTHON_BIN="$HOMEBREW_PREFIX/bin/python$PYTHON_VERSION"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "ERROR: Python $PYTHON_VERSION not found at $PYTHON_BIN"
    exit 1
fi

echo ""
echo "Setting up Python virtual environment..."

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    $PYTHON_BIN -m venv venv
fi

source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install numpy matplotlib pyneuroml

echo ""
echo "Building Sibernetic..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON3_EXECUTABLE=$(which python) \
    -DSIBERNETIC_USE_GRAPHICS=ON \
    -DSIBERNETIC_USE_PYTHON=ON

# Build
cmake --build . --parallel $(sysctl -n hw.ncpu)

echo ""
echo "=== Build Complete ==="
echo ""
echo "To run Sibernetic:"
echo "  cd build/bin"
echo "  ./Sibernetic -f worm"
echo ""
echo "Or without graphics:"
echo "  ./Sibernetic -no_g -f worm"
echo ""
