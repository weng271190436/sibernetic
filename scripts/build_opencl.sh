#!/bin/bash
# Sibernetic OpenCL Build Script
# Builds with OpenCL compute backend

set -e

echo "=== Sibernetic OpenCL Build ==="
echo ""

# Detect architecture
ARCH=$(uname -m)
if [[ "$(uname)" == "Darwin" ]]; then
    if [ "$ARCH" = "arm64" ]; then
        HOMEBREW_PREFIX="/opt/homebrew"
        echo "Detected Apple Silicon ($ARCH)"
    else
        HOMEBREW_PREFIX="/usr/local"
        echo "Detected Intel Mac ($ARCH)"
    fi

    # Install dependencies
    echo ""
    echo "Installing dependencies..."
    brew install cmake python@3.11 glew freeglut || true
else
    echo "Detected Linux ($ARCH)"
fi

# Find Python
PYTHON_BIN=""
if [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python3" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python3"
elif [[ "$(uname)" == "Darwin" ]] && [ -f "$HOMEBREW_PREFIX/opt/python@3.11/bin/python3.11" ]; then
    PYTHON_BIN="$HOMEBREW_PREFIX/opt/python@3.11/bin/python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN=$(which python3)
fi

echo "Using Python: $PYTHON_BIN"

# Install numpy
echo "Installing numpy..."
$PYTHON_BIN -m pip install numpy

# Clean and build
echo ""
echo "Building with OpenCL backend..."
rm -rf build_opencl
mkdir build_opencl
cd build_opencl

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE=$PYTHON_BIN \
    -DSIBERNETIC_USE_METAL=OFF \
    -DSIBERNETIC_USE_GRAPHICS=ON \
    -DSIBERNETIC_USE_PYTHON=ON

cmake --build . --parallel $(nproc 2>/dev/null || sysctl -n hw.ncpu)

cd ..

echo ""
echo "=== OpenCL Build Complete ==="
echo ""
echo "To run Sibernetic with OpenCL:"
echo "  cd $(pwd)"
echo "  PYTHONPATH=. ./build_opencl/bin/Sibernetic -f worm"
echo ""
