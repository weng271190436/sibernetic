#!/bin/bash
# Sibernetic Metal Build Script
# Builds with Metal compute backend instead of OpenCL

set -e

echo "=== Sibernetic Metal Build ==="
echo ""

# Check we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: Metal is only available on macOS"
    exit 1
fi

# Check for metal-cpp
if [ ! -d "metal-cpp" ]; then
    echo "Downloading metal-cpp headers..."
    curl -L https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip -o metal-cpp.zip
    unzip -q metal-cpp.zip
    rm metal-cpp.zip
    echo "metal-cpp downloaded"
fi

# Detect architecture
ARCH=$(uname -m)
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

# Ensure Metal toolchain is installed
echo "Checking Metal toolchain..."
if ! xcrun -sdk macosx metal --version &>/dev/null; then
    echo "Installing Metal toolchain (may require password)..."
    xcodebuild -downloadComponent MetalToolchain
fi

# Find Python
PYTHON_BIN=""
if [ -f "$HOMEBREW_PREFIX/opt/python@3.11/bin/python3.11" ]; then
    PYTHON_BIN="$HOMEBREW_PREFIX/opt/python@3.11/bin/python3.11"
    PIP_BIN="$HOMEBREW_PREFIX/opt/python@3.11/bin/pip3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN=$(which python3)
    PIP_BIN="$PYTHON_BIN -m pip"
fi

echo "Using Python: $PYTHON_BIN"

# Install numpy
echo "Installing numpy..."
$PIP_BIN install numpy

# Clean and build
echo ""
echo "Building with Metal backend..."
rm -rf build
mkdir build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE=$PYTHON_BIN \
    -DSIBERNETIC_USE_METAL=ON \
    -DSIBERNETIC_USE_GRAPHICS=ON \
    -DSIBERNETIC_USE_PYTHON=ON

cmake --build . --parallel $(sysctl -n hw.ncpu)

cd ..

echo ""
echo "=== Metal Build Complete ==="
echo ""
echo "To run Sibernetic with Metal:"
echo "  cd $(pwd)"
echo "  PYTHONPATH=. ./build/bin/Sibernetic -f worm"
echo ""
echo "To benchmark Metal vs OpenCL:"
echo "  ./scripts/benchmark.sh  # saves results"
echo ""
