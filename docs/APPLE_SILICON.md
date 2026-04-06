# Sibernetic - Apple Silicon Modernization

This branch (`weiweng/m5-pro-modernization`) modernizes Sibernetic to build and run on modern macOS with Apple Silicon (M1/M2/M3/M4/M5).

## Changes from upstream

### Build System
- **Added CMakeLists.txt** - Modern CMake build system replacing fragile makefiles
- Automatic detection of Apple Silicon vs Intel
- Proper Homebrew path handling for arm64
- Python 3 support (tested with 3.11+)

### Code Updates
- Fixed Python headers for Python 3 compatibility
- Removed hardcoded Python 2.7 paths
- Updated `map()` calls in Python scripts (returns iterator in Python 3)
- Fixed integer division (`//` vs `/`)

### macOS Compatibility
- Uses system OpenCL framework (deprecated but functional)
- Uses system OpenGL/GLUT frameworks
- Proper framework linking for Cocoa, IOKit, CoreFoundation

## Quick Start (macOS)

```bash
# Clone and checkout this branch
git clone https://github.com/weng271190436/sibernetic.git
cd sibernetic
git checkout weiweng/m5-pro-modernization

# Run setup script
./scripts/setup_macos.sh

# Run simulation
cd build/bin
./Sibernetic -f worm
```

## Manual Build

```bash
# Install dependencies
brew install cmake python@3.11 numpy glew freeglut

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel

# Run
./bin/Sibernetic -f worm
```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `SIBERNETIC_USE_GRAPHICS` | ON | Enable OpenGL visualization |
| `SIBERNETIC_USE_PYTHON` | ON | Enable Python muscle signal generation |
| `SIBERNETIC_BUILD_TESTS` | OFF | Build unit tests |

## Known Issues

1. **OpenCL Deprecation**: Apple deprecated OpenCL in macOS 10.14, but it still works. A Metal port is planned for Phase 3.

2. **GLUT Deprecation**: GLUT is deprecated on macOS. The simulation works but you'll see deprecation warnings.

3. **First run may be slow**: OpenCL kernel compilation happens on first run.

## Phase 2+ Roadmap

- [ ] Benchmark OpenCL performance on M5 Pro
- [ ] Port OpenCL kernels to Metal Compute Shaders  
- [ ] Replace OpenGL/GLUT with Metal/MetalKit
- [ ] Add GitHub Actions CI for macOS

## Original README

See [README_original.md](README_original.md) for the original documentation.
