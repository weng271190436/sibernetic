# Sibernetic - Apple Silicon Modernization

This branch (`weiweng/m5-pro-from-0.9.8b`) modernizes Sibernetic to build and run on modern macOS with Apple Silicon (M1/M2/M3/M4/M5).

**Based on:** `ow-0.9.8b` (latest active development branch, March 2026)

## Changes from ow-0.9.8b

### Build System
- **Added CMakeLists.txt** - Modern CMake build system replacing fragile makefiles
- Automatic detection of Apple Silicon vs Intel
- Proper Homebrew path handling for arm64
- Python 3.11+ support

### Code Updates
- Fixed Python headers for conditional compilation (`USE_PYTHON` define)
- Removed hardcoded Python 2.7 paths from C++ headers
- Fixed remaining Python 2 print statements in legacy code

### macOS Compatibility
- Uses system OpenCL framework (deprecated but functional)
- Uses system OpenGL/GLUT frameworks
- Proper framework linking for Cocoa, IOKit, CoreFoundation

## Quick Start (macOS)

```bash
# Clone and checkout this branch
git clone https://github.com/weng271190436/sibernetic.git
cd sibernetic
git checkout weiweng/m5-pro-from-0.9.8b

# Run setup script
./scripts/setup_macos.sh

# Activate venv and run
source venv/bin/activate
cd build
PYTHONPATH=.. ./bin/Sibernetic -f worm
```

## Manual Build

```bash
# Install dependencies
brew install cmake python@3.11 glew freeglut

# Create venv
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel

# Run (from build directory)
PYTHONPATH=.. ./bin/Sibernetic -f worm
```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `SIBERNETIC_USE_GRAPHICS` | ON | Enable OpenGL visualization |
| `SIBERNETIC_USE_PYTHON` | ON | Enable Python muscle signal generation |
| `SIBERNETIC_BUILD_TESTS` | OFF | Build unit tests |

## Running with c302

The 0.9.8b branch has improved c302 integration:

```bash
source venv/bin/activate
pip install pyneuroml  # if not already installed
python sibernetic_c302.py -duration 1.0 -dt 0.005
```

## Known Issues

1. **OpenCL Deprecation**: Apple deprecated OpenCL in macOS 10.14, but it still works. A Metal port is planned for Phase 3.

2. **GLUT Deprecation**: GLUT is deprecated on macOS. The simulation works but you'll see deprecation warnings.

3. **First run may be slow**: OpenCL kernel compilation happens on first run.

## Roadmap

- [x] Phase 1: CMake build system, Python 3 compatibility
- [ ] Phase 2: Benchmark OpenCL performance on M5 Pro
- [ ] Phase 3: Port OpenCL kernels to Metal Compute Shaders  
- [ ] Phase 4: Replace OpenGL/GLUT with Metal/MetalKit

## Testing

The branch includes OMV-based tests:

```bash
# Run tests (requires omv)
pip install OMV
./run_all_tests.sh
```

## Contributing Back

This branch is intended to eventually become a PR to openworm/sibernetic. When ready:

```bash
# Add upstream
git remote add upstream https://github.com/openworm/sibernetic.git

# Create PR from this branch to upstream/ow-0.9.8b
```
