# Sibernetic - Apple Silicon Modernization

This branch (`weiweng/m5-pro-from-0.9.8b`) modernizes Sibernetic to build and run on modern macOS with Apple Silicon (M1/M2/M3/M4/M5).

**Based on:** `ow-0.9.8b` (latest active development branch, March 2026)

## Why Not the Old Instructions?

The original README has outdated macOS instructions involving `makefile.OSX`, Python 2.7, and manual environment variables. **Those no longer work** on modern macOS because:

- Python 2.7 is not available on modern macOS
- The old makefile assumes Intel x86 paths
- OpenCL only runs on CPU on Apple Silicon (no GPU acceleration)

This branch provides a modern build system with **Metal GPU acceleration**.

## Changes from ow-0.9.8b

### Build System
- **CMakeLists.txt** - Modern CMake build replacing fragile makefiles
- Automatic detection of Apple Silicon vs Intel
- Proper Homebrew path handling for arm64
- Python 3.11+ support (not Python 2.7!)

### Metal GPU Backend (NEW)
- **Metal compute shaders** for SPH physics simulation
- Runs on Apple Silicon GPU (6-10x faster than OpenCL CPU fallback)
- VBO-based renderer for efficient particle visualization
- ~30 FPS for full worm simulation (169k particles)

### Performance Comparison (M5 Pro)

| Config | OpenCL (CPU) | Metal (GPU) | Speedup |
|--------|--------------|-------------|--------:|
| Demo (17k particles) | 33ms/step | 6ms/step | **5.5x** |
| Worm (169k particles) | 180ms/step | 32ms/step | **5.6x** |

## Quick Start (macOS)

```bash
# Clone and checkout this branch
git clone https://github.com/weng271190436/sibernetic.git
cd sibernetic
git checkout weiweng/m5-pro-from-0.9.8b

# Run setup script (installs deps, builds with Metal)
./scripts/setup_macos.sh

# Run the worm simulation
cd ~/repos/sibernetic
PYTHONPATH=. ./build/bin/Sibernetic -f worm
```

## Manual Build

```bash
# Install dependencies (Homebrew Python 3.11, NOT Anaconda)
brew install cmake python@3.11

# Build with Metal
mkdir -p build && cd build
cmake -DSIBERNETIC_USE_METAL=ON ..
make -j4

# Run (from repo root)
cd ~/repos/sibernetic
PYTHONPATH=. ./build/bin/Sibernetic -f worm
```

### Important: Use Homebrew Python, not Anaconda

The build requires Homebrew's Python 3.11 (`/usr/local/Cellar/python@3.11/...`), not Anaconda. Anaconda uses a different libc++ that causes linker conflicts.

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `SIBERNETIC_USE_METAL` | OFF | Use Metal GPU backend (recommended for Apple Silicon) |
| `SIBERNETIC_USE_GRAPHICS` | ON | Enable OpenGL visualization |
| `SIBERNETIC_USE_PYTHON` | ON | Enable Python muscle signal generation |

## Running Configurations

```bash
# Demo (17k particles, fluid box with elastic body)
PYTHONPATH=. ./build/bin/Sibernetic

# Worm (169k particles, full C. elegans body model)
PYTHONPATH=. ./build/bin/Sibernetic -f worm

# No graphics (headless benchmarking)
PYTHONPATH=. ./build/bin/Sibernetic -f worm -no_g
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Sibernetic                        │
├─────────────────────────────────────────────────────┤
│  owPhysicsFluidSimulator (physics orchestration)    │
├─────────────────┬───────────────────────────────────┤
│   owISolver     │  Abstract interface               │
├─────────────────┼───────────────────────────────────┤
│ owOpenCLSolver  │ owMetalSolver                     │
│ (CPU fallback)  │ (GPU - Apple Silicon)             │
├─────────────────┴───────────────────────────────────┤
│  owFastRenderer (VBO-based OpenGL)                  │
└─────────────────────────────────────────────────────┘
```

## Known Issues

1. **CPU Sort Bottleneck**: The particle sort currently uses CPU `std::sort` (~28ms for 169k particles). GPU radix sort is planned.

2. **GLUT Deprecation**: GLUT is deprecated on macOS. Works but shows warnings.

3. **First run may be slow**: Metal shader compilation happens on first run.

## Roadmap

- [x] Phase 1: CMake build system, Python 3 compatibility
- [x] Phase 2: Benchmark OpenCL performance on M5 Pro  
- [x] Phase 3: Port OpenCL kernels to Metal Compute Shaders
- [x] Phase 3b: VBO-based fast renderer (21x rendering speedup)
- [ ] Phase 4: GPU radix sort (replace 28ms CPU sort)
- [ ] Phase 5: Full Metal renderer (replace OpenGL/GLUT)

## Running with c302

The 0.9.8b branch has improved c302 integration:

```bash
pip install pyneuroml
python sibernetic_c302.py -duration 1.0 -dt 0.005
```

## Contributing Back

This branch is intended to eventually become a PR to openworm/sibernetic.

```bash
# Add upstream
git remote add upstream https://github.com/openworm/sibernetic.git

# Create PR from this branch to upstream/ow-0.9.8b
```
