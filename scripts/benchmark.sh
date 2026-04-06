#!/bin/bash
# Sibernetic Phase 2: Benchmark Script for Apple Silicon
# Run this from the sibernetic root directory

set -e

echo "=== Sibernetic Benchmark (Apple Silicon) ==="
echo ""
echo "System Info:"
echo "  $(uname -m) / $(sw_vers -productName) $(sw_vers -productVersion)"
sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "  Apple Silicon"
echo ""

# Check we're in the right place
if [ ! -f "build/bin/Sibernetic" ]; then
    echo "ERROR: Run this from the sibernetic root directory after building"
    echo "       (./scripts/setup_macos.sh)"
    exit 1
fi

RESULTS_FILE="benchmark_results_$(date +%Y%m%d_%H%M%S).txt"

echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Header for results
{
    echo "=== Sibernetic Benchmark Results ==="
    echo "Date: $(date)"
    echo "System: $(uname -m) / $(sw_vers -productName) $(sw_vers -productVersion)"
    sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "CPU: Apple Silicon"
    echo ""
} > "$RESULTS_FILE"

# Benchmark 1: Short simulation (0.1s simulated time)
echo "Benchmark 1: Short simulation (0.1s simulated time, ~20k timesteps)"
echo "  Running..."

START_TIME=$(python3 -c "import time; print(time.time())")
PYTHONPATH=. ./build/bin/Sibernetic -no_g -f worm timelimit=0.1 2>&1 | tee -a "$RESULTS_FILE.raw"
END_TIME=$(python3 -c "import time; print(time.time())")

ELAPSED=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")
echo "  Completed in ${ELAPSED}s wall clock time"
echo ""

{
    echo "--- Benchmark 1: Short (0.1s sim time) ---"
    echo "Wall clock time: ${ELAPSED}s"
    echo ""
} >> "$RESULTS_FILE"

# Benchmark 2: Medium simulation (0.5s simulated time)
echo "Benchmark 2: Medium simulation (0.5s simulated time)"
echo "  Running... (this may take a few minutes)"

START_TIME=$(python3 -c "import time; print(time.time())")
PYTHONPATH=. ./build/bin/Sibernetic -no_g -f worm timelimit=0.5 2>&1 | tee -a "$RESULTS_FILE.raw"
END_TIME=$(python3 -c "import time; print(time.time())")

ELAPSED=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")
echo "  Completed in ${ELAPSED}s wall clock time"
echo ""

{
    echo "--- Benchmark 2: Medium (0.5s sim time) ---"
    echo "Wall clock time: ${ELAPSED}s"
    echo ""
} >> "$RESULTS_FILE"

# Benchmark 3: With graphics overhead (short)
echo "Benchmark 3: With graphics (0.05s simulated time)"
echo "  A window will open - it will close automatically"
echo "  Running..."

START_TIME=$(python3 -c "import time; print(time.time())")
PYTHONPATH=. ./build/bin/Sibernetic -f worm timelimit=0.05 2>&1 | tee -a "$RESULTS_FILE.raw"
END_TIME=$(python3 -c "import time; print(time.time())")

ELAPSED=$(python3 -c "print(f'{$END_TIME - $START_TIME:.2f}')")
echo "  Completed in ${ELAPSED}s wall clock time"
echo ""

{
    echo "--- Benchmark 3: With Graphics (0.05s sim time) ---"
    echo "Wall clock time: ${ELAPSED}s"
    echo ""
} >> "$RESULTS_FILE"

# Summary
echo "=== Summary ==="
cat "$RESULTS_FILE"

echo ""
echo "Raw output saved to: $RESULTS_FILE.raw"
echo "Summary saved to: $RESULTS_FILE"
echo ""
echo "To compare with other systems, the key metric is:"
echo "  Wall clock seconds per simulated second"
echo ""
echo "For reference, the OpenWorm team reports ~10-30 minutes"
echo "for 1 second of simulated time on typical hardware."
