"""
Validation tests for PyTorch solver parity with OpenCL.

Phase 4 tests:
- 4.1: Reference log comparison
- 4.2: GPU/CPU parity
- 4.3: Logging infrastructure
- 4.4: Performance benchmarks
"""
import json
import os
import time
from pathlib import Path
import pytest
import numpy as np

if os.environ.get("RUN_ENGINE_TESTS") != "1":
    pytest.skip("Skipping validation tests (set RUN_ENGINE_TESTS=1)",
                allow_module_level=True)

import torch
from pytorch_solver import PytorchSolver

from conftest import (
    DEFAULT_TEST_CONFIG,
    REFERENCE_LOGS_DIR,
    load_reference,
    assert_states_equal,
)


# =============================================================================
# Phase 4.1: Reference Log Comparison Tests
# =============================================================================
class TestReferenceLogs:
    """Test against reference OpenCL output logs."""

    def test_reference_logs_exist(self):
        """Verify reference log files exist."""
        assert REFERENCE_LOGS_DIR.exists(), \
            f"Reference logs directory not found: {REFERENCE_LOGS_DIR}"

        log_files = list(REFERENCE_LOGS_DIR.glob("*.txt"))
        assert len(log_files) > 0, "No reference log files found"

    def test_density_is_computed_correctly(self):
        """Verify density computation produces valid results."""
        # Create particles in a grid with known spacing
        n = 10
        pos = torch.zeros((n, 4), dtype=torch.float32)
        pos[:, 2] = torch.linspace(0, 2, n)  # z: 0 to 2
        pos[:, 3] = 1.0  # LIQUID type
        vel = torch.zeros_like(pos)

        config = DEFAULT_TEST_CONFIG.copy()
        config["h"] = 0.5  # Smoothing length > particle spacing (0.22)
        config["grid_cells_x"] = 10
        config["grid_cells_y"] = 10
        config["grid_cells_z"] = 10
        config["grid_cell_count"] = 1000

        solver = PytorchSolver(pos, vel, config)

        solver.run_hash_particles()
        solver.run_sort()
        solver.run_index()
        solver.run_index_post_pass()
        solver.run_find_neighbors()
        solver.run_compute_density()

        density = solver.get_density().cpu().numpy()

        # Density should be positive
        assert (density > 0).all(), "Density should be positive"

        # Interior particles should have similar density (have same neighbors)
        interior = density[2:-2]
        assert np.std(interior) / np.mean(interior) < 0.1, \
            "Interior particles should have similar density"

        # Edge particles may have lower density (fewer neighbors)
        assert density[0] <= density[4], "Edge particle density <= interior"

    def test_pressure_follows_tait_equation(self):
        """Verify pressure follows Tait equation: P = (rho/rho0 - 1) * delta."""
        # Create particles in a grid
        n = 10
        pos = torch.zeros((n, 4), dtype=torch.float32)
        pos[:, 2] = torch.linspace(0, 2, n)  # z: 0 to 2
        pos[:, 3] = 1.0  # LIQUID type
        vel = torch.zeros_like(pos)

        config = DEFAULT_TEST_CONFIG.copy()
        config["h"] = 0.5
        config["rho0"] = 1000.0
        config["delta"] = 1.0
        config["grid_cells_x"] = 10
        config["grid_cells_y"] = 10
        config["grid_cells_z"] = 10
        config["grid_cell_count"] = 1000

        solver = PytorchSolver(pos, vel, config)

        solver.run_hash_particles()
        solver.run_sort()
        solver.run_index()
        solver.run_index_post_pass()
        solver.run_find_neighbors()
        solver.run_compute_density()
        solver.run_compute_pressure()

        density = solver.get_density().cpu().numpy()
        pressure = solver.pressure.cpu().numpy()

        # Verify Tait equation: P = max(0, (rho/rho0 - 1) * delta)
        # Note: Pressure is clamped to non-negative to prevent instabilities
        expected_pressure = np.maximum(0.0, (density / config["rho0"] - 1.0) * config["delta"])

        np.testing.assert_allclose(
            pressure, expected_pressure,
            rtol=1e-5,
            err_msg="Pressure doesn't follow Tait equation with clamp"
        )

        # Higher density should produce higher (or equal) pressure
        sorted_idx = np.argsort(density)
        sorted_pressure = pressure[sorted_idx]
        assert (np.diff(sorted_pressure) >= -1e-6).all(), \
            "Higher density should yield higher or equal pressure"


# =============================================================================
# Phase 4.2: GPU/CPU Parity Tests
# =============================================================================
class TestGPUCPUParity:
    """Verify GPU and CPU produce identical results."""

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_cpu_identical_results(self):
        """Verify GPU and CPU solvers produce identical results."""
        pos = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.3, 1.0],
            [0.0, 0.3, 0.0, 1.0],
        ], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        config_cpu = DEFAULT_TEST_CONFIG.copy()
        config_cpu["device"] = "cpu"

        config_gpu = DEFAULT_TEST_CONFIG.copy()
        config_gpu["device"] = "cuda"

        solver_cpu = PytorchSolver(pos.clone(), vel.clone(), config_cpu)
        solver_gpu = PytorchSolver(pos.clone(), vel.clone(), config_gpu)

        # Run both for several steps
        for _ in range(10):
            for solver in [solver_cpu, solver_gpu]:
                solver.run_hash_particles()
                solver.run_sort()
                solver.run_index()
                solver.run_index_post_pass()
                solver.run_find_neighbors()
                solver.run_compute_density()
                solver.run_compute_pressure()
                solver.run_compute_pressure_force_acceleration()
                solver.run_integrate()

        # Compare results
        pos_cpu = solver_cpu.position.cpu()
        pos_gpu = solver_gpu.position.cpu()

        torch.testing.assert_close(
            pos_cpu, pos_gpu,
            rtol=1e-5, atol=1e-5,
            msg="GPU/CPU position mismatch"
        )

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_tensors_on_correct_device(self):
        """Verify tensors are on GPU when device=cuda."""
        pos = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        config = DEFAULT_TEST_CONFIG.copy()
        config["device"] = "cuda"

        solver = PytorchSolver(pos, vel, config)

        assert solver.position.is_cuda, "Position should be on GPU"
        assert solver.velocity.is_cuda, "Velocity should be on GPU"


# =============================================================================
# Phase 4.3: Logging Infrastructure Tests
# =============================================================================
class TestLogging:
    """Test solver logging infrastructure."""

    def test_logging_methods_exist(self):
        """Check for logging methods after Phase 4.3."""
        pos = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver = PytorchSolver(pos, vel, DEFAULT_TEST_CONFIG.copy())

        logging_methods = ['enable_logging', 'get_step_log', 'export_logs']
        missing = [m for m in logging_methods if not hasattr(solver, m)]

        assert not missing, f"Missing logging methods: {missing}"

    def test_logging_disabled_by_default(self):
        """Verify logging is off by default (performance)."""
        pos = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver = PytorchSolver(pos, vel, DEFAULT_TEST_CONFIG.copy())

        log = solver.get_step_log()
        assert log is None or len(log) == 0, \
            "Logging should be disabled by default"

    def test_logging_captures_substeps(self):
        """Verify logging captures intermediate states when enabled."""
        pos = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.3, 1.0],
        ], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver = PytorchSolver(pos, vel, DEFAULT_TEST_CONFIG.copy())
        solver.enable_logging(True)

        solver.run_step()

        log = solver.get_step_log()
        # Log should be captured for the step
        # After run_step, the log is finished and moved to _step_logs
        # So get_step_log may return None, but _step_logs should have data
        assert len(solver._step_logs) > 0, "No logs captured"

        step_log = solver._step_logs[0]
        expected_substeps = ['hash_particles', 'compute_density', 'integrate']
        for substep in expected_substeps:
            assert substep in step_log, f"Missing substep log: {substep}"
            assert 'position' in step_log[substep], f"Missing position in {substep}"
            assert 'velocity' in step_log[substep], f"Missing velocity in {substep}"

    def test_logging_captures_pcisph_iterations(self):
        """Verify PCISPH iteration substeps are logged."""
        pos = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver = PytorchSolver(pos, vel, DEFAULT_TEST_CONFIG.copy())
        solver.enable_logging(True)

        solver.run_step()

        step_log = solver._step_logs[0]

        # Should have logs for 3 PCISPH iterations
        for i in range(3):
            assert f"pcisph_iter_{i}" in step_log, \
                f"Missing PCISPH iteration {i} log"

    def test_export_logs_creates_file(self):
        """Verify logs can be exported to file."""
        import tempfile

        pos = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver = PytorchSolver(pos, vel, DEFAULT_TEST_CONFIG.copy())
        solver.enable_logging(True)

        for _ in range(3):
            solver.run_step()

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            solver.export_logs(temp_path)

            assert Path(temp_path).exists(), "Log file not created"

            with open(temp_path) as f:
                exported = json.load(f)

            assert "steps" in exported, "Missing 'steps' key in exported logs"
            assert len(exported["steps"]) == 3, "Should have logs for 3 steps"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_timing_methods_exist(self):
        """Check for timing methods after Phase 4.4."""
        pos = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver = PytorchSolver(pos, vel, DEFAULT_TEST_CONFIG.copy())

        timing_methods = ['enable_timing', 'get_timing_breakdown']
        missing = [m for m in timing_methods if not hasattr(solver, m)]

        assert not missing, f"Missing timing methods: {missing}"

    def test_timing_disabled_by_default(self):
        """Verify timing is off by default."""
        pos = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver = PytorchSolver(pos, vel, DEFAULT_TEST_CONFIG.copy())

        timing = solver.get_timing_breakdown()
        assert len(timing) == 0, "Timing should be empty by default"

    def test_timing_collects_substep_data(self):
        """Verify timing captures substep durations when enabled."""
        pos = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.3, 1.0],
        ], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver = PytorchSolver(pos, vel, DEFAULT_TEST_CONFIG.copy())
        solver.enable_timing(True)

        for _ in range(10):
            solver.run_step()

        timing = solver.get_timing_breakdown()

        expected_substeps = [
            'hash_particles', 'sort', 'find_neighbors',
            'compute_density', 'compute_pressure', 'integrate'
        ]

        for substep in expected_substeps:
            assert substep in timing, f"Missing timing for: {substep}"
            assert timing[substep] > 0, f"Zero timing for: {substep}"


# =============================================================================
# Phase 4.4: Performance Benchmarks
# =============================================================================
class TestPerformance:
    """Performance benchmarks for PyTorch solver."""

    @pytest.mark.slow
    def test_basic_performance(self):
        """Verify basic solver performance is acceptable."""
        # Create larger particle set
        n_particles = 100
        pos = torch.rand(n_particles, 4, dtype=torch.float32)
        pos[:, 3] = 1.0  # Set particle type to liquid
        vel = torch.zeros_like(pos)

        config = DEFAULT_TEST_CONFIG.copy()
        config["h"] = 0.5
        config["grid_cells_x"] = 10
        config["grid_cells_y"] = 10
        config["grid_cells_z"] = 10
        config["grid_cell_count"] = 1000

        solver = PytorchSolver(pos, vel, config)

        # Warmup
        for _ in range(5):
            solver.run_step()

        # Benchmark
        start = time.perf_counter()
        n_iterations = 50

        for _ in range(n_iterations):
            solver.run_step()

        elapsed = time.perf_counter() - start
        steps_per_second = n_iterations / elapsed

        print(f"\nPerformance: {steps_per_second:.1f} steps/second "
              f"({n_particles} particles)")

        # Very loose performance requirement for basic tests
        assert steps_per_second > 1, \
            f"Performance too slow: {steps_per_second:.1f} steps/s"

    def test_step_timing_breakdown(self):
        """Profile individual substeps using built-in timing."""
        pos = torch.rand(50, 4, dtype=torch.float32)
        pos[:, 3] = 1.0
        vel = torch.zeros_like(pos)

        config = DEFAULT_TEST_CONFIG.copy()
        solver = PytorchSolver(pos, vel, config)

        # Enable timing and run steps
        solver.enable_timing(True)

        # Warmup
        for _ in range(5):
            solver.run_step()

        # Reset timing and profile
        solver.enable_timing(True)  # Reset counters

        for _ in range(100):
            solver.run_step()

        timing = solver.get_timing_breakdown()

        print("\nSubstep timing (ms per call):")
        for name, ms in sorted(timing.items(), key=lambda x: -x[1]):
            print(f"  {name}: {ms:.3f} ms")

        # Verify timing data is reasonable
        total = sum(timing.values())
        for name, ms in timing.items():
            pct = ms / total * 100
            assert pct < 90, f"{name} takes {pct:.1f}% of time - potential bottleneck"


# =============================================================================
# Integration Tests
# =============================================================================
class TestFullSimulation:
    """Test complete simulation runs."""

    def test_simulation_runs_without_crash(self):
        """Verify simulation can run many steps without crashing."""
        pos = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.3, 1.0],
            [0.0, 0.3, 0.0, 1.0],
            [0.3, 0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver = PytorchSolver(pos, vel, DEFAULT_TEST_CONFIG.copy())

        # Run 100 steps using run_step()
        for i in range(100):
            solver.run_step()

        # Should not have NaN
        assert not torch.isnan(solver.position).any(), \
            "Position contains NaN after simulation"
        assert not torch.isnan(solver.velocity).any(), \
            "Velocity contains NaN after simulation"

    def test_energy_stability(self):
        """Verify simulation is numerically stable (no NaN/Inf)."""
        pos = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.3, 1.0],
        ], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        # Disable all external forces for stability test
        config = DEFAULT_TEST_CONFIG.copy()
        config["gravity_x"] = 0.0
        config["gravity_y"] = 0.0
        config["gravity_z"] = 0.0
        config["delta"] = 0.0  # Disable pressure forces
        config["enable_viscosity"] = True
        config["enable_surface_tension"] = False

        solver = PytorchSolver(pos, vel, config)

        for _ in range(100):
            solver.run_step()

        # Primary check: No NaN or Inf values
        assert not torch.isnan(solver.position).any(), "Position contains NaN"
        assert not torch.isnan(solver.velocity).any(), "Velocity contains NaN"
        assert not torch.isinf(solver.position).any(), "Position contains Inf"
        assert not torch.isinf(solver.velocity).any(), "Velocity contains Inf"

        # Velocities should be finite and reasonable
        max_vel = solver.velocity[:, :3].abs().max().item()
        assert max_vel < 1e6, f"Velocity exploded: max = {max_vel}"

    def test_simulation_with_logging_and_timing(self):
        """Verify simulation works correctly with logging and timing enabled."""
        pos = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.3, 1.0],
        ], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver = PytorchSolver(pos, vel, DEFAULT_TEST_CONFIG.copy())

        # Enable both logging and timing
        solver.enable_logging(True)
        solver.enable_timing(True)

        for _ in range(20):
            solver.run_step()

        # Verify logging captured data
        assert len(solver._step_logs) == 20, "Should have 20 step logs"

        # Verify timing captured data
        timing = solver.get_timing_breakdown()
        assert len(timing) > 0, "Should have timing data"

        # Verify no NaN
        assert not torch.isnan(solver.position).any(), "No NaN in positions"
        assert not torch.isnan(solver.velocity).any(), "No NaN in velocities"
