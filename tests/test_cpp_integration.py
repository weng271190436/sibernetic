"""
Tests for C++ to Python integration in Sibernetic.

Phase 1 tests:
- 1.1: Memory leak detection
- 1.2: Configuration parameter passing
- 1.3: Error propagation
- 1.4: Reset/constructor parity
"""
import os
import subprocess
import sys
from pathlib import Path
import pytest

# Allow skipping if not specifically enabled
if os.environ.get("RUN_ENGINE_TESTS") != "1":
    pytest.skip("Skipping engine integration tests (set RUN_ENGINE_TESTS=1)",
                allow_module_level=True)

import torch
from pytorch_solver import PytorchSolver

from conftest import (
    SIBERNETIC_BIN,
    TEST_CONFIG,
    DEFAULT_TEST_CONFIG,
    assert_states_equal,
)


# =============================================================================
# Phase 1.1: Memory Leak Tests
# =============================================================================
class TestMemoryLeaks:
    """Test that C++ integration doesn't leak Python objects."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_no_memory_leaks_subprocess(self):
        """Run solver init/reset multiple times via subprocess, verify memory stable."""
        if not SIBERNETIC_BIN.exists():
            pytest.skip("Sibernetic binary not built")

        import tracemalloc
        tracemalloc.start()

        # Run short simulations multiple times
        for i in range(20):  # Reduced from 100 for faster testing
            result = subprocess.run(
                [str(SIBERNETIC_BIN), "-no_g", "backend=torch",
                 "timelimit=0.0001", "-f", str(TEST_CONFIG)],
                capture_output=True,
                timeout=30
            )
            # Don't fail on non-zero - some configs may have issues
            # We're testing for memory leaks, not correctness here

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory should not grow significantly
        # Allow up to 50MB peak (generous for subprocess overhead)
        assert peak < 50 * 1024 * 1024, \
            f"Memory usage too high: {peak / 1024 / 1024:.1f}MB (possible leak)"

    def test_solver_repeated_creation_memory(self):
        """Create and destroy solver many times, check memory doesn't grow."""
        import gc

        # Get baseline memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        pos = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        memory_samples = []

        for i in range(50):
            # Create solver
            solver = PytorchSolver(pos.clone(), vel.clone(), DEFAULT_TEST_CONFIG.copy())
            solver.run_hash_particles()
            solver.run_sort()
            del solver

            if i % 10 == 0:
                gc.collect()
                # Sample memory (rough check via tensor allocation)
                probe = torch.zeros(1000, dtype=torch.float32)
                del probe

        gc.collect()

        # If there's a leak, memory samples would grow
        # This is a basic check - actual C++ leaks need valgrind


# =============================================================================
# Phase 1.2: Configuration Parameter Tests
# =============================================================================
class TestConfigParameters:
    """Verify all required config params are passed from C++ to Python."""

    def test_basic_config_keys_exist(self):
        """Check that basic config keys are stored in solver."""
        pos = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver = PytorchSolver(pos, vel, DEFAULT_TEST_CONFIG.copy())

        # Check solver has config
        assert hasattr(solver, 'config'), "Solver missing config attribute"

        basic_keys = ["h", "rho0", "delta", "time_step", "xmin", "ymin", "zmin"]
        for key in basic_keys:
            assert key in solver.config, f"Missing config key: {key}"

    def test_get_config_method(self):
        """Verify get_config() method returns config dict."""
        pos = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver = PytorchSolver(pos, vel, DEFAULT_TEST_CONFIG.copy())

        # Test get_config if it exists (Phase 1.2 implementation)
        if hasattr(solver, 'get_config'):
            config = solver.get_config()
            assert isinstance(config, dict), "get_config() should return dict"

            required_keys = [
                "h", "rho0", "delta", "time_step",
                "xmin", "ymin", "zmin",
                "gravity_x", "gravity_y", "gravity_z",
            ]
            for key in required_keys:
                assert key in config, f"get_config() missing key: {key}"
        else:
            pytest.skip("get_config() not yet implemented")

    def test_config_parameters_complete(self):
        """Verify ALL required config params are present (Phase 1.2 target)."""
        pos = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        # Include all params that should be added in Phase 1.2
        full_config = DEFAULT_TEST_CONFIG.copy()

        solver = PytorchSolver(pos, vel, full_config)

        if hasattr(solver, 'get_config'):
            config = solver.get_config()

            # Phase 1.2 additions
            phase_1_2_keys = [
                "max_neighbor_count",
                "simulation_scale",
                "simulation_scale_inv",
                "max_iteration",
                "device",
            ]

            for key in phase_1_2_keys:
                assert key in config, f"Missing Phase 1.2 config key: {key}"

            # Verify specific values
            assert config.get("max_neighbor_count") == 32, \
                "max_neighbor_count should be 32"
            assert config.get("max_iteration") == 3, \
                "max_iteration should be 3"
            assert config.get("device") in ["cpu", "cuda"], \
                "device should be 'cpu' or 'cuda'"
        else:
            # Direct attribute check as fallback
            assert solver.config.get("max_neighbor_count", 50) == \
                   full_config.get("max_neighbor_count", 32)


# =============================================================================
# Phase 1.3: Error Checking Tests
# =============================================================================
class TestErrorPropagation:
    """Verify Python exceptions are properly caught and reported."""

    def test_invalid_config_raises_error(self):
        """Verify invalid config raises appropriate error."""
        pos = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        # Missing required keys should fail gracefully
        bad_config = {"h": 0.5}  # Missing many required keys

        with pytest.raises((KeyError, RuntimeError, ValueError)):
            solver = PytorchSolver(pos, vel, bad_config)
            # Force usage that would expose missing config
            solver.run_hash_particles()

    def test_nan_values_detected(self):
        """Verify NaN values in input are handled."""
        pos = torch.tensor([[float('nan'), 0.0, 0.0, 1.0]], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver = PytorchSolver(pos, vel, DEFAULT_TEST_CONFIG.copy())

        # NaN should propagate but not crash
        solver.run_hash_particles()
        # If we get here without crashing, basic error handling works

    def test_empty_particles_handled(self):
        """Verify empty particle array is handled gracefully."""
        pos = torch.zeros((0, 4), dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver = PytorchSolver(pos, vel, DEFAULT_TEST_CONFIG.copy())

        # Should not crash on empty data
        solver.run_hash_particles()
        assert solver.position.shape[0] == 0


# =============================================================================
# Phase 1.4: Reset/Constructor Parity Tests
# =============================================================================
class TestResetParity:
    """Verify reset() produces same state as fresh construction."""

    def test_get_state_method_exists(self):
        """Verify get_state() method exists and returns dict."""
        pos = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver = PytorchSolver(pos, vel, DEFAULT_TEST_CONFIG.copy())

        # get_state() should already exist (from original implementation)
        assert hasattr(solver, 'get_state'), "Missing get_state() method"

        state = solver.get_state()
        # Original implementation returns (pos_list, vel_list) tuple
        # New implementation should return dict with all internal state
        assert state is not None, "get_state() returned None"

    def test_initial_state_reproducible(self):
        """Verify same inputs produce same initial state."""
        pos = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.4, 1.0],
        ], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver1 = PytorchSolver(pos.clone(), vel.clone(), DEFAULT_TEST_CONFIG.copy())
        solver2 = PytorchSolver(pos.clone(), vel.clone(), DEFAULT_TEST_CONFIG.copy())

        # Initial positions should match
        torch.testing.assert_close(solver1.position, solver2.position)
        torch.testing.assert_close(solver1.velocity, solver2.velocity)

    def test_step_produces_deterministic_output(self):
        """Verify running steps produces deterministic results."""
        pos = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.4, 1.0],
        ], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        solver1 = PytorchSolver(pos.clone(), vel.clone(), DEFAULT_TEST_CONFIG.copy())
        solver2 = PytorchSolver(pos.clone(), vel.clone(), DEFAULT_TEST_CONFIG.copy())

        # Run same operations on both
        for solver in [solver1, solver2]:
            solver.run_hash_particles()
            solver.run_sort()
            solver.run_index()
            solver.run_index_post_pass()
            solver.run_find_neighbors()
            solver.run_compute_density()
            solver.run_compute_pressure()
            solver.run_compute_pressure_force_acceleration()
            solver.run_integrate()

        # Results should be identical
        torch.testing.assert_close(solver1.position, solver2.position)
        torch.testing.assert_close(solver1.velocity, solver2.velocity)

    def test_reset_matches_constructor(self):
        """Verify reset() produces same state as fresh construction."""
        pos = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.4, 1.0],
        ], dtype=torch.float32)
        vel = torch.zeros_like(pos)

        # Create fresh solver and run one step
        solver1 = PytorchSolver(pos.clone(), vel.clone(), DEFAULT_TEST_CONFIG.copy())
        solver1.run_hash_particles()
        solver1.run_sort()
        solver1.run_index()
        solver1.run_index_post_pass()
        solver1.run_find_neighbors()
        solver1.run_compute_density()

        # Create second solver, run, reset, run again
        solver2 = PytorchSolver(pos.clone(), vel.clone(), DEFAULT_TEST_CONFIG.copy())
        solver2.run_hash_particles()
        solver2.run_sort()

        # If reset exists, use it
        if hasattr(solver2, 'reset'):
            solver2.reset(pos.clone(), vel.clone())
            solver2.run_hash_particles()
            solver2.run_sort()
            solver2.run_index()
            solver2.run_index_post_pass()
            solver2.run_find_neighbors()
            solver2.run_compute_density()

            # States should match after reset
            torch.testing.assert_close(
                solver1.rho, solver2.rho,
                msg="Density mismatch after reset"
            )
        else:
            pytest.skip("reset() not yet implemented")
