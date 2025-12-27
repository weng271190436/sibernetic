"""
Shared pytest fixtures for Sibernetic PyTorch solver tests.
"""
import os
import json
from pathlib import Path
import pytest
import numpy as np

# Conditionally import torch - some tests don't need it
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Path constants
# =============================================================================
SIBERNETIC_ROOT = Path(__file__).parent.parent
SIBERNETIC_BIN = SIBERNETIC_ROOT / "Release" / "Sibernetic"
TEST_CONFIG = SIBERNETIC_ROOT / "configuration" / "test" / "test_energy"
REFERENCE_LOGS_DIR = SIBERNETIC_ROOT / "tests" / "data" / "reference_logs"


# =============================================================================
# Test configuration defaults
# =============================================================================
DEFAULT_TEST_CONFIG = {
    "xmin": 0.0,
    "ymin": 0.0,
    "zmin": 0.0,
    "hash_grid_cell_size_inv": 1.0,
    "grid_cells_x": 4,
    "grid_cells_y": 4,
    "grid_cells_z": 4,
    "grid_cell_count": 64,
    "h": 0.5,
    "mass_mult_Wpoly6Coefficient": 1.0,
    "mass_mult_gradWspikyCoefficient": 1.0,
    "rho0": 1000.0,
    "delta": 1.0,
    "time_step": 0.01,
    "gravity_x": 0.0,
    "gravity_y": -9.8,
    "gravity_z": 0.0,
    "simulation_scale": 1.0,
    "simulation_scale_inv": 1.0,
    "max_neighbor_count": 32,
    "max_iteration": 3,
    "device": "cpu",
}


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def test_config():
    """Return a copy of the default test configuration."""
    return DEFAULT_TEST_CONFIG.copy()


@pytest.fixture
def get_test_config():
    """Factory fixture to get test configs with customization."""
    def _get_config(**overrides):
        cfg = DEFAULT_TEST_CONFIG.copy()
        cfg.update(overrides)
        return cfg
    return _get_config


@pytest.fixture
def get_test_particles():
    """Return simple test particles for basic tests."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    def _get_particles(n=3):
        # Simple 3-particle setup
        pos = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.4, 1.0],
            [2.0, 2.0, 2.0, 1.0],
        ], dtype=torch.float32)
        vel = torch.zeros_like(pos)
        return pos[:n], vel[:n]

    return _get_particles


# =============================================================================
# Solver factory fixtures
# =============================================================================
@pytest.fixture
def create_test_solver(get_test_particles, get_test_config):
    """Create a PytorchSolver with default test configuration."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    from pytorch_solver import PytorchSolver

    def _create(**config_overrides):
        pos, vel = get_test_particles()
        config = get_test_config(**config_overrides)
        return PytorchSolver(pos, vel, config)

    return _create


@pytest.fixture
def create_two_particle_solver(get_test_config):
    """Create solver with exactly two particles at specified distance."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    from pytorch_solver import PytorchSolver

    def _create(distance=0.3):
        pos = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [distance, 0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        vel = torch.zeros_like(pos)
        config = get_test_config(h=max(0.5, distance * 2))  # Ensure in range
        return PytorchSolver(pos, vel, config)

    return _create


@pytest.fixture
def create_solver_with_sparse_particles(get_test_config):
    """Create solver with particles far apart (no neighbors)."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    from pytorch_solver import PytorchSolver

    def _create():
        pos = torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [10.0, 10.0, 10.0, 1.0],  # Very far apart
            [20.0, 20.0, 20.0, 1.0],
        ], dtype=torch.float32)
        vel = torch.zeros_like(pos)
        config = get_test_config()
        return PytorchSolver(pos, vel, config)

    return _create


# =============================================================================
# Reference data loading
# =============================================================================
def load_reference(filename):
    """Load reference data from tests/data/reference_logs."""
    filepath = REFERENCE_LOGS_DIR / filename
    if filepath.suffix == '.txt':
        return np.loadtxt(filepath)
    elif filepath.suffix == '.npz':
        return np.load(filepath)
    elif filepath.suffix == '.json':
        with open(filepath) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown file type: {filepath.suffix}")


def load_reference_log(log_file):
    """Load a full reference log file with positions, velocities, etc."""
    # Placeholder - implement based on actual log format
    data = {}
    data["num_steps"] = 0
    data["checkpoints"] = {}
    return data


# =============================================================================
# State comparison helpers
# =============================================================================
def assert_states_equal(state1, state2, rtol=1e-5, atol=1e-5):
    """Assert two solver states are equal within tolerance."""
    if isinstance(state1, dict) and isinstance(state2, dict):
        for key in state1:
            assert key in state2, f"Missing key in state2: {key}"
            if TORCH_AVAILABLE and torch.is_tensor(state1[key]):
                torch.testing.assert_close(
                    state1[key], state2[key],
                    rtol=rtol, atol=atol,
                    msg=f"Mismatch in {key}"
                )
            else:
                np.testing.assert_allclose(
                    state1[key], state2[key],
                    rtol=rtol, atol=atol,
                    err_msg=f"Mismatch in {key}"
                )
    elif TORCH_AVAILABLE and torch.is_tensor(state1):
        torch.testing.assert_close(state1, state2, rtol=rtol, atol=atol)
    else:
        np.testing.assert_allclose(state1, state2, rtol=rtol, atol=atol)


# =============================================================================
# Markers
# =============================================================================
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring CUDA GPU"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests that require the full Sibernetic binary"
    )
