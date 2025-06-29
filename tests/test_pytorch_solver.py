import torch
from pytorch_solver import PytorchSolver


def test_simple_flow():
    pos = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.4, 1.0], [2.0, 2.0, 2.0, 1.0]]
    )
    vel = torch.zeros_like(pos)
    cfg = {
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
        "rho0": 1.0,
        "delta": 1.0,
        "time_step": 0.01,
    }
    solver = PytorchSolver(pos, vel, cfg)
    solver.run_hash_particles()
    solver.run_sort()
    solver.run_index()
    solver.run_index_post_pass()
    solver.run_find_neighbors()
    solver.run_compute_density()
    solver.run_compute_pressure()
    solver.run_compute_pressure_force_acceleration()
    solver.run_integrate()

    # the first two particles should be neighbours
    neigh0 = solver.neighbor_map[0]
    assert 1 in neigh0[:2]
    assert solver.position.shape == pos.shape
    # velocities should change due to gravity
    assert torch.allclose(solver.velocity[0, 1], torch.tensor(-0.098), atol=1e-3)
