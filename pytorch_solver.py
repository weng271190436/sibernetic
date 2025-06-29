import torch


class PytorchSolver:
    """Simplified PCISPH solver implemented with PyTorch tensors."""

    def __init__(self, position, velocity, config):
        self.device = torch.device(config.get("device", "cpu"))
        self.config = config
        self.position = torch.as_tensor(
            position, dtype=torch.float32, device=self.device
        )
        self.velocity = torch.as_tensor(
            velocity, dtype=torch.float32, device=self.device
        )
        self.acceleration = torch.zeros_like(self.position)
        self.pressure = torch.zeros(self.position.shape[0], device=self.device)
        self.rho = torch.zeros(self.position.shape[0], device=self.device)

        self.particle_index = None
        self.sorted_position = None
        self.sorted_velocity = None
        self.particle_index_back = None
        self.grid_cell_index = None
        self.grid_cell_index_fixed = None
        self.neighbor_map = None

    # ------------------------------------------------------------------
    # Neighbour search helpers
    # ------------------------------------------------------------------
    def run_hash_particles(self):
        """Assign a grid cell index to each particle."""
        pos = self.position[:, :3]
        offset = torch.tensor(
            [self.config["xmin"], self.config["ymin"], self.config["zmin"]],
            device=self.device,
        )
        idx = torch.floor(
            (pos - offset) * self.config["hash_grid_cell_size_inv"]
        ).long()
        cell_id = (
            idx[:, 0]
            + idx[:, 1] * self.config["grid_cells_x"]
            + idx[:, 2] * self.config["grid_cells_x"] * self.config["grid_cells_y"]
        )
        ids = torch.arange(pos.shape[0], device=self.device)
        self.particle_index = torch.stack([cell_id, ids], dim=1)

    def run_sort(self):
        """Sort particle index by cell id and permute buffers accordingly."""
        order = torch.argsort(self.particle_index[:, 0])
        self.particle_index = self.particle_index[order]
        self.sorted_position = self.position[order]
        self.sorted_velocity = self.velocity[order]
        self.particle_index_back = order

    def run_index(self):
        """Compute starting index in the sorted array for each grid cell."""
        num_cells = self.config["grid_cell_count"]
        counts = torch.bincount(self.particle_index[:, 0], minlength=num_cells)
        start = torch.cumsum(
            torch.cat(
                [torch.zeros(1, device=self.device, dtype=torch.long), counts[:-1]]
            ),
            dim=0,
        )
        index = torch.where(counts > 0, start, torch.full_like(start, -1))
        index = torch.cat(
            [index, torch.tensor([self.particle_index.shape[0]], device=self.device)]
        )
        self.grid_cell_index = index

    def run_index_post_pass(self):
        """Fill empty cell slots with the next non-empty cell index."""
        fixed = self.grid_cell_index.clone()
        mask = (fixed == -1)
        fixed[mask] = torch.flip(torch.cumsum(torch.flip((fixed != -1).long() * fixed, dims=[0]), dim=0), dims=[0])[mask]
        self.grid_cell_index_fixed = fixed

    def run_find_neighbors(self):
        """Search neighbors using the hashed grid."""
        n = self.sorted_position.shape[0]
        max_n = self.config.get("max_neighbor_count", 50)
        neighbors = torch.full((n, max_n), -1, dtype=torch.long, device=self.device)
        pos = self.sorted_position[:, :3]
        h = self.config["h"]

        # Compute all pairwise distances within the search radius
        dist_matrix = torch.cdist(pos, pos)
        neighbor_mask = (dist_matrix < h) & (dist_matrix > 0)

        # Populate the neighbors tensor
        for i in range(n):
            neighbor_indices = torch.nonzero(neighbor_mask[i], as_tuple=False).squeeze(1)
            count = min(len(neighbor_indices), max_n)
            neighbors[i, :count] = neighbor_indices[:count]
        self.neighbor_map = neighbors

    # ------------------------------------------------------------------
    # PCISPH steps
    # ------------------------------------------------------------------
    def run_compute_density(self):
        """Compute particle densities from neighbor positions."""
        pos = self.sorted_position[:, :3]
        i_idx = torch.repeat_interleave(
            torch.arange(pos.shape[0], device=self.device),
            self.neighbor_map.shape[1],
        )
        j_idx = self.neighbor_map.reshape(-1)
        mask = j_idx >= 0
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        diff = pos[i_idx] - pos[j_idx]
        r2 = (diff * diff).sum(dim=1)
        w = (self.config["h"] ** 2 - r2).clamp(min=0) ** 3
        dens = torch.zeros(pos.shape[0], device=self.device)
        dens.scatter_add_(0, i_idx, w * self.config["mass_mult_Wpoly6Coefficient"])
        self.rho = dens

    def run_compute_pressure(self):
        """Compute pressure from density error."""
        self.pressure = self.config["delta"] * (self.rho - self.config["rho0"])

    def run_compute_pressure_force_acceleration(self):
        """Compute pressure forces and update acceleration."""
        pos = self.sorted_position[:, :3]
        i_idx = torch.repeat_interleave(
            torch.arange(pos.shape[0], device=self.device),
            self.neighbor_map.shape[1],
        )
        j_idx = self.neighbor_map.reshape(-1)
        mask = j_idx >= 0
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        diff = pos[i_idx] - pos[j_idx]
        dist = diff.norm(dim=1)
        dir = diff / (dist.unsqueeze(1) + 1e-12)
        grad = (
            self.config["mass_mult_gradWspikyCoefficient"]
            * (self.config["h"] - dist).clamp(min=0) ** 2
        ).unsqueeze(1) * dir
        pres = (
            self.pressure[i_idx] / (self.rho[i_idx] ** 2)
            + self.pressure[j_idx] / (self.rho[j_idx] ** 2)
        ).unsqueeze(1)
        force = -pres * grad
        acc = torch.zeros_like(self.sorted_position)
        acc.scatter_add_(0, i_idx.unsqueeze(1).expand(-1, 3), force)
        gravity = torch.tensor(
            [
                self.config.get("gravity_x", 0.0),
                self.config.get("gravity_y", -9.8),
                self.config.get("gravity_z", 0.0),
            ],
            device=self.device,
        )
        acc[:, :3] += gravity
        self.acceleration = acc

    def run_integrate(self):
        """Integrate positions and velocities using current acceleration."""
        dt = self.config["time_step"]
        self.sorted_velocity += dt * self.acceleration
        self.sorted_position += dt * self.sorted_velocity
        inv = torch.argsort(self.particle_index_back)
        self.position = self.sorted_position[inv]
        self.velocity = self.sorted_velocity[inv]
