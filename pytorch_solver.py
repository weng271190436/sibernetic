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
        # The reference engine does not preserve the placeholder fourth
        # velocity component and always outputs zeros.  Mirror this
        # behaviour so our logs match the reference files.
        if self.velocity.shape[1] > 3:
            self.velocity[:, 3] = 0.0
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
        self.boundary_force = None  # Phase 3.1: Boundary interaction force

        # Phase 4.3: Logging infrastructure
        self._logging_enabled = False
        self._step_logs = []
        self._current_step_log = {}

        # Phase 4.4: Timing infrastructure
        self._timing_enabled = False
        self._timing_data = {}
        self._step_count = 0

    # ------------------------------------------------------------------
    # Phase 4.3: Logging Infrastructure
    # ------------------------------------------------------------------
    def enable_logging(self, enabled=True):
        """Enable or disable step logging (Phase 4.3).

        When enabled, intermediate states are captured during run_step().
        Logging is disabled by default for performance.
        """
        self._logging_enabled = enabled
        if enabled:
            self._step_logs = []
            self._current_step_log = {}

    def get_step_log(self):
        """Return the current step log (Phase 4.3).

        Returns None or empty dict when logging is disabled.
        When enabled, returns dict with substep names as keys.
        """
        if not self._logging_enabled:
            return None
        return self._current_step_log if self._current_step_log else None

    def export_logs(self, filepath):
        """Export all collected logs to a JSON file (Phase 4.3).

        Args:
            filepath: Path to save the log file.
        """
        import json

        # Convert tensor data to lists for JSON serialization
        export_data = {
            "steps": []
        }

        for step_log in self._step_logs:
            step_data = {}
            for substep_name, substep_data in step_log.items():
                step_data[substep_name] = {}
                for key, value in substep_data.items():
                    if hasattr(value, 'tolist'):  # Tensor or numpy array
                        step_data[substep_name][key] = value.tolist()
                    else:
                        step_data[substep_name][key] = value
            export_data["steps"].append(step_data)

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

    def _log_substep(self, substep_name):
        """Log the current state for a substep (internal)."""
        if not self._logging_enabled:
            return

        self._current_step_log[substep_name] = {
            "position": self.position.detach().cpu().clone(),
            "velocity": self.velocity.detach().cpu().clone(),
            "density": self.rho.detach().cpu().clone() if hasattr(self, 'rho') else None,
            "pressure": self.pressure.detach().cpu().clone() if hasattr(self, 'pressure') else None,
        }

    def _finish_step_log(self):
        """Complete logging for the current step (internal)."""
        if self._logging_enabled and self._current_step_log:
            self._step_logs.append(self._current_step_log.copy())
            self._current_step_log = {}

    # ------------------------------------------------------------------
    # Phase 4.4: Timing Infrastructure
    # ------------------------------------------------------------------
    def enable_timing(self, enabled=True):
        """Enable or disable substep timing collection (Phase 4.4).

        When enabled, time is tracked for each substep.
        """
        self._timing_enabled = enabled
        if enabled:
            self._timing_data = {}
            self._step_count = 0

    def get_timing_breakdown(self):
        """Return timing breakdown in milliseconds per substep (Phase 4.4).

        Returns dict mapping substep names to average time in ms.
        """
        if not self._timing_data or self._step_count == 0:
            return {}

        return {
            name: (total_ns / self._step_count) / 1e6  # Convert to ms
            for name, total_ns in self._timing_data.items()
        }

    def _time_substep(self, substep_name, func):
        """Time a substep and optionally record it (internal)."""
        if not self._timing_enabled:
            return func()

        import time
        start = time.perf_counter_ns()
        result = func()
        elapsed = time.perf_counter_ns() - start

        if substep_name not in self._timing_data:
            self._timing_data[substep_name] = 0
        self._timing_data[substep_name] += elapsed

        return result

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

        # Clamp indices to valid grid range to prevent negative cell IDs
        grid_x = self.config["grid_cells_x"]
        grid_y = self.config["grid_cells_y"]
        grid_z = self.config["grid_cells_z"]
        idx[:, 0] = torch.clamp(idx[:, 0], 0, grid_x - 1)
        idx[:, 1] = torch.clamp(idx[:, 1], 0, grid_y - 1)
        idx[:, 2] = torch.clamp(idx[:, 2], 0, grid_z - 1)

        cell_id = (
            idx[:, 0]
            + idx[:, 1] * grid_x
            + idx[:, 2] * grid_x * grid_y
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
        mask = fixed == -1
        fixed[mask] = torch.flip(
            torch.cumsum(torch.flip((fixed != -1).long() * fixed, dims=[0]), dim=0),
            dims=[0],
        )[mask]
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
            neighbor_indices = torch.nonzero(neighbor_mask[i], as_tuple=False).squeeze(
                1
            )
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
        # Phase 2.1: Apply minimum density clamp per OpenCL implementation
        # Prevents division by zero and matches reference solver behavior
        h_scaled = self.config["h"] * self.config.get("simulation_scale", 1.0)
        dens = torch.clamp(dens, min=h_scaled ** 6)
        self.rho = dens

    def run_compute_pressure(self):
        """Compute pressure from density using Tait equation.

        P = (rho/rho0 - 1) * delta = delta * (rho - rho0) / rho0

        Note: Pressure is clamped to >= 0 to prevent instabilities from
        negative pressure when particles are sparse.
        """
        rho0 = self.config["rho0"]
        self.pressure = self.config["delta"] * (self.rho - rho0) / rho0
        # Clamp to non-negative - prevents explosive repulsion when sparse
        self.pressure = torch.clamp(self.pressure, min=0.0)

    def run_compute_pressure_force_acceleration(self):
        """Compute pressure forces and update acceleration.

        Phase 2.3: Uses correct pressure force formula matching OpenCL:
        pres_factor = 0.5 * (p[i] + p[j]) / rho[j]
        """
        pos = self.sorted_position[:, :3]
        sim_scale = self.config.get("simulation_scale", 1.0)
        h = self.config["h"]
        h_scaled = h * sim_scale

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
        # Normalized direction
        dir_vec = diff / (dist.unsqueeze(1) + 1e-12)

        # Use predicted_rho if available (from PCISPH), otherwise use rho
        rho_j = getattr(self, 'predicted_rho', self.rho)[j_idx]

        # Phase 2.3: Correct pressure factor formula per OpenCL
        # pres_factor = 0.5 * (p[i] + p[j]) / rho[j]
        pres_factor = 0.5 * (self.pressure[i_idx] + self.pressure[j_idx]) / (rho_j + 1e-12)

        # Spiky kernel gradient: (h_scaled - r)^2 * sim_scale / r
        grad_factor = (
            self.config["mass_mult_gradWspikyCoefficient"]
            * (h_scaled - dist).clamp(min=0) ** 2
            * sim_scale / (dist + 1e-12)
        )

        # Combine into force
        force = -pres_factor.unsqueeze(1) * grad_factor.unsqueeze(1) * dir_vec

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

        # Add viscosity if enabled
        if self.config.get("enable_viscosity", True):
            self.run_compute_viscosity()
            acc += self.viscosity_acceleration

        # Add surface tension if enabled
        if self.config.get("enable_surface_tension", True):
            self.run_compute_surface_tension()
            acc += self.surface_tension_acceleration

        # Add elastic forces if connections are loaded
        if hasattr(self, 'elastic_connections') and self.elastic_connections is not None:
            self.run_compute_elastic_force()
            acc += self.elastic_force

        # Add muscle forces if activation is set
        if (hasattr(self, 'elastic_connections') and self.elastic_connections is not None and
            hasattr(self, 'muscle_activation') and self.muscle_activation is not None):
            self.run_compute_muscle_force()
            acc += self.muscle_force

        # Add boundary forces (repulsion from boundary particles)
        if self.config.get("enable_boundary", True):
            self.run_compute_boundary_force()
            acc += self.boundary_force

        self.acceleration = acc

    # ------------------------------------------------------------------
    # Phase 3.2: Viscosity Forces
    # ------------------------------------------------------------------
    def run_compute_viscosity(self):
        """Compute viscosity forces between particles.

        Formula: accel += coeff * (v_j - v_i) * (h_scaled - r_ij) / 1000
        Then scaled by: 1.5 * mass_mult_divgradWviscosityCoefficient / rho[i]
        """
        pos = self.sorted_position[:, :3]
        vel = self.sorted_velocity[:, :3]
        n = pos.shape[0]

        sim_scale = self.config.get("simulation_scale", 1.0)
        h = self.config["h"]
        h_scaled = h * sim_scale

        # Get neighbor pairs
        i_idx = torch.repeat_interleave(
            torch.arange(n, device=self.device),
            self.neighbor_map.shape[1],
        )
        j_idx = self.neighbor_map.reshape(-1)
        mask = j_idx >= 0
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]

        # Get particle types in sorted order
        sorted_types = self.position[:, 3][self.particle_index_back].int()

        # Boundary particles (type=3) don't contribute to viscosity
        # In OpenCL: not_bp = (type != 3), velocity is multiplied by not_bp
        j_types = sorted_types[j_idx]
        not_bp = (j_types != 3).float()

        # Compute distances
        diff = pos[i_idx] - pos[j_idx]
        dist = diff.norm(dim=1)

        # Velocity difference (v_j - v_i)
        # For boundary particles, their velocity contribution is zeroed
        v_diff = vel[j_idx] * not_bp.unsqueeze(1) - vel[i_idx]

        # Viscosity coefficient based on particle types
        # For simplicity, use default 1.0e-4 for all pairs
        # (OpenCL uses different values for different type pairs)
        visc_coeff = self.config.get("viscosity_coefficient", 1.0e-4)

        # Viscosity kernel: (h_scaled - r_ij) / 1000
        visc_kernel = ((h_scaled - dist).clamp(min=0) / 1000.0).unsqueeze(1)

        # Compute viscosity acceleration contribution
        visc_acc = visc_coeff * v_diff * visc_kernel

        # Accumulate per particle
        visc_total = torch.zeros_like(self.sorted_position)
        visc_total.scatter_add_(0, i_idx.unsqueeze(1).expand(-1, 3), visc_acc)

        # Scale by coefficient and density
        mass_mult = self.config.get("mass_mult_divgradWviscosityCoefficient", 1.0)
        rho = getattr(self, 'predicted_rho', self.rho)

        # Final scaling: 1.5 * mass_mult_divgradWviscosityCoefficient / rho[i]
        scale = 1.5 * mass_mult / (rho.unsqueeze(1) + 1e-12)
        visc_total[:, :3] *= scale

        # Don't apply viscosity to boundary particles
        is_boundary = sorted_types == 3
        visc_total[is_boundary] = 0.0

        self.viscosity_acceleration = visc_total

    def get_viscosity_acceleration(self):
        """Return viscosity acceleration for testing (Phase 3.2)."""
        return getattr(self, 'viscosity_acceleration', None)

    # ------------------------------------------------------------------
    # Phase 3.3: Surface Tension Forces
    # ------------------------------------------------------------------
    def run_compute_surface_tension(self):
        """Compute surface tension forces between particles.

        Formula: accel += -1.7e-09 * surfTensCoeff * (h² - r²)³ * (pos_i - pos_j) / mass
        """
        pos = self.sorted_position[:, :3]
        n = pos.shape[0]

        sim_scale = self.config.get("simulation_scale", 1.0)
        h = self.config["h"]
        h_scaled = h * sim_scale
        h_scaled2 = h_scaled * h_scaled

        mass = self.config.get("mass", 1.0)
        surf_tens_coeff = self.config.get("surf_tens_coeff", 1.0)

        # Get neighbor pairs
        i_idx = torch.repeat_interleave(
            torch.arange(n, device=self.device),
            self.neighbor_map.shape[1],
        )
        j_idx = self.neighbor_map.reshape(-1)
        mask = j_idx >= 0
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]

        # Compute distances squared
        diff = pos[i_idx] - pos[j_idx]
        r2 = (diff * diff).sum(dim=1)

        # Surface tension kernel: (h² - r²)³
        surf_kernel = (h_scaled2 - r2).clamp(min=0) ** 3

        # Surface tension coefficient: -1.7e-09 * surfTensCoeff
        coeff = -1.7e-09 * surf_tens_coeff

        # Surface tension force contribution
        surf_force = coeff * surf_kernel.unsqueeze(1) * diff

        # Accumulate per particle
        surf_total = torch.zeros_like(self.sorted_position)
        surf_total.scatter_add_(0, i_idx.unsqueeze(1).expand(-1, 3), surf_force)

        # Divide by mass
        surf_total[:, :3] /= mass

        # Don't apply to boundary particles
        sorted_types = self.position[:, 3][self.particle_index_back].int()
        is_boundary = sorted_types == 3
        surf_total[is_boundary] = 0.0

        self.surface_tension_acceleration = surf_total

    def get_surface_tension_acceleration(self):
        """Return surface tension acceleration for testing (Phase 3.3)."""
        return getattr(self, 'surface_tension_acceleration', None)

    # ------------------------------------------------------------------
    # Phase 3.4: Elastic Connection Forces
    # ------------------------------------------------------------------
    def load_elastic_connections(self, connections_data):
        """Load elastic connections data for worm body simulation.

        Args:
            connections_data: Tensor of shape (num_elastic_particles, max_connections, 4)
                Each connection has: [particle_j_id, equilibrium_distance, muscle_index, unused]
                Use -1 for particle_j_id to indicate no connection.
        """
        self.elastic_connections = torch.as_tensor(
            connections_data, dtype=torch.float32, device=self.device
        )
        self.num_elastic_particles = self.elastic_connections.shape[0]
        self.max_elastic_connections = self.elastic_connections.shape[1]

    def get_elastic_connections(self):
        """Return elastic connections buffer for testing."""
        return getattr(self, 'elastic_connections', None)

    def run_compute_elastic_force(self):
        """Compute elastic spring forces between connected particles.

        Formula: acc += -(r_ij / |r_ij|) * delta_r * elasticityCoefficient
        where delta_r = |r_ij| - equilibrium_distance
        """
        if not hasattr(self, 'elastic_connections') or self.elastic_connections is None:
            self.elastic_force = torch.zeros_like(self.sorted_position)
            return

        n = self.sorted_position.shape[0]
        sim_scale = self.config.get("simulation_scale", 1.0)
        elasticity_coeff = self.config.get("elasticity_coefficient", 1.0)

        # Initialize elastic force
        elastic_force = torch.zeros_like(self.sorted_position)

        # Get elastic particle indices (particles that have elastic connections)
        num_elastic = self.num_elastic_particles
        if num_elastic == 0:
            self.elastic_force = elastic_force
            return

        # Process each elastic particle
        # Note: elastic particles are assumed to be at the beginning of the position array
        for i in range(num_elastic):
            # Get this particle's sorted index
            sorted_i = self.particle_index_back[i] if self.particle_index_back is not None else i
            if sorted_i >= n:
                continue

            pos_i = self.sorted_position[sorted_i, :3]

            for c in range(self.max_elastic_connections):
                conn = self.elastic_connections[i, c]
                j_id = int(conn[0].item())

                if j_id < 0:  # No connection
                    continue

                if j_id >= n:
                    continue

                # Get connected particle's sorted index
                sorted_j = self.particle_index_back[j_id] if self.particle_index_back is not None else j_id
                if sorted_j >= n:
                    continue

                r_ij_equilibrium = conn[1].item()  # Rest length

                pos_j = self.sorted_position[sorted_j, :3]

                # Vector from j to i, scaled
                vect_r_ij = (pos_i - pos_j) * sim_scale
                r_ij = vect_r_ij.norm()

                if r_ij < 1e-10:
                    continue

                # Displacement from equilibrium
                delta_r = r_ij - r_ij_equilibrium

                if abs(delta_r) < 1e-10:
                    continue

                # Direction (normalized)
                direction = vect_r_ij / r_ij

                # Spring force: -(direction) * delta_r * elasticity
                force = -direction * delta_r * elasticity_coeff

                elastic_force[sorted_i, :3] += force

        self.elastic_force = elastic_force

    def get_elastic_force(self):
        """Return elastic force for testing (Phase 3.4)."""
        return getattr(self, 'elastic_force', None)

    # ------------------------------------------------------------------
    # Phase 3.5: Muscle Forces
    # Muscles are springs with contraction based on activation signal
    # ------------------------------------------------------------------
    def set_muscle_activation(self, activation):
        """Set muscle activation signal for worm muscle contraction.

        Args:
            activation: Tensor of shape (num_muscles,) with values 0.0-1.0
                        indicating muscle activation level.
        """
        self.muscle_activation = torch.as_tensor(
            activation, dtype=torch.float32, device=self.device
        )
        self.num_muscles = len(self.muscle_activation)

    def get_muscle_activation(self):
        """Return muscle activation signal for testing."""
        return getattr(self, 'muscle_activation', None)

    def run_compute_muscle_force(self):
        """Compute muscle contraction forces.

        Formula from OpenCL:
            acceleration += -(vect_r_ij / r_ij) * muscle_activation_signal[i] * max_muscle_force

        The muscle index is stored in elastic_connections[..., 2] (1-indexed).
        A value of 0 or -1 means the connection is not a muscle.
        """
        if not hasattr(self, 'elastic_connections') or self.elastic_connections is None:
            self.muscle_force = torch.zeros_like(self.sorted_position)
            return

        if not hasattr(self, 'muscle_activation') or self.muscle_activation is None:
            self.muscle_force = torch.zeros_like(self.sorted_position)
            return

        n = self.sorted_position.shape[0]
        sim_scale = self.config.get("simulation_scale", 1.0)
        max_muscle_force = self.config.get("max_muscle_force", 4000.0)

        # Initialize muscle force
        muscle_force = torch.zeros_like(self.sorted_position)

        num_elastic = self.num_elastic_particles
        if num_elastic == 0:
            self.muscle_force = muscle_force
            return

        # Process each elastic particle
        for i in range(num_elastic):
            sorted_i = self.particle_index_back[i] if self.particle_index_back is not None else i
            if sorted_i >= n:
                continue

            pos_i = self.sorted_position[sorted_i, :3]

            for c in range(self.max_elastic_connections):
                conn = self.elastic_connections[i, c]
                j_id = int(conn[0].item())

                if j_id < 0:
                    continue
                if j_id >= n:
                    continue

                # Muscle index is stored in conn[2] (1-indexed, 0 = no muscle)
                muscle_idx = int(conn[2].item())
                if muscle_idx <= 0:  # Not a muscle connection
                    continue

                # Convert to 0-indexed
                muscle_idx_0 = muscle_idx - 1
                if muscle_idx_0 >= self.num_muscles:
                    continue

                # Get activation for this muscle
                activation = self.muscle_activation[muscle_idx_0].item()
                if activation <= 0.0:
                    continue

                sorted_j = self.particle_index_back[j_id] if self.particle_index_back is not None else j_id
                if sorted_j >= n:
                    continue

                pos_j = self.sorted_position[sorted_j, :3]

                # Vector from j to i, scaled
                vect_r_ij = (pos_i - pos_j) * sim_scale
                r_ij = vect_r_ij.norm()

                if r_ij < 1e-10:
                    continue

                # Direction (normalized)
                direction = vect_r_ij / r_ij

                # Muscle contraction force: pulls particles together
                # acceleration += -(direction) * activation * max_force
                force = -direction * activation * max_muscle_force

                muscle_force[sorted_i, :3] += force

        self.muscle_force = muscle_force

    def get_muscle_force(self):
        """Return muscle force for testing (Phase 3.5)."""
        return getattr(self, 'muscle_force', None)

    # ------------------------------------------------------------------
    # Phase 3.1: Boundary Particle Handling
    # Based on Ihmsen et al. 2010: "Boundary Handling and Adaptive
    # Time-stepping for PCISPH"
    # ------------------------------------------------------------------
    def get_particle_types(self):
        """Extract particle types from position.w component.

        Types:
            1 = LIQUID_PARTICLE
            2 = ELASTIC_PARTICLE
            3 = BOUNDARY_PARTICLE
        """
        return self.position[:, 3].int()

    def run_compute_boundary_force(self):
        """Compute boundary interaction forces using Ihmsen et al. 2010 method.

        For boundary particles (type=3), the velocity buffer stores the
        surface normal instead of velocity (memory optimization from OpenCL).
        """
        n = self.sorted_position.shape[0]
        pos = self.sorted_position[:, :3]
        vel = self.sorted_velocity[:, :3]

        # Get r0 from config (boundary interaction radius)
        r0 = self.config.get("r0", self.config["h"] * 0.5)

        # Get particle types in sorted order
        inv = torch.argsort(self.particle_index_back)
        sorted_types = self.position[:, 3][self.particle_index_back].int()

        # Boundary normals are stored in velocity for boundary particles
        # This is a memory optimization from the OpenCL implementation
        boundary_normals = self.sorted_velocity[:, :3]

        # Initialize boundary force buffer
        boundary_force = torch.zeros_like(self.sorted_position)

        # Get neighbor pairs
        i_idx = torch.repeat_interleave(
            torch.arange(n, device=self.device),
            self.neighbor_map.shape[1],
        )
        j_idx = self.neighbor_map.reshape(-1)
        valid = j_idx >= 0
        i_idx = i_idx[valid]
        j_idx = j_idx[valid]

        # Filter to only boundary neighbors (type == 3)
        j_types = sorted_types[j_idx]
        boundary_mask = j_types == 3
        i_boundary = i_idx[boundary_mask]
        j_boundary = j_idx[boundary_mask]

        if i_boundary.numel() == 0:
            self.boundary_force = boundary_force
            return

        # Compute distances to boundary particles
        diff = pos[i_boundary] - pos[j_boundary]
        dist = diff.norm(dim=1)

        # Ihmsen formula (10): w_c_ib = max(0, (r0 - dist) / r0)
        w_c_ib = ((r0 - dist) / r0).clamp(min=0)

        # Get boundary normals (stored in velocity for boundary particles)
        n_b = boundary_normals[j_boundary]

        # Accumulate weighted normals per particle: formula (9)
        # n_c_i = sum(n_b * w_c_ib)
        n_c_i = torch.zeros((n, 3), device=self.device)
        n_c_i.scatter_add_(0, i_boundary.unsqueeze(1).expand(-1, 3), n_b * w_c_ib.unsqueeze(1))

        # Sum of weights per particle: formula (11) sum #1
        w_c_ib_sum = torch.zeros(n, device=self.device)
        w_c_ib_sum.scatter_add_(0, i_boundary, w_c_ib)

        # Sum of weighted penetration depths: formula (11) sum #2
        w_c_ib_depth_sum = torch.zeros(n, device=self.device)
        w_c_ib_depth_sum.scatter_add_(0, i_boundary, w_c_ib * (r0 - dist))

        # Only apply correction where there are boundary interactions
        has_boundary = w_c_ib_sum > 1e-12

        # Normalize the accumulated normal
        n_c_i_norm = n_c_i.norm(dim=1, keepdim=True)
        n_c_i_normalized = n_c_i / (n_c_i_norm + 1e-12)

        # Compute displacement to push particle away from boundary
        # d_i = n_c_i * w_c_ib_depth_sum / w_c_ib_sum  (formula 11)
        displacement = n_c_i_normalized * (w_c_ib_depth_sum / (w_c_ib_sum + 1e-12)).unsqueeze(1)

        # Convert displacement to force/acceleration
        # F = m * a, where displacement happens over dt
        dt = self.config["time_step"]
        sim_scale = self.config.get("simulation_scale", 1.0)

        # Boundary force as acceleration (displacement / dt^2 for position correction)
        boundary_acc = displacement * sim_scale / (dt * dt)

        # Only apply to particles with boundary interactions and that are NOT boundary particles
        is_not_boundary = sorted_types != 3
        apply_mask = has_boundary & is_not_boundary

        boundary_force[:, :3] = torch.where(
            apply_mask.unsqueeze(1),
            boundary_acc,
            torch.zeros_like(boundary_acc)
        )

        self.boundary_force = boundary_force

    def get_boundary_force(self):
        """Return boundary force for testing (Phase 3.1)."""
        return self.boundary_force

    def run_apply_boundary_correction(self, position, velocity):
        """Apply boundary position and velocity corrections in-place.

        Args:
            position: Position tensor to correct
            velocity: Velocity tensor to correct

        Returns:
            Corrected (position, velocity) tensors
        """
        n = position.shape[0]
        pos = position[:, :3]
        vel = velocity[:, :3]

        r0 = self.config.get("r0", self.config["h"] * 0.5)
        friction = self.config.get("boundary_friction", 0.0)

        # Get particle types in sorted order
        sorted_types = self.position[:, 3][self.particle_index_back].int()
        boundary_normals = self.sorted_velocity[:, :3]

        # Get neighbor pairs
        i_idx = torch.repeat_interleave(
            torch.arange(n, device=self.device),
            self.neighbor_map.shape[1],
        )
        j_idx = self.neighbor_map.reshape(-1)
        valid = j_idx >= 0
        i_idx = i_idx[valid]
        j_idx = j_idx[valid]

        # Filter to boundary neighbors
        j_types = sorted_types[j_idx]
        boundary_mask = j_types == 3
        i_boundary = i_idx[boundary_mask]
        j_boundary = j_idx[boundary_mask]

        if i_boundary.numel() == 0:
            return position, velocity

        # Compute weighted normals and sums
        diff = pos[i_boundary] - self.sorted_position[j_boundary, :3]
        dist = diff.norm(dim=1)
        w_c_ib = ((r0 - dist) / r0).clamp(min=0)

        n_b = boundary_normals[j_boundary]

        n_c_i = torch.zeros((n, 3), device=self.device)
        n_c_i.scatter_add_(0, i_boundary.unsqueeze(1).expand(-1, 3), n_b * w_c_ib.unsqueeze(1))

        w_c_ib_sum = torch.zeros(n, device=self.device)
        w_c_ib_sum.scatter_add_(0, i_boundary, w_c_ib)

        w_c_ib_depth_sum = torch.zeros(n, device=self.device)
        w_c_ib_depth_sum.scatter_add_(0, i_boundary, w_c_ib * (r0 - dist))

        has_boundary = w_c_ib_sum > 1e-12
        is_not_boundary = sorted_types != 3
        apply_mask = has_boundary & is_not_boundary

        if not apply_mask.any():
            return position, velocity

        # Normalize accumulated normal
        n_c_i_norm = n_c_i.norm(dim=1, keepdim=True)
        n_c_i_normalized = n_c_i / (n_c_i_norm + 1e-12)

        # Position correction: push out of boundary
        displacement = n_c_i_normalized * (w_c_ib_depth_sum / (w_c_ib_sum + 1e-12)).unsqueeze(1)

        position[:, :3] = torch.where(
            apply_mask.unsqueeze(1),
            pos + displacement,
            pos
        )

        # Velocity correction: remove normal component and apply friction
        vel_normal = (vel * n_c_i_normalized).sum(dim=1, keepdim=True) * n_c_i_normalized
        vel_tangent = vel - vel_normal

        # Reflect normal component and apply friction to tangent
        corrected_vel = -vel_normal + vel_tangent * (1.0 - friction)

        velocity[:, :3] = torch.where(
            apply_mask.unsqueeze(1),
            corrected_vel,
            vel
        )

        return position, velocity

    def run_apply_hard_floor(self, position, velocity):
        """Apply hard floor constraint as failsafe.

        This prevents particles from falling through the floor
        even if they skip past boundary particles.

        Args:
            position: Position tensor
            velocity: Velocity tensor

        Returns:
            Corrected (position, velocity) tensors
        """
        floor_y = self.config.get("floor_y", 0.0)
        restitution = self.config.get("floor_restitution", 0.3)

        # Get particle types
        sorted_types = self.position[:, 3][self.particle_index_back].int()
        is_fluid = (sorted_types == 1) | (sorted_types == 2)  # Liquid or elastic

        # Find particles below floor
        below_floor = (position[:, 1] < floor_y) & is_fluid

        if below_floor.any():
            # Clamp position to floor
            position[below_floor, 1] = floor_y

            # Reflect velocity (bounce) with restitution
            velocity[below_floor, 1] = -velocity[below_floor, 1] * restitution

            # If velocity is still pointing down, zero it
            still_down = (velocity[:, 1] < 0) & below_floor
            velocity[still_down, 1] = 0.0

        return position, velocity

    def run_integrate(self, mode=None):
        """Integrate positions and velocities using leapfrog method.

        Leapfrog integration (2nd order, symplectic):
        - Mode 0: Update positions: x(t+dt) = x(t) + v(t)*dt + a(t)*dt²/2
        - Mode 1: Update velocities: v(t+dt) = v(t) + (a(t) + a(t+dt)) * dt/2
        - Mode 2 (or None): Semi-implicit Euler (fallback for single-step)

        Boundary particles (type=3) are frozen and do not move.

        Args:
            mode: Integration mode (0=position, 1=velocity, 2 or None=semi-implicit Euler)
        """
        dt = self.config["time_step"]
        sim_scale_inv = self.config.get("simulation_scale_inv", 1.0)

        # Get particle types in sorted order (type is in position[:, 3])
        # sorted_position already has types in sorted order
        sorted_types = self.sorted_position[:, 3].int()
        is_boundary = sorted_types == 3

        # Zero out acceleration for boundary particles (they don't move)
        self.acceleration[is_boundary] = 0.0

        if mode == 0:
            # Mode 0: Position update only (leapfrog first half)
            # position_t_dt = position_t + (velocity_t * dt + acceleration_t * dt²/2) * sim_scale_inv
            if not hasattr(self, 'acceleration_old'):
                self.acceleration_old = self.acceleration.clone()

            pos_delta = (self.sorted_velocity[:, :3] * dt +
                         self.acceleration_old[:, :3] * dt * dt * 0.5) * sim_scale_inv

            # Freeze boundary particles (don't apply position delta)
            pos_delta[is_boundary] = 0.0
            self.sorted_position[:, :3] += pos_delta

        elif mode == 1:
            # Mode 1: Velocity update only (leapfrog second half)
            # velocity_t_dt = velocity_t + (acceleration_t + acceleration_t_dt) * dt / 2
            if not hasattr(self, 'acceleration_old'):
                self.acceleration_old = self.acceleration.clone()

            vel_delta = (self.acceleration_old[:, :3] + self.acceleration[:, :3]) * dt * 0.5

            # Freeze boundary particles
            vel_delta[is_boundary] = 0.0
            self.sorted_velocity[:, :3] += vel_delta

            # Zero velocity for boundary particles
            self.sorted_velocity[is_boundary, :3] = 0.0

            # Apply boundary corrections after velocity update
            self.sorted_position, self.sorted_velocity = self.run_apply_boundary_correction(
                self.sorted_position, self.sorted_velocity
            )

            # Apply hard floor constraint (failsafe)
            self.sorted_position, self.sorted_velocity = self.run_apply_hard_floor(
                self.sorted_position, self.sorted_velocity
            )

            # Store current acceleration for next step
            self.acceleration_old = self.acceleration.clone()

            # Unsort back to original order
            inv = torch.argsort(self.particle_index_back)
            self.position = self.sorted_position[inv]
            self.velocity = self.sorted_velocity[inv]

        else:
            # Mode 2 or None: Semi-implicit Euler (original behavior)
            # velocity_t_dt = velocity_t + acceleration_t_dt * dt
            # position_t_dt = position_t + velocity_t_dt * dt * sim_scale_inv
            vel_delta = dt * self.acceleration[:, :3]
            vel_delta[is_boundary] = 0.0
            self.sorted_velocity[:, :3] += vel_delta

            pos_delta = dt * self.sorted_velocity[:, :3] * sim_scale_inv
            pos_delta[is_boundary] = 0.0
            self.sorted_position[:, :3] += pos_delta

            # Zero velocity for boundary particles
            self.sorted_velocity[is_boundary, :3] = 0.0

            # Apply boundary corrections
            self.sorted_position, self.sorted_velocity = self.run_apply_boundary_correction(
                self.sorted_position, self.sorted_velocity
            )

            # Apply hard floor constraint (failsafe)
            self.sorted_position, self.sorted_velocity = self.run_apply_hard_floor(
                self.sorted_position, self.sorted_velocity
            )

            # Unsort back to original order
            inv = torch.argsort(self.particle_index_back)
            self.position = self.sorted_position[inv]
            self.velocity = self.sorted_velocity[inv]

    def get_integration_mode(self):
        """Return current integration mode setting (Phase 3.6)."""
        return self.config.get("integration_mode", "euler")

    def get_state(self):
        """Return position and velocity as Python lists for easy C++ access."""
        return self.position.cpu().tolist(), self.velocity.cpu().tolist()

    def get_config(self):
        """Return the solver configuration dict (Phase 1.2)."""
        return self.config

    def get_density(self):
        """Return density buffer for testing (Phase 2.1)."""
        return self.rho

    def get_pressure_acceleration(self):
        """Return pressure acceleration for testing (Phase 2.3)."""
        return self.acceleration

    # ------------------------------------------------------------------
    # Phase 2.2: PCISPH Iterative Loop Methods
    # ------------------------------------------------------------------
    def run_pcisph_predict_positions(self):
        """Predict positions for density correction (PCISPH step 1)."""
        dt = self.config["time_step"]
        # Store predicted position based on current velocity + acceleration
        self.predicted_position = (
            self.sorted_position.clone()
            + dt * self.sorted_velocity
            + 0.5 * dt * dt * self.acceleration
        )

    def run_pcisph_predict_density(self):
        """Compute density at predicted positions (PCISPH step 2)."""
        pos = self.predicted_position[:, :3]
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
        # Apply minimum clamp
        h_scaled = self.config["h"] * self.config.get("simulation_scale", 1.0)
        self.predicted_rho = torch.clamp(dens, min=h_scaled ** 6)

    def run_pcisph_correct_pressure(self):
        """Correct pressure based on density error (PCISPH step 3)."""
        # Pressure correction: delta * (predicted_density - rest_density)
        pressure_correction = self.config["delta"] * (
            self.predicted_rho - self.config["rho0"]
        )
        # Accumulate pressure correction
        self.pressure = self.pressure + pressure_correction
        # Clamp to non-negative (same as run_compute_pressure)
        self.pressure = torch.clamp(self.pressure, min=0.0)

    def run_step(self):
        """Execute a complete PCISPH simulation step with iterations.

        Includes optional logging (Phase 4.3) and timing (Phase 4.4).
        """
        # Initial neighbor search (only once per step)
        self._time_substep("hash_particles", self.run_hash_particles)
        self._time_substep("sort", self.run_sort)
        self._time_substep("index", self.run_index)
        self._time_substep("index_post_pass", self.run_index_post_pass)
        self._time_substep("find_neighbors", self.run_find_neighbors)
        self._log_substep("hash_particles")

        # Initial density and pressure
        self._time_substep("compute_density", self.run_compute_density)
        self._time_substep("compute_pressure", self.run_compute_pressure)
        self._log_substep("compute_density")

        # Initialize predicted position for PCISPH
        self.predicted_position = self.sorted_position.clone()
        self.predicted_rho = self.rho.clone()

        # PCISPH iterations (default 3)
        max_iter = self.config.get("max_iteration", 3)
        for i in range(max_iter):
            self._time_substep("pressure_force", self.run_compute_pressure_force_acceleration)
            self._time_substep("pcisph_predict_pos", self.run_pcisph_predict_positions)
            self._time_substep("pcisph_predict_density", self.run_pcisph_predict_density)
            self._time_substep("pcisph_correct_pressure", self.run_pcisph_correct_pressure)
            self._log_substep(f"pcisph_iter_{i}")

        # Final force computation and integration
        self._time_substep("pressure_force_final", self.run_compute_pressure_force_acceleration)
        self._time_substep("integrate", self.run_integrate)
        self._log_substep("integrate")

        # Finish step logging and timing
        self._finish_step_log()
        if self._timing_enabled:
            self._step_count += 1
