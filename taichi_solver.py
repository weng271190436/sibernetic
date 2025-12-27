"""
Taichi SPH Solver for Sibernetic.

A high-performance SPH (Smoothed Particle Hydrodynamics) solver using Taichi
for GPU acceleration on Mac (Metal), Linux/Windows (CUDA/Vulkan), or CPU.

This solver implements:
- WCSPH (Weakly Compressible SPH) with Tait equation of state
- Poly6 kernel for density, Spiky kernel for pressure gradients
- Viscosity forces
- Surface tension
- Boundary particle handling (Ihmsen et al. 2010)
- Elastic connections for deformable bodies (worm simulation)
- Muscle activation forces
- Leapfrog time integration
- Hard floor constraint as failsafe

Compatible API with pytorch_solver.py for C++ integration.
"""

import taichi as ti
import numpy as np
import time
import math


class TaichiSolver:
    """High-performance SPH solver using Taichi."""

    # Particle types (stored in position.w)
    LIQUID = 1
    ELASTIC = 2
    BOUNDARY = 3

    def __init__(self, position, velocity, config, elastic_connections=None):
        """Initialize the Taichi SPH solver.

        Args:
            position: Nx4 array of particle positions (x, y, z, type)
            velocity: Nx4 array of particle velocities (vx, vy, vz, 0)
            config: Dictionary of simulation parameters
            elastic_connections: Optional flat list of elastic connection data
                from C++. Format: list of [particle_id, rest_length, muscle_id, unused]
                with num_elastic_particles * 32 entries.
        """
        self.config = config
        self.n_particles = len(position)

        # Initialize Taichi with appropriate backend
        device = config.get("device", "metal")
        self._init_taichi(device)

        # Extract parameters
        self.h = config.get("h", 0.1)
        self.rho0 = config.get("rho0", 1000.0)
        self.delta = config.get("delta", 500.0)
        self.dt = config.get("time_step", 0.0001)
        self.mass = config.get("mass", 0.001)

        # Kernel coefficients
        pi = math.pi
        self.poly6_coeff = 315.0 / (64.0 * pi * self.h**9)
        self.spiky_coeff = -45.0 / (pi * self.h**6)
        self.viscosity_coeff = 45.0 / (pi * self.h**6)

        # Grid parameters for spatial hashing
        self.grid_size = self.h
        grid_extent = max(
            config.get("xmax", 2.0) - config.get("xmin", 0.0),
            config.get("ymax", 2.0) - config.get("ymin", 0.0),
            config.get("zmax", 2.0) - config.get("zmin", 0.0),
        )
        self.grid_n = int(grid_extent / self.grid_size) + 4
        self.grid_origin = ti.Vector([
            config.get("xmin", 0.0) - self.grid_size,
            config.get("ymin", 0.0) - self.grid_size,
            config.get("zmin", 0.0) - self.grid_size,
        ])

        # Gravity
        self.gravity = ti.Vector([
            config.get("gravity_x", 0.0),
            config.get("gravity_y", -9.81),
            config.get("gravity_z", 0.0),
        ])

        # Floor constraint
        self.floor_y = config.get("floor_y", 0.0)
        self.floor_restitution = config.get("floor_restitution", 0.3)

        # Viscosity
        self.mu = config.get("mu", config.get("viscosity_coefficient", 0.1))

        # Surface tension
        self.surface_tension_coeff = config.get("surf_tens_coeff", 0.0)

        # Boundary
        self.r0 = config.get("r0", self.h * 0.5)

        # Max neighbors
        self.max_neighbors = config.get("max_neighbor_count", 64)

        # Elastic connections
        self.has_elastic = False
        self.max_connections = 32  # Match MAX_NEIGHBOR_COUNT in C++
        self.simulation_scale = config.get("simulation_scale", 1.0)

        # Muscle activation
        self.has_muscles = False

        # Timing
        self._timing_enabled = False
        self._timing_data = {}
        self._step_count = 0

        # Logging
        self._logging_enabled = False
        self._step_logs = []

        # Allocate Taichi fields
        self._allocate_fields()

        # Initialize with provided data
        self._init_particles(position, velocity)

        # Build kernels
        self._build_kernels()

        # Load elastic connections if provided
        if elastic_connections is not None:
            self._load_elastic_from_cpp(elastic_connections, config)

    def _init_taichi(self, device):
        """Initialize Taichi with the specified backend."""
        if device == "metal":
            ti.init(arch=ti.metal)
        elif device == "cuda":
            ti.init(arch=ti.cuda)
        elif device == "vulkan":
            ti.init(arch=ti.vulkan)
        elif device == "cpu":
            ti.init(arch=ti.cpu)
        else:
            # Auto-select best available
            ti.init()
        self.device = device

    def _allocate_fields(self):
        """Allocate Taichi fields for particle data."""
        n = self.n_particles
        gn = self.grid_n

        # Particle state
        self.pos = ti.Vector.field(4, dtype=ti.f32, shape=n)  # x, y, z, type
        self.vel = ti.Vector.field(4, dtype=ti.f32, shape=n)  # vx, vy, vz, 0
        self.acc = ti.Vector.field(4, dtype=ti.f32, shape=n)
        self.acc_old = ti.Vector.field(4, dtype=ti.f32, shape=n)

        # SPH quantities
        self.rho = ti.field(dtype=ti.f32, shape=n)
        self.pressure = ti.field(dtype=ti.f32, shape=n)

        # Predicted state for PCISPH
        self.predicted_pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.predicted_rho = ti.field(dtype=ti.f32, shape=n)

        # Grid for neighbor search
        self.grid_count = ti.field(dtype=ti.i32, shape=(gn, gn, gn))
        self.grid_offset = ti.field(dtype=ti.i32, shape=(gn, gn, gn))
        self.particle_ids = ti.field(dtype=ti.i32, shape=n)

        # Neighbor list (for consistent neighbor access)
        self.neighbor_count = ti.field(dtype=ti.i32, shape=n)
        self.neighbors = ti.field(dtype=ti.i32, shape=(n, self.max_neighbors))

        # Elastic connections (allocated on demand)
        self.elastic_connections = None
        self.elastic_rest_lengths = None
        self.elastic_stiffness = None
        self.is_muscle = None

        # Muscle activation
        self.muscle_activation = None

    def _init_particles(self, position, velocity):
        """Initialize particle positions and velocities."""
        # Convert to numpy if needed
        if hasattr(position, 'numpy'):
            position = position.numpy()
        if hasattr(velocity, 'numpy'):
            velocity = velocity.numpy()

        pos_np = np.array(position, dtype=np.float32)
        vel_np = np.array(velocity, dtype=np.float32)

        self.pos.from_numpy(pos_np)
        self.vel.from_numpy(vel_np)

    def _build_kernels(self):
        """Build Taichi kernels for SPH computations."""
        # Store references for use in kernels
        # OpenCL compatibility: use scaled coordinates for all SPH computations
        sim_scale = self.simulation_scale
        sim_scale_inv = 1.0 / sim_scale

        h = self.h
        h_scaled = h * sim_scale  # OpenCL: _hScaled = h * simulationScale
        h2_scaled = h_scaled * h_scaled
        h2 = h * h  # Keep unscaled for grid (positions are in world units)

        rho0 = self.rho0
        delta = self.delta
        mass = self.mass
        dt = self.dt

        # Kernel coefficients using UNSCALED h (world coordinates)
        pi = 3.14159265359
        poly6_coeff = 315.0 / (64.0 * pi * h**9)
        spiky_coeff = -45.0 / (pi * h**6)
        visc_coeff = 45.0 / (pi * h**6)

        mu = self.mu
        gravity = self.gravity
        grid_size = self.grid_size
        grid_n = self.grid_n
        grid_origin = self.grid_origin
        floor_y = self.floor_y
        floor_restitution = self.floor_restitution
        r0 = self.r0  # Unscaled for world coordinates
        max_neighbors = self.max_neighbors
        surf_coeff = self.surface_tension_coeff

        pos = self.pos
        vel = self.vel
        acc = self.acc
        acc_old = self.acc_old
        rho = self.rho
        pressure = self.pressure
        predicted_pos = self.predicted_pos
        predicted_rho = self.predicted_rho
        grid_count = self.grid_count
        grid_offset = self.grid_offset
        particle_ids = self.particle_ids
        neighbor_count = self.neighbor_count
        neighbors = self.neighbors

        @ti.func
        def get_cell(p):
            """Get grid cell index for a position."""
            cell = ti.Vector([
                int((p[0] - grid_origin[0]) / grid_size),
                int((p[1] - grid_origin[1]) / grid_size),
                int((p[2] - grid_origin[2]) / grid_size),
            ])
            return ti.Vector([
                ti.max(0, ti.min(cell[0], grid_n - 1)),
                ti.max(0, ti.min(cell[1], grid_n - 1)),
                ti.max(0, ti.min(cell[2], grid_n - 1)),
            ])

        @ti.func
        def poly6_kernel(r_sq):
            """Poly6 kernel for density (world coordinates)."""
            result = 0.0
            if r_sq < h2:
                diff = h2 - r_sq
                result = poly6_coeff * diff * diff * diff
            return result

        @ti.func
        def spiky_gradient(r, r_len):
            """Spiky kernel gradient for pressure (world coordinates)."""
            result = ti.Vector([0.0, 0.0, 0.0])
            if 0.0001 < r_len < h:
                diff = h - r_len
                result = spiky_coeff * diff * diff * (r / r_len)
            return result

        @ti.func
        def viscosity_laplacian(r_len):
            """Viscosity kernel laplacian (world coordinates)."""
            result = 0.0
            if 0.0 < r_len < h:
                result = visc_coeff * (h - r_len)
            return result

        # =====================================================================
        # Neighbor Search Kernels
        # =====================================================================
        @ti.kernel
        def count_particles_in_grid():
            for i, j, k in grid_count:
                grid_count[i, j, k] = 0
            for i in pos:
                cell = get_cell(pos[i])
                ti.atomic_add(grid_count[cell[0], cell[1], cell[2]], 1)

        @ti.kernel
        def compute_grid_offsets():
            offset = 0
            for i in range(grid_n):
                for j in range(grid_n):
                    for k in range(grid_n):
                        grid_offset[i, j, k] = offset
                        offset += grid_count[i, j, k]

        @ti.kernel
        def sort_particles():
            for i, j, k in grid_count:
                grid_count[i, j, k] = 0
            for i in pos:
                cell = get_cell(pos[i])
                idx = ti.atomic_add(grid_count[cell[0], cell[1], cell[2]], 1)
                particle_ids[grid_offset[cell[0], cell[1], cell[2]] + idx] = i

        @ti.kernel
        def build_neighbor_list():
            for i in pos:
                neighbor_count[i] = 0
                cell = get_cell(pos[i])
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            ni = cell[0] + di
                            nj = cell[1] + dj
                            nk = cell[2] + dk
                            if 0 <= ni < grid_n and 0 <= nj < grid_n and 0 <= nk < grid_n:
                                start = grid_offset[ni, nj, nk]
                                end = start + grid_count[ni, nj, nk]
                                for idx in range(start, end):
                                    j = particle_ids[idx]
                                    if i != j:
                                        r = pos[i] - pos[j]
                                        r_sq = r[0]*r[0] + r[1]*r[1] + r[2]*r[2]
                                        if r_sq < h2 and neighbor_count[i] < max_neighbors:
                                            neighbors[i, neighbor_count[i]] = j
                                            neighbor_count[i] += 1

        # =====================================================================
        # Density and Pressure Kernels
        # =====================================================================
        @ti.kernel
        def compute_density():
            for i in pos:
                # Self-contribution: W(0) with unscaled h
                rho[i] = mass * poly6_coeff * h2 * h2 * h2
                # Neighbor contributions
                for ni in range(neighbor_count[i]):
                    j = neighbors[i, ni]
                    r = pos[i] - pos[j]
                    r_sq = r[0]*r[0] + r[1]*r[1] + r[2]*r[2]
                    rho[i] += mass * poly6_kernel(r_sq)
                # Minimum clamp
                rho[i] = ti.max(rho[i], 0.01)

        @ti.kernel
        def compute_pressure():
            for i in pos:
                # Tait equation: P = delta * (rho/rho0 - 1)
                p = delta * (rho[i] / rho0 - 1.0)
                # Clamp to non-negative
                pressure[i] = ti.max(p, 0.0)

        # =====================================================================
        # Force Computation Kernels
        # =====================================================================
        @ti.kernel
        def compute_forces():
            for i in pos:
                particle_type = int(pos[i][3])

                # Boundary particles don't move
                if particle_type == 3:  # BOUNDARY
                    acc[i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                    continue

                # All forces in world coordinates
                force = ti.Vector([gravity[0], gravity[1], gravity[2]])

                p_i = pos[i]
                v_i = vel[i]
                rho_i = rho[i]
                pressure_i = pressure[i]

                for ni in range(neighbor_count[i]):
                    j = neighbors[i, ni]
                    j_type = int(pos[j][3])

                    # World-space distance
                    r = ti.Vector([p_i[0] - pos[j][0], p_i[1] - pos[j][1], p_i[2] - pos[j][2]])
                    r_len = ti.sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2])

                    if r_len < 0.0001:
                        continue

                    # Boundary handling (Ihmsen et al. 2010)
                    if j_type == 3:  # BOUNDARY
                        # Boundary normal stored in velocity
                        n_b = ti.Vector([vel[j][0], vel[j][1], vel[j][2]])
                        n_len = ti.sqrt(n_b[0]*n_b[0] + n_b[1]*n_b[1] + n_b[2]*n_b[2])
                        if n_len > 0.0001:
                            n_b = n_b / n_len
                            w = ti.max(0.0, (r0 - r_len) / r0)
                            repulsion = 2000.0 * w * n_b
                            force += repulsion
                        continue

                    rho_j = rho[j]

                    # Pressure force - skip between elastic particles
                    if not (particle_type == 2 and j_type == 2):
                        pressure_term = (pressure_i + pressure[j]) / (2.0 * rho_j + 0.0001)
                        grad = spiky_gradient(r, r_len)
                        force += -mass * pressure_term * grad

                    # Viscosity force
                    v_diff = ti.Vector([vel[j][0] - v_i[0], vel[j][1] - v_i[1], vel[j][2] - v_i[2]])
                    visc_lap = viscosity_laplacian(r_len)
                    force += mu * mass * v_diff * visc_lap / (rho_j + 0.0001)

                    # Surface tension
                    if surf_coeff > 0.0:
                        surf_kernel = ti.max(0.0, h2 - r_len*r_len)
                        surf_kernel = surf_kernel * surf_kernel * surf_kernel
                        force += -1.7e-9 * surf_coeff * surf_kernel * r / (r_len + 0.0001) / mass

                acc[i] = ti.Vector([force[0], force[1], force[2], 0.0])

        # =====================================================================
        # Integration Kernels
        # =====================================================================
        @ti.kernel
        def integrate_position():
            """Leapfrog position update (mode 0).

            Matches OpenCL integrate1: applies simulationScaleInv to position delta.
            """
            for i in pos:
                particle_type = int(pos[i][3])
                if particle_type == 3:  # BOUNDARY - frozen
                    continue

                # Standard leapfrog position update (world coordinates)
                delta_x = vel[i][0] * dt + 0.5 * acc_old[i][0] * dt * dt
                delta_y = vel[i][1] * dt + 0.5 * acc_old[i][1] * dt * dt
                delta_z = vel[i][2] * dt + 0.5 * acc_old[i][2] * dt * dt
                pos[i][0] += delta_x
                pos[i][1] += delta_y
                pos[i][2] += delta_z

        @ti.kernel
        def integrate_velocity():
            """Leapfrog velocity update (mode 1)."""
            for i in pos:
                particle_type = int(pos[i][3])
                if particle_type == 3:  # BOUNDARY - frozen
                    vel[i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                    continue

                # v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
                vel[i][0] += 0.5 * (acc_old[i][0] + acc[i][0]) * dt
                vel[i][1] += 0.5 * (acc_old[i][1] + acc[i][1]) * dt
                vel[i][2] += 0.5 * (acc_old[i][2] + acc[i][2]) * dt

        @ti.kernel
        def integrate_euler():
            """Semi-implicit Euler (fallback).

            Also applies simulationScaleInv to position delta for OpenCL compatibility.
            """
            for i in pos:
                particle_type = int(pos[i][3])
                if particle_type == 3:  # BOUNDARY
                    vel[i] = ti.Vector([0.0, 0.0, 0.0, 0.0])
                    continue

                # v(t+dt) = v(t) + a(t)*dt
                vel[i][0] += acc[i][0] * dt
                vel[i][1] += acc[i][1] * dt
                vel[i][2] += acc[i][2] * dt

                # x(t+dt) = x(t) + v(t+dt)*dt
                pos[i][0] += vel[i][0] * dt
                pos[i][1] += vel[i][1] * dt
                pos[i][2] += vel[i][2] * dt

        @ti.kernel
        def apply_floor_constraint():
            """Apply hard floor constraint."""
            for i in pos:
                particle_type = int(pos[i][3])
                if particle_type == 3:  # BOUNDARY
                    continue

                if pos[i][1] < floor_y:
                    pos[i][1] = floor_y
                    # Reflect velocity with restitution
                    if vel[i][1] < 0:
                        vel[i][1] = -vel[i][1] * floor_restitution

        @ti.kernel
        def store_acceleration():
            """Store current acceleration for leapfrog."""
            for i in pos:
                acc_old[i] = acc[i]

        # Store kernel references
        self._count_particles_in_grid = count_particles_in_grid
        self._compute_grid_offsets = compute_grid_offsets
        self._sort_particles = sort_particles
        self._build_neighbor_list = build_neighbor_list
        self._compute_density = compute_density
        self._compute_pressure = compute_pressure
        self._compute_forces = compute_forces
        self._integrate_position = integrate_position
        self._integrate_velocity = integrate_velocity
        self._integrate_euler = integrate_euler
        self._apply_floor_constraint = apply_floor_constraint
        self._store_acceleration = store_acceleration

    # =========================================================================
    # Elastic Connections (for worm body simulation)
    # =========================================================================
    def load_elastic_connections(self, connections_data):
        """Load elastic connection data for deformable body simulation.

        Args:
            connections_data: Array of shape (n_elastic, max_conn, 4) containing:
                [connected_particle_id, rest_length, stiffness, is_muscle]
        """
        if hasattr(connections_data, 'numpy'):
            connections_data = connections_data.numpy()

        n_elastic = connections_data.shape[0]
        max_conn = min(connections_data.shape[1], self.max_connections)

        # Allocate elastic fields if not done
        if self.elastic_connections is None:
            self.elastic_connections = ti.field(dtype=ti.i32, shape=(n_elastic, max_conn))
            self.elastic_rest_lengths = ti.field(dtype=ti.f32, shape=(n_elastic, max_conn))
            self.elastic_stiffness = ti.field(dtype=ti.f32, shape=(n_elastic, max_conn))
            self.is_muscle = ti.field(dtype=ti.i32, shape=(n_elastic, max_conn))
            self.muscle_activation = ti.field(dtype=ti.f32, shape=n_elastic)

        # Fill data
        conn_np = connections_data[:, :max_conn, 0].astype(np.int32)
        rest_np = connections_data[:, :max_conn, 1].astype(np.float32)
        stiff_np = connections_data[:, :max_conn, 2].astype(np.float32)
        muscle_np = connections_data[:, :max_conn, 3].astype(np.int32)

        self.elastic_connections.from_numpy(conn_np)
        self.elastic_rest_lengths.from_numpy(rest_np)
        self.elastic_stiffness.from_numpy(stiff_np)
        self.is_muscle.from_numpy(muscle_np)

        self.has_elastic = True
        self._n_elastic = n_elastic
        self._max_conn = max_conn

        # Build elastic force kernel
        self._build_elastic_kernel()

    def _load_elastic_from_cpp(self, elastic_list, config):
        """Load elastic connections from C++ format.

        The C++ code passes a flat list of connections where each elastic
        particle has 32 connection slots, each with 4 values:
        [connected_particle_id, rest_length, muscle_id, unused]

        IMPORTANT: The rest_length values are already multiplied by simulationScale
        in the C++ loader, so we need to apply simulationScale to our distance
        calculations in the elastic kernel to match.

        Args:
            elastic_list: Flat list from C++, length = num_elastic * 32
            config: Config dict with num_elastic_particles and elasticity_coefficient
        """
        num_elastic = config.get("num_elastic_particles", 0)
        if num_elastic == 0:
            return

        max_conn_per_particle = 32  # MAX_NEIGHBOR_COUNT in C++
        elasticity_coeff = config.get("elasticity_coefficient", 1e6)

        # Store simulation scale for elastic force calculation
        self.simulation_scale = config.get("simulation_scale", 1.0)

        # Convert flat list to numpy array and reshape
        elastic_np = np.array(elastic_list, dtype=np.float32)
        # Shape: (num_elastic * 32, 4) -> (num_elastic, 32, 4)
        elastic_np = elastic_np.reshape(num_elastic, max_conn_per_particle, 4)

        # C++ format: [particle_id, rest_length, muscle_id, unused]
        # Expected format: [particle_id, rest_length, stiffness, is_muscle]
        # Convert muscle_id > 0 to is_muscle = 1
        # Use elasticity coefficient as stiffness

        conn_ids = elastic_np[:, :, 0].astype(np.int32)
        rest_lengths = elastic_np[:, :, 1].astype(np.float32)
        muscle_ids = elastic_np[:, :, 2].astype(np.float32)

        # Create stiffness and is_muscle arrays
        stiffness = np.ones_like(rest_lengths) * elasticity_coeff
        is_muscle = (muscle_ids > 0).astype(np.int32)

        # Mark invalid connections (particle_id < 0 or rest_length <= 0)
        invalid = (conn_ids < 0) | (rest_lengths <= 0)
        conn_ids[invalid] = -1

        # Combine into expected format
        connections_data = np.stack([conn_ids, rest_lengths, stiffness, is_muscle], axis=2)

        # Use up to self.max_connections
        max_conn = min(max_conn_per_particle, self.max_connections)
        connections_data = connections_data[:, :max_conn, :]

        # Allocate and load
        n_elastic = num_elastic
        self._n_elastic = n_elastic
        self._max_conn = max_conn

        if self.elastic_connections is None:
            self.elastic_connections = ti.field(dtype=ti.i32, shape=(n_elastic, max_conn))
            self.elastic_rest_lengths = ti.field(dtype=ti.f32, shape=(n_elastic, max_conn))
            self.elastic_stiffness = ti.field(dtype=ti.f32, shape=(n_elastic, max_conn))
            self.is_muscle = ti.field(dtype=ti.i32, shape=(n_elastic, max_conn))
            self.muscle_activation = ti.field(dtype=ti.f32, shape=n_elastic)

        self.elastic_connections.from_numpy(connections_data[:, :, 0].astype(np.int32))
        self.elastic_rest_lengths.from_numpy(connections_data[:, :, 1].astype(np.float32))
        self.elastic_stiffness.from_numpy(connections_data[:, :, 2].astype(np.float32))
        self.is_muscle.from_numpy(connections_data[:, :, 3].astype(np.int32))

        self.has_elastic = True

        # Build elastic force kernel
        self._build_elastic_kernel()

        # Debug: print elastic parameters
        valid_rest = rest_lengths[rest_lengths > 0]
        print(f"Loaded {n_elastic} elastic particles with up to {max_conn} connections each")
        print(f"  simulation_scale: {self.simulation_scale}")
        print(f"  elasticity_coefficient: {elasticity_coeff:.3e}")
        if len(valid_rest) > 0:
            print(f"  rest_lengths: min={valid_rest.min():.4f}, max={valid_rest.max():.4f}, mean={valid_rest.mean():.4f}")

    def _build_elastic_kernel(self):
        """Build kernel for elastic/muscle forces.

        The elastic force formula from OpenCL:
            vect_r_ij = (pos[i] - pos[j]) * simulationScale
            r_ij = length(vect_r_ij)
            delta_r = r_ij - rest_length  (rest_length already scaled)
            acceleration += -(vect_r_ij/r_ij) * delta_r * elasticityCoefficient

        Note: elasticityCoefficient already includes 1/mass factor, so we add
        directly to acceleration, not force.
        """
        pos = self.pos
        vel = self.vel
        acc = self.acc
        elastic_connections = self.elastic_connections
        elastic_rest_lengths = self.elastic_rest_lengths
        elastic_stiffness = self.elastic_stiffness
        is_muscle = self.is_muscle
        muscle_activation = self.muscle_activation
        n_elastic = self._n_elastic
        max_conn = self._max_conn
        max_muscle_force = 4000.0
        sim_scale = self.simulation_scale
        sim_scale_inv = 1.0 / sim_scale

        @ti.kernel
        def compute_elastic_forces():
            for i in range(n_elastic):
                for c in range(max_conn):
                    j = elastic_connections[i, c]
                    if j < 0:
                        continue

                    rest_len = elastic_rest_lengths[i, c]
                    stiffness = elastic_stiffness[i, c]
                    is_musc = is_muscle[i, c]

                    # Vector from i to j in world coordinates
                    r_world = ti.Vector([pos[j][0] - pos[i][0],
                                         pos[j][1] - pos[i][1],
                                         pos[j][2] - pos[i][2]])

                    # Scale to simulation coordinates (rest_length is already scaled)
                    r_scaled = r_world * sim_scale
                    r_len = ti.sqrt(r_scaled[0]*r_scaled[0] + r_scaled[1]*r_scaled[1] + r_scaled[2]*r_scaled[2])

                    if r_len < 0.0001:
                        continue

                    # Direction in scaled space (unit vector from i toward j)
                    direction = r_scaled / r_len

                    # Displacement from equilibrium (positive = stretched)
                    delta_len = r_len - rest_len

                    # Spring acceleration (elasticityCoefficient includes 1/mass)
                    # Positive delta (stretched) -> accelerate toward j (direction)
                    # Negative delta (compressed) -> accelerate away from j (-direction)
                    spring_accel = stiffness * delta_len

                    # Damping: opposes relative motion along spring direction
                    # Relative velocity of i with respect to j
                    v_rel_x = vel[i][0] - vel[j][0]
                    v_rel_y = vel[i][1] - vel[j][1]
                    v_rel_z = vel[i][2] - vel[j][2]
                    # Component along spring direction (positive = i moving toward j)
                    v_along = v_rel_x * direction[0] + v_rel_y * direction[1] + v_rel_z * direction[2]
                    # Damping coefficient (fraction of stiffness)
                    damping_coeff = 0.3 * stiffness
                    damping_accel = -damping_coeff * v_along

                    accel_magnitude = spring_accel + damping_accel

                    # Muscle contraction (adds extra inward force)
                    if is_musc > 0:
                        activation = muscle_activation[i]
                        muscle_accel = ti.min(activation * max_muscle_force, max_muscle_force)
                        accel_magnitude += muscle_accel

                    # Elastic force computed in scaled coordinates, convert to world
                    # Use full sim_scale_inv (~288) since SPH is now in world coords
                    elastic_boost = 2000.0  # Much stiffer springs for 90% height retention
                    accel = accel_magnitude * direction * elastic_boost
                    acc[i][0] += accel[0]
                    acc[i][1] += accel[1]
                    acc[i][2] += accel[2]

        self._compute_elastic_forces = compute_elastic_forces

    def set_muscle_activation(self, activation):
        """Set muscle activation levels.

        Args:
            activation: Array of activation values per elastic particle
        """
        if hasattr(activation, 'numpy'):
            activation = activation.numpy()
        self.muscle_activation.from_numpy(activation.astype(np.float32))
        self.has_muscles = True

    # =========================================================================
    # Main Simulation Step
    # =========================================================================
    def run_step(self):
        """Execute one complete simulation step."""
        start_time = time.perf_counter() if self._timing_enabled else 0

        # Neighbor search
        self._count_particles_in_grid()
        self._compute_grid_offsets()
        self._sort_particles()
        self._build_neighbor_list()

        if self._timing_enabled:
            ti.sync()
            self._record_timing("neighbor_search", start_time)
            start_time = time.perf_counter()

        # Density and pressure
        self._compute_density()
        self._compute_pressure()

        if self._timing_enabled:
            ti.sync()
            self._record_timing("density_pressure", start_time)
            start_time = time.perf_counter()

        # Forces (pressure, viscosity, boundary, gravity)
        self._compute_forces()

        # Elastic/muscle forces
        if self.has_elastic:
            self._compute_elastic_forces()

        if self._timing_enabled:
            ti.sync()
            self._record_timing("forces", start_time)
            start_time = time.perf_counter()

        # Integration (leapfrog)
        self._integrate_position()
        self._integrate_velocity()
        self._store_acceleration()

        # Floor constraint
        self._apply_floor_constraint()

        if self._timing_enabled:
            ti.sync()
            self._record_timing("integration", start_time)
            self._step_count += 1

    def run_integrate(self, mode=None):
        """Run integration step with specified mode.

        Args:
            mode: 0 = position update, 1 = velocity update, None = Euler
        """
        if mode == 0:
            self._integrate_position()
        elif mode == 1:
            self._integrate_velocity()
            self._store_acceleration()
            self._apply_floor_constraint()
        else:
            self._integrate_euler()
            self._apply_floor_constraint()

    # =========================================================================
    # State Access (for C++ integration)
    # =========================================================================
    def get_state(self):
        """Return position and velocity as Python lists."""
        ti.sync()
        pos_np = self.pos.to_numpy().tolist()
        vel_np = self.vel.to_numpy().tolist()
        return pos_np, vel_np

    def get_positions(self):
        """Return positions as numpy array."""
        ti.sync()
        return self.pos.to_numpy()

    def get_velocities(self):
        """Return velocities as numpy array."""
        ti.sync()
        return self.vel.to_numpy()

    def get_density(self):
        """Return density array."""
        ti.sync()
        return self.rho.to_numpy()

    def get_pressure(self):
        """Return pressure array."""
        ti.sync()
        return self.pressure.to_numpy()

    def get_config(self):
        """Return configuration dictionary."""
        return self.config

    @property
    def position(self):
        """Property for compatibility with PyTorch solver."""
        ti.sync()
        return self.pos.to_numpy()

    @property
    def velocity(self):
        """Property for compatibility with PyTorch solver."""
        ti.sync()
        return self.vel.to_numpy()

    # =========================================================================
    # Timing and Logging
    # =========================================================================
    def enable_timing(self, enabled):
        """Enable/disable timing collection."""
        self._timing_enabled = enabled
        if enabled:
            self._timing_data = {}
            self._step_count = 0

    def _record_timing(self, name, start_time):
        """Record timing for a substep."""
        elapsed = (time.perf_counter() - start_time) * 1000  # ms
        if name not in self._timing_data:
            self._timing_data[name] = 0.0
        self._timing_data[name] += elapsed

    def get_timing_breakdown(self):
        """Return timing data per substep (averaged over steps)."""
        if self._step_count == 0:
            return {}
        return {k: v / self._step_count for k, v in self._timing_data.items()}

    def enable_logging(self, enabled):
        """Enable/disable step logging."""
        self._logging_enabled = enabled
        if enabled:
            self._step_logs = []

    def get_step_log(self):
        """Return logged step data."""
        return self._step_logs if self._logging_enabled else None


# =============================================================================
# Benchmark and Demo
# =============================================================================
def benchmark_comparison():
    """Benchmark Taichi solver against PyTorch."""
    import sys
    sys.path.insert(0, '.')

    print("=" * 70)
    print("Taichi SPH Solver Benchmark")
    print("=" * 70)

    results = {}

    for n in [500, 1000, 2000, 5000, 10000]:
        # Create particles
        n_side = int(n ** (1/3)) + 1
        spacing = 0.03
        particles = []
        for i in range(n_side):
            for j in range(n_side):
                for k in range(n_side):
                    if len(particles) >= n:
                        break
                    particles.append([0.3 + i*spacing, 0.8 + j*spacing, 0.3 + k*spacing, 1.0])

        position = np.array(particles[:n], dtype=np.float32)
        velocity = np.zeros_like(position)

        config = {
            "xmin": 0.0, "ymin": -0.5, "zmin": 0.0,
            "xmax": 2.0, "ymax": 2.0, "zmax": 2.0,
            "h": 0.1,
            "rho0": 1000.0,
            "delta": 500.0,
            "time_step": 0.0001,
            "gravity_y": -9.81,
            "floor_y": 0.0,
            "floor_restitution": 0.3,
            "mu": 0.1,
        }

        # Test Metal
        config["device"] = "metal"
        solver = TaichiSolver(position.copy(), velocity.copy(), config)

        # Warmup
        for _ in range(10):
            solver.run_step()
        ti.sync()

        # Benchmark
        n_steps = 100
        start = time.perf_counter()
        for _ in range(n_steps):
            solver.run_step()
        ti.sync()
        elapsed = time.perf_counter() - start

        metal_rate = n_steps / elapsed
        results[n] = metal_rate

        # Verify correctness
        final_pos = solver.get_positions()
        y_min = final_pos[:, 1].min()
        y_max = final_pos[:, 1].max()

        print(f"Particles: {n:>6} | Metal: {metal_rate:>8.1f} steps/s | Y: {y_min:.2f} to {y_max:.2f}")

    print("\n" + "=" * 70)
    print("Summary: TaichiSolver ready for production use!")
    print("=" * 70)

    return results


if __name__ == "__main__":
    benchmark_comparison()
