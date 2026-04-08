/*******************************************************************************
 * The MIT License (MIT)
 * Copyright (c) 2011, 2013 OpenWorm.
 * Metal port (c) 2026
 *
 * Sibernetic SPH Fluid Simulation - Metal Compute Shaders
 * Ported from OpenCL (sphFluid.cl)
 *
 * References:
 * [1] PCISPH - Solenthaler
 * [2] Particle-based fluid simulation, Muller et al., 2003
 * [3] Boundary handling - Ihmsen et al., 2010
 *******************************************************************************/

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants and Defines
// ============================================================================

constant int MAX_NEIGHBOR_COUNT = 32;

constant int BOUNDARY_PARTICLE = 3;

constant int NO_PARTICLE_ID = -1;
constant float NO_DISTANCE = -1.0f;

// ============================================================================
// Simulation Parameters (passed as buffer)
// ============================================================================

struct SimulationParams {
    float h;                    // Smoothing radius (world scale for grid)
    float hScaled;              // h * simulationScale (for SPH kernels)
    float hScaled2;             // hScaled^2
    float mass;                 // Particle mass
    float simulationScale;      // Scale factor
    float simulationScaleInv;   // 1 / simulationScale (for position updates)
    float timeStep;             // dt
    float viscosity;            // Viscosity coefficient
    float surfaceTension;       // Surface tension coefficient
    float gravity;              // Gravity acceleration
    float r0;                   // Equilibrium distance = 0.5 * h
    
    // Pre-computed coefficients matching OpenCL
    float mass_mult_Wpoly6Coefficient;
    float mass_mult_gradWspikyCoefficient;
    float mass_mult_divgradWviscosityCoefficient;
    float surfTensCoeff;
    
    uint particleCount;         // Total particles
    uint gridCellCount;         // Grid cells
    float gridMinX, gridMinY, gridMinZ;  // Grid bounds min
    float gridMaxX, gridMaxY, gridMaxZ;  // Grid bounds max
    int gridResX, gridResY, gridResZ;    // Grid resolution
    float cellSize;             // Size of each grid cell
    
    // PCISPH parameters
    float rho0;                 // Rest density
    float delta;                // PCISPH delta
    int pcisphIterations;       // Pressure correction iterations
};

// ============================================================================
// Helper Functions
// ============================================================================

// SPH Poly6 kernel (for density)
inline float Wpoly6(float r, float h) {
    if (r > h) return 0.0f;
    float h2 = h * h;
    float h9 = h2 * h2 * h2 * h2 * h;
    float coeff = 315.0f / (64.0f * M_PI_F * h9);
    float diff = h2 - r * r;
    return coeff * diff * diff * diff;
}

// SPH Spiky kernel gradient (for pressure)
inline float3 gradWspiky(float3 r_vec, float r, float h) {
    if (r > h || r < 1e-8f) return float3(0.0f);
    float h6 = h * h * h * h * h * h;
    float coeff = -45.0f / (M_PI_F * h6);
    float diff = h - r;
    return coeff * diff * diff * (r_vec / r);
}

// SPH Viscosity kernel Laplacian
inline float laplacianWviscosity(float r, float h) {
    if (r > h) return 0.0f;
    float h6 = h * h * h * h * h * h;
    float coeff = 45.0f / (M_PI_F * h6);
    return coeff * (h - r);
}

// Grid cell index from position
inline int getCellIndex(float3 pos, float3 gridMin, float cellSize, int3 gridRes) {
    int3 cell = int3((pos - gridMin) / cellSize);
    cell = clamp(cell, int3(0), gridRes - 1);
    return cell.x + cell.y * gridRes.x + cell.z * gridRes.x * gridRes.y;
}

// Helper to get grid min as float3
inline float3 getGridMin(constant SimulationParams& p) {
    return float3(p.gridMinX, p.gridMinY, p.gridMinZ);
}

inline float3 getGridMax(constant SimulationParams& p) {
    return float3(p.gridMaxX, p.gridMaxY, p.gridMaxZ);
}

inline int3 getGridRes(constant SimulationParams& p) {
    return int3(p.gridResX, p.gridResY, p.gridResZ);
}

// ============================================================================
// Kernel: Clear Buffers
// ============================================================================

kernel void clearBuffers(
    device float2* neighborMap [[buffer(0)]],
    constant uint& particleCount [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= particleCount * MAX_NEIGHBOR_COUNT) return;
    neighborMap[id] = float2(NO_PARTICLE_ID, NO_DISTANCE);
}

// ============================================================================
// Kernel: Hash Particles to Grid
// ============================================================================

kernel void hashParticles(
    device float4* position [[buffer(0)]],
    device uint2* particleIndex [[buffer(1)]],  // x = cellId, y = particleId
    constant SimulationParams& params [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.particleCount) return;
    
    float3 pos = position[id].xyz;
    int cellIndex = getCellIndex(pos, getGridMin(params), params.cellSize, getGridRes(params));
    
    // Store in particleIndex buffer (NOT position.w which has particle type)
    particleIndex[id].x = (uint)cellIndex;
    particleIndex[id].y = id;
}

// ============================================================================
// Kernel: Compute Density (PCISPH)
// Matches OpenCL pcisph_computeDensity: uses pre-scaled distances and (hScaled²-r²)³
// ============================================================================

kernel void pcisph_computeDensity(
    device float4* position [[buffer(0)]],
    device float4* velocity [[buffer(1)]],
    device float2* rhoInv [[buffer(2)]],           // x = rho, y = 1/rho
    device int* neighborMap [[buffer(3)]],
    device int* neighborCount [[buffer(4)]],
    constant SimulationParams& params [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.particleCount) return;
    
    float3 pos_i = position[id].xyz;
    float hScaled2 = params.hScaled2;
    float hScaled6 = hScaled2 * hScaled2 * hScaled2;
    float simScale = params.simulationScale;
    
    float density = 0.0f;
    
    int count = neighborCount[id];
    for (int j = 0; j < count && j < MAX_NEIGHBOR_COUNT; j++) {
        int neighborIdx = neighborMap[id * MAX_NEIGHBOR_COUNT + j];
        if (neighborIdx == NO_PARTICLE_ID || neighborIdx < 0) continue;
        
        float3 pos_j = position[neighborIdx].xyz;
        float r = length(pos_i - pos_j) * simScale;  // scaled distance
        float r2 = r * r;
        
        if (r2 < hScaled2) {
            float diff = hScaled2 - r2;
            density += diff * diff * diff;  // (hScaled² - r²)³
        }
    }
    
    // Clamp minimum density (matches OpenCL)
    if (density < hScaled6)
        density = hScaled6;
    
    density *= params.mass_mult_Wpoly6Coefficient;
    
    rhoInv[id].x = density;
    rhoInv[id].y = 1.0f / density;
}

// ============================================================================
// Kernel: Compute Forces and Init Pressure
// Matches OpenCL pcisph_computeForcesAndInitPressure:
//   viscosity: 1e-4 * (vj - vi) * (hScaled - r_ij) / 1000 * 1.5 * mass_mult_divgrad / rho_i
//   surface tension: -1.7e-9 * surfTensCoeff * (hScaled²-r²)³ * (pos_i - pos_j) / mass
//   gravity: (gravity_x, gravity_y, gravity_z)
// ============================================================================

kernel void pcisph_computeForcesAndInitPressure(
    device float4* position [[buffer(0)]],
    device float4* velocity [[buffer(1)]],
    device float4* acceleration [[buffer(2)]],
    device float* pressure [[buffer(3)]],
    device float2* rhoInv [[buffer(4)]],
    device int* neighborMap [[buffer(5)]],
    device int* neighborCount [[buffer(6)]],
    device int* particleType [[buffer(7)]],
    constant SimulationParams& params [[buffer(8)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.particleCount) return;
    
    int ptype = particleType[id];
    if (ptype == BOUNDARY_PARTICLE) {
        acceleration[id] = float4(0.0f);
        pressure[id] = 0.0f;
        return;
    }
    
    float3 pos_i = position[id].xyz;
    float3 vel_i = velocity[id].xyz;
    float rho_i = rhoInv[id].x;
    float hScaled = params.hScaled;
    float hScaled2 = params.hScaled2;
    
    float4 accel_viscosity = float4(0.0f);
    float4 accel_surfTens = float4(0.0f);
    
    int count = neighborCount[id];
    for (int j = 0; j < count && j < MAX_NEIGHBOR_COUNT; j++) {
        int neighborIdx = neighborMap[id * MAX_NEIGHBOR_COUNT + j];
        if (neighborIdx == NO_PARTICLE_ID) continue;
        
        float3 pos_j = position[neighborIdx].xyz;
        float3 vel_j = velocity[neighborIdx].xyz;
        int ntype = particleType[neighborIdx];
        float not_bp = (ntype != BOUNDARY_PARTICLE) ? 1.0f : 0.0f;
        
        float3 r_vec = pos_i - pos_j;
        float r_unscaled = length(r_vec);
        float r_ij = r_unscaled * params.simulationScale;  // scaled distance
        float r_ij2 = r_ij * r_ij;
        
        if (r_ij < hScaled) {
            // Viscosity: matches OpenCL formula
            // All particle type pairs use 1.0e-4f coefficient (simplified from OpenCL's type-based branching)
            float4 vel_diff = float4(vel_j * not_bp - vel_i, 0.0f);
            accel_viscosity += 1.0e-4f * vel_diff * (hScaled - r_ij) / 1000.0f;
            
            // Surface tension: -1.7e-9 * surfTensCoeff * (hScaled²-r²)³ * (pos_i - pos_j)
            float surffKern = (hScaled2 - r_ij2) * (hScaled2 - r_ij2) * (hScaled2 - r_ij2);
            accel_surfTens += -1.7e-09f * params.surfTensCoeff * surffKern * float4(r_vec, 0.0f);
        }
    }
    
    // Apply viscosity coefficient: *= 1.5 * mass_mult_divgradWviscosityCoefficient / rho_i
    accel_viscosity *= 1.5f * params.mass_mult_divgradWviscosityCoefficient / max(rho_i, 1e-8f);
    
    // Surface tension / mass
    accel_surfTens /= params.mass;
    
    // Total: viscosity + gravity + surface tension
    float4 accel_i = accel_viscosity;
    accel_i += float4(0.0f, params.gravity, 0.0f, 0.0f);
    accel_i += accel_surfTens;
    
    acceleration[id] = accel_i;
    pressure[id] = 0.0f;  // Initial pressure = 0
}

// ============================================================================
// Kernel: Predict Positions (PCISPH)
// ============================================================================

kernel void pcisph_predictPositions(
    device float4* position [[buffer(0)]],
    device float4* velocity [[buffer(1)]],
    device float4* acceleration [[buffer(2)]],
    device float4* positionPredicted [[buffer(3)]],
    device int* particleType [[buffer(4)]],
    constant SimulationParams& params [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.particleCount) return;
    
    if (particleType[id] == BOUNDARY_PARTICLE) {
        positionPredicted[id] = position[id];
        return;
    }
    
    // Semi-implicit Euler prediction matching OpenCL pcisph_predictPositions:
    // velocity_t_dt = velocity_t + dt * acceleration_t_dt
    // position_t_dt = position_t + dt * simulationScaleInv * velocity_t_dt
    float3 vel = velocity[id].xyz + acceleration[id].xyz * params.timeStep;
    float3 pos = position[id].xyz + vel * params.timeStep * params.simulationScaleInv;
    
    positionPredicted[id] = float4(pos, position[id].w);
}

// ============================================================================
// Kernel: Correct Pressure (PCISPH iteration)
// ============================================================================

kernel void pcisph_correctPressure(
    device float* pressure [[buffer(0)]],
    device float2* rhoInv [[buffer(1)]],
    constant SimulationParams& params [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.particleCount) return;
    
    // Use PREDICTED density (stored in .y by predictDensity kernel)
    float rho_predicted = rhoInv[id].y;
    float rhoError = rho_predicted - params.rho0;
    
    // Pressure correction (non-negative)
    float p_corr = rhoError * params.delta;
    if (p_corr < 0.0f) p_corr = 0.0f;
    pressure[id] += p_corr;
}

// ============================================================================
// Kernel: Compute Pressure Force Acceleration
// Matches OpenCL pcisph_computePressureForceAcceleration (Solenthaler variant 1):
//   value = -(hScaled-r_ij)² * 0.5 * (p_i + p_j) / rho_j
//   result += value * vr_ij / r_ij  (with close-range repulsion)
//   result *= mass_mult_gradWspikyCoefficient / rho_i
// ============================================================================

kernel void pcisph_computePressureForceAcceleration(
    device float4* position [[buffer(0)]],
    device float4* acceleration [[buffer(1)]],
    device const float4* baseAcceleration [[buffer(2)]],
    device float* pressure [[buffer(3)]],
    device float2* rhoInv [[buffer(4)]],
    device int* neighborMap [[buffer(5)]],
    device int* neighborCount [[buffer(6)]],
    device int* particleType [[buffer(7)]],
    constant SimulationParams& params [[buffer(8)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.particleCount) return;
    
    if (particleType[id] == BOUNDARY_PARTICLE) {
        return;
    }
    
    float3 pos_i = position[id].xyz;
    float pressure_i = pressure[id];
    float rho_i = rhoInv[id].x;  // predicted density
    float hScaled = params.hScaled;
    
    float4 result = float4(0.0f);
    
    int count = neighborCount[id];
    for (int j = 0; j < count && j < MAX_NEIGHBOR_COUNT; j++) {
        int neighborIdx = neighborMap[id * MAX_NEIGHBOR_COUNT + j];
        if (neighborIdx == NO_PARTICLE_ID) continue;
        
        float3 pos_j = position[neighborIdx].xyz;
        float3 r_vec_unscaled = pos_i - pos_j;
        float4 vr_ij = float4(r_vec_unscaled * params.simulationScale, 0.0f);
        float r_ij = length(vr_ij.xyz);
        
        if (r_ij < hScaled && r_ij > 0.0f) {
            float pressure_j = pressure[neighborIdx];
            float rho_j = rhoInv[neighborIdx].x;  // predicted density
            
            // Solenthaler variant 1: -(hScaled-r)² * 0.5 * (p_i+p_j) / rho_j
            float value = -(hScaled - r_ij) * (hScaled - r_ij) * 0.5f * (pressure_i + pressure_j) / max(rho_j, 1e-8f);
            
            // Close-range repulsion (r < r0 = hScaled/4)
            if (r_ij < 0.5f * (hScaled * 0.5f)) {
                value = -(hScaled * 0.25f - r_ij) * (hScaled * 0.25f - r_ij) * 0.5f * (params.rho0 * params.delta) / max(rho_j, 1e-8f);
            }
            
            result += value * vr_ij / r_ij;
        }
    }
    
    result *= params.mass_mult_gradWspikyCoefficient / max(rho_i, 1e-8f);
    
    // Output: base acceleration + pressure force
    acceleration[id] = baseAcceleration[id] + result;
}

// ============================================================================
// Helper: Boundary particle interaction (Ihmsen et al., 2010)
// Computes position correction and tangential velocity from boundary normals.
// Boundary particle "normals" are stored in the velocity buffer.
// ============================================================================

inline void computeInteractionWithBoundaryParticles(
    uint id,
    float r0,
    device int* neighborMap,
    device float4* position,
    device float4* velocity,   // boundary normals stored here
    device int* particleType,
    device int* neighborCount,
    thread float4& pos_,
    bool tangVel,
    thread float4& vel_
) {
    float4 n_c_i = float4(0.0f);
    float w_c_ib_sum = 0.0f;
    float w_c_ib_second_sum = 0.0f;
    
    int count = neighborCount[id];
    for (int nc = 0; nc < count && nc < MAX_NEIGHBOR_COUNT; nc++) {
        int id_b = neighborMap[id * MAX_NEIGHBOR_COUNT + nc];
        if (id_b == NO_PARTICLE_ID) continue;
        
        if (particleType[id_b] == BOUNDARY_PARTICLE) {
            float3 diff = pos_.xyz - position[id_b].xyz;
            float x_ib_dist = length(diff);
            
            float w_c_ib = max(0.0f, (r0 - x_ib_dist) / r0);  // Ihmsen formula (10)
            float4 n_b = velocity[id_b];  // boundary normal stored in velocity
            
            n_c_i += n_b * w_c_ib;                    // formula (9)
            w_c_ib_sum += w_c_ib;                     // formula (11) sum #1
            w_c_ib_second_sum += w_c_ib * (r0 - x_ib_dist);  // formula (11) sum #2
        }
    }
    
    float n_c_i_length_sq = dot(n_c_i.xyz, n_c_i.xyz);
    if (n_c_i_length_sq > 0.0f) {
        float n_c_i_length = sqrt(n_c_i_length_sq);
        float4 delta_pos = (n_c_i / n_c_i_length) * w_c_ib_second_sum / w_c_ib_sum;  // formula (11)
        pos_.x += delta_pos.x;
        pos_.y += delta_pos.y;
        pos_.z += delta_pos.z;
        
        if (tangVel) {
            float eps = 0.99f;  // friction coefficient
            float vel_n_len = n_c_i.x * vel_.x + n_c_i.y * vel_.y + n_c_i.z * vel_.z;
            if (vel_n_len < 0.0f) {
                vel_.x -= n_c_i.x * vel_n_len;
                vel_.y -= n_c_i.y * vel_n_len;
                vel_.z -= n_c_i.z * vel_n_len;
                vel_ = vel_ * eps;  // formula (12)
            }
        }
    }
}

// ============================================================================
// Kernel: Integrate (Leapfrog + Semi-implicit Euler)
// Matches OpenCL pcisph_integrate:
//   iterationCount==0: store acceleration and return (initialization)
//   mode==0 (Leapfrog positions): pos += (vel*dt + prevAccel*dt²/2) * scaleInv
//   mode==1 (Leapfrog velocities): vel += (prevAccel+newAccel)*dt/2 + boundary
//   mode==2 (Semi-implicit Euler): vel += acc*dt, pos += vel*dt*scaleInv + boundary
// ============================================================================

kernel void pcisph_integrate(
    device float4* position [[buffer(0)]],
    device float4* velocity [[buffer(1)]],
    device float4* acceleration [[buffer(2)]],
    constant SimulationParams& params [[buffer(3)]],
    constant int& mode [[buffer(4)]],
    device float4* prevAcceleration [[buffer(5)]],
    device int* neighborMap [[buffer(6)]],
    device int* neighborCount [[buffer(7)]],
    device int* particleType [[buffer(8)]],
    constant int& iterationCount [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.particleCount) return;
    
    if (particleType[id] == BOUNDARY_PARTICLE) {
        return;
    }
    
    // At iterationCount == 0: just store acceleration for Leapfrog bootstrap
    if (iterationCount == 0) {
        prevAcceleration[id] = acceleration[id];  // total accel (base + pressure)
        return;
    }
    
    float4 acceleration_t = prevAcceleration[id];  // previous step's total acceleration
    acceleration_t.w = 0.0f;
    float4 velocity_t = velocity[id];
    float ptype = position[id].w;
    
    if (mode == 2) {
        // Semi-implicit Euler
        float4 acceleration_t_dt = acceleration[id];
        acceleration_t_dt.w = 0.0f;
        float4 velocity_t_dt = velocity_t + acceleration_t_dt * params.timeStep;
        float4 position_t_dt = float4(position[id].xyz, 0.0f) + velocity_t_dt * params.timeStep * params.simulationScaleInv;
        
        // Boundary interaction
        computeInteractionWithBoundaryParticles(
            id, params.r0, neighborMap, position, velocity,
            particleType, neighborCount, position_t_dt, true, velocity_t_dt);
        
        velocity[id] = velocity_t_dt;
        position[id] = position_t_dt;
        position[id].w = ptype;
        prevAcceleration[id] = acceleration_t_dt;
        return;
    }
    
    // Leapfrog integration
    if (mode == 0) {
        // Position update: x(t+dt) = x(t) + (v(t)*dt + a(t)*dt²/2) * scaleInv
        float4 position_t = float4(position[id].xyz, 0.0f);
        float4 position_t_dt = position_t + (velocity_t * params.timeStep + acceleration_t * params.timeStep * params.timeStep * 0.5f) * params.simulationScaleInv;
        position[id] = position_t_dt;
        position[id].w = ptype;
    }
    else if (mode == 1) {
        // Velocity update: v(t+dt) = v(t) + (a(t) + a(t+dt))*dt/2
        float4 position_t_dt = float4(position[id].xyz, 0.0f);
        float4 acceleration_t_dt = acceleration[id];
        acceleration_t_dt.w = 0.0f;
        float4 velocity_t_dt = velocity_t + (acceleration_t + acceleration_t_dt) * params.timeStep * 0.5f;
        
        // Boundary interaction (with tangential velocity adjustment)
        computeInteractionWithBoundaryParticles(
            id, params.r0, neighborMap, position, velocity,
            particleType, neighborCount, position_t_dt, true, velocity_t_dt);
        
        velocity[id] = velocity_t_dt;
        prevAcceleration[id] = acceleration_t_dt;  // store for next step
        position[id] = position_t_dt;
        position[id].w = ptype;
    }
}

// ============================================================================
// Kernel: Find Neighbors (spatial hashing)
// ============================================================================

kernel void findNeighbors(
    device float4* position [[buffer(0)]],
    device int* cellStart [[buffer(1)]],
    device int* cellEnd [[buffer(2)]],
    device int* neighborMap [[buffer(3)]],
    device int* neighborCount [[buffer(4)]],
    device uint2* particleIndex [[buffer(5)]],  // Maps sorted index -> original particle id
    constant SimulationParams& params [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.particleCount) return;
    
    float3 pos_i = position[id].xyz;
    int count = 0;
    
    // Get cell coordinates
    int3 cell_i = int3((pos_i - getGridMin(params)) / params.cellSize);
    
    // Search neighboring cells (3x3x3)
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int3 cell_j = cell_i + int3(dx, dy, dz);
                
                // Bounds check
                if (any(cell_j < 0) || any(cell_j >= getGridRes(params))) continue;
                
                int cellIdx = cell_j.x + cell_j.y * getGridRes(params).x 
                            + cell_j.z * getGridRes(params).x * getGridRes(params).y;
                
                int start = cellStart[cellIdx];
                int end = cellEnd[cellIdx];
                
                if (start == -1) continue;
                
                for (int sortedIdx = start; sortedIdx < end && count < MAX_NEIGHBOR_COUNT; sortedIdx++) {
                    // Get original particle ID from sorted index
                    uint origId = particleIndex[sortedIdx].y;
                    
                    if (origId == id) continue;  // Skip self
                    
                    float3 pos_j = position[origId].xyz;
                    float r = length(pos_i - pos_j);
                    
                    if (r < params.h && r > 0.0001f) {
                        neighborMap[id * MAX_NEIGHBOR_COUNT + count] = (int)origId;
                        count++;
                    }
                }
            }
        }
    }
    
    neighborCount[id] = count;
    
    // Fill remaining slots with -1
    for (int i = count; i < MAX_NEIGHBOR_COUNT; i++) {
        neighborMap[id * MAX_NEIGHBOR_COUNT + i] = NO_PARTICLE_ID;
    }
}

// ============================================================================
// Kernel: Compute Elastic Forces (for worm body)
// ============================================================================

kernel void pcisph_computeElasticForces(
    device float4* position [[buffer(0)]],
    device float4* velocity [[buffer(1)]],
    device float4* acceleration [[buffer(2)]],
    device float4* elasticConnections [[buffer(3)]],  // MAX_NEIGHBOR_COUNT float4s per elastic particle
    device float* muscleActivation [[buffer(4)]],     // Muscle activation signals
    constant SimulationParams& params [[buffer(5)]],
    constant float& elasticity [[buffer(6)]],
    constant float& maxMuscleForce [[buffer(7)]],
    constant uint& numOfElasticP [[buffer(8)]],
    constant uint& muscleCount [[buffer(9)]],
    uint index [[thread_position_in_grid]]  // Index among elastic particles only
) {
    if (index >= numOfElasticP) return;
    
    // For elastic particles, index IS the particle ID (elastic particles are first in the array)
    uint id = index;
    
    float3 pos_i = position[id].xyz;
    float3 elasticForce = float3(0.0f);
    float ptype_i = position[id].w;
    
    // Loop through all elastic connections for this particle
    for (int nc = 0; nc < MAX_NEIGHBOR_COUNT; nc++) {
        float4 conn = elasticConnections[index * MAX_NEIGHBOR_COUNT + nc];
        int partnerId = int(conn.x);
        float restLength = conn.y;  // Note: y is rest length in OpenCL format
        int muscleId = int(conn.z);
        
        if (partnerId == NO_PARTICLE_ID) break;  // End of connections
        
        float3 pos_j = position[partnerId].xyz;
        float3 r_vec = (pos_i - pos_j) * params.simulationScale;  // Scale to sim coords
        float r_ij = length(r_vec);
        
        if (r_ij > 1e-8f) {
            float delta_r = r_ij - restLength;
            float3 dir = r_vec / r_ij;
            
            // Elastic spring force
            float ptype_j = position[partnerId].w;
            
            // Check if both particles are worm body (type ~2.1-2.2)
            if (ptype_i > 2.05f && ptype_i < 2.25f && ptype_j > 2.05f && ptype_j < 2.25f) {
                elasticForce -= dir * delta_r * elasticity;
            } else {
                // Agar particles get reduced elasticity
                elasticForce -= dir * delta_r * elasticity * 0.25f;
            }
            
            // Muscle force (if this connection is a muscle)
            if (muscleId > 0 && muscleId <= (int)muscleCount) {
                float activation = muscleActivation[muscleId - 1];
                if (activation > 0.0f) {
                    elasticForce -= dir * activation * maxMuscleForce;
                }
            }
        }
    }
    
    acceleration[id] += float4(elasticForce, 0.0f);
}

// ============================================================================
// Kernel: Predict Density (PCISPH)
// Matches OpenCL pcisph_predictDensity:
//   density = sum((h²-r²)³) * simulationScale⁶ * mass_mult_Wpoly6Coefficient
//   Uses predicted positions (not actual positions)
//   Writes predicted density to rhoInv[id].y (base density stays in .x)
// ============================================================================

kernel void pcisph_predictDensity(
    device float4* positionPredicted [[buffer(0)]],
    device float2* rhoInv [[buffer(1)]],
    device int* neighborMap [[buffer(2)]],
    device int* neighborCount [[buffer(3)]],
    constant SimulationParams& params [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.particleCount) return;
    
    float3 pos_i = positionPredicted[id].xyz;
    float h = params.h;  // unscaled h (positions are in world coords)
    float h2 = h * h;
    float hScaled2 = params.hScaled2;
    float hScaled6 = hScaled2 * hScaled2 * hScaled2;
    float simScale = params.simulationScale;
    float simScale6 = simScale * simScale;
    simScale6 = simScale6 * simScale6 * simScale6;
    
    float density_accum = 0.0f;
    
    int count = neighborCount[id];
    for (int n = 0; n < count && n < MAX_NEIGHBOR_COUNT; n++) {
        int j = neighborMap[id * MAX_NEIGHBOR_COUNT + n];
        if (j == NO_PARTICLE_ID) continue;
        
        float3 pos_j = positionPredicted[j].xyz;
        float3 r_vec = pos_i - pos_j;
        float r2 = dot(r_vec, r_vec);  // world coords distance squared
        
        if (r2 < h2) {
            float diff = h2 - r2;
            density_accum += diff * diff * diff;  // (h² - r²)³
        }
    }
    
    float density = density_accum * simScale6;  // scale to simulation units
    
    // Floor clamp matching OpenCL
    if (density < hScaled6) {
        density = hScaled6;
    }
    
    density *= params.mass_mult_Wpoly6Coefficient;
    
    // Write predicted density to .y (keep base density in .x)
    rhoInv[id].y = density;
}

// ============================================================================
// Kernel: Clear Membrane Buffers
// ============================================================================

kernel void clearMembraneBuffers(
    device float* membraneForces [[buffer(0)]],
    constant SimulationParams& params [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.particleCount) return;
    membraneForces[id * 4 + 0] = 0.0f;
    membraneForces[id * 4 + 1] = 0.0f;
    membraneForces[id * 4 + 2] = 0.0f;
    membraneForces[id * 4 + 3] = 0.0f;
}

// ============================================================================
// Kernel: Compute Interaction With Membranes
// ============================================================================

kernel void computeInteractionWithMembranes(
    device float4* position [[buffer(0)]],
    device int* membraneData [[buffer(1)]],  // Triangles: i, j, k indices
    device float* membraneForces [[buffer(2)]],
    constant SimulationParams& params [[buffer(3)]],
    constant int& numMembranes [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    // For each membrane triangle, compute forces
    // This is a simplified placeholder
    if ((int)id >= numMembranes) return;
    
    int i = membraneData[id * 3 + 0];
    int j = membraneData[id * 3 + 1];
    int k = membraneData[id * 3 + 2];
    
    if (i < 0 || j < 0 || k < 0) return;
    
    // TODO: Compute actual membrane forces
    // This involves computing normals, area preservation, etc.
}

// ============================================================================
// Kernel: Finalize Membrane Interaction
// ============================================================================

kernel void computeInteractionWithMembranes_finalize(
    device float4* position [[buffer(0)]],
    device float4* velocity [[buffer(1)]],
    device float* membraneForces [[buffer(2)]],
    constant SimulationParams& params [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.particleCount) return;
    
    // Apply accumulated membrane forces to velocity
    float3 force = float3(
        membraneForces[id * 4 + 0],
        membraneForces[id * 4 + 1],
        membraneForces[id * 4 + 2]
    );
    
    velocity[id].xyz += force * params.timeStep / params.mass;
}

// ============================================================================
// Bitonic Sort for particle indices
// ============================================================================

// Sort key-value pairs by key (cell index)
// Each thread handles one comparison in the bitonic network
kernel void bitonicSortStep(
    device uint2* particleIndex [[buffer(0)]],
    constant uint& j [[buffer(1)]],
    constant uint& k [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Bitonic sort step
    uint i = gid;
    uint ixj = i ^ j;  // Partner to compare with
    
    if (ixj > i) {
        // Determine sort direction based on position in bitonic sequence
        bool ascending = ((i & k) == 0);
        
        uint2 a = particleIndex[i];
        uint2 b = particleIndex[ixj];
        
        // Compare by cell index (first component)
        bool needSwap = ascending ? (a.x > b.x) : (a.x < b.x);
        
        if (needSwap) {
            particleIndex[i] = b;
            particleIndex[ixj] = a;
        }
    }
}

// Local bitonic sort within threadgroup (faster for small sequences)
kernel void bitonicSortLocal(
    device uint2* particleIndex [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]],
    threadgroup uint2* localData [[threadgroup(0)]]
) {
    // Load into local memory
    uint idx = gid;
    if (idx < count) {
        localData[lid] = particleIndex[idx];
    } else {
        localData[lid] = uint2(0xFFFFFFFF, 0);  // Sentinel for padding
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Bitonic sort within threadgroup
    for (uint k = 2; k <= tgSize; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            uint ixj = lid ^ j;
            if (ixj > lid && ixj < tgSize) {
                bool ascending = ((lid & k) == 0);
                uint2 a = localData[lid];
                uint2 b = localData[ixj];
                
                if (ascending ? (a.x > b.x) : (a.x < b.x)) {
                    localData[lid] = b;
                    localData[ixj] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    
    // Write back
    if (idx < count) {
        particleIndex[idx] = localData[lid];
    }
}
