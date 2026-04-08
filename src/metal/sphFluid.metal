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
    float mass;                 // Particle mass
    float simulationScale;      // Scale factor
    float simulationScaleInv;   // 1 / simulationScale (for position updates)
    float timeStep;             // dt
    float viscosity;            // Viscosity coefficient
    float surfaceTension;       // Surface tension coefficient
    float gravity;              // Gravity acceleration
    
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
    float hScaled = params.hScaled;
    float simScale = params.simulationScale;
    
    // Numerically stable: use normalized distances q = r/h
    // Wpoly6(r, h) = 315/(64*pi*h^3) * (1 - (r/h)^2)^3  (for r < h)
    // density = mass * sum(Wpoly6) = mass * 315/(64*pi*h^3) * sum((1-q^2)^3)
    
    float h3 = hScaled * hScaled * hScaled;
    float coeff = 315.0f / (64.0f * M_PI_F * h3) * params.mass;
    
    float density = 0.0f;
    
    // Self contribution: q = 0, (1 - 0)^3 = 1
    density += 1.0f;
    
    // Neighbor contributions
    int count = neighborCount[id];
    for (int j = 0; j < count && j < MAX_NEIGHBOR_COUNT; j++) {
        int neighborIdx = neighborMap[id * MAX_NEIGHBOR_COUNT + j];
        if (neighborIdx == NO_PARTICLE_ID || neighborIdx < 0) continue;
        
        float3 pos_j = position[neighborIdx].xyz;
        float r = length(pos_i - pos_j) * simScale;
        
        if (r < hScaled) {
            float q = r / hScaled;      // normalized distance [0, 1)
            float q2 = q * q;
            float term = 1.0f - q2;     // (1 - q²)
            density += term * term * term;  // (1 - q²)³
        }
    }
    
    density *= coeff;
    
    // Clamp to rest density minimum
    if (density < 1.0f) density = 1.0f;
    
    rhoInv[id].x = density;
    rhoInv[id].y = 1.0f / density;
}

// ============================================================================
// Kernel: Compute Forces and Init Pressure
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
    
    float3 force = float3(0.0f);
    
    // Gravity (params.gravity is already negative for downward)
    force.y += params.gravity * params.mass;
    
    // Viscosity force
    int count = neighborCount[id];
    for (int j = 0; j < count && j < MAX_NEIGHBOR_COUNT; j++) {
        int neighborIdx = neighborMap[id * MAX_NEIGHBOR_COUNT + j];
        if (neighborIdx == NO_PARTICLE_ID) continue;
        
        float3 pos_j = position[neighborIdx].xyz;
        float3 vel_j = velocity[neighborIdx].xyz;
        float rho_j = rhoInv[neighborIdx].x;
        
        float3 r_vec = pos_i - pos_j;
        float r = length(r_vec);
        
        // Viscosity
        float3 vel_diff = vel_j - vel_i;
        float lap = laplacianWviscosity(r, params.h);
        force += params.viscosity * params.mass * (vel_diff / max(rho_j, 1e-8f)) * lap;
    }
    
    acceleration[id] = float4(force / params.mass, 0.0f);
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
    device float4* velocityPredicted [[buffer(4)]],
    device int* particleType [[buffer(5)]],
    constant SimulationParams& params [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.particleCount) return;
    
    if (particleType[id] == BOUNDARY_PARTICLE) {
        positionPredicted[id] = position[id];
        velocityPredicted[id] = float4(0.0f);
        return;
    }
    
    float3 vel = velocity[id].xyz + acceleration[id].xyz * params.timeStep;
    float3 pos = position[id].xyz + vel * params.timeStep;
    
    positionPredicted[id] = float4(pos, position[id].w);
    velocityPredicted[id] = float4(vel, 0.0f);
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
    
    float rho = rhoInv[id].x;
    float rhoError = rho - params.rho0;
    
    // Pressure update
    pressure[id] += params.delta * rhoError;
    pressure[id] = max(pressure[id], 0.0f);  // No negative pressure
}

// ============================================================================
// Kernel: Compute Pressure Force Acceleration
// ============================================================================

kernel void pcisph_computePressureForceAcceleration(
    device float4* position [[buffer(0)]],
    device float4* acceleration [[buffer(1)]],
    device const float4* baseAcceleration [[buffer(2)]],  // Base accel (gravity+viscosity+elastic) - read only
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
    float rhoInvSq_i = rhoInv[id].y * rhoInv[id].y;
    
    float3 pressureForce = float3(0.0f);
    
    int count = neighborCount[id];
    for (int j = 0; j < count && j < MAX_NEIGHBOR_COUNT; j++) {
        int neighborIdx = neighborMap[id * MAX_NEIGHBOR_COUNT + j];
        if (neighborIdx == NO_PARTICLE_ID) continue;
        
        float3 pos_j = position[neighborIdx].xyz;
        float pressure_j = pressure[neighborIdx];
        float rhoInvSq_j = rhoInv[neighborIdx].y * rhoInv[neighborIdx].y;
        
        float3 r_vec = pos_i - pos_j;
        float r = length(r_vec);
        
        // Pressure gradient (symmetric formulation)
        float pressureTerm = pressure_i * rhoInvSq_i + pressure_j * rhoInvSq_j;
        float3 grad = gradWspiky(r_vec, r, params.h);
        
        pressureForce -= params.mass * pressureTerm * grad;
    }
    
    // Output: base acceleration + pressure force
    // This OVERWRITES (not accumulates) so each PCISPH iteration starts fresh
    acceleration[id] = baseAcceleration[id] + float4(pressureForce / params.mass, 0.0f);
}

// ============================================================================
// Kernel: Integrate (Final position/velocity update)
// ============================================================================

kernel void pcisph_integrate(
    device float4* position [[buffer(0)]],
    device float4* velocity [[buffer(1)]],
    device float4* acceleration [[buffer(2)]],
    constant SimulationParams& params [[buffer(3)]],
    constant int& mode [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.particleCount) return;
    
    int particleType = (int)position[id].w;
    if (particleType == BOUNDARY_PARTICLE) {
        return;
    }
    
    float3 acc = acceleration[id].xyz;
    float3 vel = velocity[id].xyz + acc * params.timeStep;
    
    // Position update: velocity is in scaled coords, position is in world coords
    // Need to multiply by simulationScaleInv to convert
    float3 pos = position[id].xyz + vel * params.timeStep * params.simulationScaleInv;
    
    // Simple boundary clamping
    float3 minBound = float3(getGridMin(params)[0], getGridMin(params)[1], getGridMin(params)[2]) + 0.01f;
    float3 maxBound = float3(getGridMax(params)[0], getGridMax(params)[1], getGridMax(params)[2]) - 0.01f;
    pos = clamp(pos, minBound, maxBound);
    
    position[id] = float4(pos, position[id].w);  // Preserve particle type
    velocity[id] = float4(vel, 0.0f);
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
    float rho = 0.0f;
    
    int count = neighborCount[id];
    for (int n = 0; n < count; n++) {
        int j = neighborMap[id * MAX_NEIGHBOR_COUNT + n];
        if (j == NO_PARTICLE_ID) continue;
        
        float3 pos_j = positionPredicted[j].xyz;
        float r = length(pos_i - pos_j);
        
        if (r < params.h) {
            float q = 1.0f - r / params.h;
            rho += params.mass * (315.0f / (64.0f * M_PI_F * pow(params.h, 3.0f))) * pow(q, 3.0f);
        }
    }
    
    // Add self contribution
    rho += params.mass * (315.0f / (64.0f * M_PI_F * pow(params.h, 3.0f)));
    
    rhoInv[id].x = rho;
    rhoInv[id].y = (rho > 1e-8f) ? 1.0f / rho : 0.0f;
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
