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
constant int MAX_MEMBRANES_INCLUDING_SAME_PARTICLE = 7;

constant int LIQUID_PARTICLE = 1;
constant int ELASTIC_PARTICLE = 2;
constant int BOUNDARY_PARTICLE = 3;

constant int NO_PARTICLE_ID = -1;
constant int NO_CELL_ID = -1;
constant float NO_DISTANCE = -1.0f;

constant int radius_segments = 30;

// ============================================================================
// Simulation Parameters (passed as buffer)
// ============================================================================

struct SimulationParams {
    float h;                    // Smoothing radius (world scale for grid)
    float hScaled;              // h * simulationScale (for SPH kernels)
    float mass;                 // Particle mass
    float simulationScale;      // Scale factor
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
    float density = 0.0f;
    float hScaled = params.hScaled;
    float simScale = params.simulationScale;
    
    // Self contribution
    density = params.mass * Wpoly6(0.0f, hScaled);
    
    // Neighbor contributions
    int count = neighborCount[id];
    for (int j = 0; j < count && j < MAX_NEIGHBOR_COUNT; j++) {
        int neighborIdx = neighborMap[id * MAX_NEIGHBOR_COUNT + j];
        if (neighborIdx == NO_PARTICLE_ID) continue;
        
        float3 pos_j = position[neighborIdx].xyz;
        float3 r_vec = pos_i - pos_j;
        float r = length(r_vec) * simScale;  // Scale distance to simulation coordinates
        
        density += params.mass * Wpoly6(r, hScaled);
    }
    
    rhoInv[id].x = density;
    rhoInv[id].y = 1.0f / max(density, 1e-8f);
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
    float rho_i = rhoInv[id].x;
    float rhoInv_i = rhoInv[id].y;
    
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
    device float* pressure [[buffer(2)]],
    device float2* rhoInv [[buffer(3)]],
    device int* neighborMap [[buffer(4)]],
    device int* neighborCount [[buffer(5)]],
    device int* particleType [[buffer(6)]],
    constant SimulationParams& params [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.particleCount) return;
    
    if (particleType[id] == BOUNDARY_PARTICLE) {
        return;
    }
    
    float3 pos_i = position[id].xyz;
    float pressure_i = pressure[id];
    float rho_i = rhoInv[id].x;
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
    
    acceleration[id] += float4(pressureForce / params.mass, 0.0f);
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
    float3 pos = position[id].xyz + vel * params.timeStep;
    
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
    device float4* elasticConnections [[buffer(3)]],  // x,y,z = partner particle ids, w = rest length
    device int* particleType [[buffer(4)]],
    constant SimulationParams& params [[buffer(5)]],
    constant float& elasticity [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= params.particleCount) return;
    
    if (particleType[id] != ELASTIC_PARTICLE) return;
    
    float3 pos_i = position[id].xyz;
    float3 elasticForce = float3(0.0f);
    
    float4 conn = elasticConnections[id];
    int partnerId = int(conn.x);
    float restLength = conn.w;
    
    if (partnerId != NO_PARTICLE_ID && restLength > 0.0f) {
        float3 pos_j = position[partnerId].xyz;
        float3 r_vec = pos_j - pos_i;
        float dist = length(r_vec);
        
        if (dist > 1e-8f) {
            float stretch = dist - restLength;
            float3 springForce = elasticity * stretch * normalize(r_vec);
            elasticForce += springForce;
        }
    }
    
    acceleration[id] += float4(elasticForce / params.mass, 0.0f);
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
