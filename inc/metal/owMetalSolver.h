/*******************************************************************************
 * The MIT License (MIT)
 * Copyright (c) 2011, 2013 OpenWorm.
 * Metal port (c) 2026
 *
 * Metal Compute Solver for Sibernetic
 * Replaces OpenCL solver with Metal on Apple Silicon
 *******************************************************************************/

#ifndef OW_METAL_SOLVER_H
#define OW_METAL_SOLVER_H

#ifdef __APPLE__

#include <vector>
#include <string>

// Forward declarations (Metal types)
namespace MTL {
    class Device;
    class CommandQueue;
    class ComputePipelineState;
    class Buffer;
    class Library;
}

// Simulation parameters (must match Metal shader struct)
struct SimulationParams {
    float h;
    float mass;
    float simulationScale;
    float timeStep;
    float viscosity;
    float surfaceTension;
    float gravity;
    
    unsigned int particleCount;
    unsigned int gridCellCount;
    float gridMin[3];
    float gridMax[3];
    int gridResolution[3];
    float cellSize;
    
    float rho0;
    float delta;
    int pcisphIterations;
};

class owConfigProperty;

class owMetalSolver {
public:
    owMetalSolver(
        const float* position_cpp,
        const float* velocity_cpp,
        owConfigProperty* config,
        const float* elasticConnectionsData_cpp,
        const int* membraneData_cpp,
        const int* particleMembranesList_cpp
    );
    
    ~owMetalSolver();
    
    // Main simulation step
    void step();
    
    // Copy data back to CPU
    void readPosition(float* position_cpp);
    void readVelocity(float* velocity_cpp);
    void readDensity(float* density_cpp);
    
    // Update muscle activations
    void updateMuscleActivation(const float* activations, int count);
    
    // Getters
    unsigned int getParticleCount() const { return particleCount; }
    
private:
    // Initialization
    void initDevice();
    void loadLibrary();
    void createPipelineStates();
    void createBuffers(
        const float* position_cpp,
        const float* velocity_cpp,
        owConfigProperty* config,
        const float* elasticConnectionsData_cpp
    );
    
    // Simulation substeps
    void dispatchHashParticles();
    void dispatchBuildGrid();
    void dispatchFindNeighbors();
    void dispatchComputeDensity();
    void dispatchComputeForces();
    void dispatchPCISPHLoop();
    void dispatchIntegrate();
    
    // Metal objects
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::Library* library;
    
    // Compute pipeline states (one per kernel)
    MTL::ComputePipelineState* clearBuffersPipeline;
    MTL::ComputePipelineState* hashParticlesPipeline;
    MTL::ComputePipelineState* findNeighborsPipeline;
    MTL::ComputePipelineState* computeDensityPipeline;
    MTL::ComputePipelineState* computeForcesPipeline;
    MTL::ComputePipelineState* predictPositionsPipeline;
    MTL::ComputePipelineState* correctPressurePipeline;
    MTL::ComputePipelineState* computePressureForcePipeline;
    MTL::ComputePipelineState* computeElasticForcesPipeline;
    MTL::ComputePipelineState* integratePipeline;
    
    // Buffers
    MTL::Buffer* positionBuffer;
    MTL::Buffer* velocityBuffer;
    MTL::Buffer* accelerationBuffer;
    MTL::Buffer* positionPredictedBuffer;
    MTL::Buffer* velocityPredictedBuffer;
    MTL::Buffer* pressureBuffer;
    MTL::Buffer* rhoInvBuffer;
    MTL::Buffer* neighborMapBuffer;
    MTL::Buffer* neighborCountBuffer;
    MTL::Buffer* cellStartBuffer;
    MTL::Buffer* cellEndBuffer;
    MTL::Buffer* particleTypeBuffer;
    MTL::Buffer* elasticConnectionsBuffer;
    MTL::Buffer* paramsBuffer;
    
    // Configuration
    SimulationParams params;
    unsigned int particleCount;
    unsigned int gridCellCount;
    
    static constexpr int MAX_NEIGHBOR_COUNT = 32;
};

#endif // __APPLE__
#endif // OW_METAL_SOLVER_H
