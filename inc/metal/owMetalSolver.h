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
#include "owSolver.h"

// Forward declarations (Metal types)
namespace MTL {
    class Device;
    class CommandQueue;
    class ComputePipelineState;
    class Buffer;
    class Library;
}

// Simulation parameters - simple packed layout matching Metal
struct SimulationParams {
    float h;
    float hScaled;  // h * simulationScale (for SPH kernels)
    float hScaled2; // hScaled^2
    float mass;
    float simulationScale;
    float simulationScaleInv;  // 1 / simulationScale (for position updates)
    float timeStep;
    float viscosity;
    float surfaceTension;
    float gravity;
    float r0;  // equilibrium distance = 0.5 * h
    
    // Pre-computed coefficients matching OpenCL
    float mass_mult_Wpoly6Coefficient;
    float mass_mult_gradWspikyCoefficient;
    float mass_mult_divgradWviscosityCoefficient;
    float surfTensCoeff;
    
    unsigned int particleCount;
    unsigned int gridCellCount;
    float gridMinX, gridMinY, gridMinZ;
    float gridMaxX, gridMaxY, gridMaxZ;
    int gridResX, gridResY, gridResZ;
    float cellSize;
    
    float rho0;
    float delta;
    int pcisphIterations;
};

class owConfigProperty;

// Metal solver - implements owISolver interface
class owMetalSolver : public owISolver {
public:
    owMetalSolver(
        const float* position_cpp,
        const float* velocity_cpp,
        owConfigProperty* config,
        const float* elasticConnectionsData_cpp = nullptr,
        const int* membraneData_cpp = nullptr,
        const int* particleMembranesList_cpp = nullptr
    );
    
    ~owMetalSolver() override;
    
    // owISolver interface implementation
    void reset(
        const float* position,
        const float* velocity,
        owConfigProperty* config,
        const float* elasticConnections = nullptr,
        const int* membraneData = nullptr,
        const int* particleMembranesList = nullptr
    ) override;
    
    // Neighbor search kernels
    unsigned int _runClearBuffers(owConfigProperty* config) override;
    unsigned int _runHashParticles(owConfigProperty* config) override;
    void _runSort(owConfigProperty* config) override;
    unsigned int _runSortPostPass(owConfigProperty* config) override;
    unsigned int _runIndexx(owConfigProperty* config) override;
    void _runIndexPostPass(owConfigProperty* config) override;
    unsigned int _runFindNeighbors(owConfigProperty* config) override;
    
    // PCISPH physics kernels
    unsigned int _run_pcisph_computeDensity(owConfigProperty* config) override;
    unsigned int _run_pcisph_computeForcesAndInitPressure(owConfigProperty* config) override;
    unsigned int _run_pcisph_computeElasticForces(owConfigProperty* config) override;
    void _saveBaseAcceleration() override;
    unsigned int _run_pcisph_predictPositions(owConfigProperty* config) override;
    unsigned int _run_pcisph_predictDensity(owConfigProperty* config) override;
    unsigned int _run_pcisph_correctPressure(owConfigProperty* config) override;
    unsigned int _run_pcisph_computePressureForceAcceleration(owConfigProperty* config) override;
    unsigned int _run_pcisph_integrate(int iterationCount, int mode, owConfigProperty* config) override;
    
    // Membrane kernels
    unsigned int _run_clearMembraneBuffers(owConfigProperty* config) override;
    unsigned int _run_computeInteractionWithMembranes(owConfigProperty* config) override;
    unsigned int _run_computeInteractionWithMembranes_finalize(owConfigProperty* config) override;
    
    // Data transfer
    void updateMuscleActivityData(float* data, owConfigProperty* config) override;
    void read_position_buffer(float* position, owConfigProperty* config) override;
    void read_velocity_buffer(float* velocity, owConfigProperty* config) override;
    void read_density_buffer(float* density, owConfigProperty* config) override;
    void read_particleIndex_buffer(unsigned int* particleIndex, owConfigProperty* config) override;
    void read_pressure_buffer(float* pressure, owConfigProperty* config) override;
    
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
    
    // Helper for dispatching compute
    void dispatchKernel(MTL::ComputePipelineState* pipeline, unsigned int count);
    
    // Metal objects
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::Library* library;
    
    // Compute pipeline states (one per kernel)
    MTL::ComputePipelineState* clearBuffersPipeline;
    MTL::ComputePipelineState* hashParticlesPipeline;
    MTL::ComputePipelineState* sortPipeline;
    MTL::ComputePipelineState* findNeighborsPipeline;
    MTL::ComputePipelineState* computeDensityPipeline;
    MTL::ComputePipelineState* computeForcesPipeline;
    MTL::ComputePipelineState* predictPositionsPipeline;
    MTL::ComputePipelineState* predictDensityPipeline;
    MTL::ComputePipelineState* correctPressurePipeline;
    MTL::ComputePipelineState* computePressureForcePipeline;
    MTL::ComputePipelineState* computeElasticForcesPipeline;
    MTL::ComputePipelineState* integratePipeline;
    MTL::ComputePipelineState* clearMembraneBuffersPipeline;
    MTL::ComputePipelineState* computeMembranesPipeline;
    MTL::ComputePipelineState* computeMembranesFinalizePipeline;
    
    // Buffers
    MTL::Buffer* positionBuffer;
    MTL::Buffer* velocityBuffer;
    MTL::Buffer* accelerationBuffer;
    MTL::Buffer* baseAccelerationBuffer;  // Stores gravity+viscosity+elastic before PCISPH loop
    MTL::Buffer* prevAccelerationBuffer;  // Stores total acceleration from previous step (for Leapfrog)
    MTL::Buffer* positionPredictedBuffer;
    MTL::Buffer* velocityPredictedBuffer;
    MTL::Buffer* pressureBuffer;
    MTL::Buffer* rhoBuffer;
    MTL::Buffer* neighborMapBuffer;
    MTL::Buffer* neighborCountBuffer;
    MTL::Buffer* cellStartBuffer;
    MTL::Buffer* cellEndBuffer;
    MTL::Buffer* particleIndexBuffer;
    MTL::Buffer* particleTypeBuffer;
    MTL::Buffer* elasticConnectionsBuffer;
    MTL::Buffer* membraneDataBuffer;
    MTL::Buffer* particleMembranesListBuffer;
    MTL::Buffer* muscleActivationBuffer;
    MTL::Buffer* membraneCorrectionBuffer;
    MTL::Buffer* paramsBuffer;
    
    // Configuration
    SimulationParams params;
    unsigned int particleCount;
    unsigned int gridCellCount;
    
    static constexpr int METAL_MAX_NEIGHBORS = 32;
    static constexpr int METAL_MAX_MEMBRANES_PER_PARTICLE = 7;
};

#endif // __APPLE__
#endif // OW_METAL_SOLVER_H
