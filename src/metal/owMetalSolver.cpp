/*******************************************************************************
 * The MIT License (MIT)
 * Copyright (c) 2011, 2013 OpenWorm.
 * Metal port (c) 2026
 *
 * Metal Compute Solver Implementation
 *******************************************************************************/

#ifdef __APPLE__

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/owMetalSolver.h"
#include "owConfigProperty.h"

#include <iostream>
#include <fstream>
#include <sstream>

// ============================================================================
// Constructor
// ============================================================================

owMetalSolver::owMetalSolver(
    const float* position_cpp,
    const float* velocity_cpp,
    owConfigProperty* config,
    const float* elasticConnectionsData_cpp,
    const int* membraneData_cpp,
    const int* particleMembranesList_cpp
) {
    std::cout << "Initializing Metal solver..." << std::endl;
    
    initDevice();
    loadLibrary();
    createPipelineStates();
    createBuffers(position_cpp, velocity_cpp, config, elasticConnectionsData_cpp);
    
    std::cout << "Metal solver initialized with " << particleCount << " particles" << std::endl;
}

// ============================================================================
// Destructor
// ============================================================================

owMetalSolver::~owMetalSolver() {
    // Release buffers
    if (positionBuffer) positionBuffer->release();
    if (velocityBuffer) velocityBuffer->release();
    if (accelerationBuffer) accelerationBuffer->release();
    if (positionPredictedBuffer) positionPredictedBuffer->release();
    if (velocityPredictedBuffer) velocityPredictedBuffer->release();
    if (pressureBuffer) pressureBuffer->release();
    if (rhoInvBuffer) rhoInvBuffer->release();
    if (neighborMapBuffer) neighborMapBuffer->release();
    if (neighborCountBuffer) neighborCountBuffer->release();
    if (cellStartBuffer) cellStartBuffer->release();
    if (cellEndBuffer) cellEndBuffer->release();
    if (particleTypeBuffer) particleTypeBuffer->release();
    if (elasticConnectionsBuffer) elasticConnectionsBuffer->release();
    if (paramsBuffer) paramsBuffer->release();
    
    // Release pipelines
    if (clearBuffersPipeline) clearBuffersPipeline->release();
    if (hashParticlesPipeline) hashParticlesPipeline->release();
    if (findNeighborsPipeline) findNeighborsPipeline->release();
    if (computeDensityPipeline) computeDensityPipeline->release();
    if (computeForcesPipeline) computeForcesPipeline->release();
    if (predictPositionsPipeline) predictPositionsPipeline->release();
    if (correctPressurePipeline) correctPressurePipeline->release();
    if (computePressureForcePipeline) computePressureForcePipeline->release();
    if (computeElasticForcesPipeline) computeElasticForcesPipeline->release();
    if (integratePipeline) integratePipeline->release();
    
    // Release core objects
    if (library) library->release();
    if (commandQueue) commandQueue->release();
    if (device) device->release();
}

// ============================================================================
// Initialization
// ============================================================================

void owMetalSolver::initDevice() {
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("Metal is not supported on this device");
    }
    
    std::cout << "Metal device: " << device->name()->utf8String() << std::endl;
    
    commandQueue = device->newCommandQueue();
    if (!commandQueue) {
        throw std::runtime_error("Failed to create Metal command queue");
    }
}

void owMetalSolver::loadLibrary() {
    NS::Error* error = nullptr;
    
    // Try to load precompiled metallib first
    NS::String* libPath = NS::String::string("sphFluid.metallib", NS::UTF8StringEncoding);
    library = device->newLibrary(libPath, &error);
    
    if (!library) {
        // Fall back to compiling from source
        std::cout << "Compiling Metal shaders from source..." << std::endl;
        
        // Read shader source
        std::ifstream file("src/metal/sphFluid.metal");
        if (!file.is_open()) {
            throw std::runtime_error("Could not open Metal shader file: src/metal/sphFluid.metal");
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string source = buffer.str();
        
        NS::String* sourceStr = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
        
        MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
        library = device->newLibrary(sourceStr, options, &error);
        options->release();
        
        if (!library) {
            std::string errMsg = "Failed to compile Metal shaders";
            if (error) {
                errMsg += ": ";
                errMsg += error->localizedDescription()->utf8String();
            }
            throw std::runtime_error(errMsg);
        }
        
        std::cout << "Metal shaders compiled successfully" << std::endl;
    }
}

void owMetalSolver::createPipelineStates() {
    NS::Error* error = nullptr;
    
    auto createPipeline = [&](const char* name) -> MTL::ComputePipelineState* {
        NS::String* funcName = NS::String::string(name, NS::UTF8StringEncoding);
        MTL::Function* func = library->newFunction(funcName);
        if (!func) {
            throw std::runtime_error(std::string("Metal function not found: ") + name);
        }
        
        MTL::ComputePipelineState* pipeline = device->newComputePipelineState(func, &error);
        func->release();
        
        if (!pipeline) {
            std::string errMsg = std::string("Failed to create pipeline for: ") + name;
            if (error) {
                errMsg += ": ";
                errMsg += error->localizedDescription()->utf8String();
            }
            throw std::runtime_error(errMsg);
        }
        
        return pipeline;
    };
    
    clearBuffersPipeline = createPipeline("clearBuffers");
    hashParticlesPipeline = createPipeline("hashParticles");
    findNeighborsPipeline = createPipeline("findNeighbors");
    computeDensityPipeline = createPipeline("pcisph_computeDensity");
    computeForcesPipeline = createPipeline("pcisph_computeForcesAndInitPressure");
    predictPositionsPipeline = createPipeline("pcisph_predictPositions");
    correctPressurePipeline = createPipeline("pcisph_correctPressure");
    computePressureForcePipeline = createPipeline("pcisph_computePressureForceAcceleration");
    computeElasticForcesPipeline = createPipeline("pcisph_computeElasticForces");
    integratePipeline = createPipeline("pcisph_integrate");
    
    std::cout << "Created " << 10 << " Metal compute pipelines" << std::endl;
}

void owMetalSolver::createBuffers(
    const float* position_cpp,
    const float* velocity_cpp,
    owConfigProperty* config,
    const float* elasticConnectionsData_cpp
) {
    particleCount = config->getParticleCount();
    gridCellCount = config->getGridCellCount();
    
    // Setup simulation parameters
    params.h = config->h;
    params.mass = config->mass;
    params.simulationScale = config->simulationScale;
    params.timeStep = config->timeStep;
    params.viscosity = config->viscosity;
    params.surfaceTension = config->surfaceTension;
    params.gravity = config->gravity;
    params.particleCount = particleCount;
    params.gridCellCount = gridCellCount;
    params.gridMin[0] = config->xmin;
    params.gridMin[1] = config->ymin;
    params.gridMin[2] = config->zmin;
    params.gridMax[0] = config->xmax;
    params.gridMax[1] = config->ymax;
    params.gridMax[2] = config->zmax;
    params.gridResolution[0] = config->gridCellsX;
    params.gridResolution[1] = config->gridCellsY;
    params.gridResolution[2] = config->gridCellsZ;
    params.cellSize = config->h;  // Cell size typically equals smoothing radius
    params.rho0 = config->rho0;
    params.delta = config->delta;
    params.pcisphIterations = 3;
    
    // Create buffers
    size_t float4Size = 4 * sizeof(float);
    size_t float2Size = 2 * sizeof(float);
    
    positionBuffer = device->newBuffer(position_cpp, particleCount * float4Size, MTL::ResourceStorageModeShared);
    velocityBuffer = device->newBuffer(velocity_cpp, particleCount * float4Size, MTL::ResourceStorageModeShared);
    accelerationBuffer = device->newBuffer(particleCount * float4Size, MTL::ResourceStorageModeShared);
    positionPredictedBuffer = device->newBuffer(particleCount * float4Size, MTL::ResourceStorageModeShared);
    velocityPredictedBuffer = device->newBuffer(particleCount * float4Size, MTL::ResourceStorageModeShared);
    pressureBuffer = device->newBuffer(particleCount * sizeof(float), MTL::ResourceStorageModeShared);
    rhoInvBuffer = device->newBuffer(particleCount * float2Size, MTL::ResourceStorageModeShared);
    
    neighborMapBuffer = device->newBuffer(particleCount * MAX_NEIGHBOR_COUNT * sizeof(int), MTL::ResourceStorageModeShared);
    neighborCountBuffer = device->newBuffer(particleCount * sizeof(int), MTL::ResourceStorageModeShared);
    
    cellStartBuffer = device->newBuffer(gridCellCount * sizeof(int), MTL::ResourceStorageModeShared);
    cellEndBuffer = device->newBuffer(gridCellCount * sizeof(int), MTL::ResourceStorageModeShared);
    
    // TODO: Copy particle types from config
    particleTypeBuffer = device->newBuffer(particleCount * sizeof(int), MTL::ResourceStorageModeShared);
    
    if (elasticConnectionsData_cpp) {
        elasticConnectionsBuffer = device->newBuffer(elasticConnectionsData_cpp, 
            particleCount * float4Size, MTL::ResourceStorageModeShared);
    } else {
        elasticConnectionsBuffer = device->newBuffer(particleCount * float4Size, MTL::ResourceStorageModeShared);
    }
    
    paramsBuffer = device->newBuffer(&params, sizeof(SimulationParams), MTL::ResourceStorageModeShared);
    
    std::cout << "Created Metal buffers for " << particleCount << " particles" << std::endl;
}

// ============================================================================
// Simulation Step
// ============================================================================

void owMetalSolver::step() {
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    // Helper to dispatch a kernel
    auto dispatch = [&](MTL::ComputePipelineState* pipeline, uint count) {
        encoder->setComputePipelineState(pipeline);
        
        NS::UInteger threadGroupSize = pipeline->maxTotalThreadsPerThreadgroup();
        if (threadGroupSize > count) threadGroupSize = count;
        
        MTL::Size gridSize = MTL::Size(count, 1, 1);
        MTL::Size groupSize = MTL::Size(threadGroupSize, 1, 1);
        
        encoder->dispatchThreads(gridSize, groupSize);
    };
    
    // 1. Hash particles to grid
    encoder->setBuffer(positionBuffer, 0, 0);
    encoder->setBuffer(cellStartBuffer, 0, 1);
    encoder->setBuffer(cellEndBuffer, 0, 2);
    encoder->setBuffer(paramsBuffer, 0, 3);
    dispatch(hashParticlesPipeline, particleCount);
    
    // 2. Find neighbors
    encoder->setBuffer(positionBuffer, 0, 0);
    encoder->setBuffer(cellStartBuffer, 0, 1);
    encoder->setBuffer(cellEndBuffer, 0, 2);
    encoder->setBuffer(neighborMapBuffer, 0, 3);
    encoder->setBuffer(neighborCountBuffer, 0, 4);
    encoder->setBuffer(paramsBuffer, 0, 5);
    dispatch(findNeighborsPipeline, particleCount);
    
    // 3. Compute density
    encoder->setBuffer(positionBuffer, 0, 0);
    encoder->setBuffer(velocityBuffer, 0, 1);
    encoder->setBuffer(rhoInvBuffer, 0, 2);
    encoder->setBuffer(neighborMapBuffer, 0, 3);
    encoder->setBuffer(neighborCountBuffer, 0, 4);
    encoder->setBuffer(paramsBuffer, 0, 5);
    dispatch(computeDensityPipeline, particleCount);
    
    // 4. Compute forces and init pressure
    encoder->setBuffer(positionBuffer, 0, 0);
    encoder->setBuffer(velocityBuffer, 0, 1);
    encoder->setBuffer(accelerationBuffer, 0, 2);
    encoder->setBuffer(pressureBuffer, 0, 3);
    encoder->setBuffer(rhoInvBuffer, 0, 4);
    encoder->setBuffer(neighborMapBuffer, 0, 5);
    encoder->setBuffer(neighborCountBuffer, 0, 6);
    encoder->setBuffer(particleTypeBuffer, 0, 7);
    encoder->setBuffer(paramsBuffer, 0, 8);
    dispatch(computeForcesPipeline, particleCount);
    
    // 5. PCISPH pressure correction loop
    for (int iter = 0; iter < params.pcisphIterations; iter++) {
        // Predict positions
        encoder->setBuffer(positionBuffer, 0, 0);
        encoder->setBuffer(velocityBuffer, 0, 1);
        encoder->setBuffer(accelerationBuffer, 0, 2);
        encoder->setBuffer(positionPredictedBuffer, 0, 3);
        encoder->setBuffer(velocityPredictedBuffer, 0, 4);
        encoder->setBuffer(particleTypeBuffer, 0, 5);
        encoder->setBuffer(paramsBuffer, 0, 6);
        dispatch(predictPositionsPipeline, particleCount);
        
        // Correct pressure
        encoder->setBuffer(pressureBuffer, 0, 0);
        encoder->setBuffer(rhoInvBuffer, 0, 1);
        encoder->setBuffer(paramsBuffer, 0, 2);
        dispatch(correctPressurePipeline, particleCount);
        
        // Compute pressure force
        encoder->setBuffer(positionBuffer, 0, 0);
        encoder->setBuffer(accelerationBuffer, 0, 1);
        encoder->setBuffer(pressureBuffer, 0, 2);
        encoder->setBuffer(rhoInvBuffer, 0, 3);
        encoder->setBuffer(neighborMapBuffer, 0, 4);
        encoder->setBuffer(neighborCountBuffer, 0, 5);
        encoder->setBuffer(particleTypeBuffer, 0, 6);
        encoder->setBuffer(paramsBuffer, 0, 7);
        dispatch(computePressureForcePipeline, particleCount);
    }
    
    // 6. Compute elastic forces (worm body)
    float elasticity = 1000.0f;  // TODO: Get from config
    encoder->setBuffer(positionBuffer, 0, 0);
    encoder->setBuffer(velocityBuffer, 0, 1);
    encoder->setBuffer(accelerationBuffer, 0, 2);
    encoder->setBuffer(elasticConnectionsBuffer, 0, 3);
    encoder->setBuffer(particleTypeBuffer, 0, 4);
    encoder->setBuffer(paramsBuffer, 0, 5);
    encoder->setBytes(&elasticity, sizeof(float), 6);
    dispatch(computeElasticForcesPipeline, particleCount);
    
    // 7. Integrate
    encoder->setBuffer(positionBuffer, 0, 0);
    encoder->setBuffer(velocityBuffer, 0, 1);
    encoder->setBuffer(accelerationBuffer, 0, 2);
    encoder->setBuffer(particleTypeBuffer, 0, 3);
    encoder->setBuffer(paramsBuffer, 0, 4);
    dispatch(integratePipeline, particleCount);
    
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}

// ============================================================================
// Data Transfer
// ============================================================================

void owMetalSolver::readPosition(float* position_cpp) {
    memcpy(position_cpp, positionBuffer->contents(), particleCount * 4 * sizeof(float));
}

void owMetalSolver::readVelocity(float* velocity_cpp) {
    memcpy(velocity_cpp, velocityBuffer->contents(), particleCount * 4 * sizeof(float));
}

void owMetalSolver::readDensity(float* density_cpp) {
    float* rhoInv = (float*)rhoInvBuffer->contents();
    for (unsigned int i = 0; i < particleCount; i++) {
        density_cpp[i] = rhoInv[i * 2];  // x component is density
    }
}

void owMetalSolver::updateMuscleActivation(const float* activations, int count) {
    // TODO: Implement muscle activation update
    // This would modify elastic connection rest lengths or apply forces
}

#endif // __APPLE__
