/*******************************************************************************
 * The MIT License (MIT)
 * Copyright (c) 2011, 2013 OpenWorm.
 * Metal port (c) 2026
 *
 * Metal Compute Solver Implementation
 * Implements owISolver interface for Apple Silicon GPU acceleration
 *******************************************************************************/

#ifdef __APPLE__

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "metal/owMetalSolver.h"
#include "owConfigProperty.h"
#include "owOpenCLConstant.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

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
) : device(nullptr), commandQueue(nullptr), library(nullptr),
    positionBuffer(nullptr), velocityBuffer(nullptr), accelerationBuffer(nullptr),
    positionPredictedBuffer(nullptr), velocityPredictedBuffer(nullptr),
    pressureBuffer(nullptr), rhoBuffer(nullptr), neighborMapBuffer(nullptr),
    neighborCountBuffer(nullptr), cellStartBuffer(nullptr), cellEndBuffer(nullptr),
    particleIndexBuffer(nullptr), particleTypeBuffer(nullptr),
    elasticConnectionsBuffer(nullptr), membraneDataBuffer(nullptr),
    particleMembranesListBuffer(nullptr), muscleActivationBuffer(nullptr),
    paramsBuffer(nullptr)
{
    std::cout << "Initializing Metal solver..." << std::endl;
    
    initDevice();
    loadLibrary();
    createPipelineStates();
    createBuffers(position_cpp, velocity_cpp, config, elasticConnectionsData_cpp);
    
    // Copy membrane data if provided
    if (membraneData_cpp && config->numOfMembranes > 0) {
        membraneDataBuffer = device->newBuffer(membraneData_cpp, 
            config->numOfMembranes * 3 * sizeof(int), MTL::ResourceStorageModeShared);
    }
    
    if (particleMembranesList_cpp && config->numOfElasticP > 0) {
        particleMembranesListBuffer = device->newBuffer(particleMembranesList_cpp,
            config->numOfElasticP * MAX_MEMBRANES_INCLUDING_SAME_PARTICLE * sizeof(int),
            MTL::ResourceStorageModeShared);
    }
    
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
    if (rhoBuffer) rhoBuffer->release();
    if (neighborMapBuffer) neighborMapBuffer->release();
    if (neighborCountBuffer) neighborCountBuffer->release();
    if (cellStartBuffer) cellStartBuffer->release();
    if (cellEndBuffer) cellEndBuffer->release();
    if (particleIndexBuffer) particleIndexBuffer->release();
    if (particleTypeBuffer) particleTypeBuffer->release();
    if (elasticConnectionsBuffer) elasticConnectionsBuffer->release();
    if (membraneDataBuffer) membraneDataBuffer->release();
    if (particleMembranesListBuffer) particleMembranesListBuffer->release();
    if (muscleActivationBuffer) muscleActivationBuffer->release();
    if (paramsBuffer) paramsBuffer->release();
    
    // Release pipelines
    if (clearBuffersPipeline) clearBuffersPipeline->release();
    if (hashParticlesPipeline) hashParticlesPipeline->release();
    if (sortPipeline) sortPipeline->release();
    if (findNeighborsPipeline) findNeighborsPipeline->release();
    if (computeDensityPipeline) computeDensityPipeline->release();
    if (computeForcesPipeline) computeForcesPipeline->release();
    if (predictPositionsPipeline) predictPositionsPipeline->release();
    if (predictDensityPipeline) predictDensityPipeline->release();
    if (correctPressurePipeline) correctPressurePipeline->release();
    if (computePressureForcePipeline) computePressureForcePipeline->release();
    if (computeElasticForcesPipeline) computeElasticForcesPipeline->release();
    if (integratePipeline) integratePipeline->release();
    if (clearMembraneBuffersPipeline) clearMembraneBuffersPipeline->release();
    if (computeMembranesPipeline) computeMembranesPipeline->release();
    if (computeMembranesFinalizePipeline) computeMembranesFinalizePipeline->release();
    
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
            std::cerr << "Warning: Metal function not found: " << name << std::endl;
            return nullptr;
        }
        
        MTL::ComputePipelineState* pipeline = device->newComputePipelineState(func, &error);
        func->release();
        
        if (!pipeline) {
            std::cerr << "Warning: Failed to create pipeline for: " << name << std::endl;
            return nullptr;
        }
        
        return pipeline;
    };
    
    clearBuffersPipeline = createPipeline("clearBuffers");
    hashParticlesPipeline = createPipeline("hashParticles");
    sortPipeline = nullptr;  // Sorting done on CPU for now
    findNeighborsPipeline = createPipeline("findNeighbors");
    computeDensityPipeline = createPipeline("pcisph_computeDensity");
    computeForcesPipeline = createPipeline("pcisph_computeForcesAndInitPressure");
    predictPositionsPipeline = createPipeline("pcisph_predictPositions");
    predictDensityPipeline = createPipeline("pcisph_predictDensity");
    correctPressurePipeline = createPipeline("pcisph_correctPressure");
    computePressureForcePipeline = createPipeline("pcisph_computePressureForceAcceleration");
    computeElasticForcesPipeline = createPipeline("pcisph_computeElasticForces");
    integratePipeline = createPipeline("pcisph_integrate");
    clearMembraneBuffersPipeline = createPipeline("clearMembraneBuffers");
    computeMembranesPipeline = createPipeline("computeInteractionWithMembranes");
    computeMembranesFinalizePipeline = createPipeline("computeInteractionWithMembranes_finalize");
    
    std::cout << "Created Metal compute pipelines" << std::endl;
}

void owMetalSolver::createBuffers(
    const float* position_cpp,
    const float* velocity_cpp,
    owConfigProperty* config,
    const float* elasticConnectionsData_cpp
) {
    particleCount = config->getParticleCount();
    gridCellCount = config->gridCellCount;
    
    // Setup simulation parameters
    params.h = config->getConst("h");
    params.mass = config->getConst("mass");
    params.simulationScale = config->getConst("simulationScale");
    params.timeStep = config->getTimeStep();
    params.viscosity = config->getConst("viscosity");
    params.surfaceTension = 0.0f;
    params.gravity = config->getConst("gravity_y");  // Y is down
    params.particleCount = particleCount;
    params.gridCellCount = gridCellCount;
    params.gridMinX = config->xmin;
    params.gridMinY = config->ymin;
    params.gridMinZ = config->zmin;
    params.gridMaxX = config->xmax;
    params.gridMaxY = config->ymax;
    params.gridMaxZ = config->zmax;
    params.gridResX = config->gridCellsX;
    params.gridResY = config->gridCellsY;
    params.gridResZ = config->gridCellsZ;
    params.cellSize = config->getConst("h");
    params.rho0 = config->getConst("rho0");
    params.delta = config->getDelta();
    params.pcisphIterations = 3;
    
    // Create buffers
    size_t float4Size = 4 * sizeof(float);
    size_t float2Size = 2 * sizeof(float);
    
    positionBuffer = device->newBuffer(position_cpp, particleCount * float4Size, MTL::ResourceStorageModeShared);
    velocityBuffer = device->newBuffer(velocity_cpp, particleCount * float4Size, MTL::ResourceStorageModeShared);
    accelerationBuffer = device->newBuffer(particleCount * float4Size, MTL::ResourceStorageModeShared);
    positionPredictedBuffer = device->newBuffer(particleCount * float4Size, MTL::ResourceStorageModeShared);
    velocityPredictedBuffer = device->newBuffer(particleCount * float4Size, MTL::ResourceStorageModeShared);
    pressureBuffer = device->newBuffer(particleCount * sizeof(float) * 2, MTL::ResourceStorageModeShared);  // Extra for membrane handling
    rhoBuffer = device->newBuffer(particleCount * float2Size, MTL::ResourceStorageModeShared);
    
    neighborMapBuffer = device->newBuffer(particleCount * MAX_NEIGHBOR_COUNT * sizeof(int), MTL::ResourceStorageModeShared);
    neighborCountBuffer = device->newBuffer(particleCount * sizeof(int), MTL::ResourceStorageModeShared);
    
    cellStartBuffer = device->newBuffer(gridCellCount * sizeof(int), MTL::ResourceStorageModeShared);
    cellEndBuffer = device->newBuffer(gridCellCount * sizeof(int), MTL::ResourceStorageModeShared);
    
    particleIndexBuffer = device->newBuffer(particleCount * sizeof(unsigned int) * 2, MTL::ResourceStorageModeShared);
    
    // Initialize particleTypeBuffer from position.w (4th float component)
    std::vector<int> particleTypes(particleCount);
    for (unsigned int i = 0; i < particleCount; i++) {
        particleTypes[i] = (int)position_cpp[i * 4 + 3];  // w component = particle type
    }
    particleTypeBuffer = device->newBuffer(particleTypes.data(), particleCount * sizeof(int), MTL::ResourceStorageModeShared);
    
    if (elasticConnectionsData_cpp && config->numOfElasticP > 0) {
        elasticConnectionsBuffer = device->newBuffer(elasticConnectionsData_cpp, 
            config->numOfElasticP * MAX_NEIGHBOR_COUNT * float4Size, MTL::ResourceStorageModeShared);
    } else {
        elasticConnectionsBuffer = device->newBuffer(particleCount * float4Size, MTL::ResourceStorageModeShared);
    }
    
    muscleActivationBuffer = device->newBuffer(config->MUSCLE_COUNT * sizeof(float), MTL::ResourceStorageModeShared);
    paramsBuffer = device->newBuffer(&params, sizeof(SimulationParams), MTL::ResourceStorageModeShared);
    
    std::cout << "Created Metal buffers for " << particleCount << " particles" << std::endl;
    std::cout << "[Metal DEBUG] params.h = " << params.h << ", params.mass = " << params.mass << ", params.rho0 = " << params.rho0 << std::endl;
}

// ============================================================================
// Helper for kernel dispatch
// ============================================================================

void owMetalSolver::dispatchKernel(MTL::ComputePipelineState* pipeline, unsigned int count) {
    if (!pipeline) return;
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(pipeline);
    
    NS::UInteger threadGroupSize = pipeline->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > count) threadGroupSize = count;
    
    MTL::Size gridSize = MTL::Size(count, 1, 1);
    MTL::Size groupSize = MTL::Size(threadGroupSize, 1, 1);
    
    encoder->dispatchThreads(gridSize, groupSize);
    encoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}

// ============================================================================
// Reset
// ============================================================================

void owMetalSolver::reset(
    const float* position,
    const float* velocity,
    owConfigProperty* config,
    const float* elasticConnections,
    const int* membraneData,
    const int* particleMembranesList
) {
    // Copy new data to buffers
    std::memcpy(positionBuffer->contents(), position, particleCount * 4 * sizeof(float));
    std::memcpy(velocityBuffer->contents(), velocity, particleCount * 4 * sizeof(float));
    
    if (elasticConnections && config->numOfElasticP > 0) {
        std::memcpy(elasticConnectionsBuffer->contents(), elasticConnections,
            config->numOfElasticP * MAX_NEIGHBOR_COUNT * 4 * sizeof(float));
    }
    
    // Update params
    params.timeStep = config->getTimeStep();
    std::memcpy(paramsBuffer->contents(), &params, sizeof(SimulationParams));
}

// ============================================================================
// Kernel Implementations (owISolver interface)
// ============================================================================

unsigned int owMetalSolver::_runClearBuffers(owConfigProperty* config) {
    // Clear grid cell buffers
    std::memset(cellStartBuffer->contents(), -1, gridCellCount * sizeof(int));
    std::memset(cellEndBuffer->contents(), -1, gridCellCount * sizeof(int));
    return 0;
}

unsigned int owMetalSolver::_runHashParticles(owConfigProperty* config) {
    if (!hashParticlesPipeline) return 1;
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(hashParticlesPipeline);
    encoder->setBuffer(positionBuffer, 0, 0);
    encoder->setBuffer(particleIndexBuffer, 0, 1);
    encoder->setBuffer(paramsBuffer, 0, 2);
    
    NS::UInteger threadGroupSize = hashParticlesPipeline->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > particleCount) threadGroupSize = particleCount;
    
    encoder->dispatchThreads(MTL::Size(particleCount, 1, 1), MTL::Size(threadGroupSize, 1, 1));
    encoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    return 0;
}

void owMetalSolver::_runSort(owConfigProperty* config) {
    // CPU-based radix sort for now (Metal parallel sort is complex)
    // This matches OpenCL behavior which also does partial CPU sort
    unsigned int* indexData = (unsigned int*)particleIndexBuffer->contents();
    
    // Simple counting sort by cell index
    std::vector<std::pair<unsigned int, unsigned int>> pairs(particleCount);
    for (unsigned int i = 0; i < particleCount; i++) {
        pairs[i] = {indexData[i * 2], indexData[i * 2 + 1]};  // cell, particle
    }
    
    std::sort(pairs.begin(), pairs.end());
    
    for (unsigned int i = 0; i < particleCount; i++) {
        indexData[i * 2] = pairs[i].first;
        indexData[i * 2 + 1] = pairs[i].second;
    }
}

unsigned int owMetalSolver::_runSortPostPass(owConfigProperty* config) {
    // Build cell start/end indices
    unsigned int* indexData = (unsigned int*)particleIndexBuffer->contents();
    int* cellStart = (int*)cellStartBuffer->contents();
    int* cellEnd = (int*)cellEndBuffer->contents();
    
    std::memset(cellStart, -1, gridCellCount * sizeof(int));
    std::memset(cellEnd, -1, gridCellCount * sizeof(int));
    
    for (unsigned int i = 0; i < particleCount; i++) {
        unsigned int cell = indexData[i * 2];
        if (cell < gridCellCount) {
            if (cellStart[cell] == -1) {
                cellStart[cell] = i;
            }
            cellEnd[cell] = i + 1;
        }
    }
    
    // Debug: check cell occupancy
    static int debugOnce = 0;
    if (debugOnce++ < 1) {
        int nonEmptyCells = 0;
        int maxOccupancy = 0;
        for (unsigned int c = 0; c < gridCellCount; c++) {
            if (cellStart[c] != -1) {
                nonEmptyCells++;
                int occ = cellEnd[c] - cellStart[c];
                if (occ > maxOccupancy) maxOccupancy = occ;
            }
        }
        std::cout << "[Metal DEBUG] Non-empty cells: " << nonEmptyCells << " / " << gridCellCount << std::endl;
        std::cout << "[Metal DEBUG] Max cell occupancy: " << maxOccupancy << std::endl;
    }
    
    return 0;
}

unsigned int owMetalSolver::_runIndexx(owConfigProperty* config) {
    // Index post-pass already handled in _runSortPostPass
    return 0;
}

void owMetalSolver::_runIndexPostPass(owConfigProperty* config) {
    // No-op for Metal, handled in sort post pass
}

unsigned int owMetalSolver::_runFindNeighbors(owConfigProperty* config) {
    if (!findNeighborsPipeline) return 1;
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(findNeighborsPipeline);
    encoder->setBuffer(positionBuffer, 0, 0);
    encoder->setBuffer(cellStartBuffer, 0, 1);
    encoder->setBuffer(cellEndBuffer, 0, 2);
    encoder->setBuffer(neighborMapBuffer, 0, 3);
    encoder->setBuffer(neighborCountBuffer, 0, 4);
    encoder->setBuffer(particleIndexBuffer, 0, 5);
    encoder->setBuffer(paramsBuffer, 0, 6);
    
    NS::UInteger threadGroupSize = findNeighborsPipeline->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > particleCount) threadGroupSize = particleCount;
    
    encoder->dispatchThreads(MTL::Size(particleCount, 1, 1), MTL::Size(threadGroupSize, 1, 1));
    encoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    return 0;
}

unsigned int owMetalSolver::_run_pcisph_computeDensity(owConfigProperty* config) {
    if (!computeDensityPipeline) return 1;
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(computeDensityPipeline);
    encoder->setBuffer(positionBuffer, 0, 0);
    encoder->setBuffer(velocityBuffer, 0, 1);  // Not used but expected by shader
    encoder->setBuffer(rhoBuffer, 0, 2);
    encoder->setBuffer(neighborMapBuffer, 0, 3);
    encoder->setBuffer(neighborCountBuffer, 0, 4);
    encoder->setBuffer(paramsBuffer, 0, 5);
    
    NS::UInteger threadGroupSize = computeDensityPipeline->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > particleCount) threadGroupSize = particleCount;
    
    encoder->dispatchThreads(MTL::Size(particleCount, 1, 1), MTL::Size(threadGroupSize, 1, 1));
    encoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    return 0;
}

unsigned int owMetalSolver::_run_pcisph_computeForcesAndInitPressure(owConfigProperty* config) {
    if (!computeForcesPipeline) return 1;
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(computeForcesPipeline);
    encoder->setBuffer(positionBuffer, 0, 0);
    encoder->setBuffer(velocityBuffer, 0, 1);
    encoder->setBuffer(accelerationBuffer, 0, 2);
    encoder->setBuffer(pressureBuffer, 0, 3);
    encoder->setBuffer(rhoBuffer, 0, 4);
    encoder->setBuffer(neighborMapBuffer, 0, 5);
    encoder->setBuffer(neighborCountBuffer, 0, 6);
    encoder->setBuffer(particleTypeBuffer, 0, 7);
    encoder->setBuffer(paramsBuffer, 0, 8);
    
    NS::UInteger threadGroupSize = computeForcesPipeline->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > particleCount) threadGroupSize = particleCount;
    
    encoder->dispatchThreads(MTL::Size(particleCount, 1, 1), MTL::Size(threadGroupSize, 1, 1));
    encoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    return 0;
}

unsigned int owMetalSolver::_run_pcisph_computeElasticForces(owConfigProperty* config) {
    if (!computeElasticForcesPipeline || config->numOfElasticP == 0) return 0;
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(computeElasticForcesPipeline);
    encoder->setBuffer(positionBuffer, 0, 0);
    encoder->setBuffer(velocityBuffer, 0, 1);
    encoder->setBuffer(accelerationBuffer, 0, 2);
    encoder->setBuffer(elasticConnectionsBuffer, 0, 3);
    encoder->setBuffer(muscleActivationBuffer, 0, 4);
    encoder->setBuffer(particleTypeBuffer, 0, 5);
    encoder->setBuffer(paramsBuffer, 0, 6);
    
    NS::UInteger threadGroupSize = computeElasticForcesPipeline->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > config->numOfElasticP) threadGroupSize = config->numOfElasticP;
    
    encoder->dispatchThreads(MTL::Size(config->numOfElasticP, 1, 1), MTL::Size(threadGroupSize, 1, 1));
    encoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    return 0;
}

unsigned int owMetalSolver::_run_pcisph_predictPositions(owConfigProperty* config) {
    if (!predictPositionsPipeline) return 1;
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(predictPositionsPipeline);
    encoder->setBuffer(positionBuffer, 0, 0);
    encoder->setBuffer(velocityBuffer, 0, 1);
    encoder->setBuffer(accelerationBuffer, 0, 2);
    encoder->setBuffer(positionPredictedBuffer, 0, 3);
    encoder->setBuffer(particleTypeBuffer, 0, 4);
    encoder->setBuffer(paramsBuffer, 0, 5);
    
    NS::UInteger threadGroupSize = predictPositionsPipeline->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > particleCount) threadGroupSize = particleCount;
    
    encoder->dispatchThreads(MTL::Size(particleCount, 1, 1), MTL::Size(threadGroupSize, 1, 1));
    encoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    return 0;
}

unsigned int owMetalSolver::_run_pcisph_predictDensity(owConfigProperty* config) {
    if (!predictDensityPipeline) return 1;
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(predictDensityPipeline);
    encoder->setBuffer(positionPredictedBuffer, 0, 0);
    encoder->setBuffer(rhoBuffer, 0, 1);
    encoder->setBuffer(neighborMapBuffer, 0, 2);
    encoder->setBuffer(neighborCountBuffer, 0, 3);
    encoder->setBuffer(paramsBuffer, 0, 4);
    
    NS::UInteger threadGroupSize = predictDensityPipeline->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > particleCount) threadGroupSize = particleCount;
    
    encoder->dispatchThreads(MTL::Size(particleCount, 1, 1), MTL::Size(threadGroupSize, 1, 1));
    encoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    return 0;
}

unsigned int owMetalSolver::_run_pcisph_correctPressure(owConfigProperty* config) {
    if (!correctPressurePipeline) return 1;
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(correctPressurePipeline);
    encoder->setBuffer(pressureBuffer, 0, 0);
    encoder->setBuffer(rhoBuffer, 0, 1);
    encoder->setBuffer(paramsBuffer, 0, 2);
    
    NS::UInteger threadGroupSize = correctPressurePipeline->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > particleCount) threadGroupSize = particleCount;
    
    encoder->dispatchThreads(MTL::Size(particleCount, 1, 1), MTL::Size(threadGroupSize, 1, 1));
    encoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    return 0;
}

unsigned int owMetalSolver::_run_pcisph_computePressureForceAcceleration(owConfigProperty* config) {
    if (!computePressureForcePipeline) return 1;
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(computePressureForcePipeline);
    encoder->setBuffer(positionBuffer, 0, 0);
    encoder->setBuffer(accelerationBuffer, 0, 1);
    encoder->setBuffer(pressureBuffer, 0, 2);
    encoder->setBuffer(rhoBuffer, 0, 3);
    encoder->setBuffer(neighborMapBuffer, 0, 4);
    encoder->setBuffer(neighborCountBuffer, 0, 5);
    encoder->setBuffer(particleTypeBuffer, 0, 6);
    encoder->setBuffer(paramsBuffer, 0, 7);
    
    NS::UInteger threadGroupSize = computePressureForcePipeline->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > particleCount) threadGroupSize = particleCount;
    
    encoder->dispatchThreads(MTL::Size(particleCount, 1, 1), MTL::Size(threadGroupSize, 1, 1));
    encoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    return 0;
}

unsigned int owMetalSolver::_run_pcisph_integrate(int iterationCount, int mode, owConfigProperty* config) {
    if (!integratePipeline) return 1;
    
    // Debug: check acceleration, velocity, density, and neighbor count values before integrate
    static int debugCount = 0;
    if (debugCount < 3) {
        float* acc = (float*)accelerationBuffer->contents();
        float* vel = (float*)velocityBuffer->contents();
        float* rho = (float*)rhoBuffer->contents();
        int* ncount = (int*)neighborCountBuffer->contents();
        std::cout << "[Metal DEBUG] Before integrate:" << std::endl;
        std::cout << "  Particle 1000 acc: (" << acc[4000] << ", " << acc[4001] << ", " << acc[4002] << ")" << std::endl;
        std::cout << "  Particle 1000 vel: (" << vel[4000] << ", " << vel[4001] << ", " << vel[4002] << ")" << std::endl;
        std::cout << "  Particle 1000 rho: " << rho[2000] << " (1/rho: " << rho[2001] << ")" << std::endl;
        std::cout << "  Particle 1000 neighbors: " << ncount[1000] << std::endl;
        debugCount++;
    }
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    
    encoder->setComputePipelineState(integratePipeline);
    encoder->setBuffer(positionBuffer, 0, 0);
    encoder->setBuffer(velocityBuffer, 0, 1);
    encoder->setBuffer(accelerationBuffer, 0, 2);
    encoder->setBuffer(paramsBuffer, 0, 3);
    encoder->setBytes(&mode, sizeof(int), 4);
    
    NS::UInteger threadGroupSize = integratePipeline->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > particleCount) threadGroupSize = particleCount;
    
    encoder->dispatchThreads(MTL::Size(particleCount, 1, 1), MTL::Size(threadGroupSize, 1, 1));
    encoder->endEncoding();
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    return 0;
}

unsigned int owMetalSolver::_run_clearMembraneBuffers(owConfigProperty* config) {
    // Clear membrane-related parts of pressure buffer
    float* pressure = (float*)pressureBuffer->contents();
    for (unsigned int i = 0; i < particleCount; i++) {
        pressure[particleCount + i] = 0.0f;  // Second half is membrane handling
    }
    return 0;
}

unsigned int owMetalSolver::_run_computeInteractionWithMembranes(owConfigProperty* config) {
    // TODO: Implement membrane interaction kernel
    // For now, skip if no membranes
    if (config->numOfMembranes == 0) return 0;
    return 0;
}

unsigned int owMetalSolver::_run_computeInteractionWithMembranes_finalize(owConfigProperty* config) {
    // TODO: Implement membrane finalization kernel
    if (config->numOfMembranes == 0) return 0;
    return 0;
}

// ============================================================================
// Data Transfer
// ============================================================================

void owMetalSolver::updateMuscleActivityData(float* data, owConfigProperty* config) {
    std::memcpy(muscleActivationBuffer->contents(), data, config->MUSCLE_COUNT * sizeof(float));
}

void owMetalSolver::read_position_buffer(float* position, owConfigProperty* config) {
    std::memcpy(position, positionBuffer->contents(), particleCount * 4 * sizeof(float));
}

void owMetalSolver::read_velocity_buffer(float* velocity, owConfigProperty* config) {
    std::memcpy(velocity, velocityBuffer->contents(), particleCount * 4 * sizeof(float));
}

void owMetalSolver::read_density_buffer(float* density, owConfigProperty* config) {
    float* rho = (float*)rhoBuffer->contents();
    for (unsigned int i = 0; i < particleCount; i++) {
        density[i] = rho[i * 2];  // x component is density
    }
}

void owMetalSolver::read_particleIndex_buffer(unsigned int* particleIndex, owConfigProperty* config) {
    std::memcpy(particleIndex, particleIndexBuffer->contents(), particleCount * 2 * sizeof(unsigned int));
}

void owMetalSolver::read_pressure_buffer(float* pressure, owConfigProperty* config) {
    std::memcpy(pressure, pressureBuffer->contents(), particleCount * sizeof(float));
}

#endif // __APPLE__
