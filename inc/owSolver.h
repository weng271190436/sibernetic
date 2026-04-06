/*******************************************************************************
 * The MIT License (MIT)
 * Copyright (c) 2011, 2013 OpenWorm.
 * Solver interface - allows switching between OpenCL and Metal backends
 ******************************************************************************/
#ifndef OW_SOLVER_H_
#define OW_SOLVER_H_

#include "owConfigProperty.h"

// Forward declarations
namespace cl { class Buffer; }

/**
 * Abstract solver interface for SPH simulation
 * Implemented by owOpenCLSolver and owMetalSolver
 */
class owISolver {
public:
    virtual ~owISolver() = default;
    
    // Buffer management
    virtual void reset(
        const float* position,
        const float* velocity,
        owConfigProperty* config,
        const float* elasticConnections,
        const int* membraneData,
        const int* particleMembranesList
    ) = 0;
    
    // Kernel execution
    virtual void _runHashParticles(owConfigProperty* config) = 0;
    virtual void _runSort(owConfigProperty* config) = 0;
    virtual void _runSortPostPass(owConfigProperty* config) = 0;
    virtual void _runIndexx(owConfigProperty* config) = 0;
    virtual void _runIndexPostPass(owConfigProperty* config) = 0;
    virtual void _runFindNeighbors(owConfigProperty* config) = 0;
    virtual void _run_pcisph_computeDensity(owConfigProperty* config) = 0;
    virtual void _run_pcisph_computeForcesAndInitPressure(owConfigProperty* config) = 0;
    virtual void _run_pcisph_computeElasticForces(owConfigProperty* config) = 0;
    virtual void _run_pcisph_predictPositions(owConfigProperty* config) = 0;
    virtual void _run_pcisph_predictDensity(owConfigProperty* config) = 0;
    virtual void _run_pcisph_correctPressure(owConfigProperty* config) = 0;
    virtual void _run_pcisph_computePressureForceAcceleration(owConfigProperty* config) = 0;
    virtual void _run_pcisph_integrate(int iteration, int mode, owConfigProperty* config) = 0;
    virtual void _run_clearMembraneBuffers(owConfigProperty* config) = 0;
    virtual void _run_computeInteractionWithMembranes(owConfigProperty* config) = 0;
    virtual void _run_computeInteractionWithMembranes_finalize(owConfigProperty* config) = 0;
    
    // Data transfer
    virtual void updateMuscleActivityData(float* data, owConfigProperty* config) = 0;
    virtual void read_position_buffer(float* position, owConfigProperty* config) = 0;
    virtual void read_density_buffer(float* density, owConfigProperty* config) = 0;
    virtual void read_particleIndex_buffer(unsigned int* particleIndex, owConfigProperty* config) = 0;
    virtual void read_velocity_buffer(float* velocity, owConfigProperty* config) = 0;
    virtual void read_elapsed_time(float* elapsed, owConfigProperty* config) = 0;
    
    // For compatibility - some code accesses buffers directly
    // Metal solver will need to provide equivalent
    virtual void* getPositionBuffer() = 0;
    virtual void* getVelocityBuffer() = 0;
    virtual void* getDensityBuffer() = 0;
    
    // Copy buffer helper - different signature for each backend
    virtual void copy_buffer_to_device(const void* host, void* device_buffer, size_t size) = 0;
    virtual void copy_buffer_from_device(void* host, void* device_buffer, size_t size) = 0;
};

#endif // OW_SOLVER_H_
