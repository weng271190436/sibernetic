/*******************************************************************************
 * The MIT License (MIT)
 * Copyright (c) 2011, 2013 OpenWorm.
 * Solver interface - allows switching between OpenCL and Metal backends
 ******************************************************************************/
#ifndef OW_SOLVER_H_
#define OW_SOLVER_H_

#include "owConfigProperty.h"

/**
 * Abstract solver interface for SPH simulation
 * Implemented by owOpenCLSolver and owMetalSolver
 */
class owISolver {
public:
    virtual ~owISolver() = default;
    
    // Initialization and reset
    virtual void reset(
        const float* position,
        const float* velocity,
        owConfigProperty* config,
        const float* elasticConnections = nullptr,
        const int* membraneData = nullptr,
        const int* particleMembranesList = nullptr
    ) = 0;
    
    // Neighbor search kernels
    virtual unsigned int _runClearBuffers(owConfigProperty* config) = 0;
    virtual unsigned int _runHashParticles(owConfigProperty* config) = 0;
    virtual void _runSort(owConfigProperty* config) = 0;
    virtual unsigned int _runSortPostPass(owConfigProperty* config) = 0;
    virtual unsigned int _runIndexx(owConfigProperty* config) = 0;
    virtual void _runIndexPostPass(owConfigProperty* config) = 0;
    virtual unsigned int _runFindNeighbors(owConfigProperty* config) = 0;
    
    // PCISPH physics kernels
    virtual unsigned int _run_pcisph_computeDensity(owConfigProperty* config) = 0;
    virtual unsigned int _run_pcisph_computeForcesAndInitPressure(owConfigProperty* config) = 0;
    virtual unsigned int _run_pcisph_computeElasticForces(owConfigProperty* config) = 0;
    virtual void _saveBaseAcceleration() = 0;  // Save base accel before PCISPH loop
    virtual unsigned int _run_pcisph_predictPositions(owConfigProperty* config) = 0;
    virtual unsigned int _run_pcisph_predictDensity(owConfigProperty* config) = 0;
    virtual unsigned int _run_pcisph_correctPressure(owConfigProperty* config) = 0;
    virtual unsigned int _run_pcisph_computePressureForceAcceleration(owConfigProperty* config) = 0;
    virtual unsigned int _run_pcisph_integrate(int iterationCount, int mode, owConfigProperty* config) = 0;
    
    // Membrane kernels
    virtual unsigned int _run_clearMembraneBuffers(owConfigProperty* config) = 0;
    virtual unsigned int _run_computeInteractionWithMembranes(owConfigProperty* config) = 0;
    virtual unsigned int _run_computeInteractionWithMembranes_finalize(owConfigProperty* config) = 0;
    
    // Data transfer
    virtual void updateMuscleActivityData(float* data, owConfigProperty* config) = 0;
    virtual void read_position_buffer(float* position, owConfigProperty* config) = 0;
    virtual void read_velocity_buffer(float* velocity, owConfigProperty* config) = 0;
    virtual void read_density_buffer(float* density, owConfigProperty* config) = 0;
    virtual void read_particleIndex_buffer(unsigned int* particleIndex, owConfigProperty* config) = 0;
    virtual void read_pressure_buffer(float* pressure, owConfigProperty* config) = 0;
};

// Factory function - creates appropriate solver based on build config
owISolver* createSolver(
    const float* position,
    const float* velocity,
    owConfigProperty* config,
    const float* elasticConnections = nullptr,
    const int* membraneData = nullptr,
    const int* particleMembranesList = nullptr
);

#endif // OW_SOLVER_H_
