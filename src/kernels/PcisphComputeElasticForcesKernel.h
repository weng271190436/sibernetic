#pragma once

// Kernel argument abstraction for the `pcisph_computeElasticForces` kernel.
//
// Computes elastic spring forces (Hooke's law) and muscle activation forces
// for the worm body simulation. Thread index is the elastic particle index
// (0..numOfElasticP-1).
//
// OpenCL signature (sphFluid.cl):
//   __kernel void pcisph_computeElasticForces(
//       __global float2 *neighborMap,            // arg 0  (unused)
//       __global float4 *sortedPosition,         // arg 1
//       __global float4 *sortedVelocity,         // arg 2  (unused)
//       __global float4 *acceleration,           // arg 3
//       __global uint   *particleIndexBack,      // arg 4
//       __global uint2  *particleIndex,          // arg 5
//       float max_muscle_force,                  // arg 6
//       float mass,                              // arg 7  (unused)
//       float simulationScale,                   // arg 8
//       uint numOfElasticP,                      // arg 9
//       __global float4 *elasticConnectionsData, // arg 10
//       uint PARTICLE_COUNT,                     // arg 11 (unused)
//       uint MUSCLE_COUNT,                       // arg 12
//       __global float *muscle_activation_signal,// arg 13
//       __global float4 *position,               // arg 14
//       float elasticityCoefficient              // arg 15
//   )
//
// Metal signature (sphFluid.metal):
//   kernel void pcisph_computeElasticForces(
//       const device float4 *sortedPosition               [[buffer(0)]],
//       device float4 *acceleration                       [[buffer(1)]],
//       const device uint *sortedParticleIdBySerialId     [[buffer(2)]],
//       const device uint2 *sortedCellAndSerialId         [[buffer(3)]],
//       constant float &maxMuscleForce                    [[buffer(4)]],
//       constant float &simulationScale                   [[buffer(5)]],
//       constant uint &numOfElasticP                      [[buffer(6)]],
//       const device float4 *elasticConnectionsData       [[buffer(7)]],
//       constant uint &muscleCount                        [[buffer(8)]],
//       const device float *muscleActivationSignal        [[buffer(9)]],
//       const device float4 *originalPosition             [[buffer(10)]],
//       constant float &elasticityCoefficient             [[buffer(11)]],
//       uint index [[thread_position_in_grid]])

#include <cstdint>

#include "../types/HostTypes.h"
#include "common/KernelArgs.h"

#ifdef SIBERNETIC_USE_METAL
#include "Metal/MTLBuffer.hpp"
#include "Metal/MTLComputeCommandEncoder.hpp"
#include "Metal/MTLDevice.hpp"
#endif

#ifdef SIBERNETIC_USE_OPENCL
#ifdef err_local
#undef err_local
#endif
#include "OpenCL/cl.hpp"
#endif

namespace Sibernetic {

// ============ Backend-agnostic input ============
struct PcisphComputeElasticForcesInput {
  Float4Span sortedPosition;             // size: particleCount
  Float4Span acceleration;               // size: particleCount (in/out)
  UInt32Span sortedParticleIdBySerialId; // size: particleCount
  UInt2Span sortedCellAndSerialId;       // size: particleCount
  float maxMuscleForce;
  float simulationScale;
  uint32_t numOfElasticP;
  Float4Span elasticConnectionsData; // size: numOfElasticP * kMaxNeighborCount
  uint32_t muscleCount;
  FloatSpan muscleActivationSignal; // size: muscleCount
  Float4Span originalPosition;      // size: particleCount (.w = type)
  float elasticityCoefficient;
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct PcisphComputeElasticForcesMetalArgs {
  MTL::Buffer *sortedPosition;             // [[buffer(0)]]
  MTL::Buffer *acceleration;               // [[buffer(1)]]  in/out
  MTL::Buffer *sortedParticleIdBySerialId; // [[buffer(2)]]
  MTL::Buffer *sortedCellAndSerialId;      // [[buffer(3)]]
  float maxMuscleForce;                    // [[buffer(4)]]
  float simulationScale;                   // [[buffer(5)]]
  uint32_t numOfElasticP;                  // [[buffer(6)]]
  MTL::Buffer *elasticConnectionsData;     // [[buffer(7)]]
  uint32_t muscleCount;                    // [[buffer(8)]]
  MTL::Buffer *muscleActivationSignal;     // [[buffer(9)]]
  MTL::Buffer *originalPosition;           // [[buffer(10)]]
  float elasticityCoefficient;             // [[buffer(11)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, sortedPosition, 0);
    bindBuffer(enc, acceleration, 1);
    bindBuffer(enc, sortedParticleIdBySerialId, 2);
    bindBuffer(enc, sortedCellAndSerialId, 3);
    bindScalar(enc, maxMuscleForce, 4);
    bindScalar(enc, simulationScale, 5);
    bindScalar(enc, numOfElasticP, 6);
    bindBuffer(enc, elasticConnectionsData, 7);
    bindScalar(enc, muscleCount, 8);
    bindBuffer(enc, muscleActivationSignal, 9);
    bindBuffer(enc, originalPosition, 10);
    bindScalar(enc, elasticityCoefficient, 11);
  }
};

inline PcisphComputeElasticForcesMetalArgs toMetalArgs(
    const PcisphComputeElasticForcesInput &input, MTL::Device *device,
    MTL::Buffer *sortedPosition, MTL::Buffer *inOutAcceleration,
    MTL::Buffer *sortedParticleIdBySerialId, MTL::Buffer *sortedCellAndSerialId,
    MTL::Buffer *elasticConnectionsData, MTL::Buffer *muscleActivationSignal,
    MTL::Buffer *originalPosition) {
  (void)device;
  PcisphComputeElasticForcesMetalArgs args{};
  args.sortedPosition = sortedPosition;
  args.acceleration = inOutAcceleration;
  args.sortedParticleIdBySerialId = sortedParticleIdBySerialId;
  args.sortedCellAndSerialId = sortedCellAndSerialId;
  args.maxMuscleForce = input.maxMuscleForce;
  args.simulationScale = input.simulationScale;
  args.numOfElasticP = input.numOfElasticP;
  args.elasticConnectionsData = elasticConnectionsData;
  args.muscleCount = input.muscleCount;
  args.muscleActivationSignal = muscleActivationSignal;
  args.originalPosition = originalPosition;
  args.elasticityCoefficient = input.elasticityCoefficient;
  return args;
}

#endif

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct PcisphComputeElasticForcesOpenCLArgs {
  cl::Buffer neighborMap;            // arg 0  (unused)
  cl::Buffer sortedPosition;         // arg 1
  cl::Buffer sortedVelocity;         // arg 2  (unused)
  cl::Buffer acceleration;           // arg 3
  cl::Buffer particleIndexBack;      // arg 4
  cl::Buffer particleIndex;          // arg 5
  float maxMuscleForce;              // arg 6
  float mass;                        // arg 7  (unused)
  float simulationScale;             // arg 8
  uint32_t numOfElasticP;            // arg 9
  cl::Buffer elasticConnectionsData; // arg 10
  uint32_t particleCount;            // arg 11 (unused)
  uint32_t muscleCount;              // arg 12
  cl::Buffer muscleActivationSignal; // arg 13
  cl::Buffer position;               // arg 14
  float elasticityCoefficient;       // arg 15

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, neighborMap, 0);
    bindBuffer(kernel, sortedPosition, 1);
    bindBuffer(kernel, sortedVelocity, 2);
    bindBuffer(kernel, acceleration, 3);
    bindBuffer(kernel, particleIndexBack, 4);
    bindBuffer(kernel, particleIndex, 5);
    bindScalar(kernel, maxMuscleForce, 6);
    bindScalar(kernel, mass, 7);
    bindScalar(kernel, simulationScale, 8);
    bindScalar(kernel, numOfElasticP, 9);
    bindBuffer(kernel, elasticConnectionsData, 10);
    bindScalar(kernel, particleCount, 11);
    bindScalar(kernel, muscleCount, 12);
    bindBuffer(kernel, muscleActivationSignal, 13);
    bindBuffer(kernel, position, 14);
    bindScalar(kernel, elasticityCoefficient, 15);
  }
};

inline PcisphComputeElasticForcesOpenCLArgs toOpenCLArgs(
    const PcisphComputeElasticForcesInput &input, cl::Context &context,
    cl::Buffer &sortedPosition, cl::Buffer &inOutAcceleration,
    cl::Buffer &sortedParticleIdBySerialId, cl::Buffer &sortedCellAndSerialId,
    cl::Buffer &elasticConnectionsData, cl::Buffer &muscleActivationSignal,
    cl::Buffer &originalPosition, uint32_t particleCount) {
  cl_int err = CL_SUCCESS;
  PcisphComputeElasticForcesOpenCLArgs args{};

  // neighborMap (arg 0): unused but required by OpenCL signature.
  float dummy2[2] = {0.0f, 0.0f};
  args.neighborMap =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(dummy2), &dummy2, &err);
  args.sortedPosition = sortedPosition;
  // sortedVelocity (arg 2): unused but required by OpenCL signature.
  float dummy4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  args.sortedVelocity =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(dummy4), &dummy4, &err);
  args.acceleration = inOutAcceleration;
  args.particleIndexBack = sortedParticleIdBySerialId;
  args.particleIndex = sortedCellAndSerialId;
  args.maxMuscleForce = input.maxMuscleForce;
  args.mass = 1.0f; // unused
  args.simulationScale = input.simulationScale;
  args.numOfElasticP = input.numOfElasticP;
  args.elasticConnectionsData = elasticConnectionsData;
  args.particleCount = particleCount;
  args.muscleCount = input.muscleCount;
  args.muscleActivationSignal = muscleActivationSignal;
  args.position = originalPosition;
  args.elasticityCoefficient = input.elasticityCoefficient;
  return args;
}

#endif

// ============ Constants ============
inline constexpr const char *kPcisphComputeElasticForcesKernelName =
    "pcisph_computeElasticForces";

} // namespace Sibernetic
