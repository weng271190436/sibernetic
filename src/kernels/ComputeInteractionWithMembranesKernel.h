#pragma once

// Kernel argument abstractions for the `computeInteractionWithMembranes` and
// `computeInteractionWithMembranes_finalize` kernels.
//
// computeInteractionWithMembranes: For each liquid particle, finds neighboring
// elastic (membrane) particles, projects the particle onto membrane planes,
// and accumulates a position correction into position[N + serialId].
//
// computeInteractionWithMembranes_finalize: Applies the accumulated delta
// from position[N + serialId] to position[serialId].
//
// OpenCL signatures (sphFluid.cl):
//   __kernel void computeInteractionWithMembranes(
//       __global float4 *position,              // arg 0  (read + write delta)
//       __global float4 *velocity,              // arg 1
//       __global float4 *sortedPosition,        // arg 2  (unused)
//       __global uint2  *particleIndex,         // arg 3
//       __global uint   *particleIndexBack,     // arg 4
//       __global float2 *neighborMap,           // arg 5
//       __global int    *particleMembranesList, // arg 6
//       __global int    *membraneData,          // arg 7
//       int PARTICLE_COUNT,                     // arg 8
//       int numOfElasticP,                      // arg 9  (unused)
//       float r0                                // arg 10
//   )
//   __kernel void computeInteractionWithMembranes_finalize(
//       __global float4 *position,              // arg 0
//       __global float4 *velocity,              // arg 1  (unused)
//       __global uint2  *particleIndex,         // arg 2
//       __global uint   *particleIndexBack,     // arg 3
//       int PARTICLE_COUNT                      // arg 4
//   )
//
// Metal signatures (sphFluid.metal):
//   kernel void computeInteractionWithMembranes(
//       device float4 *position                      [[buffer(0)]],
//       const device float4 *velocity                [[buffer(1)]],
//       const device uint2 *sortedCellAndSerialId    [[buffer(2)]],
//       const device uint *sortedParticleIdBySerialId [[buffer(3)]],
//       const device float2 *neighborMap             [[buffer(4)]],
//       const device int *particleMembranesList      [[buffer(5)]],
//       const device int *membraneData               [[buffer(6)]],
//       constant uint &particleCount                 [[buffer(7)]],
//       constant float &r0                           [[buffer(8)]],
//       uint serialId [[thread_position_in_grid]])
//
//   kernel void computeInteractionWithMembranes_finalize(
//       device float4 *position                       [[buffer(0)]],
//       const device uint *sortedParticleIdBySerialId [[buffer(1)]],
//       constant uint &particleCount                  [[buffer(2)]],
//       uint serialId [[thread_position_in_grid]])

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

// Maximum membranes sharing a single elastic particle vertex.
inline constexpr int kMaxMembranesIncludingSameParticle = 7;

// ============ computeInteractionWithMembranes ============

struct ComputeInteractionWithMembranesInput {
  Float4Span position;                   // size: 2 × particleCount (in/out)
  Float4Span velocity;                   // size: 2 × particleCount
  UInt2Span sortedCellAndSerialId;       // size: particleCount
  UInt32Span sortedParticleIdBySerialId; // size: particleCount
  Float2Span neighborMap;   // size: particleCount × kMaxNeighborCount
  Int32Span particleMembranesList; // size: numOfElasticP × 7
  Int32Span membraneData;          // size: numMembranes × 3
  uint32_t particleCount;
  float r0;
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct ComputeInteractionWithMembranesMetalArgs {
  MTL::Buffer *position;                   // [[buffer(0)]]
  MTL::Buffer *velocity;                   // [[buffer(1)]]
  MTL::Buffer *sortedCellAndSerialId;      // [[buffer(2)]]
  MTL::Buffer *sortedParticleIdBySerialId; // [[buffer(3)]]
  MTL::Buffer *neighborMap;                // [[buffer(4)]]
  MTL::Buffer *particleMembranesList;      // [[buffer(5)]]
  MTL::Buffer *membraneData;               // [[buffer(6)]]
  uint32_t particleCount;                  // [[buffer(7)]]
  float r0;                                // [[buffer(8)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, position, 0);
    bindBuffer(enc, velocity, 1);
    bindBuffer(enc, sortedCellAndSerialId, 2);
    bindBuffer(enc, sortedParticleIdBySerialId, 3);
    bindBuffer(enc, neighborMap, 4);
    bindBuffer(enc, particleMembranesList, 5);
    bindBuffer(enc, membraneData, 6);
    bindScalar(enc, particleCount, 7);
    bindScalar(enc, r0, 8);
  }
};

#endif

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct ComputeInteractionWithMembranesOpenCLArgs {
  cl::Buffer position;              // arg 0
  cl::Buffer velocity;              // arg 1
  cl::Buffer sortedPosition;        // arg 2  (unused)
  cl::Buffer particleIndex;         // arg 3
  cl::Buffer particleIndexBack;     // arg 4
  cl::Buffer neighborMap;           // arg 5
  cl::Buffer particleMembranesList; // arg 6
  cl::Buffer membraneData;          // arg 7
  int32_t particleCount;            // arg 8
  int32_t numOfElasticP;            // arg 9  (unused)
  float r0;                         // arg 10

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, position, 0);
    bindBuffer(kernel, velocity, 1);
    bindBuffer(kernel, sortedPosition, 2);
    bindBuffer(kernel, particleIndex, 3);
    bindBuffer(kernel, particleIndexBack, 4);
    bindBuffer(kernel, neighborMap, 5);
    bindBuffer(kernel, particleMembranesList, 6);
    bindBuffer(kernel, membraneData, 7);
    bindScalar(kernel, particleCount, 8);
    bindScalar(kernel, numOfElasticP, 9);
    bindScalar(kernel, r0, 10);
  }
};

#endif

inline constexpr const char *kComputeInteractionWithMembranesKernelName =
    "computeInteractionWithMembranes";

// ============ computeInteractionWithMembranes_finalize ============

struct ComputeInteractionWithMembranesFinalizeInput {
  Float4Span position;                   // size: 2 × particleCount (in/out)
  UInt32Span sortedParticleIdBySerialId; // size: particleCount
  uint32_t particleCount;
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct ComputeInteractionWithMembranesFinalizeMetalArgs {
  MTL::Buffer *position;                   // [[buffer(0)]]
  MTL::Buffer *sortedParticleIdBySerialId; // [[buffer(1)]]
  uint32_t particleCount;                  // [[buffer(2)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, position, 0);
    bindBuffer(enc, sortedParticleIdBySerialId, 1);
    bindScalar(enc, particleCount, 2);
  }
};

#endif

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct ComputeInteractionWithMembranesFinalizeOpenCLArgs {
  cl::Buffer position;          // arg 0
  cl::Buffer velocity;          // arg 1  (unused)
  cl::Buffer particleIndex;     // arg 2
  cl::Buffer particleIndexBack; // arg 3
  int32_t particleCount;        // arg 4

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, position, 0);
    bindBuffer(kernel, velocity, 1);
    bindBuffer(kernel, particleIndex, 2);
    bindBuffer(kernel, particleIndexBack, 3);
    bindScalar(kernel, particleCount, 4);
  }
};

#endif

inline constexpr const char *
    kComputeInteractionWithMembranesFinalizeKernelName =
        "computeInteractionWithMembranes_finalize";

} // namespace Sibernetic
