#pragma once

// Kernel argument abstraction for the `pcisph_predictPositions` kernel.
//
// OpenCL signature (sphFluid.cl):
//   __kernel void pcisph_predictPositions(
//       __global float4 *acceleration,         // arg 0  (3×N: non-pressure,
//       pressure, combined)
//       __global float4 *sortedPosition,        // arg 1  (2×N: current [0..N),
//       predicted [N..2N))
//       __global float4 *sortedVelocity,        // arg 2
//       __global uint2  *particleIndex,          // arg 3
//       __global uint   *particleIndexBack,      // arg 4
//       float gravitationalAccelerationX,                         // arg 5
//       float gravitationalAccelerationY,                         // arg 6
//       float gravitationalAccelerationZ,                         // arg 7
//       float simulationScaleInv,                // arg 8
//       float timeStep,                          // arg 9
//       __global float4 *originalPosition,               // arg 10
//       __global float4 *velocity,               // arg 11
//       float r0,                                // arg 12
//       __global float2 *neighborMap,            // arg 13
//       uint PARTICLE_COUNT                      // arg 14
//   )
//
// Metal signature (sphFluid.metal):
//   kernel void pcisph_predictPositions(
//       device float4 *acceleration             [[buffer(0)]],
//       device float4 *sortedPosition           [[buffer(1)]],
//       const device float4 *sortedVelocity     [[buffer(2)]],
//       const device uint2 *sortedCellAndSerialId [[buffer(3)]],
//       const device uint *sortedParticleIdBySerialId [[buffer(4)]],
//       constant float &gravitationalAccelerationX               [[buffer(5)]],
//       constant float &gravitationalAccelerationY               [[buffer(6)]],
//       constant float &gravitationalAccelerationZ               [[buffer(7)]],
//       constant float &simulationScaleInv      [[buffer(8)]],
//       constant float &timeStep                [[buffer(9)]],
//       const device float4 *originalPosition           [[buffer(10)]],
//       const device float4 *velocity           [[buffer(11)]],
//       constant float &r0                      [[buffer(12)]],
//       const device float2 *neighborMap        [[buffer(13)]],
//       constant uint &particleCount            [[buffer(14)]],
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

// ============ Backend-agnostic input ============
struct PcisphPredictPositionsInput {
  Float4Span acceleration;   // size: particleCount * 3
  Float4Span sortedPosition; // size: particleCount * 2 (in [0..N), out [N..2N))
  Float4Span sortedVelocity; // size: particleCount
  UInt2Span sortedCellAndSerialId;       // size: particleCount
  UInt32Span sortedParticleIdBySerialId; // size: particleCount
  Float4Span originalPosition;                   // size: particleCount
  Float4Span velocity;    // size: particleCount (boundary stores normals)
  Float2Span neighborMap; // size: particleCount * MAX_NEIGHBOR_COUNT
  float gravitationalAccelerationX;
  float gravitationalAccelerationY;
  float gravitationalAccelerationZ;
  float simulationScaleInv;
  float timeStep;
  float r0;
  uint32_t particleCount;
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct PcisphPredictPositionsMetalArgs {
  MTL::Buffer *acceleration;               // [[buffer(0)]]  in/out (3×N)
  MTL::Buffer *sortedPosition;             // [[buffer(1)]]  in/out (2×N)
  MTL::Buffer *sortedVelocity;             // [[buffer(2)]]
  MTL::Buffer *sortedCellAndSerialId;      // [[buffer(3)]]
  MTL::Buffer *sortedParticleIdBySerialId; // [[buffer(4)]]
  float gravitationalAccelerationX;        // [[buffer(5)]]
  float gravitationalAccelerationY;        // [[buffer(6)]]
  float gravitationalAccelerationZ;        // [[buffer(7)]]
  float simulationScaleInv;                // [[buffer(8)]]
  float timeStep;                          // [[buffer(9)]]
  MTL::Buffer *originalPosition;                   // [[buffer(10)]]
  MTL::Buffer *velocity;                   // [[buffer(11)]]
  float r0;                                // [[buffer(12)]]
  MTL::Buffer *neighborMap;                // [[buffer(13)]]
  uint32_t particleCount;                  // [[buffer(14)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, acceleration, 0);
    bindBuffer(enc, sortedPosition, 1);
    bindBuffer(enc, sortedVelocity, 2);
    bindBuffer(enc, sortedCellAndSerialId, 3);
    bindBuffer(enc, sortedParticleIdBySerialId, 4);
    bindScalar(enc, gravitationalAccelerationX, 5);
    bindScalar(enc, gravitationalAccelerationY, 6);
    bindScalar(enc, gravitationalAccelerationZ, 7);
    bindScalar(enc, simulationScaleInv, 8);
    bindScalar(enc, timeStep, 9);
    bindBuffer(enc, originalPosition, 10);
    bindBuffer(enc, velocity, 11);
    bindScalar(enc, r0, 12);
    bindBuffer(enc, neighborMap, 13);
    bindScalar(enc, particleCount, 14);
  }
};

inline PcisphPredictPositionsMetalArgs
toMetalArgs(const PcisphPredictPositionsInput &input, MTL::Device *device,
            MTL::Buffer *inOutAcceleration, MTL::Buffer *inOutSortedPosition) {
  PcisphPredictPositionsMetalArgs args{};
  args.acceleration = inOutAcceleration;
  args.sortedPosition = inOutSortedPosition;
  args.sortedVelocity = device->newBuffer(input.sortedVelocity.data(),
                                          input.sortedVelocity.size_bytes(),
                                          MTL::ResourceStorageModeShared);
  args.sortedCellAndSerialId = device->newBuffer(
      input.sortedCellAndSerialId.data(),
      input.sortedCellAndSerialId.size_bytes(), MTL::ResourceStorageModeShared);
  args.sortedParticleIdBySerialId =
      device->newBuffer(input.sortedParticleIdBySerialId.data(),
                        input.sortedParticleIdBySerialId.size_bytes(),
                        MTL::ResourceStorageModeShared);
  args.gravitationalAccelerationX = input.gravitationalAccelerationX;
  args.gravitationalAccelerationY = input.gravitationalAccelerationY;
  args.gravitationalAccelerationZ = input.gravitationalAccelerationZ;
  args.simulationScaleInv = input.simulationScaleInv;
  args.timeStep = input.timeStep;
  args.originalPosition =
      device->newBuffer(input.originalPosition.data(), input.originalPosition.size_bytes(),
                        MTL::ResourceStorageModeShared);
  args.velocity =
      device->newBuffer(input.velocity.data(), input.velocity.size_bytes(),
                        MTL::ResourceStorageModeShared);
  args.r0 = input.r0;
  args.neighborMap = device->newBuffer(input.neighborMap.data(),
                                       input.neighborMap.size_bytes(),
                                       MTL::ResourceStorageModeShared);
  args.particleCount = input.particleCount;
  return args;
}

#endif // SIBERNETIC_USE_METAL

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct PcisphPredictPositionsOpenCLArgs {
  cl::Buffer acceleration;               // arg 0
  cl::Buffer sortedPosition;             // arg 1
  cl::Buffer sortedVelocity;             // arg 2
  cl::Buffer sortedCellAndSerialId;      // arg 3
  cl::Buffer sortedParticleIdBySerialId; // arg 4
  float gravitationalAccelerationX;      // arg 5
  float gravitationalAccelerationY;      // arg 6
  float gravitationalAccelerationZ;      // arg 7
  float simulationScaleInv;              // arg 8
  float timeStep;                        // arg 9
  cl::Buffer originalPosition;                   // arg 10
  cl::Buffer velocity;                   // arg 11
  float r0;                              // arg 12
  cl::Buffer neighborMap;                // arg 13
  uint32_t particleCount;                // arg 14

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, acceleration, 0);
    bindBuffer(kernel, sortedPosition, 1);
    bindBuffer(kernel, sortedVelocity, 2);
    bindBuffer(kernel, sortedCellAndSerialId, 3);
    bindBuffer(kernel, sortedParticleIdBySerialId, 4);
    bindScalar(kernel, gravitationalAccelerationX, 5);
    bindScalar(kernel, gravitationalAccelerationY, 6);
    bindScalar(kernel, gravitationalAccelerationZ, 7);
    bindScalar(kernel, simulationScaleInv, 8);
    bindScalar(kernel, timeStep, 9);
    bindBuffer(kernel, originalPosition, 10);
    bindBuffer(kernel, velocity, 11);
    bindScalar(kernel, r0, 12);
    bindBuffer(kernel, neighborMap, 13);
    bindScalar(kernel, particleCount, 14);
  }
};

inline PcisphPredictPositionsOpenCLArgs
toOpenCLArgs(const PcisphPredictPositionsInput &input, cl::Context &context,
             cl::Buffer &inOutAcceleration, cl::Buffer &inOutSortedPosition) {
  cl_int err = CL_SUCCESS;
  PcisphPredictPositionsOpenCLArgs args{};
  args.acceleration = inOutAcceleration;
  args.sortedPosition = inOutSortedPosition;
  args.sortedVelocity =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 input.sortedVelocity.size_bytes(),
                 const_cast<HostFloat4 *>(input.sortedVelocity.data()), &err);
  args.sortedCellAndSerialId = cl::Buffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      input.sortedCellAndSerialId.size_bytes(),
      const_cast<HostUInt2 *>(input.sortedCellAndSerialId.data()), &err);
  args.sortedParticleIdBySerialId = cl::Buffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      input.sortedParticleIdBySerialId.size_bytes(),
      const_cast<uint32_t *>(input.sortedParticleIdBySerialId.data()), &err);
  args.gravitationalAccelerationX = input.gravitationalAccelerationX;
  args.gravitationalAccelerationY = input.gravitationalAccelerationY;
  args.gravitationalAccelerationZ = input.gravitationalAccelerationZ;
  args.simulationScaleInv = input.simulationScaleInv;
  args.timeStep = input.timeStep;
  args.originalPosition =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 input.originalPosition.size_bytes(),
                 const_cast<HostFloat4 *>(input.originalPosition.data()), &err);
  args.velocity =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 input.velocity.size_bytes(),
                 const_cast<HostFloat4 *>(input.velocity.data()), &err);
  args.r0 = input.r0;
  args.neighborMap =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 input.neighborMap.size_bytes(),
                 const_cast<HostFloat2 *>(input.neighborMap.data()), &err);
  args.particleCount = input.particleCount;
  return args;
}

#endif // SIBERNETIC_USE_OPENCL

// ============ Constants ============
inline constexpr const char *kPcisphPredictPositionsKernelName =
    "pcisph_predictPositions";

} // namespace Sibernetic
