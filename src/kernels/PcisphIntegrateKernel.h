#pragma once

// Kernel argument abstraction for the `pcisph_integrate` kernel.
//
// Performs leapfrog time integration for PCISPH. Has three modes:
//
//   iterationCount == 0:
//     Store combined acceleration: acceleration[2N + serialId] =
//       acceleration[sortedId] + acceleration[N + sortedId].
//     No position or velocity update.
//
//   mode == 0 (positions):
//     Leapfrog position step:
//       x(t+dt) = x(t) + v(t)*dt + a(t)*dt^2/2
//     Writes into sortedPosition[sortedId] in-place.
//
//   mode == 1 (velocities):
//     Leapfrog velocity step with boundary interaction:
//       v(t+dt) = v(t) + (a(t) + a(t+dt))*dt/2
//     Applies boundary correction, then writes velocity, position, and
//     stores combined acceleration for next step.
//
//   mode == 2 (semi-implicit Euler):
//       v(t+dt) = v(t) + a(t+dt)*dt
//       x(t+dt) = x(t) + v(t+dt)*dt
//     Applies boundary correction, then writes everything.
//
// OpenCL signature (sphFluid.cl):
//   __kernel void pcisph_integrate(
//       __global float4 *acceleration,         // arg 0  (3×N)
//       __global float4 *sortedPosition,       // arg 1
//       __global float4 *sortedVelocity,       // arg 2
//       __global uint2  *particleIndex,        // arg 3
//       __global uint   *particleIndexBack,    // arg 4
//       float gravity_x,                       // arg 5
//       float gravity_y,                       // arg 6
//       float gravity_z,                       // arg 7
//       float simulationScaleInv,              // arg 8
//       float timeStep,                        // arg 9
//       float xmin,                            // arg 10
//       float xmax,                            // arg 11
//       float ymin,                            // arg 12
//       float ymax,                            // arg 13
//       float zmin,                            // arg 14
//       float zmax,                            // arg 15
//       __global float4 *position,             // arg 16
//       __global float4 *velocity,             // arg 17
//       __global float  *rho,                  // arg 18 (unused)
//       float r0,                              // arg 19
//       __global float2 *neighborMap,          // arg 20
//       uint PARTICLE_COUNT,                   // arg 21
//       int iterationCount,                    // arg 22
//       int mode                               // arg 23
//   )
//
// Metal signature (sphFluid.metal):
//   kernel void pcisph_integrate(
//       device float4 *acceleration                      [[buffer(0)]],
//       device float4 *sortedPosition                    [[buffer(1)]],
//       const device float4 *sortedVelocity              [[buffer(2)]],
//       const device uint2 *sortedCellAndSerialId        [[buffer(3)]],
//       const device uint *sortedParticleIdBySerialId    [[buffer(4)]],
//       constant float &simulationScaleInv               [[buffer(5)]],
//       constant float &timeStep                         [[buffer(6)]],
//       device float4 *originalPosition                  [[buffer(7)]],
//       device float4 *velocity                          [[buffer(8)]],
//       constant float &r0                               [[buffer(9)]],
//       const device float2 *neighborMap                 [[buffer(10)]],
//       constant uint &particleCount                     [[buffer(11)]],
//       constant int &iterationCount                     [[buffer(12)]],
//       constant int &mode                               [[buffer(13)]],
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
struct PcisphIntegrateInput {
  Float4Span acceleration;   // size: particleCount * 3
  Float4Span sortedPosition; // size: particleCount
  Float4Span sortedVelocity; // size: particleCount
  UInt2Span sortedCellAndSerialId;       // size: particleCount
  UInt32Span sortedParticleIdBySerialId; // size: particleCount
  float simulationScaleInv;
  float timeStep;
  Float4Span originalPosition; // size: particleCount (in/out, .w = type)
  Float4Span velocity;         // size: particleCount (in/out)
  float r0;                    // boundary interaction radius
  Float2Span neighborMap;      // size: particleCount * 32
  uint32_t particleCount;
  int32_t iterationCount;
  int32_t mode;
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct PcisphIntegrateMetalArgs {
  MTL::Buffer *acceleration;               // [[buffer(0)]]  in/out (3×N)
  MTL::Buffer *sortedPosition;             // [[buffer(1)]]  in/out
  MTL::Buffer *sortedVelocity;             // [[buffer(2)]]
  MTL::Buffer *sortedCellAndSerialId;      // [[buffer(3)]]
  MTL::Buffer *sortedParticleIdBySerialId; // [[buffer(4)]]
  float simulationScaleInv;                // [[buffer(5)]]
  float timeStep;                          // [[buffer(6)]]
  MTL::Buffer *originalPosition;           // [[buffer(7)]]  in/out
  MTL::Buffer *velocity;                   // [[buffer(8)]]  in/out
  float r0;                                // [[buffer(9)]]
  MTL::Buffer *neighborMap;                // [[buffer(10)]]
  uint32_t particleCount;                  // [[buffer(11)]]
  int32_t iterationCount;                  // [[buffer(12)]]
  int32_t mode;                            // [[buffer(13)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, acceleration, 0);
    bindBuffer(enc, sortedPosition, 1);
    bindBuffer(enc, sortedVelocity, 2);
    bindBuffer(enc, sortedCellAndSerialId, 3);
    bindBuffer(enc, sortedParticleIdBySerialId, 4);
    bindScalar(enc, simulationScaleInv, 5);
    bindScalar(enc, timeStep, 6);
    bindBuffer(enc, originalPosition, 7);
    bindBuffer(enc, velocity, 8);
    bindScalar(enc, r0, 9);
    bindBuffer(enc, neighborMap, 10);
    bindScalar(enc, particleCount, 11);
    bindScalar(enc, iterationCount, 12);
    bindScalar(enc, mode, 13);
  }
};

inline PcisphIntegrateMetalArgs
toMetalArgs(const PcisphIntegrateInput &input, MTL::Device *device,
            MTL::Buffer *inOutAcceleration, MTL::Buffer *inOutSortedPosition,
            MTL::Buffer *inOutOriginalPosition, MTL::Buffer *inOutVelocity) {
  PcisphIntegrateMetalArgs args{};
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
  args.simulationScaleInv = input.simulationScaleInv;
  args.timeStep = input.timeStep;
  args.originalPosition = inOutOriginalPosition;
  args.velocity = inOutVelocity;
  args.r0 = input.r0;
  args.neighborMap = device->newBuffer(input.neighborMap.data(),
                                       input.neighborMap.size_bytes(),
                                       MTL::ResourceStorageModeShared);
  args.particleCount = input.particleCount;
  args.iterationCount = input.iterationCount;
  args.mode = input.mode;
  return args;
}

#endif // SIBERNETIC_USE_METAL

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct PcisphIntegrateOpenCLArgs {
  cl::Buffer acceleration;               // arg 0
  cl::Buffer sortedPosition;             // arg 1
  cl::Buffer sortedVelocity;             // arg 2
  cl::Buffer sortedCellAndSerialId;      // arg 3
  cl::Buffer sortedParticleIdBySerialId; // arg 4
  float gravityX;                        // arg 5
  float gravityY;                        // arg 6
  float gravityZ;                        // arg 7
  float simulationScaleInv;              // arg 8
  float timeStep;                        // arg 9
  float xmin;                            // arg 10
  float xmax;                            // arg 11
  float ymin;                            // arg 12
  float ymax;                            // arg 13
  float zmin;                            // arg 14
  float zmax;                            // arg 15
  cl::Buffer originalPosition;           // arg 16
  cl::Buffer velocity;                   // arg 17
  cl::Buffer rho;                        // arg 18 (unused)
  float r0;                              // arg 19
  cl::Buffer neighborMap;                // arg 20
  uint32_t particleCount;                // arg 21
  int32_t iterationCount;                // arg 22
  int32_t mode;                          // arg 23

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, acceleration, 0);
    bindBuffer(kernel, sortedPosition, 1);
    bindBuffer(kernel, sortedVelocity, 2);
    bindBuffer(kernel, sortedCellAndSerialId, 3);
    bindBuffer(kernel, sortedParticleIdBySerialId, 4);
    bindScalar(kernel, gravityX, 5);
    bindScalar(kernel, gravityY, 6);
    bindScalar(kernel, gravityZ, 7);
    bindScalar(kernel, simulationScaleInv, 8);
    bindScalar(kernel, timeStep, 9);
    bindScalar(kernel, xmin, 10);
    bindScalar(kernel, xmax, 11);
    bindScalar(kernel, ymin, 12);
    bindScalar(kernel, ymax, 13);
    bindScalar(kernel, zmin, 14);
    bindScalar(kernel, zmax, 15);
    bindBuffer(kernel, originalPosition, 16);
    bindBuffer(kernel, velocity, 17);
    bindBuffer(kernel, rho, 18);
    bindScalar(kernel, r0, 19);
    bindBuffer(kernel, neighborMap, 20);
    bindScalar(kernel, particleCount, 21);
    bindScalar(kernel, iterationCount, 22);
    bindScalar(kernel, mode, 23);
  }
};

inline PcisphIntegrateOpenCLArgs
toOpenCLArgs(const PcisphIntegrateInput &input, cl::Context &context,
             cl::Buffer &inOutAcceleration, cl::Buffer &inOutSortedPosition,
             cl::Buffer &inOutOriginalPosition, cl::Buffer &inOutVelocity) {
  cl_int err = CL_SUCCESS;
  PcisphIntegrateOpenCLArgs args{};
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
  args.gravityX = 0.0f;
  args.gravityY = 0.0f;
  args.gravityZ = 0.0f;
  args.simulationScaleInv = input.simulationScaleInv;
  args.timeStep = input.timeStep;
  args.xmin = 0.0f;
  args.xmax = 100.0f;
  args.ymin = 0.0f;
  args.ymax = 100.0f;
  args.zmin = 0.0f;
  args.zmax = 100.0f;
  args.originalPosition = inOutOriginalPosition;
  args.velocity = inOutVelocity;
  // rho is unused but required by the OpenCL kernel signature.
  float dummyRho = 0.0f;
  args.rho = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float), &dummyRho, &err);
  args.r0 = input.r0;
  args.neighborMap =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 input.neighborMap.size_bytes(),
                 const_cast<HostFloat2 *>(input.neighborMap.data()), &err);
  args.particleCount = input.particleCount;
  args.iterationCount = input.iterationCount;
  args.mode = input.mode;
  return args;
}

#endif // SIBERNETIC_USE_OPENCL

// ============ Constants ============
inline constexpr const char *kPcisphIntegrateKernelName = "pcisph_integrate";

} // namespace Sibernetic
