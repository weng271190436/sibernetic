#pragma once

// Kernel argument abstraction for the `pcisph_computeForcesAndInitPressure`
// kernel.
//
// OpenCL signature (sphFluid.cl):
//   __kernel void pcisph_computeForcesAndInitPressure(
//       __global float2 *neighborMap,                    // arg 0
//       __global float *rho,                             // arg 1
//       __global float *pressure,                        // arg 2  (output)
//       __global float4 *sortedPosition,                 // arg 3
//       __global float4 *sortedVelocity,                 // arg 4
//       __global float4 *acceleration,                   // arg 5  (output)
//       __global uint *sortedParticleIdBySerialId,                // arg 6
//       float surfTensCoeff,                             // arg 7
//       float mass_mult_divgradWviscosityCoefficient,    // arg 8
//       float hScaled,                                   // arg 9
//       float mu,                                        // arg 10
//       float gravity_x,                                 // arg 11
//       float gravity_y,                                 // arg 12
//       float gravity_z,                                 // arg 13
//       __global float4 *position,                       // arg 14
//       __global uint2 *particleIndex,                   // arg 15
//       uint PARTICLE_COUNT,                             // arg 16
//       float mass                                       // arg 17
//   )
//
// Metal signature (sphFluid.metal):
//   kernel void pcisph_computeForcesAndInitPressure(
//       const device float2 *neighborMap          [[buffer(0)]],
//       const device float *rho                   [[buffer(1)]],
//       device float *pressure                    [[buffer(2)]],  (output)
//       const device float4 *sortedPosition       [[buffer(3)]],
//       const device float4 *sortedVelocity       [[buffer(4)]],
//       device float4 *acceleration               [[buffer(5)]],  (output)
//       const device uint *sortedParticleIdBySerialId      [[buffer(6)]],
//       constant float &surfTensCoeff             [[buffer(7)]],
//       constant float &mass_mult_divgradWviscosityCoeff [[buffer(8)]],
//       constant float &hScaled                   [[buffer(9)]],
//       constant float &mu                        [[buffer(10)]],
//       constant float &gravity_x                 [[buffer(11)]],
//       constant float &gravity_y                 [[buffer(12)]],
//       constant float &gravity_z                 [[buffer(13)]],
//       const device float4 *position             [[buffer(14)]],
//       const device uint2 *particleIndex         [[buffer(15)]],
//       constant uint &particleCount              [[buffer(16)]],
//       constant float &mass                      [[buffer(17)]],
//       uint gid [[thread_position_in_grid]])

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
struct PcisphComputeForcesInput {
  Float2Span neighborMap;                // size: particleCount * 32
  FloatSpan rho;                         // size: particleCount
  Float4Span sortedPosition;             // size: particleCount
  Float4Span sortedVelocity;             // size: particleCount
  UInt32Span sortedParticleIdBySerialId; // size: particleCount
  Float4Span position;     // size: particleCount (original, unsorted)
  UInt2Span particleIndex; // size: particleCount
  float surfTensCoeff;
  float mass_mult_divgradWviscosityCoeff;
  float hScaled;
  float mu;
  float gravity_x;
  float gravity_y;
  float gravity_z;
  float mass;
  uint32_t particleCount;
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct PcisphComputeForcesMetalArgs {
  MTL::Buffer *neighborMap;                // [[buffer(0)]]
  MTL::Buffer *rho;                        // [[buffer(1)]]
  MTL::Buffer *pressure;                   // [[buffer(2)]] output
  MTL::Buffer *sortedPosition;             // [[buffer(3)]]
  MTL::Buffer *sortedVelocity;             // [[buffer(4)]]
  MTL::Buffer *acceleration;               // [[buffer(5)]] output
  MTL::Buffer *sortedParticleIdBySerialId; // [[buffer(6)]]
  float surfTensCoeff;                     // [[buffer(7)]]
  float mass_mult_divgradWviscosityCoeff;  // [[buffer(8)]]
  float hScaled;                           // [[buffer(9)]]
  float mu;                                // [[buffer(10)]]
  float gravity_x;                         // [[buffer(11)]]
  float gravity_y;                         // [[buffer(12)]]
  float gravity_z;                         // [[buffer(13)]]
  MTL::Buffer *position;                   // [[buffer(14)]]
  MTL::Buffer *particleIndex;              // [[buffer(15)]]
  uint32_t particleCount;                  // [[buffer(16)]]
  float mass;                              // [[buffer(17)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, neighborMap, 0);
    bindBuffer(enc, rho, 1);
    bindBuffer(enc, pressure, 2);
    bindBuffer(enc, sortedPosition, 3);
    bindBuffer(enc, sortedVelocity, 4);
    bindBuffer(enc, acceleration, 5);
    bindBuffer(enc, sortedParticleIdBySerialId, 6);
    bindScalar(enc, surfTensCoeff, 7);
    bindScalar(enc, mass_mult_divgradWviscosityCoeff, 8);
    bindScalar(enc, hScaled, 9);
    bindScalar(enc, mu, 10);
    bindScalar(enc, gravity_x, 11);
    bindScalar(enc, gravity_y, 12);
    bindScalar(enc, gravity_z, 13);
    bindBuffer(enc, position, 14);
    bindBuffer(enc, particleIndex, 15);
    bindScalar(enc, particleCount, 16);
    bindScalar(enc, mass, 17);
  }
};

inline PcisphComputeForcesMetalArgs
toMetalArgs(const PcisphComputeForcesInput &input, MTL::Device *device,
            MTL::Buffer *outPressure, MTL::Buffer *outAcceleration) {
  PcisphComputeForcesMetalArgs args{};
  args.neighborMap = device->newBuffer(input.neighborMap.data(),
                                       input.neighborMap.size_bytes(),
                                       MTL::ResourceStorageModeShared);
  args.rho = device->newBuffer(input.rho.data(), input.rho.size_bytes(),
                               MTL::ResourceStorageModeShared);
  args.pressure = outPressure;
  args.sortedPosition = device->newBuffer(input.sortedPosition.data(),
                                          input.sortedPosition.size_bytes(),
                                          MTL::ResourceStorageModeShared);
  args.sortedVelocity = device->newBuffer(input.sortedVelocity.data(),
                                          input.sortedVelocity.size_bytes(),
                                          MTL::ResourceStorageModeShared);
  args.acceleration = outAcceleration;
  args.sortedParticleIdBySerialId =
      device->newBuffer(input.sortedParticleIdBySerialId.data(),
                        input.sortedParticleIdBySerialId.size_bytes(),
                        MTL::ResourceStorageModeShared);
  args.surfTensCoeff = input.surfTensCoeff;
  args.mass_mult_divgradWviscosityCoeff =
      input.mass_mult_divgradWviscosityCoeff;
  args.hScaled = input.hScaled;
  args.mu = input.mu;
  args.gravity_x = input.gravity_x;
  args.gravity_y = input.gravity_y;
  args.gravity_z = input.gravity_z;
  args.position =
      device->newBuffer(input.position.data(), input.position.size_bytes(),
                        MTL::ResourceStorageModeShared);
  args.particleIndex = device->newBuffer(input.particleIndex.data(),
                                         input.particleIndex.size_bytes(),
                                         MTL::ResourceStorageModeShared);
  args.particleCount = input.particleCount;
  args.mass = input.mass;
  return args;
}

#endif // SIBERNETIC_USE_METAL

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct PcisphComputeForcesOpenCLArgs {
  cl::Buffer neighborMap;                 // arg 0
  cl::Buffer rho;                         // arg 1
  cl::Buffer pressure;                    // arg 2  output
  cl::Buffer sortedPosition;              // arg 3
  cl::Buffer sortedVelocity;              // arg 4
  cl::Buffer acceleration;                // arg 5  output
  cl::Buffer sortedParticleIdBySerialId;  // arg 6
  float surfTensCoeff;                    // arg 7
  float mass_mult_divgradWviscosityCoeff; // arg 8
  float hScaled;                          // arg 9
  float mu;                               // arg 10
  float gravity_x;                        // arg 11
  float gravity_y;                        // arg 12
  float gravity_z;                        // arg 13
  cl::Buffer position;                    // arg 14
  cl::Buffer particleIndex;               // arg 15
  uint32_t particleCount;                 // arg 16
  float mass;                             // arg 17

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, neighborMap, 0);
    bindBuffer(kernel, rho, 1);
    bindBuffer(kernel, pressure, 2);
    bindBuffer(kernel, sortedPosition, 3);
    bindBuffer(kernel, sortedVelocity, 4);
    bindBuffer(kernel, acceleration, 5);
    bindBuffer(kernel, sortedParticleIdBySerialId, 6);
    bindScalar(kernel, surfTensCoeff, 7);
    bindScalar(kernel, mass_mult_divgradWviscosityCoeff, 8);
    bindScalar(kernel, hScaled, 9);
    bindScalar(kernel, mu, 10);
    bindScalar(kernel, gravity_x, 11);
    bindScalar(kernel, gravity_y, 12);
    bindScalar(kernel, gravity_z, 13);
    bindBuffer(kernel, position, 14);
    bindBuffer(kernel, particleIndex, 15);
    bindScalar(kernel, particleCount, 16);
    bindScalar(kernel, mass, 17);
  }
};

inline PcisphComputeForcesOpenCLArgs
toOpenCLArgs(const PcisphComputeForcesInput &input, cl::Context &context,
             cl::Buffer &outPressure, cl::Buffer &outAcceleration) {
  cl_int err = CL_SUCCESS;
  PcisphComputeForcesOpenCLArgs args{};
  args.neighborMap =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 input.neighborMap.size_bytes(),
                 const_cast<HostFloat2 *>(input.neighborMap.data()), &err);
  args.rho = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        input.rho.size_bytes(),
                        const_cast<float *>(input.rho.data()), &err);
  args.pressure = outPressure;
  args.sortedPosition =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 input.sortedPosition.size_bytes(),
                 const_cast<HostFloat4 *>(input.sortedPosition.data()), &err);
  args.sortedVelocity =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 input.sortedVelocity.size_bytes(),
                 const_cast<HostFloat4 *>(input.sortedVelocity.data()), &err);
  args.acceleration = outAcceleration;
  args.sortedParticleIdBySerialId = cl::Buffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      input.sortedParticleIdBySerialId.size_bytes(),
      const_cast<uint32_t *>(input.sortedParticleIdBySerialId.data()), &err);
  args.surfTensCoeff = input.surfTensCoeff;
  args.mass_mult_divgradWviscosityCoeff =
      input.mass_mult_divgradWviscosityCoeff;
  args.hScaled = input.hScaled;
  args.mu = input.mu;
  args.gravity_x = input.gravity_x;
  args.gravity_y = input.gravity_y;
  args.gravity_z = input.gravity_z;
  args.position =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 input.position.size_bytes(),
                 const_cast<HostFloat4 *>(input.position.data()), &err);
  args.particleIndex =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 input.particleIndex.size_bytes(),
                 const_cast<HostUInt2 *>(input.particleIndex.data()), &err);
  args.particleCount = input.particleCount;
  args.mass = input.mass;
  return args;
}

#endif // SIBERNETIC_USE_OPENCL

// ============ Constants ============
inline constexpr const char *kPcisphComputeForcesKernelName =
    "pcisph_computeForcesAndInitPressure";

} // namespace Sibernetic
