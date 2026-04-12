#pragma once

// Kernel argument abstraction for the `sortPostPass` kernel.
//
// OpenCL signature (sphFluid.cl):
//   __kernel void sortPostPass(
//       __global uint2  *particleIndex,     // arg 0
//       __global uint   *particleIndexBack, // arg 1  (output)
//       __global float4 *position,          // arg 2
//       __global float4 *velocity,          // arg 3
//       __global float4 *sortedPosition,    // arg 4  (output)
//       __global float4 *sortedVelocity,    // arg 5  (output)
//       uint PARTICLE_COUNT                 // arg 6
//   )
//
// Metal signature (sphFluid.metal):
//   kernel void sortPostPass(
//       const device uint2 *particleIndex     [[buffer(0)]],
//       device uint *particleIndexBack        [[buffer(1)]],  (output)
//       const device float4 *position         [[buffer(2)]],
//       const device float4 *velocity         [[buffer(3)]],
//       device float4 *sortedPosition         [[buffer(4)]],  (output)
//       device float4 *sortedVelocity         [[buffer(5)]],  (output)
//       constant uint &particleCount          [[buffer(6)]],
//       uint particleId [[thread_position_in_grid]])

#include <cstdint>

#include "common/KernelArgs.h"

#ifdef SIBERNETIC_USE_METAL
#include "Foundation/NSSharedPtr.hpp"
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
struct SortPostPassInput {
  const uint32_t *particleIndex; // uint2 array, size: particleCount * 2
  const float *position;         // float4 array, size: particleCount * 4
  const float *velocity;         // float4 array, size: particleCount * 4
  uint32_t particleCount;
};

struct SortPostPassOutput {
  uint32_t *particleIndexBack; // size: particleCount
  float *sortedPosition;       // float4 array, size: particleCount * 4
  float *sortedVelocity;       // float4 array, size: particleCount * 4
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct SortPostPassMetalArgs {
  MTL::Buffer *particleIndex;     // [[buffer(0)]]
  MTL::Buffer *particleIndexBack; // [[buffer(1)]] output
  MTL::Buffer *position;          // [[buffer(2)]]
  MTL::Buffer *velocity;          // [[buffer(3)]]
  MTL::Buffer *sortedPosition;    // [[buffer(4)]] output
  MTL::Buffer *sortedVelocity;    // [[buffer(5)]] output
  uint32_t particleCount;         // [[buffer(6)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, particleIndex, 0);
    bindBuffer(enc, particleIndexBack, 1);
    bindBuffer(enc, position, 2);
    bindBuffer(enc, velocity, 3);
    bindBuffer(enc, sortedPosition, 4);
    bindBuffer(enc, sortedVelocity, 5);
    bindScalar(enc, particleCount, 6);
  }
};

inline SortPostPassMetalArgs
toMetalArgs(const SortPostPassInput &input, MTL::Device *device,
            MTL::Buffer *outputParticleIndexBack,
            MTL::Buffer *outputSortedPosition,
            MTL::Buffer *outputSortedVelocity) {
  SortPostPassMetalArgs args{};
  args.particleIndex = device->newBuffer(
      input.particleIndex, sizeof(uint32_t) * 2 * input.particleCount,
      MTL::ResourceStorageModeShared);
  args.particleIndexBack = outputParticleIndexBack;
  args.position = device->newBuffer(input.position,
                                    sizeof(float) * 4 * input.particleCount,
                                    MTL::ResourceStorageModeShared);
  args.velocity = device->newBuffer(input.velocity,
                                    sizeof(float) * 4 * input.particleCount,
                                    MTL::ResourceStorageModeShared);
  args.sortedPosition = outputSortedPosition;
  args.sortedVelocity = outputSortedVelocity;
  args.particleCount = input.particleCount;
  return args;
}

#endif

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct SortPostPassOpenCLArgs {
  cl::Buffer particleIndex;     // arg 0
  cl::Buffer particleIndexBack; // arg 1  output
  cl::Buffer position;          // arg 2
  cl::Buffer velocity;          // arg 3
  cl::Buffer sortedPosition;    // arg 4  output
  cl::Buffer sortedVelocity;    // arg 5  output
  uint32_t particleCount;       // arg 6

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, particleIndex, 0);
    bindBuffer(kernel, particleIndexBack, 1);
    bindBuffer(kernel, position, 2);
    bindBuffer(kernel, velocity, 3);
    bindBuffer(kernel, sortedPosition, 4);
    bindBuffer(kernel, sortedVelocity, 5);
    bindScalar(kernel, particleCount, 6);
  }
};

inline SortPostPassOpenCLArgs
toOpenCLArgs(const SortPostPassInput &input, cl::Context &context,
             cl::Buffer &outputParticleIndexBack,
             cl::Buffer &outputSortedPosition,
             cl::Buffer &outputSortedVelocity) {
  cl_int err = CL_SUCCESS;
  SortPostPassOpenCLArgs args{};
  args.particleIndex =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(uint32_t) * 2 * input.particleCount,
                 const_cast<uint32_t *>(input.particleIndex), &err);
  args.particleIndexBack = outputParticleIndexBack;
  args.position =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(float) * 4 * input.particleCount,
                 const_cast<float *>(input.position), &err);
  args.velocity =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(float) * 4 * input.particleCount,
                 const_cast<float *>(input.velocity), &err);
  args.sortedPosition = outputSortedPosition;
  args.sortedVelocity = outputSortedVelocity;
  args.particleCount = input.particleCount;
  return args;
}

#endif

// ============ Constants ============
inline constexpr const char *kSortPostPassKernelName = "sortPostPass";

} // namespace Sibernetic
