#pragma once

// Kernel argument abstraction for the `clearMembraneBuffers` kernel.
//
// Zeros the delta accumulator region [N..2N) of the position and velocity
// buffers. These regions store per-particle membrane interaction deltas
// that must be cleared at the start of each timestep.
//
// OpenCL signature (sphFluid.cl):
//   __kernel void clearMembraneBuffers(
//       __global float4 *position,       // arg 0  (2×N)
//       __global float4 *velocity,       // arg 1  (2×N)
//       __global float4 *sortedPosition, // arg 2  (unused)
//       int PARTICLE_COUNT               // arg 3
//   )
//
// Metal signature (sphFluid.metal):
//   kernel void clearMembraneBuffers(
//       device float4 *position          [[buffer(0)]],  (2×N)
//       device float4 *velocity          [[buffer(1)]],  (2×N)
//       constant uint &particleCount     [[buffer(2)]],
//       uint id [[thread_position_in_grid]])

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
struct ClearMembraneBuffersInput {
  Float4Span position; // size: 2 × particleCount
  Float4Span velocity; // size: 2 × particleCount
  uint32_t particleCount;
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct ClearMembraneBuffersMetalArgs {
  MTL::Buffer *position;  // [[buffer(0)]]  in/out (2×N)
  MTL::Buffer *velocity;  // [[buffer(1)]]  in/out (2×N)
  uint32_t particleCount; // [[buffer(2)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, position, 0);
    bindBuffer(enc, velocity, 1);
    bindScalar(enc, particleCount, 2);
  }
};

inline ClearMembraneBuffersMetalArgs
toMetalArgs(const ClearMembraneBuffersInput &input, MTL::Device *device,
            MTL::Buffer *inOutPosition, MTL::Buffer *inOutVelocity) {
  (void)device;
  ClearMembraneBuffersMetalArgs args{};
  args.position = inOutPosition;
  args.velocity = inOutVelocity;
  args.particleCount = input.particleCount;
  return args;
}

#endif

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct ClearMembraneBuffersOpenCLArgs {
  cl::Buffer position;       // arg 0
  cl::Buffer velocity;       // arg 1
  cl::Buffer sortedPosition; // arg 2  (unused, required by OpenCL signature)
  int32_t particleCount;     // arg 3  (int in OpenCL signature)

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, position, 0);
    bindBuffer(kernel, velocity, 1);
    bindBuffer(kernel, sortedPosition, 2);
    bindScalar(kernel, particleCount, 3);
  }
};

inline ClearMembraneBuffersOpenCLArgs
toOpenCLArgs(const ClearMembraneBuffersInput &input, cl::Context &context,
             cl::Buffer &inOutPosition, cl::Buffer &inOutVelocity) {
  cl_int err = CL_SUCCESS;
  ClearMembraneBuffersOpenCLArgs args{};
  args.position = inOutPosition;
  args.velocity = inOutVelocity;
  // sortedPosition is unused but required by the OpenCL kernel signature.
  float dummy = 0.0f;
  args.sortedPosition =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(float), &dummy, &err);
  args.particleCount = static_cast<int32_t>(input.particleCount);
  return args;
}

#endif

// ============ Constants ============
inline constexpr const char *kClearMembraneBuffersKernelName =
    "clearMembraneBuffers";

} // namespace Sibernetic
