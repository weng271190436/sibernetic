#pragma once

// Kernel argument abstraction for the `clearBuffers` kernel.
//
// Clears the neighborMap buffer to "no neighbor" sentinel values (-1, -1)
// for all particles. Each thread handles one particle's kMaxNeighborCount
// (32) float2 entries.
//
// OpenCL signature (sphFluid.cl):
//   __kernel void clearBuffers(
//       __global float2 *neighborMap,   // arg 0
//       uint PARTICLE_COUNT             // arg 1
//   )
//
// Metal signature (sphFluid.metal):
//   kernel void clearBuffers(
//       device float4 *neighborMap      [[buffer(0)]],  // cast for efficiency
//       constant uint &particleCount    [[buffer(1)]],
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
struct ClearBuffersInput {
  Float2Span neighborMap; // size: particleCount * kMaxNeighborCount
  uint32_t particleCount;
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct ClearBuffersMetalArgs {
  MTL::Buffer *neighborMap; // [[buffer(0)]]  in/out
  uint32_t particleCount;   // [[buffer(1)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, neighborMap, 0);
    bindScalar(enc, particleCount, 1);
  }
};

inline ClearBuffersMetalArgs toMetalArgs(const ClearBuffersInput &input,
                                         MTL::Device *device,
                                         MTL::Buffer *inOutNeighborMap) {
  (void)device;
  ClearBuffersMetalArgs args{};
  args.neighborMap = inOutNeighborMap;
  args.particleCount = input.particleCount;
  return args;
}

#endif

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct ClearBuffersOpenCLArgs {
  cl::Buffer neighborMap; // arg 0
  uint32_t particleCount; // arg 1

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, neighborMap, 0);
    bindScalar(kernel, particleCount, 1);
  }
};

inline ClearBuffersOpenCLArgs toOpenCLArgs(const ClearBuffersInput &input,
                                           cl::Context &context,
                                           cl::Buffer &inOutNeighborMap) {
  (void)context;
  ClearBuffersOpenCLArgs args{};
  args.neighborMap = inOutNeighborMap;
  args.particleCount = input.particleCount;
  return args;
}

#endif

// ============ Constants ============
inline constexpr const char *kClearBuffersKernelName = "clearBuffers";

} // namespace Sibernetic
