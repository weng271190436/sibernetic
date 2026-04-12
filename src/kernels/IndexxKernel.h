#pragma once

// Kernel argument abstraction for the `indexx` kernel.
//
// OpenCL signature (sphFluid.cl):
//   __kernel void indexx(
//       __global uint2 *particleIndex,   // arg 0
//       uint gridCellCount,              // arg 1
//       __global uint *gridCellIndex,    // arg 2  (output)
//       uint PARTICLE_COUNT              // arg 3
//   )
//
// Metal signature (sphFluid.metal):
//   kernel void indexx(
//       const device uint2 *particleIndex [[buffer(0)]],
//       constant uint &gridCellCount     [[buffer(1)]],
//       device uint *gridCellIndex       [[buffer(2)]],  (output)
//       constant uint &particleCount     [[buffer(3)]],
//       uint targetCellId [[thread_position_in_grid]])

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
struct IndexxInput {
  const uint32_t *particleIndex; // uint2 array: [cellId, serialId]
  uint32_t particleCount;
  uint32_t gridCellCount;
};

struct IndexxOutput {
  uint32_t *gridCellIndex; // size: gridCellCount + 1
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct IndexxMetalArgs {
  MTL::Buffer *particleIndex; // [[buffer(0)]]
  uint32_t gridCellCount;     // [[buffer(1)]]
  MTL::Buffer *gridCellIndex; // [[buffer(2)]] output
  uint32_t particleCount;     // [[buffer(3)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, particleIndex, 0);
    bindScalar(enc, gridCellCount, 1);
    bindBuffer(enc, gridCellIndex, 2);
    bindScalar(enc, particleCount, 3);
  }
};

inline IndexxMetalArgs toMetalArgs(const IndexxInput &input,
                                   MTL::Device *device,
                                   MTL::Buffer *outputGridCellIndex) {
  IndexxMetalArgs args{};
  args.particleIndex = device->newBuffer(
      input.particleIndex, sizeof(uint32_t) * 2 * input.particleCount,
      MTL::ResourceStorageModeShared);
  args.gridCellCount = input.gridCellCount;
  args.gridCellIndex = outputGridCellIndex;
  args.particleCount = input.particleCount;
  return args;
}

#endif

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct IndexxOpenCLArgs {
  cl::Buffer particleIndex; // arg 0
  uint32_t gridCellCount;   // arg 1
  cl::Buffer gridCellIndex; // arg 2  output
  uint32_t particleCount;   // arg 3

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, particleIndex, 0);
    bindScalar(kernel, gridCellCount, 1);
    bindBuffer(kernel, gridCellIndex, 2);
    bindScalar(kernel, particleCount, 3);
  }
};

inline IndexxOpenCLArgs toOpenCLArgs(const IndexxInput &input,
                                     cl::Context &context,
                                     cl::Buffer &outputGridCellIndex) {
  cl_int err = CL_SUCCESS;
  IndexxOpenCLArgs args{};
  args.particleIndex = cl::Buffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(uint32_t) * 2 * input.particleCount,
      const_cast<uint32_t *>(input.particleIndex), &err);
  args.gridCellCount = input.gridCellCount;
  args.gridCellIndex = outputGridCellIndex;
  args.particleCount = input.particleCount;
  return args;
}

#endif

// ============ Constants ============
inline constexpr const char *kIndexxKernelName = "indexx";

} // namespace Sibernetic
