#pragma once

// Kernel argument abstraction for the `hashParticles` kernel.
//
// OpenCL signature (sphFluid.cl):
//   __kernel void hashParticles(
//       __global float4 *position,       // arg 0
//       uint gridCellsX,                 // arg 1
//       uint gridCellsY,                 // arg 2
//       uint gridCellsZ,                 // arg 3
//       float hashGridCellSizeInv,       // arg 4
//       float xmin,                      // arg 5
//       float ymin,                      // arg 6
//       float zmin,                      // arg 7
//       __global uint2 *particleIndex,   // arg 8  (output)
//       uint PARTICLE_COUNT              // arg 9
//   )
//
// Metal signature (sphFluid.metal):
//   kernel void hashParticles(
//       const device float4 *position        [[buffer(0)]],
//       constant uint &gridCellsX            [[buffer(1)]],
//       constant uint &gridCellsY            [[buffer(2)]],
//       constant uint &gridCellsZ            [[buffer(3)]],
//       constant float &hashGridCellSizeInv  [[buffer(4)]],
//       constant float &xmin                 [[buffer(5)]],
//       constant float &ymin                 [[buffer(6)]],
//       constant float &zmin                 [[buffer(7)]],
//       device uint2 *particleIndex          [[buffer(8)]],  (output)
//       constant uint &particleCount         [[buffer(9)]],
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
struct HashParticlesInput {
  const float *position;   // float4 array, size: particleCount * 4
  uint32_t gridCellsX;
  uint32_t gridCellsY;
  uint32_t gridCellsZ;
  float hashGridCellSizeInv;
  float xmin;
  float ymin;
  float zmin;
  uint32_t particleCount;
};

struct HashParticlesOutput {
  uint32_t *particleIndex; // uint2 array, size: particleCount * 2
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct HashParticlesMetalArgs {
  MTL::Buffer *position;         // [[buffer(0)]]
  uint32_t gridCellsX;           // [[buffer(1)]]
  uint32_t gridCellsY;           // [[buffer(2)]]
  uint32_t gridCellsZ;           // [[buffer(3)]]
  float hashGridCellSizeInv;     // [[buffer(4)]]
  float xmin;                    // [[buffer(5)]]
  float ymin;                    // [[buffer(6)]]
  float zmin;                    // [[buffer(7)]]
  MTL::Buffer *particleIndex;    // [[buffer(8)]] output
  uint32_t particleCount;        // [[buffer(9)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, position, 0);
    bindScalar(enc, gridCellsX, 1);
    bindScalar(enc, gridCellsY, 2);
    bindScalar(enc, gridCellsZ, 3);
    bindScalar(enc, hashGridCellSizeInv, 4);
    bindScalar(enc, xmin, 5);
    bindScalar(enc, ymin, 6);
    bindScalar(enc, zmin, 7);
    bindBuffer(enc, particleIndex, 8);
    bindScalar(enc, particleCount, 9);
  }
};

inline HashParticlesMetalArgs
toMetalArgs(const HashParticlesInput &input, MTL::Device *device,
            MTL::Buffer *outputParticleIndex) {
  HashParticlesMetalArgs args{};
  args.position = device->newBuffer(input.position,
                                    sizeof(float) * 4 * input.particleCount,
                                    MTL::ResourceStorageModeShared);
  args.gridCellsX = input.gridCellsX;
  args.gridCellsY = input.gridCellsY;
  args.gridCellsZ = input.gridCellsZ;
  args.hashGridCellSizeInv = input.hashGridCellSizeInv;
  args.xmin = input.xmin;
  args.ymin = input.ymin;
  args.zmin = input.zmin;
  args.particleIndex = outputParticleIndex;
  args.particleCount = input.particleCount;
  return args;
}

#endif

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct HashParticlesOpenCLArgs {
  cl::Buffer position;       // arg 0
  uint32_t gridCellsX;      // arg 1
  uint32_t gridCellsY;      // arg 2
  uint32_t gridCellsZ;      // arg 3
  float hashGridCellSizeInv; // arg 4
  float xmin;               // arg 5
  float ymin;               // arg 6
  float zmin;               // arg 7
  cl::Buffer particleIndex; // arg 8  output
  uint32_t particleCount;   // arg 9

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, position, 0);
    bindScalar(kernel, gridCellsX, 1);
    bindScalar(kernel, gridCellsY, 2);
    bindScalar(kernel, gridCellsZ, 3);
    bindScalar(kernel, hashGridCellSizeInv, 4);
    bindScalar(kernel, xmin, 5);
    bindScalar(kernel, ymin, 6);
    bindScalar(kernel, zmin, 7);
    bindBuffer(kernel, particleIndex, 8);
    bindScalar(kernel, particleCount, 9);
  }
};

inline HashParticlesOpenCLArgs
toOpenCLArgs(const HashParticlesInput &input, cl::Context &context,
             cl::Buffer &outputParticleIndex) {
  cl_int err = CL_SUCCESS;
  HashParticlesOpenCLArgs args{};
  args.position =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(float) * 4 * input.particleCount,
                 const_cast<float *>(input.position), &err);
  args.gridCellsX = input.gridCellsX;
  args.gridCellsY = input.gridCellsY;
  args.gridCellsZ = input.gridCellsZ;
  args.hashGridCellSizeInv = input.hashGridCellSizeInv;
  args.xmin = input.xmin;
  args.ymin = input.ymin;
  args.zmin = input.zmin;
  args.particleIndex = outputParticleIndex;
  args.particleCount = input.particleCount;
  return args;
}

#endif

// ============ Constants ============
inline constexpr const char *kHashParticlesKernelName = "hashParticles";

} // namespace Sibernetic
