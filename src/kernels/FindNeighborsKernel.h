#pragma once

// Kernel argument abstraction for the `findNeighbors` kernel.
//
// OpenCL signature (sphFluid.cl):
//   __kernel void findNeighbors(
//       __global uint *gridCellIndexFixedUp,  // arg 0
//       __global float4 *sortedPosition,      // arg 1
//       uint gridCellCount,                   // arg 2
//       uint gridCellsX,                      // arg 3
//       uint gridCellsY,                      // arg 4
//       uint gridCellsZ,                      // arg 5
//       float h,                              // arg 6
//       float hashGridCellSize,               // arg 7
//       float hashGridCellSizeInv,            // arg 8
//       float simulationScale,                // arg 9
//       float xmin,                           // arg 10
//       float ymin,                           // arg 11
//       float zmin,                           // arg 12
//       __global float2 *neighborMap,         // arg 13  (output)
//       uint PARTICLE_COUNT                   // arg 14
//   )
//
// Metal signature (sphFluid.metal):
//   kernel void findNeighbors(
//       const device uint *gridCellIndicesFixedUp [[buffer(0)]],
//       const device float4 *sortedPosition      [[buffer(1)]],
//       constant uint &gridCellCount             [[buffer(2)]],
//       constant uint &gridCellsX                [[buffer(3)]],
//       constant uint &gridCellsY                [[buffer(4)]],
//       constant uint &gridCellsZ                [[buffer(5)]],
//       constant float &h                        [[buffer(6)]],
//       constant float &hashGridCellSize         [[buffer(7)]],
//       constant float &hashGridCellSizeInv      [[buffer(8)]],
//       constant float &simulationScale          [[buffer(9)]],
//       constant float &xmin                     [[buffer(10)]],
//       constant float &ymin                     [[buffer(11)]],
//       constant float &zmin                     [[buffer(12)]],
//       device float2 *neighborMap               [[buffer(13)]],  (output)
//       constant uint &particleCount             [[buffer(14)]],
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
struct FindNeighborsInput {
  const uint32_t *gridCellIndexFixedUp; // size: gridCellCount + 1
  const float *sortedPosition;          // float4 array, size: particleCount * 4
  uint32_t gridCellCount;
  uint32_t gridCellsX;
  uint32_t gridCellsY;
  uint32_t gridCellsZ;
  float h;
  float hashGridCellSize;
  float hashGridCellSizeInv;
  float simulationScale;
  float xmin;
  float ymin;
  float zmin;
  uint32_t particleCount;
};

struct FindNeighborsOutput {
  float *neighborMap; // float2 array, size: particleCount * kMaxNeighborCount * 2
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct FindNeighborsMetalArgs {
  MTL::Buffer *gridCellIndexFixedUp; // [[buffer(0)]]
  MTL::Buffer *sortedPosition;       // [[buffer(1)]]
  uint32_t gridCellCount;            // [[buffer(2)]]
  uint32_t gridCellsX;               // [[buffer(3)]]
  uint32_t gridCellsY;               // [[buffer(4)]]
  uint32_t gridCellsZ;               // [[buffer(5)]]
  float h;                           // [[buffer(6)]]
  float hashGridCellSize;            // [[buffer(7)]]
  float hashGridCellSizeInv;         // [[buffer(8)]]
  float simulationScale;             // [[buffer(9)]]
  float xmin;                        // [[buffer(10)]]
  float ymin;                        // [[buffer(11)]]
  float zmin;                        // [[buffer(12)]]
  MTL::Buffer *neighborMap;          // [[buffer(13)]] output
  uint32_t particleCount;            // [[buffer(14)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, gridCellIndexFixedUp, 0);
    bindBuffer(enc, sortedPosition, 1);
    bindScalar(enc, gridCellCount, 2);
    bindScalar(enc, gridCellsX, 3);
    bindScalar(enc, gridCellsY, 4);
    bindScalar(enc, gridCellsZ, 5);
    bindScalar(enc, h, 6);
    bindScalar(enc, hashGridCellSize, 7);
    bindScalar(enc, hashGridCellSizeInv, 8);
    bindScalar(enc, simulationScale, 9);
    bindScalar(enc, xmin, 10);
    bindScalar(enc, ymin, 11);
    bindScalar(enc, zmin, 12);
    bindBuffer(enc, neighborMap, 13);
    bindScalar(enc, particleCount, 14);
  }
};

inline FindNeighborsMetalArgs
toMetalArgs(const FindNeighborsInput &input, MTL::Device *device,
            MTL::Buffer *outputNeighborMap) {
  FindNeighborsMetalArgs args{};
  args.gridCellIndexFixedUp = device->newBuffer(
      input.gridCellIndexFixedUp,
      sizeof(uint32_t) * (input.gridCellCount + 1),
      MTL::ResourceStorageModeShared);
  args.sortedPosition = device->newBuffer(
      input.sortedPosition, sizeof(float) * 4 * input.particleCount,
      MTL::ResourceStorageModeShared);
  args.gridCellCount = input.gridCellCount;
  args.gridCellsX = input.gridCellsX;
  args.gridCellsY = input.gridCellsY;
  args.gridCellsZ = input.gridCellsZ;
  args.h = input.h;
  args.hashGridCellSize = input.hashGridCellSize;
  args.hashGridCellSizeInv = input.hashGridCellSizeInv;
  args.simulationScale = input.simulationScale;
  args.xmin = input.xmin;
  args.ymin = input.ymin;
  args.zmin = input.zmin;
  args.neighborMap = outputNeighborMap;
  args.particleCount = input.particleCount;
  return args;
}

#endif

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct FindNeighborsOpenCLArgs {
  cl::Buffer gridCellIndexFixedUp; // arg 0
  cl::Buffer sortedPosition;       // arg 1
  uint32_t gridCellCount;          // arg 2
  uint32_t gridCellsX;             // arg 3
  uint32_t gridCellsY;             // arg 4
  uint32_t gridCellsZ;             // arg 5
  float h;                         // arg 6
  float hashGridCellSize;          // arg 7
  float hashGridCellSizeInv;       // arg 8
  float simulationScale;           // arg 9
  float xmin;                      // arg 10
  float ymin;                      // arg 11
  float zmin;                      // arg 12
  cl::Buffer neighborMap;          // arg 13  output
  uint32_t particleCount;          // arg 14

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, gridCellIndexFixedUp, 0);
    bindBuffer(kernel, sortedPosition, 1);
    bindScalar(kernel, gridCellCount, 2);
    bindScalar(kernel, gridCellsX, 3);
    bindScalar(kernel, gridCellsY, 4);
    bindScalar(kernel, gridCellsZ, 5);
    bindScalar(kernel, h, 6);
    bindScalar(kernel, hashGridCellSize, 7);
    bindScalar(kernel, hashGridCellSizeInv, 8);
    bindScalar(kernel, simulationScale, 9);
    bindScalar(kernel, xmin, 10);
    bindScalar(kernel, ymin, 11);
    bindScalar(kernel, zmin, 12);
    bindBuffer(kernel, neighborMap, 13);
    bindScalar(kernel, particleCount, 14);
  }
};

inline FindNeighborsOpenCLArgs
toOpenCLArgs(const FindNeighborsInput &input, cl::Context &context,
             cl::Buffer &outputNeighborMap) {
  cl_int err = CL_SUCCESS;
  FindNeighborsOpenCLArgs args{};
  args.gridCellIndexFixedUp = cl::Buffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(uint32_t) * (input.gridCellCount + 1),
      const_cast<uint32_t *>(input.gridCellIndexFixedUp), &err);
  args.sortedPosition = cl::Buffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(float) * 4 * input.particleCount,
      const_cast<float *>(input.sortedPosition), &err);
  args.gridCellCount = input.gridCellCount;
  args.gridCellsX = input.gridCellsX;
  args.gridCellsY = input.gridCellsY;
  args.gridCellsZ = input.gridCellsZ;
  args.h = input.h;
  args.hashGridCellSize = input.hashGridCellSize;
  args.hashGridCellSizeInv = input.hashGridCellSizeInv;
  args.simulationScale = input.simulationScale;
  args.xmin = input.xmin;
  args.ymin = input.ymin;
  args.zmin = input.zmin;
  args.neighborMap = outputNeighborMap;
  args.particleCount = input.particleCount;
  return args;
}

#endif

// ============ Constants ============
inline constexpr const char *kFindNeighborsKernelName = "findNeighbors";

} // namespace Sibernetic
