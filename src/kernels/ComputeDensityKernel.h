#pragma once

// Kernel argument abstraction for the `pcisph_computeDensity` kernel.
//
// OpenCL signature (sphFluid.cl):
//   __kernel void pcisph_computeDensity(
//       __global float2 *neighborMap,           // arg 0
//       float mass_mult_Wpoly6Coefficient,      // arg 1
//       float hScaled2,                         // arg 2
//       __global float *rho,                    // arg 3  (output)
//       __global uint *particleIndexBack,       // arg 4
//       uint PARTICLE_COUNT                     // arg 5
//   )
//
// Metal signature (sphFluid.metal):
//   kernel void pcisph_computeDensity(
//       const device float2 *neighborMap           [[buffer(0)]],
//       constant float &massMultWpoly6Coefficient  [[buffer(1)]],
//       constant float &hScaled2                   [[buffer(2)]],
//       device float *rho                          [[buffer(3)]],  (output)
//       const device uint *particleIndexBack       [[buffer(4)]],
//       constant uint &particleCount               [[buffer(5)]],
//       uint serialId [[thread_position_in_grid]])

#include <cstdint>

#include "../types/HostTypes.h"
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
struct ComputeDensityInput {
  Float2Span neighborMap;            // size: particleCount * 32
  float massMultWpoly6Coefficient;
  float hScaled2;
  UInt32Span particleIndexBack;      // size: particleCount
  uint32_t particleCount;
};

struct ComputeDensityOutput {
  float *rho; // size: particleCount
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct ComputeDensityMetalArgs {
  MTL::Buffer *neighborMap;           // [[buffer(0)]]
  float massMultWpoly6Coefficient;    // [[buffer(1)]]
  float hScaled2;                     // [[buffer(2)]]
  MTL::Buffer *rho;                   // [[buffer(3)]] output
  MTL::Buffer *particleIndexBack;     // [[buffer(4)]]
  uint32_t particleCount;             // [[buffer(5)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, neighborMap, 0);
    bindScalar(enc, massMultWpoly6Coefficient, 1);
    bindScalar(enc, hScaled2, 2);
    bindBuffer(enc, rho, 3);
    bindBuffer(enc, particleIndexBack, 4);
    bindScalar(enc, particleCount, 5);
  }
};

inline ComputeDensityMetalArgs
toMetalArgs(const ComputeDensityInput &input, MTL::Device *device,
            MTL::Buffer *outputRho) {
  ComputeDensityMetalArgs args{};
  args.neighborMap = device->newBuffer(
      input.neighborMap.data(),
      input.neighborMap.size_bytes(),
      MTL::ResourceStorageModeShared);
  args.massMultWpoly6Coefficient = input.massMultWpoly6Coefficient;
  args.hScaled2 = input.hScaled2;
  args.rho = outputRho;
  args.particleIndexBack = device->newBuffer(
      input.particleIndexBack.data(), input.particleIndexBack.size_bytes(),
      MTL::ResourceStorageModeShared);
  args.particleCount = input.particleCount;
  return args;
}

#endif

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct ComputeDensityOpenCLArgs {
  cl::Buffer neighborMap;           // arg 0
  float massMultWpoly6Coefficient;  // arg 1
  float hScaled2;                   // arg 2
  cl::Buffer rho;                   // arg 3  output
  cl::Buffer particleIndexBack;     // arg 4
  uint32_t particleCount;           // arg 5

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, neighborMap, 0);
    bindScalar(kernel, massMultWpoly6Coefficient, 1);
    bindScalar(kernel, hScaled2, 2);
    bindBuffer(kernel, rho, 3);
    bindBuffer(kernel, particleIndexBack, 4);
    bindScalar(kernel, particleCount, 5);
  }
};

inline ComputeDensityOpenCLArgs
toOpenCLArgs(const ComputeDensityInput &input, cl::Context &context,
             cl::Buffer &outputRho) {
  cl_int err = CL_SUCCESS;
  ComputeDensityOpenCLArgs args{};
  args.neighborMap = cl::Buffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      input.neighborMap.size_bytes(),
      const_cast<HostFloat2 *>(input.neighborMap.data()), &err);
  args.massMultWpoly6Coefficient = input.massMultWpoly6Coefficient;
  args.hScaled2 = input.hScaled2;
  args.rho = outputRho;
  args.particleIndexBack = cl::Buffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      input.particleIndexBack.size_bytes(),
      const_cast<uint32_t *>(input.particleIndexBack.data()), &err);
  args.particleCount = input.particleCount;
  return args;
}

#endif

// ============ Constants ============
inline constexpr const char *kComputeDensityKernelName =
    "pcisph_computeDensity";

} // namespace Sibernetic
