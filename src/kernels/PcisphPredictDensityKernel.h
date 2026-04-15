#pragma once

// Kernel argument abstraction for the `pcisph_predictDensity` kernel.
//
// Computes predicted density from predicted positions (sortedPosition[N..2N))
// and writes the result to rho[N..2N). Used inside the PCISPH pressure
// correction loop after pcisph_predictPositions updates the predicted
// positions.
//
// OpenCL signature (sphFluid.cl):
//   __kernel void pcisph_predictDensity(
//       __global float2 *neighborMap,              // arg 0
//       __global uint   *particleIndexBack,        // arg 1
//       float mass_mult_Wpoly6Coefficient,         // arg 2
//       float h,                                   // arg 3  (smoothing radius)
//       float rho0,                                // arg 4  (unused, rest
//       density) float simulationScale,                     // arg 5
//       __global float4 *sortedPosition,           // arg 6  (2×N)
//       __global float  *pressure,                 // arg 7  (unused)
//       __global float  *rho,                      // arg 8  (2×N, output)
//       uint PARTICLE_COUNT                        // arg 9
//   )
//
// Metal signature (sphFluid.metal):
//   kernel void pcisph_predictDensity(
//       const device float2 *neighborMap                  [[buffer(0)]],
//       const device uint   *sortedParticleIdBySerialId   [[buffer(1)]],
//       constant float &massMultWpoly6Coefficient         [[buffer(2)]],
//       constant float &h                                 [[buffer(3)]],
//       constant float &restDensity                       [[buffer(4)]],
//       constant float &simulationScale                   [[buffer(5)]],
//       const device float4 *sortedPosition               [[buffer(6)]],
//       device float *rho                                 [[buffer(7)]],
//       constant uint &particleCount                      [[buffer(8)]],
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
struct PcisphPredictDensityInput {
  Float2Span neighborMap;                // size: particleCount * 32
  UInt32Span sortedParticleIdBySerialId; // size: particleCount
  float massMultWpoly6Coefficient;
  float h;                   // smoothing radius (unscaled)
  float restDensity;         // reference density (passed but unused by kernel)
  float simulationScale;     // grid-to-simulation scale factor
  Float4Span sortedPosition; // size: particleCount * 2
  uint32_t particleCount;
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct PcisphPredictDensityMetalArgs {
  MTL::Buffer *neighborMap;                // [[buffer(0)]]
  MTL::Buffer *sortedParticleIdBySerialId; // [[buffer(1)]]
  float massMultWpoly6Coefficient;         // [[buffer(2)]]
  float h;                                 // [[buffer(3)]]
  float restDensity;                       // [[buffer(4)]]
  float simulationScale;                   // [[buffer(5)]]
  MTL::Buffer *sortedPosition;             // [[buffer(6)]]
  MTL::Buffer *rho;                        // [[buffer(7)]] output
  uint32_t particleCount;                  // [[buffer(8)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, neighborMap, 0);
    bindBuffer(enc, sortedParticleIdBySerialId, 1);
    bindScalar(enc, massMultWpoly6Coefficient, 2);
    bindScalar(enc, h, 3);
    bindScalar(enc, restDensity, 4);
    bindScalar(enc, simulationScale, 5);
    bindBuffer(enc, sortedPosition, 6);
    bindBuffer(enc, rho, 7);
    bindScalar(enc, particleCount, 8);
  }
};

inline PcisphPredictDensityMetalArgs
toMetalArgs(const PcisphPredictDensityInput &input, MTL::Device *device,
            MTL::Buffer *outputRho) {
  PcisphPredictDensityMetalArgs args{};
  args.neighborMap = device->newBuffer(input.neighborMap.data(),
                                       input.neighborMap.size_bytes(),
                                       MTL::ResourceStorageModeShared);
  args.sortedParticleIdBySerialId =
      device->newBuffer(input.sortedParticleIdBySerialId.data(),
                        input.sortedParticleIdBySerialId.size_bytes(),
                        MTL::ResourceStorageModeShared);
  args.massMultWpoly6Coefficient = input.massMultWpoly6Coefficient;
  args.h = input.h;
  args.restDensity = input.restDensity;
  args.simulationScale = input.simulationScale;
  args.sortedPosition = device->newBuffer(input.sortedPosition.data(),
                                          input.sortedPosition.size_bytes(),
                                          MTL::ResourceStorageModeShared);
  args.rho = outputRho;
  args.particleCount = input.particleCount;
  return args;
}

#endif // SIBERNETIC_USE_METAL

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct PcisphPredictDensityOpenCLArgs {
  cl::Buffer neighborMap;                // arg 0
  cl::Buffer sortedParticleIdBySerialId; // arg 1
  float massMultWpoly6Coefficient;       // arg 2
  float h;                               // arg 3
  float restDensity;                     // arg 4
  float simulationScale;                 // arg 5
  cl::Buffer sortedPosition;             // arg 6
  cl::Buffer pressure;                   // arg 7 (unused dummy)
  cl::Buffer rho;                        // arg 8 output
  uint32_t particleCount;                // arg 9

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, neighborMap, 0);
    bindBuffer(kernel, sortedParticleIdBySerialId, 1);
    bindScalar(kernel, massMultWpoly6Coefficient, 2);
    bindScalar(kernel, h, 3);
    bindScalar(kernel, restDensity, 4);
    bindScalar(kernel, simulationScale, 5);
    bindBuffer(kernel, sortedPosition, 6);
    bindBuffer(kernel, pressure, 7);
    bindBuffer(kernel, rho, 8);
    bindScalar(kernel, particleCount, 9);
  }
};

inline PcisphPredictDensityOpenCLArgs
toOpenCLArgs(const PcisphPredictDensityInput &input, cl::Context &context,
             cl::Buffer &outputRho) {
  cl_int err = CL_SUCCESS;
  PcisphPredictDensityOpenCLArgs args{};
  args.neighborMap =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 input.neighborMap.size_bytes(),
                 const_cast<HostFloat2 *>(input.neighborMap.data()), &err);
  args.sortedParticleIdBySerialId = cl::Buffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      input.sortedParticleIdBySerialId.size_bytes(),
      const_cast<uint32_t *>(input.sortedParticleIdBySerialId.data()), &err);
  args.massMultWpoly6Coefficient = input.massMultWpoly6Coefficient;
  args.h = input.h;
  args.restDensity = input.restDensity;
  args.simulationScale = input.simulationScale;
  args.sortedPosition =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 input.sortedPosition.size_bytes(),
                 const_cast<HostFloat4 *>(input.sortedPosition.data()), &err);
  // pressure is unused but required by the OpenCL kernel signature.
  float dummyPressure = 0.0f;
  args.pressure = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(float), &dummyPressure, &err);
  args.rho = outputRho;
  args.particleCount = input.particleCount;
  return args;
}

#endif // SIBERNETIC_USE_OPENCL

// ============ Constants ============
inline constexpr const char *kPcisphPredictDensityKernelName =
    "pcisph_predictDensity";

} // namespace Sibernetic
