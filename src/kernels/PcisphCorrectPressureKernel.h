#pragma once

// Kernel argument abstraction for the `pcisph_correctPressure` kernel.
//
// Reads predicted density from rho[N..2N) (written by pcisph_predictDensity),
// computes the density error against the reference density rho0, scales by the
// precomputed correction factor delta, clamps to non-negative, and accumulates
// the correction into pressure[0..N).
//
// OpenCL signature (sphFluid.cl):
//   __kernel void pcisph_correctPressure(
//       __global uint  *particleIndexBack,  // arg 0
//       float           rho0,               // arg 1 (reference density)
//       __global float *pressure,           // arg 2 (input/output: += corr)
//       __global float *rho,                // arg 3 (reads [N..2N))
//       float           delta,              // arg 4 (pressure correction factor)
//       uint            PARTICLE_COUNT      // arg 5
//   )
//
// Metal signature (sphFluid.metal):
//   kernel void pcisph_correctPressure(
//       const device uint  *particleIndexBack  [[buffer(0)]],
//       constant float &rho0                   [[buffer(1)]],
//       device float *pressure                 [[buffer(2)]],
//       const device float *rho                [[buffer(3)]],
//       constant float &delta                  [[buffer(4)]],
//       constant uint  &particleCount          [[buffer(5)]],
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
struct PcisphCorrectPressureInput {
  UInt32Span particleIndexBack; // size: particleCount
  float rho0;                   // reference density
  FloatSpan pressure;           // size: particleCount (input + output)
  FloatSpan rho;                // size: particleCount * 2 (reads [N..2N))
  float delta;                  // pressure correction factor
  uint32_t particleCount;
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct PcisphCorrectPressureMetalArgs {
  MTL::Buffer *particleIndexBack; // [[buffer(0)]]
  float rho0;                     // [[buffer(1)]]
  MTL::Buffer *pressure;          // [[buffer(2)]] input/output
  MTL::Buffer *rho;               // [[buffer(3)]]
  float delta;                    // [[buffer(4)]]
  uint32_t particleCount;         // [[buffer(5)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, particleIndexBack, 0);
    bindScalar(enc, rho0, 1);
    bindBuffer(enc, pressure, 2);
    bindBuffer(enc, rho, 3);
    bindScalar(enc, delta, 4);
    bindScalar(enc, particleCount, 5);
  }
};

inline PcisphCorrectPressureMetalArgs
toMetalArgs(const PcisphCorrectPressureInput &input, MTL::Device *device,
            MTL::Buffer *pressureBuf, MTL::Buffer *rhoBuf) {
  PcisphCorrectPressureMetalArgs args{};
  args.particleIndexBack =
      device->newBuffer(input.particleIndexBack.data(),
                        input.particleIndexBack.size_bytes(),
                        MTL::ResourceStorageModeShared);
  args.rho0 = input.rho0;
  args.pressure = pressureBuf;
  args.rho = rhoBuf;
  args.delta = input.delta;
  args.particleCount = input.particleCount;
  return args;
}

#endif // SIBERNETIC_USE_METAL

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct PcisphCorrectPressureOpenCLArgs {
  cl::Buffer particleIndexBack; // arg 0
  float rho0;                   // arg 1
  cl::Buffer pressure;          // arg 2 input/output
  cl::Buffer rho;               // arg 3
  float delta;                  // arg 4
  uint32_t particleCount;       // arg 5

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, particleIndexBack, 0);
    bindScalar(kernel, rho0, 1);
    bindBuffer(kernel, pressure, 2);
    bindBuffer(kernel, rho, 3);
    bindScalar(kernel, delta, 4);
    bindScalar(kernel, particleCount, 5);
  }
};

inline PcisphCorrectPressureOpenCLArgs
toOpenCLArgs(const PcisphCorrectPressureInput &input, cl::Context &context,
             cl::Buffer &pressureBuf, cl::Buffer &rhoBuf) {
  cl_int err = CL_SUCCESS;
  PcisphCorrectPressureOpenCLArgs args{};
  args.particleIndexBack = cl::Buffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      input.particleIndexBack.size_bytes(),
      const_cast<uint32_t *>(input.particleIndexBack.data()), &err);
  args.rho0 = input.rho0;
  args.pressure = pressureBuf;
  args.rho = rhoBuf;
  args.delta = input.delta;
  args.particleCount = input.particleCount;
  return args;
}

#endif // SIBERNETIC_USE_OPENCL

// ============ Constants ============
inline constexpr const char *kPcisphCorrectPressureKernelName =
    "pcisph_correctPressure";

} // namespace Sibernetic
