#pragma once

// Kernel argument abstraction for the
// `pcisph_computePressureForceAcceleration` kernel.
//
// Computes the pressure-gradient force acceleration for each fluid particle
// and writes it into acceleration[N..2N). Boundary particles get zero.
// This kernel runs inside the PCISPH predict-correct loop, after
// pcisph_correctPressure has updated the pressure field.
//
// The pressure force follows Solenthaler & Pajarola (2009) formula (5):
//   F_pressure_i = -m * sum_j (p_i + p_j) / (2 * rho_j) * grad_W_spiky(r_ij)
// with a close-range correction when r_ij < hScaled/4 for stability.
//
// OpenCL signature (sphFluid.cl):
//   __kernel void pcisph_computePressureForceAcceleration(
//       __global float2 *neighborMap,                 // arg 0
//       __global float  *pressure,                    // arg 1
//       __global float  *rho,                         // arg 2  (reads [N..2N))
//       __global float4 *sortedPosition,              // arg 3
//       __global float4 *sortedVelocity,              // arg 4  (unused)
//       __global uint   *particleIndexBack,           // arg 5
//       float            delta,                       // arg 6
//       float            mass_mult_gradWspikyCoefficient, // arg 7
//       float            h,                           // arg 8  (unscaled)
//       float            simulationScale,             // arg 9
//       float            mu,                          // arg 10 (unused)
//       __global float4 *acceleration,                // arg 11 (writes
//       [N..2N)) float            rho0,                        // arg 12
//       __global float4 *position,                    // arg 13 (particle type)
//       __global uint2  *particleIndex,               // arg 14
//       uint             PARTICLE_COUNT               // arg 15
//   )
//
// Metal signature (sphFluid.metal):
//   kernel void pcisph_computePressureForceAcceleration(
//       const device float2 *neighborMap                       [[buffer(0)]],
//       const device float  *pressure                          [[buffer(1)]],
//       const device float  *rho                               [[buffer(2)]],
//       const device float4 *sortedPosition                    [[buffer(3)]],
//       const device uint   *sortedParticleIdBySerialId        [[buffer(4)]],
//       constant float &delta                                  [[buffer(5)]],
//       constant float &massMultGradWspikyCoefficient          [[buffer(6)]],
//       constant float &h                                      [[buffer(7)]],
//       constant float &simulationScale                        [[buffer(8)]],
//       constant float &restDensity                            [[buffer(9)]],
//       device float4 *acceleration                            [[buffer(10)]],
//       const device float4 *originalPosition                  [[buffer(11)]],
//       const device uint2  *sortedCellAndSerialId             [[buffer(12)]],
//       constant uint  &particleCount                          [[buffer(13)]],
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
struct PcisphComputePressureForceAccelerationInput {
  Float2Span neighborMap;    // size: particleCount * 32
  FloatSpan pressure;        // size: particleCount
  FloatSpan rho;             // size: particleCount * 2 (reads [N..2N))
  Float4Span sortedPosition; // size: particleCount
  UInt32Span sortedParticleIdBySerialId; // size: particleCount
  float delta;
  float massMultGradWspikyCoefficient;
  float h; // smoothing radius (unscaled)
  float simulationScale;
  float restDensity; // rho0
  Float4Span
      originalPosition; // size: particleCount (for type check, .w = type)
  UInt2Span sortedCellAndSerialId; // size: particleCount
  uint32_t particleCount;
};

// ============ Metal ============
#ifdef SIBERNETIC_USE_METAL

struct PcisphComputePressureForceAccelerationMetalArgs {
  MTL::Buffer *neighborMap;                // [[buffer(0)]]
  MTL::Buffer *pressure;                   // [[buffer(1)]]
  MTL::Buffer *rho;                        // [[buffer(2)]]
  MTL::Buffer *sortedPosition;             // [[buffer(3)]]
  MTL::Buffer *sortedParticleIdBySerialId; // [[buffer(4)]]
  float delta;                             // [[buffer(5)]]
  float massMultGradWspikyCoefficient;     // [[buffer(6)]]
  float h;                                 // [[buffer(7)]]
  float simulationScale;                   // [[buffer(8)]]
  float restDensity;                       // [[buffer(9)]]
  MTL::Buffer *acceleration;               // [[buffer(10)]] output
  MTL::Buffer *originalPosition;           // [[buffer(11)]]
  MTL::Buffer *sortedCellAndSerialId;      // [[buffer(12)]]
  uint32_t particleCount;                  // [[buffer(13)]]

  void bind(MTL::ComputeCommandEncoder *enc) const {
    bindBuffer(enc, neighborMap, 0);
    bindBuffer(enc, pressure, 1);
    bindBuffer(enc, rho, 2);
    bindBuffer(enc, sortedPosition, 3);
    bindBuffer(enc, sortedParticleIdBySerialId, 4);
    bindScalar(enc, delta, 5);
    bindScalar(enc, massMultGradWspikyCoefficient, 6);
    bindScalar(enc, h, 7);
    bindScalar(enc, simulationScale, 8);
    bindScalar(enc, restDensity, 9);
    bindBuffer(enc, acceleration, 10);
    bindBuffer(enc, originalPosition, 11);
    bindBuffer(enc, sortedCellAndSerialId, 12);
    bindScalar(enc, particleCount, 13);
  }
};

inline PcisphComputePressureForceAccelerationMetalArgs
toMetalArgs(const PcisphComputePressureForceAccelerationInput &input,
            MTL::Device *device, MTL::Buffer *accelerationBuf) {
  PcisphComputePressureForceAccelerationMetalArgs args{};
  args.neighborMap = device->newBuffer(input.neighborMap.data(),
                                       input.neighborMap.size_bytes(),
                                       MTL::ResourceStorageModeShared);
  args.pressure =
      device->newBuffer(input.pressure.data(), input.pressure.size_bytes(),
                        MTL::ResourceStorageModeShared);
  args.rho = device->newBuffer(input.rho.data(), input.rho.size_bytes(),
                               MTL::ResourceStorageModeShared);
  args.sortedPosition = device->newBuffer(input.sortedPosition.data(),
                                          input.sortedPosition.size_bytes(),
                                          MTL::ResourceStorageModeShared);
  args.sortedParticleIdBySerialId =
      device->newBuffer(input.sortedParticleIdBySerialId.data(),
                        input.sortedParticleIdBySerialId.size_bytes(),
                        MTL::ResourceStorageModeShared);
  args.delta = input.delta;
  args.massMultGradWspikyCoefficient = input.massMultGradWspikyCoefficient;
  args.h = input.h;
  args.simulationScale = input.simulationScale;
  args.restDensity = input.restDensity;
  args.acceleration = accelerationBuf;
  args.originalPosition = device->newBuffer(input.originalPosition.data(),
                                            input.originalPosition.size_bytes(),
                                            MTL::ResourceStorageModeShared);
  args.sortedCellAndSerialId = device->newBuffer(
      input.sortedCellAndSerialId.data(),
      input.sortedCellAndSerialId.size_bytes(), MTL::ResourceStorageModeShared);
  args.particleCount = input.particleCount;
  return args;
}

#endif // SIBERNETIC_USE_METAL

// ============ OpenCL ============
#ifdef SIBERNETIC_USE_OPENCL

struct PcisphComputePressureForceAccelerationOpenCLArgs {
  cl::Buffer neighborMap;                // arg 0
  cl::Buffer pressure;                   // arg 1
  cl::Buffer rho;                        // arg 2
  cl::Buffer sortedPosition;             // arg 3
  cl::Buffer sortedVelocity;             // arg 4 (unused dummy)
  cl::Buffer sortedParticleIdBySerialId; // arg 5
  float delta;                           // arg 6
  float massMultGradWspikyCoefficient;   // arg 7
  float h;                               // arg 8
  float simulationScale;                 // arg 9
  float mu;                              // arg 10 (unused)
  cl::Buffer acceleration;               // arg 11 output
  float restDensity;                     // arg 12
  cl::Buffer originalPosition;           // arg 13
  cl::Buffer sortedCellAndSerialId;      // arg 14
  uint32_t particleCount;                // arg 15

  void bind(cl::Kernel &kernel) const {
    bindBuffer(kernel, neighborMap, 0);
    bindBuffer(kernel, pressure, 1);
    bindBuffer(kernel, rho, 2);
    bindBuffer(kernel, sortedPosition, 3);
    bindBuffer(kernel, sortedVelocity, 4);
    bindBuffer(kernel, sortedParticleIdBySerialId, 5);
    bindScalar(kernel, delta, 6);
    bindScalar(kernel, massMultGradWspikyCoefficient, 7);
    bindScalar(kernel, h, 8);
    bindScalar(kernel, simulationScale, 9);
    bindScalar(kernel, mu, 10);
    bindBuffer(kernel, acceleration, 11);
    bindScalar(kernel, restDensity, 12);
    bindBuffer(kernel, originalPosition, 13);
    bindBuffer(kernel, sortedCellAndSerialId, 14);
    bindScalar(kernel, particleCount, 15);
  }
};

inline PcisphComputePressureForceAccelerationOpenCLArgs
toOpenCLArgs(const PcisphComputePressureForceAccelerationInput &input,
             cl::Context &context, cl::Buffer &accelerationBuf) {
  cl_int err = CL_SUCCESS;
  PcisphComputePressureForceAccelerationOpenCLArgs args{};
  args.neighborMap =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 input.neighborMap.size_bytes(),
                 const_cast<HostFloat2 *>(input.neighborMap.data()), &err);
  args.pressure = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             input.pressure.size_bytes(),
                             const_cast<float *>(input.pressure.data()), &err);
  args.rho = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        input.rho.size_bytes(),
                        const_cast<float *>(input.rho.data()), &err);
  args.sortedPosition =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 input.sortedPosition.size_bytes(),
                 const_cast<HostFloat4 *>(input.sortedPosition.data()), &err);
  // sortedVelocity is unused but required by the OpenCL kernel signature.
  HostFloat4 dummyVel = {0.0f, 0.0f, 0.0f, 0.0f};
  args.sortedVelocity =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 sizeof(HostFloat4), &dummyVel, &err);
  args.sortedParticleIdBySerialId = cl::Buffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      input.sortedParticleIdBySerialId.size_bytes(),
      const_cast<uint32_t *>(input.sortedParticleIdBySerialId.data()), &err);
  args.delta = input.delta;
  args.massMultGradWspikyCoefficient = input.massMultGradWspikyCoefficient;
  args.h = input.h;
  args.simulationScale = input.simulationScale;
  args.mu = 0.0f; // unused
  args.acceleration = accelerationBuf;
  args.restDensity = input.restDensity;
  args.originalPosition =
      cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                 input.originalPosition.size_bytes(),
                 const_cast<HostFloat4 *>(input.originalPosition.data()), &err);
  args.sortedCellAndSerialId = cl::Buffer(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      input.sortedCellAndSerialId.size_bytes(),
      const_cast<HostUInt2 *>(input.sortedCellAndSerialId.data()), &err);
  args.particleCount = input.particleCount;
  return args;
}

#endif // SIBERNETIC_USE_OPENCL

// ============ Constants ============
inline constexpr const char *kPcisphComputePressureForceAccelerationKernelName =
    "pcisph_computePressureForceAcceleration";

} // namespace Sibernetic
