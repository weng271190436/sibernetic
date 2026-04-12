#pragma once

#include <cstdint>

#ifdef SIBERNETIC_USE_METAL
#include "Metal/MTLBuffer.hpp"
#include "Metal/MTLComputeCommandEncoder.hpp"
#include "Metal/MTLDevice.hpp"
#endif

#ifdef SIBERNETIC_USE_OPENCL
// macOS defines err_local as a macro in <err.h>; it collides with
// the local variable name used inside OpenCL C++ headers.
#ifdef err_local
#undef err_local
#endif
#include "OpenCL/cl.hpp"
#endif

namespace Sibernetic {

// Maximum number of neighbors stored per particle in neighborMap.
inline constexpr int kMaxNeighborCount = 32;

// Sentinel value indicating an empty neighbor slot.
inline constexpr int kNoParticleId = -1;

// ============ Metal helpers ============
#ifdef SIBERNETIC_USE_METAL

inline void bindBuffer(MTL::ComputeCommandEncoder *enc, MTL::Buffer *buf,
                       uint32_t index) {
  enc->setBuffer(buf, 0, index);
}

template <typename T>
inline void bindScalar(MTL::ComputeCommandEncoder *enc, const T &value,
                       uint32_t index) {
  enc->setBytes(&value, sizeof(T), index);
}

#endif

// ============ OpenCL helpers ============
#ifdef SIBERNETIC_USE_OPENCL

inline void bindBuffer(cl::Kernel &kernel, const cl::Buffer &buf,
                       cl_uint index) {
  kernel.setArg(index, buf);
}

template <typename T>
inline void bindScalar(cl::Kernel &kernel, const T &value, cl_uint index) {
  kernel.setArg(index, value);
}

#endif

} // namespace Sibernetic
