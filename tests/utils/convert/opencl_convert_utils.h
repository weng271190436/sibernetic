#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

// macOS defines err_local as a macro in <err.h>; it collides with
// the local variable name used inside OpenCL C++ headers.
#ifdef err_local
#undef err_local
#endif

#include "../types/types.h"
#include <OpenCL/opencl.h>

namespace SiberneticTest {

inline std::vector<cl_float4>
toCLFloat4Vector(const std::vector<HostFloat4> &src) {
  std::vector<cl_float4> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i].s[0] = src[i][0];
    out[i].s[1] = src[i][1];
    out[i].s[2] = src[i][2];
    out[i].s[3] = src[i][3];
  }
  return out;
}

inline std::vector<cl_uint2>
toCLUInt2Vector(const std::vector<HostUInt2> &src) {
  std::vector<cl_uint2> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i].s[0] = src[i][0];
    out[i].s[1] = src[i][1];
  }
  return out;
}

inline std::vector<HostFloat4>
toHostFloat4Vector(const std::vector<cl_float4> &src) {
  std::vector<HostFloat4> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = {src[i].s[0], src[i].s[1], src[i].s[2], src[i].s[3]};
  }
  return out;
}

inline std::vector<HostUInt2>
toHostUInt2Vector(const std::vector<cl_uint2> &src) {
  std::vector<HostUInt2> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = {src[i].s[0], src[i].s[1]};
  }
  return out;
}

inline std::vector<uint32_t>
toHostUInt32Vector(const std::vector<cl_uint> &src) {
  std::vector<uint32_t> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = src[i];
  }
  return out;
}

template <typename TVec, size_t N>
inline std::vector<std::array<float, N>>
toHostFloatArrayVector(const std::vector<TVec> &src) {
  std::vector<std::array<float, N>> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    for (size_t j = 0; j < N; ++j) {
      out[i][j] = src[i].s[j];
    }
  }
  return out;
}

inline std::vector<std::array<float, 2>>
toHostFloat2ArrayVector(const std::vector<cl_float2> &src) {
  return toHostFloatArrayVector<cl_float2, 2>(src);
}

} // namespace SiberneticTest
