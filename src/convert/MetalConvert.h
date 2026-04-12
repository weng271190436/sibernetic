#pragma once

#include <cstddef>
#include <vector>

#include <simd/simd.h>

#include "../types/HostTypes.h"
#include "../types/MetalTypes.h"

namespace Sibernetic::Metal {

// ============ encode: Host → Metal ============

inline std::vector<MetalFloat4>
encode(const std::vector<HostFloat4> &src) {
  std::vector<MetalFloat4> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = simd_make_float4(src[i][0], src[i][1], src[i][2], src[i][3]);
  }
  return out;
}

inline std::vector<MetalFloat2>
encode(const std::vector<HostFloat2> &src) {
  std::vector<MetalFloat2> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = simd_make_float2(src[i][0], src[i][1]);
  }
  return out;
}

inline std::vector<MetalUInt2>
encode(const std::vector<HostUInt2> &src) {
  std::vector<MetalUInt2> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = simd_make_uint2(src[i][0], src[i][1]);
  }
  return out;
}

inline std::vector<MetalUInt4>
encode(const std::vector<HostUInt4> &src) {
  std::vector<MetalUInt4> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = simd_make_uint4(src[i][0], src[i][1], src[i][2], src[i][3]);
  }
  return out;
}

// ============ decode: Metal → Host ============

inline std::vector<HostFloat4> decode(const MetalFloat4 *src, size_t n) {
  std::vector<HostFloat4> out(n);
  for (size_t i = 0; i < n; ++i) {
    out[i] = {src[i].x, src[i].y, src[i].z, src[i].w};
  }
  return out;
}

inline std::vector<HostFloat2> decode(const MetalFloat2 *src, size_t n) {
  std::vector<HostFloat2> out(n);
  for (size_t i = 0; i < n; ++i) {
    out[i] = {src[i].x, src[i].y};
  }
  return out;
}

inline std::vector<HostUInt2> decode(const MetalUInt2 *src, size_t n) {
  std::vector<HostUInt2> out(n);
  for (size_t i = 0; i < n; ++i) {
    out[i] = {src[i].x, src[i].y};
  }
  return out;
}

inline std::vector<HostUInt4> decode(const MetalUInt4 *src, size_t n) {
  std::vector<HostUInt4> out(n);
  for (size_t i = 0; i < n; ++i) {
    out[i] = {src[i].x, src[i].y, src[i].z, src[i].w};
  }
  return out;
}

// Generic scalar decode (no conversion needed).
template <typename T>
inline std::vector<T> decode(const T *src, size_t n) {
  return std::vector<T>(src, src + n);
}

} // namespace Sibernetic::Metal
