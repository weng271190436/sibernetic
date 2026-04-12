#pragma once

#include <cstddef>
#include <vector>

#include <simd/simd.h>

#include "../types/HostTypes.h"
#include "../types/MetalTypes.h"

namespace Sibernetic::Metal {

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
