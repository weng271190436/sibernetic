#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "../types/metal_types.h"
#include "../types/types.h"

namespace SiberneticTest {

inline std::vector<MetalFloat4>
toMetalFloat4Vector(const std::vector<HostFloat4> &src) {
  std::vector<MetalFloat4> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i].s[0] = src[i][0];
    out[i].s[1] = src[i][1];
    out[i].s[2] = src[i][2];
    out[i].s[3] = src[i][3];
  }
  return out;
}

inline std::vector<MetalUInt2>
toMetalUInt2Vector(const std::vector<HostUInt2> &src) {
  std::vector<MetalUInt2> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i].s[0] = src[i][0];
    out[i].s[1] = src[i][1];
  }
  return out;
}

inline std::vector<HostFloat4> toHostFloat4Vector(const MetalFloat4 *src,
                                                  size_t n) {
  std::vector<HostFloat4> out(n);
  for (size_t i = 0; i < n; ++i) {
    out[i] = {src[i].s[0], src[i].s[1], src[i].s[2], src[i].s[3]};
  }
  return out;
}

inline std::vector<HostUInt2> toHostUInt2Vector(const MetalUInt2 *src,
                                                size_t n) {
  std::vector<HostUInt2> out(n);
  for (size_t i = 0; i < n; ++i) {
    out[i] = {src[i].s[0], src[i].s[1]};
  }
  return out;
}

template <typename T>
inline std::vector<T> toHostVector(const T *src, size_t n) {
  return std::vector<T>(src, src + n);
}

template <typename TVec, size_t N>
inline std::vector<std::array<float, N>> toHostFloatArrayVector(const TVec *src,
                                                                size_t n) {
  std::vector<std::array<float, N>> out(n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < N; ++j) {
      out[i][j] = src[i].s[j];
    }
  }
  return out;
}

inline std::vector<std::array<float, 2>>
toHostFloat2ArrayVector(const MetalFloat2 *src, size_t n) {
  return toHostFloatArrayVector<MetalFloat2, 2>(src, n);
}

} // namespace SiberneticTest
