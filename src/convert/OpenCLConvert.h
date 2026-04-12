#pragma once

#include <cstddef>
#include <vector>

#ifdef err_local
#undef err_local
#endif
#include <OpenCL/opencl.h>

#include "../types/HostTypes.h"

namespace Sibernetic::OpenCL {

// ============ decode: OpenCL → Host ============

inline std::vector<HostFloat4> decode(const std::vector<cl_float4> &src) {
  std::vector<HostFloat4> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = {src[i].s[0], src[i].s[1], src[i].s[2], src[i].s[3]};
  }
  return out;
}

inline std::vector<HostFloat2> decode(const std::vector<cl_float2> &src) {
  std::vector<HostFloat2> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = {src[i].s[0], src[i].s[1]};
  }
  return out;
}

inline std::vector<HostUInt2> decode(const std::vector<cl_uint2> &src) {
  std::vector<HostUInt2> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = {src[i].s[0], src[i].s[1]};
  }
  return out;
}

inline std::vector<HostUInt4> decode(const std::vector<cl_uint4> &src) {
  std::vector<HostUInt4> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = {src[i].s[0], src[i].s[1], src[i].s[2], src[i].s[3]};
  }
  return out;
}

} // namespace Sibernetic::OpenCL
