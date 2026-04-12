#pragma once

#include <cstddef>
#include <vector>

#ifdef err_local
#undef err_local
#endif
#include <OpenCL/opencl.h>

#include "../types/HostTypes.h"

namespace Sibernetic::OpenCL {

// ============ encode: Host → OpenCL ============

inline std::vector<cl_float4>
encode(const std::vector<HostFloat4> &src) {
  std::vector<cl_float4> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i].s[0] = src[i][0];
    out[i].s[1] = src[i][1];
    out[i].s[2] = src[i][2];
    out[i].s[3] = src[i][3];
  }
  return out;
}

inline std::vector<cl_float2>
encode(const std::vector<HostFloat2> &src) {
  std::vector<cl_float2> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i].s[0] = src[i][0];
    out[i].s[1] = src[i][1];
  }
  return out;
}

inline std::vector<cl_uint2>
encode(const std::vector<HostUInt2> &src) {
  std::vector<cl_uint2> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i].s[0] = src[i][0];
    out[i].s[1] = src[i][1];
  }
  return out;
}

inline std::vector<cl_uint4>
encode(const std::vector<HostUInt4> &src) {
  std::vector<cl_uint4> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i].s[0] = src[i][0];
    out[i].s[1] = src[i][1];
    out[i].s[2] = src[i][2];
    out[i].s[3] = src[i][3];
  }
  return out;
}

// ============ decode: OpenCL → Host ============

inline std::vector<HostFloat4>
decode(const std::vector<cl_float4> &src) {
  std::vector<HostFloat4> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = {src[i].s[0], src[i].s[1], src[i].s[2], src[i].s[3]};
  }
  return out;
}

inline std::vector<HostFloat2>
decode(const std::vector<cl_float2> &src) {
  std::vector<HostFloat2> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = {src[i].s[0], src[i].s[1]};
  }
  return out;
}

inline std::vector<HostUInt2>
decode(const std::vector<cl_uint2> &src) {
  std::vector<HostUInt2> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = {src[i].s[0], src[i].s[1]};
  }
  return out;
}

inline std::vector<HostUInt4>
decode(const std::vector<cl_uint4> &src) {
  std::vector<HostUInt4> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = {src[i].s[0], src[i].s[1], src[i].s[2], src[i].s[3]};
  }
  return out;
}

} // namespace Sibernetic::OpenCL
