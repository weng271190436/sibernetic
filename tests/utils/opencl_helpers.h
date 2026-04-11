#pragma once

#include <stdexcept>
#include <vector>

#include "opencl_context.h"
#include "types.h"

namespace SiberneticTest {

// Creates a CL_MEM_READ_ONLY buffer pre-loaded with `data`.
template <typename T>
cl::Buffer makeOpenCLReadBuffer(cl::Context &ctx, const std::vector<T> &data,
                                cl_int &err) {
  return cl::Buffer(
      ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * data.size(),
      const_cast<void *>(static_cast<const void *>(data.data())), &err);
}

// Creates a CL_MEM_WRITE_ONLY output buffer of `bytes` bytes.
inline cl::Buffer makeOpenCLWriteBuffer(cl::Context &ctx, size_t bytes,
                                        cl_int &err) {
  return cl::Buffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
}

// Enqueues a 1D kernel, waits for completion, and throws a kernel-specific
// error if either step fails.
inline void runOpenCL1DKernel(cl::CommandQueue &queue, cl::Kernel &kernel,
                              size_t workItems,
                              const char *kernelNameForErrors) {
  if (queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(workItems),
                                 cl::NullRange) != CL_SUCCESS ||
      queue.finish() != CL_SUCCESS) {
    throw std::runtime_error(std::string("Failed to execute ") +
                             kernelNameForErrors + " kernel");
  }
}

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

inline std::vector<cl_uint2> toCLUInt2Vector(const std::vector<HostUInt2> &src) {
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

inline std::vector<HostUInt2> toHostUInt2Vector(const std::vector<cl_uint2> &src) {
  std::vector<HostUInt2> out(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    out[i] = {src[i].s[0], src[i].s[1]};
  }
  return out;
}

} // namespace SiberneticTest