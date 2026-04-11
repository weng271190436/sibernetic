#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "../context/opencl_context.h"

namespace SiberneticTest {

// A host buffer uploaded to the GPU as a read-only input.
struct CLInputBuffer {
  cl_uint argIndex;
  std::vector<std::byte> bytes;

  template <typename T>
  static CLInputBuffer make(cl_uint idx, const std::vector<T> &v) {
    CLInputBuffer b;
    b.argIndex = idx;
    b.bytes.resize(sizeof(T) * v.size());
    std::memcpy(b.bytes.data(), v.data(), b.bytes.size());
    return b;
  }
};

// A write-only GPU buffer; after dispatch the kernel output is copied to dest.
struct CLOutputBuffer {
  cl_uint argIndex;
  size_t byteSize;
  void *dest; // caller-owned storage, must be >= byteSize bytes
};

inline cl::Buffer makeOpenCLInputBuffer(cl::Context &context,
                                        const CLInputBuffer &input,
                                        const std::string &kernelName) {
  cl_int err = CL_SUCCESS;
  cl::Buffer buf(
      context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input.bytes.size(),
      const_cast<void *>(static_cast<const void *>(input.bytes.data())), &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to create input buffer arg " +
                             std::to_string(input.argIndex) + " for " +
                             kernelName);
  }
  return buf;
}

inline cl::Buffer makeOpenCLOutputBuffer(cl::Context &context,
                                         const CLOutputBuffer &output,
                                         const std::string &kernelName) {
  cl_int err = CL_SUCCESS;
  cl::Buffer buf(context, CL_MEM_WRITE_ONLY, output.byteSize, nullptr, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to create output buffer arg " +
                             std::to_string(output.argIndex) + " for " +
                             kernelName);
  }
  return buf;
}

} // namespace SiberneticTest
