#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "opencl_context.h"

// macOS defines err_local as a macro in <err.h>; it collides with
// the local variable name used inside OpenCL C++ headers.
#ifdef err_local
#undef err_local
#endif

#include "../../inc/OpenCL/cl.hpp"
#include "types.h"

namespace SiberneticTest {

// A typed scalar value stored as raw bytes; use CLScalarArg::make<T>() to
// construct.
struct CLScalarArg {
  cl_uint argIndex;
  std::vector<std::byte> bytes;

  template <typename T> static CLScalarArg make(cl_uint idx, T val) {
    CLScalarArg a;
    a.argIndex = idx;
    a.bytes.resize(sizeof(T));
    std::memcpy(a.bytes.data(), &val, sizeof(T));
    return a;
  }
};

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

// All the information needed to dispatch one 1-D kernel.
struct CLKernelSpec {
  std::string kernelName;
  size_t workItems;
  std::vector<CLScalarArg> scalarArgs;
  std::vector<CLInputBuffer> inputBuffers;
  std::vector<CLOutputBuffer> outputBuffers;
};

inline void runCLKernelSpec(const CLKernelSpec &spec);

// Describes how one raw OpenCL output buffer maps into a result field.
template <typename TResult, typename TDevice, typename THost>
struct CLOutputFieldBinding {
  cl_uint argIndex;
  std::vector<TDevice> deviceData;
  std::vector<THost> TResult::*resultField;
  std::function<std::vector<THost>(const std::vector<TDevice> &)> convert;

  CLOutputBuffer asOutputBuffer() {
    return {argIndex, sizeof(TDevice) * deviceData.size(), deviceData.data()};
  }

  void commit(TResult &result) const {
    result.*resultField = convert(deviceData);
  }
};

template <typename TResult, typename TDevice, typename THost,
          typename TConvertFn>
inline CLOutputFieldBinding<TResult, TDevice, THost>
makeCLOutputFieldBinding(cl_uint argIndex, size_t elementCount,
                         std::vector<THost> TResult::*resultField,
                         TConvertFn convert) {
  return {argIndex, std::vector<TDevice>(elementCount), resultField,
          std::move(convert)};
}

template <typename TResult, typename TBindings>
inline void collectCLOutputBuffers(std::vector<CLOutputBuffer> &outBuffers,
                                   TBindings &binding) {
  outBuffers.push_back(binding.asOutputBuffer());
}

template <typename TResult, typename TFirstBinding, typename... TRestBindings>
inline void collectCLOutputBuffers(std::vector<CLOutputBuffer> &outBuffers,
                                   TFirstBinding &first,
                                   TRestBindings &...rest) {
  outBuffers.push_back(first.asOutputBuffer());
  collectCLOutputBuffers<TResult>(outBuffers, rest...);
}

template <typename TResult, typename TBinding>
inline void commitCLOutputBindings(TResult &result, TBinding &binding) {
  binding.commit(result);
}

template <typename TResult, typename TFirstBinding, typename... TRestBindings>
inline void commitCLOutputBindings(TResult &result, TFirstBinding &first,
                                   TRestBindings &...rest) {
  first.commit(result);
  commitCLOutputBindings(result, rest...);
}

template <typename TResult, typename... TOutputBindings>
inline void runCLKernelSpecAndStore(std::string kernelName, size_t workItems,
                                    std::vector<CLScalarArg> scalarArgs,
                                    std::vector<CLInputBuffer> inputBuffers,
                                    TResult &result,
                                    TOutputBindings &...outputBindings) {
  std::vector<CLOutputBuffer> outputBuffers;
  outputBuffers.reserve(sizeof...(outputBindings));
  collectCLOutputBuffers<TResult>(outputBuffers, outputBindings...);

  runCLKernelSpec({std::move(kernelName), workItems, std::move(scalarArgs),
                   std::move(inputBuffers), std::move(outputBuffers)});

  commitCLOutputBindings(result, outputBindings...);
}

// Dispatches the kernel described by spec and reads all outputs back.
inline void runCLKernelSpec(const CLKernelSpec &spec) {
  OpenCLKernelContext opencl;

  cl_int err = CL_SUCCESS;
  cl::Kernel kernel(opencl.program(), spec.kernelName.c_str(), &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to create kernel: " + spec.kernelName);
  }

  // Bind scalar args via the raw C API so we don't need per-type overloads.
  for (const auto &s : spec.scalarArgs) {
    if (clSetKernelArg(kernel(), s.argIndex, s.bytes.size(), s.bytes.data()) !=
        CL_SUCCESS) {
      throw std::runtime_error("Failed to set scalar arg " +
                               std::to_string(s.argIndex) + " for " +
                               spec.kernelName);
    }
  }

  // Create and bind input buffers.
  std::vector<cl::Buffer> inputBufs;
  inputBufs.reserve(spec.inputBuffers.size());
  for (const auto &ib : spec.inputBuffers) {
    cl_int berr = CL_SUCCESS;
    inputBufs.push_back(cl::Buffer(
        opencl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        ib.bytes.size(),
        const_cast<void *>(static_cast<const void *>(ib.bytes.data())), &berr));
    if (berr != CL_SUCCESS ||
        kernel.setArg(ib.argIndex, inputBufs.back()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to set input buffer arg " +
                               std::to_string(ib.argIndex) + " for " +
                               spec.kernelName);
    }
  }

  // Create and bind output buffers.
  std::vector<cl::Buffer> outputBufs;
  outputBufs.reserve(spec.outputBuffers.size());
  for (const auto &ob : spec.outputBuffers) {
    cl_int berr = CL_SUCCESS;
    outputBufs.push_back(cl::Buffer(opencl.context(), CL_MEM_WRITE_ONLY,
                                    ob.byteSize, nullptr, &berr));
    if (berr != CL_SUCCESS ||
        kernel.setArg(ob.argIndex, outputBufs.back()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to set output buffer arg " +
                               std::to_string(ob.argIndex) + " for " +
                               spec.kernelName);
    }
  }

  // Dispatch and wait.
  if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                          cl::NDRange(spec.workItems),
                                          cl::NullRange) != CL_SUCCESS ||
      opencl.queue().finish() != CL_SUCCESS) {
    throw std::runtime_error("Failed to execute " + spec.kernelName +
                             " kernel");
  }

  // Read outputs back.
  for (size_t i = 0; i < spec.outputBuffers.size(); ++i) {
    if (opencl.queue().enqueueReadBuffer(
            outputBufs[i], CL_TRUE, 0, spec.outputBuffers[i].byteSize,
            spec.outputBuffers[i].dest) != CL_SUCCESS) {
      throw std::runtime_error("Failed to read output buffer arg " +
                               std::to_string(spec.outputBuffers[i].argIndex) +
                               " for " + spec.kernelName);
    }
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

} // namespace SiberneticTest