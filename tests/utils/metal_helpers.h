#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../../metal-cpp/Foundation/NSSharedPtr.hpp"
#include "../../metal-cpp/Metal/MTLBuffer.hpp"
#include "../../metal-cpp/Metal/MTLComputeCommandEncoder.hpp"
#include "../../metal-cpp/Metal/MTLDevice.hpp"

#include "metal_context.h"
#include "metal_types.h"
#include "types.h"

namespace SiberneticTest {

enum class MetalKernelArgKind { Scalar, Input, Output };

struct MetalKernelArg {
  uint32_t argIndex;
  MetalKernelArgKind kind;
  std::optional<std::vector<uint8_t>> scalarBytes;
  std::optional<NS::SharedPtr<MTL::Buffer>> buffer;
};

template <typename T>
inline MetalKernelArg makeMetalScalarArg(uint32_t argIndex, const T &value) {
  MetalKernelArg arg{};
  arg.argIndex = argIndex;
  arg.kind = MetalKernelArgKind::Scalar;
  arg.scalarBytes = std::vector<uint8_t>(sizeof(T));
  std::memcpy(arg.scalarBytes->data(), &value, sizeof(T));
  return arg;
}

inline MetalKernelArg makeMetalInputArg(uint32_t argIndex,
                                        NS::SharedPtr<MTL::Buffer> buffer) {
  MetalKernelArg arg{};
  arg.argIndex = argIndex;
  arg.kind = MetalKernelArgKind::Input;
  arg.buffer = std::move(buffer);
  return arg;
}

inline MetalKernelArg makeMetalOutputArg(uint32_t argIndex,
                                         NS::SharedPtr<MTL::Buffer> buffer) {
  MetalKernelArg arg{};
  arg.argIndex = argIndex;
  arg.kind = MetalKernelArgKind::Output;
  arg.buffer = std::move(buffer);
  return arg;
}

inline void bindMetalKernelArgs(MTL::ComputeCommandEncoder *enc,
                                const std::vector<MetalKernelArg> &args) {
  std::vector<MetalKernelArg> sortedArgs = args;
  std::sort(sortedArgs.begin(), sortedArgs.end(),
            [](const MetalKernelArg &a, const MetalKernelArg &b) {
              return a.argIndex < b.argIndex;
            });

  for (const auto &arg : sortedArgs) {
    if (arg.kind == MetalKernelArgKind::Scalar) {
      if (!arg.scalarBytes.has_value()) {
        throw std::runtime_error("Invalid Metal scalar arg descriptor");
      }
      enc->setBytes(arg.scalarBytes->data(), arg.scalarBytes->size(),
                    arg.argIndex);
      continue;
    }

    if (!arg.buffer.has_value() || !arg.buffer->get()) {
      throw std::runtime_error("Invalid Metal buffer arg descriptor");
    }
    enc->setBuffer(arg.buffer->get(), 0, arg.argIndex);
  }
}

template <typename TResult, typename TDevice, typename THost>
struct MetalOutputFieldBinding {
  uint32_t argIndex;
  NS::SharedPtr<MTL::Buffer> buffer;
  size_t elementCount;
  std::vector<THost> TResult::*resultField;
  std::function<std::vector<THost>(const TDevice *, size_t)> convert;

  MetalKernelArg asKernelArg() const {
    return makeMetalOutputArg(argIndex, buffer);
  }

  void commit(TResult &result) const {
    const auto *src = reinterpret_cast<const TDevice *>(buffer->contents());
    result.*resultField = convert(src, elementCount);
  }
};

template <typename TResult, typename TDevice, typename THost,
          typename TConvertFn>
inline MetalOutputFieldBinding<TResult, TDevice, THost>
makeMetalOutputFieldBinding(uint32_t argIndex,
                            NS::SharedPtr<MTL::Buffer> buffer,
                            size_t elementCount,
                            std::vector<THost> TResult::*resultField,
                            TConvertFn convert) {
  return {argIndex, std::move(buffer), elementCount, resultField,
          std::move(convert)};
}

template <typename TResult, typename TBinding>
inline void appendMetalOutputArg(std::vector<MetalKernelArg> &args,
                                 const TBinding &binding) {
  args.push_back(binding.asKernelArg());
}

template <typename TResult, typename TFirstBinding, typename... TRestBindings>
inline void appendMetalOutputArg(std::vector<MetalKernelArg> &args,
                                 const TFirstBinding &first,
                                 const TRestBindings &...rest) {
  args.push_back(first.asKernelArg());
  appendMetalOutputArg<TResult>(args, rest...);
}

template <typename TResult, typename TBinding>
inline void commitMetalOutputs(TResult &result, const TBinding &binding) {
  binding.commit(result);
}

template <typename TResult, typename TFirstBinding, typename... TRestBindings>
inline void commitMetalOutputs(TResult &result, const TFirstBinding &first,
                               const TRestBindings &...rest) {
  first.commit(result);
  commitMetalOutputs(result, rest...);
}

template <typename TResult, typename... TOutputBindings>
inline void runMetalKernelSpecAndStore(MetalKernelContext &metal,
                                       uint32_t threadCount,
                                       std::vector<MetalKernelArg> args,
                                       TResult &result,
                                       const TOutputBindings &...outputs) {
  args.reserve(args.size() + sizeof...(outputs));
  appendMetalOutputArg<TResult>(args, outputs...);

  metal.dispatch(threadCount, [&](MTL::ComputeCommandEncoder *enc) {
    bindMetalKernelArgs(enc, args);
  });

  commitMetalOutputs(result, outputs...);
}

template <typename T>
NS::SharedPtr<MTL::Buffer> makeMetalInputBuffer(MTL::Device *device,
                                                const std::vector<T> &data) {
  NS::SharedPtr<MTL::Buffer> buf = NS::TransferPtr(device->newBuffer(
      data.data(), sizeof(T) * data.size(), MTL::ResourceStorageModeShared));
  if (!buf.get()) {
    throw std::runtime_error("Failed to create Metal input buffer");
  }
  return buf;
}

inline NS::SharedPtr<MTL::Buffer> makeMetalOutputBuffer(MTL::Device *device,
                                                        size_t bytes) {
  NS::SharedPtr<MTL::Buffer> buf =
      NS::TransferPtr(device->newBuffer(bytes, MTL::ResourceStorageModeShared));
  if (!buf.get()) {
    throw std::runtime_error("Failed to create Metal output buffer");
  }
  return buf;
}

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

} // namespace SiberneticTest