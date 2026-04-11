#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../../../metal-cpp/Foundation/NSSharedPtr.hpp"
#include "../../../metal-cpp/Metal/MTLBuffer.hpp"
#include "../../../metal-cpp/Metal/MTLComputeCommandEncoder.hpp"

#include "../context/metal_context.h"

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
inline void
runMetalKernelSpecAndStore(MetalKernelContext &metal, uint32_t threadCount,
                           std::vector<MetalKernelArg> args, TResult &result,
                           const TOutputBindings &...outputs) {
  args.reserve(args.size() + sizeof...(outputs));
  appendMetalOutputArg<TResult>(args, outputs...);

  metal.dispatch(threadCount, [&](MTL::ComputeCommandEncoder *enc) {
    bindMetalKernelArgs(enc, args);
  });

  commitMetalOutputs(result, outputs...);
}

} // namespace SiberneticTest
