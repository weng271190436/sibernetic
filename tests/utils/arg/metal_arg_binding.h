#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../../metal-cpp/Foundation/NSSharedPtr.hpp"
#include "../../../metal-cpp/Metal/MTLBuffer.hpp"
#include "../../../metal-cpp/Metal/MTLComputeCommandEncoder.hpp"

#include "../buffer/metal_buffer_utils.h"
#include "../context/metal_context.h"

namespace SiberneticTest {

enum class MetalKernelArgKind { Scalar, Input, Output };

struct MetalKernelArg {
  uint32_t argIndex;
  MetalKernelArgKind kind;
  std::optional<std::vector<uint8_t>> scalarBytes;
  std::optional<NS::SharedPtr<MTL::Buffer>> buffer;
};

struct MetalScalarArg {
  uint32_t argIndex;
  std::vector<uint8_t> bytes;

  template <typename T>
  static MetalScalarArg make(uint32_t idx, const T &value) {
    MetalScalarArg arg{};
    arg.argIndex = idx;
    arg.bytes.resize(sizeof(T));
    std::memcpy(arg.bytes.data(), &value, sizeof(T));
    return arg;
  }
};

struct MetalInputBufferArg {
  uint32_t argIndex;
  NS::SharedPtr<MTL::Buffer> buffer;
};

struct MetalInputHostBuffer {
  uint32_t argIndex;
  std::vector<std::byte> bytes;

  template <typename T>
  static MetalInputHostBuffer make(uint32_t idx, const std::vector<T> &data) {
    MetalInputHostBuffer input{};
    input.argIndex = idx;
    input.bytes.resize(sizeof(T) * data.size());
    std::memcpy(input.bytes.data(), data.data(), input.bytes.size());
    return input;
  }
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

template <typename TResult, typename TDevice, typename THost>
struct MetalOutputFieldSpec {
  uint32_t argIndex;
  size_t elementCount;
  std::vector<THost> TResult::*resultField;
  std::function<std::vector<THost>(const TDevice *, size_t)> convert;

  MetalOutputFieldBinding<TResult, TDevice, THost>
  makeBinding(MTL::Device *device) const {
    NS::SharedPtr<MTL::Buffer> buffer =
        makeMetalOutputBuffer(device, sizeof(TDevice) * elementCount);
    return makeMetalOutputFieldBinding<TResult, TDevice, THost>(
        argIndex, std::move(buffer), elementCount, resultField, convert);
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

template <typename TResult, typename TDevice, typename THost,
          typename TConvertFn>
inline MetalOutputFieldSpec<TResult, TDevice, THost>
makeMetalOutputFieldSpec(uint32_t argIndex, size_t elementCount,
                         std::vector<THost> TResult::*resultField,
                         TConvertFn convert) {
  return {argIndex, elementCount, resultField, std::move(convert)};
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

template <typename TResult, typename... TOutputBindings>
inline void
runMetalKernelSpecAndStore(MetalKernelContext &metal, uint32_t threadCount,
                           std::vector<MetalScalarArg> scalarArgs,
                           std::vector<MetalInputBufferArg> inputArgs,
                           TResult &result, const TOutputBindings &...outputs) {
  std::vector<MetalKernelArg> args;
  args.reserve(scalarArgs.size() + inputArgs.size());

  for (auto &s : scalarArgs) {
    MetalKernelArg arg{};
    arg.argIndex = s.argIndex;
    arg.kind = MetalKernelArgKind::Scalar;
    arg.scalarBytes = std::move(s.bytes);
    args.push_back(std::move(arg));
  }

  for (auto &in : inputArgs) {
    args.push_back(makeMetalInputArg(in.argIndex, std::move(in.buffer)));
  }

  runMetalKernelSpecAndStore(metal, threadCount, std::move(args), result,
                             outputs...);
}

inline std::vector<MetalInputBufferArg>
makeMetalInputBufferArgs(MTL::Device *device,
                         std::vector<MetalInputHostBuffer> inputBuffers) {
  std::vector<MetalInputBufferArg> inputArgs;
  inputArgs.reserve(inputBuffers.size());
  for (const auto &input : inputBuffers) {
    NS::SharedPtr<MTL::Buffer> buffer = NS::TransferPtr(
        device->newBuffer(input.bytes.data(), input.bytes.size(),
                          MTL::ResourceStorageModeShared));
    if (!buffer.get()) {
      throw std::runtime_error("Failed to create Metal input buffer");
    }
    inputArgs.push_back({input.argIndex, std::move(buffer)});
  }
  return inputArgs;
}

template <typename TResult, typename... TOutputSpecs>
inline void
runMetalKernelSpecAndStore(const std::string &kernelName, uint32_t threadCount,
                           std::vector<MetalScalarArg> scalarArgs,
                           std::vector<MetalInputHostBuffer> inputBuffers,
                           TResult &result,
                           const TOutputSpecs &...outputSpecs) {
  MetalKernelContext metal(kernelName.c_str());
  auto *device = metal.device().get();
  std::vector<MetalInputBufferArg> inputArgs =
      makeMetalInputBufferArgs(device, std::move(inputBuffers));

  runMetalKernelSpecAndStore(metal, threadCount, std::move(scalarArgs),
                             std::move(inputArgs), result,
                             outputSpecs.makeBinding(device)...);
}

} // namespace SiberneticTest
