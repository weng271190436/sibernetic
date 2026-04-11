#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../../../metal-cpp/Foundation/NSSharedPtr.hpp"
#include "../../../metal-cpp/Metal/MTLBuffer.hpp"
#include "../../../metal-cpp/Metal/MTLDevice.hpp"

namespace SiberneticTest {

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

} // namespace SiberneticTest
