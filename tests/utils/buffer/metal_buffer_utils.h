#pragma once

#include <cstddef>
#include <stdexcept>

#include "../../../metal-cpp/Foundation/NSSharedPtr.hpp"
#include "../../../metal-cpp/Metal/MTLBuffer.hpp"
#include "../../../metal-cpp/Metal/MTLDevice.hpp"

namespace SiberneticTest {

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
