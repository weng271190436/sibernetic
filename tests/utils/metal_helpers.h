#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../../metal-cpp/Foundation/NSSharedPtr.hpp"
#include "../../metal-cpp/Metal/MTLBuffer.hpp"
#include "../../metal-cpp/Metal/MTLDevice.hpp"

#include "metal_types.h"
#include "types.h"

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