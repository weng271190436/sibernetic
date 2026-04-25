#pragma once

#include <vector>

#include "../../src/convert/MetalConvert.h"
#include "../../src/kernels/ClearBuffersKernel.h"
#include "../utils/context/metal_context.h"
#include "clear_buffers_test_common.h"

namespace SiberneticTest {

class MetalClearBuffersRunner : public ClearBuffersRunner {
public:
  ClearBuffersResult run(const ClearBuffersCase &tc) override {
    auto input = tc.toInput();
    // Read back the full buffer, which may be larger than
    // particleCount * kMaxNeighborCount when testing bounds guard.
    const size_t bufferEntries = tc.neighborMap.size();

    MetalKernelContext metal(Sibernetic::kClearBuffersKernelName);
    auto *device = metal.device();

    // neighborMap is read/write — pre-populated with input data.
    // Use tc.neighborMap directly for the full buffer size.
    auto inOutNeighborMap = NS::TransferPtr(device->newBuffer(
        tc.neighborMap.data(), tc.neighborMap.size() * sizeof(HostFloat2),
        MTL::ResourceStorageModeShared));

    auto args = Sibernetic::toMetalArgs(input, device, inOutNeighborMap.get());

    metal.dispatch(input.particleCount,
                   [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    ClearBuffersResult result;
    const auto *ptr = reinterpret_cast<const Sibernetic::MetalFloat2 *>(
        inOutNeighborMap->contents());
    result.neighborMap = Sibernetic::Metal::decode(ptr, bufferEntries);
    return result;
  }
};

} // namespace SiberneticTest
