#pragma once

#include <vector>

#include "../../src/convert/MetalConvert.h"
#include "../../src/kernels/ClearMembraneBuffersKernel.h"
#include "../utils/context/metal_context.h"
#include "clear_membrane_buffers_test_common.h"

namespace SiberneticTest {

class MetalClearMembraneBuffersRunner : public ClearMembraneBuffersRunner {
public:
  ClearMembraneBuffersResult run(const ClearMembraneBuffersCase &tc) override {
    auto input = tc.toInput();
    const uint32_t N = input.particleCount;

    MetalKernelContext metal(Sibernetic::kClearMembraneBuffersKernelName);
    auto *device = metal.device();

    auto inOutPosition = NS::TransferPtr(
        device->newBuffer(input.position.data(), input.position.size_bytes(),
                          MTL::ResourceStorageModeShared));
    auto inOutVelocity = NS::TransferPtr(
        device->newBuffer(input.velocity.data(), input.velocity.size_bytes(),
                          MTL::ResourceStorageModeShared));

    auto args = Sibernetic::toMetalArgs(input, device, inOutPosition.get(),
                                        inOutVelocity.get());

    metal.dispatch(N, [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    ClearMembraneBuffersResult result;
    const auto *posPtr = reinterpret_cast<const Sibernetic::MetalFloat4 *>(
        inOutPosition->contents());
    result.position = Sibernetic::Metal::decode(posPtr, 2 * N);

    const auto *velPtr = reinterpret_cast<const Sibernetic::MetalFloat4 *>(
        inOutVelocity->contents());
    result.velocity = Sibernetic::Metal::decode(velPtr, 2 * N);

    return result;
  }
};

} // namespace SiberneticTest
