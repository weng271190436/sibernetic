#pragma once

#include <vector>

#include "../../src/kernels/SortPostPassKernel.h"
#include "../../src/convert/MetalConvert.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
#include "../utils/types/metal_types.h"
#include "sort_post_pass_test_common.h"

namespace SiberneticTest {

class MetalSortPostPassRunner : public SortPostPassRunner {
public:
  SortPostPassResult run(const SortPostPassCase &tc) override {
    const uint32_t particleCount =
        static_cast<uint32_t>(tc.particleIndex.size());

    Sibernetic::SortPostPassInput input{};
    input.particleIndex = tc.particleIndex;
    input.position = tc.position;
    input.velocity = tc.velocity;
    input.particleCount = particleCount;

    MetalKernelContext metal(Sibernetic::kSortPostPassKernelName);
    auto *device = metal.device();

    auto outIndexBack =
        makeMetalOutputBuffer(device, sizeof(uint32_t) * particleCount);
    auto outSortedPosition =
        makeMetalOutputBuffer(device, sizeof(MetalFloat4) * particleCount);
    auto outSortedVelocity =
        makeMetalOutputBuffer(device, sizeof(MetalFloat4) * particleCount);

    auto args = Sibernetic::toMetalArgs(input, device, outIndexBack.get(),
                                        outSortedPosition.get(),
                                        outSortedVelocity.get());

    metal.dispatch(particleCount,
                   [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    SortPostPassResult result;
    const auto *idxPtr =
        reinterpret_cast<const uint32_t *>(outIndexBack->contents());
    result.particleIndexBack = Sibernetic::Metal::decode(idxPtr, particleCount);
    const auto *posPtr =
        reinterpret_cast<const MetalFloat4 *>(outSortedPosition->contents());
    result.sortedPosition = Sibernetic::Metal::decode(posPtr, particleCount);
    const auto *velPtr =
        reinterpret_cast<const MetalFloat4 *>(outSortedVelocity->contents());
    result.sortedVelocity = Sibernetic::Metal::decode(velPtr, particleCount);
    return result;
  }
};

} // namespace SiberneticTest
