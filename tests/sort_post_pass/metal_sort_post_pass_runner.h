#pragma once

#include <vector>

#include "../../src/kernels/SortPostPassKernel.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
#include "../utils/convert/metal_convert_utils.h"
#include "../utils/types/metal_types.h"
#include "sort_post_pass_test_common.h"

namespace SiberneticTest {

class MetalSortPostPassRunner : public SortPostPassRunner {
public:
  SortPostPassResult run(const SortPostPassCase &tc) override {
    const uint32_t particleCount =
        static_cast<uint32_t>(tc.particleIndex.size());

    std::vector<MetalUInt2> particleIndex =
        toMetalUInt2Vector(tc.particleIndex);
    std::vector<MetalFloat4> position = toMetalFloat4Vector(tc.position);
    std::vector<MetalFloat4> velocity = toMetalFloat4Vector(tc.velocity);

    Sibernetic::SortPostPassInput input{};
    input.particleIndex =
        reinterpret_cast<const uint32_t *>(particleIndex.data());
    input.position = reinterpret_cast<const float *>(position.data());
    input.velocity = reinterpret_cast<const float *>(velocity.data());
    input.particleCount = particleCount;

    MetalKernelContext metal(Sibernetic::kSortPostPassKernelName);
    auto *device = metal.device().get();

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
    result.particleIndexBack = toHostVector(idxPtr, particleCount);
    const auto *posPtr =
        reinterpret_cast<const MetalFloat4 *>(outSortedPosition->contents());
    result.sortedPosition = toHostFloat4Vector(posPtr, particleCount);
    const auto *velPtr =
        reinterpret_cast<const MetalFloat4 *>(outSortedVelocity->contents());
    result.sortedVelocity = toHostFloat4Vector(velPtr, particleCount);
    return result;
  }
};

} // namespace SiberneticTest
