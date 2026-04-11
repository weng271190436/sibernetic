#pragma once

#include <vector>

#include "../utils/metal_context.h"
#include "../utils/metal_helpers.h"
#include "../utils/metal_types.h"
#include "sort_post_pass_test_common.h"

namespace SiberneticTest {

class MetalSortPostPassRunner : public SortPostPassRunner {
public:
  SortPostPassResult run(const SortPostPassCase &tc) override {
    MetalKernelContext metal("sortPostPass");
    auto *dev = metal.device().get();
    const uint32_t n = static_cast<uint32_t>(tc.particleIndex.size());

    std::vector<MetalUInt2> particleIndex =
        toMetalUInt2Vector(tc.particleIndex);
    std::vector<MetalFloat4> position = toMetalFloat4Vector(tc.position);
    std::vector<MetalFloat4> velocity = toMetalFloat4Vector(tc.velocity);

    auto particleIndexBuf = makeMetalInputBuffer(dev, particleIndex);
    auto particleIndexBackBuf =
        makeMetalOutputBuffer(dev, sizeof(uint32_t) * n);
    auto positionBuf = makeMetalInputBuffer(dev, position);
    auto velocityBuf = makeMetalInputBuffer(dev, velocity);
    auto sortedPositionBuf =
        makeMetalOutputBuffer(dev, sizeof(MetalFloat4) * n);
    auto sortedVelocityBuf =
        makeMetalOutputBuffer(dev, sizeof(MetalFloat4) * n);

    metal.dispatch(n, [&](MTL::ComputeCommandEncoder *enc) {
      enc->setBuffer(particleIndexBuf.get(), 0, 0);
      enc->setBuffer(particleIndexBackBuf.get(), 0, 1);
      enc->setBuffer(positionBuf.get(), 0, 2);
      enc->setBuffer(velocityBuf.get(), 0, 3);
      enc->setBuffer(sortedPositionBuf.get(), 0, 4);
      enc->setBuffer(sortedVelocityBuf.get(), 0, 5);
      enc->setBytes(&n, sizeof(n), 6);
    });

    SortPostPassResult result;
    result.particleIndexBack.resize(n);
    const auto *outPos =
        reinterpret_cast<const MetalFloat4 *>(sortedPositionBuf->contents());
    const auto *outVel =
        reinterpret_cast<const MetalFloat4 *>(sortedVelocityBuf->contents());
    const auto *outBack =
        reinterpret_cast<const uint32_t *>(particleIndexBackBuf->contents());
    result.sortedPosition = toHostFloat4Vector(outPos, n);
    result.sortedVelocity = toHostFloat4Vector(outVel, n);
    for (size_t i = 0; i < n; ++i) {
      result.particleIndexBack[i] = outBack[i];
    }
    return result;
  }
};

} // namespace SiberneticTest
