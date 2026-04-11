#pragma once

#include <vector>

#include "../utils/metal_test_utils.h"
#include "sort_post_pass_test_common.h"

namespace SiberneticTest {

class MetalSortPostPassRunner : public SortPostPassRunner {
public:
  SortPostPassResult run(const SortPostPassCase &tc) override {
    MetalKernelContext metal("sortPostPassMetal");
    auto *dev = metal.device().get();
    const uint32_t n = static_cast<uint32_t>(tc.particleIndex.size());

    std::vector<MetalUInt2> particleIndex(n);
    std::vector<MetalFloat4> position(n), velocity(n);
    for (size_t i = 0; i < n; ++i) {
      particleIndex[i].s[0] = tc.particleIndex[i][0];
      particleIndex[i].s[1] = tc.particleIndex[i][1];
      position[i].s[0] = tc.position[i][0];
      position[i].s[1] = tc.position[i][1];
      position[i].s[2] = tc.position[i][2];
      position[i].s[3] = tc.position[i][3];
      velocity[i].s[0] = tc.velocity[i][0];
      velocity[i].s[1] = tc.velocity[i][1];
      velocity[i].s[2] = tc.velocity[i][2];
      velocity[i].s[3] = tc.velocity[i][3];
    }

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
    result.sortedPosition.resize(n);
    result.sortedVelocity.resize(n);
    result.particleIndexBack.resize(n);
    const auto *outPos =
        reinterpret_cast<const MetalFloat4 *>(sortedPositionBuf->contents());
    const auto *outVel =
        reinterpret_cast<const MetalFloat4 *>(sortedVelocityBuf->contents());
    const auto *outBack =
        reinterpret_cast<const uint32_t *>(particleIndexBackBuf->contents());
    for (size_t i = 0; i < n; ++i) {
      result.sortedPosition[i] = {outPos[i].s[0], outPos[i].s[1],
                                  outPos[i].s[2], outPos[i].s[3]};
      result.sortedVelocity[i] = {outVel[i].s[0], outVel[i].s[1],
                                  outVel[i].s[2], outVel[i].s[3]};
      result.particleIndexBack[i] = outBack[i];
    }
    return result;
  }
};

} // namespace SiberneticTest
