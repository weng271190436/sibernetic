#pragma once

#include <vector>

#include "../utils/arg/metal_arg_binding.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
#include "../utils/convert/metal_convert_utils.h"
#include "../utils/types/metal_types.h"
#include "sort_post_pass_test_common.h"

namespace SiberneticTest {

inline std::vector<uint32_t>
convertMetalSortPostPassIndexBack(const uint32_t *src, size_t n) {
  return std::vector<uint32_t>(src, src + n);
}

inline std::vector<HostFloat4>
convertMetalSortPostPassFloat4(const MetalFloat4 *src, size_t n) {
  return toHostFloat4Vector(src, n);
}

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

    SortPostPassResult result;
    auto outIndexBack =
        makeMetalOutputFieldBinding<SortPostPassResult, uint32_t, uint32_t>(
            1, particleIndexBackBuf, n, &SortPostPassResult::particleIndexBack,
            convertMetalSortPostPassIndexBack);
    auto outSortedPosition =
        makeMetalOutputFieldBinding<SortPostPassResult, MetalFloat4,
                                    HostFloat4>(
            4, sortedPositionBuf, n, &SortPostPassResult::sortedPosition,
            convertMetalSortPostPassFloat4);
    auto outSortedVelocity =
        makeMetalOutputFieldBinding<SortPostPassResult, MetalFloat4,
                                    HostFloat4>(
            5, sortedVelocityBuf, n, &SortPostPassResult::sortedVelocity,
            convertMetalSortPostPassFloat4);

    std::vector<MetalKernelArg> args = {
        makeMetalInputArg(0, particleIndexBuf),
        makeMetalInputArg(2, positionBuf),
        makeMetalInputArg(3, velocityBuf),
        makeMetalScalarArg(6, n),
    };

    runMetalKernelSpecAndStore(metal, n, std::move(args), result, outIndexBack,
                               outSortedPosition, outSortedVelocity);
    return result;
  }
};

} // namespace SiberneticTest
