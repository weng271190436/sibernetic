#pragma once

#include <vector>

#include "../utils/arg/metal_arg_binding.h"
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

    SortPostPassResult result;
    auto outIndexBack =
        makeMetalOutputFieldSpec<SortPostPassResult, uint32_t, uint32_t>(
            1, particleCount, &SortPostPassResult::particleIndexBack,
            toHostVector<uint32_t>);
    auto outSortedPosition =
        makeMetalOutputFieldSpec<SortPostPassResult, MetalFloat4, HostFloat4>(
            4, particleCount, &SortPostPassResult::sortedPosition,
            toHostFloat4Vector);
    auto outSortedVelocity =
        makeMetalOutputFieldSpec<SortPostPassResult, MetalFloat4, HostFloat4>(
            5, particleCount, &SortPostPassResult::sortedVelocity,
            toHostFloat4Vector);

    runMetalKernelSpecAndStore("sortPostPass", particleCount,
                               {
                                   MetalScalarArg::make(6, particleCount),
                               },
                               {
                                   MetalInputHostBuffer::make(0, particleIndex),
                                   MetalInputHostBuffer::make(2, position),
                                   MetalInputHostBuffer::make(3, velocity),
                               },
                               result, outIndexBack, outSortedPosition,
                               outSortedVelocity);
    return result;
  }
};

} // namespace SiberneticTest
