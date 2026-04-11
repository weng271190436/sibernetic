#pragma once

#include <vector>

#include "../utils/metal_context.h"
#include "../utils/metal_helpers.h"
#include "../utils/metal_types.h"
#include "indexx_test_common.h"

namespace SiberneticTest {

inline std::vector<uint32_t> convertMetalIndexxGridCellIndex(const uint32_t *src,
                                                             size_t n) {
  return std::vector<uint32_t>(src, src + n);
}

class MetalIndexxRunner : public IndexxRunner {
public:
  IndexxResult run(const IndexxCase &tc) override {
    MetalKernelContext metal("indexx");
    auto *dev = metal.device().get();

    std::vector<MetalUInt2> particleIndex =
        toMetalUInt2Vector(tc.particleIndex);

    const uint32_t gridCellCount = tc.gridCellCount;
    const uint32_t particleCount =
        static_cast<uint32_t>(tc.particleIndex.size());
    const uint32_t threadCount = gridCellCount + 1u;

    auto particleIndexBuf = makeMetalInputBuffer(dev, particleIndex);
    auto gridCellIndexBuf =
        makeMetalOutputBuffer(dev, sizeof(uint32_t) * threadCount);

    IndexxResult result;
    auto outGridCellIndex =
        makeMetalOutputFieldBinding<IndexxResult, uint32_t, uint32_t>(
            2, gridCellIndexBuf, threadCount, &IndexxResult::gridCellIndex,
            convertMetalIndexxGridCellIndex);

    std::vector<MetalKernelArg> args = {
        makeMetalInputArg(0, particleIndexBuf),
        makeMetalScalarArg(1, gridCellCount),
        makeMetalScalarArg(3, particleCount),
    };

    runMetalKernelSpecAndStore(metal, threadCount, std::move(args), result,
                               outGridCellIndex);
    return result;
  }
};

} // namespace SiberneticTest