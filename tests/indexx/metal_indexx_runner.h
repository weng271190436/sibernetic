#pragma once

#include <vector>

#include "../utils/metal_context.h"
#include "../utils/metal_helpers.h"
#include "../utils/metal_types.h"
#include "indexx_test_common.h"

namespace SiberneticTest {

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

    metal.dispatch(threadCount, [&](MTL::ComputeCommandEncoder *enc) {
      enc->setBuffer(particleIndexBuf.get(), 0, 0);
      enc->setBytes(&gridCellCount, sizeof(gridCellCount), 1);
      enc->setBuffer(gridCellIndexBuf.get(), 0, 2);
      enc->setBytes(&particleCount, sizeof(particleCount), 3);
    });

    IndexxResult result;
    result.gridCellIndex.resize(threadCount);
    const auto *out =
        reinterpret_cast<const uint32_t *>(gridCellIndexBuf->contents());
    for (size_t i = 0; i < threadCount; ++i) {
      result.gridCellIndex[i] = out[i];
    }
    return result;
  }
};

} // namespace SiberneticTest