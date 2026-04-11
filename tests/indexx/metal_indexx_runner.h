#pragma once

#include <vector>

#include "../utils/arg/metal_arg_binding.h"
#include "../utils/convert/metal_convert_utils.h"
#include "../utils/types/metal_types.h"
#include "indexx_test_common.h"

namespace SiberneticTest {

inline std::vector<uint32_t>
convertMetalIndexxGridCellIndex(const uint32_t *src, size_t n) {
  return std::vector<uint32_t>(src, src + n);
}

class MetalIndexxRunner : public IndexxRunner {
public:
  IndexxResult run(const IndexxCase &tc) override {
    std::vector<MetalUInt2> particleIndex =
        toMetalUInt2Vector(tc.particleIndex);

    const uint32_t gridCellCount = tc.gridCellCount;
    const uint32_t particleCount =
        static_cast<uint32_t>(tc.particleIndex.size());
    const uint32_t gridCellIndexCount = gridCellCount + 1u;
    const uint32_t threadCount = gridCellIndexCount;

    IndexxResult result;
    auto outGridCellIndex =
        makeMetalOutputFieldSpec<IndexxResult, uint32_t, uint32_t>(
            2, gridCellIndexCount, &IndexxResult::gridCellIndex,
            convertMetalIndexxGridCellIndex);

    runMetalKernelSpecAndStore("indexx", threadCount,
                               {
                                   MetalScalarArg::make(1, gridCellCount),
                                   MetalScalarArg::make(3, particleCount),
                               },
                               {
                                   MetalInputHostBuffer::make(0, particleIndex),
                               },
                               result, outGridCellIndex);
    return result;
  }
};

} // namespace SiberneticTest