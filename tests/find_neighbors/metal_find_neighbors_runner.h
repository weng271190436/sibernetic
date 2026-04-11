#pragma once

#include <vector>

#include "../utils/arg/metal_arg_binding.h"
#include "../utils/convert/metal_convert_utils.h"
#include "../utils/types/metal_types.h"
#include "find_neighbors_test_common.h"

namespace SiberneticTest {

inline std::vector<FindNeighborsEntry>
convertMetalFindNeighborsMap(const MetalFloat2 *src, size_t n) {
  std::vector<FindNeighborsEntry> out(n);
  for (size_t i = 0; i < n; ++i) {
    out[i] = {src[i].s[0], src[i].s[1]};
  }
  return out;
}

class MetalFindNeighborsRunner : public FindNeighborsRunner {
public:
  FindNeighborsResult run(const FindNeighborsCase &tc) override {
    const uint32_t particleCount =
        static_cast<uint32_t>(tc.sortedPosition.size());
    const size_t neighborCount = static_cast<size_t>(particleCount) * 32u;

    std::vector<uint32_t> gridCellIndex = tc.gridCellIndexFixedUp;
    std::vector<MetalFloat4> sortedPosition =
        toMetalFloat4Vector(tc.sortedPosition);

    FindNeighborsResult result;
    auto outNeighborMap =
        makeMetalOutputFieldSpec<FindNeighborsResult, MetalFloat2,
                                 FindNeighborsEntry>(
            13, neighborCount, &FindNeighborsResult::neighborMap,
            convertMetalFindNeighborsMap);

    runMetalKernelSpecAndStore(
        "findNeighbors", particleCount,
        {
            MetalScalarArg::make(2, tc.gridCellCount),
            MetalScalarArg::make(3, tc.gridCellsX),
            MetalScalarArg::make(4, tc.gridCellsY),
            MetalScalarArg::make(5, tc.gridCellsZ),
            MetalScalarArg::make(6, tc.h),
            MetalScalarArg::make(7, tc.hashGridCellSize),
            MetalScalarArg::make(8, tc.hashGridCellSizeInv),
            MetalScalarArg::make(9, tc.simulationScale),
            MetalScalarArg::make(10, tc.xmin),
            MetalScalarArg::make(11, tc.ymin),
            MetalScalarArg::make(12, tc.zmin),
            MetalScalarArg::make(14, particleCount),
        },
        {
            MetalInputHostBuffer::make(0, gridCellIndex),
            MetalInputHostBuffer::make(1, sortedPosition),
        },
        result, outNeighborMap);
    return result;
  }
};

} // namespace SiberneticTest
