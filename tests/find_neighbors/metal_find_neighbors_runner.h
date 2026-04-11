#pragma once

#include <vector>

#include "../utils/arg/metal_arg_binding.h"
#include "../utils/convert/metal_convert_utils.h"
#include "../utils/types/metal_types.h"
#include "find_neighbors_test_common.h"

namespace SiberneticTest {

class MetalFindNeighborsRunner : public FindNeighborsRunner {
public:
  FindNeighborsResult run(const FindNeighborsCase &tc) override {
    const uint32_t particleCount =
        static_cast<uint32_t>(tc.sortedPosition.size());
    // findNeighbors writes a fixed-width neighbor table: 32 float2 entries per
    // particle (kMaxNeighborCount in the kernel).
    const size_t neighborCount = static_cast<size_t>(particleCount) * 32u;

    std::vector<uint32_t> gridCellIndex = tc.gridCellIndexFixedUp;
    std::vector<MetalFloat4> sortedPosition =
        toMetalFloat4Vector(tc.sortedPosition);

    FindNeighborsResult result;
    auto outNeighborMap =
        makeMetalOutputFieldSpec<FindNeighborsResult, MetalFloat2,
                                 std::array<float, 2>>(
            13, neighborCount, &FindNeighborsResult::neighborMap,
            toHostFloat2ArrayVector);

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
