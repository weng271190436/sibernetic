#pragma once

#include <vector>

#include "../utils/arg/metal_arg_binding.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
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
    MetalKernelContext metal("findNeighbors");
    auto *dev = metal.device().get();

    const uint32_t particleCount =
        static_cast<uint32_t>(tc.sortedPosition.size());
    const size_t neighborCount = static_cast<size_t>(particleCount) * 32u;

    std::vector<uint32_t> gridCellIndex = tc.gridCellIndexFixedUp;
    std::vector<MetalFloat4> sortedPosition =
        toMetalFloat4Vector(tc.sortedPosition);

    auto gridCellIndexBuf = makeMetalInputBuffer(dev, gridCellIndex);
    auto sortedPositionBuf = makeMetalInputBuffer(dev, sortedPosition);
    auto neighborMapBuf =
        makeMetalOutputBuffer(dev, sizeof(MetalFloat2) * neighborCount);

    const uint32_t gridCellCount = tc.gridCellCount;
    const uint32_t gridCellsX = tc.gridCellsX;
    const uint32_t gridCellsY = tc.gridCellsY;
    const uint32_t gridCellsZ = tc.gridCellsZ;
    const float h = tc.h;
    const float hashGridCellSize = tc.hashGridCellSize;
    const float hashGridCellSizeInv = tc.hashGridCellSizeInv;
    const float simulationScale = tc.simulationScale;
    const float xmin = tc.xmin;
    const float ymin = tc.ymin;
    const float zmin = tc.zmin;

    FindNeighborsResult result;
    auto outNeighborMap =
        makeMetalOutputFieldBinding<FindNeighborsResult, MetalFloat2,
                                    FindNeighborsEntry>(
            13, neighborMapBuf, neighborCount,
            &FindNeighborsResult::neighborMap, convertMetalFindNeighborsMap);

    std::vector<MetalKernelArg> args = {
        makeMetalInputArg(0, gridCellIndexBuf),
        makeMetalInputArg(1, sortedPositionBuf),
        makeMetalScalarArg(2, gridCellCount),
        makeMetalScalarArg(3, gridCellsX),
        makeMetalScalarArg(4, gridCellsY),
        makeMetalScalarArg(5, gridCellsZ),
        makeMetalScalarArg(6, h),
        makeMetalScalarArg(7, hashGridCellSize),
        makeMetalScalarArg(8, hashGridCellSizeInv),
        makeMetalScalarArg(9, simulationScale),
        makeMetalScalarArg(10, xmin),
        makeMetalScalarArg(11, ymin),
        makeMetalScalarArg(12, zmin),
        makeMetalScalarArg(14, particleCount),
    };

    runMetalKernelSpecAndStore(metal, particleCount, std::move(args), result,
                               outNeighborMap);
    return result;
  }
};

} // namespace SiberneticTest
