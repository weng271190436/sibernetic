#pragma once

#include <vector>

#include "../utils/metal_context.h"
#include "../utils/metal_helpers.h"
#include "../utils/metal_types.h"
#include "find_neighbors_test_common.h"

namespace SiberneticTest {

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

    metal.dispatch(particleCount, [&](MTL::ComputeCommandEncoder *enc) {
      enc->setBuffer(gridCellIndexBuf.get(), 0, 0);
      enc->setBuffer(sortedPositionBuf.get(), 0, 1);
      enc->setBytes(&gridCellCount, sizeof(gridCellCount), 2);
      enc->setBytes(&gridCellsX, sizeof(gridCellsX), 3);
      enc->setBytes(&gridCellsY, sizeof(gridCellsY), 4);
      enc->setBytes(&gridCellsZ, sizeof(gridCellsZ), 5);
      enc->setBytes(&h, sizeof(h), 6);
      enc->setBytes(&hashGridCellSize, sizeof(hashGridCellSize), 7);
      enc->setBytes(&hashGridCellSizeInv, sizeof(hashGridCellSizeInv), 8);
      enc->setBytes(&simulationScale, sizeof(simulationScale), 9);
      enc->setBytes(&xmin, sizeof(xmin), 10);
      enc->setBytes(&ymin, sizeof(ymin), 11);
      enc->setBytes(&zmin, sizeof(zmin), 12);
      enc->setBuffer(neighborMapBuf.get(), 0, 13);
      enc->setBytes(&particleCount, sizeof(particleCount), 14);
    });

    FindNeighborsResult result;
    result.neighborMap.resize(neighborCount);
    const auto *out =
        reinterpret_cast<const MetalFloat2 *>(neighborMapBuf->contents());
    for (size_t i = 0; i < neighborCount; ++i) {
      result.neighborMap[i] = {out[i].s[0], out[i].s[1]};
    }
    return result;
  }
};

} // namespace SiberneticTest
