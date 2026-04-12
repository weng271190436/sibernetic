#pragma once

#include <vector>

#include "../../src/kernels/FindNeighborsKernel.h"
#include "../../src/convert/MetalConvert.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
#include "../utils/types/metal_types.h"
#include "find_neighbors_test_common.h"

namespace SiberneticTest {

class MetalFindNeighborsRunner : public FindNeighborsRunner {
public:
  FindNeighborsResult run(const FindNeighborsCase &tc) override {
    const uint32_t particleCount =
        static_cast<uint32_t>(tc.sortedPosition.size());
    const size_t neighborCount = static_cast<size_t>(particleCount) * 32u;

    Sibernetic::FindNeighborsInput input{};
    input.gridCellIndexFixedUp = tc.gridCellIndexFixedUp;
    input.sortedPosition = tc.sortedPosition;
    input.gridCellCount = tc.gridCellCount;
    input.gridCellsX = tc.gridCellsX;
    input.gridCellsY = tc.gridCellsY;
    input.gridCellsZ = tc.gridCellsZ;
    input.h = tc.h;
    input.hashGridCellSize = tc.hashGridCellSize;
    input.hashGridCellSizeInv = tc.hashGridCellSizeInv;
    input.simulationScale = tc.simulationScale;
    input.xmin = tc.xmin;
    input.ymin = tc.ymin;
    input.zmin = tc.zmin;
    input.particleCount = particleCount;

    MetalKernelContext metal(Sibernetic::kFindNeighborsKernelName);
    auto *device = metal.device();

    auto outputNeighborMap =
        makeMetalOutputBuffer(device, sizeof(MetalFloat2) * neighborCount);

    auto args =
        Sibernetic::toMetalArgs(input, device, outputNeighborMap.get());

    metal.dispatch(particleCount,
                   [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    FindNeighborsResult result;
    const auto *ptr =
        reinterpret_cast<const MetalFloat2 *>(outputNeighborMap->contents());
    result.neighborMap = Sibernetic::Metal::decode(ptr, neighborCount);
    return result;
  }
};

} // namespace SiberneticTest
