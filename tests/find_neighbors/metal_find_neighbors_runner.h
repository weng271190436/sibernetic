#pragma once

#include <vector>

#include "../../src/convert/MetalConvert.h"
#include "../../src/kernels/FindNeighborsKernel.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
#include "../utils/types/metal_types.h"
#include "find_neighbors_test_common.h"

namespace SiberneticTest {

class MetalFindNeighborsRunner : public FindNeighborsRunner {
public:
  FindNeighborsResult run(const FindNeighborsCase &tc) override {
    auto input = tc.toInput();
    const uint32_t particleCount = input.particleCount;
    const size_t neighborCount = static_cast<size_t>(particleCount) * 32u;

    MetalKernelContext metal(Sibernetic::kFindNeighborsKernelName);
    auto *device = metal.device();

    auto outputNeighborMap =
        makeMetalOutputBuffer(device, sizeof(MetalFloat2) * neighborCount);

    auto args = Sibernetic::toMetalArgs(input, device, outputNeighborMap.get());

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
