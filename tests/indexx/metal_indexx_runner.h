#pragma once

#include <vector>

#include "../../src/kernels/IndexxKernel.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
#include "../../src/convert/MetalConvert.h"
#include "../utils/types/metal_types.h"
#include "indexx_test_common.h"

namespace SiberneticTest {

class MetalIndexxRunner : public IndexxRunner {
public:
  IndexxResult run(const IndexxCase &tc) override {
    const uint32_t particleCount =
        static_cast<uint32_t>(tc.particleIndex.size());
    const uint32_t gridCellIndexCount = tc.gridCellCount + 1u;
    const uint32_t threadCount = gridCellIndexCount;

    auto particleIndex = Sibernetic::Metal::encode(tc.particleIndex);

    Sibernetic::IndexxInput input{};
    input.particleIndex =
        reinterpret_cast<const uint32_t *>(particleIndex.data());
    input.particleCount = particleCount;
    input.gridCellCount = tc.gridCellCount;

    MetalKernelContext metal(Sibernetic::kIndexxKernelName);
    auto *device = metal.device().get();

    auto outputGridCellIndex =
        makeMetalOutputBuffer(device, sizeof(uint32_t) * gridCellIndexCount);

    auto args =
        Sibernetic::toMetalArgs(input, device, outputGridCellIndex.get());

    metal.dispatch(threadCount,
                   [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    IndexxResult result;
    const auto *ptr =
        reinterpret_cast<const uint32_t *>(outputGridCellIndex->contents());
    result.gridCellIndex = Sibernetic::Metal::decode(ptr, gridCellIndexCount);
    return result;
  }
};

} // namespace SiberneticTest