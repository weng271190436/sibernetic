#pragma once

#include <vector>

#include "../../src/convert/MetalConvert.h"
#include "../../src/kernels/IndexxKernel.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
#include "indexx_test_common.h"

namespace SiberneticTest {

class MetalIndexxRunner : public IndexxRunner {
public:
  IndexxResult run(const IndexxCase &tc) override {
    auto input = tc.toInput();
    const uint32_t gridCellIndexCount = input.gridCellCount + 1u;
    const uint32_t threadCount = gridCellIndexCount;

    MetalKernelContext metal(Sibernetic::kIndexxKernelName);
    auto *device = metal.device();

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