#pragma once

#include <vector>

#include "../../src/convert/MetalConvert.h"
#include "../../src/kernels/PcisphPredictPositionsKernel.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
#include "pcisph_predict_positions_test_common.h"

namespace SiberneticTest {

class MetalPcisphPredictPositionsRunner
    : public PcisphPredictPositionsRunner {
public:
  PcisphPredictPositionsResult
  run(const PcisphPredictPositionsCase &tc) override {
    auto input = tc.toInput();
    const uint32_t N = input.particleCount;

    MetalKernelContext metal(Sibernetic::kPcisphPredictPositionsKernelName);
    auto *device = metal.device();

    // acceleration is read/write (3×N float4)
    auto inOutAcceleration = NS::TransferPtr(device->newBuffer(
        input.acceleration.data(), input.acceleration.size_bytes(),
        MTL::ResourceStorageModeShared));
    // sortedPosition is read/write (2×N float4)
    auto inOutSortedPosition = NS::TransferPtr(device->newBuffer(
        input.sortedPosition.data(), input.sortedPosition.size_bytes(),
        MTL::ResourceStorageModeShared));

    auto args =
        Sibernetic::toMetalArgs(input, device, inOutAcceleration.get(),
                                inOutSortedPosition.get());

    metal.dispatch(N,
                   [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    // Read back predicted positions from sortedPosition[N..2N)
    PcisphPredictPositionsResult result;
    const auto *sortedPosPtr =
        reinterpret_cast<const Sibernetic::MetalFloat4 *>(
            inOutSortedPosition->contents());
    result.predictedPosition =
        Sibernetic::Metal::decode(sortedPosPtr + N, N);
    return result;
  }
};

} // namespace SiberneticTest
