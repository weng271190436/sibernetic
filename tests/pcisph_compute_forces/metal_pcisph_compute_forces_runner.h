#pragma once

#include <vector>

#include "../../src/convert/MetalConvert.h"
#include "../../src/kernels/PcisphComputeForcesKernel.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
#include "pcisph_compute_forces_test_common.h"

namespace SiberneticTest {

class MetalPcisphComputeForcesRunner : public PcisphComputeForcesRunner {
public:
  PcisphComputeForcesResult run(const PcisphComputeForcesCase &tc) override {
    auto input = tc.toInput();

    MetalKernelContext metal(Sibernetic::kPcisphComputeForcesKernelName);
    auto *device = metal.device();

    // Output buffers
    auto outPressure =
        makeMetalOutputBuffer(device, sizeof(float) * input.particleCount);
    auto outAcceleration = makeMetalOutputBuffer(
        device, sizeof(float) * 4 * input.particleCount * 2);

    auto args = Sibernetic::toMetalArgs(input, device, outPressure.get(),
                                        outAcceleration.get());

    metal.dispatch(input.particleCount,
                   [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    // Read back
    PcisphComputeForcesResult result;
    result.pressure = Sibernetic::Metal::decode(
        reinterpret_cast<const float *>(outPressure->contents()),
        input.particleCount);
    result.acceleration = Sibernetic::Metal::decode(
        reinterpret_cast<const Sibernetic::MetalFloat4 *>(
            outAcceleration->contents()),
        input.particleCount * 2);
    return result;
  }
};

} // namespace SiberneticTest
