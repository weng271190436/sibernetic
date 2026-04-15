#pragma once

#include <vector>

#include "../../src/convert/MetalConvert.h"
#include "../../src/kernels/PcisphPredictDensityKernel.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
#include "pcisph_predict_density_test_common.h"

namespace SiberneticTest {

class MetalPcisphPredictDensityRunner : public PcisphPredictDensityRunner {
public:
  PcisphPredictDensityResult run(const PcisphPredictDensityCase &tc) override {
    auto input = tc.toInput();
    const uint32_t N = input.particleCount;

    MetalKernelContext metal(Sibernetic::kPcisphPredictDensityKernelName);
    auto *device = metal.device();

    // rho is the output buffer (2×N floats). Initialize to zero.
    auto outputRho = makeMetalOutputBuffer(device, static_cast<size_t>(N) * 2 *
                                                       sizeof(float));

    auto args = Sibernetic::toMetalArgs(input, device, outputRho.get());

    metal.dispatch(N, [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    // Read back predicted density from rho[N..2N).
    PcisphPredictDensityResult result;
    const auto *rhoPtr = reinterpret_cast<const float *>(outputRho->contents());
    result.predictedRho = Sibernetic::Metal::decode(rhoPtr + N, N);
    return result;
  }
};

} // namespace SiberneticTest
