#pragma once

#include <vector>

#include "../../src/convert/MetalConvert.h"
#include "../../src/kernels/ComputeDensityKernel.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
#include "compute_density_test_common.h"

namespace SiberneticTest {

class MetalComputeDensityRunner : public ComputeDensityRunner {
public:
  ComputeDensityResult run(const ComputeDensityCase &tc) override {
    auto input = tc.toInput();
    const uint32_t particleCount = input.particleCount;
    if (input.neighborMap.size() != static_cast<size_t>(particleCount) * 32u) {
      throw std::runtime_error("neighborMap size must be particleCount * 32");
    }

    MetalKernelContext metal(Sibernetic::kComputeDensityKernelName);
    auto *device = metal.device();

    // Create output buffer.
    auto outputRho =
        makeMetalOutputBuffer(device, sizeof(float) * particleCount);

    // Convert to Metal args.
    auto args = Sibernetic::toMetalArgs(input, device, outputRho.get());

    // Dispatch.
    metal.dispatch(particleCount,
                   [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    // Read back results.
    ComputeDensityResult result;
    const auto *rhoPtr = reinterpret_cast<const float *>(outputRho->contents());
    result.rho = Sibernetic::Metal::decode(rhoPtr, particleCount);
    return result;
  }
};

} // namespace SiberneticTest
