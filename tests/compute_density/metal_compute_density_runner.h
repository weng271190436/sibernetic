#pragma once

#include <vector>

#include "../../src/kernels/ComputeDensityKernel.h"
#include "../../src/convert/MetalConvert.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
#include "compute_density_test_common.h"

namespace SiberneticTest {

class MetalComputeDensityRunner : public ComputeDensityRunner {
public:
  ComputeDensityResult run(const ComputeDensityCase &tc) override {
    const uint32_t particleCount =
        static_cast<uint32_t>(tc.particleIndexBack.size());
    if (tc.neighborMap.size() != static_cast<size_t>(particleCount) * 32u) {
      throw std::runtime_error("neighborMap size must be particleCount * 32");
    }

    MetalKernelContext metal(Sibernetic::kComputeDensityKernelName);
    auto *device = metal.device();

    // Build backend-agnostic input.
    Sibernetic::ComputeDensityInput input{};
    input.neighborMap = tc.neighborMap;
    input.massMultWpoly6Coefficient = tc.massMultWpoly6Coefficient;
    input.hScaled2 = tc.hScaled2;
    input.particleIndexBack = tc.particleIndexBack;
    input.particleCount = particleCount;

    // Create output buffer.
    auto outputRho =
        makeMetalOutputBuffer(device, sizeof(float) * particleCount);

    // Convert to Metal args.
    auto args =
        Sibernetic::toMetalArgs(input, device, outputRho.get());

    // Dispatch.
    metal.dispatch(particleCount,
                   [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    // Read back results.
    ComputeDensityResult result;
    const auto *rhoPtr =
        reinterpret_cast<const float *>(outputRho->contents());
    result.rho = Sibernetic::Metal::decode(rhoPtr, particleCount);
    return result;
  }
};

} // namespace SiberneticTest
