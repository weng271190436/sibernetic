#pragma once

#include <vector>

#include "../../src/convert/MetalConvert.h"
#include "../../src/kernels/HashParticlesKernel.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
#include "../utils/types/metal_types.h"
#include "hash_particles_test_common.h"

namespace SiberneticTest {

class MetalHashParticlesRunner : public HashParticlesRunner {
public:
  HashParticlesResult run(const HashParticlesCase &tc) override {
    auto input = tc.toInput();
    const uint32_t particleCount = input.particleCount;

    MetalKernelContext metal(Sibernetic::kHashParticlesKernelName);
    auto *device = metal.device();

    auto outputParticleIndex =
        makeMetalOutputBuffer(device, sizeof(MetalUInt2) * particleCount);

    auto args =
        Sibernetic::toMetalArgs(input, device, outputParticleIndex.get());

    metal.dispatch(particleCount,
                   [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    HashParticlesResult result;
    const auto *ptr =
        reinterpret_cast<const MetalUInt2 *>(outputParticleIndex->contents());
    result.particleIndex = Sibernetic::Metal::decode(ptr, particleCount);
    return result;
  }
};

} // namespace SiberneticTest
