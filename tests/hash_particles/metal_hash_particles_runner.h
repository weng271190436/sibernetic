#pragma once

#include <vector>

#include "../../src/kernels/HashParticlesKernel.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
#include "../utils/convert/metal_convert_utils.h"
#include "../utils/types/metal_types.h"
#include "hash_particles_test_common.h"

namespace SiberneticTest {

class MetalHashParticlesRunner : public HashParticlesRunner {
public:
  HashParticlesResult run(const HashParticlesCase &tc) override {
    std::vector<MetalFloat4> positions = toMetalFloat4Vector(tc.positions);
    const uint32_t particleCount = static_cast<uint32_t>(positions.size());

    Sibernetic::HashParticlesInput input{};
    input.position = reinterpret_cast<const float *>(positions.data());
    input.gridCellsX = tc.gridCellsX;
    input.gridCellsY = tc.gridCellsY;
    input.gridCellsZ = tc.gridCellsZ;
    input.hashGridCellSizeInv = tc.hashGridCellSizeInv;
    input.xmin = tc.xmin;
    input.ymin = tc.ymin;
    input.zmin = tc.zmin;
    input.particleCount = particleCount;

    MetalKernelContext metal(Sibernetic::kHashParticlesKernelName);
    auto *device = metal.device().get();

    auto outputParticleIndex = makeMetalOutputBuffer(
        device, sizeof(MetalUInt2) * particleCount);

    auto args =
        Sibernetic::toMetalArgs(input, device, outputParticleIndex.get());

    metal.dispatch(particleCount,
                   [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    HashParticlesResult result;
    const auto *ptr = reinterpret_cast<const MetalUInt2 *>(
        outputParticleIndex->contents());
    result.particleIndex = toHostUInt2Vector(ptr, particleCount);
    return result;
  }
};

} // namespace SiberneticTest
