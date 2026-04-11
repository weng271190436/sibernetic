#pragma once

#include <vector>

#include "../utils/arg/metal_arg_binding.h"
#include "../utils/convert/metal_convert_utils.h"
#include "../utils/types/metal_types.h"
#include "hash_particles_test_common.h"

namespace SiberneticTest {

inline std::vector<HostUInt2>
convertMetalHashParticleIndex(const MetalUInt2 *src, size_t n) {
  return toHostUInt2Vector(src, n);
}

class MetalHashParticlesRunner : public HashParticlesRunner {
public:
  HashParticlesResult run(const HashParticlesCase &tc) override {
    std::vector<MetalFloat4> positions = toMetalFloat4Vector(tc.positions);

    const uint32_t particleCount = static_cast<uint32_t>(positions.size());

    HashParticlesResult result;
    auto outParticleIndex =
        makeMetalOutputFieldSpec<HashParticlesResult, MetalUInt2, HostUInt2>(
            8, particleCount, &HashParticlesResult::particleIndex,
            convertMetalHashParticleIndex);

    runMetalKernelSpecAndStore(
        "hashParticles", particleCount,
        {
            MetalScalarArg::make(1, tc.gridCellsX),
            MetalScalarArg::make(2, tc.gridCellsY),
            MetalScalarArg::make(3, tc.gridCellsZ),
            MetalScalarArg::make(4, tc.hashGridCellSizeInv),
            MetalScalarArg::make(5, tc.xmin),
            MetalScalarArg::make(6, tc.ymin),
            MetalScalarArg::make(7, tc.zmin),
            MetalScalarArg::make(9, particleCount),
        },
        {
            MetalInputHostBuffer::make(0, positions),
        },
        result, outParticleIndex);
    return result;
  }
};

} // namespace SiberneticTest
