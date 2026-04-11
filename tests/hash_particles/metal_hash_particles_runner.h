#pragma once

#include <vector>

#include "../utils/arg/metal_arg_binding.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
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
    MetalKernelContext metal("hashParticles");
    auto *dev = metal.device().get();

    std::vector<MetalFloat4> positions = toMetalFloat4Vector(tc.positions);

    const uint32_t particleCount = static_cast<uint32_t>(positions.size());
    auto positionBuf = makeMetalInputBuffer(dev, positions);
    auto particleIndexBuf =
        makeMetalOutputBuffer(dev, sizeof(MetalUInt2) * particleCount);

    const uint32_t gridCellsX = tc.gridCellsX;
    const uint32_t gridCellsY = tc.gridCellsY;
    const uint32_t gridCellsZ = tc.gridCellsZ;
    const float hashGridCellSizeInv = tc.hashGridCellSizeInv;
    const float xmin = tc.xmin;
    const float ymin = tc.ymin;
    const float zmin = tc.zmin;

    HashParticlesResult result;
    auto outParticleIndex =
        makeMetalOutputFieldBinding<HashParticlesResult, MetalUInt2, HostUInt2>(
            8, particleIndexBuf, particleCount,
            &HashParticlesResult::particleIndex, convertMetalHashParticleIndex);

    std::vector<MetalKernelArg> args = {
        makeMetalInputArg(0, positionBuf),
        makeMetalScalarArg(1, gridCellsX),
        makeMetalScalarArg(2, gridCellsY),
        makeMetalScalarArg(3, gridCellsZ),
        makeMetalScalarArg(4, hashGridCellSizeInv),
        makeMetalScalarArg(5, xmin),
        makeMetalScalarArg(6, ymin),
        makeMetalScalarArg(7, zmin),
        makeMetalScalarArg(9, particleCount),
    };

    runMetalKernelSpecAndStore(metal, particleCount, std::move(args), result,
                               outParticleIndex);
    return result;
  }
};

} // namespace SiberneticTest
