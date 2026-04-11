#pragma once

#include <vector>

#include "../utils/metal_context.h"
#include "../utils/metal_helpers.h"
#include "../utils/metal_types.h"
#include "hash_particles_test_common.h"

namespace SiberneticTest {

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

    metal.dispatch(particleCount, [&](MTL::ComputeCommandEncoder *enc) {
      enc->setBuffer(positionBuf.get(), 0, 0);
      enc->setBytes(&gridCellsX, sizeof(gridCellsX), 1);
      enc->setBytes(&gridCellsY, sizeof(gridCellsY), 2);
      enc->setBytes(&gridCellsZ, sizeof(gridCellsZ), 3);
      enc->setBytes(&hashGridCellSizeInv, sizeof(hashGridCellSizeInv), 4);
      enc->setBytes(&xmin, sizeof(xmin), 5);
      enc->setBytes(&ymin, sizeof(ymin), 6);
      enc->setBytes(&zmin, sizeof(zmin), 7);
      enc->setBuffer(particleIndexBuf.get(), 0, 8);
      enc->setBytes(&particleCount, sizeof(particleCount), 9);
    });

    HashParticlesResult result;
    const auto *out =
        reinterpret_cast<const MetalUInt2 *>(particleIndexBuf->contents());
    result.particleIndex = toHostUInt2Vector(out, particleCount);
    return result;
  }
};

} // namespace SiberneticTest
