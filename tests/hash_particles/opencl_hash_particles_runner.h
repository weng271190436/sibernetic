#pragma once

#include <vector>

#include "../utils/arg/opencl_arg_binding.h"
#include "../utils/convert/opencl_convert_utils.h"
#include "hash_particles_test_common.h"

namespace SiberneticTest {

inline std::vector<HostUInt2>
convertHashParticleIndex(const std::vector<cl_uint2> &src) {
  return toHostUInt2Vector(src);
}

class OpenCLHashParticlesRunner : public HashParticlesRunner {
public:
  HashParticlesResult run(const HashParticlesCase &tc) override {
    std::vector<cl_float4> clPositions = toCLFloat4Vector(tc.positions);
    const cl_uint particleCount = static_cast<cl_uint>(clPositions.size());
    HashParticlesResult result;
    auto outParticleIndex =
        makeCLOutputFieldBinding<HashParticlesResult, cl_uint2, HostUInt2>(
            8, particleCount, &HashParticlesResult::particleIndex,
            convertHashParticleIndex);

    runCLKernelSpecAndStore(
        "hashParticles", particleCount,
        {
            CLScalarArg::make<cl_uint>(1, static_cast<cl_uint>(tc.gridCellsX)),
            CLScalarArg::make<cl_uint>(2, static_cast<cl_uint>(tc.gridCellsY)),
            CLScalarArg::make<cl_uint>(3, static_cast<cl_uint>(tc.gridCellsZ)),
            CLScalarArg::make<cl_float>(
                4, static_cast<cl_float>(tc.hashGridCellSizeInv)),
            CLScalarArg::make<cl_float>(5, static_cast<cl_float>(tc.xmin)),
            CLScalarArg::make<cl_float>(6, static_cast<cl_float>(tc.ymin)),
            CLScalarArg::make<cl_float>(7, static_cast<cl_float>(tc.zmin)),
            CLScalarArg::make<cl_uint>(9, particleCount),
        },
        {
            CLInputBuffer::make<cl_float4>(0, clPositions),
        },
        result, outParticleIndex);

    return result;
  }
};

} // namespace SiberneticTest
