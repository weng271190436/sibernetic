#pragma once

#include <vector>

#include "../utils/arg/metal_arg_binding.h"
#include "../utils/convert/metal_convert_utils.h"
#include "../utils/types/metal_types.h"
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

    std::vector<MetalFloat2> neighborMap(tc.neighborMap.size());
    for (size_t i = 0; i < tc.neighborMap.size(); ++i) {
      neighborMap[i].s[0] = tc.neighborMap[i][0];
      neighborMap[i].s[1] = tc.neighborMap[i][1];
    }

    std::vector<uint32_t> particleIndexBack = tc.particleIndexBack;

    ComputeDensityResult result;
    auto outRho = makeMetalOutputFieldSpec<ComputeDensityResult, float, float>(
        3, particleCount, &ComputeDensityResult::rho,
        [](const float *src, size_t n) { return toHostVector(src, n); });

    runMetalKernelSpecAndStore(
        "pcisph_computeDensity", particleCount,
        {
            MetalScalarArg::make(1, tc.massMultWpoly6Coefficient),
            MetalScalarArg::make(2, tc.hScaled2),
            MetalScalarArg::make(5, particleCount),
        },
        {
            MetalInputHostBuffer::make(0, neighborMap),
            MetalInputHostBuffer::make(4, particleIndexBack),
        },
        result, outRho);

    return result;
  }
};

} // namespace SiberneticTest
