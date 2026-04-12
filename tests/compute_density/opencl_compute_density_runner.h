#pragma once

#include <vector>

#include "../utils/arg/opencl_arg_binding.h"
#include "compute_density_test_common.h"

namespace SiberneticTest {

class OpenCLComputeDensityRunner : public ComputeDensityRunner {
public:
  ComputeDensityResult run(const ComputeDensityCase &tc) override {
    const cl_uint particleCount =
        static_cast<cl_uint>(tc.particleIndexBack.size());
    if (tc.neighborMap.size() != static_cast<size_t>(particleCount) * 32u) {
      throw std::runtime_error("neighborMap size must be particleCount * 32");
    }

    std::vector<cl_float2> clNeighborMap(tc.neighborMap.size());
    for (size_t i = 0; i < tc.neighborMap.size(); ++i) {
      clNeighborMap[i].s[0] = tc.neighborMap[i][0];
      clNeighborMap[i].s[1] = tc.neighborMap[i][1];
    }

    std::vector<cl_uint> clParticleIndexBack(tc.particleIndexBack.begin(),
                                             tc.particleIndexBack.end());

    ComputeDensityResult result;
    auto outRho =
        makeCLOutputFieldBinding<ComputeDensityResult, cl_float, float>(
            3, particleCount, &ComputeDensityResult::rho,
            [](const std::vector<cl_float> &src) {
              return std::vector<float>(src.begin(), src.end());
            });

    runCLKernelSpecAndStore(
        "pcisph_computeDensity", particleCount,
        {
            CLScalarArg::make<cl_float>(1, tc.massMultWpoly6Coefficient),
            CLScalarArg::make<cl_float>(2, tc.hScaled2),
            CLScalarArg::make<cl_uint>(5, particleCount),
        },
        {
            CLInputBuffer::make<cl_float2>(0, clNeighborMap),
            CLInputBuffer::make<cl_uint>(4, clParticleIndexBack),
        },
        result, outRho);

    return result;
  }
};

} // namespace SiberneticTest
