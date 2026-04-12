#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/kernels/ComputeDensityKernel.h"
#include "../../src/types/HostTypes.h"
#include "../utils/common/backend_param_test.h"
#include "../utils/types/types.h"

namespace SiberneticTest {

struct ComputeDensityResult {
  std::vector<float> rho;
};

struct ComputeDensityCase {
  using InputType = Sibernetic::ComputeDensityInput;
  using ResultType = ComputeDensityResult;

  const char *name;
  std::vector<Sibernetic::HostFloat2> neighborMap; // size particleCount * 32
  float massMultWpoly6Coefficient;
  float hScaled2;
  std::vector<uint32_t> particleIndexBack;
  std::vector<float> expectedRho;

  InputType toInput() const {
    return {
        .neighborMap = neighborMap,
        .massMultWpoly6Coefficient = massMultWpoly6Coefficient,
        .hScaled2 = hScaled2,
        .particleIndexBack = particleIndexBack,
        .particleCount = static_cast<uint32_t>(particleIndexBack.size()),
    };
  }

  void verify(const ResultType &result) const {
    ASSERT_EQ(result.rho.size(), expectedRho.size());
    for (size_t i = 0; i < expectedRho.size(); ++i) {
      EXPECT_NEAR(result.rho[i], expectedRho[i], 1e-6f)
          << "rho mismatch at index " << i;
    }
  }
};

static_assert(SiberneticTest::KernelTestCase<ComputeDensityCase>);

class ComputeDensityRunner
    : public TestRunner<ComputeDensityCase, ComputeDensityResult> {};

inline std::vector<std::array<float, 2>>
makeNeighborMap(uint32_t particleCount) {
  return std::vector<Sibernetic::HostFloat2>(
      static_cast<size_t>(particleCount) * 32u, {-1.0f, -1.0f});
}

inline void setNeighbor(std::vector<Sibernetic::HostFloat2> &neighborMap,
                        uint32_t particleId, uint32_t slot, int32_t neighborId,
                        float distance) {
  neighborMap[static_cast<size_t>(particleId) * 32u + slot] = {
      static_cast<float>(neighborId), distance};
}

struct ComputeDensityTestCommon {
  using Case = ComputeDensityCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = [] {
      std::vector<Case> cases;

      {
        constexpr uint32_t kParticleCount = 3u;
        auto neighborMap = makeNeighborMap(kParticleCount);

        setNeighbor(neighborMap, 0u, 0u, 1, 0.0f); // contributes
        setNeighbor(neighborMap, 0u, 1u, 2, 0.5f); // contributes
        setNeighbor(neighborMap, 0u, 2u, 2, 1.5f); // outside h, ignored
        setNeighbor(neighborMap, 1u, 0u, 0, 0.5f); // contributes
        setNeighbor(neighborMap, 2u, 0u, 0, 1.5f); // outside h, ignored

        cases.push_back(ComputeDensityCase{
            "FloorAndCutoffBehavior",
            neighborMap,
            /*massMultWpoly6Coefficient=*/10.0f,
            /*hScaled2=*/1.0f,
            /*particleIndexBack=*/{0u, 1u, 2u},
            /*expectedRho=*/{14.21875f, 10.0f, 10.0f},
        });
      }

      {
        constexpr uint32_t kParticleCount = 3u;
        auto neighborMap = makeNeighborMap(kParticleCount);

        // particle 0: 3 neighbors at r=0.2 -> density sum 3 * 0.46^3
        setNeighbor(neighborMap, 0u, 0u, 1, 0.2f);
        setNeighbor(neighborMap, 0u, 1u, 2, 0.2f);
        setNeighbor(neighborMap, 0u, 2u, 5, 0.2f);

        // particle 1: no neighbors -> floor only

        // particle 2: one neighbor below floor
        setNeighbor(neighborMap, 2u, 0u, 0, 0.3f);

        cases.push_back(ComputeDensityCase{
            "UsesParticleIndexBackMapping",
            neighborMap,
            /*massMultWpoly6Coefficient=*/2.0f,
            /*hScaled2=*/0.5f,
            /*particleIndexBack=*/{2u, 0u, 1u},
            /*expectedRho=*/{0.584016f, 0.25f, 0.25f},
        });
      }

      return cases;
    }();

    return kCases;
  }

  static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
    return info.param.name;
  }
};

static_assert(
    SiberneticTest::SibTestCommon<SiberneticTest::ComputeDensityTestCommon>);

} // namespace SiberneticTest
