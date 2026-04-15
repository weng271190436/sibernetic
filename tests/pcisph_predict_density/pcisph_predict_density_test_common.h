#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/kernels/PcisphPredictDensityKernel.h"
#include "../../src/types/HostTypes.h"
#include "../utils/common/backend_param_test.h"
#include "../utils/neighbormap/neighbor_map_helpers.h"

namespace SiberneticTest {

struct PcisphPredictDensityResult {
  std::vector<float> predictedRho; // rho[N..2N)
};

struct PcisphPredictDensityCase {
  using InputType = Sibernetic::PcisphPredictDensityInput;
  using ResultType = PcisphPredictDensityResult;

  const char *name;

  // Input data
  std::vector<Sibernetic::HostFloat2> neighborMap;  // size: N * 32
  std::vector<uint32_t> sortedParticleIdBySerialId; // size: N
  float massMultWpoly6Coefficient;
  float h;
  float rho0;
  float simulationScale;
  std::vector<Sibernetic::HostFloat4> sortedPosition; // size: 2*N

  // Expected output
  std::vector<float> expectedPredictedRho; // size: N (rho[N..2N))

  InputType toInput() const {
    return {
        .neighborMap = neighborMap,
        .sortedParticleIdBySerialId = sortedParticleIdBySerialId,
        .massMultWpoly6Coefficient = massMultWpoly6Coefficient,
        .h = h,
        .rho0 = rho0,
        .simulationScale = simulationScale,
        .sortedPosition = sortedPosition,
        .particleCount =
            static_cast<uint32_t>(sortedParticleIdBySerialId.size()),
    };
  }

  void verify(const ResultType &result) const {
    ASSERT_EQ(result.predictedRho.size(), expectedPredictedRho.size());
    for (size_t i = 0; i < expectedPredictedRho.size(); ++i) {
      EXPECT_NEAR(result.predictedRho[i], expectedPredictedRho[i], 1e-2f)
          << "predictedRho mismatch at [" << i << "]";
    }
  }
};

static_assert(SiberneticTest::KernelTestCase<PcisphPredictDensityCase>);

class PcisphPredictDensityRunner {
public:
  virtual ~PcisphPredictDensityRunner() = default;
  virtual PcisphPredictDensityResult
  run(const PcisphPredictDensityCase &tc) = 0;
};

struct PcisphPredictDensityTestCommon {
  using Case = PcisphPredictDensityCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = [] {
      std::vector<Case> cases;

      // All tests use: h=1, simulationScale=1, coeff=2.
      // hScaled=1, hScaled6=1, simulationScale6=1.
      // Min density = hScaled6 * coeff = 2.0.
      constexpr float kH = 1.0f;
      constexpr float kSimScale = 1.0f;
      constexpr float kCoeff = 2.0f;
      constexpr float kMinDensity = 1.0f * kCoeff; // hScaled6 * coeff

      // Helper: make sortedPosition of size 2*N.
      // First half (current positions) is unused by the kernel but must exist.
      // Second half (predicted positions) is set from `predicted`.
      auto makeSortedPos =
          [](const std::vector<Sibernetic::HostFloat4> &predicted) {
            const size_t N = predicted.size();
            std::vector<Sibernetic::HostFloat4> sp(
                N * 2, Sibernetic::HostFloat4{0, 0, 0, 0});
            for (size_t i = 0; i < N; ++i) {
              sp[N + i] = predicted[i];
            }
            return sp;
          };

      // ---- Test 1: SingleParticleMinDensity ----
      // No neighbors → density_accum=0, clamped to hScaled6, * coeff = 2.0.
      {
        constexpr uint32_t N = 1;
        auto neighborMap = makeNeighborMap(N);

        cases.push_back({
            .name = "SingleParticleMinDensity",
            .neighborMap = neighborMap,
            .sortedParticleIdBySerialId = {0},
            .massMultWpoly6Coefficient = kCoeff,
            .h = kH,
            .rho0 = 1000.0f,
            .simulationScale = kSimScale,
            .sortedPosition = makeSortedPos({{1.0f, 2.0f, 3.0f, 0.0f}}),
            .expectedPredictedRho = {kMinDensity},
        });
      }

      // ---- Test 2: TwoNeighborsClose ----
      // Particle 0 has 2 neighbors at distance 0.1 (grid space).
      // Each contributes (h²-d²)³ = (1-0.01)³ = 0.970299.
      // density_accum = 2 * 0.970299 = 1.940598 > hScaled6 = 1 → not clamped.
      // density = 1.940598 * coeff = 3.881196.
      // Particles 1 and 2 have no neighbors → min density.
      {
        constexpr uint32_t N = 3;
        auto neighborMap = makeNeighborMap(N);
        setNeighbor(neighborMap, 0, 0, 1, 0.1f);
        setNeighbor(neighborMap, 0, 1, 2, 0.1f);

        const float d2 = 0.01f;
        const float delta = 1.0f - d2;
        const float contrib = delta * delta * delta; // 0.970299
        const float expected0 = 2.0f * contrib * kCoeff;

        cases.push_back({
            .name = "TwoNeighborsClose",
            .neighborMap = neighborMap,
            .sortedParticleIdBySerialId = {0, 1, 2},
            .massMultWpoly6Coefficient = kCoeff,
            .h = kH,
            .rho0 = 1000.0f,
            .simulationScale = kSimScale,
            .sortedPosition = makeSortedPos({
                {0.0f, 0.0f, 0.0f, 0.0f},
                {0.1f, 0.0f, 0.0f, 0.0f},
                {-0.1f, 0.0f, 0.0f, 0.0f},
            }),
            .expectedPredictedRho = {expected0, kMinDensity, kMinDensity},
        });
      }

      // ---- Test 3: TwoParticlesFar ----
      // Neighbor at distance 2.0 > h=1.0, no contribution.
      {
        constexpr uint32_t N = 2;
        auto neighborMap = makeNeighborMap(N);
        setNeighbor(neighborMap, 0, 0, 1, 2.0f);

        cases.push_back({
            .name = "TwoParticlesFar",
            .neighborMap = neighborMap,
            .sortedParticleIdBySerialId = {0, 1},
            .massMultWpoly6Coefficient = kCoeff,
            .h = kH,
            .rho0 = 1000.0f,
            .simulationScale = kSimScale,
            .sortedPosition = makeSortedPos({
                {0.0f, 0.0f, 0.0f, 0.0f},
                {2.0f, 0.0f, 0.0f, 0.0f},
            }),
            .expectedPredictedRho = {kMinDensity, kMinDensity},
        });
      }

      // ---- Test 4: NeighborAtExactCutoff ----
      // Distance == h: r_ij2 == h2, excluded by strict < comparison.
      {
        constexpr uint32_t N = 2;
        auto neighborMap = makeNeighborMap(N);
        setNeighbor(neighborMap, 0, 0, 1, 1.0f);

        cases.push_back({
            .name = "NeighborAtExactCutoff",
            .neighborMap = neighborMap,
            .sortedParticleIdBySerialId = {0, 1},
            .massMultWpoly6Coefficient = kCoeff,
            .h = kH,
            .rho0 = 1000.0f,
            .simulationScale = kSimScale,
            .sortedPosition = makeSortedPos({
                {0.0f, 0.0f, 0.0f, 0.0f},
                {1.0f, 0.0f, 0.0f, 0.0f},
            }),
            .expectedPredictedRho = {kMinDensity, kMinDensity},
        });
      }

      // ---- Test 5: NonIdentityIndexBack ----
      // sortedParticleIdBySerialId = {2, 0, 1}: serial→sorted mapping is
      // scrambled. Particle sorted 2 (serial 0) has 2 neighbors (sorted 0 and
      // sorted 1). Predicted positions:
      //   sorted 0: (0, 0, 0)
      //   sorted 1: (0.1, 0, 0)
      //   sorted 2: (0.2, 0, 0)
      // Sorted 2 → neighbor sorted 0: d=0.2, r_ij2=0.04
      //   contrib = (1-0.04)³ = 0.884736
      // Sorted 2 → neighbor sorted 1: d=0.1, r_ij2=0.01
      //   contrib = (1-0.01)³ = 0.970299
      // density_accum = 1.855035 > 1.0 → not clamped.
      {
        constexpr uint32_t N = 3;
        auto neighborMap = makeNeighborMap(N);
        // Sorted particle 2 has neighbors at slots 0 and 1.
        setNeighbor(neighborMap, 2, 0, 0, 0.2f);
        setNeighbor(neighborMap, 2, 1, 1, 0.1f);

        const float d02 = 0.04f; // distance 0.2 squared
        const float d12 = 0.01f; // distance 0.1 squared
        const float delta0 = 1.0f - d02;
        const float delta1 = 1.0f - d12;
        const float accum = delta0 * delta0 * delta0 + delta1 * delta1 * delta1;
        const float expected2 = accum * kCoeff;

        cases.push_back({
            .name = "NonIdentityIndexBack",
            .neighborMap = neighborMap,
            .sortedParticleIdBySerialId = {2, 0, 1},
            .massMultWpoly6Coefficient = kCoeff,
            .h = kH,
            .rho0 = 1000.0f,
            .simulationScale = kSimScale,
            .sortedPosition = makeSortedPos({
                {0.0f, 0.0f, 0.0f, 0.0f},
                {0.1f, 0.0f, 0.0f, 0.0f},
                {0.2f, 0.0f, 0.0f, 0.0f},
            }),
            // rho[N+0], rho[N+1], rho[N+2]
            .expectedPredictedRho = {kMinDensity, kMinDensity, expected2},
        });
      }

      // ---- Test 6: NonUnitSimulationScale ----
      // simulationScale=2 exercises the simulationScale^6 multiplication and
      // the hScaled^6 floor at a non-trivial value.
      // hScaled = h * simScale = 2, hScaled6 = 64.
      // simulationScale6 = 64.
      // Particle 0 has 2 neighbors each at grid-distance 0.1.
      // Each contrib = (1-0.01)^3 = 0.970299.
      // density_accum = 1.940598 → * 64 = 124.198 > 64 → not clamped.
      // density = 124.198 * coeff = 248.397.
      // Particles 1,2: no neighbors → floor 64 * coeff = 128.
      {
        constexpr uint32_t N = 3;
        constexpr float simScale = 2.0f;
        constexpr float simScale6 =
            simScale * simScale * simScale * simScale * simScale * simScale;
        constexpr float hScaled6 = kH * simScale * kH * simScale * kH *
                                   simScale * kH * simScale * kH * simScale *
                                   kH * simScale;
        const float floorDensity = hScaled6 * kCoeff;

        auto neighborMap = makeNeighborMap(N);
        setNeighbor(neighborMap, 0, 0, 1, 0.1f);
        setNeighbor(neighborMap, 0, 1, 2, 0.1f);

        const float d2 = 0.01f;
        const float delta = 1.0f - d2;
        const float contrib = delta * delta * delta;
        const float expected0 = 2.0f * contrib * simScale6 * kCoeff;

        cases.push_back({
            .name = "NonUnitSimulationScale",
            .neighborMap = neighborMap,
            .sortedParticleIdBySerialId = {0, 1, 2},
            .massMultWpoly6Coefficient = kCoeff,
            .h = kH,
            .rho0 = 1000.0f,
            .simulationScale = simScale,
            .sortedPosition = makeSortedPos({
                {0.0f, 0.0f, 0.0f, 0.0f},
                {0.1f, 0.0f, 0.0f, 0.0f},
                {-0.1f, 0.0f, 0.0f, 0.0f},
            }),
            .expectedPredictedRho = {expected0, floorDensity, floorDensity},
        });
      }

      // ---- Test 7: ThreeDimensionalDistance ----
      // Neighbor offsets have nonzero x, y, and z components so that all three
      // terms of the dot-product distance are exercised.
      // Particle 0 has 2 neighbors:
      //   neighbor 1 at (0.1, 0.2, 0.1): d²=0.06, delta=0.94, (0.94)³
      //   neighbor 2 at (0.2, 0.1, 0.2): d²=0.09, delta=0.91, (0.91)³
      // density_accum = 0.830584 + 0.753571 = 1.584155 > 1 → not clamped.
      // density = 1.584155 * coeff = 3.168310.
      {
        constexpr uint32_t N = 3;
        auto neighborMap = makeNeighborMap(N);
        setNeighbor(neighborMap, 0, 0, 1, 0.1f); // distance field unused
        setNeighbor(neighborMap, 0, 1, 2, 0.1f);

        const float d2_a = 0.1f * 0.1f + 0.2f * 0.2f + 0.1f * 0.1f; // 0.06
        const float d2_b = 0.2f * 0.2f + 0.1f * 0.1f + 0.2f * 0.2f; // 0.09
        const float delta_a = 1.0f - d2_a;
        const float delta_b = 1.0f - d2_b;
        const float accum =
            delta_a * delta_a * delta_a + delta_b * delta_b * delta_b;
        const float expected0 = accum * kCoeff;

        cases.push_back({
            .name = "ThreeDimensionalDistance",
            .neighborMap = neighborMap,
            .sortedParticleIdBySerialId = {0, 1, 2},
            .massMultWpoly6Coefficient = kCoeff,
            .h = kH,
            .rho0 = 1000.0f,
            .simulationScale = kSimScale,
            .sortedPosition = makeSortedPos({
                {0.0f, 0.0f, 0.0f, 0.0f},
                {0.1f, 0.2f, 0.1f, 0.0f},
                {0.2f, 0.1f, 0.2f, 0.0f},
            }),
            .expectedPredictedRho = {expected0, kMinDensity, kMinDensity},
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

} // namespace SiberneticTest
