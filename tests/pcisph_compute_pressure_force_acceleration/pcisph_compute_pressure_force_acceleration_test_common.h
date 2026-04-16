#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/kernels/PcisphComputePressureForceAccelerationKernel.h"
#include "../utils/common/backend_param_test.h"

namespace SiberneticTest {

struct PcisphComputePressureForceAccelerationResult {
  // acceleration[N..2N) — the pressure-force acceleration for each particle.
  std::vector<Sibernetic::HostFloat4> pressureAcceleration;
};

struct PcisphComputePressureForceAccelerationCase {
  using InputType = Sibernetic::PcisphComputePressureForceAccelerationInput;
  using ResultType = PcisphComputePressureForceAccelerationResult;

  const char *name;

  // Input data
  std::vector<Sibernetic::HostFloat2> neighborMap;
  std::vector<float> pressure;
  std::vector<float> rho; // size: 2*N (reads [N..2N))
  std::vector<Sibernetic::HostFloat4> sortedPosition;
  std::vector<uint32_t> sortedParticleIdBySerialId;
  float delta;
  float massMultGradWspikyCoefficient;
  float h;
  float simulationScale;
  float restDensity;
  std::vector<Sibernetic::HostFloat4> originalPosition; // .w = type
  std::vector<Sibernetic::HostUInt2> sortedCellAndSerialId;

  // Expected output
  std::vector<Sibernetic::HostFloat4> expectedPressureAcceleration;

  InputType toInput() const {
    return {
        .neighborMap = neighborMap,
        .pressure = pressure,
        .rho = rho,
        .sortedPosition = sortedPosition,
        .sortedParticleIdBySerialId = sortedParticleIdBySerialId,
        .delta = delta,
        .massMultGradWspikyCoefficient = massMultGradWspikyCoefficient,
        .h = h,
        .simulationScale = simulationScale,
        .restDensity = restDensity,
        .originalPosition = originalPosition,
        .sortedCellAndSerialId = sortedCellAndSerialId,
        .particleCount =
            static_cast<uint32_t>(sortedParticleIdBySerialId.size()),
    };
  }

  void verify(const ResultType &result) const {
    ASSERT_EQ(result.pressureAcceleration.size(),
              expectedPressureAcceleration.size());
    for (size_t i = 0; i < expectedPressureAcceleration.size(); ++i) {
      for (int c = 0; c < 4; ++c) {
        EXPECT_NEAR(result.pressureAcceleration[i][c],
                    expectedPressureAcceleration[i][c], 1e-4f)
            << "pressureAcceleration mismatch at [" << i << "][" << c << "]";
      }
    }
  }
};

static_assert(
    SiberneticTest::KernelTestCase<PcisphComputePressureForceAccelerationCase>);

class PcisphComputePressureForceAccelerationRunner {
public:
  virtual ~PcisphComputePressureForceAccelerationRunner() = default;
  virtual PcisphComputePressureForceAccelerationResult
  run(const PcisphComputePressureForceAccelerationCase &tc) = 0;
};

// Helper: create an empty neighborMap for N particles (all slots = no
// particle).
inline std::vector<Sibernetic::HostFloat2>
makeEmptyNeighborMap(uint32_t particleCount) {
  std::vector<Sibernetic::HostFloat2> nm(static_cast<size_t>(particleCount) *
                                         32);
  for (auto &e : nm) {
    e = {-1.0f, -1.0f};
  }
  return nm;
}

struct PcisphComputePressureForceAccelerationTestCommon {
  using Case = PcisphComputePressureForceAccelerationCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = [] {
      std::vector<Case> cases;

      // Common parameters used across tests.
      const float h = 2.0f;
      const float simScale = 0.5f;
      const float hScaled = h * simScale; // 1.0
      const float rho0 = 1000.0f;
      const float delta = 0.5f;
      const float gradCoeff = -1.0f; // massMultGradWspikyCoefficient

      // ---- Test 1: BoundaryParticleOutputsZero ----
      // Particle with originalPosition.w == 3 → acceleration = 0.
      {
        auto nm = makeEmptyNeighborMap(1);
        cases.push_back({
            .name = "BoundaryParticleOutputsZero",
            .neighborMap = nm,
            .pressure = {10.0f},
            .rho = {0.0f, 1000.0f},
            .sortedPosition = {{1.0f, 2.0f, 3.0f, 0.0f}},
            .sortedParticleIdBySerialId = {0},
            .delta = delta,
            .massMultGradWspikyCoefficient = gradCoeff,
            .h = h,
            .simulationScale = simScale,
            .restDensity = rho0,
            .originalPosition = {{1.0f, 2.0f, 3.0f, 3.0f}}, // type 3 = boundary
            .sortedCellAndSerialId = {{0, 0}},
            .expectedPressureAcceleration = {{0.0f, 0.0f, 0.0f, 0.0f}},
        });
      }

      // ---- Test 2: SingleParticleNoNeighbors ----
      // Fluid particle with no neighbors → acceleration = 0.
      {
        auto nm = makeEmptyNeighborMap(1);
        cases.push_back({
            .name = "SingleParticleNoNeighbors",
            .neighborMap = nm,
            .pressure = {50.0f},
            .rho = {0.0f, 1000.0f},
            .sortedPosition = {{1.0f, 2.0f, 3.0f, 0.0f}},
            .sortedParticleIdBySerialId = {0},
            .delta = delta,
            .massMultGradWspikyCoefficient = gradCoeff,
            .h = h,
            .simulationScale = simScale,
            .restDensity = rho0,
            .originalPosition = {{1.0f, 2.0f, 3.0f, 2.1f}}, // fluid type
            .sortedCellAndSerialId = {{0, 0}},
            .expectedPressureAcceleration = {{0.0f, 0.0f, 0.0f, 0.0f}},
        });
      }

      // ---- Test 3: TwoParticlesSymmetricPressure ----
      // Two particles at equal pressure → equal and opposite forces (Newton 3).
      // Particle 0 at (0,0,0), particle 1 at (1,0,0). Identity index map.
      // hScaled = 1.0, distance along x in grid space = 1.0,
      // r_ij in simulation space = 1.0 * 0.5 = 0.5 (< hScaled = 1.0).
      // Wait - neighborMap stores distances already in simulation units.
      // So r_ij = 0.5 (set in neighborMap).
      //
      // For particle 0 (neighbor = particle 1, r = 0.5):
      //   value = -(1.0 - 0.5)^2 * 0.5 * (100 + 100) / 1000
      //         = -0.25 * 0.5 * 200 / 1000 = -0.025
      //   direction = (pos[0] - pos[1]) * simScale = (-1,0,0) * 0.5 =
      //   (-0.5,0,0) contribution = -0.025 * (-0.5,0,0) / 0.5 = (0.025, 0, 0)
      //   result = (0.025, 0, 0) * gradCoeff / rho_predicted[0]
      //          = (0.025, 0, 0) * (-1.0) / 1000 = (-0.000025, 0, 0)
      //
      // For particle 1 (neighbor = particle 0, r = 0.5):
      //   direction = (pos[1] - pos[0]) * simScale = (0.5,0,0)
      //   contribution = -0.025 * (0.5,0,0) / 0.5 = (-0.025, 0, 0)
      //   result = (-0.025, 0, 0) * (-1.0) / 1000 = (0.000025, 0, 0)
      {
        const uint32_t N = 2;
        auto nm = makeEmptyNeighborMap(N);
        // Particle 0's neighbor: particle 1 at distance 0.5
        nm[0 * 32 + 0] = {1.0f, 0.5f};
        // Particle 1's neighbor: particle 0 at distance 0.5
        nm[1 * 32 + 0] = {0.0f, 0.5f};

        cases.push_back({
            .name = "TwoParticlesSymmetricPressure",
            .neighborMap = nm,
            .pressure = {100.0f, 100.0f},
            .rho = {0.0f, 0.0f, 1000.0f, 1000.0f}, // [N..2N) = predicted
            .sortedPosition = {{0.0f, 0.0f, 0.0f, 0.0f},
                               {1.0f, 0.0f, 0.0f, 0.0f}},
            .sortedParticleIdBySerialId = {0, 1},
            .delta = delta,
            .massMultGradWspikyCoefficient = gradCoeff,
            .h = h,
            .simulationScale = simScale,
            .restDensity = rho0,
            .originalPosition = {{0.0f, 0.0f, 0.0f, 2.1f},
                                 {1.0f, 0.0f, 0.0f, 2.1f}},
            .sortedCellAndSerialId = {{0, 0}, {0, 1}},
            .expectedPressureAcceleration = {{-0.000025f, 0.0f, 0.0f, 0.0f},
                                             {0.000025f, 0.0f, 0.0f, 0.0f}},
        });
      }

      // ---- Test 4: TwoParticlesUnequalPressure ----
      // p0 = 200, p1 = 0. Same geometry as test 3.
      // For particle 0 (neighbor = 1, r = 0.5):
      //   value = -(0.5)^2 * 0.5 * (200 + 0) / 1000 = -0.025
      //   direction = (-0.5, 0, 0), contrib = -0.025 * (-0.5,0,0) / 0.5
      //            = (0.025, 0, 0)
      //   result = 0.025 * (-1) / 1000 = -0.000025
      //
      // For particle 1 (neighbor = 0, r = 0.5):
      //   value = -(0.5)^2 * 0.5 * (0 + 200) / 1000 = -0.025
      //   direction = (0.5, 0, 0), contrib = -0.025 * (0.5,0,0) / 0.5
      //            = (-0.025, 0, 0)
      //   result = -0.025 * (-1) / 1000 = 0.000025
      //
      // Same magnitude as symmetric case because the formula uses (p_i+p_j)/2.
      {
        const uint32_t N = 2;
        auto nm = makeEmptyNeighborMap(N);
        nm[0 * 32 + 0] = {1.0f, 0.5f};
        nm[1 * 32 + 0] = {0.0f, 0.5f};

        cases.push_back({
            .name = "TwoParticlesUnequalPressure",
            .neighborMap = nm,
            .pressure = {200.0f, 0.0f},
            .rho = {0.0f, 0.0f, 1000.0f, 1000.0f},
            .sortedPosition = {{0.0f, 0.0f, 0.0f, 0.0f},
                               {1.0f, 0.0f, 0.0f, 0.0f}},
            .sortedParticleIdBySerialId = {0, 1},
            .delta = delta,
            .massMultGradWspikyCoefficient = gradCoeff,
            .h = h,
            .simulationScale = simScale,
            .restDensity = rho0,
            .originalPosition = {{0.0f, 0.0f, 0.0f, 2.1f},
                                 {1.0f, 0.0f, 0.0f, 2.1f}},
            .sortedCellAndSerialId = {{0, 0}, {0, 1}},
            .expectedPressureAcceleration = {{-0.000025f, 0.0f, 0.0f, 0.0f},
                                             {0.000025f, 0.0f, 0.0f, 0.0f}},
        });
      }

      // ---- Test 5: NeighborAtCutoffIgnored ----
      // r_ij == hScaled → skipped (distance >= hScaled check).
      {
        const uint32_t N = 2;
        auto nm = makeEmptyNeighborMap(N);
        // Particle 0's neighbor at exactly hScaled distance
        nm[0 * 32 + 0] = {1.0f, hScaled}; // r = 1.0 == hScaled
        nm[1 * 32 + 0] = {0.0f, hScaled};

        cases.push_back({
            .name = "NeighborAtCutoffIgnored",
            .neighborMap = nm,
            .pressure = {100.0f, 100.0f},
            .rho = {0.0f, 0.0f, 1000.0f, 1000.0f},
            .sortedPosition = {{0.0f, 0.0f, 0.0f, 0.0f},
                               {2.0f, 0.0f, 0.0f, 0.0f}},
            .sortedParticleIdBySerialId = {0, 1},
            .delta = delta,
            .massMultGradWspikyCoefficient = gradCoeff,
            .h = h,
            .simulationScale = simScale,
            .restDensity = rho0,
            .originalPosition = {{0.0f, 0.0f, 0.0f, 2.1f},
                                 {2.0f, 0.0f, 0.0f, 2.1f}},
            .sortedCellAndSerialId = {{0, 0}, {0, 1}},
            .expectedPressureAcceleration = {{0.0f, 0.0f, 0.0f, 0.0f},
                                             {0.0f, 0.0f, 0.0f, 0.0f}},
        });
      }

      // ---- Test 6: CloseRangeCorrection ----
      // r_ij < hScaled/4 = 0.25 → uses close-range formula.
      // Particle 0 at (0,0,0), particle 1 at (0.2,0,0).
      // r_ij = 0.1 (in neighborMap, already simulation-space distance).
      // hScaled = 1.0, hScaled/4 = 0.25. r_ij = 0.1 < 0.25 → close-range.
      //
      // Close-range value:
      //   value = -(0.25 - 0.1)^2 * 0.5 * (rho0*delta) / rho_predicted[j]
      //         = -(0.15)^2 * 0.5 * (1000*0.5) / 1000
      //         = -0.0225 * 0.5 * 500 / 1000 = -0.005625
      //   direction = (0 - 0.2, 0, 0) * 0.5 = (-0.1, 0, 0)
      //   contrib = -0.005625 * (-0.1, 0, 0) / 0.1 = (0.005625, 0, 0)
      //   result = 0.005625 * (-1) / 1000 = -0.000005625
      {
        const uint32_t N = 2;
        auto nm = makeEmptyNeighborMap(N);
        nm[0 * 32 + 0] = {1.0f, 0.1f};
        nm[1 * 32 + 0] = {0.0f, 0.1f};

        // Close-range for particle 1 (neighbor = 0):
        //   direction = (0.2-0, 0, 0)*0.5 = (0.1, 0, 0)
        //   contrib = -0.005625 * (0.1, 0, 0) / 0.1 = (-0.005625, 0, 0)
        //   result = -0.005625 * (-1) / 1000 = 0.000005625
        cases.push_back({
            .name = "CloseRangeCorrection",
            .neighborMap = nm,
            .pressure = {100.0f, 100.0f},
            .rho = {0.0f, 0.0f, 1000.0f, 1000.0f},
            .sortedPosition = {{0.0f, 0.0f, 0.0f, 0.0f},
                               {0.2f, 0.0f, 0.0f, 0.0f}},
            .sortedParticleIdBySerialId = {0, 1},
            .delta = delta,
            .massMultGradWspikyCoefficient = gradCoeff,
            .h = h,
            .simulationScale = simScale,
            .restDensity = rho0,
            .originalPosition = {{0.0f, 0.0f, 0.0f, 2.1f},
                                 {0.2f, 0.0f, 0.0f, 2.1f}},
            .sortedCellAndSerialId = {{0, 0}, {0, 1}},
            .expectedPressureAcceleration = {{-5.625e-6f, 0.0f, 0.0f, 0.0f},
                                             {5.625e-6f, 0.0f, 0.0f, 0.0f}},
        });
      }

      // ---- Test 7: NonIdentityIndexBack ----
      // 3 particles with scrambled sortedParticleIdBySerialId = {2, 0, 1}.
      // Only particle at sorted id 0 has a neighbor; others have none.
      // serial 0 → sorted 2, serial 1 → sorted 0, serial 2 → sorted 1.
      // sortedCellAndSerialId: sorted 0 → serial 1, sorted 1 → serial 2,
      //                        sorted 2 → serial 0.
      //
      // Particle serial 1 → sorted 0: fluid type, has neighbor at sorted 1.
      //   r = 0.5, hScaled = 1.0
      //   value = -(0.5)^2 * 0.5 * (50 + 75) / 800 = -0.25*0.5*125/800
      //         = -0.01953125
      //   direction = (pos[0] - pos[1]) * 0.5 = (0-1,0,0)*0.5 = (-0.5,0,0)
      //   contrib = -0.01953125 * (-0.5,0,0) / 0.5 = (0.01953125, 0, 0)
      //   result = 0.01953125 * (-1) / rho_predicted[sorted 0]
      //          = 0.01953125 * (-1) / 900 = -2.170139e-5
      //
      // Particle serial 0 → sorted 2: no neighbors → 0
      // Particle serial 2 → sorted 1: no neighbors → 0
      {
        const uint32_t N = 3;
        auto nm = makeEmptyNeighborMap(N);
        // sorted 0 sees sorted 1 at distance 0.5
        nm[0 * 32 + 0] = {1.0f, 0.5f};

        cases.push_back({
            .name = "NonIdentityIndexBack",
            .neighborMap = nm,
            .pressure = {50.0f, 75.0f, 30.0f}, // indexed by sorted id
            .rho = {0.0f, 0.0f, 0.0f, 900.0f, 800.0f,
                    1000.0f}, // [N..2N) predicted densities
            .sortedPosition = {{0.0f, 0.0f, 0.0f, 0.0f},  // sorted 0
                               {1.0f, 0.0f, 0.0f, 0.0f},  // sorted 1
                               {3.0f, 0.0f, 0.0f, 0.0f}}, // sorted 2
            .sortedParticleIdBySerialId = {2, 0, 1},
            .delta = delta,
            .massMultGradWspikyCoefficient = gradCoeff,
            .h = h,
            .simulationScale = simScale,
            .restDensity = rho0,
            .originalPosition = {{3.0f, 0.0f, 0.0f, 2.1f},  // serial 0
                                 {0.0f, 0.0f, 0.0f, 2.1f},  // serial 1
                                 {1.0f, 0.0f, 0.0f, 2.1f}}, // serial 2
            // sorted 0 → serial 1, sorted 1 → serial 2, sorted 2 → serial 0
            .sortedCellAndSerialId = {{0, 1}, {0, 2}, {0, 0}},
            // Expected: sorted 0 gets force, sorted 1 and 2 get zero.
            // result.x = 0.01953125 * (-1) / 900 = -2.170139e-5
            .expectedPressureAcceleration = {{-2.170139e-5f, 0.0f, 0.0f, 0.0f},
                                             {0.0f, 0.0f, 0.0f, 0.0f},
                                             {0.0f, 0.0f, 0.0f, 0.0f}},
        });
      }

      // ---- Test 8: MultipleNeighbors ----
      // Particle 0 has two neighbors (1 and 2) to verify accumulation.
      // Particle 0 at (0,0,0), particle 1 at (1,0,0), particle 2 at (0,1,0).
      // Both neighbors at r = 0.5 in simulation space.
      // Each contributes the same magnitude along orthogonal axes.
      //
      // For particle 0:
      //   Neighbor 1: value = -(0.5)^2*0.5*(100+100)/1000 = -0.025
      //     dir = (-0.5,0,0), contrib = -0.025*(-0.5,0,0)/0.5 = (0.025,0,0)
      //   Neighbor 2: value = -0.025
      //     dir = (0,-0.5,0), contrib = -0.025*(0,-0.5,0)/0.5 = (0,0.025,0)
      //   Sum = (0.025, 0.025, 0)
      //   result = (0.025,0.025,0) * (-1)/1000 = (-2.5e-5, -2.5e-5, 0, 0)
      {
        const uint32_t N = 3;
        auto nm = makeEmptyNeighborMap(N);
        // Particle 0: two neighbors
        nm[0 * 32 + 0] = {1.0f, 0.5f};
        nm[0 * 32 + 1] = {2.0f, 0.5f};
        // Particle 1: neighbor 0
        nm[1 * 32 + 0] = {0.0f, 0.5f};
        // Particle 2: neighbor 0
        nm[2 * 32 + 0] = {0.0f, 0.5f};

        cases.push_back({
            .name = "MultipleNeighbors",
            .neighborMap = nm,
            .pressure = {100.0f, 100.0f, 100.0f},
            .rho = {0.0f, 0.0f, 0.0f, 1000.0f, 1000.0f, 1000.0f},
            .sortedPosition = {{0.0f, 0.0f, 0.0f, 0.0f},
                               {1.0f, 0.0f, 0.0f, 0.0f},
                               {0.0f, 1.0f, 0.0f, 0.0f}},
            .sortedParticleIdBySerialId = {0, 1, 2},
            .delta = delta,
            .massMultGradWspikyCoefficient = gradCoeff,
            .h = h,
            .simulationScale = simScale,
            .restDensity = rho0,
            .originalPosition = {{0.0f, 0.0f, 0.0f, 2.1f},
                                 {1.0f, 0.0f, 0.0f, 2.1f},
                                 {0.0f, 1.0f, 0.0f, 2.1f}},
            .sortedCellAndSerialId = {{0, 0}, {0, 1}, {0, 2}},
            .expectedPressureAcceleration = {{-2.5e-5f, -2.5e-5f, 0.0f, 0.0f},
                                             {2.5e-5f, 0.0f, 0.0f, 0.0f},
                                             {0.0f, 2.5e-5f, 0.0f, 0.0f}},
        });
      }

      // ---- Test 9: DiagonalDisplacement ----
      // Two particles along (1,1,0) diagonal to verify multi-component forces.
      // Particle 0 at (0,0,0), particle 1 at (1,1,0).
      // r_ij = sqrt(1^2+1^2)*simScale = sqrt(2)*0.5 ≈ 0.7071.
      // hScaled = 1.0. r < hScaled, not close-range (0.7071 > 0.25).
      //
      // For particle 0 (neighbor = 1):
      //   value = -(1.0 - 0.7071)^2 * 0.5 * (100+100) / 1000
      //         = -(0.2929)^2 * 0.1 = -0.085786 * 0.1 = -0.0085786
      //   dir = (0-1, 0-1, 0)*0.5 = (-0.5, -0.5, 0)
      //   contrib = -0.0085786 * (-0.5,-0.5,0) / 0.7071
      //           = (0.006066, 0.006066, 0)
      //   result = (0.006066, 0.006066, 0) * (-1) / 1000
      //          = (-6.066e-6, -6.066e-6, 0, 0)
      {
        const uint32_t N = 2;
        const float r_ij = std::sqrt(2.0f) * simScale; // ≈ 0.7071
        auto nm = makeEmptyNeighborMap(N);
        nm[0 * 32 + 0] = {1.0f, r_ij};
        nm[1 * 32 + 0] = {0.0f, r_ij};

        // Precise expected: (hScaled - r)^2 * 0.5 * 200 / 1000 * dir / r
        // (1 - sqrt(2)/2)^2 = (2 - sqrt(2))^2 / 4 = (4 - 4*sqrt(2) + 2) / 4
        //                   = (6 - 4*sqrt(2)) / 4
        // value = -(6-4*sqrt(2))/4 * 0.1
        // dir_component / r = -0.5 / (sqrt(2)/2) = -1/sqrt(2)
        // contrib_x = value * (-1/sqrt(2))
        // expected_x = contrib_x * gradCoeff / 1000
        const float hMinusR = hScaled - r_ij;
        const float value = -hMinusR * hMinusR * 0.5f * 200.0f / 1000.0f;
        const float dirX = -0.5f;
        const float contribX = value * dirX / r_ij;
        const float expectedX = contribX * gradCoeff / 1000.0f;

        cases.push_back({
            .name = "DiagonalDisplacement",
            .neighborMap = nm,
            .pressure = {100.0f, 100.0f},
            .rho = {0.0f, 0.0f, 1000.0f, 1000.0f},
            .sortedPosition = {{0.0f, 0.0f, 0.0f, 0.0f},
                               {1.0f, 1.0f, 0.0f, 0.0f}},
            .sortedParticleIdBySerialId = {0, 1},
            .delta = delta,
            .massMultGradWspikyCoefficient = gradCoeff,
            .h = h,
            .simulationScale = simScale,
            .restDensity = rho0,
            .originalPosition = {{0.0f, 0.0f, 0.0f, 2.1f},
                                 {1.0f, 1.0f, 0.0f, 2.1f}},
            .sortedCellAndSerialId = {{0, 0}, {0, 1}},
            .expectedPressureAcceleration = {{expectedX, expectedX, 0.0f, 0.0f},
                                             {-expectedX, -expectedX, 0.0f,
                                              0.0f}},
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
