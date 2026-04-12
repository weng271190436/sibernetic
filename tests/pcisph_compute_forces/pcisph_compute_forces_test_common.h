#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/kernels/PcisphComputeForcesKernel.h"
#include "../../src/types/HostTypes.h"
#include "../utils/common/backend_param_test.h"
#include "../utils/neighbormap/neighbor_map_helpers.h"

namespace SiberneticTest {

struct PcisphComputeForcesResult {
  std::vector<float> pressure;
  std::vector<Sibernetic::HostFloat4> acceleration; // size: particleCount * 2
};

struct PcisphComputeForcesCase {
  using InputType = Sibernetic::PcisphComputeForcesInput;
  using ResultType = PcisphComputeForcesResult;

  const char *name;

  // Input data
  std::vector<Sibernetic::HostFloat2> neighborMap;
  std::vector<float> rho;
  std::vector<Sibernetic::HostFloat4> sortedPosition;
  std::vector<Sibernetic::HostFloat4> sortedVelocity;
  std::vector<uint32_t> sortedParticleIdBySerialId;
  std::vector<Sibernetic::HostFloat4> position;
  std::vector<Sibernetic::HostUInt2> particleIndex;
  float surfTensCoeff;
  float mass_mult_divgradWviscosityCoeff;
  float hScaled;
  float mu;
  float gravity_x;
  float gravity_y;
  float gravity_z;
  float mass;

  // Expected output
  std::vector<float> expectedPressure;
  std::vector<Sibernetic::HostFloat4> expectedAcceleration;

  InputType toInput() const {
    return {
        .neighborMap = neighborMap,
        .rho = rho,
        .sortedPosition = sortedPosition,
        .sortedVelocity = sortedVelocity,
        .sortedParticleIdBySerialId = sortedParticleIdBySerialId,
        .position = position,
        .particleIndex = particleIndex,
        .surfTensCoeff = surfTensCoeff,
        .mass_mult_divgradWviscosityCoeff = mass_mult_divgradWviscosityCoeff,
        .hScaled = hScaled,
        .mu = mu,
        .gravity_x = gravity_x,
        .gravity_y = gravity_y,
        .gravity_z = gravity_z,
        .mass = mass,
        .particleCount = static_cast<uint32_t>(rho.size()),
    };
  }

  void verify(const ResultType &result) const {
    // Check pressure initialized to 0
    ASSERT_EQ(result.pressure.size(), expectedPressure.size());
    for (size_t i = 0; i < expectedPressure.size(); ++i) {
      EXPECT_NEAR(result.pressure[i], expectedPressure[i], 1e-6f)
          << "pressure mismatch at " << i;
    }

    // Check acceleration
    ASSERT_EQ(result.acceleration.size(), expectedAcceleration.size());
    for (size_t i = 0; i < expectedAcceleration.size(); ++i) {
      for (int c = 0; c < 4; ++c) {
        EXPECT_NEAR(result.acceleration[i][c], expectedAcceleration[i][c],
                    1e-5f)
            << "acceleration mismatch at [" << i << "][" << c << "]";
      }
    }
  }
};

static_assert(SiberneticTest::KernelTestCase<PcisphComputeForcesCase>);

class PcisphComputeForcesRunner {
public:
  virtual ~PcisphComputeForcesRunner() = default;
  virtual PcisphComputeForcesResult run(const PcisphComputeForcesCase &tc) = 0;
};

struct PcisphComputeForcesTestCommon {
  using Case = PcisphComputeForcesCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = [] {
      std::vector<Case> cases;

      // Test 1: Boundary particle (should output zeros)
      {
        constexpr uint32_t N = 1;
        auto neighborMap = makeNeighborMap(N);

        cases.push_back({
            .name = "BoundaryParticleOutputsZero",
            .neighborMap = neighborMap,
            .rho = {1000.0f},
            .sortedPosition = {{0, 0, 0, 0}},
            .sortedVelocity = {{1, 2, 3, 0}},
            .sortedParticleIdBySerialId = {0},
            .position = {{0, 0, 0, 3}}, // .w = 3 = BOUNDARY
            .particleIndex = {{0, 0}},
            .surfTensCoeff = 0.1f,
            .mass_mult_divgradWviscosityCoeff = 1.0f,
            .hScaled = 0.1f,
            .mu = 0.01f,
            .gravity_x = 0.0f,
            .gravity_y = -9.8f,
            .gravity_z = 0.0f,
            .mass = 1.0f,
            .expectedPressure = {0.0f},
            .expectedAcceleration = {{0, 0, 0, 0}, {0, 0, 0, 0}},
        });
      }

      // Test 2: Single liquid particle, no neighbors, just gravity
      {
        constexpr uint32_t N = 1;
        auto neighborMap = makeNeighborMap(N);

        cases.push_back({
            .name = "SingleParticleGravityOnly",
            .neighborMap = neighborMap,
            .rho = {1000.0f},
            .sortedPosition = {{0, 0, 0, 0}},
            .sortedVelocity = {{0, 0, 0, 0}},
            .sortedParticleIdBySerialId = {0},
            .position = {{0, 0, 0, 1}}, // .w = 1 = LIQUID
            .particleIndex = {{0, 0}},
            .surfTensCoeff = 0.1f,
            .mass_mult_divgradWviscosityCoeff = 1.0f,
            .hScaled = 0.1f,
            .mu = 0.01f,
            .gravity_x = 0.0f,
            .gravity_y = -9.8f,
            .gravity_z = 0.0f,
            .mass = 1.0f,
            .expectedPressure = {0.0f},
            .expectedAcceleration = {{0, -9.8f, 0, 0}, {0, 0, 0, 0}},
        });
      }

      // Test 3: Non-identity serial->sorted map with no neighbors
      {
        constexpr uint32_t N = 2;
        auto neighborMap = makeNeighborMap(N);

        cases.push_back({
            .name = "NoNeighborsNonIdentityMapGravityOnly",
            .neighborMap = neighborMap,
            .rho = {1000.0f, 1000.0f},
            .sortedPosition = {{0, 0, 0, 0}, {1, 0, 0, 0}},
            .sortedVelocity = {{0, 0, 0, 0}, {0, 0, 0, 0}},
            .sortedParticleIdBySerialId = {1, 0},
            // sortedCellAndSerialId: sorted[0] -> serial 1, sorted[1] -> serial
            // 0
            .position = {{0, 0, 0, 1}, {1, 0, 0, 1}},
            .particleIndex = {{0, 1}, {0, 0}},
            .surfTensCoeff = 0.1f,
            .mass_mult_divgradWviscosityCoeff = 1.0f,
            .hScaled = 0.1f,
            .mu = 0.01f,
            .gravity_x = 0.0f,
            .gravity_y = -9.8f,
            .gravity_z = 0.0f,
            .mass = 1.0f,
            .expectedPressure = {0.0f, 0.0f},
            .expectedAcceleration =
                {
                    {0, -9.8f, 0, 0},
                    {0, -9.8f, 0, 0},
                    {0, 0, 0, 0},
                    {0, 0, 0, 0},
                },
        });
      }

      // Test 4: Boundary/liquid behavior must follow serial ids from
      // sortedCellAndSerialId
      {
        constexpr uint32_t N = 2;
        auto neighborMap = makeNeighborMap(N);

        cases.push_back({
            .name = "BoundaryClassificationUsesSerialIdFromSortedTuple",
            .neighborMap = neighborMap,
            .rho = {1000.0f, 1000.0f},
            .sortedPosition = {{0, 0, 0, 0}, {1, 0, 0, 0}},
            .sortedVelocity = {{0, 0, 0, 0}, {0, 0, 0, 0}},
            .sortedParticleIdBySerialId = {1, 0},
            // sorted[0] -> serial 1 (liquid), sorted[1] -> serial 0 (boundary)
            .position = {{0, 0, 0, 3}, {1, 0, 0, 1}},
            .particleIndex = {{0, 1}, {0, 0}},
            .surfTensCoeff = 0.1f,
            .mass_mult_divgradWviscosityCoeff = 1.0f,
            .hScaled = 0.1f,
            .mu = 0.01f,
            .gravity_x = 0.0f,
            .gravity_y = -9.8f,
            .gravity_z = 0.0f,
            .mass = 1.0f,
            .expectedPressure = {0.0f, 0.0f},
            .expectedAcceleration =
                {
                    {0, -9.8f, 0, 0},
                    {0, 0, 0, 0},
                    {0, 0, 0, 0},
                    {0, 0, 0, 0},
                },
        });
      }

      // Test 5: Two-particle symmetric viscosity interaction (no
      // gravity/surface tension)
      {
        constexpr uint32_t N = 2;
        auto neighborMap = makeNeighborMap(N);
        setNeighbor(neighborMap, 0, 0, 1, 0.5f);
        setNeighbor(neighborMap, 1, 0, 0, 0.5f);

        cases.push_back({
            .name = "TwoParticleViscositySymmetry",
            .neighborMap = neighborMap,
            .rho = {1.0f, 1.0f},
            .sortedPosition = {{0, 0, 0, 0}, {1, 0, 0, 0}},
            .sortedVelocity = {{0, 0, 0, 0}, {2, 0, 0, 0}},
            .sortedParticleIdBySerialId = {0, 1},
            .position = {{0, 0, 0, 1}, {1, 0, 0, 1}},
            .particleIndex = {{0, 0}, {0, 1}},
            .surfTensCoeff = 0.0f,
            .mass_mult_divgradWviscosityCoeff = 100000.0f,
            .hScaled = 1.0f,
            .mu = 0.01f,
            .gravity_x = 0.0f,
            .gravity_y = 0.0f,
            .gravity_z = 0.0f,
            .mass = 1.0f,
            .expectedPressure = {0.0f, 0.0f},
            .expectedAcceleration =
                {
                    {0.015f, 0, 0, 0},
                    {-0.015f, 0, 0, 0},
                    {0, 0, 0, 0},
                    {0, 0, 0, 0},
                },
        });
      }

      // Test 6: Neighbor at cutoff distance is excluded (r == hScaled)
      {
        constexpr uint32_t N = 2;
        auto neighborMap = makeNeighborMap(N);
        setNeighbor(neighborMap, 0, 0, 1, 1.0f);
        setNeighbor(neighborMap, 1, 0, 0, 1.0f);

        cases.push_back({
            .name = "NeighborAtCutoffIsIgnored",
            .neighborMap = neighborMap,
            .rho = {1000.0f, 1000.0f},
            .sortedPosition = {{0, 0, 0, 0}, {1, 0, 0, 0}},
            .sortedVelocity = {{0, 0, 0, 0}, {3, 0, 0, 0}},
            .sortedParticleIdBySerialId = {0, 1},
            .position = {{0, 0, 0, 1}, {1, 0, 0, 1}},
            .particleIndex = {{0, 0}, {0, 1}},
            .surfTensCoeff = 0.0f,
            .mass_mult_divgradWviscosityCoeff = 100000.0f,
            .hScaled = 1.0f,
            .mu = 0.01f,
            .gravity_x = 0.0f,
            .gravity_y = 0.0f,
            .gravity_z = 0.0f,
            .mass = 1.0f,
            .expectedPressure = {0.0f, 0.0f},
            .expectedAcceleration =
                {
                    {0, 0, 0, 0},
                    {0, 0, 0, 0},
                    {0, 0, 0, 0},
                    {0, 0, 0, 0},
                },
        });
      }

      // Test 7: Worm<->Agar pair uses low viscosity branch (1e-5)
      {
        constexpr uint32_t N = 2;
        auto neighborMap = makeNeighborMap(N);
        setNeighbor(neighborMap, 0, 0, 1, 0.5f);
        setNeighbor(neighborMap, 1, 0, 0, 0.5f);

        cases.push_back({
            .name = "WormAgarUsesLowViscosityBranch",
            .neighborMap = neighborMap,
            .rho = {1.0f, 1.0f},
            .sortedPosition = {{0, 0, 0, 0}, {1, 0, 0, 0}},
            .sortedVelocity = {{0, 0, 0, 0}, {2, 0, 0, 0}},
            .sortedParticleIdBySerialId = {0, 1},
            // 2.1 = worm range, 2.3 = agar range
            .position = {{0, 0, 0, 2.1f}, {1, 0, 0, 2.3f}},
            .particleIndex = {{0, 0}, {0, 1}},
            .surfTensCoeff = 0.0f,
            .mass_mult_divgradWviscosityCoeff = 100000.0f,
            .hScaled = 1.0f,
            .mu = 0.01f,
            .gravity_x = 0.0f,
            .gravity_y = 0.0f,
            .gravity_z = 0.0f,
            .mass = 1.0f,
            .expectedPressure = {0.0f, 0.0f},
            .expectedAcceleration =
                {
                    {0.0015f, 0, 0, 0},
                    {-0.0015f, 0, 0, 0},
                    {0, 0, 0, 0},
                    {0, 0, 0, 0},
                },
        });
      }

      // Test 8: Surface tension contributes along -(p_i - p_j) and sets w=0
      {
        constexpr uint32_t N = 2;
        auto neighborMap = makeNeighborMap(N);
        setNeighbor(neighborMap, 0, 0, 1, 0.5f);
        setNeighbor(neighborMap, 1, 0, 0, 0.5f);

        cases.push_back({
            .name = "SurfaceTensionOnlySymmetry",
            .neighborMap = neighborMap,
            .rho = {1.0f, 1.0f},
            .sortedPosition = {{0, 0, 0, 0}, {1, 0, 0, 0}},
            .sortedVelocity = {{0, 0, 0, 0}, {0, 0, 0, 0}},
            .sortedParticleIdBySerialId = {0, 1},
            .position = {{0, 0, 0, 1}, {1, 0, 0, 1}},
            .particleIndex = {{0, 0}, {0, 1}},
            .surfTensCoeff = 1.0f,
            .mass_mult_divgradWviscosityCoeff = 0.0f,
            .hScaled = 1.0f,
            .mu = 0.01f,
            .gravity_x = 0.0f,
            .gravity_y = 0.0f,
            .gravity_z = 0.0f,
            .mass = 1.0f,
            .expectedPressure = {0.0f, 0.0f},
            .expectedAcceleration =
                {
                    {2.86875e-10f, 0, 0, 0},
                    {-2.86875e-10f, 0, 0, 0},
                    {0, 0, 0, 0},
                    {0, 0, 0, 0},
                },
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

static_assert(SiberneticTest::SibTestCommon<PcisphComputeForcesTestCommon>);

} // namespace SiberneticTest
