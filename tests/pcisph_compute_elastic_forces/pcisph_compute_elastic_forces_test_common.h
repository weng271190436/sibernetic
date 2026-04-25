#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/kernels/PcisphComputeElasticForcesKernel.h"
#include "../utils/common/backend_param_test.h"
#include "../utils/types/types.h"

namespace SiberneticTest {

struct PcisphComputeElasticForcesResult {
  std::vector<HostFloat4> acceleration;
};

struct PcisphComputeElasticForcesCase {
  using InputType = Sibernetic::PcisphComputeElasticForcesInput;
  using ResultType = PcisphComputeElasticForcesResult;

  const char *name;

  // Per-particle arrays (size: particleCount).
  std::vector<HostFloat4> sortedPosition;
  std::vector<HostFloat4> acceleration; // initial values (in/out)
  std::vector<uint32_t> sortedParticleIdBySerialId;
  std::vector<HostUInt2> sortedCellAndSerialId;
  std::vector<HostFloat4> originalPosition; // .w = particle type

  // Elastic connection data (size: numOfElasticP * kMaxNeighborCount).
  std::vector<HostFloat4> elasticConnectionsData;

  // Muscle activation (size: muscleCount).
  std::vector<float> muscleActivationSignal;

  float maxMuscleForce = 0.0f;
  float simulationScale = 1.0f;
  uint32_t numOfElasticP = 0;
  uint32_t muscleCount = 0;
  float elasticityCoefficient = 1.0f;

  // Expected acceleration for each particle after kernel execution.
  std::vector<HostFloat4> expectedAcceleration;
  float tolerance = 1e-4f;

  InputType toInput() const {
    return {
        .sortedPosition = sortedPosition,
        .acceleration = acceleration,
        .sortedParticleIdBySerialId = sortedParticleIdBySerialId,
        .sortedCellAndSerialId = sortedCellAndSerialId,
        .maxMuscleForce = maxMuscleForce,
        .simulationScale = simulationScale,
        .numOfElasticP = numOfElasticP,
        .elasticConnectionsData = elasticConnectionsData,
        .muscleCount = muscleCount,
        .muscleActivationSignal = muscleActivationSignal,
        .originalPosition = originalPosition,
        .elasticityCoefficient = elasticityCoefficient,
    };
  }

  void verify(const ResultType &result) const {
    ASSERT_EQ(result.acceleration.size(), acceleration.size());
    for (size_t i = 0; i < expectedAcceleration.size(); ++i) {
      for (int c = 0; c < 4; ++c) {
        EXPECT_NEAR(result.acceleration[i][c], expectedAcceleration[i][c],
                    tolerance)
            << "acceleration[" << i << "][" << c << "]";
      }
    }
  }
};

static_assert(SiberneticTest::KernelTestCase<PcisphComputeElasticForcesCase>);

class PcisphComputeElasticForcesRunner
    : public TestRunner<PcisphComputeElasticForcesCase,
                        PcisphComputeElasticForcesResult> {};

// Helper: build a float4 from (x,y,z,w).
inline HostFloat4 f4(float x, float y, float z, float w) {
  return {x, y, z, w};
}

// Helper: build an identity sort mapping (serial == sorted).
inline void identitySort(uint32_t N, std::vector<uint32_t> &idBySerial,
                         std::vector<HostUInt2> &cellAndSerial) {
  idBySerial.resize(N);
  cellAndSerial.resize(N);
  for (uint32_t i = 0; i < N; ++i) {
    idBySerial[i] = i;
    cellAndSerial[i] = {0, i}; // cell=0, serialId=i
  }
}

// Helper: fill elasticConnectionsData with kNoParticleId sentinels.
inline std::vector<HostFloat4>
emptyConnections(uint32_t numElastic,
                 int maxNeighbors = Sibernetic::kMaxNeighborCount) {
  const size_t total =
      static_cast<size_t>(numElastic) * static_cast<size_t>(maxNeighbors);
  std::vector<HostFloat4> data(total);
  for (auto &e : data) {
    e = {static_cast<float>(Sibernetic::kNoParticleId), 0.0f, 0.0f, 0.0f};
  }
  return data;
}

// Helper: set one connection entry.
inline void setConnection(std::vector<HostFloat4> &data, uint32_t elasticIndex,
                          int slot, int connectedSerialId,
                          float equilibriumDist, int muscleId = 0) {
  const int idx =
      static_cast<int>(elasticIndex) * Sibernetic::kMaxNeighborCount + slot;
  data[static_cast<size_t>(idx)] = {static_cast<float>(connectedSerialId),
                                    equilibriumDist,
                                    static_cast<float>(muscleId), 0.0f};
}

struct PcisphComputeElasticForcesTestCommon {
  using Case = PcisphComputeElasticForcesCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = {

        // 1. No elastic particles → acceleration unchanged.
        [] {
          PcisphComputeElasticForcesCase tc{};
          tc.name = "NoElasticParticles";
          tc.numOfElasticP = 0;
          // Still need 1 particle for valid buffers.
          tc.sortedPosition = {f4(0, 0, 0, 0)};
          tc.acceleration = {f4(1, 2, 3, 0)};
          tc.originalPosition = {f4(0, 0, 0, 2.1f)};
          identitySort(1, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);
          tc.elasticConnectionsData = emptyConnections(0);
          tc.expectedAcceleration = {f4(1, 2, 3, 0)};
          return tc;
        }(),

        // 2. Single elastic particle with no connections → unchanged.
        [] {
          PcisphComputeElasticForcesCase tc{};
          tc.name = "SingleParticleNoConnections";
          tc.numOfElasticP = 1;
          tc.sortedPosition = {f4(1, 0, 0, 0)};
          tc.acceleration = {f4(0.5f, 0.5f, 0.5f, 0)};
          tc.originalPosition = {f4(1, 0, 0, 2.1f)};
          identitySort(1, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);
          tc.elasticConnectionsData = emptyConnections(1);
          tc.expectedAcceleration = {f4(0.5f, 0.5f, 0.5f, 0)};
          return tc;
        }(),

        // 3. Two worm-body particles, spring stretched beyond equilibrium.
        // Particle 0 at (0,0,0), particle 1 at (3,0,0).
        // Distance = 3 * simulationScale = 3.
        // Equilibrium = 2. delta = 3 - 2 = 1.
        // direction from 0→1 subtraction: (0-3)*simScale = (-3,0,0),
        // normalized = (-1,0,0).
        // F = -(-1,0,0) * 1.0 * 1.0 = (1,0,0) → attractive toward particle 1.
        // But kernel does: accel[0] -= direction * delta * k
        //   = accel[0] -= (-1,0,0) * 1 * 1 = accel[0] += (1,0,0)
        [] {
          PcisphComputeElasticForcesCase tc{};
          tc.name = "TwoConnectedParticles_Stretched";
          tc.numOfElasticP = 2;
          tc.simulationScale = 1.0f;
          tc.elasticityCoefficient = 1.0f;
          tc.sortedPosition = {f4(0, 0, 0, 0), f4(3, 0, 0, 0)};
          tc.acceleration = {f4(0, 0, 0, 0), f4(0, 0, 0, 0)};
          tc.originalPosition = {f4(0, 0, 0, 2.1f), f4(3, 0, 0, 2.1f)};
          identitySort(2, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);

          tc.elasticConnectionsData = emptyConnections(2);
          setConnection(tc.elasticConnectionsData, 0, 0, 1, 2.0f);
          setConnection(tc.elasticConnectionsData, 1, 0, 0, 2.0f);

          // Particle 0: direction = (0-3)/3 = (-1,0,0), delta=1
          //   accel -= (-1,0,0)*1*1 → accel += (1,0,0)
          // Particle 1: direction = (3-0)/3 = (1,0,0), delta=1
          //   accel -= (1,0,0)*1*1 → accel -= (1,0,0)
          tc.expectedAcceleration = {f4(1, 0, 0, 0), f4(-1, 0, 0, 0)};
          return tc;
        }(),

        // 4. Two worm-body particles, spring compressed.
        // Particle 0 at (0,0,0), particle 1 at (1,0,0).
        // Distance = 1. Equilibrium = 2. delta = 1 - 2 = -1.
        // direction from 0 perspective: (0-1)/1 = (-1,0,0).
        // accel[0] -= (-1,0,0) * (-1) * 1 = accel[0] -= (1,0,0)
        //   → repulsive, pushes apart.
        [] {
          PcisphComputeElasticForcesCase tc{};
          tc.name = "TwoConnectedParticles_Compressed";
          tc.numOfElasticP = 2;
          tc.simulationScale = 1.0f;
          tc.elasticityCoefficient = 1.0f;
          tc.sortedPosition = {f4(0, 0, 0, 0), f4(1, 0, 0, 0)};
          tc.acceleration = {f4(0, 0, 0, 0), f4(0, 0, 0, 0)};
          tc.originalPosition = {f4(0, 0, 0, 2.1f), f4(1, 0, 0, 2.1f)};
          identitySort(2, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);

          tc.elasticConnectionsData = emptyConnections(2);
          setConnection(tc.elasticConnectionsData, 0, 0, 1, 2.0f);
          setConnection(tc.elasticConnectionsData, 1, 0, 0, 2.0f);

          // Particle 0: dir=(-1,0,0), delta=-1 → -= (-1)*(-1)*1 = -= (1,0,0)
          // Particle 1: dir=(1,0,0), delta=-1 → -= (1)*(-1)*1 = += (1,0,0)
          tc.expectedAcceleration = {f4(-1, 0, 0, 0), f4(1, 0, 0, 0)};
          return tc;
        }(),

        // 5. WormBodyVsAgar: Particle 0 is worm (type 2.1), particle 1 is
        // agar (type 3.1). Elasticity should be 0.25x.
        [] {
          PcisphComputeElasticForcesCase tc{};
          tc.name = "WormBodyVsAgar";
          tc.numOfElasticP = 1;
          tc.simulationScale = 1.0f;
          tc.elasticityCoefficient = 4.0f; // k=4, agar → 4*0.25=1.0
          tc.sortedPosition = {f4(0, 0, 0, 0), f4(3, 0, 0, 0)};
          tc.acceleration = {f4(0, 0, 0, 0), f4(0, 0, 0, 0)};
          // Particle 0: worm (2.1), Particle 1: agar/boundary (3.1)
          tc.originalPosition = {f4(0, 0, 0, 2.1f), f4(3, 0, 0, 3.1f)};
          identitySort(2, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);

          tc.elasticConnectionsData = emptyConnections(1);
          setConnection(tc.elasticConnectionsData, 0, 0, 1, 2.0f);

          // dir=(-1,0,0), delta=1, k=4*0.25=1.0
          // accel[0] -= (-1,0,0)*1*1 → += (1,0,0)
          tc.expectedAcceleration = {f4(1, 0, 0, 0), f4(0, 0, 0, 0)};
          return tc;
        }(),

        // 6. MuscleActivation: muscle is activated, adds contraction force.
        [] {
          PcisphComputeElasticForcesCase tc{};
          tc.name = "MuscleActivation";
          tc.numOfElasticP = 1;
          tc.simulationScale = 1.0f;
          tc.elasticityCoefficient = 0.0f; // zero elasticity to isolate muscle
          tc.maxMuscleForce = 10.0f;
          tc.muscleCount = 1;
          tc.muscleActivationSignal = {0.5f}; // 50% activation
          tc.sortedPosition = {f4(0, 0, 0, 0), f4(4, 0, 0, 0)};
          tc.acceleration = {f4(0, 0, 0, 0), f4(0, 0, 0, 0)};
          tc.originalPosition = {f4(0, 0, 0, 2.1f), f4(4, 0, 0, 2.1f)};
          identitySort(2, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);

          tc.elasticConnectionsData = emptyConnections(1);
          // muscleId=1 (1-indexed)
          setConnection(tc.elasticConnectionsData, 0, 0, 1, 4.0f, 1);

          // dir = (0-4)/4 = (-1,0,0), delta = 4-4 = 0 → no spring force
          // muscle: accel -= (-1,0,0) * 0.5 * 10 = accel += (5,0,0)
          tc.expectedAcceleration = {f4(5, 0, 0, 0), f4(0, 0, 0, 0)};
          return tc;
        }(),

        // 7. MuscleNotActivated: muscle activation = 0, no extra force.
        [] {
          PcisphComputeElasticForcesCase tc{};
          tc.name = "MuscleNotActivated";
          tc.numOfElasticP = 1;
          tc.simulationScale = 1.0f;
          tc.elasticityCoefficient = 1.0f;
          tc.maxMuscleForce = 10.0f;
          tc.muscleCount = 1;
          tc.muscleActivationSignal = {0.0f}; // no activation
          tc.sortedPosition = {f4(0, 0, 0, 0), f4(3, 0, 0, 0)};
          tc.acceleration = {f4(0, 0, 0, 0), f4(0, 0, 0, 0)};
          tc.originalPosition = {f4(0, 0, 0, 2.1f), f4(3, 0, 0, 2.1f)};
          identitySort(2, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);

          tc.elasticConnectionsData = emptyConnections(1);
          setConnection(tc.elasticConnectionsData, 0, 0, 1, 2.0f, 1);

          // Spring only: dir=(-1,0,0), delta=1, k=1
          // accel -= (-1,0,0)*1*1 → += (1,0,0)
          // Muscle: activation=0 → no extra force.
          tc.expectedAcceleration = {f4(1, 0, 0, 0), f4(0, 0, 0, 0)};
          return tc;
        }(),

        // 8. SimulationScale != 1: positions are in grid space, distances
        // are scaled by simulationScale before comparing to equilibrium.
        // Particle 0 at (0,0,0), particle 1 at (6,0,0), simScale=0.5.
        // Scaled distance = 6*0.5 = 3. Equilibrium = 2. delta = 1.
        // Scaled displacement = (0-6)*0.5 = (-3,0,0), dir = (-1,0,0).
        // accel -= (-1,0,0)*1*1 → += (1,0,0)
        [] {
          PcisphComputeElasticForcesCase tc{};
          tc.name = "SimulationScaleNonUnit";
          tc.numOfElasticP = 2;
          tc.simulationScale = 0.5f;
          tc.elasticityCoefficient = 1.0f;
          tc.sortedPosition = {f4(0, 0, 0, 0), f4(6, 0, 0, 0)};
          tc.acceleration = {f4(0, 0, 0, 0), f4(0, 0, 0, 0)};
          tc.originalPosition = {f4(0, 0, 0, 2.1f), f4(6, 0, 0, 2.1f)};
          identitySort(2, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);
          tc.elasticConnectionsData = emptyConnections(2);
          setConnection(tc.elasticConnectionsData, 0, 0, 1, 2.0f);
          setConnection(tc.elasticConnectionsData, 1, 0, 0, 2.0f);
          tc.expectedAcceleration = {f4(1, 0, 0, 0), f4(-1, 0, 0, 0)};
          return tc;
        }(),

        // 9. Non-identity sort: serial order ≠ sorted order.
        // Serial 0 → sorted 1, serial 1 → sorted 0.
        // Sorted positions: [0]=(5,0,0), [1]=(0,0,0).
        // Elastic particle 0 (serial 0, sorted 1) at (0,0,0).
        // Connected to serial 1 (sorted 0) at (5,0,0).
        // dir = (0-5)/5 = (-1,0,0), delta = 5-3 = 2, k=1.
        // accel[sorted=1] -= (-1,0,0)*2*1 → += (2,0,0)
        [] {
          PcisphComputeElasticForcesCase tc{};
          tc.name = "NonIdentitySort";
          tc.numOfElasticP = 1;
          tc.simulationScale = 1.0f;
          tc.elasticityCoefficient = 1.0f;
          // Sorted order: sorted[0]=particle at (5,0,0), sorted[1]=(0,0,0)
          tc.sortedPosition = {f4(5, 0, 0, 0), f4(0, 0, 0, 0)};
          tc.acceleration = {f4(0, 0, 0, 0), f4(0, 0, 0, 0)};
          tc.originalPosition = {f4(0, 0, 0, 2.1f), f4(5, 0, 0, 2.1f)};
          // serial 0 → sorted 1, serial 1 → sorted 0
          tc.sortedParticleIdBySerialId = {1, 0};
          tc.sortedCellAndSerialId = {{0, 1}, {0, 0}};
          tc.elasticConnectionsData = emptyConnections(1);
          setConnection(tc.elasticConnectionsData, 0, 0, 1, 3.0f);
          // Result lands in sorted[1] (serial 0).
          tc.expectedAcceleration = {f4(0, 0, 0, 0), f4(2, 0, 0, 0)};
          return tc;
        }(),

        // 10. Multiple connections per particle: accumulate forces from 2
        // neighbors.
        // Particle 0 at origin, particle 1 at (2,0,0), particle 2 at (0,3,0).
        // Connection 0→1: dist=2, eq=1, delta=1, dir=(-1,0,0)
        //   accel -= (-1,0,0)*1*1 → += (1,0,0)
        // Connection 0→2: dist=3, eq=1, delta=2, dir=(0,-1,0)
        //   accel -= (0,-1,0)*2*1 → += (0,2,0)
        // Total: (1,2,0,0)
        [] {
          PcisphComputeElasticForcesCase tc{};
          tc.name = "MultipleConnectionsAccumulate";
          tc.numOfElasticP = 1;
          tc.simulationScale = 1.0f;
          tc.elasticityCoefficient = 1.0f;
          tc.sortedPosition = {f4(0, 0, 0, 0), f4(2, 0, 0, 0), f4(0, 3, 0, 0)};
          tc.acceleration = {f4(0, 0, 0, 0), f4(0, 0, 0, 0), f4(0, 0, 0, 0)};
          tc.originalPosition = {f4(0, 0, 0, 2.1f), f4(2, 0, 0, 2.1f),
                                 f4(0, 3, 0, 2.1f)};
          identitySort(3, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);
          tc.elasticConnectionsData = emptyConnections(1);
          setConnection(tc.elasticConnectionsData, 0, 0, 1, 1.0f);
          setConnection(tc.elasticConnectionsData, 0, 1, 2, 1.0f);
          tc.expectedAcceleration = {f4(1, 2, 0, 0), f4(0, 0, 0, 0),
                                     f4(0, 0, 0, 0)};
          return tc;
        }(),

        // 11. 3D displacement: positions differ along all three axes.
        // Particle 0 at origin, particle 1 at (1,2,2).
        // dist = sqrt(1+4+4) = 3. eq = 1. delta = 2.
        // dir = (0-1, 0-2, 0-2)/3 = (-1/3, -2/3, -2/3).
        // accel -= dir*2*1 → += (2/3, 4/3, 4/3)
        [] {
          PcisphComputeElasticForcesCase tc{};
          tc.name = "Displacement3D";
          tc.numOfElasticP = 1;
          tc.simulationScale = 1.0f;
          tc.elasticityCoefficient = 1.0f;
          tc.sortedPosition = {f4(0, 0, 0, 0), f4(1, 2, 2, 0)};
          tc.acceleration = {f4(0, 0, 0, 0), f4(0, 0, 0, 0)};
          tc.originalPosition = {f4(0, 0, 0, 2.1f), f4(1, 2, 2, 2.1f)};
          identitySort(2, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);
          tc.elasticConnectionsData = emptyConnections(1);
          setConnection(tc.elasticConnectionsData, 0, 0, 1, 1.0f);
          tc.expectedAcceleration = {
              f4(2.0f / 3.0f, 4.0f / 3.0f, 4.0f / 3.0f, 0), f4(0, 0, 0, 0)};
          return tc;
        }(),

        // 12. Pre-existing acceleration: elastic force adds to nonzero
        // initial value.
        // Same geometry as test 3 (stretched, delta=1, dir=(-1,0,0))
        // but initial accel = (10, 20, 30, 0).
        // Result = (10+1, 20, 30, 0).
        [] {
          PcisphComputeElasticForcesCase tc{};
          tc.name = "PreExistingAcceleration";
          tc.numOfElasticP = 1;
          tc.simulationScale = 1.0f;
          tc.elasticityCoefficient = 1.0f;
          tc.sortedPosition = {f4(0, 0, 0, 0), f4(3, 0, 0, 0)};
          tc.acceleration = {f4(10, 20, 30, 0), f4(0, 0, 0, 0)};
          tc.originalPosition = {f4(0, 0, 0, 2.1f), f4(3, 0, 0, 2.1f)};
          identitySort(2, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);
          tc.elasticConnectionsData = emptyConnections(1);
          setConnection(tc.elasticConnectionsData, 0, 0, 1, 2.0f);
          tc.expectedAcceleration = {f4(11, 20, 30, 0), f4(0, 0, 0, 0)};
          return tc;
        }(),

        // 13. Coincident particles (r=0): should be skipped, no NaN.
        [] {
          PcisphComputeElasticForcesCase tc{};
          tc.name = "CoincidentParticlesSkipped";
          tc.numOfElasticP = 1;
          tc.simulationScale = 1.0f;
          tc.elasticityCoefficient = 1.0f;
          tc.sortedPosition = {f4(5, 5, 5, 0), f4(5, 5, 5, 0)};
          tc.acceleration = {f4(1, 2, 3, 0), f4(0, 0, 0, 0)};
          tc.originalPosition = {f4(5, 5, 5, 2.1f), f4(5, 5, 5, 2.1f)};
          identitySort(2, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);
          tc.elasticConnectionsData = emptyConnections(1);
          setConnection(tc.elasticConnectionsData, 0, 0, 1, 1.0f);
          // r=0 → skipped, accel unchanged.
          tc.expectedAcceleration = {f4(1, 2, 3, 0), f4(0, 0, 0, 0)};
          return tc;
        }(),

    };
    return kCases;
  }

  static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
    return info.param.name;
  }
};

static_assert(
    SiberneticTest::SibTestCommon<PcisphComputeElasticForcesTestCommon>);

} // namespace SiberneticTest
