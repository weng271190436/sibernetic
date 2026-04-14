#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/kernels/PcisphPredictPositionsKernel.h"
#include "../../src/types/HostTypes.h"
#include "../utils/common/backend_param_test.h"
#include "../utils/neighbormap/neighbor_map_helpers.h"

namespace SiberneticTest {

struct PcisphPredictPositionsResult {
  std::vector<Sibernetic::HostFloat4>
      predictedPosition; // sortedPosition[N..2N)
};

struct PcisphPredictPositionsCase {
  using InputType = Sibernetic::PcisphPredictPositionsInput;
  using ResultType = PcisphPredictPositionsResult;

  const char *name;

  // Input data
  std::vector<Sibernetic::HostFloat4> acceleration; // size: 3*N
  std::vector<Sibernetic::HostFloat4>
      sortedPosition; // size: 2*N (second half is output)
  std::vector<Sibernetic::HostFloat4> sortedVelocity;       // size: N
  std::vector<Sibernetic::HostUInt2> sortedCellAndSerialId; // size: N
  std::vector<uint32_t> sortedParticleIdBySerialId;         // size: N
  std::vector<Sibernetic::HostFloat4> position;    // size: N (original)
  std::vector<Sibernetic::HostFloat4> velocity;    // size: N (boundary normals)
  std::vector<Sibernetic::HostFloat2> neighborMap; // size: N * 32
  float gravitationalAccelerationX;
  float gravitationalAccelerationY;
  float gravitationalAccelerationZ;
  float simulationScaleInv;
  float timeStep;
  float r0;

  // Expected output
  std::vector<Sibernetic::HostFloat4> expectedPredictedPosition;

  InputType toInput() const {
    return {
        .acceleration = acceleration,
        .sortedPosition = sortedPosition,
        .sortedVelocity = sortedVelocity,
        .sortedCellAndSerialId = sortedCellAndSerialId,
        .sortedParticleIdBySerialId = sortedParticleIdBySerialId,
        .position = position,
        .velocity = velocity,
        .neighborMap = neighborMap,
        .gravitationalAccelerationX = gravitationalAccelerationX,
        .gravitationalAccelerationY = gravitationalAccelerationY,
        .gravitationalAccelerationZ = gravitationalAccelerationZ,
        .simulationScaleInv = simulationScaleInv,
        .timeStep = timeStep,
        .r0 = r0,
        .particleCount = static_cast<uint32_t>(sortedVelocity.size()),
    };
  }

  void verify(const ResultType &result) const {
    ASSERT_EQ(result.predictedPosition.size(),
              expectedPredictedPosition.size());
    for (size_t i = 0; i < expectedPredictedPosition.size(); ++i) {
      for (int c = 0; c < 4; ++c) {
        EXPECT_NEAR(result.predictedPosition[i][c],
                    expectedPredictedPosition[i][c], 1e-4f)
            << "predictedPosition mismatch at [" << i << "][" << c << "]";
      }
    }
  }
};

static_assert(SiberneticTest::KernelTestCase<PcisphPredictPositionsCase>);

class PcisphPredictPositionsRunner {
public:
  virtual ~PcisphPredictPositionsRunner() = default;
  virtual PcisphPredictPositionsResult
  run(const PcisphPredictPositionsCase &tc) = 0;
};

struct PcisphPredictPositionsTestCommon {
  using Case = PcisphPredictPositionsCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = [] {
      std::vector<Case> cases;

      // Helper: make an acceleration buffer of size 3*N initialized to zero.
      auto makeAccel = [](uint32_t N) {
        return std::vector<Sibernetic::HostFloat4>(
            static_cast<size_t>(N) * 3, Sibernetic::HostFloat4{0, 0, 0, 0});
      };
      // Helper: make a sortedPosition buffer of size 2*N; second half zeroed.
      auto makeSortedPos = [](const std::vector<Sibernetic::HostFloat4> &pos) {
        std::vector<Sibernetic::HostFloat4> sp = pos;
        sp.resize(pos.size() * 2, Sibernetic::HostFloat4{0, 0, 0, 0});
        return sp;
      };

      // ---- Test 1: BoundaryParticleUnchanged ----
      // Boundary particle (position.w == 3) just copies its current position.
      {
        constexpr uint32_t N = 1;
        auto neighborMap = makeNeighborMap(N);
        auto accel = makeAccel(N);
        std::vector<Sibernetic::HostFloat4> pos = {{5.0f, 6.0f, 7.0f, 0.0f}};
        auto sortedPos = makeSortedPos(pos);

        cases.push_back({
            .name = "BoundaryParticleUnchanged",
            .acceleration = accel,
            .sortedPosition = sortedPos,
            .sortedVelocity = {{1.0f, 2.0f, 3.0f, 0.0f}},
            .sortedCellAndSerialId = {{0, 0}}, // cell 0, serial 0
            .sortedParticleIdBySerialId = {0},
            .position = {{5.0f, 6.0f, 7.0f, 3.0f}}, // .w=3 => BOUNDARY
            .velocity = {{0, 1, 0, 0}},             // normal
            .neighborMap = neighborMap,
            .gravitationalAccelerationX = 0.0f,
            .gravitationalAccelerationY = -9.8f,
            .gravitationalAccelerationZ = 0.0f,
            .simulationScaleInv = 1.0f,
            .timeStep = 0.001f,
            .r0 = 0.1f,
            .expectedPredictedPosition = {{5.0f, 6.0f, 7.0f, 0.0f}},
        });
      }

      // ---- Test 2: SingleParticleEulerIntegration ----
      // No neighbors. Acceleration in all three segments sums to gravity-like.
      // v_new = v + dt * (acc[2N+serial] + acc[id] + acc[N+id])
      //       = (0,0,0) + 0.001 * (0,-9.8,0) = (0,-0.0098,0)
      // x_new = x + dt * simScaleInv * v_new
      //       = (1,2,3) + 0.001 * 1.0 * (0,-0.0098,0) = (1,1.9999902,3)
      {
        constexpr uint32_t N = 1;
        auto neighborMap = makeNeighborMap(N);
        auto accel = makeAccel(N);
        // acc[2*N + 0] = combined from previous step: (0, 0, 0, 0)
        // acc[0] = non-pressure: (0, -9.8, 0, 0)
        // acc[N+0] = pressure: (0, 0, 0, 0)
        accel[0] = {0.0f, -9.8f, 0.0f, 0.0f}; // acc[id=0]

        std::vector<Sibernetic::HostFloat4> pos = {{1.0f, 2.0f, 3.0f, 0.0f}};
        auto sortedPos = makeSortedPos(pos);

        // v_new = (0,0,0) + 0.001 * ((0,-9.8,0) + (0,0,0)) = (0,-0.0098,0)
        // x_new = (1,2,3) + 0.001 * 1.0 * (0,-0.0098,0) = (1, 1.9999902, 3)
        float vy = -9.8f * 0.001f;
        float yNew = 2.0f + 0.001f * 1.0f * vy;

        cases.push_back({
            .name = "SingleParticleEulerIntegration",
            .acceleration = accel,
            .sortedPosition = sortedPos,
            .sortedVelocity = {{0.0f, 0.0f, 0.0f, 0.0f}},
            .sortedCellAndSerialId = {{0, 0}},
            .sortedParticleIdBySerialId = {0},
            .position = {{1.0f, 2.0f, 3.0f, 1.0f}}, // .w=1 => LIQUID
            .velocity = {{0, 0, 0, 0}},
            .neighborMap = neighborMap,
            .gravitationalAccelerationX = 0.0f,
            .gravitationalAccelerationY = 0.0f,
            .gravitationalAccelerationZ = 0.0f,
            .simulationScaleInv = 1.0f,
            .timeStep = 0.001f,
            .r0 = 0.1f,
            .expectedPredictedPosition = {{1.0f, yNew, 3.0f, 0.0f}},
        });
      }

      // ---- Test 3: ZeroAccelerationVelocityOnly ----
      // All acceleration zero; only velocity contributes.
      // v_new = v + dt * 0 = v = (2, 0, 0)
      // x_new = x + dt * simScaleInv * v_new = (0,0,0) + 0.01*0.5*(2,0,0) =
      // (0.01,0,0)
      {
        constexpr uint32_t N = 1;
        auto neighborMap = makeNeighborMap(N);
        auto accel = makeAccel(N);
        std::vector<Sibernetic::HostFloat4> pos = {{0.0f, 0.0f, 0.0f, 0.0f}};
        auto sortedPos = makeSortedPos(pos);

        float dt = 0.01f;
        float simScaleInv = 0.5f;
        float xNew = 0.0f + dt * simScaleInv * 2.0f;

        cases.push_back({
            .name = "ZeroAccelerationVelocityOnly",
            .acceleration = accel,
            .sortedPosition = sortedPos,
            .sortedVelocity = {{2.0f, 0.0f, 0.0f, 0.0f}},
            .sortedCellAndSerialId = {{0, 0}},
            .sortedParticleIdBySerialId = {0},
            .position = {{0.0f, 0.0f, 0.0f, 1.0f}}, // LIQUID
            .velocity = {{0, 0, 0, 0}},
            .neighborMap = neighborMap,
            .gravitationalAccelerationX = 0.0f,
            .gravitationalAccelerationY = 0.0f,
            .gravitationalAccelerationZ = 0.0f,
            .simulationScaleInv = simScaleInv,
            .timeStep = dt,
            .r0 = 0.1f,
            .expectedPredictedPosition = {{xNew, 0.0f, 0.0f, 0.0f}},
        });
      }

      // ---- Test 4: NonIdentityIndexMapping ----
      // 2 particles where sortedParticleIdBySerialId swaps them.
      // Thread 0 -> sorted id 1 (serial 0, liquid)
      // Thread 1 -> sorted id 0 (serial 1, liquid)
      // Both have same acceleration; verify positions land correctly.
      {
        constexpr uint32_t N = 2;
        auto neighborMap = makeNeighborMap(N);
        auto accel = makeAccel(N);
        // acc[0] = (1,0,0), acc[1] = (0,1,0) (non-pressure for sorted 0,1)
        accel[0] = {1.0f, 0.0f, 0.0f, 0.0f};
        accel[1] = {0.0f, 1.0f, 0.0f, 0.0f};

        std::vector<Sibernetic::HostFloat4> pos = {
            {10.0f, 0.0f, 0.0f, 0.0f}, // sorted 0
            {0.0f, 10.0f, 0.0f, 0.0f}, // sorted 1
        };
        auto sortedPos = makeSortedPos(pos);

        float dt = 0.001f;
        float simScaleInv = 1.0f;

        // Thread 0: id = sortedParticleIdBySerialId[0] = 1
        //   serial = sortedCellAndSerialId[1].y = 0 (liquid)
        //   acc_t_dt = acc[1] + acc[N+1] = (0,1,0) + (0,0,0) = (0,1,0)
        //   v_new = (0,0,0) + 0.001*(0,1,0) = (0,0.001,0)
        //   x_new = (0,10,0) + 0.001*1.0*(0,0.001,0) = (0,10.000001,0)
        //   -> written to sortedPos[N+1]
        //
        // Thread 1: id = sortedParticleIdBySerialId[1] = 0
        //   serial = sortedCellAndSerialId[0].y = 1 (liquid)
        //   acc_t_dt = acc[0] + acc[N+0] = (1,0,0) + (0,0,0) = (1,0,0)
        //   v_new = (0,0,0) + 0.001*(1,0,0) = (0.001,0,0)
        //   x_new = (10,0,0) + 0.001*1.0*(0.001,0,0) = (10.000001,0,0)
        //   -> written to sortedPos[N+0]
        float vx1 = 1.0f * dt;
        float xNew0 = 10.0f + dt * simScaleInv * vx1;
        float vy1 = 1.0f * dt;
        float yNew1 = 10.0f + dt * simScaleInv * vy1;

        cases.push_back({
            .name = "NonIdentityIndexMapping",
            .acceleration = accel,
            .sortedPosition = sortedPos,
            .sortedVelocity = {{0, 0, 0, 0}, {0, 0, 0, 0}},
            .sortedCellAndSerialId =
                {{0, 1}, {0, 0}}, // sorted[0]->serial 1, sorted[1]->serial 0
            .sortedParticleIdBySerialId =
                {1, 0}, // serial 0->sorted 1, serial 1->sorted 0
            .position = {{0, 10, 0, 1}, {10, 0, 0, 1}}, // both LIQUID
            .velocity = {{0, 0, 0, 0}, {0, 0, 0, 0}},
            .neighborMap = neighborMap,
            .gravitationalAccelerationX = 0.0f,
            .gravitationalAccelerationY = 0.0f,
            .gravitationalAccelerationZ = 0.0f,
            .simulationScaleInv = simScaleInv,
            .timeStep = dt,
            .r0 = 0.1f,
            // output: sortedPos[N+0] for sorted id 0, sortedPos[N+1] for sorted
            // id 1
            .expectedPredictedPosition = {{xNew0, 0.0f, 0.0f, 0.0f},
                                          {0.0f, yNew1, 0.0f, 0.0f}},
        });
      }

      // ---- Test 5: BoundaryCollisionPushesAway ----
      // A liquid particle is near a boundary particle. The boundary interaction
      // should push it away from the wall.
      {
        constexpr uint32_t N = 2;
        auto neighborMap = makeNeighborMap(N);
        auto accel = makeAccel(N);

        // Sorted particle 0 is liquid, sorted particle 1 is boundary.
        // Make particle 0 a neighbor of itself via neighborMap (sorted 0 sees
        // sorted 1).
        float dist = 0.04f; // less than r0=0.1
        setNeighbor(neighborMap, 0, 0, 1, dist);

        std::vector<Sibernetic::HostFloat4> pos = {
            {0.5f, 0.5f, 0.5f, 0.0f},  // sorted 0 (liquid)
            {0.5f, 0.46f, 0.5f, 0.0f}, // sorted 1 (boundary, below)
        };
        auto sortedPos = makeSortedPos(pos);

        float dt = 0.001f;
        float simScaleInv = 1.0f;
        float r0val = 0.1f;

        // The boundary particle stores its outward normal in velocity.
        // Normal pointing +y (away from boundary wall).
        // position[1].w = 3 (boundary).
        // After Euler step with no acceleration, predicted = original.
        // Then boundary interaction pushes it in +y direction.
        // We just verify y_predicted > y_original (pushed away).

        cases.push_back({
            .name = "BoundaryCollisionPushesAway",
            .acceleration = accel,
            .sortedPosition = sortedPos,
            .sortedVelocity = {{0, 0, 0, 0}, {0, 0, 0, 0}},
            .sortedCellAndSerialId =
                {{0, 0}, {0, 1}}, // sorted[0]->serial 0, sorted[1]->serial 1
            .sortedParticleIdBySerialId = {0, 1},
            .position = {{0.5f, 0.5f, 0.5f, 1.0f},    // serial 0: LIQUID
                         {0.5f, 0.46f, 0.5f, 3.0f}},  // serial 1: BOUNDARY
            .velocity = {{0, 0, 0, 0}, {0, 1, 0, 0}}, // boundary normal: +y
            .neighborMap = neighborMap,
            .gravitationalAccelerationX = 0.0f,
            .gravitationalAccelerationY = 0.0f,
            .gravitationalAccelerationZ = 0.0f,
            .simulationScaleInv = simScaleInv,
            .timeStep = dt,
            .r0 = r0val,
            // Exact value: Euler gives (0.5, 0.5, 0.5) (no accel).
            // Boundary pushes +y. w_c = (0.1-0.04)/0.1 = 0.6
            // push = n_hat * (w_c * (r0-dist)) / w_c = (r0-dist) = 0.06
            // y_new = 0.5 + 0.06 = 0.56
            .expectedPredictedPosition = {{0.5f, 0.56f, 0.5f, 0.0f},
                                          {0.5f, 0.46f, 0.5f, 0.0f}},
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
