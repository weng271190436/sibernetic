#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/kernels/PcisphIntegrateKernel.h"
#include "../../src/types/HostTypes.h"
#include "../utils/common/backend_param_test.h"

namespace SiberneticTest {

struct PcisphIntegrateResult {
  // Outputs that may be written depending on mode:
  std::vector<Sibernetic::HostFloat4> acceleration; // full 3×N buffer
  std::vector<Sibernetic::HostFloat4> sortedPosition;
  std::vector<Sibernetic::HostFloat4> originalPosition;
  std::vector<Sibernetic::HostFloat4> velocity;
};

struct PcisphIntegrateCase {
  using InputType = Sibernetic::PcisphIntegrateInput;
  using ResultType = PcisphIntegrateResult;

  const char *name;

  // Input data
  std::vector<Sibernetic::HostFloat4> acceleration; // size: 3*N
  std::vector<Sibernetic::HostFloat4> sortedPosition;
  std::vector<Sibernetic::HostFloat4> sortedVelocity;
  std::vector<Sibernetic::HostUInt2> sortedCellAndSerialId;
  std::vector<uint32_t> sortedParticleIdBySerialId;
  float simulationScaleInv;
  float deltaTime;
  std::vector<Sibernetic::HostFloat4> originalPosition;
  std::vector<Sibernetic::HostFloat4> velocity;
  float r0;
  std::vector<Sibernetic::HostFloat2> neighborMap;
  int32_t timestepIndex;
  int32_t mode;

  // Which outputs to verify (order must match declaration order for
  // designated initializer warnings).
  bool checkAcceleration = false;
  bool checkSortedPosition = false;
  bool checkOriginalPosition = false;
  bool checkVelocity = false;

  // Expected outputs (only checked when corresponding flag is true)
  std::vector<Sibernetic::HostFloat4> expectedAcceleration;
  std::vector<Sibernetic::HostFloat4> expectedSortedPosition;
  std::vector<Sibernetic::HostFloat4> expectedOriginalPosition;
  std::vector<Sibernetic::HostFloat4> expectedVelocity;

  InputType toInput() const {
    return {
        .acceleration = acceleration,
        .sortedPosition = sortedPosition,
        .sortedVelocity = sortedVelocity,
        .sortedCellAndSerialId = sortedCellAndSerialId,
        .sortedParticleIdBySerialId = sortedParticleIdBySerialId,
        .simulationScaleInv = simulationScaleInv,
        .deltaTime = deltaTime,
        .originalPosition = originalPosition,
        .velocity = velocity,
        .r0 = r0,
        .neighborMap = neighborMap,
        .particleCount =
            static_cast<uint32_t>(sortedParticleIdBySerialId.size()),
        .timestepIndex = timestepIndex,
        .mode = mode,
    };
  }

  void verify(const ResultType &result) const {
    if (checkAcceleration) {
      ASSERT_EQ(result.acceleration.size(), expectedAcceleration.size());
      for (size_t i = 0; i < expectedAcceleration.size(); ++i) {
        for (int c = 0; c < 4; ++c) {
          EXPECT_NEAR(result.acceleration[i][c], expectedAcceleration[i][c],
                      1e-4f)
              << "acceleration mismatch at [" << i << "][" << c << "]";
        }
      }
    }
    if (checkSortedPosition) {
      ASSERT_EQ(result.sortedPosition.size(), expectedSortedPosition.size());
      for (size_t i = 0; i < expectedSortedPosition.size(); ++i) {
        for (int c = 0; c < 4; ++c) {
          EXPECT_NEAR(result.sortedPosition[i][c], expectedSortedPosition[i][c],
                      1e-4f)
              << "sortedPosition mismatch at [" << i << "][" << c << "]";
        }
      }
    }
    if (checkOriginalPosition) {
      ASSERT_EQ(result.originalPosition.size(),
                expectedOriginalPosition.size());
      for (size_t i = 0; i < expectedOriginalPosition.size(); ++i) {
        for (int c = 0; c < 4; ++c) {
          EXPECT_NEAR(result.originalPosition[i][c],
                      expectedOriginalPosition[i][c], 1e-4f)
              << "originalPosition mismatch at [" << i << "][" << c << "]";
        }
      }
    }
    if (checkVelocity) {
      ASSERT_EQ(result.velocity.size(), expectedVelocity.size());
      for (size_t i = 0; i < expectedVelocity.size(); ++i) {
        for (int c = 0; c < 4; ++c) {
          EXPECT_NEAR(result.velocity[i][c], expectedVelocity[i][c], 1e-4f)
              << "velocity mismatch at [" << i << "][" << c << "]";
        }
      }
    }
  }
};

static_assert(SiberneticTest::KernelTestCase<PcisphIntegrateCase>);

class PcisphIntegrateRunner {
public:
  virtual ~PcisphIntegrateRunner() = default;
  virtual PcisphIntegrateResult run(const PcisphIntegrateCase &tc) = 0;
};

// Helper: create an empty neighborMap for N particles (all slots = no
// particle).
inline std::vector<Sibernetic::HostFloat2>
makeEmptyIntegrateNeighborMap(uint32_t particleCount) {
  std::vector<Sibernetic::HostFloat2> nm(static_cast<size_t>(particleCount) *
                                         32);
  for (auto &e : nm) {
    e = {-1.0f, -1.0f};
  }
  return nm;
}

struct PcisphIntegrateTestCommon {
  using Case = PcisphIntegrateCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = [] {
      std::vector<Case> cases;

      // Common parameters
      const float simScaleInv = 2.0f; // simulationScaleInv = 1/simScale
      const float dt = 0.5f;
      const float r0 = 0.5f;

      // Helper: make 3×N acceleration buffer initialized to zero.
      auto makeAccel = [](uint32_t N) {
        return std::vector<Sibernetic::HostFloat4>(
            static_cast<size_t>(N) * 3, Sibernetic::HostFloat4{0, 0, 0, 0});
      };

      // ---- Test 1: BoundaryParticleUnchanged ----
      // Boundary particle (type 3) should be skipped entirely.
      {
        const uint32_t N = 1;
        auto nm = makeEmptyIntegrateNeighborMap(N);
        auto accel = makeAccel(N);
        accel[0] = {1.0f, 2.0f, 3.0f, 0.0f}; // non-pressure
        accel[1] = {4.0f, 5.0f, 6.0f, 0.0f}; // pressure

        cases.push_back({
            .name = "BoundaryParticleUnchanged",
            .acceleration = accel,
            .sortedPosition = {{10.0f, 20.0f, 30.0f, 3.0f}},
            .sortedVelocity = {{1.0f, 1.0f, 1.0f, 0.0f}},
            .sortedCellAndSerialId = {{0, 0}},
            .sortedParticleIdBySerialId = {0},
            .simulationScaleInv = simScaleInv,
            .deltaTime = dt,
            .originalPosition = {{10.0f, 20.0f, 30.0f, 3.0f}}, // boundary
            .velocity = {{0.0f, 0.0f, 0.0f, 0.0f}},
            .r0 = r0,
            .neighborMap = nm,
            .timestepIndex = 1,
            .mode = 0,
            // Boundary → no writes at all. sortedPosition, velocity,
            // acceleration should be unchanged.
            .checkAcceleration = true,
            .checkSortedPosition = true,
            .checkVelocity = true,
            .expectedAcceleration = accel,
            .expectedSortedPosition = {{10.0f, 20.0f, 30.0f, 3.0f}},
            .expectedVelocity = {{0.0f, 0.0f, 0.0f, 0.0f}},
        });
      }

      // ---- Test 2: IterationZeroStoresAcceleration ----
      // timestepIndex == 0 → store acc[id] + acc[N+id] into acc[2N+serialId].
      {
        const uint32_t N = 1;
        auto nm = makeEmptyIntegrateNeighborMap(N);
        auto accel = makeAccel(N);
        accel[0] = {1.0f, 2.0f, 3.0f, 0.0f}; // non-pressure at sorted id 0
        accel[1] = {4.0f, 5.0f, 6.0f, 0.0f}; // pressure at sorted id 0

        auto expectedAccel = accel;
        // acc[2N + serialId] = acc[0] + acc[1] = (5, 7, 9, 0)
        expectedAccel[2] = {5.0f, 7.0f, 9.0f, 0.0f};

        cases.push_back({
            .name = "IterationZeroStoresAcceleration",
            .acceleration = accel,
            .sortedPosition = {{1.0f, 2.0f, 3.0f, 0.0f}},
            .sortedVelocity = {{0.0f, 0.0f, 0.0f, 0.0f}},
            .sortedCellAndSerialId = {{0, 0}},
            .sortedParticleIdBySerialId = {0},
            .simulationScaleInv = simScaleInv,
            .deltaTime = dt,
            .originalPosition = {{1.0f, 2.0f, 3.0f, 2.0f}}, // fluid
            .velocity = {{0.0f, 0.0f, 0.0f, 0.0f}},
            .r0 = r0,
            .neighborMap = nm,
            .timestepIndex = 0,
            .mode = 0, // mode is irrelevant for timestepIndex==0
            .checkAcceleration = true,
            .expectedAcceleration = expectedAccel,
        });
      }

      // ---- Test 3: Mode0PositionUpdate ----
      // Leapfrog position update: x(t+dt) = x(t) + v*dt + a*dt^2/2
      //   (all scaled by simulationScaleInv)
      // With v = (1,0,0), a_prev = (2,0,0), dt = 0.5, simScaleInv = 2:
      //   dx = (v*dt + a*dt^2/2) * simScaleInv
      //      = (1*0.5 + 2*0.5*0.5/2) * 2 = (0.5 + 0.25) * 2 = 1.5
      //   x(t+dt) = 10 + 1.5 = 11.5
      // sortedPosition[id].w should be set to particleType.
      {
        const uint32_t N = 1;
        auto nm = makeEmptyIntegrateNeighborMap(N);
        auto accel = makeAccel(N);
        // acc[2N + 0] = previous combined acceleration
        accel[2] = {2.0f, 0.0f, 0.0f, 0.0f};

        cases.push_back({
            .name = "Mode0PositionUpdate",
            .acceleration = accel,
            .sortedPosition = {{10.0f, 0.0f, 0.0f, 0.0f}},
            .sortedVelocity = {{1.0f, 0.0f, 0.0f, 0.0f}},
            .sortedCellAndSerialId = {{0, 0}},
            .sortedParticleIdBySerialId = {0},
            .simulationScaleInv = simScaleInv,
            .deltaTime = dt,
            .originalPosition = {{10.0f, 0.0f, 0.0f, 2.0f}}, // fluid type 2
            .velocity = {{0.0f, 0.0f, 0.0f, 0.0f}},
            .r0 = r0,
            .neighborMap = nm,
            .timestepIndex = 1,
            .mode = 0,
            .checkSortedPosition = true,
            // x(t+dt) = 10 + (1*0.5 + 2*0.25/2)*2 = 10 + 1.5 = 11.5
            // .w = particleType = 2.0
            .expectedSortedPosition = {{11.5f, 0.0f, 0.0f, 2.0f}},
        });
      }

      // ---- Test 4: Mode1VelocityUpdate ----
      // Leapfrog velocity update:
      //   a(t+dt) = acc[id] + acc[N+id]
      //   v(t+dt) = v(t) + (a(t) + a(t+dt)) * dt / 2
      //
      // No boundary neighbors → computeInteractionWithBoundaryParticles is
      // identity. Writes: velocity[serialId], acceleration[2N+serialId],
      //   position[serialId] (= sortedPosition[id]), position[serialId].w =
      //   type.
      //
      // v(t) = (1,0,0), a_prev = (2,0,0), a_current = (0,4,0)
      //   v(t+dt) = (1,0,0) + ((2,0,0) + (0,4,0)) * 0.5 / 2
      //           = (1,0,0) + (2,4,0) * 0.25 = (1.5, 1.0, 0)
      //
      // position[serialId] = sortedPosition[id] (after mode 0 updated it)
      {
        const uint32_t N = 1;
        auto nm = makeEmptyIntegrateNeighborMap(N);
        auto accel = makeAccel(N);
        accel[0] = {0.0f, 4.0f, 0.0f, 0.0f}; // non-pressure
        accel[1] = {0.0f, 0.0f, 0.0f, 0.0f}; // pressure
        accel[2] = {2.0f, 0.0f, 0.0f, 0.0f}; // previous combined

        auto expectedAccel = accel;
        // a(t+dt) = acc[0] + acc[1] = (0, 4, 0, 0)
        expectedAccel[2] = {0.0f, 4.0f, 0.0f, 0.0f};

        cases.push_back({
            .name = "Mode1VelocityUpdate",
            .acceleration = accel,
            .sortedPosition = {{11.5f, 0.0f, 0.0f, 0.0f}},
            .sortedVelocity = {{1.0f, 0.0f, 0.0f, 0.0f}},
            .sortedCellAndSerialId = {{0, 0}},
            .sortedParticleIdBySerialId = {0},
            .simulationScaleInv = simScaleInv,
            .deltaTime = dt,
            .originalPosition = {{10.0f, 0.0f, 0.0f, 2.0f}}, // fluid
            .velocity = {{0.0f, 0.0f, 0.0f, 0.0f}},
            .r0 = r0,
            .neighborMap = nm,
            .timestepIndex = 1,
            .mode = 1,
            .checkAcceleration = true,
            .checkOriginalPosition = true,
            .checkVelocity = true,
            .expectedAcceleration = expectedAccel,
            // position[serialId] = sortedPosition[id]
            // position[serialId].w = particleType = 2.0
            .expectedOriginalPosition = {{11.5f, 0.0f, 0.0f, 2.0f}},
            // v(t+dt) = (1,0,0) + ((2,0,0) + (0,4,0)) * 0.5 * 0.5
            //         = (1,0,0) + (0.5, 1.0, 0) = (1.5, 1.0, 0)
            .expectedVelocity = {{1.5f, 1.0f, 0.0f, 0.0f}},
        });
      }

      // ---- Test 5: Mode2SemiImplicitEuler ----
      // Semi-implicit Euler:
      //   a(t+dt) = acc[id] + acc[N+id]
      //   v(t+dt) = v(t) + a(t+dt) * dt
      //   x(t+dt) = x(t) + v(t+dt) * dt * simScaleInv
      //
      // v(t) = (0,0,1), a_current = (0,0,2), dt = 0.5, simScaleInv = 2
      //   v(t+dt) = (0,0,1) + (0,0,2)*0.5 = (0,0,2)
      //   x(t+dt) = (5,5,5) + (0,0,2)*0.5*2 = (5,5,7)
      {
        const uint32_t N = 1;
        auto nm = makeEmptyIntegrateNeighborMap(N);
        auto accel = makeAccel(N);
        accel[0] = {0.0f, 0.0f, 2.0f, 0.0f}; // non-pressure
        accel[1] = {0.0f, 0.0f, 0.0f, 0.0f}; // pressure
        accel[2] = {0.0f, 0.0f, 0.0f, 0.0f}; // previous (unused in mode 2)

        auto expectedAccel = accel;
        // a(t+dt) stored at [2N+serialId]
        expectedAccel[2] = {0.0f, 0.0f, 2.0f, 0.0f};

        cases.push_back({
            .name = "Mode2SemiImplicitEuler",
            .acceleration = accel,
            .sortedPosition = {{5.0f, 5.0f, 5.0f, 0.0f}},
            .sortedVelocity = {{0.0f, 0.0f, 1.0f, 0.0f}},
            .sortedCellAndSerialId = {{0, 0}},
            .sortedParticleIdBySerialId = {0},
            .simulationScaleInv = simScaleInv,
            .deltaTime = dt,
            .originalPosition = {{5.0f, 5.0f, 5.0f, 2.0f}}, // fluid
            .velocity = {{0.0f, 0.0f, 0.0f, 0.0f}},
            .r0 = r0,
            .neighborMap = nm,
            .timestepIndex = 1,
            .mode = 2,
            .checkAcceleration = true,
            .checkOriginalPosition = true,
            .checkVelocity = true,
            .expectedAcceleration = expectedAccel,
            // x(t+dt) = (5,5,5) + (0,0,2)*0.5*2 = (5,5,7), .w = 2.0
            .expectedOriginalPosition = {{5.0f, 5.0f, 7.0f, 2.0f}},
            // v(t+dt) = (0,0,1) + (0,0,2)*0.5 = (0,0,2)
            .expectedVelocity = {{0.0f, 0.0f, 2.0f, 0.0f}},
        });
      }

      // ---- Test 6: NonIdentityIndexMapping ----
      // Two fluid particles where sorted order is reversed from serial order.
      // serial 0 → sorted 1, serial 1 → sorted 0.
      // Mode 0 position update to verify correct index indirection.
      //
      // Particle serial 0 (sorted 1):
      //   sortedPosition[1] = (20, 0, 0), sortedVelocity[1] = (0, 1, 0)
      //   acc_prev = acceleration[2N + 0] = acceleration[4] = (0, 0, 4)
      //   dx = (v*dt + a*dt^2/2) * simScaleInv
      //      = ((0,1,0)*0.5 + (0,0,4)*0.125) * 2 = ((0,0.5,0) + (0,0,0.5))
      //      * 2 = (0, 1, 1) x(t+dt) = (20, 1, 1), .w = 2.0
      //
      // Particle serial 1 (sorted 0):
      //   sortedPosition[0] = (10, 0, 0), sortedVelocity[0] = (1, 0, 0)
      //   acc_prev = acceleration[2N + 1] = acceleration[5] = (2, 0, 0)
      //   dx = ((1,0,0)*0.5 + (2,0,0)*0.125) * 2 = (0.75) * 2 = (1.5, 0, 0)
      //   x(t+dt) = (11.5, 0, 0), .w = 2.0
      {
        const uint32_t N = 2;
        auto nm = makeEmptyIntegrateNeighborMap(N);
        auto accel = makeAccel(N);
        // acc_prev for serial 0 → acc[2N + 0] = acc[4]
        accel[4] = {0.0f, 0.0f, 4.0f, 0.0f};
        // acc_prev for serial 1 → acc[2N + 1] = acc[5]
        accel[5] = {2.0f, 0.0f, 0.0f, 0.0f};

        cases.push_back({
            .name = "NonIdentityIndexMapping",
            .acceleration = accel,
            // sorted 0 = serial 1, sorted 1 = serial 0
            .sortedPosition = {{10.0f, 0.0f, 0.0f, 0.0f},
                               {20.0f, 0.0f, 0.0f, 0.0f}},
            .sortedVelocity = {{1.0f, 0.0f, 0.0f, 0.0f},
                               {0.0f, 1.0f, 0.0f, 0.0f}},
            // sorted 0 → serial 1, sorted 1 → serial 0
            .sortedCellAndSerialId = {{0, 1}, {0, 0}},
            // serial 0 → sorted 1, serial 1 → sorted 0
            .sortedParticleIdBySerialId = {1, 0},
            .simulationScaleInv = simScaleInv,
            .deltaTime = dt,
            .originalPosition = {{20.0f, 0.0f, 0.0f, 2.0f},
                                 {10.0f, 0.0f, 0.0f, 2.0f}},
            .velocity = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}},
            .r0 = r0,
            .neighborMap = nm,
            .timestepIndex = 1,
            .mode = 0,
            .checkSortedPosition = true,
            // sorted 0 (serial 1): x = 10 + 1.5 = 11.5, .w = type of
            //   serial 1 = 2.0
            // sorted 1 (serial 0): x = (20, 1, 1), .w = type of serial 0 =
            //   2.0
            .expectedSortedPosition = {{11.5f, 0.0f, 0.0f, 2.0f},
                                       {20.0f, 1.0f, 1.0f, 2.0f}},
        });
      }

      // ---- Test 7: AccelerationWZeroed ----
      // Verifies that .w of acceleration_prev and acceleration_current is
      // zeroed before use. Feed non-zero .w into acceleration entries;
      // result should be same as if .w were 0.
      // Mode 1: v(t+dt) = v(t) + (a_prev + a_cur) * dt / 2
      // With a_prev = (1, 0, 0, 99), a_cur = (0, 2, 0, 77):
      //   zeroed: a_prev.w = 0, a_cur.w = 0
      //   v(t+dt) = (0,0,0) + ((1,0,0,0) + (0,2,0,0)) * 0.25 = (0.25, 0.5, 0,
      //   0) stored a_cur at [2N] should have .w = 0
      {
        const uint32_t N = 1;
        auto nm = makeEmptyIntegrateNeighborMap(N);
        auto accel = makeAccel(N);
        accel[0] = {0.0f, 2.0f, 0.0f, 77.0f}; // non-pressure, bad .w
        accel[1] = {0.0f, 0.0f, 0.0f, 0.0f};  // pressure
        accel[2] = {1.0f, 0.0f, 0.0f, 99.0f}; // previous combined, bad .w

        auto expectedAccel = accel;
        // a_cur = acc[0] + acc[1] = (0,2,0,77), then .w zeroed → (0,2,0,0)
        expectedAccel[2] = {0.0f, 2.0f, 0.0f, 0.0f};

        cases.push_back({
            .name = "AccelerationWZeroed",
            .acceleration = accel,
            .sortedPosition = {{5.0f, 5.0f, 5.0f, 0.0f}},
            .sortedVelocity = {{0.0f, 0.0f, 0.0f, 0.0f}},
            .sortedCellAndSerialId = {{0, 0}},
            .sortedParticleIdBySerialId = {0},
            .simulationScaleInv = simScaleInv,
            .deltaTime = dt,
            .originalPosition = {{5.0f, 5.0f, 5.0f, 2.0f}},
            .velocity = {{0.0f, 0.0f, 0.0f, 0.0f}},
            .r0 = r0,
            .neighborMap = nm,
            .timestepIndex = 1,
            .mode = 1,
            .checkAcceleration = true,
            .checkVelocity = true,
            .expectedAcceleration = expectedAccel,
            // v(t+dt) = (0,0,0) + ((1,0,0) + (0,2,0)) * 0.5 * 0.5
            //         = (0.25, 0.5, 0, 0)
            .expectedVelocity = {{0.25f, 0.5f, 0.0f, 0.0f}},
        });
      }

      // ---- Test 8: MixedFluidAndBoundary ----
      // Two particles: serial 0 = fluid, serial 1 = boundary.
      // Identity index mapping (sorted = serial).
      // Mode 0. Only the fluid particle should be updated; boundary
      // particle's sortedPosition remains unchanged.
      {
        const uint32_t N = 2;
        auto nm = makeEmptyIntegrateNeighborMap(N);
        auto accel = makeAccel(N);
        accel[4] = {0.0f, 0.0f, 2.0f, 0.0f}; // a_prev for serial 0
        accel[5] = {0.0f, 0.0f, 0.0f, 0.0f}; // a_prev for serial 1 (unused)

        cases.push_back({
            .name = "MixedFluidAndBoundary",
            .acceleration = accel,
            .sortedPosition = {{1.0f, 2.0f, 3.0f, 0.0f},
                               {10.0f, 20.0f, 30.0f, 3.0f}},
            .sortedVelocity = {{1.0f, 0.0f, 0.0f, 0.0f},
                               {0.0f, 0.0f, 0.0f, 0.0f}},
            .sortedCellAndSerialId = {{0, 0}, {0, 1}},
            .sortedParticleIdBySerialId = {0, 1},
            .simulationScaleInv = simScaleInv,
            .deltaTime = dt,
            .originalPosition = {{1.0f, 2.0f, 3.0f, 2.0f},     // fluid
                                 {10.0f, 20.0f, 30.0f, 3.0f}}, // boundary
            .velocity = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}},
            .r0 = r0,
            .neighborMap = nm,
            .timestepIndex = 1,
            .mode = 0,
            .checkSortedPosition = true,
            // Fluid (sorted 0): dx = (1*0.5 + 2*0.125)*2 = 1.5 in z
            //   Wait — a_prev = (0,0,2), v = (1,0,0)
            //   dx = ((1,0,0)*0.5 + (0,0,2)*0.125) * 2
            //      = (0.5, 0, 0.25) * 2 = (1.0, 0, 0.5)
            //   x(t+dt) = (2.0, 2.0, 3.5), .w = 2.0
            // Boundary (sorted 1): unchanged = (10, 20, 30, 3)
            .expectedSortedPosition = {{2.0f, 2.0f, 3.5f, 2.0f},
                                       {10.0f, 20.0f, 30.0f, 3.0f}},
        });
      }

      // ---- Test 9: Mode1BoundaryInteraction ----
      // Fluid particle with a boundary neighbor. Mode 1 calls
      // computeInteractionWithBoundaryParticles with correctVelocity=true.
      //
      // Setup (N=2, identity mapping):
      //   Particle 0: fluid at sortedPos (1, 0, 0)
      //   Particle 1: boundary at originalPos (0, 0, 0)
      //   Boundary normal: velocity[1] = (1, 0, 0) (outward)
      //   r0 = 2.0 (so distance = 1 < r0)
      //   neighborMap: particle 0 sees particle 1 at distance 1
      //
      // Leapfrog velocity (before boundary):
      //   a_prev = (0,0,0), a_cur = (0,0,0) → v(t+dt) = v(t) = (-2,0,0)
      //   position_t_dt = sortedPosition[0] = (1,0,0)
      //
      // Boundary interaction (Ihmsen):
      //   d = (1,0,0) - (0,0,0) = (1,0,0), dist = 1
      //   w_c_ib = (2-1)/2 = 0.5
      //   n_c_i = (1,0,0)*0.5 = (0.5,0,0)
      //   w_c_ib_sum = 0.5, w_c_ib_second_sum = 0.5*(2-1) = 0.5
      //   n_c_i_length = 0.5
      //   deltaPos = (0.5/0.5) * 0.5 / 0.5 = (1,0,0)
      //   position += (1,0,0) → (2,0,0)
      //
      //   Velocity correction (correctVelocity=true):
      //   dot(n_c_i, v) = dot((0.5,0,0), (-2,0,0)) = -1 < 0
      //   v -= n_c_i * (-1) = (-2,0,0) + (0.5,0,0) = (-1.5,0,0)
      //   v *= 0.99 → (-1.485, 0, 0)
      {
        const uint32_t N = 2;
        const float bigR0 = 2.0f;
        auto nm = makeEmptyIntegrateNeighborMap(N);
        // Particle 0 sees particle 1 at distance 1.0
        nm[0 * 32 + 0] = {1.0f, 1.0f};
        auto accel = makeAccel(N);
        // All accelerations zero → pure velocity + boundary test.

        auto expectedAccel = accel;
        // a_cur for serial 0 = acc[0]+acc[2] = (0,0,0,0)
        expectedAccel[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        cases.push_back({
            .name = "Mode1BoundaryInteraction",
            .acceleration = accel,
            .sortedPosition = {{1.0f, 0.0f, 0.0f, 0.0f},
                               {0.0f, 0.0f, 0.0f, 3.0f}},
            .sortedVelocity = {{-2.0f, 0.0f, 0.0f, 0.0f},
                               {0.0f, 0.0f, 0.0f, 0.0f}},
            .sortedCellAndSerialId = {{0, 0}, {0, 1}},
            .sortedParticleIdBySerialId = {0, 1},
            .simulationScaleInv = simScaleInv,
            .deltaTime = dt,
            .originalPosition = {{1.0f, 0.0f, 0.0f, 2.0f},  // fluid
                                 {0.0f, 0.0f, 0.0f, 3.0f}}, // boundary
            // velocity[1] = boundary outward normal
            .velocity = {{0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f}},
            .r0 = bigR0,
            .neighborMap = nm,
            .timestepIndex = 1,
            .mode = 1,
            .checkAcceleration = true,
            .checkOriginalPosition = true,
            .checkVelocity = true,
            .expectedAcceleration = expectedAccel,
            // position = (1,0,0) + (1,0,0) boundary push = (2,0,0), .w = 2
            .expectedOriginalPosition = {{2.0f, 0.0f, 0.0f, 2.0f},
                                         {0.0f, 0.0f, 0.0f, 3.0f}},
            // v = (-2,0,0) → (-1.5,0,0) → *0.99 = (-1.485, 0, 0)
            .expectedVelocity = {{-1.485f, 0.0f, 0.0f, 0.0f},
                                 {1.0f, 0.0f, 0.0f, 0.0f}},
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
