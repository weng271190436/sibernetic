#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/kernels/ComputeInteractionWithMembranesKernel.h"
#include "../utils/common/backend_param_test.h"
#include "../utils/types/types.h"

namespace SiberneticTest {

// ── Result types ────────────────────────────────────────────────────────────

struct ComputeInteractionWithMembranesResult {
  std::vector<HostFloat4> position; // size: 2 × particleCount
};

struct ComputeInteractionWithMembranesFinalizeResult {
  std::vector<HostFloat4> position; // size: 2 × particleCount
};

// ── Helper: create a float4 ─────────────────────────────────────────────────

inline HostFloat4 f4(float x, float y, float z, float w) {
  return {x, y, z, w};
}

inline HostFloat2 f2(float x, float y) { return {x, y}; }

// Build identity sort: sortedParticleIdBySerialId[i] = i,
// sortedCellAndSerialId[i] = (0, i).
inline void identitySort(uint32_t n, std::vector<uint32_t> &backMap,
                          std::vector<HostUInt2> &cellAndSerial) {
  backMap.resize(n);
  cellAndSerial.resize(n);
  for (uint32_t i = 0; i < n; ++i) {
    backMap[i] = i;
    cellAndSerial[i] = {0, i};
  }
}

// Fill neighborMap with sentinels.
inline std::vector<HostFloat2>
emptyNeighborMap(uint32_t particleCount) {
  std::vector<HostFloat2> nm(
      particleCount * Sibernetic::kMaxNeighborCount, {-1.0f, -1.0f});
  return nm;
}

// Set a single neighbor entry in the neighborMap.
inline void setNeighbor(std::vector<HostFloat2> &nm, uint32_t sortedId,
                         int slot, int neighborSortedId, float distance) {
  nm[sortedId * Sibernetic::kMaxNeighborCount + slot] = {
      static_cast<float>(neighborSortedId), distance};
}

// ── computeInteractionWithMembranes test case ───────────────────────────────

struct ComputeInteractionWithMembranesCase {
  using InputType = Sibernetic::ComputeInteractionWithMembranesInput;
  using ResultType = ComputeInteractionWithMembranesResult;

  const char *name;
  std::vector<HostFloat4> position; // size: 2 × particleCount
  std::vector<HostFloat4> velocity; // size: 2 × particleCount
  std::vector<HostUInt2> sortedCellAndSerialId;
  std::vector<uint32_t> sortedParticleIdBySerialId;
  std::vector<HostFloat2> neighborMap;
  std::vector<int32_t> particleMembranesList;
  std::vector<int32_t> membraneData;
  uint32_t particleCount;
  float r0;

  // Expected delta in position[N + serialId] after the kernel runs.
  // Only .xyz is checked (delta). If empty, all deltas should be zero.
  std::vector<HostFloat4> expectedDelta;

  InputType toInput() const {
    return {
        .position = position,
        .velocity = velocity,
        .sortedCellAndSerialId = sortedCellAndSerialId,
        .sortedParticleIdBySerialId = sortedParticleIdBySerialId,
        .neighborMap = neighborMap,
        .particleMembranesList = particleMembranesList,
        .membraneData = membraneData,
        .particleCount = particleCount,
        .r0 = r0,
    };
  }

  void verify(const ResultType &result) const {
    const uint32_t N = particleCount;
    ASSERT_EQ(result.position.size(), 2 * N);

    // Check [0..N) unchanged.
    for (uint32_t i = 0; i < N; ++i) {
      for (int c = 0; c < 4; ++c) {
        EXPECT_FLOAT_EQ(result.position[i][c], position[i][c])
            << "position[" << i << "][" << c << "] should be unchanged";
      }
    }

    // Check [N..2N) delta region.
    if (expectedDelta.empty()) {
      // All deltas should be zero.
      for (uint32_t i = 0; i < N; ++i) {
        for (int c = 0; c < 3; ++c) {
          EXPECT_NEAR(result.position[N + i][c], 0.0f, 1e-6f)
              << "position[N+" << i << "][" << c << "] should be zero";
        }
      }
    } else {
      ASSERT_EQ(expectedDelta.size(), N);
      for (uint32_t i = 0; i < N; ++i) {
        for (int c = 0; c < 3; ++c) {
          EXPECT_NEAR(result.position[N + i][c], expectedDelta[i][c], 1e-5f)
              << "position[N+" << i << "][" << c << "] delta mismatch";
        }
      }
    }
  }
};

static_assert(SiberneticTest::KernelTestCase<ComputeInteractionWithMembranesCase>);

class ComputeInteractionWithMembranesRunner
    : public TestRunner<ComputeInteractionWithMembranesCase,
                        ComputeInteractionWithMembranesResult> {};

// ── computeInteractionWithMembranes_finalize test case ──────────────────────

struct ComputeInteractionWithMembranesFinalizeCase {
  using InputType = Sibernetic::ComputeInteractionWithMembranesFinalizeInput;
  using ResultType = ComputeInteractionWithMembranesFinalizeResult;

  const char *name;
  std::vector<HostFloat4> position; // size: 2 × particleCount
  std::vector<uint32_t> sortedParticleIdBySerialId;
  uint32_t particleCount;

  // Expected position[0..N) after finalization.
  std::vector<HostFloat4> expectedPosition;

  InputType toInput() const {
    return {
        .position = position,
        .sortedParticleIdBySerialId = sortedParticleIdBySerialId,
        .particleCount = particleCount,
    };
  }

  void verify(const ResultType &result) const {
    const uint32_t N = particleCount;
    ASSERT_EQ(result.position.size(), 2 * N);

    for (uint32_t i = 0; i < N; ++i) {
      for (int c = 0; c < 4; ++c) {
        EXPECT_NEAR(result.position[i][c], expectedPosition[i][c], 1e-6f)
            << "position[" << i << "][" << c << "] mismatch";
      }
    }
  }
};

static_assert(
    SiberneticTest::KernelTestCase<ComputeInteractionWithMembranesFinalizeCase>);

class ComputeInteractionWithMembranesFinalizeRunner
    : public TestRunner<ComputeInteractionWithMembranesFinalizeCase,
                        ComputeInteractionWithMembranesFinalizeResult> {};

// ── Test cases: computeInteractionWithMembranes ─────────────────────────────

struct ComputeInteractionWithMembranesTestCommon {
  using Case = ComputeInteractionWithMembranesCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = {

        // Boundary particle: no delta produced.
        [] {
          Case tc{};
          tc.name = "BoundaryParticleSkipped";
          tc.particleCount = 1;
          tc.r0 = 0.5f;
          // type 3.0 = boundary
          tc.position = {f4(1.0f, 1.0f, 1.0f, 3.0f),
                         f4(0.0f, 0.0f, 0.0f, 0.0f)};
          tc.velocity = {f4(0.0f, 0.0f, 0.0f, 0.0f),
                         f4(0.0f, 0.0f, 0.0f, 0.0f)};
          identitySort(1, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);
          tc.neighborMap = emptyNeighborMap(1);
          tc.particleMembranesList = {-1, -1, -1, -1, -1, -1, -1};
          tc.membraneData = {};
          return tc;
        }(),

        // Elastic particle (type 2): skipped (not liquid).
        [] {
          Case tc{};
          tc.name = "NonLiquidParticleSkipped";
          tc.particleCount = 1;
          tc.r0 = 0.5f;
          tc.position = {f4(1.0f, 1.0f, 1.0f, 2.1f),
                         f4(0.0f, 0.0f, 0.0f, 0.0f)};
          tc.velocity = {f4(0.0f, 0.0f, 0.0f, 0.0f),
                         f4(0.0f, 0.0f, 0.0f, 0.0f)};
          identitySort(1, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);
          tc.neighborMap = emptyNeighborMap(1);
          tc.particleMembranesList = {-1, -1, -1, -1, -1, -1, -1};
          tc.membraneData = {};
          return tc;
        }(),

        // Liquid particle with no elastic neighbors: no delta.
        [] {
          Case tc{};
          tc.name = "LiquidNoMembraneNeighbors";
          tc.particleCount = 2;
          tc.r0 = 0.5f;
          // Particle 0: liquid, particle 1: liquid (not elastic)
          tc.position = {f4(1.0f, 1.0f, 0.0f, 1.0f),
                         f4(2.0f, 1.0f, 0.0f, 1.0f),
                         f4(0.0f, 0.0f, 0.0f, 0.0f),
                         f4(0.0f, 0.0f, 0.0f, 0.0f)};
          tc.velocity.resize(4, f4(0, 0, 0, 0));
          identitySort(2, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);
          tc.neighborMap = emptyNeighborMap(2);
          // Particle 0 sees particle 1 as neighbor, but particle 1 is liquid,
          // not elastic.
          setNeighbor(tc.neighborMap, 0, 0, 1, 0.1f);
          // No elastic particles → no membranes list needed beyond sentinels.
          tc.particleMembranesList = {
              -1, -1, -1, -1, -1, -1, -1, // particle 0
              -1, -1, -1, -1, -1, -1, -1, // particle 1
          };
          tc.membraneData = {};
          return tc;
        }(),

        // Liquid particle near a membrane triangle → should get a position
        // delta. Setup: 4 particles. Particle 0 is liquid at (0.5, 0.5, 0.1).
        // Particles 1, 2, 3 are elastic, forming a triangle in the z=0 plane.
        // Membrane triangle: vertices 1, 2, 3.
        // Particle 0's projection onto that plane is (0.5, 0.5, 0).
        // Normal = (0, 0, 1), length = 0.1.
        [] {
          Case tc{};
          tc.name = "LiquidNearMembrane";
          tc.particleCount = 4;
          tc.r0 = 2.0f;
          // Particle 0: liquid at (0.5, 0.5, 0.1)
          // Particles 1,2,3: elastic triangle in z=0 plane
          tc.position = {
              f4(0.5f, 0.5f, 0.1f, 1.0f), // 0: liquid
              f4(0.0f, 0.0f, 0.0f, 2.1f), // 1: elastic
              f4(1.0f, 0.0f, 0.0f, 2.1f), // 2: elastic
              f4(0.0f, 1.0f, 0.0f, 2.1f), // 3: elastic
              f4(0.0f, 0.0f, 0.0f, 0.0f), // delta for 0
              f4(0.0f, 0.0f, 0.0f, 0.0f), // delta for 1
              f4(0.0f, 0.0f, 0.0f, 0.0f), // delta for 2
              f4(0.0f, 0.0f, 0.0f, 0.0f), // delta for 3
          };
          tc.velocity.resize(8, f4(0, 0, 0, 0));
          identitySort(4, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);
          tc.neighborMap = emptyNeighborMap(4);
          setNeighbor(tc.neighborMap, 0, 0, 1, 0.707f);
          setNeighbor(tc.neighborMap, 0, 1, 2, 0.707f);
          setNeighbor(tc.neighborMap, 0, 2, 3, 0.707f);

          // particleMembranesList: particles 1,2,3 each belong to membrane 0.
          // Particle 0 has no membranes.
          tc.particleMembranesList = {
              -1, -1, -1, -1, -1, -1, -1, // particle 0: liquid, no membranes
              0,  -1, -1, -1, -1, -1, -1,  // particle 1: membrane 0
              0,  -1, -1, -1, -1, -1, -1,  // particle 2: membrane 0
              0,  -1, -1, -1, -1, -1, -1,  // particle 3: membrane 0
          };
          // membraneData: membrane 0 → vertices 1, 2, 3.
          tc.membraneData = {1, 2, 3};

          // Expected delta: The kernel computes displacement as a float4 with
          // .z zeroed but .w still present (type difference: 1.0 − 2.1 = −1.1).
          // The 4D dot product gives dist = sqrt(0.5 + 1.21) = sqrt(1.71).
          // With r0 = 2.0: w = (2.0 − sqrt(1.71)) / 2.0.
          // All 3 neighbors produce the same distance and normal (0,0,1).
          // When normals and distances are identical, the Ihmsen formula
          // simplifies to delta = normal * (r0 − dist).
          const float dw = 1.0f - 2.1f; // type difference in .w
          const float dist = std::sqrt(0.5f + dw * dw);
          const float dz = tc.r0 - dist;
          tc.expectedDelta = {
              f4(0.0f, 0.0f, dz, 0.0f),   // particle 0: pushed in +z
              f4(0.0f, 0.0f, 0.0f, 0.0f), // particle 1: elastic, skipped
              f4(0.0f, 0.0f, 0.0f, 0.0f), // particle 2: elastic, skipped
              f4(0.0f, 0.0f, 0.0f, 0.0f), // particle 3: elastic, skipped
          };
          return tc;
        }(),

        // Liquid particle near two separate membrane triangles sharing an edge.
        // Tests that normals from multiple triangles per neighbor are averaged.
        [] {
          Case tc{};
          tc.name = "MultipleMembranePlanes";
          tc.particleCount = 5;
          tc.r0 = 2.0f;
          // Particle 0: liquid at (0.5, 0.5, 0.1)
          // Particles 1-4: elastic, forming two triangles sharing edge 1-2.
          // Triangle 0: (1, 2, 3) in z=0 plane.
          // Triangle 1: (1, 2, 4) also in z=0 plane (same normal).
          tc.position = {
              f4(0.5f, 0.5f, 0.1f, 1.0f), // 0: liquid
              f4(0.0f, 0.0f, 0.0f, 2.1f), // 1: elastic
              f4(1.0f, 0.0f, 0.0f, 2.1f), // 2: elastic
              f4(0.0f, 1.0f, 0.0f, 2.1f), // 3: elastic
              f4(1.0f, 1.0f, 0.0f, 2.1f), // 4: elastic
              // delta region
              f4(0.0f, 0.0f, 0.0f, 0.0f),
              f4(0.0f, 0.0f, 0.0f, 0.0f),
              f4(0.0f, 0.0f, 0.0f, 0.0f),
              f4(0.0f, 0.0f, 0.0f, 0.0f),
              f4(0.0f, 0.0f, 0.0f, 0.0f),
          };
          tc.velocity.resize(10, f4(0, 0, 0, 0));
          identitySort(5, tc.sortedParticleIdBySerialId,
                       tc.sortedCellAndSerialId);
          tc.neighborMap = emptyNeighborMap(5);
          // Particle 0 neighbors: 1 and 2 (shared edge).
          setNeighbor(tc.neighborMap, 0, 0, 1, 0.707f);
          setNeighbor(tc.neighborMap, 0, 1, 2, 0.707f);

          // Particle 1 belongs to both triangles, particle 2 belongs to both.
          tc.particleMembranesList = {
              -1, -1, -1, -1, -1, -1, -1, // 0: liquid
              0,  1,  -1, -1, -1, -1, -1,  // 1: membranes 0 and 1
              0,  1,  -1, -1, -1, -1, -1,  // 2: membranes 0 and 1
              0,  -1, -1, -1, -1, -1, -1,  // 3: membrane 0 only
              1,  -1, -1, -1, -1, -1, -1,  // 4: membrane 1 only
          };
          tc.membraneData = {
              1, 2, 3, // membrane 0
              1, 2, 4, // membrane 1
          };

          // Both triangles are in z=0 plane → same normal (0,0,1).
          // Each of the 2 neighbors (1, 2) contributes 2 membrane triangles,
          // averaged to the same (0,0,1) normal.
          // Distance is 4D: displacement .z=0 but .w = 1.0−2.1 = −1.1.
          const float dw = 1.0f - 2.1f;
          const float d = std::sqrt(0.5f + dw * dw);
          const float dz = tc.r0 - d;
          tc.expectedDelta = {
              f4(0.0f, 0.0f, dz, 0.0f),
              f4(0.0f, 0.0f, 0.0f, 0.0f),
              f4(0.0f, 0.0f, 0.0f, 0.0f),
              f4(0.0f, 0.0f, 0.0f, 0.0f),
              f4(0.0f, 0.0f, 0.0f, 0.0f),
          };
          return tc;
        }(),
    };
    return kCases;
  }

  static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
    return info.param.name;
  }
};

static_assert(SiberneticTest::SibTestCommon<
              ComputeInteractionWithMembranesTestCommon>);

// ── Test cases: computeInteractionWithMembranes_finalize ────────────────────

struct ComputeInteractionWithMembranesFinalizeTestCommon {
  using Case = ComputeInteractionWithMembranesFinalizeCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = {

        // Boundary particle: position unchanged.
        [] {
          Case tc{};
          tc.name = "BoundaryParticleSkipped";
          tc.particleCount = 1;
          tc.position = {f4(1.0f, 2.0f, 3.0f, 3.0f),
                         f4(0.5f, 0.5f, 0.5f, 0.0f)};
          tc.sortedParticleIdBySerialId = {0};
          tc.expectedPosition = {f4(1.0f, 2.0f, 3.0f, 3.0f)};
          return tc;
        }(),

        // Liquid particle: position += delta.
        [] {
          Case tc{};
          tc.name = "AppliesDelta";
          tc.particleCount = 1;
          tc.position = {f4(1.0f, 2.0f, 3.0f, 1.0f),
                         f4(0.1f, 0.2f, 0.3f, 0.0f)};
          tc.sortedParticleIdBySerialId = {0};
          tc.expectedPosition = {f4(1.1f, 2.2f, 3.3f, 1.0f)};
          return tc;
        }(),

        // Zero delta: position unchanged.
        [] {
          Case tc{};
          tc.name = "ZeroDelta";
          tc.particleCount = 1;
          tc.position = {f4(5.0f, 6.0f, 7.0f, 1.0f),
                         f4(0.0f, 0.0f, 0.0f, 0.0f)};
          tc.sortedParticleIdBySerialId = {0};
          tc.expectedPosition = {f4(5.0f, 6.0f, 7.0f, 1.0f)};
          return tc;
        }(),
    };
    return kCases;
  }

  static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
    return info.param.name;
  }
};

static_assert(SiberneticTest::SibTestCommon<
              ComputeInteractionWithMembranesFinalizeTestCommon>);

} // namespace SiberneticTest
