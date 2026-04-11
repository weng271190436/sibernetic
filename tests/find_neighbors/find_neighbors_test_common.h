#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../utils/common/backend_param_test.h"
#include "../utils/types/types.h"

namespace SiberneticTest {

using FindNeighborsFloat4 = HostFloat4;
using FindNeighborsEntry = std::array<float, 2>; // [neighborId, distance]

struct FindNeighborsCase : public TestCase {
  const char *name;
  std::vector<uint32_t> gridCellIndexFixedUp; // size gridCellCount + 1
  std::vector<FindNeighborsFloat4> sortedPosition;
  uint32_t gridCellCount;
  uint32_t gridCellsX;
  uint32_t gridCellsY;
  uint32_t gridCellsZ;
  float h;
  float hashGridCellSize;
  float hashGridCellSizeInv;
  float simulationScale;
  float xmin;
  float ymin;
  float zmin;
  std::vector<std::array<int32_t, 2>> expectedPrimaryNeighborIds;
  std::vector<std::array<float, 2>> expectedPrimaryNeighborDistances;
};

struct FindNeighborsResult : public TestResult {
  std::vector<FindNeighborsEntry> neighborMap; // size particleCount * 32
};

class FindNeighborsRunner
    : public TestRunner<FindNeighborsCase, FindNeighborsResult> {};

inline FindNeighborsFloat4 fnPos(float x, float y, float z, float cellId) {
  return {x, y, z, cellId};
}

inline std::vector<uint32_t> fixedUpSingleOccupiedCell(uint32_t gridCellCount,
                                                       uint32_t occupiedCell,
                                                       uint32_t particleCount) {
  std::vector<uint32_t> out(gridCellCount + 1u, particleCount);
  for (uint32_t c = 0; c <= occupiedCell && c < gridCellCount; ++c) {
    out[c] = 0u;
  }
  out[gridCellCount] = particleCount;
  return out;
}

struct FindNeighborsTestCommon {
  using Case = FindNeighborsCase;
  using Result = FindNeighborsResult;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = {
        FindNeighborsCase{
            {},
            "SameCell_TwoNeighborsEach",
            fixedUpSingleOccupiedCell(/*gridCellCount=*/512u,
                                      /*occupiedCell=*/73u,
                                      /*particleCount=*/3u),
            {
                fnPos(1.80f, 1.80f, 1.80f, 73.0f),
                fnPos(1.90f, 1.80f, 1.80f, 73.0f),
                fnPos(1.80f, 1.90f, 1.80f, 73.0f),
            },
            /*gridCellCount=*/512u,
            /*gridCellsX=*/8u,
            /*gridCellsY=*/8u,
            /*gridCellsZ=*/8u,
            /*h=*/0.25f,
            /*hashGridCellSize=*/1.0f,
            /*hashGridCellSizeInv=*/1.0f,
            /*simulationScale=*/1.0f,
            /*xmin=*/0.0f,
            /*ymin=*/0.0f,
            /*zmin=*/0.0f,
            /*expectedPrimaryNeighborIds=*/{{{1, 2}}, {{0, 2}}, {{0, 1}}},
            /*expectedPrimaryNeighborDistances=*/
            {{{0.1f, 0.1f}}, {{0.1f, 0.14142136f}}, {{0.1f, 0.14142136f}}},
        },
    };
    return kCases;
  }

  static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
    return info.param.name;
  }

  static void expect(const Case &tc, const Result &result) {
    const size_t particleCount = tc.sortedPosition.size();
    ASSERT_EQ(tc.expectedPrimaryNeighborIds.size(), particleCount);
    ASSERT_EQ(tc.expectedPrimaryNeighborDistances.size(), particleCount);
    ASSERT_EQ(result.neighborMap.size(), particleCount * 32u);

    for (size_t p = 0; p < particleCount; ++p) {
      const size_t base = p * 32u;
      for (size_t k = 0; k < 2u; ++k) {
        const int gotId = static_cast<int>(result.neighborMap[base + k][0]);
        const int expId = tc.expectedPrimaryNeighborIds[p][k];
        EXPECT_EQ(gotId, expId) << "particle " << p << ", slot " << k;

        const float gotDist = result.neighborMap[base + k][1];
        const float expDist = tc.expectedPrimaryNeighborDistances[p][k];
        EXPECT_NEAR(gotDist, expDist, 1e-5f)
            << "particle " << p << ", slot " << k;
      }

      for (size_t k = 2u; k < 32u; ++k) {
        const int gotId = static_cast<int>(result.neighborMap[base + k][0]);
        EXPECT_EQ(gotId, -1) << "particle " << p << ", slot " << k;
        EXPECT_EQ(result.neighborMap[base + k][1], -1.0f)
            << "particle " << p << ", slot " << k;
      }
    }
  }
};

static_assert(
    SiberneticTest::SibTestCommon<SiberneticTest::FindNeighborsTestCommon>);

} // namespace SiberneticTest
