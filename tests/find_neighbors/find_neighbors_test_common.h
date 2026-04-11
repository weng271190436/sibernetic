#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../utils/common/backend_param_test.h"
#include "../utils/types/types.h"

namespace SiberneticTest {

using FindNeighborsFloat4 = HostFloat4;

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
  std::vector<std::array<float, 2>> neighborMap; // size particleCount * 32
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

    using NeighborEntry = std::pair<int, float>; // {id, distance}
    const auto byId = [](const NeighborEntry &a, const NeighborEntry &b) {
      return a.first < b.first;
    };

    for (size_t p = 0; p < particleCount; ++p) {
      const size_t base = p * 32u;

      // Collect non-sentinel entries from the output.
      std::vector<NeighborEntry> got;
      for (size_t s = 0; s < 32u; ++s) {
        const int id = static_cast<int>(result.neighborMap[base + s][0]);
        if (id != -1) {
          got.push_back({id, result.neighborMap[base + s][1]});
        }
      }

      // Build expected entries.
      std::vector<NeighborEntry> expected;
      for (size_t k = 0; k < tc.expectedPrimaryNeighborIds[p].size(); ++k) {
        expected.push_back({tc.expectedPrimaryNeighborIds[p][k],
                            tc.expectedPrimaryNeighborDistances[p][k]});
      }

      std::sort(got.begin(), got.end(), byId);
      std::sort(expected.begin(), expected.end(), byId);

      ASSERT_EQ(got.size(), expected.size()) << "particle " << p;
      for (size_t k = 0; k < expected.size(); ++k) {
        EXPECT_EQ(got[k].first, expected[k].first)
            << "particle " << p << ", sorted slot " << k;
        EXPECT_NEAR(got[k].second, expected[k].second, 1e-5f)
            << "particle " << p << ", sorted slot " << k;
      }
    }
  }
};

static_assert(
    SiberneticTest::SibTestCommon<SiberneticTest::FindNeighborsTestCommon>);

} // namespace SiberneticTest
