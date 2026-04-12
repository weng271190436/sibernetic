#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/kernels/FindNeighborsKernel.h"
#include "../../src/types/HostTypes.h"
#include "../utils/common/backend_param_test.h"
#include "../utils/types/types.h"

namespace SiberneticTest {

using FindNeighborsFloat4 = HostFloat4;

struct FindNeighborsResult {
  std::vector<std::array<float, 2>> neighborMap; // size particleCount * 32
};

struct FindNeighborsCase {
  using InputType = Sibernetic::FindNeighborsInput;
  using ResultType = FindNeighborsResult;

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

  InputType toInput() const {
    return {
        .gridCellIndexFixedUp = gridCellIndexFixedUp,
        .sortedPosition = sortedPosition,
        .gridCellCount = gridCellCount,
        .gridCellsX = gridCellsX,
        .gridCellsY = gridCellsY,
        .gridCellsZ = gridCellsZ,
        .h = h,
        .hashGridCellSize = hashGridCellSize,
        .hashGridCellSizeInv = hashGridCellSizeInv,
        .simulationScale = simulationScale,
        .xmin = xmin,
        .ymin = ymin,
        .zmin = zmin,
        .particleCount = static_cast<uint32_t>(sortedPosition.size()),
    };
  }

  void verify(const ResultType &result) const {
    const size_t particleCount = sortedPosition.size();
    ASSERT_EQ(expectedPrimaryNeighborIds.size(), particleCount);
    ASSERT_EQ(expectedPrimaryNeighborDistances.size(), particleCount);
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
      for (size_t k = 0; k < expectedPrimaryNeighborIds[p].size(); ++k) {
        expected.push_back({expectedPrimaryNeighborIds[p][k],
                            expectedPrimaryNeighborDistances[p][k]});
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

static_assert(SiberneticTest::KernelTestCase<FindNeighborsCase>);

class FindNeighborsRunner
    : public TestRunner<FindNeighborsCase, FindNeighborsResult> {};

inline FindNeighborsFloat4 fnPos(float x, float y, float z, float cellId) {
  return {x, y, z, cellId};
}

inline std::vector<uint32_t> fixedUpFromOccupiedCellCounts(
    uint32_t gridCellCount,
    const std::vector<std::pair<uint32_t, uint32_t>> &occupiedCellCounts) {
  std::vector<uint32_t> counts(gridCellCount, 0u);
  for (const auto &entry : occupiedCellCounts) {
    const uint32_t cell = entry.first;
    const uint32_t count = entry.second;
    if (cell < gridCellCount) {
      counts[cell] = count;
    }
  }

  std::vector<uint32_t> out(gridCellCount + 1u, 0u);
  for (uint32_t c = 0; c < gridCellCount; ++c) {
    out[c + 1u] = out[c] + counts[c];
  }
  return out;
}

inline std::vector<uint32_t> fixedUpSingleOccupiedCell(uint32_t gridCellCount,
                                                       uint32_t occupiedCell,
                                                       uint32_t particleCount) {
  return fixedUpFromOccupiedCellCounts(gridCellCount,
                                       {{occupiedCell, particleCount}});
}

struct FindNeighborsTestCommon {
  using Case = FindNeighborsCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = {
        FindNeighborsCase{
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
        FindNeighborsCase{
            "SameCell_IncludesDistanceNearH",
            fixedUpSingleOccupiedCell(/*gridCellCount=*/512u,
                                      /*occupiedCell=*/73u,
                                      /*particleCount=*/3u),
            {
                fnPos(1.00f, 1.00f, 1.00f, 73.0f),
                fnPos(1.249f, 1.00f, 1.00f, 73.0f),
                fnPos(1.10f, 1.00f, 1.00f, 73.0f),
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
            {{{0.249f, 0.10f}}, {{0.249f, 0.149f}}, {{0.10f, 0.149f}}},
        },
        FindNeighborsCase{
            "TwoClusters_DistantCells_NoCrossTalk",
            fixedUpFromOccupiedCellCounts(
                /*gridCellCount=*/512u,
                {
                    {73u, 3u},
                    {310u, 3u},
                }),
            {
                fnPos(1.80f, 1.80f, 1.80f, 73.0f),
                fnPos(1.90f, 1.80f, 1.80f, 73.0f),
                fnPos(1.80f, 1.90f, 1.80f, 73.0f),
                fnPos(6.80f, 6.80f, 4.80f, 310.0f),
                fnPos(6.90f, 6.80f, 4.80f, 310.0f),
                fnPos(6.80f, 6.90f, 4.80f, 310.0f),
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
            /*expectedPrimaryNeighborIds=*/
            {{{1, 2}}, {{0, 2}}, {{0, 1}}, {{4, 5}}, {{3, 5}}, {{3, 4}}},
            /*expectedPrimaryNeighborDistances=*/
            {{{0.1f, 0.1f}},
             {{0.1f, 0.14142136f}},
             {{0.1f, 0.14142136f}},
             {{0.1f, 0.1f}},
             {{0.1f, 0.14142136f}},
             {{0.1f, 0.14142136f}}},
        },
        FindNeighborsCase{
            "AdjacentCells_CrossCellNeighborsFound",
            fixedUpFromOccupiedCellCounts(
                /*gridCellCount=*/512u,
                {
                    {73u, 2u},
                    {74u, 1u},
                }),
            {
                fnPos(1.95f, 1.80f, 1.80f, 73.0f),
                fnPos(1.90f, 1.80f, 1.80f, 73.0f),
                fnPos(2.05f, 1.80f, 1.80f, 74.0f),
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
            {{{0.05f, 0.10f}}, {{0.05f, 0.15f}}, {{0.10f, 0.15f}}},
        },
    };
    return kCases;
  }

  static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
    return info.param.name;
  }
};

static_assert(
    SiberneticTest::SibTestCommon<SiberneticTest::FindNeighborsTestCommon>);

} // namespace SiberneticTest
