#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../utils/backend_param_test.h"
#include "../utils/types.h"

namespace SiberneticTest {

using IndexxParticleIndexEntry = HostUInt2; // [cellId, serialId]

struct IndexxCase : public TestCase {
  const char *name;
  std::vector<IndexxParticleIndexEntry> particleIndex; // already sorted by cellId
  uint32_t gridCellCount;
  std::vector<uint32_t> expectedGridCellIndex; // size gridCellCount + 1
};

struct IndexxResult : public TestResult {
  std::vector<uint32_t> gridCellIndex;
};

class IndexxRunner : public TestRunner<IndexxCase, IndexxResult> {};

struct IndexxTestCommon {
  using Case = IndexxCase;
  using Result = IndexxResult;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = {
        IndexxCase{{},
                   "SparseCells_5",
                   {{0u, 10u}, {0u, 11u}, {2u, 12u}, {4u, 13u}},
                   5u,
                   {0u, UINT32_MAX, 2u, UINT32_MAX, 3u, 4u}},
        IndexxCase{{},
                   "DensePrefix_4",
                   {{0u, 2u}, {1u, 0u}, {1u, 1u}, {2u, 3u}},
                   4u,
                   {0u, 1u, 3u, UINT32_MAX, 4u}},
    };
    return kCases;
  }

  static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
    return info.param.name;
  }

  static void expect(const Case &tc, const Result &result) {
    ASSERT_EQ(result.gridCellIndex.size(), tc.expectedGridCellIndex.size());
    for (size_t i = 0; i < tc.expectedGridCellIndex.size(); ++i) {
      EXPECT_EQ(result.gridCellIndex[i], tc.expectedGridCellIndex[i])
          << "gridCellIndex[" << i << "]";
    }
  }
};

static_assert(SiberneticTest::SibTestCommon<SiberneticTest::IndexxTestCommon>);

} // namespace SiberneticTest