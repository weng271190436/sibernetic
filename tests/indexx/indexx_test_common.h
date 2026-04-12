#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/kernels/IndexxKernel.h"
#include "../../src/types/HostTypes.h"
#include "../utils/common/backend_param_test.h"
#include "../utils/types/types.h"

namespace SiberneticTest {

using IndexxParticleIndexEntry = HostUInt2; // [cellId, serialId]

struct IndexxResult {
  std::vector<uint32_t> gridCellIndex;
};

struct IndexxCase {
  using InputType = Sibernetic::IndexxInput;
  using ResultType = IndexxResult;

  const char *name;
  std::vector<IndexxParticleIndexEntry>
      particleIndex; // already sorted by cellId
  uint32_t gridCellCount;
  std::vector<uint32_t> expectedGridCellIndex; // size gridCellCount + 1

  InputType toInput() const {
    return {
        .particleIndex = particleIndex,
        .particleCount = static_cast<uint32_t>(particleIndex.size()),
        .gridCellCount = gridCellCount,
    };
  }

  void verify(const ResultType &result) const {
    ASSERT_EQ(result.gridCellIndex.size(), expectedGridCellIndex.size());
    for (size_t i = 0; i < expectedGridCellIndex.size(); ++i) {
      EXPECT_EQ(result.gridCellIndex[i], expectedGridCellIndex[i])
          << "gridCellIndex[" << i << "]";
    }
  }
};

static_assert(SiberneticTest::KernelTestCase<IndexxCase>);

class IndexxRunner : public TestRunner<IndexxCase, IndexxResult> {};

struct IndexxTestCommon {
  using Case = IndexxCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = {
        IndexxCase{"SparseCells_5",
                   {{0u, 10u}, {0u, 11u}, {2u, 12u}, {4u, 13u}},
                   5u,
                   {0u, UINT32_MAX, 2u, UINT32_MAX, 3u, 4u}},
        IndexxCase{"DensePrefix_4",
                   {{0u, 2u}, {1u, 0u}, {1u, 1u}, {2u, 3u}},
                   4u,
                   {0u, 1u, 3u, UINT32_MAX, 4u}},
        IndexxCase{"LongDuplicateRun_Middle",
                   {{0u, 7u},
                    {1u, 8u},
                    {2u, 9u},
                    {2u, 10u},
                    {2u, 11u},
                    {2u, 12u},
                    {4u, 13u}},
                   5u,
                   {0u, 1u, 2u, UINT32_MAX, 6u, 7u}},
        IndexxCase{"LastOccupiedCell_HasRun",
                   {{0u, 1u}, {0u, 2u}, {2u, 3u}, {5u, 4u}, {5u, 5u}},
                   6u,
                   {0u, UINT32_MAX, 2u, UINT32_MAX, UINT32_MAX, 3u, 5u}},
        IndexxCase{"SingleParticle_NonZeroCell",
                   {{3u, 42u}},
                   5u,
                   {0u, UINT32_MAX, UINT32_MAX, 0u, UINT32_MAX, 1u}},
        IndexxCase{"AllParticlesInOneNonZeroCell",
                   {{2u, 0u}, {2u, 1u}, {2u, 2u}, {2u, 3u}},
                   5u,
                   {0u, UINT32_MAX, 0u, UINT32_MAX, UINT32_MAX, 4u}},
        IndexxCase{"NoMatchForMostCells_NonEmptyInput",
                   {{3u, 21u}, {3u, 22u}},
                   5u,
                   {0u, UINT32_MAX, UINT32_MAX, 0u, UINT32_MAX, 2u}},
        IndexxCase{"CellZeroForcedToZeroEvenWhenEmpty",
                   {{2u, 10u}, {2u, 11u}, {3u, 12u}},
                   4u,
                   {0u, UINT32_MAX, 0u, 2u, 3u}},
    };
    return kCases;
  }

  static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
    return info.param.name;
  }
};

static_assert(SiberneticTest::SibTestCommon<SiberneticTest::IndexxTestCommon>);

} // namespace SiberneticTest