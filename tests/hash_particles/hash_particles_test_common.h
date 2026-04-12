#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/kernels/HashParticlesKernel.h"
#include "../../src/types/HostTypes.h"
#include "../utils/common/backend_param_test.h"
#include "../utils/types/types.h"

namespace SiberneticTest {

using HashParticlesPosition = HostFloat4;
using HashParticlesIndexEntry = HostUInt2;

struct HashParticlesResult {
  std::vector<HashParticlesIndexEntry> particleIndex;
};

struct HashParticlesCase {
  using InputType = Sibernetic::HashParticlesInput;
  using ResultType = HashParticlesResult;

  const char *name;
  std::vector<HashParticlesPosition> positions;
  uint32_t gridCellsX;
  uint32_t gridCellsY;
  uint32_t gridCellsZ;
  float hashGridCellSizeInv;
  float xmin;
  float ymin;
  float zmin;
  std::vector<uint32_t> expectedCellIds;

  InputType toInput() const {
    return {
        .position = positions,
        .gridCellsX = gridCellsX,
        .gridCellsY = gridCellsY,
        .gridCellsZ = gridCellsZ,
        .hashGridCellSizeInv = hashGridCellSizeInv,
        .xmin = xmin,
        .ymin = ymin,
        .zmin = zmin,
        .particleCount = static_cast<uint32_t>(positions.size()),
    };
  }

  void verify(const ResultType &result) const {
    ASSERT_EQ(positions.size(), expectedCellIds.size());
    ASSERT_EQ(result.particleIndex.size(), expectedCellIds.size());

    const uint32_t particleCount =
        static_cast<uint32_t>(result.particleIndex.size());
    for (uint32_t i = 0; i < particleCount; ++i) {
      EXPECT_EQ(result.particleIndex[i][0], expectedCellIds[i]);
      EXPECT_EQ(result.particleIndex[i][1], i);
    }
  }
};

static_assert(SiberneticTest::KernelTestCase<HashParticlesCase>);

class HashParticlesRunner
    : public TestRunner<HashParticlesCase, HashParticlesResult> {};

inline HashParticlesPosition makeFloat4(float x, float y, float z,
                                        float w = 0.0f) {
  return {x, y, z, w};
}

struct HashParticlesTestCommon {
  using Case = HashParticlesCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = {
        HashParticlesCase{
            "UnitCellSize_4x4x4",
            {makeFloat4(0.1f, 0.1f, 0.1f), makeFloat4(1.2f, 0.1f, 0.1f),
             makeFloat4(0.2f, 1.7f, 0.1f), makeFloat4(2.8f, 3.1f, 1.0f)},
            4,
            4,
            4,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            {
                0, // p0 -> cell (0,0,0)
                1, // p1 -> cell (1,0,0)
                4, // p2 -> cell (0,1,0)
                30 // p3 -> cell (2,3,1)
            }},
        HashParticlesCase{
            "HalfCellSize_8x8x8",
            {makeFloat4(0.1f, 0.1f, 0.1f), makeFloat4(0.6f, 0.1f, 0.1f),
             makeFloat4(0.1f, 0.6f, 0.1f), makeFloat4(1.1f, 1.1f, 0.6f)},
            8,
            8,
            8,
            2.0f,
            0.0f,
            0.0f,
            0.0f,
            {
                0, // p0 -> cell (0,0,0)
                1, // p1 -> cell (1,0,0)
                8, // p2 -> cell (0,1,0)
                82 // p3 -> cell (2,2,1)
            }},
    };
    return kCases;
  }

  static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
    return info.param.name;
  }
};

static_assert(
    SiberneticTest::SibTestCommon<SiberneticTest::HashParticlesTestCommon>);

} // namespace SiberneticTest