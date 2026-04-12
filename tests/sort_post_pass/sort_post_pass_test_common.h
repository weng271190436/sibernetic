#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/kernels/SortPostPassKernel.h"
#include "../../src/types/HostTypes.h"
#include "../utils/common/backend_param_test.h"
#include "../utils/types/types.h"

namespace SiberneticTest {

using SortPostPassFloat4 = HostFloat4;
using SortPostPassParticleIndex = HostUInt2; // [cellId, serialId]

struct SortPostPassResult {
  std::vector<SortPostPassFloat4> sortedPosition;
  std::vector<SortPostPassFloat4> sortedVelocity;
  std::vector<uint32_t> particleIndexBack;
};

struct SortPostPassCase {
  using InputType = Sibernetic::SortPostPassInput;
  using ResultType = SortPostPassResult;

  const char *name;
  std::vector<SortPostPassParticleIndex> particleIndex; // sorted input
  std::vector<SortPostPassFloat4> position;             // original positions
  std::vector<SortPostPassFloat4> velocity;             // original velocities
  // expected outputs
  std::vector<SortPostPassFloat4> expectedSortedPosition; // xyz + cellId as .w
  std::vector<SortPostPassFloat4> expectedSortedVelocity;
  std::vector<uint32_t> expectedParticleIndexBack; // originalId -> sortedIndex

  InputType toInput() const {
    return {
        .particleIndex = particleIndex,
        .position = position,
        .velocity = velocity,
        .particleCount = static_cast<uint32_t>(particleIndex.size()),
    };
  }

  void verify(const ResultType &result) const {
    const uint32_t n = static_cast<uint32_t>(particleIndex.size());
    ASSERT_EQ(result.sortedPosition.size(), n);
    ASSERT_EQ(result.sortedVelocity.size(), n);
    ASSERT_EQ(result.particleIndexBack.size(), n);

    for (uint32_t i = 0; i < n; ++i) {
      EXPECT_EQ(result.sortedPosition[i][0], expectedSortedPosition[i][0])
          << "sortedPosition[" << i << "].x";
      EXPECT_EQ(result.sortedPosition[i][1], expectedSortedPosition[i][1])
          << "sortedPosition[" << i << "].y";
      EXPECT_EQ(result.sortedPosition[i][2], expectedSortedPosition[i][2])
          << "sortedPosition[" << i << "].z";
      EXPECT_EQ(result.sortedPosition[i][3], expectedSortedPosition[i][3])
          << "sortedPosition[" << i << "].w (cellId)";

      EXPECT_EQ(result.sortedVelocity[i][0], expectedSortedVelocity[i][0])
          << "sortedVelocity[" << i << "].x";
      EXPECT_EQ(result.sortedVelocity[i][1], expectedSortedVelocity[i][1])
          << "sortedVelocity[" << i << "].y";
      EXPECT_EQ(result.sortedVelocity[i][2], expectedSortedVelocity[i][2])
          << "sortedVelocity[" << i << "].z";
      EXPECT_EQ(result.sortedVelocity[i][3], expectedSortedVelocity[i][3])
          << "sortedVelocity[" << i << "].w";

      EXPECT_EQ(result.particleIndexBack[i], expectedParticleIndexBack[i])
          << "particleIndexBack[" << i << "]";
    }
  }
};

static_assert(SiberneticTest::KernelTestCase<SortPostPassCase>);

class SortPostPassRunner
    : public TestRunner<SortPostPassCase, SortPostPassResult> {};

struct SortPostPassTestCommon {
  using Case = SortPostPassCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = {
        // Particles already in sorted order: particleIndex[i] = {i, i}
        SortPostPassCase{
            "Identity_3P",
            /*particleIndex=*/{{0u, 0u}, {1u, 1u}, {2u, 2u}},
            /*position=*/
            {{1.0f, 2.0f, 3.0f, 9.0f},
             {4.0f, 5.0f, 6.0f, 9.0f},
             {7.0f, 8.0f, 9.0f, 9.0f}},
            /*velocity=*/
            {{10.0f, 20.0f, 30.0f, 0.0f},
             {40.0f, 50.0f, 60.0f, 0.0f},
             {70.0f, 80.0f, 90.0f, 0.0f}},
            /*expectedSortedPosition=*/
            {{1.0f, 2.0f, 3.0f, 0.0f}, // sorted[0] = position[0], .w = cellId 0
             {4.0f, 5.0f, 6.0f, 1.0f}, // sorted[1] = position[1], .w = cellId 1
             {7.0f, 8.0f, 9.0f, 2.0f}},
            /*expectedSortedVelocity=*/
            {{10.0f, 20.0f, 30.0f, 0.0f},
             {40.0f, 50.0f, 60.0f, 0.0f},
             {70.0f, 80.0f, 90.0f, 0.0f}},
            /*expectedParticleIndexBack=*/{0u, 1u, 2u},
        },
        // Reverse order: sorted[0] = original 2, sorted[1] = original 1, ...
        SortPostPassCase{
            "Reversed_3P",
            /*particleIndex=*/{{0u, 2u}, {1u, 1u}, {2u, 0u}},
            /*position=*/
            {{1.0f, 2.0f, 3.0f, 9.0f},
             {4.0f, 5.0f, 6.0f, 9.0f},
             {7.0f, 8.0f, 9.0f, 9.0f}},
            /*velocity=*/
            {{10.0f, 20.0f, 30.0f, 0.0f},
             {40.0f, 50.0f, 60.0f, 0.0f},
             {70.0f, 80.0f, 90.0f, 0.0f}},
            /*expectedSortedPosition=*/
            {{7.0f, 8.0f, 9.0f, 0.0f}, // sorted[0] = position[2], .w = cellId 0
             {4.0f, 5.0f, 6.0f, 1.0f}, // sorted[1] = position[1], .w = cellId 1
             {1.0f, 2.0f, 3.0f, 2.0f}},
            /*expectedSortedVelocity=*/
            {{70.0f, 80.0f, 90.0f, 0.0f}, // velocity[2]
             {40.0f, 50.0f, 60.0f, 0.0f}, // velocity[1]
             {10.0f, 20.0f, 30.0f, 0.0f}},
            /*expectedParticleIndexBack=*/{2u, 1u, 0u},
        },
    };
    return kCases;
  }

  static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
    return info.param.name;
  }
};

static_assert(
    SiberneticTest::SibTestCommon<SiberneticTest::SortPostPassTestCommon>);

} // namespace SiberneticTest
