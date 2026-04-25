#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/kernels/ClearBuffersKernel.h"
#include "../utils/common/backend_param_test.h"
#include "../utils/types/types.h"

namespace SiberneticTest {

struct ClearBuffersResult {
  std::vector<HostFloat2> neighborMap;
};

struct ClearBuffersCase {
  using InputType = Sibernetic::ClearBuffersInput;
  using ResultType = ClearBuffersResult;

  const char *name;
  std::vector<HostFloat2> neighborMap; // initial buffer contents
  uint32_t particleCount;
  // Optional: expected values for entries beyond particleCount *
  // kMaxNeighborCount. When non-empty, verify checks that trailing entries
  // match these values (i.e. were NOT touched by the kernel).
  std::vector<HostFloat2> expectedTrailingEntries;

  InputType toInput() const {
    return {
        .neighborMap = neighborMap,
        .particleCount = particleCount,
    };
  }

  void verify(const ResultType &result) const {
    const size_t clearedEntries =
        static_cast<size_t>(particleCount) * Sibernetic::kMaxNeighborCount;
    const size_t totalEntries = clearedEntries + expectedTrailingEntries.size();
    ASSERT_EQ(result.neighborMap.size(), totalEntries);
    for (size_t i = 0; i < clearedEntries; ++i) {
      EXPECT_FLOAT_EQ(result.neighborMap[i][0], -1.0f)
          << "neighborMap[" << i << "].x";
      EXPECT_FLOAT_EQ(result.neighborMap[i][1], -1.0f)
          << "neighborMap[" << i << "].y";
    }
    for (size_t i = 0; i < expectedTrailingEntries.size(); ++i) {
      const size_t idx = clearedEntries + i;
      EXPECT_FLOAT_EQ(result.neighborMap[idx][0], expectedTrailingEntries[i][0])
          << "trailing neighborMap[" << idx << "].x should be untouched";
      EXPECT_FLOAT_EQ(result.neighborMap[idx][1], expectedTrailingEntries[i][1])
          << "trailing neighborMap[" << idx << "].y should be untouched";
    }
  }
};

static_assert(SiberneticTest::KernelTestCase<ClearBuffersCase>);

class ClearBuffersRunner
    : public TestRunner<ClearBuffersCase, ClearBuffersResult> {};

struct ClearBuffersTestCommon {
  using Case = ClearBuffersCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = {
        // 1 particle: 32 float2 entries, all initially zero.
        [] {
          ClearBuffersCase tc{};
          tc.name = "SingleParticle";
          tc.particleCount = 1;
          tc.neighborMap.assign(static_cast<size_t>(tc.particleCount) *
                                    Sibernetic::kMaxNeighborCount,
                                HostFloat2{0.0f, 0.0f});
          return tc;
        }(),
        // 4 particles: 128 float2 entries, all initially zero.
        [] {
          ClearBuffersCase tc{};
          tc.name = "MultipleParticles";
          tc.particleCount = 4;
          tc.neighborMap.assign(static_cast<size_t>(tc.particleCount) *
                                    Sibernetic::kMaxNeighborCount,
                                HostFloat2{0.0f, 0.0f});
          return tc;
        }(),
        // 2 particles with non-zero initial data that should be cleared.
        [] {
          ClearBuffersCase tc{};
          tc.name = "PreExistingDataCleared";
          tc.particleCount = 2;
          const size_t totalEntries = static_cast<size_t>(tc.particleCount) *
                                      Sibernetic::kMaxNeighborCount;
          tc.neighborMap.resize(totalEntries);
          for (size_t i = 0; i < totalEntries; ++i) {
            tc.neighborMap[i] = {static_cast<float>(i),
                                 static_cast<float>(i) * 0.5f};
          }
          return tc;
        }(),
        // Buffer holds 4 particles' worth of data but particleCount is 2.
        // First 64 entries should be cleared; last 64 should be untouched.
        [] {
          ClearBuffersCase tc{};
          tc.name = "BoundsGuard";
          tc.particleCount = 2;
          constexpr uint32_t kBufferParticles = 4;
          constexpr size_t kTotalEntries =
              kBufferParticles * Sibernetic::kMaxNeighborCount;
          tc.neighborMap.resize(kTotalEntries);
          for (size_t i = 0; i < kTotalEntries; ++i) {
            tc.neighborMap[i] = {static_cast<float>(i + 1),
                                 static_cast<float>(i + 1) * 0.25f};
          }
          // Trailing entries (beyond particleCount * kMaxNeighborCount)
          // should remain unchanged.
          constexpr size_t kClearedEntries =
              static_cast<size_t>(2) * Sibernetic::kMaxNeighborCount;
          tc.expectedTrailingEntries.assign(
              tc.neighborMap.begin() + kClearedEntries, tc.neighborMap.end());
          return tc;
        }(),
    };
    return kCases;
  }

  static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
    return info.param.name;
  }
};

static_assert(SiberneticTest::SibTestCommon<ClearBuffersTestCommon>);

} // namespace SiberneticTest
