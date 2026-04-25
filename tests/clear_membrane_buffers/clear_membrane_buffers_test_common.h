#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/kernels/ClearMembraneBuffersKernel.h"
#include "../utils/common/backend_param_test.h"
#include "../utils/types/types.h"

namespace SiberneticTest {

struct ClearMembraneBuffersResult {
  std::vector<HostFloat4> position;
  std::vector<HostFloat4> velocity;
};

struct ClearMembraneBuffersCase {
  using InputType = Sibernetic::ClearMembraneBuffersInput;
  using ResultType = ClearMembraneBuffersResult;

  const char *name;
  std::vector<HostFloat4> position; // size: 2 × particleCount
  std::vector<HostFloat4> velocity; // size: 2 × particleCount
  uint32_t particleCount;

  InputType toInput() const {
    return {
        .position = position,
        .velocity = velocity,
        .particleCount = particleCount,
    };
  }

  void verify(const ResultType &result) const {
    const size_t N = particleCount;
    ASSERT_EQ(result.position.size(), 2 * N);
    ASSERT_EQ(result.velocity.size(), 2 * N);

    // [0..N) should be unchanged from input.
    for (size_t i = 0; i < N; ++i) {
      for (int c = 0; c < 4; ++c) {
        EXPECT_FLOAT_EQ(result.position[i][c], position[i][c])
            << "position[" << i << "][" << c << "] should be unchanged";
        EXPECT_FLOAT_EQ(result.velocity[i][c], velocity[i][c])
            << "velocity[" << i << "][" << c << "] should be unchanged";
      }
    }

    // [N..2N) should be zeroed.
    for (size_t i = N; i < 2 * N; ++i) {
      for (int c = 0; c < 4; ++c) {
        EXPECT_FLOAT_EQ(result.position[i][c], 0.0f)
            << "position[" << i << "][" << c << "] should be zeroed";
        EXPECT_FLOAT_EQ(result.velocity[i][c], 0.0f)
            << "velocity[" << i << "][" << c << "] should be zeroed";
      }
    }
  }
};

static_assert(SiberneticTest::KernelTestCase<ClearMembraneBuffersCase>);

class ClearMembraneBuffersRunner
    : public TestRunner<ClearMembraneBuffersCase, ClearMembraneBuffersResult> {
};

struct ClearMembraneBuffersTestCommon {
  using Case = ClearMembraneBuffersCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = {
        // 1 particle: position[1] and velocity[1] should be zeroed.
        [] {
          ClearMembraneBuffersCase tc{};
          tc.name = "SingleParticle";
          tc.particleCount = 1;
          tc.position = {{1.0f, 2.0f, 3.0f, 4.0f}, {5.0f, 6.0f, 7.0f, 8.0f}};
          tc.velocity = {{0.1f, 0.2f, 0.3f, 0.4f}, {0.5f, 0.6f, 0.7f, 0.8f}};
          return tc;
        }(),
        // 4 particles: [4..8) region zeroed in both buffers.
        [] {
          ClearMembraneBuffersCase tc{};
          tc.name = "MultipleParticles";
          tc.particleCount = 4;
          tc.position.resize(8);
          tc.velocity.resize(8);
          for (size_t i = 0; i < 8; ++i) {
            float v = static_cast<float>(i + 1);
            tc.position[i] = {v, v * 2, v * 3, v * 4};
            tc.velocity[i] = {v * 0.1f, v * 0.2f, v * 0.3f, v * 0.4f};
          }
          return tc;
        }(),
        // Verify [0..N) is not modified — original data preserved.
        [] {
          ClearMembraneBuffersCase tc{};
          tc.name = "PreservesOriginalData";
          tc.particleCount = 2;
          tc.position = {{10.0f, 20.0f, 30.0f, 2.1f},
                         {40.0f, 50.0f, 60.0f, 2.2f},
                         {99.0f, 88.0f, 77.0f, 66.0f},
                         {55.0f, 44.0f, 33.0f, 22.0f}};
          tc.velocity = {{1.0f, 2.0f, 3.0f, 0.0f},
                         {4.0f, 5.0f, 6.0f, 0.0f},
                         {7.0f, 8.0f, 9.0f, 1.0f},
                         {10.0f, 11.0f, 12.0f, 1.0f}};
          return tc;
        }(),
    };
    return kCases;
  }

  static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
    return info.param.name;
  }
};

static_assert(SiberneticTest::SibTestCommon<ClearMembraneBuffersTestCommon>);

} // namespace SiberneticTest
