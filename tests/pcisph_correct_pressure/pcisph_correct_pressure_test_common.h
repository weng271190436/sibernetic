#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/kernels/PcisphCorrectPressureKernel.h"
#include "../utils/common/backend_param_test.h"

namespace SiberneticTest {

struct PcisphCorrectPressureResult {
  std::vector<float> pressure; // updated pressure[0..N)
};

struct PcisphCorrectPressureCase {
  using InputType = Sibernetic::PcisphCorrectPressureInput;
  using ResultType = PcisphCorrectPressureResult;

  const char *name;

  // Input data
  std::vector<uint32_t> sortedParticleIdBySerialId; // size: N
  float restDensity;                                // reference density
  std::vector<float> pressure;                      // size: N (initial values)
  std::vector<float> rho; // size: 2*N (kernel reads [N..2N))
  float delta;            // pressure correction factor

  // Expected output
  std::vector<float> expectedPressure; // size: N

  InputType toInput() const {
    return {
        .sortedParticleIdBySerialId = sortedParticleIdBySerialId,
        .restDensity = restDensity,
        .pressure = pressure,
        .rho = rho,
        .delta = delta,
        .particleCount =
            static_cast<uint32_t>(sortedParticleIdBySerialId.size()),
    };
  }

  void verify(const ResultType &result) const {
    ASSERT_EQ(result.pressure.size(), expectedPressure.size());
    for (size_t i = 0; i < expectedPressure.size(); ++i) {
      EXPECT_NEAR(result.pressure[i], expectedPressure[i], 1e-5f)
          << "pressure mismatch at [" << i << "]";
    }
  }
};

static_assert(SiberneticTest::KernelTestCase<PcisphCorrectPressureCase>);

class PcisphCorrectPressureRunner {
public:
  virtual ~PcisphCorrectPressureRunner() = default;
  virtual PcisphCorrectPressureResult
  run(const PcisphCorrectPressureCase &tc) = 0;
};

struct PcisphCorrectPressureTestCommon {
  using Case = PcisphCorrectPressureCase;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = [] {
      std::vector<Case> cases;

      // ---- Test 1: PositiveDensityError ----
      // rho_predicted (1100) > rho0 (1000) → positive correction.
      // rho_err = 1100 - 1000 = 100, p_corr = 100 * 0.5 = 50.
      // pressure = 0 + 50 = 50.
      {
        cases.push_back({
            .name = "PositiveDensityError",
            .sortedParticleIdBySerialId = {0},
            .restDensity = 1000.0f,
            .pressure = {0.0f},
            .rho = {0.0f, 1100.0f}, // [0..N) unused, [N..2N) predicted
            .delta = 0.5f,
            .expectedPressure = {50.0f},
        });
      }

      // ---- Test 2: NegativeDensityErrorClamped ----
      // rho_predicted (900) < rho0 (1000) → negative correction clamped to 0.
      // rho_err = 900 - 1000 = -100, p_corr = -100 * 0.5 = -50 → clamped 0.
      // pressure = 0 + 0 = 0 (unchanged).
      {
        cases.push_back({
            .name = "NegativeDensityErrorClamped",
            .sortedParticleIdBySerialId = {0},
            .restDensity = 1000.0f,
            .pressure = {0.0f},
            .rho = {0.0f, 900.0f},
            .delta = 0.5f,
            .expectedPressure = {0.0f},
        });
      }

      // ---- Test 3: ZeroDensityError ----
      // rho_predicted == rho0 → zero correction.
      // rho_err = 0, p_corr = 0, pressure unchanged.
      {
        cases.push_back({
            .name = "ZeroDensityError",
            .sortedParticleIdBySerialId = {0},
            .restDensity = 1000.0f,
            .pressure = {5.0f},
            .rho = {0.0f, 1000.0f},
            .delta = 0.5f,
            .expectedPressure = {5.0f},
        });
      }

      // ---- Test 4: AccumulatesOnExistingPressure ----
      // Initial pressure = 25, positive correction = 100 * 0.5 = 50.
      // pressure = 25 + 50 = 75.
      {
        cases.push_back({
            .name = "AccumulatesOnExistingPressure",
            .sortedParticleIdBySerialId = {0},
            .restDensity = 1000.0f,
            .pressure = {25.0f},
            .rho = {0.0f, 1100.0f},
            .delta = 0.5f,
            .expectedPressure = {75.0f},
        });
      }

      // ---- Test 5: NonIdentityIndexBack ----
      // 3 particles with scrambled sortedParticleIdBySerialId: {2, 0, 1}.
      // gid 0 → id 2: rho[3+2]=rho[5]=1200, err=200, corr=200*0.5=100,
      //               pressure[2]=10+100=110
      // gid 1 → id 0: rho[3+0]=rho[3]=800, err=-200, corr=0 (clamped),
      //               pressure[0]=20+0=20
      // gid 2 → id 1: rho[3+1]=rho[4]=1050, err=50, corr=50*0.5=25,
      //               pressure[1]=30+25=55
      {
        cases.push_back({
            .name = "NonIdentityIndexBack",
            .sortedParticleIdBySerialId = {2, 0, 1},
            .restDensity = 1000.0f,
            .pressure = {20.0f, 30.0f, 10.0f},
            // rho[0..3) unused first half, rho[3..6) predicted densities
            .rho = {0.0f, 0.0f, 0.0f, 800.0f, 1050.0f, 1200.0f},
            .delta = 0.5f,
            .expectedPressure = {20.0f, 55.0f, 110.0f},
        });
      }

      // ---- Test 6: SmallPositiveError ----
      // rho_predicted barely exceeds rho0: 1000.25 vs 1000.
      // rho_err = 0.25, p_corr = 0.25 * 0.5 = 0.125.
      // Uses power-of-two fractions for exact float32 representation.
      // Verifies small corrections near the clamp boundary aren't lost.
      {
        cases.push_back({
            .name = "SmallPositiveError",
            .sortedParticleIdBySerialId = {0},
            .restDensity = 1000.0f,
            .pressure = {0.0f},
            .rho = {0.0f, 1000.25f},
            .delta = 0.5f,
            .expectedPressure = {0.125f},
        });
      }

      // ---- Test 7: LargeDelta ----
      // Same density error as Test 1 but with delta = 3.7 instead of 0.5.
      // rho_err = 100, p_corr = 100 * 3.7 = 370.
      // pressure = 0 + 370 = 370.
      {
        cases.push_back({
            .name = "LargeDelta",
            .sortedParticleIdBySerialId = {0},
            .restDensity = 1000.0f,
            .pressure = {0.0f},
            .rho = {0.0f, 1100.0f},
            .delta = 3.7f,
            .expectedPressure = {370.0f},
        });
      }

      return cases;
    }();
    return kCases;
  }

  static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
    return info.param.name;
  }
};

} // namespace SiberneticTest
