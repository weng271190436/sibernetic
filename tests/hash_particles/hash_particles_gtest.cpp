#define CL_TARGET_OPENCL_VERSION 120

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "hash_particles_test_common.h"
#include "opencl_hash_particles_runner.h"
#include "metal_hash_particles_runner.h"

using namespace SiberneticTest;

class HashParticlesBackendParamTest
  : public ::testing::Test,
      public ::testing::WithParamInterface<HashParticlesCase> {};

TEST_P(HashParticlesBackendParamTest, AllBackends) {
  const HashParticlesCase &tc = GetParam();

  std::vector<std::unique_ptr<HashParticlesRunner>> runners;
  runners.push_back(std::make_unique<OpenCLHashParticlesRunner>());
  runners.push_back(std::make_unique<MetalHashParticlesRunner>());

  for (auto &runner : runners) {
    HashParticlesResult result;
    ASSERT_NO_THROW(result = runner->run(tc));
    expectHashParticlesResultMatches(tc, result);
  }
}

INSTANTIATE_TEST_SUITE_P(HashParticlesCases, HashParticlesBackendParamTest,
                         ::testing::ValuesIn(hashParticlesCases()),
                         hashParticlesCaseName);
