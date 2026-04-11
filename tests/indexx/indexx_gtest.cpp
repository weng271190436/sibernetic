#define CL_TARGET_OPENCL_VERSION 120

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "indexx_test_common.h"
#include "metal_indexx_runner.h"
#include "opencl_indexx_runner.h"

using namespace SiberneticTest;

class IndexxBackendParamTest : public ::testing::Test,
                               public ::testing::WithParamInterface<IndexxCase> {};

TEST_P(IndexxBackendParamTest, AllBackends) {
  const IndexxCase &tc = GetParam();

  std::vector<std::unique_ptr<IndexxRunner>> runners;
  runners.push_back(std::make_unique<OpenCLIndexxRunner>());
  runners.push_back(std::make_unique<MetalIndexxRunner>());

  for (auto &runner : runners) {
    IndexxResult result;
    ASSERT_NO_THROW(result = runner->run(tc));
    expectIndexxResultMatches(tc, result);
  }
}

INSTANTIATE_TEST_SUITE_P(IndexxCases, IndexxBackendParamTest,
                         ::testing::ValuesIn(indexxCases()), indexxCaseName);