#define CL_TARGET_OPENCL_VERSION 120

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "sort_post_pass_test_common.h"
#include "opencl_sort_post_pass_runner.h"
#include "metal_sort_post_pass_runner.h"

using namespace SiberneticTest;

class SortPostPassBackendParamTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<SortPostPassCase> {};

TEST_P(SortPostPassBackendParamTest, AllBackends) {
  const SortPostPassCase &tc = GetParam();

  std::vector<std::unique_ptr<SortPostPassRunner>> runners;
  runners.push_back(std::make_unique<OpenCLSortPostPassRunner>());
  runners.push_back(std::make_unique<MetalSortPostPassRunner>());

  for (auto &runner : runners) {
    SortPostPassResult result;
    ASSERT_NO_THROW(result = runner->run(tc));
    expectSortPostPassResultMatches(tc, result);
  }
}

INSTANTIATE_TEST_SUITE_P(SortPostPassCases, SortPostPassBackendParamTest,
                         ::testing::ValuesIn(sortPostPassCases()),
                         sortPostPassCaseName);
