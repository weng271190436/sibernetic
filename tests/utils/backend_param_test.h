#pragma once

#include <memory>  // IWYU pragma: keep
#include <vector>  // IWYU pragma: keep

#include <gtest/gtest.h>

#define SIB_DEFINE_BACKEND_PARAM_TEST(                                        \
    SuiteName, CaseType, RunnerBaseType, OpenCLRunnerType, MetalRunnerType,  \
    CasesFn, CaseNameFn, ExpectFn)                                            \
  class SuiteName : public ::testing::Test,                                   \
                    public ::testing::WithParamInterface<CaseType> {};         \
                                                                               \
  TEST_P(SuiteName, AllBackends) {                                             \
    const CaseType &tc = GetParam();                                           \
                                                                               \
    std::vector<std::unique_ptr<RunnerBaseType>> runners;                      \
    runners.push_back(std::make_unique<OpenCLRunnerType>());                   \
    runners.push_back(std::make_unique<MetalRunnerType>());                    \
                                                                               \
    for (auto &runner : runners) {                                             \
      ASSERT_NO_THROW({                                                        \
        auto result = runner->run(tc);                                         \
        ExpectFn(tc, result);                                                  \
      });                                                                      \
    }                                                                          \
  }                                                                            \
                                                                               \
  INSTANTIATE_TEST_SUITE_P(SuiteName##Cases, SuiteName,                        \
                           ::testing::ValuesIn(CasesFn()), CaseNameFn)
