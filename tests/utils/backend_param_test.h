#pragma once

#include <concepts>
#include <memory> // IWYU pragma: keep
#include <string>
#include <vector> // IWYU pragma: keep

#include <gtest/gtest.h>

namespace SiberneticTest {

template <typename T>
concept SibTestCommon = requires {
  typename T::Case;
  typename T::Result;
  { T::cases() } -> std::same_as<const std::vector<typename T::Case> &>;
  {
    T::caseName(
        std::declval<const ::testing::TestParamInfo<typename T::Case> &>())
  } -> std::same_as<std::string>;
  {
    T::expect(std::declval<const typename T::Case &>(),
              std::declval<const typename T::Result &>())
  } -> std::same_as<void>;
};

} // namespace SiberneticTest

#define SIB_DEFINE_BACKEND_PARAM_TEST(SuiteName, TestCommon, RunnerBaseType,   \
                                      OpenCLRunnerType, MetalRunnerType)       \
  static_assert(SiberneticTest::SibTestCommon<TestCommon>,                     \
                #TestCommon " does not satisfy SibTestCommon");                \
  class SuiteName : public ::testing::Test,                                    \
                    public ::testing::WithParamInterface<TestCommon::Case> {}; \
                                                                               \
  TEST_P(SuiteName, AllBackends) {                                             \
    const TestCommon::Case &tc = GetParam();                                   \
                                                                               \
    std::vector<std::unique_ptr<RunnerBaseType>> runners;                      \
    runners.push_back(std::make_unique<OpenCLRunnerType>());                   \
    runners.push_back(std::make_unique<MetalRunnerType>());                    \
                                                                               \
    for (auto &runner : runners) {                                             \
      ASSERT_NO_THROW({                                                        \
        auto result = runner->run(tc);                                         \
        TestCommon::expect(tc, result);                                        \
      });                                                                      \
    }                                                                          \
  }                                                                            \
                                                                               \
  INSTANTIATE_TEST_SUITE_P(SuiteName##Cases, SuiteName,                        \
                           ::testing::ValuesIn(TestCommon::cases()),           \
                           TestCommon::caseName)
