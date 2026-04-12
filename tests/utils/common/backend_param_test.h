#pragma once

#include <concepts>
#include <memory> // IWYU pragma: keep
#include <ostream>
#include <string>
#include <vector> // IWYU pragma: keep

#include <gtest/gtest.h>

namespace SiberneticTest {

// Every test case must provide toInput() and verify().
template <typename T>
concept KernelTestCase = requires(const T &tc) {
  { tc.name } -> std::convertible_to<const char *>;
  typename T::InputType;
  typename T::ResultType;
  { tc.toInput() } -> std::same_as<typename T::InputType>;
  { tc.verify(std::declval<const typename T::ResultType &>()) } ->
      std::same_as<void>;
};

template <typename T>
concept SibTestCommon = requires {
  typename T::Case;
  requires KernelTestCase<typename T::Case>;
  { T::cases() } -> std::same_as<const std::vector<typename T::Case> &>;
  {
    T::caseName(
        std::declval<const ::testing::TestParamInfo<typename T::Case> &>())
  } -> std::same_as<std::string>;
};

// GTest pretty-printer for parameterized test case structs that expose a
// human-readable `name` field.  GTest discovers this function via ADL
// (argument-dependent lookup) when formatting test parameter values in
// assertion failure messages — it is never called explicitly in test code.
template <typename T>
  requires requires(const T &tc) {
    { tc.name } -> std::convertible_to<const char *>;
  }
inline void PrintTo(const T &tc, std::ostream *os) {
  *os << ((tc.name != nullptr) ? tc.name : "<unnamed>");
}

} // namespace SiberneticTest

#define SIB_DEFINE_BACKEND_PARAM_TEST(SuiteName, TestCommon, RunnerBaseType,   \
                                      OpenCLRunnerType, MetalRunnerType)       \
  static_assert(SiberneticTest::SibTestCommon<TestCommon>,                     \
                #TestCommon " does not satisfy SibTestCommon");                \
  class SuiteName : public ::testing::Test,                                    \
                    public ::testing::WithParamInterface<TestCommon::Case> {}; \
                                                                               \
  TEST_P(SuiteName, AllBackends) {                                             \
    const auto &tc = GetParam();                                               \
                                                                               \
    std::vector<std::pair<const char *, std::unique_ptr<RunnerBaseType>>>      \
        runners;                                                               \
    runners.emplace_back(#OpenCLRunnerType,                                    \
                         std::make_unique<OpenCLRunnerType>());                \
    runners.emplace_back(#MetalRunnerType,                                     \
                         std::make_unique<MetalRunnerType>());                 \
                                                                               \
    for (auto &runnerEntry : runners) {                                        \
      SCOPED_TRACE(std::string("backend=") + runnerEntry.first);               \
      ASSERT_NO_THROW({                                                        \
        auto result = runnerEntry.second->run(tc);                             \
        tc.verify(result);                                                     \
      });                                                                      \
    }                                                                          \
  }                                                                            \
                                                                               \
  INSTANTIATE_TEST_SUITE_P(SuiteName##Cases, SuiteName,                        \
                           ::testing::ValuesIn(TestCommon::cases()),           \
                           TestCommon::caseName)
