#pragma once

namespace SiberneticTest {

struct TestCase {};

struct TestResult {};

template <typename TCase, typename TResult> class TestRunner {
public:
  TestRunner() = default;
  virtual ~TestRunner() = default;
  virtual TResult run(const TCase &tc) = 0;
};

} // namespace SiberneticTest
