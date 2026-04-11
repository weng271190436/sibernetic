#pragma once

#include <array>
#include <cstdint>

namespace SiberneticTest {

using HostFloat4 = std::array<float, 4>;
using HostUInt2 = std::array<uint32_t, 2>;

struct TestCase {};

struct TestResult {};

template <typename TCase, typename TResult> class TestRunner {
public:
  TestRunner() = default;
  virtual ~TestRunner() = default;
  virtual TResult run(const TCase &tc) = 0;
};

} // namespace SiberneticTest
