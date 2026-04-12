#pragma once

#include "../../../src/types/HostTypes.h"

namespace SiberneticTest {

using Sibernetic::HostFloat2;
using Sibernetic::HostFloat4;
using Sibernetic::HostUInt2;
using Sibernetic::HostUInt4;

struct TestCase {};

struct TestResult {};

template <typename TCase, typename TResult> class TestRunner {
public:
  TestRunner() = default;
  virtual ~TestRunner() = default;
  virtual TResult run(const TCase &tc) = 0;
};

} // namespace SiberneticTest
