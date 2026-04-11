# Kernel Tests

Tests verify that GPU kernel functions produce correct output on both the OpenCL
and Metal backends. They use [GoogleTest](https://github.com/google/googletest)
and are run with `make test` from the repo root.

## Directory structure

```
tests/
  metal_private_impl.cpp      # Owns NS/MTL_PRIVATE_IMPLEMENTATION (one per binary)
  utils/
    types.h                   # Host aliases + base structs: TestCase/TestResult/TestRunner<>
    test_utils.h              # readTextFile(), logging macros
    backend_param_test.h      # SibTestCommon concept + SIB_DEFINE_BACKEND_PARAM_TEST
    opencl_context.h          # pickDevice(), OpenCLKernelContext
    opencl_helpers.h          # OpenCL buffer + conversion helpers + runOpenCL1DKernel
    metal_context.h           # MetalKernelContext (dispatch, pipeline setup)
    metal_helpers.h           # Metal buffer + conversion helpers
    metal_types.h             # MetalFloat4, MetalUInt2
  hash_particles/             # One directory per kernel under test
    hash_particles_test_common.h      # Shared: Case/Result types, test data, assertions
    opencl_hash_particles_runner.h    # OpenCL backend runner
    metal_hash_particles_runner.h     # Metal backend runner
    hash_particles_gtest.cpp          # GoogleTest entry point
  sort_post_pass/             # sortPostPass kernel tests
    ...
```

## How to add tests for a new kernel

### 1. Create a directory

```
tests/my_kernel/
```

### 2. Write `my_kernel_test_common.h`

Define the input/output contract and test data, shared by all backends.

Use a `MyKernelTestCommon` struct that satisfies `SibTestCommon`:

```cpp
#pragma once
#include <gtest/gtest.h>
#include "../utils/backend_param_test.h"
#include "../utils/types.h"

namespace SiberneticTest {

struct MyKernelCase : public TestCase {
  const char *name;
  // ... kernel inputs and expected outputs ...
};

struct MyKernelResult : public TestResult {
  // ... fields read back from the GPU ...
};

class MyKernelRunner
    : public TestRunner<MyKernelCase, MyKernelResult> {};

struct MyKernelTestCommon {
  using Case = MyKernelCase;
  using Result = MyKernelResult;

  static const std::vector<Case> &cases() {
    static const std::vector<Case> kCases = {
        // Note: first element must be {} to initialize the TestCase base
        MyKernelCase{{}, "CaseName", /* inputs */, /* expected */},
    };
    return kCases;
  }

  static std::string caseName(const ::testing::TestParamInfo<Case> &info) {
    return info.param.name;
  }

  static void expect(const Case &tc, const Result &result) {
    // EXPECT_EQ / ASSERT_EQ assertions
  }
};

static_assert(SiberneticTest::SibTestCommon<SiberneticTest::MyKernelTestCommon>);

} // namespace SiberneticTest
```

> **Aggregate-init note:** Because `MyKernelCase` inherits from `TestCase`, C++17
> aggregate initialization requires an explicit `{}` as the first element of the
> braced initializer to initialize the base-class subobject.

### 3. Write `opencl_my_kernel_runner.h`

```cpp
#pragma once
#include "../utils/opencl_context.h"
#include "../utils/opencl_helpers.h"
#include "my_kernel_test_common.h"

namespace SiberneticTest {

class OpenCLMyKernelRunner : public MyKernelRunner {
public:
  MyKernelResult run(const MyKernelCase &tc) override {
    OpenCLKernelContext opencl;
    cl_int err = CL_SUCCESS;
    cl::Kernel kernel(opencl.program(), "myKernelFunctionName", &err);

    // 1. Convert input data to CL types (or use helper converters)
    // 2. Create buffers using helpers:
    auto inputBuf  = makeOpenCLReadBuffer(opencl.context(), clInputData, err);
    auto outputBuf = makeOpenCLWriteBuffer(opencl.context(), outputBytes, err);
    // 3. Set kernel args, enqueue, finish
    // 4. Read back and fill MyKernelResult
    MyKernelResult result;
    return result;
  }
};

} // namespace SiberneticTest
```

The kernel is compiled from `src/sphFluid.cl` by default; pass a different path
to `opencl.compileProgramFromSourceFile()` if needed.

### 4. Write `metal_my_kernel_runner.h`

```cpp
#pragma once
#include "../utils/metal_context.h"
#include "../utils/metal_helpers.h"
#include "../utils/metal_types.h"
#include "my_kernel_test_common.h"

namespace SiberneticTest {

class MetalMyKernelRunner : public MyKernelRunner {
public:
  MyKernelResult run(const MyKernelCase &tc) override {
    MetalKernelContext metal("myMetalFunctionName");
    auto *dev = metal.device().get();
    const uint32_t n = ...;

    // 1. Convert input data to MetalFloat4 / MetalUInt2 (or use helper converters)
    // 2. Create buffers using helpers:
    auto inputBuf  = makeMetalInputBuffer(dev, inputVec);
    auto outputBuf = makeMetalOutputBuffer(dev, outputBytes);
    // 3. Dispatch — only setBuffer/setBytes calls needed inside the lambda:
    metal.dispatch(n, [&](MTL::ComputeCommandEncoder *enc) {
      enc->setBuffer(inputBuf.get(), 0, 0);
      enc->setBuffer(outputBuf.get(), 0, 1);
      // ...
    });
    // 4. Read back contents and fill MyKernelResult
    MyKernelResult result;
    return result;
  }
};

} // namespace SiberneticTest
```

> **`MetalKernelContext` kernel name:** Pass the Metal function name as the
> constructor argument: `MetalKernelContext metal("myMetalFunctionName")`.

### 5. Write `my_kernel_gtest.cpp`

```cpp
#define CL_TARGET_OPENCL_VERSION 120

#include "../utils/backend_param_test.h"
#include "my_kernel_test_common.h"
#include "opencl_my_kernel_runner.h"
#include "metal_my_kernel_runner.h"

using namespace SiberneticTest;

SIB_DEFINE_BACKEND_PARAM_TEST(MyKernelBackendParamTest, MyKernelTestCommon,
                              MyKernelRunner, OpenCLMyKernelRunner,
                              MetalMyKernelRunner);
```

### 6. Register the new `.cpp` in the build

In `makefile.OSX`, append the new source file to `OPENCL_TEST_SRC`:

```makefile
OPENCL_TEST_SRC := tests/hash_particles/hash_particles_gtest.cpp \
                   tests/my_kernel/my_kernel_gtest.cpp
```

The pattern rule `$(OPENCL_TEST_OBJDIR)/%.o: tests/%.cpp` picks up any `.cpp`
listed there automatically; no other changes are needed.

## Running tests

```bash
# Install GoogleTest once (macOS)
brew install googletest

# Build and run all kernel tests
make test
```

## Notes

- Tests now build with C++20 (`CXX_STANDARD ?= c++20` in `makefile.OSX`) to
  support concept-based `SibTestCommon` checks.
