#pragma once

#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "test_utils.h"

#include "../../inc/OpenCL/cl.hpp"

namespace SiberneticTest {

inline cl::Device pickDevice() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.empty()) {
    throw std::runtime_error("No OpenCL platform found");
  }

  auto pickFirstDeviceOfType = [&platforms](cl_device_type deviceType,
                                            const char *suffix,
                                            cl::Device &outDevice) -> bool {
    for (const auto &platform : platforms) {
      std::vector<cl::Device> devices;
      platform.getDevices(deviceType, &devices);
      if (!devices.empty()) {
        const auto &d = devices.front();
        SIB_TEST_LOG_COLOR(SIB_ANSI_GREEN, " DEVICE ",
                           d.getInfo<CL_DEVICE_NAME>()
                               << " (" << platform.getInfo<CL_PLATFORM_NAME>()
                               << ")" << suffix);
        outDevice = d;
        return true;
      }
    }
    return false;
  };

  cl::Device selected;
  if (pickFirstDeviceOfType(CL_DEVICE_TYPE_GPU, "", selected)) {
    return selected;
  }

  if (pickFirstDeviceOfType(CL_DEVICE_TYPE_ALL, " [fallback]", selected)) {
    return selected;
  }

  throw std::runtime_error("No OpenCL device found");
}

class OpenCLKernelFixture : public ::testing::Test {
protected:
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;
  cl::Program program;

  cl::Kernel createKernel(const char *kernelName) const {
    cl_int err = CL_SUCCESS;
    cl::Kernel kernel(program, kernelName, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error(std::string("Failed to create kernel: ") +
                               kernelName);
    }
    return kernel;
  }

  virtual const char *kernelSourcePath() const { return "src/sphFluid.cl"; }

  void SetUp() override {
    ASSERT_NO_THROW(device = pickDevice());

    cl_int err = CL_SUCCESS;
    context = cl::Context(device, nullptr, nullptr, nullptr, &err);
    ASSERT_EQ(err, CL_SUCCESS);

    queue = cl::CommandQueue(context, device, 0, &err);
    ASSERT_EQ(err, CL_SUCCESS);

    const std::string kernelSource = readTextFile(kernelSourcePath());
    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.size()});

    program = cl::Program(context, sources, &err);
    ASSERT_EQ(err, CL_SUCCESS);

    err = program.build({device});
    if (err != CL_SUCCESS) {
      const std::string log =
          program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
      FAIL() << "OpenCL build failed: " << log;
    }
  }
};

} // namespace SiberneticTest