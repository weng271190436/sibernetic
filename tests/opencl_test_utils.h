#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../inc/OpenCL/cl.hpp"

#define SIB_ANSI_RESET "\x1b[0m"
#define SIB_ANSI_GREEN "\x1b[32m"

#define SIB_TEST_LOG_COLOR(color, label, msg_stream)                           \
  do {                                                                         \
    std::ostringstream sib_test_log_oss;                                       \
    sib_test_log_oss << msg_stream;                                            \
    std::cerr << (color) << "[ " << (label) << " ]"                            \
              << ((color)[0] != '\0' ? SIB_ANSI_RESET : "") << " "             \
              << sib_test_log_oss.str() << "\n"                                \
              << std::flush;                                                   \
  } while (0)

#define SIB_TEST_LOG(label, msg_stream)                                        \
  SIB_TEST_LOG_COLOR("", label, msg_stream)

namespace SiberneticTest {

inline std::string readTextFile(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

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
