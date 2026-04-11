#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "test_utils.h"

// macOS defines err_local as a macro in <err.h>; it collides with
// the local variable name used inside OpenCL C++ headers.
#ifdef err_local
#undef err_local
#endif

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

class OpenCLKernelContext {
public:
  static constexpr const char *defaultKernelSourcePath() {
    return "src/sphFluid.cl";
  }

  OpenCLKernelContext() {
    device_ = pickDevice();

    cl_int err = CL_SUCCESS;
    context_ = cl::Context(device_, nullptr, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create OpenCL context");
    }

    queue_ = cl::CommandQueue(context_, device_, 0, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create OpenCL command queue");
    }

    program_ = compileProgramFromSourceFile(defaultKernelSourcePath());
  }

  const cl::Device &device() const { return device_; }
  cl::Context &context() { return context_; }
  cl::CommandQueue &queue() { return queue_; }
  cl::Program &program() { return program_; }

  cl::Program
  compileProgramFromSourceFile(const std::string &sourcePath) const {
    const std::string kernelSource = readTextFile(sourcePath);
    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.size()});

    cl_int err = CL_SUCCESS;
    cl::Program program(context_, sources, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create OpenCL program");
    }

    err = program.build({device_});
    if (err != CL_SUCCESS) {
      const std::string log =
          program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_);
      throw std::runtime_error(std::string("OpenCL build failed: ") + log);
    }

    return program;
  }

private:
  cl::Device device_;
  cl::Context context_;
  cl::CommandQueue queue_;
  cl::Program program_;
};

} // namespace SiberneticTest