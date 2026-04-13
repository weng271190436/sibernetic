#pragma once

#include <string>

#ifdef err_local
#undef err_local
#endif
#include <OpenCL/cl.hpp>

namespace Sibernetic {

class OpenCLBackend {
public:
  static constexpr const char *kDefaultKernelPath = "src/sphFluid.cl";

  explicit OpenCLBackend(const char *kernelPath = kDefaultKernelPath);

  OpenCLBackend(const OpenCLBackend &) = delete;
  OpenCLBackend &operator=(const OpenCLBackend &) = delete;

  cl::Device &device() { return device_; }
  cl::Context &context() { return context_; }
  cl::CommandQueue &queue() { return queue_; }
  cl::Program &program() { return program_; }

  cl::Kernel createKernel(const char *kernelName);
  void dispatch(cl::Kernel &kernel, uint32_t globalSize);

private:
  static cl::Device pickDevice();
  cl::Program compileProgram(const std::string &sourcePath);

  cl::Device device_;
  cl::Context context_;
  cl::CommandQueue queue_;
  cl::Program program_;
};

} // namespace Sibernetic
