#include "OpenCLBackend.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace Sibernetic {

namespace {
std::string readFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}
} // namespace

cl::Device OpenCLBackend::pickDevice() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  if (platforms.empty()) {
    throw std::runtime_error("No OpenCL platform found");
  }

  for (const auto &platform : platforms) {
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (!devices.empty()) {
      const auto &d = devices.front();
      std::cerr << "[\x1b[32m  DEVICE  \x1b[0m] " << d.getInfo<CL_DEVICE_NAME>()
                << " (" << platform.getInfo<CL_PLATFORM_NAME>() << ")\n"
                << std::flush;
      return d;
    }
  }

  for (const auto &platform : platforms) {
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (!devices.empty()) {
      const auto &d = devices.front();
      std::cerr << "[\x1b[32m  DEVICE  \x1b[0m] " << d.getInfo<CL_DEVICE_NAME>()
                << " (" << platform.getInfo<CL_PLATFORM_NAME>()
                << ") [fallback]\n"
                << std::flush;
      return d;
    }
  }

  throw std::runtime_error("No OpenCL device found");
}

OpenCLBackend::OpenCLBackend(const char *kernelPath) {
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

  program_ = compileProgram(kernelPath);
}

cl::Program OpenCLBackend::compileProgram(const std::string &sourcePath) {
  std::string source = readFile(sourcePath);
  cl::Program::Sources sources;
  sources.push_back({source.c_str(), source.size()});

  cl_int err = CL_SUCCESS;
  cl::Program prog(context_, sources, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to create OpenCL program");
  }

  err = prog.build({device_});
  if (err != CL_SUCCESS) {
    std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_);
    throw std::runtime_error("OpenCL build failed: " + log);
  }
  return prog;
}

cl::Kernel OpenCLBackend::createKernel(const char *kernelName) {
  cl_int err = CL_SUCCESS;
  cl::Kernel kernel(program_, kernelName, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(std::string("Failed to create kernel: ") +
                             kernelName);
  }
  return kernel;
}

void OpenCLBackend::dispatch(cl::Kernel &kernel, uint32_t globalSize) {
  cl_int err = queue_.enqueueNDRangeKernel(
      kernel, cl::NullRange, cl::NDRange(globalSize), cl::NullRange);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to enqueue kernel");
  }
  queue_.finish();
}

} // namespace Sibernetic
