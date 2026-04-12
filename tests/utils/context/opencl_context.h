#pragma once

#include <memory>

#include "../../../src/backend/OpenCLBackend.h"

namespace SiberneticTest {

class OpenCLKernelContext {
public:
  OpenCLKernelContext()
      : backend_(std::make_unique<Sibernetic::OpenCLBackend>()) {}

  cl::Device &device() { return backend_->device(); }
  cl::Context &context() { return backend_->context(); }
  cl::CommandQueue &queue() { return backend_->queue(); }
  cl::Program &program() { return backend_->program(); }

private:
  std::unique_ptr<Sibernetic::OpenCLBackend> backend_;
};

} // namespace SiberneticTest