#pragma once

#include <functional>
#include <memory>

#include "../../../src/backend/MetalBackend.h"

namespace SiberneticTest {

class MetalKernelContext {
public:
  explicit MetalKernelContext(const char *kernelName)
      : backend_(std::make_unique<Sibernetic::MetalBackend>()),
        pipeline_(backend_->getPipeline(kernelName)) {}

  MTL::Device *device() const { return backend_->device(); }

  void dispatch(uint32_t threadCount,
                std::function<void(MTL::ComputeCommandEncoder *)> setupArgs) {
    backend_->dispatch(pipeline_, threadCount, std::move(setupArgs));
  }

private:
  std::unique_ptr<Sibernetic::MetalBackend> backend_;
  MTL::ComputePipelineState *pipeline_;
};

} // namespace SiberneticTest