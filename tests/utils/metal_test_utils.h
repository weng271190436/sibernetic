#pragma once

#include <stdexcept>
#include <string>

#include "test_utils.h"

#include "../../metal-cpp/Foundation/NSSharedPtr.hpp"
#include "../../metal-cpp/Metal/MTLCommandBuffer.hpp"
#include "../../metal-cpp/Metal/MTLCommandQueue.hpp"
#include "../../metal-cpp/Metal/MTLComputeCommandEncoder.hpp"
#include "../../metal-cpp/Metal/MTLComputePipeline.hpp"
#include "../../metal-cpp/Metal/MTLDevice.hpp"
#include "../../metal-cpp/Metal/MTLLibrary.hpp"

namespace SiberneticTest {

struct MetalFloat4 {
  float s[4];
};

struct MetalUInt2 {
  uint32_t s[2];
};

class MetalKernelContext {
public:
  static constexpr const char *defaultLibrarySourcePath() {
    return "src/metal/sphFluid.metal";
  }

  explicit MetalKernelContext(const char *kernelFunctionName) {
    autoreleasePool_ = NS::AutoreleasePool::alloc()->init();
    device_ = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
    if (device_.get() == nullptr) {
      if (autoreleasePool_ != nullptr) {
        autoreleasePool_->release();
        autoreleasePool_ = nullptr;
      }
      throw std::runtime_error("No Metal device found");
    }

    library_ = compileLibraryFromSourceFile(defaultLibrarySourcePath());
    function_ = createFunction(kernelFunctionName);
    pipeline_ = createComputePipeline(function_);
    queue_ = createCommandQueue();
  }

  ~MetalKernelContext() { releaseResources(); }

private:
  void releaseResources() {
    if (autoreleasePool_ != nullptr) {
      autoreleasePool_->release();
      autoreleasePool_ = nullptr;
    }
  }

public:
  const NS::SharedPtr<MTL::Device> &device() const { return device_; }
  const NS::SharedPtr<MTL::Library> &library() const { return library_; }
  const NS::SharedPtr<MTL::Function> &function() const { return function_; }
  const NS::SharedPtr<MTL::ComputePipelineState> &pipeline() const {
    return pipeline_;
  }
  const NS::SharedPtr<MTL::CommandQueue> &queue() const { return queue_; }

private:
  NS::SharedPtr<MTL::Library>
  compileLibraryFromSourceFile(const std::string &sourcePath) const {
    // metal-cpp error reporting uses raw out-parameters.
    NS::Error *error = nullptr;
    const std::string shaderSource = readTextFile(sourcePath);
    // NS::String::string(...) returns an autoreleased object used as input only.
    NS::String *source =
        NS::String::string(shaderSource.c_str(), NS::UTF8StringEncoding);

    MTL::Library *library = device_->newLibrary(source, nullptr, &error);
    if (library == nullptr) {
      std::string msg = "Failed to compile Metal shader";
      if (error != nullptr && error->localizedDescription() != nullptr) {
        msg += ": ";
        msg += error->localizedDescription()->utf8String();
      }
      throw std::runtime_error(msg);
    }

    // newLibrary returns +1 retained ownership; transfer it into SharedPtr.
    return NS::TransferPtr(library);
  }

  NS::SharedPtr<MTL::Function> createFunction(const char *functionName) const {
    MTL::Function *function = library_->newFunction(
        NS::String::string(functionName, NS::UTF8StringEncoding));
    if (function == nullptr) {
      throw std::runtime_error(std::string("Metal function ") + functionName +
                               " not found");
    }
    // newFunction returns +1 retained ownership; transfer it into SharedPtr.
    return NS::TransferPtr(function);
  }

  NS::SharedPtr<MTL::ComputePipelineState>
  createComputePipeline(const NS::SharedPtr<MTL::Function> &function) const {
    NS::Error *error = nullptr;
    MTL::ComputePipelineState *pipeline =
        device_->newComputePipelineState(function.get(), &error);
    if (pipeline == nullptr) {
      std::string msg = "Failed to create Metal compute pipeline";
      if (error != nullptr && error->localizedDescription() != nullptr) {
        msg += ": ";
        msg += error->localizedDescription()->utf8String();
      }
      throw std::runtime_error(msg);
    }
    // newComputePipelineState returns +1 retained ownership; transfer it.
    return NS::TransferPtr(pipeline);
  }

  NS::SharedPtr<MTL::CommandQueue> createCommandQueue() const {
    MTL::CommandQueue *queue = device_->newCommandQueue();
    if (queue == nullptr) {
      throw std::runtime_error("Failed to create Metal command queue");
    }
    // newCommandQueue returns +1 retained ownership; transfer it into SharedPtr.
    return NS::TransferPtr(queue);
  }

private:
  NS::AutoreleasePool *autoreleasePool_ = nullptr;
  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::Library> library_;
  NS::SharedPtr<MTL::Function> function_;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
  NS::SharedPtr<MTL::CommandQueue> queue_;
};

} // namespace SiberneticTest
