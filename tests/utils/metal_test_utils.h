#pragma once

#include <stdexcept>
#include <string>

#include "test_utils.h"

#include "../../metal-cpp/Foundation/NSSharedPtr.hpp"
#include "../../metal-cpp/Metal/MTLCommandQueue.hpp"
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

  MetalKernelContext() {
    autoreleasePool_ = NS::AutoreleasePool::alloc()->init();
    device_ = MTL::CreateSystemDefaultDevice();
    if (device_ == nullptr) {
      if (autoreleasePool_ != nullptr) {
        autoreleasePool_->release();
        autoreleasePool_ = nullptr;
      }
      throw std::runtime_error("No Metal device found");
    }
  }

  ~MetalKernelContext() { releaseResources(); }

private:
  void releaseResources() {
    if (device_ != nullptr) {
      device_->release();
      device_ = nullptr;
    }
    if (autoreleasePool_ != nullptr) {
      autoreleasePool_->release();
      autoreleasePool_ = nullptr;
    }
  }

public:
  MTL::Device *device() const { return device_; }

  NS::SharedPtr<MTL::Library> compileDefaultLibrary() const {
    return compileLibraryFromSourceFile(defaultLibrarySourcePath());
  }

  NS::SharedPtr<MTL::Library>
  compileLibraryFromSourceFile(const std::string &sourcePath) const {
    NS::Error *error = nullptr;
    const std::string shaderSource = readTextFile(sourcePath);
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

    return NS::TransferPtr(library);
  }

  NS::SharedPtr<MTL::Function> createFunction(MTL::Library *library,
                                              const char *functionName) const {
    MTL::Function *function = library->newFunction(
        NS::String::string(functionName, NS::UTF8StringEncoding));
    if (function == nullptr) {
      throw std::runtime_error(std::string("Metal function ") + functionName +
                               " not found");
    }
    return NS::TransferPtr(function);
  }

  NS::SharedPtr<MTL::ComputePipelineState>
  createComputePipeline(MTL::Function *function) const {
    NS::Error *error = nullptr;
    MTL::ComputePipelineState *pipeline =
        device_->newComputePipelineState(function, &error);
    if (pipeline == nullptr) {
      std::string msg = "Failed to create Metal compute pipeline";
      if (error != nullptr && error->localizedDescription() != nullptr) {
        msg += ": ";
        msg += error->localizedDescription()->utf8String();
      }
      throw std::runtime_error(msg);
    }
    return NS::TransferPtr(pipeline);
  }

  NS::SharedPtr<MTL::CommandQueue> createCommandQueue() const {
    MTL::CommandQueue *queue = device_->newCommandQueue();
    if (queue == nullptr) {
      throw std::runtime_error("Failed to create Metal command queue");
    }
    return NS::TransferPtr(queue);
  }

private:
  NS::AutoreleasePool *autoreleasePool_ = nullptr;
  MTL::Device *device_ = nullptr;
};

} // namespace SiberneticTest
