#pragma once

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>

#include "../common/test_utils.h"

#include "../../../metal-cpp/Foundation/NSSharedPtr.hpp"
#include "../../../metal-cpp/Metal/MTLCommandBuffer.hpp"
#include "../../../metal-cpp/Metal/MTLCommandQueue.hpp"
#include "../../../metal-cpp/Metal/MTLComputeCommandEncoder.hpp"
#include "../../../metal-cpp/Metal/MTLComputePipeline.hpp"
#include "../../../metal-cpp/Metal/MTLDevice.hpp"
#include "../../../metal-cpp/Metal/MTLLibrary.hpp"
#include "../../../metal-cpp/Metal/MTLTypes.hpp"

namespace SiberneticTest {

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

  void dispatch(uint32_t threadCount,
                std::function<void(MTL::ComputeCommandEncoder *)> setupArgs) {
    MTL::CommandBuffer *commandBufferRaw = queue_->commandBuffer();
    if (!commandBufferRaw) {
      throw std::runtime_error("Failed to create Metal command buffer");
    }
    NS::SharedPtr<MTL::CommandBuffer> commandBuffer =
        NS::RetainPtr(commandBufferRaw);

    MTL::ComputeCommandEncoder *encoderRaw =
        commandBuffer->computeCommandEncoder();
    if (!encoderRaw) {
      throw std::runtime_error("Failed to create Metal compute encoder");
    }
    NS::SharedPtr<MTL::ComputeCommandEncoder> encoder =
        NS::RetainPtr(encoderRaw);

    encoder->setComputePipelineState(pipeline_.get());
    setupArgs(encoder.get());

    const NS::UInteger threads = static_cast<NS::UInteger>(threadCount);
    const NS::UInteger tgWidth = std::max<NS::UInteger>(
        1, std::min<NS::UInteger>(threads,
                                  pipeline_->maxTotalThreadsPerThreadgroup()));
    encoder->dispatchThreads(MTL::Size::Make(threads, 1, 1),
                             MTL::Size::Make(tgWidth, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    if (commandBuffer->status() != MTL::CommandBufferStatusCompleted) {
      throw std::runtime_error(
          "Metal command buffer did not complete successfully");
    }
  }

private:
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

  NS::SharedPtr<MTL::Function> createFunction(const char *functionName) const {
    MTL::Function *function = library_->newFunction(
        NS::String::string(functionName, NS::UTF8StringEncoding));
    if (function == nullptr) {
      throw std::runtime_error(std::string("Metal function ") + functionName +
                               " not found");
    }
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
  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::Library> library_;
  NS::SharedPtr<MTL::Function> function_;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline_;
  NS::SharedPtr<MTL::CommandQueue> queue_;
};

} // namespace SiberneticTest