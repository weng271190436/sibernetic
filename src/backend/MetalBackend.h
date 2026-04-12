#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "../metal-cpp/Foundation/NSSharedPtr.hpp"
#include "../metal-cpp/Metal/MTLCommandBuffer.hpp"
#include "../metal-cpp/Metal/MTLCommandQueue.hpp"
#include "../metal-cpp/Metal/MTLComputeCommandEncoder.hpp"
#include "../metal-cpp/Metal/MTLComputePipeline.hpp"
#include "../metal-cpp/Metal/MTLDevice.hpp"
#include "../metal-cpp/Metal/MTLLibrary.hpp"

namespace Sibernetic {

class MetalBackend {
public:
  static constexpr const char *kDefaultLibraryPath =
      "src/metal/sphFluid.metal";

  explicit MetalBackend(const char *libraryPath = kDefaultLibraryPath);
  ~MetalBackend();

  MetalBackend(const MetalBackend &) = delete;
  MetalBackend &operator=(const MetalBackend &) = delete;

  MTL::Device *device() const { return device_.get(); }
  MTL::CommandQueue *queue() const { return queue_.get(); }

  MTL::ComputePipelineState *getPipeline(const char *kernelName);

  void dispatch(const char *kernelName, uint32_t threadCount,
                std::function<void(MTL::ComputeCommandEncoder *)> setupArgs);
  void dispatch(MTL::ComputePipelineState *pipeline, uint32_t threadCount,
                std::function<void(MTL::ComputeCommandEncoder *)> setupArgs);

private:
  NS::SharedPtr<MTL::Library> compileLibrary(const std::string &sourcePath);
  NS::SharedPtr<MTL::ComputePipelineState>
  createPipeline(const char *functionName);

  NS::AutoreleasePool *autoreleasePool_ = nullptr;
  NS::SharedPtr<MTL::Device> device_;
  NS::SharedPtr<MTL::Library> library_;
  NS::SharedPtr<MTL::CommandQueue> queue_;
  std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>>
      pipelines_;
};

} // namespace Sibernetic
