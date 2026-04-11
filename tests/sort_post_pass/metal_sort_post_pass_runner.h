#pragma once

#include <stdexcept>
#include <vector>

#include "../utils/metal_test_utils.h"
#include "sort_post_pass_test_common.h"

#include "../../metal-cpp/Foundation/NSSharedPtr.hpp"
#include "../../metal-cpp/Metal/MTLBuffer.hpp"
#include "../../metal-cpp/Metal/MTLCommandBuffer.hpp"
#include "../../metal-cpp/Metal/MTLCommandQueue.hpp"
#include "../../metal-cpp/Metal/MTLComputeCommandEncoder.hpp"
#include "../../metal-cpp/Metal/MTLComputePipeline.hpp"
#include "../../metal-cpp/Metal/MTLDevice.hpp"
#include "../../metal-cpp/Metal/MTLTypes.hpp"

namespace SiberneticTest {

class MetalSortPostPassRunner : public SortPostPassRunner {
public:
  SortPostPassResult run(const SortPostPassCase &tc) override {
    MetalKernelContext metal("sortPostPassMetal");
    const NS::SharedPtr<MTL::Device> &device = metal.device();
    const NS::SharedPtr<MTL::ComputePipelineState> &pipeline = metal.pipeline();
    const NS::SharedPtr<MTL::CommandQueue> &queue = metal.queue();

    const size_t n = tc.particleIndex.size();

    std::vector<MetalUInt2> particleIndex(n);
    std::vector<MetalFloat4> position(n);
    std::vector<MetalFloat4> velocity(n);
    for (size_t i = 0; i < n; ++i) {
      particleIndex[i].s[0] = tc.particleIndex[i][0];
      particleIndex[i].s[1] = tc.particleIndex[i][1];
      position[i].s[0] = tc.position[i][0];
      position[i].s[1] = tc.position[i][1];
      position[i].s[2] = tc.position[i][2];
      position[i].s[3] = tc.position[i][3];
      velocity[i].s[0] = tc.velocity[i][0];
      velocity[i].s[1] = tc.velocity[i][1];
      velocity[i].s[2] = tc.velocity[i][2];
      velocity[i].s[3] = tc.velocity[i][3];
    }

    NS::SharedPtr<MTL::Buffer> particleIndexBuf = NS::TransferPtr(
        device->newBuffer(particleIndex.data(), sizeof(MetalUInt2) * n,
                          MTL::ResourceStorageModeShared));
    if (!particleIndexBuf.get()) {
      throw std::runtime_error("Failed to create Metal particleIndex buffer");
    }

    NS::SharedPtr<MTL::Buffer> particleIndexBackBuf = NS::TransferPtr(
        device->newBuffer(sizeof(uint32_t) * n, MTL::ResourceStorageModeShared));
    if (!particleIndexBackBuf.get()) {
      throw std::runtime_error(
          "Failed to create Metal particleIndexBack buffer");
    }

    NS::SharedPtr<MTL::Buffer> positionBuf = NS::TransferPtr(
        device->newBuffer(position.data(), sizeof(MetalFloat4) * n,
                          MTL::ResourceStorageModeShared));
    if (!positionBuf.get()) {
      throw std::runtime_error("Failed to create Metal position buffer");
    }

    NS::SharedPtr<MTL::Buffer> velocityBuf = NS::TransferPtr(
        device->newBuffer(velocity.data(), sizeof(MetalFloat4) * n,
                          MTL::ResourceStorageModeShared));
    if (!velocityBuf.get()) {
      throw std::runtime_error("Failed to create Metal velocity buffer");
    }

    NS::SharedPtr<MTL::Buffer> sortedPositionBuf = NS::TransferPtr(
        device->newBuffer(sizeof(MetalFloat4) * n,
                          MTL::ResourceStorageModeShared));
    if (!sortedPositionBuf.get()) {
      throw std::runtime_error(
          "Failed to create Metal sortedPosition buffer");
    }

    NS::SharedPtr<MTL::Buffer> sortedVelocityBuf = NS::TransferPtr(
        device->newBuffer(sizeof(MetalFloat4) * n,
                          MTL::ResourceStorageModeShared));
    if (!sortedVelocityBuf.get()) {
      throw std::runtime_error(
          "Failed to create Metal sortedVelocity buffer");
    }

    const uint32_t particleCount = static_cast<uint32_t>(n);

    MTL::CommandBuffer *commandBufferRaw = queue->commandBuffer();
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

    encoder->setComputePipelineState(pipeline.get());
    encoder->setBuffer(particleIndexBuf.get(), 0, 0);
    encoder->setBuffer(particleIndexBackBuf.get(), 0, 1);
    encoder->setBuffer(positionBuf.get(), 0, 2);
    encoder->setBuffer(velocityBuf.get(), 0, 3);
    encoder->setBuffer(sortedPositionBuf.get(), 0, 4);
    encoder->setBuffer(sortedVelocityBuf.get(), 0, 5);
    encoder->setBytes(&particleCount, sizeof(particleCount), 6);

    const NS::UInteger threads = static_cast<NS::UInteger>(particleCount);
    const NS::UInteger tgWidth = std::max<NS::UInteger>(
        1, std::min<NS::UInteger>(threads,
                                  pipeline->maxTotalThreadsPerThreadgroup()));
    encoder->dispatchThreads(MTL::Size::Make(threads, 1, 1),
                             MTL::Size::Make(tgWidth, 1, 1));
    encoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    if (commandBuffer->status() != MTL::CommandBufferStatusCompleted) {
      throw std::runtime_error(
          "Metal command buffer did not complete successfully");
    }

    SortPostPassResult result;
    result.sortedPosition.resize(n);
    result.sortedVelocity.resize(n);
    result.particleIndexBack.resize(n);

    const auto *outPos =
        reinterpret_cast<const MetalFloat4 *>(sortedPositionBuf->contents());
    const auto *outVel =
        reinterpret_cast<const MetalFloat4 *>(sortedVelocityBuf->contents());
    const auto *outBack =
        reinterpret_cast<const uint32_t *>(particleIndexBackBuf->contents());

    for (size_t i = 0; i < n; ++i) {
      result.sortedPosition[i] = {outPos[i].s[0], outPos[i].s[1],
                                   outPos[i].s[2], outPos[i].s[3]};
      result.sortedVelocity[i] = {outVel[i].s[0], outVel[i].s[1],
                                   outVel[i].s[2], outVel[i].s[3]};
      result.particleIndexBack[i] = outBack[i];
    }

    return result;
  }
};

} // namespace SiberneticTest
