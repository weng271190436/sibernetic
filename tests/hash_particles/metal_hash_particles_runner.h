#pragma once

#include <stdexcept>
#include <vector>

#include "../utils/metal_test_utils.h"
#include "hash_particles_test_common.h"

#include "../../metal-cpp/Foundation/NSSharedPtr.hpp"
#include "../../metal-cpp/Metal/MTLBuffer.hpp"
#include "../../metal-cpp/Metal/MTLCommandBuffer.hpp"
#include "../../metal-cpp/Metal/MTLCommandQueue.hpp"
#include "../../metal-cpp/Metal/MTLComputeCommandEncoder.hpp"
#include "../../metal-cpp/Metal/MTLComputePipeline.hpp"
#include "../../metal-cpp/Metal/MTLDevice.hpp"
#include "../../metal-cpp/Metal/MTLTypes.hpp"

namespace SiberneticTest {

class MetalHashParticlesRunner : public HashParticlesRunner {
public:
  HashParticlesResult run(const HashParticlesCase &tc) override {
    MetalKernelContext metal("hashParticlesMetal");
    const NS::SharedPtr<MTL::Device> &device = metal.device();
    const NS::SharedPtr<MTL::ComputePipelineState> &pipeline = metal.pipeline();
    const NS::SharedPtr<MTL::CommandQueue> &queue = metal.queue();

    std::vector<MetalFloat4> positions(tc.positions.size());
    for (size_t i = 0; i < tc.positions.size(); ++i) {
      positions[i].s[0] = tc.positions[i][0];
      positions[i].s[1] = tc.positions[i][1];
      positions[i].s[2] = tc.positions[i][2];
      positions[i].s[3] = tc.positions[i][3];
    }

    HashParticlesResult result;
    result.particleIndex.resize(positions.size());

    NS::SharedPtr<MTL::Buffer> positionBuffer = NS::TransferPtr(
      device->newBuffer(positions.data(),
                sizeof(MetalFloat4) * positions.size(),
                MTL::ResourceStorageModeShared));
    if (positionBuffer.get() == nullptr) {
      throw std::runtime_error("Failed to create Metal position buffer");
    }

    NS::SharedPtr<MTL::Buffer> particleIndexBuffer = NS::TransferPtr(
      device->newBuffer(sizeof(MetalUInt2) * result.particleIndex.size(),
                MTL::ResourceStorageModeShared));
    if (particleIndexBuffer.get() == nullptr) {
      throw std::runtime_error("Failed to create Metal particleIndex buffer");
    }

    const uint32_t gridCellsX = tc.gridCellsX;
    const uint32_t gridCellsY = tc.gridCellsY;
    const uint32_t gridCellsZ = tc.gridCellsZ;
    const float hashGridCellSizeInv = tc.hashGridCellSizeInv;
    const float xmin = tc.xmin;
    const float ymin = tc.ymin;
    const float zmin = tc.zmin;
    const uint32_t particleCount = static_cast<uint32_t>(positions.size());

    MTL::CommandBuffer *commandBufferRaw = queue->commandBuffer();
    if (commandBufferRaw == nullptr) {
      throw std::runtime_error("Failed to create Metal command buffer");
    }
    NS::SharedPtr<MTL::CommandBuffer> commandBuffer =
      NS::RetainPtr(commandBufferRaw);

    MTL::ComputeCommandEncoder *encoderRaw =
      commandBuffer->computeCommandEncoder();
    if (encoderRaw == nullptr) {
      throw std::runtime_error("Failed to create Metal compute encoder");
    }
    NS::SharedPtr<MTL::ComputeCommandEncoder> encoder =
      NS::RetainPtr(encoderRaw);
    encoder->setComputePipelineState(pipeline.get());
    encoder->setBuffer(positionBuffer.get(), 0, 0);
    encoder->setBytes(&gridCellsX, sizeof(gridCellsX), 1);
    encoder->setBytes(&gridCellsY, sizeof(gridCellsY), 2);
    encoder->setBytes(&gridCellsZ, sizeof(gridCellsZ), 3);
    encoder->setBytes(&hashGridCellSizeInv, sizeof(hashGridCellSizeInv), 4);
    encoder->setBytes(&xmin, sizeof(xmin), 5);
    encoder->setBytes(&ymin, sizeof(ymin), 6);
    encoder->setBytes(&zmin, sizeof(zmin), 7);
    encoder->setBuffer(particleIndexBuffer.get(), 0, 8);
    encoder->setBytes(&particleCount, sizeof(particleCount), 9);

    const NS::UInteger threads = static_cast<NS::UInteger>(particleCount);
    const NS::UInteger tgWidth =
        std::max<NS::UInteger>(
            1,
            std::min<NS::UInteger>(threads,
                                   pipeline->maxTotalThreadsPerThreadgroup()));
    encoder->dispatchThreads(MTL::Size::Make(threads, 1, 1),
                             MTL::Size::Make(tgWidth, 1, 1));
    encoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    if (commandBuffer->status() != MTL::CommandBufferStatusCompleted) {
      throw std::runtime_error("Metal command buffer did not complete successfully");
    }

    const auto *deviceOut = reinterpret_cast<const MetalUInt2 *>(
        particleIndexBuffer->contents());
    for (size_t i = 0; i < result.particleIndex.size(); ++i) {
      result.particleIndex[i][0] = deviceOut[i].s[0];
      result.particleIndex[i][1] = deviceOut[i].s[1];
    }

    return result;
  }
};

} // namespace SiberneticTest
