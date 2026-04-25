#pragma once

#include <vector>

#include "../../src/convert/MetalConvert.h"
#include "../../src/kernels/PcisphComputeElasticForcesKernel.h"
#include "../utils/context/metal_context.h"
#include "pcisph_compute_elastic_forces_test_common.h"

namespace SiberneticTest {

class MetalPcisphComputeElasticForcesRunner
    : public PcisphComputeElasticForcesRunner {
public:
  PcisphComputeElasticForcesResult
  run(const PcisphComputeElasticForcesCase &tc) override {
    auto input = tc.toInput();
    const uint32_t N = static_cast<uint32_t>(tc.sortedPosition.size());

    MetalKernelContext metal(Sibernetic::kPcisphComputeElasticForcesKernelName);
    auto *device = metal.device();

    auto sortedPosition = NS::TransferPtr(device->newBuffer(
        input.sortedPosition.data(), input.sortedPosition.size_bytes(),
        MTL::ResourceStorageModeShared));
    auto acceleration = NS::TransferPtr(device->newBuffer(
        input.acceleration.data(), input.acceleration.size_bytes(),
        MTL::ResourceStorageModeShared));
    auto sortedParticleIdBySerialId = NS::TransferPtr(
        device->newBuffer(input.sortedParticleIdBySerialId.data(),
                          input.sortedParticleIdBySerialId.size_bytes(),
                          MTL::ResourceStorageModeShared));
    auto sortedCellAndSerialId = NS::TransferPtr(
        device->newBuffer(input.sortedCellAndSerialId.data(),
                          input.sortedCellAndSerialId.size_bytes(),
                          MTL::ResourceStorageModeShared));
    // elasticConnectionsData: may be empty if numOfElasticP == 0.
    Sibernetic::HostFloat4 dummyConn = {0.0f, 0.0f, 0.0f, 0.0f};
    auto elasticConnectionsData = NS::TransferPtr(
        device->newBuffer(input.elasticConnectionsData.size_bytes() > 0
                              ? input.elasticConnectionsData.data()
                              : &dummyConn,
                          input.elasticConnectionsData.size_bytes() > 0
                              ? input.elasticConnectionsData.size_bytes()
                              : sizeof(dummyConn),
                          MTL::ResourceStorageModeShared));

    // muscleActivationSignal: may be empty if muscleCount == 0.
    MTL::Buffer *muscleSignalRaw = nullptr;
    NS::SharedPtr<MTL::Buffer> muscleSignalOwner;
    if (input.muscleCount > 0) {
      muscleSignalOwner = NS::TransferPtr(
          device->newBuffer(input.muscleActivationSignal.data(),
                            input.muscleActivationSignal.size_bytes(),
                            MTL::ResourceStorageModeShared));
      muscleSignalRaw = muscleSignalOwner.get();
    } else {
      // Metal requires a non-null buffer; create a 4-byte placeholder.
      float dummy = 0.0f;
      muscleSignalOwner = NS::TransferPtr(device->newBuffer(
          &dummy, sizeof(dummy), MTL::ResourceStorageModeShared));
      muscleSignalRaw = muscleSignalOwner.get();
    }

    auto originalPosition = NS::TransferPtr(device->newBuffer(
        input.originalPosition.data(), input.originalPosition.size_bytes(),
        MTL::ResourceStorageModeShared));

    auto args = Sibernetic::toMetalArgs(
        input, device, sortedPosition.get(), acceleration.get(),
        sortedParticleIdBySerialId.get(), sortedCellAndSerialId.get(),
        elasticConnectionsData.get(), muscleSignalRaw, originalPosition.get());

    const uint32_t threadCount = input.numOfElasticP;
    if (threadCount > 0) {
      metal.dispatch(threadCount,
                     [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });
    }

    PcisphComputeElasticForcesResult result;
    const auto *accelPtr = reinterpret_cast<const Sibernetic::MetalFloat4 *>(
        acceleration->contents());
    result.acceleration = Sibernetic::Metal::decode(accelPtr, N);

    return result;
  }
};

} // namespace SiberneticTest
