#pragma once

#include <vector>

#include "../../src/convert/MetalConvert.h"
#include "../../src/kernels/PcisphIntegrateKernel.h"
#include "../utils/context/metal_context.h"
#include "pcisph_integrate_test_common.h"

namespace SiberneticTest {

class MetalPcisphIntegrateRunner : public PcisphIntegrateRunner {
public:
  PcisphIntegrateResult run(const PcisphIntegrateCase &tc) override {
    auto input = tc.toInput();
    const uint32_t N = input.particleCount;

    MetalKernelContext metal(Sibernetic::kPcisphIntegrateKernelName);
    auto *device = metal.device();

    // acceleration is read/write (3×N float4)
    auto inOutAcceleration = NS::TransferPtr(device->newBuffer(
        input.acceleration.data(), input.acceleration.size_bytes(),
        MTL::ResourceStorageModeShared));
    // sortedPosition is read/write
    auto inOutSortedPosition = NS::TransferPtr(device->newBuffer(
        input.sortedPosition.data(), input.sortedPosition.size_bytes(),
        MTL::ResourceStorageModeShared));
    // originalPosition is read/write
    auto inOutOriginalPosition = NS::TransferPtr(device->newBuffer(
        input.originalPosition.data(), input.originalPosition.size_bytes(),
        MTL::ResourceStorageModeShared));
    // velocity is read/write
    auto inOutVelocity = NS::TransferPtr(
        device->newBuffer(input.velocity.data(), input.velocity.size_bytes(),
                          MTL::ResourceStorageModeShared));

    auto args = Sibernetic::toMetalArgs(
        input, device, inOutAcceleration.get(), inOutSortedPosition.get(),
        inOutOriginalPosition.get(), inOutVelocity.get());

    metal.dispatch(N, [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    PcisphIntegrateResult result;

    // Read back all output buffers.
    const auto *accelPtr = reinterpret_cast<const Sibernetic::MetalFloat4 *>(
        inOutAcceleration->contents());
    result.acceleration = Sibernetic::Metal::decode(accelPtr, 3 * N);

    const auto *sortedPosPtr =
        reinterpret_cast<const Sibernetic::MetalFloat4 *>(
            inOutSortedPosition->contents());
    result.sortedPosition = Sibernetic::Metal::decode(sortedPosPtr, N);

    const auto *origPosPtr = reinterpret_cast<const Sibernetic::MetalFloat4 *>(
        inOutOriginalPosition->contents());
    result.originalPosition = Sibernetic::Metal::decode(origPosPtr, N);

    const auto *velPtr = reinterpret_cast<const Sibernetic::MetalFloat4 *>(
        inOutVelocity->contents());
    result.velocity = Sibernetic::Metal::decode(velPtr, N);

    return result;
  }
};

} // namespace SiberneticTest
