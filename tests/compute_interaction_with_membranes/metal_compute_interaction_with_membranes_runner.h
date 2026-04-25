#pragma once

#include <vector>

#include "../../src/convert/MetalConvert.h"
#include "../../src/kernels/ComputeInteractionWithMembranesKernel.h"
#include "../utils/context/metal_context.h"
#include "compute_interaction_with_membranes_test_common.h"

namespace SiberneticTest {

// ── computeInteractionWithMembranes Metal runner ────────────────────────────

class MetalComputeInteractionWithMembranesRunner
    : public ComputeInteractionWithMembranesRunner {
public:
  ComputeInteractionWithMembranesResult
  run(const ComputeInteractionWithMembranesCase &tc) override {
    auto input = tc.toInput();
    const uint32_t N = input.particleCount;

    MetalKernelContext metal(
        Sibernetic::kComputeInteractionWithMembranesKernelName);
    auto *device = metal.device();

    // position: 2×N float4 (in/out).
    auto positionBuf = NS::TransferPtr(device->newBuffer(
        input.position.data(), input.position.size_bytes(),
        MTL::ResourceStorageModeShared));
    auto velocityBuf = NS::TransferPtr(device->newBuffer(
        input.velocity.data(), input.velocity.size_bytes(),
        MTL::ResourceStorageModeShared));
    auto sortedCellBuf = NS::TransferPtr(device->newBuffer(
        input.sortedCellAndSerialId.data(),
        input.sortedCellAndSerialId.size_bytes(),
        MTL::ResourceStorageModeShared));
    auto backMapBuf = NS::TransferPtr(device->newBuffer(
        input.sortedParticleIdBySerialId.data(),
        input.sortedParticleIdBySerialId.size_bytes(),
        MTL::ResourceStorageModeShared));
    auto neighborMapBuf = NS::TransferPtr(device->newBuffer(
        input.neighborMap.data(), input.neighborMap.size_bytes(),
        MTL::ResourceStorageModeShared));

    // particleMembranesList and membraneData may be empty; use dummy buffers.
    auto memListBuf = NS::TransferPtr(
        input.particleMembranesList.empty()
            ? device->newBuffer(4, MTL::ResourceStorageModeShared)
            : device->newBuffer(input.particleMembranesList.data(),
                                input.particleMembranesList.size_bytes(),
                                MTL::ResourceStorageModeShared));
    auto memDataBuf = NS::TransferPtr(
        input.membraneData.empty()
            ? device->newBuffer(4, MTL::ResourceStorageModeShared)
            : device->newBuffer(input.membraneData.data(),
                                input.membraneData.size_bytes(),
                                MTL::ResourceStorageModeShared));

    Sibernetic::ComputeInteractionWithMembranesMetalArgs args{};
    args.position = positionBuf.get();
    args.velocity = velocityBuf.get();
    args.sortedCellAndSerialId = sortedCellBuf.get();
    args.sortedParticleIdBySerialId = backMapBuf.get();
    args.neighborMap = neighborMapBuf.get();
    args.particleMembranesList = memListBuf.get();
    args.membraneData = memDataBuf.get();
    args.particleCount = N;
    args.r0 = input.r0;

    metal.dispatch(N,
                   [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    ComputeInteractionWithMembranesResult result;
    const auto *posPtr = reinterpret_cast<const Sibernetic::MetalFloat4 *>(
        positionBuf->contents());
    result.position = Sibernetic::Metal::decode(posPtr, 2 * N);
    return result;
  }
};

// ── computeInteractionWithMembranes_finalize Metal runner ───────────────────

class MetalComputeInteractionWithMembranesFinalizeRunner
    : public ComputeInteractionWithMembranesFinalizeRunner {
public:
  ComputeInteractionWithMembranesFinalizeResult
  run(const ComputeInteractionWithMembranesFinalizeCase &tc) override {
    auto input = tc.toInput();
    const uint32_t N = input.particleCount;

    MetalKernelContext metal(
        Sibernetic::kComputeInteractionWithMembranesFinalizeKernelName);
    auto *device = metal.device();

    auto positionBuf = NS::TransferPtr(device->newBuffer(
        input.position.data(), input.position.size_bytes(),
        MTL::ResourceStorageModeShared));
    auto backMapBuf = NS::TransferPtr(device->newBuffer(
        input.sortedParticleIdBySerialId.data(),
        input.sortedParticleIdBySerialId.size_bytes(),
        MTL::ResourceStorageModeShared));

    Sibernetic::ComputeInteractionWithMembranesFinalizeMetalArgs args{};
    args.position = positionBuf.get();
    args.sortedParticleIdBySerialId = backMapBuf.get();
    args.particleCount = N;

    metal.dispatch(N,
                   [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    ComputeInteractionWithMembranesFinalizeResult result;
    const auto *posPtr = reinterpret_cast<const Sibernetic::MetalFloat4 *>(
        positionBuf->contents());
    result.position = Sibernetic::Metal::decode(posPtr, 2 * N);
    return result;
  }
};

} // namespace SiberneticTest
