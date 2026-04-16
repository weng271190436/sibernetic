#pragma once

#include <vector>

#include "../../src/convert/MetalConvert.h"
#include "../../src/kernels/PcisphComputePressureForceAccelerationKernel.h"
#include "../utils/buffer/metal_buffer_utils.h"
#include "../utils/context/metal_context.h"
#include "pcisph_compute_pressure_force_acceleration_test_common.h"

namespace SiberneticTest {

class MetalPcisphComputePressureForceAccelerationRunner
    : public PcisphComputePressureForceAccelerationRunner {
public:
  PcisphComputePressureForceAccelerationResult
  run(const PcisphComputePressureForceAccelerationCase &tc) override {
    auto input = tc.toInput();
    const uint32_t N = input.particleCount;

    MetalKernelContext metal(
        Sibernetic::kPcisphComputePressureForceAccelerationKernelName);
    auto *device = metal.device();

    // Acceleration buffer: 2*N float4 entries. Kernel writes [N..2N).
    auto accelerationBuf = makeMetalOutputBuffer(
        device, sizeof(float) * 4 * static_cast<size_t>(N) * 2);

    auto args = Sibernetic::toMetalArgs(input, device, accelerationBuf.get());

    metal.dispatch(N, [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    // Read back acceleration[N..2N).
    PcisphComputePressureForceAccelerationResult result;
    const auto *accelPtr = reinterpret_cast<const Sibernetic::MetalFloat4 *>(
        accelerationBuf->contents());
    result.pressureAcceleration = Sibernetic::Metal::decode(accelPtr + N, N);
    return result;
  }
};

} // namespace SiberneticTest
