#pragma once

#include <vector>

#include "../../src/convert/MetalConvert.h"
#include "../../src/kernels/PcisphCorrectPressureKernel.h"
#include "../utils/context/metal_context.h"
#include "pcisph_correct_pressure_test_common.h"

namespace SiberneticTest {

class MetalPcisphCorrectPressureRunner : public PcisphCorrectPressureRunner {
public:
  PcisphCorrectPressureResult
  run(const PcisphCorrectPressureCase &tc) override {
    auto input = tc.toInput();
    const uint32_t N = input.particleCount;

    MetalKernelContext metal(Sibernetic::kPcisphCorrectPressureKernelName);
    auto *device = metal.device();

    // Pressure buffer: copy input values (kernel modifies in-place).
    auto pressureBuf = NS::TransferPtr(device->newBuffer(
        const_cast<float *>(input.pressure.data()), input.pressure.size_bytes(),
        MTL::ResourceStorageModeShared));

    // Rho buffer: 2*N floats, kernel reads [N..2N).
    auto rhoBuf = NS::TransferPtr(device->newBuffer(
        const_cast<float *>(input.rho.data()), input.rho.size_bytes(),
        MTL::ResourceStorageModeShared));

    auto args =
        Sibernetic::toMetalArgs(input, device, pressureBuf.get(), rhoBuf.get());

    metal.dispatch(N, [&](MTL::ComputeCommandEncoder *enc) { args.bind(enc); });

    // Read back pressure[0..N).
    PcisphCorrectPressureResult result;
    const auto *pressurePtr =
        reinterpret_cast<const float *>(pressureBuf->contents());
    result.pressure = Sibernetic::Metal::decode(pressurePtr, N);
    return result;
  }
};

} // namespace SiberneticTest
