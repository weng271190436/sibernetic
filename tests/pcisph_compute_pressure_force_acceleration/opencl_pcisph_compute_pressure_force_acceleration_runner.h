#pragma once

#include <vector>

#include "../../src/kernels/PcisphComputePressureForceAccelerationKernel.h"
#include "../utils/context/opencl_context.h"
#include "pcisph_compute_pressure_force_acceleration_test_common.h"

namespace SiberneticTest {

class OpenCLPcisphComputePressureForceAccelerationRunner
    : public PcisphComputePressureForceAccelerationRunner {
public:
  PcisphComputePressureForceAccelerationResult
  run(const PcisphComputePressureForceAccelerationCase &tc) override {
    auto input = tc.toInput();
    const cl_uint N = static_cast<cl_uint>(input.particleCount);

    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;

    // Acceleration buffer: 2*N float4 entries. Kernel writes [N..2N).
    cl::Buffer accelerationBuf(opencl.context(), CL_MEM_READ_WRITE,
                               sizeof(float) * 4 * static_cast<size_t>(N) * 2,
                               nullptr, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create acceleration buffer");
    }

    auto args =
        Sibernetic::toOpenCLArgs(input, opencl.context(), accelerationBuf);

    cl::Kernel kernel(
        opencl.program(),
        Sibernetic::kPcisphComputePressureForceAccelerationKernelName, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to create pcisph_computePressureForceAcceleration kernel");
    }
    args.bind(kernel);

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(N),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to execute pcisph_computePressureForceAcceleration kernel");
    }

    // Read back acceleration[N..2N).
    PcisphComputePressureForceAccelerationResult result;
    result.pressureAcceleration.resize(N);
    const size_t offsetBytes = sizeof(float) * 4 * N;
    const size_t sizeBytes = sizeof(float) * 4 * N;
    if (opencl.queue().enqueueReadBuffer(
            accelerationBuf, CL_TRUE, offsetBytes, sizeBytes,
            result.pressureAcceleration.data()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to read acceleration output buffer");
    }
    return result;
  }
};

} // namespace SiberneticTest
