#pragma once

#include <vector>

#include "../../src/kernels/PcisphComputeForcesKernel.h"
#include "../utils/context/opencl_context.h"
#include "pcisph_compute_forces_test_common.h"

namespace SiberneticTest {

class OpenCLPcisphComputeForcesRunner : public PcisphComputeForcesRunner {
public:
  PcisphComputeForcesResult run(const PcisphComputeForcesCase &tc) override {
    auto input = tc.toInput();
    const cl_uint particleCount = static_cast<cl_uint>(input.particleCount);

    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;
    cl::Buffer outPressure(opencl.context(), CL_MEM_WRITE_ONLY,
                           sizeof(float) * particleCount, nullptr, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create pressure output buffer");
    }
    cl::Buffer outAcceleration(opencl.context(), CL_MEM_WRITE_ONLY,
                               sizeof(float) * 4 * particleCount * 2, nullptr,
                               &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create acceleration output buffer");
    }

    auto args = Sibernetic::toOpenCLArgs(input, opencl.context(), outPressure,
                                         outAcceleration);

    cl::Kernel kernel(opencl.program(),
                      Sibernetic::kPcisphComputeForcesKernelName, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to create pcisph_computeForcesAndInitPressure kernel");
    }
    args.bind(kernel);

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(particleCount),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to execute pcisph_computeForcesAndInitPressure kernel");
    }

    PcisphComputeForcesResult result;
    result.pressure.resize(particleCount);
    if (opencl.queue().enqueueReadBuffer(
            outPressure, CL_TRUE, 0, sizeof(float) * particleCount,
            result.pressure.data()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to read pressure output buffer");
    }
    result.acceleration.resize(static_cast<size_t>(particleCount) * 2);
    if (opencl.queue().enqueueReadBuffer(
            outAcceleration, CL_TRUE, 0, sizeof(float) * 4 * particleCount * 2,
            result.acceleration.data()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to read acceleration output buffer");
    }
    return result;
  }
};

} // namespace SiberneticTest
