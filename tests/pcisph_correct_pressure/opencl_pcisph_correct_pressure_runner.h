#pragma once

#include <vector>

#include "../../src/kernels/PcisphCorrectPressureKernel.h"
#include "../utils/context/opencl_context.h"
#include "pcisph_correct_pressure_test_common.h"

namespace SiberneticTest {

class OpenCLPcisphCorrectPressureRunner : public PcisphCorrectPressureRunner {
public:
  PcisphCorrectPressureResult
  run(const PcisphCorrectPressureCase &tc) override {
    auto input = tc.toInput();
    const cl_uint N = static_cast<cl_uint>(input.particleCount);

    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;

    // Pressure buffer: copy input values (kernel modifies in-place).
    std::vector<float> pressureInit(input.pressure.begin(),
                                    input.pressure.end());
    cl::Buffer pressureBuf(opencl.context(),
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           pressureInit.size() * sizeof(float),
                           pressureInit.data(), &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create pressure buffer");
    }

    // Rho buffer: 2*N floats, kernel reads [N..2N).
    std::vector<float> rhoInit(input.rho.begin(), input.rho.end());
    cl::Buffer rhoBuf(opencl.context(),
                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      rhoInit.size() * sizeof(float), rhoInit.data(), &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create rho buffer");
    }

    auto args =
        Sibernetic::toOpenCLArgs(input, opencl.context(), pressureBuf, rhoBuf);

    cl::Kernel kernel(opencl.program(),
                      Sibernetic::kPcisphCorrectPressureKernelName, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to create pcisph_correctPressure kernel");
    }
    args.bind(kernel);

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(N),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to execute pcisph_correctPressure kernel");
    }

    // Read back pressure[0..N).
    PcisphCorrectPressureResult result;
    result.pressure.resize(N);
    if (opencl.queue().enqueueReadBuffer(pressureBuf, CL_TRUE, 0,
                                         sizeof(float) * N,
                                         result.pressure.data()) !=
        CL_SUCCESS) {
      throw std::runtime_error("Failed to read pressure output buffer");
    }
    return result;
  }
};

} // namespace SiberneticTest
