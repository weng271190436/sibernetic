#pragma once

#include <vector>

#include "../../src/kernels/PcisphPredictDensityKernel.h"
#include "../utils/context/opencl_context.h"
#include "pcisph_predict_density_test_common.h"

namespace SiberneticTest {

class OpenCLPcisphPredictDensityRunner : public PcisphPredictDensityRunner {
public:
  PcisphPredictDensityResult run(const PcisphPredictDensityCase &tc) override {
    auto input = tc.toInput();
    const cl_uint N = static_cast<cl_uint>(input.particleCount);

    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;

    // rho is the output buffer (2×N floats). Initialize to zero.
    std::vector<float> rhoInit(static_cast<size_t>(N) * 2, 0.0f);
    cl::Buffer outputRho(opencl.context(),
                         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                         rhoInit.size() * sizeof(float), rhoInit.data(), &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create rho buffer");
    }

    auto args = Sibernetic::toOpenCLArgs(input, opencl.context(), outputRho);

    cl::Kernel kernel(opencl.program(),
                      Sibernetic::kPcisphPredictDensityKernelName, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create pcisph_predictDensity kernel");
    }
    args.bind(kernel);

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(N),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to execute pcisph_predictDensity kernel");
    }

    // Read back predicted density from rho[N..2N).
    PcisphPredictDensityResult result;
    result.predictedRho.resize(N);
    if (opencl.queue().enqueueReadBuffer(
            outputRho, CL_TRUE,
            sizeof(float) * N, // offset to second half
            sizeof(float) * N, result.predictedRho.data()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to read rho output buffer");
    }
    return result;
  }
};

} // namespace SiberneticTest
