#pragma once

#include <vector>

#include "../../src/kernels/ComputeDensityKernel.h"
#include "../utils/context/opencl_context.h"
#include "compute_density_test_common.h"

namespace SiberneticTest {

class OpenCLComputeDensityRunner : public ComputeDensityRunner {
public:
  ComputeDensityResult run(const ComputeDensityCase &tc) override {
    const cl_uint particleCount =
        static_cast<cl_uint>(tc.particleIndexBack.size());
    if (tc.neighborMap.size() != static_cast<size_t>(particleCount) * 32u) {
      throw std::runtime_error("neighborMap size must be particleCount * 32");
    }

    // Convert host float2 array to flat floats for the Input struct.
    std::vector<cl_float2> clNeighborMap(tc.neighborMap.size());
    for (size_t i = 0; i < tc.neighborMap.size(); ++i) {
      clNeighborMap[i].s[0] = tc.neighborMap[i][0];
      clNeighborMap[i].s[1] = tc.neighborMap[i][1];
    }

    OpenCLKernelContext opencl;

    // Build backend-agnostic input.
    Sibernetic::ComputeDensityInput input{};
    input.neighborMap =
        reinterpret_cast<const float *>(clNeighborMap.data());
    input.massMultWpoly6Coefficient = tc.massMultWpoly6Coefficient;
    input.hScaled2 = tc.hScaled2;
    input.particleIndexBack = tc.particleIndexBack.data();
    input.particleCount = particleCount;

    // Create output buffer.
    cl_int err = CL_SUCCESS;
    cl::Buffer outputRho(opencl.context(), CL_MEM_WRITE_ONLY,
                         sizeof(float) * particleCount, nullptr, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create output rho buffer");
    }

    // Convert to OpenCL args.
    auto args =
        Sibernetic::toOpenCLArgs(input, opencl.context(), outputRho);

    // Create kernel, bind, and dispatch.
    cl::Kernel kernel(opencl.program(), Sibernetic::kComputeDensityKernelName,
                      &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create compute density kernel");
    }
    args.bind(kernel);

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(particleCount),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error("Failed to execute compute density kernel");
    }

    // Read back results.
    ComputeDensityResult result;
    result.rho.resize(particleCount);
    if (opencl.queue().enqueueReadBuffer(outputRho, CL_TRUE, 0,
                                         sizeof(float) * particleCount,
                                         result.rho.data()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to read rho output buffer");
    }
    return result;
  }
};

} // namespace SiberneticTest
