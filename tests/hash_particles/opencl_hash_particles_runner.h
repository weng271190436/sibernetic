#pragma once

#include <vector>

#include "../../src/kernels/HashParticlesKernel.h"
#include "../../src/convert/OpenCLConvert.h"
#include "../utils/context/opencl_context.h"
#include "hash_particles_test_common.h"

namespace SiberneticTest {

class OpenCLHashParticlesRunner : public HashParticlesRunner {
public:
  HashParticlesResult run(const HashParticlesCase &tc) override {
    auto input = tc.toInput();
    const cl_uint particleCount = static_cast<cl_uint>(input.particleCount);

    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;
    cl::Buffer outputParticleIndex(opencl.context(), CL_MEM_WRITE_ONLY,
                                   sizeof(cl_uint2) * particleCount, nullptr,
                                   &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create output particleIndex buffer");
    }

    auto args =
        Sibernetic::toOpenCLArgs(input, opencl.context(), outputParticleIndex);

    cl::Kernel kernel(opencl.program(), Sibernetic::kHashParticlesKernelName,
                      &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create hashParticles kernel");
    }
    args.bind(kernel);

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(particleCount),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error("Failed to execute hashParticles kernel");
    }

    HashParticlesResult result;
    std::vector<cl_uint2> clOutput(particleCount);
    if (opencl.queue().enqueueReadBuffer(
            outputParticleIndex, CL_TRUE, 0,
            sizeof(cl_uint2) * particleCount,
            clOutput.data()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to read particleIndex output buffer");
    }
    result.particleIndex = Sibernetic::OpenCL::decode(clOutput);
    return result;
  }
};

} // namespace SiberneticTest
