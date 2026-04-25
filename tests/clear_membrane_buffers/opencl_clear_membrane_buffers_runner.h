#pragma once

#include <vector>

#include "../../src/kernels/ClearMembraneBuffersKernel.h"
#include "../utils/context/opencl_context.h"
#include "clear_membrane_buffers_test_common.h"

namespace SiberneticTest {

class OpenCLClearMembraneBuffersRunner : public ClearMembraneBuffersRunner {
public:
  ClearMembraneBuffersResult
  run(const ClearMembraneBuffersCase &tc) override {
    auto input = tc.toInput();
    const uint32_t N = input.particleCount;

    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;

    cl::Buffer inOutPosition(
        opencl.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        input.position.size_bytes(),
        const_cast<Sibernetic::HostFloat4 *>(input.position.data()), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create position buffer");

    cl::Buffer inOutVelocity(
        opencl.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        input.velocity.size_bytes(),
        const_cast<Sibernetic::HostFloat4 *>(input.velocity.data()), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create velocity buffer");

    auto args = Sibernetic::toOpenCLArgs(input, opencl.context(),
                                         inOutPosition, inOutVelocity);

    cl::Kernel kernel(opencl.program(),
                      Sibernetic::kClearMembraneBuffersKernelName, &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create clearMembraneBuffers kernel");
    args.bind(kernel);

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(N),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to execute clearMembraneBuffers kernel");
    }

    ClearMembraneBuffersResult result;
    result.position.resize(2 * N);
    if (opencl.queue().enqueueReadBuffer(inOutPosition, CL_TRUE, 0,
                                         sizeof(float) * 4 * 2 * N,
                                         result.position.data()) != CL_SUCCESS)
      throw std::runtime_error("Failed to read position output buffer");

    result.velocity.resize(2 * N);
    if (opencl.queue().enqueueReadBuffer(inOutVelocity, CL_TRUE, 0,
                                         sizeof(float) * 4 * 2 * N,
                                         result.velocity.data()) != CL_SUCCESS)
      throw std::runtime_error("Failed to read velocity output buffer");

    return result;
  }
};

} // namespace SiberneticTest
