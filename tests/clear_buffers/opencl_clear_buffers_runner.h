#pragma once

#include <vector>

#include "../../src/kernels/ClearBuffersKernel.h"
#include "../utils/context/opencl_context.h"
#include "clear_buffers_test_common.h"

namespace SiberneticTest {

class OpenCLClearBuffersRunner : public ClearBuffersRunner {
public:
  ClearBuffersResult run(const ClearBuffersCase &tc) override {
    auto input = tc.toInput();
    // Read back the full buffer, which may be larger than
    // particleCount * kMaxNeighborCount when testing bounds guard.
    const size_t bufferEntries = tc.neighborMap.size();
    const size_t bufferBytes = bufferEntries * sizeof(Sibernetic::HostFloat2);

    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;

    // neighborMap is read/write — pre-populated with input data.
    // Use tc.neighborMap directly for the full buffer size.
    cl::Buffer inOutNeighborMap(
        opencl.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        bufferBytes,
        const_cast<Sibernetic::HostFloat2 *>(tc.neighborMap.data()), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create neighborMap buffer");

    auto args =
        Sibernetic::toOpenCLArgs(input, opencl.context(), inOutNeighborMap);

    cl::Kernel kernel(opencl.program(), Sibernetic::kClearBuffersKernelName,
                      &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create clearBuffers kernel");
    args.bind(kernel);

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(input.particleCount),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error("Failed to execute clearBuffers kernel");
    }

    ClearBuffersResult result;
    result.neighborMap.resize(bufferEntries);
    if (opencl.queue().enqueueReadBuffer(inOutNeighborMap, CL_TRUE, 0,
                                         bufferBytes,
                                         result.neighborMap.data()) !=
        CL_SUCCESS)
      throw std::runtime_error("Failed to read neighborMap output buffer");

    return result;
  }
};

} // namespace SiberneticTest
