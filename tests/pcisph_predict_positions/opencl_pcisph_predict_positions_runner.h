#pragma once

#include <vector>

#include "../../src/kernels/PcisphPredictPositionsKernel.h"
#include "../utils/context/opencl_context.h"
#include "pcisph_predict_positions_test_common.h"

namespace SiberneticTest {

class OpenCLPcisphPredictPositionsRunner : public PcisphPredictPositionsRunner {
public:
  PcisphPredictPositionsResult
  run(const PcisphPredictPositionsCase &tc) override {
    auto input = tc.toInput();
    const cl_uint N = static_cast<cl_uint>(input.particleCount);

    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;

    // acceleration is read/write (3×N float4)
    cl::Buffer inOutAcceleration(
        opencl.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        input.acceleration.size_bytes(),
        const_cast<Sibernetic::HostFloat4 *>(input.acceleration.data()), &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create acceleration buffer");
    }

    // sortedPosition is read/write (2×N float4)
    cl::Buffer inOutSortedPosition(
        opencl.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        input.sortedPosition.size_bytes(),
        const_cast<Sibernetic::HostFloat4 *>(input.sortedPosition.data()),
        &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create sortedPosition buffer");
    }

    auto args = Sibernetic::toOpenCLArgs(
        input, opencl.context(), inOutAcceleration, inOutSortedPosition);

    cl::Kernel kernel(opencl.program(),
                      Sibernetic::kPcisphPredictPositionsKernelName, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to create pcisph_predictPositions kernel");
    }
    args.bind(kernel);

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(N),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to execute pcisph_predictPositions kernel");
    }

    // Read back predicted positions from sortedPosition[N..2N)
    PcisphPredictPositionsResult result;
    result.predictedPosition.resize(N);
    if (opencl.queue().enqueueReadBuffer(
            inOutSortedPosition, CL_TRUE,
            sizeof(float) * 4 * N, // offset to second half
            sizeof(float) * 4 * N,
            result.predictedPosition.data()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to read sortedPosition output buffer");
    }
    return result;
  }
};

} // namespace SiberneticTest
