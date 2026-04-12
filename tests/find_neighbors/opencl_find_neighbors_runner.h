#pragma once

#include <vector>

#include "../../src/kernels/FindNeighborsKernel.h"
#include "../../src/convert/OpenCLConvert.h"
#include "../utils/context/opencl_context.h"
#include "find_neighbors_test_common.h"

namespace SiberneticTest {

class OpenCLFindNeighborsRunner : public FindNeighborsRunner {
public:
  FindNeighborsResult run(const FindNeighborsCase &tc) override {
    auto input = tc.toInput();
    const cl_uint particleCount = static_cast<cl_uint>(input.particleCount);
    const size_t neighborCount = static_cast<size_t>(particleCount) * 32u;

    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;
    cl::Buffer outputNeighborMap(opencl.context(), CL_MEM_WRITE_ONLY,
                                 sizeof(cl_float2) * neighborCount, nullptr,
                                 &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create output neighborMap buffer");
    }

    auto args =
        Sibernetic::toOpenCLArgs(input, opencl.context(), outputNeighborMap);

    cl::Kernel kernel(opencl.program(), Sibernetic::kFindNeighborsKernelName,
                      &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create findNeighbors kernel");
    }
    args.bind(kernel);

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(particleCount),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error("Failed to execute findNeighbors kernel");
    }

    FindNeighborsResult result;
    std::vector<cl_float2> clNeighborMap(neighborCount);
    if (opencl.queue().enqueueReadBuffer(
            outputNeighborMap, CL_TRUE, 0,
            sizeof(cl_float2) * neighborCount,
            clNeighborMap.data()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to read neighborMap output buffer");
    }
    result.neighborMap = Sibernetic::OpenCL::decode(clNeighborMap);
    return result;
  }
};

} // namespace SiberneticTest
