#pragma once

#include <vector>

#include "../../src/kernels/SortPostPassKernel.h"
#include "../../src/convert/OpenCLConvert.h"
#include "../utils/context/opencl_context.h"
#include "sort_post_pass_test_common.h"

namespace SiberneticTest {

class OpenCLSortPostPassRunner : public SortPostPassRunner {
public:
  SortPostPassResult run(const SortPostPassCase &tc) override {
    auto input = tc.toInput();
    const cl_uint particleCount = static_cast<cl_uint>(input.particleCount);

    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;
    cl::Buffer outIndexBack(opencl.context(), CL_MEM_WRITE_ONLY,
                            sizeof(uint32_t) * particleCount, nullptr, &err);
    cl::Buffer outSortedPos(opencl.context(), CL_MEM_WRITE_ONLY,
                            sizeof(cl_float4) * particleCount, nullptr, &err);
    cl::Buffer outSortedVel(opencl.context(), CL_MEM_WRITE_ONLY,
                            sizeof(cl_float4) * particleCount, nullptr, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create sortPostPass output buffers");
    }

    auto args = Sibernetic::toOpenCLArgs(input, opencl.context(), outIndexBack,
                                         outSortedPos, outSortedVel);

    cl::Kernel kernel(opencl.program(), Sibernetic::kSortPostPassKernelName,
                      &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create sortPostPass kernel");
    }
    args.bind(kernel);

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(particleCount),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error("Failed to execute sortPostPass kernel");
    }

    SortPostPassResult result;
    result.particleIndexBack.resize(particleCount);
    opencl.queue().enqueueReadBuffer(outIndexBack, CL_TRUE, 0,
                                     sizeof(uint32_t) * particleCount,
                                     result.particleIndexBack.data());

    std::vector<cl_float4> clSortedPos(particleCount);
    opencl.queue().enqueueReadBuffer(outSortedPos, CL_TRUE, 0,
                                     sizeof(cl_float4) * particleCount,
                                     clSortedPos.data());
    result.sortedPosition = Sibernetic::OpenCL::decode(clSortedPos);

    std::vector<cl_float4> clSortedVel(particleCount);
    opencl.queue().enqueueReadBuffer(outSortedVel, CL_TRUE, 0,
                                     sizeof(cl_float4) * particleCount,
                                     clSortedVel.data());
    result.sortedVelocity = Sibernetic::OpenCL::decode(clSortedVel);
    return result;
  }
};

} // namespace SiberneticTest
