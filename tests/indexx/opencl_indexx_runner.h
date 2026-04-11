#pragma once

#include <stdexcept>
#include <vector>

#include "../utils/opencl_context.h"
#include "../utils/opencl_helpers.h"
#include "indexx_test_common.h"

namespace SiberneticTest {

class OpenCLIndexxRunner : public IndexxRunner {
public:
  IndexxResult run(const IndexxCase &tc) override {
    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;
    cl::Kernel kernel(opencl.program(), "indexx", &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create kernel: indexx");
    }

    std::vector<cl_uint2> clParticleIndex(tc.particleIndex.size());
    for (size_t i = 0; i < tc.particleIndex.size(); ++i) {
      clParticleIndex[i].s[0] = tc.particleIndex[i][0];
      clParticleIndex[i].s[1] = tc.particleIndex[i][1];
    }

    const cl_uint gridCellCount = tc.gridCellCount;
    const cl_uint particleCount = static_cast<cl_uint>(tc.particleIndex.size());
    const size_t outputCount = static_cast<size_t>(gridCellCount) + 1u;

    auto particleIndexBuf =
        makeOpenCLReadBuffer(opencl.context(), clParticleIndex, err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create particleIndex buffer");
    }
    auto gridCellIndexBuf =
        makeOpenCLWriteBuffer(opencl.context(), sizeof(cl_uint) * outputCount, err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create gridCellIndex buffer");
    }

    if (kernel.setArg(0, particleIndexBuf) != CL_SUCCESS ||
        kernel.setArg(1, gridCellCount) != CL_SUCCESS ||
        kernel.setArg(2, gridCellIndexBuf) != CL_SUCCESS ||
        kernel.setArg(3, particleCount) != CL_SUCCESS) {
      throw std::runtime_error("Failed to set kernel args for indexx");
    }

    runOpenCL1DKernel(opencl.queue(), kernel, outputCount, "indexx");

    IndexxResult result;
    result.gridCellIndex.resize(outputCount);
    if (opencl.queue().enqueueReadBuffer(gridCellIndexBuf, CL_TRUE, 0,
                                         sizeof(cl_uint) * outputCount,
                                         result.gridCellIndex.data()) !=
        CL_SUCCESS) {
      throw std::runtime_error("Failed to read gridCellIndex buffer");
    }
    return result;
  }
};

} // namespace SiberneticTest