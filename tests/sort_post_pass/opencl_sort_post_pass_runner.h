#pragma once

#include <stdexcept>
#include <vector>

#include "../utils/opencl_context.h"
#include "../utils/opencl_helpers.h"
#include "sort_post_pass_test_common.h"

namespace SiberneticTest {

class OpenCLSortPostPassRunner : public SortPostPassRunner {
public:
  SortPostPassResult run(const SortPostPassCase &tc) override {
    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;
    cl::Kernel kernel(opencl.program(), "sortPostPass", &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create kernel: sortPostPass");
    }

    const cl_uint n = static_cast<cl_uint>(tc.particleIndex.size());

    std::vector<cl_uint2> clParticleIndex(n);
    std::vector<cl_float4> clPosition(n), clVelocity(n);
    for (size_t i = 0; i < n; ++i) {
      clParticleIndex[i].s[0] = tc.particleIndex[i][0];
      clParticleIndex[i].s[1] = tc.particleIndex[i][1];
      clPosition[i].s[0] = tc.position[i][0];
      clPosition[i].s[1] = tc.position[i][1];
      clPosition[i].s[2] = tc.position[i][2];
      clPosition[i].s[3] = tc.position[i][3];
      clVelocity[i].s[0] = tc.velocity[i][0];
      clVelocity[i].s[1] = tc.velocity[i][1];
      clVelocity[i].s[2] = tc.velocity[i][2];
      clVelocity[i].s[3] = tc.velocity[i][3];
    }

    auto particleIndexBuf =
        makeOpenCLReadBuffer(opencl.context(), clParticleIndex, err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create particleIndex buffer");
    auto particleIndexBackBuf =
        makeOpenCLWriteBuffer(opencl.context(), sizeof(cl_uint) * n, err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create particleIndexBack buffer");
    auto positionBuf = makeOpenCLReadBuffer(opencl.context(), clPosition, err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create position buffer");
    auto velocityBuf = makeOpenCLReadBuffer(opencl.context(), clVelocity, err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create velocity buffer");
    auto sortedPositionBuf =
        makeOpenCLWriteBuffer(opencl.context(), sizeof(cl_float4) * n, err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create sortedPosition buffer");
    auto sortedVelocityBuf =
        makeOpenCLWriteBuffer(opencl.context(), sizeof(cl_float4) * n, err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create sortedVelocity buffer");

    if (kernel.setArg(0, particleIndexBuf) != CL_SUCCESS ||
        kernel.setArg(1, particleIndexBackBuf) != CL_SUCCESS ||
        kernel.setArg(2, positionBuf) != CL_SUCCESS ||
        kernel.setArg(3, velocityBuf) != CL_SUCCESS ||
        kernel.setArg(4, sortedPositionBuf) != CL_SUCCESS ||
        kernel.setArg(5, sortedVelocityBuf) != CL_SUCCESS ||
        kernel.setArg(6, n) != CL_SUCCESS) {
      throw std::runtime_error("Failed to set kernel args for sortPostPass");
    }

    runOpenCL1DKernel(opencl.queue(), kernel, n, "sortPostPass");

    SortPostPassResult result;
    result.sortedPosition.resize(n);
    result.sortedVelocity.resize(n);
    result.particleIndexBack.resize(n);
    std::vector<cl_float4> clSortedPos(n), clSortedVel(n);
    std::vector<cl_uint> clIndexBack(n);
    if (opencl.queue().enqueueReadBuffer(sortedPositionBuf, CL_TRUE, 0,
                                         sizeof(cl_float4) * n,
                                         clSortedPos.data()) != CL_SUCCESS ||
        opencl.queue().enqueueReadBuffer(sortedVelocityBuf, CL_TRUE, 0,
                                         sizeof(cl_float4) * n,
                                         clSortedVel.data()) != CL_SUCCESS ||
        opencl.queue().enqueueReadBuffer(particleIndexBackBuf, CL_TRUE, 0,
                                         sizeof(cl_uint) * n,
                                         clIndexBack.data()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to read sortPostPass output buffers");
    }
    for (size_t i = 0; i < n; ++i) {
      result.sortedPosition[i] = {clSortedPos[i].s[0], clSortedPos[i].s[1],
                                  clSortedPos[i].s[2], clSortedPos[i].s[3]};
      result.sortedVelocity[i] = {clSortedVel[i].s[0], clSortedVel[i].s[1],
                                  clSortedVel[i].s[2], clSortedVel[i].s[3]};
      result.particleIndexBack[i] = clIndexBack[i];
    }
    return result;
  }
};

} // namespace SiberneticTest
