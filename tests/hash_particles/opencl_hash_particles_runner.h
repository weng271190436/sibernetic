#pragma once

#include <stdexcept>
#include <vector>

#include "../utils/opencl_test_utils.h"
#include "hash_particles_test_common.h"

namespace SiberneticTest {

class OpenCLHashParticlesRunner : public HashParticlesRunner {
public:
  OpenCLHashParticlesRunner() = default;

  HashParticlesResult run(const HashParticlesCase &tc) override {
    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;
    cl::Kernel kernel(opencl.program(), "hashParticles", &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create kernel: hashParticles");
    }

    std::vector<cl_float4> positions(tc.positions.size());
    for (size_t i = 0; i < tc.positions.size(); ++i) {
      positions[i].s[0] = tc.positions[i][0];
      positions[i].s[1] = tc.positions[i][1];
      positions[i].s[2] = tc.positions[i][2];
      positions[i].s[3] = tc.positions[i][3];
    }

    HashParticlesResult result;
    result.particleIndex.resize(positions.size());

    cl::Buffer positionBuffer(opencl.context(),
                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                  sizeof(cl_float4) * positions.size(),
                  positions.data(), &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create position buffer");
    }

    cl::Buffer particleIndexBuffer(
      opencl.context(), CL_MEM_WRITE_ONLY,
        sizeof(cl_uint2) * result.particleIndex.size(), nullptr, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create particleIndex buffer");
    }

    const cl_uint gridCellsX = static_cast<cl_uint>(tc.gridCellsX);
    const cl_uint gridCellsY = static_cast<cl_uint>(tc.gridCellsY);
    const cl_uint gridCellsZ = static_cast<cl_uint>(tc.gridCellsZ);
    const cl_float hashGridCellSizeInv =
        static_cast<cl_float>(tc.hashGridCellSizeInv);
    const cl_float xmin = static_cast<cl_float>(tc.xmin);
    const cl_float ymin = static_cast<cl_float>(tc.ymin);
    const cl_float zmin = static_cast<cl_float>(tc.zmin);
    const cl_uint particleCount = static_cast<cl_uint>(positions.size());

    if (kernel.setArg(0, positionBuffer) != CL_SUCCESS ||
        kernel.setArg(1, gridCellsX) != CL_SUCCESS ||
        kernel.setArg(2, gridCellsY) != CL_SUCCESS ||
        kernel.setArg(3, gridCellsZ) != CL_SUCCESS ||
        kernel.setArg(4, hashGridCellSizeInv) != CL_SUCCESS ||
        kernel.setArg(5, xmin) != CL_SUCCESS ||
        kernel.setArg(6, ymin) != CL_SUCCESS ||
        kernel.setArg(7, zmin) != CL_SUCCESS ||
        kernel.setArg(8, particleIndexBuffer) != CL_SUCCESS ||
        kernel.setArg(9, particleCount) != CL_SUCCESS) {
      throw std::runtime_error("Failed to set kernel args for hashParticles");
    }

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(particleCount),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error("Failed to execute hashParticles kernel");
    }

    std::vector<cl_uint2> clResult(result.particleIndex.size());
    if (opencl.queue().enqueueReadBuffer(particleIndexBuffer, CL_TRUE, 0,
                                         sizeof(cl_uint2) * clResult.size(),
                                         clResult.data()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to read particleIndex buffer");
    }

    for (size_t i = 0; i < clResult.size(); ++i) {
      result.particleIndex[i][0] = clResult[i].s[0];
      result.particleIndex[i][1] = clResult[i].s[1];
    }

    return result;
  }
};

} // namespace SiberneticTest
