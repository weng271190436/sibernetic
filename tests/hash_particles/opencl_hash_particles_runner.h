#pragma once

#include <stdexcept>
#include <vector>

#include "../utils/opencl_helpers.h"
#include "hash_particles_test_common.h"

namespace SiberneticTest {

class OpenCLHashParticlesRunner : public HashParticlesRunner {
public:
  HashParticlesResult run(const HashParticlesCase &tc) override {
    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;
    cl::Kernel kernel(opencl.program(), "hashParticles", &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create kernel: hashParticles");
    }

    std::vector<cl_float4> positions = toCLFloat4Vector(tc.positions);

    const cl_uint particleCount = static_cast<cl_uint>(positions.size());
    auto positionBuf = makeOpenCLReadBuffer(opencl.context(), positions, err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create position buffer");
    }
    auto particleIndexBuf = makeOpenCLWriteBuffer(
        opencl.context(), sizeof(cl_uint2) * particleCount, err);
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

    if (kernel.setArg(0, positionBuf) != CL_SUCCESS ||
        kernel.setArg(1, gridCellsX) != CL_SUCCESS ||
        kernel.setArg(2, gridCellsY) != CL_SUCCESS ||
        kernel.setArg(3, gridCellsZ) != CL_SUCCESS ||
        kernel.setArg(4, hashGridCellSizeInv) != CL_SUCCESS ||
        kernel.setArg(5, xmin) != CL_SUCCESS ||
        kernel.setArg(6, ymin) != CL_SUCCESS ||
        kernel.setArg(7, zmin) != CL_SUCCESS ||
        kernel.setArg(8, particleIndexBuf) != CL_SUCCESS ||
        kernel.setArg(9, particleCount) != CL_SUCCESS) {
      throw std::runtime_error("Failed to set kernel args for hashParticles");
    }

    runOpenCL1DKernel(opencl.queue(), kernel, particleCount, "hashParticles");

    HashParticlesResult result;
    std::vector<cl_uint2> clResult(particleCount);
    if (opencl.queue().enqueueReadBuffer(particleIndexBuf, CL_TRUE, 0,
                                         sizeof(cl_uint2) * particleCount,
                                         clResult.data()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to read particleIndex buffer");
    }
    result.particleIndex = toHostUInt2Vector(clResult);
    return result;
  }
};

} // namespace SiberneticTest
