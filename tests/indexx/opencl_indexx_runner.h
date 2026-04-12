#pragma once

#include <vector>

#include "../../src/kernels/IndexxKernel.h"
#include "../utils/buffer/opencl_buffer_utils.h"
#include "../utils/context/opencl_context.h"
#include "../utils/convert/opencl_convert_utils.h"
#include "indexx_test_common.h"

namespace SiberneticTest {

class OpenCLIndexxRunner : public IndexxRunner {
public:
  IndexxResult run(const IndexxCase &tc) override {
    const cl_uint particleCount = static_cast<cl_uint>(tc.particleIndex.size());
    const size_t gridCellIndexCount = static_cast<size_t>(tc.gridCellCount) + 1u;
    const size_t threadCount = gridCellIndexCount;

    std::vector<cl_uint2> clParticleIndex = toCLUInt2Vector(tc.particleIndex);

    Sibernetic::IndexxInput input{};
    input.particleIndex =
        reinterpret_cast<const uint32_t *>(clParticleIndex.data());
    input.particleCount = particleCount;
    input.gridCellCount = tc.gridCellCount;

    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;
    cl::Buffer outputGridCellIndex(opencl.context(), CL_MEM_WRITE_ONLY,
                                   sizeof(uint32_t) * gridCellIndexCount,
                                   nullptr, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create output gridCellIndex buffer");
    }

    auto args =
        Sibernetic::toOpenCLArgs(input, opencl.context(), outputGridCellIndex);

    cl::Kernel kernel(opencl.program(), Sibernetic::kIndexxKernelName, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create indexx kernel");
    }
    args.bind(kernel);

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(threadCount),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error("Failed to execute indexx kernel");
    }

    IndexxResult result;
    result.gridCellIndex.resize(gridCellIndexCount);
    if (opencl.queue().enqueueReadBuffer(
            outputGridCellIndex, CL_TRUE, 0,
            sizeof(uint32_t) * gridCellIndexCount,
            result.gridCellIndex.data()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to read gridCellIndex output buffer");
    }
    return result;
  }
};

} // namespace SiberneticTest