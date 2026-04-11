#pragma once

#include <stdexcept>
#include <vector>

#include "../utils/opencl_context.h"
#include "../utils/opencl_helpers.h"
#include "find_neighbors_test_common.h"

namespace SiberneticTest {

class OpenCLFindNeighborsRunner : public FindNeighborsRunner {
public:
  FindNeighborsResult run(const FindNeighborsCase &tc) override {
    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;
    cl::Kernel kernel(opencl.program(), "findNeighbors", &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create kernel: findNeighbors");
    }

    const cl_uint particleCount = static_cast<cl_uint>(tc.sortedPosition.size());
    const size_t neighborCount = static_cast<size_t>(particleCount) * 32u;

    std::vector<cl_uint> clGridCellIndex(tc.gridCellIndexFixedUp.begin(),
                                         tc.gridCellIndexFixedUp.end());
    std::vector<cl_float4> clSortedPosition = toCLFloat4Vector(tc.sortedPosition);

    auto gridCellIndexBuf =
        makeOpenCLReadBuffer(opencl.context(), clGridCellIndex, err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create gridCellIndexFixedUp buffer");
    }
    auto sortedPositionBuf =
        makeOpenCLReadBuffer(opencl.context(), clSortedPosition, err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create sortedPosition buffer");
    }
    auto neighborMapBuf =
        makeOpenCLWriteBuffer(opencl.context(), sizeof(cl_float2) * neighborCount, err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create neighborMap buffer");
    }

    const cl_uint gridCellCount = tc.gridCellCount;
    const cl_uint gridCellsX = tc.gridCellsX;
    const cl_uint gridCellsY = tc.gridCellsY;
    const cl_uint gridCellsZ = tc.gridCellsZ;
    const cl_float h = tc.h;
    const cl_float hashGridCellSize = tc.hashGridCellSize;
    const cl_float hashGridCellSizeInv = tc.hashGridCellSizeInv;
    const cl_float simulationScale = tc.simulationScale;
    const cl_float xmin = tc.xmin;
    const cl_float ymin = tc.ymin;
    const cl_float zmin = tc.zmin;

    if (kernel.setArg(0, gridCellIndexBuf) != CL_SUCCESS ||
        kernel.setArg(1, sortedPositionBuf) != CL_SUCCESS ||
        kernel.setArg(2, gridCellCount) != CL_SUCCESS ||
        kernel.setArg(3, gridCellsX) != CL_SUCCESS ||
        kernel.setArg(4, gridCellsY) != CL_SUCCESS ||
        kernel.setArg(5, gridCellsZ) != CL_SUCCESS ||
        kernel.setArg(6, h) != CL_SUCCESS ||
        kernel.setArg(7, hashGridCellSize) != CL_SUCCESS ||
        kernel.setArg(8, hashGridCellSizeInv) != CL_SUCCESS ||
        kernel.setArg(9, simulationScale) != CL_SUCCESS ||
        kernel.setArg(10, xmin) != CL_SUCCESS ||
        kernel.setArg(11, ymin) != CL_SUCCESS ||
        kernel.setArg(12, zmin) != CL_SUCCESS ||
        kernel.setArg(13, neighborMapBuf) != CL_SUCCESS ||
        kernel.setArg(14, particleCount) != CL_SUCCESS) {
      throw std::runtime_error("Failed to set kernel args for findNeighbors");
    }

    runOpenCL1DKernel(opencl.queue(), kernel, particleCount, "findNeighbors");

    std::vector<cl_float2> clNeighborMap(neighborCount);
    if (opencl.queue().enqueueReadBuffer(neighborMapBuf, CL_TRUE, 0,
                                         sizeof(cl_float2) * neighborCount,
                                         clNeighborMap.data()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to read neighborMap buffer");
    }

    FindNeighborsResult result;
    result.neighborMap.resize(neighborCount);
    for (size_t i = 0; i < neighborCount; ++i) {
      result.neighborMap[i] = {clNeighborMap[i].s[0], clNeighborMap[i].s[1]};
    }
    return result;
  }
};

} // namespace SiberneticTest
