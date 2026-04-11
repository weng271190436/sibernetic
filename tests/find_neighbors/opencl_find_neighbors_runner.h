#pragma once

#include <vector>

#include "../utils/arg/opencl_arg_binding.h"
#include "../utils/convert/opencl_convert_utils.h"
#include "find_neighbors_test_common.h"

namespace SiberneticTest {

class OpenCLFindNeighborsRunner : public FindNeighborsRunner {
public:
  FindNeighborsResult run(const FindNeighborsCase &tc) override {
    const cl_uint particleCount =
        static_cast<cl_uint>(tc.sortedPosition.size());
    const size_t neighborCount = static_cast<size_t>(particleCount) * 32u;

    std::vector<cl_uint> clGridCellIndex(tc.gridCellIndexFixedUp.begin(),
                                         tc.gridCellIndexFixedUp.end());
    std::vector<cl_float4> clSortedPosition =
        toCLFloat4Vector(tc.sortedPosition);
    FindNeighborsResult result;
    auto outNeighborMap =
        makeCLOutputFieldBinding<FindNeighborsResult, cl_float2,
                                 std::array<float, 2>>(
            13, neighborCount, &FindNeighborsResult::neighborMap,
            static_cast<std::vector<std::array<float, 2>> (*)(
                const std::vector<cl_float2> &)>(toHostFloat2ArrayVector));
    runCLKernelSpecAndStore(
        "findNeighbors", particleCount,
        {
            CLScalarArg::make<cl_uint>(2, tc.gridCellCount),
            CLScalarArg::make<cl_uint>(3, tc.gridCellsX),
            CLScalarArg::make<cl_uint>(4, tc.gridCellsY),
            CLScalarArg::make<cl_uint>(5, tc.gridCellsZ),
            CLScalarArg::make<cl_float>(6, tc.h),
            CLScalarArg::make<cl_float>(7, tc.hashGridCellSize),
            CLScalarArg::make<cl_float>(8, tc.hashGridCellSizeInv),
            CLScalarArg::make<cl_float>(9, tc.simulationScale),
            CLScalarArg::make<cl_float>(10, tc.xmin),
            CLScalarArg::make<cl_float>(11, tc.ymin),
            CLScalarArg::make<cl_float>(12, tc.zmin),
            CLScalarArg::make<cl_uint>(14, particleCount),
        },
        {
            CLInputBuffer::make<cl_uint>(0, clGridCellIndex),
            CLInputBuffer::make<cl_float4>(1, clSortedPosition),
        },
        result, outNeighborMap);

    return result;
  }
};

} // namespace SiberneticTest
