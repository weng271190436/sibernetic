#pragma once

#include <vector>

#include "../utils/arg/opencl_arg_binding.h"
#include "../utils/convert/opencl_convert_utils.h"
#include "sort_post_pass_test_common.h"

namespace SiberneticTest {

inline std::vector<uint32_t>
convertSortPostPassIndexBack(const std::vector<cl_uint> &src) {
  return toHostUInt32Vector(src);
}

inline std::vector<HostFloat4>
convertSortPostPassFloat4(const std::vector<cl_float4> &src) {
  return toHostFloat4Vector(src);
}

class OpenCLSortPostPassRunner : public SortPostPassRunner {
public:
  SortPostPassResult run(const SortPostPassCase &tc) override {
    const cl_uint particleCount = static_cast<cl_uint>(tc.particleIndex.size());

    std::vector<cl_uint2> clParticleIndex = toCLUInt2Vector(tc.particleIndex);
    std::vector<cl_float4> clPosition = toCLFloat4Vector(tc.position);
    std::vector<cl_float4> clVelocity = toCLFloat4Vector(tc.velocity);

    SortPostPassResult result;
    auto outIndexBack =
        makeCLOutputFieldBinding<SortPostPassResult, cl_uint, uint32_t>(
            1, particleCount, &SortPostPassResult::particleIndexBack,
            convertSortPostPassIndexBack);
    auto outSortedPos =
        makeCLOutputFieldBinding<SortPostPassResult, cl_float4, HostFloat4>(
            4, particleCount, &SortPostPassResult::sortedPosition,
            convertSortPostPassFloat4);
    auto outSortedVel =
        makeCLOutputFieldBinding<SortPostPassResult, cl_float4, HostFloat4>(
            5, particleCount, &SortPostPassResult::sortedVelocity,
            convertSortPostPassFloat4);

    runCLKernelSpecAndStore(
        "sortPostPass", particleCount,
        {
            CLScalarArg::make<cl_uint>(6, particleCount),
        },
        {
            CLInputBuffer::make<cl_uint2>(0, clParticleIndex),
            CLInputBuffer::make<cl_float4>(2, clPosition),
            CLInputBuffer::make<cl_float4>(3, clVelocity),
        },
        result, outIndexBack, outSortedPos, outSortedVel);

    return result;
  }
};

} // namespace SiberneticTest
