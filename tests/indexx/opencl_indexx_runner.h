#pragma once

#include <vector>

#include "../utils/arg/opencl_arg_binding.h"
#include "../utils/convert/opencl_convert_utils.h"
#include "indexx_test_common.h"

namespace SiberneticTest {
inline std::vector<uint32_t>
convertIndexxGridCellIndex(const std::vector<cl_uint> &src) {
  return toHostUInt32Vector(src);
}

class OpenCLIndexxRunner : public IndexxRunner {
public:
  IndexxResult run(const IndexxCase &tc) override {
    std::vector<cl_uint2> clParticleIndex = toCLUInt2Vector(tc.particleIndex);

    const cl_uint gridCellCount = tc.gridCellCount;
    const cl_uint particleCount = static_cast<cl_uint>(tc.particleIndex.size());
    const size_t outputCount = static_cast<size_t>(gridCellCount) + 1u;

    IndexxResult result;
    auto outGridCellIndex =
        makeCLOutputFieldBinding<IndexxResult, cl_uint, uint32_t>(
            2, outputCount, &IndexxResult::gridCellIndex,
            convertIndexxGridCellIndex);

    runCLKernelSpecAndStore(
        "indexx", outputCount,
        {
            CLScalarArg::make<cl_uint>(1, gridCellCount),
            CLScalarArg::make<cl_uint>(3, particleCount),
        },
        {
            CLInputBuffer::make<cl_uint2>(0, clParticleIndex),
        },
        result, outGridCellIndex);

    return result;
  }
};

} // namespace SiberneticTest