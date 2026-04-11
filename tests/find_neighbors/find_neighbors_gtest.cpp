#define CL_TARGET_OPENCL_VERSION 120

#include "../utils/common/backend_param_test.h"
#include "find_neighbors_test_common.h"
#include "metal_find_neighbors_runner.h"
#include "opencl_find_neighbors_runner.h"

using namespace SiberneticTest;

SIB_DEFINE_BACKEND_PARAM_TEST(FindNeighborsBackendParamTest,
                              FindNeighborsTestCommon, FindNeighborsRunner,
                              OpenCLFindNeighborsRunner,
                              MetalFindNeighborsRunner);
