#define CL_TARGET_OPENCL_VERSION 120

#include "../utils/common/backend_param_test.h"
#include "metal_sort_post_pass_runner.h"
#include "opencl_sort_post_pass_runner.h"
#include "sort_post_pass_test_common.h"

using namespace SiberneticTest;

SIB_DEFINE_BACKEND_PARAM_TEST(SortPostPassBackendParamTest,
                              SortPostPassTestCommon, SortPostPassRunner,
                              OpenCLSortPostPassRunner,
                              MetalSortPostPassRunner);
