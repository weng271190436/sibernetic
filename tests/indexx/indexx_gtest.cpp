#define CL_TARGET_OPENCL_VERSION 120

#include "../utils/backend_param_test.h"
#include "indexx_test_common.h"
#include "metal_indexx_runner.h"
#include "opencl_indexx_runner.h"

using namespace SiberneticTest;

SIB_DEFINE_BACKEND_PARAM_TEST(IndexxBackendParamTest, IndexxTestCommon,
                              IndexxRunner, OpenCLIndexxRunner,
                              MetalIndexxRunner);