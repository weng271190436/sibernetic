#define CL_TARGET_OPENCL_VERSION 120

#include "../utils/common/backend_param_test.h"
#include "clear_buffers_test_common.h"
#include "metal_clear_buffers_runner.h"
#include "opencl_clear_buffers_runner.h"

using namespace SiberneticTest;

SIB_DEFINE_BACKEND_PARAM_TEST(ClearBuffersBackendParamTest,
                              ClearBuffersTestCommon, ClearBuffersRunner,
                              OpenCLClearBuffersRunner,
                              MetalClearBuffersRunner);
