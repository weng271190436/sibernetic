#define CL_TARGET_OPENCL_VERSION 120

#include "../utils/common/backend_param_test.h"
#include "metal_pcisph_integrate_runner.h"
#include "opencl_pcisph_integrate_runner.h"
#include "pcisph_integrate_test_common.h"

using namespace SiberneticTest;

SIB_DEFINE_BACKEND_PARAM_TEST(PcisphIntegrateBackendParamTest,
                              PcisphIntegrateTestCommon,
                              PcisphIntegrateRunner,
                              OpenCLPcisphIntegrateRunner,
                              MetalPcisphIntegrateRunner);
