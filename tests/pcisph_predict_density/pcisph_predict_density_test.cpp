#define CL_TARGET_OPENCL_VERSION 120

#include "../utils/common/backend_param_test.h"
#include "metal_pcisph_predict_density_runner.h"
#include "opencl_pcisph_predict_density_runner.h"
#include "pcisph_predict_density_test_common.h"

using namespace SiberneticTest;

SIB_DEFINE_BACKEND_PARAM_TEST(PcisphPredictDensityBackendParamTest,
                              PcisphPredictDensityTestCommon,
                              PcisphPredictDensityRunner,
                              OpenCLPcisphPredictDensityRunner,
                              MetalPcisphPredictDensityRunner);
