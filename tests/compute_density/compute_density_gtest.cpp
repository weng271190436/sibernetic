#define CL_TARGET_OPENCL_VERSION 120

#include "../utils/common/backend_param_test.h"
#include "compute_density_test_common.h"
#include "metal_compute_density_runner.h"
#include "opencl_compute_density_runner.h"

using namespace SiberneticTest;

SIB_DEFINE_BACKEND_PARAM_TEST(ComputeDensityBackendParamTest,
                              ComputeDensityTestCommon, ComputeDensityRunner,
                              OpenCLComputeDensityRunner,
                              MetalComputeDensityRunner);
