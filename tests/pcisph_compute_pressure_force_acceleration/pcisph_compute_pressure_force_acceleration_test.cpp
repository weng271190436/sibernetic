#define CL_TARGET_OPENCL_VERSION 120

#include "../utils/common/backend_param_test.h"
#include "metal_pcisph_compute_pressure_force_acceleration_runner.h"
#include "opencl_pcisph_compute_pressure_force_acceleration_runner.h"
#include "pcisph_compute_pressure_force_acceleration_test_common.h"

using namespace SiberneticTest;

SIB_DEFINE_BACKEND_PARAM_TEST(
    PcisphComputePressureForceAccelerationBackendParamTest,
    PcisphComputePressureForceAccelerationTestCommon,
    PcisphComputePressureForceAccelerationRunner,
    OpenCLPcisphComputePressureForceAccelerationRunner,
    MetalPcisphComputePressureForceAccelerationRunner);
