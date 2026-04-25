#define CL_TARGET_OPENCL_VERSION 120

#include "../utils/common/backend_param_test.h"
#include "compute_interaction_with_membranes_test_common.h"
#include "metal_compute_interaction_with_membranes_runner.h"
#include "opencl_compute_interaction_with_membranes_runner.h"

using namespace SiberneticTest;

SIB_DEFINE_BACKEND_PARAM_TEST(
    ComputeInteractionWithMembranesBackendParamTest,
    ComputeInteractionWithMembranesTestCommon,
    ComputeInteractionWithMembranesRunner,
    OpenCLComputeInteractionWithMembranesRunner,
    MetalComputeInteractionWithMembranesRunner);

SIB_DEFINE_BACKEND_PARAM_TEST(
    ComputeInteractionWithMembranesFinalizeBackendParamTest,
    ComputeInteractionWithMembranesFinalizeTestCommon,
    ComputeInteractionWithMembranesFinalizeRunner,
    OpenCLComputeInteractionWithMembranesFinalizeRunner,
    MetalComputeInteractionWithMembranesFinalizeRunner);
