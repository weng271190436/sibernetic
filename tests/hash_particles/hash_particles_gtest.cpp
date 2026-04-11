#define CL_TARGET_OPENCL_VERSION 120

#include "../utils/backend_param_test.h"
#include "hash_particles_test_common.h"
#include "metal_hash_particles_runner.h"
#include "opencl_hash_particles_runner.h"

using namespace SiberneticTest;

SIB_DEFINE_BACKEND_PARAM_TEST(
    HashParticlesBackendParamTest, HashParticlesTestCommon, HashParticlesRunner,
    OpenCLHashParticlesRunner, MetalHashParticlesRunner);
