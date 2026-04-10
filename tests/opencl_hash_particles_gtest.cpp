#define CL_TARGET_OPENCL_VERSION 120

#include <vector>

#include <gtest/gtest.h>

#include "opencl_test_utils.h"

using namespace SiberneticTest;

namespace {

struct HashParticlesCase {
  const char *name;
  std::vector<cl_float4> positions;
  cl_uint gridCellsX;
  cl_uint gridCellsY;
  cl_uint gridCellsZ;
  cl_float hashGridCellSizeInv;
  cl_float xmin;
  cl_float ymin;
  cl_float zmin;
  std::vector<cl_uint> expectedCellIds;
};

struct HashParticlesResult {
  std::vector<cl_uint2> particleIndex;
};

class HashParticlesRunner {
public:
  virtual ~HashParticlesRunner() = default;
  virtual HashParticlesResult run(const HashParticlesCase &tc) = 0;
};

class OpenCLHashParticlesRunner : public HashParticlesRunner {
public:
  OpenCLHashParticlesRunner(cl::Context &context, cl::CommandQueue &queue,
                            cl::Program &program)
      : context_(context), queue_(queue), program_(program) {}

  HashParticlesResult run(const HashParticlesCase &tc) override {
    cl_int err = CL_SUCCESS;
    cl::Kernel kernel(program_, "hashParticles", &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create kernel: hashParticles");
    }

    std::vector<cl_float4> positions = tc.positions;
    HashParticlesResult result;
    result.particleIndex.resize(positions.size());

    cl::Buffer positionBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_float4) * positions.size(),
                              positions.data(), &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create position buffer");
    }

    cl::Buffer particleIndexBuffer(
        context_, CL_MEM_WRITE_ONLY,
        sizeof(cl_uint2) * result.particleIndex.size(), nullptr, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create particleIndex buffer");
    }

    const cl_uint particleCount = static_cast<cl_uint>(positions.size());

    if (kernel.setArg(0, positionBuffer) != CL_SUCCESS ||
        kernel.setArg(1, tc.gridCellsX) != CL_SUCCESS ||
        kernel.setArg(2, tc.gridCellsY) != CL_SUCCESS ||
        kernel.setArg(3, tc.gridCellsZ) != CL_SUCCESS ||
        kernel.setArg(4, tc.hashGridCellSizeInv) != CL_SUCCESS ||
        kernel.setArg(5, tc.xmin) != CL_SUCCESS ||
        kernel.setArg(6, tc.ymin) != CL_SUCCESS ||
        kernel.setArg(7, tc.zmin) != CL_SUCCESS ||
        kernel.setArg(8, particleIndexBuffer) != CL_SUCCESS ||
        kernel.setArg(9, particleCount) != CL_SUCCESS) {
      throw std::runtime_error("Failed to set kernel args for hashParticles");
    }

    if (queue_.enqueueNDRangeKernel(kernel, cl::NullRange,
                                    cl::NDRange(particleCount),
                                    cl::NullRange) != CL_SUCCESS ||
        queue_.finish() != CL_SUCCESS) {
      throw std::runtime_error("Failed to execute hashParticles kernel");
    }

    if (queue_.enqueueReadBuffer(particleIndexBuffer, CL_TRUE, 0,
                                 sizeof(cl_uint2) * result.particleIndex.size(),
                                 result.particleIndex.data()) != CL_SUCCESS) {
      throw std::runtime_error("Failed to read particleIndex buffer");
    }

    return result;
  }

private:
  cl::Context &context_;
  cl::CommandQueue &queue_;
  cl::Program &program_;
};

cl_float4 makeFloat4(cl_float x, cl_float y, cl_float z, cl_float w = 0.0f) {
  cl_float4 v;
  v.s[0] = x;
  v.s[1] = y;
  v.s[2] = z;
  v.s[3] = w;
  return v;
}

class OpenCLHashParticlesParamTest
    : public OpenCLKernelFixture,
      public ::testing::WithParamInterface<HashParticlesCase> {};

} // namespace

TEST_P(OpenCLHashParticlesParamTest, ProducesExpectedCellAndSerialIds) {
  const HashParticlesCase &tc = GetParam();
  ASSERT_EQ(tc.positions.size(), tc.expectedCellIds.size());

  OpenCLHashParticlesRunner runner(context, queue, program);
  HashParticlesResult result;
  ASSERT_NO_THROW(result = runner.run(tc));

  const cl_uint particleCount =
      static_cast<cl_uint>(result.particleIndex.size());
  for (cl_uint i = 0; i < particleCount; ++i) {
    EXPECT_EQ(result.particleIndex[i].s[0], tc.expectedCellIds[i]);
    EXPECT_EQ(result.particleIndex[i].s[1], i);
  }
}

INSTANTIATE_TEST_SUITE_P(
    HashParticlesCases, OpenCLHashParticlesParamTest,
    ::testing::Values(
        HashParticlesCase{
            "UnitCellSize_4x4x4",
            {makeFloat4(0.1f, 0.1f, 0.1f), makeFloat4(1.2f, 0.1f, 0.1f),
             makeFloat4(0.2f, 1.7f, 0.1f), makeFloat4(2.8f, 3.1f, 1.0f)},
            4,
            4,
            4,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            {
                0, // p0 -> cell (0,0,0)
                1, // p1 -> cell (1,0,0)
                4, // p2 -> cell (0,1,0)
                30 // p3 -> cell (2,3,1)
            }},
        HashParticlesCase{
            "HalfCellSize_8x8x8",
            {makeFloat4(0.1f, 0.1f, 0.1f), makeFloat4(0.6f, 0.1f, 0.1f),
             makeFloat4(0.1f, 0.6f, 0.1f), makeFloat4(1.1f, 1.1f, 0.6f)},
            8,
            8,
            8,
            2.0f,
            0.0f,
            0.0f,
            0.0f,
            {
                0, // p0 -> cell (0,0,0)
                1, // p1 -> cell (1,0,0)
                8, // p2 -> cell (0,1,0)
                82 // p3 -> cell (2,2,1)
            }}),
    [](const ::testing::TestParamInfo<HashParticlesCase> &info) {
      return info.param.name;
    });
