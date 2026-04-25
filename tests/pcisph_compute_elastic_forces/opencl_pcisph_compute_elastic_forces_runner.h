#pragma once

#include <vector>

#include "../../src/kernels/PcisphComputeElasticForcesKernel.h"
#include "../utils/context/opencl_context.h"
#include "pcisph_compute_elastic_forces_test_common.h"

namespace SiberneticTest {

class OpenCLPcisphComputeElasticForcesRunner
    : public PcisphComputeElasticForcesRunner {
public:
  PcisphComputeElasticForcesResult
  run(const PcisphComputeElasticForcesCase &tc) override {
    auto input = tc.toInput();
    const uint32_t N = static_cast<uint32_t>(tc.sortedPosition.size());

    OpenCLKernelContext opencl;
    cl_int err = CL_SUCCESS;

    cl::Buffer sortedPosition(
        opencl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input.sortedPosition.size_bytes(),
        const_cast<Sibernetic::HostFloat4 *>(input.sortedPosition.data()),
        &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create sortedPosition buffer");

    cl::Buffer acceleration(
        opencl.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        input.acceleration.size_bytes(),
        const_cast<Sibernetic::HostFloat4 *>(input.acceleration.data()), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create acceleration buffer");

    cl::Buffer sortedParticleIdBySerialId(
        opencl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input.sortedParticleIdBySerialId.size_bytes(),
        const_cast<uint32_t *>(input.sortedParticleIdBySerialId.data()), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create sortedParticleIdBySerialId");

    cl::Buffer sortedCellAndSerialId(
        opencl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input.sortedCellAndSerialId.size_bytes(),
        const_cast<Sibernetic::HostUInt2 *>(input.sortedCellAndSerialId.data()),
        &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create sortedCellAndSerialId");

    Sibernetic::HostFloat4 dummyConn = {0.0f, 0.0f, 0.0f, 0.0f};
    cl::Buffer elasticConnectionsData(
        opencl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input.elasticConnectionsData.size_bytes() > 0
            ? input.elasticConnectionsData.size_bytes()
            : sizeof(dummyConn),
        input.elasticConnectionsData.size_bytes() > 0
            ? const_cast<Sibernetic::HostFloat4 *>(
                  input.elasticConnectionsData.data())
            : &dummyConn,
        &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create elasticConnectionsData");

    // muscleActivationSignal: may be empty.
    float dummySignal = 0.0f;
    cl::Buffer muscleActivationSignal(
        opencl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input.muscleCount > 0 ? input.muscleActivationSignal.size_bytes()
                              : sizeof(float),
        input.muscleCount > 0
            ? const_cast<float *>(input.muscleActivationSignal.data())
            : &dummySignal,
        &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create muscleActivationSignal");

    cl::Buffer originalPosition(
        opencl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input.originalPosition.size_bytes(),
        const_cast<Sibernetic::HostFloat4 *>(input.originalPosition.data()),
        &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create originalPosition buffer");

    auto args = Sibernetic::toOpenCLArgs(
        input, opencl.context(), sortedPosition, acceleration,
        sortedParticleIdBySerialId, sortedCellAndSerialId,
        elasticConnectionsData, muscleActivationSignal, originalPosition, N);

    cl::Kernel kernel(opencl.program(),
                      Sibernetic::kPcisphComputeElasticForcesKernelName, &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error(
          "Failed to create pcisph_computeElasticForces kernel");
    args.bind(kernel);

    const uint32_t threadCount = input.numOfElasticP;
    if (threadCount > 0) {
      if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                              cl::NDRange(threadCount),
                                              cl::NullRange) != CL_SUCCESS ||
          opencl.queue().finish() != CL_SUCCESS) {
        throw std::runtime_error(
            "Failed to execute pcisph_computeElasticForces kernel");
      }
    }

    PcisphComputeElasticForcesResult result;
    result.acceleration.resize(N);
    if (opencl.queue().enqueueReadBuffer(
            acceleration, CL_TRUE, 0, sizeof(float) * 4 * N,
            result.acceleration.data()) != CL_SUCCESS)
      throw std::runtime_error("Failed to read acceleration output buffer");

    return result;
  }
};

} // namespace SiberneticTest
