#pragma once

#include <vector>

#include "../../src/kernels/PcisphIntegrateKernel.h"
#include "../utils/context/opencl_context.h"
#include "pcisph_integrate_test_common.h"

namespace SiberneticTest {

class OpenCLPcisphIntegrateRunner : public PcisphIntegrateRunner {
public:
  PcisphIntegrateResult run(const PcisphIntegrateCase &tc) override {
    auto input = tc.toInput();
    const cl_uint N = static_cast<cl_uint>(input.particleCount);

    OpenCLKernelContext opencl;

    cl_int err = CL_SUCCESS;

    // acceleration is read/write (3×N float4)
    cl::Buffer inOutAcceleration(
        opencl.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        input.acceleration.size_bytes(),
        const_cast<Sibernetic::HostFloat4 *>(input.acceleration.data()), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create acceleration buffer");

    // sortedPosition is read/write
    cl::Buffer inOutSortedPosition(
        opencl.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        input.sortedPosition.size_bytes(),
        const_cast<Sibernetic::HostFloat4 *>(input.sortedPosition.data()),
        &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create sortedPosition buffer");

    // originalPosition is read/write
    cl::Buffer inOutOriginalPosition(
        opencl.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        input.originalPosition.size_bytes(),
        const_cast<Sibernetic::HostFloat4 *>(input.originalPosition.data()),
        &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create originalPosition buffer");

    // velocity is read/write
    cl::Buffer inOutVelocity(
        opencl.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        input.velocity.size_bytes(),
        const_cast<Sibernetic::HostFloat4 *>(input.velocity.data()), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create velocity buffer");

    auto args = Sibernetic::toOpenCLArgs(input, opencl.context(),
                                         inOutAcceleration, inOutSortedPosition,
                                         inOutOriginalPosition, inOutVelocity);

    cl::Kernel kernel(opencl.program(), Sibernetic::kPcisphIntegrateKernelName,
                      &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create pcisph_integrate kernel");
    args.bind(kernel);

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(N),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error("Failed to execute pcisph_integrate kernel");
    }

    PcisphIntegrateResult result;

    // Read back all output buffers.
    result.acceleration.resize(3 * N);
    if (opencl.queue().enqueueReadBuffer(
            inOutAcceleration, CL_TRUE, 0, sizeof(float) * 4 * 3 * N,
            result.acceleration.data()) != CL_SUCCESS)
      throw std::runtime_error("Failed to read acceleration output buffer");

    result.sortedPosition.resize(N);
    if (opencl.queue().enqueueReadBuffer(
            inOutSortedPosition, CL_TRUE, 0, sizeof(float) * 4 * N,
            result.sortedPosition.data()) != CL_SUCCESS)
      throw std::runtime_error("Failed to read sortedPosition output buffer");

    result.originalPosition.resize(N);
    if (opencl.queue().enqueueReadBuffer(
            inOutOriginalPosition, CL_TRUE, 0, sizeof(float) * 4 * N,
            result.originalPosition.data()) != CL_SUCCESS)
      throw std::runtime_error("Failed to read originalPosition output buffer");

    result.velocity.resize(N);
    if (opencl.queue().enqueueReadBuffer(inOutVelocity, CL_TRUE, 0,
                                         sizeof(float) * 4 * N,
                                         result.velocity.data()) != CL_SUCCESS)
      throw std::runtime_error("Failed to read velocity output buffer");

    return result;
  }
};

} // namespace SiberneticTest
