#pragma once

#include <stdexcept>
#include <vector>

#include "../../src/kernels/ComputeInteractionWithMembranesKernel.h"
#include "../utils/context/opencl_context.h"
#include "compute_interaction_with_membranes_test_common.h"

namespace SiberneticTest {

// ── computeInteractionWithMembranes OpenCL runner ───────────────────────────

class OpenCLComputeInteractionWithMembranesRunner
    : public ComputeInteractionWithMembranesRunner {
public:
  ComputeInteractionWithMembranesResult
  run(const ComputeInteractionWithMembranesCase &tc) override {
    auto input = tc.toInput();
    const uint32_t N = input.particleCount;

    OpenCLKernelContext opencl;
    cl_int err = CL_SUCCESS;

    cl::Buffer positionBuf(
        opencl.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        input.position.size_bytes(),
        const_cast<Sibernetic::HostFloat4 *>(input.position.data()), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create position buffer");

    cl::Buffer velocityBuf(
        opencl.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        input.velocity.size_bytes(),
        const_cast<Sibernetic::HostFloat4 *>(input.velocity.data()), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create velocity buffer");

    // sortedPosition (arg 2): unused but required by OpenCL signature.
    float dummy4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    cl::Buffer sortedPositionBuf(opencl.context(),
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(dummy4), &dummy4, &err);

    // particleIndex (arg 3) = sortedCellAndSerialId
    cl::Buffer particleIndexBuf(
        opencl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input.sortedCellAndSerialId.size_bytes(),
        const_cast<Sibernetic::HostUInt2 *>(input.sortedCellAndSerialId.data()),
        &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create particleIndex buffer");

    // particleIndexBack (arg 4) = sortedParticleIdBySerialId
    cl::Buffer particleIndexBackBuf(
        opencl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input.sortedParticleIdBySerialId.size_bytes(),
        const_cast<uint32_t *>(input.sortedParticleIdBySerialId.data()), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create particleIndexBack buffer");

    cl::Buffer neighborMapBuf(
        opencl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input.neighborMap.size_bytes(),
        const_cast<Sibernetic::HostFloat2 *>(input.neighborMap.data()), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create neighborMap buffer");

    // particleMembranesList may be empty.
    int32_t dummyInt = -1;
    cl::Buffer memListBuf(
        opencl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input.particleMembranesList.empty()
            ? sizeof(dummyInt)
            : input.particleMembranesList.size_bytes(),
        input.particleMembranesList.empty()
            ? reinterpret_cast<void *>(&dummyInt)
            : const_cast<int32_t *>(input.particleMembranesList.data()),
        &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create particleMembranesList buffer");

    cl::Buffer memDataBuf(
        opencl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input.membraneData.empty() ? sizeof(dummyInt)
                                   : input.membraneData.size_bytes(),
        input.membraneData.empty()
            ? reinterpret_cast<void *>(&dummyInt)
            : const_cast<int32_t *>(input.membraneData.data()),
        &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create membraneData buffer");

    Sibernetic::ComputeInteractionWithMembranesOpenCLArgs args{};
    args.position = positionBuf;
    args.velocity = velocityBuf;
    args.sortedPosition = sortedPositionBuf;
    args.particleIndex = particleIndexBuf;
    args.particleIndexBack = particleIndexBackBuf;
    args.neighborMap = neighborMapBuf;
    args.particleMembranesList = memListBuf;
    args.membraneData = memDataBuf;
    args.particleCount = static_cast<int32_t>(N);
    args.numOfElasticP = 0; // unused by kernel
    args.r0 = input.r0;

    cl::Kernel kernel(opencl.program(),
                      Sibernetic::kComputeInteractionWithMembranesKernelName,
                      &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error(
          "Failed to create computeInteractionWithMembranes kernel");
    args.bind(kernel);

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(N),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to execute computeInteractionWithMembranes kernel");
    }

    ComputeInteractionWithMembranesResult result;
    result.position.resize(2 * N);
    if (opencl.queue().enqueueReadBuffer(positionBuf, CL_TRUE, 0,
                                         sizeof(float) * 4 * 2 * N,
                                         result.position.data()) != CL_SUCCESS)
      throw std::runtime_error("Failed to read position output buffer");

    return result;
  }
};

// ── computeInteractionWithMembranes_finalize OpenCL runner ──────────────────

class OpenCLComputeInteractionWithMembranesFinalizeRunner
    : public ComputeInteractionWithMembranesFinalizeRunner {
public:
  ComputeInteractionWithMembranesFinalizeResult
  run(const ComputeInteractionWithMembranesFinalizeCase &tc) override {
    auto input = tc.toInput();
    const uint32_t N = input.particleCount;

    OpenCLKernelContext opencl;
    cl_int err = CL_SUCCESS;

    cl::Buffer positionBuf(
        opencl.context(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        input.position.size_bytes(),
        const_cast<Sibernetic::HostFloat4 *>(input.position.data()), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("Failed to create position buffer");

    // velocity (arg 1): unused but required by OpenCL signature.
    float dummy4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    cl::Buffer velocityBuf(opencl.context(),
                           CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(dummy4), &dummy4, &err);

    // Build sortedCellAndSerialId from sortedParticleIdBySerialId.
    // For the finalize kernel, particleIndex[sortedId].y must map back to
    // serialId. With identity sort: particleIndex[i] = (0, i).
    std::vector<Sibernetic::HostUInt2> cellAndSerial(N);
    for (uint32_t i = 0; i < N; ++i) {
      cellAndSerial[i] = {0, i};
    }
    cl::Buffer particleIndexBuf(
        opencl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        cellAndSerial.size() * sizeof(Sibernetic::HostUInt2),
        cellAndSerial.data(), &err);

    cl::Buffer particleIndexBackBuf(
        opencl.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        input.sortedParticleIdBySerialId.size_bytes(),
        const_cast<uint32_t *>(input.sortedParticleIdBySerialId.data()), &err);

    Sibernetic::ComputeInteractionWithMembranesFinalizeOpenCLArgs args{};
    args.position = positionBuf;
    args.velocity = velocityBuf;
    args.particleIndex = particleIndexBuf;
    args.particleIndexBack = particleIndexBackBuf;
    args.particleCount = static_cast<int32_t>(N);

    cl::Kernel kernel(
        opencl.program(),
        Sibernetic::kComputeInteractionWithMembranesFinalizeKernelName, &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error(
          "Failed to create computeInteractionWithMembranes_finalize kernel");
    args.bind(kernel);

    if (opencl.queue().enqueueNDRangeKernel(kernel, cl::NullRange,
                                            cl::NDRange(N),
                                            cl::NullRange) != CL_SUCCESS ||
        opencl.queue().finish() != CL_SUCCESS) {
      throw std::runtime_error(
          "Failed to execute computeInteractionWithMembranes_finalize kernel");
    }

    ComputeInteractionWithMembranesFinalizeResult result;
    result.position.resize(2 * N);
    if (opencl.queue().enqueueReadBuffer(positionBuf, CL_TRUE, 0,
                                         sizeof(float) * 4 * 2 * N,
                                         result.position.data()) != CL_SUCCESS)
      throw std::runtime_error("Failed to read position output buffer");

    return result;
  }
};

} // namespace SiberneticTest
