#include <metal_stdlib>
using namespace metal;

// Flattens 3D grid coordinates (x,y,z) into a linear cell id.
inline int cellId(int3 cellFactors, uint gridCellsX, uint gridCellsY) {
  return cellFactors.x + cellFactors.y * static_cast<int>(gridCellsX) +
         cellFactors.z * static_cast<int>(gridCellsX) *
             static_cast<int>(gridCellsY);
}

// Computes each particle's spatial hash cell and writes (cellId, serialId)
// into particleIndex. serialId is the original particle index in position[].
kernel void hashParticles(const device float4 *position [[buffer(0)]],
                          constant uint &gridCellsX [[buffer(1)]],
                          constant uint &gridCellsY [[buffer(2)]],
                          constant uint &gridCellsZ [[buffer(3)]],
                          constant float &hashGridCellSizeInv [[buffer(4)]],
                          constant float &xmin [[buffer(5)]],
                          constant float &ymin [[buffer(6)]],
                          constant float &zmin [[buffer(7)]],
                          device uint2 *particleIndex [[buffer(8)]],
                          constant uint &particleCount [[buffer(9)]],
                          uint id [[thread_position_in_grid]]) {
  if (id >= particleCount) {
    return;
  }

  (void)gridCellsZ;
  (void)xmin;
  (void)ymin;
  (void)zmin;

  const float4 p = position[id];
  int3 cellFactors;
  cellFactors.x = static_cast<int>(p.x * hashGridCellSizeInv);
  cellFactors.y = static_cast<int>(p.y * hashGridCellSizeInv);
  cellFactors.z = static_cast<int>(p.z * hashGridCellSizeInv);

  // Keep low 24 bits to match existing OpenCL behavior.
  const int cell = cellId(cellFactors, gridCellsX, gridCellsY) & 0x00ffffff;
  particleIndex[id] = uint2(static_cast<uint>(cell), id);
}

// After particleIndex is sorted by cell id, position/velocity are still in
// original serial order. This pass gathers them into sortedPosition/
// sortedVelocity and builds a reverse map (serialId -> sorted index).
//
// Example:
//   sorted particleIndex: [(1,1), (1,2), (2,3), (3,0), ...]
//   position:             [pos_0, pos_1, pos_2, pos_3, ...]
//   sortedPosition:       [pos_1, pos_2, pos_3, pos_0, ...]
kernel void sortPostPass(const device uint2 *particleIndex [[buffer(0)]],
                         device uint *particleIndexBack [[buffer(1)]],
                         const device float4 *position [[buffer(2)]],
                         const device float4 *velocity [[buffer(3)]],
                         device float4 *sortedPosition [[buffer(4)]],
                         device float4 *sortedVelocity [[buffer(5)]],
                         constant uint &particleCount [[buffer(6)]],
                         uint id [[thread_position_in_grid]]) {
  if (id >= particleCount) {
    return;
  }

  const uint2 spi = particleIndex[id];
  const uint serialId = spi.y; // PI_SERIAL_ID
  const uint cellId = spi.x;   // PI_CELL_ID

  float4 pos = position[serialId];
  pos.w = float(cellId); // POSITION_CELL_ID

  sortedPosition[id] = pos;
  sortedVelocity[id] = velocity[serialId];
  particleIndexBack[serialId] = id;
}

// For each cell id, finds the first index in sorted particleIndex whose
// particle belongs to that cell. Empty cells get UINT_MAX; slot gridCellCount
// stores PARTICLE_COUNT as the end sentinel.
kernel void indexx(const device uint2 *particleIndex [[buffer(0)]],
                   constant uint &gridCellCount [[buffer(1)]],
                   device uint *gridCellIndex [[buffer(2)]],
                   constant uint &particleCount [[buffer(3)]],
                   uint id [[thread_position_in_grid]]) {
  if (id > gridCellCount) {
    return;
  }
  if (id == gridCellCount) {
    gridCellIndex[id] = particleCount;
    return;
  }
  if (id == 0) {
    gridCellIndex[id] = 0;
    return;
  }

  int low = 0;
  int high = static_cast<int>(particleCount) - 1;
  bool converged = false;
  int cellIndex = -1;
  while (!converged) {
    if (low > high) {
      converged = true;
      cellIndex = -1;
      continue;
    }

    const int idx = ((high - low) >> 1) + low;
    const uint2 sample = particleIndex[idx];
    const int sampleCellId = static_cast<int>(sample.x);
    const bool isHigh = (sampleCellId > static_cast<int>(id));
    const bool isLow = (sampleCellId < static_cast<int>(id));
    const bool isMiddle = !(isHigh || isLow);

    high = isHigh ? idx - 1 : high;
    low = isLow ? idx + 1 : low;

    const bool zeroCase = (idx == 0 && isMiddle);
    const int sampleM1CellId =
        zeroCase ? -1 : static_cast<int>(particleIndex[idx - 1].x);
    converged = isMiddle && (zeroCase || sampleM1CellId < sampleCellId);
    cellIndex = converged ? idx : cellIndex;
    high = (isMiddle && !converged) ? idx - 1 : high;
  }

  gridCellIndex[id] = static_cast<uint>(cellIndex);
}
