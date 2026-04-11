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
  // This limits the simulation to at most 2^24 = 16,777,216 grid cells
  // (e.g. a 256x256x256 grid). Typical Sibernetic runs use ~32^3-64^3
  // cells, so this ceiling is never approached in practice.
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
// TODO(weiweng): Revisit this after full Metal kernel port. Consider replacing
// per-cell binary search with a linear boundary-scan build:
// 1) initialize gridCellIndex to UINT_MAX,
// 2) launch one thread per sorted particle index,
// 3) write start index when cell id changes,
// 4) set gridCellIndex[gridCellCount] = particleCount.
kernel void indexx(const device uint2 *particleIndex [[buffer(0)]],
                   constant uint &gridCellCount [[buffer(1)]],
                   device uint *gridCellIndex [[buffer(2)]],
                   constant uint &particleCount [[buffer(3)]],
                   uint targetCellId [[thread_position_in_grid]]) {
  if (targetCellId > gridCellCount) {
    return;
  }
  if (targetCellId == gridCellCount) {
    gridCellIndex[targetCellId] = particleCount;
    return;
  }
  if (targetCellId == 0) {
    gridCellIndex[targetCellId] = 0;
    return;
  }

  int low = 0;
  int high = static_cast<int>(particleCount) - 1;
  int cellIndex = -1;
  while (low <= high) {
    const int idx = low + (high - low) / 2;
    const int sampleCellId = static_cast<int>(particleIndex[idx].x);

    if (sampleCellId < static_cast<int>(targetCellId)) {
      low = idx + 1;
    } else if (sampleCellId > static_cast<int>(targetCellId)) {
      high = idx - 1;
    } else {
      // Found a match; check if it's the leftmost occurrence.
      // Guard idx-1 access: when idx==0 there is no previous element,
      // so this is trivially the left boundary.
      const bool isLeftBoundary =
          (idx == 0) ||
          (static_cast<int>(particleIndex[idx - 1].x) < sampleCellId);
      if (isLeftBoundary) {
        cellIndex = idx;
        break;
      }
      high = idx - 1; // keep searching left for the first occurrence
    }
  }

  // cellIndex == -1 when not found; cast to uint gives 0xFFFFFFFF == UINT_MAX.
  gridCellIndex[targetCellId] = static_cast<uint>(cellIndex);
}
