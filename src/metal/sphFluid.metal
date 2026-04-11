#include <metal_stdlib>
using namespace metal;

inline int cellId(int3 cellFactors, uint gridCellsX, uint gridCellsY) {
  return cellFactors.x + cellFactors.y * static_cast<int>(gridCellsX) +
         cellFactors.z * static_cast<int>(gridCellsX) * static_cast<int>(gridCellsY);
}

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

  const int cell = cellId(cellFactors, gridCellsX, gridCellsY) & 0x00ffffff;
  particleIndex[id] = uint2(static_cast<uint>(cell), id);
}

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
