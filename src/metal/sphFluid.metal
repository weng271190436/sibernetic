#include <metal_stdlib>
using namespace metal;

constant int kMaxNeighborCount = 32;
constant int kNoParticleId = -1;
// Each particle probes the 2×2×2 octant of cells overlapping its h-radius ball.
constant int kNeighborCellCount = 8;

// Flattens 3D grid coordinates (x,y,z) into a linear cell id.
inline int cellId(int3 cellCoordinates, uint gridCellsX, uint gridCellsY) {
  return cellCoordinates.x + cellCoordinates.y * static_cast<int>(gridCellsX) +
         cellCoordinates.z * static_cast<int>(gridCellsX) *
             static_cast<int>(gridCellsY);
}

// Converts a particle's world position into 3D grid cell coordinates.
// Divides each position component by the inverse cell size (effectively scaling
// by 1/cellSize to get which cell the particle belongs to).
inline int3 cellCoordinates(float4 position, float hashGridCellSizeInv) {
  return int3(static_cast<int>(position.x * hashGridCellSizeInv),
              static_cast<int>(position.y * hashGridCellSizeInv),
              static_cast<int>(position.z * hashGridCellSizeInv));
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
                          uint particleId [[thread_position_in_grid]]) {
  if (particleId >= particleCount) {
    return;
  }

  (void)gridCellsZ;
  (void)xmin;
  (void)ymin;
  (void)zmin;

  const float4 p = position[particleId];
  int3 cellCoordinates;
  cellCoordinates.x = static_cast<int>(p.x * hashGridCellSizeInv);
  cellCoordinates.y = static_cast<int>(p.y * hashGridCellSizeInv);
  cellCoordinates.z = static_cast<int>(p.z * hashGridCellSizeInv);

  // Keep low 24 bits to match existing OpenCL behavior.
  // This limits the simulation to at most 2^24 = 16,777,216 grid cells
  // (e.g. a 256x256x256 grid). Typical Sibernetic runs use ~32^3-64^3
  // cells, so this ceiling is never approached in practice.
  const int cell = cellId(cellCoordinates, gridCellsX, gridCellsY) & 0x00ffffff;
  particleIndex[particleId] = uint2(static_cast<uint>(cell), particleId);
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
                         uint particleId [[thread_position_in_grid]]) {
  if (particleId >= particleCount) {
    return;
  }

  const uint2 spi = particleIndex[particleId];
  const uint serialId = spi.y; // PI_SERIAL_ID
  const uint cellId = spi.x;   // PI_CELL_ID

  float4 pos = position[serialId];
  pos.w = float(cellId); // POSITION_CELL_ID

  sortedPosition[particleId] = pos;
  sortedVelocity[particleId] = velocity[serialId];
  particleIndexBack[serialId] = particleId;
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

// Sifts the element at index i downward to restore the max-heap property.
// closestDistances is the key; larger = farther = worse.
inline void heapSiftDown(thread int *closestIndexes,
                         thread float *closestDistances, int i) {
  while (true) {
    int largest = i;
    const int left = 2 * i + 1;
    const int right = 2 * i + 2;
    if (left < kMaxNeighborCount &&
        closestDistances[left] > closestDistances[largest]) {
      largest = left;
    }
    if (right < kMaxNeighborCount &&
        closestDistances[right] > closestDistances[largest]) {
      largest = right;
    }
    if (largest == i)
      break;
    const int tmpIdx = closestIndexes[i];
    const float tmpDist = closestDistances[i];
    closestIndexes[i] = closestIndexes[largest];
    closestDistances[i] = closestDistances[largest];
    closestIndexes[largest] = tmpIdx;
    closestDistances[largest] = tmpDist;
    i = largest;
  }
}

// Scans all particles in probeCellId and maintains closestDistances/
// closestIndexes as a max-heap of the kMaxNeighborCount closest neighbors seen
// so far. The heap root (index 0) always holds the current farthest (worst)
// neighbor, so a single comparison against it decides whether to admit a
// candidate; heapSiftDown then restores the invariant in O(log k).
inline void updateNeighborHeapFromCell(
    int probeCellId, const device uint *gridCellIndices,
    float4 myParticlePosition, int myParticleId,
    const device float4 *sortedPosition, device float2 *neighborMap,
    thread int *closestIndexes, thread float *closestDistances) {
  const int baseParticleId = static_cast<int>(gridCellIndices[probeCellId]);
  const int nextParticleId = static_cast<int>(gridCellIndices[probeCellId + 1]);
  const int particleCountThisCell = nextParticleId - baseParticleId;

  for (int i = 0; i < particleCountThisCell; ++i) {
    const int neighborParticleId = baseParticleId + i;
    if (myParticleId == neighborParticleId) {
      continue;
    }

    const float4 d = myParticlePosition - sortedPosition[neighborParticleId];
    const float distanceSquared = d.x * d.x + d.y * d.y + d.z * d.z;
    // The root is always the worst kept neighbor; only admit candidates that
    // beat it, then sift down to restore the heap invariant.
    if (distanceSquared <= closestDistances[0]) {
      closestDistances[0] = distanceSquared;
      closestIndexes[0] = neighborParticleId;
      heapSiftDown(closestIndexes, closestDistances, 0);
    }
  }
}

// Computes the linear cell ID of a neighbor cell by applying (deltaX, deltaY,
// deltaZ) offsets to the given cell ID, with periodic boundary wrapping.
inline int neighborCellId(int cell, int deltaX, int deltaY, int deltaZ,
                          uint gridCellsX, uint gridCellsY,
                          uint gridCellCount) {
  const int dx = deltaX;
  const int dy = deltaY * static_cast<int>(gridCellsX);
  const int dz =
      deltaZ * static_cast<int>(gridCellsX) * static_cast<int>(gridCellsY);
  int newCell = cell + dx + dy + dz;
  const int gridCellCountInt = static_cast<int>(gridCellCount);
  newCell = (newCell < 0) ? (newCell + gridCellCountInt) : newCell;
  newCell =
      (newCell >= gridCellCountInt) ? (newCell - gridCellCountInt) : newCell;
  return newCell;
}

kernel void findNeighbors(const device uint *gridCellIndicesFixedUp
                          [[buffer(0)]],
                          const device float4 *sortedPosition [[buffer(1)]],
                          constant uint &gridCellCount [[buffer(2)]],
                          constant uint &gridCellsX [[buffer(3)]],
                          constant uint &gridCellsY [[buffer(4)]],
                          constant uint &gridCellsZ [[buffer(5)]],
                          constant float &h [[buffer(6)]],
                          constant float &hashGridCellSize [[buffer(7)]],
                          constant float &hashGridCellSizeInv [[buffer(8)]],
                          constant float &simulationScale [[buffer(9)]],
                          constant float &xmin [[buffer(10)]],
                          constant float &ymin [[buffer(11)]],
                          constant float &zmin [[buffer(12)]],
                          device float2 *neighborMap [[buffer(13)]],
                          constant uint &particleCount [[buffer(14)]],
                          uint particleId [[thread_position_in_grid]]) {
  if (particleId >= particleCount) {
    return;
  }

  (void)gridCellsZ;

  // "FixedUp" means empty-cell holes were filled in a post-pass so each cell
  // has a valid start pointer. Empty cells satisfy start[c] == start[c + 1],
  // producing a zero-length [start, end) range during neighbor scans.
  const device uint *gridCellIndices = gridCellIndicesFixedUp;
  const float4 position = sortedPosition[particleId];
  const int particleCellId = static_cast<int>(position.w) & 0x00ffffff;

  const float rThresholdSquared = h * h;
  float closestDistances[kMaxNeighborCount];
  int closestIndexes[kMaxNeighborCount];
  // All slots start at rThresholdSquared, forming a trivial valid max-heap
  // (all values equal). Empty slots (kNoParticleId) act as sentinels that
  // accept any particle within the cutoff radius until they are displaced.
  for (int k = 0; k < kMaxNeighborCount; ++k) {
    closestDistances[k] = rThresholdSquared;
    closestIndexes[k] = kNoParticleId;
  }

  // Determine which neighboring cells to probe based on position inside cell.
  const float4 p = position - float4(xmin, ymin, zmin, 0.0f);
  const int3 cellCoords = cellCoordinates(position, hashGridCellSizeInv);
  const float3 cellMinCorner =
      float3(cellCoords.x * hashGridCellSize, cellCoords.y * hashGridCellSize,
             cellCoords.z * hashGridCellSize);
  const bool3 isInLowerHalf =
      bool3((p.x - cellMinCorner.x) < h, (p.y - cellMinCorner.y) < h,
            (p.z - cellMinCorner.z) < h);
  // Match OpenCL semantics: lower-half probes negative direction (-1),
  // upper-half probes positive direction (+1).
  const int3 delta = int3(isInLowerHalf.x ? -1 : 1, isInLowerHalf.y ? -1 : 1,
                          isInLowerHalf.z ? -1 : 1);

  // Build the kNeighborCellCount neighbor cell IDs. Each entry covers one
  // combination of (delta.x or 0, delta.y or 0, delta.z or 0), selected by
  // the 3 low bits of i: bit 0 → x, bit 1 → y, bit 2 → z.
  int neighborCellIds[kNeighborCellCount];
  for (int i = 0; i < kNeighborCellCount; ++i) {
    const int dx = (i & 1) ? delta.x : 0;
    const int dy = (i & 2) ? delta.y : 0;
    const int dz = (i & 4) ? delta.z : 0;
    neighborCellIds[i] = neighborCellId(particleCellId, dx, dy, dz, gridCellsX,
                                        gridCellsY, gridCellCount);
  }

  for (int i = 0; i < kNeighborCellCount; ++i) {
    updateNeighborHeapFromCell(neighborCellIds[i], gridCellIndices, position,
                               static_cast<int>(particleId), sortedPosition,
                               neighborMap, closestIndexes, closestDistances);
  }

  for (int i = 0; i < kMaxNeighborCount; ++i) {
    float2 neighborData;
    neighborData.x = static_cast<float>(closestIndexes[i]);
    if (closestIndexes[i] >= 0) {
      // Positions are in hash-grid space; multiply by simulationScale to
      // convert the distance into simulation units for the SPH kernels.
      neighborData.y = sqrt(closestDistances[i]) * simulationScale;
    } else {
      neighborData.y = -1.0f;
    }
    neighborMap[particleId * kMaxNeighborCount + static_cast<uint>(i)] =
        neighborData;
  }
}

// Computes per-particle density by summing poly6 neighbor contributions
// within hScaled, then applying the hScaled^6 minimum-density floor.
//
// Arguments:
// buffer(0) neighborMap: packed table of 32 float2 entries per particle.
//   x = neighbor particle id (or -1 sentinel), y = neighbor distance.
// buffer(1) massMultWpoly6Coefficient: mass * 315 / (64 * pi * hScaled^9),
//   where hScaled = h * simulationScale. Multiplying poly6Sum by this converts
//   the raw kernel sum into density (mass/volume).
// buffer(2) hScaled2: (h * simulationScale)^2, matching the scale of distances
// in neighborMap.
// buffer(3) rho: output density array, indexed by sorted particle id.
// buffer(4) particleIndexBack: map from serial (original) particle id -> sorted
// index.
// buffer(5) particleCount: number of particles to process.
// thread_position_in_grid serialId: serial (original) particle index for this
// thread.
kernel void
pcisph_computeDensity(const device float2 *neighborMap [[buffer(0)]],
                      constant float &massMultWpoly6Coefficient [[buffer(1)]],
                      constant float &hScaled2 [[buffer(2)]],
                      device float *rho [[buffer(3)]],
                      const device uint *particleIndexBack [[buffer(4)]],
                      constant uint &particleCount [[buffer(5)]],
                      uint serialId [[thread_position_in_grid]]) {
  if (serialId >= particleCount) {
    return;
  }

  const uint particleId = particleIndexBack[serialId];
  const uint idx = particleId * static_cast<uint>(kMaxNeighborCount);
  // poly6Sum = sum_j (hScaled2 - r_j^2)^3  for all neighbors j with r_j <
  // hScaled. Floored to hScaled^6 = (hScaled2)^3, which equals the poly6
  // self-contribution at r=0. This prevents poly6Sum from being zero when a
  // particle has no neighbors, which would otherwise produce rho=0 and a
  // numerically explosive pressure correction.
  float poly6Sum = 0.0f;
  const float hScaled6 = hScaled2 * hScaled2 * hScaled2;

  for (int nc = 0; nc < kMaxNeighborCount; ++nc) {
    const float2 neighbor = neighborMap[idx + static_cast<uint>(nc)];
    if (static_cast<int>(neighbor.x) != kNoParticleId) {
      float r2 = neighbor.y;
      r2 *= r2;
      if (r2 < hScaled2) {
        const float delta = hScaled2 - r2;
        poly6Sum += delta * delta * delta;
      }
    }
  }

  if (poly6Sum < hScaled6) {
    poly6Sum = hScaled6;
  }
  rho[particleId] = poly6Sum * massMultWpoly6Coefficient;
}
