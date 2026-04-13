#include <metal_stdlib>
using namespace metal;

constant int kMaxNeighborCount = 32;
constant int kNoParticleId = -1;
// Each particle probes the 2×2×2 octant of cells overlapping its h-radius ball.
constant int kNeighborCellCount = 8;
constant int kBoundaryParticle = 3;

// Empirical surface tension scaling constant.
// See the surface tension force comment below for full attribution.
constant float kSurfaceTensionScale = -1.7e-09f;
// Empirical viscosity scale factor baked into this codebase.
// Not derived from the Müller (2003) viscosity formulation.
constant float kViscosityScale = 1.5f / 1000.0f;

// Particle type ranges are encoded per particle in configuration files
// ([position] section, 4th column) and loaded into originalPosition[].w.
constant float kWormTypeMin = 2.05f;
constant float kWormTypeMax = 2.25f;
constant float kFluidTypeMin = 2.25f;
constant float kFluidTypeMax = 2.35f;

inline bool isWormType(float particleType) {
  return particleType > kWormTypeMin && particleType < kWormTypeMax;
}

inline bool isFluidType(float particleType) {
  return particleType > kFluidTypeMin && particleType < kFluidTypeMax;
}

// Shared Poly6 radial core term: (hScaled^2 - r^2)^3.
// Reusing this in density and surface tension is correct because both
// operators are built from the same compact-support Poly6 distance weight;
// they differ in how this scalar is used (density sums scalar mass weights,
// while surface tension multiplies by a direction vector and a different
// physical coefficient).
inline float poly6Core(float distanceSquared, float hScaledSquared) {
  const float delta = hScaledSquared - distanceSquared;
  return delta * delta * delta;
}

// neighborMap: A packed table of neighbor information for SPH force kernels.
// Structure: float2 array with kMaxNeighborCount entries per particle.
//   Indexing: neighborMap[sortedParticleId * kMaxNeighborCount + slotIndex]
//   Each float2 tuple is:
//     .x = neighbor sorted particle ID (or kNoParticleId=-1 if empty)
//     .y = distance to neighbor in simulation units (or -1.0f if empty)
//
// Built by findNeighbors kernel:
//   - Probes neighboring spatial hash cells to find particles within h-radius
//   - Maintains a max-heap of kMaxNeighborCount closest neighbors
//   - Empty slots at the end have .x = kNoParticleId, .y = -1.0f
//
// Consumed by SPH kernels (pcisph_computeDensity,
// pcisph_computeForcesAndInitPressure):
//   - Iterate over slots; skip when .x == kNoParticleId
//   - Use .x to index into sorted position/velocity/density buffers
//   - Use .y for distance-dependent kernel evaluations (cutoff check at
//   hScaled)

// Flattens 3D grid coordinates (x,y,z) into a linear cell id.
inline int cellId(int3 cellCoordinates, uint gridCellsX, uint gridCellsY) {
  return cellCoordinates.x + cellCoordinates.y * static_cast<int>(gridCellsX) +
         cellCoordinates.z * static_cast<int>(gridCellsX) *
             static_cast<int>(gridCellsY);
}

// Converts a particle's world position into 3D grid cell coordinates.
// Divides each position component by the inverse cell size (effectively scaling
// by 1/cellSize to get which cell the particle belongs to).
// Only .x/.y/.z are used; .w is ignored, so this works for both
// originalPosition (where .w may be particle type) and sortedPosition
// (where .w may be cell ID).
inline int3 cellCoordinates(float4 position, float hashGridCellSizeInv) {
  return int3(static_cast<int>(position.x * hashGridCellSizeInv),
              static_cast<int>(position.y * hashGridCellSizeInv),
              static_cast<int>(position.z * hashGridCellSizeInv));
}

// Computes each particle's spatial hash cell and writes (cellId, serialId)
// into sortedCellAndSerialId. serialId is the original particle index in
// originalPosition[].
kernel void hashParticles(const device float4 *originalPosition [[buffer(0)]],
                          constant uint &gridCellsX [[buffer(1)]],
                          constant uint &gridCellsY [[buffer(2)]],
                          constant uint &gridCellsZ [[buffer(3)]],
                          constant float &hashGridCellSizeInv [[buffer(4)]],
                          constant float &xmin [[buffer(5)]],
                          constant float &ymin [[buffer(6)]],
                          constant float &zmin [[buffer(7)]],
                          device uint2 *sortedCellAndSerialId [[buffer(8)]],
                          constant uint &particleCount [[buffer(9)]],
                          uint serialId [[thread_position_in_grid]]) {
  if (serialId >= particleCount) {
    return;
  }

  (void)gridCellsZ;
  (void)xmin;
  (void)ymin;
  (void)zmin;

  const float4 p = originalPosition[serialId];
  int3 cellCoordinates;
  cellCoordinates.x = static_cast<int>(p.x * hashGridCellSizeInv);
  cellCoordinates.y = static_cast<int>(p.y * hashGridCellSizeInv);
  cellCoordinates.z = static_cast<int>(p.z * hashGridCellSizeInv);

  // Keep low 24 bits to match existing OpenCL behavior.
  // This limits the simulation to at most 2^24 = 16,777,216 grid cells
  // (e.g. a 256x256x256 grid). Typical Sibernetic runs use ~32^3-64^3
  // cells, so this ceiling is never approached in practice.
  const int cell = cellId(cellCoordinates, gridCellsX, gridCellsY) & 0x00ffffff;
  sortedCellAndSerialId[serialId] = uint2(static_cast<uint>(cell), serialId);
}

// After sortedCellAndSerialId is sorted by cell id, position/velocity are still
// in original serial order. This pass gathers them into sortedPosition/
// sortedVelocity and builds a reverse map (serialId -> sorted index).
//
// Example:
//   sorted sortedCellAndSerialId: [(1,1), (1,2), (2,3), (3,0), ...]
//   position:             [pos_0, pos_1, pos_2, pos_3, ...]
//   sortedPosition:       [pos_1, pos_2, pos_3, pos_0, ...]
kernel void sortPostPass(const device uint2 *sortedCellAndSerialId
                         [[buffer(0)]],
                         device uint *sortedParticleIdBySerialId [[buffer(1)]],
                         const device float4 *originalPosition [[buffer(2)]],
                         const device float4 *velocity [[buffer(3)]],
                         device float4 *sortedPosition [[buffer(4)]],
                         device float4 *sortedVelocity [[buffer(5)]],
                         constant uint &particleCount [[buffer(6)]],
                         uint sortedParticleId [[thread_position_in_grid]]) {
  if (sortedParticleId >= particleCount) {
    return;
  }

  const uint2 spi = sortedCellAndSerialId[sortedParticleId];
  const uint serialId = spi.y; // PI_SERIAL_ID
  const uint cellId = spi.x;   // PI_CELL_ID

  float4 pos = originalPosition[serialId];
  // Store cell ID in sorted position's .w component for neighbor lookups.
  // (Original position[].w still holds particle type; only sortedPosition[]
  // gets cell ID overwritten here.)
  pos.w = float(cellId); // POSITION_CELL_ID

  sortedPosition[sortedParticleId] = pos;
  sortedVelocity[sortedParticleId] = velocity[serialId];
  sortedParticleIdBySerialId[serialId] = sortedParticleId;
}

// For each cell id, finds the first index in sorted sortedCellAndSerialId whose
// particle belongs to that cell. Empty cells get UINT_MAX; slot gridCellCount
// stores PARTICLE_COUNT as the end sentinel.
// TODO(weiweng): Revisit this after full Metal kernel port. Consider replacing
// per-cell binary search with a linear boundary-scan build:
// 1) initialize gridCellIndex to UINT_MAX,
// 2) launch one thread per sorted particle index,
// 3) write start index when cell id changes,
// 4) set gridCellIndex[gridCellCount] = particleCount.
kernel void indexx(const device uint2 *sortedCellAndSerialId [[buffer(0)]],
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
    const int sampleCellId = static_cast<int>(sortedCellAndSerialId[idx].x);

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
          (static_cast<int>(sortedCellAndSerialId[idx - 1].x) < sampleCellId);
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
    float4 myParticlePosition, int mySortedParticleId,
    const device float4 *sortedPosition, device float2 *neighborMap,
    thread int *closestIndexes, thread float *closestDistances) {
  const int baseSortedParticleId =
      static_cast<int>(gridCellIndices[probeCellId]);
  const int nextSortedParticleId =
      static_cast<int>(gridCellIndices[probeCellId + 1]);
  const int particleCountThisCell = nextSortedParticleId - baseSortedParticleId;

  for (int i = 0; i < particleCountThisCell; ++i) {
    const int neighborSortedParticleId = baseSortedParticleId + i;
    if (mySortedParticleId == neighborSortedParticleId) {
      continue;
    }

    const float4 d =
        myParticlePosition - sortedPosition[neighborSortedParticleId];
    const float distanceSquared = d.x * d.x + d.y * d.y + d.z * d.z;
    // The root is always the worst kept neighbor; only admit candidates that
    // beat it strictly (< not <=) so exact-boundary particles at
    // distanceSquared == rThresholdSquared are excluded: both the viscosity
    // and surface tension kernels evaluate to zero at that distance, so they
    // contribute nothing and would waste a neighbor slot.
    if (distanceSquared < closestDistances[0]) {
      closestDistances[0] = distanceSquared;
      closestIndexes[0] = neighborSortedParticleId;
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
                          uint sortedParticleId [[thread_position_in_grid]]) {
  if (sortedParticleId >= particleCount) {
    return;
  }

  (void)gridCellsZ;

  // "FixedUp" means empty-cell holes were filled in a post-pass so each cell
  // has a valid start pointer. Empty cells satisfy start[c] == start[c + 1],
  // producing a zero-length [start, end) range during neighbor scans.
  const device uint *gridCellIndices = gridCellIndicesFixedUp;
  const float4 position = sortedPosition[sortedParticleId];
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
                               static_cast<int>(sortedParticleId),
                               sortedPosition, neighborMap, closestIndexes,
                               closestDistances);
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
    neighborMap[sortedParticleId * kMaxNeighborCount + static_cast<uint>(i)] =
        neighborData;
  }
}

// Computes per-particle density by summing poly6 neighbor contributions
// within hScaled, then applying the hScaled^6 minimum-density floor.
//
// Arguments:
// buffer(0) neighborMap: packed table of 32 float2 entries per particle.
//   x = neighbor sorted particle id (or -1 sentinel), y = neighbor distance.
// buffer(1) massMultWpoly6Coefficient: mass * 315 / (64 * pi * hScaled^9),
//   where hScaled = h * simulationScale. Multiplying poly6Sum by this converts
//   the raw kernel sum into density (mass/volume).
// buffer(2) hScaled2: (h * simulationScale)^2, matching the scale of distances
// in neighborMap.
// buffer(3) rho: output density array, indexed by sorted particle id.
// buffer(4) sortedParticleIdBySerialId: map from serial (original) particle id
// -> sorted index. buffer(5) particleCount: number of particles to process.
// thread_position_in_grid serialId: serial (original) particle index for this
// thread.
kernel void pcisph_computeDensity(
    const device float2 *neighborMap [[buffer(0)]],
    constant float &massMultWpoly6Coefficient [[buffer(1)]],
    constant float &hScaled2 [[buffer(2)]], device float *rho [[buffer(3)]],
    const device uint *sortedParticleIdBySerialId [[buffer(4)]],
    constant uint &particleCount [[buffer(5)]],
    uint serialId [[thread_position_in_grid]]) {
  if (serialId >= particleCount) {
    return;
  }

  const uint sortedParticleId = sortedParticleIdBySerialId[serialId];
  const uint idx = sortedParticleId * static_cast<uint>(kMaxNeighborCount);
  // poly6NeighborContributionSum = sum_j (hScaled2 - r_j^2)^3 for all
  // neighbors j with r_j <
  // hScaled. Floored to the Poly6 self-contribution at r=0:
  //   poly6SelfContribution = (hScaled2 - 0)^3 = hScaled^6.
  // This prevents the density kernel sum from becoming zero when a particle
  // has no
  // neighbors, which would otherwise produce rho=0 and a numerically
  // explosive pressure correction.
  float poly6NeighborContributionSum = 0.0f;
  const float poly6SelfContribution = poly6Core(0.0f, hScaled2);

  for (int neighborSlot = 0; neighborSlot < kMaxNeighborCount; ++neighborSlot) {
    const float2 neighbor = neighborMap[idx + static_cast<uint>(neighborSlot)];
    if (static_cast<int>(neighbor.x) != kNoParticleId) {
      float distanceToNeighbor2 = neighbor.y;
      distanceToNeighbor2 *= distanceToNeighbor2;
      if (distanceToNeighbor2 < hScaled2) {
        poly6NeighborContributionSum +=
            poly6Core(distanceToNeighbor2, hScaled2);
      }
    }
  }

  if (poly6NeighborContributionSum < poly6SelfContribution) {
    poly6NeighborContributionSum = poly6SelfContribution;
  }
  rho[sortedParticleId] =
      poly6NeighborContributionSum * massMultWpoly6Coefficient;
}

// Writes the non-pressure acceleration and zeros the pressure-force slot and
// pressure scalar for sortedParticleId.
//
// The acceleration buffer is a single allocation of 2*particleCount entries
// split into two logical halves:
//   [0 .. N-1]   non-pressure acceleration (viscosity + gravity + surface
//                tension) — written as nonPressureAccel (float4(0) for
//                boundary particles, total_accel for fluid particles).
//   [N .. 2N-1]  pressure acceleration accumulated by the PCISPH
//                predict/correct iterations that follow this kernel — zeroed
//                here so each timestep starts from a clean slate.
// Both are written unconditionally because sorted slots are reassigned every
// timestep and may have held a different particle's data in the prior step.
inline void writeAccelerationAndInitPressure(device float4 *acceleration,
                                             device float *pressure,
                                             uint particleCount,
                                             uint sortedParticleId,
                                             float4 nonPressureAccel) {
  acceleration[sortedParticleId] = nonPressureAccel;
  acceleration[particleCount + sortedParticleId] = float4(0.0f);
  pressure[sortedParticleId] = 0.0f;
}

/// PCISPH: Compute viscosity + surface tension + gravity forces.
/// Initializes pressure to 0 and acceleration[N..2N] to 0.
kernel void pcisph_computeForcesAndInitPressure(
    const device float2 *neighborMap [[buffer(0)]],
    const device float *rho [[buffer(1)]], device float *pressure [[buffer(2)]],
    const device float4 *sortedPosition [[buffer(3)]],
    const device float4 *sortedVelocity [[buffer(4)]],
    device float4 *acceleration [[buffer(5)]],
    const device uint *sortedParticleIdBySerialId [[buffer(6)]],
    constant float &surfTensCoeff [[buffer(7)]],
    constant float &massMultLaplacianWviscosityCoeff [[buffer(8)]],
    constant float &hScaled [[buffer(9)]], constant float &mu [[buffer(10)]],
    constant float &gravitationalAccelerationX [[buffer(11)]],
    constant float &gravitationalAccelerationY [[buffer(12)]],
    constant float &gravitationalAccelerationZ [[buffer(13)]],
    const device float4 *originalPosition [[buffer(14)]],
    const device uint2 *sortedCellAndSerialId [[buffer(15)]],
    constant uint &particleCount [[buffer(16)]],
    constant float &mass [[buffer(17)]],
    uint serialId [[thread_position_in_grid]]) {
  if (serialId >= particleCount)
    return;

  uint sortedParticleId = sortedParticleIdBySerialId[serialId];

  // Boundary particles don't move.
  // Note: originalPosition[].w still holds the original particle type;
  // sortedPosition[].w was overwritten with cell ID in sortPostPass. Match the
  // OpenCL behavior by classifying via integer cast (i.e. floor/truncate for
  // positive values) instead of exact float equality. Configuration
  // files (such as demo1) store subtype tags in the fractional part (e.g. 3.1),
  // so exact comparison with 3.0 would miss boundary-like values that should
  // map to boundary class 3.
  if (int(originalPosition[serialId].w) == kBoundaryParticle) {
    // Boundary particles never move, but sortedParticleId changes every
    // timestep as the sort reassigns sorted slots. The slot this particle
    // occupies now may have held a non-boundary particle last timestep, so
    // its acceleration entry could contain stale non-zero values. Write zeros
    // unconditionally to own the output slot cleanly.
    writeAccelerationAndInitPressure(acceleration, pressure, particleCount,
                                     sortedParticleId, float4(0.0f));
    return;
  }

  int neighborMapOffset = sortedParticleId * kMaxNeighborCount;
  float hScaled2 = hScaled * hScaled;

  float4 accelViscosity = float4(0.0f);
  float4 accelSurfaceTension = float4(0.0f);

  // Loop through neighbors
  for (int neighborSlot = 0; neighborSlot < kMaxNeighborCount; ++neighborSlot) {
    float2 neighborEntry = neighborMap[neighborMapOffset + neighborSlot];
    int neighborSortedParticleId = int(neighborEntry.x);
    if (neighborSortedParticleId == kNoParticleId)
      continue;

    float distanceToNeighbor = neighborEntry.y;
    // findNeighbors admits only neighbors with distanceSquared <= h*h, so
    // distanceToNeighbor <= hScaled is guaranteed. The >= check handles the
    // exact-equality boundary case: at distanceToNeighbor == hScaled both
    // the viscosity term (hScaled - distanceToNeighbor = 0) and the surface
    // tension kernel ((hScaled2 - distanceToNeighbor2) = 0) contribute
    // nothing, so skipping saves the work.
    if (distanceToNeighbor >= hScaled)
      continue;

    float distanceToNeighbor2 = distanceToNeighbor * distanceToNeighbor;

    uint neighborSerialId = sortedCellAndSerialId[neighborSortedParticleId].y;
    // 1.0 for fluid neighbors, 0.0 for boundary neighbors. Boundary particles
    // are stationary walls whose velocity contribution to viscosity is zeroed
    // out by multiplying vj by this mask.
    float neighborFluidMask =
        (int(originalPosition[neighborSerialId].w) != kBoundaryParticle) ? 1.0f
                                                                         : 0.0f;

    float4 particleVelocity = sortedVelocity[sortedParticleId];
    float4 neighborVelocity = sortedVelocity[neighborSortedParticleId];
    // originalPosition[].w holds the particle type (unchanged by sortPostPass).
    float particleType = originalPosition[serialId].w;
    float neighborType = originalPosition[neighborSerialId].w;
    // Müller et al. (2003), written as a neighbor sum:
    //   a_i += sum_j [ (mu / rho_j) * (m_j / rho_i) * (v_j - v_i)
    //                  * laplacian_W_visc(r_ij, h) ]
    // where laplacian_W_visc(r, h) = (45 / (pi * h^6)) * (h - r).
    // This loop body computes one j-term at a time; the sum_j is realized by
    // iterating over neighbor slots and accumulating into accelViscosity.
    // In this implementation, massMultLaplacianWviscosityCoeff packs
    // m_j * (45 / (pi * hScaled^6)) (with constant particle mass), while
    // 1 / rho_i is applied separately in the finalize step after the loop.
    //
    // Here viscCoeff replaces mu / rho_j: mu (the true dynamic viscosity,
    // Pa·s) is passed as buffer(10) but is not used. Instead, viscCoeff is
    // an empirical per-interaction-type tuning constant that lumps together
    // mu and the neighbor density rho_j into a single scalar.
    float viscCoeff = 1.0e-4f; // default (fluid–fluid or fluid–boundary)

    // Worm body ([kWormTypeMin, kWormTypeMax]) <->
    // fluid medium ([kFluidTypeMin, kFluidTypeMax]): use lower
    // interface viscosity so the worm can slide through the medium.
    // If this interface viscosity is too high, drag becomes too strong and
    // locomotion is over-damped.
    bool isWormParticle = isWormType(particleType);
    bool isWormNeighbor = isWormType(neighborType);
    bool isFluidParticle = isFluidType(particleType);
    bool isFluidNeighbor = isFluidType(neighborType);

    if ((isWormParticle && isFluidNeighbor) ||
        (isFluidParticle && isWormNeighbor)) {
      viscCoeff = 1.0e-5f; // worm-fluid interface: 10x lower viscosity
    }

    // This loop accumulates only the pairwise velocity-shape part:
    //   sum_j [ viscCoeff_ij * (v_j - v_i) * (hScaled - r_ij) ]
    // where viscCoeff_ij is an empirical stand-in for (mu / rho_j).
    //
    // Compared with Muller viscosity:
    //   a_i += sum_j [ (mu/rho_j) * (m_j/rho_i) * (v_j-v_i)
    //                  * (45/(pi*h^6)) * (h-r_ij) ]
    // this line intentionally omits ((m_j*45)/(pi*hScaled^6)) and (1/rho_i);
    // both are applied after the neighbor loop via:
    //   accelViscosity *= (1.5/1000) * massMultLaplacianWviscosityCoeff /
    //                     rho_i
    // so rho_i and kernel/mass scaling are deferred to the finalize step.
    accelViscosity +=
        viscCoeff * (neighborVelocity * neighborFluidMask - particleVelocity) *
        (hScaled - distanceToNeighbor);

    // Surface tension force (legacy Sibernetic form).
    // Discrete pairwise form used here:
    //   a_i^surf += C_st * s_ij * (x_i - x_j),
    //   s_ij = (hScaled^2 - r_ij^2)^3,
    // where C_st = (-1.7e-09) * surfTensCoeff / mass (the /mass is applied
    // in the finalize step below).
    // Literature attribution: this term is historically tagged in the OpenCL
    // implementation as "formula (16) [5]" from Becker and Teschner,
    // "Weakly compressible SPH for free surface flows" (SCA 2007,
    // DOI: 10.2312/SCA/SCA07/209-218). The -1.7e-09 scalar is a project-
    // specific empirical scaling, not a universal constant from the paper.
    const float surfKern = poly6Core(distanceToNeighbor2, hScaled2);
    // sortedPosition[].w stores cell ID (not position), so the .w part of this
    // subtraction is not physically meaningful; we zero accelSurfaceTension.w
    // after the loop.
    accelSurfaceTension += kSurfaceTensionScale * surfTensCoeff * surfKern *
                           (sortedPosition[sortedParticleId] -
                            sortedPosition[neighborSortedParticleId]);
  }

  // Finalize surface tension
  accelSurfaceTension.w = 0.0f;
  accelSurfaceTension /= mass;
  // Finalize viscosity
  // 1.5f/1000.0f is an empirical scale factor from this codebase,
  // not a coefficient from the Müller viscosity derivation.
  accelViscosity *= kViscosityScale * massMultLaplacianWviscosityCoeff /
                    rho[sortedParticleId];

  // Total acceleration = viscosity + gravitational acceleration +
  // surface tension
  float4 totalAccel = accelViscosity;
  totalAccel += float4(gravitationalAccelerationX, gravitationalAccelerationY,
                       gravitationalAccelerationZ, 0.0f);
  totalAccel += accelSurfaceTension;

  writeAccelerationAndInitPressure(acceleration, pressure, particleCount,
                                   sortedParticleId, totalAccel);
}
