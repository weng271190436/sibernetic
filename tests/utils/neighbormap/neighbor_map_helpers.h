#pragma once

#include <cstdint>
#include <vector>

#include "../../src/types/HostTypes.h"

namespace SiberneticTest {

/// Helper to create an empty neighbor map (all slots = no particle).
/// Each particle gets kMaxNeighborCount=32 float2 entries.
/// Empty slots are initialized to {-1.0f, -1.0f} where:
///   .x = kNoParticleId = -1 (sentinel for "no neighbor")
///   .y = -1.0f (sentinel for "no distance data")
inline std::vector<Sibernetic::HostFloat2>
makeNeighborMap(uint32_t particleCount) {
  return std::vector<Sibernetic::HostFloat2>(
      static_cast<size_t>(particleCount) * 32u, {-1.0f, -1.0f});
}

/// Helper to inject a neighbor entry into the neighbor map.
/// Writes {neighborId, distance} into the specified slot for a particle.
///
/// Args:
///   neighborMap: The neighbor map vector (size >= particleCount * 32)
///   particleId: Sorted particle index (0-based)
///   slot: Neighbor slot index (0-31), filled left-to-right by findNeighbors
///   neighborId: Neighbor sorted particle ID (must be >= 0; -1 reserved for
///   empty) distance: Distance to neighbor in simulation units (>= 0)
inline void setNeighbor(std::vector<Sibernetic::HostFloat2> &neighborMap,
                        uint32_t particleId, uint32_t slot, int32_t neighborId,
                        float distance) {
  neighborMap[static_cast<size_t>(particleId) * 32u + slot] = {
      static_cast<float>(neighborId), distance};
}

} // namespace SiberneticTest
