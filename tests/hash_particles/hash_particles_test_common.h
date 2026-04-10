#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace SiberneticTest {

struct HashParticlesFloat4 {
  float s[4];
};

struct HashParticlesUInt2 {
  uint32_t s[2];
};

struct HashParticlesCase {
  const char *name;
  std::vector<HashParticlesFloat4> positions;
  uint32_t gridCellsX;
  uint32_t gridCellsY;
  uint32_t gridCellsZ;
  float hashGridCellSizeInv;
  float xmin;
  float ymin;
  float zmin;
  std::vector<uint32_t> expectedCellIds;
};

struct HashParticlesResult {
  std::vector<HashParticlesUInt2> particleIndex;
};

class HashParticlesRunner {
public:
  virtual ~HashParticlesRunner() = default;
  virtual HashParticlesResult run(const HashParticlesCase &tc) = 0;
};

inline HashParticlesFloat4 makeFloat4(float x, float y, float z,
                                      float w = 0.0f) {
  HashParticlesFloat4 v;
  v.s[0] = x;
  v.s[1] = y;
  v.s[2] = z;
  v.s[3] = w;
  return v;
}

inline void expectHashParticlesResultMatches(const HashParticlesCase &tc,
                                             const HashParticlesResult &result) {
  ASSERT_EQ(tc.positions.size(), tc.expectedCellIds.size());
  ASSERT_EQ(result.particleIndex.size(), tc.expectedCellIds.size());

  const uint32_t particleCount =
      static_cast<uint32_t>(result.particleIndex.size());
  for (uint32_t i = 0; i < particleCount; ++i) {
    EXPECT_EQ(result.particleIndex[i].s[0], tc.expectedCellIds[i]);
    EXPECT_EQ(result.particleIndex[i].s[1], i);
  }
}

inline const std::vector<HashParticlesCase> &hashParticlesCases() {
  static const std::vector<HashParticlesCase> kCases = {
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
          }},
  };
  return kCases;
}

inline std::string hashParticlesCaseName(
    const ::testing::TestParamInfo<HashParticlesCase> &info) {
  return info.param.name;
}

} // namespace SiberneticTest