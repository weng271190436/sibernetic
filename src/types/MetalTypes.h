#pragma once

// Apple's SIMD types are memory-compatible with Metal shader types
// and provide nice accessors (.x, .y, .z, .w) plus SIMD operations.
#include <simd/simd.h>

namespace Sibernetic {

// Re-export Apple's types for consistency
using MetalFloat4 = simd_float4;
using MetalFloat2 = simd_float2;
using MetalUInt2 = simd_uint2;
using MetalUInt4 = simd_uint4;

} // namespace Sibernetic
