#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace Sibernetic {

using HostFloat4 = std::array<float, 4>;
using HostFloat2 = std::array<float, 2>;
using HostUInt2 = std::array<uint32_t, 2>;
using HostUInt4 = std::array<uint32_t, 4>;

// Span aliases for Input structs
using Float4Span = std::span<const HostFloat4>;
using Float2Span = std::span<const HostFloat2>;
using UInt2Span = std::span<const HostUInt2>;
using UInt32Span = std::span<const uint32_t>;

} // namespace Sibernetic
