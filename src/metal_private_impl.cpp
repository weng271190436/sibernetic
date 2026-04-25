// This translation unit owns the metal-cpp and Foundation private symbol
// definitions. NS_PRIVATE_IMPLEMENTATION and MTL_PRIVATE_IMPLEMENTATION must be
// defined before any metal-cpp headers are included in exactly one .cpp file
// per linked binary.
#ifdef SIBERNETIC_USE_METAL

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "../metal-cpp/Foundation/Foundation.hpp" // IWYU pragma: keep
#include "../metal-cpp/Metal/Metal.hpp"           // IWYU pragma: keep

#endif // SIBERNETIC_USE_METAL
