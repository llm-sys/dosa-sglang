#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace record4 {
__device__ __forceinline__ uint32_t ordered_cell(float x) {
  uint32_t bits = __float_as_uint(x);
  if ((bits & 0x7FFFFFFFu) == 0) bits = 0;
  return ((bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u)) >> 14;
}

__device__ __forceinline__ uint32_t encode(float score, float threshold) {
  const int delta = 0xC00 + int(ordered_cell(threshold)) - int(ordered_cell(score));
  return uint32_t(max(delta, 0));
}

__device__ __forceinline__ uint32_t full_code(int32_t word) {
  const uint32_t cp = uint32_t(word) >> 20;
  // Includes reserved FFF: invalid encodings never decode to a finite value.
  return cp > 0xC00u ? 0xFFFFFFu : (0xC00000u | (cp << 6));
}
}  // namespace record4
