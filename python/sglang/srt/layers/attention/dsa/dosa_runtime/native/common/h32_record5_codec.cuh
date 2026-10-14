#pragma once
#include "record5_codec.cuh"

// Completed original H32 scores use a global order cell. This is a different
// mathematical domain from Full's bounded surrogate, with the same5B layout.
namespace h32_record5 {
inline constexpr uint32_t Min = 0x00800u, Max = 0xFF7FFu;
inline constexpr uint32_t Invalid = record5::Invalid;

__device__ __forceinline__ uint32_t encode(float score) {
  uint32_t bits = __float_as_uint(score);
  if ((bits & 0x7FFFFFFFu) == 0) bits = 0;
  const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  return (~ordered) >> 12;
}

__device__ __forceinline__ uint32_t pack_word(uint32_t cp, uint32_t id) {
  return record5::pack_word(cp,id);
}
}  // namespace h32_record5
