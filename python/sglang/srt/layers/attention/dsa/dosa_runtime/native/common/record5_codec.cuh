#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace record5 {
inline constexpr uint32_t Base=0xC00000u, Invalid=0xFFFFFu;
// A caller first applies its domain-specific encoding. Both record writers
// then use this same low8/high12 plus exact20-bit-ID storage representation.
__device__ __forceinline__ uint32_t pack_word(uint32_t code,uint32_t id) {
  return ((code>>8)<<20)|(id&0xFFFFFu);
}
// Input remains the original canonicalized ordered-F32 high24 code.
__device__ __forceinline__ uint32_t compact(uint32_t c) {
  return c<=Base ? 0u : (c<Base+Invalid ? c-Base : Invalid);
}
__device__ __forceinline__ uint32_t full_code(uint8_t low,int32_t packed) {
  const uint32_t c=(static_cast<uint32_t>(packed)>>20)<<8 | uint32_t(low);
  return c==Invalid ? 0xFFFFFFu : (Base|c);
}
}
