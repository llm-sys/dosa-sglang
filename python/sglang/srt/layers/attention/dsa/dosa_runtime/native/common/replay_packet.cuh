#pragma once
#include "h32_common.cuh"

namespace dosa_h32_replay {
using namespace certicascade_streaming_exact;

__device__ __forceinline__ int swizzle(int row, int vec) {
  return row * 8 + (vec ^ (row % 8));
}

// Both recovery policies issue the same H32 packet and consume the same TMEM
// result. The caller retains its active-row reduction and output publication.
template <class Shared, class Inputs>
__device__ __forceinline__ void load_packet(
    Shared& s, const Inputs& a, int64_t key, bool active,
    int t, int tm, int seq, float (&acc)[kHeads]) {
  // No out-of-interval K/scale read, including a partial final packet.
  #pragma unroll
  for(int v=0;v<8;++v)
    reinterpret_cast<uint4*>(s.keys)[swizzle(t,v)]=active
        ? reinterpret_cast<const uint4*>(a.k+uint64_t(key)*128)[v]
        : make_uint4(0,0,0,0);
  s.scale[t]=active?a.scale[key]:0.0f;
  cutlass::arch::fence_view_async_shared();
  __syncthreads();
  if(t==0) {
    deep_gemm::ptx::tcgen05_after_thread_sync();
    issue_h32_mma<128,kHeads>(reinterpret_cast<cutlass::float_e4m3_t*>(s.keys),
        reinterpret_cast<cutlass::float_e4m3_t*>(s.q),0,tm,
        cute::cast_smem_ptr_to_uint(&s.done));
  }
  s.done.wait(seq&1);
  deep_gemm::ptx::tcgen05_after_thread_sync();
  tmem_load_heads(tm,acc);
}
}  // namespace dosa_h32_replay
