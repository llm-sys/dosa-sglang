#pragma once
#include "../../../../../srt/layers/attention/dsa/dosa_runtime/native/common/score_emit_loop.cuh"
#include <deep_gemm/ptx/ld_st.cuh>

namespace dsa_litetopk {

// Full's qualified dense loader has four raw-K stages and two UMMA buffers
// per math warpgroup. Only loading, metadata and ownership live here. The
// common consumer owns all head reduction, predicates and record emission.
template <int Queries, int RawStages, int UmmaStages, int TileKeys,
          int WindowTiles, bool Coalesced>
struct DenseScoreAccess {
  using Barrier = cutlass::arch::ClusterTransactionBarrier;
  static_assert(Queries == 4 && TileKeys == 256);

  const float (&weights)[Queries][32];
  float (&gates)[Queries];
  const float (&affine_origins)[Queries];
  const uint32_t (&starts)[Queries];
  const uint32_t (&spans)[Queries];
  const uint32_t *gate_mailbox;
  const float *key_scales;
  Barrier *full_umma, *empty_umma, *empty_raw;
  uint32_t first_key, tiles, key_count, query_base, tensor_memory;
  uint32_t probe_group, probe_add_max;
  uint64_t probe_magic;
  uint32_t key_offset = 0, accumulator_slot = 0, raw_slot = 0;
  float scale = 0.0f;

  __device__ __forceinline__ int tile_count() const { return int(tiles); }

  __device__ __forceinline__ void before_tile(int tile) {
    // Keep the accepted gate-load placement, before the full-UMMA wait.
    if (tile != 0 && tile % 8 == 0) {
      const float4 gate = deep_gemm::ptx::ld_shared(
          reinterpret_cast<const float4 *>(gate_mailbox));
      gates[0] = gate.x;
      gates[1] = gate.y;
      gates[2] = gate.z;
      gates[3] = gate.w;
    }
    const uint32_t thread = threadIdx.x;
    key_offset = first_key + uint32_t(tile) * TileKeys + thread;
    scale = key_offset < key_count ? key_scales[key_offset] : 0.0f;
    accumulator_slot = (thread / 128) * UmmaStages + uint32_t(tile) % UmmaStages;
    raw_slot = uint32_t(tile) % RawStages;
    full_umma[accumulator_slot].wait((uint32_t(tile) / UmmaStages) & 1u);
    deep_gemm::ptx::tcgen05_after_thread_sync();
    // A flush borrows dead raw stage3 and the scale tail as drain scratch.
    // The final partial window has no future producer. Return this raw credit
    // only after the writer's CTA rendezvous, copy and proxy fence complete.
    if constexpr (Coalesced) {
      if (!flush_tile(tile)) empty_raw[raw_slot].arrive();
    } else {
      empty_raw[raw_slot].arrive();
    }
  }

  __device__ __forceinline__ uint32_t tmem_base(int query_first) const {
    return tensor_memory + accumulator_slot * Queries * 32 + query_first * 32;
  }
  __device__ __forceinline__ void after_tmem_read(bool last) {
    if (last) {
      deep_gemm::ptx::tcgen05_before_thread_sync();
      empty_umma[accumulator_slot].arrive();
    }
  }
  __device__ __forceinline__ void after_emit(int tile) {
    if constexpr (Coalesced) {
      if (flush_tile(tile)) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        empty_raw[raw_slot].arrive();
      }
    }
  }
  __device__ __forceinline__ bool flush_tile(int tile) const {
    return (tile + 1) % WindowTiles == 0 || uint32_t(tile + 1) == tiles;
  }

  __device__ __forceinline__ int row_base() const { return int(query_base); }
  __device__ __forceinline__ bool query_active(int) const { return true; }
  __device__ __forceinline__ bool visible(int query) const {
    return (key_offset - starts[query]) < spans[query];
  }
  __device__ __forceinline__ float weight(int query, int head) const {
    return weights[query][head];
  }
  __device__ __forceinline__ float lane_scale() const { return scale; }
  __device__ __forceinline__ float cutoff(int query) const { return gates[query]; }
  __device__ __forceinline__ float affine_bias(int query) const {
    return affine_origins[query];
  }
  __device__ __forceinline__ uint32_t key_label() const {
    return mapped_key(key_offset);
  }
  __device__ __forceinline__ uint32_t key_for_record(int first_tile, uint32_t tag) const {
    return mapped_key(first_key + (uint32_t(first_tile) + tag) * TileKeys + threadIdx.x);
  }
  __device__ __forceinline__ uint32_t mapped_key(uint32_t key) const {
    if (probe_group != 0) {
      const uint32_t group = uint32_t((uint64_t(key) * probe_magic) >> 42);
      key += min((group + 1) * 64u, probe_add_max);
    }
    return key;
  }
};
} // namespace dsa_litetopk
