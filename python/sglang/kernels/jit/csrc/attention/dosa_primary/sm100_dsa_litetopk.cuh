#pragma once
#include "record5_codec.cuh"
#include "dense_score_access.cuh"

#include <cuda_fp16.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace dsa_litetopk {

using namespace deep_gemm;

using dosa_score_emit::issue_h32_mma;


inline constexpr uint32_t kEmitChunkBlocks = 256;
inline constexpr uint32_t kEmitLaneSlots = 18;
inline constexpr uint32_t kWarpQueueCap = 64;
inline constexpr uint32_t kMathRegisters = 240;
inline constexpr uint32_t kSpecializedRegisters = 24;
inline constexpr uint32_t kTmemRows = 4;
inline constexpr uint32_t kUmmaStages = 2;
inline constexpr uint32_t kSparseRefreshNs = 512;
inline constexpr uint32_t kSparseRefreshIdleNs = 2048;
// A ring slot can only equal this pattern if a +NaN score passed the gate,
// which both gate comparisons reject; the refresher treats it as "empty".
inline constexpr uint32_t kRingSentinel = 0xffffffffu;
// Gate updates only need to keep pace with the 65,536-token flush windows,
// so the refresher can poll orders of magnitude slower than the dynamic
// histogram daemon and stay invisible to the math warps' issue ports.
inline constexpr uint32_t kRingActiveNs = 2048;
inline constexpr uint32_t kRingIdleNs = 2048;
inline constexpr uint32_t kPagedCandidateInlineRecords = 8192;
inline constexpr uint32_t kPagedCandidatePageRecords = 4096;
inline constexpr uint32_t kPagedCandidateOverflowFlag = 1u << 31;

inline constexpr int32_t kPagedCandidateEmpty = -1;
inline constexpr int32_t kPagedCandidateBusy = -2;
inline constexpr int32_t kPagedCandidateFailed = -3;
inline constexpr uint32_t kPagedCandidateBadArguments = 1u << 0;
inline constexpr uint32_t kPagedCandidatePoolExhausted = 1u << 1;
inline constexpr uint32_t kPagedCandidateLogicalOverflow = 1u << 2;

// Five-byte record: cand_val holds the low8 of the compacted FP24 surrogate
// code; cand_idx holds its high12 above the exact20-bit KV index. The local
// ring retains the uncompressed FP24 code and an8-bit block-in-window tag.
using CandidateValue = uint8_t;
constexpr uint32_t kCandidateIndexBits = 20;
constexpr uint32_t kCandidateIndexMask = (1u << kCandidateIndexBits) - 1u;

CUTLASS_DEVICE uint32_t candidate_pack_index(const uint32_t payload,
                                             const uint32_t kv_index) {
  return record5::pack_word(record5::compact(payload),kv_index);
}

CUTLASS_DEVICE uint32_t candidate_ordered_fp32_code(const float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

CUTLASS_DEVICE uint32_t candidate_fp24_code(const float value) {
  return candidate_ordered_fp32_code(value==0.0f?0.0f:value) >> 8;
}

// Full-width counterpart of the dense selector's coarse ordering. Smaller
// unsigned keys correspond to larger FP32 values.
CUTLASS_DEVICE uint32_t candidate_load_score_code(const CandidateValue value,
                                                  const int32_t packed_idx) {
  return record5::full_code(value,packed_idx);
}

CUTLASS_DEVICE float candidate_decode_score(const CandidateValue value,
                                            const int32_t packed_idx) {
  const uint32_t code = candidate_load_score_code(value, packed_idx);
  const uint32_t ordered = code << 8;
  const uint32_t bits =
      (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return __uint_as_float(bits);
}

CUTLASS_DEVICE int32_t candidate_decode_index(const int32_t packed_idx) {
  return static_cast<int32_t>(static_cast<uint32_t>(packed_idx) &
                              kCandidateIndexMask);
}

CUTLASS_DEVICE void store_candidate(CandidateValue *value_dst,
                                    int32_t *index_dst, const float bq,
                                    const uint32_t kv_index) {
  const uint32_t payload = candidate_fp24_code(bq);
  __stcs(value_dst, static_cast<CandidateValue>(record5::compact(payload)));
  __stcs(index_dst,
         static_cast<int32_t>(candidate_pack_index(payload, kv_index)));
}

CUTLASS_DEVICE void paged_candidate_set_status(int32_t *pool_status,
                                               const uint32_t status) {
  if (pool_status != nullptr) {
    atomicOr(reinterpret_cast<unsigned int *>(pool_status),
             static_cast<unsigned int>(status));
  }
}

// The histogram keeps unexpanded surrogate-value bins. Publish only its
// outward expanded edge, so consumers do not carry four epsilon registers.
CUTLASS_DEVICE float surrogate_initialize_edge(float theta, float epsilon,
                                               int32_t *pool_status,
                                               float &two_epsilon) {
  two_epsilon = __uint_as_float(0x7f800000u);
  if (!(epsilon >= 0.0f) || !isfinite(epsilon)) {
    paged_candidate_set_status(pool_status, 8u);
    return two_epsilon;
  }
  two_epsilon = __fmul_ru(2.0f, epsilon);
  const float edge = __fadd_ru(theta, two_epsilon);
  if (!isfinite(edge)) {
    paged_candidate_set_status(pool_status, 8u);
    two_epsilon = __uint_as_float(0x7f800000u);
    return two_epsilon;
  }
  return edge;
}

template <uint32_t kNumHeads, uint32_t kHeadDim, uint32_t BLOCK_Q,
          uint32_t BLOCK_KV, uint32_t kNumQStages, uint32_t kNumKVStages,
          uint32_t kNumSMs, uint32_t kNumSpecializedThreads,
          uint32_t kNumMathThreads,
          uint32_t kNumMathWarpGroups = kNumMathThreads / 128,
          bool kOnlineFixedBuckets = false, bool kStaticHotGate = false,
          bool kStaticHotNoHist = false, bool kStaticHotExactGate = false,
          bool kPagedCandidates = false, bool kPooledSegments = false,
          bool kRingRefresh = false, uint32_t kUmmaStagesT = kUmmaStages,
          uint32_t kEmitLaneSlotsT = kEmitLaneSlots,
          uint32_t kMathRegistersT = kMathRegisters,
          uint32_t kMinBlocksPerSm = 1, uint32_t kTmemRowsT = kTmemRows,
          bool kSharedWeightEpilogue = false, bool kSplitTmemLoad = false,
          bool kCoalescedDrain = false>
CUTLASS_GLOBAL
__launch_bounds__(kNumSpecializedThreads + kNumMathThreads, kMinBlocksPerSm) void sm100_dsa_litetopk(
    const uint32_t seq_len, const uint32_t seq_len_kv,
    uint32_t *cu_seq_len_k_start, uint32_t *cu_seq_len_k_end,
    const float *__restrict__ origin,    // [seq_len]
    const float *__restrict__ inv_delta, // [seq_len]
    int32_t *__restrict__ th_bucket,     // [seq_len]
    int32_t *__restrict__ bcount,        // [seq_len, num_buckets]
    const uint32_t num_buckets, const uint32_t topk,
    const uint32_t refresh_every, const uint32_t num_kv_splits,
    const uint32_t probe_group,            // compacted-space group size
                                           // (pstp-1)*64; 0 = no probe
                                           // compaction (identity map)
    const uint64_t probe_magic,            // ceil(2^42/probe_group):
                                           // exact div via mul-shift
    const uint32_t probe_add_max,          // npage*64 cap for the map
    CandidateValue *__restrict__ cand_val, // [seq_len, cand_cap]
    int32_t *__restrict__ cand_idx,        // [seq_len, cand_cap]
    int32_t *__restrict__ cand_cnt,        // [seq_len]
    const uint32_t cand_cap,
    const __grid_constant__ cute::TmaDescriptor tensor_map_q,
    const __grid_constant__ cute::TmaDescriptor tensor_map_kv,
    // fp8: per-token fp32 dequant scales. fp4: the
    // UE8M0 SF-KV int32 stream (same 1-D shape).
    const __grid_constant__ cute::TmaDescriptor tensor_map_kv_scales,
    const __grid_constant__ cute::TmaDescriptor tensor_map_weights,
    // fp4 only: per-(token, head) packed UE8M0 SF-Q
    // int32 stream; fp8 callers pass any map (unread).
    const __grid_constant__ cute::TmaDescriptor tensor_map_sf_q,
    CandidateValue *__restrict__ overflow_val = nullptr,
    int32_t *__restrict__ overflow_idx = nullptr,
    int32_t *__restrict__ page_table = nullptr,
    uint32_t *__restrict__ pool_head = nullptr,
    int32_t *__restrict__ pool_status = nullptr, const uint32_t pool_pages = 0,
    CandidateValue *__restrict__ = nullptr, int32_t *__restrict__ = nullptr,
    uint64_t *__restrict__ = nullptr, uint32_t *__restrict__ = nullptr,
    int32_t *__restrict__ = nullptr, const uint32_t = 0, const uint32_t = 0,
    uint64_t *__restrict__ = nullptr, int32_t *__restrict__ = nullptr,
    uint32_t *__restrict__ = nullptr, uint32_t *__restrict__ = nullptr,
    const uint32_t = 0, const float *__restrict__ epsilon = nullptr,
    const float *__restrict__ raw_kv_scales = nullptr) {
  static_assert(
      !kOnlineFixedBuckets && kStaticHotGate && kStaticHotNoHist &&
          !kStaticHotExactGate && kPagedCandidates && !kPooledSegments &&
          kRingRefresh && kUmmaStagesT == kUmmaStages &&
          kEmitLaneSlotsT == kEmitLaneSlots &&
          kMathRegistersT == kMathRegisters && kMinBlocksPerSm == 1 &&
          kTmemRowsT == kTmemRows && !kSharedWeightEpilogue && !kSplitTmemLoad,
      "only the qualified paged/ring-refresh production policy is built");
  const bool valid =
      cand_val != nullptr && cand_idx != nullptr && overflow_val != nullptr &&
      overflow_idx != nullptr && page_table != nullptr &&
      pool_head != nullptr && pool_status != nullptr && pool_pages != 0 &&
      cand_cap >= kPagedCandidateInlineRecords && num_kv_splits == 1 &&
      epsilon != nullptr && raw_kv_scales != nullptr;
  if (!valid) {
    if (threadIdx.x == 0) {
      paged_candidate_set_status(pool_status, kPagedCandidateBadArguments);
    }
    return;
  }
  const auto num_q_blocks = math::ceil_div(seq_len, BLOCK_Q);
  using Barrier = cutlass::arch::ClusterTransactionBarrier;

  const auto warp_idx = cutlass::canonical_warp_idx_sync();
  const auto warpgroup_idx = warp_idx / 4;
  const auto lane_idx = ptx::get_lane_idx();
  constexpr uint32_t kSpecWarpStart = kNumMathWarpGroups * 4;
  constexpr uint32_t kNumMathWarps = kNumMathThreads / 32;
  constexpr uint32_t kNumUmmaStages = kUmmaStagesT;
  constexpr uint32_t kActiveEmitLaneSlots = kEmitLaneSlotsT;
  constexpr uint32_t kNumUmmaBuffers = kNumMathWarpGroups * kNumUmmaStages;
  // fp4 replaces the per-warpgroup UMMA double buffers with a shared
  // 3-stage TMEM accumulator pool (the SF columns take the freed space);
  constexpr uint32_t kNumAccumBufs = kNumUmmaBuffers;
  constexpr uint32_t kActiveGateStride = 8u;
  constexpr uint32_t kActiveEmitChunkBlocks = kEmitChunkBlocks;
  DG_STATIC_ASSERT(
      (BLOCK_Q == 4 || BLOCK_Q == 2) && kNumMathWarps == 8,
      "LiteTopK requires BLOCK_Q in {4 (H=32), 2 (H=64)} and 8 math warps");
  DG_STATIC_ASSERT(
      BLOCK_KV == 256,
      "the local ring infers the KV tile coordinate from its owner lane");
  static_assert(32u * kActiveEmitLaneSlots <= kPagedCandidatePageRecords,
                "one warp-row reservation must intersect at most two pages");

  DG_STATIC_ASSERT(kNumSpecializedThreads == 128 and kNumMathThreads % 128 == 0,
                   "Invalid threads");

  if (warp_idx == kSpecWarpStart) {
    cute::prefetch_tma_descriptor(&tensor_map_q);
    cute::prefetch_tma_descriptor(&tensor_map_kv);
    // The original immutable FP32 scale now follows the generic global path.
    cute::prefetch_tma_descriptor(&tensor_map_weights);
  }

  // fp4 packs two e2m1 elements per byte; SF regions are UE8M0 bytes
  // packed 4-per-int32 and padded to the 128-element UTCCP granule.
  static constexpr uint32_t kKvElemBytesNum = 2u; // per 2 elems
  static constexpr uint32_t kNumUTCCPAlignedElems = 128;
  static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE =
      BLOCK_Q * kNumHeads * kHeadDim * kKvElemBytesNum / 2;
  static constexpr uint32_t SMEM_WEIGHT_SIZE_PER_STAGE =
      BLOCK_Q * kNumHeads * sizeof(float);
  static constexpr uint32_t SMEM_KV_SIZE_PER_STAGE =
      BLOCK_KV * kHeadDim * kKvElemBytesNum / 2;
  static constexpr uint32_t kNumSFQ =
      math::constexpr_align(BLOCK_Q * kNumHeads, kNumUTCCPAlignedElems);
  static constexpr uint32_t kNumSFKV =
      math::constexpr_align(BLOCK_KV, kNumUTCCPAlignedElems);
  // fp8: one fp32 dequant scale per token. fp4: one int32 (4x UE8M0) per
  // token, padded to the UTCCP granule.
  static constexpr uint32_t SMEM_KV_SCALE_SIZE_PER_STAGE =
      BLOCK_KV * sizeof(float);
  static constexpr uint32_t SMEM_SF_Q_SIZE = 0u;
  static constexpr uint32_t ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE =
      math::constexpr_align(SMEM_KV_SCALE_SIZE_PER_STAGE, 512u);

  extern __shared__ __align__(512) uint8_t smem_buffer[];
  DG_STATIC_ASSERT(SMEM_Q_SIZE_PER_STAGE % 512 == 0, "Unaligned TMA swizzling");
  DG_STATIC_ASSERT(SMEM_WEIGHT_SIZE_PER_STAGE % 512 == 0,
                   "Unaligned TMA swizzling");
  DG_STATIC_ASSERT(SMEM_KV_SIZE_PER_STAGE % 512 == 0,
                   "Unaligned TMA swizzling");

  constexpr uint32_t kNumAccumTmemCols = BLOCK_Q * kNumHeads * kNumAccumBufs;
  constexpr uint32_t kTmemStartColOfSFQ = kNumAccumTmemCols;
  constexpr uint32_t kTmemStartColOfSFKV = kNumAccumTmemCols + kNumSFQ / 32;
  constexpr uint32_t kNumTmemCols = kNumAccumTmemCols;
  DG_STATIC_ASSERT(kNumTmemCols <= 512, "Too many tensor memory");

  auto smem_q = utils::PatternVisitor([&](const uint32_t &i) {
    return reinterpret_cast<__nv_fp8_e4m3 *>(smem_buffer +
                                             SMEM_Q_SIZE_PER_STAGE * i);
  });
  auto smem_weights = utils::PatternVisitor([&](const uint32_t &i) {
    return reinterpret_cast<float *>(smem_buffer +
                                     SMEM_Q_SIZE_PER_STAGE * kNumQStages +
                                     SMEM_WEIGHT_SIZE_PER_STAGE * i);
  });
  auto smem_kv = utils::PatternVisitor([&](const uint32_t &i) {
    return reinterpret_cast<__nv_fp8_e4m3 *>(
        smem_buffer + (SMEM_Q_SIZE_PER_STAGE * kNumQStages +
                       SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
                       SMEM_KV_SIZE_PER_STAGE * i));
  });
  // Full windows end on raw stage3. The final partial window has no
  // future producer writes. Delay that tile's raw credit until the drain ends.
  constexpr uint32_t kDrainRecordsPerWarp = kActiveEmitLaneSlots * 32u;
  DG_STATIC_ASSERT(kNumKVStages == 4 && kActiveEmitChunkBlocks % 4 == 0,
                   "drain scratch requires the qualified four-stage schedule");
  DG_STATIC_ASSERT(kNumMathWarps * kDrainRecordsPerWarp * sizeof(uint2) <=
                   SMEM_KV_SIZE_PER_STAGE +
                   ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * kNumKVStages,
                   "dead last raw stage and unused scale tail must fit scratch");
  auto drain_records = reinterpret_cast<uint2*>(smem_kv[kNumKVStages - 1]);

  auto smem_kv_scales = utils::PatternVisitor([&](const uint32_t &i) {
    return reinterpret_cast<float *>(smem_buffer +
                                     SMEM_Q_SIZE_PER_STAGE * kNumQStages +
                                     SMEM_WEIGHT_SIZE_PER_STAGE * kNumQStages +
                                     SMEM_KV_SIZE_PER_STAGE * kNumKVStages +
                                     ALIGNED_SMEM_KV_SCALE_SIZE_PER_STAGE * i);
  });

  auto smem_sf_q = reinterpret_cast<uint32_t *>(
      reinterpret_cast<uint8_t *>(smem_kv_scales[kNumKVStages]));
  auto barrier_ptr = reinterpret_cast<Barrier *>(
      reinterpret_cast<uint8_t *>(smem_sf_q) + SMEM_SF_Q_SIZE);
  auto full_q_barriers =
      utils::PatternVisitor([&](const uint32_t &i) { return barrier_ptr + i; });
  auto empty_q_barriers = utils::PatternVisitor(
      [&](const uint32_t &i) { return barrier_ptr + (kNumQStages + i); });
  auto full_kv_barriers = utils::PatternVisitor(
      [&](const uint32_t &i) { return barrier_ptr + (kNumQStages * 2 + i); });
  auto empty_kv_barriers = utils::PatternVisitor([&](const uint32_t &i) {
    return barrier_ptr + (kNumQStages * 2 + kNumKVStages + i);
  });
  auto full_umma_barriers = utils::PatternVisitor([&](const uint32_t &i) {
    return barrier_ptr + (kNumQStages * 2 + kNumKVStages * 2 + i);
  });
  auto empty_umma_barriers = utils::PatternVisitor([&](const uint32_t &i) {
    return barrier_ptr +
           (kNumQStages * 2 + kNumKVStages * 2 + kNumAccumBufs + i);
  });

  // FP8 has no separate scale-factor-ready barriers. Keep the host SMEM
  // formula and the device high-water mark identical; the old unused
  // kNumKVStages-sized gap only happened to fit inside allocation rounding.
  auto tmem_ptr_in_smem = reinterpret_cast<uint32_t *>(
      barrier_ptr + kNumQStages * 2 + kNumKVStages * 2 + kNumAccumBufs * 2);
  auto scan_done_flag = reinterpret_cast<volatile int *>(tmem_ptr_in_smem + 1);
  auto warpq_count = reinterpret_cast<int32_t *>(tmem_ptr_in_smem + 4);
  // Chunked emit does not use the legacy warp-queue count words. Reuse four
  // of them as a CTA-local Gate4 mailbox. Values are positive float edge
  // bit-patterns, so unsigned atomicMin is exactly a monotonic gate tighten.
  auto sparse_gate_bits = reinterpret_cast<uint32_t *>(warpq_count);
  auto emit_smem_records =
      reinterpret_cast<uint32_t *>(warpq_count + kNumMathWarps * BLOCK_Q);
  auto smem_hist = reinterpret_cast<int32_t *>(
      emit_smem_records + kNumMathWarps * BLOCK_Q * kActiveEmitLaneSlots * 32u);
  // Ring-refresher scratch sits after the refresher's histogram (which
  // reuses the dynamic-path histogram region): per-(math warp, row) flush
  // sequence words, the spare warps' shadow copies, and per-lane
  // consumed-slot cursors.
  auto ring_seq = reinterpret_cast<uint32_t *>(smem_hist + BLOCK_Q * 256);
  auto ring_shadow = ring_seq + kNumMathWarps * BLOCK_Q;
  auto ring_pending =
      reinterpret_cast<int32_t *>(ring_shadow + kNumMathWarps * BLOCK_Q);
  auto ring_stale = ring_pending + BLOCK_Q;
  auto ring_last = ring_stale + BLOCK_Q;
  auto ring_progress = reinterpret_cast<uint8_t *>(ring_last + BLOCK_Q);
  // Dedicated immutable tail, after every live progress byte. All four words
  // are initialized before the role-split CTA barrier below.
  auto shared_two_epsilon = reinterpret_cast<uint32_t *>(
      ring_progress + kNumMathWarps * BLOCK_Q * 32u);

  DG_STATIC_ASSERT(kNumSpecializedThreads % 128 == 0 and
                       kNumSpecializedThreads >= 64,
                   "Invalid threads");
  if (warp_idx == kSpecWarpStart and cute::elect_one_sync()) {
#pragma unroll
    for (uint32_t i = 0; i < kNumQStages; ++i) {
      full_q_barriers[i]->init(1);
      empty_q_barriers[i]->init(kNumMathThreads + 32);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumKVStages; ++i) {
      full_kv_barriers[i]->init(1);
      empty_kv_barriers[i]->init(kNumMathThreads);
    }
    *scan_done_flag = 0;
    cutlass::arch::fence_barrier_init();
  }
  if (warp_idx == kSpecWarpStart + 1) {
    if (cute::elect_one_sync()) {
#pragma unroll
      for (uint32_t i = 0; i < kNumAccumBufs; ++i) {
        full_umma_barriers[i]->init(1);
        empty_umma_barriers[i]->init(128);
      }
      cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm tmem_allocator;
    tmem_allocator.allocate(kNumTmemCols, tmem_ptr_in_smem);
  }
  for (uint32_t idx = threadIdx.x;
       idx < kNumMathWarps * BLOCK_Q * kActiveEmitLaneSlots * 32u;
       idx += blockDim.x) {
    emit_smem_records[idx] = kRingSentinel;
  }
  // Preload the seed sample's full bucket histogram as the daemon's
  // refresh base. The sample records are genuine row records in the
  // same (origin, inv) bucket space and the exact-once scan starts after
  // the sampled prefix, so ring harvests never recount them; base plus
  // harvested stays a subset of the row's true records and every
  // publish remains a provable upper bound.
  for (uint32_t idx = threadIdx.x; idx < BLOCK_Q * num_buckets;
       idx += blockDim.x) {
    const uint32_t r = idx / num_buckets;
    const uint32_t row_q = static_cast<uint32_t>(blockIdx.x) * BLOCK_Q + r;
    smem_hist[idx] =
        row_q < seq_len
            ? __ldcg(bcount + static_cast<uint64_t>(row_q) * num_buckets +
                     (idx - r * num_buckets))
            : 0;
  }
  for (uint32_t idx = threadIdx.x; idx < kNumMathWarps * BLOCK_Q;
       idx += blockDim.x) {
    ring_seq[idx] = 0u;
    ring_shadow[idx] = 0u;
  }
  for (uint32_t idx = threadIdx.x; idx < BLOCK_Q; idx += blockDim.x) {
    ring_pending[idx] = 0;
    ring_last[idx] = 0;
    ring_stale[idx] = blockIdx.x * BLOCK_Q + idx < seq_len ? 0 : 2;
  }
  for (uint32_t idx = threadIdx.x; idx < kNumMathWarps * BLOCK_Q * 32u;
       idx += blockDim.x) {
    ring_progress[idx] = 0u;
  }
  if (threadIdx.x < BLOCK_Q) {
    const uint32_t row_q =
        static_cast<uint32_t>(blockIdx.x) * BLOCK_Q + threadIdx.x;
    const uint32_t row = min(row_q, seq_len - 1);
    int gate = __ldcg(th_bucket + row);
    if (row_q < seq_len) {
      const int seed_count = __ldcg(cand_cnt + row_q);
      const bool bad_seed = seed_count < 2048 || seed_count > 8192 ||
                            gate < 0 || gate >= int(num_buckets);
      if (bad_seed) {
        // Each C1 row has one CTA and this precedes its emission barrier.
        // A late negative/huge count must not become an atomic output offset.
        cand_cnt[row_q] = 0;
        paged_candidate_set_status(pool_status, 8u);
        gate = 0;
      }
    }
    // A padded row can borrow the final real row's metadata; sanitize its
    // gate arithmetic too, even though it never emits candidate records.
    if (gate < 0 || gate >= int(num_buckets)) gate = 0;
    float two_epsilon;
    const float edge = surrogate_initialize_edge(
        static_cast<float>(gate + 1), epsilon[row], pool_status, two_epsilon);
    ptx::st_shared(shared_two_epsilon + threadIdx.x, __float_as_uint(two_epsilon));
    ptx::st_shared(sparse_gate_bits + threadIdx.x, __float_as_uint(edge));
  }
  // One uniform release for the specialized warpgroup. The existing CTA
  // initialization barrier also completes the warpgroup synchronization.
  if (warp_idx >= kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_dealloc<kSpecializedRegisters>();
  }
  __syncthreads();
  const uint32_t tmem_base = ptx::ld_shared(tmem_ptr_in_smem);

  constexpr uint32_t kNumSpecializedRegisters = kSpecializedRegisters;
  constexpr uint32_t kNumMathRegisters = kMathRegistersT;

  // V1 KV-split scheduling: blockIdx.x = q-block (one per CTA), blockIdx.y =
  // contiguous KV sub-window. Split boundaries are BLOCK_KV-aligned.
  const uint32_t block_q_idx = blockIdx.x;
  const uint32_t kv_split = blockIdx.y;
  uint32_t seq_k_start[BLOCK_Q], seq_k_end[BLOCK_Q];
  const auto load_schedule =
      [&](const uint32_t block_q_idx) -> cute::tuple<uint32_t, uint32_t> {
    uint32_t start = cute::numeric_limits<uint32_t>::max();
    uint32_t end = cute::numeric_limits<uint32_t>::min();

#pragma unroll
    for (uint32_t i = 0; i < BLOCK_Q; ++i) {
      const auto q_idx = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
      seq_k_start[i] = cu_seq_len_k_start[q_idx];
      seq_k_end[i] = cu_seq_len_k_end[q_idx];
      if (block_q_idx * BLOCK_Q + i >= seq_len) {
        // Padded row of a ragged final q-block: empty, aggregation-neutral.
        seq_k_start[i] = seq_len_kv;
        seq_k_end[i] = 0;
      }
      start = min(start, min(seq_k_start[i], seq_len_kv));
      end = max(end, min(seq_k_end[i], seq_len_kv));
    }
    start = start / 4 * 4; // TMA alignment for SF KV
    if (num_kv_splits > 1) {
      // Partition the actual suffix relative to its 4-aligned start.
      // Absolute BLOCK_KV boundaries overlap when start%256 != 0: the
      // final 256-wide TMA tile of one split then crosses split_hi and
      // the following split reads those records again.  Relative tile
      // boundaries keep ownership disjoint for radix/prefix-cache bases
      // such as 64/128/192 while the existing per-row range check masks
      // the final partial tile.
      const uint32_t suffix_blocks =
          end > start ? math::ceil_div(end - start, BLOCK_KV) : 0;
      const uint32_t blocks_per_split =
          math::ceil_div(suffix_blocks, num_kv_splits);
      const uint32_t split_lo =
          min(start + kv_split * blocks_per_split * BLOCK_KV, end);
      const uint32_t split_hi =
          min(start + (kv_split + 1) * blocks_per_split * BLOCK_KV, end);
      start = split_lo;
      end = split_hi;
    }
    const uint32_t nkv =
        (end > start) ? math::ceil_div(end - start, BLOCK_KV) : 0;
    return {start, nkv};
  };

  const auto get_kv_pipeline =
      [&](const uint32_t &kv_block_idx) -> cute::tuple<uint32_t, uint32_t> {
    return {kv_block_idx % kNumKVStages, (kv_block_idx / kNumKVStages) & 1};
  };

  constexpr uint32_t UMMA_M = 128;
  constexpr uint32_t UMMA_K = 32 / sizeof(cutlass::float_e4m3_t);
  constexpr uint32_t UMMA_N = BLOCK_Q * kNumHeads;

  if (warp_idx == kSpecWarpStart) {

    if (cute::elect_one_sync()) {
      if (block_q_idx < num_q_blocks) {
        // Q + weights once for this q-block. Inner extents are BYTES
        // for the uint8 maps, so fp4's packed rows halve them.
        constexpr uint32_t kKvInnerBytes = kHeadDim * kKvElemBytesNum / 2;
        tma::copy<kKvInnerBytes, BLOCK_Q * kNumHeads, kKvInnerBytes>(
            &tensor_map_q, full_q_barriers[0], smem_q[0], 0,
            block_q_idx * BLOCK_Q * kNumHeads);
        tma::copy<kNumHeads, BLOCK_Q, 0>(&tensor_map_weights,
                                         full_q_barriers[0], smem_weights[0], 0,
                                         block_q_idx * BLOCK_Q);
        full_q_barriers[0]->arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE +
                                                 SMEM_WEIGHT_SIZE_PER_STAGE);

        CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
        for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks;
             ++kv_block_idx) {
          CUTE_TIE_DECL(get_kv_pipeline(kv_block_idx), kv_stage_idx, kv_phase);
          empty_kv_barriers[kv_stage_idx]->wait(kv_phase ^ 1);

          tma::copy<kKvInnerBytes, BLOCK_KV, kKvInnerBytes>(
              &tensor_map_kv, full_kv_barriers[kv_stage_idx],
              smem_kv[kv_stage_idx], 0, kv_start + kv_block_idx * BLOCK_KV);
          // Only raw K belongs to this TMA transaction. Scales are immutable
          // generic-global reads by the owning math lanes.
          full_kv_barriers[kv_stage_idx]->arrive_and_expect_tx(
              SMEM_KV_SIZE_PER_STAGE);
        }
      }
    }
  } else if (warp_idx == kSpecWarpStart + 1) {

    DG_TRAP_ONLY_DEVICE_ASSERT(tmem_base % kNumTmemCols == 0 &&
                               tmem_base + kNumTmemCols <= 512);

    auto instr_desc = cute::UMMA::make_instr_desc<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float, UMMA_M, UMMA_N,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
    auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);

    if (block_q_idx < num_q_blocks) {
      CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
      full_q_barriers[0]->wait(0);

      const bool issue_lane = cute::elect_one_sync();
      for (uint32_t kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        const uint32_t kvg=kv_block_idx;
        CUTE_TIE_DECL(get_kv_pipeline(kvg),kv_stage_idx,kv_phase);
        if(issue_lane) {
          full_kv_barriers[kv_stage_idx]->wait(kv_phase);
          const uint32_t umma_stage=kvg%kNumUmmaStages;
          const uint32_t umma_phase=(kvg/kNumUmmaStages)&1;
          #pragma unroll
          for(uint32_t i=0;i<kNumMathWarpGroups;++i) {
            const uint32_t buf=i*kNumUmmaStages+umma_stage;
            empty_umma_barriers[buf]->wait(umma_phase^1);
            ptx::tcgen05_after_thread_sync();
            issue_h32_mma<128,UMMA_N>(smem_kv[kv_stage_idx],smem_q[0],i*128,
                tmem_base+buf*UMMA_N,
                uint32_t(__cvta_generic_to_shared(full_umma_barriers[buf])));
          }
        }
        __syncwarp();
      }
      empty_q_barriers[0]->arrive();
    }
  } else if (warp_idx == kSpecWarpStart + 2 or warp_idx == kSpecWarpStart + 3) {
    // Spare-warp threshold-refresh daemon. Keeping refresh off the math
    // warps avoids placing histogram scan latency on their critical path.

    if (block_q_idx < num_q_blocks) {
      const uint32_t spare_id = warp_idx - (kSpecWarpStart + 2); // 0 or 1
      // The production ring daemon tightens from emitted records.
      // fixed scan-time gate. Avoid every intermediate prefix scan and
      // publish the tight full-scan boundary certificate only once.
      // Ring-readback refresher: histogram already-emitted smem
      // ring records (word-atomic reads) and tighten the CTA gate
      // mailbox with zero additions to the math-warp hit path.
      // Counting a SUBSET of emitted records keeps every published
      // edge a provable lower bound on the row's current K-th
      // best, and the per-(math warp, row) flush seqlock discards
      // any read that raced a drain so no record counts twice.
      //
      // The production daemon is always active. Quiet passes use a
      // fixed 2048 ns sleep; keeping this compile-time removes the
      // former launch-time daemon/pacing A/B branches.
      uint32_t pass = 0;
      while (*scan_done_flag == 0) {
        const uint32_t local_row =
            BLOCK_Q == 4 ? spare_id + ((pass & 1u) << 1) : spare_id;
        ++pass;
        uint32_t fresh_total = 0;
        if (block_q_idx * BLOCK_Q + local_row < seq_len) {
          for (uint32_t w = 0; w < kNumMathWarps; ++w) {
            const uint32_t pair = w * BLOCK_Q + local_row;
            uint32_t s0 = 0u;
            if (lane_idx == 0) {
              s0 = *reinterpret_cast<volatile uint32_t *>(ring_seq + pair);
            }
            s0 = __shfl_sync(0xffffffffu, s0, 0);
            if (s0 != ring_shadow[pair]) {
              // New flush epoch: the drained column is
              // sentinel again and refills from zero.
              ring_progress[pair * 32u + lane_idx] = 0u;
              __syncwarp();
              if (lane_idx == 0)
                ring_shadow[pair] = s0;
              __syncwarp();
            }
            // Deep harvest: drain the lane's whole staged
            // column per visit (up to kActiveEmitLaneSlots) instead
            // of one record per pass. The flood-time daemon
            // bottleneck is harvest bandwidth, not publish
            // cadence; reads and smem atomics only, so the
            // math warps' issue ports stay untouched.
            uint32_t progress = ring_progress[pair * 32u + lane_idx];
            uint32_t took = 0;
            while (progress < kActiveEmitLaneSlots) {
              const uint32_t record = *reinterpret_cast<volatile uint32_t *>(
                  emit_smem_records +
                  (pair * kActiveEmitLaneSlots + progress) * 32u + lane_idx);
              if (record == kRingSentinel)
                break;
              // Lanes leave this loop at different sentinel
              // positions.  Read one sequence value per
              // currently-active subgroup so every lane that
              // validates this record makes the same epoch
              // decision without naming inactive lanes.
              const unsigned seq_mask = __activemask();
              const int seq_reader = __ffs(seq_mask) - 1;
              uint32_t s1 = 0u;
              if (static_cast<int>(lane_idx) == seq_reader) {
                s1 = *reinterpret_cast<volatile uint32_t *>(ring_seq + pair);
              }
              s1 = __shfl_sync(seq_mask, s1, seq_reader);
              if (s1 != s0) {
                // Raced a flush drain: discard this
                // read; the epoch reset above replays
                // the refilled column next visit.
                break;
              }
              // Invert the truncated fp24 code back to
              // the affine bucket value. Truncation only
              // rounds toward better scores and never
              // crosses an integer boundary (codes and
              // integer edges share the 0x100-aligned
              // grid), so unshifted bins support a
              // pad-free published edge. The comparison
              // ladder (not fmin/fmax) routes NaN and
              // out-of-range scores to the top bin, which
              // the publish guard below never counts.
              uint32_t fbits = record & 0xffffff00u;
              fbits = (fbits & 0x80000000u) ? (fbits ^ 0x80000000u) : ~fbits;
              const float value = __uint_as_float(fbits);
              uint32_t bucket;
              if (value < 0.0f) {
                bucket = 0u;
              } else if (value < static_cast<float>(num_buckets - 1)) {
                bucket = static_cast<uint32_t>(value);
              } else {
                bucket = num_buckets - 1;
              }
              atomicAdd(smem_hist + local_row * num_buckets + bucket, 1);
              ++progress;
              ++took;
            }
            if (took != 0u) {
              ring_progress[pair * 32u + lane_idx] =
                  static_cast<uint8_t>(progress);
            }
            const uint32_t fresh = __reduce_add_sync(0xffffffffu, took);
            if (lane_idx == 0 && fresh != 0u) {
              ring_pending[local_row] += static_cast<int32_t>(fresh);
            }
            fresh_total += fresh;
          }
          __syncwarp();
          // The seed base already carries K records, so each
          // topk/16 batch of fresh evidence can move the
          // boundary.
          const int32_t pend = ring_pending[local_row];
          if (pend - ring_last[local_row] >= static_cast<int32_t>(topk) / 16) {
            if (lane_idx == 0)
              ring_last[local_row] = pend;
            __syncwarp();
            // Smallest bucket prefix holding at least K ring
            // records. The truncated codes and the ordered
            // codes of integers <= 256 both sit on the
            // 0x100-aligned grid, so a counted record's
            // true code is < ord(found + 1) exactly and the
            // bare bucket upper boundary is a provably safe
            // published edge — no truncation pad needed.
            int carry = 0;
            int found = -1;
            for (uint32_t bbase = 0; bbase < num_buckets && found < 0;
                 bbase += 32) {
              const uint32_t b = bbase + lane_idx;
              const int v =
                  b < num_buckets ? smem_hist[local_row * num_buckets + b] : 0;
              const int group_sum = __reduce_add_sync(0xffffffffu, v);
              if (carry + group_sum < static_cast<int>(topk)) {
                carry += group_sum;
                continue;
              }
              int prefix = v;
#pragma unroll
              for (int off = 1; off < 32; off <<= 1) {
                const int nsh = __shfl_up_sync(0xffffffffu, prefix, off);
                if (static_cast<int>(lane_idx) >= off)
                  prefix += nsh;
              }
              const bool hit = carry + prefix >= static_cast<int>(topk);
              const unsigned hm = __ballot_sync(0xffffffffu, hit);
              if (hm) {
                found = static_cast<int>(bbase) + __ffs(hm) - 1;
              } else {
                carry += __shfl_sync(0xffffffffu, prefix, 31);
              }
            }
            if (lane_idx == 0) {
              bool published = false;
              // The top bin collects NaN and clamped
              // out-of-range records, so an edge may only
              // be published from a strictly lower bin.
              if (found >= 0 && found < static_cast<int>(num_buckets) - 1) {
                const float two_epsilon = __uint_as_float(
                    ptx::ld_shared(shared_two_epsilon + local_row));
                const uint32_t edge = __float_as_uint(__fadd_ru(
                    static_cast<float>(found + 1), two_epsilon));
                published =
                    edge < atomicMin(sparse_gate_bits + local_row, edge);
              }
              // Early warm-start scans that fail to
              // tighten only lack fresh evidence, not
              // convergence; they must not retire the
              // daemon.
              ring_stale[local_row] = published
                                          ? 0
                                          : (pend >= static_cast<int32_t>(topk)
                                                 ? ring_stale[local_row] + 1
                                                 : ring_stale[local_row]);
            }
            __syncwarp();
          }
        }
        // Quiet pacing. The CTA retires promptly once the math
        // warps finish; bursts skip pacing and chain straight
        // into the next pass.
        if (fresh_total < 64u) {
          uint32_t slept = 0;
          while (slept < kRingIdleNs && *scan_done_flag == 0) {
            __nanosleep(kRingActiveNs);
            slept += kRingActiveNs;
          }
        }
      }
    }
  } else if (warp_idx < kSpecWarpStart) {
    cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

    // Bucket comparisons use the affine score space. Raw float bit
    // patterns are nonuniform across exponent ranges and cannot preserve
    // this threshold contract.
    float weight_storage[BLOCK_Q][kNumHeads];
    float o_reg[BLOCK_Q], inv_reg[BLOCK_Q], vth_reg[BLOCK_Q];
    uint32_t kstart_reg[BLOCK_Q],
        kspan_reg[BLOCK_Q]; // unsigned range-check trick

    if (block_q_idx < num_q_blocks) {
      CUTE_TIE_DECL(load_schedule(block_q_idx), kv_start, num_kv_blocks);
      full_q_barriers[0]->wait(0);

// Qualified baseline: keep all four rows resident in registers.
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < kNumHeads; ++j) {
          weight_storage[i][j] =
              ptx::ld_shared(smem_weights[0] + i * kNumHeads + j);
        }
      }
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
        const uint32_t rq = min(block_q_idx * BLOCK_Q + i, seq_len - 1);
        o_reg[i] = origin[rq];
        inv_reg[i] = inv_delta[rq];
        vth_reg[i] = -o_reg[i] * inv_reg[i];
        o_reg[i] = 0.0f; // gate closed until the first consume
        kstart_reg[i] = seq_k_start[i];
        kspan_reg[i] =
            seq_k_end[i] > seq_k_start[i] ? seq_k_end[i] - seq_k_start[i] : 0;
      }
// The metadata envelope uses these actual RN folded coefficients.
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_Q; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < kNumHeads; ++j)
          weight_storage[i][j] = __fmul_rn(weight_storage[i][j], -inv_reg[i]);
      }
      const float4 initial_gate =
          ptx::ld_shared(reinterpret_cast<const float4 *>(sparse_gate_bits));
      o_reg[0] = initial_gate.x;
      o_reg[1] = initial_gate.y;
      o_reg[2] = initial_gate.z;
      o_reg[3] = initial_gate.w;

      using Access = DenseScoreAccess<BLOCK_Q, kNumKVStages, kNumUmmaStages,
                                      BLOCK_KV, kActiveEmitChunkBlocks, kCoalescedDrain>;
      Access access{weight_storage, o_reg, vth_reg, kstart_reg, kspan_reg,
                    sparse_gate_bits, raw_kv_scales, full_umma_barriers[0],
                    empty_umma_barriers[0], empty_kv_barriers[0], kv_start,
                    num_kv_blocks, seq_len_kv, block_q_idx * BLOCK_Q, tmem_base,
                    probe_group, probe_add_max, probe_magic};
      using Geometry = dosa_score_emit::Geometry<
          BLOCK_Q, kNumMathThreads, BLOCK_Q, BLOCK_Q * kNumHeads,
          kActiveEmitLaneSlots, kActiveEmitChunkBlocks, kCoalescedDrain, kRingRefresh, 1>;
      const dosa_score_emit::RecordOutput output{
          cand_val, overflow_val, cand_idx, overflow_idx, cand_cnt, nullptr,
          page_table, pool_status, pool_head, int(seq_len), int(cand_cap), int(pool_pages),
          kPagedCandidateInlineRecords, kPagedCandidateInlineRecords};
      const dosa_score_emit::EmitStorage emit{
          emit_smem_records, drain_records, ring_seq, nullptr, nullptr};
      dosa_score_emit::run_score_emit<Geometry, dosa_score_emit::Domain::FullSurrogate24>(
          access, output, emit);

      // Mandatory full metadata already certifies finite healthy results.
      // Rejected rows retain independent error through bounded scan to replay.
      // Every actual chunk was published at its final KV block.
      empty_q_barriers[0]->arrive();
    }

    // Signal the refresh daemon, then free tensor memory.
    cutlass::arch::NamedBarrier(kNumMathThreads, 0).sync();
    if constexpr (!kStaticHotNoHist || kRingRefresh) {
      if (threadIdx.x == 0) {
        __threadfence_block();
        *scan_done_flag = 1;
        if constexpr (kRingRefresh) {
          // Prompt-wake a daemon parked on a stage-0 parity: odd
          // count so a straggler that missed transient flips
          // still sees the final parity flipped. The in-wait
          // exit-flag recheck makes this an optimization, not a
          // correctness requirement.
          asm volatile("" ::: "memory");
#pragma unroll
          for (int poke = 0; poke < 5; ++poke) {
            full_kv_barriers[0]->arrive();
          }
        }
      }
    }
    if (warp_idx == 0)
      cute::TMEM::Allocator1Sm().free(tmem_base, kNumTmemCols);
  }
}

} // namespace dsa_litetopk
