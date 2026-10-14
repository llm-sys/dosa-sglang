// Qualified FP8/H32 full scorer, checked8K seed, and record5 coarse selection.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <torch/extension.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "sm100_dsa_litetopk.cuh"
#include "coarse_selector.cuh"

namespace {
using CandidateValue = dsa_litetopk::CandidateValue;

// Paged-candidate post-processing ABI.  Keep these constants beside the host
// validation so the producer, finalizer, selector, and Python-owned workspace
// cannot silently drift apart.
constexpr int kPagedCandidateInline = 8192;
constexpr int kPagedCandidatePage = 4096;
constexpr int kPagedCandidatePageShift = 12;

static_assert((1 << kPagedCandidatePageShift) == kPagedCandidatePage,
              "paged candidate page size must be a power of two");
static_assert(kPagedCandidateInline % kPagedCandidatePage == 0,
              "paged candidate inline capacity must be page aligned");

struct PagedCandidateShape {
  int rows;
  int overflow_pages;
  int table_pages_per_row;
  int logical_cap;
};

static bool tensor_storage_ranges_overlap(const torch::Tensor &a,
                                          const torch::Tensor &b);

static CandidateValue *candidate_data_ptr(torch::Tensor &tensor) {
  return reinterpret_cast<CandidateValue *>(tensor.data_ptr<uint8_t>());
}

static void check_candidate_dtype(const torch::Tensor &tensor) {
  TORCH_CHECK(tensor.scalar_type() == torch::kByte,
              "cand_val must use uint8 as opaque record5 storage");
}

static PagedCandidateShape check_paged_candidate_layout(
    const torch::Tensor &inline_val, const torch::Tensor &inline_idx,
    const torch::Tensor &overflow_val, const torch::Tensor &overflow_idx,
    const torch::Tensor &page_table, int64_t logical_cap64) {
  TORCH_CHECK(inline_val.is_cuda() && inline_idx.is_cuda() &&
                  overflow_val.is_cuda() && overflow_idx.is_cuda() &&
                  page_table.is_cuda(),
              "paged candidate tensors must be CUDA");
  TORCH_CHECK(inline_val.device() == inline_idx.device() &&
                  inline_val.device() == overflow_val.device() &&
                  inline_val.device() == overflow_idx.device() &&
                  inline_val.device() == page_table.device(),
              "paged candidate tensors must be on one CUDA device");
  TORCH_CHECK(inline_val.is_contiguous() && inline_idx.is_contiguous() &&
                  overflow_val.is_contiguous() &&
                  overflow_idx.is_contiguous() && page_table.is_contiguous(),
              "paged candidate tensors must be contiguous");
  check_candidate_dtype(inline_val);
  check_candidate_dtype(overflow_val);
  TORCH_CHECK(inline_idx.scalar_type() == torch::kInt &&
                  overflow_idx.scalar_type() == torch::kInt &&
                  page_table.scalar_type() == torch::kInt,
              "paged candidate indices and page_table must be int32");
  TORCH_CHECK(inline_val.dim() == 2 &&
                  inline_val.size(1) == kPagedCandidateInline &&
                  inline_idx.sizes() == inline_val.sizes(),
              "inline_val/inline_idx must be [Q,8192]");
  TORCH_CHECK(overflow_val.dim() == 2 &&
                  overflow_val.size(1) == kPagedCandidatePage &&
                  overflow_idx.sizes() == overflow_val.sizes(),
              "overflow_val/overflow_idx must be [P,4096]");
  TORCH_CHECK(page_table.dim() == 2 && page_table.size(0) == inline_val.size(0),
              "page_table must be [Q,Npages]");
  TORCH_CHECK(!tensor_storage_ranges_overlap(inline_val, inline_idx) &&
                  !tensor_storage_ranges_overlap(inline_val, overflow_val) &&
                  !tensor_storage_ranges_overlap(inline_val, overflow_idx) &&
                  !tensor_storage_ranges_overlap(inline_val, page_table) &&
                  !tensor_storage_ranges_overlap(inline_idx, overflow_val) &&
                  !tensor_storage_ranges_overlap(inline_idx, overflow_idx) &&
                  !tensor_storage_ranges_overlap(inline_idx, page_table) &&
                  !tensor_storage_ranges_overlap(overflow_val, overflow_idx) &&
                  !tensor_storage_ranges_overlap(overflow_val, page_table) &&
                  !tensor_storage_ranges_overlap(overflow_idx, page_table),
              "paged candidate arenas and page_table must not alias");
  TORCH_CHECK(inline_val.size(0) > 0 &&
                  inline_val.size(0) <= std::numeric_limits<int>::max() &&
                  overflow_val.size(0) <= std::numeric_limits<int>::max() &&
                  page_table.size(1) <= std::numeric_limits<int>::max(),
              "paged candidate dimensions exceed the int32 ABI");
  TORCH_CHECK(logical_cap64 >= 1 &&
                  logical_cap64 < std::numeric_limits<int32_t>::max(),
              "logical_cap must leave room for an int32 overflow sentinel");
  const int64_t overflow_capacity =
      std::max<int64_t>(logical_cap64 - kPagedCandidateInline, 0);
  const int64_t required_table_pages =
      (overflow_capacity + kPagedCandidatePage - 1) / kPagedCandidatePage;
  TORCH_CHECK(page_table.size(1) == required_table_pages,
              "page_table width must exactly match logical_cap");
  return {
      static_cast<int>(inline_val.size(0)),
      static_cast<int>(overflow_val.size(0)),
      static_cast<int>(page_table.size(1)),
      static_cast<int>(logical_cap64),
  };
}

static bool tensor_storage_ranges_overlap(const torch::Tensor &a,
                                          const torch::Tensor &b) {
  if (a.numel() == 0 || b.numel() == 0)
    return false;
  const uintptr_t a_begin = reinterpret_cast<uintptr_t>(a.data_ptr());
  const uintptr_t b_begin = reinterpret_cast<uintptr_t>(b.data_ptr());
  const uintptr_t a_end =
      a_begin + static_cast<uintptr_t>(a.numel()) * a.element_size();
  const uintptr_t b_end =
      b_begin + static_cast<uintptr_t>(b.numel()) * b.element_size();
  return a_begin < b_end && b_begin < a_end;
}

static void *driver_handle() {
  static void *h = nullptr;
  if (!h) {
    h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    TORCH_CHECK(h, "failed to load libcuda.so.1");
  }
  return h;
}

static CUresult enc_tiled(CUtensorMap *tm, CUtensorMapDataType dt,
                          cuuint32_t rank, void *addr, const cuuint64_t *dims,
                          const cuuint64_t *strides, const cuuint32_t *box,
                          const cuuint32_t *estrides, CUtensorMapInterleave il,
                          CUtensorMapSwizzle sw, CUtensorMapL2promotion l2,
                          CUtensorMapFloatOOBfill oob) {
  using FT = CUresult (*)(CUtensorMap *, CUtensorMapDataType, cuuint32_t,
                          void *, const cuuint64_t *, const cuuint64_t *,
                          const cuuint32_t *, const cuuint32_t *,
                          CUtensorMapInterleave, CUtensorMapSwizzle,
                          CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  static FT f = nullptr;
  if (!f) {
    f = reinterpret_cast<FT>(dlsym(driver_handle(), "cuTensorMapEncodeTiled"));
    TORCH_CHECK(f, "failed to load cuTensorMapEncodeTiled");
  }
  return f(tm, dt, rank, addr, dims, strides, box, estrides, il, sw, l2, oob);
}

static CUtensorMap make_2d(void *ptr, CUtensorMapDataType dt, int elem_size,
                           int gmem_inner, int gmem_outer, int smem_inner,
                           int smem_outer, long gmem_outer_stride,
                           int swizzle_mode) {
  if (swizzle_mode != 0)
    smem_inner = swizzle_mode / elem_size;
  CUtensorMap tm;
  const cuuint64_t gdims[2] = {(cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer};
  const cuuint32_t sdims[2] = {(cuuint32_t)smem_inner, (cuuint32_t)smem_outer};
  const cuuint64_t gstrides[1] = {(cuuint64_t)(gmem_outer_stride * elem_size)};
  const cuuint32_t estrides[2] = {1, 1};
  CUtensorMapSwizzle swizzle = swizzle_mode == 128  ? CU_TENSOR_MAP_SWIZZLE_128B
                               : swizzle_mode == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                               : swizzle_mode == 32
                                   ? CU_TENSOR_MAP_SWIZZLE_32B
                                   : CU_TENSOR_MAP_SWIZZLE_NONE;
  CUresult r = enc_tiled(&tm, dt, 2, ptr, gdims, gstrides, sdims, estrides,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                         CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                         CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return tm;
}

static inline int align_up(int x, int a) { return (x + a - 1) / a * a; }

constexpr int NUM_HEADS = 32;
constexpr int HEAD_DIM = 128;
constexpr int BLOCK_Q = 4; // 128 q*h rows per UMMA tile / 32 heads
constexpr int BLOCK_KV = 256;
constexpr int NUM_Q_STAGES = 1; // one q-block per CTA
constexpr int NUM_KV_STAGES = 4;
constexpr int SPEC_THREADS = 128;
constexpr int MATH_THREADS = 256; // 2 math warpgroups on SM100
constexpr int NUM_SMS = 148;      // B200

constexpr int kSeedThreads = 256;

template <int kRetainedHead, int BT, bool Checked>
__global__ void seed_prep_kernel(
    const float *__restrict__ slog, const int64_t slog_stride, const int NB,
    const int K,
    const float headroom, // extend the bucket scale ABOVE the sample max by
                          // headroom*span (absolute, resolution-preserving
                          // when NB is scaled up with it): drifted scores
                          // land in real buckets instead of clamping to
                          // bucket 0 where refresh can never resolve them
    float *__restrict__ origin, float *__restrict__ inv_delta,
    int32_t *__restrict__ th_bucket, CandidateValue *__restrict__ cand_val,
    int32_t *__restrict__ cand_idx, int32_t *__restrict__ cand_cnt,
    const int cand_cap, const int physical_index_base,
    int32_t *__restrict__ bcount_out, int32_t *__restrict__ seed_error) {
  constexpr int NSUB = 4; // sub-histograms to spread smem atomic conflicts
  static_assert(kRetainedHead == 8192 || kRetainedHead == 12288,
                "production seed supports only the qualified 8K/12K layouts");
  constexpr int kRetainVecs = kRetainedHead / (BT * 4);
  const int row = gridDim.x - 1 - blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const float *srow = slog + (size_t)row * slog_stride;
  extern __shared__ int s_hist[]; // NSUB * NB ints

  // pass 1: min/max of the row's FINITE scores (vectorized). -inf appears
  // when the caller passes clean_logits=True full-row logits (dense-select
  // mode) for the out-of-range causal tail; it must not poison the range.
  __shared__ float s_mx[BT / 32];
  __shared__ float s_mn[BT / 32];
  float mx = -INFINITY, mn = INFINITY;
  bool bad_seed_value = false;
  const auto acc = [&](const float s) {
    if constexpr (Checked) {
      const uint32_t bits = __float_as_uint(s), mag = bits & 0x7fffffffu;
      bad_seed_value |= !isfinite(s) || bits == 0x80000000u ||
                        (mag != 0 && mag < 0x00800000u);
    }
    if (isfinite(s)) {
      mx = fmaxf(mx, s);
      mn = fminf(mn, s);
    }
  };
  // The production 8K and 12K specializations retain respectively eight
  // and twelve float4 values per thread. Keep them live across the CTA
  // reduction so histogram construction and emission never reread the
  // materialized prefix logits. Missing tail lanes carry -inf and are
  // ignored by the generic <=8K compatibility path.
  static_assert(BT == 256 || BT == 384 || BT == 512,
                "retained HOT seed requires a qualified CTA size");
  static_assert(BT % (NSUB * 32) == 0,
                "each seed sub-histogram must own whole warps");
  float4 retained[kRetainVecs];
#pragma unroll
  for (int it = 0; it < kRetainVecs; ++it) {
    const int j = tid * 4 + it * BT * 4;
    const float4 s4 = *reinterpret_cast<const float4 *>(srow + j);
    retained[it] = s4;
    acc(s4.x);
    acc(s4.y);
    acc(s4.z);
    acc(s4.w);
  }
  if constexpr (Checked) {
    // The values are already retained for min/max, histogram and emission.
    // Keep failure outside allocator scratch, which scan resets independently.
    const bool warp_bad = __any_sync(0xffffffffu, bad_seed_value);
    if (warp_bad && lane == 0) atomicOr(seed_error + row, 1);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, off));
  }
  if (lane == 0) {
    s_mx[tid >> 5] = mx;
    s_mn[tid >> 5] = mn;
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int wgi = 1; wgi < BT / 32; ++wgi) {
      s_mx[0] = fmaxf(s_mx[0], s_mx[wgi]);
      s_mn[0] = fminf(s_mn[0], s_mn[wgi]);
    }
  }
  __syncthreads();
  float o = -s_mx[0];        // min over x = -score
  const float hi = -s_mn[0]; // max over x
  const float span = fmaxf(hi - o, 1e-20f);
  o -= headroom * span; // forward (above-max) drift headroom
  float inv = (NB - 1) / (span * (1.0f + headroom));
  const float vth = -o * inv;

  // pass 2: histogram in [o, inv] bucket space, NSUB sub-histograms to cut
  // smem atomic conflicts, vectorized loads.
  for (int b = tid; b < NSUB * NB; b += BT)
    s_hist[b] = 0;
  __syncthreads();
  int *my_hist = s_hist + (tid / (BT / NSUB)) * NB;
  const auto bucket_of = [&](const float s) -> int {
    // Use the byte-for-byte arithmetic contract consumed by both the
    // seed emitter and the main scan.  Computing (-s - o) * inv as two
    // rounded operations can put a boundary value one bucket below its
    // FMA result: the histogram would then certify K records while the
    // emitter rejects one of them, producing a silent underfill.
    const float bq = fmaf(-s, inv, vth);
    if (bq < 0.0f) return 0;
    if (bq < static_cast<float>(NB-1)) return static_cast<int>(bq);
    return NB-1; // defined under/overflow bucket, never cast huge/NaN to int
  };
#pragma unroll
  for (int it = 0; it < kRetainVecs; ++it) {
    const float4 s4 = retained[it];
    if (isfinite(s4.x))
      atomicAdd(&my_hist[bucket_of(s4.x)], 1);
    if (isfinite(s4.y))
      atomicAdd(&my_hist[bucket_of(s4.y)], 1);
    if (isfinite(s4.z))
      atomicAdd(&my_hist[bucket_of(s4.z)], 1);
    if (isfinite(s4.w))
      atomicAdd(&my_hist[bucket_of(s4.w)], 1);
  }
  __syncthreads();
  // merge sub-histograms into s_hist[0..NB)
  for (int b = tid; b < NB; b += BT) {
    int c = s_hist[b];
#pragma unroll
    for (int g = 1; g < NSUB; ++g)
      c += s_hist[g * NB + b];
    s_hist[b] = c;
  }
  __syncthreads();
  if (bcount_out != nullptr) {
    // Full-row overwrite of the sample histogram. The ring daemon warm-
    // starts its refresh base from these counts: they are genuine row
    // records in the final (origin, inv) bucket space, so adding them to
    // the daemon's subset cum can only tighten the published edge safely
    // — provided the main scan starts after the sampled prefix (the
    // exact-once contract), or the same records would count twice.
    for (int b = tid; b < NB; b += BT)
      bcount_out[(size_t)row * NB + b] = s_hist[b];
  }
  // Coarse K-th estimate on the single (o, inv) scale built above. There
  // is deliberately NO scale rebuild: th_bucket, origin/inv, the emitted
  // candidates, and the exported bcount histogram must all share one
  // bucket space — the ring warm-start base is only sound under that
  // identity. Headroom above the sample max keeps drifted scores out of
  // bucket 0 where refresh could never resolve them.
  // The production U16 contract uses emit_limit==0 and a single KV split.
  // Its scan covers the complete KV range and initializes the CTA-local
  // histogram itself, so writing Q*NB zeros to global memory is dead work.
  // Find the first histogram prefix that reaches K in parallel.  The old
  // single-thread walk serialized 256 dependent shared-memory loads while
  // the other 1023 threads waited.  NB <= BT gives every bin one owner;
  // the half-open prefix ranges are disjoint, so exactly one thread writes
  // the same threshold as the serial "first cumulative sum >= K" rule.
  __shared__ int s_th;
  __shared__ int s_wsum[BT / 32];
  if (tid == 0)
    s_th = NB - 1;
  const int h = (tid < NB) ? s_hist[tid] : 0;
  int x = h;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, off);
    if ((tid & 31) >= off)
      x += y;
  }
  if ((tid & 31) == 31)
    s_wsum[tid >> 5] = x;
  __syncthreads();
  int base = 0;
#pragma unroll
  for (int w = 0; w < BT / 32; ++w)
    if (w < (tid >> 5))
      base += s_wsum[w];
  const int incl = base + x;
  const int excl = incl - h;
  if (tid < NB && excl < K && K <= incl)
    s_th = tid;
  __syncthreads();
  if (tid == 0) {
    th_bucket[row] = s_th;
    origin[row] = o;
    inv_delta[row] = inv;
  }
  __syncthreads();
  // The HOT scores retained above are the physical prefix
  // [physical_index_base, physical_index_base + kRetainedHead). Emit their
  // passing records now, then let the main producer start after the prefix.
  int emitted_before = 0;
  const float gate_edge = static_cast<float>(s_th + 1);
  const uint64_t row_base = static_cast<uint64_t>(row) * cand_cap;
#pragma unroll
  for (int it = 0; it < kRetainVecs; ++it) {
    const int j0 = tid * 4 + it * BT * 4;
    const float4 s4 = retained[it];
    const float score[4] = {s4.x, s4.y, s4.z, s4.w};
    float bq[4];
    bool pass[4];
    int local_count = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      bq[k] = fmaf(-score[k], inv, vth);
      pass[k] = isfinite(score[k]) &&
                __float_as_int(bq[k]) < __float_as_int(gate_edge);
      local_count += pass[k] ? 1 : 0;
    }

    int warp_inclusive = local_count;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, warp_inclusive, off);
      if (lane >= off)
        warp_inclusive += other;
    }
    if (lane == 31)
      s_wsum[tid >> 5] = warp_inclusive;
    __syncthreads();

    int warp_before = 0;
#pragma unroll
    for (int w = 0; w < BT / 32; ++w) {
      if (w < (tid >> 5))
        warp_before += s_wsum[w];
    }
    const int thread_base =
        emitted_before + warp_before + warp_inclusive - local_count;
    int local_rank = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      if (pass[k]) {
        const int out = thread_base + local_rank++;
        if (out < cand_cap) {
          const uint32_t physical_idx =
              static_cast<uint32_t>(physical_index_base + j0 + k);
          dsa_litetopk::store_candidate(cand_val + row_base + out,
                                        cand_idx + row_base + out, bq[k],
                                        physical_idx);
        }
      }
    }

    int block_total = 0;
#pragma unroll
    for (int w = 0; w < BT / 32; ++w)
      block_total += s_wsum[w];
    emitted_before += block_total;
    // Do not let an early warp overwrite s_wsum for the next retained
    // group while a slower warp still consumes this group's totals.
    __syncthreads();
  }
  if (tid == 0) {
    cand_cnt[row] = emitted_before;
    if constexpr (Checked) {
      const float offset = __fmul_rn(-o, inv);
      const bool bad = !(isfinite(inv) && inv > 0 && isfinite(o) &&
                         isfinite(offset)) || s_th < 0 || s_th >= NB ||
                       emitted_before < 2048 || emitted_before > 8192;
      if (bad) atomicOr(seed_error + row, 1);
    }
  }
}

static int compute_smem_bytes() {
  const int esz_fp8 = 1, esz_f32 = 4;
  const int smem_q = BLOCK_Q * NUM_HEADS * HEAD_DIM * esz_fp8;
  const int smem_w = BLOCK_Q * NUM_HEADS * esz_f32;
  const int smem_kv = BLOCK_KV * HEAD_DIM * esz_fp8;
  const int smem_ks = align_up(BLOCK_KV * esz_f32, 512);
  const int num_barriers = NUM_Q_STAGES * 2 + NUM_KV_STAGES * 2 +
                           (MATH_THREADS / 128) * dsa_litetopk::kUmmaStages * 2;
  const int smem_barriers = num_barriers * 8;
  const int smem_slots =
      4 * (int)sizeof(uint32_t); // tmem ptr + daemon mailboxes
  constexpr int emit_record_bytes = (int)sizeof(uint32_t);
  const int smem_warpq = (MATH_THREADS / 32) * BLOCK_Q *
                         ((int)sizeof(int32_t) + dsa_litetopk::kEmitLaneSlots *
                                                     32 * emit_record_bytes);
  const int smem_hist = BLOCK_Q * 256 * (int)sizeof(int32_t);
  return NUM_Q_STAGES * smem_q + NUM_Q_STAGES * smem_w +
         NUM_KV_STAGES * smem_kv + NUM_KV_STAGES * smem_ks + smem_barriers +
         smem_slots + smem_warpq + smem_hist;
}

static int compute_paged_smem_bytes() {
  constexpr int ring_bytes =
      2 * 8 * BLOCK_Q * static_cast<int>(sizeof(uint32_t)) +
      3 * BLOCK_Q * static_cast<int>(sizeof(int32_t)) + 8 * BLOCK_Q * 32 +
      BLOCK_Q * static_cast<int>(sizeof(float));
  return compute_smem_bytes() + ring_bytes;
}

template <bool Checked>
void launch_seed_prep(const float *slog, int64_t slog_stride, int Q, int head,
                      int NB, int K, float headroom, float *origin,
                      float *inv_delta, int32_t *th_bucket,
                      CandidateValue *cand_val, int32_t *cand_idx,
                      int32_t *cand_cnt, int cand_cap, int physical_index_base,
                      int32_t *bcount, cudaStream_t stream, int32_t *seed_error) {
  const int seed_smem = 4 * NB * static_cast<int>(sizeof(int));
  seed_prep_kernel<8192, kSeedThreads, Checked>
      <<<Q, kSeedThreads, seed_smem, stream>>>(
          slog, slog_stride, NB, K, headroom, origin, inv_delta, th_bucket,
          cand_val, cand_idx, cand_cnt, cand_cap, physical_index_base,
          bcount, seed_error);
}

// Fused seed/prep: sample scores -> (origin, inv_delta, th_bucket, cand_val,
// cand_idx, cand_cnt, bcount), everything the scan needs, in one launch.
template <bool Checked>
void seed_prep_impl(torch::Tensor slog, int64_t num_buckets64,
                         int64_t topk64, int64_t cand_cap64,
                         int64_t emit_limit64, double headroom,
                         int64_t probe_stride_tok64, int64_t hist_stride64,
                         torch::Tensor origin, torch::Tensor inv_delta,
                         torch::Tensor th_bucket, torch::Tensor bcount,
                         torch::Tensor cand_val, torch::Tensor cand_idx,
                         torch::Tensor cand_cnt, torch::Tensor seed_error) {
  TORCH_CHECK(slog.is_cuda() && slog.dim() == 2, "slog must be CUDA [Q, head]");
  TORCH_CHECK(origin.is_cuda() && inv_delta.is_cuda() && th_bucket.is_cuda() &&
                  bcount.is_cuda() && cand_val.is_cuda() &&
                  cand_idx.is_cuda() && cand_cnt.is_cuda(),
              "seed prep outputs must be CUDA tensors");
  TORCH_CHECK(slog.device() == origin.device() &&
                  slog.device() == inv_delta.device() &&
                  slog.device() == th_bucket.device() &&
                  slog.device() == bcount.device() &&
                  slog.device() == cand_val.device() &&
                  slog.device() == cand_idx.device() &&
                  slog.device() == cand_cnt.device(),
              "seed prep tensors must be on one CUDA device");
  TORCH_CHECK(origin.is_contiguous() && inv_delta.is_contiguous() &&
                  th_bucket.is_contiguous() && bcount.is_contiguous() &&
                  cand_val.is_contiguous() && cand_idx.is_contiguous() &&
                  cand_cnt.is_contiguous(),
              "seed prep outputs must be contiguous");
  TORCH_CHECK(slog.scalar_type() == torch::kFloat, "slog must be fp32 scores");
  TORCH_CHECK(slog.stride(1) == 1, "slog rows must be inner-contiguous");
  const int Q = (int)slog.size(0);
  const int head = (int)slog.size(1);
  const int NB = (int)num_buckets64;
  const int K = (int)topk64;
  const int cap = (int)cand_cap64;
  if constexpr (Checked) {
    TORCH_CHECK(head == 8192 && K == 2048 && cap == 8192,
                "certified checked seed requires 8K HOT and top2048");
    TORCH_CHECK(seed_error.is_cuda() && seed_error.device() == slog.device() &&
                seed_error.is_contiguous() && seed_error.scalar_type() == torch::kInt &&
                seed_error.numel() == Q, "checked seed error must be same-device int32[Q]");
  }
  TORCH_CHECK(head >= K && (head == 8192 || head == 12288),
              "production seed prep requires HOT=8192 or HOT=12288");
  TORCH_CHECK(NB >= 3 && NB <= 256, "num_buckets out of range");
  TORCH_CHECK(K >= 1 && cap >= K, "need cap >= topk >= 1");
  TORCH_CHECK(
      origin.scalar_type() == torch::kFloat &&
          inv_delta.scalar_type() == torch::kFloat &&
          th_bucket.scalar_type() == torch::kInt &&
          bcount.scalar_type() == torch::kInt &&
          cand_idx.scalar_type() == torch::kInt &&
          cand_cnt.scalar_type() == torch::kInt,
      "seed prep affine outputs must be fp32 and metadata/indices int32");
  TORCH_CHECK(origin.dim() == 1 && origin.numel() >= Q &&
                  inv_delta.dim() == 1 && inv_delta.numel() >= Q &&
                  th_bucket.dim() == 1 && th_bucket.numel() >= Q &&
                  cand_cnt.dim() == 1 && cand_cnt.numel() >= Q,
              "origin/inv_delta/th_bucket/cand_cnt must cover Q rows");
  TORCH_CHECK(cand_val.dim() == 2 && cand_val.size(0) >= Q &&
                  cand_val.size(1) == cap &&
                  cand_idx.sizes() == cand_val.sizes(),
              "cand_val/cand_idx must be [>=Q,cand_cap]");
  check_candidate_dtype(cand_val);
  TORCH_CHECK(bcount.dim() == 2 && bcount.size(0) >= Q && bcount.size(1) == NB,
              "bcount must be [>=Q,num_buckets]");
  TORCH_CHECK((slog.stride(0) % 4) == 0 &&
                  (reinterpret_cast<uintptr_t>(slog.data_ptr()) % 16) == 0,
              "slog rows must be 16B aligned");
  const c10::cuda::CUDAGuard device_guard(slog.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  TORCH_CHECK(emit_limit64 == head,
              "production seed prep requires exact-once HOT emission");
  TORCH_CHECK(hist_stride64 == 1,
              "production seed prep requires hist_stride=1");
  const int64_t physical_index_base64 = probe_stride_tok64;
  TORCH_CHECK(
      physical_index_base64 >= 0 &&
          physical_index_base64 + head <=
              (int64_t{1} << dsa_litetopk::kCandidateIndexBits),
      "HOT physical index range exceeds the packed 20-bit candidate ABI");
  launch_seed_prep<Checked>(slog.data_ptr<float>(), slog.stride(0), Q, head, NB, K,
                   static_cast<float>(headroom), origin.data_ptr<float>(),
                   inv_delta.data_ptr<float>(), th_bucket.data_ptr<int32_t>(),
                   candidate_data_ptr(cand_val), cand_idx.data_ptr<int32_t>(),
                   cand_cnt.data_ptr<int32_t>(), cap,
                   static_cast<int>(physical_index_base64),
                   bcount.data_ptr<int32_t>(), stream,
                   Checked ? seed_error.data_ptr<int32_t>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void seed_prep_checked(torch::Tensor slog,int64_t num_buckets64,int64_t topk64,
    int64_t cand_cap64,int64_t emit_limit64,double headroom,
    int64_t probe_stride_tok64,int64_t hist_stride64,torch::Tensor origin,
    torch::Tensor inv_delta,torch::Tensor th_bucket,torch::Tensor bcount,
    torch::Tensor cand_val,torch::Tensor cand_idx,torch::Tensor cand_cnt,torch::Tensor seed_error) {
  seed_prep_impl<true>(slog,num_buckets64,topk64,cand_cap64,emit_limit64,headroom,
      probe_stride_tok64,hist_stride64,origin,inv_delta,th_bucket,bcount,
      cand_val,cand_idx,cand_cnt,seed_error);
}

void surrogate_scan(
    torch::Tensor q, torch::Tensor kv, torch::Tensor kv_scales,
    torch::Tensor weights, torch::Tensor cu_start, torch::Tensor cu_end,
    torch::Tensor origin, torch::Tensor inv_delta, torch::Tensor th_bucket,
    torch::Tensor inline_val, torch::Tensor inline_idx,
    torch::Tensor overflow_val, torch::Tensor overflow_idx,
    torch::Tensor page_table, torch::Tensor pool_head,
    torch::Tensor pool_status, torch::Tensor cand_cnt, torch::Tensor bcount,
    int64_t logical_cap64, int64_t num_buckets64, int64_t topk64,
    torch::Tensor epsilon) {
  const PagedCandidateShape shape =
      check_paged_candidate_layout(inline_val, inline_idx, overflow_val,
                                   overflow_idx, page_table, logical_cap64);
  TORCH_CHECK(
      q.is_cuda() && kv.is_cuda() && kv_scales.is_cuda() && weights.is_cuda() &&
          cu_start.is_cuda() && cu_end.is_cuda() && origin.is_cuda() &&
          inv_delta.is_cuda() && th_bucket.is_cuda() && pool_head.is_cuda() &&
          pool_status.is_cuda() && cand_cnt.is_cuda() && bcount.is_cuda(),
      "paged static-HOT scan tensors must be CUDA");
  TORCH_CHECK(
      q.device() == kv.device() && q.device() == kv_scales.device() &&
          q.device() == weights.device() && q.device() == cu_start.device() &&
          q.device() == cu_end.device() && q.device() == origin.device() &&
          q.device() == inv_delta.device() &&
          q.device() == th_bucket.device() &&
          q.device() == inline_val.device() &&
          q.device() == pool_head.device() &&
          q.device() == pool_status.device() &&
          q.device() == cand_cnt.device() && q.device() == bcount.device(),
      "paged static-HOT scan tensors must be on one CUDA device");
  TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() &&
                  kv_scales.is_contiguous() && weights.is_contiguous() &&
                  cu_start.is_contiguous() && cu_end.is_contiguous() &&
                  origin.is_contiguous() && inv_delta.is_contiguous() &&
                  th_bucket.is_contiguous() && pool_head.is_contiguous() &&
                  pool_status.is_contiguous() && cand_cnt.is_contiguous() &&
                  bcount.is_contiguous(),
              "paged static-HOT scan tensors must be contiguous");
  TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn &&
                  kv.scalar_type() == torch::kFloat8_e4m3fn,
              "q/kv must be fp8_e4m3fn");
  TORCH_CHECK(kv_scales.scalar_type() == torch::kFloat &&
                  weights.scalar_type() == torch::kFloat &&
                  origin.scalar_type() == torch::kFloat &&
                  inv_delta.scalar_type() == torch::kFloat,
              "kv_scales/weights/origin/inv_delta must be fp32");
  TORCH_CHECK(cu_start.scalar_type() == torch::kInt &&
                  cu_end.scalar_type() == torch::kInt &&
                  th_bucket.scalar_type() == torch::kInt &&
                  pool_head.scalar_type() == torch::kInt &&
                  pool_status.scalar_type() == torch::kInt &&
                  cand_cnt.scalar_type() == torch::kInt &&
                  bcount.scalar_type() == torch::kInt,
              "paged scan range, allocator, count, and metadata must be int32");
  TORCH_CHECK(q.dim() == 3 && kv.dim() == 2,
              "q/kv must be [Q,32,128] and [S,128]");
  const int seq_len = static_cast<int>(q.size(0));
  const int seq_len_kv = static_cast<int>(kv.size(0));
  TORCH_CHECK(seq_len > 0 && seq_len == shape.rows && seq_len_kv > 0,
              "paged candidate rows must equal Q and Q/S must be nonzero");
  TORCH_CHECK(q.size(1) == NUM_HEADS && q.size(2) == HEAD_DIM &&
                  kv.size(1) == HEAD_DIM,
              "paged static-HOT path requires GLM DSA H=32,D=128");
  TORCH_CHECK(seq_len_kv <= (1 << dsa_litetopk::kCandidateIndexBits),
              "packed candidates support at most 1M KV positions");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == seq_len &&
                  weights.size(1) == NUM_HEADS,
              "weights must be [Q,32]");
  TORCH_CHECK(kv_scales.dim() == 1 && cu_start.dim() == 1 &&
                  cu_start.numel() == seq_len && cu_end.dim() == 1 &&
                  cu_end.numel() == seq_len,
              "kv_scales must be 1-D and cu_start/cu_end must have Q elements");
  TORCH_CHECK(origin.dim() == 1 && origin.numel() == seq_len &&
                  inv_delta.dim() == 1 && inv_delta.numel() == seq_len &&
                  th_bucket.dim() == 1 && th_bucket.numel() == seq_len &&
                  cand_cnt.dim() == 1 && cand_cnt.numel() == seq_len,
              "origin/inv_delta/th_bucket/cand_cnt must have Q elements");
  TORCH_CHECK(bcount.dim() == 2 && bcount.size(0) == seq_len &&
                  bcount.size(1) == num_buckets64,
              "bcount must be [Q,num_buckets]");
  TORCH_CHECK(pool_head.numel() == 1 && pool_status.numel() == 1,
              "pool_head/pool_status must be one-element int32 tensors");
  TORCH_CHECK(!tensor_storage_ranges_overlap(pool_head, pool_status) &&
                  !tensor_storage_ranges_overlap(pool_head, inline_val) &&
                  !tensor_storage_ranges_overlap(pool_head, inline_idx) &&
                  !tensor_storage_ranges_overlap(pool_head, overflow_val) &&
                  !tensor_storage_ranges_overlap(pool_head, overflow_idx) &&
                  !tensor_storage_ranges_overlap(pool_head, page_table) &&
                  !tensor_storage_ranges_overlap(pool_head, cand_cnt) &&
                  !tensor_storage_ranges_overlap(pool_status, inline_val) &&
                  !tensor_storage_ranges_overlap(pool_status, inline_idx) &&
                  !tensor_storage_ranges_overlap(pool_status, overflow_val) &&
                  !tensor_storage_ranges_overlap(pool_status, overflow_idx) &&
                  !tensor_storage_ranges_overlap(pool_status, page_table) &&
                  !tensor_storage_ranges_overlap(pool_status, cand_cnt),
              "pool_head/pool_status must own disjoint allocator storage");
  TORCH_CHECK(shape.logical_cap >= kPagedCandidateInline &&
                  shape.overflow_pages > 0 && shape.table_pages_per_row > 0,
              "paged scan requires logical_cap>8192 and a nonempty overflow "
              "pool/table");
  TORCH_CHECK(num_buckets64 >= 3 && num_buckets64 <= 256,
              "paged static-HOT path requires 3 <= num_buckets <= 256");
  TORCH_CHECK(topk64 >= 1 && topk64 <= logical_cap64,
              "topk must be in [1,logical_cap]");

  TORCH_CHECK(epsilon.is_cuda() && epsilon.device() == q.device() &&
                  epsilon.scalar_type() == torch::kFloat &&
                  epsilon.dim() == 1 && epsilon.is_contiguous() &&
                  epsilon.numel() >= seq_len,
              "surrogate epsilon must be contiguous CUDA FP32 [>=Q]");
  const int num_buckets = static_cast<int>(num_buckets64);
  const int topk = static_cast<int>(topk64);
  const int esz_fp8 = 1;
  const int esz_f32 = 4;
  const int ks_aligned = align_up(seq_len_kv, 16 / esz_f32);
  TORCH_CHECK(kv_scales.numel() >= ks_aligned,
              "kv_scales storage is shorter than the aligned KV length");

  c10::cuda::CUDAGuard device_guard(q.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(
      page_table.data_ptr<int32_t>(), 0xff,
      static_cast<size_t>(page_table.numel()) * sizeof(int32_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(pool_head.data_ptr<int32_t>(), 0,
                                 sizeof(int32_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(pool_status.data_ptr<int32_t>(), 0,
                                 sizeof(int32_t), stream));

  auto tm_q = make_2d(q.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8,
                      HEAD_DIM, seq_len * NUM_HEADS, HEAD_DIM,
                      BLOCK_Q * NUM_HEADS, HEAD_DIM, HEAD_DIM);
  auto tm_kv =
      make_2d(kv.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, esz_fp8, HEAD_DIM,
              seq_len_kv, HEAD_DIM, BLOCK_KV, HEAD_DIM, HEAD_DIM);
  auto tm_ks = make_2d(kv_scales.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                       esz_f32, ks_aligned, 1, BLOCK_KV, 1, 0, 0);
  auto tm_w =
      make_2d(weights.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, esz_f32,
              NUM_HEADS, seq_len, NUM_HEADS, BLOCK_Q, NUM_HEADS, 0);

  const int smem = compute_paged_smem_bytes();
  const auto launch = [&](auto coalesced) {
    constexpr bool Coalesced = decltype(coalesced)::value;
  auto kernel = &dsa_litetopk::sm100_dsa_litetopk<
      NUM_HEADS, HEAD_DIM, BLOCK_Q, BLOCK_KV, NUM_Q_STAGES, NUM_KV_STAGES,
      NUM_SMS, SPEC_THREADS, MATH_THREADS, MATH_THREADS / 128, false, true,
      true, false, true, false, true,
      dsa_litetopk::kUmmaStages, dsa_litetopk::kEmitLaneSlots,
      dsa_litetopk::kMathRegisters, 1, dsa_litetopk::kTmemRows,
      false, false, Coalesced>;
  C10_CUDA_CHECK(
      cudaFuncSetAttribute(reinterpret_cast<void *>(kernel),
                           cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
  dim3 grid(static_cast<unsigned>(num_q_blocks), 1u, 1u);
  kernel<<<grid, SPEC_THREADS + MATH_THREADS, smem, stream>>>(
      static_cast<uint32_t>(seq_len), static_cast<uint32_t>(seq_len_kv),
      reinterpret_cast<uint32_t *>(cu_start.data_ptr<int>()),
      reinterpret_cast<uint32_t *>(cu_end.data_ptr<int>()),
      origin.data_ptr<float>(), inv_delta.data_ptr<float>(),
      th_bucket.data_ptr<int32_t>(), bcount.data_ptr<int32_t>(),
      static_cast<uint32_t>(num_buckets), static_cast<uint32_t>(topk), 1u, 1u,
      0u, 0ULL, 0u, candidate_data_ptr(inline_val),
      inline_idx.data_ptr<int32_t>(), cand_cnt.data_ptr<int32_t>(),
      static_cast<uint32_t>(shape.logical_cap), tm_q, tm_kv, tm_ks, tm_w, tm_q,
      candidate_data_ptr(overflow_val), overflow_idx.data_ptr<int32_t>(),
      page_table.data_ptr<int32_t>(),
      reinterpret_cast<uint32_t *>(pool_head.data_ptr<int32_t>()),
      pool_status.data_ptr<int32_t>(),
      static_cast<uint32_t>(shape.overflow_pages), nullptr, nullptr, nullptr,
      nullptr, nullptr, 0u, 0u, nullptr, nullptr, nullptr, nullptr, 0u,
      epsilon.data_ptr<float>(), kv_scales.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  // Only the qualified natural Q16K shape changes its drain geometry.
  if (seq_len == 16384) launch(std::true_type{});
  else launch(std::false_type{});
}

template <bool EmitOutput, bool ContainedBand = false>
void h2048_paged_impl(
    torch::Tensor inline_val, torch::Tensor inline_idx,
    torch::Tensor overflow_val, torch::Tensor overflow_idx,
    torch::Tensor page_table, torch::Tensor cand_cnt, torch::Tensor pool_status,
    torch::Tensor out_idx, torch::Tensor status,
    torch::Tensor diagnostic_scratch, int64_t logical_cap64,
    int64_t index_limit64, const std::vector<torch::Tensor>& band_tensors = {}) {
  constexpr int kDiagnosticIntsPerRow = 5;
  static_assert(dsa_litetopk::kPagedCandidateInlineRecords ==
                static_cast<uint32_t>(kPagedCandidateInline));
  static_assert(dsa_litetopk::kPagedCandidatePageRecords ==
                static_cast<uint32_t>(kPagedCandidatePage));
  static_assert(dsa_litetopk::kCandidateIndexBits == 20,
                "paged h2048 requires the production 20-bit physical ID ABI");
  static_assert(sizeof(CandidateValue) == sizeof(uint8_t));

  const PagedCandidateShape shape =
      check_paged_candidate_layout(inline_val, inline_idx, overflow_val,
                                   overflow_idx, page_table, logical_cap64);
  TORCH_CHECK(cand_cnt.is_cuda() && pool_status.is_cuda() &&
                  out_idx.is_cuda() && status.is_cuda() &&
                  diagnostic_scratch.is_cuda(),
              "paged h2048 selector tensors must be CUDA");
  TORCH_CHECK(inline_val.device() == cand_cnt.device() &&
                  inline_val.device() == pool_status.device() &&
                  inline_val.device() == out_idx.device() &&
                  inline_val.device() == status.device() &&
                  inline_val.device() == diagnostic_scratch.device(),
              "paged h2048 selector tensors must be on one CUDA device");
  TORCH_CHECK(cand_cnt.is_contiguous() && pool_status.is_contiguous() &&
                  out_idx.is_contiguous() && status.is_contiguous() &&
                  diagnostic_scratch.is_contiguous(),
              "paged h2048 selector tensors must be contiguous");
  TORCH_CHECK(cand_cnt.scalar_type() == torch::kInt &&
                  pool_status.scalar_type() == torch::kInt &&
                  out_idx.scalar_type() == torch::kInt &&
                  status.scalar_type() == torch::kInt &&
                  diagnostic_scratch.scalar_type() == torch::kInt,
              "paged h2048 metadata and output must be int32");
  TORCH_CHECK(logical_cap64 >= kPagedCandidateInline &&
                  logical_cap64 <= h2048_safe_topk::kMaxCap,
              "paged h2048 logical_cap must be in [8192,1M]");
  TORCH_CHECK(cand_cnt.dim() == 1 && cand_cnt.numel() == shape.rows &&
                  status.dim() == 1 && status.numel() == shape.rows,
              "paged h2048 cand_cnt/status must have Q elements");
  TORCH_CHECK(pool_status.numel() == 1,
              "paged h2048 pool_status must contain one int32 value");
  TORCH_CHECK(out_idx.dim() == 2 && out_idx.size(0) == shape.rows &&
                  out_idx.size(1) >= 1 &&
                  out_idx.size(1) <= h2048_safe_topk::overflow::kTopK &&
                  out_idx.size(1) <= kPagedCandidateInline,
              "paged h2048 output must be [Q,topk<=2048]");
  TORCH_CHECK(
      diagnostic_scratch.numel() >=
          static_cast<int64_t>(shape.rows) * kDiagnosticIntsPerRow,
      "paged h2048 diagnostic scratch must contain at least Q*5 int32 values");
  TORCH_CHECK(index_limit64 > 0 &&
                  index_limit64 <=
                      (int64_t{1} << dsa_litetopk::kCandidateIndexBits),
              "paged h2048 index_limit must be in [1,1M]");
  TORCH_CHECK(
      !tensor_storage_ranges_overlap(out_idx, inline_val) &&
          !tensor_storage_ranges_overlap(out_idx, inline_idx) &&
          !tensor_storage_ranges_overlap(out_idx, overflow_val) &&
          !tensor_storage_ranges_overlap(out_idx, overflow_idx) &&
          !tensor_storage_ranges_overlap(out_idx, page_table) &&
          !tensor_storage_ranges_overlap(out_idx, cand_cnt) &&
          !tensor_storage_ranges_overlap(out_idx, pool_status) &&
          !tensor_storage_ranges_overlap(out_idx, status) &&
          !tensor_storage_ranges_overlap(out_idx, diagnostic_scratch) &&
          !tensor_storage_ranges_overlap(pool_status, status),
      "paged h2048 output/status storage must not alias selector inputs");

  h2048_safe_topk::CoarseBandOutputs band{};
  if constexpr(ContainedBand) {
    TORCH_CHECK(band_tensors.size()==10 && out_idx.size(1)==2048,"coarse band requires ten workspace fields and top2048");
    for(const auto&t:band_tensors)TORCH_CHECK(t.is_cuda()&&t.device()==inline_val.device()&&t.is_contiguous(),"coarse band device/layout");
    const auto&x=band_tensors;int q=shape.rows;
    for(int j:{0,1,3,4,5,7,8,9})TORCH_CHECK(x[j].scalar_type()==torch::kInt,"coarse band integer dtype");
    TORCH_CHECK(x[2].scalar_type()==torch::kFloat&&x[6].scalar_type()==torch::kFloat,"coarse band float dtype");
    for(int j:{0,1,2,3,5,8,9})TORCH_CHECK(x[j].numel()==q,"coarse band row shape");
    TORCH_CHECK(x[4].dim()==2&&x[4].size(0)==q&&x[4].size(1)==4096&&
      x[6].dim()==2&&x[6].size(0)==q&&x[6].size(1)>=6144&&x[7].sizes()==x[6].sizes(),"coarse band capacity");
    band={x[0].data_ptr<int>(),x[1].data_ptr<int>(),pool_status.data_ptr<int>(),
      x[2].data_ptr<float>(),x[3].data_ptr<int>(),x[4].data_ptr<int>(),x[5].data_ptr<int>(),
      x[7].data_ptr<int>(),x[8].data_ptr<int>(),x[9].data_ptr<int>(),x[6].data_ptr<float>(),
      int(x[4].size(1)),int(x[6].size(1))};
  }
  const int topk = static_cast<int>(out_idx.size(1));
  const int index_limit = static_cast<int>(index_limit64);
  c10::cuda::CUDAGuard device_guard(inline_val.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // raw<=8192 follows the frozen h2048 fast path byte-for-byte.  The extra
  // template policy only changes raw>inline from a fatal slab overflow into
  // the one recoverable bit consumed by the page-aware kernel below.
  h2048_safe_topk::coarse_tiering_topk_kernel<EmitOutput,ContainedBand><<<shape.rows, 256, 0, stream>>>(
      reinterpret_cast<const uint8_t *>(inline_val.data_ptr<uint8_t>()),
      inline_idx.data_ptr<int32_t>(), cand_cnt.data_ptr<int32_t>(),
      out_idx.data_ptr<int32_t>(), status.data_ptr<int32_t>(),
      diagnostic_scratch.data_ptr<int32_t>(), shape.rows, kPagedCandidateInline,
      topk, index_limit, band);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  h2048_safe_topk::overflow::overflow_exact_topk_paged_kernel<EmitOutput><<<
      shape.rows, h2048_safe_topk::overflow::kThreads, 0, stream>>>(
      reinterpret_cast<const uint8_t *>(inline_val.data_ptr<uint8_t>()),
      inline_idx.data_ptr<int32_t>(),
      reinterpret_cast<const uint8_t *>(overflow_val.data_ptr<uint8_t>()),
      overflow_idx.data_ptr<int32_t>(), page_table.data_ptr<int32_t>(),
      cand_cnt.data_ptr<int32_t>(), pool_status.data_ptr<int32_t>(),
      out_idx.data_ptr<int32_t>(), status.data_ptr<int32_t>(),
      diagnostic_scratch.data_ptr<int32_t>(), shape.rows, kPagedCandidateInline,
      kPagedCandidatePage, shape.overflow_pages, shape.table_pages_per_row,
      shape.logical_cap, index_limit, topk);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void h2048_coarse_band_paged_(torch::Tensor inline_val,torch::Tensor inline_idx,
    torch::Tensor overflow_val,torch::Tensor overflow_idx,torch::Tensor page_table,
    torch::Tensor cand_cnt,torch::Tensor pool_status,torch::Tensor out_idx,
    torch::Tensor status,torch::Tensor diagnostic_scratch,int64_t logical_cap64,
    int64_t index_limit64,const std::vector<torch::Tensor>& band) {
  h2048_paged_impl<false,true>(inline_val,inline_idx,overflow_val,overflow_idx,page_table,cand_cnt,
      pool_status,out_idx,status,diagnostic_scratch,logical_cap64,index_limit64,band);
}


} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("seed_prep_checked", &seed_prep_checked);
  m.def("surrogate_scan", &surrogate_scan);
  m.def("h2048_coarse_band_paged_", &h2048_coarse_band_paged_);
  m.def("surrogate_dynamic_smem_bytes", &compute_paged_smem_bytes);
  m.def("requires_full_surrogate_metadata", []() { return true; });
  m.def("candidate_record_bytes", []() { return 5; });
}
