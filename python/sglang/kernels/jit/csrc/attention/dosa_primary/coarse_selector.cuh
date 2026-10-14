#pragma once
#include "../../../../../srt/layers/attention/dsa/dosa_runtime/native/common/record5_ops.cuh"
// Qualified record5 coarse/halo selection and exact paged-overflow fallback.
namespace h2048_safe_topk {

constexpr int kBins = 256;
constexpr int kMaxCap = 1 << 20;
constexpr uint32_t kPhysicalMask = (1u << 20) - 1u;
constexpr uint32_t kScoreMask = 0xffu;
constexpr uint32_t kReservedPackedMask = 0u;

enum StatusBits : uint32_t {
  kBadCount = 1u << 0,
  kNonFinite = 1u << 1,
  kBadPhysical = 1u << 2,
  kBadMapped = 1u << 3,
  kBadCertificate = 1u << 4,
  // Matches qrita_overflow_fallback_safety.cu's recoverable contract.
  kBoundaryOverflow = 1u << 5,
};

constexpr uint32_t kNonOverflowStatusMask =
    kBadCount | kNonFinite | kBadPhysical | kBadMapped | kBadCertificate;
static_assert(kBoundaryOverflow == 32u);
static_assert((kNonOverflowStatusMask & kBoundaryOverflow) == 0u);

__device__ __forceinline__ uint32_t candidate_score_code(uint8_t value,
                                                         int32_t packed_index) {
  return record5::Policy<record5::Domain::Surrogate24>::code(value, packed_index);
}

__device__ __forceinline__ float decode_score_code(uint32_t code) {
  const uint32_t ordered = code << 8;
  const uint32_t bits =
      (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return __uint_as_float(bits);
}

__device__ __forceinline__ int coarse_bucket_scaled(uint32_t code) {
  constexpr int Scale = 8;
  constexpr int bins = kBins * Scale;
  const float value = decode_score_code(code);
  return value < 0.0f
             ? 0
             : (value >= static_cast<float>(kBins)
                    ? bins - 1
                    : static_cast<int>(value * static_cast<float>(Scale)));
}

__device__ __forceinline__ uint32_t fp24_code(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  return ordered >> 8;
}

template <int Bins>
__device__ __forceinline__ void
find_radix_digit(const uint32_t *__restrict__ hist,
                 uint32_t *__restrict__ desired, uint32_t *__restrict__ rank,
                 uint32_t *__restrict__ selected_count, int shift) {
  const int tid = static_cast<int>(threadIdx.x);
  if (tid >= 32)
    return;
  constexpr unsigned kFull = 0xffffffffu;
  static_assert(Bins == 256 || Bins == 512 || Bins == 1024 || Bins == 2048 ||
                Bins == 4096);
  constexpr int kGroupBins = Bins / 32;
  constexpr int kItemsPerLane = (kGroupBins + 31) / 32;
  const int lane = tid;
  const int first = lane * kGroupBins;
  uint32_t group_count = 0u;
#pragma unroll
  for (int i = 0; i < kGroupBins; ++i)
    group_count += hist[first + i];
  uint32_t inclusive = group_count;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint32_t other = __shfl_up_sync(kFull, inclusive, offset);
    if (lane >= offset)
      inclusive += other;
  }
  const uint32_t target = *rank;
  const unsigned group_mask = __ballot_sync(kFull, inclusive >= target);
  if (target == 0u || group_mask == 0u)
    return;
  const int winning_group = __ffs(group_mask) - 1;
  const uint32_t group_before =
      __shfl_sync(kFull, inclusive - group_count, winning_group);
  // For 2048/4096 bins the winning coarse group contains 64/128 bins.
  // Give each lane a contiguous 2/4-bin segment, scan segment totals, then
  // let the winning lane locate the exact bin locally.  The same code also
  // handles the 8-bin exact-radix histogram used below.
  const int segment_offset = lane * kItemsPerLane;
  const bool segment_valid = segment_offset < kGroupBins;
  const int segment_first = winning_group * kGroupBins + segment_offset;
  uint32_t segment_count = 0u;
#pragma unroll
  for (int i = 0; i < kItemsPerLane; ++i) {
    if (segment_offset + i < kGroupBins) {
      segment_count += hist[segment_first + i];
    }
  }
  uint32_t segment_inclusive = segment_count;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint32_t other = __shfl_up_sync(kFull, segment_inclusive, offset);
    if (lane >= offset)
      segment_inclusive += other;
  }
  const unsigned segment_mask = __ballot_sync(
      kFull, segment_valid && group_before + segment_inclusive >= target);
  if (segment_mask == 0u)
    return;
  const int winning_lane = __ffs(segment_mask) - 1;
  const uint32_t segment_before =
      __shfl_sync(kFull, segment_inclusive - segment_count, winning_lane);

  uint32_t local_digit = 0u;
  uint32_t local_before = 0u;
  uint32_t local_count = 0u;
  if (lane == winning_lane) {
    const uint32_t local_target = target - group_before - segment_before;
    uint32_t running = 0u;
    bool found = false;
#pragma unroll
    for (int i = 0; i < kItemsPerLane; ++i) {
      const bool valid = segment_offset + i < kGroupBins;
      const uint32_t count = valid ? hist[segment_first + i] : 0u;
      if (!found && valid && running + count >= local_target) {
        local_digit = static_cast<uint32_t>(segment_offset + i);
        local_before = running;
        local_count = count;
        found = true;
      }
      running += count;
    }
  }
  local_digit = __shfl_sync(kFull, local_digit, winning_lane);
  local_before = __shfl_sync(kFull, local_before, winning_lane);
  local_count = __shfl_sync(kFull, local_count, winning_lane);
  if (lane == 0) {
    const uint32_t digit =
        static_cast<uint32_t>(winning_group * kGroupBins) + local_digit;
    *desired |= digit << static_cast<uint32_t>(shift);
    *rank = target - group_before - segment_before - local_before;
    *selected_count = local_count;
  }
}

struct CoarseBandOutputs {
  const int *starts,*ends,*pool_status;
  const float *epsilon;
  int *error,*boundary_ids,*boundary_counts,*temporary_ids,*temporary_counts,*done;
  float *temporary;
  int boundary_stride,temporary_stride;
};
__device__ __forceinline__ float code_upper_endpoint(uint32_t code) {
  const uint32_t ordered=(code<<8)|255u;
  return __uint_as_float((ordered&0x80000000u)?(ordered^0x80000000u):~ordered);
}

template <bool EmitOutput, bool ContainedBand = false>
__global__ __launch_bounds__(256) void coarse_tiering_topk_kernel(
    const uint8_t *__restrict__ values,
    const int32_t *__restrict__ packed_indices,
    const int32_t *__restrict__ counts, int32_t *__restrict__ output,
    int32_t *__restrict__ status, int32_t *__restrict__ diagnostics, int rows,
    int cap, int topk, int sequence_length, CoarseBandOutputs band) {
  static_assert(!EmitOutput && ContainedBand,
                "This module builds only conservative coarse-band selection");
  constexpr int kThreads = 256;
  constexpr int kBoundaryCapacity = 512;
  constexpr int kHistScale = 8;
  constexpr int kCoarseBins = kBins * kHistScale;
  constexpr unsigned kFull = 0xffffffffu;
  const int logical_block = static_cast<int>(blockIdx.x);
  // Real count argmax is near the end; longest rows launch first.
  const int row = rows - 1 - logical_block;
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  if (row < 0 || row >= rows)
    return;

  __shared__ uint32_t hist[kCoarseBins];
  __shared__ uint32_t boundary_code[kBoundaryCapacity];
  // Candidate rows may use the production 196608-record capacity.  A uint16
  // slot silently wrapped beyond the old 65536-record qualification shape.
  __shared__ uint32_t boundary_slot[kBoundaryCapacity];
  __shared__ uint32_t s_status;
  __shared__ uint32_t s_desired;
  __shared__ uint32_t s_rank;
  __shared__ uint32_t s_selected_count;
  __shared__ int s_count;
  __shared__ int s_coarse_bucket;
  __shared__ int s_coarse_upper;
  __shared__ int s_coarse_lt;
  __shared__ int s_coarse_need;
  __shared__ int s_boundary_count;
  __shared__ int s_strict_cursor;
  __shared__ int s_boundary_cursor;
  __shared__ int s_boundary_lt_cursor;
  __shared__ int fused_b,fused_e,contained;
  __shared__ float band_lo,band_hi,band_epsilon;

  if (tid == 0) {
    const int raw_count = counts[row];
    if constexpr(ContainedBand) band.done[row]=0;
    s_status = 0u;
    if (raw_count < topk || raw_count < 0) {
      s_status |= kBadCount;
    } else if (raw_count > cap) {
      // The inline slab is only a storage tier in the paged ABI. Mark the row
      // for the exact page-aware fallback.
      s_status |= kBoundaryOverflow;
    }
    if constexpr(ContainedBand) {
      if(band.starts[row]<0||band.ends[row]<band.starts[row]||band.ends[row]>sequence_length)
        s_status|=kBadPhysical;
    }
    s_count = max(0, min(raw_count, cap));
    s_desired = 0u;
    s_rank = static_cast<uint32_t>(topk);
    s_selected_count = 0u;
    s_coarse_bucket = -1;
    s_coarse_upper = -1;
    s_coarse_lt = 0;
    s_coarse_need = 0;
    s_boundary_count = 0;
    s_strict_cursor = 0;
    s_boundary_cursor = 0;
    s_boundary_lt_cursor = 0;
    status[row] = 0;
  }
  for (int i = tid; i < kCoarseBins; i += kThreads)
    hist[i] = 0u;
  __syncthreads();

  const int count = s_count;
  const int64_t row_base = static_cast<int64_t>(row) * cap;
  const int64_t out_base = static_cast<int64_t>(row) * topk;
  if (s_status != 0u) {
    if (tid == 0)
      status[row] = static_cast<int32_t>(s_status);
    return;
  }

  DOSA_RECORD5_INDEPENDENT_WORDS(kThreads,tid,packed_indices,row_base,count,word,slot,
    const uint8_t value=values[row_base+slot];
    const int32_t packed_index=int32_t(word);
    const uint32_t physical=word&kPhysicalMask;
    if((word&kReservedPackedMask)!=0u) {atomicOr(&s_status,kBadMapped);continue;}
    bool bad_physical=physical>=uint32_t(sequence_length);
    if constexpr(ContainedBand)
      bad_physical|=physical<uint32_t(band.starts[row])||physical>=uint32_t(band.ends[row]);
    if(bad_physical) {atomicOr(&s_status,kBadPhysical);continue;}
    const uint32_t code=candidate_score_code(value,packed_index);
    const float decoded=decode_score_code(code);
    if(!isfinite(decoded)) {atomicOr(&s_status,kNonFinite);continue;}
    atomicAdd(hist+coarse_bucket_scaled(code),1u);
  );
  __syncthreads();
  if (s_status == 0u) {
    find_radix_digit<kCoarseBins>(hist, &s_desired, &s_rank, &s_selected_count,
                                  0);
  }
  __syncthreads();
  if (tid == 0 && s_status == 0u) {
    s_coarse_bucket = static_cast<int>(s_desired);
    s_coarse_need = static_cast<int>(s_rank);
    s_coarse_lt = topk - s_coarse_need;
    s_boundary_count = static_cast<int>(s_selected_count);
    s_coarse_upper = s_coarse_bucket + 1;
    if constexpr (ContainedBand) {
      // Enclose the central kth bucket with one complete neighbor on each
      // side. Rank and strict prefix change by exactly the added left count.
      const int center = s_coarse_bucket;
      const int left_count = center > 0 ? int(hist[center - 1]) : 0;
      const int right_count = center + 1 < kCoarseBins ? int(hist[center + 1]) : 0;
      s_coarse_bucket = max(0, center - 1);
      s_coarse_upper = min(kCoarseBins, center + 2);
      s_coarse_lt -= left_count;
      s_coarse_need += left_count;
      s_boundary_count += left_count + right_count;
    }
    if (s_coarse_bucket < 0 || s_coarse_bucket >= kCoarseBins ||
        s_coarse_lt < 0 || s_coarse_lt >= topk || s_coarse_need <= 0 ||
        s_coarse_need > s_boundary_count) {
      s_status |= kBadCertificate;
    }
    if (s_boundary_count > kBoundaryCapacity) {
      s_status |= kBoundaryOverflow;
    }
  }
  __syncthreads();
  if (s_status != 0u) {
    if (tid == 0) {
      status[row] = static_cast<int32_t>(s_status);
      int32_t *diag = diagnostics + static_cast<int64_t>(row) * 5;
      diag[0] = count;
      diag[1] = s_coarse_bucket;
      diag[2] = s_coarse_lt;
      diag[3] = s_coarse_need;
      diag[4] = s_boundary_count;
    }
    return;
  }

  // Emit physical IDs below the coarse boundary and gather the boundary.
  const int threshold_bucket = s_coarse_bucket;
  const float threshold_edge =
      static_cast<float>(threshold_bucket) / static_cast<float>(kHistScale);
  const float next_threshold_edge =
      static_cast<float>(s_coarse_upper) / static_cast<float>(kHistScale);
  const uint32_t threshold_code = fp24_code(threshold_edge);
  const uint32_t next_threshold_code = fp24_code(next_threshold_edge);
  const int warp = tid >> 5;
  auto collect_record=[&](unsigned word,int slot,bool valid) {
    const int32_t packed_index=int32_t(word);
    const uint32_t code=valid?candidate_score_code(values[row_base+slot],packed_index):0u;
    const bool is_strict =
        valid && threshold_bucket > 0 && code < threshold_code;
    const bool is_boundary =
        valid &&
        (s_coarse_upper == kCoarseBins || code < next_threshold_code) &&
        (threshold_bucket == 0 || code >= threshold_code);
    const unsigned strict_mask=__ballot_sync(kFull,is_strict);
    const unsigned boundary_mask=__ballot_sync(kFull,is_boundary);
    DOSA_RECORD5_RESERVE(kFull,lane,strict_mask,boundary_mask,
        &s_strict_cursor,&s_boundary_cursor,strict_base,boundary_base,lane_before);
    if constexpr (EmitOutput || ContainedBand) if (is_strict) {
      const int pos = strict_base+__popc(strict_mask&lane_before);
      if (pos < topk) {
        output[out_base + pos] = static_cast<int32_t>(
            static_cast<uint32_t>(packed_index) & kPhysicalMask);
      }
    }
    if (is_boundary) {
      const int pos = boundary_base+__popc(boundary_mask&lane_before);
      if (pos < kBoundaryCapacity) {
        boundary_code[pos] = code;
        if constexpr (EmitOutput || ContainedBand)
          boundary_slot[pos] = ContainedBand ? (uint32_t(packed_index)&kPhysicalMask) : uint32_t(slot);
      }
    }
  };
  DOSA_RECORD5_COLLECTIVE_WORDS(kThreads,warp,lane,packed_indices,row_base,count,collect_record);
  __syncthreads();
  if (tid == 0 && (s_strict_cursor != s_coarse_lt ||
                   s_boundary_cursor != s_boundary_count)) {
    s_status |= kBadCertificate;
  }
  __syncthreads();

  if (tid == 0) {
    s_desired = 0u;
    s_rank = static_cast<uint32_t>(s_coarse_need);
    s_selected_count = 0u;
  }
  __syncthreads();
#pragma unroll
  for (int pass = 0; pass < 3; ++pass) {
    for (int i = tid; i < kBins; i += kThreads)
      hist[i] = 0u;
    __syncthreads();
    const uint32_t desired = s_desired;
    for (int j = tid; j < s_boundary_count; j += kThreads) {
      const uint32_t code = boundary_code[j];
      bool keep = true;
      if (pass == 1)
        keep = (code >> 16) == (desired >> 16);
      if (pass == 2)
        keep = (code >> 8) == (desired >> 8);
      if (keep) {
        atomicAdd(hist + ((code >> (16 - 8 * pass)) & 0xffu), 1u);
      }
    }
    __syncthreads();
    find_radix_digit<kBins>(hist, &s_desired, &s_rank, &s_selected_count,
                            16 - pass * 8);
    __syncthreads();
  }
  if (tid == 0 && (s_rank == 0u || s_rank > s_selected_count ||
                   s_rank > static_cast<uint32_t>(s_coarse_need))) {
    s_status |= kBadCertificate;
  }
  if constexpr (!EmitOutput) if (tid == 0) s_boundary_cursor = 0;
  __syncthreads();

  const uint32_t exact_pivot = s_desired;
  const int exact_equal_take = static_cast<int>(s_rank);
  const int boundary_strict = s_coarse_need - exact_equal_take;
  if constexpr (!EmitOutput) {
    // Only the exact pivot is consumed. Recount the actual boundary support
    // instead of quadratic stable equal-ID ranking and global ID writes.
    unsigned nlt=0,neq=0;
    for(int j=tid;j<s_boundary_count;j+=kThreads) {
      uint32_t code=boundary_code[j];nlt+=code<exact_pivot;neq+=code==exact_pivot;
    }
    nlt=__reduce_add_sync(kFull,nlt);neq=__reduce_add_sync(kFull,neq);
    if(lane==0) {
      if(nlt)atomicAdd(&s_boundary_lt_cursor,int(nlt));
      if(neq)atomicAdd(&s_boundary_cursor,int(neq));
    }
  }
  __syncthreads();
  if (tid == 0) {
    if (s_boundary_lt_cursor != boundary_strict) {
      s_status |= kBadCertificate;
    }
    if constexpr (!EmitOutput) {
      if(s_boundary_cursor < exact_equal_take ||
          s_boundary_cursor != int(s_selected_count)) s_status |= kBadCertificate;
    }
    status[row] = static_cast<int32_t>(s_status);
    int32_t *diag = diagnostics + static_cast<int64_t>(row) * 5;
    diag[0] = count;
    diag[1] = threshold_bucket;
    diag[2] = s_coarse_lt;
    diag[3] = s_boundary_count;
    diag[4] = static_cast<int32_t>(exact_pivot); // exact kth CODE, not count
  }
  if constexpr(ContainedBand) {
    __syncthreads();
    if(tid==0) {
      band_epsilon=band.epsilon[row];
      band_lo=__fsub_rd(decode_score_code(exact_pivot),band_epsilon);
      band_hi=__fadd_ru(code_upper_endpoint(exact_pivot),band_epsilon);
      bool lower_ok=threshold_bucket==0 ||
          __fadd_ru(code_upper_endpoint(threshold_code-1),band_epsilon)<band_lo;
      bool upper_ok=s_coarse_upper==kCoarseBins ||
          __fsub_rd(decode_score_code(next_threshold_code),band_epsilon)>band_hi;
      contained=s_status==0 && band.error[row]==0 && *band.pool_status==0 &&
          record5::Policy<record5::Domain::Surrogate24>::radius_valid(band_epsilon)&&isfinite(band_lo)&&isfinite(band_hi)&&
          lower_ok&&upper_ok;
      fused_b=s_coarse_lt;fused_e=0;
    }
    __syncthreads();
    if(contained) {
      for(int j=tid;j<s_boundary_count;j+=kThreads) {
        uint32_t code=boundary_code[j],id=boundary_slot[j];
        float lo=decode_score_code(code),hi=code_upper_endpoint(code);
        if(!isfinite(lo)||!isfinite(hi)||id<uint32_t(band.starts[row])||id>=uint32_t(band.ends[row])) {
          atomicOr(&s_status,kBadCertificate);continue;
        }
        if(__fadd_ru(hi,band_epsilon)<band_lo) {
          int pos=atomicAdd(&fused_b,1);
          if(pos<topk)output[out_base+pos]=int(id);else atomicOr(&s_status,kBadCertificate);
        } else if(__fsub_rd(lo,band_epsilon)<=band_hi) {
          int pos=atomicAdd(&fused_e,1);
          if(pos<band.boundary_stride)band.boundary_ids[int64_t(row)*band.boundary_stride+pos]=int(id);
          else atomicOr(&s_status,kBadCertificate);
        }
      }
      __syncthreads();
      if(tid==0 && (fused_b<0||fused_b>=topk||fused_e<=0||fused_e>band.boundary_stride||
                    fused_b+fused_e<topk||fused_b+fused_e>band.temporary_stride))
        s_status|=kBadCertificate;
      __syncthreads();
      if(s_status==0) {
        if(fused_e>128)for(int j=tid;j<fused_b;j+=kThreads) {
          band.temporary[int64_t(row)*band.temporary_stride+j]=FLT_MAX;
          band.temporary_ids[int64_t(row)*band.temporary_stride+j]=output[out_base+j];
        }
        if(tid==0) {
          band.boundary_counts[row]=fused_e;
          band.temporary_counts[row]=fused_e<=128?0:fused_b;
          int32_t*diag=diagnostics+int64_t(row)*5;
          diag[0]=fused_b;diag[1]=fused_e;diag[2]=count;diag[3]=fused_b+fused_e;
          band.done[row]=1;
        }
      } else if(tid==0) {status[row]=int(s_status);atomicOr(band.error+row,1);}
    }
  }

}

namespace overflow {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kRadixBits = 12;
constexpr int kRadixBins = 1 << kRadixBits;
constexpr int kBinsPerThread = kRadixBins / kThreads;
constexpr int kTopK = 2048;
constexpr uint32_t kPhysicalMask = (1u << 20) - 1u;
constexpr uint32_t kScoreMask = 0xffu;
constexpr uint32_t kReservedPackedMask = 0u;

enum StatusBits : uint32_t {
  kBadCount = 1u << 0,
  kNonFinite = 1u << 1,
  kBadPhysical = 1u << 2,
  kBadMapped = 1u << 3,
  kHistogramFailure = 1u << 4,
  kBoundaryOverflow = 1u << 5,
  kCompactFailure = 1u << 6,
  kPagedPoolFailure = 1u << 7,
  kPagedInvalidPage = 1u << 8,
};

__device__ __forceinline__ uint32_t candidate_score_code(uint8_t value,
                                                         int32_t packed_index) {
  return record5::Policy<record5::Domain::Surrogate24>::code(value, packed_index);
}

__device__ __forceinline__ float decode_candidate_score(uint32_t code) {
  const uint32_t ordered = code << 8;
  const uint32_t bits =
      (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
  return __uint_as_float(bits);
}

__device__ __forceinline__ int block_exclusive_sum(int value,
                                                   int *warp_prefix) {
  const int lane = static_cast<int>(threadIdx.x) & 31;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  int inclusive = value;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset)
      inclusive += other;
  }
  if (lane == 31)
    warp_prefix[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    const int original = lane < kWarps ? warp_prefix[lane] : 0;
    int warp_inclusive = original;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const int other = __shfl_up_sync(0xffffffffu, warp_inclusive, offset);
      if (lane >= offset)
        warp_inclusive += other;
    }
    if (lane < kWarps)
      warp_prefix[lane] = warp_inclusive - original;
  }
  __syncthreads();
  return warp_prefix[warp] + inclusive - value;
}

__device__ __forceinline__ void
select_histogram_bin(const int *histogram, int target, int *warp_prefix,
                     volatile int *selected_bin,
                     volatile int *selected_count_lt) {
  const int begin = static_cast<int>(threadIdx.x) * kBinsPerThread;
  int segment_sum = 0;
#pragma unroll
  for (int i = 0; i < kBinsPerThread; ++i) {
    segment_sum += histogram[begin + i];
  }
  const int segment_lt = block_exclusive_sum(segment_sum, warp_prefix);
  if (target > segment_lt && target <= segment_lt + segment_sum) {
    int local_lt = 0;
#pragma unroll
    for (int i = 0; i < kBinsPerThread; ++i) {
      const int count = histogram[begin + i];
      if (target <= segment_lt + local_lt + count) {
        *selected_bin = begin + i;
        *selected_count_lt = segment_lt + local_lt;
        break;
      }
      local_lt += count;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void
paged_high_histogram_span(const uint8_t *__restrict__ values,
                          const int32_t *__restrict__ packed_indices, int count,
                          int sequence_length, int *__restrict__ histogram,
                          uint32_t *__restrict__ block_status) {
  const int tid = static_cast<int>(threadIdx.x);
  for (int col = tid; col < count; col += kThreads) {
    const int32_t packed = packed_indices[col];
    if ((static_cast<uint32_t>(packed) & kReservedPackedMask) != 0u) {
      atomicOr(block_status, static_cast<uint32_t>(kBadMapped));
      continue;
    }
    const uint32_t physical = static_cast<uint32_t>(packed) & kPhysicalMask;
    const uint32_t code = candidate_score_code(values[col], packed);
    if (!isfinite(decode_candidate_score(code))) {
      atomicOr(block_status, static_cast<uint32_t>(kNonFinite));
    }
    if (physical >= static_cast<uint32_t>(sequence_length)) {
      atomicOr(block_status, static_cast<uint32_t>(kBadPhysical));
    }
    atomicAdd(histogram + (code >> kRadixBits), 1);
  }
}

__device__ __forceinline__ void
paged_low_histogram_span(const uint8_t *__restrict__ values,
                         const int32_t *__restrict__ packed_indices, int count,
                         int high_bin, int *__restrict__ histogram) {
  const int tid = static_cast<int>(threadIdx.x);
  for (int col = tid; col < count; col += kThreads) {
    const int32_t packed = packed_indices[col];
    const uint32_t code = candidate_score_code(values[col], packed);
    if (static_cast<int>(code >> kRadixBits) == high_bin) {
      atomicAdd(histogram + (code & (kRadixBins - 1)), 1);
    }
  }
}

// Independent cold-repair histogram.  The ordinary fallback selects two
// 12-bit digits with the block-prefix helper above.  A compact-certificate
// mismatch retries with three 8-bit digits and a scalar histogram scan, so
// the repair shares neither selector state nor prefix-scan machinery with the
// failed path.  Candidate validation and page validation have already
// completed before repair is entered.
__device__ __forceinline__ void
paged_repair_histogram_span(const uint8_t *__restrict__ values,
                            const int32_t *__restrict__ packed_indices,
                            int count, int pass, uint32_t prefix,
                            int *__restrict__ histogram) {
  const int tid = static_cast<int>(threadIdx.x);
  const int shift = 16 - pass * 8;
  for (int col = tid; col < count; col += kThreads) {
    const int32_t packed = packed_indices[col];
    const uint32_t code = candidate_score_code(values[col], packed);
    const bool matches_prefix = pass == 0 ||
                                (pass == 1 && (code >> 16) == (prefix >> 16)) ||
                                (pass == 2 && (code >> 8) == (prefix >> 8));
    if (matches_prefix) {
      atomicAdd(histogram + ((code >> shift) & 0xffu), 1);
    }
  }
}

// Compact one contiguous logical span while carrying the output cursors into
// the next span.  Inline records are visited first, followed by overflow pages
// in logical page order, so exact-pivot ties retain the contiguous h2048
// fallback's stable logical-ordinal rule even when physical page ids differ.
template <bool EmitOutput>
__device__ __forceinline__ void paged_stable_compact_span(
    const uint8_t *__restrict__ values,
    const int32_t *__restrict__ packed_indices, int count,
    uint32_t threshold_code, int32_t *__restrict__ output, int topk,
    volatile int *warp_lt, volatile int *warp_eq, volatile int *tile_lt,
    volatile int *tile_eq, volatile int *base_lt, volatile int *base_eq) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const unsigned lane_before =
      lane == 0 ? 0u : ((1u << static_cast<uint32_t>(lane)) - 1u);
  if constexpr (!EmitOutput) {
    unsigned nlt=0,neq=0;
    for(int col=tid;col<count;col+=kThreads) {
      uint32_t code=candidate_score_code(values[col],packed_indices[col]);
      nlt+=code<threshold_code;neq+=code==threshold_code;
    }
    nlt=__reduce_add_sync(0xffffffffu,nlt);
    neq=__reduce_add_sync(0xffffffffu,neq);
    if(lane==0) {
      if(nlt)atomicAdd(const_cast<int*>(base_lt),int(nlt));
      if(neq)atomicAdd(const_cast<int*>(base_eq),int(neq));
    }
    __syncthreads();
    return;
  }

}

template <bool EmitOutput>
__device__ __forceinline__ void
paged_fail_row(int32_t *__restrict__ output, int topk,
               int32_t *__restrict__ status, int row, uint32_t block_status,
               int32_t *__restrict__ diagnostics, int raw_count,
               uint32_t producer_status, int bad_page_ordinal,
               int bad_page_id) {
  const int tid = static_cast<int>(threadIdx.x);
  __syncthreads();
  if (tid == 0) {
    status[row] = static_cast<int32_t>(block_status);
    int32_t *const diag = diagnostics + static_cast<int64_t>(row) * 5;
    diag[0] = raw_count;
    diag[1] = static_cast<int32_t>(producer_status);
    diag[2] = bad_page_ordinal;
    diag[3] = bad_page_id;
    diag[4] = 0;
  }
}

// Exact cold selector for the paged candidate ABI.  The ordinary row executes
// the frozen coarse h2048 kernel over the inline tier and never reaches this
// body.  A raw count beyond the inline tier (or a valid coarse-boundary
// overflow) is selected over inline+pages with the same high12/low12 radix
// rule and physical-winner contract as the contiguous fallback.
template <bool EmitOutput>
__global__ __launch_bounds__(kThreads) void overflow_exact_topk_paged_kernel(
    const uint8_t *__restrict__ inline_values,
    const int32_t *__restrict__ inline_indices,
    const uint8_t *__restrict__ overflow_values,
    const int32_t *__restrict__ overflow_indices,
    const int32_t *__restrict__ page_table, const int32_t *__restrict__ counts,
    const int32_t *__restrict__ pool_status, int32_t *__restrict__ output,
    int32_t *__restrict__ status, int32_t *__restrict__ diagnostics, int rows,
    int inline_cap, int overflow_page_size, int overflow_pages,
    int table_pages_per_row, int logical_cap, int sequence_length,
    int topk_arg = kTopK) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= rows)
    return;

  const uint32_t input_status = static_cast<uint32_t>(status[row]);
  const uint32_t producer_status = static_cast<uint32_t>(*pool_status);
  if (producer_status == 0u && (input_status & kBoundaryOverflow) == 0u) {
    return;
  }

  __shared__ int histogram[kRadixBins];
  __shared__ int warp_scratch[kWarps];
  // Snapshot the pass-1 winner separately from the selector output that pass 2
  // resets.  In the old code each warp loaded selected_bin independently, so a
  // late warp could observe the reset -1 and omit its low12 contributions.
  __shared__ volatile int selected_bin;
  __shared__ volatile int selected_count_lt;
  __shared__ volatile int selected_high_bin;
  __shared__ volatile int high_count_lt;
  __shared__ volatile int remaining_rank;
  __shared__ volatile int first_count_lt;
  __shared__ volatile uint32_t threshold_code;
  // These cursors are written by lane 0 and consumed by every warp through an
  // inlined helper.  Keep the objects themselves volatile as well as the
  // helper aliases: otherwise NVCC may reuse a non-volatile direct load in the
  // final certificate even though the helper updated the shared object.
  __shared__ volatile int warp_lt[kWarps];
  __shared__ volatile int warp_eq[kWarps];
  __shared__ volatile int tile_lt;
  __shared__ volatile int tile_eq;
  __shared__ volatile int base_lt;
  __shared__ volatile int base_eq;
  __shared__ volatile uint32_t repair_prefix;
  __shared__ volatile int repair_rank;
  __shared__ volatile int repair_failed;
  __shared__ int page_id;
  __shared__ int bad_page_ordinal;
  __shared__ int bad_page_id;
  __shared__ uint32_t block_status;

  const int tid = static_cast<int>(threadIdx.x);
  const int count = counts[row];
  const int64_t inline_row = static_cast<int64_t>(row) * inline_cap;
  const int64_t output_row = static_cast<int64_t>(row) * topk_arg;
  const int inline_count = max(0, min(count, inline_cap));
  const int overflow_count = max(count - inline_cap, 0);
  const int num_pages =
      overflow_count > 0
          ? (overflow_count + overflow_page_size - 1) / overflow_page_size
          : 0;

  if (tid == 0) {
    block_status = input_status & ~kBoundaryOverflow;
    if (producer_status != 0u) {
      // Preserve the producer's raw status bits in the upper half while
      // reserving a stable selector-level bit for fail-closed dispatch.
      block_status |= kPagedPoolFailure | ((producer_status & 0x7fffu) << 16);
    }
    if (count < topk_arg || count < 0 || count > logical_cap ||
        num_pages > table_pages_per_row) {
      block_status |= kBadCount;
    }
    selected_bin = -1;
    selected_count_lt = -1;
    selected_high_bin = -1;
    high_count_lt = -1;
    remaining_rank = -1;
    first_count_lt = -1;
    threshold_code = 0u;
    bad_page_ordinal = -1;
    bad_page_id = -1;
  }
  __syncthreads();

  if (block_status != 0u) {
    paged_fail_row<EmitOutput>(output + output_row, topk_arg, status, row, block_status,
                   diagnostics, count, producer_status, bad_page_ordinal,
                   bad_page_id);
    return;
  }

  for (int bin = tid; bin < kRadixBins; bin += kThreads) {
    histogram[bin] = 0;
  }
  __syncthreads();

  // Pass 1: high 12 bits, including independent record validation.
  paged_high_histogram_span(inline_values + inline_row,
                            inline_indices + inline_row, inline_count,
                            sequence_length, histogram, &block_status);
  __syncthreads();
  for (int logical_page = 0; logical_page < num_pages; ++logical_page) {
    if (tid == 0) {
      page_id = page_table[static_cast<int64_t>(row) * table_pages_per_row +
                           logical_page];
      if (page_id < 0 || page_id >= overflow_pages) {
        block_status |= kPagedInvalidPage;
        bad_page_ordinal = logical_page;
        bad_page_id = page_id;
      }
    }
    __syncthreads();
    if (block_status != 0u)
      break;
    const int page_count = min(
        overflow_count - logical_page * overflow_page_size, overflow_page_size);
    const int64_t page_base =
        static_cast<int64_t>(page_id) * overflow_page_size;
    paged_high_histogram_span(overflow_values + page_base,
                              overflow_indices + page_base, page_count,
                              sequence_length, histogram, &block_status);
    __syncthreads();
  }

  if (block_status != 0u) {
    paged_fail_row<EmitOutput>(output + output_row, topk_arg, status, row, block_status,
                   diagnostics, count, producer_status, bad_page_ordinal,
                   bad_page_id);
    return;
  }

  select_histogram_bin(histogram, topk_arg, warp_scratch, &selected_bin,
                       &selected_count_lt);
  if (tid == 0) {
    if (selected_bin < 0 || selected_count_lt < 0 ||
        selected_count_lt >= topk_arg) {
      block_status |= kHistogramFailure;
    } else {
      selected_high_bin = selected_bin;
      high_count_lt = selected_count_lt;
      remaining_rank = topk_arg - selected_count_lt;
    }
  }
  __syncthreads();
  if (block_status != 0u) {
    paged_fail_row<EmitOutput>(output + output_row, topk_arg, status, row, block_status,
                   diagnostics, count, producer_status, bad_page_ordinal,
                   bad_page_id);
    return;
  }
  for (int bin = tid; bin < kRadixBins; bin += kThreads) {
    histogram[bin] = 0;
  }
  if (tid == 0) {
    selected_bin = -1;
    selected_count_lt = -1;
  }
  __syncthreads();
  // selected_bin is reset by warp 0 above.  All warps consume the immutable
  // pass-1 snapshot only after that reset is block-visible; otherwise a late
  // warp can observe -1 and silently omit its low12 histogram contribution.
  const int high_bin = selected_high_bin;
  const int low_target = remaining_rank;

  // Pass 2: low 12 bits inside the selected high bucket.
  paged_low_histogram_span(inline_values + inline_row,
                           inline_indices + inline_row, inline_count, high_bin,
                           histogram);
  __syncthreads();
  for (int logical_page = 0; logical_page < num_pages; ++logical_page) {
    if (tid == 0) {
      page_id = page_table[static_cast<int64_t>(row) * table_pages_per_row +
                           logical_page];
      if (page_id < 0 || page_id >= overflow_pages) {
        block_status |= kPagedInvalidPage;
        bad_page_ordinal = logical_page;
        bad_page_id = page_id;
      }
    }
    __syncthreads();
    if (block_status != 0u)
      break;
    const int page_count = min(
        overflow_count - logical_page * overflow_page_size, overflow_page_size);
    const int64_t page_base =
        static_cast<int64_t>(page_id) * overflow_page_size;
    paged_low_histogram_span(overflow_values + page_base,
                             overflow_indices + page_base, page_count, high_bin,
                             histogram);
    __syncthreads();
  }
  if (block_status != 0u) {
    paged_fail_row<EmitOutput>(output + output_row, topk_arg, status, row, block_status,
                   diagnostics, count, producer_status, bad_page_ordinal,
                   bad_page_id);
    return;
  }

  select_histogram_bin(histogram, low_target, warp_scratch, &selected_bin,
                       &selected_count_lt);
  if (tid == 0) {
    if (selected_bin < 0 || selected_count_lt < 0 ||
        selected_count_lt >= low_target) {
      block_status |= kHistogramFailure;
    } else {
      threshold_code = (static_cast<uint32_t>(high_bin) << kRadixBits) |
                       static_cast<uint32_t>(selected_bin);
      // Avoid a cross-pass shared read-modify-write.  The final certificate
      // remains fail-closed if the compact recount disagrees with this sum.
      first_count_lt = high_count_lt + selected_count_lt;
      base_lt = 0;
      base_eq = first_count_lt;
    }
  }
  __syncthreads();
  if (block_status != 0u) {
    paged_fail_row<EmitOutput>(output + output_row, topk_arg, status, row, block_status,
                   diagnostics, count, producer_status, bad_page_ordinal,
                   bad_page_id);
    return;
  }

  // Pass 3: stable exact compact in logical inline/page order.
  paged_stable_compact_span<EmitOutput>(
      inline_values + inline_row, inline_indices + inline_row, inline_count,
      threshold_code, output + output_row, topk_arg, warp_lt, warp_eq, &tile_lt,
      &tile_eq, &base_lt, &base_eq);
  for (int logical_page = 0; logical_page < num_pages; ++logical_page) {
    if (tid == 0) {
      page_id = page_table[static_cast<int64_t>(row) * table_pages_per_row +
                           logical_page];
      if (page_id < 0 || page_id >= overflow_pages) {
        block_status |= kPagedInvalidPage;
        bad_page_ordinal = logical_page;
        bad_page_id = page_id;
      }
    }
    __syncthreads();
    if (block_status != 0u)
      break;
    const int page_count = min(
        overflow_count - logical_page * overflow_page_size, overflow_page_size);
    const int64_t page_base =
        static_cast<int64_t>(page_id) * overflow_page_size;
    paged_stable_compact_span<EmitOutput>(
        overflow_values + page_base, overflow_indices + page_base, page_count,
        threshold_code, output + output_row, topk_arg, warp_lt, warp_eq,
        &tile_lt, &tile_eq, &base_lt, &base_eq);
  }
  __syncthreads();
  if (tid == 0 && block_status == 0u) {
    if (base_lt != first_count_lt || base_lt > topk_arg || base_eq < topk_arg) {
      block_status |= kCompactFailure;
    }
  }
  __syncthreads();

  // A status-64 certificate is recoverable without poisoning the request.
  // Retry only that row with an independent 3x8-bit radix selector.  Each
  // 256-bin digit is chosen by thread 0's scalar scan, deliberately avoiding
  // both the failed 12-bit block-prefix selector and its shared control words.
  // The normal path pays only this uniform branch; no extra kernel is launched.
  if (block_status == kCompactFailure) {
    if (tid == 0) {
      repair_prefix = 0u;
      repair_rank = topk_arg;
      repair_failed = 0;
    }
    __syncthreads();

#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
      for (int bin = tid; bin < 256; bin += kThreads) {
        histogram[bin] = 0;
      }
      __syncthreads();

      const uint32_t pass_prefix = repair_prefix;
      paged_repair_histogram_span(inline_values + inline_row,
                                  inline_indices + inline_row, inline_count,
                                  pass, pass_prefix, histogram);
      __syncthreads();
      for (int logical_page = 0; logical_page < num_pages; ++logical_page) {
        const int repair_page_id =
            page_table[static_cast<int64_t>(row) * table_pages_per_row +
                       logical_page];
        const int page_count =
            min(overflow_count - logical_page * overflow_page_size,
                overflow_page_size);
        const int64_t page_base =
            static_cast<int64_t>(repair_page_id) * overflow_page_size;
        paged_repair_histogram_span(overflow_values + page_base,
                                    overflow_indices + page_base, page_count,
                                    pass, pass_prefix, histogram);
        __syncthreads();
      }

      if (tid == 0) {
        const int target = repair_rank;
        int count_before = 0;
        int winning_bin = -1;
        int winning_count = 0;
        int winning_before = 0;
        for (int bin = 0; bin < 256; ++bin) {
          const int bin_count = histogram[bin];
          if (winning_bin < 0 && target > count_before &&
              target <= count_before + bin_count) {
            winning_bin = bin;
            winning_count = bin_count;
            winning_before = count_before;
          }
          count_before += bin_count;
        }
        if (winning_bin < 0 || winning_count <= 0) {
          repair_failed = 1;
        } else {
          const int shift = 16 - pass * 8;
          repair_prefix =
              pass_prefix | (static_cast<uint32_t>(winning_bin) << shift);
          repair_rank = target - winning_before;
        }
      }
      __syncthreads();
      if (repair_failed != 0)
        break;
    }

    if (tid == 0 && repair_failed == 0) {
      if (repair_rank <= 0 || repair_rank > topk_arg) {
        repair_failed = 1;
      } else {
        threshold_code = repair_prefix;
        first_count_lt = topk_arg - repair_rank;
        base_lt = 0;
        base_eq = first_count_lt;
      }
    }
    __syncthreads();

    if (repair_failed == 0) {
      paged_stable_compact_span<EmitOutput>(
          inline_values + inline_row, inline_indices + inline_row, inline_count,
          threshold_code, output + output_row, topk_arg, warp_lt, warp_eq,
          &tile_lt, &tile_eq, &base_lt, &base_eq);
      for (int logical_page = 0; logical_page < num_pages; ++logical_page) {
        const int repair_page_id =
            page_table[static_cast<int64_t>(row) * table_pages_per_row +
                       logical_page];
        const int page_count =
            min(overflow_count - logical_page * overflow_page_size,
                overflow_page_size);
        const int64_t page_base =
            static_cast<int64_t>(repair_page_id) * overflow_page_size;
        paged_stable_compact_span<EmitOutput>(
            overflow_values + page_base, overflow_indices + page_base,
            page_count, threshold_code, output + output_row, topk_arg, warp_lt,
            warp_eq, &tile_lt, &tile_eq, &base_lt, &base_eq);
      }
      __syncthreads();
      if (tid == 0) {
        if (base_lt == first_count_lt && base_lt < topk_arg &&
            base_eq >= topk_arg) {
          block_status &= ~static_cast<uint32_t>(kCompactFailure);
        } else {
          repair_failed = 1;
        }
      }
    }
    __syncthreads();
  }

  if (block_status != 0u) {
    paged_fail_row<EmitOutput>(output + output_row, topk_arg, status, row, block_status,
                   diagnostics, count, producer_status, bad_page_ordinal,
                   bad_page_id);
    if (tid == 0 && (block_status & kCompactFailure) != 0u) {
      // Preserve the exact cursor certificate for a fail-closed diagnostic.
      // diagnostic_scratch is dead after selection and has five ints per row.
      int32_t *const diag = diagnostics + static_cast<int64_t>(row) * 5;
      diag[1] = first_count_lt;
      diag[2] = base_lt;
      diag[3] = base_eq;
      diag[4] = static_cast<int32_t>(threshold_code);
    }
    return;
  }

  if (tid == 0) {
    status[row] = 0;
    int32_t *const diag = diagnostics + static_cast<int64_t>(row) * 5;
    diag[0] = count;
    diag[1] = static_cast<int32_t>(producer_status);
    diag[2] = bad_page_ordinal;
    diag[3] = bad_page_id;
    diag[4] = static_cast<int32_t>(threshold_code); // exact kth CODE
  }
}

} // namespace overflow

static_assert(kBoundaryOverflow == overflow::kBoundaryOverflow);
static_assert(kBoundaryOverflow == 32u);

} // namespace h2048_safe_topk
