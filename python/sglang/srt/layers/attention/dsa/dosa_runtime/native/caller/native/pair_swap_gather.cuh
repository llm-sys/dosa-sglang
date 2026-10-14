#pragma once
// Qualified single-request planner/gather. Inactive tiered routines omitted.
namespace pair_swap_gather {
namespace cg = cooperative_groups;
constexpr int kPlanThreads = 256;
constexpr int kGatherBlockY = 32;
constexpr int kGatherThreadsX = 8;
constexpr int kGatherVecBytes = 16;
constexpr int kSeedHotSize = 8192;
constexpr int kMaxHotSize = 12288;

__global__ __launch_bounds__(kPlanThreads, 1) void cooperative_plan_kernel(
    const int64_t *__restrict__ hot, int *__restrict__ hot_epoch,
    int *__restrict__ permutation, int *__restrict__ swap_a,
    int *__restrict__ swap_b, int *__restrict__ counts, int hot_size,
    int window_start, int common_end, int epoch) {
  cg::grid_group grid = cg::this_grid();
  const int i = static_cast<int>(blockIdx.x) * kPlanThreads + threadIdx.x;
  const int lane = threadIdx.x & 31;

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    counts[1] = 0;
    counts[2] = 0;
  }
  const int old_count = max(0, min(counts[0], hot_size));
  if (i < old_count) {
    const int a = swap_a[i];
    const int b = swap_b[i];
    permutation[a] = a;
    permutation[b] = b;
  }
  if (i < hot_size) {
    const int64_t value = static_cast<int64_t>(hot[i]);
    if (value >= window_start && value < common_end) {
      const int previous =
          atomicExch(hot_epoch + static_cast<int>(value), epoch);
      if (previous == epoch) {
        atomicOr(counts + 3, 1);
      }
    } else {
      atomicOr(counts + 3, 2);
    }
  }
  grid.sync();

  const int64_t hot_value64 = static_cast<int64_t>(hot[i]);
  const bool hot_in_range =
      hot_value64 >= window_start && hot_value64 < common_end;
  const int hot_value =
      hot_in_range ? static_cast<int>(hot_value64) : window_start;
  const int window_value = window_start + i;
  const bool take_a = hot_in_range && hot_value >= window_start + hot_size;
  const bool take_b = hot_epoch[window_value] != epoch;

  const unsigned a_mask = __ballot_sync(0xffffffffu, take_a);
  int a_base = 0;
  if (lane == 0 && a_mask != 0) {
    a_base = atomicAdd(counts + 1, __popc(a_mask));
  }
  a_base = __shfl_sync(0xffffffffu, a_base, 0);
  if (take_a) {
    const int rank = __popc(a_mask & ((1u << lane) - 1u));
    swap_a[a_base + rank] = hot_value;
  }

  const unsigned b_mask = __ballot_sync(0xffffffffu, take_b);
  int b_base = 0;
  if (lane == 0 && b_mask != 0) {
    b_base = atomicAdd(counts + 2, __popc(b_mask));
  }
  b_base = __shfl_sync(0xffffffffu, b_base, 0);
  if (take_b) {
    const int rank = __popc(b_mask & ((1u << lane) - 1u));
    swap_b[b_base + rank] = window_value;
  }
  grid.sync();

  const int pair_count = min(counts[1], counts[2]);
  const int metadata_error = counts[3];
  if (metadata_error != 0 || counts[1] != counts[2]) {
    if (i == 0 && counts[1] != counts[2]) {
      atomicOr(counts + 3, 8);
    }
    asm volatile("trap;");
    return;
  }
  if (i < pair_count) {
    const int a = swap_a[i];
    const int b = swap_b[i];
    permutation[a] = b;
    permutation[b] = a;
  }
  if (i == 0) {
    counts[0] = pair_count;
  }
}

__global__ void paged_gather_kernel(
    const char *__restrict__ kv_cache, char *__restrict__ dst_k,
    char *__restrict__ dst_scale, const int *__restrict__ block_table,
    const int *__restrict__ permutation, int64_t token_stride, int64_t head_dim,
    int64_t block_stride, int64_t cache_block_size, int num_tokens,
    int quant_block_size) {
  const int dst_token = blockIdx.x * blockDim.y + threadIdx.y;
  const int head_idx =
      (blockIdx.y * blockDim.x + threadIdx.x) * kGatherVecBytes;

  // Each warp has four independent 8-lane token groups.  The x=0 lane
  // performs the permutation lookup and broadcasts it within its group.
  int source =
      threadIdx.x == 0 && dst_token < num_tokens ? permutation[dst_token] : -1;
  source = __shfl_sync(0xffffffffu, source, 0, kGatherThreadsX);
  if (head_idx >= head_dim || dst_token >= num_tokens) {
    return;
  }
  if (source < 0 || source >= num_tokens) {
    asm volatile("trap;");
    return;
  }

  const int block_idx = block_table[source / cache_block_size];
  const int64_t src_block_offset = block_idx * block_stride;
  const int64_t cache_inblock_offset =
      (source % cache_block_size) * head_dim + head_idx;
  const int64_t src_inblock_offset = src_block_offset + cache_inblock_offset;
  const int64_t dst_inblock_offset =
      static_cast<int64_t>(dst_token) * token_stride + head_idx;

  *reinterpret_cast<float4 *>(dst_k + dst_inblock_offset) =
      *reinterpret_cast<const float4 *>(kv_cache + src_inblock_offset);
  if (threadIdx.x == 0) {
    const int64_t src_scale_offset =
        src_block_offset + cache_block_size * head_dim +
        cache_inblock_offset * 4 / quant_block_size;
    *reinterpret_cast<float *>(dst_scale +
                               dst_inblock_offset * 4 / quant_block_size) =
        *reinterpret_cast<const float *>(kv_cache + src_scale_offset);
  }
}

void validate_plan(const torch::Tensor &hot, const torch::Tensor &hot_epoch,
                   const torch::Tensor &permutation,
                   const torch::Tensor &swap_a, const torch::Tensor &swap_b,
                   const torch::Tensor &counts, int64_t window_start,
                   int64_t common_end, int64_t epoch) {
  TORCH_CHECK(hot.is_cuda() && hot_epoch.is_cuda() && permutation.is_cuda() &&
                  swap_a.is_cuda() && swap_b.is_cuda() && counts.is_cuda(),
              "all pair-swap tensors must be CUDA");
  TORCH_CHECK(hot.scalar_type() == torch::kLong &&
                  hot_epoch.scalar_type() == torch::kInt &&
                  permutation.scalar_type() == torch::kInt &&
                  swap_a.scalar_type() == torch::kInt &&
                  swap_b.scalar_type() == torch::kInt &&
                  counts.scalar_type() == torch::kInt,
              "hot must be int64; pair-swap workspaces must be int32");
  TORCH_CHECK(hot.is_contiguous() && hot_epoch.is_contiguous() &&
                  permutation.is_contiguous() && swap_a.is_contiguous() &&
                  swap_b.is_contiguous() && counts.is_contiguous(),
              "all pair-swap tensors must be contiguous");
  TORCH_CHECK(hot.dim() == 1 && hot_epoch.dim() == 1 &&
                  permutation.dim() == 1 && swap_a.dim() == 1 &&
                  swap_b.dim() == 1 && counts.dim() == 1,
              "all pair-swap tensors must be vectors");
  TORCH_CHECK(hot.device() == hot_epoch.device() &&
                  hot.device() == permutation.device() &&
                  hot.device() == swap_a.device() &&
                  hot.device() == swap_b.device() &&
                  hot.device() == counts.device(),
              "all pair-swap tensors must be on one CUDA device");
  TORCH_CHECK(hot_epoch.numel() == permutation.numel(),
              "epoch and permutation lengths must match");
  TORCH_CHECK(swap_a.numel() >= hot.numel() && swap_b.numel() >= hot.numel(),
              "swap workspaces must hold HOT entries");
  TORCH_CHECK(hot.numel() == kSeedHotSize || hot.numel() == kMaxHotSize,
              "production pair-swap planner requires exactly 8192 or 12288 hot "
              "indices");
  TORCH_CHECK(counts.numel() >= 4, "counts must hold four int32 values");
  TORCH_CHECK(window_start >= 0 && window_start + hot.numel() <= common_end &&
                  common_end <= permutation.numel() &&
                  permutation.numel() <= std::numeric_limits<int>::max(),
              "expected 0 <= window_start, window_start + HOT <= common_end "
              "<= sequence length");
  TORCH_CHECK(epoch > 0 && epoch <= std::numeric_limits<int>::max(),
              "epoch must be positive int32");
}

void validate_gather(const torch::Tensor &kv_cache, const torch::Tensor &dst_k,
                     const torch::Tensor &dst_scale,
                     const torch::Tensor &block_table,
                     const torch::Tensor &permutation) {
  TORCH_CHECK(kv_cache.is_cuda() && dst_k.is_cuda() && dst_scale.is_cuda() &&
                  block_table.is_cuda() && permutation.is_cuda(),
              "all gather tensors must be CUDA");
  TORCH_CHECK(kv_cache.device() == dst_k.device() &&
                  kv_cache.device() == dst_scale.device() &&
                  kv_cache.device() == block_table.device() &&
                  kv_cache.device() == permutation.device(),
              "all gather tensors must be on one CUDA device");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8 &&
                  dst_k.scalar_type() == torch::kUInt8 &&
                  dst_scale.scalar_type() == torch::kUInt8,
              "cache/value/scale storage must be uint8");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt &&
                  permutation.scalar_type() == torch::kInt,
              "block table and permutation must be int32");
  TORCH_CHECK(dst_k.is_contiguous() && dst_scale.is_contiguous() &&
                  block_table.is_contiguous() && permutation.is_contiguous(),
              "all gather tensors must be contiguous");
  TORCH_CHECK(kv_cache.dim() == 3 && kv_cache.stride(2) == 1 &&
                  kv_cache.stride(1) == kv_cache.size(2) &&
                  kv_cache.stride(0) >= kv_cache.size(1) * kv_cache.size(2),
              "kv_cache blocks must be internally contiguous "
              "(dim0 may be strided: cross-layer cache slices)");
  TORCH_CHECK(kv_cache.dim() == 3 && dst_k.dim() == 2 && dst_scale.dim() == 2 &&
                  block_table.dim() == 2 && block_table.size(0) == 1 &&
                  permutation.dim() == 1,
              "invalid gather ranks");
  TORCH_CHECK(permutation.numel() == dst_k.size(0),
              "permutation must have one entry per destination token");
  // fp8 rows are 128 bytes, fp4 rows are 64 packed e2m1 bytes; both carry
  // 4 scale bytes per token and the kernel derives row bytes at runtime.
  TORCH_CHECK(dst_scale.size(0) == dst_k.size(0) && dst_scale.size(1) == 4 &&
                  (dst_k.size(1) == 128 || dst_k.size(1) == 64),
              "production gather outputs must be K uint8 [S,128|64] and scale "
              "bytes [S,4]");
  TORCH_CHECK(
      dst_k.size(0) <= block_table.size(1) * kv_cache.size(1),
      "single-request block table does not cover the destination sequence");
}

void launch_plan(const torch::Tensor &hot, const torch::Tensor &hot_epoch,
                 const torch::Tensor &permutation, const torch::Tensor &swap_a,
                 const torch::Tensor &swap_b, const torch::Tensor &counts,
                 int window_start, int common_end, int epoch,
                 cudaStream_t stream) {
  int hot_size = static_cast<int>(hot.numel());
  const int blocks = (hot_size + kPlanThreads - 1) / kPlanThreads;
  const int64_t *hot_ptr = hot.data_ptr<int64_t>();
  int *hot_epoch_ptr = hot_epoch.data_ptr<int>();
  int *permutation_ptr = permutation.data_ptr<int>();
  int *swap_a_ptr = swap_a.data_ptr<int>();
  int *swap_b_ptr = swap_b.data_ptr<int>();
  int *counts_ptr = counts.data_ptr<int>();
  void *args[] = {
      &hot_ptr,    &hot_epoch_ptr, &permutation_ptr, &swap_a_ptr, &swap_b_ptr,
      &counts_ptr, &hot_size,      &window_start,    &common_end, &epoch,
  };
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<const void *>(cooperative_plan_kernel), dim3(blocks),
      dim3(kPlanThreads), args, 0, stream));
}

void launch_gather(const torch::Tensor &kv_cache, const torch::Tensor &dst_k,
                   const torch::Tensor &dst_scale,
                   const torch::Tensor &block_table,
                   const torch::Tensor &permutation, cudaStream_t stream) {
  const int num_tokens = static_cast<int>(dst_k.size(0));
  const int head_dim = static_cast<int>(dst_k.size(1));
  const int quant_block_size =
      static_cast<int>(head_dim * 4 / dst_scale.size(1));
  const dim3 grid((num_tokens + kGatherBlockY - 1) / kGatherBlockY,
                  (head_dim + kGatherThreadsX * kGatherVecBytes - 1) /
                      (kGatherThreadsX * kGatherVecBytes));
  const dim3 block(kGatherThreadsX, kGatherBlockY);
  paged_gather_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const char *>(kv_cache.data_ptr<uint8_t>()),
      reinterpret_cast<char *>(dst_k.data_ptr<uint8_t>()),
      reinterpret_cast<char *>(dst_scale.data_ptr<uint8_t>()),
      block_table.data_ptr<int>(), permutation.data_ptr<int>(), dst_k.stride(0),
      head_dim, kv_cache.stride(0), kv_cache.size(1), num_tokens,
      quant_block_size);
}

// One host transaction; field algorithms and launch geometry remain separate.
// Both validators bind the planner and gather to the same permutation device.
template<auto CheckFields, auto Gather, typename... Fields>
void plan_and_gather(
    const torch::Tensor &hot, const torch::Tensor &hot_epoch,
    const torch::Tensor &permutation, const torch::Tensor &swap_a,
    const torch::Tensor &swap_b, const torch::Tensor &counts,
    int64_t window_start, int64_t common_end, int64_t epoch,
    const torch::Tensor &cache, const torch::Tensor &dst_k,
    const torch::Tensor &dst_scale, const torch::Tensor &table,
    const Fields &... fields) {
  validate_plan(hot, hot_epoch, permutation, swap_a, swap_b, counts,
                window_start, common_end, epoch);
  validate_gather(cache, dst_k, dst_scale, table, permutation);
  CheckFields(dst_k, fields...);
  const c10::cuda::CUDAGuard guard(hot.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  launch_plan(hot, hot_epoch, permutation, swap_a, swap_b, counts,
              static_cast<int>(window_start), static_cast<int>(common_end),
              static_cast<int>(epoch), stream);
  Gather(cache, dst_k, dst_scale, table, permutation, fields..., stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void no_gather_fields(const torch::Tensor &) {}
inline constexpr auto plan_and_permuted_paged_gather_out =
    plan_and_gather<no_gather_fields, launch_gather>;

#include "gather_norm.cuh"
#include "gather_rho.cuh"

} // namespace pair_swap_gather
