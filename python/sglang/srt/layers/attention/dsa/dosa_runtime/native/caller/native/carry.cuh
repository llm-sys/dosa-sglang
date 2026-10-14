#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace {
constexpr int kCarryTileItems = 8192;
constexpr int kCarryMaxItems = 1 << 20;
constexpr int kCarryMaxK = 12288;
constexpr int kCarryMaxVote = 8192;
constexpr int kCarryMaxBlocks =
    (kCarryMaxItems + kCarryTileItems - 1) / kCarryTileItems;
constexpr int kCarryThreads = 256;
constexpr int kCarryWarps = kCarryThreads / 32;

enum CarryStateOffset : int {
  kCarryTicket = 0,
  kCarryThreshold = 1,
  kCarryTieBlock = 2,
  kCarryTieTake = 3,
  kCarryOutK = 4,
  kCarryNumBlocks = 5,
  kCarryBlockOffsets = 6,
};
constexpr int kCarryStateInts = kCarryBlockOffsets + kCarryMaxBlocks + 1;

__device__ __forceinline__ int carry_warp_sum(int value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}


#include "prefix_plan.cuh"

#include "warp_emit.cuh"



}

template <bool ResetVotes = true>
void carry_votes_topk_reset_litetopk_impl_(
    torch::Tensor votes, torch::Tensor out_idx, torch::Tensor partial,
    torch::Tensor state, int64_t k64, int64_t max_vote64, int64_t min_index64) {
  TORCH_CHECK(votes.is_cuda() && out_idx.is_cuda() && partial.is_cuda() &&
                  state.is_cuda(),
              "votes/out_idx/partial/state must be CUDA tensors");
  TORCH_CHECK(votes.is_contiguous() && out_idx.is_contiguous() &&
                  partial.is_contiguous() && state.is_contiguous(),
              "votes/out_idx/partial/state must be contiguous");
  TORCH_CHECK(votes.scalar_type() == torch::kInt, "votes must be int32");
  TORCH_CHECK(out_idx.scalar_type() == torch::kLong, "out_idx must be int64");
  TORCH_CHECK(partial.scalar_type() == torch::kShort, "partial must be int16");
  TORCH_CHECK(state.scalar_type() == torch::kInt, "state must be int32");
  TORCH_CHECK(votes.device() == out_idx.device() &&
                  votes.device() == partial.device() &&
                  votes.device() == state.device(),
              "votes/out_idx/partial/state must be on the same CUDA device");
  TORCH_CHECK(votes.dim() == 1, "votes must be a 1-D histogram");
  TORCH_CHECK(out_idx.dim() == 1, "out_idx must be 1-D");
  TORCH_CHECK(partial.dim() == 2, "partial must be [blocks,bins]");
  TORCH_CHECK(state.dim() == 1 && state.numel() >= kCarryStateInts,
              "state is too small for the carry top-k ABI");

  const int64_t count64 = votes.numel();
  TORCH_CHECK(count64 >= 1 && count64 <= kCarryMaxItems,
              "votes length must be in [1,1048576]");
  TORCH_CHECK(k64 >= 1 && k64 <= kCarryMaxK, "k must be in [1,12288]");
  TORCH_CHECK(max_vote64 >= 1 && max_vote64 <= kCarryMaxVote,
              "max_vote must be in [1,8192]");
  TORCH_CHECK(min_index64 >= 0 && min_index64 < count64,
              "min_index must be in [0,votes.numel())");
  const int count = static_cast<int>(count64);
  const int min_index = static_cast<int>(min_index64);
  const int eligible = count - min_index;
  const int out_k = static_cast<int>(std::min(k64, static_cast<int64_t>(eligible)));
  const int max_vote = static_cast<int>(max_vote64);
  const int bins = max_vote + 1;
  const int blocks = (count + kCarryTileItems - 1) / kCarryTileItems;
  TORCH_CHECK(out_idx.numel() == out_k,
              "out_idx must have min(k,votes.numel()-min_index) elements");
  TORCH_CHECK(partial.size(0) >= blocks && partial.size(1) >= bins,
              "partial must provide at least [ceil(N/8192),max_vote+1]");

  const int partial_stride = static_cast<int>(partial.size(1));
  const size_t dynamic_smem = static_cast<size_t>(bins) * sizeof(uint32_t);
  const c10::cuda::CUDAGuard device_guard(votes.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  carry_votes_plan_prefix_kernel<<<blocks, kCarryThreads, dynamic_smem,
                                     stream>>>(
      votes.data_ptr<int32_t>(), count, min_index, out_k, max_vote,
      partial.data_ptr<int16_t>(), partial_stride, state.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  carry_votes_emit_warp_kernel<ResetVotes>
      <<<blocks, kCarryThreads, 0, stream>>>(
          votes.data_ptr<int32_t>(), count, min_index, max_vote,
          out_idx.data_ptr<int64_t>(), state.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void carry_votes_topk_reset_litetopk_(torch::Tensor votes,
                                      torch::Tensor out_idx,
                                      torch::Tensor partial,
                                      torch::Tensor state, int64_t k64,
                                      int64_t max_vote64, int64_t min_index64) {
  carry_votes_topk_reset_litetopk_impl_<true>(votes, out_idx, partial, state,
                                              k64, max_vote64, min_index64);
}

void carry_votes_topk_noreset_litetopk_(
    torch::Tensor votes, torch::Tensor out_idx, torch::Tensor partial,
    torch::Tensor state, int64_t k64, int64_t max_vote64, int64_t min_index64) {
  carry_votes_topk_reset_litetopk_impl_<false>(votes, out_idx, partial, state,
                                               k64, max_vote64, min_index64);
}
