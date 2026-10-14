#pragma once
#include <torch/extension.h>
#include "head_geometry.cuh"

namespace dosa_replay_layout {
// A null pointer is the legacy actual-view ABI. The capacity ABI supplies a
// stable device scalar whose value is ordered before every recovery launch.
inline const int* logical_pointer(const torch::Tensor& owner,
                                  const torch::Tensor& logical_nk) {
  if (!logical_nk.defined()) return nullptr;
  TORCH_CHECK(logical_nk.is_cuda() && logical_nk.device() == owner.device() &&
              logical_nk.scalar_type() == torch::kInt32 &&
              logical_nk.dim() == 1 && logical_nk.numel() == 1 &&
              logical_nk.is_contiguous(), "logical_nk must be CUDA int32[1]");
  return logical_nk.data_ptr<int>();
}

// Common tensor contract; the caller keeps its policy-specific batch limits.
inline void check_inputs(
    const torch::Tensor& q, const torch::Tensor& k,
    const torch::Tensor& scale, const torch::Tensor& weights,
    const torch::Tensor& starts, const torch::Tensor& ends,
    const torch::Tensor& error_status, const torch::Tensor& dense_scores,
    const torch::Tensor& counts, int64_t nq, int64_t nk) {
  TORCH_CHECK(q.is_cuda() && q.dim()==3 && q.size(1)==dosa_geometry::Heads && q.size(2)==128
      && q.element_size()==1 && q.is_contiguous(),"q must be contiguous CUDA [Q,model-heads,128] E4 bytes");
  TORCH_CHECK(k.is_cuda() && k.device()==q.device() && k.dim()==2 && k.size(1)==128
      && k.element_size()==1 && k.is_contiguous(),"k must be contiguous CUDA [NK,128] E4 bytes");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr())%16==0
      && reinterpret_cast<uintptr_t>(k.data_ptr())%16==0,"replay E4 base pointers require 16-byte alignment");
  for(const auto &x:{scale,weights,dense_scores})
    TORCH_CHECK(x.is_cuda() && x.device()==q.device() && x.scalar_type()==torch::kFloat32
        && x.is_contiguous(),"scale/weights/dense must be contiguous CUDA F32");
  for(const auto &x:{starts,ends,error_status,counts})
    TORCH_CHECK(x.is_cuda() && x.device()==q.device() && x.scalar_type()==torch::kInt32
        && x.is_contiguous() && x.numel()==nq,"interval/status/count tensors must be CUDA int32 [Q]");
  TORCH_CHECK(scale.numel()==nk && weights.dim()==2 && weights.size(0)==nq && weights.size(1)==dosa_geometry::Heads
      && dense_scores.dim()==2 && dense_scores.size(0)==nq && dense_scores.size(1)==nk,
      "replay score/weight shapes");
  TORCH_CHECK(nk<=INT32_MAX && nq<=INT32_MAX,"replay int32 logical dimensions");
}

__device__ __forceinline__ int logical_extent(const int* logical_nk,
                                             int storage_capacity) {
  return logical_nk ? *logical_nk : storage_capacity;
}

__device__ __forceinline__ bool valid_interval(int begin, int end,
                                              int logical_nk,
                                              int storage_capacity) {
  return logical_nk >= 0 && logical_nk <= storage_capacity &&
         begin >= 0 && begin <= end && end <= logical_nk;
}
}  // namespace dosa_replay_layout
