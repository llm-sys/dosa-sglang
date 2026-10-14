#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <climits>

#include "pair_swap_gather.cuh"
#include "carry.cuh"
#include "map_votes.cuh"
#include "stock_prior.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("supports_zero_recent_rows", [](){ return true; });
  m.def("key_rho_contract",[](){return "rms128-ks-positive-le2p64-v1";});
  m.def("plan_and_permuted_paged_gather_out",
        pair_swap_gather::plan_and_permuted_paged_gather_out);
  m.def("plan_and_permuted_paged_gather_norm_out",
        pair_swap_gather::plan_and_permuted_paged_gather_norm_out);
  m.def("plan_and_permuted_paged_gather_rho_out",
        pair_swap_gather::plan_and_permuted_paged_gather_rho_out);
  m.def("carry_votes_topk_reset_", &carry_votes_topk_reset_litetopk_);
  m.def("carry_votes_topk_noreset_", &carry_votes_topk_noreset_litetopk_);
  m.def("map_topk_vote_stats_litetopk_", &map_topk_vote_stats_litetopk_);
  m.def("map_vote_page_", &map_vote_page_);
  m.def("stock_paged_to_votes_", &stock_prior::paged_to_votes_);
  m.attr("carry_variant") = "prefix-plan042641-warp-emitter061236";
}
