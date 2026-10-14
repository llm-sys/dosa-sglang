#pragma once
namespace dsa_litetopk { constexpr uint32_t kCandidateIndexMask=(1u<<20)-1u; }





__global__ void map_topk_indices_and_accumulate_votes_litetopk_kernel(
    int32_t *__restrict__ out_idx, const int32_t *__restrict__ index_map,
    const int32_t *__restrict__ status, int32_t *__restrict__ votes,
    int64_t total, int rows, int index_map_size, int topk, int votes_len,
    int vote_recent_rows, const int32_t *__restrict__ cand_cnt,
    int32_t *__restrict__ stat_run_max, int32_t *__restrict__ stat_over,
    int stat_watermark) {
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int64_t global_thread =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t vote_begin =
      static_cast<int64_t>(rows - vote_recent_rows) * topk;
  // A bad/underfilled row may contain selector padding rather than K valid
  // physical indices.  Preserve fail-closed behavior before treating the
  // complete output matrix as mappable winners.  The candidate-count
  // telemetry rides the same one-thread-per-row sweep.
  int stat_local_max = 0;
  int stat_local_over = 0;
  for (int row = static_cast<int>(global_thread); row < rows;
       row += static_cast<int>(step)) {
    if (status[row] != 0) {
      asm volatile("trap;");
      return;
    }
    if (cand_cnt != nullptr) {
      const int c = cand_cnt[row];
      stat_local_max = c > stat_local_max ? c : stat_local_max;
      stat_local_over += c > stat_watermark ? 1 : 0;
    }
  }
  if (cand_cnt != nullptr) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const int m = __shfl_down_sync(0xffffffffu, stat_local_max, off);
      stat_local_max = m > stat_local_max ? m : stat_local_max;
      stat_local_over += __shfl_down_sync(0xffffffffu, stat_local_over, off);
    }
    __shared__ int stat_smax[32], stat_sover[32];
    const int wid = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
      stat_smax[wid] = stat_local_max;
      stat_sover[wid] = stat_local_over;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int m = 0, ov = 0;
      const int warps = (blockDim.x + 31) >> 5;
      for (int w = 0; w < warps; ++w) {
        m = stat_smax[w] > m ? stat_smax[w] : m;
        ov += stat_sover[w];
      }
      if (m > 0)
        atomicMax(stat_run_max, m);
      if (ov > 0)
        atomicAdd(stat_over, ov);
    }
  }
  for (int64_t linear = global_thread; linear < total; linear += step) {
    const int32_t physical_idx = out_idx[linear];
    if (static_cast<uint32_t>(physical_idx) >=
        static_cast<uint32_t>(index_map_size)) {
      asm volatile("trap;");
      return;
    }
    const int32_t original_idx = index_map[physical_idx];
    if (static_cast<uint32_t>(original_idx) >=
            static_cast<uint32_t>(index_map_size) ||
        static_cast<uint32_t>(original_idx) >
            dsa_litetopk::kCandidateIndexMask) {
      asm volatile("trap;");
      return;
    }
    out_idx[linear] = original_idx;

    if (votes != nullptr && votes_len > 0) {
      // The next chunk is best predicted by the most recent query
      // window.  Mapping already visits every Q*K winner, so voting all
      // winners from the last 1536 rows only adds the atomics; it needs
      // no extra winner read or launch.
      if (linear >= vote_begin) {
        const int32_t vote_idx =
            original_idx < 0
                ? 0
                : (original_idx >= votes_len ? votes_len - 1 : original_idx);
        atomicAdd(votes + vote_idx, 1);
      }
    }
  }
}

void map_topk_vote_stats_litetopk_(
    torch::Tensor out_idx, torch::Tensor index_map, torch::Tensor status,
    torch::Tensor votes, int64_t vote_recent_rows64, torch::Tensor cand_cnt,
    torch::Tensor run_max, torch::Tensor over_events, int64_t watermark64) {
  TORCH_CHECK(cand_cnt.is_cuda() && run_max.is_cuda() && over_events.is_cuda(),
              "stats tensors must be CUDA");
  TORCH_CHECK(cand_cnt.is_contiguous() && run_max.is_contiguous() &&
                  over_events.is_contiguous(),
              "stats tensors must be contiguous");
  TORCH_CHECK(cand_cnt.scalar_type() == torch::kInt &&
                  run_max.scalar_type() == torch::kInt &&
                  over_events.scalar_type() == torch::kInt,
              "stats tensors must be int32");
  TORCH_CHECK(cand_cnt.numel() >= out_idx.size(0) && run_max.numel() >= 1 &&
                  over_events.numel() >= 1,
              "stats tensors too small");
  TORCH_CHECK(out_idx.is_cuda() && out_idx.is_contiguous() &&
                  out_idx.scalar_type() == torch::kInt && out_idx.dim() == 2 &&
                  out_idx.numel() > 0,
              "out_idx must be a nonempty contiguous int32 [R,K]");
  TORCH_CHECK(status.numel() == out_idx.size(0) && votes.dim() == 1 &&
                  vote_recent_rows64 >= 0 &&
                  vote_recent_rows64 <= out_idx.size(0),
              "map stats: bad status/votes/recent-rows");

  constexpr int kThreads = 256;
  constexpr int kBlocksPerSm = 8;
  constexpr int kProductionSms = 148;
  const int64_t total = out_idx.numel();
  const int blocks = static_cast<int>(std::min<int64_t>(
      (total + kThreads - 1) / kThreads, kProductionSms * kBlocksPerSm));
  const int votes_len = static_cast<int>(votes.numel());
  const c10::cuda::CUDAGuard device_guard(out_idx.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  map_topk_indices_and_accumulate_votes_litetopk_kernel<<<blocks, kThreads, 0,
                                                          stream>>>(
      out_idx.data_ptr<int32_t>(), index_map.data_ptr<int32_t>(),
      status.data_ptr<int32_t>(),
      votes_len > 0 ? votes.data_ptr<int32_t>() : nullptr, total,
      static_cast<int>(out_idx.size(0)), static_cast<int>(index_map.numel()),
      static_cast<int>(out_idx.size(1)), votes_len,
      static_cast<int>(vote_recent_rows64), cand_cnt.data_ptr<int32_t>(),
      run_max.data_ptr<int32_t>(), over_events.data_ptr<int32_t>(),
      static_cast<int>(watermark64));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__global__ void map_and_page_kernel(
    int32_t *__restrict__ out_idx, const int32_t *__restrict__ index_map,
    const int32_t *__restrict__ status, int32_t *__restrict__ votes,
    int64_t total, int rows, int index_map_size, int topk, int votes_len,
    int vote_recent_rows, const int32_t *__restrict__ cand_cnt,
    int32_t *__restrict__ stat_run_max, int32_t *__restrict__ stat_over,
    int stat_watermark, const int32_t* page_table) {
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int64_t global_thread =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t vote_begin =
      static_cast<int64_t>(rows - vote_recent_rows) * topk;
  // A bad/underfilled row may contain selector padding rather than K valid
  // physical indices.  Preserve fail-closed behavior before treating the
  // complete output matrix as mappable winners.  The candidate-count
  // telemetry rides the same one-thread-per-row sweep.
  int stat_local_max = 0;
  int stat_local_over = 0;
  for (int row = static_cast<int>(global_thread); row < rows;
       row += static_cast<int>(step)) {
    if (status[row] != 0) {
      asm volatile("trap;");
      return;
    }
    if (cand_cnt != nullptr) {
      const int c = cand_cnt[row];
      stat_local_max = c > stat_local_max ? c : stat_local_max;
      stat_local_over += c > stat_watermark ? 1 : 0;
    }
  }
  if (cand_cnt != nullptr) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const int m = __shfl_down_sync(0xffffffffu, stat_local_max, off);
      stat_local_max = m > stat_local_max ? m : stat_local_max;
      stat_local_over += __shfl_down_sync(0xffffffffu, stat_local_over, off);
    }
    __shared__ int stat_smax[32], stat_sover[32];
    const int wid = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
      stat_smax[wid] = stat_local_max;
      stat_sover[wid] = stat_local_over;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int m = 0, ov = 0;
      const int warps = (blockDim.x + 31) >> 5;
      for (int w = 0; w < warps; ++w) {
        m = stat_smax[w] > m ? stat_smax[w] : m;
        ov += stat_sover[w];
      }
      if (m > 0)
        atomicMax(stat_run_max, m);
      if (ov > 0)
        atomicAdd(stat_over, ov);
    }
  }
  for (int64_t linear = global_thread; linear < total; linear += step) {
    const int32_t physical_idx = out_idx[linear];
    if (static_cast<uint32_t>(physical_idx) >=
        static_cast<uint32_t>(index_map_size)) {
      asm volatile("trap;");
      return;
    }
    const int32_t original_idx = index_map[physical_idx];
    if (static_cast<uint32_t>(original_idx) >=
            static_cast<uint32_t>(index_map_size) ||
        static_cast<uint32_t>(original_idx) >
            dsa_litetopk::kCandidateIndexMask) {
      asm volatile("trap;");
      return;
    }
    out_idx[linear] = page_table[original_idx];

    if (votes != nullptr && votes_len > 0) {
      // The next chunk is best predicted by the most recent query
      // window.  Mapping already visits every Q*K winner, so voting all
      // winners from the last 1536 rows only adds the atomics; it needs
      // no extra winner read or launch.
      if (linear >= vote_begin) {
        const int32_t vote_idx =
            original_idx < 0
                ? 0
                : (original_idx >= votes_len ? votes_len - 1 : original_idx);
        atomicAdd(votes + vote_idx, 1);
      }
    }
  }
}

void map_vote_page_(
    torch::Tensor out_idx, torch::Tensor index_map, torch::Tensor status,
    torch::Tensor votes, int64_t vote_recent_rows64, torch::Tensor cand_cnt,
    torch::Tensor run_max, torch::Tensor over_events, int64_t watermark64,
    torch::Tensor page_table) {
  TORCH_CHECK(cand_cnt.is_cuda() && run_max.is_cuda() && over_events.is_cuda(),
              "stats tensors must be CUDA");
  TORCH_CHECK(cand_cnt.is_contiguous() && run_max.is_contiguous() &&
                  over_events.is_contiguous(),
              "stats tensors must be contiguous");
  TORCH_CHECK(cand_cnt.scalar_type() == torch::kInt &&
                  run_max.scalar_type() == torch::kInt &&
                  over_events.scalar_type() == torch::kInt,
              "stats tensors must be int32");
  TORCH_CHECK(cand_cnt.numel() >= out_idx.size(0) && run_max.numel() >= 1 &&
                  over_events.numel() >= 1,
              "stats tensors too small");
  TORCH_CHECK(out_idx.is_cuda() && out_idx.is_contiguous() &&
                  out_idx.scalar_type() == torch::kInt && out_idx.dim() == 2 &&
                  out_idx.numel() > 0,
              "out_idx must be a nonempty contiguous int32 [R,K]");
  TORCH_CHECK(status.numel() == out_idx.size(0) && votes.dim() == 1 &&
                  vote_recent_rows64 > 0 &&
                  vote_recent_rows64 <= out_idx.size(0),
              "map stats: bad status/votes/recent-rows");

  TORCH_CHECK(page_table.is_cuda() && page_table.device()==out_idx.device() &&
      page_table.scalar_type()==torch::kInt && page_table.is_contiguous() &&
      page_table.numel()>=index_map.numel(), "C1 original PAGED row must cover index_map");
  constexpr int kThreads = 256;
  constexpr int kBlocksPerSm = 8;
  constexpr int kProductionSms = 148;
  const int64_t total = out_idx.numel();
  const int blocks = static_cast<int>(std::min<int64_t>(
      (total + kThreads - 1) / kThreads, kProductionSms * kBlocksPerSm));
  const int votes_len = static_cast<int>(votes.numel());
  const c10::cuda::CUDAGuard device_guard(out_idx.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  map_and_page_kernel<<<blocks, kThreads, 0,
                                                          stream>>>(
      out_idx.data_ptr<int32_t>(), index_map.data_ptr<int32_t>(),
      status.data_ptr<int32_t>(),
      votes_len > 0 ? votes.data_ptr<int32_t>() : nullptr, total,
      static_cast<int>(out_idx.size(0)), static_cast<int>(index_map.numel()),
      static_cast<int>(out_idx.size(1)), votes_len,
      static_cast<int>(vote_recent_rows64), cand_cnt.data_ptr<int32_t>(),
      run_max.data_ptr<int32_t>(), over_events.data_ptr<int32_t>(),
      static_cast<int>(watermark64),page_table.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
