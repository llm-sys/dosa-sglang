#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <utility>
#include "head_geometry.cuh"

namespace dosa_emit_host {

// Callers check the x15/stream6 ABI before their family-specific checks.
// Args and PagedArgs keep the existing per-kernel namespace and device layout.
template <typename Args, typename PagedArgs>
std::pair<Args, PagedArgs> bind(
    int paid, const std::vector<torch::Tensor>& x,
    const std::vector<torch::Tensor>& streams, const torch::Tensor& pool,
    const torch::Tensor& table, const torch::Tensor& head,
    const torch::Tensor& status,
    const torch::Tensor& error, int64_t seed, const torch::Tensor& words) {
  int nq = x[0].size(0), nk = x[1].size(0), H = paid + 2;
  TORCH_CHECK(dosa_geometry::valid_certificate_budget(paid) && nq % 4 == 0 &&
                  seed >= 0 && seed <= nk,
              "family/Q/seed contract");
  for (const auto& t : x)
    TORCH_CHECK(t.is_cuda() && t.device() == x[0].device() && t.is_contiguous(),
                "contiguous input device");
  for (const auto& t : streams)
    TORCH_CHECK(t.is_cuda() && t.device() == x[0].device() && t.is_contiguous() &&
                    t.scalar_type() == torch::kInt32,
                "int32 stream device");
  for (const auto& t : {pool, table, head, status, error, words})
    TORCH_CHECK(t.is_cuda() && t.device() == x[0].device() && t.is_contiguous() &&
                    t.scalar_type() == torch::kInt32,
                "int32 pool device");
  TORCH_CHECK(x[0].scalar_type() == torch::kUInt8 &&
                  x[1].scalar_type() == torch::kUInt8 &&
                  x[5].scalar_type() == torch::kUInt8 &&
                  x[0].dim() == 3 && x[0].size(2) == 128 &&
                  (x[0].size(1) == 32 || x[0].size(1) == 64) &&
                  x[1].numel() == int64_t(nk) * 128 &&
                  x[5].numel() == int64_t(nq) * H * 128,
              "Q/K/fine shape");
  TORCH_CHECK(x[2].numel() >= nk && x[3].numel() >= nk &&
                  x[4].numel() == int64_t(nq) * x[0].size(1) &&
                  x[6].numel() == int64_t(nq) * H &&
                  x[7].numel() == int64_t(nq) * (H >= 22 ? 96 : 64),
              "coefficient shape");
  for (int i : {8, 9, 10, 11, 12, 14})
    TORCH_CHECK(x[i].numel() == nq, "query shape");
  TORCH_CHECK(x[13].numel() == nq / 4 && pool.dim() == 2 &&
                  pool.size(1) == 128 && pool.size(0) > 0,
              "budget/pool page shape");
  TORCH_CHECK(table.dim() == 2 && table.size(0) == nq + nq / 4 &&
                  table.size(1) >= (nk - seed + 127) / 128,
              "row page table capacity");
  TORCH_CHECK(head.numel() == 1 && status.numel() == 1 && error.numel() == nq,
              "pool/error status shape");
  TORCH_CHECK(seed % 256 == 0 && words.dim() == 3 && words.size(0) == nq / 4 &&
                  words.size(1) == (nk - seed + 31) / 32 && words.size(2) == 4,
              "aligned seed and complete word matrix");
  TORCH_CHECK(streams[0].numel() == nq && streams[3].numel() == nq / 4 &&
                  streams[2].numel() == 1 && streams[5].numel() == 1,
              "stream counts");
  for (int i : {1, 4})
    TORCH_CHECK(streams[i].dim() == 2 && streams[i].size(1) == 2 &&
                    streams[i].size(0) > 0,
                "int2 task shape");

  Args a{};
  a.q = reinterpret_cast<const uint4*>(x[0].data_ptr());
  a.k = reinterpret_cast<const uint4*>(x[1].data_ptr());
  a.ks = x[2].data_ptr<float>();
  a.rho = reinterpret_cast<const uint32_t*>(x[3].data_ptr<int>());
  a.w = x[4].data_ptr<float>();
  a.fine = reinterpret_cast<const uint4*>(x[5].data_ptr());
  a.fine_scale = x[6].data_ptr<float>();
  a.meta = x[7].data_ptr<float>();
  a.raw = x[8].data_ptr<float>();
  a.out = x[9].data_ptr<float>();
  a.hint = x[10].data_ptr<int>();
  a.starts = x[11].data_ptr<int>();
  a.ends = x[12].data_ptr<int>();
  a.budget = x[13].data_ptr<int>();
  a.tau = x[14].data_ptr<float>();
  a.nq = nq;
  a.nk = nk;
  a.seed = seed;
  a.error = error.data_ptr<int>();

  PagedArgs pg{};
  pg.pool = pool.data_ptr<int>();
  pg.table = table.data_ptr<int>();
  pg.head = head.data_ptr<int>();
  pg.status = status.data_ptr<int>();
  pg.error = error.data_ptr<int>();
  pg.solo_counts = streams[0].data_ptr<int>();
  pg.solo_tasks = reinterpret_cast<int2*>(streams[1].data_ptr<int>());
  pg.solo_n = streams[2].data_ptr<int>();
  pg.common_counts = streams[3].data_ptr<int>();
  pg.common_tasks = reinterpret_cast<int2*>(streams[4].data_ptr<int>());
  pg.common_n = streams[5].data_ptr<int>();
  pg.words = reinterpret_cast<unsigned*>(words.data_ptr<int>());
  pg.word_count = words.size(1);
  pg.nq = nq;
  pg.nk = nk;
  pg.max_records = nk - seed;
  pg.columns = table.size(1);
  pg.pages = pool.size(0);
  pg.solo_task_cap = streams[1].size(0);
  pg.common_task_cap = streams[4].size(0);
  return {a, pg};
}

}  // namespace dosa_emit_host
