#pragma once

// Exact inverse of the request's PAGED table. Only the existing stock
// selector's completed output is consumed; this does not select new scores.
namespace stock_prior {
constexpr int kThreads = 256;
constexpr int kInvalidTable = 1;
constexpr int kDuplicatePage = 2;
constexpr int kInvalidWinner = 4;
constexpr int kInvalidVotes = 8;
constexpr int kHashFull = 16;

__device__ __forceinline__ uint32_t slot_of(int page, int slots) {
  return (uint32_t(page) * 2654435761u) & uint32_t(slots - 1);
}

__global__ void initialize(int* keys, int* values, int slots, int* status) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < slots;
       i += blockDim.x * gridDim.x) {
    keys[i] = -1;
    values[i] = INT_MAX;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) *status = 0;
}

__global__ void build_inverse(const int* table, int pages, int* keys,
                              int* values, int slots, int* status) {
  for (int logical = blockIdx.x * blockDim.x + threadIdx.x; logical < pages;
       logical += blockDim.x * gridDim.x) {
    const int page = table[logical];
    if (page < 0 || page > INT_MAX / 64) {
      atomicOr(status, kInvalidTable);
      continue;
    }
    uint32_t slot = slot_of(page, slots);
    bool found = false;
    for (int n = 0; n < slots; ++n, slot = (slot + 1) & (slots - 1)) {
      const int old = atomicCAS(keys + slot, -1, page);
      if (old == -1) {
        // A duplicate may have marked this -1 before the owner arrives.
        // atomicMin preserves that rejection under either publication order.
        atomicMin(values + slot, logical);
        found = true;
        break;
      }
      if (old == page) {
        atomicMin(values + slot, -1);
        atomicOr(status, kDuplicatePage);
        found = true;
        break;
      }
    }
    if (!found) atomicOr(status, kHashFull);
  }
}

__global__ void inverse_votes(const int* paged, int items, const int* keys,
                              const int* values, int slots, int nv,
                              int* votes, int* status) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < items;
       i += blockDim.x * gridDim.x) {
    const int physical = paged[i];
    if (physical < 0) {
      atomicOr(status, kInvalidWinner);
      continue;
    }
    const int page = physical / 64;
    uint32_t slot = slot_of(page, slots);
    int logical_page = -1;
    for (int n = 0; n < slots; ++n, slot = (slot + 1) & (slots - 1)) {
      const int key = keys[slot];
      if (key == page) {
        logical_page = values[slot];
        break;
      }
      if (key == -1) break;
    }
    const int64_t logical = int64_t(logical_page) * 64 + physical % 64;
    if (logical_page < 0 || logical < 0 || logical >= nv) {
      atomicOr(status, kInvalidWinner);
      continue;
    }
    atomicAdd(votes + logical, 1);
  }
}

__global__ void bound_votes(int* votes, int nv, int max_vote, int* status) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nv;
       i += blockDim.x * gridDim.x) {
    const int v = votes[i];
    if (v < 0 || v > max_vote) {
      atomicOr(status, kInvalidVotes);
      // Keep the original carry selector's histogram access in bounds even
      // for rejected input. A flagged result is never adopted as a prior.
      votes[i] = min(max(v, 0), max_vote);
    }
  }
}

void paged_to_votes_(torch::Tensor snapshot, torch::Tensor hash_storage,
                     torch::Tensor votes, torch::Tensor status,
                     int64_t recent64) {
  TORCH_CHECK(snapshot.is_cuda() && hash_storage.is_cuda() && votes.is_cuda()
              && status.is_cuda(), "stock prior operands must be CUDA");
  TORCH_CHECK(snapshot.device() == votes.device()
              && hash_storage.device() == votes.device()
              && status.device() == votes.device(), "stock prior device mismatch");
  for (const auto& x : {snapshot, hash_storage, votes, status}) {
    TORCH_CHECK(x.scalar_type() == torch::kInt && x.is_contiguous()
                && x.dim() == 1, "stock prior operands must be contiguous I32 vectors");
  }
  const int64_t nv64 = votes.numel();
  TORCH_CHECK(nv64 >= 12288 && nv64 <= 1048576, "stock prior K capacity");
  TORCH_CHECK(recent64 >= 1 && recent64 <= 1536, "stock prior recent rows");
  const int nv = int(nv64), recent = int(recent64);
  const int pages = (nv + 63) / 64;
  int slots = 1;
  while (slots < pages * 2) slots *= 2;
  TORCH_CHECK(snapshot.numel() == int64_t(recent) * 2048 + pages,
              "stock prior snapshot shape");
  TORCH_CHECK(hash_storage.numel() >= int64_t(slots) * 2 && status.numel() == 1,
              "stock prior workspace capacity");
  const c10::cuda::CUDAGuard guard(votes.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  int* keys = hash_storage.data_ptr<int>();
  int* values = keys + slots;
  int* flags = status.data_ptr<int>();
  initialize<<<std::min(128, (slots + 255) / 256), kThreads, 0, stream>>>(
      keys, values, slots, flags);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  build_inverse<<<(pages + 255) / 256, kThreads, 0, stream>>>(
      snapshot.data_ptr<int>() + recent * 2048, pages, keys, values, slots, flags);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  inverse_votes<<<std::min(256, (recent * 2048 + 255) / 256), kThreads, 0, stream>>>(
      snapshot.data_ptr<int>(), recent * 2048, keys, values, slots, nv,
      votes.data_ptr<int>(), flags);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  bound_votes<<<std::min(256, (nv + 255) / 256), kThreads, 0, stream>>>(
      votes.data_ptr<int>(), nv, recent, flags);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace stock_prior
