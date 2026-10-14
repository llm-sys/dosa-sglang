#pragma once
#include "h32_common.cuh"
#include "../common/replay_layout.cuh"
#include "../common/replay_packet.cuh"

namespace conditional_h32_replay {
using namespace certicascade_streaming_exact;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
struct alignas(1024) Smem {
  alignas(1024) uint8_t keys[128*128], q[kHeads*128];
  alignas(128) float weights[kHeads], scale[128];
  Barrier done;
  uint32_t tmem;
};
struct Args {
  const uint8_t *q, *k;
  const float *scale, *weights;
  const int *starts, *ends;
  int *error;
  float *dense;
  int *counts;
  int nk;  // Physical K storage and dense row pitch.
  const int* logical_nk;
};
using dosa_h32_replay::swizzle;

CUTLASS_GLOBAL void __launch_bounds__(128,4) replay_kernel(Args a) {
  const int row=blockIdx.x, t=threadIdx.x, warp=t/32;
  // CTA-uniform status. Normal operation is O(Q), with no Q/K reads or TMEM.
  if(a.error[row]==0) { if(t==0)a.counts[row]=0; return; }
  const int begin=a.starts[row], end=a.ends[row];
  const int logical=dosa_replay_layout::logical_extent(a.logical_nk,a.nk);
  if(!dosa_replay_layout::valid_interval(begin,end,logical,a.nk)) {
    if(t==0) { a.counts[row]=0;atomicOr(a.error+row,2); }
    // Bit 2 means invalid interval / unrecoverable input, never clip or publish.
    return;
  }
  float *dst=a.dense+uint64_t(row)*a.nk;
  // FI receives count=end and therefore preserves implicit original key IDs.
  for(int64_t key=t;key<begin;key+=128)dst[key]=-INFINITY;
  if(begin==end) { if(t==0)a.counts[row]=end; return; }

  extern __shared__ __align__(1024) uint8_t buf[];
  auto &s=*reinterpret_cast<Smem*>(buf);
  if(t==0) { s.done.init(1);cutlass::arch::fence_barrier_init(); }
  if(warp==0) {
    cute::TMEM::Allocator1Sm().allocate(kHeads,&s.tmem);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  for(int i=t;i<kHeads*8;i+=128)
    reinterpret_cast<uint4*>(s.q)[swizzle(i/8,i%8)]=
        reinterpret_cast<const uint4*>(a.q+uint64_t(row)*kHeads*128)[i];
  if(t<kHeads)s.weights[t]=a.weights[row*kHeads+t];
  __syncthreads();
  const int tm=deep_gemm::ptx::ld_shared(&s.tmem);
  int seq=0;
  for(int64_t first=begin;first<end;first+=128,++seq) {
    const int64_t key=first+t;
    const bool active=key<end;
    float acc[kHeads];
    dosa_h32_replay::load_packet(s, a, key, active, t, tm, seq, acc);
    if(active)dst[key]=exact_h32_reduce(acc,s.weights,s.scale[t]);
    deep_gemm::ptx::tcgen05_before_thread_sync();
    __syncthreads();
  }
  if(warp==0)cute::TMEM::Allocator1Sm().free(tm,kHeads);
  if(t==0)a.counts[row]=end;
}
}

static void replay_impl(torch::Tensor q,torch::Tensor k,torch::Tensor scale,torch::Tensor weights,
            torch::Tensor starts,torch::Tensor ends,torch::Tensor error_status,
            torch::Tensor dense_scores,torch::Tensor counts,torch::Tensor logical_nk) {
  using namespace conditional_h32_replay;
  c10::cuda::CUDAGuard guard(q.device());
  const int64_t nq=q.size(0), nk=k.size(0);
  dosa_replay_layout::check_inputs(
      q, k, scale, weights, starts, ends, error_status, dense_scores, counts, nq, nk);
  const int* logical=dosa_replay_layout::logical_pointer(q,logical_nk);
  if(nq==0)return;
  Args a{static_cast<const uint8_t*>(q.data_ptr()),static_cast<const uint8_t*>(k.data_ptr()),
      scale.data_ptr<float>(),weights.data_ptr<float>(),starts.data_ptr<int>(),ends.data_ptr<int>(),
      error_status.data_ptr<int>(),dense_scores.data_ptr<float>(),counts.data_ptr<int>(),int(nk),logical};
  C10_CUDA_CHECK(cudaFuncSetAttribute(replay_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Smem)));
  replay_kernel<<<nq,128,sizeof(Smem),at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void replay(torch::Tensor q,torch::Tensor k,torch::Tensor scale,torch::Tensor weights,
            torch::Tensor starts,torch::Tensor ends,torch::Tensor error_status,
            torch::Tensor dense_scores,torch::Tensor counts) {
  replay_impl(q,k,scale,weights,starts,ends,error_status,dense_scores,counts,torch::Tensor());
}

void replay_capacity(torch::Tensor q,torch::Tensor k,torch::Tensor scale,torch::Tensor weights,
            torch::Tensor starts,torch::Tensor ends,torch::Tensor error_status,
            torch::Tensor dense_scores,torch::Tensor counts,torch::Tensor logical_nk) {
  TORCH_CHECK(logical_nk.defined(),"capacity replay requires logical_nk");
  replay_impl(q,k,scale,weights,starts,ends,error_status,dense_scores,counts,logical_nk);
}
