#include "h32_common.cuh"
#include "../common/replay_layout.cuh"
#include "../common/replay_packet.cuh"
namespace partitioned_h32_replay {
using namespace certicascade_streaming_exact;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
struct alignas(1024) Smem {
  alignas(1024) uint8_t keys[128*128], q[kHeads*128];
  alignas(128) float weights[kHeads], scale[128];
  Barrier done;
  uint32_t tmem;
  int job;
};
struct Args {
  const uint8_t *q, *k;
  const float *scale, *weights;
  const int *starts, *ends;
  int *error;
  float *dense;
  int *counts;
  int nk;  // Physical K storage and dense row pitch.
  const int2* tasks;
  int* state;
  int* row_slot;
  float* leaf;
  int* fast_rows;
  int* fast_state;
  int* finite;
  int* leaf_lengths;
  int chunks;
  const int* logical_nk;
};
using dosa_h32_replay::swizzle;


constexpr int PartitionKeys=4096;
// One reservation per failed query; normal rows never read Q/K or emit jobs.
__global__ void prepare_jobs(Args a,int nq,int2*tasks){
  int row=blockIdx.x*blockDim.x+threadIdx.x;
  if(row>=nq)return;
  a.counts[row]=0;a.row_slot[row]=-1;
  if(a.error[row]==0)return;
  int begin=a.starts[row],end=a.ends[row];
  int logical=dosa_replay_layout::logical_extent(a.logical_nk,a.nk);
  if(!dosa_replay_layout::valid_interval(begin,end,logical,a.nk)){atomicOr(a.error+row,2);return;}
  if(end>0){
    int slot=atomicAdd(a.state+2,1);a.row_slot[row]=slot;
    if(slot<8){a.state[11+slot]=row;if(begin>0||end<2048)atomicOr(a.state+3+slot,1);}
  }
  int chunks=int((int64_t(end)-begin+PartitionKeys-1)/PartitionKeys);
  chunks=max(chunks,1); // An empty visible interval can still need its -Inf prefix.
  int base=atomicAdd(a.state,chunks);
  for(int j=0;j<chunks;++j)tasks[base+j]=make_int2(row,int(int64_t(begin)+int64_t(j)*PartitionKeys));
}
CUTLASS_GLOBAL void __launch_bounds__(128,4) replay_worker(Args a){
  const int t=threadIdx.x,warp=t/32,task_count=a.state[0];
  if(blockIdx.x>=task_count)return;
  extern __shared__ __align__(1024) uint8_t buf[];
  auto&s=*reinterpret_cast<Smem*>(buf);
  if(t==0){s.done.init(1);cutlass::arch::fence_barrier_init();}
  if(warp==0){cute::TMEM::Allocator1Sm().allocate(kHeads,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();}
  __syncthreads();
  const int tm=deep_gemm::ptx::ld_shared(&s.tmem);
  int seq=0;
  for(;;){
    if(t==0)s.job=atomicAdd(a.state+1,1);
    __syncthreads();
    int job=s.job;
    if(job>=task_count)break;
    int2 task=a.tasks[job];int row=task.x;
    int begin=a.starts[row],end=a.ends[row],segment_begin=task.y;
    int segment_end=int(min(int64_t(end),int64_t(segment_begin)+PartitionKeys));
    float*dst=a.dense+uint64_t(row)*a.nk;
    int fast_slot=a.row_slot[row];
    bool emit_leaf=a.state[2]<=8&&fast_slot>=0&&fast_slot<8;
    // Exactly the first partition owns the original-ID -Inf prefix. Other
    // partitions write only their disjoint visible ranges; no clearing race.
    if(segment_begin==begin)for(int64_t key=t;key<begin;key+=128)dst[key]=-INFINITY;
    if(segment_begin>=segment_end){__syncthreads();continue;}
  for(int i=t;i<kHeads*8;i+=128)
    reinterpret_cast<uint4*>(s.q)[swizzle(i/8,i%8)]=
        reinterpret_cast<const uint4*>(a.q+uint64_t(row)*kHeads*128)[i];
  if(t<kHeads)s.weights[t]=a.weights[row*kHeads+t];
  __syncthreads();
  for(int64_t first=segment_begin;first<segment_end;first+=128,++seq) {
    const int64_t key=first+t;
    const bool active=key<segment_end;
    float acc[kHeads];
    dosa_h32_replay::load_packet(s, a, key, active, t, tm, seq, acc);
    if(active){
      float score=exact_h32_reduce(acc,s.weights,s.scale[t]);dst[key]=score;
      if(emit_leaf){
        a.leaf[uint64_t(fast_slot)*a.chunks*32768+key]=score;
        if(!isfinite(score))atomicOr(a.state+3+fast_slot,1);
      }
    }
    deep_gemm::ptx::tcgen05_before_thread_sync();
    __syncthreads();
  }

  }
  if(warp==0)cute::TMEM::Allocator1Sm().free(tm,kHeads);
}
// Publish counts only after every worker finished all required dense writes.
__global__ void publish_counts(Args a,int nq){
  int index=blockIdx.x*blockDim.x+threadIdx.x,total=a.state[2];
  if(index<nq){
    int count=0;
    if(a.error[index]!=0){
      int begin=a.starts[index],end=a.ends[index];
      int logical=dosa_replay_layout::logical_extent(a.logical_nk,a.nk);
      if(dosa_replay_layout::valid_interval(begin,end,logical,a.nk))count=end;
    }
    a.counts[index]=count;
  }
  if(index==0)*a.fast_state=total;
  if(index<8){
    a.fast_rows[index]=index<total?a.state[11+index]:-1;
    a.finite[index]=total<=8&&index<total&&a.state[3+index]==0;
  }
  if(index<8*a.chunks){
    int slot=index/a.chunks,seg=index%a.chunks;
    bool valid=total<=8&&slot<total&&a.state[3+slot]==0;
    a.leaf_lengths[index]=valid?max(0,min(32768,a.ends[a.state[11+slot]]-seg*32768)):0;
  }
}
} // namespace partitioned_h32_replay
static void replay_and_leaf_impl(torch::Tensor q,torch::Tensor k,torch::Tensor scale,torch::Tensor weights,
            torch::Tensor starts,torch::Tensor ends,torch::Tensor error_status,
            torch::Tensor dense_scores,torch::Tensor counts,torch::Tensor leaf,torch::Tensor fast_rows,
            torch::Tensor fast_state,torch::Tensor finite,torch::Tensor leaf_lengths,
            torch::Tensor logical_nk) {
  using namespace partitioned_h32_replay;
  c10::cuda::CUDAGuard guard(q.device());
  const int64_t nq=q.size(0), nk=k.size(0);
  dosa_replay_layout::check_inputs(
      q, k, scale, weights, starts, ends, error_status, dense_scores, counts, nq, nk);
  TORCH_CHECK(nq>0&&nq<=512&&nk>0&&nk<=1048576,"bounded recovery batch");
  const int* logical=dosa_replay_layout::logical_pointer(q,logical_nk);
  const int chunks=int((nk+32767)/32768);
  TORCH_CHECK(leaf.is_cuda()&&leaf.device()==q.device()&&leaf.scalar_type()==torch::kFloat32&&leaf.is_contiguous()&&leaf.dim()==2&&leaf.size(0)==8*chunks&&leaf.size(1)==32768,"same-shape leaf buffer");
  for(const auto&x:{fast_rows,fast_state,finite,leaf_lengths})
    TORCH_CHECK(x.is_cuda()&&x.device()==q.device()&&x.scalar_type()==torch::kInt32&&x.is_contiguous(),"leaf metadata tensors");
  TORCH_CHECK(fast_rows.numel()==8&&fast_state.numel()==1&&finite.numel()==8&&leaf_lengths.numel()==8*chunks,"leaf metadata lengths");
  const int64_t max_tasks=nq*std::max<int64_t>(1,(nk+PartitionKeys-1)/PartitionKeys);
  const int blocks=int(std::min<int64_t>(max_tasks,4*at::cuda::getDeviceProperties(q.get_device())->multiProcessorCount));
  TORCH_CHECK(max_tasks<=INT32_MAX-blocks,"bounded int32 replay task cursor");
  auto tasks=torch::empty({max_tasks,2},error_status.options());
  auto state=torch::zeros({19},error_status.options());
  auto row_slot=torch::empty({nq},error_status.options());
  Args a{static_cast<const uint8_t*>(q.data_ptr()),static_cast<const uint8_t*>(k.data_ptr()),
      scale.data_ptr<float>(),weights.data_ptr<float>(),starts.data_ptr<int>(),ends.data_ptr<int>(),
      error_status.data_ptr<int>(),dense_scores.data_ptr<float>(),counts.data_ptr<int>(),int(nk),
      reinterpret_cast<const int2*>(tasks.data_ptr<int>()),state.data_ptr<int>(),row_slot.data_ptr<int>(),
      leaf.data_ptr<float>(),fast_rows.data_ptr<int>(),fast_state.data_ptr<int>(),finite.data_ptr<int>(),leaf_lengths.data_ptr<int>(),chunks,logical};
  auto stream=at::cuda::getCurrentCUDAStream();
  prepare_jobs<<<(nq+127)/128,128,0,stream>>>(a,int(nq),reinterpret_cast<int2*>(tasks.data_ptr<int>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaFuncSetAttribute(replay_worker,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Smem)));
  replay_worker<<<blocks,128,sizeof(Smem),stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  publish_counts<<<(std::max<int64_t>(nq,8*chunks)+127)/128,128,0,stream>>>(a,int(nq));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void replay_and_leaf(torch::Tensor q,torch::Tensor k,torch::Tensor scale,torch::Tensor weights,
            torch::Tensor starts,torch::Tensor ends,torch::Tensor error_status,
            torch::Tensor dense_scores,torch::Tensor counts,torch::Tensor leaf,torch::Tensor fast_rows,
            torch::Tensor fast_state,torch::Tensor finite,torch::Tensor leaf_lengths) {
  replay_and_leaf_impl(q,k,scale,weights,starts,ends,error_status,dense_scores,counts,
      leaf,fast_rows,fast_state,finite,leaf_lengths,torch::Tensor());
}

void replay_and_leaf_capacity(torch::Tensor q,torch::Tensor k,torch::Tensor scale,torch::Tensor weights,
            torch::Tensor starts,torch::Tensor ends,torch::Tensor error_status,
            torch::Tensor dense_scores,torch::Tensor counts,torch::Tensor leaf,torch::Tensor fast_rows,
            torch::Tensor fast_state,torch::Tensor finite,torch::Tensor leaf_lengths,
            torch::Tensor logical_nk) {
  TORCH_CHECK(logical_nk.defined(),"capacity leaf replay requires logical_nk");
  replay_and_leaf_impl(q,k,scale,weights,starts,ends,error_status,dense_scores,counts,
      leaf,fast_rows,fast_state,finite,leaf_lengths,logical_nk);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("replay_and_leaf",&replay_and_leaf);m.def("replay_and_leaf_capacity",&replay_and_leaf_capacity);m.attr("partition_keys")=4096;m.attr("grid_sms_multiplier")=4;}
