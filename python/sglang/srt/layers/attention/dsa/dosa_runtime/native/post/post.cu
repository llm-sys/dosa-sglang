// Original record5 collector and fused-large H32 completion.
#include "h32_common.cuh"
#include "../common/record5_codec.cuh"
#include <cub/block/block_radix_sort.cuh>
#include <cfloat>

struct PagedCandidateShape {int rows,overflow_pages,table_pages_per_row,logical_cap;};
static PagedCandidateShape check_paged_candidate_layout(
    const torch::Tensor& iv,const torch::Tensor& ii,const torch::Tensor& ov,
    const torch::Tensor& oi,const torch::Tensor& table,int64_t cap) {
  for(const auto&x:{iv,ii,ov,oi,table})
    TORCH_CHECK(x.is_cuda()&&x.device()==iv.device()&&x.is_contiguous(),"candidate device/layout");
  TORCH_CHECK(iv.scalar_type()==torch::kByte&&ov.scalar_type()==torch::kByte&&
      ii.scalar_type()==torch::kInt&&oi.scalar_type()==torch::kInt&&table.scalar_type()==torch::kInt,"candidate dtype");
  TORCH_CHECK(iv.dim()==2&&iv.size(1)==8192&&ii.sizes()==iv.sizes()&&
      ov.dim()==2&&ov.size(1)==4096&&oi.sizes()==ov.sizes()&&table.dim()==2&&
      table.size(0)==iv.size(0)&&cap>0&&cap<=(1<<20)&&
      table.size(1)==(std::max<int64_t>(cap-8192,0)+4095)/4096,"candidate shape/capacity");
  return {int(iv.size(0)),int(ov.size(0)),int(table.size(1)),int(cap)};
}

namespace small_boundary {
constexpr int Wanted=2048, Inline=8192, Page=4096;
__device__ __forceinline__ void fail_row(float*values,int*ids,int*counts,
    int*boundary_counts,int*error,int row,int stride,int t) {
  for(int j=t;j<Wanted;j+=blockDim.x) {
    values[int64_t(row)*stride+j]=-INFINITY;ids[int64_t(row)*stride+j]=-1;
  }
  if(t==0){counts[row]=Wanted;boundary_counts[row]=0;atomicOr(error+row,1);}
}
__global__ void collect_kernel(const uint8_t*inline_values,const int*inline_ids,
    const uint8_t*overflow_values,const int*overflow_ids,const int*table,
    const int*counts,const int*pool_status,const int*selector_status,int*diag,
    const int*starts,const int*ends,int*boundary_ids,int*boundary_counts,
    int2*tasks,int*task_count,float*temporary,int*temporary_ids,int*temporary_counts,
    int*error,int table_stride,int pool_pages,int logical_cap,int index_limit,
    int boundary_cap,int temporary_stride,const float*epsilon,int*output) {
  int row=blockIdx.x,t=threadIdx.x;__shared__ int higher,equal,bad,count;
  if(row==0&&t==0)*task_count=0;
  if(reinterpret_cast<const int*>(tasks)[row]==1 && error[row]==0 &&
      selector_status[row]==0 && *pool_status==0)return;
  __shared__ uint32_t pivot;
  __shared__ float radius,lower_cut,upper_cut;
  if(t==0) {
    higher=equal=0;count=counts[row];pivot=uint32_t(diag[row*5+4]);
    radius=epsilon[row];
    uint32_t lo_word=pivot<<8,hi_word=lo_word|255u;
    float lo=__uint_as_float((lo_word&0x80000000u)?(lo_word^0x80000000u):~lo_word);
    float hi=__uint_as_float((hi_word&0x80000000u)?(hi_word^0x80000000u):~hi_word);
    lower_cut=__fsub_rd(lo,radius);upper_cut=__fadd_ru(hi,radius);
    bad=!isfinite(radius)||radius<0||!isfinite(lower_cut)||!isfinite(upper_cut)||error[row]!=0||selector_status[row]!=0||*pool_status!=0||
        count<Wanted||count>logical_cap||pivot>=0x1000000u||
        starts[row]<0||ends[row]<starts[row]||ends[row]>index_limit;
  }
  __syncthreads();
  if(bad){fail_row(temporary,temporary_ids,temporary_counts,boundary_counts,
                  error,row,temporary_stride,t);return;}
  for(int slot=t;slot<count;slot+=blockDim.x) {
    uint8_t raw_value;int packed;
    if(slot<Inline) {
      int64_t at=int64_t(row)*Inline+slot;
      raw_value=inline_values[at];packed=inline_ids[at];
    } else {
      int ordinal=(slot-Inline)/Page,offset=(slot-Inline)%Page;
      if(ordinal>=table_stride){atomicOr(&bad,1);continue;}
      int page=table[int64_t(row)*table_stride+ordinal];
      if(page<0||page>=pool_pages){atomicOr(&bad,1);continue;}
      int64_t at=int64_t(page)*Page+offset;
      raw_value=overflow_values[at];packed=overflow_ids[at];
    }
    uint32_t word=uint32_t(packed),id=word&0xfffffu;
    uint32_t code=record5::full_code(raw_value, packed);
    uint32_t ord=code<<8,hi_ord=ord|255u;
    float lo=__uint_as_float((ord&0x80000000u)?(ord^0x80000000u):~ord);
    float hi=__uint_as_float((hi_ord&0x80000000u)?(hi_ord^0x80000000u):~hi_ord);
    if(id>=uint32_t(index_limit)||
       id<uint32_t(starts[row])||id>=uint32_t(ends[row])||
       !isfinite(lo)||!isfinite(hi)) {atomicOr(&bad,1);continue;}
    // True t is contained in this entire FP24 cell plus/minus radius.
    // Strictly certain winners can use a final-only proxy; every uncertain
    // possible winner is refined. Equality always belongs to the band.
    if(__fadd_ru(hi,radius)<lower_cut) {
      int pos=atomicAdd(&higher,1);
      if(pos<Wanted)output[int64_t(row)*Wanted+pos]=int(id);
      else atomicOr(&bad,1);
    } else if(__fsub_rd(lo,radius)<=upper_cut) {
      int pos=atomicAdd(&equal,1);
      if(pos<boundary_cap)boundary_ids[int64_t(row)*boundary_cap+pos]=int(id);
      else atomicOr(&bad,1);
    }
  }
  __syncthreads();
  if(t==0) {
    bad|=higher<0||higher>=Wanted||equal<=0||equal>boundary_cap||
         higher+equal<Wanted||higher+equal>temporary_stride;
    diag[row*5]=higher;diag[row*5+1]=equal;diag[row*5+2]=count;
    diag[row*5+3]=higher+equal;
  }
  __syncthreads();
  if(bad){fail_row(temporary,temporary_ids,temporary_counts,boundary_counts,
                  error,row,temporary_stride,t);return;}
  // Only large-E rows need the generic FI proxy merge. Small rows never
  // materialize or read B scores; B is already in final output coordinates.
  if(equal>128)for(int j=t;j<higher;j+=blockDim.x) {
    temporary[int64_t(row)*temporary_stride+j]=FLT_MAX;
    temporary_ids[int64_t(row)*temporary_stride+j]=output[int64_t(row)*Wanted+j];
  }
  if(t==0) {
    boundary_counts[row]=equal;temporary_counts[row]=equal<=128?0:higher;
    // Row-owned H32 completion consumes boundary counts directly.
    // Preserve a defined zero diagnostic without a separate memset launch.
    if(row==0)*task_count=0;
  }
}
} // namespace small_boundary

void collect_band(torch::Tensor inline_val,torch::Tensor inline_idx,
    torch::Tensor overflow_val,torch::Tensor overflow_idx,torch::Tensor page_table,
    torch::Tensor counts,torch::Tensor pool_status,torch::Tensor selector_status,
    torch::Tensor diagnostics,torch::Tensor starts,torch::Tensor ends,
    torch::Tensor boundary_ids,torch::Tensor boundary_counts,torch::Tensor tasks,
    torch::Tensor task_count,torch::Tensor temporary,torch::Tensor temporary_ids,
    torch::Tensor temporary_counts,torch::Tensor error,int64_t logical_cap,
    int64_t index_limit,torch::Tensor epsilon,torch::Tensor output) {
  auto shape=check_paged_candidate_layout(inline_val,inline_idx,overflow_val,
                                         overflow_idx,page_table,logical_cap);
  int q=shape.rows,bcap=boundary_ids.size(1),stride=temporary.size(1);
  TORCH_CHECK(output.is_cuda()&&output.device()==inline_val.device()&&
      output.is_contiguous()&&output.scalar_type()==torch::kInt&&
      output.dim()==2&&output.size(0)==q&&output.size(1)==2048,
      "one contiguous 2048-ID output row per query");
  TORCH_CHECK(epsilon.is_cuda()&&epsilon.device()==inline_val.device()&&
      epsilon.scalar_type()==torch::kFloat&&epsilon.is_contiguous()&&epsilon.numel()==q,
      "one affine error radius per query");
  for(const auto&t:{counts,pool_status,selector_status,diagnostics,starts,ends,
      boundary_ids,boundary_counts,tasks,task_count,temporary,temporary_ids,
      temporary_counts,error})
    TORCH_CHECK(t.is_cuda()&&t.device()==inline_val.device()&&t.is_contiguous(),
                "interval tensors must be contiguous on the candidate device");
  TORCH_CHECK(counts.numel()==q&&selector_status.numel()==q&&starts.numel()==q&&
      ends.numel()==q&&boundary_counts.numel()==q&&temporary_counts.numel()==q&&
      error.numel()==q&&pool_status.numel()==1&&task_count.numel()==1&&
      diagnostics.numel()>=q*5,"interval row metadata shape");
  for(const auto&t:{counts,pool_status,selector_status,diagnostics,starts,ends,
      boundary_ids,boundary_counts,tasks,task_count,temporary_ids,temporary_counts,error})
    TORCH_CHECK(t.scalar_type()==torch::kInt,"interval integer dtype");
  TORCH_CHECK(boundary_ids.dim()==2&&boundary_ids.size(0)==q&&bcap>0&&
      temporary.dim()==2&&temporary.size(0)==q&&stride>=2048&&
      temporary.scalar_type()==torch::kFloat&&temporary_ids.sizes()==temporary.sizes()&&
      tasks.dim()==2&&tasks.size(1)==2&&tasks.size(0)>=int64_t(q)*((bcap+127)/128)&&
      index_limit>0&&index_limit<=(1<<20),"boundary/temporary/task capacity");
  c10::cuda::CUDAGuard guard(inline_val.device());auto stream=c10::cuda::getCurrentCUDAStream();
  small_boundary::collect_kernel<<<q,256,0,stream>>>(
    reinterpret_cast<const uint8_t*>(inline_val.data_ptr<uint8_t>()),inline_idx.data_ptr<int>(),
    reinterpret_cast<const uint8_t*>(overflow_val.data_ptr<uint8_t>()),overflow_idx.data_ptr<int>(),
    page_table.data_ptr<int>(),counts.data_ptr<int>(),pool_status.data_ptr<int>(),
    selector_status.data_ptr<int>(),diagnostics.data_ptr<int>(),starts.data_ptr<int>(),
    ends.data_ptr<int>(),boundary_ids.data_ptr<int>(),boundary_counts.data_ptr<int>(),
    reinterpret_cast<int2*>(tasks.data_ptr<int>()),task_count.data_ptr<int>(),
    temporary.data_ptr<float>(),temporary_ids.data_ptr<int>(),temporary_counts.data_ptr<int>(),
    error.data_ptr<int>(),shape.table_pages_per_row,shape.overflow_pages,shape.logical_cap,
    index_limit,bcap,stride,epsilon.data_ptr<float>(),output.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace small_boundary {
__global__ void publish_replay_kernel(const int*recovered,int*output,const int*error) {
  int row=blockIdx.x;if(error[row]==0)return;
  for(int j=threadIdx.x;j<Wanted;j+=blockDim.x)
    output[int64_t(row)*Wanted+j]=recovered[int64_t(row)*Wanted+j];
}
} // namespace small_boundary

static void check_final_rows(torch::Tensor values,torch::Tensor ids,
    torch::Tensor starts,torch::Tensor ends,torch::Tensor error,
    torch::Tensor output,torch::Tensor tau) {
  int q=values.size(0);
  for(const auto&x:{values,ids,starts,ends,error,output,tau})
    TORCH_CHECK(x.is_cuda()&&x.device()==values.device()&&x.is_contiguous(),"finalization device/layout");
  TORCH_CHECK(values.dim()==2&&values.scalar_type()==torch::kFloat&&
      ids.scalar_type()==torch::kInt&&ids.sizes()==values.sizes()&&values.size(1)>=6144&&
      starts.scalar_type()==torch::kInt&&ends.scalar_type()==torch::kInt&&error.scalar_type()==torch::kInt&&
      starts.numel()==q&&ends.numel()==q&&error.numel()==q&&
      output.scalar_type()==torch::kInt&&output.dim()==2&&output.size(0)==q&&output.size(1)==2048&&
      tau.scalar_type()==torch::kFloat&&tau.numel()==q,"finalization dtype/shape");
}

void publish_replay(torch::Tensor recovered,torch::Tensor output,torch::Tensor error) {
  TORCH_CHECK(recovered.is_cuda()&&output.is_cuda()&&error.is_cuda()&&
      recovered.device()==output.device()&&error.device()==output.device()&&
      recovered.is_contiguous()&&output.is_contiguous()&&error.is_contiguous()&&
      recovered.scalar_type()==torch::kInt&&output.scalar_type()==torch::kInt&&error.scalar_type()==torch::kInt&&
      output.dim()==2&&output.size(1)==2048&&recovered.sizes()==output.sizes()&&error.numel()==output.size(0),"replay publish layout");
  c10::cuda::CUDAGuard guard(output.device());
  small_boundary::publish_replay_kernel<<<output.size(0),128,0,c10::cuda::getCurrentCUDAStream()>>>(
      recovered.data_ptr<int>(),output.data_ptr<int>(),error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace fused_boundary_h32 {
using namespace certicascade_streaming_exact;
using Barrier=cutlass::arch::ClusterTransactionBarrier;
constexpr int BatchQueries=256/kHeads, KeysPerQuery=128/BatchQueries;
constexpr int QueriesPerWarp=32/KeysPerQuery, GroupDoneBit=1<<(BatchQueries/2);
using Sort=cub::BlockRadixSort<float,128,1,int>;
struct alignas(1024) Smem {
  union Work {alignas(1024) uint8_t keys[128*128];typename Sort::TempStorage sort;} work;
  alignas(1024) uint8_t q[kHeads*128];
  alignas(128) float weights[kHeads],scale[128];
  alignas(128) int ids[128],coordinates[128];
  Barrier gathered,done;
  uint32_t tmem;
};
struct Args {
  const uint8_t*q;
  const float*scale,*weights;
  const int*boundary_ids,*boundary_counts,*diag,*starts,*ends;
  int*error,*output;
  float*values;
  int*ids,*counts,*fi_lengths;
  float*tau;
  int nk,stride;
  int64_t boundary_stride;
  int rows;
  int*group_done;
};
__device__ __forceinline__ int swizzle(int row,int vec){return row*8+(vec^(row%8));}
template<int Copies>__device__ __forceinline__ void copy_q_async(uint8_t* dst,const uint8_t* src,int t){
 #pragma unroll
 for(int it=0;it<Copies;++it){
  int i=t+it*128;
  uint32_t to=cute::cast_smem_ptr_to_uint(reinterpret_cast<uint4*>(dst)+swizzle(i/8,i%8));
  const uint4* from=reinterpret_cast<const uint4*>(src)+i;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"::"r"(to),"l"(from):"memory");
  if((it&7)==7||it==Copies-1)asm volatile("cp.async.commit_group;":::"memory");
 }
}

__device__ __forceinline__ void gather(Smem&s,const cute::TmaDescriptor*map) {
  #pragma unroll
  for(int r=0;r<128;r+=4) {
    int4 coord=*reinterpret_cast<const int4*>(s.coordinates+r);
    deep_gemm::ptx::tma_gather4(map,s.gathered,s.work.keys+r*128,0,coord,
        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL));
  }
  s.gathered.arrive_and_expect_tx(128*128);
}

struct alignas(1024) PairSmem {
  alignas(1024) uint8_t keys[128*128],q[2*kHeads*128];
  alignas(128) float weights[2*kHeads],scale[128],exchange_score[128];
  alignas(128) int ids[128],coordinates[128],exchange_id[128];
  int b[2],e[2],bad[2];Barrier gathered,done;uint32_t tmem;
};
__device__ __forceinline__ void complete_pair64(Args a,const cute::TmaDescriptor*map,int first) {
  const int t=threadIdx.x,warp=t/32,qr=t/64,slot=t%64,row=first+qr;
  extern __shared__ __align__(1024) uint8_t buffer[];
  auto&s=*reinterpret_cast<PairSmem*>(buffer);
  if(t<2){int r=first+t;s.b[t]=a.diag[r*5];s.e[t]=a.diag[r*5+1];s.bad[t]=a.error[r]!=0;a.fi_lengths[r]=0;}
  if(t==0){s.gathered.init(1);s.done.init(1);cutlass::arch::fence_barrier_init();cute::prefetch_tma_descriptor(map);}
  __syncthreads();
  bool active=slot<s.e[qr]&&!s.bad[qr];
  int id=active?a.boundary_ids[uint64_t(row)*a.boundary_stride+slot]:0;
  bool bad=active&&(id<a.starts[row]||id>=a.ends[row]||id<0||id>=a.nk);
  if(bad){atomicOr(s.bad+qr,1);atomicOr(a.error+row,1);}
  __syncthreads();
  // A late bad ID invalidates only its query. Both query columns still drain
  // the shared MMA; the healthy partner uses its complete original key set.
  active=slot<s.e[qr]&&!s.bad[qr];id=active?id:0;
  s.ids[t]=active?id:-1;s.coordinates[t]=id;s.scale[t]=active?a.scale[id]:0.f;
  copy_q_async<kHeads/8>(s.q,a.q+uint64_t(first)*kHeads*128,t);
  if(t<2*kHeads)s.weights[t]=a.weights[first*kHeads+t];
  if(warp==0){cute::TMEM::Allocator1Sm().allocate(2*kHeads,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();}
  asm volatile("cp.async.wait_group 0;":::"memory");
  cutlass::arch::fence_view_async_shared();__syncthreads();
  const uint32_t tm=deep_gemm::ptx::ld_shared(&s.tmem);
  if(t==0){
    #pragma unroll
    for(int r=0;r<128;r+=4){int4 coordinates=*reinterpret_cast<const int4*>(s.coordinates+r);
      deep_gemm::ptx::tma_gather4(map,s.gathered,s.keys+r*128,0,coordinates,
        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL));}
    s.gathered.arrive_and_expect_tx(128*128);s.gathered.wait(0);
    deep_gemm::ptx::tcgen05_after_thread_sync();
    issue_h32_mma<128,2*kHeads>(reinterpret_cast<cutlass::float_e4m3_t*>(s.keys),
      reinterpret_cast<cutlass::float_e4m3_t*>(s.q),0,tm,cute::cast_smem_ptr_to_uint(&s.done));
  }
  s.done.wait(0);deep_gemm::ptx::tcgen05_after_thread_sync();
  float accum[kHeads];tmem_load_heads(tm+qr*kHeads,accum);
  float score=exact_h32_reduce(accum,s.weights+qr*kHeads,s.scale[t]);
  if(active&&!isfinite(score)){atomicOr(s.bad+qr,1);atomicOr(a.error+row,1);}
  deep_gemm::ptx::tcgen05_before_thread_sync();__syncthreads();
  if(warp==0)cute::TMEM::Allocator1Sm().free(tm,2*kHeads);
  __syncthreads();
  score=active&&!s.bad[qr]?score:-INFINITY;id=active&&!s.bad[qr]?s.ids[t]:-1;
  #pragma unroll
  for(int width=2;width<=64;width*=2){
    #pragma unroll
    for(int step=width/2;step>0;step/=2){
      float other;int other_id;
      if(step==32){
        s.exchange_score[t]=score;s.exchange_id[t]=id;__syncthreads();
        other=s.exchange_score[t^32];other_id=s.exchange_id[t^32];
        // This is the only cross-warp exchange; these shared slots are never
        // overwritten again, so no second read-completion barrier is needed.
      }else{other=__shfl_xor_sync(0xffffffffu,score,step);other_id=__shfl_xor_sync(0xffffffffu,id,step);}
      bool take_best=((slot&step)==0)==((slot&width)==0);
      bool other_better=other>score||(other==score&&other_id<id);
      if(take_best?other_better:!other_better){score=other;id=other_id;}
    }
  }
  const int needed=2048-s.b[qr];
  if(!s.bad[qr]){
    if(slot<needed)a.output[uint64_t(row)*2048+s.b[qr]+slot]=id;
    if(slot==needed-1)a.tau[row]=score;
    if(slot==0)a.counts[row]=s.e[qr];
  }
}

__device__ __forceinline__ uint32_t exact_ordered(float x) {
 uint32_t u=__float_as_uint(x);return (u&0x80000000u)?~u:(u^0x80000000u);
}
__device__ __forceinline__ void finish_large_radix(Args a,int row,int b,int e,uint32_t* work){
 const int t=threadIdx.x,lane=t&31,needed=2048-b;uint32_t*state=work+256;
 bool bad=false;
 for(int j=t;j<b+e;j+=128){float v=a.values[uint64_t(row)*a.stride+j];int id=a.ids[uint64_t(row)*a.stride+j];
  bad|=!isfinite(v)||id<a.starts[row]||id>=a.ends[row];}
 if(__syncthreads_or(bad)){if(t==0)atomicOr(a.error+row,1);return;}
 if(t==0){state[0]=0;state[1]=needed;state[2]=0;state[3]=0;state[4]=0;state[5]=0;}
 __syncthreads();
 #pragma unroll
 for(int pass=0;pass<4;++pass){
  const int shift=24-8*pass;const uint32_t prefix_mask=shift==24?0u:(0xffffffffu<<(shift+8));
  work[t]=0;work[t+128]=0;__syncthreads();
  const uint32_t prefix=state[0];
  for(int j=t;j<e;j+=128){uint32_t code=exact_ordered(a.values[uint64_t(row)*a.stride+b+j]);
   if((code&prefix_mask)==prefix)atomicAdd(work+((code>>shift)&255u),1u);}
  __syncthreads();
  if(t<32){
   uint32_t local=0;const int first=255-lane*8;
   #pragma unroll
   for(int j=0;j<8;++j)local+=work[first-j];
   uint32_t inclusive=local;
   #pragma unroll
   for(int off=1;off<32;off<<=1){uint32_t x=__shfl_up_sync(0xffffffffu,inclusive,off);if(lane>=off)inclusive+=x;}
   uint32_t rank=state[1];unsigned hits=__ballot_sync(0xffffffffu,inclusive>=rank);
   int owner=hits?__ffs(hits)-1:0;uint32_t digit=0,next_rank=0,equal=0;
   if(lane==owner){uint32_t remaining=rank-(inclusive-local);
    #pragma unroll
    for(int j=0;j<8;++j){uint32_t count=work[first-j];if(remaining>count)remaining-=count;else{digit=first-j;next_rank=remaining;equal=count;break;}}
   }
   digit=__shfl_sync(0xffffffffu,digit,owner);next_rank=__shfl_sync(0xffffffffu,next_rank,owner);equal=__shfl_sync(0xffffffffu,equal,owner);
   if(lane==0){if(!hits||!rank||!next_rank||next_rank>equal)state[3]=1;state[0]|=digit<<shift;state[1]=next_rank;state[2]=equal;}
  }
  __syncthreads();
 }
 if(state[3]){if(t==0)atomicOr(a.error+row,1);return;}
 const uint32_t pivot=state[0],equal_take=state[1];const int strict=needed-int(equal_take);
 if(strict<0||equal_take==0){if(t==0)atomicOr(a.error+row,1);return;}
 for(int j=t;j<e;j+=128){
  int id=a.ids[uint64_t(row)*a.stride+b+j];uint32_t code=exact_ordered(a.values[uint64_t(row)*a.stride+b+j]);
  if(code>pivot){uint32_t pos=atomicAdd(state+4,1u);if(pos<uint32_t(strict))a.output[uint64_t(row)*2048+b+pos]=id;else atomicOr(state+3,1u);}
  else if(code==pivot){uint32_t pos=atomicAdd(state+5,1u);if(pos<equal_take)a.output[uint64_t(row)*2048+b+strict+pos]=id;}
 }
 for(int j=t;j<b;j+=128)a.output[uint64_t(row)*2048+j]=a.ids[uint64_t(row)*a.stride+j];
 __syncthreads();
 if(t==0){
  if(state[3]||state[4]!=uint32_t(strict)||state[5]!=state[2])atomicOr(a.error+row,1);
  else a.tau[row]=__uint_as_float((pivot&0x80000000u)?(pivot^0x80000000u):~pivot);
 }
}
template<bool FusedLarge=false>
CUTLASS_GLOBAL void __launch_bounds__(128,4) complete(
    Args a,const __grid_constant__ cute::TmaDescriptor map) {
  int row=blockIdx.x,t=threadIdx.x,warp=t/32;
  const int plan=a.group_done[row/BatchQueries];
  if(plan&GroupDoneBit)return;
  if(plan&(1<<((row%BatchQueries)/2))){
    if((row&1)==0)complete_pair64(a,&map,row);
    return;
  }
  int b=a.diag[row*5],e=a.diag[row*5+1];
  if(t==0)a.fi_lengths[row]=0;
  if(a.error[row]!=0)return;
  bool invalid=b<0||b>=2048||e<=0||e>4096||b+e<2048||
      b+e>a.stride||a.boundary_counts[row]!=e||a.starts[row]<0||
      a.ends[row]<a.starts[row]||a.ends[row]>a.nk;
  if(invalid){if(t==0)atomicOr(a.error+row,1);return;}
  extern __shared__ __align__(1024) uint8_t buffer[];
  auto&s=*reinterpret_cast<Smem*>(buffer);
  if(t==0) {
    s.gathered.init(1);s.done.init(1);
    cutlass::arch::fence_barrier_init();cute::prefetch_tma_descriptor(&map);
  }
  for(int i=t;i<kHeads*8;i+=128)
    reinterpret_cast<uint4*>(s.q)[swizzle(i/8,i%8)]=
        reinterpret_cast<const uint4*>(a.q+uint64_t(row)*kHeads*128)[i];
  if(t<kHeads)s.weights[t]=a.weights[row*kHeads+t];
  if(warp==0) {
    cute::TMEM::Allocator1Sm().allocate(kHeads,&s.tmem);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  cutlass::arch::fence_view_async_shared();__syncthreads();
  const uint32_t tm=deep_gemm::ptx::ld_shared(&s.tmem);
  float small_score[1]={-INFINITY};int small_id[1]={-1};
  bool failed=false;
  for(int begin=0,packet=0;begin<e;begin+=128,++packet) {
    bool active=begin+t<e;
    int id=active?a.boundary_ids[uint64_t(row)*a.boundary_stride+begin+t]:0;
    bool bad=active&&(id<a.starts[row]||id>=a.ends[row]||id<0||id>=a.nk);
    // Validate logical coordinates before TMA or the scale load.
    if(__syncthreads_or(bad)) {if(t==0)atomicOr(a.error+row,1);failed=true;break;}
    s.ids[t]=id;s.coordinates[t]=id;s.scale[t]=active?a.scale[id]:0.f;
    cutlass::arch::fence_view_async_shared();__syncthreads();
    if(t==0) {
      gather(s,&map);s.gathered.wait(packet&1);
      deep_gemm::ptx::tcgen05_after_thread_sync();
      issue_h32_mma<128,kHeads>(reinterpret_cast<cutlass::float_e4m3_t*>(s.work.keys),
          reinterpret_cast<cutlass::float_e4m3_t*>(s.q),0,tm,cute::cast_smem_ptr_to_uint(&s.done));
    }
    s.done.wait(packet&1);deep_gemm::ptx::tcgen05_after_thread_sync();
    float accum[kHeads];
    tmem_load_heads(tm,accum);
    float score=exact_h32_reduce(accum,s.weights,s.scale[t]);
    if(__syncthreads_or(active&&!isfinite(score))) {
      if(t==0)atomicOr(a.error+row,1);failed=true;break;
    }
    if(e<=128) {small_score[0]=active?score:-INFINITY;small_id[0]=active?id:-1;}
    else if(active) {
      uint64_t at=uint64_t(row)*a.stride+b+begin+t;
      a.values[at]=score;a.ids[at]=id;
    }
    deep_gemm::ptx::tcgen05_before_thread_sync();__syncthreads();
  }
  // No MMA, TMA or TMEM consumer survives this barrier. Reuse dead K SMEM
  // for CUB and release TMEM before sorting the per-thread H32 registers.
  deep_gemm::ptx::tcgen05_before_thread_sync();__syncthreads();
  if(warp==0)cute::TMEM::Allocator1Sm().free(tm,kHeads);
  __syncthreads();
  if(failed)return;
  if(e<=128) {
    Sort(s.work.sort).SortDescending(small_score,small_id);
    int needed=2048-b;
    if(t<needed)a.output[uint64_t(row)*2048+b+t]=small_id[0];
    if(t==needed-1)a.tau[row]=small_score[0];
    if(t==0)a.counts[row]=e;
  } else {
    if(t==0){a.counts[row]=b+e;if constexpr(!FusedLarge)a.fi_lengths[row]=b+e;}
    if constexpr(FusedLarge){__syncthreads();finish_large_radix(a,row,b,e,reinterpret_cast<uint32_t*>(s.work.keys));}
  }
}
struct alignas(1024) BatchSmem {
  alignas(1024) uint8_t keys[128*128],q[256*128];
  alignas(128) float weights[256],scale[128];
  alignas(128) int ids[128],coordinates[128];
  int b[BatchQueries],e[BatchQueries];
  Barrier gathered,done;
  uint32_t tmem;
};

CUTLASS_GLOBAL void __launch_bounds__(128,2) complete_batch8(
    Args a,const __grid_constant__ cute::TmaDescriptor map) {
  const int group=blockIdx.x,first=group*BatchQueries,t=threadIdx.x,warp=t/32;
  // Freeze the Q2 decisions before any later completion can report a fault.
  // A complete B8 group remains first priority and uses its original body.
  bool invalid=false,eligible64=false;
  if(t<BatchQueries){
    int row=first+t;
    if(row>=a.rows)invalid=true;
    else{
      int b=a.diag[row*5],e=a.diag[row*5+1];
      bool valid=a.error[row]==0&&b>=0&&b<2048&&e>0&&b+e>=2048&&
        b+e<=a.stride&&a.boundary_counts[row]==e&&a.starts[row]>=0&&
        a.ends[row]>=a.starts[row]&&a.ends[row]<=a.nk;
      invalid=!valid||e>KeysPerQuery;eligible64=valid&&e<=64;
    }
  }
  const unsigned mask=__ballot_sync(0xffffffffu,eligible64);
  if(t==0){int plan=0;
    #pragma unroll
    for(int p=0;p<BatchQueries/2;++p)if(((mask>>(2*p))&3u)==3u)plan|=1<<p;
    a.group_done[group]=plan;
  }
  if(__syncthreads_or(invalid))return;
  extern __shared__ __align__(1024) uint8_t buffer[];
  auto&s=*reinterpret_cast<BatchSmem*>(buffer);
  if(t<BatchQueries) {
    s.b[t]=a.diag[(first+t)*5];s.e[t]=a.diag[(first+t)*5+1];
    a.fi_lengths[first+t]=0;
  }
  if(t==0) {
    s.gathered.init(1);s.done.init(1);
    cutlass::arch::fence_barrier_init();cute::prefetch_tma_descriptor(&map);
  }
  __syncthreads();
  int qr=t/KeysPerQuery,row=first+qr,slot=t%KeysPerQuery;
  const bool active=slot<s.e[qr];
  int id=active?a.boundary_ids[uint64_t(row)*a.boundary_stride+slot]:0;
  bool bad=active&&(id<a.starts[row]||id>=a.ends[row]||id<0||id>=a.nk);
  if(bad)atomicOr(a.error+row,1);
  if(__syncthreads_or(bad))return;
  s.ids[t]=active?id:-1;s.coordinates[t]=id;s.scale[t]=active?a.scale[id]:0.f;
  copy_q_async<16>(s.q,a.q+uint64_t(first)*kHeads*128,t);
  for(int i=t;i<256;i+=128)s.weights[i]=a.weights[first*kHeads+i];
  if(warp==0) {
    cute::TMEM::Allocator1Sm().allocate(256,&s.tmem);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  asm volatile("cp.async.wait_group 0;":::"memory");
  cutlass::arch::fence_view_async_shared();__syncthreads();
  uint32_t tm=deep_gemm::ptx::ld_shared(&s.tmem);
  if(t==0) {
    #pragma unroll
    for(int r=0;r<128;r+=4) {
      int4 coordinates=*reinterpret_cast<const int4*>(s.coordinates+r);
      deep_gemm::ptx::tma_gather4(&map,s.gathered,s.keys+r*128,0,coordinates,
        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL));
    }
    s.gathered.arrive_and_expect_tx(128*128);s.gathered.wait(0);
    deep_gemm::ptx::tcgen05_after_thread_sync();
    issue_h32_mma<128,256>(reinterpret_cast<cutlass::float_e4m3_t*>(s.keys),
      reinterpret_cast<cutlass::float_e4m3_t*>(s.q),0,tm,
      cute::cast_smem_ptr_to_uint(&s.done));
  }
  s.done.wait(0);deep_gemm::ptx::tcgen05_after_thread_sync();
  float score=-INFINITY;
  #pragma unroll
  for(int half=0;half<QueriesPerWarp;++half) {
    // Column addresses are uniform within each warp. Its lower/upper half
    // retains the corresponding query's original-H32 result.
    int query=warp*QueriesPerWarp+half;
    float accum[kHeads];
    tmem_load_heads(tm+query*kHeads,accum);
    float v=exact_h32_reduce(accum,s.weights+query*kHeads,s.scale[t]);
    if((t%32)/KeysPerQuery==half)score=active?v:-INFINITY;
  }
  bad=active&&!isfinite(score);
  if(bad)atomicOr(a.error+row,1);
  bool failed=__syncthreads_or(bad);
  deep_gemm::ptx::tcgen05_before_thread_sync();__syncthreads();
  if(warp==0)cute::TMEM::Allocator1Sm().free(tm,256);
  __syncthreads();
  if(failed)return;
  id=s.ids[t];
  #pragma unroll
  for(int width=2;width<=KeysPerQuery;width*=2) {
    #pragma unroll
    for(int step=width/2;step>0;step/=2) {
      float other=__shfl_xor_sync(0xffffffffu,score,step,KeysPerQuery);
      int other_id=__shfl_xor_sync(0xffffffffu,id,step,KeysPerQuery);
      bool take_best=((slot&step)==0)==((slot&width)==0);
      bool other_better=other>score||(other==score&&other_id<id);
      if(take_best?other_better:!other_better){score=other;id=other_id;}
    }
  }
  int needed=2048-s.b[qr];
  if(slot<needed)a.output[uint64_t(row)*2048+s.b[qr]+slot]=id;
  if(slot==needed-1)a.tau[row]=score;
  if(slot==0)a.counts[row]=s.e[qr];
  __syncthreads();
  if(t==0)a.group_done[group]=GroupDoneBit;
}
} // namespace fused_boundary_h32

template<bool FusedLarge=false>void complete_h32(torch::Tensor q,torch::Tensor k,torch::Tensor scale,torch::Tensor weights,
    torch::Tensor boundary_ids,torch::Tensor boundary_counts,torch::Tensor diag,
    torch::Tensor starts,torch::Tensor ends,torch::Tensor error,torch::Tensor output,
    torch::Tensor values,torch::Tensor ids,torch::Tensor counts,torch::Tensor fi_lengths,
    torch::Tensor tau) {
  check_final_rows(values,ids,starts,ends,error,output,tau);int rows=values.size(0);
  for(const auto&x:{q,k,scale,weights,boundary_ids,boundary_counts,diag,counts,fi_lengths})
    TORCH_CHECK(x.is_cuda()&&x.device()==values.device()&&x.is_contiguous(),"H32 finalization device/layout");
  TORCH_CHECK(q.dim()==3&&q.size(0)==rows&&q.size(1)==dosa_geometry::Heads&&q.size(2)==128&&
      k.dim()==2&&k.size(1)==128&&k.size(0)>0&&k.size(0)<=(1<<20)&&
      q.element_size()==1&&k.element_size()==1&&scale.scalar_type()==torch::kFloat&&
      scale.numel()==k.size(0)&&weights.scalar_type()==torch::kFloat&&
      weights.sizes()==torch::IntArrayRef({rows,dosa_geometry::Heads}),"original H32 shape/dtype");
  for(const auto&x:{boundary_ids,boundary_counts,diag,counts,fi_lengths})
    TORCH_CHECK(x.scalar_type()==torch::kInt,"H32 finalization integer metadata");
  TORCH_CHECK(boundary_ids.dim()==2&&boundary_ids.size(0)==rows&&boundary_ids.size(1)==4096&&
      boundary_counts.numel()==rows&&counts.numel()==rows&&fi_lengths.numel()==rows&&diag.numel()>=int64_t(rows)*5,"boundary row capacity");
  using namespace fused_boundary_h32;
  c10::cuda::CUDAGuard guard(q.device());
  auto groups=torch::empty({(rows+BatchQueries-1)/BatchQueries},error.options());
  Args a{static_cast<const uint8_t*>(q.data_ptr()),scale.data_ptr<float>(),weights.data_ptr<float>(),
      boundary_ids.data_ptr<int>(),boundary_counts.data_ptr<int>(),diag.data_ptr<int>(),
      starts.data_ptr<int>(),ends.data_ptr<int>(),error.data_ptr<int>(),output.data_ptr<int>(),
      values.data_ptr<float>(),ids.data_ptr<int>(),counts.data_ptr<int>(),fi_lengths.data_ptr<int>(),
      tau.data_ptr<float>(),int(k.size(0)),int(values.size(1)),boundary_ids.stride(0),rows,
      groups.data_ptr<int>()};
  auto map=make_map(k.data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT8,{128,(cuuint64_t)k.size(0)},
      {128},{128,1},CU_TENSOR_MAP_SWIZZLE_128B);
  C10_CUDA_CHECK(cudaFuncSetAttribute(complete<FusedLarge>,cudaFuncAttributeMaxDynamicSharedMemorySize,std::max(sizeof(Smem),sizeof(PairSmem))));
  C10_CUDA_CHECK(cudaFuncSetAttribute(complete_batch8,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(BatchSmem)));
  complete_batch8<<<(rows+BatchQueries-1)/BatchQueries,128,sizeof(BatchSmem),c10::cuda::getCurrentCUDAStream()>>>(a,map);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  complete<FusedLarge><<<rows,128,std::max(sizeof(Smem),sizeof(PairSmem)),c10::cuda::getCurrentCUDAStream()>>>(a,map);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


#include "replay.cuh"
#include "conditional.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("replay_capacity", &replay_capacity);
  m.attr("requires_coarse_band") = true;
  m.def("collect_band", &collect_band);
  m.def("complete_h32_fused_large", &complete_h32<true>);
  m.def("publish_replay", &publish_replay);
  m.def("replay", &replay);
  m.def("attach_if_any", &attach_if_any);
  m.def("attach_if_any_not_in", &attach_if_any_not_in);
}
