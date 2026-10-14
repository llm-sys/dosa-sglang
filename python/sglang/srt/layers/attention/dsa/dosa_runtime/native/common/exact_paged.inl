// Shared static/device-extent implementation. Included inside the selected
// namespace policy; keep the distinct live metadata pipeline and guards.

namespace DOSA_PAGED_NAMESPACE {
template<int BQ,bool Record4=false,bool Record5=false>
CUTLASS_GLOBAL void __launch_bounds__(128,4) exact(Args a,const __grid_constant__ cute::TmaDescriptor map){
  static_assert(!(Record4&&Record5));
  constexpr int TmemColumns = Record5 && BQ*kHeads > 128 ? 128 : BQ*kHeads;
#if DOSA_PAGED_LIVE_EXTENT
  // The descriptor keeps physical capacity; every candidate is checked
  // against this same epoch's logical extent before scale/TMA access.
  const int logical=*a.live_nk;
  if(logical<a.candidate_seed||logical>a.nk){
    for(int qr=blockIdx.x*blockDim.x+threadIdx.x;qr<a.qrows;qr+=gridDim.x*blockDim.x)atomicOr(a.error+qr,2);
    return;
  }
  a.nk=logical;a.table_width=(logical-a.candidate_seed+PacketIds-1)/PacketIds;
#else
#endif
  int task_count=*a.packs,group_ids=PacketIds*a.packs[2];
  if(blockIdx.x>=task_count)return;
  extern __shared__ __align__(1024) uint8_t buf[];
  auto&s=*reinterpret_cast<Smem<BQ>*>(buf);
  int t=threadIdx.x,warp=t/32,lane=t%32;
  if(t==0){
    s.gathered[0].init(1);s.gathered[1].init(1);s.done.init(1);
    cutlass::arch::fence_barrier_init();cute::prefetch_tma_descriptor(&map);
  }
  if(warp==0){cute::TMEM::Allocator1Sm().allocate(TmemColumns,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();}
  __syncthreads();
  int tm=deep_gemm::ptx::ld_shared(&s.tmem);
  uint32_t phase_bits=0,done_sequence=0;
  for(;;){
    if(t==0)s.job=atomicAdd(a.cursor,1);
    __syncthreads();
    int job=s.job;
    if(job>=task_count)break;
    int2 task=a.tasks[job];
    int row=task.x,begin=task.y,stop=min(begin+group_ids,a.counts[row]);
    if(begin>=stop){__syncthreads();continue;}
    int packets=(stop-begin+PacketIds-1)/PacketIds;
    int failed_columns=0;
    if constexpr(BQ==1){
      if(t==0){s.skip=0;for(int qb=0;qb<BQ;++qb)if(row*BQ+qb<a.qrows&&a.error[row*BQ+qb]!=0)s.skip=1;}
      __syncthreads();
      if(s.skip){if(t==0)flag_group<BQ>(a,row,4);__syncthreads();continue;}
    }else{
      if(t==0){s.skip=0;for(int qb=0;qb<BQ;++qb)if(row*BQ+qb<a.qrows&&a.error[row*BQ+qb]!=0)s.skip|=1<<qb;}
      __syncthreads();failed_columns=s.skip;
      const int real_columns=min(BQ,a.qrows-row*BQ);
      if(failed_columns==((1<<real_columns)-1)){if(t==0)flag_group<BQ>(a,row,4);__syncthreads();continue;}
    }
    if constexpr (Record5) {
      // Both solo BQ1 and common BQ4 execute the same shared tile, H32 and
      // writer body. The access object contains only the original loader
      // schedule, metadata and address views.
      PagedScoreAccess<BQ, true> access(
          s, a, &map, {row, begin, stop, uint32_t(failed_columns)},
          uint32_t(tm), phase_bits, done_sequence, t);
      access.initialize_work();
      using Geometry = dosa_score_emit::Geometry<
          BQ, 128, 1, 16, 1, 1, false, false, BQ == 4 ? 4 : 1>;
      using G = Geometry;
      constexpr auto D = dosa_score_emit::Domain::OriginalH32Global20;
      auto &a = access;
      auto output = access.output_view();
      auto storage = access.emit_storage();
      using namespace dosa_score_emit;
#if DOSA_PAGED_LIVE_EXTENT
#define DOSA_SCORE_EMIT_AFTER_QUERY(access,query) do { \
  (access).finish_future_scale(query); \
  (access).advance_query_stage(query); \
} while (0)
#else
#define DOSA_SCORE_EMIT_AFTER_QUERY(access,query) (access).advance_query_stage(query)
#endif
#include "../common/score_emit_body.inl"
#undef DOSA_SCORE_EMIT_AFTER_QUERY
    } else {
      // Legacy explicit F32/record4 compatibility, pending codec integration.
      // Record5 has no score/emit implementation here: both BQ1/BQ4 above
      // execute the public common body.
      for(int i=t;i<BQ*kHeads*8;i+=128)
        reinterpret_cast<uint4*>(s.q)[swizzle(i/8,i%8)]=(row*BQ+i/(kHeads*8)<a.qrows)?reinterpret_cast<const uint4*>(a.q+uint64_t(row)*BQ*kHeads*128)[i]:make_uint4(0,0,0,0);
      for(int h=t;h<BQ*kHeads;h+=128)s.weights[h]=(row*BQ+h/kHeads<a.qrows)?a.w[row*BQ*kHeads+h]:0;
      prepare_metadata(s,a,row,begin,stop,0,t);
      if(packets>1)prepare_metadata(s,a,row,begin+PacketIds,stop,1,t);
      if(t==0)s.gathered[0].arrive_and_expect_tx(128*128);
      cutlass::arch::fence_view_async_shared();__syncthreads();
      gather(s,&map,0,t);
      for(int packet=0;packet<packets;packet++){
        int slot=packet&1,valid=min(PacketIds,stop-begin-packet*PacketIds);
        if(t==0){
          s.gathered[slot].wait((phase_bits>>slot)&1u);phase_bits^=1u<<slot;
          deep_gemm::ptx::tcgen05_after_thread_sync();
          issue_h32_mma<128,BQ*kHeads>(reinterpret_cast<cutlass::float_e4m3_t*>(s.keys[slot]),reinterpret_cast<cutlass::float_e4m3_t*>(s.q),0,tm,cute::cast_smem_ptr_to_uint(&s.done));
          if(packet+1<packets)s.gathered[slot^1].arrive_and_expect_tx(128*128);
        }
        __syncthreads();
        if(packet+1<packets)gather(s,&map,slot^1,t);
        FutureMetadata future{};
        if(packet+2<packets)future=load_future<BQ>(a,row,begin+(packet+2)*PacketIds,stop,t);
        s.done.wait(done_sequence&1);deep_gemm::ptx::tcgen05_after_thread_sync();
        #pragma unroll
        for(int qb=0;qb<BQ;qb++){
          float acc[kHeads];int output_row=row*BQ+qb;
          if(output_row>=a.qrows)continue;
          if constexpr(BQ>1){if(failed_columns&(1<<qb))continue;}
          tmem_load_heads(tm+qb*kHeads,acc);
          float score=exact_h32_reduce(acc,s.weights+qb*kHeads,s.scale[slot][t]);
          if(t<valid&&!isfinite(score))atomicOr(a.error+output_row,8);
          bool keep=t<valid&&a.error[output_row]==0&&(!isfinite(a.tau[output_row])||score>=a.tau[output_row]);
          uint32_t mask=__ballot_sync(0xffffffffu,keep),n=__popc(mask),base=0;
          if(lane==0&&n)base=atomicAdd(a.outcounts+output_row,n);
          base=__shfl_sync(0xffffffffu,base,0);
          if(keep){
            int64_t column=int64_t(base)+__popc(lane?mask&((1u<<lane)-1):0);
            if(column>=0&&column<a.output_cap){
              if constexpr(Record4){
                const int id=s.ids[slot][t]+a.origin;
                const uint32_t cp=record4::encode(score,a.tau[output_row]);
                const bool good=isfinite(score)&&isfinite(a.tau[output_row])&&score>=a.tau[output_row]&&
                    __float_as_uint(score)!=0x80000000u&&id>=0&&uint32_t(id)<0x100000u&&cp<=0xC00u;
                a.outids[int64_t(output_row)*a.output_id_stride+column]=int32_t(((good?cp:0xFFFu)<<20)|(uint32_t(id)&0xFFFFFu));
                if(!good)atomicOr(a.error+output_row,32);
              }else{
                static_cast<float*>(a.scores)[int64_t(output_row)*a.score_stride+column]=score;
                a.outids[int64_t(output_row)*a.output_id_stride+column]=s.ids[slot][t]+a.origin;
              }
            }else atomicOr(a.error+output_row,16);
          }
        }
        if(packet+2<packets)publish_future(s,a,row,slot,t,future);
        deep_gemm::ptx::tcgen05_before_thread_sync();__syncthreads();
        done_sequence++;
      }
    }
    // All issued gathers have been consumed. Per-slot phases remain global
    // across groups, including groups with an odd number of packets.
  }
  if(warp==0)cute::TMEM::Allocator1Sm().free(tm,TmemColumns);
}
}

#if DOSA_PAGED_LIVE_EXTENT
namespace live_record5_host {
#else
#endif
template<int BQ,bool Record4=false,bool Record5=false>
void exact_paged_completion(torch::Tensor q,torch::Tensor k,torch::Tensor scale,torch::Tensor weights,
            torch::Tensor tau,torch::Tensor ids,torch::Tensor counts,torch::Tensor tasks,
            torch::Tensor packs,torch::Tensor scores,torch::Tensor outids,torch::Tensor outcounts,int64_t origin,int64_t sms,
#if DOSA_PAGED_LIVE_EXTENT
            torch::Tensor page_table,int64_t pool_pages,torch::Tensor starts,torch::Tensor ends,torch::Tensor error,torch::Tensor live_nk,int64_t seed){
#else
            torch::Tensor page_table,int64_t pool_pages,torch::Tensor starts,torch::Tensor ends,torch::Tensor error){
#endif
  using namespace DOSA_PAGED_NAMESPACE;
  static_assert(!(Record4&&Record5));
  TORCH_CHECK(q.is_cuda()&&q.dim()==3&&q.size(1)==kHeads&&q.size(2)==128&&q.element_size()==1&&q.is_contiguous(),"q CUDA contiguous [Q,model-heads,128] bytes");
  c10::cuda::CUDAGuard guard(q.device());
  for(const auto&x:{k,scale,weights,tau,ids,counts,tasks,packs,scores,outids,outcounts,page_table,starts,ends,error})
    TORCH_CHECK(x.is_cuda()&&x.device()==q.device(),"paged exact tensors share a CUDA device");
  int64_t qr=q.size(0),streams=(qr+BQ-1)/BQ;
  TORCH_CHECK(qr>0&&qr<=std::numeric_limits<int>::max()&&sms>0&&sms<65536,"positive supported Q/SM count");
  TORCH_CHECK(k.dim()==2&&k.size(1)==128&&k.element_size()==1&&k.is_contiguous()&&k.size(0)>0&&k.size(0)<=((Record4||Record5)?(1<<20):std::numeric_limits<int>::max()),"K contiguous byte [K,128]");
  TORCH_CHECK(scale.scalar_type()==torch::kFloat&&scale.is_contiguous()&&scale.numel()>=k.size(0),"contiguous F32 K scales");
  TORCH_CHECK(weights.scalar_type()==torch::kFloat&&weights.sizes()==torch::IntArrayRef({qr,kHeads})&&weights.is_contiguous(),"contiguous model-head F32 weights");
  TORCH_CHECK(tau.scalar_type()==torch::kFloat&&tau.is_contiguous()&&tau.numel()>=qr,"F32 Q thresholds");
  for(const auto&x:{ids,counts,tasks,packs,outids,outcounts,page_table,starts,ends,error})TORCH_CHECK(x.scalar_type()==torch::kInt,"integer paged metadata");
  TORCH_CHECK(ids.dim()==2&&ids.size(1)==128&&ids.stride(1)==1&&ids.stride(0)>=128&&pool_pages>=0&&pool_pages<=ids.size(0),"128-ID pool pages and capacity");
  TORCH_CHECK(page_table.dim()==2&&page_table.size(0)>=streams&&page_table.size(1)>0&&page_table.size(1)<=std::numeric_limits<int>::max()&&page_table.stride(1)==1&&page_table.stride(0)>=page_table.size(1),"stream page table view");
  for(const auto&x:{counts,packs,outcounts,starts,ends,error})TORCH_CHECK(x.is_contiguous(),"contiguous scalar/row metadata");
  TORCH_CHECK(counts.numel()>=streams&&packs.numel()>=1&&outcounts.numel()>=qr&&starts.numel()>=qr&&ends.numel()>=qr&&error.numel()>=qr,"row/count metadata capacity");
  TORCH_CHECK(tasks.dim()==2&&tasks.size(1)==2&&tasks.is_contiguous()&&tasks.size(0)<=std::numeric_limits<int>::max(),"contiguous int2 tasks");
  TORCH_CHECK(scores.scalar_type()==(Record5?torch::kByte:torch::kFloat)&&scores.dim()==2&&outids.dim()==2&&scores.size(0)>=qr&&outids.size(0)>=qr&&scores.size(1)>0&&outids.size(1)>0&&scores.stride(1)==1&&outids.stride(1)==1&&scores.stride(0)>=scores.size(1)&&outids.stride(0)>=outids.size(1),"score/ID output row views");
  TORCH_CHECK(origin>=0&&origin<=((Record4||Record5)?(1<<20):std::numeric_limits<int>::max())-k.size(0),"visible global IDs fit int32");
  if constexpr(Record4||Record5) TORCH_CHECK(outids.size(1)<=(1<<20),"packed output row capacity fits int32");
#if DOSA_PAGED_LIVE_EXTENT
  TORCH_CHECK(live_nk.is_cuda()&&live_nk.device()==q.device()&&live_nk.scalar_type()==torch::kInt32&&live_nk.is_contiguous()&&live_nk.dim()==1&&live_nk.numel()==1,"stable device int32[1] live NK");
  TORCH_CHECK(seed>=0&&seed<=k.size(0)&&seed%256==0&&page_table.size(1)>=(k.size(0)-seed+127)/128,"fixed physical table capacity and seed");
#else
#endif
  auto group_tasks=torch::empty_like(tasks);
  auto state=torch::zeros({3},counts.options());
  filter_groups<BQ><<<4*sms,128,0,at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const int2*>(tasks.data_ptr<int>()),packs.data_ptr<int>(),
      reinterpret_cast<int2*>(group_tasks.data_ptr<int>()),state.data_ptr<int>(),8*sms,
#if DOSA_PAGED_LIVE_EXTENT
      tasks.size(0),counts.data_ptr<int>(),page_table.size(1),streams,qr,error.data_ptr<int>(),live_nk.data_ptr<int>(),int(k.size(0)),int(seed));
#else
      tasks.size(0),counts.data_ptr<int>(),page_table.size(1),streams,qr,error.data_ptr<int>());
#endif
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  Args a{static_cast<const uint8_t*>(q.data_ptr()),weights.data_ptr<float>(),scale.data_ptr<float>(),tau.data_ptr<float>(),ids.data_ptr<int>(),counts.data_ptr<int>(),state.data_ptr<int>(),reinterpret_cast<const int2*>(group_tasks.data_ptr<int>()),scores.data_ptr(),outids.data_ptr<int>(),outcounts.data_ptr<int>(),int(k.size(0)),int(scores.size(1)),int(origin),ids.stride(0),state.data_ptr<int>()+1,
#if DOSA_PAGED_LIVE_EXTENT
      page_table.data_ptr<int>(),starts.data_ptr<int>(),ends.data_ptr<int>(),error.data_ptr<int>(),int(qr),int(streams),int(page_table.size(1)),int(pool_pages),int(Record4?outids.size(1):std::min(scores.size(1),outids.size(1))),page_table.stride(0),ids.stride(0),scores.stride(0),outids.stride(0),live_nk.data_ptr<int>(),int(seed)};
#else
      page_table.data_ptr<int>(),starts.data_ptr<int>(),ends.data_ptr<int>(),error.data_ptr<int>(),int(qr),int(streams),int(page_table.size(1)),int(pool_pages),int(Record4?outids.size(1):std::min(scores.size(1),outids.size(1))),page_table.stride(0),ids.stride(0),scores.stride(0),outids.stride(0)};
#endif
  auto map=make_map(k.data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT8,{128,(cuuint64_t)k.size(0)},{128},{128,1},CU_TENSOR_MAP_SWIZZLE_128B);
  auto fn=exact<BQ,Record4,Record5>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Smem<BQ>)));
  fn<<<4*sms,128,sizeof(Smem<BQ>),at::cuda::getCurrentCUDAStream()>>>(a,map);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#if DOSA_PAGED_LIVE_EXTENT

} // live_record5_host
#else
#endif
